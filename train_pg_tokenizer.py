import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils.losses as losses 
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from utils.word_vectorizer import WordVectorizer, WordVectorizer_only_text_token
from utils.misc import T2M_CONTACT_JOINTS
from utils.codebook import group_id_to_full_group_name, vq_to_range
from utils.eval_trans import tensorboard_add_image_pw_sim, tensorboard_add_image_tsne # , tensorboard_and_wandb_add_tsne
from utils.motion_process import recover_from_ric
from utils.trainUtil import update_lr_warm_up
from utils.loss_wrapper import *

import options.option_vq as option_vq
from dataset import dataset_PC, dataset_TM_eval
from options.get_eval_option import get_opt

from models.evaluator_wrapper import EvaluatorModelWrapper
import models.motion_dec as motion_dec
from models.pg_tokenizer import PoseGuidedTokenizer

import yaml
from datetime import datetime

def drop_out_residual(pdrop_res):
    return np.random.random() < pdrop_res

def gram(codebook):
    code_range = list(vq_to_range.items())[:-2]
    cat_centroid = []
    
    for idx, cat_range in code_range:
        end, start = cat_range

        group_vecs = codebook[start:end+1, :]
        
        mean_group_vec = torch.mean(group_vecs, dim=0)
        cat_centroid.append(mean_group_vec)

    cat_centroid = torch.stack(cat_centroid, dim=0)
    
    norms = torch.norm(cat_centroid, dim=1, keepdim=True)

    normalized_cat_centroid = cat_centroid / norms 

    gram_matrix = normalized_cat_centroid @ normalized_cat_centroid.T  # shape [K, K]
    return gram_matrix

def load_loss_wrappers(logger, cfg, external_params=None):

    wrappers = {}
    for item in cfg['loss_wrappers']:
        wrapper_type = item['type']
        name = item.get('name', wrapper_type)
        params = item.get('params', {})

        cls = WRAPPER_CLASS_MAP.get(wrapper_type)
        if cls is None:
            raise ValueError(f"Unknown wrapper type: {wrapper_type}")
        else:
            logger.info(f'loaded loss type: {name}')
            # logger.info(f'params: {params}')

        wrappers[name] = cls(**params)

    return wrappers

def flatten_and_sum_losses(losses, is_use_in_loss):
    total = 0.0
    for loss, use in zip(losses, is_use_in_loss):
        if not use:
            continue
        total += loss if torch.is_tensor(loss) else float(loss)
    return total

def reset(met_dict):
    for name, _ in met_dict.items():
        met_dict[name] = 0.
    return met_dict

from utils.loss_wrapper import (
    ReConsLossWrapper,
    OrthogonalLossWrapper,
)

WRAPPER_CLASS_MAP = {
    'ReConsLossWrapper': ReConsLossWrapper,
    'OrthogonalLossWrapper': OrthogonalLossWrapper,
}

args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.out_dir = os.path.join(args.out_dir, date_time)

os.makedirs(args.out_dir, exist_ok = True)

json_path = os.path.join(args.out_dir, 'arguments.yaml')

with open(args.loss_cfg_path, 'r') as f:
    loss_cfg = yaml.safe_load(f)

with open(json_path, 'w') as f:
    dict_args = vars(args)
    dict_args['loss_cfg'] = loss_cfg
    json.dump(dict_args, f, indent=2)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)

val_shuffle = args.val_shuffle

logger.info(f"val_shuffle:{val_shuffle}")
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
logger.info(f"## args.use_full_sequence:{args.use_full_sequence} ##")
logger.info(f"## not args.disable_align_root:{not args.disable_align_root} ##")
logger.info(f"## args.use_rvq:{args.use_rvq} ##")
logger.info(f"## args.params_soft_ent_loss:{args.params_soft_ent_loss} ##")
logger.info(f"## args.rvq_init_method:{args.rvq_init_method} ##")

if args.use_word_only:
    w_vectorizer = WordVectorizer_only_text_token('./glove', 'our_vab')
else:
    w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit': 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    args.max_motion_len = 196
else:
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22
    args.max_motion_len = 196

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Dataloader ---- #####
train_loader = dataset_PC.DATALoader(args.dataname,
                                    args.batch_size,
                                    window_size=args.window_size,
                                    unit_length=2**args.down_t,
                                    num_workers=args.num_workers,
                                    use_full_sequence=args.use_full_sequence,
                                    meta_dir=args.meta_dir,
                                    codes_folder_name=args.codes_folder_name,
                                    )

train_loader_iter = dataset_PC.cycle(train_loader)

val_loader = dataset_TM_eval.DATALoader(args.dataname, 
                                        False,
                                        32,
                                        w_vectorizer,
                                        codebook_size=args.nb_code,
                                        val_shuffle=val_shuffle,
                                        unit_length=2**args.down_t,
                                        num_workers=args.num_workers,
                                        use_keywords=args.use_keywords,
                                        use_word_only=args.use_word_only,
                                        meta_dir=args.meta_dir,
                                        codes_folder_name=args.codes_folder_name,
                                        )


##### ---- Network ---- #####
net = PoseGuidedTokenizer(
                    args, 
                    args.nb_code,                      # nb_code
                    args.code_dim,                    # code_dim
                    args.output_emb_width,            # output_emb_width
                    args.down_t,                      # down_t
                    args.stride_t,                    # stride_t
                    args.width,                       # width
                    args.depth,                       # depth
                    args.dilation_growth_rate,        # dilation_growth_rate
                    args.vq_act,                      # activation
                    args.vq_norm,                     # norm
                    num_quantizers=args.rvq_num_quantizers,
                    shared_codebook=args.rvq_shared_codebook,
                    quantize_dropout_prob=args.rvq_quantize_dropout_prob,
                    quantize_dropout_cutoff_index=args.rvq_quantize_dropout_cutoff_index,
                    rvq_nb_code=args.rvq_nb_code,
                    mu=args.rvq_mu,
                    residual_ratio=args.rvq_residual_ratio,
                    vq_loss_beta=args.rvq_vq_loss_beta,
                    quantizer_type=args.rvq_quantizer_type,
                    params_soft_ent_loss=args.params_soft_ent_loss,
                    use_ema=(not args.unuse_ema),
                    init_method=args.rvq_init_method
                    )

if args.resume_pth: 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)

    if 'nb_iter' in ckpt:
        loaded_nb_iter = ckpt['nb_iter']
        logger.info(f"Resumed at iteration {loaded_nb_iter}")
    else:
        loaded_nb_iter = 1  
        logger.warning("nb_iter not found in checkpoint, starting from 1")
else:
    loaded_nb_iter = 1  
    logger.warning("nb_iter not found in checkpoint, starting from 1")
    
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

wrappers = load_loss_wrappers(logger, loss_cfg)
eval_loss_wrapper = load_loss_wrappers(logger, loss_cfg)

##### ------ warm-up ------- #####

avg_loss_total = 0.
avg_rvq_commit = 0.
avg_perplexity = 0.

avg_metrics_dict = {
    "Loss_Total":0.,
}

avg_residual_metric_dict = {
    "RVQ_Commit_Loss":0.,
    "Perplexity":0.,
    "Ent_loss_sub_1_samp_loss":0.,
    "Ent_loss_sub_2_avg_loss":0.,
    "Ent_loss_total":0.,
    "N_samples_residual_trained": 0
}


for nb_iter in range(loaded_nb_iter, args.warm_up_iter):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    gt_motion, code_indices = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float()
        
    p_drop_res = drop_out_residual(args.pdrop_res)
    pred_motion, codebook, output = net(code_indices.cuda().float(), gt_motion.cuda().float(), detach_p_latent=args.detach_p_latent, drop_out_residual_quantization=p_drop_res)
    
    if not p_drop_res:
        perplexity = output['perplexity']
        rvq_commit_loss = output['vq_loss']

        sub_samp_ent_loss = output['all_ent_sub_samp_loss']
        sub_batch_ent_loss = output['all_ent_sub_avg_loss']
        ent_loss = output['all_ent_loss']
    
    writer.add_scalar(f'./Warmup/Drop_out_residual_quantization', float(p_drop_res), nb_iter)
    logger.info(f"Warmup. Iter {nb_iter} : Drop_out_residual_quantization {float(p_drop_res)}")

    loss_dict = {}
    is_use_in_loss = []

    for name, wrapper in wrappers.items():
        loss_name = str(wrapper)
        use_in_loss = wrapper.is_use_in_loss()

        loss_ = wrapper.update(pred_motion, gt_motion)

        loss_dict.update(loss_) 

        if use_in_loss:
            is_use_in_loss += [True for _ in range(len(list(loss_.values())))]
        else:
            is_use_in_loss += [False for _ in range(len(list(loss_.values())))]

    loss_list = list(loss_dict.values())

    loss = flatten_and_sum_losses(loss_list, is_use_in_loss)

    if args.use_rvq and not p_drop_res:
        loss += args.rvq_commit * rvq_commit_loss
        loss += args.params_soft_ent_loss * ent_loss

        avg_residual_metric_dict['RVQ_Commit_Loss'] += args.rvq_commit * rvq_commit_loss.item()
        avg_residual_metric_dict['Perplexity'] += perplexity.item() if args.use_rvq and args.rvq_name == 'rptc' else 0.
        avg_residual_metric_dict['Ent_loss_sub_1_samp_loss'] += args.params_soft_ent_loss * sub_samp_ent_loss.item()
        avg_residual_metric_dict['Ent_loss_sub_2_avg_loss'] += args.params_soft_ent_loss * sub_batch_ent_loss.item()
        avg_residual_metric_dict['Ent_loss_total'] += args.params_soft_ent_loss * ent_loss.item()
        avg_residual_metric_dict['N_samples_residual_trained'] += 1

    avg_metrics_dict['Loss_Total'] += loss.item()
    
    ## update ##
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ## record ##
    if nb_iter % args.print_iter == 0:

        for met_name, met_val in avg_metrics_dict.items():
            print_value = met_val / args.print_iter

            writer.add_scalar(f'./Warmup/{met_name}', print_value, nb_iter)
            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t {met_name}.  {print_value:.5f}")
        
        for met_name, met_val in avg_residual_metric_dict.items():

            if met_name == 'N_samples_residual_trained':
                print_value = met_val
            else:
                print_value = met_val / avg_residual_metric_dict['N_samples_residual_trained'] if avg_residual_metric_dict['N_samples_residual_trained'] > 0 else 0.    

            writer.add_scalar(f'./Warmup/{met_name}', print_value, nb_iter)
            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t {met_name}.  {print_value:.5f}")
    
        for name, wrapper in wrappers.items():
            avg_loss_dict = wrapper.state()
            loss_name = str(wrapper)

            centroid_gram_mat = gram(codebook)
            tensorboard_add_image_pw_sim(writer=writer, codebook=codebook, tag="./Image/Codebook_pairwise_similarity(warm_up)", nb_iter=nb_iter)
            tensorboard_add_image_pw_sim(writer=writer, codebook=centroid_gram_mat, tag="./Image/Code_group_centroid_pairwise_simiarity(warm_up)", nb_iter=nb_iter)
            tensorboard_add_image_tsne(writer=writer, codebook=codebook, tag="./Image/Codebook_tsne(warm_up)", nb_iter=nb_iter, title="T-sne of Pose Codebook (Warmup)", mode=args.codes_folder_name)

            for name, avg_loss in avg_loss_dict.items():
                avg_loss /= args.print_iter
                writer.add_scalar(f'./Warmup/{name}', avg_loss, nb_iter)
                logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t {name}.  {avg_loss:.5f}")

            wrapper.reset()
            avg_loss_dict = wrapper.state()

        avg_metrics_dict = reset(avg_metrics_dict)
        avg_residual_metric_dict = reset(avg_residual_metric_dict)


##### ---- Training ---- #####

for name, wrapper in wrappers.items():
    avg_loss_dict = wrapper.reset()

best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, writer, logger = eval_trans.evaluation_dec(args, args.out_dir, val_loader, net, logger, writer, 0, use_rvq=args.use_rvq, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_mpjpe=0, best_pampjpe=0, best_accel=0, eval_wrapper=eval_wrapper, unit_length=2**args.down_t, max_motion_len=args.max_motion_len, split='Validation', use_aggregator=use_aggregator, num_joints=args.nb_joints, align_root=(not args.disable_align_root), drop_out_residual_quantization=False)
unit_length = 2**args.down_t

avg_metrics_dict = reset(avg_metrics_dict)
avg_residual_metric_dict = reset(avg_residual_metric_dict)

best_val_loss = 999
for nb_iter in range(1, args.total_iter + 1):
    
    # predict
    gt_motion, code_indices = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float() # bs, nb_joints, joints_dim, seq_len
    

    if args.use_rvq:
        if args.rvq_name == 'vanilla':
            pred_motion, codebook, rvq_commit_loss = net(code_indices.cuda().float())
        elif args.rvq_name == 'rptc':
            p_drop_res = drop_out_residual(args.pdrop_res)
            pred_motion, codebook, output, proj_rel_pos = net(code_indices.cuda().float(), gt_motion.cuda().float(),  detach_p_latent=args.detach_p_latent, drop_out_residual_quantization=p_drop_res)
            
            if not p_drop_res:
                perplexity = output['perplexity']
                rvq_commit_loss = output['vq_loss']
                
                sub_samp_ent_loss = output['all_ent_sub_samp_loss']
                sub_batch_ent_loss = output['all_ent_sub_avg_loss']
                ent_loss = output['all_ent_loss']
    else:
        pred_motion, codebook = net(code_indices.cuda().float())

    loss_dict = {}
    is_use_in_loss = []

    for name, wrapper in wrappers.items():
        loss_name = str(wrapper)
        use_in_loss = wrapper.is_use_in_loss()
        loss_ = wrapper.update(pred_motion, gt_motion)
        loss_dict.update(loss_)

        if use_in_loss:
            is_use_in_loss += [True for _ in range(len(list(loss_.values())))]
        else:
            is_use_in_loss += [False for _ in range(len(list(loss_.values())))]
    
    loss_list = list(loss_dict.values())
    loss = flatten_and_sum_losses(loss_list, is_use_in_loss)

    if args.use_rvq and not p_drop_res:
        loss += args.rvq_commit * rvq_commit_loss
        loss += args.params_soft_ent_loss * ent_loss

        avg_residual_metric_dict['RVQ_Commit_Loss'] += args.rvq_commit * rvq_commit_loss.item()
        avg_residual_metric_dict['Perplexity'] += perplexity.item()
        avg_residual_metric_dict['Ent_loss_sub_1_samp_loss'] += args.params_soft_ent_loss * sub_samp_ent_loss.item()
        avg_residual_metric_dict['Ent_loss_sub_2_avg_loss'] += args.params_soft_ent_loss * sub_batch_ent_loss.item()
        avg_residual_metric_dict['Ent_loss_total'] += args.params_soft_ent_loss * ent_loss.item()
        avg_residual_metric_dict['N_samples_residual_trained'] += 1

    avg_metrics_dict['Loss_Total'] += loss.item()
        
    
    # update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # record
    if nb_iter % args.print_iter ==  0 :

        for met_name, met_val in avg_metrics_dict.items():
            print_value = met_val / args.print_iter
            
            writer.add_scalar(f'./Train/{met_name}', print_value, nb_iter)
            logger.info(f"Train. Iter {nb_iter} : \t {met_name} .  {print_value:.5f}")

        for met_name, met_val in avg_residual_metric_dict.items():

            if met_name == 'N_samples_residual_trained':
                print_value = met_val
            else:
                print_value = met_val / avg_residual_metric_dict['N_samples_residual_trained'] if avg_residual_metric_dict['N_samples_residual_trained'] > 0 else 0.    

            writer.add_scalar(f'./Train/{met_name}', print_value, nb_iter)
            logger.info(f"Train. Iter {nb_iter} :  \t {met_name}.  {print_value:.5f}")

        for name, wrapper in wrappers.items():
            avg_loss_dict = wrapper.state()
            weight_dict = wrapper.return_weights()
            is_use_in_loss = wrapper.is_use_in_loss()

            for name, avg_loss in avg_loss_dict.items():
                avg_loss /= args.print_iter

                if is_use_in_loss:
                    writer.add_scalar(f'./Train/{name}', avg_loss, nb_iter)
                    logger.info(f"Train. Iter {nb_iter} : \t {name}.  {avg_loss:.5f}")
                else:
                    writer.add_scalar(f'./Train/{name}(Implicit)', avg_loss, nb_iter)
                    logger.info(f"Train. Iter {nb_iter} : \t {name}(Implicit).  {avg_loss:.5f}")
                
            for name, weight in weight_dict.items():
                writer.add_scalar(f'./Params/{name}', weight, nb_iter)

        for name, wrapper in wrappers.items():
            avg_loss_dict = wrapper.reset()
        
        avg_metrics_dict = reset(avg_metrics_dict)
        avg_residual_metric_dict = reset(avg_residual_metric_dict)

    # Validation
    if nb_iter % args.eval_iter == 0:

        centroid_gram_mat = gram(codebook)
        tensorboard_add_image_pw_sim(writer=writer, codebook=codebook, tag="./Image/Codebook_pairwise_similarity", nb_iter=nb_iter)
        tensorboard_add_image_pw_sim(writer=writer, codebook=centroid_gram_mat, tag="./Image/Code_group_centroid_pairwise_simiarity", nb_iter=nb_iter)
        tensorboard_add_image_tsne(writer=writer, codebook=codebook, tag="./Image/Codebook_tsne", nb_iter=nb_iter, title="T-sne of Pose Codebook", mode=args.codes_folder_name)

        def eval(args, net, val_loader, writer, logger, eval_loss_wrapper, pdrop_res=False, unit_length=4):
            
            if pdrop_res:
                avg_eval_metrics_dict = {
                    "Loss_Total":0.,
                }
            else:
                avg_eval_metrics_dict = {
                    "Loss_Total":0.,
                    "RVQ_Commit_Loss":0.,
                    "Perplexity":0.,
                    "Ent_loss_sub_1_samp_loss":0.,
                    "Ent_loss_sub_2_avg_loss":0.,
                    "Ent_loss_total":0.
                }

            eval_loss_log_list = []
            eval_loss_dict = {}
            is_use_in_loss = []
            avg_eval_loss_std = 0

            eval_case_id = 'No RVQ Validation' if pdrop_res else 'Validation'
            
            # evaluate validation loss
            with torch.no_grad():
                net.eval()

                batch_iter = 0.
                for batch in val_loader:
                    batch_iter += 1
                    word_embeddings, pos_one_hots, caption, sent_len, eval_gt_motion, m_length, _, _, eval_code_indices, _, _ = batch
                    eval_gt_motion = eval_gt_motion.cuda().float()

                    perplexity = torch.tensor(0., device=eval_gt_motion.device)
                    eval_rvq_commit_loss = torch.tensor(0., device=eval_gt_motion.device)
                    sub_samp_ent_loss = torch.tensor(0., device=eval_gt_motion.device)
                    sub_batch_ent_loss = torch.tensor(0., device=eval_gt_motion.device)
                    ent_loss = torch.tensor(0., device=eval_gt_motion.device)

                    if args.use_rvq:
                        eval_pred_motion, codebook_eval, eval_out = net(eval_code_indices[:,::unit_length,:].cuda().float(), eval_gt_motion, detach_p_latent=args.detach_p_latent, drop_out_residual_quantization=pdrop_res) 
                        
                        if not pdrop_res:
                            perplexity = eval_out['perplexity']
                            eval_rvq_commit_loss = eval_out['vq_loss']
                            sub_samp_ent_loss = eval_out['all_ent_sub_samp_loss']
                            sub_batch_ent_loss = eval_out['all_ent_sub_avg_loss']
                            ent_loss = eval_out['all_ent_loss']
                    
                    for name, wrapper in eval_loss_wrapper.items():
                        eval_loss_name = str(wrapper)
                        use_in_loss = wrapper.is_use_in_loss()
                        eval_loss_ = wrapper.update(eval_pred_motion, eval_gt_motion)
                        eval_loss_dict.update(eval_loss_)

                        if use_in_loss:
                            is_use_in_loss += [True for _ in range(len(list(eval_loss_.values())))]
                        else:
                            is_use_in_loss += [False for _ in range(len(list(eval_loss_.values())))]
                
                    eval_loss_list = list(eval_loss_dict.values())
                    eval_loss = flatten_and_sum_losses(eval_loss_list, is_use_in_loss)

                    if args.use_rvq and not pdrop_res:
                        eval_loss += args.rvq_commit * eval_rvq_commit_loss
                        eval_loss += args.params_soft_ent_loss * ent_loss

                        avg_eval_metrics_dict['RVQ_Commit_Loss'] += args.rvq_commit * eval_rvq_commit_loss.item()
                        avg_eval_metrics_dict['Perplexity'] += perplexity.item()
                        avg_eval_metrics_dict['Ent_loss_sub_1_samp_loss'] += args.params_soft_ent_loss * sub_samp_ent_loss.item()
                        avg_eval_metrics_dict['Ent_loss_sub_2_avg_loss'] += args.params_soft_ent_loss * sub_batch_ent_loss.item()
                        avg_eval_metrics_dict['Ent_loss_total'] += args.params_soft_ent_loss * ent_loss.item()

                    avg_eval_metrics_dict['Loss_Total'] += eval_loss.item()

                writer.add_scalar(f'./{eval_case_id}/Loss_Standard_Deviation(Recons + Vel)', (avg_eval_loss_std/batch_iter), nb_iter)

                for met_name, met_val in avg_eval_metrics_dict.items():
                    print_value = met_val / batch_iter

                    writer.add_scalar(f'./{eval_case_id}/{met_name}', print_value, nb_iter)
                    logger.info(f"{eval_case_id}. Iter {nb_iter} : \t {met_name}.  {print_value:.5f}")

                for name, wrapper in eval_loss_wrapper.items():
                    avg_loss_dict = wrapper.state()
                    is_use_in_loss = wrapper.is_use_in_loss()

                    for name, eval_avg_loss in avg_loss_dict.items():
                        eval_avg_loss /= batch_iter

                        if is_use_in_loss:
                            writer.add_scalar(f'./{eval_case_id}/{name}', eval_avg_loss, nb_iter)
                            logger.info(f"{eval_case_id}. Iter {nb_iter} : \t {name}.  {eval_avg_loss:.5f}")
                        else:
                            writer.add_scalar(f'./{eval_case_id}/{name}(Implicit)', eval_avg_loss, nb_iter)
                            logger.info(f"{eval_case_id}. Iter {nb_iter} : \t {name}(Implicit).  {eval_avg_loss:.5f}")

                    wrapper.reset()

                batch_iter = 0
            
            avg_eval_metrics_dict = reset(avg_eval_metrics_dict)
            
            return eval_loss_log_list

        # w/o residual dropout
        eval_loss_log_list = eval(args, net, val_loader, writer, logger, eval_loss_wrapper, pdrop_res=True, unit_length=2**args.down_t)
        eval_trans.evaluation_dec(args, args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, use_rvq=args.use_rvq, eval_wrapper=eval_wrapper, unit_length=2**args.down_t, max_motion_len=args.max_motion_len, eval_loss_list=None, num_joints=args.nb_joints, align_root=(not args.disable_align_root), split='No RVQ Validation', save=False, drop_out_residual_quantization=True)
        
        # w residual dropout 
        eval_loss_log_list = eval(args, net, val_loader, writer, logger, eval_loss_wrapper, pdrop_res=False, unit_length=2**args.down_t)
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, writer, logger = eval_trans.evaluation_dec(args, args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, use_rvq=args.use_rvq, eval_wrapper=eval_wrapper, unit_length=2**args.down_t, max_motion_len=args.max_motion_len, eval_loss_list=None, num_joints=args.nb_joints, align_root=(not args.disable_align_root), split='Validation', drop_out_residual_quantization=False)