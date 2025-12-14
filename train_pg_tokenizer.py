import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.motion_dec as motion_dec
import models_rvq.motion_dec_rvq as motion_rvq_dec
import models_rptc.motion_rptc as motion_rptc
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_PC, dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
from utils.word_vectorizer import WordVectorizer_only_text_token
from utils.misc import T2M_CONTACT_JOINTS
from utils.codebook import group_id_to_full_group_name
##### ---- WANDB ---- #####
import wandb
import random
import yaml
from utils.eval_trans import tensorboard_add_image_pw_sim, tensorboard_add_image_tsne # , tensorboard_and_wandb_add_tsne

from datetime import datetime
from utils.motion_process import recover_from_ric

### 2025_05_20 ###
from utils.trainUtil import update_lr_warm_up, gradient_based_dynamic_weighting
from utils.loss_wrapper import *

import yaml
# from utils.losses import *

from utils.loss_wrapper import (
    ReConsLossWrapper,
    ReconsJointFormatLossWrapper,
    ReconsJointWiseLossWrapper,
    ReconsJointGroupLossWrapper,
    GroupWiseL1LossWrapper,
    OrthogonalLossWrapper,
    GroupAwareContrastiveLossWrapper,
    DisentangleLossWrapper,
    DisentangleLossBatchWrapper,
    HTDLossWrapper
    
)

WRAPPER_CLASS_MAP = {
    'ReConsLossWrapper': ReConsLossWrapper,
    'ReconsJointFormatLossWrapper': ReconsJointFormatLossWrapper,
    'ReconsJointWiseLossWrapper': ReconsJointWiseLossWrapper,
    'ReconsJointGroupLossWrapper': ReconsJointGroupLossWrapper,
    'GroupWiseL1LossWrapper': GroupWiseL1LossWrapper,
    'OrthogonalLossWrapper': OrthogonalLossWrapper,
    'GroupAwareContrastiveLossWrapper': GroupAwareContrastiveLossWrapper,
    'DisentangleLossWrapper': DisentangleLossWrapper,
    'DisentangleLossBatchWrapper': DisentangleLossBatchWrapper,
    'HTDLossWrapper': HTDLossWrapper
}

args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.out_dir = os.path.join(args.out_dir, date_time)

os.makedirs(args.out_dir, exist_ok = True)

json_path = os.path.join(args.out_dir, 'arguments.yaml')

if args.cfg_cla_path:
    with open(args.cfg_cla_path, 'r') as f:
        args.cfg_cla = yaml.safe_load(f)
    use_aggregator = True
else:
    args.cfg_cla = None
    use_aggregator = False

with open(args.loss_cfg_path, 'r') as f:
    loss_cfg = yaml.safe_load(f)

with open(json_path, 'w') as f:
    dict_args = vars(args)
    dict_args['cfg_cla'] = args.cfg_cla
    dict_args['loss_cfg'] = loss_cfg
    json.dump(dict_args, f, indent=2)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)

val_shuffle = args.val_shuffle
logger.info(f"val_shuffle:{val_shuffle}")
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# run = wandb.init(project="decoder", config = vars(args), name=f'exp_{args.exp_name}')

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
logger.info(f"## args.use_full_sequence:{args.use_full_sequence} ##")
# logger.info(f"## args.use_dynamic_weight:{args.use_dynamic_weight} ##")
# logger.info(f"## args.norm_dynamic_weight:{args.norm_dynamic_weight} ##")
logger.info(f"## not args.disable_align_root:{not args.disable_align_root} ##")
logger.info(f"## args.randomize_window_size:{args.randomize_window_size} ##")
logger.info(f"## args.use_rvq:{args.use_rvq} ##")
logger.info(f"## args.params_soft_ent_loss:{args.params_soft_ent_loss} ##")
logger.info(f"## args.rvq_init_method:{args.rvq_init_method} ##")


##### ---- Dataloader ---- #####
train_loader = dataset_PC.DATALoader(args.dataname,
                                    args.batch_size,
                                    window_size=args.window_size,
                                    unit_length=2**args.down_t,
                                    num_workers=args.num_workers,
                                    use_full_sequence=args.use_full_sequence,
                                    meta_dir=args.meta_dir,
                                    codes_folder_name=args.codes_folder_name,
                                    target_del_semantic_code=args.target_del_semantic_code,
                                    delete_mode=args.delete_mode,
                                    randomize_window_size=args.randomize_window_size)

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
                                        target_del_semantic_code=args.target_del_semantic_code,
                                        delete_mode=args.delete_mode)

aggregate_mode = args.aggregate_mode


import numpy as np 
def drop_out_residual(pdrop_res):
    return np.random.random() < pdrop_res
from utils.codebook import vq_to_range

def gram(codebook):
    code_range = list(vq_to_range.items())[:-2]
    cat_centroid = []
    
    for idx, cat_range in code_range:
        end, start = cat_range

        group_vecs = codebook[start:end+1, :]
        
        mean_group_vec = torch.mean(group_vecs, dim=0)
        cat_centroid.append(mean_group_vec)

    # 
    cat_centroid = torch.stack(cat_centroid, dim=0)
    
    # norm
    norms = torch.norm(cat_centroid, dim=1, keepdim=True)

    # norm2
    normalized_cat_centroid = cat_centroid / norms 

    # 
    gram_matrix = normalized_cat_centroid @ normalized_cat_centroid.T  # shape [K, K]
    return gram_matrix

##### ---- Network ---- #####
if args.use_rvq:
    if args.rvq_name == 'vanilla':
        net = motion_rvq_dec.MotionDecRVQ(
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
                        args.cfg_cla,                     # cfg_cla
                        aggregate_mode=aggregate_mode,    # aggregate_mode
                        num_quantizers=args.rvq_num_quantizers,
                        shared_codebook=args.rvq_shared_codebook,
                        quantize_dropout_prob=args.rvq_quantize_dropout_prob,
                        quantize_dropout_cutoff_index=args.rvq_quantize_dropout_cutoff_index,
                        rvq_nb_code=args.rvq_nb_code,
                        mu=args.rvq_mu
                        )
    elif args.rvq_name == 'rptc':

        print(f"## NOTE: rvq_resi_beta:{args.rvq_resi_beta}")
        net = motion_rptc.PoseGuidedTokenizer(
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
                        args.cfg_cla,                     # cfg_cla
                        aggregate_mode=aggregate_mode,    # aggregate_mode
                        num_quantizers=args.rvq_num_quantizers,
                        shared_codebook=args.rvq_shared_codebook,
                        quantize_dropout_prob=args.rvq_quantize_dropout_prob,
                        quantize_dropout_cutoff_index=args.rvq_quantize_dropout_cutoff_index,
                        rvq_nb_code=args.rvq_nb_code,
                        mu=args.rvq_mu,
                        resi_beta=args.rvq_resi_beta,
                        vq_loss_beta=args.rvq_vq_loss_beta,
                        quantizer_type=args.rvq_quantizer_type,
                        params_soft_ent_loss=args.params_soft_ent_loss,
                        use_ema=(not args.unuse_ema),
                        init_method=args.rvq_init_method,
                        )

else:
    net = motion_dec.MotionDec(
                    args, 
                    args.nb_code,
                    args.code_dim,
                    args.output_emb_width,
                    args.down_t,
                    args.stride_t,
                    args.width,
                    args.depth,
                    args.dilation_growth_rate,
                    args.vq_act,
                    args.vq_norm,
                    args.cfg_cla,
                    aggregate_mode=aggregate_mode)
    
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

### Define Loss ###

# if args.loss_format == 'h3d':
#     Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)


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

wrappers = load_loss_wrappers(logger, loss_cfg)
eval_loss_wrapper = load_loss_wrappers(logger, loss_cfg)

##### ------ warm-up ------- #####

avg_loss_total = 0.
avg_rvq_commit = 0.
avg_perplexity = 0.

def reset(met_dict):
    for name, _ in met_dict.items():
        met_dict[name] = 0.
    return met_dict

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

avg_other_metric_dict = {

}

def callback(writer, nb_iter, phase='Warmup', **kwargs):
    codebook_grad = kwargs.get('w_codebook').grad

    for id, (end, start) in vq_to_range.items():
        if id < 70:
            grad_norm = torch.norm(codebook_grad[start:end+1, :], p=2, dim=1)
            group_name = group_id_to_full_group_name[id]
            writer.add_scalar(f'./{phase}/Codebook_grad_norms_group_{group_name}', grad_norm.mean().item(), nb_iter)
        else:
            break


for nb_iter in range(loaded_nb_iter, args.warm_up_iter):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    if args.randomize_window_size:
        gt_motion, code_indices, mask = next(train_loader_iter)
        mask = mask.cuda()
    else:
        gt_motion, code_indices = next(train_loader_iter)

    gt_motion = gt_motion.cuda().float()
    

    # predict
    if args.use_rvq:
        if args.rvq_name == 'vanilla':
            pred_motion, codebook, rvq_commit_loss = net(code_indices.cuda().float())
        elif args.rvq_name == 'rptc':
            # pred_motion, codebook, rvq_commit_loss, perplexity = net(gt_motion.cuda().float(), code_indices.cuda().float(), detach_p_latent=args.detach_p_latent)
            p_drop_res = drop_out_residual(args.pdrop_res)
            pred_motion, codebook, output, proj_rel_pos = net(code_indices.cuda().float(), gt_motion.cuda().float(), detach_p_latent=args.detach_p_latent, drop_out_residual_quantization=p_drop_res)
            
            if not p_drop_res:
                perplexity = output['perplexity']
                rvq_commit_loss = output['vq_loss']

                sub_samp_ent_loss = output['all_ent_sub_samp_loss']
                sub_batch_ent_loss = output['all_ent_sub_avg_loss']
                ent_loss = output['all_ent_loss']
            
            writer.add_scalar(f'./Warmup/Drop_out_residual_quantization', float(p_drop_res), nb_iter)
            logger.info(f"Warmup. Iter {nb_iter} : Drop_out_residual_quantization {float(p_drop_res)}")

    else:
        pred_motion, codebook = net(code_indices.cuda().float())

    loss_dict = {}
    is_use_in_loss = []
    for name, wrapper in wrappers.items():
        loss_name = str(wrapper)
        use_in_loss = wrapper.is_use_in_loss()

        if loss_name == 'recons_jgl':
            loss_ = wrapper.update(net, pred_motion, gt_motion)
        elif loss_name == 'orthogonal_loss' or loss_name == 'contrastive_loss':
            loss_ = wrapper.update(codebook)
        elif loss_name == 'recons_jfl':
            loss_ = wrapper.update(val_loader, pred_motion, gt_motion)
        elif loss_name == 'group_wise_loss':
            loss_ = wrapper.update(pred_motion[:, ::(2**args.down_t), :], code_indices)
        elif loss_name == 'disentangle_loss':
            loss_ = wrapper.update(codebook)
        elif loss_name == 'disentangle_loss_batch':
            loss_ = wrapper.update(code_indices, codebook)
        elif loss_name == 'htd_loss':
            loss_ = wrapper.update(pred_motion, proj_rel_pos, gt_motion)
        else:
            if args.randomize_window_size:
                loss_ = wrapper.update(pred_motion, gt_motion, mask)
            else:
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

    callback(writer, nb_iter, phase='Warmup', w_codebook=codebook)

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

            # if args.use_rvq:
            #     logger.info(f"Warmup. Iter {nb_iter} : lr {current_lr:.5f} \t RVQ_Commit_Loss .  {avg_rvq_commit:.5f}")
            #     logger.info(f"Warmup. Iter {nb_iter} : lr {current_lr:.5f} \t Perplexity .  {avg_perplexity:.5f}")
    

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
                # wandb.log({"Rec Loss": avg_loss})

            wrapper.reset()
            avg_loss_dict = wrapper.state()

        avg_metrics_dict = reset(avg_metrics_dict)
        avg_residual_metric_dict = reset(avg_residual_metric_dict)


def inv_transform(data, mean, std):
    return data * std + mean

##### ---- Training ---- #####

for name, wrapper in wrappers.items():
    avg_loss_dict = wrapper.reset()

best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, writer, logger = eval_trans.evaluation_dec(args, args.out_dir, val_loader, net, logger, writer, 0, use_rvq=args.use_rvq, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_mpjpe=0, best_pampjpe=0, best_accel=0, eval_wrapper=eval_wrapper, unit_length=2**args.down_t, max_motion_len=args.max_motion_len, split='Validation', use_aggregator=use_aggregator, num_joints=args.nb_joints, align_root=(not args.disable_align_root), drop_out_residual_quantization=False)
unit_length = 2**args.down_t

avg_metrics_dict = reset(avg_metrics_dict)
avg_residual_metric_dict = reset(avg_residual_metric_dict)

best_val_loss = 999
# start_iter = 
for nb_iter in range(1, args.total_iter + 1):
    
    # predict
    # gt_motion, code_indices = next(train_loader_iter)
    
    if args.randomize_window_size:
        gt_motion, code_indices, mask = next(train_loader_iter)
        mask = mask.cuda()
    else:
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
        
        if loss_name == 'recons_jgl':
            loss_ = wrapper.update(net, pred_motion, gt_motion)
        elif loss_name == 'orthogonal_loss' or loss_name == 'contrastive_loss':
            loss_ = wrapper.update(codebook)
        elif loss_name == 'recons_jfl':
            loss_ = wrapper.update(val_loader, pred_motion, gt_motion)
        elif loss_name == 'group_wise_loss':
            loss_ = wrapper.update(pred_motion[:, ::(2**args.down_t), :], code_indices)
        elif loss_name == 'disentangle_loss':
            loss_ = wrapper.update(codebook)
        elif loss_name == 'disentangle_loss_batch':
            loss_ = wrapper.update(code_indices, codebook)
        elif loss_name == 'htd_loss':
            loss_ = wrapper.update(pred_motion, proj_rel_pos, gt_motion)
        else:
            if args.randomize_window_size:
                loss_ = wrapper.update(pred_motion, gt_motion, mask)
            else:
                loss_ = wrapper.update(pred_motion, gt_motion)

        loss_dict.update(loss_)

        if use_in_loss:
            is_use_in_loss += [True for _ in range(len(list(loss_.values())))]
        else:
            is_use_in_loss += [False for _ in range(len(list(loss_.values())))]
    
    loss_list = list(loss_dict.values())
    # print(f"#DEBUG loss_list:{loss_list}")
    loss = flatten_and_sum_losses(loss_list, is_use_in_loss)

    # print(f"#DEBUG loss:{loss}")

    

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
        
    # avg_loss_total += loss.item()    
    
    # update
    optimizer.zero_grad()
    loss.backward()
    
    callback(writer, nb_iter, phase='Train', w_codebook=codebook)

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
                
                # wandb.log({f"{name}": avg_loss})
                

            for name, weight in weight_dict.items():
                writer.add_scalar(f'./Params/{name}', weight, nb_iter)

        for name, wrapper in wrappers.items():
            avg_loss_dict = wrapper.reset()
        
        avg_metrics_dict = reset(avg_metrics_dict)
        avg_residual_metric_dict = reset(avg_residual_metric_dict)

    if nb_iter % args.eval_iter == 0:

        centroid_gram_mat = gram(codebook)
        tensorboard_add_image_pw_sim(writer=writer, codebook=codebook, tag="./Image/Codebook_pairwise_similarity", nb_iter=nb_iter)
        tensorboard_add_image_pw_sim(writer=writer, codebook=centroid_gram_mat, tag="./Image/Code_group_centroid_pairwise_simiarity", nb_iter=nb_iter)
        tensorboard_add_image_tsne(writer=writer, codebook=codebook, tag="./Image/Codebook_tsne", nb_iter=nb_iter, title="T-sne of Pose Codebook", mode=args.codes_folder_name)

        def eval(args, net, val_loader, writer, logger, eval_loss_wrapper, pdrop_res=False, unit_length=4, use_aggregator=False):
            
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

                    #### padding mask ####
                    mask = (torch.arange(args.max_motion_len).unsqueeze(0) < m_length.unsqueeze(1)) # 
                    mask = mask.cuda().float() # bs, n_seq

                        
                    # make frame_wise attn mask
                    if use_aggregator:
                        fw_attn_mask = mask[:, ::unit_length]
                        fw_attn_mask = fw_attn_mask.cuda().float()
                    else:
                        fw_attn_mask = None

                    # predict

                    perplexity = torch.tensor(0., device=eval_gt_motion.device)
                    eval_rvq_commit_loss = torch.tensor(0., device=eval_gt_motion.device)
                    sub_samp_ent_loss = torch.tensor(0., device=eval_gt_motion.device)
                    sub_batch_ent_loss = torch.tensor(0., device=eval_gt_motion.device)
                    ent_loss = torch.tensor(0., device=eval_gt_motion.device)

                
                    if args.use_rvq:
                        if args.rvq_name == 'vanilla':
                            eval_pred_motion, codebook_eval, eval_rvq_commit_loss = net(eval_code_indices[:,::unit_length,:].cuda().float(), fw_mask=fw_attn_mask) 
                        elif args.rvq_name == 'rptc':
                            eval_pred_motion, codebook_eval, eval_out, eval_proj_rel_pos = net(eval_code_indices[:,::unit_length,:].cuda().float(), eval_gt_motion, detach_p_latent=args.detach_p_latent, drop_out_residual_quantization=pdrop_res) 
                            
                            if not pdrop_res:
                                perplexity = eval_out['perplexity']
                                eval_rvq_commit_loss = eval_out['vq_loss']
                                sub_samp_ent_loss = eval_out['all_ent_sub_samp_loss']
                                sub_batch_ent_loss = eval_out['all_ent_sub_avg_loss']
                                ent_loss = eval_out['all_ent_loss']
                    else:
                        eval_pred_motion, codebook_eval = net(eval_code_indices[:,::unit_length,:].cuda().float(), fw_mask=fw_attn_mask) 
                    
                    print_log = True
                    for name, wrapper in eval_loss_wrapper.items():
                        eval_loss_name = str(wrapper)
                        use_in_loss = wrapper.is_use_in_loss()

                        # print(f"#DEBUG: eval_pred_motion.shape {eval_pred_motion.shape}")
                        # print(f"#DEBUG: eval_gt_motion.shape {eval_gt_motion.shape}")
                        
                        if eval_loss_name == 'recons_jgl':
                            eval_loss_ = wrapper.update(net, eval_pred_motion, eval_gt_motion, mask, validation=True)
                        elif eval_loss_name in ['orthogonal_loss', 'contrastive_loss', 'disentangle_loss']:
                            # eval_loss_ = wrapper.update(codebook)
                            continue
                        elif eval_loss_name == 'recons_jfl':
                            eval_loss_ = wrapper.update(val_loader, eval_pred_motion, eval_gt_motion, mask)
                        elif eval_loss_name == 'group_wise_loss':
                            eval_loss_ = wrapper.update(eval_pred_motion[:, ::(2**args.down_t), :], eval_code_indices, mask)
                        elif eval_loss_name == 'disentangle_loss_batch':
                            eval_loss_ = wrapper.update(eval_code_indices, codebook_eval)
                        elif loss_name == 'htd_loss':
                            eval_loss_ = wrapper.update(eval_pred_motion, eval_proj_rel_pos, eval_gt_motion)
                        else:
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
                        # avg_eval_rvq_commit += args.rvq_commit * eval_rvq_commit_loss.item()

                        # avg_eval_perplexity += eval_perplexity.item() if args.use_rvq and args.rvq_name == 'rptc' else 0.
                        avg_eval_metrics_dict['RVQ_Commit_Loss'] += args.rvq_commit * eval_rvq_commit_loss.item()
                        avg_eval_metrics_dict['Perplexity'] += perplexity.item() if args.use_rvq and args.rvq_name == 'rptc' else 0.
                        
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
                    # print(f"# DEBUG: avg_loss_dict:{avg_loss_dict}")
                    is_use_in_loss = wrapper.is_use_in_loss()

                    if name in ['orthogonal_loss', 'contrastive_loss', 'disentangle_loss']:
                        continue

                    for name, eval_avg_loss in avg_loss_dict.items():
                        # print(f"# DEBUG: eval_avg_loss:{eval_avg_loss}")
                        eval_avg_loss /= batch_iter

                        # print(f"# DEBUG: eval_avg_loss/batch_iter:{eval_avg_loss}")

                        if is_use_in_loss:
                            # change 10. wrapper list에 대해서 logging
                            writer.add_scalar(f'./{eval_case_id}/{name}', eval_avg_loss, nb_iter)
                            logger.info(f"{eval_case_id}. Iter {nb_iter} : \t {name}.  {eval_avg_loss:.5f}")
                        else:
                            writer.add_scalar(f'./{eval_case_id}/{name}(Implicit)', eval_avg_loss, nb_iter)
                            logger.info(f"{eval_case_id}. Iter {nb_iter} : \t {name}(Implicit).  {eval_avg_loss:.5f}")
                        
                        # wandb.log({f"{eval_case_id} {name}": eval_avg_loss})

                    wrapper.reset()

                batch_iter = 0
            
            avg_eval_metrics_dict = reset(avg_eval_metrics_dict)
            
            return eval_loss_log_list

        if args.use_rvq:
            eval_loss_log_list = eval(args, net, val_loader, writer, logger, eval_loss_wrapper, pdrop_res=True, unit_length=2**args.down_t, use_aggregator=use_aggregator)
            eval_trans.evaluation_dec(args, args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, use_rvq=args.use_rvq, eval_wrapper=eval_wrapper, unit_length=2**args.down_t, max_motion_len=args.max_motion_len, use_aggregator=use_aggregator, eval_loss_list=None, num_joints=args.nb_joints, align_root=(not args.disable_align_root), split='No RVQ Validation', save=False, drop_out_residual_quantization=True)

        eval_loss_log_list = eval(args, net, val_loader, writer, logger, eval_loss_wrapper, pdrop_res=False, unit_length=2**args.down_t, use_aggregator=use_aggregator)
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, writer, logger = eval_trans.evaluation_dec(args, args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, use_rvq=args.use_rvq, eval_wrapper=eval_wrapper, unit_length=2**args.down_t, max_motion_len=args.max_motion_len, use_aggregator=use_aggregator, eval_loss_list=None, num_joints=args.nb_joints, align_root=(not args.disable_align_root), split='Validation', drop_out_residual_quantization=False)