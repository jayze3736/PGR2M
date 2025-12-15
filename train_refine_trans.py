import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import torch
import numpy as np
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
from tqdm import tqdm
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import random
import math
from einops import repeat

########################## Base Transformer ##########################
import models.t2m_trans as t2m
from utils.codebook import *

########################## Residual Transformer ##########################
import options.option_residual_transformer as option_res_trans # 
from models.pg_tokenizer import PoseGuidedTokenizer
from dataset import dataset_RTM_train #
from dataset import dataset_RTM_eval # 
import models.rt2m_trans as r_trans

########################## Utils ##########################
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
import utils.losses as losses
from utils.masking_utils import mask_residual_codes, corrupt_residual_codes
from utils.file_utils.misc import get_model_parameters_info
from utils.misc import load_loss_wrappers
from utils.trainUtil import update_lr_warm_up, lr_lambda
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper

################ Loss Wrapper ################
from utils.rt2m_loss_wrapper import (
    CEWithLogitsLossWrapper
)

WRAPPER_CLASS_MAP = {
    'CEWithLogitsLossWrapper':CEWithLogitsLossWrapper,
}

########################## Functions ##########################

def cosine_q_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def q_schedule(bs, low, high, device='cpu'):
    # Replaced `uniform` with `torch.rand`
    noise = torch.rand((bs,), device=device)
    schedule = 1 - cosine_q_schedule(noise)
    return torch.round(schedule * (high - low)) + low

def create_pad_mask(max_len, m_tokens_len):
    pad_mask = torch.arange(max_len, device=m_tokens_len.device).expand(len(m_tokens_len), max_len) < m_tokens_len.unsqueeze(1)
    pad_mask = pad_mask.float()
    return pad_mask

def sampling(logits):
    output = logits.clone()
    for cat in vq_to_range:
        if cat < cat_num: 
            end, start = vq_to_range[cat]
            idx = torch.argmax(output[:,start:end+1], dim = -1)
            output[:,start:end+1] = 0
            output[torch.arange(output.shape[0]),start+idx] = 1
        else:
            end, start = vq_to_range[cat]
            output[:,start:end+1] = (torch.nn.functional.sigmoid(output[:,start:end+1]) > 0.5)

    return output

def flatten_and_sum_losses(losses, is_use_in_loss):
    total = 0.0
    for loss, use in zip(losses, is_use_in_loss):
        if not use:
            continue
        total += loss if torch.is_tensor(loss) else float(loss)
    return total

def calc_acc(gt, logits, ignore_index):
    gt_index = gt.argmax(dim=-1)  # (B, T)
    pred = logits.argmax(dim=-1)  
    mask = (gt_index != ignore_index)  
    correct = (pred == gt_index) & mask
    num_correct = correct.sum().item()
    num_valid = mask.sum().item()
    accuracy = num_correct / num_valid if num_valid > 0 else 0.0
    return accuracy

def get_cfg_ckpt_path(folder_path):
    if folder_path is None:
        return None, None
    else:
        ckpt_path = pjoin(folder_path, 'net_best_fid.pth')
        config_path = pjoin(folder_path, 'arguments.yaml')
    
    return config_path, ckpt_path

def mask_posecodes(gt_label, current_sampling_prob, pkeep):
    if np.random.random() >= current_sampling_prob:
        masked_out = True
        if pkeep == -1:
            proba = np.random.rand(1)[0] 
            mask = torch.bernoulli(proba * torch.ones(gt_label.shape,
                                                            device=gt_label.device))
        else: 
            mask = torch.bernoulli(pkeep * torch.ones(gt_label.shape,
                                                            device=gt_label.device))
        mask = mask.round().to(dtype=torch.int64)
        r_indices = torch.randn(gt_label.shape, device = gt_label.device)
        a_indices = mask*gt_label+(1-mask)*r_indices
        for cat in vq_to_range:
            if cat < cat_num:                
                end, start = vq_to_range[cat]
                idx = torch.argmax(a_indices[:,:,start:end+1], dim = -1, keepdim=True)
                a_indices[:,:,start:end+1] = 0
                a_indices.scatter_(-1, start+idx, 1)
    else:
        masked_out = False
        a_indices = gt_label

    return a_indices, masked_out

def cosine_schedule(t, T, tau_plus, tau_minus):
  cos_term = np.cos(np.pi * t / T) 
  result = (tau_plus - tau_minus) * (1 + cos_term) / 2 + tau_minus
  return result

def linear_schedule(t, T, tau_plus, tau_minus):
  T_safe = max(T, 1)
  linear_part = tau_plus - (tau_plus - tau_minus) * (t / T_safe)
  result = np.where(t < T, linear_part, tau_minus)
  return result

def fixseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_pose_code_emb_layer(args, trans_ckpt):
    weights_to_load = {}

    if args.load_pretrained_pose_code_emb:
        source_state_dict = trans_ckpt['trans']

        source_weight_key = 'trans_base.tok_emb.weight'
        source_bias_key = 'trans_base.tok_emb.bias'

        target_weight_key = 'proc_in.pose_tok_emb.weight'
        target_bias_key = 'proc_in.pose_tok_emb.bias'

        if source_weight_key in source_state_dict:
            weights_to_load[target_weight_key] = source_state_dict[source_weight_key]
            print(f"Found and mapped: {source_weight_key} -> {target_weight_key}")

        if source_bias_key in source_state_dict:
            weights_to_load[target_bias_key] = source_state_dict[source_bias_key]
            print(f"Found and mapped: {source_bias_key} -> {target_bias_key}")
        
    return weights_to_load


#############################################################################
args = option_res_trans.get_args_parser()
fixseed(args.seed)

mode = args.cat_mode

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

args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

########################## Evaluator & CLIP model ##########################

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

wrappers = load_loss_wrappers(logger, loss_cfg, WRAPPER_CLASS_MAP)
eval_loss_wrapper = load_loss_wrappers(logger, loss_cfg, WRAPPER_CLASS_MAP)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

########################## Load Base Transformer ##########################
t2m_config, t2m_checkpoint_path = get_cfg_ckpt_path(args.t2m_checkpoint_folder)

if t2m_config is None or t2m_checkpoint_path is None:
    raise ValueError("Decoder config or checkpoint path is None. Please provide a valid folder path.")

with open(t2m_config, 'r') as f:
    arg_dict = yaml.safe_load(f)
    
t2m_args = argparse.Namespace(**arg_dict)

if args.use_keywords:
    num_keywords = 11
else:
    num_keywords = 0

trans_net = t2m.BaseTrans(num_vq=t2m_args.nb_code, 
                                embed_dim=t2m_args.embed_dim_gpt, 
                                clip_dim=t2m_args.clip_dim, 
                                block_size=t2m_args.block_size, 
                                num_layers=t2m_args.num_layers, 
                                n_head=t2m_args.n_head_gpt, 
                                drop_out_rate=t2m_args.drop_out_rate, 
                                fc_rate=t2m_args.ff_rate)

print ('loading transformer checkpoint from {}'.format(t2m_checkpoint_path))
trans_ckpt = torch.load(t2m_checkpoint_path, map_location='cpu')
trans_net.load_state_dict(trans_ckpt['trans'], strict=True)

trans_net.cuda()
trans_net.eval()

for p in trans_net.parameters():
    p.requires_grad = False

if 'nb_iter' in trans_ckpt:
    logger.info(f"Transformer ckpt Loaded at iteration {trans_ckpt['nb_iter']}")

print("## NOTICE: your Base Transformer parameters")
m_params = get_model_parameters_info(trans_net)
print(m_params)

########################## Load Decoder ##########################
dec_config, dec_checkpoint_path = get_cfg_ckpt_path(args.dec_checkpoint_folder)

if dec_config is None or dec_checkpoint_path is None:
    raise ValueError("Decoder config or checkpoint path is None. Please provide a valid folder path.")

with open(dec_config, 'r') as f:
    arg_dict = yaml.safe_load(f)

dec_args = argparse.Namespace(**arg_dict)
net = PoseGuidedTokenizer(
                dec_args, 
                dec_args.nb_code,                      # nb_code
                dec_args.code_dim,                    # code_dim
                dec_args.output_emb_width,            # output_emb_width
                dec_args.down_t,                      # down_t
                dec_args.stride_t,                    # stride_t
                dec_args.width,                       # width
                dec_args.depth,                       # depth
                dec_args.dilation_growth_rate,        # dilation_growth_rate
                dec_args.vq_act,                      # activation
                dec_args.vq_norm,                     # norm
                num_quantizers=dec_args.rvq_num_quantizers,
                shared_codebook=dec_args.rvq_shared_codebook,
                quantize_dropout_prob=dec_args.rvq_quantize_dropout_prob,
                quantize_dropout_cutoff_index=dec_args.rvq_quantize_dropout_cutoff_index,
                rvq_nb_code=dec_args.rvq_nb_code,
                mu=dec_args.rvq_mu,
                residual_ratio=dec_args.rvq_residual_ratio,
                vq_loss_beta=dec_args.rvq_vq_loss_beta,
                quantizer_type=dec_args.rvq_quantizer_type,
                params_soft_ent_loss=dec_args.params_soft_ent_loss,
                use_ema= (not dec_args.unuse_ema),
                init_method=dec_args.rvq_init_method
                )
    
print ('loading decoder checkpoint from {}'.format(dec_checkpoint_path))
ckpt = torch.load(dec_checkpoint_path, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

for p in net.parameters():
    p.requires_grad = False

print("## NOTICE: your Decoder parameters")
m_params = get_model_parameters_info(net)
print(m_params)

########################## Data Loader ##########################
num_workers = args.num_workers
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')

val_loader = dataset_RTM_eval.DATALoader(args.dataname, 
                                        False, 
                                        32,
                                        w_vectorizer,
                                        codebook_size=dec_args.nb_code,
                                        pg_tokenizer=net,
                                        rvq_codebook_size=dec_args.rvq_nb_code,
                                        num_quantizer=dec_args.rvq_num_quantizers,
                                        num_workers=args.num_workers,
                                        codes_folder_name=args.codes_folder_name,
                                        cache_file_name=args.eval_cache_file_name,
                                        use_keywords=args.use_keywords)

train_loader = dataset_RTM_train.DATALoader(args.dataname, 
                                           args.batch_size, 
                                           dec_args.nb_code,
                                           pg_tokenizer=net,
                                           unit_length=2**args.down_t, 
                                           rvq_codebook_size=dec_args.rvq_nb_code,
                                           num_quantizer=dec_args.rvq_num_quantizers,
                                           num_workers=args.num_workers,
                                           codes_folder_name=args.codes_folder_name,
                                           cache_file_name=args.train_cache_file_name,
                                           use_keywords=args.use_keywords)

train_loader_iter = dataset_RTM_train.cycle(train_loader)

########################## Load RefineTrans ##########################

refine_trans = r_trans.RefineTrans(num_vq=args.nb_code, 
                                    num_rvq=dec_args.rvq_nb_code,
                                    embed_dim=args.embed_dim_gpt, 
                                    clip_dim=args.clip_dim, 
                                    block_size=args.block_size, 
                                    num_layers=args.num_layers, 
                                    n_head=args.n_head_gpt, 
                                    drop_out_rate=args.drop_out_rate, 
                                    fc_rate=args.ff_rate,
                                    num_key=num_keywords,
                                    mode=mode,
                                    num_quantizer=dec_args.rvq_num_quantizers,
                                    share_weight=args.share_weight)

weights_to_load = parse_pose_code_emb_layer(args, trans_ckpt)
if weights_to_load:
    incompatible_keys = refine_trans.load_state_dict(weights_to_load, strict=False)
    print("\nWeight loading process finished.")
    print(" - Successfully loaded keys:", incompatible_keys.missing_keys)
    print(" - Unexpected keys in source:", incompatible_keys.unexpected_keys)

    if args.freeze_pose_code_emb:
        print("Freezing the 'proc_in.pose_tok_emb' layer...")
        for param in refine_trans.proc_in.pose_tok_emb.parameters():
            param.requires_grad = False


########################## Optimizer & lr_scheduler ##########################
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, refine_trans, args.optimizer)
adjusted_milestones = [m - args.warm_up_iter for m in args.lr_scheduler if m > args.warm_up_iter]

if not args.start_warm_up:
    args.warm_up_iter = 0

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda step: lr_lambda(
        step,
        warm_up_iter=args.warm_up_iter,
        gamma=args.gamma,
        milestones=adjusted_milestones
    )
)

###############################################################################

if args.resume_trans is not None: 
    logger.info('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    refine_trans.load_state_dict(ckpt['trans'], strict=True)

    if 'nb_iter' in ckpt:
        loaded_nb_iter = ckpt['nb_iter']
        logger.info(f"Resumed at iteration {loaded_nb_iter}")
    else:
        loaded_nb_iter = 1 
        logger.warning("nb_iter not found in checkpoint, starting from 1")
else:
    loaded_nb_iter = 1 
    logger.warning("nb_iter not found in checkpoint, starting from 1")

refine_trans.train()
refine_trans.cuda()

print("## NOTICE: your model Residual Transformer parameters")
m_params = get_model_parameters_info(refine_trans)
print(m_params)
      
########################## Train Start ##########################
nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

best_fid=1000
best_iter=0
best_div=100
best_top1=0
best_top2=0
best_top3=0
best_matching=100
offset = num_keywords + 2

pad_index = dec_args.rvq_nb_code + 1

# Define the exponential decay schedule parameters
decay_factor = 0.99  # Exponential decay factor (adjust as needed)
min_sampling_prob = args.min_sampling_prob  # Minimum sampling probability
start_sampling_prob = 1.0  # Initial sampling probability (start with full teacher forcing)
current_sampling_prob = start_sampling_prob
thresh = torch.nn.Threshold(0.5,0)
avg_acc = 0.
cat_num = len(codes) # number of pose code categories

steps = np.arange(args.total_iter + 1)
scheduled_sampling_prob = linear_schedule(steps, args.warm_up_iter, start_sampling_prob, min_sampling_prob)

if args.schedule_masking_prob:
    scheduled_masking_prob = linear_schedule(steps, int(args.total_iter/2), 0.0, args.masking_prob)
else:
    scheduled_masking_prob = linear_schedule(steps, int(args.total_iter/2), args.masking_prob, args.masking_prob)

if args.schedule_pkeep:
    scheduled_pkeep = linear_schedule(steps, int(args.total_iter/2), 1.0, args.pkeep)
else:
    scheduled_pkeep = linear_schedule(steps, int(args.total_iter/2), args.pkeep, args.pkeep)

best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, val_acc, val_loss = eval_trans.evaluation_residual_transformer(args, args.out_dir, val_loader, net, trans_net, refine_trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, optimizer=optimizer, scheduler=scheduler, num_keywords=num_keywords)

if args.resume_trans is not None:
    start_iter = loaded_nb_iter
else:
    start_iter = 1

avg_acc = 0.
logger.info(f"## Model Main Training Start From:{start_iter} iter ##")
logger.info(f"args.start_warm_up: {args.start_warm_up}")

for nb_iter in range(start_iter, args.total_iter + 1):

    if nb_iter > 0: 
        if args.scheduled_sampling:
            current_sampling_prob = scheduled_sampling_prob[nb_iter] # 
        else:
            current_sampling_prob = max(min_sampling_prob, current_sampling_prob * decay_factor)

    # batch load
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len, keyword_embeddings, gt_motion, m_length, m_res_tokens_new = batch # bs x 1, bs x 55 x 512, keyword_embeddings
    m_tokens, m_tokens_len, gt_motion, m_res_tokens_new = m_tokens.cuda(), m_tokens_len.cuda(), gt_motion.cuda(), m_res_tokens_new.cuda()
    m_tokens = m_tokens.float()
    m_res_tokens_new = m_res_tokens_new.float()
    bs = m_tokens.shape[0]
    # target = m_tokens.cuda()   # (bs, t, code_num+2)

    # text embedding
    text = clip.tokenize(clip_text, truncate=True).cuda()
    feat_clip_text = clip_model.encode_text(text).float().unsqueeze(1)
    
    if args.use_keywords:
        feat_clip_text = torch.cat((feat_clip_text, keyword_embeddings.float().cuda()), dim=1)  # bs x 12 x 512

    pose_codes = m_tokens[:, :-2, :] # (bs, t-2, code_num)

    a_pose_codes, masked_out = mask_posecodes(pose_codes, current_sampling_prob, scheduled_pkeep[nb_iter])
    if not args.start_warm_up and nb_iter < args.warm_up_iter:
        writer.add_scalar(f'./Train/Sampling_Prob', current_sampling_prob, nb_iter)
        logger.info(f"## Notice: current_sampling_prob:{current_sampling_prob} at Iteration {nb_iter} ##")
        
        if masked_out:
            logger.info(f"## Notice: Masking is Applied. ##")
            writer.add_scalar(f'./Train/Masking', 1.0, nb_iter)
        else:
            logger.info(f"## Notice: Masking is not Applied. ##")
            writer.add_scalar(f'./Train/Masking', 0.0, nb_iter)

    active_q_layers = q_schedule(bs=bs, low=0, high=dec_args.rvq_num_quantizers-1, device=m_res_tokens_new.device) # (bs, )

    pad_mask = create_pad_mask(args.block_size, (m_tokens_len + offset).cuda()) # (bs, n) -> (bs, 52)
    q_non_pad_mask = repeat(pad_mask, 'b n -> b n q', q=dec_args.rvq_num_quantizers)
    mo_pad_mask = q_non_pad_mask[:, offset:, :] # (bs, 50)

    ##################################

    oh_vec = m_res_tokens_new[:, :-2, :, :] # bs, n-2, q, cb
    all_indices = torch.argmax(oh_vec, dim=-1) # bs, n-2, q
    
    tgt_indices = torch.where(mo_pad_mask.bool(), all_indices, pad_index) 
    target = tgt_indices[torch.arange(bs), :, active_q_layers.long()]  # (b, n)

    if args.mask_residual_code:
        all_indices = corrupt_residual_codes(all_indices=all_indices, rvq_nb_code=dec_args.rvq_nb_code, masking_prob=scheduled_masking_prob[nb_iter], pad_index=pad_index)
        all_indices = all_indices

    ##################################

    cls_pred = refine_trans(all_indices, a_pose_codes.float(), feat_clip_text, active_q_layers, mask=pad_mask)
    cls_pred = cls_pred.contiguous()

    loss_dict = {}
    is_use_in_loss = []

    pred = cls_pred[:, offset:, :] # num_keywords(11) + main_sentence(1) + q_id(1)
    tgt = target # 51

    for name, wrapper in wrappers.items():
        loss_name = str(wrapper)
        avg_loss_dict = wrapper.state()
        use_in_loss = wrapper.is_use_in_loss()

        n_loss_term = len(avg_loss_dict.values())

        if use_in_loss:
            is_use_in_loss += [True for _ in range(n_loss_term)]
        else:
            is_use_in_loss += [False for _ in range(n_loss_term)]
        
        if 'ce' in loss_name:
            loss_ = wrapper.update(pred, tgt, ignore_index=pad_index) # ignore_index=64
        
        for key, value in loss_.items():
            if key in loss_dict:
                loss_dict[key] += value  
            else:
                loss_dict[key] = value  

    loss_list = list(loss_dict.values())
    loss_cls = flatten_and_sum_losses(loss_list, is_use_in_loss)  # batch size로 나눠주기

    optimizer.zero_grad()
    loss_cls.backward()
    
    with torch.no_grad():
        if (nb_iter % args.print_iter == 0):

            if masked_out:
                writer.add_scalar(f'./Train/Masking', 1.0, nb_iter)
            else:
                writer.add_scalar(f'./Train/Masking', 0.0, nb_iter)

            writer.add_scalar(f'./Train/masking_prob', scheduled_masking_prob[nb_iter], nb_iter)
            writer.add_scalar(f'./Train/pkeep', scheduled_pkeep[nb_iter], nb_iter)

            
    optimizer.step()
    scheduler.step()

    # Logging
    if nb_iter % args.print_iter ==  0:
        writer.add_scalar(f'./Train/Total_loss', loss_cls.item(), nb_iter)
        logger.info(f"Train. Iter {nb_iter}: Total_loss. {loss_cls.item():.5f}")

        writer.add_scalar(f'./Train/Sampling_Prob', current_sampling_prob, nb_iter)
        logger.info(f"## Notice: current_sampling_prob:{current_sampling_prob} at Iteration {nb_iter} ##")

        writer.add_scalar("./Train/LR", optimizer.param_groups[0]['lr'], nb_iter)

        for name, wrapper in wrappers.items():
            loss_name = str(wrapper)
            avg_loss_dict = wrapper.state()
            loss_weight = wrapper.return_weights()

            for ls_name, avg_loss in avg_loss_dict.items():
                avg_loss /= args.print_iter

                writer.add_scalar(f'./Train/{ls_name}', avg_loss, nb_iter)
                logger.info(f"Train. Iter {nb_iter} : {ls_name}. {avg_loss:.5f}")
            
            for weight_name, val in loss_weight.items():
                writer.add_scalar(f'./Params_loss_weight/{weight_name}', val, nb_iter)

        avg_acc = 0.
        right_num = 0
        nb_sample_train = 0
        
        if not args.scheduled_sampling:
            current_sampling_prob = max(min_sampling_prob, current_sampling_prob * decay_factor)

        for name, wrapper in wrappers.items():
            wrapper.reset()
    
    with torch.no_grad():
        if nb_iter % args.eval_loss_iter ==  0:

            eval_right_num_by_cat = {}

            for id, name in group_id_to_full_group_name.items():
                eval_right_num_by_cat[name] = 0

            refine_trans.eval() # 

            eval_loss_dict = {} # 
            
            total_n_eval_sample = 0
            avg_eval_acc = 0.
            eval_right_num = 0
            nb_sample_eval = 0
            total_eval_iter = 0
            
            eval_is_use_in_loss = []

            for name, wrapper in eval_loss_wrapper.items():
                loss_name = str(wrapper)
                avg_loss_dict = wrapper.state()
                use_in_loss = wrapper.is_use_in_loss()

                n_loss_term = len(avg_loss_dict.values())

                if use_in_loss:
                    eval_is_use_in_loss += [True for _ in range(n_loss_term)]
                else:
                    eval_is_use_in_loss += [False for _ in range(n_loss_term)]


            for batch in val_loader:
                word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, indices, keyword_embeddings, m_tokens_new, m_res_tokens_new = batch
                m_tokens_len =  m_length /(2**args.down_t)
                eval_m_tokens, m_tokens_len, m_res_tokens_new = m_tokens_new.cuda(), m_tokens_len.cuda(), m_res_tokens_new.cuda()
                m_res_tokens_new = m_res_tokens_new.float()
                eval_m_tokens = eval_m_tokens.float()
                pose = pose.cuda()

                bs = eval_m_tokens.shape[0]

                # text embedding
                text = clip.tokenize(clip_text, truncate=True).cuda()
                feat_clip_text = clip_model.encode_text(text).float().unsqueeze(1)

                if args.use_keywords:
                    feat_clip_text = torch.cat((feat_clip_text, keyword_embeddings.float().cuda()), dim=1)  # bs x 12 x 512
        
                eval_pose_codes = eval_m_tokens[:, :-2, :] # (bs, t-1, code_num+2)    

                active_q_layers = q_schedule(bs=bs, low=0, high=dec_args.rvq_num_quantizers-1, device=m_res_tokens_new.device) # (bs, )
                pad_mask = create_pad_mask(args.block_size, (m_tokens_len + offset).cuda()) # (bs, n)
                q_non_pad_mask = repeat(pad_mask, 'b n -> b n q', q=dec_args.rvq_num_quantizers)
                mo_pad_mask = q_non_pad_mask[:, offset:, :]

                oh_vec = m_res_tokens_new[:, :-2, :, :] # bs, n-2, q, cb
                all_indices = torch.argmax(oh_vec, dim=-1) # bs, n-2, q
                
                tgt_indices = torch.where(mo_pad_mask.bool(), all_indices, pad_index)
                eval_target = tgt_indices[torch.arange(bs), :, active_q_layers.long()]  # (b, n) 

                if args.eval_masking:
                    eval_a_pose_codes, masked_out = mask_posecodes(eval_pose_codes, current_sampling_prob, scheduled_pkeep[nb_iter])
                    if args.mask_residual_code:
                        all_indices = corrupt_residual_codes(all_indices=all_indices, rvq_nb_code=dec_args.rvq_nb_code, masking_prob=scheduled_masking_prob[nb_iter], pad_index=pad_index)
                else:
                    eval_a_pose_codes = eval_pose_codes   

                #
                cls_pred = refine_trans(all_indices, eval_a_pose_codes, feat_clip_text, active_q_layers, mask=pad_mask)
                cls_pred = cls_pred.contiguous()
                
                eval_pred = cls_pred[:, offset:, :]
                eval_tgt = eval_target
            
                for name, wrapper in eval_loss_wrapper.items():
                    loss_name = str(wrapper)

                    if 'ce' in loss_name:
                        loss_ = wrapper.update(eval_pred, eval_tgt, ignore_index=pad_index) # ignore_index=64
                    
                    for key, value in loss_.items():
                        if key in eval_loss_dict:
                            eval_loss_dict[key] += value
                        else:
                            eval_loss_dict[key] = value  
                
                total_eval_iter += 1
                total_n_eval_sample += bs

            eval_loss_list = list(eval_loss_dict.values())
            eval_loss_cls = flatten_and_sum_losses(eval_loss_list, eval_is_use_in_loss)  # batch size로 나눠주기
            eval_loss_cls /= total_n_eval_sample
            
            writer.add_scalar(f'./Validation/Total_loss', eval_loss_cls.item(), nb_iter)
            logger.info(f"Validation. Iter {nb_iter}: Total_loss. {eval_loss_cls.item():.5f}")

            for name, wrapper in eval_loss_wrapper.items():
                loss_name = str(wrapper)
                avg_loss_dict = wrapper.state()
                loss_weight = wrapper.return_weights()

                for ls_name, avg_loss in avg_loss_dict.items():
                    avg_loss /= total_n_eval_sample 

                    writer.add_scalar(f'./Validation/{ls_name}', avg_loss, nb_iter)
                    logger.info(f"Validation. Iter {nb_iter} : {ls_name}. {avg_loss:.5f}")

            for name, wrapper in eval_loss_wrapper.items():
                wrapper.reset()
            
            refine_trans.train()
    
    if nb_iter % args.eval_iter ==  0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, val_acc, val_loss = eval_trans.evaluation_residual_transformer(args, args.out_dir, val_loader, net, trans_net, refine_trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, optimizer=optimizer, scheduler=scheduler, num_keywords=num_keywords)
            
    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            
