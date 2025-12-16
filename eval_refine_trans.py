
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip
from tqdm import tqdm
import argparse
import yaml 
import options.option_residual_transformer as option_res_trans # 
from models.pg_tokenizer import PoseGuidedTokenizer
import utils.utils_model as utils_model
from utils.codebook import *
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
import models.pg_tokenizer as pg_tokenizer
import models.rt2m_trans as r_trans
import models.t2m_trans as t2m
from utils.file_utils.misc import get_model_parameters_info
import random

def fixseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from os.path import join as pjoin
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_res_trans.get_args_parser()
# torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
logger.info(f"## args.mm_mode:{args.mm_mode} ##")

##### ---- Dataset ---- #####
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer, codebook_size=392, use_keywords=args.use_keywords)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip_model.eval() 

for p in clip_model.parameters():
    p.requires_grad = False

# print(f"clip model device: {next(clip_model.parameters()).device}")

mode = args.cat_mode

print("# NOTE: category mode: ", mode)
########################## Load Base Transformer ##########################

def get_cfg_ckpt_path(folder_path, reference='net_best_fid.pth'):
    if folder_path is None:
        return None, None
    else:
        ckpt_path = pjoin(folder_path, reference)
        config_path = pjoin(folder_path, 'arguments.yaml')
    
    return config_path, ckpt_path

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
                    use_ema=(not dec_args.unuse_ema),
                    init_method=dec_args.rvq_init_method
                    )
    
print ('loading decoder checkpoint from {}'.format(dec_checkpoint_path))
ckpt = torch.load(dec_checkpoint_path, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

for p in net.parameters():
    p.requires_grad = False

print("## Notice: args.use_keywords:", args.use_keywords)

########################## Load RefineTrans ##########################

r_trans_config, r_trans_checkpoint_path = get_cfg_ckpt_path(args.residual_t2m_checkpoint_folder, reference=f"{args.eval_reference}.pth")

if r_trans_config is None or r_trans_checkpoint_path is None:
    raise ValueError("Residual Transformer config or checkpoint path is None. Please provide a valid folder path.")

with open(r_trans_config, 'r') as f:
    arg_dict = yaml.safe_load(f)

r_trans_args = argparse.Namespace(**arg_dict)

res_trans_net = r_trans.RefineTrans(num_vq=r_trans_args.nb_code, 
                                    num_rvq=dec_args.rvq_nb_code,
                                    embed_dim=r_trans_args.embed_dim_gpt, 
                                    clip_dim=r_trans_args.clip_dim, 
                                    block_size=r_trans_args.block_size, 
                                    num_layers=r_trans_args.num_layers, 
                                    n_head=r_trans_args.n_head_gpt, 
                                    drop_out_rate=r_trans_args.drop_out_rate, 
                                    fc_rate=r_trans_args.ff_rate,
                                    num_key=num_keywords,
                                    mode=mode,
                                    num_quantizer=dec_args.rvq_num_quantizers,
                                    share_weight=r_trans_args.share_weight)


print ('loading residual transformer checkpoint from {}'.format(r_trans_checkpoint_path))
r_trans_ckpt = torch.load(r_trans_checkpoint_path, map_location='cpu')
res_trans_net.load_state_dict(r_trans_ckpt['r_trans'], strict=True)

res_trans_net.cuda()
res_trans_net.eval()

print("## NOTICE: your model parameters")
m_params = get_model_parameters_info(res_trans_net)
print(m_params)

if 'nb_iter' in r_trans_ckpt:
    logger.info(f"Residual Transformer ckpt Loaded at iteration {r_trans_ckpt['nb_iter']}")

fixseed(args.seed)

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
repeat_time = 20

for i in tqdm(range(repeat_time)):
    print(f"{i}th evaluation")
    if args.eval_mode == 'fast':
        print("## Fast evaluation mode ##")
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_residual_transformer_test_fast(args.out_dir, 
                                                                                                                                                                val_loader, 
                                                                                                                                                                net, 
                                                                                                                                                                trans_net, 
                                                                                                                                                                res_trans_net, 
                                                                                                                                                                logger, 
                                                                                                                                                                writer, 
                                                                                                                                                                0,
                                                                                                                                                                clip_model=clip_model,
                                                                                                                                                                best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_multi=0, 
                                                                                                                                                                eval_wrapper=eval_wrapper, 
                                                                                                                                                                draw=False, 
                                                                                                                                                                savegif=False, 
                                                                                                                                                                save=False, 
                                                                                                                                                                savenpy=False, 
                                                                                                                                                                mm_mode=args.mm_mode,
                                                                                                                                                                use_keywords=args.use_keywords)
    else:
        print("## Normal evaluation mode ##")
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_residual_transformer_test(args.out_dir, 
                                                                                                                                                                val_loader, 
                                                                                                                                                                net, 
                                                                                                                                                                trans_net, 
                                                                                                                                                                res_trans_net, 
                                                                                                                                                                logger, 
                                                                                                                                                                writer, 
                                                                                                                                                                0,
                                                                                                                                                                clip_model=clip_model,
                                                                                                                                                                best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_multi=0, 
                                                                                                                                                                eval_wrapper=eval_wrapper, 
                                                                                                                                                                draw=False, 
                                                                                                                                                                savegif=False, 
                                                                                                                                                                save=False, 
                                                                                                                                                                savenpy=False, 
                                                                                                                                                                mm_mode=args.mm_mode,
                                                                                                                                                                use_keywords=args.use_keywords)
    
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
    multi.append(best_multi)

print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)

if args.mm_mode:
    print('multi: ', sum(multi)/repeat_time)
    multi = np.array(multi)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)

if args.mm_mode:
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
else:
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"

logger.info(msg_final)
