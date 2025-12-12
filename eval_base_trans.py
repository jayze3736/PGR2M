
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip
import utils.utils_model as utils_model

from tqdm import tqdm # 
import argparse # 
import yaml # 

import options.option_transformer as option_trans
from utils.codebook import *
import models.motion_dec as motion_dec
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval
import models.t2m_trans as trans

from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import random # 

import warnings
warnings.filterwarnings('ignore')

from os.path import join as pjoin
import models_rptc.motion_rptc as motion_rptc

def fixseed(seed): # 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
logger.info(f"## args.mm_mode:{args.mm_mode} ##")
logger.info(f"## args.use_keywords:{args.use_keywords} ##")

##### ---- Dataset ---- #####
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, 
                                        w_vectorizer, 
                                        codebook_size=392,
                                        use_keywords=args.use_keywords) # 

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip_model.eval() # 평가 모드 진입: batch 정규화나 드롭아웃과 같은 출력에 있어서 영향을 주는 랜덤 요소를 제거
for p in clip_model.parameters():
    p.requires_grad = False 

def get_cfg_ckpt_path(folder_path): # 

    if folder_path is None:
        return None, None
    else:
        ckpt_path = pjoin(folder_path, 'net_best_fid.pth')
        config_path = pjoin(folder_path, 'arguments.yaml')
    
    return config_path, ckpt_path

# print(f"clip model device: {next(clip_model.parameters()).device}")

t2m_config, t2m_checkpoint_path = get_cfg_ckpt_path(args.t2m_checkpoint_folder)

if t2m_config is None or t2m_checkpoint_path is None:
    raise ValueError("T2M config or checkpoint path is None. Please provide a valid folder path.")

with open(t2m_config, 'r') as f:
    arg_dict = yaml.safe_load(f)
t2m_args = argparse.Namespace(**arg_dict)

#################################            
trans_net = trans.MotionTrans(num_vq=t2m_args.nb_code, 
                                embed_dim=t2m_args.embed_dim_gpt, 
                                clip_dim=t2m_args.clip_dim, 
                                block_size=t2m_args.block_size, 
                                num_layers=t2m_args.num_layers, 
                                n_head=t2m_args.n_head_gpt, 
                                drop_out_rate=t2m_args.drop_out_rate, 
                                fc_rate=t2m_args.ff_rate)

mode = args.cat_mode
print("# NOTE: category mode: ", mode)
print(f"Loading decoder config and checkpoint from {args.dec_checkpoint_folder}")
dec_config, dec_checkpoint_path = get_cfg_ckpt_path(args.dec_checkpoint_folder)

if dec_config is None or dec_checkpoint_path is None:
    raise ValueError("Decoder config or checkpoint path is None. Please provide a valid folder path.")

with open(dec_config, 'r') as f:
    arg_dict = yaml.safe_load(f)
dec_args = argparse.Namespace(**arg_dict)

if dec_args.use_rvq:
    net = motion_rptc.ResidualPoseTemporalComplementor(
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
                    dec_args.cfg_cla,                     # cfg_cla
                    aggregate_mode=None,    # aggregate_mode
                    num_quantizers=dec_args.rvq_num_quantizers,
                    shared_codebook=dec_args.rvq_shared_codebook,
                    quantize_dropout_prob=dec_args.rvq_quantize_dropout_prob,
                    quantize_dropout_cutoff_index=dec_args.rvq_quantize_dropout_cutoff_index,
                    rvq_nb_code=dec_args.rvq_nb_code,
                    mu=dec_args.rvq_mu,
                    resi_beta=dec_args.rvq_resi_beta,
                    vq_loss_beta=dec_args.rvq_vq_loss_beta,
                    quantizer_type=dec_args.rvq_quantizer_type,
                    params_soft_ent_loss=dec_args.params_soft_ent_loss,
                    use_ema=(not getattr(dec_args, 'unuse_ema', True)),
                    init_method=getattr(dec_args, 'rvq_init_method', 'enc')
                    )
    # codebook = None
    print ('loading checkpoint from {}'.format(dec_checkpoint_path))
    ckpt = torch.load(dec_checkpoint_path, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()
else:

    print(f"Loading baseline decoder")
    net = motion_dec.MotionDec(dec_args,
                        dec_args.nb_code,
                        dec_args.code_dim,
                        dec_args.output_emb_width,
                        dec_args.down_t,
                        dec_args.stride_t,
                        dec_args.width,
                        dec_args.depth,
                        dec_args.dilation_growth_rate)  

    print ('loading checkpoint from {}'.format(dec_checkpoint_path))
    ckpt = torch.load(dec_checkpoint_path, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()

print ('loading transformer checkpoint from {}'.format(t2m_checkpoint_path))
ckpt = torch.load(t2m_checkpoint_path, map_location='cpu')
trans_net.load_state_dict(ckpt['trans'], strict=True)

trans_net.eval()
trans_net.cuda()

# 시드는 모델 초기화때도 발생함. 따라서 모델 로드 이후에 시드를 고정하는게 맞음

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
repeat_time = 20

fixseed(args.seed) # seed는 모델 초기화 이후 고정시키도록 하여 배치에서 샘플을 가져올때 동일한 순서로 가져오도록 함

for i in tqdm(range(repeat_time)):
    print(f"{i}th evaluation")
    if args.eval_mode == 'fast':
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer_test_fast(args.out_dir, 
                                                                                                                                                        val_loader, 
                                                                                                                                                        net, 
                                                                                                                                                        trans_net, 
                                                                                                                                                        logger, 
                                                                                                                                                        writer, 
                                                                                                                                                        0, 
                                                                                                                                                        best_fid=1000, 
                                                                                                                                                        best_iter=0, 
                                                                                                                                                        best_div=100, 
                                                                                                                                                        best_top1=0, 
                                                                                                                                                        best_top2=0, 
                                                                                                                                                        best_top3=0, 
                                                                                                                                                        best_matching=100, 
                                                                                                                                                        best_multi=0, 
                                                                                                                                                        eval_wrapper=eval_wrapper, 
                                                                                                                                                        draw=False, 
                                                                                                                                                        savegif=False, 
                                                                                                                                                        save=False, 
                                                                                                                                                        savenpy=False, 
                                                                                                                                                        mm_mode=args.mm_mode,
                                                                                                                                                        text_encoding_method='baseline', 
                                                                                                                                                        use_rptc=dec_args.use_rvq,
                                                                                                                                                        text_encoder=clip_model, 
                                                                                                                                                        block_size=args.block_size,
                                                                                                                                                        max_token_len=49,
                                                                                                                                                        end_token_id=392,
                                                                                                                                                        use_keywords=args.use_keywords)
    else:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer_test(args.out_dir, 
                                                                                                                                                        val_loader, 
                                                                                                                                                        net, 
                                                                                                                                                        trans_net, 
                                                                                                                                                        logger, 
                                                                                                                                                        writer, 
                                                                                                                                                        0, 
                                                                                                                                                        best_fid=1000, 
                                                                                                                                                        best_iter=0, 
                                                                                                                                                        best_div=100, 
                                                                                                                                                        best_top1=0, 
                                                                                                                                                        best_top2=0, 
                                                                                                                                                        best_top3=0, 
                                                                                                                                                        best_matching=100, 
                                                                                                                                                        best_multi=0, 
                                                                                                                                                        eval_wrapper=eval_wrapper, 
                                                                                                                                                        draw=False, 
                                                                                                                                                        savegif=False, 
                                                                                                                                                        save=False, 
                                                                                                                                                        savenpy=False, 
                                                                                                                                                        mm_mode=args.mm_mode,
                                                                                                                                                        text_encoding_method='baseline', 
                                                                                                                                                        use_rptc=dec_args.use_rvq,
                                                                                                                                                        text_encoder=clip_model, 
                                                                                                                                                        block_size=args.block_size,
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
print('multi: ', sum(multi)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
multi = np.array(multi)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)
