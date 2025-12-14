import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.motion_dec as motion_dec
import models.motion_enc as motion_enc
import models_rvq.motion_dec_rvq as motion_rvq_dec
import models_rptc.motion_rptc as motion_rptc
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
import yaml
import argparse
from os.path import join as pjoin
warnings.filterwarnings('ignore')
import numpy as np
import random

def fixseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
# torch.manual_seed(args.seed)
fixseed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

def get_cfg_ckpt_path(folder_path):
    if folder_path is None:
        return None, None
    else:
        ckpt_path = pjoin(folder_path, 'net_best_fid.pth')
        config_path = pjoin(folder_path, 'arguments.yaml')
    
    return config_path, ckpt_path

# load config of cla
if args.cfg_cla_path:
    with open(args.cfg_cla_path, 'r') as f:
        args.cfg_cla = yaml.safe_load(f)
    use_aggregator = True
else:
    args.cfg_cla = None
    use_aggregator = False


##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Dataloader ---- #####
args.nb_joints = 21 if args.dataname == 'kit' else 22

val_loader = dataset_TM_eval.DATALoader(args.dataname, 
                                        True, 
                                        32, 
                                        w_vectorizer,
                                        codebook_size=args.nb_code, 
                                        unit_length=2**args.down_t, 
                                        use_keywords=args.use_keywords,
                                        use_word_only=args.use_word_only,
                                        val_shuffle=True,
                                        codes_folder_name=args.codes_folder_name)

aggregate_mode = args.aggregate_mode


dec_config, dec_checkpoint_path = get_cfg_ckpt_path(args.dec_checkpoint_folder)

if dec_config is None or dec_checkpoint_path is None:
    raise ValueError("Decoder config or checkpoint path is None. Please provide a valid folder path.")

with open(dec_config, 'r') as f:
    arg_dict = yaml.safe_load(f)

dec_args = argparse.Namespace(**arg_dict)

print("dec_args.rvq_name: ", dec_args.rvq_name)
print("dec_args.use_rvq: ", dec_args.use_rvq)

##### ---- Network ---- #####
if dec_args.use_rvq:
    if dec_args.rvq_name == 'vanilla':
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
    elif dec_args.rvq_name == 'rptc':

        print(f"## NOTE: rvq_resi_beta:{dec_args.rvq_resi_beta}")
        net = motion_rptc.PoseGuidedTokenizer(
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
                    quantizer_type=getattr(dec_args, 'rvq_quantizer_type', 'hard'),
                    params_soft_ent_loss=0.0,
                    use_ema=(not getattr(dec_args, 'unuse_ema', False)),
                    init_method=getattr(dec_args, 'rvq_init_method', 'enc'),  # 'enc', 'xavier', 'uniform',
                    )
else:
    net = motion_dec.MotionDec(
                    dec_args, 
                    dec_args.nb_code,
                    dec_args.code_dim,
                    dec_args.output_emb_width,
                    dec_args.down_t,
                    dec_args.stride_t,
                    dec_args.width,
                    dec_args.depth,
                    dec_args.dilation_growth_rate,
                    dec_args.vq_act,
                    dec_args.vq_norm,
                    dec_args.cfg_cla, 
                    aggregate_mode=aggregate_mode)

if dec_checkpoint_path is not None: 
    logger.info('loading checkpoint from {}'.format(dec_checkpoint_path))
    ckpt = torch.load(dec_checkpoint_path, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)

net.eval() # 문제 없음(나중에 evaluation_dec 에서 바뀜)
net.cuda()

fid = []
div = []
top1 = []
top2 = []
top3 = []
mpjpe = []
pampjpe = []
accel = []
matching = []
repeat_time = 20

print("(not getattr(dec_args, 'unuse_ema', False))", (not getattr(dec_args, 'unuse_ema', False)))
print("args.force_drop_residual_quantization: ", args.force_drop_residual_quantization)

for i in range(repeat_time):
    # best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, writer, logger = eval_trans.evaluation_dec(args, args.out_dir, val_loader, net, logger, writer, 0, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, eval_wrapper=eval_wrapper, unit_length=2**args.down_t, max_motion_len=args.max_motion_len, use_aggregator=use_aggregator, eval_loss_list=eval_loss_log_list, num_joints=args.nb_joints, align_root=(not args.disable_align_root))
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, writer, logger = eval_trans.evaluation_dec(dec_args, args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_mpjpe=1000, best_pampjpe=1000, best_accel=1000, eval_wrapper=eval_wrapper, use_rvq=dec_args.use_rvq, draw=False, save=False, savenpy=(i==0), num_joints=args.nb_joints, split='Test', drop_out_residual_quantization=args.force_drop_residual_quantization, align_root=(not args.disable_align_root))
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    mpjpe.append(best_mpjpe)
    pampjpe.append(best_pampjpe)
    accel.append(best_accel)
    matching.append(best_matching)

print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('mpjpe: ', sum(mpjpe)/repeat_time)
print('pampjpe: ', sum(pampjpe)/repeat_time)
print('accel: ', sum(accel)/repeat_time)
print('matching: ', sum(matching)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
mpjpe = np.array(mpjpe)
pampjpe = np.array(pampjpe)
accel = np.array(accel)
matching = np.array(matching)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, MPJPE. {np.mean(mpjpe):.3f}, conf. {np.std(mpjpe)*1.96/np.sqrt(repeat_time):.3f}, PA-MPJPE. {np.mean(pampjpe):.3f}, conf. {np.std(pampjpe)*1.96/np.sqrt(repeat_time):.3f}, ACCEL. {np.mean(accel):.3f}, conf. {np.std(accel)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)
