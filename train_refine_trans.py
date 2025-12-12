import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
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

########################## Base Transformer 관련 ##########################
import models.t2m_trans as t2m
from utils.codebook import *

########################## Residual Transformer 관련 ##########################
import options.option_residual_transformer as option_res_trans # 
import models_rptc.motion_rptc as motion_dec
from dataset import dataset_TM_train_rtpc #
from dataset import dataset_TM_eval_rtpc # 
import models_rptc.rt2m_trans as r_trans

########################## Util 관련 ##########################
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
import utils.losses as losses
from utils.masking_utils import mask_residual_codes, corrupt_residual_codes
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
from utils.file_utils.misc import get_model_parameters_info
warnings.filterwarnings('ignore')
from datetime import datetime
from utils.misc import load_loss_wrappers
from utils.misc import compute_gradient_norm
from utils.trainUtil import update_lr_warm_up, lr_lambda
import random
import math
from einops import repeat

################ Loss Wrapper ################
from utils.rt2m_loss_wrapper import (
    ReConsT2MLossWrapper,
    CEWithLogitsLossWrapper
)
WRAPPER_CLASS_MAP = {
    'CEWithLogitsLossWrapper':CEWithLogitsLossWrapper,
    'ReConsT2MLossWrapper':ReConsT2MLossWrapper
}

########################## Util Functions ##########################

"""
loss.backward()
adaptive_clip_(model.parameters(), clip_factor=0.01)  # 보통 0.01~0.05 -> 
optimizer.step()
"""

def adaptive_clip_(params, clip_factor=0.01, eps=1e-3):
    for p in params:
        if p.grad is None: 
            continue
        p_norm = p.norm()
        g_norm = p.grad.norm()
        max_norm = (p_norm + eps) * clip_factor # 
        if g_norm > max_norm:
            p.grad.mul_(max_norm / (g_norm + 1e-6))

def cosine_q_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def q_schedule(bs, low, high, device='cpu'):
    # Replaced `uniform` with `torch.rand`
    noise = torch.rand((bs,), device=device)
    schedule = 1 - cosine_q_schedule(noise)
    return torch.round(schedule * (high - low)) + low

# def schedule_q(num_quantizer):
#     target_q_id = random.randint(1, num_quantizer)
#     return target_q_id

def create_pad_mask(max_len, m_tokens_len):
    pad_mask = torch.arange(max_len, device=m_tokens_len.device).expand(len(m_tokens_len), max_len) < m_tokens_len.unsqueeze(1) # block size가 최대길이라서, m_tokens_len보다 큰 부분은 mask 처리
    pad_mask = pad_mask.float()
    return pad_mask

def sampling(logits):
    output = logits.clone()
    for cat in vq_to_range:
        if cat < cat_num: # 이 부분 카테고리 수에 따라 달라짐
            end, start = vq_to_range[cat] # 주어진 category range를 나타냄
            # 그래서 예를 들어 start: 0, end: 18으로 주어진 범위는 L-arm에 대한 angle category 범주를 의미함
            idx = torch.argmax(output[:,start:end+1], dim = -1)
            # 실제로 0, 1로 정규화하는 부분
            output[:,start:end+1] = 0
            output[torch.arange(output.shape[0]),start+idx] = 1
        else:
            # Optional 코드로 추정됨
            # 사전에 허용한 이외의 카테고리인 경우 sigmoid함수를 거쳐서 
            # pad, end token에 대한 예측 확률
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
    
    # 예측값
    pred = logits.argmax(dim=-1)  # (B, T)
    # ignore_index=64인 위치는 accuracy 계산에서 제외
    mask = (gt_index != ignore_index)  # (B, T), True: 유효, False: ignore

    # 정답 비교 (mask 적용)
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

    # 처음에는 정답만, 후반에는 masking을 많이
    if np.random.random() >= current_sampling_prob:

        masked_out = True

        if pkeep == -1:
            proba = np.random.rand(1)[0] 
            mask = torch.bernoulli(proba * torch.ones(gt_label.shape,
                                                            device=gt_label.device))
        else: # 정해진 pkeep 확률로 masking할 토큰을 결정
            mask = torch.bernoulli(pkeep * torch.ones(gt_label.shape,
                                                            device=gt_label.device))

        # 혹시 모르니 정수가 되도록 반올림?
        mask = mask.round().to(dtype=torch.int64)

        # 0 ~ 1 사이의 랜덤한 숫자 샘플링(노이즈?)
        r_indices = torch.randn(gt_label.shape, device = gt_label.device)

        # 기존 target logit에 masking을 적용하고 masking이 적용되지 않은 logit에는 노이즈를 추가
        # a_indices를 추가하는 이유는 이후 trans_net에서 teacher forcing을 시키기 위해 정답 토큰을 넣을텐데, 이때 정답을 그대로 넣으면 실제 test랑 일치하지 않을테니 일부러 노이즈를 줘서
        # 
        a_indices = mask*gt_label+(1-mask)*r_indices

        # Mutual exclusivity
        for cat in vq_to_range:
            if cat < cat_num:                
                end, start = vq_to_range[cat]

                # 주어진 category 범위에서 logit이 가장 높은 target token을 1로 나머지는 0으로 만듦
                idx = torch.argmax(a_indices[:,:,start:end+1], dim = -1, keepdim=True)

                a_indices[:,:,start:end+1] = 0
                a_indices.scatter_(-1, start+idx, 1)
    else:
        masked_out = False
        a_indices = gt_label

    return a_indices, masked_out

def cosine_schedule(t, T, tau_plus, tau_minus):
  cos_term = np.cos(np.pi * t / T) # cosine 반주기 사용
  result = (tau_plus - tau_minus) * (1 + cos_term) / 2 + tau_minus
  return result

def linear_schedule(t, T, tau_plus, tau_minus):
  # 웜업 구간에 대한 선형 보간 계산
  # T가 0일 경우를 대비해 1로 처리
  T_safe = max(T, 1)
  linear_part = tau_plus - (tau_plus - tau_minus) * (t / T_safe)
  
  # t가 T보다 작으면 linear_part를, 크거나 같으면 tau_minus를 사용
  result = np.where(t < T, linear_part, tau_minus)
  
  return result

########################## Load Options & etc ##########################

def fixseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = option_res_trans.get_args_parser()
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
fixseed(args.seed)
mode = args.cat_mode

if mode == None:
    from utils.codebook import *
elif 'v2' in mode or 'v3' in mode:
    from utils.codebook_v2_v3 import *
elif 'v4' in mode:
    from utils.codebook_v4 import *
elif 'v5' in mode:
    from utils.codebook_v5 import *
elif 'v6' in mode:
    from utils.codebook_v6 import *
elif 'v7' in mode:
    from utils.codebook_v7 import *
else:
    from utils.codebook import *

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

args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
print(f"## Notice: args.enable_pos_emb_additional:{not args.disable_pos_emb_additional}")
print(f"## Notice: args.eval_masking:{args.eval_masking}")
print(f"## Notice: args.start_warm_up:{args.start_warm_up}")
print(f"## Notice: args.soft_label_folder_name:{args.soft_label_folder_name}")

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

trans_net = t2m.MotionTrans(num_vq=t2m_args.nb_code, 
                                embed_dim=t2m_args.embed_dim_gpt, 
                                clip_dim=t2m_args.clip_dim, 
                                block_size=t2m_args.block_size, 
                                num_layers=t2m_args.num_layers, 
                                n_head=t2m_args.n_head_gpt, 
                                drop_out_rate=t2m_args.drop_out_rate, 
                                fc_rate=t2m_args.ff_rate,
                                num_key=11,
                                mode=mode,
                                token_emb_name=t2m_args.token_emb_layer,
                                pos_emb_additional=getattr(
                                    t2m_args, "pos_emb_additional",
                                    not getattr(t2m_args, "disable_pos_emb_additional", True)  # 과거 플래그 호환
                                ),
                                pos_emb_rope=getattr(
                                    t2m_args, "pos_emb_rope",
                                    getattr(t2m_args, "use_rope_pos_emb", False)  # 과거 이름 호환
                                ),
                                pos_emb_offset=getattr(
                                    t2m_args, "pos_emb_offset",
                                    getattr(t2m_args, "pos_emb_rope_offset", 11)  # 둘 다 없으면 11
                                ),
                                mask_only_motion_tokens=getattr(t2m_args, "mask_only_motion_tokens", False),
                                init_prior=getattr(t2m_args, "init_prior", False),
                                codebook=None,
                                frozen=getattr(t2m_args, "frozen", getattr(t2m_args, "froze_codebook", True)),
                                graph_based_reasoning=getattr(
                                    t2m_args, "graph_based_reasoning",
                                    (getattr(t2m_args, "text_encoding_method", "").lower() == "graph_reasoning")
                                ),
                                block_attend_cond2cond=getattr(t2m_args, "block_attend_cond2cond", False)) # True -> APE, False -> ROPE

print ('loading transformer checkpoint from {}'.format(t2m_checkpoint_path))
trans_ckpt = torch.load(t2m_checkpoint_path, map_location='cpu')
trans_net.load_state_dict(trans_ckpt['trans'], strict=True)

# eval mode로 고정
trans_net.cuda()
trans_net.eval()

# trans_net의 파라메터는 업데이트 하지 않음
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
net = motion_dec.ResidualPoseTemporalComplementor(dec_args, 
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
                    quantizer_type=dec_args.rvq_quantizer_type,
                    params_soft_ent_loss=dec_args.params_soft_ent_loss,
                    use_ema=(not dec_args.unuse_ema) if dec_args.unuse_ema is not None else False,
                    init_method=getattr(dec_args, 'rvq_init_method', 'enc'),  # 'enc', 'xavier', 'uniform',
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

print("## Notice: args.use_keywords:", args.use_keywords)

########################## Data Loader ##########################
num_workers = args.num_workers
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval_rtpc.DATALoader(args.dataname, 
                                        False, 
                                        32,
                                        w_vectorizer,
                                        codebook_size=dec_args.nb_code,
                                        rtpc_net=net,
                                        rvq_codebook_size=dec_args.rvq_nb_code,
                                        num_quantizer=dec_args.rvq_num_quantizers,
                                        num_workers=args.num_workers,
                                        codes_folder_name=args.codes_folder_name,
                                        soft_label_folder_name=args.soft_label_folder_name,
                                        cache_file_name=args.eval_cache_file_name,
                                        use_keywords=args.use_keywords)

train_loader = dataset_TM_train_rtpc.DATALoader(args.dataname, 
                                           args.batch_size, 
                                           dec_args.nb_code,
                                           rtpc_net=net,
                                           unit_length=2**args.down_t, 
                                           rvq_codebook_size=dec_args.rvq_nb_code,
                                           num_quantizer=dec_args.rvq_num_quantizers,
                                           num_workers=args.num_workers,
                                           codes_folder_name=args.codes_folder_name,
                                           soft_label_folder_name=args.soft_label_folder_name,
                                           cache_file_name=args.train_cache_file_name,
                                           use_keywords=args.use_keywords)

train_loader_iter = dataset_TM_train_rtpc.cycle(train_loader)

########################## Load RTransformer ##########################

# ToDo: 점검 필요

res_trans_net = r_trans.RTransformer(num_vq=args.nb_code, 
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


##################### INIT #######################
print("## Notice: args.load_pretrained_pose_code_emb:", args.load_pretrained_pose_code_emb)
print("## Notice: args.freeze_pose_code_emb:", args.freeze_pose_code_emb)

if args.load_pretrained_pose_code_emb:
    source_state_dict = trans_ckpt['trans']

    source_weight_key = 'trans_base.tok_emb.weight'
    source_bias_key = 'trans_base.tok_emb.bias'

    target_weight_key = 'proc_in.pose_tok_emb.weight'
    target_bias_key = 'proc_in.pose_tok_emb.bias'

    weights_to_load = {}

    if source_weight_key in source_state_dict:
        weights_to_load[target_weight_key] = source_state_dict[source_weight_key]
        print(f"Found and mapped: {source_weight_key} -> {target_weight_key}")

    if source_bias_key in source_state_dict:
        weights_to_load[target_bias_key] = source_state_dict[source_bias_key]
        print(f"Found and mapped: {source_bias_key} -> {target_bias_key}")

    # 4. 생성된 딕셔너리가 비어있지 않은 경우에만 가중치를 로드합니다.
    if weights_to_load:
        # strict=False를 사용하여 지정된 가중치만 로드합니다.
        incompatible_keys = res_trans_net.load_state_dict(weights_to_load, strict=False)
        print("\nWeight loading process finished.")
        print(" - Successfully loaded keys:", incompatible_keys.missing_keys) # 로드된 키는 missing_keys 목록에서 제외됨
        print(" - Unexpected keys in source:", incompatible_keys.unexpected_keys)

        if args.freeze_pose_code_emb:
            print("Freezing the 'proc_in.pose_tok_emb' layer...")
            
            # 방법 1: 특정 모듈의 모든 파라미터 고정 (가장 일반적)
            for param in res_trans_net.proc_in.pose_tok_emb.parameters():
                param.requires_grad = False


########################## Optimizer & lr_scheduler ##########################
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, res_trans_net, args.optimizer)

adjusted_milestones = [m - args.warm_up_iter for m in args.lr_scheduler if m > args.warm_up_iter]

# def lr_lambda(current_step: int, gamma, ):
#     # Warm-up 구간
#     if current_step < args.warm_up_iter:
#         return float(current_step) / float(max(1, args.warm_up_iter))
#     # Warm-up 이후 MultiStepLR 동작 구간
#     return args.gamma ** len([m for m in adjusted_milestones if m <= current_step])

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma) # 지정한 args.lr_scheduler 스텝에서 gamma 배로 감소

if not args.start_warm_up:
    args.warm_up_iter = 0

print(f"arg.start_warm_up_iter:{args.start_warm_up}")

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
    res_trans_net.load_state_dict(ckpt['trans'], strict=True)

    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        # 3) 옵티마이저 state 텐서들을 디바이스로 이동
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(torch.device('cuda'))
    
    if 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
        logger.info("Scheduler state loaded from checkpoint.")

    if 'nb_iter' in ckpt:
        loaded_nb_iter = ckpt['nb_iter']
        logger.info(f"Resumed at iteration {loaded_nb_iter}")
    else:
        loaded_nb_iter = 1  # 혹은 초기값
        logger.warning("nb_iter not found in checkpoint, starting from 1")
else:
    loaded_nb_iter = 1  # 혹은 초기값
    logger.warning("nb_iter not found in checkpoint, starting from 1")

res_trans_net.train()
res_trans_net.cuda()

print("## NOTICE: your model Residual Transformer parameters")
m_params = get_model_parameters_info(res_trans_net)
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
offset = num_keywords + 2 # 2 = text caption + q id embedding

pad_index = dec_args.rvq_nb_code + 1

# Define the exponential decay schedule parameters
decay_factor = 0.99  # Exponential decay factor (adjust as needed)
min_sampling_prob = args.min_sampling_prob  # Minimum sampling probability
start_sampling_prob = 1.0  # Initial sampling probability (start with full teacher forcing)
current_sampling_prob = start_sampling_prob
thresh = torch.nn.Threshold(0.5,0)
avg_acc = 0.
cat_num = len(codes) # number of pose code categories

# warm up까지
steps = np.arange(args.total_iter + 1) # 전체 이터레이션에 대한 스케줄 생성
scheduled_sampling_prob = linear_schedule(steps, args.warm_up_iter, start_sampling_prob, min_sampling_prob)

if args.schedule_masking_prob:
    scheduled_masking_prob = linear_schedule(steps, int(args.total_iter/2), 0.0, args.masking_prob)
else:
    scheduled_masking_prob = linear_schedule(steps, int(args.total_iter/2), args.masking_prob, args.masking_prob)

if args.schedule_pkeep:
    scheduled_pkeep = linear_schedule(steps, int(args.total_iter/2), 1.0, args.pkeep)
else:
    scheduled_pkeep = linear_schedule(steps, int(args.total_iter/2), args.pkeep, args.pkeep)

# nb_iter_print_grad_early_step = 500

best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, val_acc, val_loss = eval_trans.evaluation_residual_transformer(args, args.out_dir, val_loader, net, trans_net, res_trans_net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, optimizer=optimizer, scheduler=scheduler, log_cat_right_num=args.log_cat_right_num, cat_mode=args.codes_folder_name, num_keywords=num_keywords)

if args.resume_trans is not None:
    start_iter = loaded_nb_iter
else:
    start_iter = 1

avg_acc = 0.
logger.info(f"## Model Main Training Start From:{start_iter} iter ##")
logger.info(f"args.start_warm_up: {args.start_warm_up}")
print("WARNING!! args.clip_grad:", args.clip_grad)

# nb_iter: number of iteration
for nb_iter in range(start_iter, args.total_iter + 1):

    if nb_iter > 0: 
        if args.scheduled_sampling:
            current_sampling_prob = scheduled_sampling_prob[nb_iter] # 
        else:
            current_sampling_prob = max(min_sampling_prob, current_sampling_prob * decay_factor)
    
    # if args.start_warm_up and nb_iter < args.warm_up_iter:
    #     optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr) #

    # batch load
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len, keyword_embeddings, gt_motion, m_length, m_res_tokens_new = batch # bs x 1, bs x 55 x 512, keyword_embeddings
    m_tokens, m_tokens_len, gt_motion, m_res_tokens_new = m_tokens.cuda(), m_tokens_len.cuda(), gt_motion.cuda(), m_res_tokens_new.cuda()
    m_tokens = m_tokens.float()
    m_res_tokens_new = m_res_tokens_new.float()
    bs = m_tokens.shape[0]
    # target = m_tokens.cuda()   # (bs, t, code_num+2) -> PAD, END 포함

    # text embedding
    text = clip.tokenize(clip_text, truncate=True).cuda()
    feat_clip_text = clip_model.encode_text(text).float().unsqueeze(1)
    
    if args.use_keywords:
        feat_clip_text = torch.cat((feat_clip_text, keyword_embeddings.float().cuda()), dim=1)  # bs x 12 x 512

    # PAD, END 토큰이 포함된 time step 제외(-2)(motion 길이가 최대인 경우에는 필수적으로 PAD가 들어가 있음)
    pose_codes = m_tokens[:, :-2, :] # (bs, t-2, code_num) -> PAD, END 제외

    a_pose_codes, masked_out = mask_posecodes(pose_codes, current_sampling_prob, scheduled_pkeep[nb_iter])

        # 초반까지는 주기 1로 출력
    if not args.start_warm_up and nb_iter < args.warm_up_iter:
        writer.add_scalar(f'./Train/Sampling_Prob', current_sampling_prob, nb_iter)
        logger.info(f"## Notice: current_sampling_prob:{current_sampling_prob} at Iteration {nb_iter} ##")
        
        if masked_out:
            logger.info(f"## Notice: Masking is Applied. ##")
            writer.add_scalar(f'./Train/Masking', 1.0, nb_iter)
        else:
            logger.info(f"## Notice: Masking is not Applied. ##")
            writer.add_scalar(f'./Train/Masking', 0.0, nb_iter)

    # text embedding과 기존 motion token을 입력으로 예측을 수행
    # teacher forcing을 위한 정답 logit(토큰) 입력, 하지만 이때 정답 토큰에는 일부 [MASK]가 적용되어있음(corrupt)

    active_q_layers = q_schedule(bs=bs, low=0, high=dec_args.rvq_num_quantizers-1, device=m_res_tokens_new.device) # (bs, )

    pad_mask = create_pad_mask(args.block_size, (m_tokens_len + offset).cuda()) # (bs, n) -> (bs, 52)
    q_non_pad_mask = repeat(pad_mask, 'b n -> b n q', q=dec_args.rvq_num_quantizers)
    mo_pad_mask = q_non_pad_mask[:, offset:, :] # (bs, 50)

    ##################################

    # if active_q_layers > 1:
    #     input_r_codes = m_res_tokens_new[:, :-2, :, :]  # 여기서는 autoregresive하지 않으니까 seq len에 대해서 slicing x
        
    #     if args.mask_residual_code:
    #         input_r_codes = corrupt_residual_codes(r_codes=input_r_codes, rvq_nb_code=dec_args.rvq_nb_code, masking_prob=scheduled_masking_prob[nb_iter], pad_index=pad_index)
    # else:
    #     input_r_codes = None
    
    # target = m_res_tokens_new[:, :-2, active_q_layers-1, :] # one-hot 형태, 하지만 loss 계산시에는 정수로 입력

    ## target 구성 과정
    oh_vec = m_res_tokens_new[:, :-2, :, :] # bs, n-2, q, cb
    all_indices = torch.argmax(oh_vec, dim=-1) # bs, n-2, q(정수형)
    
    tgt_indices = torch.where(mo_pad_mask.bool(), all_indices, pad_index) # PAD 토큰 채우기
    target = tgt_indices[torch.arange(bs), :, active_q_layers.long()]  # (b, n) -> 정답, loss 계산에만 사용

    if args.mask_residual_code:
        all_indices = corrupt_residual_codes(all_indices=all_indices, rvq_nb_code=dec_args.rvq_nb_code, masking_prob=scheduled_masking_prob[nb_iter], pad_index=pad_index)
        all_indices = all_indices

    ##################################

    cls_pred = res_trans_net(all_indices, a_pose_codes.float(), feat_clip_text, active_q_layers, mask=pad_mask)
    cls_pred = cls_pred.contiguous()

    loss_dict = {}
    is_use_in_loss = []

    pred = cls_pred[:, offset:, :] # num_keywords(11) + main_sentence(1) + q_id(1) 제거 (49)
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
        
        # 누적이 필요함
        for key, value in loss_.items():
            if key in loss_dict:
                loss_dict[key] += value  # 누적
            else:
                loss_dict[key] = value   # 새로 추가

    loss_list = list(loss_dict.values())

    # 이 부분 확인 필요
    loss_cls = flatten_and_sum_losses(loss_list, is_use_in_loss)  # batch size로 나눠주기
    # loss_cls /= bs

    optimizer.zero_grad()
    loss_cls.backward()

    if args.clip_grad:
        adaptive_clip_(res_trans_net.parameters(), clip_factor=0.01)
    
    with torch.no_grad():
        if (nb_iter % args.print_iter == 0):
            base_grad_norm = compute_gradient_norm(res_trans_net.trans_base, norm_type=2)
            # head_grad_norm = compute_gradient_norm(res_trans_net.trans_head, norm_type=2)

            # Logging
            writer.add_scalar(f'./Train/Grad_norm(Base)', base_grad_norm, nb_iter)
            logger.info(f"Train. Iter {nb_iter}: Grad_norm(Base). {base_grad_norm:.5f}")

            if masked_out:
                writer.add_scalar(f'./Train/Masking', 1.0, nb_iter)
            else:
                writer.add_scalar(f'./Train/Masking', 0.0, nb_iter)

            writer.add_scalar(f'./Train/masking_prob', scheduled_masking_prob[nb_iter], nb_iter)
            writer.add_scalar(f'./Train/pkeep', scheduled_pkeep[nb_iter], nb_iter)

            
    optimizer.step()
    # warm up을 하지 않았거나, nb_iter이 warm up 단계를 지난 경우에만 scheduler 진행
    # if (not args.start_warm_up) or (nb_iter >= args.warm_up_iter):
    scheduler.step()

    # Logging
    # nb_iter += 1
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
                # avg_loss /= bs # 앞에서는 sample별 loss를 계산하고 합산하기 때문에, 뒤에서 bs로 나눠줘야 함

                writer.add_scalar(f'./Train/{ls_name}', avg_loss, nb_iter)
                logger.info(f"Train. Iter {nb_iter} : {ls_name}. {avg_loss:.5f}")
            
            for weight_name, val in loss_weight.items():
                writer.add_scalar(f'./Params_loss_weight/{weight_name}', val, nb_iter)

        # wandb.log({"Train/Loss": avg_loss_cls, "Train/Accuracy":avg_acc})

        # 초기화 process
        avg_acc = 0.
        right_num = 0
        nb_sample_train = 0
        
        if not args.scheduled_sampling:
            current_sampling_prob = max(min_sampling_prob, current_sampling_prob * decay_factor)

        for name, wrapper in wrappers.items():
            wrapper.reset()
    
    # validation set에 대한 loss 검증
    with torch.no_grad():
        if nb_iter % args.eval_loss_iter ==  0:

            eval_right_num_by_cat = {}

            for id, name in group_id_to_full_group_name.items():
                eval_right_num_by_cat[name] = 0

            res_trans_net.eval() # 평가 모드 진입

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

                # batch data 불러오기
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
            
                # 일단 모델에 입력되는 것은 과거 시퀀스이기 때문에 :-1까지의 범위내로 슬라이싱을 해야함
                # 그리고 예측을 할때는 1: 부터 예측하니까
                eval_pose_codes = eval_m_tokens[:, :-2, :] # (bs, t-1, code_num+2)    

                # validation sample에 대하여 masking
                # eval_a_indices = mask_posecodes(input_index, current_sampling_prob, args.pkeep)
                # --> masking이 없음

                active_q_layers = q_schedule(bs=bs, low=0, high=dec_args.rvq_num_quantizers-1, device=m_res_tokens_new.device) # (bs, )
                pad_mask = create_pad_mask(args.block_size, (m_tokens_len + offset).cuda()) # (bs, n)
                q_non_pad_mask = repeat(pad_mask, 'b n -> b n q', q=dec_args.rvq_num_quantizers)
                mo_pad_mask = q_non_pad_mask[:, offset:, :]
                # mo_pad_mask = pad_mask[:, offset:, :]

                # if active_q_layers > 1:
                #     eval_input_r_codes = m_res_tokens_new[:, :-2, :(active_q_layers-1), :]  
                # else:
                #     eval_input_r_codes = None

                # # 똑같이 맞춰주는 case
                
                # eval_r_target = m_res_tokens_new[:, :-2, (active_q_layers-1), :]
    
                ### NEW way
                oh_vec = m_res_tokens_new[:, :-2, :, :] # bs, n-2, q, cb
                all_indices = torch.argmax(oh_vec, dim=-1) # bs, n-2, q
                
                tgt_indices = torch.where(mo_pad_mask.bool(), all_indices, pad_index) # PAD 토큰 채우기
                eval_target = tgt_indices[torch.arange(bs), :, active_q_layers.long()]  # (b, n) -> 정답, loss 계산에만 사용

                if args.eval_masking:
                    eval_a_pose_codes, masked_out = mask_posecodes(eval_pose_codes, current_sampling_prob, scheduled_pkeep[nb_iter])
                    if args.mask_residual_code:
                        all_indices = corrupt_residual_codes(all_indices=all_indices, rvq_nb_code=dec_args.rvq_nb_code, masking_prob=scheduled_masking_prob[nb_iter], pad_index=pad_index)
                else:
                    eval_a_pose_codes = eval_pose_codes   

                # cls_pred = res_trans_net(eval_a_pose_codes.float(), feat_clip_text) # logit
                cls_pred = res_trans_net(all_indices, eval_a_pose_codes, feat_clip_text, active_q_layers, mask=pad_mask)
                cls_pred = cls_pred.contiguous()
                
                eval_pred = cls_pred[:, offset:, :] # pad는 이후 ignore index에서 처리됨, mask 토큰으로 예측되어있을 경우 어쨌든 바로 잡힘
                eval_tgt = eval_target
            
                # 샘플별 validation loss 취득 및 update
                for name, wrapper in eval_loss_wrapper.items():
                    loss_name = str(wrapper)

                    if 'ce' in loss_name:
                        loss_ = wrapper.update(eval_pred, eval_tgt, ignore_index=pad_index) # ignore_index=64
                    
                    # 누적이 필요함
                    for key, value in loss_.items():
                        if key in eval_loss_dict:
                            eval_loss_dict[key] += value  # 누적
                        else:
                            eval_loss_dict[key] = value   # 새로 추가
                
                ################
                total_eval_iter += 1
                ################
                total_n_eval_sample += bs

            # 취득된 결과 종합 및 logging
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
                    avg_loss /= total_n_eval_sample # 앞에서는 sample별 loss를 계산하고 합산하기 때문에, 뒤에서 bs로 나눠줘야 함

                    writer.add_scalar(f'./Validation/{ls_name}', avg_loss, nb_iter)
                    logger.info(f"Validation. Iter {nb_iter} : {ls_name}. {avg_loss:.5f}")

                # return records

            for name, wrapper in eval_loss_wrapper.items():
                wrapper.reset()
            
            res_trans_net.train()
    
        # validation set에 대한 motion generation quality 측정
    if nb_iter % args.eval_iter ==  0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, val_acc, val_loss = eval_trans.evaluation_residual_transformer(args, args.out_dir, val_loader, net, trans_net, res_trans_net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, optimizer=optimizer, scheduler=scheduler, log_cat_right_num=args.log_cat_right_num, cat_mode=args.codes_folder_name, num_keywords=num_keywords)
        # wandb.log({"Val/best_fid": best_fid, "Val/best_top1": best_top1, "Val/best_top2": best_top2,"Val/best_top3": best_top3})
            
    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            

# wandb.finish()