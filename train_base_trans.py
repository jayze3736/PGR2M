import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from os.path import join as pjoin

import json
import clip
from tqdm import tqdm
import argparse

import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
import utils.losses as losses 
from utils.codebook import *

import yaml
from models.pg_tokenizer import PoseGuidedTokenizer
import models.motion_dec as motion_dec
import options.option_transformer as option_trans

from dataset import dataset_TM_train, dataset_TM_eval

import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper

from datetime import datetime
import random

def fixseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_dec_cfg_ckpt_path(folder_path):

    if folder_path is None:
        return None, None
    else:
        ckpt_path = pjoin(folder_path, 'net_best_fid.pth')
        config_path = pjoin(folder_path, 'arguments.yaml')
    
    return config_path, ckpt_path

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
fixseed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')

####### save configs #######
date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.out_dir = os.path.join(args.out_dir, date_time)
os.makedirs(args.out_dir, exist_ok = True)
json_path = os.path.join(args.out_dir, 'arguments.yaml')

with open(json_path, 'w') as f:
    dict_args = vars(args)
    json.dump(dict_args, f, indent=2)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer, codebook_size=args.nb_code, use_keywords=args.use_keywords)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
text_encoder, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
text_encoder.eval()
for p in text_encoder.parameters():
    p.requires_grad = False

dec_config, dec_checkpoint_path = get_dec_cfg_ckpt_path(args.dec_checkpoint_folder)

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
                resi_beta=dec_args.rvq_resi_beta,
                vq_loss_beta=dec_args.rvq_vq_loss_beta,
                quantizer_type=dec_args.rvq_quantizer_type,
                params_soft_ent_loss=dec_args.params_soft_ent_loss,
                use_ema= (not args.unuse_ema),
                init_method=dec_args.init_method
                )

print ('loading decoder checkpoint from {}'.format(dec_checkpoint_path))
ckpt = torch.load(dec_checkpoint_path, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

trans_net = trans.BaseTrans(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_net.load_state_dict(ckpt['trans'], strict=True)

trans_net.train()
trans_net.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_net, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_bce = torch.nn.BCEWithLogitsLoss()
Loss = losses.ReConsLoss('l1_smooth', 22)

nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, unit_length=2**args.down_t, use_keywords=args.use_keywords)
train_loader_iter = dataset_TM_train.cycle(train_loader)

##### ---- Training ---- #####
best_fid=1000
best_iter=0
best_div=100
best_top1=0
best_top2=0
best_top3=0
best_matching=100
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, val_acc, val_loss = eval_trans.evaluation_transformer(args, args.out_dir, val_loader, net, trans_net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, text_encoder=text_encoder, eval_wrapper=eval_wrapper, optimizer=optimizer, scheduler=scheduler, log_cat_right_num=args.log_cat_right_num, cat_mode=args.codes_folder_name, use_rptc=dec_args.use_rvq)
        

# Define the exponential decay schedule parameters
decay_factor = 0.99  # Exponential decay factor (adjust as needed)
min_sampling_prob = 0.1
current_sampling_prob = 1.0  # Initial sampling probability (start with full teacher forcing)
thresh = torch.nn.Threshold(0.5,0)
avg_acc = 0.
cat_num = 70 # number of pose code categories
while nb_iter <= args.total_iter:
    if nb_iter > 0:
        current_sampling_prob = max(min_sampling_prob, current_sampling_prob * decay_factor)
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len, keyword_embeddings, gt_motion, m_length = batch # bs x 1, bs x 55 x 512, keyword_embeddings

    m_tokens, m_tokens_len, gt_motion = m_tokens.cuda(), m_tokens_len.cuda(), gt_motion.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens.cuda()   # (bs, t, code_num+2)

    text = clip.tokenize(clip_text, truncate=True).cuda()
    feat_clip_text = text_encoder.encode_text(text).float().unsqueeze(1)

    if args.use_keywords:
        feat_clip_text = torch.cat((feat_clip_text, keyword_embeddings.float().cuda()), dim = 1) #bs x 12 x 512

    input_index = target[:,:-1,:] # (bs, t-1, code_num+2)

    if np.random.random() >= current_sampling_prob:
        if args.pkeep == -1:
            proba = np.random.rand(1)[0]
            mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                            device=input_index.device))
        else:
            mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                            device=input_index.device))
        mask = mask.round().to(dtype=torch.int64)
        r_indices = torch.randn(input_index.shape, device = input_index.device)
        a_indices = mask*input_index+(1-mask)*r_indices

        # Mutual exclusivity
        for cat in vq_to_range:
            if cat < cat_num:
                end, start = vq_to_range[cat]
                idx = torch.argmax(a_indices[:,:,start:end+1], dim = -1, keepdim=True)
                a_indices[:,:,start:end+1] = 0
                a_indices.scatter_(-1, start+idx, 1)
            else:
                end, start = vq_to_range[cat]
                a_indices[:,:,start:end+1] = torch.nn.functional.sigmoid(a_indices[:,:,start:end+1]) > 0.5
    else:
        a_indices = input_index
    
    cls_pred = trans_net(a_indices.float(), feat_clip_text) 
    cls_pred = cls_pred.contiguous()
    
    if args.use_keywords:
        offset = 11 #number of keywords
    else:
        offset = 0
        
    loss_cls = 0.0

    for i in range(bs):
        pred = cls_pred[i][offset:m_tokens_len[i] + 1 + offset] 
        tgt = target[i][:m_tokens_len[i] + 1].float() 
        loss_cls += loss_bce(pred,tgt)/bs

        if nb_iter % args.print_iter ==  0:
            for cat in vq_to_range:
                if cat < cat_num:
                    end, start = vq_to_range[cat]
                    right_num += (torch.argmax(pred[:,start:end+1], dim = -1) == torch.argmax(tgt[:,start:end+1], dim = -1)).sum().item()

            nb_sample_train += ((m_tokens_len[i]+1)*cat_num).item()

    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_cls.item()

    nb_iter += 1
    if nb_iter % args.print_iter ==  0 :
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        avg_acc = 0.
        right_num = 0
        nb_sample_train = 0
        current_sampling_prob = max(min_sampling_prob, current_sampling_prob * decay_factor)

    if nb_iter % args.eval_iter ==  0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, val_acc, val_loss = eval_trans.evaluation_transformer(args, 
                                                                                                                                                             args.out_dir, 
                                                                                                                                                             val_loader, 
                                                                                                                                                             net, 
                                                                                                                                                             trans_net, 
                                                                                                                                                             logger, 
                                                                                                                                                             writer, 
                                                                                                                                                             nb_iter, 
                                                                                                                                                             best_fid, 
                                                                                                                                                             best_iter, 
                                                                                                                                                             best_div, 
                                                                                                                                                             best_top1, 
                                                                                                                                                             best_top2, 
                                                                                                                                                             best_top3, 
                                                                                                                                                             best_matching, 
                                                                                                                                                             text_encoder=text_encoder, 
                                                                                                                                                             eval_wrapper=eval_wrapper,
                                                                                                                                                             optimizer=optimizer, 
                                                                                                                                                             scheduler=scheduler, 
                                                                                                                                                             cat_mode=args.codes_folder_name)
        
    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            
