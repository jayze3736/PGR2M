# non-AR
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding 
from functools import partial
from einops import rearrange, repeat
from models_rptc.utils.rt2m_utils import Attend
from utils.trainUtil import create_pad_mask

class RTransformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                num_rvq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                num_key=11,
                mode=None,
                num_quantizer=2,
                share_weight=False, 
                shared_codebook=False):
        super().__init__()

        # input: text feature, pose code feature

        # output: residual pose code
        self.proc_in = InputProcessor(num_vq, num_rvq, clip_dim, embed_dim, block_size, share_weight, shared_codebook, num_quantizer)
        self.proc_out = OutputProcessor(num_rvq, embed_dim, share_weight, shared_codebook, num_quantizer)
        self.trans_base = TransBase(embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, num_quantizer, input_processor=self.proc_in)

        self.block_size = block_size + 1 # 62 (motion len) / 4(down sampling rate) = 16(code_sequence_length)
        self.num_vq = num_vq
        self.num_rvq = num_rvq
        self.num_keywords = num_key # body part 수를 의미(아마도? 10 + 1?)
        self.mode = mode
        self.num_quantizer = num_quantizer
        self.cond_num = self.num_keywords + 2
        self.max_m_token_len = self.block_size - self.cond_num # 63 - 13 = 50
        print("Note: self.block_size:", self.block_size)
        print("Note: self.max_m_token_len:", self.max_m_token_len)
        self.share_weight = share_weight

        if self.share_weight:
            self.embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.num_quantizer, num_rvq+2, embed_dim)))
            self.process_embed_proj_weight()

    def process_embed_proj_weight(self):
        if self.share_weight:
            # if not self.registered:
            self.proc_out.output_proj_weight = self.embed_proj_shared_weight
            self.proc_in.token_embed_weight = self.embed_proj_shared_weight

    def get_block_size(self):
        return self.block_size

    def forward(self, r_codes, p_codes, clip_feature, active_q_ids, mask=None, inference=False): # bs x T x code_num+2
        
        # mask: bs x T
        if mask is not None:
            bs, n_seq = mask.shape
            mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            mask = mask.view(bs, 1, n_seq, n_seq)
        
        active_q_ids = active_q_ids.long()
        x = self.trans_base(r_codes, p_codes, clip_feature, active_q_ids, pad_mask=mask, inference=inference) # bs x T x embedd_dim
        logits = self.proc_out(x, active_q_ids)

        return logits # bs x T x code_num + 1

    """
    토큰 샘플링이 이루어지는 함수
    이전 Transformer가 예측한 pose code들과 clip_feature를 입력으로 residual_token을 순차적으로 예측
    """

    # forward --> 한번에, logit 나옴, sample(softmax, categorical)
    # 입력: text, pose code, trans_net --> trans_net의 inference를 먼저 진행해야함
    # 
    
    # @torch.no_grad()
    # def sample(self, clip_feature, trans, if_categorial=False, temperature=1.0, offset=11, if_residual_categorical=False):
    #     """
    #     pose_codes : (B, T, num_vq+2)   # M-Transformer가 만든 base k-hot/soft one-hot
    #     clip_feature: (B, clip_dim)
    #     trans: pose code 예측 모델
    #     num_res_layers: 예측할 residual 레이어 수 (None이면 self.num_quantizers-1 사용)
    #     sampler: "argmax" | "gumbel"
    #     return: residual one-hots (B, T, num_res_layers, num_rvq)
    #     """
        
    #     cond = { 'clip_feat_text': clip_feature }
    #     bs = clip_feature.shape[0]

    #     # pretrained t2m model을 사용하여 pose code 예측
    #     pred_p_codes, _ = trans.sample(cond, if_categorial=if_categorial) # t2m model이 항상 keywords를 사용한다는 가정
        
    #     text_feat = clip_feature[:, 0:offset+1, :] # offset에 따라서 keyword embedding 사용 유무 결정
    #     pred_p_codes[:, :, -2:] = 0 # 마지막 1개는 사용하지 않음(PAD, END), END 토큰 자리도 사용 x, one-hot vector

    #     m_tokens_len = pred_p_codes.shape[1]
    #     pad_len = self.max_m_token_len - m_tokens_len
    #     pad_pred_p_codes = torch.nn.functional.pad(pred_p_codes, (0,0,0, pad_len), value=0) 
    #     pad_mask = create_pad_mask(self.block_size, torch.tensor([m_tokens_len + self.cond_num], device=clip_feature.device)) # (bs, n)

    #     # 생성한 pose code에 대해서 max 길이만큼 뒤에 padding 추가
    #     # pred_p_codes = F.pad(pred_p_codes, (0,0,0,self.block_size - m_tokens_len), value=0) # (bs, block_size, num_vq+2)
    #     # pad_mask = create_pad_mask(self.block_size, (m_tokens_len + offset).cuda()) # (bs, n)

    #     # pad mask 생성

    #     pred_r_indices = None
    #     for target_q_id in range(0, self.num_quantizer): # 0번째 residual code를 예측부터 시작, num_quantizer - 1까지 예측
    #         q_cond = torch.full((bs,), target_q_id, device=clip_feature.device) # (1, )
    #         logits = self.forward(pred_r_indices, pad_pred_p_codes, text_feat, q_cond, mask=pad_mask, inference=True) # bs x T x code_num+1
    #         logits = logits[:, offset+2:, :-2] # key_word=11개, sentence = 1개, q_id = 1개 = 13개 

    #         # softmax
    #         if if_residual_categorical:
    #             prob = F.softmax(logits / temperature, dim=-1) # bs x T x code_num+1
    #             pred_r_indice = torch.distributions.Categorical(prob).sample() # bs x T
    #         else:
    #             idx = torch.argmax(logits, dim=-1)          # [bs, T]
    #             pred_r_indice = idx.view(idx.size(0), -1)  # 항상 [bs, T]

    #         # pred_r_indice = pred_r_indice[...,] # 마지막 PAD 토큰 제거
    #         # pred_r_indices = torch.stack(pred_r_indices, pred_r_indice) if pred_r_indices is not None else pred_r_indice # bs x T x n_q x nb_code

    #         pred_r_indices = torch.cat([pred_r_indices, pred_r_indice.unsqueeze(2)], dim=2) if pred_r_indices is not None else pred_r_indice.unsqueeze(2) # bs x T x n_q -> 정수형 코드

    #         # 요구 입력 shape: bs, T, n_q, nb_code -> one-hot으로 줘야함
    #         # 스택쌓아서 줘야함
        
    #     pred_r_indices = pred_r_indices[:, :m_tokens_len, :] # pad 부분 자르기
    #     pred_r_codes = F.one_hot(pred_r_indices, num_classes=self.num_rvq).float() # bs x T x n_q x nb_code -> one-hot 형태로 변환
        
    #     return pred_p_codes, pred_r_codes
    

    @torch.no_grad()
    def sample(self, clip_feature, trans, if_categorial=False, temperature=1.0, offset=11, if_residual_categorical=False):
        """
        pose_codes : (B, T, num_vq+2)   # M-Transformer가 만든 base k-hot/soft one-hot
        clip_feature: (B, clip_dim)
        trans: pose code 예측 모델
        num_res_layers: 예측할 residual 레이어 수 (None이면 self.num_quantizers-1 사용)
        sampler: "argmax" | "gumbel"
        return: residual one-hots (B, T, num_res_layers, num_rvq)
        """
    
        bs = clip_feature.shape[0]

        # pretrained t2m model을 사용하여 pose code 예측
        pred_p_codes = trans.sample(clip_feature, if_categorial=if_categorial) # t2m model이 항상 keywords를 사용한다는 가정

        
        
        text_feat = clip_feature[:, 0:offset+1, :] # offset에 따라서 keyword embedding 사용 유무 결정
        pred_p_codes[:, :, -2:] = 0 # 마지막 1개는 사용하지 않음(PAD, END), END 토큰 자리도 사용 x, one-hot vector
        pred_r_indices = None

        for target_q_id in range(0, self.num_quantizer): # 0번째 residual code를 예측부터 시작, num_quantizer - 1까지 예측
            q_cond = torch.full((bs,), target_q_id, device=clip_feature.device) # (1, )
            logits = self.forward(pred_r_indices, pred_p_codes, text_feat, q_cond, mask=None, inference=True) # bs x T x code_num+1
            logits = logits[:, offset+2:, :-2] # key_word=11개, sentence = 1개, q_id = 1개 = 13개 # 

            # softmax
            if if_residual_categorical:
                prob = F.softmax(logits / temperature, dim=-1) # bs x T x code_num+1
                pred_r_indice = torch.distributions.Categorical(prob).sample() # bs x T
            else:
                idx = torch.argmax(logits, dim=-1)          # [bs, T]
                pred_r_indice = idx.view(idx.size(0), -1)  # 항상 [bs, T]

            # pred_r_indice = pred_r_indice[...,] # 마지막 PAD 토큰 제거
            # pred_r_indices = torch.stack(pred_r_indices, pred_r_indice) if pred_r_indices is not None else pred_r_indice # bs x T x n_q x nb_code

            pred_r_indices = torch.cat([pred_r_indices, pred_r_indice.unsqueeze(2)], dim=2) if pred_r_indices is not None else pred_r_indice.unsqueeze(2) # bs x T x n_q -> 정수형 코드

            # 요구 입력 shape: bs, T, n_q, nb_code -> one-hot으로 줘야함
            # 스택쌓아서 줘야함
        
        pred_r_codes = F.one_hot(pred_r_indices, num_classes=self.num_rvq).float() # bs x T x code_num x nb_code -> one-hot 형태로 변환

        return pred_p_codes, pred_r_codes
    
    @torch.no_grad()
    def sample_fast(self, clip_feature, trans, if_categorial=False, temperature=1.0, offset=11, if_residual_categorical=False):
        """
        pose_codes : (B, T, num_vq+2)   # M-Transformer가 만든 base k-hot/soft one-hot
        clip_feature: (B, clip_dim)
        trans: pose code 예측 모델
        num_res_layers: 예측할 residual 레이어 수 (None이면 self.num_quantizers-1 사용)
        sampler: "argmax" | "gumbel"
        return: residual one-hots (B, T, num_res_layers, num_rvq)
        """
    
        bs = clip_feature.shape[0]
        text_feat = clip_feature[:, 0:offset+1, :] # offset에 따라서 keyword embedding 사용 유무 결정

        # pretrained t2m model을 사용하여 pose code 예측
        pred_p_codes = trans.sample_fast(clip_feature, if_categorial=if_categorial) # t2m model이 항상 keywords를 사용한다는 가정, bs x 50 x (num_vq+2)

        is_end_token = (pred_p_codes[:, :, self.num_vq] == 1) 
        has_end_token = torch.any(is_end_token, dim=1)
        pred_mo_lens = torch.argmax(is_end_token.int(), dim=1)

        pred_mo_lens[~has_end_token] = self.max_m_token_len # 최대 시퀀스 길이로 설정

        indices = torch.arange(self.block_size, device=pred_p_codes.device)
        pred_mo_lens_tensor = pred_mo_lens.to(pred_p_codes.device).unsqueeze(1)
        mask = (indices < (pred_mo_lens_tensor + self.cond_num)) # bs x 50
        
        pred_p_codes[:, :, -2:] = 0 
        pred_r_indices = None

        for target_q_id in range(0, self.num_quantizer): # 0번째 residual code를 예측부터 시작, num_quantizer - 1까지 예측
            q_cond = torch.full((bs,), target_q_id, device=clip_feature.device) # (1, )
            logits = self.forward(pred_r_indices, pred_p_codes, text_feat, q_cond, mask=mask, inference=True) # bs x (49 + 13) x code_num+1
            logits = logits[:, offset+2:, :-2] # key_word=11개, sentence = 1개, q_id = 1개 = 13개 # 

            # softmax
            if if_residual_categorical:
                prob = F.softmax(logits / temperature, dim=-1) # bs x T x code_num+1
                pred_r_indice = torch.distributions.Categorical(prob).sample() # bs x T
            else:
                idx = torch.argmax(logits, dim=-1)          # [bs, T]
                pred_r_indice = idx.view(idx.size(0), -1)  # 항상 [bs, T]

            # pred_r_indice = pred_r_indice[...,] # 마지막 PAD 토큰 제거
            # pred_r_indices = torch.stack(pred_r_indices, pred_r_indice) if pred_r_indices is not None else pred_r_indice # bs x T x n_q x nb_code

            pred_r_indices = torch.cat([pred_r_indices, pred_r_indice.unsqueeze(2)], dim=2) if pred_r_indices is not None else pred_r_indice.unsqueeze(2) # bs x T x n_q -> 정수형 코드

            # 요구 입력 shape: bs, T, n_q, nb_code -> one-hot으로 줘야함
            # 스택쌓아서 줘야함

        pred_r_codes = F.one_hot(pred_r_indices, num_classes=self.num_rvq).float() # bs x T x code_num x nb_code -> one-hot 형태로 변환

        return pred_p_codes, pred_r_codes, pred_mo_lens.cpu().numpy()



# 임력만을 전문적으로 임베딩
class InputProcessor(nn.Module):
    def __init__(self, num_vq, num_rvq, clip_dim, embed_dim, block_size, share_weight, shared_codebook, num_quantizers):
        super().__init__()
        self.num_vq = num_vq
        self.num_rvq =num_rvq
        self.num_quantizers = num_quantizers
        self.share_weight = share_weight
        self.shared_codebook = shared_codebook
        self.embed_dim = embed_dim
        self.block_size = block_size
        
        # 입력과 관련된 
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pose_tok_emb = nn.Linear(num_vq+2, embed_dim) # pose code embedding
        self.quant_emb = nn.Linear(self.num_quantizers, embed_dim) # quantizer layer 순서를 나타내는 조건
        self.pos_embed = pos_encoding.PositionEmbedding(self.block_size, embed_dim, 0.0, False)
        self.encode_quant = partial(F.one_hot, num_classes=self.num_quantizers)
        
        self.init_()
    
    # 임베딩 테이블 초기화
    def init_(self):
        
        if self.shared_codebook:
            # input embedding
            # 동일하게 초기화한 embedding을 num_quantizer 개수만큼 반복
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.num_rvq+2, self.embed_dim)))
            self.token_embed_weight = token_embed.expand(self.num_quantizers-1, self.num_rvq+2, self.embed_dim)
        else:
            # projection layer
            # share weight: 각 quantizer level마다 동일한 projection layer를 사용할 것인가?
            if self.share_weight:
                self.token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_rvq+2, self.embed_dim)))
            else: # 거의 main
                # quantizer별로 다른 embedding table 사용
                self.token_embed_weight = torch.normal(mean=0, std=0.02, size=(self.num_quantizers-1, self.num_rvq+2, self.embed_dim))
                self.token_embed_weight = nn.Parameter(self.token_embed_weight)

            
    
    def forward(self, clip_feature, q_ids, p_codes, r_codes):
        return self.emb_cond(clip_feature), self.emb_q_ids(q_ids), self.emb_pose_codes(p_codes, q_ids), self.emb_res_codes(r_codes)

    def emb_cond(self, clip_feature):
        cond_embeddings = self.cond_emb(clip_feature) # text emb
        return cond_embeddings
    
    def emb_q_ids(self, q_ids):
        # q_ids: (bs, )

        # q_onehot = F.one_hot(q_id, num_classes=self.num_quantizers).float()
        # q_onehot = q_onehot.to(device=pose_codes.device)
        # q_emb = self.quant_emb(q_onehot).unsqueeze(1) # bs x 1 x embed_dim
        q_ids = q_ids.long()
        q_oh = self.encode_quant(q_ids) # one-hot
        q_oh = q_oh.float()
        q_emb = self.quant_emb(q_oh) # one-hot -> emb

        return q_emb
    
    def emb_pose_codes(self, p_codes):
        # one hot 형태로 입력됨
        p_emb = self.pose_tok_emb(p_codes)
        return p_emb
    
    # 
    def emb_res_codes(self, all_indices, active_q_layers, inference=False):
        # all_indices: bs, n_seq, n_q
        # q_id: int -> 입력되는 res code의 quantizer layer id
        active_q_layers = active_q_layers.long()
        bs, T, Q = all_indices.shape # 정수형 index를 입력으로 받음
        
        ######################## r_codes(one hot)를 정수형으로 변환 ########################
        # all_indices = torch.argmax(r_codes, dim=-1)

        ########################## gather codes ##########################
        token_embed_repeat = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs) # batch size에 따라 반복

        # all_indices에서 q layer가 1일 경우 이 부분은 사라지게 됨
        # train에서는 all_indices가 전체 layer의 코드를 가지지면, inference시에는 하나씩 생성을 하기 때문에 all indices가 다름
    
        # training
        if not inference:
            gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed_repeat.shape[2]) # 코드 차원을 임베딩 차원에 따라 복사
        else:
            gather_indices = repeat(all_indices, 'b n q -> b n d q', d=token_embed_repeat.shape[2]) # 코드 차원을 임베딩 차원에 따라 복사

        # 복제를 하는 이유는 gather 함수가 하나의 대응되는 스칼라값을 가져오기 때문에 임베딩 벡터 전체를 가져오려면 동일한 차원만큼 자리를 만들어줘야함
        all_codes = token_embed_repeat.gather(1, gather_indices)  # (b, n, d, q-1) or (b, n, d, q)(inference mode) # codebook 차원에 대해서 indices에 해당하는 벡터를 가져옴

        ######################### 주어진 q_id까지 embedding의 history sum을 구함 #########################
        cumsum_codes = torch.cumsum(all_codes, dim=-1) # bs x t x dim x n_q-1 or bs x t x dim x n_q(inference mode)
        
        ########################## q_ids 위치에 해당하는 임베딩을 전달 ##########################
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers-1] # bs x t x dim # embedding table zero index base, q_id는 1부터 시작

        return history_sum

    def emb_positional(self, x):
        # position embedding
        x = self.pos_embed(x)
        return x
        

# 출력만을 전문적으로 임베딩
class OutputProcessor(nn.Module):
    def __init__(self, num_rvq, embed_dim, share_weight, shared_codebook, num_quantizers):
        super().__init__()
        self.num_rvq = num_rvq
        self.num_quantizers = num_quantizers
        self.share_weight = share_weight
        self.shared_codebook = shared_codebook
        self.embed_dim = embed_dim
        
        # condition embedding
        self.init_()
        

    # 임베딩 테이블 초기화
    def init_(self):
        if self.shared_codebook:
            # input embedding
            # 동일하게 초기화한 embedding을 num_quantizer 개수만큼 반복
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.num_rvq+2, self.embed_dim)))
            self.token_embed_weight = token_embed.expand(self.num_quantizers-1, self.num_rvq+2, self.embed_dim)

            # projection layer
            # share weight: 각 quantizer level마다 동일한 projection layer를 사용할 것인가?
            if self.share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                self.output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.num_rvq+2, self.embed_dim)))
                self.output_bias = nn.Parameter(torch.zeros(size=(self.num_rvq+2,)))
                
                # output_proj_bias = 0
                self.output_proj_weight = self.output_proj.expand(self.num_quantizers, self.num_rvq+2, self.embed_dim)
                self.output_proj_bias = self.output_bias.expand(self.num_quantizers, self.num_rvq+2)
        else:
            # projection layer
            # share weight: 각 quantizer level마다 동일한 projection layer를 사용할 것인가?
            if self.share_weight:
                self.output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_rvq+2, self.embed_dim)))
                self.output_proj_bias = None
                self.registered = False
            else: # 거의 main
                self.output_proj_weight = torch.normal(mean=0, std=0.02,
                                                    size=(self.num_quantizers, self.num_rvq+2, self.embed_dim))

                self.output_proj_weight = nn.Parameter(self.output_proj_weight)
                self.output_proj_bias = nn.Parameter(torch.zeros(size=(self.num_quantizers, self.num_rvq+2)))


        
    def forward(self, logits, active_q_layers):
        '''
        :logits: (bs, seqlen, code_dim)
        :active_q_layers: (bs, )

        :return:
            logits (bs, seqlen, dim)
        '''
        # (num_qlayers-1, num_token, code_dim) -> (bs, ntoken, code_dim)

        output_proj_weight = self.output_proj_weight[active_q_layers]
        # (num_qlayers, ntoken) -> (bs, ntoken)
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[active_q_layers]

        output = torch.einsum('bnc, bsc->bsn', output_proj_weight, logits)

        if output_proj_bias is not None:
            output += output + output_proj_bias.unsqueeze(1)
        
        return output
    


class TransBase(nn.Module):

    def __init__(self, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                num_quantizers=2,
                input_processor:InputProcessor=None
                ):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.num_quantizers = num_quantizers
        self.proc_in = input_processor
        
        # transformer backbone
        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.block_size = block_size

        self.drop = nn.Dropout(drop_out_rate)
        self.apply(self._init_weights)


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    # 목표는 clip feature와, 앞서 Transformer의 Autoregressive 결과물을 입력으로 residual token을 예측하는 구조
    def forward(self, res_codes, pose_codes, clip_feature, active_q_layers, pad_mask=None, inference=False): # pad mask 필요
        
        # res_codes는 None이 될 수 있음
        if res_codes is not None:
            B, T, C = res_codes.size() #bs, t, code_num+1 -> one hot 형태
            assert T <= self.block_size, "Cannot forward, model block size is exhausted."

        # condition 1: text embedding
        cond_embeddings = self.proc_in.emb_cond(clip_feature) # bs, key_word_num+1, embed_dim
        q_emb = self.proc_in.emb_q_ids(active_q_layers).unsqueeze(1)
        pose_code_embeddings = self.proc_in.emb_pose_codes(pose_codes) # bs, n_seq, embed_dim
        
        if res_codes is not None:
            zero_mask = (active_q_layers == 0)
            r_token_embeddings = self.proc_in.emb_res_codes(res_codes, active_q_layers, inference=inference) # 입력하는 res_codes 
            history_sum = pose_code_embeddings + r_token_embeddings
            token_embeddings = torch.where(zero_mask.view(B, 1, 1), pose_code_embeddings, history_sum) # zero 인 위치에는 pose code embedding만, 아닌 경우 pose code와 합산한 embedding을 사용
        else:
            token_embeddings = pose_code_embeddings

        token_embeddings = self.proc_in.emb_positional(token_embeddings)
        
        x = torch.cat([cond_embeddings, q_emb, token_embeddings], dim=1) #bs x t+1 x embedd_dim #.unsqueeze(1)

        for blk in self.blocks:
            x = blk(x, mask=pad_mask)

        return x



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, flash=True):
        super().__init__()
        assert embed_dim % n_head == 0
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.resid_drop = nn.Dropout(drop_out_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # ToDo: key_padding_mask 필요, Attention Module 교체 필요
        self.attn = Attend(dropout=drop_out_rate, flash=flash)  # flash=True 시 오류 발생 가능

        self.n_head = n_head

    # @torch.cuda.amp.autocast(enabled=False)  # 선택: 안정적 softmax용
    def forward(self, x, key_padding_mask=None):
        """
        x: (B, T, C)
        key_padding_mask: (B, T)에서 유효 토큰은 1, pad는 0 (선택)
        """
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # ToDo: key_padding_mask 필요, Attention Module 교체 필요

        if key_padding_mask is not None:
            key_padding_mask = (key_padding_mask[:,:,:T,:T] == 0) # B

        y, att = self.attn(q, k, v, key_padding_mask=key_padding_mask)  # (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y, att


class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, block_size, n_head, drop_out_rate, flash=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )


    def forward(self, x, mask=None):
        output, att = self.attn(self.ln1(x), mask)
        x = x + output
        x = x + self.mlp(self.ln2(x))
        return x
