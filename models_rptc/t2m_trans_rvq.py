# non-AR
import os

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding 
import time
import torch.nn.functional as F
from functools import partial

class RVQMotionTrans(nn.Module):

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
                num_quantizer=2):
        super().__init__()
        self.trans_base = CrossCondTransBase(num_vq, num_rvq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, num_quantizer)
        self.trans_head = CrossCondTransHead(num_rvq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size # 64 (motion len) / 4(down sampling rate) = 16(code_sequence_length)
        self.num_vq = num_vq
        self.num_rvq = num_rvq
        self.num_keywords = num_key # body part 수를 의미(아마도? 10 + 1?)
        self.mode = mode
        self.num_quantizer = num_quantizer
        self.cond_num = self.num_keywords + 1 + 1 # num key word + global text emb + q_id emb

    def get_block_size(self):
        return self.block_size

    def forward(self, r_codes, pose_codes, clip_feature, q_ids): # bs x T x code_num+2
        feat = self.trans_base(r_codes, pose_codes, clip_feature, q_ids) # bs x T x embedd_dim
        logits = self.trans_head(feat)
        return logits # bs x T x code_num + 1
    
    """
    토큰 샘플링이 이루어지는 함수
    이전 Transformer가 예측한 pose code들과 clip_feature를 입력으로 residual_token을 순차적으로 예측
    """
    @torch.no_grad()
    def sample(self, clip_feature, pose_codes, if_categorial=False, m_length=None,
           num_res_layers=None, sampler="argmax", temperature=1.0):
        """
        pose_codes : (B, T, num_vq+2)   # M-Transformer가 만든 base k-hot/soft one-hot
        clip_feature: (B, clip_dim)
        num_res_layers: 예측할 residual 레이어 수 (None이면 self.num_quantizers-1 사용)
        sampler: "argmax" | "gumbel"
        return: residual one-hots (B, T, num_res_layers, num_rvq)
        """
        device = pose_codes.device
        B, T, _ = pose_codes.size()

        # 예측할 residual 레이어 개수 (base 제외)
        if num_res_layers is None:
            num_res_layers = self.num_quantizers - 1

        residuals = []  # 레이어별로 (B,T,num_rvq) 텐서 누적

        for j in range(1, num_res_layers + 1):
            # 현재 레이어 ID 토큰
            q_ids = torch.full((B,), j, dtype=torch.long, device=device)

            # 이전 레이어들 r_codes(=residuals) 합산 조건 + [COND],[Q] 토큰 붙여 인코딩
            feat = self.trans_base(
                r_codes=residuals,            # 누적된 residual들을 조건으로
                pose_codes=pose_codes,        # base 토큰 시퀀스
                clip_feature=clip_feature,
                q_ids=q_ids
            )  
            

            # 현재 레이어 j의 residual 로짓
            logits = self.trans_head(feat)     # (B, T, num_rvq)

            # 병렬 시퀀스 예측이므로 전체 T를 한 번에 샘플
            if sampler == "argmax":
                pred_id = logits.argmax(dim=-1)                         # (B, T)
            elif sampler == "gumbel":
                g = -torch.empty_like(logits).exponential_().log()      # Gumbel(0,1)
                pred_id = ((logits + g) / max(temperature, 1e-5)).argmax(dim=-1)
            else:
                raise ValueError("sampler must be 'argmax' or 'gumbel'")

            # one-hot로 보관(다음 레이어 조건으로 쓰기 쉬움)
            # r_j = F.one_hot(pred_id, num_classes=logits.size(-1)).float()  # (B, T, num_rvq)
            r_j = F.one_hot(pred_id, num_classes=logits.size(-1)).float()  # (B, T, num_rvq)
            
            r_j = r_j[:, self.cond_num:, :]
            residuals.append(r_j)

        # (B, T, num_res_layers, num_rvq)로 묶어서 반환
        return torch.stack(residuals, dim=2)
        

class CrossConditionalSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % n_head == 0
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop  = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)

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

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        

        # ✅ causal 제거: 전체 위치에 양방향 어텐션 허용
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))  # 삭제

        # ✅ (옵션) 패딩 마스크 적용: pad 위치를 보지 않도록
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) {1: keep, 0: pad}
            # att shape 맞추기: (B, 1, 1, T)
            pad = (key_padding_mask == 0).unsqueeze(1).unsqueeze(2)  # True at pads
            att = att.masked_fill(pad, float('-inf'))

        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                 # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y, att


class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        output, att = self.attn(self.ln1(x))
        x = x + output
        x = x + self.mlp(self.ln2(x))
        return x

class CrossCondTransBase(nn.Module):

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
                num_quantizers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_quantizers = num_quantizers
        self.pose_tok_emb = nn.Linear(num_vq+2, embed_dim)
        self.r_tok_emb = nn.Linear(num_rvq+2, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.quant_emb = nn.Linear(self.num_quantizers, embed_dim)

        self.encode_quant = partial(F.one_hot, num_classes=self.num_quantizers)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

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
    def forward(self, r_codes, pose_codes, clip_feature, q_ids):

        q_onehot = F.one_hot(q_ids, num_classes=self.num_quantizers).float()

        q_onehot = q_onehot.to(device=pose_codes.device)
        q_emb = self.quant_emb(q_onehot).unsqueeze(1) # bs x 1 x embed_dim

        # pose_codes는 예측이 끝난 상태

        if len(r_codes) == 0:            
            b, t, C = pose_codes.size() #bs, t, code_num+2 -> one hot 형태
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."

            cond_embeddings = self.cond_emb(clip_feature) # .unsqueeze(1) #-> bs x 1 x embed_dim #.unsqueeze(1)
            token_embeddings = self.pose_tok_emb(pose_codes)
            token_embeddings = torch.cat([cond_embeddings, q_emb, token_embeddings], dim=1)
        else:
            b, t, C = pose_codes.size() #bs, t, code_num+2 -> one hot 형태
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            cond_embeddings = self.cond_emb(clip_feature)

            token_embeddings = self.pose_tok_emb(pose_codes) 
            
            if len(r_codes) == 1:
                r_codes = r_codes[0]
            else:
                if isinstance(r_codes, (list, tuple)):
                    stacked = torch.stack(r_codes, dim=1)
                    r_codes = torch.cumsum(stacked, dim=1)
                    r_codes = r_codes[:, -1, :, :]

            r_token_embeddings = self.r_tok_emb(r_codes)

            token_embeddings = token_embeddings + r_token_embeddings

            token_embeddings = torch.cat([cond_embeddings, q_emb, token_embeddings], dim=1) #bs x t+1 x embedd_dim #.unsqueeze(1)       
            
        x = self.pos_embed(token_embeddings) # ->
        x = self.blocks(x)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_rvq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_rvq+2, bias=False)
        self.block_size = block_size

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

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits #bs x t+1 x code_num+2

    


        

