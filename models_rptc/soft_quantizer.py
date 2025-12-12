import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from models_rptc.quantizer_utils import QuanizerBasedAttention

class SoftQuantize(nn.Module):
    def __init__(self, nb_code, code_dim, attn_dim, mu, beta, norm_type, init_method, entropy_temperature=0.01, sample_minimization_weight=1.0, batch_maximization_weight=1.0):
        super(SoftQuantize, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu  ##TO_DO
        self.beta = beta
        self.reset_codebook()
        self.attn = QuanizerBasedAttention(hidden_dim=code_dim, norm_type=norm_type, attn_dim=attn_dim)
        self.entropy_temperature = entropy_temperature
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight
        self.init_method = init_method
    
    def get_codebook_value_emb(self):
        """
        Return the current codebook tensor.
        (nn.Parameter → Tensor 로 바로 접근 가능)
        """
        
        return self.attn.to_v(self.codebook)

    def reset_codebook(self):
        self.init = False
        # 임시로 zero로 초기화
        # self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=True))
        self.register_parameter('codebook', nn.Parameter(torch.zeros(self.nb_code, self.code_dim)))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
            
        return out

    def init_codebook(self, x=None):
        """
        Initialize codebook with different strategies.
        mode:
            - 'enc': use data-driven tiling (default, needs x)
            - 'xavier': Xavier normal initialization
            - 'uniform': Uniform[-scale, scale] initialization
        """
        if self.init_method == 'enc':
            out = self._tile(x)
            with torch.no_grad():
                self.codebook.data.copy_(out[:self.nb_code])
        elif self.init_method == 'xavier':
            # Xavier normal initialization
            nn.init.xavier_normal_(self.codebook.data)
        elif self.init_method == 'uniform':
            # Uniform distribution in [-scale, scale]
            scale = 1.0 / self.code_dim**0.5
            nn.init.uniform_(self.codebook.data, -scale, scale)
        else:
            raise ValueError(f"Unknown mode: {self.init_method}")

        # codebook

        # self.code_sum = self.codebook.clone()
        # self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def compute_entropy_loss(
        self,
        logits,
        temperature=0.01,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        eps=1e-5,
    ):
        """
        Entropy loss of unnormalized logits

        logits: Affinities are over the last dimension

        https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
        LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
        """
        probs = F.softmax(logits / temperature, -1)
        log_probs = F.log_softmax(logits / temperature + eps, -1)

        avg_probs = reduce(probs, "... D -> D", "mean")

        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

        sample_entropy = -torch.sum(probs * log_probs, -1)
        sample_entropy = torch.mean(sample_entropy)

        loss = (sample_minimization_weight * sample_entropy) - (
            batch_maximization_weight * avg_entropy
        )

        result_loss = {
            "sample_entropy": sample_entropy,
            "avg_entropy": avg_entropy,
            "loss": loss
        }

        return result_loss

    def soft_quantize(self, x):
        # N X C
        k_w = self.codebook

        # x: (NT, C)
        logits, code_idx, z_q, z_q_2 = self.attn(x, k_w)
        # z_q
        # z_q_2

        ent_loss = self.compute_entropy_loss(logits=logits.reshape(-1, self.nb_code), temperature=self.entropy_temperature, sample_minimization_weight=self.sample_minimization_weight, batch_maximization_weight=self.batch_maximization_weight) # logits [b d h w] -> [b * h * w, n

        return code_idx, z_q, z_q_2, ent_loss
    
    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    # 실제로 codebook을 업데이트하는 부분

    # 3d -> 2d
    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(-1, x.shape[-1])
        x = rearrange(x, 'n c t -> (n t) c') # 
        return x

    def forward(self, x, return_idx=False, temperature=0.):
        N, width, T = x.shape
        output = {}

        x = self.preprocess(x)

        if self.training and not self.init:
            self.init_codebook(x)

        # code_idx, x_d, x_d_2, ent_loss = self.soft_quantize(x)
        code_idx, x_d, x_d_2, ent_loss = self.soft_quantize(x)
        # x_d = self.dequantize(code_idx)

        # codebook내의 code가 얼만큼 고르게 사용되고 있는지를 나타내는 지표
        # if self.training:
        #     perplexity = self.update_codebook(x, code_idx)
        # else:

        perplexity = self.compute_perplexity(code_idx)

        # commit_loss = F.mse_loss(x, x_d.detach()) # It's right. the t2m-gpt paper is wrong on embed loss and commitment loss.
        
        # following CoDA loss
        vq_loss = torch.mean((x_d - x)**2) + torch.mean((x_d_2.detach()-x)**2) + self.beta * \
                    torch.mean((x_d_2 - x.detach()) ** 2)
        
        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()

        output['x_d'] = x_d
        output['ent_loss'] = ent_loss
        output['perplexity'] = perplexity
        output['vq_loss'] = vq_loss
        
        # print(code_idx[0])
        if return_idx:
            output['code_idx'] = code_idx    
        
        return output


class SoftQuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, attn_dim, mu, beta, norm_type, entropy_temperature=0.01, sample_minimization_weight=1.0, batch_maximization_weight=1.0, init_method='enc'):
        super(SoftQuantizeEMAReset, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu  ##TO_DO
        self.beta = beta
        self.reset_codebook()
        self.attn = QuanizerBasedAttention(hidden_dim=code_dim, norm_type=norm_type, attn_dim=attn_dim)
        self.entropy_temperature = entropy_temperature
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        # 임시로 zero로 초기화
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
            
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def compute_entropy_loss(
        self,
        logits,
        temperature=0.01,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        eps=1e-5,
    ):
        """
        Entropy loss of unnormalized logits

        logits: Affinities are over the last dimension

        https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
        LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
        """
        probs = F.softmax(logits / temperature, -1)
        log_probs = F.log_softmax(logits / temperature + eps, -1)

        avg_probs = reduce(probs, "... D -> D", "mean")

        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

        sample_entropy = -torch.sum(probs * log_probs, -1)
        sample_entropy = torch.mean(sample_entropy)

        loss = (sample_minimization_weight * sample_entropy) - (
            batch_maximization_weight * avg_entropy
        )

        result_loss = {
            "sample_entropy": sample_entropy,
            "avg_entropy": avg_entropy,
            "loss": loss
        }

        return result_loss

    def soft_quantize(self, x):
        # N X C
        k_w = self.codebook

        # x: (NT, C)
        logits, code_idx, z_q, z_q_2 = self.attn(x, k_w)
        # z_q
        # z_q_2

        ent_loss = self.compute_entropy_loss(logits=logits.reshape(-1, self.nb_code), temperature=self.entropy_temperature, sample_minimization_weight=self.sample_minimization_weight, batch_maximization_weight=self.batch_maximization_weight) # logits [b d h w] -> [b * h * w, n

        return code_idx, z_q, z_q_2, ent_loss
    
    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    # 실제로 codebook을 업데이트하는 부분
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * code_rand

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(-1, x.shape[-1])
        x = rearrange(x, 'n c t -> (n t) c') # 
        return x

    def forward(self, x, return_idx=False, temperature=0.):
        N, width, T = x.shape
        output = {}

        x = self.preprocess(x)

        if self.training and not self.init:
            self.init_codebook(x)

        # code_idx, x_d, x_d_2, ent_loss = self.soft_quantize(x)
        code_idx, x_d, x_d_2, ent_loss = self.soft_quantize(x)
        # x_d = self.dequantize(code_idx)

        # codebook내의 code가 얼만큼 고르게 사용되고 있는지를 나타내는 지표
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # commit_loss = F.mse_loss(x, x_d.detach()) # It's right. the t2m-gpt paper is wrong on embed loss and commitment loss.
        
        # following CoDA loss

        # vq_loss = torch.mean((x_d - x)**2) + torch.mean((x_d_2.detach()-x)**2) + self.beta * \
        #             torch.mean((x_d_2 - x.detach()) ** 2)
        
        vq_loss = torch.mean((x_d.detach()-x)**2)
        
        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()

        output['x_d'] = x_d
        output['ent_loss'] = ent_loss
        output['perplexity'] = perplexity
        output['vq_loss'] = vq_loss
        
        # print(code_idx[0])
        if return_idx:
            output['code_idx'] = code_idx    
        
        return output
    
class SoftQuantizeEMA(SoftQuantizeEMAReset):
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * self.codebook

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity
