"""
Various positional encodings for the transformer.
"""
import math
from einops import repeat, rearrange
import torch
from torch import nn

def PE1d_sincos(seq_length, dim):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                         -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding (standard), learned.
    """
    def __init__(self, seq_length, dim, dropout, grad=False):
        super().__init__()
        self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x.shape: bs, seq_len, feat_dim
        l = x.shape[1] # length
        x = x.permute(1, 0, 2) + self.embed[:l].expand(x.permute(1, 0, 2).shape)
        x = self.dropout(x.permute(1, 0, 2))
    
        return x

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, max_seq_length, n_heads, dim, offset=0):
        super().__init__()

        # batch_size = 1
        self.n_heads = n_heads
        self.seq_len = max_seq_length
        self.theta = 10000
        self.dim = dim
        self.head_dim = dim // n_heads
        self.offset = offset

        pos = torch.arange(max_seq_length)
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))

        t = torch.einsum('n,d->nd', pos.to(freqs.dtype), freqs)

        # cos/sin: (1, 1, seq_len, head_dim)  로 브로드캐스트 되도록 준비
        cos = repeat(t.cos(), 'n d -> 1 1 n (d r)', r=2)
        sin = repeat(t.sin(), 'n d -> 1 1 n (d r)', r=2)

        # 편의상 패키징
        self.rot_cache = (cos, sin)
        
    
    def rotate_half(self, x):
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')
    
    def apply_rotary_emb(self, freqs: tuple, t: torch.Tensor, start_index: int = 0):
        cos, sin = freqs                             # (1, 1, seq, D) 
        T = t.shape[2] # (B, nh, T, hs)
        cos = cos[..., :T, :].to(t.device)  # (1, 1, seq, D)
        sin = sin[..., :T, :].to(t.device)  # (1, 1, seq, D)
        rot_dim = cos.shape[-1]
        end_index = start_index + rot_dim
        assert t.shape[-1] >= end_index, \
            f'feature dimension {t.shape[-1]} is not of sufficient size to {end_index}'

        t_left, t_mid, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        rotated = (t_mid * cos) + (self.rotate_half(t_mid) * sin)

        if self.offset > 0:
            idx = torch.arange(T, device=t.device)
            m = (idx >= self.offset).view(1, 1, T, 1)  # (1,1,T,1) → 브로드캐스트
            t_mid = torch.where(m, rotated, t_mid)
        else:
            t_mid = rotated

        return torch.cat((t_left, t_mid, t_right), dim=-1)

    def forward(self, q, k):
        rot_q = self.apply_rotary_emb(self.rot_cache, q)
        rot_k = self.apply_rotary_emb(self.rot_cache, k)

        return rot_q, rot_k
    
