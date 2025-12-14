import random
from math import ceil
from functools import partial
from itertools import zip_longest
from random import randrange
from torch import einsum
import torch
from torch import nn
import torch.nn.functional as F
from models.vq.quantizer import QuantizeEMAReset, SoftQuantize
from einops import rearrange, repeat, pack, unpack

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# main class

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        quantizer_type='hard',
        params_soft_ent_loss=0.0,
        use_ema=True,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        if quantizer_type == 'soft':
            self.quantizer = SoftQuantize
        else:
            self.quantizer = QuantizeEMAReset

        if shared_codebook:
            layer = self.quantizer(**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([self.quantizer(**kwargs) for _ in range(num_quantizers)])
        
        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob
        self.params_soft_ent_loss = params_soft_ent_loss

            
    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks # 'q c d'
    
    def get_codes_from_indices(self, indices): #indices shape 'b n q' # dequantize

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct
        
        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # get ready for gathering

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

        # take care of quantizer dropout

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later
        all_codes = codebooks.gather(2, gather_indices) # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.)

        return all_codes # 'q b n d'

    def get_codebook_entry(self, indices): #indices shape 'b n q'
        all_codes = self.get_codes_from_indices(indices) #'q b n d'
        latent = torch.sum(all_codes, dim=0) #'b n d'
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes = False, sample_codebook_temp = None, force_dropout_index=-1):
        num_quant, quant_dropout_prob, device = self.num_quantizers, self.quantize_dropout_prob, x.device

        quantized_out = 0. 
        residual = x 

        all_losses = []
        all_indices = []
        all_perplexity = []

        all_ent_sub_samp_loss = []
        all_ent_sub_avg_loss = []
        all_ent_loss = []

        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        start_drop_quantize_index = num_quant
        # To ensure the first-k layers learn things as much as possible, we randomly dropout the last q - k layers
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant) # keep quant layers <= quantize_dropout_cutoff_index, TODO vary in batch
            null_indices_shape = [x.shape[0], x.shape[-1]] # 'b*n'
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)

        if force_dropout_index >= 0: # 
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index # 5
            null_indices_shape = [x.shape[0], x.shape[-1]]  # 'b*n'
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

        for quantizer_index, layer in enumerate(self.layers): # 0, 1, 2, 3, 4, 5

            if should_quantize_dropout and quantizer_index > start_drop_quantize_index: 
                all_indices.append(null_indices)
                continue

            output = layer(residual, return_idx=True, temperature=sample_codebook_temp) #single quantizer

            quantized = output['x_d']
            embed_indices = output['code_idx']
            loss = output['vq_loss']
            perplexity = output['perplexity']

            if self.params_soft_ent_loss > 0:
                ent_out = output['ent_loss'] # dict
                ent_sub_samp_loss = ent_out['sample_entropy']
                ent_sub_avg_loss = ent_out['avg_entropy']
                ent_loss = ent_out['loss']
            else:
                ent_sub_samp_loss = torch.tensor(0.0, device=device)
                ent_sub_avg_loss = torch.tensor(0.0, device=device)
                ent_loss = torch.tensor(0.0, device=device)

            all_ent_sub_samp_loss.append(ent_sub_samp_loss)
            all_ent_sub_avg_loss.append(ent_sub_avg_loss)
            all_ent_loss.append(ent_loss)

            residual -= quantized.detach() # x - x_d -> residual
            quantized_out += quantized # x_d + r_1 + r_2 + ... 

            
            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)

        # stack all losses and indices
        all_indices = torch.stack(all_indices, dim=-1) 
        all_losses = sum(all_losses)/len(all_losses) 
        all_perplexity = sum(all_perplexity)/len(all_perplexity) 
        
        ret = {}

        ret = {
            "quantized_out": quantized_out,
            "all_indices": all_indices,
            "vq_loss": all_losses,
            "perplexity": all_perplexity
        }

        if return_all_codes:
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            # ret = (*ret, all_codes)
            ret['all_codes'] = all_codes

        
        all_ent_sub_samp_loss = sum(all_ent_sub_samp_loss)/len(all_ent_sub_samp_loss) 
        all_ent_sub_avg_loss = sum(all_ent_sub_avg_loss)/len(all_ent_sub_avg_loss) 
        all_ent_loss = sum(all_ent_loss)/len(all_ent_loss) 

        ret['all_ent_sub_samp_loss'] = all_ent_sub_samp_loss
        ret['all_ent_sub_avg_loss'] = all_ent_sub_avg_loss
        ret['all_ent_loss'] = all_ent_loss

        return ret
    
    def quantize(self, x, return_latent=False):
        all_indices = []
        quantized_out = 0.
        residual = x
        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):

            quantized, *rest = layer(residual, return_idx=True) #single quantizer

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx

    def get_sum_residual_features(self, residual_codes):

        bs, n_seq, n_q, dim = residual_codes.shape # one-hot
        residual_codes = residual_codes.float()
        residual_feats = None

        for i_q in range(n_q):
            
            one_hot = residual_codes[:, :, i_q].reshape(-1, dim) # (bs * n_seq, n_code)
            value = self.layers[i_q].get_codebook_value_emb() # (n_code, dim)

            z_q = einsum('b n, n d -> b d', one_hot, value)

            z_q = z_q.reshape(bs, n_seq, -1).permute(0, 2, 1).contiguous() # (bs, dim, n_seq)

            if residual_feats is None:
                residual_feats = z_q
            else:
                residual_feats = residual_feats + z_q
        
        return residual_feats