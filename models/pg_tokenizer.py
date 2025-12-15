import torch.nn as nn
import torch
import torch.nn.functional as F
from models.vq.residual_vq import ResidualVQ
from models.vq.encdec import Encoder
from models.vq.encdec import Decoder
from models.vq.pose_code_enc import PoseCodeEncoder

class PoseGuidedTokenizer(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 num_quantizers=3,
                 shared_codebook=False,
                 quantize_dropout_prob=0.2,
                 quantize_dropout_cutoff_index=0,
                 rvq_nb_code=64,           
                 mu=0.99,
                 residual_ratio=1.0,
                 vq_loss_beta=1.0,
                 quantizer_type='hard',
                 params_soft_ent_loss=0.0,
                 use_ema=True,
                 init_method='enc',  # 'enc', 'xavier', 'uniform',
                 ):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.rvq_nb_code = rvq_nb_code
        self.nb_joints = 21 if args.dataname == 'kit' else 2
        self.residual_ratio = residual_ratio

        self.encoder = Encoder(251 if args.dataname == 'kit' else 263, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(251 if args.dataname == 'kit' else 263, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.pc_encoder = PoseCodeEncoder(args, nb_code=nb_code, code_dim=code_dim)

        # ric_pos_dim = 3 * (self.nb_joints - 1)
        # self.proj = nn.Linear(code_dim, ric_pos_dim) # 512 -> rel pos dim
        self.vq_loss_beta = vq_loss_beta
        
        # rvq
        if quantizer_type == 'soft':
            self.rvq = ResidualVQ(
                num_quantizers=num_quantizers,
                shared_codebook=shared_codebook,
                quantize_dropout_prob=quantize_dropout_prob,
                quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
                nb_code=rvq_nb_code,
                code_dim=self.code_dim,
                attn_dim=self.code_dim,
                quantizer_type=quantizer_type,
                beta=self.vq_loss_beta,
                norm_type='rms_norm',
                entropy_temperature=0.01, 
                sample_minimization_weight=1.0,
                batch_maximization_weight=1.0,
                params_soft_ent_loss=params_soft_ent_loss,
                use_ema=use_ema,
                init_method=init_method
            )
        else:
            self.rvq = ResidualVQ(
                num_quantizers=num_quantizers,
                shared_codebook=shared_codebook,
                quantize_dropout_prob=quantize_dropout_prob,
                quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
                nb_code=rvq_nb_code,
                code_dim=self.code_dim,
                mu=mu,
                quantizer_type=quantizer_type
            )

    def tokenize(self, motion=None, code_indices = None, detach_p_latent = False, return_one_hot=False):

        with torch.no_grad():
            p_latent = self.pc_encoder(code_indices)
            B, T, C = p_latent.shape
            
            z_latent = self.encoder(motion)

            if detach_p_latent:
                r_latent = z_latent - p_latent.detach()    
            else:
                r_latent = z_latent - p_latent 

            r_latent = r_latent.view(B, T, -1).permute(0, 2, 1).contiguous()
            p_latent = p_latent.view(B, T, -1).permute(0, 2, 1).contiguous()

            # r_quatized, indices, loss, perplexity, all_codes = self.rvq(r_latent, return_all_codes=True, sample_codebook_temp=1.0)
            out = self.rvq(r_latent, return_all_codes=True, sample_codebook_temp=1.0)

            if return_one_hot:
                out['all_indices'] = F.one_hot(out['all_indices'], num_classes=self.rvq_nb_code).float()

        return out['all_codes'], out['all_indices']
    
    def inference(self, residual_codes=None, code_indices = None, drop_out_residual_quantization=False):
        p_latent = self.pc_encoder(code_indices)
        B, T, C = p_latent.shape

        p_latent = p_latent.view(B, T, -1).permute(0, 2, 1).contiguous()

        r_latent = self.rvq.get_sum_residual_features(residual_codes)

        if drop_out_residual_quantization:
            z_c_latent = p_latent
        else:
            z_c_latent = p_latent + self.residual_ratio * r_latent

        x_decoder = self.decoder(z_c_latent)
        x_out = self.postprocess(x_decoder)
        
        return x_out

    def forward(self, code_indices, motion=None, detach_p_latent = False, drop_out_residual_quantization=False, force_dropout_index=-1):
    
        p_latent = self.pc_encoder(code_indices)
        B, T, C = p_latent.shape
        
        if drop_out_residual_quantization:
            out = {}
            p_latent = p_latent.view(B, T, -1).permute(0, 2, 1).contiguous()
            z_c_latent = p_latent
        else:
            z_latent = self.encoder(motion)

            if detach_p_latent:
                r_latent = z_latent - p_latent.detach() 
            else:
                r_latent = z_latent - p_latent 

            r_latent = r_latent.view(B, T, -1).permute(0, 2, 1).contiguous()
            p_latent = p_latent.view(B, T, -1).permute(0, 2, 1).contiguous()

            # Retrieve latent representation as linear combination of codes
            # r_quatized, indices, loss, perplexity, all_codes = self.rvq(r_latent, return_all_codes=True, sample_codebook_temp=1.0)
            out = self.rvq(r_latent, return_all_codes=True, sample_codebook_temp=1.0, force_dropout_index=force_dropout_index)

            r_quatized = out['quantized_out']

            z_c_latent = p_latent + self.residual_ratio * r_quatized
        
        ## decoder

        x_decoder = self.decoder(z_c_latent)
        x_out = self.postprocess(x_decoder)
        
        return x_out, self.pc_encoder.codebook.weight, out
        
    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        # x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x




