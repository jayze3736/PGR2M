
import torch.nn as nn
import torch

class PoseCodeEncoder(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512):
        super().__init__()
        
        self.code_dim = code_dim
        self.num_code = nb_code
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        
        # pose codebook
        self.codebook = nn.Embedding(self.num_code, self.code_dim)
        # Initialized random/uniformly
        self.codebook.weight.data.uniform_(-1.0 / self.num_code, 1.0 / self.code_dim)

        # channel positional encoding 추가
        
    def forward(self, code_indices):
        # x shape: Bs, seq_len, category_num

        B, T, C = code_indices.shape # bs, T, code_num

        codes_flattened = code_indices.contiguous().view(-1, self.num_code) # REC: 사전에 저장된 pose code를 flatten

        # Retrieve latent representation as linear combination of codes
        z = torch.matmul(codes_flattened, self.codebook.weight).view((-1, self.code_dim))
        z = z.view(B, T, -1).permute(0, 2, 1).contiguous()

        # z_out = self.postprocess(z)
        z_out = z
        z_out = self.postprocess(z_out)
        return z_out


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        # x = x.permute(0, 2, 1)a
        # make complete frame wise mask
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x
