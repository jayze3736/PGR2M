import torch
import torch.nn as nn
from posescript.coding import Coder

class PoseCoder(nn.Module):
    def __init__(self, dataname, device):
        super().__init__()
        self.dataname = dataname
        self.device = device
        
        if self.dataname == 't2m':
            self.n_joint = 22 # 
        elif self.dataname == 'kit':
            self.n_joint = 21
        else:
            raise NotImplementedError()
        
        self.posecoder = Coder(dataname)
        

    @torch.no_grad()
    def forward(self, x):
        
        x = x.to(self.device)
        kh_pose_codes = []

        if x.dim() == 4: # if raw motion is entered
            print("## raw motion detected convert to pose code")
            bs, seq_len, n_joint, _ = x.shape

            
            if n_joint == self.n_joint:
                # normalize sacle 
                scale = 1

                R = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32).to(self.device)

                for b in range(bs):
                    
                    if self.dataname == "kit":
                        x[b] = self.posecoder.format_kit(x[b])

                    x[b] = x[b].reshape((-1,22,3))

                    for frame in range(seq_len):
                        x[b, frame] = (R @ x[b, frame].T).T # x[b, frame] = pose data
                        # x[b, frame] = (np.array([[-1,0,0],[0,1,0],[0,0,-1]]) @ x[b, frame].T).T

                    # pred_xyz
                    x[b] = x[b] / scale

                    _oh = torch.tensor(self.posecoder.motion2poseqs(x[b]), dtype=torch.float32).to(self.device) 
                    kh_pose_codes.append(_oh) # bs, n_seq, n_cat

                    kh_pose_codes = torch.stack(kh_pose_codes)
                    print("completed")
            else:
                raise Exception("## Error -> Wrong skeleton format ##")
            
        elif x.dim() == 3:
            kh_pose_codes = x

        return kh_pose_codes # (bs, selected_cat_num)
    
    def k_hot2_index(self, kh_pose_codes):
        # kh_pose_codes = kh_pose_codes.clone()
        bs, seq_len, n_cat = kh_pose_codes.shape 
        
        _sum = kh_pose_codes.sum(dim=-1) 

        uq_values = _sum.unique()

        assert len(uq_values) == 2 # uq_values should 1 and other number

        sel_cat = int(max(uq_values).item()) 

        mask_idx = (_sum == 1).bool() 
        code_seqs = torch.arange(n_cat, device=self.device).repeat(bs, seq_len, 1) 

        dummy_tensor = torch.tensor(([0] * (n_cat - sel_cat) + [1] * sel_cat), dtype=bool, device=self.device) 
        kh_pose_codes = kh_pose_codes.bool() 
        kh_pose_codes[mask_idx, :] = dummy_tensor.repeat(mask_idx.sum(), 1) # advanced indexing & insert dummy tensor for fitting the number of selected categories
        
        code_seqs = code_seqs[kh_pose_codes] # masking
        code_seqs = code_seqs.view(bs, seq_len, -1) # bs, seq_len, sel_cat 
        
        # pad_tokens = torch.tensor([n_cat - 1] * sel_cat, dtype=code_seqs.dtype, device=self.device) # create pad tokens
        pad_tokens = torch.tensor([n_cat - 1] * sel_cat, dtype=code_seqs.dtype, device=self.device) # create pad tokens
        code_seqs[mask_idx, :] = pad_tokens.repeat(mask_idx.sum(), 1)

        return code_seqs
    

    def index_2_k_hot(self, x, num_classes):
        bs, seq_len, n_sel_cat = x.shape 
        k_hot = torch.zeros(size=(bs, seq_len, num_classes + 2), dtype=torch.float, device=self.device)
        k_hot.scatter_(dim=2, index=x, value=1.0)
        
        return k_hot