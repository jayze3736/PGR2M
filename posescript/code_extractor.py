import torch
import torch.nn as nn
from posescript.coding import Coder
from utils.motion_process import recover_from_ric
import numpy as np
from os.path import join as pjoin

class CodeExtractor(nn.Module):
    def __init__(self, dataname):
        super(CodeExtractor, self).__init__()
        self.dataname = dataname
        
        if self.dataname == 't2m':
            self.n_joint = 22 #
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif self.dataname == 'kit':
            self.n_joint = 21
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta' 

        else:
            raise NotImplementedError()

        self.mean = torch.from_numpy(np.load(pjoin(self.meta_dir, 'mean.npy')))
        self.std = torch.from_numpy(np.load(pjoin(self.meta_dir, 'std.npy')))
        
        self.posecoder = Coder(dataname)

    def extract_pose_code(self, motion):
        device = motion.device

        bs, seq_len, n_joint, _ = motion.shape    
        kh_pose_codes = []
        
        if n_joint == self.n_joint:
            # normalize sacle 
            scale = 1

            R = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32).to(device)

            for b in range(bs):
                
                # # kit 일 경우
                if self.dataname == "kit":
                    motion[b] = self.posecoder.format_kit(motion[b])

                motion[b] = motion[b].reshape((-1,22,3))

                for frame in range(seq_len):
                    motion[b, frame] = (R @ motion[b, frame].T).T # motion[b, frame] = pose data
                    # motion[b, frame] = (np.array([[-1,0,0],[0,1,0],[0,0,-1]]) @ motion[b, frame].T).T

                # pred_xyz
                motion[b] = motion[b] / scale

                _oh = torch.tensor(self.posecoder.motion2poseqs(motion[b], device=device), dtype=torch.float32) # .to(device) 
                
                kh_pose_codes.append(_oh) # bs, n_seq, n_cat

            output = torch.stack(kh_pose_codes)

            return output
        else:
            raise Exception("## Error -> Wrong skeleton format ##")

    def forward(self, motion):
        if motion.dim() == 4: # if raw motion is entered
            output = self.extract_pose_code(motion)
        elif motion.dim() == 3: # h3d format -> joint representation
            
            pred_denorm = self.inv_transform(motion)
            
            pred_xyz = recover_from_ric(pred_denorm.float(), self.n_joint)
                   
            output = self.extract_pose_code(pred_xyz) 
        else:
            raise Exception()
        
        return output

    def inv_transform(self, data):
        device = data.device
        self.std = self.std.to(device)
        self.mean = self.mean.to(device)
        return data * self.std + self.mean