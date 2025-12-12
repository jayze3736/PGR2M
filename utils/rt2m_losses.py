import torch
import torch.nn as nn
import torch.nn.functional as F

class ReConsT2MLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, vel_mode='v1'):
        super(ReConsT2MLoss, self).__init__()
        
        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss(reduction="none")
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss(reduction="none")
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction="none")
        else:
            raise ValueError(f"Invalid recons_loss: {recons_loss}")
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4  # (nb_joints-1)*12 + 4 + 3 + 4
        self.vel_mode = vel_mode

    def forward(self, motion_pred, motion_gt, mask=None, mean_by_sample=False):
        """
        motion_pred, motion_gt: (n_seq, dim)
        """
        loss = self.Loss(motion_pred[..., :self.motion_dim],
                         motion_gt[..., :self.motion_dim])
        
        stepwise_loss = loss.mean(dim=-1)

        if mask is not None:    
            stepwise_loss = stepwise_loss * mask
        
        if mean_by_sample:
            final_loss = stepwise_loss.mean(dim=-1)
        else:
            # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
            final_loss = stepwise_loss.mean()

        return final_loss


class CEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(CEWithLogitsLoss, self).__init__()
        # self.margin = margin
        # 카테고리 그룹 중심 별

    def forward(self, pred, target, ignore_index=None):
        # logits: (B, T, C), gt: (B, T) - gt가 one-hot 또는 soft label인 경우
        
        # class index로 변환
        target_index = target

        # reshape for cross entropy
        pred_reshaped = pred.reshape(-1, pred.size(-1))  # (B*T, C)
        target_reshaped = target_index.reshape(-1)  # (B*T,)

        loss = F.cross_entropy(pred_reshaped, target_reshaped, ignore_index=ignore_index)

        # mask로 적용 필요
        return loss
        
