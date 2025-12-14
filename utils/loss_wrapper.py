from .base_loss_wrapper import BaseLossWrapper 
from .motion_process import recover_from_ric
from .losses import *
from .misc import T2M_ID2JOINTNAME, T2M_CONTACT_JOINTS
from .trainUtil import gradient_based_dynamic_weighting
import torch

class ReConsLossWrapper(BaseLossWrapper):
    def __init__(self, recons_loss, nb_joints, weight={}, vel_mode='v1', mean_by_sample=None, use_in_loss=True):
        self.loss = ReConsLoss(recons_loss, nb_joints, vel_mode)
        super().__init__(self.loss, use_in_loss, weight)
        self.recons_loss = recons_loss
        self.nb_joints = nb_joints
        self.mean_by_sample = mean_by_sample
        self.vel_mode = vel_mode
        
    def update(self, pred, gt, mask=None):
        # forward
        
        loss = self.weight['loss_recons'] * self.loss(pred, gt, mask, self.mean_by_sample)
        self.avg_loss += loss
        
        vel_loss = self.weight['loss_vel'] * self.loss.forward_vel(pred, gt, mask, self.mean_by_sample)
        self.avg_vel_loss += vel_loss

        losses = {} 
        losses['recons'] = loss
        losses['recons_vel'] = vel_loss

        return losses
    
    def state(self):
        return {
            'Loss_Recons': self.avg_loss,
            'Loss_Vel': self.avg_vel_loss
        }

    def reset(self):
        self.avg_loss = 0
        self.avg_vel_loss = 0
        
    def return_weights(self):
        return self.weight
    
    def __str__(self):
        return 'recons'

class OrthogonalLossWrapper(BaseLossWrapper):
    def __init__(self, mode='group', codebook_mode='v1', weight={}, use_in_loss=True):
        self.loss = OrthogonalLoss(mode, codebook_mode)
        super().__init__(self.loss, use_in_loss, weight)

        
    def update(self, codebook):
        # forward
        
        loss_orthogonal, self.centroid_gram_mat = self.loss(codebook)
        loss = self.weight['ortho_loss'] * loss_orthogonal
        self.avg_loss += loss
    
        losses = {} 
        losses['orthogonal_loss'] = loss
        
        return losses
    
    def state(self):
        return {'Loss_Codebook_Orthogonal': self.avg_loss}
    
    def reset(self):
        self.avg_loss = 0
        
    def return_weights(self):
        return self.weight

    def __str__(self):
        return 'orthogonal_loss'
    
    def gram_matrix(self):
        return self.centroid_gram_mat
