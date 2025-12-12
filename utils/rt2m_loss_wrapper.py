from .base_loss_wrapper import BaseLossWrapper 
from .rt2m_losses import ReConsT2MLoss, CEWithLogitsLoss
import torch

class ReConsT2MLossWrapper(BaseLossWrapper):
    def __init__(self, cat_mode='v1', recons_loss='l1_smooth', nb_joints=22, weight={}, use_in_loss=True):
        self.loss = ReConsT2MLoss(recons_loss, nb_joints, cat_mode)
        super().__init__(self.loss, use_in_loss, weight)
        
    def update(self, pred, target, mask=None):
        # forward
        val_loss = self.loss(pred, target)

        loss = self.weight['loss_recons_t2m'] * val_loss
        self.avg_loss += loss

        losses = {} 
        losses['recons_t2m'] = loss

        return losses
    
    def state(self):
        return {
            'Loss_ReConsT2MLoss': self.avg_loss,
        }

    def reset(self):
        self.avg_loss = 0
        
    def return_weights(self):
        return self.weight

    def __str__(self):
        return 'recons_t2m'
    

class CEWithLogitsLossWrapper(BaseLossWrapper):
    def __init__(self, weight={}, use_in_loss=True):
        self.loss = CEWithLogitsLoss()
        super().__init__(self.loss, use_in_loss, weight)
        
    def update(self, pred, target, ignore_index=None):
        # forward
        # vel loss를 사용할 경우
        val_loss = self.loss(pred, target, ignore_index=ignore_index)

        loss = self.weight['loss_ce'] * val_loss
        self.avg_loss += loss

        losses = {} 
        losses['ce'] = loss

        return losses
    
    def state(self):
        return {
            'Loss_CEWithLogits': self.avg_loss,
        }

    def reset(self):
        self.avg_loss = 0
        
    def return_weights(self):
        return self.weight

    def __str__(self):
        return 'ce'
    
