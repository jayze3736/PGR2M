from .base_loss_wrapper import BaseLossWrapper 
from .motion_process import recover_from_ric
from .losses import *
from .misc import T2M_ID2JOINTNAME, T2M_CONTACT_JOINTS
from .trainUtil import gradient_based_dynamic_weighting
import torch
from posescript.code_extractor import CodeExtractor

class FocalFrequencyLossWrapper(BaseLossWrapper):
    def __init__(self, loss_weight=1.0, alpha=1.0, ave_spectrum=False, log_matrix=False, batch_matrix=False, use_in_loss=True):
        self.ffl = FocalFrequencyLoss(loss_weight, alpha, ave_spectrum, log_matrix, batch_matrix)
        super().__init__(self.ffl, use_in_loss, weight={})
        
    def update(self, en_feat, de_feat, device):
        
        loss = torch.tensor([0.], requires_grad = True, device=device)
        loss_terms = []
        for i in range(len(en_feat)):
            loss = loss + self.ffl(de_feat[i], en_feat[i])
            loss_terms.append(self.ffl(de_feat[i], en_feat[i]))

        loss = loss / len(en_feat) 
        self.avg_loss += self.weight['loss_ffl'] * loss

        losses = {} 
        losses['ffl'] = loss

        return losses
    
    def state(self):
        return {
            'FFL': self.avg_loss,
        }

    def reset(self):
        self.avg_loss = 0
        
    def return_weights(self):
        return self.weight
    
    def __str__(self):
        return 'focal_frequency_loss'

class HTDLossWrapper(BaseLossWrapper):
    def __init__(self, weight={}, use_in_loss=True, mocons_loss_type='l1_smooth', pocons_loss_type='l1_smooth', unit_length=4, dataname='t2m'):
        self.loss = HTDLoss(mocons_loss=mocons_loss_type, pocons_loss=pocons_loss_type, unit_length=unit_length, dataname=dataname)
        super().__init__(self.loss, use_in_loss, weight)
        self.avg_pose_pair_loss = 0.
        self.unit_length = unit_length
        self.joints_num = 21 if dataname == 'kit' else 22

    def update(self, pred_motion, pred_rel_pose, gt_motion):
        # forward
        
        mocons_loss = self.weight['loss_mocons'] * self.loss(pred_motion, gt_motion)
        pocons_loss = self.weight['loss_pocons'] * self.loss.forward_pose_diff(pred_rel_pose, gt_motion[:, ::self.unit_length, 4:4+3 * (self.joints_num - 1)])
        self.avg_loss += mocons_loss
        self.avg_pose_pair_loss += pocons_loss

        losses = {} 
        losses['loss_motion_recons'] = mocons_loss
        losses['loss_pose_pair'] = pocons_loss

        return losses
    
    def state(self):
        return {
            'Loss_Motion_Reconstruction': self.avg_loss,
            'Loss_Pose_Pair': self.avg_pose_pair_loss,
        }

    def reset(self):
        self.avg_loss = 0
        self.avg_pose_pair_loss = 0

    def return_weights(self):
        return self.weight
    
    def __str__(self):
        return 'htd_loss'

class DisentangleLossBatchWrapper(BaseLossWrapper):
    def __init__(self, group_num=70, weight={}, use_in_loss=True):
        self.loss = DisentangleLossBatch(group_num)
        super().__init__(self.loss, use_in_loss, weight)

    def update(self, pose_code, codebook):
        # forward
        
        loss = self.weight['loss_disentangle_batch'] * self.loss(pose_code, codebook)
        self.avg_loss += loss

        losses = {} 
        losses['loss_disentangle_batch'] = loss

        return losses
    
    def state(self):
        return {
            'Loss_Disentangle_Batch': self.avg_loss,
        }

    def reset(self):
        self.avg_loss = 0
        
    def return_weights(self):
        return self.weight
    
    def __str__(self):
        return 'disentangle_loss_batch'

class DisentangleLossWrapper(BaseLossWrapper):
    def __init__(self, weight={}, use_in_loss=True):
        self.loss = DisentangleLoss()
        super().__init__(self.loss, use_in_loss, weight)

    def update(self, codebook):
        # forward
        
        loss = self.weight['loss_disentangle'] * self.loss(codebook)
        self.avg_loss += loss

        losses = {} 
        losses['loss_disentangle'] = loss

        return losses
    
    def state(self):
        return {
            'Loss_Disentangle': self.avg_loss,
        }

    def reset(self):
        self.avg_loss = 0
        
    def return_weights(self):
        return self.weight
    
    def __str__(self):
        return 'disentangle_loss'

class GroupAwareContrastiveLossWrapper(BaseLossWrapper):
    def __init__(self, m_pos, m_neg_sim, lam_neg, weight={}, use_in_loss=True):
        self.loss = GroupAwareContrastiveLoss(m_pos, m_neg_sim, lam_neg)
        super().__init__(self.loss, use_in_loss, weight)

    def update(self, codebook):
        # forward
        
        loss = self.weight['loss_contrastive'] * self.loss(codebook)
        self.avg_loss += loss

        losses = {} 
        losses['loss_contrastive'] = loss

        return losses
    
    def state(self):
        return {
            'Loss_Contrastive': self.avg_loss,
        }

    def reset(self):
        self.avg_loss = 0
        
    def return_weights(self):
        return self.weight
    
    def __str__(self):
        return 'contrastive_loss'

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

class ReconsJointFormatLossWrapper(BaseLossWrapper):
    def __init__(self, recons_loss, nb_joints, weight={}, vel_mode='v1', mean_by_sample=None, use_in_loss=True):
        self.loss = ReCons_Joint_Format_Loss(recons_loss, nb_joints)
        super().__init__(self.loss, use_in_loss, weight)
        self.recons_loss = recons_loss
        self.nb_joints = nb_joints
        self.mean_by_sample = mean_by_sample
        self.vel_mode = vel_mode

    def inv_transform(self, data, mean, std):
        return data * std + mean
        
    def update(self, val_loader, pred, gt, mask=None):
            
        # if h3d format
        if pred.dim() == 3:
            mean = torch.from_numpy(val_loader.dataset.mean).to(pred.device)
            std = torch.from_numpy(val_loader.dataset.std).to(pred.device)

            pred_denorm = self.inv_transform(pred, mean, std)
            pred = recover_from_ric(pred_denorm.float().cuda(), joints_num=self.nb_joints)

            gt_denorm = self.inv_transform(gt, mean, std)
            gt = recover_from_ric(gt_denorm.float().cuda(), joints_num=self.nb_joints)

        # forward
        
        loss = self.weight['loss_recons'] * self.loss(pred, gt, mask, self.mean_by_sample)
        self.avg_loss += loss
        
        # vel_loss = self.loss.forward_vel(pred, gt, self.vel_mode, mask, self.mean_by_sample)
        # self.avg_vel_loss += self.weight['loss_vel'] * vel_loss

        losses = {} 
        losses['recons_jfl'] = loss
        # losses['recons_jfl_vel'] = vel_loss

        return losses
    
    def state(self):
        return {
            'Loss_Recons(Joint_Format)': self.avg_loss,
            # 'Loss_Vel(Joint_Format)': self.avg_vel_loss
        }

    def reset(self):
        self.avg_loss = 0
        self.avg_vel_loss = 0
        
    def return_weights(self):
        return self.weight

    def __str__(self):
        return 'recons_jfl'


class ReconsJointWiseLossWrapper(BaseLossWrapper):
    def __init__(self, recons_loss, nb_joints, mode, weight={}, vel_mode='v1', use_in_loss=True):
        self.loss = JointWise_ReConsLoss(recons_loss, nb_joints, contact_joints=T2M_CONTACT_JOINTS)
        super().__init__(self.loss, use_in_loss, weight)
        self.recons_loss = recons_loss
        self.nb_joints = nb_joints
        self.vel_mode = vel_mode

        self.avg_joint_loss = {}
        self.avg_joint_vel_loss = {}

        for id, name in T2M_ID2JOINTNAME.items():
            self.avg_joint_loss[name] = 0.
            self.avg_joint_vel_loss[name] = 0. 
            
    def update(self, pred, gt, mask=None):
        # forward
        
        joint_losses = self.loss(pred, gt, mask, out_list=False)
        joint_vel_losses = self.loss.forward_vel(pred, gt, self.vel_mode, mask, out_list=False)

        # for id, name in T2M_ID2JOINTNAME.items():
        #     self.avg_joint_loss[name] += self.weight['loss_recons'] * joint_losses[id]
        #     self.avg_joint_vel_loss[name] += self.weight['loss_vel'] * joint_vel_losses[id]

        # losses = {} 
        # losses['recons_jwl'] = joint_losses
        # losses['recons_jwl_vel'] = joint_vel_losses

        return joint_losses, joint_vel_losses
    
    def state(self):
        raise NotImplementedError()
        return {
            'Loss_Recons': self.avg_joint_loss,
            'Loss_Vel': self.avg_joint_vel_loss
        }
    
    def reset(self):
        raise NotImplementedError()
        self.avg_joint_loss = 0
        self.avg_joint_vel_loss = 0
        
    def return_weights(self):
        return self.weight
    
    def __str__(self):
        return 'recons_jwl'


class ReconsJointGroupLossWrapper(BaseLossWrapper):
    def __init__(self, recons_loss, nb_joints, mode, weight={}, vel_mode='v1', mean_by_sample=None, enable_dynamic_weights=False, disable_vel_loss=False, norm_dynamic_weight=False, use_in_loss=True):
        self.loss = JointGroup_ReConsLoss(recons_loss, nb_joints, contact_joints=T2M_CONTACT_JOINTS)
        super().__init__(self.loss, use_in_loss, weight)
        self.recons_loss = recons_loss
        self.nb_joints = nb_joints
        self.mean_by_sample = mean_by_sample
        self.vel_mode = vel_mode
        self.enable_dynamic_weights = enable_dynamic_weights
        self.disable_vel_loss = disable_vel_loss
        self.norm_dynamic_weight = norm_dynamic_weight

        self.avg_recons_plv = 0.
        self.avg_recons_jt = 0.
        self.avg_recons_cjt = 0.

        if not self.disable_vel_loss:
            self.avg_recons_vel_plv = 0.
            self.avg_recons_vel_jt = 0.
            self.avg_recons_vel_cjt = 0.
        
    def update(self, net, pred, gt, mask=None, validation=False):

        loss_list = []

        # forward
        group_name, loss_joint_groups = self.loss(pred, gt, mask, self.mean_by_sample)
        loss_names = ['loss_plv', 'loss_jt', 'loss_cjt']

        # unpack losses
        for i, name in enumerate(loss_names):
            loss = loss_joint_groups[i]
            loss_list.append(loss)

        if not self.disable_vel_loss:
            group_name, loss_vel_joint_groups = self.loss.forward_vel(pred, gt, self.vel_mode, mask, self.mean_by_sample)
            for i, name in enumerate(loss_names):
                loss = loss_vel_joint_groups[i]
                loss_list.append(loss)

        # dynamic weighting
        if validation:
            if not self.disable_vel_loss:
                weights = [1.0 for _ in range(len(loss_names) * 2)]
            else:
                weights = [1.0 for _ in range(len(loss_names))]
        else:
            weights = gradient_based_dynamic_weighting(net, loss_list, self.norm_dynamic_weight)
            

        # apply weights
        if self.enable_dynamic_weights:
            for i, name in enumerate(loss_names):
                self.weight[name] = weights[i]
                if not self.disable_vel_loss:
                    self.weight[f'{name}_vel'] = weights[i + 3]

        # accumulate weighted losses
        self.avg_recons_plv += self.weight['loss_plv'] * loss_joint_groups[0]
        self.avg_recons_jt  += self.weight['loss_jt']  * loss_joint_groups[1]
        self.avg_recons_cjt += self.weight['loss_cjt'] * loss_joint_groups[2]

        if not self.disable_vel_loss:
            self.avg_recons_vel_plv += self.weight['loss_plv_vel'] * loss_vel_joint_groups[0]
            self.avg_recons_vel_jt  += self.weight['loss_jt_vel']  * loss_vel_joint_groups[1]
            self.avg_recons_vel_cjt += self.weight['loss_cjt_vel'] * loss_vel_joint_groups[2]

        weighted_losses = {}
        for i, name in enumerate(loss_names):
            weighted_losses[name] = (self.weight[name] * loss_list[i])
            if not self.disable_vel_loss:
                weighted_losses[f'{name}_vel'] = (self.weight[f'{name}_vel'] * loss_list[i + 3])
        
        return weighted_losses
    
    def state(self):

        if not self.disable_vel_loss:
            return {
                'Pelvis Feature loss': self.avg_recons_plv,
                'Joint Feature loss': self.avg_recons_jt,
                'Contact Joint Feature loss': self.avg_recons_cjt,
                'Pelvis Feature Vel loss': self.avg_recons_vel_plv,
                'Joint Feature Vel loss': self.avg_recons_vel_jt,
                'Contact Joint Vel Feature': self.avg_recons_vel_cjt,
            }
        else:
            return {
                'Pelvis Feature loss': self.avg_recons_plv,
                'Joint Feature loss': self.avg_recons_jt,
                'Contact Joint Feature loss': self.avg_recons_cjt
            }
        
    def reset(self):
        self.avg_recons_plv = 0.
        self.avg_recons_jt = 0.
        self.avg_recons_cjt = 0.

        self.avg_recons_vel_plv = 0.
        self.avg_recons_vel_jt = 0.
        self.avg_recons_vel_cjt = 0.
    
    def return_weights(self):
        return self.weight
    
    def __str__(self):
        return 'recons_jgl'


class GroupWiseL1LossWrapper(BaseLossWrapper):
    def __init__(self, recons_loss, weight={}, mean_by_frame=None, use_in_loss=True):
        self.loss = GroupWiseL1Loss(recons_loss)
        super().__init__(self.loss, use_in_loss, weight)
        self.recons_loss = recons_loss
        self.mean_by_frame = mean_by_frame
        self.centroid_gram_mat = None
        self.code_ext = CodeExtractor('t2m')
        
    def update(self, pred_motion, gt_codes, mask=None):
        # forward
        

        pred_codes = self.code_ext(pred_motion)

        loss = self.weight['group_wise_loss'] * self.loss(pred_codes, gt_codes, mask, self.mean_by_frame)
        self.avg_loss += loss

        losses = {} 
        losses['group_wise_loss'] = loss
        
        return losses
    
    def state(self):
        return {'Loss_Groupwise_Recons': self.avg_loss}

    def reset(self):
        self.avg_loss = 0
        
    def return_weights(self):
        return self.weight

    def __str__(self):
        return 'group_wise_loss'

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
