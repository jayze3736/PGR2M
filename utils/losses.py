import torch
import torch.nn as nn
from .misc import kh2index
from .misc import JOINT_GROUP, T2M_ID2JOINTNAME, T2M_CONTACT_JOINTS
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# from utils.codebook import *
import torch_dct as dct

IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

"""
쉽게 얘기하면, 데이터를 주파수 도메인으로 변환하고, 실수축, 허수축으로 만들어진 주파수 도메인에서의 벡터 distance가 원본과 가까워지도록 학습시키는 loss 함수
"""
class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, alpha=1.0, ave_spectrum=False, log_matrix=False, batch_matrix=False, mode='dct', method='l1'):
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = alpha
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.mode = mode
        
        if method == 'l1_smooth':
            self.loss = torch.nn.SmoothL1Loss(reduction="none")
        elif method == 'l1':
            self.loss = torch.nn.L1Loss(reduction="none")
        else:
            self.loss = torch.nn.MSELoss(reduction="none")

    def tensor2freq(self, x):
        x = x.permute(0, 2, 1)  # (B, D, C) -> (B, C, D)
        _, n_seq, dim = x.shape
        
        if self.mode == 'dct':
            freq = dct.dct(x, norm='ortho') # (B, F, C)
        elif self.mode == 'fft':
            freq = torch.fft.rfft(x, dim=1, norm='ortho') # (B, F//2, C)
            freq = torch.stack([freq.real, freq.imag], -1)
        
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            if self.mode == 'dct':

                # recon_freq: (B, F, C), real_freq: (B, F, C)
                matrix_tmp = torch.abs(recon_freq - real_freq) ** self.alpha
                # whether to adjust the spectrum weight matrix by logarithm
                if self.log_matrix:
                    matrix_tmp = torch.log(matrix_tmp + 1.0)

                # whether to calculate the spectrum weight matrix using batch-based statistics
                if self.batch_matrix:
                    matrix_tmp = matrix_tmp / matrix_tmp.max()
                else:
                    # Normalize across frequency and channel dimensions for each batch item
                    matrix_tmp = matrix_tmp / matrix_tmp.flatten(1).max(dim=1)[0].view(-1, 1, 1)

                matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
                matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
                weight_matrix = matrix_tmp.clone().detach()

                assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                    'The values of spectrum weight matrix should be in the range [0, 1], '
                    'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

            elif self.mode == 'fft':
                # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
                matrix_tmp = (recon_freq - real_freq) ** 2
                matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

                # whether to adjust the spectrum weight matrix by logarithm
                if self.log_matrix:
                    matrix_tmp = torch.log(matrix_tmp + 1.0)

                # whether to calculate the spectrum weight matrix using batch-based statistics
                if self.batch_matrix:
                    matrix_tmp = matrix_tmp / matrix_tmp.max()
                else:
                    matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, None, None]

                matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
                matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
                weight_matrix = matrix_tmp.clone().detach()

                assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                    'The values of spectrum weight matrix should be in the range [0, 1], '
                    'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))
            

        # frequency distance using (squared) Euclidean distance
        if self.mode == 'dct':
            # freq_distance = (recon_freq - real_freq) ** 2
            freq_distance = self.loss(recon_freq, real_freq)
        elif self.mode == 'fft':
            tmp = (recon_freq - real_freq) ** 2
            freq_distance = tmp[..., 0] + tmp[..., 1]

        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (B, C, D). Predicted tensor.
            target (torch.Tensor): of shape (B, C, D). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        
        # k개의 주파수 성분만 추출할지는 나중에 결정
        
        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        
        return self.loss_formulation(pred_freq, target_freq, matrix)

class HTDLoss(nn.Module):
    def __init__(self, mocons_loss='l1_smooth', pocons_loss='l1', unit_length=4, dataname='t2m'):
        super(HTDLoss, self).__init__()

        if mocons_loss == 'l1':
            self.mocons_loss = torch.nn.L1Loss(reduction="mean")
        elif mocons_loss == 'l2':
            self.mocons_loss = torch.nn.MSELoss(reduction="mean")
        elif mocons_loss == 'l1_smooth':
            self.mocons_loss = torch.nn.SmoothL1Loss(reduction="mean")

        if pocons_loss == 'l1':
            self.pocons_loss = torch.nn.L1Loss(reduction="mean")
        elif pocons_loss == 'l2':
            self.pocons_loss = torch.nn.MSELoss(reduction="mean")
        elif pocons_loss == 'l1_smooth':
            self.pocons_loss = torch.nn.SmoothL1Loss(reduction="mean")
    
        if dataname == 't2m':
            self.joints_num = 22
        elif dataname == 'kit-ml':
            self.joints_num = 21
        
        self.unit_length = unit_length
        self.motion_dim = (self.joints_num - 1) * 12 + 4 + 3 + 4
    
    def forward_pose_diff(self, pred_pose, gt_pose):
        down_gt_pose = gt_pose # ric
        down_pred_pose = pred_pose # ric

        gt_pairwise_pose_diff = down_gt_pose.unsqueeze(2) - down_gt_pose.unsqueeze(1)
        pred_pairwise_pose_diff = down_pred_pose.unsqueeze(2) - down_pred_pose.unsqueeze(1)

        pred_l2_norm_diff = torch.norm(pred_pairwise_pose_diff, p=2, dim=3)
        pred_lower_triangular_l2 = torch.tril(pred_l2_norm_diff)

        gt_l2_norm_diff = torch.norm(gt_pairwise_pose_diff, p=2, dim=3)
        gt_lower_triangular_l2 = torch.tril(gt_l2_norm_diff)

        pocons_loss = self.pocons_loss(gt_lower_triangular_l2, pred_lower_triangular_l2) # 다운 샘플링된 위치에서, 포즈가 동일해야 함
        return pocons_loss

    def forward(self, pred_motion, gt_motion):
        loss = self.mocons_loss(pred_motion[..., :self.motion_dim],
                         gt_motion[..., :self.motion_dim])
        
        return loss

class DisentangleLossBatch(nn.Module):
    def __init__(self, group_num):
        super(DisentangleLossBatch, self).__init__()

        from utils.codebook import vq_to_range 
        self.vq_to_range = vq_to_range
        self.group_num = group_num

        # 카테고리 그룹 중심 별
    def forward(self, pose_code, codebook):

        B, N, D = pose_code.shape
        pose_code_flat = pose_code.view(B * N, D)

        # codebook의 entry들을 단위 벡터(방향 벡터)로 정규화
        normalized_codebook = F.normalize(codebook, p=2, dim=1)
        
        # 활성화된 k개의 벡터 인덱스 추출
        hot_indices = torch.topk(pose_code_flat, k=self.group_num, dim=1).indices
        indices_expanded = hot_indices.unsqueeze(-1).expand(-1, -1, normalized_codebook.shape[1])
        indices_expanded = indices_expanded.cuda()

        # codebook으로부터 벡터 get
        retrieved_code = normalized_codebook.expand(B*N, -1, -1).gather(1, indices_expanded)
        retrieved_code_transposed = retrieved_code.transpose(1, 2)
        
        # 내적
        inner_product_matrix = torch.matmul(retrieved_code, retrieved_code_transposed)

        # pad mask
        masked_score = inner_product_matrix

        # 평균 
        mean_score = torch.mean(masked_score, dim = 0)
        eye = torch.eye(self.group_num, device=mean_score.device)

        # 자신과는 내적이 어쩌피 1, 나머지들 카테고리와의 내적값이 0에 가까워지도록 loss 설계
        loss = F.l1_loss(mean_score, eye, reduction='sum')

        return loss

class DisentangleLoss(nn.Module):
    def __init__(self):
        super(DisentangleLoss, self).__init__()

        from utils.codebook import vq_to_range 
        self.vq_to_range = vq_to_range
        # self.margin = margin
        
        lens = [(item[0] - item[1]) + 1 for item in vq_to_range.values()]
        lens = lens[:-3]
        self.lengths = torch.tensor(lens)
        starts = torch.cat([torch.tensor([0]), self.lengths.cumsum(0)[:-1]])
        
        self.intervals = [(int(s), int(s+l)) for s, l in zip(starts, self.lengths)]

        # 카테고리 그룹 중심 별
    def forward(self, codebook):
        
        # 방향성에 대해서만 codebook에 대한 제약을 주기
        normalized_codebook = F.normalize(codebook, p=2, dim=1)

        chunks_A = [normalized_codebook[s:e] for (s, e) in self.intervals]                 # [(L_g, 512)] * 70
        A_pad = pad_sequence(chunks_A, batch_first=True)            # (G, L_max, 512)

        chunks_C = [normalized_codebook[s:e] for (s, e) in self.intervals]
        C_pad = pad_sequence(chunks_C, batch_first=True)            # (G, L_max, 512)

        #  주의: L_g << 512 이므로 열-그람(512x512) Cholesky는 실패 -> 행-그람(L_g x L_g) 사용
        Gram_row = A_pad @ A_pad.transpose(-1, -2)                  # (G, L_max, L_max)
        eps = 1e-8
        I = torch.eye(Gram_row.size(-1), device=A_pad.device, dtype=A_pad.dtype)
        L = torch.linalg.cholesky(Gram_row + eps*I)                 # (G, L_max, L_max) lower

        # 행-정규직교 기저 Q_pad (각 행이 512차원, 서로 직교/단위)
        Q_pad = torch.linalg.solve_triangular(L, A_pad, upper=False)   # (G, L_max, 512)

        # --- 마스크 (구간별 유효 행만 사용) ---
        lengths_t = torch.tensor(self.lengths, device=A_pad.device, dtype=torch.long)  # (70,)
        j = torch.arange(A_pad.size(1), device=A_pad.device)                       # (L_max,)
        mask = (j.unsqueeze(0) < lengths_t.unsqueeze(1))                           # (G, L_max) True/False

        # --- 핵심: "자기 vs 타 카테고리" 교차 내적 (512축에 대해)
        # cross[g, i, h, j] = < C_pad[g, i, :], Q_pad[h, j, :] >
        energy = torch.einsum('gip, hjp -> gihj', C_pad, Q_pad)       # (G, L_max, G, L_max)

        # 유효 행만 남기도록 마스킹 (행·열 모두)
        row_mask  = mask[:, :, None, None]   # (G, L_max, 1, 1)
        col_mask  = mask[None, None, :, :]   # (1, 1, G, L_max)
        mask = (row_mask & col_mask).float()

        # 자기 카테고리 제외(대각선 0)
        G = lengths_t.numel()
        offdiag = (~torch.eye(G, dtype=torch.bool, device=A_pad.device)) \
                    .unsqueeze(1).unsqueeze(-1)                      # (G,1,G,1)
        mask = mask * offdiag

        # 에너지(= 투영 크기^2) 총합: 0이면 "자기집합 ⟂ (타카테고리들의 span)"
        loss = (energy.pow(2) * mask).sum()

        return loss


class GroupAwareContrastiveLoss(nn.Module):
    def __init__(self, mode, m_pos=0.5, m_neg_sim=0.1, lam_neg=1.0):
        super(GroupAwareContrastiveLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg_sim = m_neg_sim
        self.lam_neg = lam_neg

        if 'v2' in mode or 'v3' in mode:
            from utils.codebook_v2_v3 import vq_to_range
        elif 'v4' in mode:
            from utils.codebook_v4 import vq_to_range
        elif 'v5' in mode:
            from utils.codebook_v5 import vq_to_range
        elif 'v6' in mode:
            from utils.codebook_v6 import vq_to_range
        elif 'v7' in mode:
            from utils.codebook_v7 import vq_to_range
        else:
            from utils.codebook import vq_to_range 
        
        self.vq_to_range = vq_to_range
        # self.margin = margin

        # 카테고리 그룹 중심 별

    def forward(self, codebook: torch.Tensor, device=None, max_i=70):
        """
        z: (N, d) 원본 임베딩 (정규화 X) — positive는 '거리'만 제약
        vq_to_range[i] -> (start, end): i가 속한 positive 그룹 범위
        m_pos: positive 반지름
        m_ang: negative 직교 허용치 (|cos| <= m_ang)
        lam  : negative 직교 항 가중치
        """
        if device is None: device = codebook.device
        N = codebook.size(0)

        # negative용 코사인 유사도: 정규화된 복사본으로만 계산
        zn = F.normalize(codebook, p=2, dim=-1)
        cos_mat = zn @ zn.T

        loss, cnt = 0.0, 0
        arangeN = torch.arange(N, device=device)

        for i in range(N):
            if i > max_i: break

            end, start = self.vq_to_range[i]
            group_mask = torch.zeros(N, dtype=torch.bool, device=device)
            group_mask[start:end+1] = True
            group_mask[i] = False

            pos_idx = arangeN[group_mask]
            neg_idx = arangeN[~group_mask]

            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            # positive: 거리만 가깝게
            D = (codebook[i] - codebook[pos_idx]).norm(dim=-1)                 # 유클리드 거리
            pos_pull = torch.relu(D - self.m_pos).pow(2).mean()       # 반지름 밖만 당김

            # negative: 직교(코사인 절댓값 ↓)
            neg_cos = cos_mat[i, neg_idx].clamp(-1, 1) # 자기 자신과 negative pair들간의 유사도
            ortho = torch.relu(neg_cos.abs() - self.m_neg_sim).pow(2).mean()

            loss += pos_pull + self.lam_neg * ortho
            cnt += 1

        return loss / cnt if cnt > 0 else torch.tensor(0.0, device=device, requires_grad=True)
        
    # def forward(self, codebook: torch.Tensor):
    #     """
    #     codebook: (N, D) float tensor of code vectors
    #     group_ids: (N,) long tensor, 0~69 group indices
    #     """
    #     N, D = codebook.shape

    #     # sim 기반인데, 거리 기반으로도 해볼 필요가 있을듯
    #     sim_matrix = F.cosine_similarity(codebook[:, None, :], codebook[None, :, :], dim=-1)  # (N, N) -> 392 x 392

    #     loss = 0.0
    #     valid_count = 0
    #     device = codebook.device

    #     for i in range(N):
    #         if i > 70:
    #             break
            
    #         end, start = vq_to_range[i]  # 주의: 보통은 (start, end) 순서입니다
    #         # anchor_group = list(range(start, end + 1))  # 예: [30, 31, ..., 45]

    #         # (N,) boolean mask: anchor_group에 속하는 인덱스만 True
    #         group_mask = torch.zeros(N, dtype=torch.bool, device=device)
    #         group_mask[start:end + 1] = True
    #         group_mask[i] = False

    #         pos_mask = group_mask

    #         # positive mask: 같은 그룹이면서 자기 자신은 제외
    #         # pos_mask = group_mask & (torch.arange(N) != i)

    #         # negative mask: 다른 그룹
    #         neg_mask = ~group_mask  # group_mask의 반전

    #         pos_sims = sim_matrix[i][pos_mask] # positive인 애들과의 거리
    #         neg_sims = sim_matrix[i][neg_mask] # negative인 애들과의 거리
    #         # 둘다 앵커는 i번째 pose code

    #         if len(pos_sims) == 0 or len(neg_sims) == 0:
    #             continue  # skip this anchor

    #         numerator = torch.exp(pos_sims / self.tau).sum()
    #         denominator = torch.exp(torch.cat([pos_sims, neg_sims]) / self.tau).sum()
    #         loss += -torch.log(numerator / denominator)
    #         valid_count += 1

    #     return loss / valid_count



class GroupWiseL1Loss(nn.Module):
    def __init__(self, recons_loss='l1'):
        super(GroupWiseL1Loss, self).__init__()
        
        if recons_loss == 'l1':
            self.loss = torch.nn.L1Loss(reduction="none")
        elif recons_loss == 'l2':
            self.loss = torch.nn.MSELoss(reduction="none")
        elif recons_loss == 'l1_smooth':
            self.loss = torch.nn.SmoothL1Loss(reduction="none")
        else:
            raise ValueError(f"Invalid recons_loss: {recons_loss}")

        # 카테고리 그룹 중심 별
    
    def forward(self, pred_code, gt_code, mask=None, mean_by_frame=False):
        
        pred_indices = kh2index(pred_code).float()
        gt_indices = kh2index(gt_code).float()

        loss = self.loss(pred_indices, gt_indices) # (bs, n_seq, n_cat_group)

        if mean_by_frame:
            final_loss = loss.mean(dim=1)
        else:
            stepwise_loss = loss.mean(dim=-1) # (bs, n_seq)
            # padding mask
            if mask is not None:    
                stepwise_loss = stepwise_loss * mask
                final_loss = stepwise_loss.mean()
            else:
                # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
                final_loss = stepwise_loss.mean()

        return final_loss


class OrthogonalLoss(nn.Module):
    def __init__(self, mode='group', codebook_mode=None):
        super(OrthogonalLoss, self).__init__()
        
        from utils.codebook import vq_to_range 

        self.vq_to_range = vq_to_range

        # 카테고리 그룹 중심 별
        self.mode = mode

    def forward(self, codebook):
        
        if self.mode == 'group':
            # 마지막 두개는 제외(PAD, MASK)
            code_range = list(self.vq_to_range.items())[:-2]
            cat_centroid = []
            
            # category 그룹 별 centroid 구하기
            for idx, cat_range in code_range:
                end, start = cat_range

                group_vecs = codebook[start:end+1, :]
                
                mean_group_vec = torch.mean(group_vecs, dim=0)
                cat_centroid.append(mean_group_vec)

            # 
            cat_centroid = torch.stack(cat_centroid, dim=0)
            
            # norm
            norms = torch.norm(cat_centroid, dim=1, keepdim=True)

            # norm2
            normalized_cat_centroid = cat_centroid / norms # 방향 벡터(원점 기준)

            # 
            gram_matrix = normalized_cat_centroid @ normalized_cat_centroid.T  # shape [K, K]

            # 
            identity_matrix = torch.eye(gram_matrix.size(0), device=normalized_cat_centroid.device)

            # 차이를 계산하여 Frobenius Norm의 제곱 계산
            loss = torch.norm(gram_matrix - identity_matrix, p='fro')**2
        else:
            raise NotImplementedError()

        return loss, gram_matrix
    

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, vel_mode='v1'):
        super(ReConsLoss, self).__init__()
        
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
        motion_pred, motion_gt: (bs, n_seq, dim)
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
    
    def forward_vel(self, motion_pred, motion_gt, mask=None, mean_by_sample=False):
        """
        motion_pred, motion_gt: (bs, n_seq, dim)
        """

        if self.vel_mode == 'v1':
            loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4],
                            motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        elif self.vel_mode == 'v2': # local vel
            vel_start = 4 + (self.nb_joints - 1) * 9 # root(4) + ric(3 * joint - 1) + rot(6 * joint - 1)
            loss = self.Loss(motion_pred[..., vel_start : vel_start + 3 * self.nb_joints],
                         motion_gt[..., vel_start : vel_start + 3 * self.nb_joints])
        elif self.vel_mode == 'v3': # global vel
            vel_gt = motion_gt[:,1:,:] - motion_gt[:, :-1, :]
            vel_pred = motion_pred[:,1:,:] - motion_pred[:, :-1, :]
            loss = self.Loss(vel_pred, vel_gt)

        else:
            raise NotImplementedError()
        
        stepwise_loss = loss.mean(dim=-1)


        if mask is not None:
            if self.vel_mode == 'v3':
                mask = mask[:, :-1] # 마지막꺼 제외
            stepwise_loss = stepwise_loss * mask
        
        if mean_by_sample:
            final_loss = stepwise_loss.mean(dim=-1)
        else:
            # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
            final_loss = stepwise_loss.mean()

        return final_loss
    

    def forward_output_sample_losses(self, motion_pred, motion_gt, mask=None, mean_by_sample=False):
        """
        motion_pred, motion_gt: (bs, n_seq, dim)
        """
        loss = self.Loss(motion_pred[..., :self.motion_dim],
                         motion_gt[..., :self.motion_dim])
        
        loss = loss.mean(dim=-1) # (bs, n_seq)

        if mean_by_sample:
            loss = loss.mean(dim=-1)
        
        # if mask is not None:    
        #     stepwise_loss = stepwise_loss * mask
        #     final_loss = stepwise_loss.mean()
        # else:
        #     # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
        #     final_loss = stepwise_loss.mean()

        return loss

    def forward_output_sample_vel_losses(self, motion_pred, motion_gt, mask=None, vel_mode='v2', mean_by_sample=False):
        """
        motion_pred, motion_gt: (bs, n_seq, dim)
    """
        if vel_mode == 'v1':
            loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4],
                            motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        elif vel_mode == 'v2': # local vel
            vel_start = 4 + (self.nb_joints - 1) * 9 # root(4) + ric(3 * joint - 1) + rot(6 * joint - 1)
            loss = self.Loss(motion_pred[..., vel_start : vel_start + 3 * self.nb_joints],
                         motion_gt[..., vel_start : vel_start + 3 * self.nb_joints])
        elif vel_mode == 'v3': # global vel
            vel_gt = motion_gt[:,1:,:] - motion_gt[:, :-1, :]
            vel_pred = motion_pred[:,1:,:] - motion_pred[:, :-1, :]
            loss = self.Loss(vel_pred, vel_gt)
        else:
            raise NotImplementedError()
        
        loss = loss.mean(dim=-1)

        if mean_by_sample:
            loss = loss.mean(dim=-1)
        
        return loss


class ReCons_Joint_Format_Loss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(ReCons_Joint_Format_Loss, self).__init__()
        
        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss(reduction="none")
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss(reduction="none")
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction="none")
        else:
            raise ValueError(f"Invalid recons_loss: {recons_loss}")
        
        self.nb_joints = nb_joints
        

    def forward(self, motion_pred, motion_gt, mask=None, mean_by_sample=False):
        """
        motion_pred, motion_gt: (bs, n_seq, 22, 3)
        """
        loss = self.Loss(motion_pred, motion_gt) # bs, n_seq, 22, 3
        # print(f"forward: loss.shape = {loss.shape}")
        
        stepwise_loss = loss.mean(dim=(2, 3)) # bs, n_seq

        # print(f"forward: stepwise_loss.shape = {stepwise_loss.shape}")


        if mask is not None:
            stepwise_loss = stepwise_loss * mask

        if mean_by_sample:
            final_loss = stepwise_loss.mean(dim=-1)
        else:
            final_loss = stepwise_loss.mean()

        return final_loss
    
    def forward_vel(self, motion_pred, motion_gt, vel_mode, mask=None, mean_by_sample=False):
        """
        motion_pred, motion_gt: (bs, n_seq, dim)
        """

        if vel_mode == 'v1':
            loss = self.Loss(motion_pred,
                            motion_gt)
        elif vel_mode == 'v2': # local vel
            vel_start = 4 + (self.nb_joints - 1) * 9 # root(4) + ric(3 * joint - 1) + rot(6 * joint - 1)
            loss = self.Loss(motion_pred,
                         motion_gt)
        elif vel_mode == 'v3': # global vel
            vel_gt = motion_gt[:,1:,:] - motion_gt[:, :-1, :]
            vel_pred = motion_pred[:,1:,:] - motion_pred[:, :-1, :]
            loss = self.Loss(vel_pred, vel_gt)

        else:
            raise NotImplementedError()
        
        stepwise_loss = loss.mean(dim=(2, 3))

        if mask is not None:
            if vel_mode == 'v3':
                mask = mask[:, :-1] # 마지막꺼 제외
            stepwise_loss = stepwise_loss * mask

        if mean_by_sample:
            final_loss = stepwise_loss.mean(dim=-1)
        else:
            final_loss = stepwise_loss.mean()


        return final_loss

    def forward_by_joint(self, motion_pred, motion_gt, mask=None):

        """
        motion_pred, motion_gt: (bs, n_seq, 22, 3)
        """
        loss = self.Loss(motion_pred, motion_gt) # bs, n_seq, 22, 3
        # print(f"forward: loss.shape = {loss.shape}")
        
        stepwise_loss = loss.mean(dim=-1) # bs, n_seq

        # print(f"forward: stepwise_loss.shape = {stepwise_loss.shape}")

       
        if mask is not None:
            # mask: bs, n_seq
            stepwise_loss = stepwise_loss * mask.unsqueeze(-1) # bs, n_seq, 22

        loss_by_joints = stepwise_loss.mean(dim=(0, 1)) # 22

        result = {}
        for id, loss in enumerate(loss_by_joints):
            joint_name = T2M_ID2JOINTNAME[id]
            result[joint_name] = loss.item()

        return result
    
    def forward_vel_by_joint(self, motion_pred, motion_gt, vel_mode, mask=None):
        """
        motion_pred, motion_gt: (bs, n_seq, dim)
        """

        if vel_mode == 'v1':
            loss = self.Loss(motion_pred,
                            motion_gt)
        elif vel_mode == 'v2': # local vel
            vel_start = 4 + (self.nb_joints - 1) * 9 # root(4) + ric(3 * joint - 1) + rot(6 * joint - 1)
            loss = self.Loss(motion_pred,
                         motion_gt)
        elif vel_mode == 'v3': # global vel
            vel_gt = motion_gt[:,1:,:] - motion_gt[:, :-1, :]
            vel_pred = motion_pred[:,1:,:] - motion_pred[:, :-1, :]
            loss = self.Loss(vel_pred, vel_gt)

        else:
            raise NotImplementedError()
        
        stepwise_loss = loss.mean(dim=-1)

        if mask is not None:
            if vel_mode == 'v3':
                mask = mask[:, :-1] # 마지막꺼 제외
            # mask: bs, n_seq
            stepwise_loss = stepwise_loss * mask.unsqueeze(-1)

        loss_by_joints = stepwise_loss.mean(dim=(0, 1))

        result = {}
        for id, loss in enumerate(loss_by_joints):
            joint_name = T2M_ID2JOINTNAME[id]
            result[joint_name] = loss.item()

        return result




class JointWise_ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, contact_joints):
        super(JointWise_ReConsLoss, self).__init__()
        
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
        self.contact_joints = contact_joints

        # self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4  # (nb_joints-1)*12 + 4 + 3 + 4

    def forward(self, m_pred, m_gt, mask=None, out_list=False):
        """
        motion_pred, motion_gt: (bs, n_seq, dim)
        """

        splitted_joint_feats = self.split_joint_feats(m_pred, self.nb_joints, self.contact_joints)
        m_gt = self.split_joint_feats(m_gt, self.nb_joints, self.contact_joints)

        assert len(splitted_joint_feats) == len(m_gt)

        joint_losses = []

        for pred, gt in zip(splitted_joint_feats, m_gt):

            loss = self.Loss(pred, gt)
        
            stepwise_loss = loss.mean(dim=-1)

            if out_list:
                joint_loss = stepwise_loss
            else:
                if mask is not None:    
                    stepwise_loss = stepwise_loss * mask
                    joint_loss = stepwise_loss.mean()
                else:
                    # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
                    joint_loss = stepwise_loss.mean()
        

            joint_losses.append(joint_loss)

        return joint_losses
    
    def forward_vel(self, m_pred, m_gt, vel_mode, mask=None, out_list=False):
        """
        motion_pred, motion_gt: (bs, n_seq, dim)
        """

        splitted_joint_feats = self.split_joint_feats(m_pred, self.nb_joints, self.contact_joints)
        m_gt = self.split_joint_feats(m_gt, self.nb_joints, self.contact_joints)


        assert len(splitted_joint_feats) == len(m_gt)

        joint_vel_losses = []

        if mask is not None and vel_mode == 'v3':
            mask = mask[:, :-1] # 마지막꺼 제외

        for i, (pred, gt) in enumerate(zip(splitted_joint_feats, m_gt)):
            # root는 따로 처리
            
            if vel_mode == 'v1':
                
                # if i == 0:
                #     vel_start = 4
                # else:
                #     vel_start = 9

                # loss = self.Loss(splitted_joint_feats[..., 4 : (self.nb_joints - 1) * 3 + 4],
                #                 ground_truth[..., 4 : (self.nb_joints - 1) * 3 + 4]) # root를 제외한 차분
                
                raise NotImplementedError()
            elif vel_mode == 'v2':

                if i == 0:
                    vel_start = 4
                else:
                    vel_start = 9

                loss = self.Loss(pred[..., vel_start : vel_start + 3],
                                gt[..., vel_start : vel_start + 3]) # root를 제외한 차분

                stepwise_loss = loss.mean(dim=-1)
                
                if out_list:
                    joint_vel_loss = stepwise_loss
                else:
                    if mask is not None:    
                        stepwise_loss = stepwise_loss * mask
                        joint_vel_loss = stepwise_loss.mean()
                    else:
                        # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
                        joint_vel_loss = stepwise_loss.mean()

                joint_vel_losses.append(joint_vel_loss)

            elif vel_mode == 'v3':
                vel_gt = gt[:,1:,:] - gt[:, :-1, :]
                vel_pred = pred[:,1:,:] - pred[:, :-1, :]
                loss = self.Loss(vel_pred, vel_gt)

                

                stepwise_loss = loss.mean(dim=-1)
                
                if out_list:
                    joint_vel_loss = stepwise_loss
                else:
                    if mask is not None:
                        stepwise_loss = stepwise_loss * mask
                        joint_vel_loss = stepwise_loss.mean()
                    else:
                        # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
                        joint_vel_loss = stepwise_loss.mean()

                joint_vel_losses.append(joint_vel_loss)
        
        return joint_vel_losses

    def split_joint_feats(self, x, joints_num, contact_joints):
        """
        x: [bs, nframes, pose_dim]
        
        nfeats = 12J + 1
            - root_rot_velocity (B, seq_len, 1)
            - root_linear_velocity (B, seq_len, 2)
            - root_y (B, seq_len, 1)
            - ric_data (B, seq_len, (joint_num - 1)*3)
            - rot_data (B, seq_len, (joint_num - 1)*6)
            - local_velocity (B, seq_len, joint_num*3)
            - foot contact (B, seq_len, 4)
        """
        B, T, D = x.size()

        # split
        root, ric, rot, vel, contact = torch.split(x, [4, 3 * (joints_num - 1), 6 * (joints_num - 1), 3 * joints_num, 4], dim=-1)
        
        ric = ric.reshape(B, T, joints_num - 1, 3)

        rot = rot.reshape(B, T, joints_num - 1, 6)

        vel = vel.reshape(B, T, joints_num, 3)

        # joint-wise input
        joints = [torch.cat([root, vel[:, :, 0]], dim=-1)] # [B, T, 7(4 + 3)]] # 

        for i in range(1, joints_num):
            joints.append(torch.cat([ric[:, :, i - 1], rot[:, :, i - 1], vel[:, :, i]], dim=-1)) # joint별 ric, rot, vel을 합침

        for cidx, jidx in enumerate(contact_joints):
            joints[jidx] = torch.cat([joints[jidx], contact[:, :, cidx, None]], dim=-1)
        
        return joints


    def merge_splitted_joint_feats(self, x, joints_num, contact_joints): # x should be list
        """
        x: [bs, nframes, joints_num]
        """
        # B, T, J = x.size()
        J = len(x)
        
        root = x[0]
        B, T, _ = root.shape
        

        ric_list, rot_list, vel_list = [], [], []
        for i in range(1, joints_num):
            ric = x[i][:, :, :3]
            rot = x[i][:, :, 3:9]
            vel = x[i][:, :, 9:12]

            ric_list.append(ric)
            rot_list.append(rot)
            vel_list.append(vel)

        contact = [x[i][:, :, -1] for i in contact_joints]

        ric = torch.stack(ric_list, dim=2).reshape(B, T, (J - 1) * 3)
        rot = torch.stack(rot_list, dim=2).reshape(B, T, (J - 1) * 6)
        vel = torch.stack(vel_list, dim=2).reshape(B, T, (J - 1) * 3)
        contact = torch.stack(contact, dim=2).reshape(B, T, len(contact_joints)) # 지면과 컨택(contact) 했는지를 나타내는 feature

        motion = torch.cat([
            root[..., :4], # root
            ric, # ric
            rot, # rot
            torch.cat([root[..., 4:], vel], dim=-1), # vel
            contact, # contact
        ], dim=-1)

        return motion
    
import torch
import torch.nn as nn

class JointGroup_ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints ,contact_joints):
        super(JointGroup_ReConsLoss, self).__init__()
        
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
        self.contact_joints = contact_joints

        # self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4  # (nb_joints-1)*12 + 4 + 3 + 4

    def forward(self, m_pred, m_gt, mask=None, out_list=False):
        """
        motion_pred, motion_gt: (bs, n_seq, dim)
        """

        # joint별로 나누기 -> 리스트
        splitted_joint_feats = self.split_joint_feats(m_pred, self.nb_joints, self.contact_joints)
        m_gt = self.split_joint_feats(m_gt, self.nb_joints, self.contact_joints)

        assert len(splitted_joint_feats) == len(m_gt)

        group_names = []
        joint_group_losses = []

        for group_name, joint_ids in JOINT_GROUP.items():

            joint_group_loss = []
            
            for id in joint_ids:
                gt = m_gt[id]
                pred = splitted_joint_feats[id]

                loss = self.Loss(pred, gt) # 개별 loss term

                stepwise_loss = loss.mean(dim=-1) # mean 계산

                if out_list:
                    joint_loss = stepwise_loss
                else:
                    if mask is not None:    
                        stepwise_loss = stepwise_loss * mask
                        joint_loss = stepwise_loss.mean()
                    else:
                        # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
                        joint_loss = stepwise_loss.mean()
                
                joint_group_loss.append(joint_loss)

            loss = torch.stack(joint_group_loss, dim=0).mean(dim=0) # 그룹별 평균

            group_names.append(group_name)
            joint_group_losses.append(loss)

        return group_names, joint_group_losses
    
    def forward_vel(self, m_pred, m_gt, vel_mode, mask=None, out_list=False):
        """
        motion_pred, motion_gt: (bs, n_seq, dim)
        """

        splitted_joint_feats = self.split_joint_feats(m_pred, self.nb_joints, self.contact_joints)
        m_gt = self.split_joint_feats(m_gt, self.nb_joints, self.contact_joints)

        assert len(splitted_joint_feats) == len(m_gt)

        group_names = []
        joint_group_vel_losses = []

        if mask is not None and vel_mode == 'v3':
            mask = mask[:, :-1] # 마지막꺼 제외

        for group_name, joint_ids in JOINT_GROUP.items():

            joint_group_vel_loss = []

            # 결론적으로는 v1 -> local vel matching mode이잖음
            
            
            for id in joint_ids:
                gt = m_gt[id]
                pred = splitted_joint_feats[id]

                if vel_mode == 'v1':
                    loss = self.Loss(pred[..., 4 : (self.nb_joints - 1) * 3 + 4],
                                    gt[..., 4 : (self.nb_joints - 1) * 3 + 4]) # root를 제외한 차분

                    stepwise_loss = loss.mean(dim=-1)
                    
                    if out_list:
                        joint_vel_loss = stepwise_loss
                    else:
                        if mask is not None:    
                            stepwise_loss = stepwise_loss * mask
                            joint_vel_loss = stepwise_loss.mean()
                        else:
                            # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
                            joint_vel_loss = stepwise_loss.mean()

                    joint_group_vel_loss.append(joint_vel_loss)
                
                elif vel_mode == 'v2':

                    # 각 group에서 local velocity를 나타내는 위치를 의미
                    if T2M_ID2JOINTNAME[id] == 'pelvis': # pelvis인 경우
                        vel_start = 4
                    else:
                        vel_start = 9

                    loss = self.Loss(pred[..., vel_start : vel_start + 3],
                                    gt[..., vel_start : vel_start + 3]) # root를 제외한 차분

                    stepwise_loss = loss.mean(dim=-1)
                    
                    if out_list:
                        joint_vel_loss = stepwise_loss
                    else:
                        if mask is not None:    
                            stepwise_loss = stepwise_loss * mask
                            joint_vel_loss = stepwise_loss.mean()
                        else:
                            # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
                            joint_vel_loss = stepwise_loss.mean()

                    joint_group_vel_loss.append(joint_vel_loss)

                elif vel_mode == 'v3':
                    vel_gt = gt[:,1:,:] - gt[:, :-1, :]
                    vel_pred = pred[:,1:,:] - pred[:, :-1, :]
                    loss = self.Loss(vel_pred, vel_gt)

                    stepwise_loss = loss.mean(dim=-1)
                    
                    if out_list:
                        joint_vel_loss = stepwise_loss
                    else:
                        if mask is not None:
                            stepwise_loss = stepwise_loss * mask
                            joint_vel_loss = stepwise_loss.mean()
                        else:
                            # 4) n_seq, batch 차원 전체에 대해 평균 => 최종 스칼라
                            joint_vel_loss = stepwise_loss.mean()

                
                    joint_group_vel_loss.append(joint_vel_loss)

            loss = torch.stack(joint_group_vel_loss, dim=0).mean(dim=0) # 그룹별 평균 loss
            group_names.append(group_name)
            joint_group_vel_losses.append(loss)

    
        return group_names, joint_group_vel_losses

    def split_joint_feats(self, x, joints_num, contact_joints):
        """
        x: [bs, nframes, pose_dim]
        
        nfeats = 12J + 1
            - root_rot_velocity (B, seq_len, 1)
            - root_linear_velocity (B, seq_len, 2)
            - root_y (B, seq_len, 1)
            - ric_data (B, seq_len, (joint_num - 1)*3)
            - rot_data (B, seq_len, (joint_num - 1)*6)
            - local_velocity (B, seq_len, joint_num*3)
            - foot contact (B, seq_len, 4)
        """
        B, T, D = x.size()

        # split
        root, ric, rot, vel, contact = torch.split(x, [4, 3 * (joints_num - 1), 6 * (joints_num - 1), 3 * joints_num, 4], dim=-1)
        
        ric = ric.reshape(B, T, joints_num - 1, 3)

        rot = rot.reshape(B, T, joints_num - 1, 6)

        vel = vel.reshape(B, T, joints_num, 3)

        # joint-wise input
        joints = [torch.cat([root, vel[:, :, 0]], dim=-1)] # [B, T, 7(4 + 3)]] # 

        for i in range(1, joints_num):
            joints.append(torch.cat([ric[:, :, i - 1], rot[:, :, i - 1], vel[:, :, i]], dim=-1)) # joint별 ric, rot, vel을 합침

        for cidx, jidx in enumerate(contact_joints):
            joints[jidx] = torch.cat([joints[jidx], contact[:, :, cidx, None]], dim=-1)
        
        return joints


    def merge_splitted_joint_feats(self, x, joints_num, contact_joints): # x should be list
        """
        x: [bs, nframes, joints_num]
        """
        # B, T, J = x.size()
        J = len(x)
        
        root = x[0]
        B, T, _ = root.shape
        

        ric_list, rot_list, vel_list = [], [], []
        for i in range(1, joints_num):
            ric = x[i][:, :, :3]
            rot = x[i][:, :, 3:9]
            vel = x[i][:, :, 9:12]

            ric_list.append(ric)
            rot_list.append(rot)
            vel_list.append(vel)

        contact = [x[i][:, :, -1] for i in contact_joints]

        ric = torch.stack(ric_list, dim=2).reshape(B, T, (J - 1) * 3)
        rot = torch.stack(rot_list, dim=2).reshape(B, T, (J - 1) * 6)
        vel = torch.stack(vel_list, dim=2).reshape(B, T, (J - 1) * 3)
        contact = torch.stack(contact, dim=2).reshape(B, T, len(contact_joints)) # 지면과 컨택(contact) 했는지를 나타내는 feature

        motion = torch.cat([
            root[..., :4], # root
            ric, # ric
            rot, # rot
            torch.cat([root[..., 4:], vel], dim=-1), # vel
            contact, # contact
        ], dim=-1)

        return motion
        