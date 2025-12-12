import torch
from typing import List

def align_by_parts(joints, align_inds=None): 
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1) # pelvis = root
    return joints - torch.unsqueeze(pelvis, dim=1) # joint position - root

def batch_compute_similarity_transform_torch(S1, S2): 

    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat, (scale, R, t)


def compute_mpjpe(preds,
                  target,
                  valid_mask=None,
                  pck_joints=None,
                  sample_wise=True): 
    """
    Mean per-joint position error (i.e. mean Euclidean distance)
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape,
                                              target.shape)  # BxJx3
    mpjpe = torch.norm(preds - target, p=2, dim=-1)  # BxJ

    if pck_joints is None:
        if sample_wise:
            mpjpe_seq = ((mpjpe * valid_mask.float()).sum(-1) /
                         valid_mask.float().sum(-1)
                         if valid_mask is not None else mpjpe.mean(-1))
        else:
            mpjpe_seq = mpjpe[valid_mask] if valid_mask is not None else mpjpe
        return mpjpe_seq
    else:
        mpjpe_pck_seq = mpjpe[:, pck_joints]
        return mpjpe_pck_seq
    
# joint per error
def compute_jpe(preds,
                target,
                valid_mask=None,
                pck_joints=None,
                sample_wise=False):  # valid_mask: (B,) or (B, J)
    """
    Compute MPJPE (Euclidean joint error) either:
      - per joint across the batch (default)
      - per joint per sample (if sample_wise=True)

    Args:
        preds, target: (B, J, 3)
        valid_mask: optional (B,) or (B, J)
        pck_joints: optional List[int], subset of joints to evaluate
        sample_wise: whether to return (B, J) instead of (J,)

    Returns:
        If sample_wise=False: (J,) or (len(pck_joints),)
        If sample_wise=True:  (B, J) or (B, len(pck_joints))
    """
    assert preds.shape == target.shape, f"{preds.shape} != {target.shape}"
    jpe = torch.norm(preds - target, p=2, dim=-1)  # (B, J)

    if pck_joints is not None:
        jpe = jpe[:, pck_joints]
        if valid_mask is not None and valid_mask.dim() == 2:
            valid_mask = valid_mask[:, pck_joints]

    if valid_mask is not None:
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(1)  # (B, 1)
        valid_mask = valid_mask.float()
        jpe = jpe * valid_mask

        if sample_wise:
            return jpe  # shape: (B, J_selected)
        else:
            return jpe.sum(dim=0) / (valid_mask.sum(dim=0) + 1e-6)
    else:
        return jpe if sample_wise else jpe.mean(dim=0)



def calc_mpjpe(preds, target, align_inds=[0], sample_wise=True, trans=None): 
    # Expects LxJx3
    valid_mask = target[:, :, 0] != -2.0 
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    if align_inds is not None:
        preds_aligned = align_by_parts(preds, align_inds=align_inds)
        if trans is not None:
            preds_aligned += trans
        target_aligned = align_by_parts(target, align_inds=align_inds)
    else:
        preds_aligned, target_aligned = preds, target
    mpjpe_each = compute_mpjpe(preds_aligned,
                               target_aligned,
                               valid_mask=valid_mask,
                               sample_wise=sample_wise)
    return mpjpe_each


def calc_jpe(preds, target, align_inds=[0], sample_wise=True, trans=None): 
    # Expects LxJx3
    valid_mask = target[:, :, 0] != -2.0 
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    if align_inds is not None:
        preds_aligned = align_by_parts(preds, align_inds=align_inds)
        if trans is not None:
            preds_aligned += trans
        target_aligned = align_by_parts(target, align_inds=align_inds)
    else:
        preds_aligned, target_aligned = preds, target
    jpe_each = compute_jpe(preds_aligned,
                            target_aligned,
                            valid_mask=valid_mask,
                            sample_wise=sample_wise)
    return jpe_each

def calc_pampjpe(preds, target, sample_wise=True, return_transform_mat=False): 
    # Expects BxJx3
    target, preds = target.float(), preds.float()
    # extracting the keypoints that all samples have valid annotations
    # valid_mask = (target[:, :, 0] != -2.).sum(0) == len(target)
    # preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(preds[:, valid_mask], target[:, valid_mask])
    # pa_mpjpe_each = compute_mpjpe(preds_tranformed, target[:, valid_mask], sample_wise=sample_wise)

    preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(
        preds, target)
    pa_mpjpe_each = compute_mpjpe(preds_tranformed,
                                  target,
                                  sample_wise=sample_wise)

    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each

def calc_accel(preds, target): 
    """
    Mean joint acceleration error
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape,
                                              target.shape)  # BxJx3
    assert preds.dim() == 3
    # Expects BxJx3
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
    accel_pred = preds[:-2] - 2 * preds[1:-1] + preds[2:]
    normed = torch.linalg.norm(accel_pred - accel_gt, dim=-1)
    accel_seq = normed.mean(1)
    return accel_seq


def compute_joint_metrics(m_org, m_pred, jointstype, m_lengths: List[int], align_root: bool = True, force_in_meter=False) -> dict:

    if force_in_meter:
        factor = 1000.0
    else:
        factor = 1.0

    assert m_pred.shape == m_org.shape
    assert m_pred.dim() == 4
    
    bs, n_seq, n_joint, n_xyz = m_pred.shape

    if align_root and jointstype in ['mmm', 'humanml3d']:
        align_inds = [0]
    else:
        align_inds = None
    
    mpjpe = torch.tensor([0.0], device=m_org.device)
    pampjpe = torch.tensor([0.0], device=m_org.device)
    accel = torch.tensor([0.0], device=m_org.device)
    jpe = torch.tensor([0.0 for _ in range(n_joint)], device=m_org.device)

    count = 0 
    count_seq = len(m_lengths) 

    for i in range(bs):
        mpjpe += torch.sum(
            calc_mpjpe(m_pred[i], m_org[i], align_inds=align_inds))
        pampjpe += torch.sum(calc_pampjpe(m_pred[i], m_org[i]))
        accel += torch.sum(calc_accel(m_pred[i], m_org[i]))
        jpe += torch.sum(calc_jpe(m_pred[i], m_org[i], align_inds=align_inds), dim=0)
        count += m_lengths[i]

    result = {
        'mpjpe': mpjpe.item() / (count * factor) ,
        'pampjpe': pampjpe.item() / (count * factor),
        'accel': accel.item() / (count - 2 * count_seq) * factor,
        'jpe': (jpe / (count * factor)).cpu() ,
    }

    return result
    

    # mr_metrics["MPJPE"] = self.MPJPE / count * factor
    # mr_metrics["PAMPJPE"] = self.PAMPJPE / count * factor
    # mr_metrics["ACCEL"] = self.ACCEL / (count - 2 * count_seq) * factor