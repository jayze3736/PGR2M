import torch
import torch.nn.functional as F
from os.path import join as pjoin
from common.skeleton import Skeleton
import numpy as np
from common.quaternion import *
from utils.paramUtil import *

T2M_ID2JOINTNAME = {
        0: 'pelvis',
        1: 'left_hip',
        2: 'right_hip',
        3: 'spine1',
        4: 'left_knee',
        5: 'right_knee',
        6: 'spine2',
        7: 'left_ankle',
        8: 'right_ankle',
        9: 'spine3',
        10: 'left_foot',
        11: 'right_foot',
        12: 'neck',
        13: 'left_collar',
        14: 'right_collar',
        15: 'head',
        16: 'left_shoulder',
        17: 'right_shoulder',
        18: 'left_elbow',
        19: 'right_elbow',
        20: 'left_wrist',
        21: 'right_wrist'
}

T2M_CONTACT_JOINTS = [7, 10, 8, 11]

JOINT_GROUP = {
    'pelvis': [0],
    'normal_joints': [i for i in list(T2M_ID2JOINTNAME.keys()) if i not in T2M_CONTACT_JOINTS and i != 0],
    'contact_joints': T2M_CONTACT_JOINTS
}

def load_loss_wrappers(logger, cfg, WRAPPER_CLASS_MAP, external_params=None):

    wrappers = {}
    for item in cfg['loss_wrappers']:
        wrapper_type = item['type']
        name = item.get('name', wrapper_type)
        params = item.get('params', {})

        cls = WRAPPER_CLASS_MAP.get(wrapper_type)
        if cls is None:
            raise ValueError(f"Unknown wrapper type: {wrapper_type}")
        else:
            
            if logger is not None:
                logger.info(f'loaded loss type: {name}')
            # logger.info(f'params: {params}')

        wrappers[name] = cls(**params)

    return wrappers

def kh2index(k_hot_tensor: torch.Tensor):
    """
    k_hot_tensor: (B, T, D) — batch of k-hot vectors
    Returns: (B, T, K) — indices of 1s
    """
    k = k_hot_tensor.sum(dim=-1)[0,0].int().item()
    indices = torch.topk(k_hot_tensor, k=k, dim=-1, sorted=True)[1]
    sorted_indices = torch.sort(indices, dim=-1)[0]
    return sorted_indices

def idx2khot(index_tensor, num_classes):
    """
    k_hot_tensor: (B, T, D) — batch of k-hot vectors
    Returns: (B, T, K) — indices of 1s
    """

    bs, n_seq, d = index_tensor.shape

    indices_flat = index_tensor.view(-1, d)

    one_hot = F.one_hot(indices_flat, num_classes=num_classes)

    # sum over d → shape: (bs * n_seq, num_classes)
    k_hot_flat = one_hot.sum(dim=1)
    k_hot = k_hot_flat.view(bs, n_seq, num_classes)

    return k_hot

### PORTED From Humanml3D

# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22
# ds_num = 8
# data_dir = './joints/'
# save_dir1 = './HumanML3D/new_joints/'
# save_dir2 = './HumanML3D/new_joint_vecs/'

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain

# Get offsets of target skeleton
position = np.load('/data/CoMo/dataset/HumanML3D/new_joints/000021.npy')
position = position.reshape(len(position), -1, 3)
example_data = torch.from_numpy(position)

tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
# (joints_num, 3)
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
# print(tgt_offsets)

def get_rotation_invariant_position(positions: torch.Tensor) -> torch.Tensor:
    """
    Convert global joint positions to rotation-invariant, root-relative positions (HumanML3D-style `jp`).

    Args:
        positions (torch.Tensor): (T, J, 3), global joint positions
        face_joint_indx (tuple): indices (r_hip, l_hip, r_shoulder, l_shoulder)

    Returns:
        torch.Tensor: (T, J-1, 3), rotation-invariant joint positions (excluding root)
    """
    positions = positions.clone()
    T, J, _ = positions.shape

    # Step 1: Put root at origin (XZ only)
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]

    # Step 2: Estimate body-facing direction
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across_hips = positions[:, r_hip] - positions[:, l_hip]
    across_shoulders = positions[:, sdr_r] - positions[:, sdr_l]
    across = across_hips + across_shoulders
    across = across / torch.norm(across, dim=1, keepdim=True)

    # Step 3: Compute rotation quaternion to align facing to Z+
    up = torch.tensor([0., 1., 0.], device=positions.device).expand(T, 3)
    forward = torch.cross(up, across, dim=1)
    forward = forward / torch.norm(forward, dim=1, keepdim=True)

    target = torch.tensor([0., 0., 1.], device=positions.device).expand(T, 3)
    rot_quat = qbetween(forward, target)  # (T, 4) – torch version of qbetween_np

    # Step 4: Apply rotation to all joints
    rot_quat = rot_quat.unsqueeze(1).expand(-1, J, -1)  # (T, J, 4)
    positions = qrot(rot_quat, positions)  # torch version of quaternion rotation

    # Step 5: Remove root
    jp = positions[:, :]  # (T, J-1, 3)

    return jp

def compute_gradient_norm(model, norm_type=2):
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None]
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)  # p-norm 계산
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm