import torch
import numpy as np
from utils.codebook import *
import torch.nn.functional as F

def generate_frame_noise_mask(data, alpha, m_length):

    """
    입력: 
        data (torch.Tensor): (bs, n_seq, features)
        alpha (float): frame 비율 [0.1, 0.9]
        m_length (torch.Tensor): (bs,) 각 시퀀스의 실제 길이
    출력: 
        data shape(bs, n_seq, features)와 동일한 이진 마스크
    """
    bs, n_seq, _ = data.shape
    # Create a mask of shape (bs, n_seq)
    mask = torch.zeros(bs, n_seq, device=data.device)
    
    for i in range(bs):
        length = m_length[i].item()
        num_to_mask = int(length * alpha)
        
        # Generate unique random indices to mask
        if num_to_mask > 0:
            indices_to_mask = torch.randperm(length, device=data.device)[:num_to_mask]
            mask[i, indices_to_mask] = 1.0
            
    # Expand mask to match data shape
    frame_mask = mask.unsqueeze(-1).expand_as(data).bool()
    return frame_mask

def inject_noise_per_frame(data, vq_to_range, beta, max_cat_num, random_noise=False):
    """
    데이터의 각 프레임에 대해 독립적으로 카테고리를 샘플링하고 노이즈를 주입합니다.
    """
    bs, n_seq, _ = data.shape
    corrupted_data = data.clone()
    original_data = data.clone()

    for i in range(bs):
        for j in range(n_seq):
            # 각 프레임마다 노이즈를 주입할 카테고리 수와 ID를 새로 샘플링
            sampled_beta = np.random.choice(beta)
            target_cat_num = int(sampled_beta * max_cat_num)
            target_cats = np.random.choice(max_cat_num, target_cat_num, replace=False)

            # 해당 프레임의 선택된 카테고리에 노이즈 주입
            for cat_id in target_cats:
                if cat_id in vq_to_range and cat_id < max_cat_num: # Pose code 카테고리만 처리
                    end, start = vq_to_range[cat_id]
                    
                    # 노이즈 생성
                    if random_noise:
                        noise = torch.rand_like(corrupted_data[i, j, start:end+1])
                    else:
                        noise = torch.randn_like(corrupted_data[i, j, start:end+1])
                    
                    corrupted_data[i, j, start:end+1] = noise
                    
                    # 원본 정답 위치를 찾아서 -inf로 마스킹
                    original_answer_mask = original_data[i, j, start:end+1].bool()
                    corrupted_data[i, j, start:end+1][original_answer_mask] = -float('inf')

    # 전체 데이터에 대해 hard label로 정규화
    for cat in vq_to_range:
        if cat < max_cat_num: # Check if cat is a pose code category
            end, start = vq_to_range[cat]
            
            # Get the index of the max value in the category range
            idx = torch.argmax(corrupted_data[:, :, start:end+1], dim=-1, keepdim=True)
            
            # Create a one-hot vector
            one_hot = torch.zeros_like(corrupted_data[:, :, start:end+1])
            one_hot.scatter_(-1, idx, 1)
            
            corrupted_data[:, :, start:end+1] = one_hot

    return corrupted_data


def inject_noise_per_frame_fast(data, vq_to_range, beta, max_cat_num, random_noise=True, group_ids:list=None):
    """
    데이터의 각 프레임에 대해 독립적으로 카테고리를 샘플링하고 노이즈를 주입합니다. (벡터화 버전)
    """
    bs, n_seq, n_feats = data.shape
    device = data.device


    if group_ids is None:
        # 1. 모든 프레임에 대해 한 번에 노이즈를 주입할 카테고리 결정
        # (bs, n_seq, 1) 크기의 랜덤 beta 샘플링
        beta_indices = torch.randint(0, len(beta), (bs, n_seq, 1), device=device)
        sampled_betas = torch.tensor(beta, device=device)[beta_indices]
        
        # 각 프레임별로 훼손할 카테고리 개수 계산
        num_to_corrupt = (sampled_betas * max_cat_num).floor().long() # (bs, n_seq, 1)

        # 훼손할 카테고리를 선택하기 위한 랜덤 순열 생성
        
        # (bs, n_seq, max_cat_num)
        cat_rand_perms = torch.rand(bs, n_seq, max_cat_num, device=device).argsort(dim=-1)

        # 각 프레임별로 훼손할 카테고리 마스크 생성
        # arange(max_cat_num) -> (max_cat_num)
        # cat_rand_perms < num_to_corrupt
        # (bs, n_seq, max_cat_num) < (bs, n_seq, 1) -> (bs, n_seq, max_cat_num)
        cat_corruption_mask = torch.arange(max_cat_num, device=device).expand_as(cat_rand_perms) < num_to_corrupt
        cat_corruption_mask = cat_corruption_mask.gather(-1, cat_rand_perms.argsort(-1))

        
    else:
        bs, n_seq, dim = data.shape
        cat_corruption_mask = torch.zeros((bs, n_seq, max_cat_num), device=data.device)
        cat_corruption_mask[..., group_ids] = 1

    # 2. 카테고리 마스크를 전체 피처 차원으로 확장
    feature_corruption_mask = torch.zeros_like(data, dtype=torch.bool)
    
    for cat_id in range(max_cat_num):
        if cat_id in vq_to_range:
            end, start = vq_to_range[cat_id]
            # (bs, n_seq) -> (bs, n_seq, 1) -> (bs, n_seq, cat_dim)
            cat_mask_expanded = cat_corruption_mask[:, :, cat_id].unsqueeze(-1).expand(-1, -1, end - start + 1)
            feature_corruption_mask[:, :, start:end+1] = cat_mask_expanded

    

    # 3. 노이즈 생성 및 적용
    if random_noise:
        noise = torch.rand_like(data)
    else:
        noise = torch.randn_like(data)
    
    # feature_corruption_mask가 True인 위치에만 노이즈 적용
    corrupted_data = torch.where(feature_corruption_mask, noise, data)

    # 4. 원본 정답 위치를 -inf로 마스킹
    original_answer_mask = data.bool()
    # 노이즈가 주입된 위치이면서, 원래 정답이었던 위치를 True로 설정 -> 노이즈가 주입되었다면, 정답인 부분은 선택해서는 안됨
    mask_for_inf = feature_corruption_mask & original_answer_mask
    corrupted_data[mask_for_inf] = -float('inf')

    # 5. 전체 데이터에 대해 hard label로 정규화
    final_data = data.clone()

    for cat in vq_to_range:
        if cat < max_cat_num:  # Pose code category
            end, start = vq_to_range[cat]
            cat_slice = corrupted_data[:, :, start:end+1]
            idx = torch.argmax(cat_slice, dim=-1, keepdim=True)
            
            one_hot = torch.zeros_like(cat_slice)
            one_hot.scatter_(-1, idx, 1)
            final_data[:, :, start:end+1] = one_hot
        
    return final_data

def mask_cond(text_cond, cond_drop_prob, force_mask=False, mask_only_keywords=True):
    bs, num_key, d =  text_cond.shape

    if force_mask:
        return torch.zeros_like(text_cond)
    elif cond_drop_prob > 0.:
        mask = torch.bernoulli(torch.ones(bs, num_key, device=text_cond.device) * cond_drop_prob).view(bs, num_key, 1)
        
        if mask_only_keywords:
            mask[:, 0, :] = 0. # main text sentence는 dropout 하지 않음

        return text_cond * (1. - mask)
    else:
        return text_cond
    
@torch.no_grad()
def mask_ver_vanilla(gt_label, current_sampling_prob, pkeep, cat_num, corrupt_pad_end=True):

    # 처음에는 정답만, 후반에는 masking을 많이
    if np.random.random() >= current_sampling_prob:

        masked_out = True

        if pkeep == -1:
            proba = np.random.rand(1)[0] 
            mask = torch.bernoulli(proba * torch.ones(gt_label.shape,
                                                            device=gt_label.device))
        else: # 정해진 pkeep 확률로 masking할 토큰을 결정
            mask = torch.bernoulli(pkeep * torch.ones(gt_label.shape,
                                                            device=gt_label.device))
            
        if not corrupt_pad_end:
            mask[:,:,-2:] = 1.0 # pad, end token은 corrupt 대상에서 제외

        # 혹시 모르니 정수가 되도록 반올림?
        mask = mask.round().to(dtype=torch.int64)

        # 0 ~ 1 사이의 랜덤한 숫자 샘플링(노이즈?)
        r_indices = torch.randn(gt_label.shape, device = gt_label.device)

        # 기존 target logit에 masking을 적용하고 masking이 적용되지 않은 logit에는 노이즈를 추가
        # a_indices를 추가하는 이유는 이후 trans_net에서 teacher forcing을 시키기 위해 정답 토큰을 넣을텐데, 이때 정답을 그대로 넣으면 실제 test랑 일치하지 않을테니 일부러 노이즈를 줘서
        # 
        a_indices = mask*gt_label+(1-mask)*r_indices

        # Mutual exclusivity
        for cat in vq_to_range:
            if cat < cat_num:                
                end, start = vq_to_range[cat]

                # 주어진 category 범위에서 logit이 가장 높은 target token을 1로 나머지는 0으로 만듦
                idx = torch.argmax(a_indices[:,:,start:end+1], dim = -1, keepdim=True)

                a_indices[:,:,start:end+1] = 0
                a_indices.scatter_(-1, start+idx, 1)
            else:
                end, start = vq_to_range[cat]
                a_indices[:,:,start:end+1] = torch.nn.functional.sigmoid(a_indices[:,:,start:end+1]) > 0.5
                
    else:
        masked_out = False
        a_indices = gt_label

    return a_indices, masked_out

@torch.no_grad()
def mask_ver_symmetric_distance(gt_label, current_sampling_prob, pkeep, gamma, cat_num, tau=1.0, corrupt_pad_end=True):
    
    # mask는 카테고리 마스크가 필요함
    if np.random.random() >= current_sampling_prob:

        masked_out = True
        mask = torch.zeros(gt_label.shape, device=gt_label.device)
        
        for cat in range(cat_num):
            if cat in vq_to_range:
                end, start = vq_to_range[cat]
                out = 1 if torch.rand(1).item() < pkeep else 0
                mask[..., start:end+1] = out * torch.ones(mask[..., start:end+1].shape,
                                                            device=mask.device)


        if not corrupt_pad_end:
            mask[:,:,-2:] = 1.0 # pad, end token은 corrupt 대상에서 제외
        
        mask = mask.round().to(dtype=torch.int64)
        
        r_indices = torch.zeros_like(gt_label, device=gt_label.device)  # (bs, n_seq, dim)

        for group_id, (end, start) in vq_to_range.items():

            if group_id < len(codes):
                # gt_label 일부 추출
                cat_vec = gt_label[..., start:end+1]   # (bs, n_seq, cat_dim)
                bs, n_seq, cat_dim = cat_vec.shape

                # pivot index (argmax)
                pivot = torch.argmax(cat_vec, dim=-1)  # (bs, n_seq)
                pivot_repeat = pivot.unsqueeze(-1).repeat(1, 1, cat_dim)  # (bs, n_seq, cat_dim)

                # row index 생성
                row_idx = torch.arange(cat_dim, device=cat_vec.device)  # (cat_dim,)
                row_idx = row_idx.unsqueeze(0).unsqueeze(0).expand(bs, n_seq, cat_dim)  # (bs, n_seq, cat_dim)

                # 거리 계산
                sub = row_idx - pivot_repeat
                sub = -1 * torch.abs(sub)  # (bs, n_seq, cat_dim)

                # logits, probs
                logits = sub * gamma + pivot_repeat  # (bs, n_seq, cat_dim)
                probs = torch.softmax(logits / tau, dim=-1)  # (bs, n_seq, cat_dim)

                # categorical sampling
                idx = torch.distributions.Categorical(probs=probs).sample()  # (bs, n_seq)

                # scatter로 one-hot 채우기
                r_indices[:, :, start:end+1] = 0
                r_indices.scatter_(-1, (start + idx).unsqueeze(-1), 1)
        
        a_indices = mask*gt_label+(1-mask)*r_indices
    else:
        masked_out = False
        a_indices = gt_label
    
    
    return a_indices, masked_out

# @torch.no_grad()
# def confidence_based_mask_ver_symmetric_distance(gt_label, current_sampling_prob, conf, gamma, cat_num, tau=1.0, corrupt_pad_end=True):
    
#     # mask는 카테고리 마스크가 필요함
#     if np.random.random() >= current_sampling_prob:

#         masked_out = True
#         mask = torch.zeros(gt_label.shape, device=gt_label.device)
#         conf = conf.view(1, 1, -1)
#         masked_conf = gt_label * conf

#         for cat in range(cat_num):
#             if cat in vq_to_range:
#                 end, start = vq_to_range[cat]
#                 category_confidences = masked_conf[..., start:end+1]
#                 flattened_confidences = category_confidences.view(-1, category_confidences.shape[-1])
#                 summed_confidences = flattened_confidences.sum(dim=0)
                        
#                 valid_token_mask = (category_confidences != 0).float()
#                 flattened_mask = valid_token_mask.view(-1, valid_token_mask.shape[-1])
#                 valid_token_count = flattened_mask.sum(dim=0)
            
#                 epsilon = 1e-9
#                 average_confidence_per_category = (1 - summed_confidences / (valid_token_count + epsilon))
#                 mask[..., start:end+1] = average_confidence_per_category

#         if not corrupt_pad_end:
#             mask[:,:,-2:] = 1.0 # pad, end token은 corrupt 대상에서 제외
        
#         mask = mask.round().to(dtype=torch.int64)
        
#         r_indices = torch.zeros_like(gt_label, device=gt_label.device)  # (bs, n_seq, dim)

#         for group_id, (end, start) in vq_to_range.items():

#             if group_id < len(codes):
#                 # gt_label 일부 추출
#                 cat_vec = gt_label[..., start:end+1]   # (bs, n_seq, cat_dim)
                
#                 bs, n_seq, cat_dim = cat_vec.shape

#                 # pivot index (argmax)
#                 pivot = torch.argmax(cat_vec, dim=-1)  # (bs, n_seq)
#                 pivot_repeat = pivot.unsqueeze(-1).repeat(1, 1, cat_dim)  # (bs, n_seq, cat_dim)

#                 # row index 생성
#                 row_idx = torch.arange(cat_dim, device=cat_vec.device)  # (cat_dim,)
#                 row_idx = row_idx.unsqueeze(0).unsqueeze(0).expand(bs, n_seq, cat_dim)  # (bs, n_seq, cat_dim)

#                 # 거리 계산
#                 sub = row_idx - pivot_repeat
#                 sub = -1 * torch.abs(sub)  # (bs, n_seq, cat_dim)

#                 # logits, probs
#                 logits = sub * gamma + pivot_repeat  # (bs, n_seq, cat_dim)
#                 probs = torch.softmax(logits / tau, dim=-1)  # (bs, n_seq, cat_dim)

#                 # categorical sampling
#                 idx = torch.distributions.Categorical(probs=probs).sample()  # (bs, n_seq) 

#                 # scatter로 one-hot 채우기
#                 r_indices[:, :, start:end+1] = 0
#                 r_indices.scatter_(-1, (start + idx).unsqueeze(-1), 1)
        
#         a_indices = mask*gt_label+(1-mask)*r_indices
#     else:
#         masked_out = False
#         a_indices = gt_label
    
    
#     return a_indices, masked_out

@torch.no_grad()
def confidence_based_mask_ver_symmetric_distance(gt_label, current_sampling_prob, conf, gamma, cat_num, tau=1.0, corrupt_pad_end=True):
    
    # mask는 카테고리 마스크가 필요함
    if np.random.random() >= current_sampling_prob:

        masked_out = True
        mask = torch.zeros(gt_label.shape, device=gt_label.device)
        conf = conf.view(1, 1, -1)
        masked_conf = gt_label * conf

        for cat in range(cat_num):
            if cat in vq_to_range:
                end, start = vq_to_range[cat]
                category_confidences = masked_conf[..., start:end+1]
                flattened_confidences = category_confidences.view(-1, category_confidences.shape[-1])
                summed_confidences = flattened_confidences.sum(dim=0)
                        
                valid_token_mask = (category_confidences != 0).float()
                flattened_mask = valid_token_mask.view(-1, valid_token_mask.shape[-1])
                valid_token_count = flattened_mask.sum(dim=0)
            
                epsilon = 1e-9
                average_confidence_per_category = summed_confidences / (valid_token_count + epsilon)
                # mask[..., start:end+1] = average_confidence_per_category
                
                # 0인 부분 제외하고 confidence 추출
                average_confidence_per_category = gt_label[:, :, start:end+1] * average_confidence_per_category 
                average_confidence_per_category = average_confidence_per_category.sum(dim=-1)

                # 각 위치별로 confidence 값에 따라 마스킹 결정
                corrupting_dir = torch.bernoulli(average_confidence_per_category) # (bs, n_seq)

                # 마스킹을 카테고리 차원으로 확장, 만약 해당 카테고리 마스킹 확정시, 해당 카테고리 logit 전체 마스킹
                corrupting_dir = corrupting_dir.unsqueeze(2).repeat(1, 1, end-start+1)
                
                mask[:, :, start:end+1] = corrupting_dir

        if not corrupt_pad_end:
            mask[:,:,-2:] = 0.0 # pad, end token은 corrupt 대상에서 제외
        
        mask = mask.round().to(dtype=torch.int64)
        
        r_indices = torch.zeros_like(gt_label, device=gt_label.device)  # (bs, n_seq, dim)

        for group_id, (end, start) in vq_to_range.items():

            if group_id < len(codes):
                # gt_label 일부 추출
                cat_vec = gt_label[..., start:end+1]   # (bs, n_seq, cat_dim)
                
                bs, n_seq, cat_dim = cat_vec.shape

                # pivot index (argmax)
                pivot = torch.argmax(cat_vec, dim=-1)  # (bs, n_seq)
                pivot_repeat = pivot.unsqueeze(-1).repeat(1, 1, cat_dim)  # (bs, n_seq, cat_dim)

                # row index 생성
                row_idx = torch.arange(cat_dim, device=cat_vec.device)  # (cat_dim,)
                row_idx = row_idx.unsqueeze(0).unsqueeze(0).expand(bs, n_seq, cat_dim)  # (bs, n_seq, cat_dim)

                # 거리 계산
                sub = row_idx - pivot_repeat
                sub = -1 * torch.abs(sub)  # (bs, n_seq, cat_dim)

                # logits, probs
                logits = sub * gamma + pivot_repeat  # (bs, n_seq, cat_dim)
                probs = torch.softmax(logits / tau, dim=-1)  # (bs, n_seq, cat_dim)

                # categorical sampling
                idx = torch.distributions.Categorical(probs=probs).sample()  # (bs, n_seq)

                # scatter로 one-hot 채우기
                r_indices[:, :, start:end+1] = 0
                r_indices.scatter_(-1, (start + idx).unsqueeze(-1), 1)
        
        a_indices = (1-mask)*gt_label+mask*r_indices
    else:
        masked_out = False
        a_indices = gt_label
    
    
    return a_indices, masked_out

def mask_residual_codes(r_codes, mask_token_id, masking_prob):
    num_classes = r_codes.shape[-1]
    input_indices = torch.argmax(r_codes, dim=-1)
    prob_mask = torch.rand(input_indices.shape, device=input_indices.device)
    masked_positions = (prob_mask < masking_prob)
    masked_input_ids = input_indices.clone()
    masked_input_ids[masked_positions] = mask_token_id

    # Convert the integer token IDs to one-hot vectors
    one_hot_masked_input = F.one_hot(masked_input_ids, num_classes=num_classes).float()
    return one_hot_masked_input

def corrupt_residual_codes(all_indices, rvq_nb_code, masking_prob, pad_index):
    
    # 패딩이 아닌 위치에 대한 마스크 생성
    non_pad_mask = (all_indices != pad_index)
    
    # 마스킹할 위치 결정
    prob_mask = torch.rand(all_indices.shape, device=all_indices.device)
    masked_positions = (prob_mask < masking_prob) 
    
    # 패딩 위치를 제외하고 최종 마스킹 위치 결정
    final_masked_positions = masked_positions & non_pad_mask
    
    masked_input_ids = all_indices.clone()
    
    # 마스킹될 위치에 대한 랜덤 토큰 생성
    num_masked = final_masked_positions.sum()
    random_tokens = torch.randint(0, rvq_nb_code, (num_masked,), device=all_indices.device)
    
    # 마스킹된 위치를 랜덤 토큰으로 교체
    masked_input_ids[final_masked_positions] = random_tokens

    # 정수 토큰 ID를 원-핫 벡터로 변환
    
    return masked_input_ids