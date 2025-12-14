import torch
import numpy as np
from utils.codebook import *
import torch.nn.functional as F

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