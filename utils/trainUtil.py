import torch

def create_pad_mask(max_len, m_tokens_len):
    pad_mask = torch.arange(max_len, device=m_tokens_len.device).expand(len(m_tokens_len), max_len) < m_tokens_len.unsqueeze(1) # block size가 최대길이라서, m_tokens_len보다 큰 부분은 mask 처리
    pad_mask = pad_mask.float()
    return pad_mask

def lr_lambda(current_step: int, warm_up_iter: int, gamma: float, milestones: list):
    if current_step < warm_up_iter:
        return float(current_step) / float(max(1, warm_up_iter))
    return gamma ** len([m for m in milestones if m <= current_step])

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gradient_based_dynamic_weighting(model, losses:list, normalize_weights=False):

    # loss에 대한 마지막 layer의 output weight의 gradient의 크기를 구함
    grads = [
        torch.autograd.grad(loss, model.decoder.model[-1].parameters(), create_graph=True)[0]
        for loss in losses
    ]
     
    # gradient norm 계산
    grad_norms = [g.norm().detach() for g in grads]

    # 평균 norm
    avg_norm = sum(grad_norms) / len(grad_norms)

    # 각 손실에 대한 동적 가중치 계산 (norm 기준 정규화)
    weights = [avg_norm / (g + 1e-8) for g in grad_norms]

    # weight 정규화 (선택적) → 전체 합이 1이 되도록 할 경우
    # weight_sum = sum(weights)
    # weights = [w / weight_sum for w in weights]

    if normalize_weights:
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

    # w_pelvis = weights[0],
    # w_normal_joint = weights[1],
    # w_contact_joint = weights[2],
    # w_vel_pelvis = weights[3],
    # w_vel_normal_joint = weights[4],
    # w_vel_contact_joint = weights[5]
    
    return weights


@torch.no_grad()
def calc_confidence(pred, target, mode='bce', mask=None):
    if mode == 'bce':
        p = torch.sigmoid(pred)
        conf = p * target + (1 - p) * (1 - target)
        conf = conf * mask.unsqueeze(-1) 
        conf = conf.view(-1, conf.size(-1)).sum(dim=0) / mask.sum() 
        conf = torch.sigmoid(conf)
    elif mode == 'ce':
        raise NotImplementedError

    return conf

@torch.no_grad()
def make_mask(max_seq_len, m_tokens_len):
    indices = torch.arange(max_seq_len, device=m_tokens_len.device)
    mask = indices < m_tokens_len.unsqueeze(1)
    return mask
