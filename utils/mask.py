from utils.masking_utils import mask_ver_vanilla, mask_ver_symmetric_distance, confidence_based_mask_ver_symmetric_distance
import torch

class Mask():
    def __init__(self, cat_num, corrupt_pad_end):
        self.cat_num = cat_num
        # args.mask_method
        self.corrupt_pad_end = corrupt_pad_end
    
    @torch.no_grad()
    def apply(self, x, current_sampling_prob, pkeep, gamma, confidence=None, mask_method='vanilla'):
        if mask_method == 'vanilla':
            return mask_ver_vanilla(x, current_sampling_prob, pkeep, cat_num=self.cat_num, corrupt_pad_end=self.corrupt_pad_end)
        elif mask_method == 'symmetric_distance':
            return mask_ver_symmetric_distance(x, current_sampling_prob, pkeep, gamma=gamma, cat_num=self.cat_num, corrupt_pad_end=self.corrupt_pad_end)
        elif mask_method == 'confidence_based_symmetric_distance':
            return confidence_based_mask_ver_symmetric_distance(x, current_sampling_prob, confidence, gamma=gamma, cat_num=self.cat_num, corrupt_pad_end=self.corrupt_pad_end)
        else:
            raise ValueError(f"Unknown mask method: {mask_method}")