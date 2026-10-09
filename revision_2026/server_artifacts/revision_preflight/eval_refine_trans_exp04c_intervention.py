
import os, sys
from pathlib import Path

PGR = Path(".")
CLIP_ROOT = PGR / "revision_preflight" / "third_party" / "openai_clip_dcba3cb2e282"

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.chdir(PGR)
sys.path.insert(0, str(PGR))
sys.path.insert(0, str(CLIP_ROOT))

import numpy as np
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import dataset.dataset_TM_eval as _tm_eval
if not hasattr(_tm_eval, "get_subsamples_loader"):
    def _unused_subsamples(*args, **kwargs):
        raise NotImplementedError("Exp.04C disables qualitative drawing.")
    _tm_eval.get_subsamples_loader = _unused_subsamples
    print("[OK] Added dataset_TM_eval.get_subsamples_loader import shim")

import torch
import models.rt2m_trans as _rt

condition = os.environ.get("EXP04C_CONDITION", "full").strip().lower()
valid = {"full", "no_text", "no_pose", "no_stage", "no_history"}
if condition not in valid:
    raise ValueError(f"Unknown EXP04C_CONDITION={condition!r}; expected {sorted(valid)}")

_orig_cond = _rt.InputProcessor.emb_cond
_orig_pose = _rt.InputProcessor.emb_pose_codes
_orig_stage = _rt.InputProcessor.emb_q_ids
_orig_hist = _rt.InputProcessor.emb_res_codes

if condition == "no_text":
    def _zero_cond(self, clip_feature):
        return torch.zeros_like(_orig_cond(self, clip_feature))
    _rt.InputProcessor.emb_cond = _zero_cond

elif condition == "no_pose":
    def _zero_pose(self, p_codes):
        return torch.zeros_like(_orig_pose(self, p_codes))
    _rt.InputProcessor.emb_pose_codes = _zero_pose

elif condition == "no_stage":
    def _zero_stage(self, q_ids):
        return torch.zeros_like(_orig_stage(self, q_ids))
    _rt.InputProcessor.emb_q_ids = _zero_stage

elif condition == "no_history":
    def _zero_history(self, all_indices, active_q_layers, inference=False):
        return torch.zeros_like(
            _orig_hist(self, all_indices, active_q_layers, inference=inference)
        )
    _rt.InputProcessor.emb_res_codes = _zero_history

print(f"[OK] Exp.04C intervention installed: {condition}")

if os.environ.get("EXP04C_DRYRUN", "0") == "1":
    proc = _rt.InputProcessor(
        num_vq=392, num_rvq=512, clip_dim=512, embed_dim=1024,
        block_size=62, share_weight=False, shared_codebook=False,
        num_quantizers=6
    )
    with torch.no_grad():
        if condition == "no_text":
            y = proc.emb_cond(torch.randn(2, 12, 512))
            assert torch.count_nonzero(y).item() == 0
        elif condition == "no_pose":
            y = proc.emb_pose_codes(torch.randn(2, 49, 394))
            assert torch.count_nonzero(y).item() == 0
        elif condition == "no_stage":
            y = proc.emb_q_ids(torch.tensor([0, 3]))
            assert torch.count_nonzero(y).item() == 0
        elif condition == "no_history":
            idx = torch.zeros(2, 49, 1, dtype=torch.long)
            y = proc.emb_res_codes(idx, torch.ones(2, dtype=torch.long), inference=True)
            assert torch.count_nonzero(y).item() == 0
        else:
            y = proc.emb_cond(torch.randn(2, 12, 512))
            assert y.shape == (2, 12, 1024)
    print(f"[OK] Dry-run contract passed: {condition}")
    raise SystemExit(0)

src_path = PGR / "eval_refine_trans.py"
src = src_path.read_text()

src = src.replace(
    'os.environ["CUDA_VISIBLE_DEVICES"] = "1"',
    '# EXP04C: CUDA visibility is controlled by the wrapper'
)

needle = "repeat_time = 20"
if src.count(needle) != 1:
    raise RuntimeError(f"Expected exactly one {needle!r}, found {src.count(needle)}")
src = src.replace(
    needle,
    'repeat_time = int(os.environ.get("EXP04C_REPEATS", "20"))'
)

code = compile(src, str(src_path), "exec")
exec(code, {"__name__": "__main__", "__file__": str(src_path)})
