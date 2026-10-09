
import os, sys, runpy, types
from pathlib import Path
import numpy as np

PGR = Path(".")
CLIP_ROOT = PGR / "revision_preflight" / "third_party" / "openai_clip_dcba3cb2e282"

os.chdir(PGR)
sys.path.insert(0, str(PGR))
sys.path.insert(0, str(CLIP_ROOT))

# NumPy 2.x compatibility used by older upstream code.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import ftfy
import clip
print("[OK] CLIP:", clip.__file__)

def _unreleased_subsamples_loader(*args, **kwargs):
    raise NotImplementedError(
        "get_subsamples_loader was not released publicly. "
        "Exp.04A disables qualitative drawing, so this helper must not be called."
    )

# Import shims required only because utils/eval_trans.py imports these symbols eagerly.
import dataset.dataset_TM_eval as _tm_eval
if not hasattr(_tm_eval, "get_subsamples_loader"):
    _tm_eval.get_subsamples_loader = _unreleased_subsamples_loader
    print("[OK] Added dataset_TM_eval.get_subsamples_loader import shim")

try:
    import dataset.dataset_RTM_eval as _rtm_eval
except ModuleNotFoundError:
    import dataset as _dataset_pkg
    _rtm_eval = types.ModuleType("dataset.dataset_RTM_eval")
    _rtm_eval.get_subsamples_loader = _unreleased_subsamples_loader
    sys.modules["dataset.dataset_RTM_eval"] = _rtm_eval
    setattr(_dataset_pkg, "dataset_RTM_eval", _rtm_eval)
    print("[OK] Added dataset_RTM_eval import-compatible module")
else:
    if not hasattr(_rtm_eval, "get_subsamples_loader"):
        _rtm_eval.get_subsamples_loader = _unreleased_subsamples_loader
        print("[OK] Added dataset_RTM_eval.get_subsamples_loader import shim")

# Import after shims are installed.
import utils.eval_trans as _eval_trans

_original_evaluation_transformer = _eval_trans.evaluation_transformer

def _evaluation_transformer_no_draw(*args, **kwargs):
    # Quantitative evaluation and checkpoint selection are unchanged.
    # Skip only the qualitative 3D/TensorBoard rendering path.
    kwargs["draw"] = False
    return _original_evaluation_transformer(*args, **kwargs)

_eval_trans.evaluation_transformer = _evaluation_transformer_no_draw
print("[OK] Forced evaluation_transformer(draw=False): quantitative metrics retained; visualization skipped")

runpy.run_path(str(PGR / "train_base_trans.py"), run_name="__main__")
