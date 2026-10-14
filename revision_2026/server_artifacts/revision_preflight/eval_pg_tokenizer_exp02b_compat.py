import os, sys, runpy
from pathlib import Path
import numpy as np

PGR = Path(".")
CLIP_ROOT = PGR / "revision_preflight" / "third_party" / "openai_clip_dcba3cb2e282"

os.chdir(PGR)
sys.path.insert(0, str(PGR))

# Exp.00S12 recovered and pinned the exact OpenAI CLIP checkout.
# utils/eval_trans.py imports `clip` transitively, so expose it here.
if not CLIP_ROOT.exists():
    raise RuntimeError(
        f"Recovered OpenAI CLIP checkout is missing: {CLIP_ROOT}. "
        "Do not install an unrelated PyPI package named `clip`."
    )
sys.path.insert(0, str(CLIP_ROOT))

try:
    import ftfy  # required by the pinned OpenAI CLIP tokenizer
except ModuleNotFoundError as e:
    raise RuntimeError(
        "Missing ftfy in evaluator subprocess. Re-run notebook cell 3; "
        "it installs the pinned ftfy==6.1.1 dependency."
    ) from e

# Public utils/eval_trans.py imports two helper symbols that were not released
# with the public repository. They are unrelated to tokenizer evaluation_dec(),
# which is the only path used by eval_pg_tokenizer.py here. Add import-only
# placeholders so the public module can load. If another code path ever tries
# to call them, fail loudly instead of silently substituting behavior.
def _unreleased_subsamples_loader(*args, **kwargs):
    raise NotImplementedError(
        "get_subsamples_loader was not released in the public PGR2M dataset loaders "
        "and is not required by Exp.02B tokenizer evaluation_dec."
    )

import dataset.dataset_TM_eval as _tm_eval
if not hasattr(_tm_eval, "get_subsamples_loader"):
    _tm_eval.get_subsamples_loader = _unreleased_subsamples_loader

# The released repository may omit dataset.dataset_RTM_eval entirely, while
# utils/eval_trans.py imports it at module import time. Create an in-memory,
# import-only placeholder module. This does NOT change repository files.
import types as _types
import dataset as _dataset_pkg

try:
    import dataset.dataset_RTM_eval as _rtm_eval
    if not hasattr(_rtm_eval, "get_subsamples_loader"):
        _rtm_eval.get_subsamples_loader = _unreleased_subsamples_loader
except ModuleNotFoundError:
    _rtm_eval = _types.ModuleType("dataset.dataset_RTM_eval")
    _rtm_eval.get_subsamples_loader = _unreleased_subsamples_loader
    sys.modules["dataset.dataset_RTM_eval"] = _rtm_eval
    setattr(_dataset_pkg, "dataset_RTM_eval", _rtm_eval)

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

runpy.run_path(str(PGR / "eval_pg_tokenizer.py"), run_name="__main__")
