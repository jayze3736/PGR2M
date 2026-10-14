import os
import sys
import runpy
import types
import random
import inspect
import numpy as np
from pathlib import Path

PGR_ROOT = Path(__file__).resolve().parents[1]
CLIP_VENDOR = Path('revision_preflight/third_party/openai_clip_dcba3cb2e282')

# Exact pinned OpenAI CLIP source required by PGR²M.
if str(CLIP_VENDOR) not in sys.path:
    sys.path.insert(0, str(CLIP_VENDOR))

# Expose repository root to imports such as `import utils.losses`.
if str(PGR_ROOT) not in sys.path:
    sys.path.insert(0, str(PGR_ROOT))

os.chdir(PGR_ROOT)

# Legacy NumPy alias used by PGR²M quaternion utilities.
if not hasattr(np, "float"):
    np.float = float

# ------------------------------------------------------------------
# Compatibility for private dataset helper symbols omitted from the
# public PGR²M repository.
# ------------------------------------------------------------------
from torch.utils.data import DataLoader, Subset

def _compat_get_subsamples_loader(source_loader, num_samples=4, seed=123):
    dataset = source_loader.dataset
    n = min(int(num_samples), len(dataset))
    if n <= 0:
        raise ValueError("Cannot sample from an empty dataset.")

    rng = random.Random(int(seed))
    indices = rng.sample(range(len(dataset)), n)
    subset = Subset(dataset, indices)

    collate_fn = getattr(source_loader, "collate_fn", None)

    return DataLoader(
        subset,
        batch_size=n,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

import dataset.dataset_TM_eval as _tm_eval
if not hasattr(_tm_eval, "get_subsamples_loader"):
    _tm_eval.get_subsamples_loader = _compat_get_subsamples_loader

try:
    import dataset.dataset_RTM_eval as _rtm_eval
except ModuleNotFoundError:
    _rtm_eval = types.ModuleType("dataset.dataset_RTM_eval")
    _rtm_eval.__file__ = "<runtime compatibility stub for tokenizer training>"
    sys.modules["dataset.dataset_RTM_eval"] = _rtm_eval

if not hasattr(_rtm_eval, "get_subsamples_loader"):
    _rtm_eval.get_subsamples_loader = _compat_get_subsamples_loader

# ------------------------------------------------------------------
# scikit-learn TSNE API compatibility.
# Historical PGR²M uses n_iter=1000; newer sklearn uses max_iter.
# Patch only the symbol imported by utils.eval_trans, at runtime.
# ------------------------------------------------------------------
import sklearn
import sklearn.manifold as _sk_manifold

_OriginalTSNE = _sk_manifold.TSNE
_tsne_params = inspect.signature(_OriginalTSNE).parameters

if "n_iter" not in _tsne_params and "max_iter" in _tsne_params:
    def _CompatTSNE(*args, **kwargs):
        if "n_iter" in kwargs and "max_iter" not in kwargs:
            kwargs["max_iter"] = kwargs.pop("n_iter")
        return _OriginalTSNE(*args, **kwargs)
    _sk_manifold.TSNE = _CompatTSNE

# ------------------------------------------------------------------
# Matplotlib Axes3D compatibility.
# Historical PGR²M uses `ax.lines = []` and `ax.collections = []`.
# Modern Matplotlib exposes these properties as read-only.
# ------------------------------------------------------------------
from mpl_toolkits.mplot3d.axes3d import Axes3D as _Axes3D

def _install_empty_list_setter(cls, attr_name):
    original = getattr(cls, attr_name, None)

    # The property may be inherited from matplotlib.axes.Axes.
    if not isinstance(original, property):
        for base in cls.__mro__[1:]:
            candidate = base.__dict__.get(attr_name)
            if isinstance(candidate, property):
                original = candidate
                break

    if isinstance(original, property) and original.fset is None:
        def _setter(self, value, _attr=attr_name):
            # PGR²M assigns only [] here. Preserve that historical intent.
            if value is None or len(value) == 0:
                artists = list(getattr(self, _attr))
                for artist in artists:
                    try:
                        artist.remove()
                    except Exception:
                        pass
                return
            raise AttributeError(
                "Compatibility setter only supports empty assignment for " + str(_attr) + "."
            )

        setattr(
            cls,
            attr_name,
            property(
                original.fget,
                _setter,
                original.fdel,
                original.__doc__,
            ),
        )
        return True
    return False

_MPL_LINES_PATCHED = _install_empty_list_setter(_Axes3D, "lines")
_MPL_COLLECTIONS_PATCHED = _install_empty_list_setter(_Axes3D, "collections")

# Import eval_trans once now so its local `TSNE` symbol receives the
# compatibility adapter before train_pg_tokenizer imports from it.
import utils.eval_trans as _eval_trans


# Exp.02 matched-RNG control: seed Python random immediately before training.
import random as _exp02_python_random
_exp02_python_random.seed(123)

runpy.run_path(str(PGR_ROOT / "train_pg_tokenizer.py"), run_name="__main__")
