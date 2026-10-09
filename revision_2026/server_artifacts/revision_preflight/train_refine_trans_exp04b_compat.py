import os, sys, runpy
from pathlib import Path
import numpy as np

PGR = Path('.')
CLIP_ROOT = PGR/'revision_preflight/third_party/openai_clip_dcba3cb2e282'
os.chdir(PGR)
sys.path.insert(0, str(PGR))
sys.path.insert(0, str(CLIP_ROOT))

if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool

import ftfy
import clip
print('[OK] CLIP:', clip.__file__)

def _unused_subsamples(*args, **kwargs):
    raise NotImplementedError('Qualitative drawing is disabled in Exp.04B.')

import dataset.dataset_TM_eval as _tm_eval
if not hasattr(_tm_eval, 'get_subsamples_loader'):
    _tm_eval.get_subsamples_loader = _unused_subsamples
    print('[OK] Added dataset_TM_eval.get_subsamples_loader import shim')

import dataset.dataset_RTM_eval as _rtm_eval
import utils.eval_trans as _eval_trans
_orig = _eval_trans.evaluation_residual_transformer

def _eval_no_draw(*args, **kwargs):
    kwargs['draw'] = False
    return _orig(*args, **kwargs)

_eval_trans.evaluation_residual_transformer = _eval_no_draw
print('[OK] Forced evaluation_residual_transformer(draw=False)')

runpy.run_path(str(PGR/'train_refine_trans.py'), run_name='__main__')
