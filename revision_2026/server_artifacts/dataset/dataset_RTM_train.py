import inspect
import torch
from dataset import dataset_TM_train as _tm


def _call_supported(fn, kwargs):
    sig = inspect.signature(fn)
    return fn(**{k:v for k,v in kwargs.items() if k in sig.parameters})


def _augment(pg_tokenizer, batch, rvq_codebook_size, num_quantizer):
    caption, m_tokens, m_tokens_len, keyword_embeddings, gt_motion, m_length = batch[:6]

    # m_tokens is [B,51,394]. Refine training later removes the last two time slots,
    # leaving 49 positions. The tokenizer itself receives only the 392 pose-code dims.
    pose_codes = m_tokens[:, :-2, :-2].float().cuda(non_blocking=False)
    n_tok = pose_codes.shape[1]

    # Tokenizer temporal stride is x4 for down_t=2. Use exactly 49*4=196 frames.
    motion = gt_motion[:, :n_tok*4, :].float().cuda(non_blocking=False)

    with torch.no_grad():
        _, residual_onehot = pg_tokenizer.tokenize(
            motion=motion,
            code_indices=pose_codes,
            return_one_hot=True,
        )

    assert residual_onehot.ndim == 4, residual_onehot.shape
    assert residual_onehot.shape[1] == n_tok, (residual_onehot.shape, n_tok)
    assert residual_onehot.shape[2] == num_quantizer, residual_onehot.shape
    assert residual_onehot.shape[3] == rvq_codebook_size, residual_onehot.shape

    # train_refine_trans.py slices m_res_tokens_new[:, :-2], so append two dummy slots.
    pad = torch.zeros(
        residual_onehot.shape[0], 2, num_quantizer, rvq_codebook_size,
        device=residual_onehot.device, dtype=residual_onehot.dtype
    )
    residual_new = torch.cat([residual_onehot, pad], dim=1)

    return (caption, m_tokens, m_tokens_len, keyword_embeddings,
            gt_motion, m_length, residual_new)


class _RTMLoader:
    def __init__(self, base_loader, pg_tokenizer, rvq_codebook_size, num_quantizer):
        self.base_loader = base_loader
        self.dataset = base_loader.dataset
        self.pg_tokenizer = pg_tokenizer
        self.rvq_codebook_size = rvq_codebook_size
        self.num_quantizer = num_quantizer

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        for batch in self.base_loader:
            yield _augment(self.pg_tokenizer, batch,
                           self.rvq_codebook_size, self.num_quantizer)


def DATALoader(dataset_name, batch_size, codebook_size, pg_tokenizer,
               unit_length=4, rvq_codebook_size=512, num_quantizer=6,
               num_workers=0, codes_folder_name='codes', cache_file_name=None,
               use_keywords=False):
    # Force workers=0: residual target generation uses the CUDA tokenizer.
    base = _call_supported(_tm.DATALoader, dict(
        dataset_name=dataset_name,
        batch_size=batch_size,
        codebook_size=codebook_size,
        unit_length=unit_length,
        num_workers=0,
        use_keywords=use_keywords,
    ))
    return _RTMLoader(base, pg_tokenizer, rvq_codebook_size, num_quantizer)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
