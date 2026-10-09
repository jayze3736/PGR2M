import inspect
import torch
from dataset import dataset_TM_eval as _tm


def _call_supported(fn, kwargs):
    sig = inspect.signature(fn)
    return fn(**{k:v for k,v in kwargs.items() if k in sig.parameters})


def _augment(pg_tokenizer, batch, codebook_size, rvq_codebook_size, num_quantizer):
    # Validated TM evaluator contract: the first ten fields are standard.
    word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, indices, keyword_embeddings = batch[:10]

    # The reconstructed TM evaluator returns frame-level 392-D pose codes over 196 frames.
    # Downsample by 4 to match the tokenizer latent length (49 positions).
    pose_codes = indices[:, :196:4, :codebook_size].float().cuda(non_blocking=False)
    n_tok = pose_codes.shape[1]
    motion = pose[:, :n_tok*4, :].float().cuda(non_blocking=False)

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

    # Refine pose embedding expects num_vq+2=394 dimensions.
    pose_new = torch.zeros(
        pose_codes.shape[0], n_tok+2, codebook_size+2,
        device=pose_codes.device, dtype=pose_codes.dtype
    )
    pose_new[:, :n_tok, :codebook_size] = pose_codes

    residual_new = torch.zeros(
        residual_onehot.shape[0], n_tok+2, num_quantizer, rvq_codebook_size,
        device=residual_onehot.device, dtype=residual_onehot.dtype
    )
    residual_new[:, :n_tok] = residual_onehot

    return (word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length,
            token, name, indices, keyword_embeddings, pose_new, residual_new)


class _RTMEvalLoader:
    def __init__(self, base_loader, pg_tokenizer, codebook_size,
                 rvq_codebook_size, num_quantizer):
        self.base_loader = base_loader
        self.dataset = base_loader.dataset
        self.pg_tokenizer = pg_tokenizer
        self.codebook_size = codebook_size
        self.rvq_codebook_size = rvq_codebook_size
        self.num_quantizer = num_quantizer

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        for batch in self.base_loader:
            yield _augment(self.pg_tokenizer, batch, self.codebook_size,
                           self.rvq_codebook_size, self.num_quantizer)


def DATALoader(dataset_name, is_test, batch_size, w_vectorizer,
               codebook_size=392, pg_tokenizer=None, rvq_codebook_size=512,
               num_quantizer=6, num_workers=0, codes_folder_name='codes',
               cache_file_name=None, use_keywords=False):
    base = _call_supported(_tm.DATALoader, dict(
        dataset_name=dataset_name,
        is_test=is_test,
        batch_size=batch_size,
        w_vectorizer=w_vectorizer,
        codebook_size=codebook_size,
        num_workers=0,
        unit_length=4,
        use_keywords=use_keywords,
    ))
    return _RTMEvalLoader(base, pg_tokenizer, codebook_size,
                          rvq_codebook_size, num_quantizer)


def get_subsamples_loader(*args, **kwargs):
    raise NotImplementedError(
        'Qualitative drawing is disabled in Exp.04B; this unreleased helper is not used.'
    )
