# Reconstruction and protocol notes

## Public/recovered boundary

The experiments in this directory build on the public PGR2M repository and a recovered
revision environment. The following categories must remain distinguishable in the public
release:

1. **Public original code** — files already present in the PGR2M repository.
2. **Revision experiment code** — notebooks/scripts added for the 2026 major revision.
3. **Operational reconstructions** — missing loader/config/support files reconstructed from
   public interfaces or public upstream code so the released stack can execute.

Do not describe category (3) as recovered private source.

## Known reconstructed support used during the revision

Depending on the experiment, the runtime used reconstructed versions of:

- `dataset/dataset_PC.py`
- `dataset/dataset_TM_eval.py`
- `dataset/dataset_TM_train.py`
- `dataset/dataset_RTM_train.py`
- `dataset/dataset_RTM_eval.py`
- tokenizer/refine loss YAMLs when the exact original YAML was absent
- compatibility wrappers for NumPy 2.x, the pinned OpenAI CLIP checkout, and qualitative
  evaluation helpers missing from the public release

The notebooks document the exact provenance and assumptions for these components.

## Reconstructed files published in this repository

These files are operational reconstructions derived from released interfaces/public upstream
code and are not claimed to be byte-identical copies of unreleased originals. The author's
canonical versions were not available at the time of release.

| File | Status record |
|---|---|
| `server_artifacts/dataset/dataset_PC.py` | `server_artifacts/dataset/reconstructed_loaders.provenance.json` |
| `server_artifacts/dataset/dataset_TM_train.py` | same |
| `server_artifacts/dataset/dataset_TM_eval.py` | same |
| `server_artifacts/dataset/dataset_RTM_train.py` | same |
| `server_artifacts/dataset/dataset_RTM_eval.py` | same |
| `server_artifacts/revision_preflight/exp04b_rt2m_losses.yaml` (stands in for the unreleased `configs/exp_rt2m/losses.yaml`) | `server_artifacts/revision_preflight/exp04b_rt2m_losses.provenance.json` |
| `configs/exp_dec/losses.yaml` | `configs/exp_dec/losses.provenance.json` |

The file contents are kept byte-identical to what ran on the revision server, so that the SHA256
values recorded inside the config-freeze JSONs still match. The reconstruction label therefore
lives in the `*.provenance.json` files next to them instead of inside the files.

Known non-files: `revision_preflight/train_pg_tokenizer_exp00s12_compat.py` appears in the
collection tool's file list but never existed on the revision server.

## Protocol cautions

- Do not mix absolute generation metrics from the reconstructed revision stack with the
  original paper's reported Table-I/Table-III values without an explicit protocol note.
- Exp.04C is a **fixed-checkpoint causal intervention**, not a retrained architecture ablation.
- Exp.05/05B are controlled pose-code editing studies on four angle groups and fixed trained
  tokenizer checkpoints.
