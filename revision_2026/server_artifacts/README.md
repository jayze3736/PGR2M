# Server artifacts (2026 major revision)

Small files exported from the revision server with `../tools/collect_server_revision_artifacts.py`.
They document how the revision experiments were run. No datasets, checkpoints, TensorBoard
logs, or third-party source are included.

## Layout

| Folder | Content |
|---|---|
| `dataset/` | Reconstructed dataset loaders (`dataset_PC`, `dataset_TM_train`, `dataset_TM_eval`, `dataset_RTM_train`, `dataset_RTM_eval`). To use them, copy the files to `dataset/` at the repository root. |
| `revision_preflight/` | Compatibility wrappers and experiment runners (`train_*_compat.py`, `eval_refine_trans_exp04c_intervention.py`, `run_exp05*.py`) and the reconstructed Refine-Transformer loss config. |
| `revision_results/exp01_residual_depth/` | Exp.01 summaries, bootstrap tables, provenance. |
| `revision_results/exp02_tau/`, `exp03_entropy/` | Config freezes, validation audits/trajectories, test freezes, 20-repeat test runs and reviewer tables. |
| `revision_results/exp04a_base/`, `exp04b_refine/` | Config freezes and completion audits. |
| `revision_results/exp04c_conditioning/` | Protocol freeze, completion audit, repeat metrics, summaries. |
| `revision_results/exp05_editing/production/`, `exp05b_reencode/production/` | Protocol freeze, completion audits, per-sample and per-target results, paired comparisons. |

## Reconstructions

`dataset/*.py` and `revision_preflight/exp04b_rt2m_losses.yaml` are operational reconstructions
derived from released interfaces/public upstream code and are not claimed to be byte-identical
copies of unreleased originals. See `dataset/reconstructed_loaders.provenance.json`,
`revision_preflight/exp04b_rt2m_losses.provenance.json`, and
`../provenance/RECONSTRUCTION_NOTES.md`.

## Paths

The server stores everything under absolute paths. In the published files they were rewritten
to be relative to the repository root:

| Server path | Published as |
|---|---|
| PGR2M root directory | repository root (`.`; paths such as `revision_results/...`) |
| revision virtual environment | `venv/` |
| upstream HumanML3D preprocessing checkout | `third_party/` |
| dataset / AMASS root | `dataset/` |

Only path text and the site-specific kernel name changed. File-by-file original and published
SHA256 values are in [`../provenance/SANITIZATION_MAP.csv`](../provenance/SANITIZATION_MAP.csv).
Files with zero replacements are byte-identical to the server copies, so SHA256 values recorded
inside the freeze/audit JSONs (loaders, loss config, per-sample result CSV) still match.

## Left out on purpose

- Pre-experiment diagnostics (`exp00*` audits, `exp00u_*` trajectories).
- `smoke/` test runs of Exp.05 and Exp.05B; only the `production/` runs are included.
- `*_partial.csv` files, which are byte-identical to the matching `*_per_sample.csv`.
- Launch and queue-state logs under `revision_logs/`, and `exp01_progress.json`.
