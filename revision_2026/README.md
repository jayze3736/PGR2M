# PGR2M 2026 Major-Revision Experiments

This directory contains the additional experiments used for the 2026 major revision of
**PGR2M: Pose-Guided Residual Refinement for Interpretable Text-to-Motion Generation and Editing**.

The goal is to make the revision evidence auditable without adding licensed datasets,
large checkpoints, TensorBoard logs, or other machine-specific artifacts to Git.

## Contents

### Notebooks

| Experiment | Purpose |
|---|---|
| Exp.01 | Residual-depth and temporal-detail analysis |
| Exp.02 / Exp.02B | Four-condition residual-dropout sensitivity (τ=0.1/0.3/0.7/0.9) and 20-repeat test evaluation |
| **Exp.02C** | **Matched τ=0.0 control, SHA256-verified checkpoint orthogonality and 20-repeat evaluation; completes Table V** |
| Exp.03 / Exp.03B | Entropy-regularization sensitivity and test evaluation |
| Exp.04A | Base Transformer recovery/retraining used by revision analysis |
| Exp.04B | Refine Transformer recovery/retraining used by revision analysis |
| Exp.04C | Fixed-checkpoint conditioning intervention (`full`, `no_text`, `no_pose`, `no_stage`, `no_history`) |
| Exp.05 | Quantitative editing success and spatial/temporal preservation |
| Exp.05B | Residual policy comparison (`pose_only`, `retain_residual`, `reencode_residual`) |

All **11** notebooks in this public handoff have execution outputs removed. The code and protocol
cells are preserved. **Exp02C was added on 2026-10-10** after the original student handoff.

### Exp02C verified result (added after original handoff)

- Training uses `--pdrop-res 0.0`, retains **RVQ quantizer dropout 0.2**, seed 123 and 300,000 iterations.
- Best-validation-FID checkpoint at **39,000** (`net_best_fid.pth`); SHA256
  `c1761ac08c32a8ae249118d6827eb68e461940029822ad17f102de6497df6ad8`.
- Pose codebook `(392, 512)`; orthogonality **0.109110** (squared off-diagonal cosine similarity).
- 20 test repeats: FID **0.006540 ± 0.000197**; MPJPE **0.027437 ± 0.000033**;
  PA-MPJPE **0.014018 ± 0.000003**; ACCEL **0.005174 ± 0.000002**.
- `results/exp02_tau_test_summary.csv` intentionally remains the original **four-condition** artifact.
  The new `results/exp02c_five_tau_test_summary.csv` holds all **five** conditions.
  An independent Exp02C-only summary and provenance JSON are also provided.
- The CI is over test evaluation repeats **at one training seed**; do not equate with multi-seed training uncertainty.
- These runs must not be confused or numerically pooled with the legacy paper Table III ablation.

### Result summaries

`results/` contains compact CSV tables corresponding to the reviewer-facing revision
experiments. These are intended as small provenance artifacts, not substitutes for raw
licensed datasets or model checkpoints.

### Provenance

`provenance/` records the environment, checkpoint hashes, and the distinction between
public original files and operational reconstructions.

### Server collection tool

`tools/collect_server_revision_artifacts.py` can be run on the revision server to export the
exact small runtime-support files, config-freeze JSONs, audit JSONs, and CSV results that
exist under the PGR2M root directory (`--pgr-root`, default `.`), including the Exp02C-specific
compatibility wrappers and `revision_results/exp02c_tau0/` result/audit exports where present.
It deliberately excludes checkpoints, datasets, raw logs, and third-party source trees.

### Server artifacts

`server_artifacts/` holds the small files exported from the revision server with that tool:
the reconstructed dataset loaders, a reconstructed loss config, revision compatibility/runner
scripts, and the config-freeze, protocol-freeze, completion-audit and result files of each
experiment. See [`server_artifacts/README.md`](server_artifacts/README.md) for what was
included or left out, and how absolute server paths were made repo-relative.

### Configs

`configs/exp_dec/losses.yaml` is the tokenizer loss config used by the revision. It is a
reconstruction; `configs/exp_dec/losses.provenance.json` records that status.

## External data / assets

Do **not** commit the following into this repository:

- HumanML3D / AMASS data
- generated `new_joints`, `new_joint_vecs`, pose-code arrays, or large cached tensors
- pretrained/checkpoint `.pth` / `.pt` files
- evaluator weights and GloVe assets when their redistribution terms do not permit it
- TensorBoard event files or full training logs
- the vendored OpenAI CLIP checkout

Instead, keep download/setup instructions and checksum manifests.

## Reproducibility note

The public repository snapshot omitted several runtime loaders/config files that were
required to execute training. Revision experiments therefore used clearly labeled
operational reconstructions. See `provenance/RECONSTRUCTION_NOTES.md`. If the authors have
the canonical original versions, publish those instead and update the provenance note.
