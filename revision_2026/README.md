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
| Exp.02 / Exp.02B | Residual-dropout threshold sensitivity and 20-repeat test evaluation |
| Exp.03 / Exp.03B | Entropy-regularization sensitivity and test evaluation |
| Exp.04A | Base Transformer recovery/retraining used by revision analysis |
| Exp.04B | Refine Transformer recovery/retraining used by revision analysis |
| Exp.04C | Fixed-checkpoint conditioning intervention (`full`, `no_text`, `no_pose`, `no_stage`, `no_history`) |
| Exp.05 | Quantitative editing success and spatial/temporal preservation |
| Exp.05B | Residual policy comparison (`pose_only`, `retain_residual`, `reencode_residual`) |

All notebooks in this public handoff have execution outputs removed. The code and protocol
cells are preserved.

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
exist under the PGR2M root directory (`--pgr-root`, default `.`). It deliberately excludes
checkpoints, datasets, raw logs, and third-party source trees.

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
