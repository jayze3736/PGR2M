# Revision experiment environment

The 2026 revision experiments were executed in the following recovered environment.

- OS: Linux container
- Python: 3.11.16
- PyTorch: 2.5.1+cu121
- torchvision: 0.20.1+cu121
- NumPy: 2.1.2
- CUDA runtime used by PyTorch: 12.1
- GPU: NVIDIA A100-SXM4-80GB
- Jupyter kernel display name: omitted (site-specific name of the revision server)
- Kernel executable used during the revision: a Python 3.11 virtual environment on the revision server (site-specific path omitted; shown as `venv/bin/python` in the published files)
- `torch-dct`: 0.1.6
- OpenAI CLIP: revision environment used a pinned checkout under
  `revision_preflight/third_party/openai_clip_dcba3cb2e282`.

## Important provenance note

Several dataset-loader/support files required by the public repository were not present in
the released repository snapshot. During the revision, operational, contract-validated
reconstructions were created from the released code interfaces and upstream public code.
These are **not claimed to be byte-identical private originals**.

The server-side collection tool in `../tools/collect_server_revision_artifacts.py` can copy
the exact runtime support files and small audit/result artifacts that were used in the
revision. Keep reconstructed files clearly labeled when publishing them.
