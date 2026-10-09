#!/usr/bin/env python3
"""
Collect small, GitHub-suitable PGR2M revision artifacts from the revision server.

This tool intentionally excludes:
- datasets
- checkpoints (.pth/.pt)
- full training logs / TensorBoard events
- third-party source trees
- raw large caches

It copies only small code/config/audit/result files needed to document the 2026 revision.
"""
from __future__ import annotations
from pathlib import Path
import argparse, shutil, json, hashlib

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_copy(src: Path, dst_root: Path, pgr: Path, manifest: list):
    if not src.exists() or not src.is_file():
        return
    if src.suffix.lower() in {".pth",".pt",".ckpt"}:
        return
    rel = src.relative_to(pgr)
    dst = dst_root / "small_files" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append({
        "source": str(src),
        "relative_path": rel.as_posix(),
        "bytes": src.stat().st_size,
        "sha256": sha256(src),
    })

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pgr-root", default=".")
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    pgr=Path(args.pgr_root).resolve()
    out=Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    (out/"small_files").mkdir(parents=True)

    manifest=[]

    # Reconstructed/runtime support files used by the revision.
    explicit = [
        "dataset/dataset_PC.py",
        "dataset/dataset_TM_eval.py",
        "dataset/dataset_TM_train.py",
        "dataset/dataset_RTM_train.py",
        "dataset/dataset_RTM_eval.py",
        "configs/exp_dec/losses.yaml",
        "revision_preflight/exp04b_rt2m_losses.yaml",
        "revision_preflight/train_pg_tokenizer_exp00s12_compat.py",
        "revision_preflight/train_base_trans_exp04a_compat.py",
        "revision_preflight/train_refine_trans_exp04b_compat.py",
        "revision_preflight/eval_refine_trans_exp04c_intervention.py",
        "revision_preflight/run_exp05_editing_quant.py",
        "revision_preflight/run_exp05b_reencode.py",
    ]
    for rel in explicit:
        safe_copy(pgr/rel, out, pgr, manifest)

    # Small result/config/audit artifacts.
    roots = [
        pgr/"revision_results",
        pgr/"revision_logs",
    ]
    allowed_suffixes={".json",".csv",".yaml",".yml",".txt"}
    for base in roots:
        if not base.exists():
            continue
        for src in base.rglob("*"):
            if not src.is_file():
                continue
            if src.suffix.lower() not in allowed_suffixes:
                continue
            # Skip full logs / queue logs. Keep manifests, audits, configs, summaries.
            name=src.name.lower()
            if name.endswith(".log") or "train.log" in name or "queue.log" in name:
                continue
            if src.stat().st_size > 5_000_000:
                continue
            safe_copy(src, out, pgr, manifest)

    # Human-readable manifest.
    (out/"MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = """# Server-exported PGR2M revision artifacts

This directory was produced by `collect_server_revision_artifacts.py`.

Before committing:
1. Review every file.
2. Keep reconstructed support files clearly labeled.
3. Do not add datasets, checkpoints, raw logs, secrets, or third-party source.
4. Prefer canonical author-owned source over reconstructed support if available.
"""
    (out/"README.md").write_text(readme, encoding="utf-8")
    print(f"[OK] Exported {len(manifest)} small files to {out}")

if __name__ == "__main__":
    main()
