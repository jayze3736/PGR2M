
import os, sys, json, time, random, hashlib, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PGR = Path(".")
os.chdir(PGR)
sys.path.insert(0, str(PGR))

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

from models.pg_tokenizer import PoseGuidedTokenizer
from utils.motion_process import recover_from_ric
from utils.codebook import vq_to_range

TARGETS = {
    "left_knee":  dict(group_id=0, triplet=(1,4,7), target_joints=(1,4,7)),
    "right_knee": dict(group_id=1, triplet=(2,5,8), target_joints=(2,5,8)),
    "left_elbow": dict(group_id=2, triplet=(16,18,20), target_joints=(16,18,20)),
    "right_elbow":dict(group_id=3, triplet=(17,19,21), target_joints=(17,19,21)),
}
ANGLE_THRESHOLDS = np.arange(10, 180, 10, dtype=np.float32)

def fixseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build_tokenizer(folder):
    import argparse as _argparse
    folder = Path(folder)
    a = _argparse.Namespace(**yaml.safe_load((folder / "arguments.yaml").read_text()))
    net = PoseGuidedTokenizer(
        a,
        a.nb_code,
        a.code_dim,
        a.output_emb_width,
        a.down_t,
        a.stride_t,
        a.width,
        a.depth,
        a.dilation_growth_rate,
        a.vq_act,
        a.vq_norm,
        num_quantizers=a.rvq_num_quantizers,
        shared_codebook=a.rvq_shared_codebook,
        quantize_dropout_prob=a.rvq_quantize_dropout_prob,
        quantize_dropout_cutoff_index=a.rvq_quantize_dropout_cutoff_index,
        rvq_nb_code=a.rvq_nb_code,
        mu=a.rvq_mu,
        residual_ratio=a.rvq_residual_ratio,
        vq_loss_beta=a.rvq_vq_loss_beta,
        quantizer_type=a.rvq_quantizer_type,
        params_soft_ent_loss=a.params_soft_ent_loss,
        use_ema=(not a.unuse_ema),
        init_method=a.rvq_init_method,
    )
    ckpt_path = folder / "net_best_fid.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.cuda().eval()
    for p in net.parameters():
        p.requires_grad = False
    return net, a, ckpt_path, ckpt

def recover_xyz(motion_norm, mean_np, std_np):
    arr = motion_norm.detach().float().cpu().numpy()
    arr = arr * std_np + mean_np
    xyz = recover_from_ric(torch.from_numpy(arr).float().unsqueeze(0), 22)
    return xyz.squeeze(0).cpu().numpy()

def angle_deg(xyz, triplet):
    a, b, c = triplet
    v1 = xyz[:, c] - xyz[:, b]
    v2 = xyz[:, a] - xyz[:, b]
    v1 = v1 / np.clip(np.linalg.norm(v1, axis=1, keepdims=True), 1e-8, None)
    v2 = v2 / np.clip(np.linalg.norm(v2, axis=1, keepdims=True), 1e-8, None)
    cosv = np.clip((v1 * v2).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosv))

def angle_category(angles):
    return np.digitize(angles, ANGLE_THRESHOLDS, right=True).astype(np.int64)

def edit_pose_tokens(pose_tokens, group_id, tok_start, tok_end, target_cat):
    out = pose_tokens.copy()
    g_end, g_start = vq_to_range[group_id]
    assert (g_end - g_start + 1) == 18
    out[tok_start:tok_end, g_start:g_end+1] = 0.0
    out[tok_start:tok_end, g_start + target_cat] = 1.0
    return out

def mpjpe_mm(a, b, frame_mask=None, joint_ids=None):
    if joint_ids is None:
        joint_ids = np.arange(a.shape[1])
    if frame_mask is None:
        frame_mask = np.ones(a.shape[0], dtype=bool)
    aa = a[frame_mask][:, joint_ids]
    bb = b[frame_mask][:, joint_ids]
    if aa.size == 0:
        return float("nan")
    return float(np.linalg.norm(aa - bb, axis=-1).mean() * 1000.0)

def bootstrap_mean_ci(values, B=1000, seed=2027):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=np.float64)
    for i in range(B):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=400)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--tokenizer-dir", type=str, required=True)
parser.add_argument("--data-root", type=str, required=True)
parser.add_argument("--exp05-rows", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--bootstrap", type=int, default=1000)
args = parser.parse_args()

fixseed(args.seed)
DATA = Path(args.data_root)
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

mean_np = np.load(DATA / "Mean.npy").astype(np.float32)
std_np = np.load(DATA / "Std.npy").astype(np.float32)

old = pd.read_csv(args.exp05_rows)
if set(old["strategy"].unique()) != {"pose_only", "retain_residual"}:
    raise RuntimeError(f"Unexpected Exp.05 strategies: {old['strategy'].unique()}")

# One canonical row per sample; use pose_only only to avoid duplicates.
protocol = old[old.strategy == "pose_only"].copy()
protocol = protocol.sort_values("sample_index").reset_index(drop=True)

if len(protocol) < args.n_samples:
    raise RuntimeError(f"Exp.05 has only {len(protocol)} canonical samples")
protocol = protocol.iloc[:args.n_samples].copy()

net, net_args, tokenizer_ckpt, tokenizer_state = build_tokenizer(args.tokenizer_dir)
print("[OK] Tokenizer loaded:", tokenizer_ckpt)
print("[OK] Tokenizer checkpoint iter:", tokenizer_state.get("nb_iter"))
print("[OK] Tokenizer SHA256:", sha256(tokenizer_ckpt))

rows = []
t0_all = time.time()

for i, pr in protocol.iterrows():
    sid = str(pr["sample_id"])
    target_name = str(pr["target_name"])
    target = TARGETS[target_name]

    motion_raw = np.load(DATA / "new_joint_vecs" / f"{sid}.npy").astype(np.float32)
    pose_full = np.load(DATA / "codes" / f"{sid}.npy").astype(np.float32)

    L = int(pr["n_frames"])
    L = min(L, len(motion_raw), len(pose_full))
    L = (L // 4) * 4

    motion_raw = motion_raw[:L]
    pose_tokens = pose_full[:L:4]

    tok_start = int(pr["tok_start"])
    tok_end = int(pr["tok_end"])
    target_cat = int(pr["target_cat"])
    source_med = int(pr["source_median_cat"])

    if tok_end > len(pose_tokens):
        raise RuntimeError(f"{sid}: Exp.05 window exceeds pose token length")

    edited_tokens = edit_pose_tokens(
        pose_tokens, target["group_id"], tok_start, tok_end, target_cat
    )

    motion_norm = ((motion_raw - mean_np) / std_np).astype(np.float32)

    motion_t = torch.from_numpy(motion_norm).float().unsqueeze(0).cuda()
    pose_t = torch.from_numpy(pose_tokens).float().unsqueeze(0).cuda()
    edited_t = torch.from_numpy(edited_tokens).float().unsqueeze(0).cuda()

    with torch.no_grad():
        # Standard unedited full reconstruction.
        _, original_residual = net.tokenize(
            motion=motion_t,
            code_indices=pose_t,
            return_one_hot=True,
        )
        base_norm = net.inference(
            residual_codes=original_residual,
            code_indices=pose_t,
        )

        # Re-encode the residual relative to the EDITED pose latent.
        _, reencoded_residual = net.tokenize(
            motion=motion_t,
            code_indices=edited_t,
            return_one_hot=True,
        )
        edit_norm = net.inference(
            residual_codes=reencoded_residual,
            code_indices=edited_t,
        )

    gt_xyz = recover_xyz(torch.from_numpy(motion_norm), mean_np, std_np)
    base_xyz = recover_xyz(base_norm[0], mean_np, std_np)
    edit_xyz = recover_xyz(edit_norm[0], mean_np, std_np)

    T = min(len(gt_xyz), len(base_xyz), len(edit_xyz), L)
    gt_xyz = gt_xyz[:T]
    base_xyz = base_xyz[:T]
    edit_xyz = edit_xyz[:T]

    fs = min(int(pr["frame_start"]), T)
    fe = min(int(pr["frame_end"]), T)
    inside = np.zeros(T, dtype=bool)
    inside[fs:fe] = True
    outside = ~inside
    if inside.sum() == 0:
        raise RuntimeError(f"{sid}: empty edit window")

    target_joints = np.asarray(target["target_joints"], dtype=np.int64)
    nontarget_joints = np.asarray(
        [j for j in range(22) if j not in set(target_joints.tolist())],
        dtype=np.int64,
    )

    base_ang = angle_deg(base_xyz, target["triplet"])
    edit_ang = angle_deg(edit_xyz, target["triplet"])
    base_cat = angle_category(base_ang)
    edit_cat = angle_category(edit_ang)

    base_err = np.abs(base_cat[inside] - target_cat)
    edit_err = np.abs(edit_cat[inside] - target_cat)

    sign = 1.0 if target_cat > source_med else -1.0
    signed_progress_deg = sign * float(
        edit_ang[inside].mean() - base_ang[inside].mean()
    )

    row = {
        "sample_id": sid,
        "sample_index": int(pr["sample_index"]),
        "strategy": "reencode_residual",
        "target_name": target_name,
        "group_id": int(target["group_id"]),
        "n_frames": T,
        "n_tokens": len(pose_tokens),
        "tok_start": tok_start,
        "tok_end": tok_end,
        "frame_start": fs,
        "frame_end": fe,
        "source_median_cat": source_med,
        "target_cat": target_cat,
        "target_description": str(pr["target_description"]),
        "source_target_cat_gap": int(pr["source_target_cat_gap"]),
        "strict_success_rate": float(np.mean(edit_cat[inside] == target_cat)),
        "toward_target_rate": float(np.mean(edit_err < base_err)),
        "not_worse_rate": float(np.mean(edit_err <= base_err)),
        "category_error_reduction": float(base_err.mean() - edit_err.mean()),
        "signed_angle_progress_deg": signed_progress_deg,
        "direction_success": float(signed_progress_deg > 0),
        "outside_window_mpjpe_mm": mpjpe_mm(edit_xyz, base_xyz, outside, None),
        "inside_nontarget_mpjpe_mm": mpjpe_mm(edit_xyz, base_xyz, inside, nontarget_joints),
        "inside_target_mpjpe_mm": mpjpe_mm(edit_xyz, base_xyz, inside, target_joints),
        "locality_ratio": float(
            mpjpe_mm(edit_xyz, base_xyz, inside, target_joints)
            / max(mpjpe_mm(edit_xyz, base_xyz, inside, nontarget_joints), 1e-8)
        ),
        "baseline_recon_mpjpe_mm": mpjpe_mm(base_xyz, gt_xyz),
    }
    rows.append(row)

    if (len(rows) % 10 == 0) or (len(rows) == args.n_samples):
        pd.DataFrame(rows).to_csv(OUT / "exp05b_reencode_partial.csv", index=False)
        elapsed = (time.time() - t0_all) / 60.0
        print(
            f"[RUN] {len(rows)}/{args.n_samples} motions "
            f"({100*len(rows)/args.n_samples:.1f}%) elapsed={elapsed:.1f} min",
            flush=True,
        )

new = pd.DataFrame(rows)
if len(new) != args.n_samples:
    raise RuntimeError(f"Expected {args.n_samples} rows, got {len(new)}")

new.to_csv(OUT / "exp05b_reencode_per_sample.csv", index=False)

# Audit that the unedited baseline matches Exp.05 retain_residual closely.
old_ret = old[old.strategy == "retain_residual"][
    ["sample_id", "baseline_recon_mpjpe_mm"]
].copy()
chk = new[["sample_id", "baseline_recon_mpjpe_mm"]].merge(
    old_ret, on="sample_id", suffixes=("_new", "_exp05")
)
baseline_abs_diff = np.abs(
    chk["baseline_recon_mpjpe_mm_new"] - chk["baseline_recon_mpjpe_mm_exp05"]
)
baseline_max_abs_diff = float(baseline_abs_diff.max())
baseline_mean_abs_diff = float(baseline_abs_diff.mean())

print(
    "[AUDIT] baseline reconstruction diff vs Exp.05 retain_residual:",
    "mean_abs=", baseline_mean_abs_diff,
    "max_abs=", baseline_max_abs_diff,
)

# Combine all three strategies.
old_keep = old[old["sample_id"].isin(new["sample_id"])].copy()
combined = pd.concat([old_keep, new], ignore_index=True)
combined.to_csv(OUT / "exp05b_all_three_per_sample.csv", index=False)

METRICS = [
    "strict_success_rate",
    "toward_target_rate",
    "not_worse_rate",
    "category_error_reduction",
    "signed_angle_progress_deg",
    "direction_success",
    "outside_window_mpjpe_mm",
    "inside_nontarget_mpjpe_mm",
    "inside_target_mpjpe_mm",
    "locality_ratio",
    "baseline_recon_mpjpe_mm",
]

strategies = ["pose_only", "retain_residual", "reencode_residual"]

summary_rows = []
for strategy in strategies:
    d = combined[combined.strategy == strategy]
    row = {"strategy": strategy, "n": len(d)}
    for mi, metric in enumerate(METRICS):
        mean, lo, hi = bootstrap_mean_ci(
            d[metric].to_numpy(),
            B=args.bootstrap,
            seed=2027 + mi,
        )
        row[f"{metric}_mean"] = mean
        row[f"{metric}_ci95_lo"] = lo
        row[f"{metric}_ci95_hi"] = hi
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "exp05b_three_policy_summary.csv", index=False)

by_target_rows = []
for strategy in strategies:
    for target_name in TARGETS:
        d = combined[
            (combined.strategy == strategy)
            & (combined.target_name == target_name)
        ]
        row = {"strategy": strategy, "target_name": target_name, "n": len(d)}
        for mi, metric in enumerate(METRICS):
            mean, lo, hi = bootstrap_mean_ci(
                d[metric].to_numpy(),
                B=args.bootstrap,
                seed=3027 + mi,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        by_target_rows.append(row)

by_target = pd.DataFrame(by_target_rows)
by_target.to_csv(OUT / "exp05b_three_policy_by_target.csv", index=False)

# Paired comparisons.
wide = combined.pivot(index="sample_id", columns="strategy", values=METRICS)

pairs = [
    ("retain_residual", "pose_only"),
    ("reencode_residual", "pose_only"),
    ("reencode_residual", "retain_residual"),
]

paired_rows = []
for a, b in pairs:
    for mi, metric in enumerate(METRICS):
        x = (wide[(metric, a)] - wide[(metric, b)]).to_numpy()
        mean, lo, hi = bootstrap_mean_ci(
            x,
            B=args.bootstrap,
            seed=4027 + mi + 100 * pairs.index((a,b)),
        )
        paired_rows.append({
            "metric": metric,
            "comparison": f"{a} - {b}",
            "mean_diff": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
        })

paired = pd.DataFrame(paired_rows)
paired.to_csv(OUT / "exp05b_three_policy_paired.csv", index=False)

audit = {
    "experiment": "Exp05B residual re-encoding policy",
    "n_samples": args.n_samples,
    "seed": args.seed,
    "source_exp05_rows": str(args.exp05_rows),
    "source_exp05_rows_sha256": sha256(args.exp05_rows),
    "tokenizer_checkpoint": str(tokenizer_ckpt),
    "tokenizer_sha256": sha256(tokenizer_ckpt),
    "tokenizer_checkpoint_iter": tokenizer_state.get("nb_iter"),
    "strategies": strategies,
    "baseline_recon_mean_abs_diff_vs_exp05_retain": baseline_mean_abs_diff,
    "baseline_recon_max_abs_diff_vs_exp05_retain": baseline_max_abs_diff,
    "elapsed_minutes": (time.time() - t0_all) / 60.0,
    "outputs": {
        "reencode_per_sample": str(OUT / "exp05b_reencode_per_sample.csv"),
        "all_three_per_sample": str(OUT / "exp05b_all_three_per_sample.csv"),
        "summary": str(OUT / "exp05b_three_policy_summary.csv"),
        "by_target": str(OUT / "exp05b_three_policy_by_target.csv"),
        "paired": str(OUT / "exp05b_three_policy_paired.csv"),
    },
}
(OUT / "exp05b_completion_audit.json").write_text(json.dumps(audit, indent=2))

print("\n=== EXP.05B THREE-POLICY SUMMARY ===")
print(summary.to_string(index=False))

print("\n=== EXP.05B PAIRED POLICY DIFFERENCES ===")
print(paired.to_string(index=False))

print("\n[OK] EXP05B_COMPLETE")
