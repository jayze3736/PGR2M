
import os, sys, json, math, time, argparse, hashlib, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PGR = Path(".")
os.chdir(PGR)
sys.path.insert(0, str(PGR))

# NumPy 2.x compatibility for older released code.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

from models.pg_tokenizer import PoseGuidedTokenizer
from utils.motion_process import recover_from_ric
from utils.codebook import vq_to_range

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
    cfg = Path(folder) / "arguments.yaml"
    ckpt_path = Path(folder) / "net_best_fid.pth"
    a = _argparse.Namespace(**yaml.safe_load(cfg.read_text()))

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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.cuda().eval()
    for p in net.parameters():
        p.requires_grad = False
    return net, a, ckpt_path, ckpt

def denorm_motion(x, mean, std):
    return x * mean.new_tensor(std) + mean.new_tensor(mean.cpu().numpy()) if False else None

def recover_xyz(motion_norm, mean_np, std_np):
    """motion_norm: [T,263] tensor on CPU or GPU."""
    arr = motion_norm.detach().float().cpu().numpy()
    arr = arr * std_np + mean_np
    xyz = recover_from_ric(torch.from_numpy(arr).float().unsqueeze(0), 22)
    return xyz.squeeze(0).cpu().numpy()

def angle_deg(xyz, triplet):
    """Geometric angle at the middle joint, in degrees."""
    a, b, c = triplet
    v1 = xyz[:, c] - xyz[:, b]
    v2 = xyz[:, a] - xyz[:, b]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / np.clip(n1, 1e-8, None)
    v2 = v2 / np.clip(n2, 1e-8, None)
    cosv = np.clip((v1 * v2).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosv))

ANGLE_THRESHOLDS = np.arange(10, 180, 10, dtype=np.float32)

def angle_category(angles):
    # categories: <=10 -> 0, <=20 -> 1, ..., >170 -> 17
    return np.digitize(angles, ANGLE_THRESHOLDS, right=True).astype(np.int64)

TARGETS = [
    dict(group_id=0, name="left_knee",  triplet=(1,4,7),   target_joints=(1,4,7)),
    dict(group_id=1, name="right_knee", triplet=(2,5,8),   target_joints=(2,5,8)),
    dict(group_id=2, name="left_elbow", triplet=(16,18,20), target_joints=(16,18,20)),
    dict(group_id=3, name="right_elbow",triplet=(17,19,21), target_joints=(17,19,21)),
]

def choose_target_cat(source_median, shift_bins):
    source_median = int(source_median)
    if source_median <= 8:
        return min(17, source_median + shift_bins)
    return max(0, source_median - shift_bins)

def target_description(cat):
    if cat == 0:
        return "<=10 deg"
    if cat == 17:
        return ">170 deg (near straight)"
    lo = cat * 10
    hi = (cat + 1) * 10
    return f"({lo},{hi}] deg"

def edit_pose_tokens(pose_tokens, group_id, tok_start, tok_end, target_cat):
    out = pose_tokens.copy()
    g_end, g_start = vq_to_range[group_id]
    assert (g_end - g_start + 1) == 18
    out[tok_start:tok_end, g_start:g_end+1] = 0.0
    out[tok_start:tok_end, g_start + target_cat] = 1.0
    return out

def pose_group_categories(pose_tokens, group_id):
    g_end, g_start = vq_to_range[group_id]
    block = pose_tokens[:, g_start:g_end+1]
    if not np.all(block.sum(axis=1) > 0.5):
        raise RuntimeError(f"Invalid one-hot pose-code block for group {group_id}")
    return np.argmax(block, axis=1)

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
parser.add_argument("--shift-bins", type=int, default=4)
parser.add_argument("--window-frac", type=float, default=0.30)
parser.add_argument("--max-frames", type=int, default=196)
parser.add_argument("--min-frames", type=int, default=64)
parser.add_argument("--tokenizer-dir", type=str, required=True)
parser.add_argument("--data-root", type=str, default="dataset/HumanML3D")
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--bootstrap", type=int, default=1000)
args = parser.parse_args()

fixseed(args.seed)
device = torch.device("cuda:0")
DATA = Path(args.data_root)
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

mean_np = np.load(DATA / "Mean.npy").astype(np.float32)
std_np = np.load(DATA / "Std.npy").astype(np.float32)
assert mean_np.shape == (263,), mean_np.shape
assert std_np.shape == (263,), std_np.shape

net, net_args, tokenizer_ckpt, tokenizer_state = build_tokenizer(args.tokenizer_dir)
print("[OK] Tokenizer loaded:", tokenizer_ckpt)
print("[OK] Tokenizer checkpoint iter:", tokenizer_state.get("nb_iter"))
print("[OK] Tokenizer SHA256:", sha256(tokenizer_ckpt))

with open(DATA / "test.txt", "r") as f:
    test_ids = [x.strip() for x in f if x.strip()]

eligible = []
for sid in sorted(test_ids):
    mp = DATA / "new_joint_vecs" / f"{sid}.npy"
    cp = DATA / "codes" / f"{sid}.npy"
    if not mp.exists() or not cp.exists():
        continue
    try:
        lm = len(np.load(mp, mmap_mode="r"))
        lc = len(np.load(cp, mmap_mode="r"))
    except Exception:
        continue
    L = min(lm, lc, args.max_frames)
    L = (L // 4) * 4
    if L >= args.min_frames:
        eligible.append(sid)

rng = np.random.default_rng(args.seed)
eligible = np.asarray(eligible, dtype=object)
rng.shuffle(eligible)

if len(eligible) < args.n_samples:
    raise RuntimeError(f"Only {len(eligible)} eligible test motions for requested {args.n_samples}")

selected = eligible[:args.n_samples].tolist()
(OUT / "exp05_selected_samples.json").write_text(json.dumps({
    "seed": args.seed,
    "n_samples": args.n_samples,
    "selected": selected,
}, indent=2))

config = {
    "experiment": "Exp05 quantitative controlled editing",
    "n_samples": args.n_samples,
    "seed": args.seed,
    "shift_bins": args.shift_bins,
    "nominal_shift_deg": args.shift_bins * 10,
    "window_frac": args.window_frac,
    "max_frames": args.max_frames,
    "min_frames": args.min_frames,
    "targets": TARGETS,
    "strategies": ["pose_only", "retain_residual"],
    "tokenizer_checkpoint": str(tokenizer_ckpt),
    "tokenizer_sha256": sha256(tokenizer_ckpt),
    "mean_path": str(DATA / "Mean.npy"),
    "std_path": str(DATA / "Std.npy"),
}
(OUT / "exp05_protocol_freeze.json").write_text(json.dumps(config, indent=2))

rows = []
error_rows = []
t0_all = time.time()

for sidx, sid in enumerate(selected):
    target = TARGETS[sidx % len(TARGETS)]
    group_id = target["group_id"]

    try:
        motion_raw = np.load(DATA / "new_joint_vecs" / f"{sid}.npy").astype(np.float32)
        pose_full = np.load(DATA / "codes" / f"{sid}.npy").astype(np.float32)

        L = min(len(motion_raw), len(pose_full), args.max_frames)
        L = (L // 4) * 4
        motion_raw = motion_raw[:L]
        pose_tokens = pose_full[:L:4]

        ntok = len(pose_tokens)
        win = max(4, int(round(ntok * args.window_frac)))
        win = min(win, max(4, ntok - 2))
        tok_start = max(1, (ntok - win) // 2)
        tok_end = min(ntok - 1, tok_start + win)
        if tok_end <= tok_start:
            raise RuntimeError(f"Bad edit window: ntok={ntok}, {tok_start}:{tok_end}")

        source_cats = pose_group_categories(pose_tokens, group_id)
        source_med = int(np.rint(np.median(source_cats[tok_start:tok_end])))
        target_cat = choose_target_cat(source_med, args.shift_bins)

        edited_tokens = edit_pose_tokens(
            pose_tokens, group_id, tok_start, tok_end, target_cat
        )

        motion_norm = ((motion_raw - mean_np) / std_np).astype(np.float32)

        motion_t = torch.from_numpy(motion_norm).float().unsqueeze(0).cuda()
        pose_t = torch.from_numpy(pose_tokens).float().unsqueeze(0).cuda()
        edited_t = torch.from_numpy(edited_tokens).float().unsqueeze(0).cuda()

        with torch.no_grad():
            # Released editing behavior: residual branch dropped.
            base_pose_norm, *_ = net(
                code_indices=pose_t,
                motion=None,
                drop_out_residual_quantization=True,
            )
            edit_pose_norm, *_ = net(
                code_indices=edited_t,
                motion=None,
                drop_out_residual_quantization=True,
            )

            # Alternative residual policy: retain original residual tokens.
            _, residual_onehot = net.tokenize(
                motion=motion_t,
                code_indices=pose_t,
                return_one_hot=True,
            )
            base_ret_norm = net.inference(
                residual_codes=residual_onehot,
                code_indices=pose_t,
            )
            edit_ret_norm = net.inference(
                residual_codes=residual_onehot,
                code_indices=edited_t,
            )

        decoded = {
            "pose_only": (base_pose_norm[0], edit_pose_norm[0]),
            "retain_residual": (base_ret_norm[0], edit_ret_norm[0]),
        }

        # Ground truth reconstruction reference from the same RIC representation.
        gt_xyz = recover_xyz(torch.from_numpy(motion_norm), mean_np, std_np)

        frame_start = tok_start * 4
        frame_end = min(tok_end * 4, L)

        for strategy, (base_norm, edit_norm) in decoded.items():
            base_xyz = recover_xyz(base_norm, mean_np, std_np)
            edit_xyz = recover_xyz(edit_norm, mean_np, std_np)

            T = min(len(base_xyz), len(edit_xyz), len(gt_xyz), L)
            base_xyz = base_xyz[:T]
            edit_xyz = edit_xyz[:T]
            gt_xyz_s = gt_xyz[:T]

            fs = min(frame_start, T)
            fe = min(frame_end, T)
            inside = np.zeros(T, dtype=bool)
            inside[fs:fe] = True
            outside = ~inside

            target_joints = np.asarray(target["target_joints"], dtype=np.int64)
            nontarget_joints = np.asarray(
                [j for j in range(22) if j not in set(target_joints.tolist())],
                dtype=np.int64,
            )

            base_ang = angle_deg(base_xyz, target["triplet"])
            edit_ang = angle_deg(edit_xyz, target["triplet"])
            base_cat = angle_category(base_ang)
            edit_cat = angle_category(edit_ang)

            if inside.sum() == 0:
                raise RuntimeError("Empty inside edit window")

            base_err = np.abs(base_cat[inside] - target_cat)
            edit_err = np.abs(edit_cat[inside] - target_cat)

            sign = 1.0 if target_cat > source_med else -1.0
            signed_progress_deg = sign * float(
                edit_ang[inside].mean() - base_ang[inside].mean()
            )

            strict_success = float(np.mean(edit_cat[inside] == target_cat))
            toward_rate = float(np.mean(edit_err < base_err))
            not_worse_rate = float(np.mean(edit_err <= base_err))
            cat_error_reduction = float(base_err.mean() - edit_err.mean())
            direction_success = float(signed_progress_deg > 0)

            outside_mpjpe = mpjpe_mm(edit_xyz, base_xyz, outside, None)
            inside_non = mpjpe_mm(
                edit_xyz, base_xyz, inside, nontarget_joints
            )
            inside_target = mpjpe_mm(
                edit_xyz, base_xyz, inside, target_joints
            )
            locality_ratio = float(inside_target / max(inside_non, 1e-8))
            baseline_recon = mpjpe_mm(base_xyz, gt_xyz_s)

            row = {
                "sample_id": sid,
                "sample_index": sidx,
                "strategy": strategy,
                "target_name": target["name"],
                "group_id": group_id,
                "n_frames": T,
                "n_tokens": ntok,
                "tok_start": tok_start,
                "tok_end": tok_end,
                "frame_start": fs,
                "frame_end": fe,
                "source_median_cat": source_med,
                "target_cat": target_cat,
                "target_description": target_description(target_cat),
                "source_target_cat_gap": abs(target_cat - source_med),
                "strict_success_rate": strict_success,
                "toward_target_rate": toward_rate,
                "not_worse_rate": not_worse_rate,
                "category_error_reduction": cat_error_reduction,
                "signed_angle_progress_deg": signed_progress_deg,
                "direction_success": direction_success,
                "outside_window_mpjpe_mm": outside_mpjpe,
                "inside_nontarget_mpjpe_mm": inside_non,
                "inside_target_mpjpe_mm": inside_target,
                "locality_ratio": locality_ratio,
                "baseline_recon_mpjpe_mm": baseline_recon,
            }
            rows.append(row)

        if (sidx + 1) % 10 == 0 or (sidx + 1) == len(selected):
            pd.DataFrame(rows).to_csv(OUT / "exp05_per_sample_partial.csv", index=False)
            elapsed = (time.time() - t0_all) / 60
            print(
                f"[RUN] {sidx+1}/{len(selected)} motions "
                f"({100*(sidx+1)/len(selected):.1f}%) elapsed={elapsed:.1f} min",
                flush=True,
            )

    except Exception as e:
        error_rows.append({
            "sample_id": sid,
            "sample_index": sidx,
            "target_name": target["name"],
            "error": repr(e),
        })
        pd.DataFrame(error_rows).to_csv(OUT / "exp05_errors.csv", index=False)
        print(f"[ERROR] {sid}: {repr(e)}", flush=True)
        raise

df = pd.DataFrame(rows)
if len(df) != args.n_samples * 2:
    raise RuntimeError(f"Expected {args.n_samples*2} rows, got {len(df)}")

per_sample = OUT / "exp05_per_sample.csv"
df.to_csv(per_sample, index=False)

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

summary_rows = []
for strategy in ["pose_only", "retain_residual"]:
    d = df[df.strategy == strategy]
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
summary.to_csv(OUT / "exp05_summary.csv", index=False)

by_target_rows = []
for strategy in ["pose_only", "retain_residual"]:
    for target_name in [t["name"] for t in TARGETS]:
        d = df[(df.strategy == strategy) & (df.target_name == target_name)]
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
by_target.to_csv(OUT / "exp05_by_target.csv", index=False)

# Paired residual-policy difference: retain_residual - pose_only.
wide = df.pivot(index="sample_id", columns="strategy", values=METRICS)
paired_rows = []
for mi, metric in enumerate(METRICS):
    x = (
        wide[(metric, "retain_residual")]
        - wide[(metric, "pose_only")]
    ).to_numpy()
    mean, lo, hi = bootstrap_mean_ci(
        x, B=args.bootstrap, seed=4027 + mi
    )
    paired_rows.append({
        "metric": metric,
        "difference": "retain_residual - pose_only",
        "mean_diff": mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
    })

paired = pd.DataFrame(paired_rows)
paired.to_csv(OUT / "exp05_residual_policy_paired.csv", index=False)

audit = {
    **config,
    "completed_samples": args.n_samples,
    "rows": len(df),
    "elapsed_minutes": (time.time() - t0_all) / 60,
    "per_sample_csv": str(per_sample),
    "summary_csv": str(OUT / "exp05_summary.csv"),
    "by_target_csv": str(OUT / "exp05_by_target.csv"),
    "residual_policy_paired_csv": str(OUT / "exp05_residual_policy_paired.csv"),
    "errors": len(error_rows),
}
(OUT / "exp05_completion_audit.json").write_text(json.dumps(audit, indent=2))

print("\n=== EXP.05 SUMMARY ===")
print(summary.to_string(index=False))
print("\n=== RESIDUAL POLICY PAIRED DIFFERENCE ===")
print(paired.to_string(index=False))
print("\n[OK] EXP05_COMPLETE")
