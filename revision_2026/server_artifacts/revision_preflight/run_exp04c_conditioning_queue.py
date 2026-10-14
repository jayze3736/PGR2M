
import os, subprocess, json
from pathlib import Path

PGR = Path(".")
PY = Path("venv/bin/python")
PRE = PGR / "revision_preflight"
RESULTS = PGR / "revision_results" / "exp04c_conditioning"
LOGDIR = PGR / "revision_logs" / "exp04c_conditioning"
EVAL_OUT = RESULTS / "eval_runs"
WRAPPER = PRE / "eval_refine_trans_exp04c_intervention.py"

b = json.loads((PGR/"revision_results"/"exp04b_refine"/"exp04b_refine_completion_audit.json").read_text())
REFINE_DIR = Path(b["run_dir"])
BASE_DIR = Path(b["base_dir"])
TOKENIZER_DIR = Path(b["tokenizer_dir"])

conditions = ["full", "no_text", "no_pose", "no_stage", "no_history"]
repeats = 20
seed = 123

def args_for(cond):
    return [
        "--exp-name", f"exp04c_{cond}",
        "--out-dir", str(EVAL_OUT),
        "--dataname", "t2m",
        "--dec-checkpoint-folder", str(TOKENIZER_DIR),
        "--t2m-checkpoint-folder", str(BASE_DIR),
        "--residual-t2m-checkpoint-folder", str(REFINE_DIR),
        "--eval-reference", "net_best_fid",
        "--eval-mode", "fast",
        "--use-keywords",
        "--seed", str(seed),
    ]

for cond in conditions:
    log = LOGDIR / f"exp04c_{cond}.log"
    old = log.read_text(errors="replace") if log.exists() else ""
    if "final result:" in old and "conf." in old:
        print(f"[SKIP] {cond}: completed", flush=True)
        continue

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["EXP04C_CONDITION"] = cond
    env["EXP04C_REPEATS"] = str(repeats)
    env["EXP04C_DRYRUN"] = "0"

    cmd = [str(PY), "-u", str(WRAPPER), *args_for(cond)]
    print(f"[RUN] {cond}", flush=True)
    print(" ".join(cmd), flush=True)

    with open(log, "w") as f:
        p = subprocess.run(
            cmd, cwd=PGR, env=env,
            stdout=f, stderr=subprocess.STDOUT
        )

    print(f"[DONE] {cond}: returncode={p.returncode}", flush=True)
    if p.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-100:])
        print(tail, flush=True)
        raise SystemExit(p.returncode)

print("[OK] EXP04C_QUEUE_COMPLETE", flush=True)
