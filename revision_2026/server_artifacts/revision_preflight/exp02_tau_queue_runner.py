
from pathlib import Path
from datetime import datetime
import subprocess, json, re, sys, traceback

CFG_PATH = Path(sys.argv[1])
cfg = json.loads(CFG_PATH.read_text())

cwd = Path(cfg["cwd"])
out_root = Path(cfg["out_root"])
logdir = Path(cfg["logdir"])
state_path = Path(cfg["state_path"])
total_iter = int(cfg["total_iter"])

logdir.mkdir(parents=True, exist_ok=True)
out_root.mkdir(parents=True, exist_ok=True)

def now():
    return datetime.now().isoformat(timespec="seconds")

def load_state():
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except Exception:
            pass
    return {
        "created_at": now(),
        "queue_status": "pending",
        "runs": {}
    }

def save_state(state):
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)

def max_train_iter(log_path):
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    vals = [
        int(x) for x in re.findall(
            r"^\d{4}-\d{2}-\d{2} .*? Train\. Iter\s+(\d+)\s*:",
            text,
            flags=re.M
        )
    ]
    return max(vals) if vals else None

state = load_state()
state["queue_status"] = "running"
state["started_or_resumed_at"] = now()
save_state(state)

for spec in cfg["runs"]:
    tau = float(spec["tau"])
    key = str(tau)
    exp_name = spec["exp_name"]
    cmd = spec["command"]

    prior = state["runs"].get(key, {})
    if prior.get("status") == "complete":
        print(f"[SKIP] tau={tau} already complete", flush=True)
        continue

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = logdir / f"exp02_tau_{spec['tag']}_{stamp}.log"

    state["runs"][key] = {
        "tau": tau,
        "exp_name": exp_name,
        "status": "running",
        "started_at": now(),
        "log": str(run_log),
        "command": cmd,
    }
    save_state(state)

    print(f"[START] tau={tau}", flush=True)
    print(" ".join(cmd), flush=True)

    try:
        with open(run_log, "w") as f:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        rc = int(proc.returncode)
    except Exception as e:
        rc = -999
        with open(run_log, "a") as f:
            f.write("\nQUEUE RUNNER EXCEPTION:\n")
            f.write(traceback.format_exc())

    max_iter = max_train_iter(run_log)

    exp_root = out_root / exp_name
    dirs = sorted(
        [p for p in exp_root.glob("*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime
    ) if exp_root.exists() else []
    run_dir = dirs[-1] if dirs else None

    checkpoint_exists = bool(run_dir and (run_dir / "net_last.pth").exists())
    complete = (rc == 0 and max_iter == total_iter and checkpoint_exists)

    rec = state["runs"][key]
    rec.update({
        "returncode": rc,
        "max_train_iter": max_iter,
        "run_dir": str(run_dir) if run_dir else None,
        "net_last_exists": checkpoint_exists,
        "finished_at": now(),
        "status": "complete" if complete else "failed",
    })
    save_state(state)

    print(
        f"[END] tau={tau} rc={rc} max_iter={max_iter} "
        f"checkpoint={checkpoint_exists} status={rec['status']}",
        flush=True
    )

    if not complete:
        state["queue_status"] = "failed"
        state["failed_tau"] = tau
        state["finished_at"] = now()
        save_state(state)
        sys.exit(1)

state["queue_status"] = "complete"
state["finished_at"] = now()
save_state(state)
print("[QUEUE COMPLETE] all tau runs finished", flush=True)
