
import json, re, subprocess, sys, time
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
state_path = Path(cfg["state_path"])
master_log = Path(cfg["master_log"])
jobs = cfg["jobs"]

train_iter_pat = re.compile(r"Train\. Iter\s+(\d+)\s*:")

def max_iter_from_log(path):
    path = Path(path)
    if not path.exists():
        return None
    m = None
    for line in path.read_text(errors="replace").splitlines():
        z = train_iter_pat.search(line)
        if z:
            m = int(z.group(1))
    return m

def has_checkpoint(exp_name):
    base = Path("exp_result_revision_entropy_sensitivity") / exp_name
    return any(base.rglob("net_best_fid.pth")) if base.exists() else False

def load_state():
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {
        "queue_status": "running",
        "runs": {
            str(j["entropy_weight"]): {
                "status": "pending",
                "log": j["log"],
                "exp_name": j["exp_name"],
            } for j in jobs
        }
    }

def save(s):
    state_path.write_text(json.dumps(s, indent=2))

def master(msg):
    with master_log.open("a") as f:
        f.write(msg.rstrip() + "\n")
        f.flush()

state = load_state()
state["queue_status"] = "running"
save(state)

for j in jobs:
    k = str(j["entropy_weight"])
    rec = state["runs"].setdefault(k, {})
    log_path = Path(j["log"])

    # Re-launch safety: skip already-complete production runs.
    if rec.get("status") == "complete":
        mi = max_iter_from_log(log_path)
        ck = has_checkpoint(j["exp_name"])
        if mi == 300000 and ck:
            master(f"[SKIP] lambda_ent={j['entropy_weight']} already complete")
            continue

    rec.update({
        "status": "running",
        "log": j["log"],
        "exp_name": j["exp_name"],
        "started_at": time.time(),
    })
    save(state)

    master("[START] lambda_ent={} {}".format(
        j["entropy_weight"], " ".join(j["command"])
    ))

    with log_path.open("w") as f:
        p = subprocess.run(
            j["command"],
            cwd=".",
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    mi = max_iter_from_log(log_path)
    ck = has_checkpoint(j["exp_name"])
    ok = (p.returncode == 0 and mi == 300000 and ck)

    rec.update({
        "returncode": p.returncode,
        "max_iter": mi,
        "checkpoint": ck,
        "finished_at": time.time(),
        "status": "complete" if ok else "failed",
    })
    save(state)

    master(
        f"[END] lambda_ent={j['entropy_weight']} rc={p.returncode} "
        f"max_iter={mi} checkpoint={ck} status={rec['status']}"
    )

    if not ok:
        state["queue_status"] = "failed"
        save(state)
        master("[QUEUE FAILED]")
        sys.exit(1)

state["queue_status"] = "complete"
save(state)
master("[QUEUE COMPLETE] all new entropy runs finished")
