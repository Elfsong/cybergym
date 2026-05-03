"""Resume launcher for groups 1-4 (group 0 already complete).

g1 fires immediately, then g2/g3/g4 at +5h gaps. Same parallel=4 / timeout=2400 /
effort=low / claude-opus-4-7 config as the master scheduler. Uses the patched
result parser (fixed multi-submit bug) since each launch spawns a new Python
process.
"""
from __future__ import annotations
import os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/nvidia/Projects/cybergym")
LOG_ROOT = Path("/lp-dev/cybergym_data/pagent")
GROUPS = [1, 2, 3, 4]
GAP_SECONDS = 5 * 3600

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def launch_group(gi: int) -> int:
    out_dir = f"/data/cybergym_data/cybergym-eval-data/eval_cc_opus47_100_g{gi}"
    log_path = LOG_ROOT / f"run_cc_opus47_100_g{gi}.log"
    cmd = [
        "uv", "run", "python", "run_eval_claude_code_tasks.py",
        "--model", "claude-opus-4-7",
        "--tasks-file", f"TASKS_EVAL_100_seed42_g{gi}",
        "--parallel", "4",
        "--out-dir", out_dir,
        "--max-iter", "72",
        "--timeout", "2400",
        "--effort", "low",
    ]
    print(f"[launch] {now_iso()}  group {gi}", flush=True)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[launch] group {gi} PID={proc.pid}  log: {log_path}", flush=True)
    return proc.pid

def main() -> None:
    t0 = time.time()
    print(f"[plan] resume scheduler started at {now_iso()}", flush=True)
    for i, gi in enumerate(GROUPS):
        target = t0 + i * GAP_SECONDS
        print(f"  g{gi}: {datetime.fromtimestamp(target, tz=timezone.utc).isoformat(timespec='seconds')}", flush=True)
    print("", flush=True)
    for i, gi in enumerate(GROUPS):
        target = t0 + i * GAP_SECONDS
        wait_s = target - time.time()
        if wait_s > 0:
            print(f"[plan] sleeping {wait_s:.0f}s until group {gi}", flush=True)
            time.sleep(wait_s)
        launch_group(gi)
    print(f"[plan] {now_iso()}  all 4 groups dispatched", flush=True)

if __name__ == "__main__":
    main()
