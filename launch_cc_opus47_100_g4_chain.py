"""Chain g4 right after g3 finishes (no inter-group wait).

Watches the in-flight g3 PID 3811273; when it exits, sleeps 10s for cleanup,
then launches g4 with parallel=8 / effort=low / timeout=2400 / max-iter=72.
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/nvidia/Projects/cybergym")
LOG_ROOT = Path("/lp-dev/cybergym_data/pagent")
G3_PID = 3811273


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wait_for_pid(pid: int, label: str) -> None:
    print(f"[wait] {now_iso()}  waiting for {label} PID={pid} to exit", flush=True)
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"[wait] {now_iso()}  {label} (PID {pid}) is gone", flush=True)
            return
        time.sleep(15)


def launch_group(gi: int) -> int:
    out_dir = f"/data/cybergym_data/cybergym-eval-data/eval_cc_opus47_100_g{gi}"
    log_path = LOG_ROOT / f"run_cc_opus47_100_g{gi}.log"
    cmd = [
        "uv", "run", "python", "run_eval_claude_code_tasks.py",
        "--model", "claude-opus-4-7",
        "--tasks-file", f"TASKS_EVAL_100_seed42_g{gi}",
        "--parallel", "8",
        "--out-dir", out_dir,
        "--max-iter", "72",
        "--timeout", "2400",
        "--effort", "low",
    ]
    print(f"[launch] {now_iso()}  group {gi}  parallel=8", flush=True)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[launch] group {gi} PID={proc.pid}  log: {log_path}", flush=True)
    return proc.pid


def main() -> None:
    print(f"[plan] g4 chain scheduler started at {now_iso()}", flush=True)
    print(f"[plan] schedule: chain g4 right after g3 (PID {G3_PID})", flush=True)
    wait_for_pid(G3_PID, "g3")
    time.sleep(10)
    g4_pid = launch_group(4)
    print(f"[plan] {now_iso()}  g4 dispatched (PID {g4_pid})", flush=True)


if __name__ == "__main__":
    main()
