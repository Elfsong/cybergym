"""Chain launcher for groups 2/3/4: each waits for the previous to finish,
then launches with parallel=8 (vs g0/g1 which used parallel=4).

Waits for g1 first (PID 639790), then on each subsequent group waits for the
freshly-spawned previous group's PID to exit before launching the next.
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/nvidia/Projects/cybergym")
LOG_ROOT = Path("/lp-dev/cybergym_data/pagent")
G1_PID = 639790  # the in-flight group 1 runner; chain after it finishes
GROUPS = [2, 3, 4]
PARALLEL = 8


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wait_for_pid(pid: int, label: str) -> None:
    print(f"[wait] {now_iso()}  waiting for {label} PID={pid} to exit", flush=True)
    while True:
        try:
            os.kill(pid, 0)  # signal 0 = check existence
        except ProcessLookupError:
            print(f"[wait] {now_iso()}  {label} (PID {pid}) is gone", flush=True)
            return
        time.sleep(30)


def launch_group(gi: int) -> int:
    out_dir = f"/data/cybergym_data/cybergym-eval-data/eval_cc_opus47_100_g{gi}"
    log_path = LOG_ROOT / f"run_cc_opus47_100_g{gi}.log"
    cmd = [
        "uv", "run", "python", "run_eval_claude_code_tasks.py",
        "--model", "claude-opus-4-7",
        "--tasks-file", f"TASKS_EVAL_100_seed42_g{gi}",
        "--parallel", str(PARALLEL),
        "--out-dir", out_dir,
        "--max-iter", "72",
        "--timeout", "2400",
        "--effort", "low",
    ]
    print(f"[launch] {now_iso()}  group {gi}  parallel={PARALLEL}", flush=True)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[launch] group {gi} PID={proc.pid}  log: {log_path}", flush=True)
    return proc.pid


def main() -> None:
    print(f"[plan] chain scheduler started at {now_iso()}", flush=True)
    print(f"[plan] config: parallel={PARALLEL}, effort=low, timeout=2400, max-iter=72", flush=True)
    print(f"[plan] chain: wait g1 (PID {G1_PID}) → g2 → g3 → g4 (each waits for prev)", flush=True)
    print("", flush=True)

    prev_pid = G1_PID
    prev_label = "g1"
    for gi in GROUPS:
        wait_for_pid(prev_pid, prev_label)
        # Small buffer so output flushes / docker cleanup
        time.sleep(10)
        new_pid = launch_group(gi)
        prev_pid = new_pid
        prev_label = f"g{gi}"

    print(f"[plan] {now_iso()}  all 3 follow-up groups dispatched (the last is still running)", flush=True)


if __name__ == "__main__":
    main()
