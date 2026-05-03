"""Launch g3 immediately, then g4 = (g3 finish) + 3h.

Both groups run with parallel=8 / effort=low / timeout=2400 / max-iter=72.
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/nvidia/Projects/cybergym")
LOG_ROOT = Path("/lp-dev/cybergym_data/pagent")
WAIT_AFTER_G3_FINISH = 3 * 3600   # 3h


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
        time.sleep(30)


def sleep_until(target_epoch: float, label: str) -> None:
    while True:
        now = time.time()
        if now >= target_epoch:
            return
        wait_s = target_epoch - now
        print(f"[wait] {now_iso()}  sleeping {wait_s:.0f}s until {label} "
              f"({datetime.fromtimestamp(target_epoch, tz=timezone.utc).isoformat(timespec='seconds')})",
              flush=True)
        time.sleep(min(wait_s, 600))


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
    print(f"[plan] g3-now-g4-3h scheduler started at {now_iso()}", flush=True)
    print(f"[plan] schedule: g3 NOW; g4 = (g3 finish) + 3h", flush=True)
    print("", flush=True)

    g3_pid = launch_group(3)

    # Wait for g3 to finish
    wait_for_pid(g3_pid, "g3")
    g3_finish = time.time()
    print(f"[plan] g3 finished at {now_iso()}; sleeping 3h before g4", flush=True)

    sleep_until(g3_finish + WAIT_AFTER_G3_FINISH, "g4 launch")
    g4_pid = launch_group(4)

    print(f"[plan] {now_iso()}  g4 dispatched (PID {g4_pid})", flush=True)


if __name__ == "__main__":
    main()
