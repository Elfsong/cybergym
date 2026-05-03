"""Sequential 5-group launcher for Claude Code + Claude Opus 4.7 on the
100-task seed-42 sample.

Each group: 20 tasks, parallel=4, timeout=2400s, effort=low.
Gap between group launches: 5 hours (matches Anthropic Claude Max rolling cap).
Total wall-clock plan: ~22 hours.

Each group writes to its own out-dir so partial results survive a per-group
failure. Launches use nohup-equivalent (start_new_session=True + redirected
stdout/stderr) so they detach from this script's process tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/nvidia/Projects/cybergym")
LOG_ROOT = Path("/lp-dev/cybergym_data/pagent")
N_GROUPS = 5
GAP_SECONDS = 5 * 3600  # 5h between group launches


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def launch_group(gi: int) -> int:
    out_dir = f"/data/cybergym_data/cybergym-eval-data/eval_cc_opus47_100_g{gi}"
    log_path = LOG_ROOT / f"run_cc_opus47_100_g{gi}.log"
    tasks_file = f"TASKS_EVAL_100_seed42_g{gi}"

    cmd = [
        "uv", "run", "python", "run_eval_claude_code_tasks.py",
        "--model", "claude-opus-4-7",
        "--tasks-file", tasks_file,
        "--parallel", "4",
        "--out-dir", out_dir,
        "--max-iter", "72",
        "--timeout", "2400",
        "--effort", "low",
    ]
    print(f"[launch] {now_iso()}  group {gi}  ({tasks_file} → {out_dir})", flush=True)
    print(f"[launch] cmd: {' '.join(cmd)}", flush=True)
    print(f"[launch] log: {log_path}", flush=True)

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[launch] group {gi} PID={proc.pid}", flush=True)
    return proc.pid


def main() -> None:
    t0 = time.time()
    print(f"[plan] Master scheduler started at {now_iso()}; "
          f"5 groups × {GAP_SECONDS//3600}h gap; expected total ~22h", flush=True)
    print(f"[plan] launch schedule:", flush=True)
    for gi in range(N_GROUPS):
        target = t0 + gi * GAP_SECONDS
        print(f"  g{gi}: {datetime.fromtimestamp(target, tz=timezone.utc).isoformat(timespec='seconds')}", flush=True)
    print("", flush=True)

    pids = []
    for gi in range(N_GROUPS):
        target = t0 + gi * GAP_SECONDS
        wait_s = target - time.time()
        if wait_s > 0:
            print(f"[plan] sleeping {wait_s:.0f}s until group {gi} launch", flush=True)
            time.sleep(wait_s)
        pid = launch_group(gi)
        pids.append((gi, pid))

    print(f"[plan] {now_iso()}  all {N_GROUPS} groups dispatched", flush=True)
    for gi, pid in pids:
        print(f"  g{gi}: PID={pid}", flush=True)


if __name__ == "__main__":
    main()
