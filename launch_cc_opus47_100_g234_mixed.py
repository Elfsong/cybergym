"""Mixed-cadence launcher for groups 2/3/4 of the Claude Opus 4.7 sweep.

Schedule:
  g2 — chains immediately on g1 finishing (PID watch)
  g3 — launches 5h after g2 launch time (timer)
  g4 — launches 5h after g3 launch time (timer)

All three groups run with parallel=8 (vs g0/g1 which used parallel=4).
The 5h gap on g3/g4 keeps each within its own Anthropic Claude Max
rolling 5-hour window.
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/nvidia/Projects/cybergym")
LOG_ROOT = Path("/lp-dev/cybergym_data/pagent")
G1_PID = 639790                # in-flight g1 runner
GAP_SECONDS = 5 * 3600         # 5h gap between g2→g3 and g3→g4 launch times
PARALLEL = 8


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
        time.sleep(min(wait_s, 600))   # wake every <=10 min so log shows progress


def launch_group(gi: int) -> tuple[int, float]:
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
    return proc.pid, time.time()


def main() -> None:
    print(f"[plan] mixed scheduler started at {now_iso()}", flush=True)
    print(f"[plan] config: parallel={PARALLEL}, effort=low, timeout=2400, max-iter=72", flush=True)
    print(f"[plan] schedule: wait g1 → g2 (chained); g3 = g2_launch + 5h; g4 = g3_launch + 5h", flush=True)
    print("", flush=True)

    # g2: chain immediately after g1 finishes
    wait_for_pid(G1_PID, "g1")
    time.sleep(10)   # buffer for cleanup
    g2_pid, g2_launched_at = launch_group(2)

    # g3: 5h after g2 launch
    sleep_until(g2_launched_at + GAP_SECONDS, "g3 launch")
    g3_pid, g3_launched_at = launch_group(3)

    # g4: 5h after g3 launch
    sleep_until(g3_launched_at + GAP_SECONDS, "g4 launch")
    g4_pid, g4_launched_at = launch_group(4)

    print(f"[plan] {now_iso()}  all 3 groups dispatched (g4 still running)", flush=True)


if __name__ == "__main__":
    main()
