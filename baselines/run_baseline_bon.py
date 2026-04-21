"""Best-of-N baseline driver for Phase 4 agent rollouts on TASKS_EVAL.

Launches N independent rollouts per task for one of two baselines:

  --baseline pagent    : OpenHands + Qwen3.5-27B + PAGENT SA injection + discipline prompt
                         (output: /lp-dev/cybergym_data/pagent/phase4/...)

  --baseline openhands : OpenHands + Qwen3.5-27B, vanilla default prompt, no SA hints
                         (output: /lp-dev/cybergym_data/openhands_baseline/phase4/...)

Usage:
    uv run python baselines/run_baseline_bon.py \
        --baseline pagent \
        --tasks TASKS_EVAL \
        --n 8 \
        --parallel 32

    # Resume is automatic — any (task, attempt) whose milestone.json
    # already exists under out_root is skipped.

Reports incremental progress to stdout and a per-baseline results JSONL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Make runner scripts importable
REPO_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT / "baselines/pagent/runner"))


CYBERGYM_DB = Path("/lp-dev/cybergym_data/pagent/db/db.json")
PAGENT_OUT_ROOT   = Path("/lp-dev/cybergym_data/pagent/phase4")
PAGENT_LOG_ROOT   = Path("/lp-dev/cybergym_data/pagent/logs/phase4")
OH_OUT_ROOT       = Path("/lp-dev/cybergym_data/openhands_baseline/phase4")
OH_LOG_ROOT       = Path("/lp-dev/cybergym_data/openhands_baseline/logs/phase4")


def _roots(baseline: str) -> tuple[Path, Path, bool]:
    if baseline == "pagent":
        return PAGENT_OUT_ROOT, PAGENT_LOG_ROOT, True       # inject_sa=True
    elif baseline == "openhands":
        return OH_OUT_ROOT, OH_LOG_ROOT, False              # inject_sa=False
    raise ValueError(baseline)


def load_task_to_project() -> dict[str, str]:
    with open(CYBERGYM_DB) as f:
        by_project = json.load(f)
    out = {}
    for proj, tids in by_project.items():
        for tid in tids:
            out[tid] = proj
    return out


def load_tasks(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def _worker(task_id: str, project: str, attempt_idx: int,
            baseline: str, max_iter: int, timeout: int) -> dict:
    """Runs in a subprocess. Catches everything so the pool doesn't crash."""
    start = time.monotonic()
    try:
        from phase4_agent import run_one
        out_root, log_root, inject_sa = _roots(baseline)
        r = run_one(task_id, project,
                    attempt_idx=attempt_idx,
                    out_root=out_root,
                    log_root=log_root,
                    inject_sa=inject_sa,
                    max_iter=max_iter, timeout=timeout)
        r.setdefault("duration", int(time.monotonic() - start))
        r["baseline"] = baseline
        return r
    except Exception as e:
        return {
            "task_id": task_id, "project": project,
            "attempt_idx": attempt_idx, "baseline": baseline,
            "status": "worker_exception",
            "duration": int(time.monotonic() - start),
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "milestone": 0, "rc": -1,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, choices=("pagent", "openhands"))
    p.add_argument("--tasks", type=Path,
                   default=REPO_ROOT / "TASKS_EVAL",
                   help="Path to task list (one task_id per line)")
    p.add_argument("--n", type=int, default=8,
                   help="Number of rollouts per task (Best-of-N).")
    p.add_argument("--parallel", type=int, default=32,
                   help="Concurrent rollouts across all (task, attempt) pairs.")
    p.add_argument("--max-iter", type=int, default=72)
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument("--projects", type=str, default=None,
                   help="Optional comma-separated project filter")
    p.add_argument("--only-task", type=str, default=None,
                   help="Optional: run a single task_id (smoke test).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if milestone.json exists")
    args = p.parse_args()

    out_root, log_root, inject_sa = _roots(args.baseline)
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    all_tasks = load_tasks(args.tasks)
    if args.only_task:
        all_tasks = [t for t in all_tasks if t == args.only_task]
    task_to_proj = load_task_to_project()
    proj_filter = set(args.projects.split(",")) if args.projects else None

    # Flatten (task, attempt_idx) pairs
    plan: list[tuple[str, str, int]] = []
    skipped_existing = 0
    no_proj = 0
    wrong_proj = 0
    for tid in all_tasks:
        proj = task_to_proj.get(tid)
        if proj is None:
            no_proj += 1
            continue
        if proj_filter and proj not in proj_filter:
            wrong_proj += 1
            continue
        num = tid.split(":", 1)[1]
        for k in range(args.n):
            out_dir = out_root / proj / num / f"attempt_{k}"
            milestone_path = out_dir / "milestone.json"
            if not args.force and milestone_path.exists():
                skipped_existing += 1
                continue
            plan.append((tid, proj, k))

    results_jsonl = out_root.parent / f"results_{args.baseline}.jsonl"
    summary_json  = out_root.parent / f"summary_{args.baseline}.json"

    print(f"=== Baseline BoN driver :: {args.baseline} ===")
    print(f"  tasks file   : {args.tasks} ({len(all_tasks)} tasks)")
    print(f"  N per task   : {args.n}")
    print(f"  parallel     : {args.parallel}")
    print(f"  max_iter     : {args.max_iter}")
    print(f"  timeout      : {args.timeout}")
    print(f"  out_root     : {out_root}")
    print(f"  results JSONL: {results_jsonl}")
    print(f"  plan size    : {len(plan)}  "
          f"(skipped_existing={skipped_existing}, no_proj={no_proj}, "
          f"wrong_proj={wrong_proj})")

    if args.dry_run:
        for tid, proj, k in plan[:10]:
            print(f"  {tid:<25} {proj:<20} attempt={k}")
        return

    if not plan:
        print("Nothing to do.")
        return

    t0 = time.monotonic()
    completed = 0
    ok = 0
    fail = 0
    milestones: list[int] = []

    with open(results_jsonl, "a") as out_f, \
         ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {}
        for tid, proj, k in plan:
            fut = pool.submit(_worker, tid, proj, k,
                              args.baseline, args.max_iter, args.timeout)
            futures[fut] = (tid, proj, k)

        for fut in as_completed(futures):
            tid, proj, k = futures[fut]
            completed += 1
            try:
                r = fut.result()
            except Exception as e:
                r = {"task_id": tid, "project": proj, "attempt_idx": k,
                     "baseline": args.baseline, "status": "pool_exception",
                     "duration": 0, "error": str(e),
                     "milestone": 0, "rc": -1}

            out_f.write(json.dumps(r) + "\n")
            out_f.flush()

            mstone = r.get("milestone", 0) or 0
            milestones.append(mstone)
            status = r.get("status", "?")
            mark = "✓" if status in ("ok", "cached") else "✗"
            if mark == "✓":
                ok += 1
            else:
                fail += 1

            elapsed = time.monotonic() - t0
            rate = completed / max(elapsed, 1e-9) * 3600
            remain = len(plan) - completed
            eta = remain / max(completed, 1) * elapsed / 60

            print(f"[{completed:4}/{len(plan)}] {mark} "
                  f"{tid:<25} a={k} {proj:<20} "
                  f"{status:<14} m={mstone} "
                  f"{r.get('duration', 0):>5}s   "
                  f"(rate={rate:.1f}/h ETA={eta:.0f}min)")
            sys.stdout.flush()

    total = time.monotonic() - t0
    avg_m = sum(milestones) / len(milestones) if milestones else 0.0
    print(f"\n=== {args.baseline} BoN done: {ok} ok / {fail} fail / {len(plan)} total "
          f"in {total/60:.1f} min ===")
    print(f"  mean milestone: {avg_m:.2f}")

    # Aggregate Best-of-N per task using everything we have in out_root
    bon = _aggregate_bon(out_root, all_tasks, task_to_proj, args.n)
    with open(summary_json, "w") as f:
        json.dump(bon, f, indent=2)
    print(f"  summary → {summary_json}")


def _aggregate_bon(out_root: Path, tasks: list[str],
                   task_to_proj: dict[str, str], n: int) -> dict:
    """Compute per-task Best-of-N + mean; then overall aggregates."""
    per_task = {}
    total_bon_sum = 0
    total_mean_sum = 0.0
    covered = 0
    for tid in tasks:
        proj = task_to_proj.get(tid)
        if not proj:
            continue
        num = tid.split(":", 1)[1]
        attempts = []
        for k in range(n):
            mpath = out_root / proj / num / f"attempt_{k}" / "milestone.json"
            if mpath.exists():
                try:
                    d = json.loads(mpath.read_text())
                    attempts.append(int(d.get("milestone", 0) or 0))
                except Exception:
                    pass
        if not attempts:
            per_task[tid] = {"project": proj, "attempts": 0,
                             "best": 0, "mean": 0.0}
            continue
        best = max(attempts)
        mean = sum(attempts) / len(attempts)
        per_task[tid] = {"project": proj, "attempts": len(attempts),
                         "best": best, "mean": mean}
        total_bon_sum += best
        total_mean_sum += mean
        covered += 1

    overall = {
        "covered_tasks": covered,
        "total_tasks_in_eval": len(tasks),
        "mean_of_best_N": (total_bon_sum / covered) if covered else 0.0,
        "mean_of_mean_N": (total_mean_sum / covered) if covered else 0.0,
        "per_task": per_task,
    }
    return overall


if __name__ == "__main__":
    main()
