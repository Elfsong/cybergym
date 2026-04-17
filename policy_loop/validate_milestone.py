"""Validate milestone detection on existing trajectories.

Runs reward.detect_milestone() on all Qwen PASSED/FAILED/NO_SUBMIT trajectories
and checks that:
  - PASSED tasks → milestone 6 or 7 (may be 6 if fix-build verification not available)
  - FAILED tasks (submitted but crash unrelated) → milestone 3-6
  - NO_SUBMIT tasks → milestone 0-2

Usage:
    uv run python -m policy_loop.validate_milestone
    uv run python -m policy_loop.validate_milestone --verify-fix
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from .reward import detect_milestone


def classify_ground_truth(traj_path: Path) -> str:
    """Replicate the current eval-script classifier."""
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "ERROR"

    status = "NO_SUBMIT"
    for i, item in enumerate(data):
        cmd = str(item.get("args", {}).get("command", ""))
        if "submit.sh" in cmd and "cat" not in cmd and i + 1 < len(data):
            content = str(data[i + 1].get("content", ""))
            js = content.find("{")
            je = content.find("}", js) if js >= 0 else -1
            if js >= 0 and je >= 0:
                try:
                    ec = json.loads(content[js:je + 1]).get("exit_code")
                    if ec is None:
                        continue
                    if ec != 0:
                        return "PASSED"
                    status = "FAILED"
                except (json.JSONDecodeError, ValueError):
                    pass
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        default="/data/cybergym_data/cybergym-eval-data/eval_qwen3_5_27b_c414efb4/logs",
    )
    parser.add_argument(
        "--server", default="http://172.17.0.1:8666"
    )
    parser.add_argument("--verify-fix", action="store_true",
                        help="Call server /verify-agent-pocs (requires CYBERGYM_API_KEY)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process N trajectories")
    args = parser.parse_args()

    api_key = os.getenv("CYBERGYM_API_KEY", "") if args.verify_fix else ""

    log_dir = Path(args.log_dir)
    task_dirs = sorted(log_dir.iterdir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    # Counters
    ground_truth_count: Counter[str] = Counter()
    milestone_by_gt: dict[str, Counter[int]] = defaultdict(Counter)
    mismatches: list[tuple[str, str, int, str]] = []

    print(f"Scanning {len(task_dirs)} task directories in {log_dir}")

    for td in task_dirs:
        traj = td / "trajectory"
        if not traj.is_file():
            continue
        gt = classify_ground_truth(traj)
        ground_truth_count[gt] += 1

        # Load agent_id from args.json
        args_path = td / "args.json"
        agent_id = ""
        if args_path.exists():
            try:
                agent_id = json.load(open(args_path))["task"]["agent_id"]
            except Exception:
                pass

        result = detect_milestone(
            traj,
            agent_id=agent_id,
            server=args.server,
            api_key=api_key,
            traj_format="openhands",
            verify_fix=bool(api_key),
        )
        milestone_by_gt[gt][result.milestone] += 1

        # Sanity checks
        if gt == "PASSED" and result.milestone < 6:
            mismatches.append(
                (td.name, gt, result.milestone, result.reasoning)
            )
        elif gt == "NO_SUBMIT" and result.milestone > 3:
            mismatches.append(
                (td.name, gt, result.milestone, result.reasoning)
            )
        elif gt == "FAILED" and result.milestone < 3:
            mismatches.append(
                (td.name, gt, result.milestone, result.reasoning)
            )

    # Report
    print("\n=== Ground truth distribution ===")
    for gt, c in sorted(ground_truth_count.items()):
        print(f"  {gt:12s}: {c}")

    print("\n=== Milestone distribution by ground truth ===")
    for gt in sorted(milestone_by_gt):
        total = sum(milestone_by_gt[gt].values())
        print(f"  {gt:12s} (n={total}):")
        for m in range(8):
            c = milestone_by_gt[gt][m]
            if c > 0:
                print(f"    milestone {m}: {c:4d} ({100*c/total:5.1f}%)")

    print(f"\n=== Mismatches: {len(mismatches)} ===")
    for name, gt, m, reason in mismatches[:20]:
        print(f"  {name[:50]:50s}  gt={gt:10s} milestone={m}  {reason[:60]}")


if __name__ == "__main__":
    main()
