#!/usr/bin/env python3
"""Patch trajectory files to fill in accumulated_cost based on model pricing.

vLLM-served models report accumulated_cost=0.0 in trajectories. This script
reads each trajectory, computes the correct cost from token counts and model
pricing, and writes the patched data back in place.

Usage:
    uv run python3 patch_trajectory_cost.py --logs_dir eval_minimax_m2_5/logs
    uv run python3 patch_trajectory_cost.py --logs_dir eval_minimax_m2_5/logs --dry-run
"""

import argparse
import glob
import json
import os
import sys

from pricing import compute_cost as _compute_cost


def patch_trajectory(traj_path: str, dry_run: bool = False) -> tuple[bool, str]:
    """Patch a single trajectory file. Returns (changed, message)."""
    with open(traj_path) as f:
        data = json.load(f)

    # Detect model from the first llm_metrics entry
    model = None
    for entry in data:
        m = entry.get("llm_metrics")
        if m:
            usage = m.get("accumulated_token_usage", {})
            model = usage.get("model", "")
            break

    if not model:
        return False, "no llm_metrics found"

    # Quick check: does pricing exist for this model?
    if _compute_cost(model) == 0.0:
        return False, f"no pricing for model: {model}"

    # Check if already patched (last accumulated_cost > 0)
    for entry in reversed(data):
        m = entry.get("llm_metrics")
        if m and "accumulated_cost" in m:
            if m["accumulated_cost"] > 0:
                return False, "already patched"
            break

    # Patch all llm_metrics entries
    patched = 0
    for entry in data:
        m = entry.get("llm_metrics")
        if m and "accumulated_token_usage" in m:
            usage = m["accumulated_token_usage"]
            m["accumulated_cost"] = round(_compute_cost(
                model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cache_read_tokens=usage.get("cache_read_tokens", 0),
                cache_write_tokens=usage.get("cache_write_tokens", 0),
            ), 6)
            patched += 1

    if patched == 0:
        return False, "nothing to patch"

    if not dry_run:
        with open(traj_path, "w") as f:
            json.dump(data, f)

    # Report final cost
    final_cost = 0.0
    for entry in reversed(data):
        m = entry.get("llm_metrics")
        if m and "accumulated_cost" in m:
            final_cost = m["accumulated_cost"]
            break

    return True, f"patched {patched} entries, final cost=${final_cost:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Patch trajectory cost data")
    parser.add_argument("--logs_dir", required=True, help="Path to logs directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if not os.path.isdir(args.logs_dir):
        print(f"Error: {args.logs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    trajectories = glob.glob(os.path.join(args.logs_dir, "*/trajectory"))
    if not trajectories:
        print("No trajectory files found.")
        sys.exit(0)

    print(f"Found {len(trajectories)} trajectory files in {args.logs_dir}")
    if args.dry_run:
        print("(DRY RUN — no files will be modified)\n")

    changed = skipped = errors = 0
    total_cost = 0.0

    for tp in sorted(trajectories):
        folder = os.path.basename(os.path.dirname(tp))
        try:
            did_change, msg = patch_trajectory(tp, dry_run=args.dry_run)
            if did_change:
                changed += 1
                # Extract cost from message
                if "cost=$" in msg:
                    cost_str = msg.split("cost=$")[1]
                    total_cost += float(cost_str)
                print(f"  {folder:<50} {msg}")
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"  {folder:<50} ERROR: {e}")

    print(f"\nDone. Changed: {changed}  Skipped: {skipped}  Errors: {errors}")
    print(f"Total cost across patched trajectories: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
