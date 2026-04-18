#!/usr/bin/env python3
"""Extract oracle strategies from MiniMax PASSED trajectories for Group B tasks."""

import json
import os
from pathlib import Path

import requests

VLLM_URL = "http://localhost:8001/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-27B"
MINIMAX_LOG_DIR = "/data/cybergym_data/cybergym-eval-data/eval_minimax_m2_5_ef24bc78/logs"


def summarize_trajectory(traj: list[dict]) -> str:
    """Summarize a PASSED trajectory into key early analysis + last 100 execution steps."""
    lines = []
    for entry in traj:
        source = entry.get("source", "")
        action = entry.get("action", "")
        args = entry.get("args", {})
        content = entry.get("content", "")
        observation = entry.get("observation", "")

        if source == "agent" and action == "run":
            cmd = args.get("command", "")
            if cmd:
                lines.append(f"[CMD] {cmd[:300]}")
        elif source == "agent" and action == "read":
            path = args.get("path", "")
            lines.append(f"[READ] {path}")
        elif source == "agent" and action == "finish":
            lines.append("[FINISH]")
        elif observation == "run" and content:
            text = str(content)[:200]
            lines.append(f"[OUTPUT] {text}")

    if len(lines) <= 140:
        return "\n".join(lines)

    head = lines[:20]
    middle = lines[20:-100]
    n = max(1, len(middle) // 20)
    sampled = [middle[i] for i in range(0, len(middle), n)]
    tail = lines[-100:]
    return "\n".join(
        head
        + [f"\n... (sampled {len(sampled)}/{len(middle)} middle steps) ..."]
        + sampled
        + [f"\n... (last 100 steps) ...\n"]
        + tail
    )


def extract_strategy(task_id: str, traj_summary: str) -> str:
    """Use Qwen 27B to summarize a trajectory into a strategy."""
    prompt = f"""You are analyzing a successful vulnerability reproduction trajectory. Summarize it into a high-level strategy plan (200-300 words) that another agent could follow to reproduce the same vulnerability.

Your strategy should describe:
1. Which source files to examine and why
2. What input format to construct (binary structure, headers, etc.)
3. How to build the proof-of-concept (PoC) file
4. How to validate that the target crash is triggered

Be specific about the approach but do NOT write exact commands. Focus on the reasoning and approach.

## Task: {task_id}

## Trajectory (key actions):
{traj_summary}

## Strategy:"""

    resp = requests.post(
        VLLM_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "temperature": 0.3,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main():
    tasks_file = Path(__file__).parent / "tasks.json"
    with open(tasks_file) as f:
        tasks = json.load(f)

    # Only Group B tasks (cross-oracle from MiniMax)
    group_b = [t for t in tasks if t["group"] == "B_cross_oracle"]
    print(f"Extracting strategies for {len(group_b)} Group B tasks from MiniMax trajectories")

    strategies = []
    for i, task in enumerate(group_b):
        task_id = task["task_id"]
        traj_dir = task["dir"]
        traj_path = os.path.join(MINIMAX_LOG_DIR, traj_dir, "trajectory")

        print(f"[{i+1}/{len(group_b)}] Extracting strategy for {task_id}...", flush=True)

        try:
            with open(traj_path) as f:
                traj = json.load(f)

            summary = summarize_trajectory(traj)
            strategy = extract_strategy(task_id, summary)

            strategies.append({
                "task_id": task_id,
                "dir": traj_dir,
                "source": "minimax",
                "strategy": strategy,
            })
            print(f"  OK ({len(strategy)} chars)", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            strategies.append({
                "task_id": task_id,
                "dir": traj_dir,
                "source": "minimax",
                "strategy": None,
                "error": str(e),
            })

    # Merge with existing Group A strategies
    existing_path = Path(__file__).parent / "oracle_strategies.json"
    with open(existing_path) as f:
        existing = json.load(f)

    # Keep only Group A strategies (from Qwen)
    group_a_tasks = {t["task_id"] for t in tasks if t["group"] == "A_self_oracle"}
    group_a_strategies = [s for s in existing if s["task_id"] in group_a_tasks]

    # Merge
    all_strategies = group_a_strategies + strategies
    with open(existing_path, "w") as f:
        json.dump(all_strategies, f, indent=2)

    print(f"\nMerged: {len(group_a_strategies)} Group A + {len(strategies)} Group B = {len(all_strategies)} total")
    print(f"Saved to {existing_path}")


if __name__ == "__main__":
    main()
