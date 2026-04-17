#!/usr/bin/env python3
"""Extract oracle strategies from PASSED trajectories using Qwen 27B via vLLM."""

import json
import os
import sys
from pathlib import Path

import requests

VLLM_URL = "http://localhost:8001/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-27B"
LOG_DIR = "/data/cybergym_data/cybergym-eval-data/eval_qwen3_5_27b_c414efb4/logs"


def summarize_trajectory(traj: list[dict]) -> str:
    """Summarize a PASSED trajectory's key actions into a concise narrative."""
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
            # Command output — truncate
            text = str(content)[:200]
            lines.append(f"[OUTPUT] {text}")

    return "\n".join(lines[-60:])  # Keep last 60 entries to fit context


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
            "max_tokens": 512,
            "temperature": 0.3,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main():
    tasks_file = Path(__file__).parent.parent / "sensitivity_tasks.json"
    with open(tasks_file) as f:
        tasks = json.load(f)

    output_file = Path(__file__).parent / "oracle_strategies.json"
    strategies = []

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        traj_dir = task["dir"]
        traj_path = os.path.join(LOG_DIR, traj_dir, "trajectory")

        print(f"[{i+1}/{len(tasks)}] Extracting strategy for {task_id}...", flush=True)

        try:
            with open(traj_path) as f:
                traj = json.load(f)

            summary = summarize_trajectory(traj)
            strategy = extract_strategy(task_id, summary)

            strategies.append({
                "task_id": task_id,
                "dir": traj_dir,
                "strategy": strategy,
            })
            print(f"  OK ({len(strategy)} chars)", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            strategies.append({
                "task_id": task_id,
                "dir": traj_dir,
                "strategy": None,
                "error": str(e),
            })

    with open(output_file, "w") as f:
        json.dump(strategies, f, indent=2)
    print(f"\nSaved {len(strategies)} strategies to {output_file}")


if __name__ == "__main__":
    main()
