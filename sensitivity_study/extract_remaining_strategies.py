#!/usr/bin/env python3
"""Extract oracle strategies for tasks that don't have them yet."""

import json
import os
from pathlib import Path

import requests

VLLM_URL = "http://localhost:8001/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-27B"
QWEN_LOG_DIR = "/data/cybergym_data/cybergym-eval-data/eval_qwen3_5_27b_c414efb4/logs"
MINIMAX_LOG_DIR = "/data/cybergym_data/cybergym-eval-data/eval_minimax_m2_5_ef24bc78/logs"


def summarize_trajectory(traj):
    lines = []
    for entry in traj:
        source = entry.get("source", "")
        action = entry.get("action", "")
        args = entry.get("args", {})
        content = entry.get("content", "")
        observation = entry.get("observation", "")
        if source == "agent" and action == "run":
            cmd = args.get("command", "")
            if cmd: lines.append(f"[CMD] {cmd[:300]}")
        elif source == "agent" and action == "read":
            lines.append(f"[READ] {args.get('path', '')}")
        elif source == "agent" and action == "finish":
            lines.append("[FINISH]")
        elif observation == "run" and content:
            lines.append(f"[OUTPUT] {str(content)[:200]}")
    return "\n".join(lines[-60:])


def extract_strategy(task_id, traj_summary):
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

    resp = requests.post(VLLM_URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512, "temperature": 0.3,
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main():
    need_path = Path(__file__).parent / "need_extraction.json"
    with open(need_path) as f:
        tasks = json.load(f)

    existing_path = Path(__file__).parent / "oracle_strategies.json"
    with open(existing_path) as f:
        existing = json.load(f)

    new_strategies = []
    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        source = task["source"]
        log_dir = QWEN_LOG_DIR if source == "qwen" else MINIMAX_LOG_DIR
        traj_path = os.path.join(log_dir, task["dir"], "trajectory")

        print(f"[{i+1}/{len(tasks)}] {task_id} ({source})...", flush=True)
        try:
            with open(traj_path) as f:
                traj = json.load(f)
            summary = summarize_trajectory(traj)
            strategy = extract_strategy(task_id, summary)
            new_strategies.append({
                "task_id": task_id, "dir": task["dir"],
                "source": source, "strategy": strategy,
            })
            print(f"  OK ({len(strategy)} chars)", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            new_strategies.append({
                "task_id": task_id, "dir": task["dir"],
                "source": source, "strategy": None, "error": str(e),
            })

    all_strategies = existing + new_strategies
    with open(existing_path, "w") as f:
        json.dump(all_strategies, f, indent=2)
    print(f"\nTotal: {len(existing)} existing + {len(new_strategies)} new = {len(all_strategies)}")


if __name__ == "__main__":
    main()
