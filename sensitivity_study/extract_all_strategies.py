#!/usr/bin/env python3
"""Extract oracle strategies for all 100 sensitivity tasks (fresh, with improved summarizer)."""

import json
import os
from pathlib import Path

import requests

VLLM_URL = "http://localhost:8001/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-27B"


def summarize_trajectory(traj: list[dict]) -> str:
    """Summarize trajectory: head 20 + sampled middle ~20 + last 100."""
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


def summarize_cc_trajectory(traj_path: str) -> str:
    """Summarize a Claude Code stream-json trajectory."""
    events = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except:
                    continue

    lines = []
    for e in events:
        if e.get("type") == "assistant":
            content = e.get("message", {}).get("content", [])
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "tool_use":
                        name = c.get("name", "")
                        inp = c.get("input", {})
                        if name == "Bash":
                            lines.append(f"[CMD] {inp.get('command', '')[:300]}")
                        elif name == "Read":
                            lines.append(f"[READ] {inp.get('file_path', '')}")
                        elif name in ("Write", "Edit"):
                            lines.append(f"[{name.upper()}] {inp.get('file_path', '')[:200]}")
        elif e.get("type") == "user":
            content = e.get("message", {}).get("content", [])
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "tool_result":
                        text = str(c.get("content", ""))[:200]
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
    prompt = f"""You are analyzing a successful vulnerability reproduction trajectory. Summarize it into a high-level strategy plan (200-500 words) that another agent could follow to reproduce the same vulnerability.

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
        "max_tokens": 8192, "temperature": 0.3,
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main():
    tasks_file = Path(__file__).parent / "tasks.json"
    with open(tasks_file) as f:
        tasks = json.load(f)

    output_file = Path(__file__).parent / "oracle_strategies.json"
    strategies = []

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        source = task["source"]
        log_dir = task["log_dir"]
        traj_dir = task["dir"]

        print(f"[{i+1}/{len(tasks)}] {task_id} ({source})...", flush=True)

        try:
            if source == "cc":
                traj_path = os.path.join(log_dir, traj_dir, "trajectory.jsonl")
                summary = summarize_cc_trajectory(traj_path)
            else:
                traj_path = os.path.join(log_dir, traj_dir, "trajectory")
                with open(traj_path) as f:
                    traj = json.load(f)
                summary = summarize_trajectory(traj)

            strategy = extract_strategy(task_id, summary)
            strategies.append({
                "task_id": task_id,
                "dir": traj_dir,
                "source": source,
                "group": task["group"],
                "strategy": strategy,
            })
            print(f"  OK ({len(strategy)} chars)", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            strategies.append({
                "task_id": task_id, "dir": traj_dir,
                "source": source, "group": task["group"],
                "strategy": None, "error": str(e),
            })

    with open(output_file, "w") as f:
        json.dump(strategies, f, indent=2)

    ok = sum(1 for s in strategies if s.get("strategy"))
    print(f"\nDone: {ok}/{len(strategies)} strategies extracted")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
