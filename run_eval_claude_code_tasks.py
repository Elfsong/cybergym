#!/usr/bin/env python3
"""CyberGym parallel evaluation runner for Claude Code.

Each task is assigned to an independent Claude Code agent that runs to
completion and saves its trajectory immediately.

Usage:
    uv run python3 run_eval_claude_code_tasks.py
    uv run python3 run_eval_claude_code_tasks.py --model sonnet --parallel 8
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from cybergym.task.gen_task import generate_task
from cybergym.task.types import TaskConfig, TaskDifficulty

SCRIPT_DIR = Path(__file__).parent
PROMPT_PATH = SCRIPT_DIR / "examples" / "agents" / "claude_code" / "prompt.txt"


class _Tee:
    """Write to two streams simultaneously."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()


def parse_tasks_file(path: str) -> list[str]:
    """Read task IDs from a file (one per line, # comments and blanks skipped)."""
    tasks = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                tasks.append(stripped)
    return tasks


def parse_trajectory(traj_path: str) -> dict:
    """Parse a Claude Code stream-json trajectory for results."""
    events = []
    try:
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        return {"status": "NO_TRAJECTORY", "steps": 0, "cost": 0, "submits": 0}

    # Count tool-use turns (nested inside assistant events)
    steps = 0
    for e in events:
        if e.get("type") == "assistant":
            content = e.get("message", {}).get("content", [])
            if isinstance(content, list):
                steps += sum(1 for c in content if c.get("type") == "tool_use")

    # Find submit.sh calls and their results
    # CC format: tool_use in assistant.message.content[], tool_result in user.message.content[]
    poc_status = "NO_SUBMIT"
    submit_count = 0

    for i, event in enumerate(events):
        # Check assistant events for Bash tool_use with submit.sh
        if event.get("type") != "assistant":
            continue
        content = event.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue

        has_submit = False
        for c in content:
            if c.get("type") == "tool_use" and c.get("name") == "Bash":
                cmd = c.get("input", {}).get("command", "")
                if "submit.sh" in cmd and "cat" not in cmd:
                    has_submit = True
                    break

        if not has_submit:
            continue

        submit_count += 1
        # Search subsequent user events for tool_result containing server response
        for j in range(i + 1, min(i + 10, len(events))):
            if events[j].get("type") != "user":
                continue
            user_content = events[j].get("message", {}).get("content", [])
            if not isinstance(user_content, list):
                continue
            for uc in user_content:
                if uc.get("type") != "tool_result":
                    continue
                result_text = str(uc.get("content", ""))
                js = result_text.find('{"task_id"')
                if js < 0:
                    js = result_text.find('"exit_code"')
                    if js >= 0:
                        js = result_text.rfind("{", 0, js)
                if js >= 0:
                    je = result_text.find("}", js)
                    if je >= 0:
                        try:
                            result_json = json.loads(result_text[js:je + 1])
                            ec = result_json.get("exit_code")
                            if ec is not None and ec != 0:
                                poc_status = "PASSED"
                            elif ec == 0 and poc_status != "PASSED":
                                poc_status = "FAILED"
                        except (json.JSONDecodeError, ValueError):
                            pass
            if poc_status == "PASSED":
                break

    # Extract cost from final result event
    cost = 0.0
    input_tokens = output_tokens = 0
    for event in reversed(events):
        if event.get("type") == "result":
            cost = event.get("total_cost_usd", 0.0)
            usage = event.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            break

    return {
        "status": poc_status,
        "steps": steps,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "submits": submit_count,
    }


def summarize_task(task_id: str, wall_time: int, log_dir: str) -> tuple:
    """Parse trajectory and print summary. Returns (status, cost, input_tokens, output_tokens)."""
    task_norm = task_id.replace(":", "_")
    wt_str = f"{wall_time // 60}m{wall_time % 60:02d}s"

    candidates = glob.glob(os.path.join(log_dir, task_norm + "-*", "trajectory.jsonl"))
    if not candidates:
        print(f"  {task_norm:<25} time: {wt_str:>7}  steps:    ?  NO_TRAJECTORY", flush=True)
        return "OTHER", 0.0, 0, 0

    traj_path = max(candidates, key=os.path.getmtime)
    result = parse_trajectory(traj_path)

    markers = {"PASSED": "\u2713", "FAILED": "\u2717", "NO_SUBMIT": "\u2014", "NO_TRAJECTORY": "?"}
    marker = markers.get(result["status"], "?")

    cost_str = f"${result['cost']:.4f}" if result["cost"] > 0 else "\u2014"
    print(
        f"  {task_norm:<25} time: {wt_str:>7}  steps: {result['steps']:>4}  cost: {cost_str:>8}"
        f"  input: {result['input_tokens']:>8}  output: {result['output_tokens']:>7}"
        f"  submits: {result['submits']:>2}  {marker} {result['status']}",
        flush=True,
    )
    return result["status"], result["cost"], result["input_tokens"], result["output_tokens"]


def run_task(
    task_num: int,
    task_id: str,
    total: int,
    *,
    model: str,
    log_dir: str,
    tmp_dir: str,
    data_dir: str,
    server: str,
    difficulty: str,
    timeout: int,
    max_budget_usd: float,
    effort: str,
    stagger: float,
    prompt_file: str,
) -> tuple[int, str, int]:
    """Run a single task with Claude Code."""
    if stagger > 0:
        time.sleep(task_num * stagger)

    print(f"[{task_num}/{total}] [{datetime.now():%Y-%m-%d %H:%M:%S}] Starting: {task_id}", flush=True)
    start = time.monotonic()

    agent_id = uuid4().hex
    task_norm = task_id.replace(":", "_")
    sub_dir = f"{task_norm}-{agent_id}"

    workspace_dir = Path(tmp_dir) / sub_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    task_log_dir = Path(log_dir) / sub_dir
    task_log_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Generate task workspace
        task = generate_task(TaskConfig(
            task_id=task_id,
            out_dir=workspace_dir,
            data_dir=Path(data_dir),
            server=server,
            difficulty=TaskDifficulty(difficulty),
            agent_id=agent_id,
        ))

        # 2. Extract repo-vul.tar.gz → src-vul/
        tarball = workspace_dir / "repo-vul.tar.gz"
        if tarball.exists():
            src_vul = workspace_dir / "src-vul"
            src_vul.mkdir(exist_ok=True)
            subprocess.run(
                ["tar", "xzf", str(tarball), "-C", str(src_vul)],
                check=True, capture_output=True,
            )

        # 3. Save args.json
        with open(task_log_dir / "args.json", "w") as f:
            json.dump({
                "agent": f"claude-code:{model}",
                "task": {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "checksum": task.checksum,
                    "server": server,
                    "difficulty": difficulty,
                },
                "agent_args": {
                    "model": model,
                    "timeout": timeout,
                    "max_budget_usd": max_budget_usd,
                    "effort": effort,
                },
            }, f, indent=2)

        # 4. Read prompt
        prompt_text = Path(prompt_file).read_text()

        # 5. Run Claude Code
        claude_bin = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
        trajectory_path = task_log_dir / "trajectory.jsonl"

        cmd = [
            "timeout", str(timeout),
            claude_bin,
            "-p", prompt_text,
            "--output-format", "stream-json",
            "--verbose",
            "--model", model,
            "--allowedTools", "Bash,Read,Write,Edit",
            "--permission-mode", "bypassPermissions",
            "--max-budget-usd", str(max_budget_usd),
            "--no-session-persistence",
            "--effort", effort,
        ]

        with open(trajectory_path, "w") as traj_file:
            subprocess.run(
                cmd,
                cwd=str(workspace_dir),
                stdout=traj_file,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=timeout + 60,
            )

    except Exception as e:
        print(f"  [{task_num}/{total}] {task_id}: error: {e}", flush=True)

    # Cleanup workspace (keep logs)
    workspace_parent = Path(tmp_dir) / sub_dir
    if workspace_parent.exists():
        shutil.rmtree(workspace_parent, ignore_errors=True)

    elapsed = int(time.monotonic() - start)
    return task_num, task_id, elapsed


def main():
    parser = argparse.ArgumentParser(description="CyberGym parallel evaluation runner (Claude Code)")
    parser.add_argument("--model", default="opus")
    parser.add_argument("--data-dir", default="/data/cybergym_data/cybergym-benchmark-data/data")
    parser.add_argument("--out-dir", default="/data/cybergym_data/cybergym-eval-data/eval_claude_code")
    parser.add_argument("--server-ip", default="172.17.0.1")
    parser.add_argument("--server-port", default="8666")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-budget-usd", type=float, default=5.0)
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "max"])
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--tasks-file", default=None)
    parser.add_argument("--stagger", type=float, default=0.5)
    args = parser.parse_args()

    # Append run UUID to out-dir
    run_id = uuid4().hex[:8]
    args.out_dir = f"{args.out_dir}_{run_id}"

    # Task list
    tasks_file = args.tasks_file or str(SCRIPT_DIR / "TASKS")
    if not Path(tasks_file).exists():
        print(f"Error: TASKS file not found: {tasks_file}", file=sys.stderr)
        sys.exit(1)
    tasks = parse_tasks_file(tasks_file)
    if not tasks:
        print(f"Error: No tasks found in {tasks_file}", file=sys.stderr)
        sys.exit(1)

    total = len(tasks)
    log_dir = f"{args.out_dir}/logs"
    tmp_dir = f"{args.out_dir}/tmp"
    os.makedirs(args.out_dir, exist_ok=True)
    (SCRIPT_DIR / "run_id.txt").write_text(run_id + "\n")

    prompt_file = str(PROMPT_PATH)
    server = f"http://{args.server_ip}:{args.server_port}"

    # Tee stdout/stderr to run.log
    run_log_path = os.path.join(args.out_dir, "run.log")
    run_log = open(run_log_path, "w")
    sys.stdout = _Tee(sys.stdout, run_log)
    sys.stderr = _Tee(sys.stderr, run_log)

    print(f"Running {total} tasks (parallel: {args.parallel})")
    print(f"Model: claude-code:{args.model} (effort: {args.effort})")
    print(f"Budget: ${args.max_budget_usd}/task, timeout: {args.timeout}s")
    print(f"Output: {args.out_dir}")
    print("=" * 60)

    # Run all tasks
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                run_task, i + 1, tid, total,
                model=args.model,
                log_dir=log_dir,
                tmp_dir=tmp_dir,
                data_dir=args.data_dir,
                server=server,
                difficulty=args.difficulty,
                timeout=args.timeout,
                max_budget_usd=args.max_budget_usd,
                effort=args.effort,
                stagger=args.stagger,
                prompt_file=prompt_file,
            ): tid
            for i, tid in enumerate(tasks)
        }
        for fut in as_completed(futures):
            task_num, task_id, elapsed = fut.result()
            result = summarize_task(task_id, elapsed, log_dir)
            results.append(result)

    # Tally results
    pass_count = sum(1 for r in results if r[0] == "PASSED")
    fail_count = sum(1 for r in results if r[0] == "FAILED")
    other_count = len(results) - pass_count - fail_count
    total_cost = sum(r[1] for r in results)
    total_input = sum(r[2] for r in results)
    total_output = sum(r[3] for r in results)

    print("=" * 60)
    print(f"All {total} tasks completed. Passed: {pass_count}  Failed: {fail_count}  Other: {other_count}")
    print("-" * 60)
    print(f"Total cost:              ${total_cost:.4f}")
    print(f"Total input tokens:      {total_input:,}")
    print(f"Total output tokens:     {total_output:,}")
    print(f"Avg cost per task:       ${total_cost / max(len(results), 1):.4f}")
    print("-" * 60)
    print(f"Results in {log_dir}/")


if __name__ == "__main__":
    main()
