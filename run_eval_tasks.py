#!/usr/bin/env python3
"""CyberGym parallel evaluation runner."""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def parse_tasks_file(path: str) -> list[str]:
    """Read task IDs from a file (one per line, # comments and blanks skipped)."""
    tasks = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                tasks.append(stripped)
    return tasks


def summarize_task(task_id: str, wall_time: int, log_dir: str) -> tuple:
    """Parse trajectory to extract status, cost, and token usage."""
    task_norm = task_id.replace(":", "_")
    wt_str = f"{wall_time // 60}m{wall_time % 60:02d}s"

    candidates = glob.glob(os.path.join(log_dir, task_norm + "-*", "trajectory"))
    if not candidates:
        print(f"  {task_norm:<25} time: {wt_str:>7}  steps:    ?  NO_TRAJECTORY", flush=True)
        return "OTHER", 0.0, 0, 0, 0

    traj_path = max(candidates, key=os.path.getmtime)
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except Exception:
        print(f"  {task_norm:<25} time: {wt_str:>7}  steps:    ?  !  ERROR", flush=True)
        return "OTHER", 0.0, 0, 0, 0

    steps = len([e for e in data if e.get("action") and e.get("source") == "agent"])

    poc_status = "NO_SUBMIT"
    for i, item in enumerate(data):
        cmd = str(item.get("args", {}).get("command", ""))
        if "submit.sh" in cmd and "cat" not in cmd:
            if i + 1 < len(data):
                content = str(data[i + 1].get("content", ""))
                try:
                    json_start = content.find("{")
                    if json_start < 0:
                        continue
                    json_end = content.find("}", json_start)
                    if json_end < 0:
                        continue
                    result = json.loads(content[json_start : json_end + 1])
                    ec = result.get("exit_code", None)
                    if ec is None:
                        continue
                    if ec != 0:
                        poc_status = "PASSED"
                        break
                    else:
                        poc_status = "FAILED"
                except Exception:
                    pass

    markers = {"PASSED": "\u2713", "FAILED": "\u2717", "NO_SUBMIT": "\u2014"}
    marker = markers.get(poc_status, "?")

    cost = 0.0
    prompt_tokens = completion_tokens = cache_read_tokens = 0
    for e in reversed(data):
        m = e.get("llm_metrics")
        if m and "accumulated_cost" in m:
            cost = m["accumulated_cost"]
            usage = m.get("accumulated_token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cache_read_tokens = usage.get("cache_read_tokens", 0)
            break

    cost_str = f"${cost:.4f}"
    print(
        f"  {task_norm:<25} time: {wt_str:>7}  steps: {steps:>4}  cost: {cost_str:>8}"
        f"  prompt: {prompt_tokens:>8}  compl: {completion_tokens:>7}"
        f"  cache: {cache_read_tokens:>8}  {marker} {poc_status}",
        flush=True,
    )
    return poc_status, cost, prompt_tokens, completion_tokens, cache_read_tokens


def run_task(
    task_num: int,
    task_id: str,
    total: int,
    *,
    model: str,
    base_url: str,
    log_dir: str,
    tmp_dir: str,
    data_dir: str,
    server: str,
    timeout: str,
    max_iter: str,
    max_output_tokens: str,
    silent: str,
    difficulty: str,
    verbose: bool,
) -> tuple[int, str, int]:
    print(f"[{task_num}/{total}] [{datetime.now():%Y-%m-%d %H:%M:%S}] Starting: {task_id}", flush=True)
    start = time.monotonic()
    cmd = [
        os.path.expanduser("~/.local/bin/uv"), "run", "python3",
        "examples/agents/openhands/run.py",
        "--model", model,
        "--base_url", base_url,
        "--log_dir", log_dir,
        "--tmp_dir", tmp_dir,
        "--data_dir", data_dir,
        "--task_id", task_id,
        "--server", server,
        "--timeout", timeout,
        "--max_iter", max_iter,
        "--max_output_tokens", max_output_tokens,
        "--silent", silent,
        "--difficulty", difficulty,
    ]
    stderr = None if verbose else subprocess.DEVNULL
    try:
        subprocess.run(cmd, stderr=stderr, timeout=int(timeout) + 300)
    except Exception as e:
        print(f"  [{task_num}/{total}] {task_id}: process error: {e}", flush=True)
    elapsed = int(time.monotonic() - start)
    return task_num, task_id, elapsed


def main():
    parser = argparse.ArgumentParser(description="CyberGym parallel evaluation runner")
    parser.add_argument("--model", default="openai/MiniMaxAI/MiniMax-M2.5")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data-dir", default="/data/cybergym_data/cybergym-benchmark-data/data")
    parser.add_argument("--out-dir", default="/data/cybergym_data/cybergym-eval-data/eval_minimax_m2_5")
    parser.add_argument("--server-ip", default="172.17.0.1")
    parser.add_argument("--server-port", default="8666")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--timeout", default="1800")
    parser.add_argument("--max-iter", default="72")
    parser.add_argument("--max-output-tokens", default="8192")
    parser.add_argument("--parallel", type=int, default=36)
    parser.add_argument("--tasks-file", default=None, help="Path to TASKS file (default: TASKS in script dir)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Environment
    os.environ.setdefault("LLM_API_KEY", "EMPTY")

    # Task list
    script_dir = Path(__file__).parent
    tasks_file = args.tasks_file or str(script_dir / "TASKS")
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
    silent = "false" if args.verbose else "true"
    server = f"http://{args.server_ip}:{args.server_port}"

    print(f"Running {total} tasks (parallel: {args.parallel}, mode: {'verbose' if args.verbose else 'concise'})")
    print(f"Model: {args.model} via vLLM ({args.base_url})")
    print("===========================================================")

    # Run all tasks with bounded parallelism
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                run_task, i + 1, tid, total,
                model=args.model,
                base_url=args.base_url,
                log_dir=log_dir,
                tmp_dir=tmp_dir,
                data_dir=args.data_dir,
                server=server,
                timeout=args.timeout,
                max_iter=args.max_iter,
                max_output_tokens=args.max_output_tokens,
                silent=silent,
                difficulty=args.difficulty,
                verbose=args.verbose,
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
    total_cost = sum(r[1] for r in results)
    total_prompt = sum(r[2] for r in results)
    total_compl = sum(r[3] for r in results)
    total_cache = sum(r[4] for r in results)
    total_tokens = total_prompt + total_compl
    other_count = len(results) - pass_count - fail_count

    print("===========================================================")
    print(f"All {total} tasks completed. Passed: {pass_count}  Failed: {fail_count}  Other: {other_count}")
    print("-----------------------------------------------------------")
    print(f"Total cost:              ${total_cost:.4f}")
    print(f"Total prompt tokens:     {total_prompt}")
    print(f"Total completion tokens: {total_compl}")
    print(f"Total cache read tokens: {total_cache}")
    print(f"Total tokens:            {total_tokens}")
    print("-----------------------------------------------------------")
    print(f"Results in {log_dir}/")


if __name__ == "__main__":
    main()
