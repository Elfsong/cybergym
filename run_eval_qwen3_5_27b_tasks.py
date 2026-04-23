#!/usr/bin/env python3
"""CyberGym parallel evaluation runner for Qwen3.5-27B."""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import docker
import docker.errors


from pricing import compute_cost


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


def summarize_task(task_id: str, wall_time: int, log_dir: str, model: str) -> tuple:
    """Parse trajectory to extract status, cost, and token usage."""
    task_norm = task_id.replace(":", "_")
    wt_str = f"{wall_time // 60}m{wall_time % 60:02d}s"

    candidates = glob.glob(os.path.join(log_dir, task_norm + "-*", "trajectory"))
    if not candidates:
        print(f"  {task_norm:<25} time: {wt_str:>7}  steps:    ?  NO_TRAJECTORY", flush=True)
        return "OTHER", 0.0, 0, 0, 0, 0

    traj_path = max(candidates, key=os.path.getmtime)
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except Exception:
        print(f"  {task_norm:<25} time: {wt_str:>7}  steps:    ?  !  ERROR", flush=True)
        return "OTHER", 0.0, 0, 0, 0, 0

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

    # Patch accumulated_cost in every llm_metrics entry and write back
    dirty = False
    prompt_tokens = completion_tokens = cache_read_tokens = cache_write_tokens = 0
    for e in data:
        m = e.get("llm_metrics")
        if m and "accumulated_token_usage" in m:
            usage = m["accumulated_token_usage"]
            new_cost = compute_cost(
                model,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("cache_read_tokens", 0),
                usage.get("cache_write_tokens", 0),
            )
            if m.get("accumulated_cost", 0) != new_cost:
                m["accumulated_cost"] = round(new_cost, 6)
                dirty = True
            # Keep the last entry's tokens for the summary
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cache_read_tokens = usage.get("cache_read_tokens", 0)
            cache_write_tokens = usage.get("cache_write_tokens", 0)

    if dirty:
        with open(traj_path, "w") as f:
            json.dump(data, f)

    cost = compute_cost(model, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens)
    cost_str = f"${cost:.4f}"
    print(
        f"  {task_norm:<25} time: {wt_str:>7}  steps: {steps:>4}  cost: {cost_str:>8}"
        f"  prompt: {prompt_tokens:>8}  compl: {completion_tokens:>7}"
        f"  cache_r: {cache_read_tokens:>8}  cache_w: {cache_write_tokens:>8}  {marker} {poc_status}",
        flush=True,
    )
    return poc_status, cost, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens


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
    stagger: float,
) -> tuple[int, str, int]:
    if stagger > 0:
        time.sleep(task_num * stagger)
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
    task_norm = task_id.replace(":", "_")
    stderr = None if verbose else subprocess.DEVNULL
    try:
        subprocess.run(cmd, stderr=stderr, timeout=int(timeout) + 300)
    except Exception as e:
        print(f"  [{task_num}/{total}] {task_id}: process error: {e}", flush=True)
        _cleanup_orphaned_container(task_norm, log_dir)
    _recover_trajectory(task_norm, log_dir)
    elapsed = int(time.monotonic() - start)
    return task_num, task_id, elapsed


def _cleanup_orphaned_container(task_norm: str, log_dir: str):
    """Clean up Docker container left behind when the outer subprocess was killed."""
    pat = re.compile(
        r"runtime ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-[0-9a-f]{16})"
    )
    candidates = glob.glob(os.path.join(log_dir, task_norm + "-*", "logs", "*.log"))
    if not candidates:
        return
    log_path = max(candidates, key=os.path.getmtime)
    container_id = None
    try:
        with open(log_path) as f:
            for line in f:
                match = pat.search(line)
                if match:
                    container_id = match.group(1)
                    break
    except Exception:
        return
    if not container_id:
        return
    try:
        client = docker.from_env()
        container = client.containers.get(f"openhands-runtime-{container_id}")
        container.remove(force=True)
        print(f"  Cleaned up orphaned container: {container_id}", flush=True)
    except docker.errors.NotFound:
        pass
    except Exception as e:
        print(f"  Failed to cleanup container {container_id}: {e}", flush=True)


def _recover_trajectory(task_norm: str, log_dir: str):
    """Rebuild trajectory from event files when openhands was killed before writing it."""
    candidates = glob.glob(os.path.join(log_dir, task_norm + "-*"))
    if not candidates:
        return
    task_dir = max(candidates, key=os.path.getmtime)
    traj_path = os.path.join(task_dir, "trajectory")
    if os.path.exists(traj_path):
        return
    session_dirs = glob.glob(os.path.join(task_dir, "file", "sessions", "*", "events"))
    if not session_dirs:
        return
    best_dir = max(session_dirs, key=lambda d: len(os.listdir(d)))
    event_files = glob.glob(os.path.join(best_dir, "*.json"))
    if not event_files:
        return
    event_files.sort(key=lambda p: int(os.path.basename(p).replace(".json", "")))
    trajectory = []
    for ef in event_files:
        try:
            with open(ef) as f:
                trajectory.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    if trajectory:
        with open(traj_path, "w") as f:
            json.dump(trajectory, f)
        steps = len([e for e in trajectory if e.get("action") and e.get("source") == "agent"])
        print(f"  Recovered trajectory: {len(trajectory)} events, {steps} steps", flush=True)


def main():
    parser = argparse.ArgumentParser(description="CyberGym parallel evaluation runner (Qwen3.5-27B)")
    parser.add_argument("--model", default="openai/Qwen/Qwen3.5-27B")
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--data-dir", default="/data/cybergym_data/cybergym-benchmark-data/data")
    parser.add_argument("--out-dir", default="/data/cybergym_data/cybergym-eval-data/eval_qwen3_5_27b")
    parser.add_argument("--server-ip", default="172.17.0.1")
    parser.add_argument("--server-port", default="8666")
    parser.add_argument("--difficulty", default="level1")
    parser.add_argument("--timeout", default="2400")
    parser.add_argument("--max-iter", default="72")
    parser.add_argument("--max-output-tokens", default="4096")
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--tasks-file", default=None, help="Path to TASKS file (default: TASKS in script dir)")
    parser.add_argument("--stagger", type=float, default=1.0, help="Seconds between task launches to avoid Docker startup storm (0 to disable)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Append run UUID to out-dir so each run is isolated
    run_id = uuid4().hex[:8]
    args.out_dir = f"{args.out_dir}_{run_id}"

    # Environment
    os.environ.setdefault("LLM_API_KEY", "EMPTY")

    # Task list
    script_dir = Path(__file__).parent
    tasks_file = args.tasks_file or str(script_dir / "TASKS_TRAIN")
    if not Path(tasks_file).exists():
        print(f"Error: TASKS_TRAIN file not found: {tasks_file}", file=sys.stderr)
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

    # Tee stdout/stderr to a log file inside out-dir
    run_log_path = os.path.join(args.out_dir, "run.log")
    run_log = open(run_log_path, "w")
    sys.stdout = _Tee(sys.stdout, run_log)
    sys.stderr = _Tee(sys.stderr, run_log)

    print(f"Running {total} tasks (parallel: {args.parallel}, mode: {'verbose' if args.verbose else 'concise'})")
    print(f"Model: {args.model} via vLLM ({args.base_url})")
    print(f"Output: {args.out_dir}")
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
                stagger=args.stagger,
            ): tid
            for i, tid in enumerate(tasks)
        }
        for fut in as_completed(futures):
            task_num, task_id, elapsed = fut.result()
            result = summarize_task(task_id, elapsed, log_dir, args.model)
            results.append(result)

    # Tally results
    pass_count = sum(1 for r in results if r[0] == "PASSED")
    fail_count = sum(1 for r in results if r[0] == "FAILED")
    total_cost = sum(r[1] for r in results)
    total_prompt = sum(r[2] for r in results)
    total_compl = sum(r[3] for r in results)
    total_cache_read = sum(r[4] for r in results)
    total_cache_write = sum(r[5] for r in results)
    total_tokens = total_prompt + total_compl
    other_count = len(results) - pass_count - fail_count

    print("===========================================================")
    print(f"All {total} tasks completed. Passed: {pass_count}  Failed: {fail_count}  Other: {other_count}")
    print("-----------------------------------------------------------")
    print(f"Total cost:               ${total_cost:.4f}")
    print(f"Total prompt tokens:      {total_prompt:,}")
    print(f"Total completion tokens:  {total_compl:,}")
    print(f"Total cache read tokens:  {total_cache_read:,}")
    print(f"Total cache write tokens: {total_cache_write:,}")
    print(f"Total tokens:             {total_tokens:,}")
    print(f"Avg cost per task:        ${total_cost / max(len(results), 1):.4f}")
    print("-----------------------------------------------------------")
    print(f"Results in {log_dir}/")


if __name__ == "__main__":
    main()
