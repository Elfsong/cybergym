#!/usr/bin/env python3
"""CyberGym Level 3 baseline via DashScope API (qwen3.5-27b).

Reuses the OpenHands-subprocess pattern from run_eval_qwen3_5_27b_tasks.py
but points LiteLLM at DashScope (https://dashscope.aliyuncs.com/...) instead
of the local vLLM. Picks up DASHSCOPE_API_KEY from .env.

Example:
    uv run python run_eval_qwen_api_level3.py --n-tasks 32 --parallel 32
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

# Load .env before anything imports pricing.py (keeps behavior consistent
# with the CLI tools in this repo that expect DASHSCOPE_API_KEY / HF_TOKEN /...).
def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

# Expose the DashScope key under the name OpenHands/LiteLLM expect.
if "DASHSCOPE_API_KEY" not in os.environ:
    print("FATAL: DASHSCOPE_API_KEY missing from .env", file=sys.stderr)
    sys.exit(2)
os.environ["LLM_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]

# Reuse helpers from the existing runner (same OpenHands subprocess plumbing,
# trajectory recovery, summarize, orphan-container cleanup).
from run_eval_qwen3_5_27b_tasks import (  # noqa: E402
    _Tee,
    parse_tasks_file,
    run_task,
    summarize_task,
)


DEFAULTS = {
    "model":             "openai/qwen3.5-27b",
    "base_url":          "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "difficulty":        "level3",
    "tasks_file":        str(Path(__file__).parent / "TASKS_EVAL"),
    "out_dir_prefix":    "/data/cybergym_data/cybergym-eval-data/eval_qwen_api_level3",
    "timeout":           "2400",
    "max_iter":          "72",
    "max_output_tokens": "8192",
    "parallel":          32,
    "stagger":           1.0,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-tasks", type=int, default=32,
                   help="Number of tasks to sample from --tasks-file (None = all)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for task sampling")
    p.add_argument("--parallel", type=int, default=DEFAULTS["parallel"])
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--base-url", default=DEFAULTS["base_url"])
    p.add_argument("--difficulty", default=DEFAULTS["difficulty"])
    p.add_argument("--tasks-file", default=DEFAULTS["tasks_file"])
    p.add_argument("--out-dir", default=None,
                   help="Base output dir (run_id suffix is always appended)")
    p.add_argument("--data-dir", default="/data/cybergym_data/cybergym-benchmark-data/data")
    p.add_argument("--server-ip", default="172.17.0.1")
    p.add_argument("--server-port", default="8666")
    p.add_argument("--timeout", default=DEFAULTS["timeout"])
    p.add_argument("--max-iter", default=DEFAULTS["max_iter"])
    p.add_argument("--max-output-tokens", default=DEFAULTS["max_output_tokens"])
    p.add_argument("--stagger", type=float, default=DEFAULTS["stagger"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    out_base = args.out_dir or DEFAULTS["out_dir_prefix"]
    run_id = uuid4().hex[:8]
    out_dir = f"{out_base}_{run_id}"

    if not Path(args.tasks_file).exists():
        print(f"Error: tasks file not found: {args.tasks_file}", file=sys.stderr)
        sys.exit(1)
    all_tasks = parse_tasks_file(args.tasks_file)
    if args.n_tasks and args.n_tasks < len(all_tasks):
        rng = random.Random(args.seed)
        tasks = rng.sample(all_tasks, args.n_tasks)
    else:
        tasks = list(all_tasks)

    total = len(tasks)
    log_dir = f"{out_dir}/logs"
    tmp_dir = f"{out_dir}/tmp"
    os.makedirs(out_dir, exist_ok=True)
    silent = "false" if args.verbose else "true"
    server = f"http://{args.server_ip}:{args.server_port}"

    run_log = open(os.path.join(out_dir, "run.log"), "w")
    sys.stdout = _Tee(sys.stdout, run_log)
    sys.stderr = _Tee(sys.stderr, run_log)

    print(f"DashScope Level {args.difficulty[-1]} baseline ({args.model})")
    print(f"Tasks: {total} (sampled with seed={args.seed} from {args.tasks_file})")
    print(f"Parallel: {args.parallel}  timeout={args.timeout}s  max_iter={args.max_iter}")
    print(f"Output: {out_dir}")
    # Save the exact task list so the smoke can be replayed deterministically.
    with open(os.path.join(out_dir, "tasks.list"), "w") as f:
        f.write("\n".join(tasks) + "\n")
    print("===========================================================")

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
            _, task_id, elapsed = fut.result()
            results.append(summarize_task(task_id, elapsed, log_dir, args.model))

    passed = sum(1 for r in results if r[0] == "PASSED")
    failed = sum(1 for r in results if r[0] == "FAILED")
    other  = len(results) - passed - failed
    total_cost    = sum(r[1] for r in results)
    total_prompt  = sum(r[2] for r in results)
    total_compl   = sum(r[3] for r in results)

    print("===========================================================")
    print(f"All {total} tasks done. Passed: {passed}  Failed: {failed}  Other: {other}")
    print(f"Total cost: ${total_cost:.4f}  avg ${total_cost / max(len(results),1):.4f}/task")
    print(f"Total prompt: {total_prompt:,}  completion: {total_compl:,}")
    print(f"Results in {log_dir}/")


if __name__ == "__main__":
    main()
