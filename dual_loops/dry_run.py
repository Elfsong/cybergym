"""Dry-run test: skip Tinker, use hand-written strategies, run executor on 2 tasks.

This validates the end-to-end pipeline (executor + reward + archive) without
needing the Tinker SDK installed. Useful for local debugging.

Usage:
    uv run python -m dual_loops.dry_run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from .archive import Archive
from .config import Config
from .executor import execute_strategies
from .planner import StrategyToExecute
from .reward import compute_reward, detect_milestone
from .utils import get_task_description, setup_logging

logger = logging.getLogger("dual_loops.dry_run")


MOCK_STRATEGIES = [
    "Examine the README.md to identify the vulnerable function. Extract the "
    "source tarball and locate the target file. Look at the fuzzer harness to "
    "understand the input format. Construct a minimal PoC file using python "
    "struct.pack to craft the binary header. Submit the PoC and check the exit "
    "code. If it fails, examine the sanitizer output and adjust the byte layout.",
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", nargs="+", default=["arvo:8933"])
    parser.add_argument("--strategies", type=int, default=1, help="strategies per task")
    parser.add_argument(
        "--model", default=None,
        help="Override executor model (e.g., openai/Qwen/Qwen3.5-27B)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Override executor base_url (e.g., http://localhost:8001/v1)",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-iter", type=int, default=30)
    args = parser.parse_args()

    config = Config()
    config.executor_parallel = len(args.task_ids) * args.strategies
    config.executor_timeout = args.timeout
    config.executor_max_iter = args.max_iter
    if args.model:
        config.executor_model = args.model
    if args.base_url:
        config.executor_base_url = args.base_url

    config.ensure_dirs()
    setup_logging(config.log_path)

    logger.info(f"Dry-run: {len(args.task_ids)} tasks × {args.strategies} strategies")
    logger.info(f"Output: {config.output_dir}")

    # Build mock StrategyToExecute objects
    strategies: list[StrategyToExecute] = []
    for tid in args.task_ids:
        desc = get_task_description(tid, config.data_dir)
        logger.info(f"  {tid}: {desc[:100]}")
        for g in range(args.strategies):
            mock_text = MOCK_STRATEGIES[g % len(MOCK_STRATEGIES)]
            strategies.append(StrategyToExecute(
                task_id=tid,
                strategy=mock_text,
                tokens=[],            # not needed for dry-run (no GRPO)
                logprobs=[],
                prompt=None,
                prompt_length=0,
                group_id=g,
            ))

    # Execute
    round_dir = config.output_dir / "round_000"
    t0 = time.monotonic()
    results = execute_strategies(strategies, config, round_dir)
    elapsed = int(time.monotonic() - t0)
    logger.info(f"Execution: {len(results)} rollouts in {elapsed}s")

    # Score
    for r in results:
        if r.trajectory_path is None:
            logger.warning(f"No trajectory for {r.strategy.task_id}")
            continue
        ms = detect_milestone(
            r.trajectory_path,
            r.agent_id,
            config.server,
            config.cybergym_api_key,
            traj_format="openhands",
            verify_fix=bool(config.cybergym_api_key),
        )
        reward = compute_reward(ms.milestone)
        logger.info(
            f"  {r.strategy.task_id} [g{r.strategy.group_id}]  "
            f"milestone={ms.milestone}  reward={reward:.1f}  "
            f"({ms.reasoning})"
        )

    logger.info("Dry-run complete")


if __name__ == "__main__":
    asyncio.run(main())
