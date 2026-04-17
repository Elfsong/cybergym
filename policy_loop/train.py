"""Main iterative offline GRPO training loop.

Usage:
    uv run python -m policy_loop.train
    uv run python -m policy_loop.train --num-rounds 3 --batch-size 10

Each round:
    1. Sample batch_size tasks
    2. Generate K strategies per task (Tinker, on-policy)
    3. Execute all K*N strategies via MiniMax (parallel subprocesses)
    4. Score each: milestone → reward
    5. Compute GRPO advantages (per task group), gradient step
    6. Save checkpoint + metrics
    7. (Phase 2) append to experience archive
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

from .archive import Archive
from .config import Config
from .executor import ExecutionResult, execute_strategies
from .planner import Planner, StrategyToExecute, Task
from .reward import MILESTONE_REWARDS, compute_reward, detect_milestone
from .utils import (
    get_task_description,
    parse_tasks_file,
    save_json,
    save_jsonl,
    setup_logging,
)

logger = logging.getLogger("policy_loop.train")


def build_tasks(task_ids: list[str], config: Config, archive: Archive | None) -> list[Task]:
    """Materialize Task objects with descriptions + optional retrieved history."""
    tasks: list[Task] = []
    for tid in task_ids:
        desc = get_task_description(tid, config.data_dir)
        prior: list[tuple[str, int]] = []
        if archive is not None and config.archive_enabled:
            prior = archive.retrieve(
                tid,
                n=config.archive_n,
                tournament_size=config.archive_tournament_size,
                min_milestone=config.archive_min_milestone,
            )
        tasks.append(Task(task_id=tid, description=desc, prior_strategies=prior))
    return tasks


def score_results(
    results: list[ExecutionResult],
    config: Config,
) -> list[tuple[StrategyToExecute, float, int]]:
    """Score every execution result: (strategy, reward, milestone)."""
    rewarded: list[tuple[StrategyToExecute, float, int]] = []
    for r in results:
        if r.trajectory_path is None:
            # No trajectory → milestone 0
            milestone = 0
            ms_reward = MILESTONE_REWARDS[0]
            rewarded.append((r.strategy, ms_reward, milestone))
            continue
        ms_result = detect_milestone(
            r.trajectory_path,
            r.agent_id,
            config.server,
            config.cybergym_api_key,
            traj_format="openhands",
            verify_fix=True,
        )
        reward = compute_reward(
            ms_result.milestone,
            adherence=1.0,
            novelty=0.0,
            lambda_adherence=config.lambda_adherence,
            alpha_novelty=config.alpha_novelty,
        )
        logger.debug(
            f"{r.strategy.task_id} [g{r.strategy.group_id}] milestone={ms_result.milestone} "
            f"reward={reward:.2f} — {ms_result.reasoning}"
        )
        rewarded.append((r.strategy, reward, ms_result.milestone))
    return rewarded


async def run_round(
    round_idx: int,
    planner: Planner,
    archive: Archive | None,
    config: Config,
    all_task_ids: list[str],
    rng: random.Random,
) -> dict:
    """One full GRPO round."""
    round_dir = config.output_dir / f"round_{round_idx:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    logger.info(f"=== ROUND {round_idx + 1}/{config.num_rounds} ===")

    # 1) Sample tasks
    batch_ids = rng.sample(all_task_ids, min(config.batch_size, len(all_task_ids)))
    tasks = build_tasks(batch_ids, config, archive)
    logger.info(f"Sampled {len(tasks)} tasks for round {round_idx}")

    # 2) Generate K strategies per task (on-policy)
    t_gen = time.monotonic()
    strategies = await planner.generate_strategies(tasks)
    gen_seconds = int(time.monotonic() - t_gen)
    save_json(
        [
            {
                "task_id": s.task_id,
                "group_id": s.group_id,
                "strategy": s.strategy,
                "n_tokens": len(s.tokens),
            }
            for s in strategies
        ],
        round_dir / "strategies.json",
    )
    logger.info(f"Generation: {len(strategies)} strategies in {gen_seconds}s")

    # 3) Execute via MiniMax (slow)
    t_exec = time.monotonic()
    results = execute_strategies(strategies, config, round_dir)
    exec_seconds = int(time.monotonic() - t_exec)
    logger.info(f"Execution: {len(results)} rollouts in {exec_seconds}s")

    # 4) Score
    rewarded = score_results(results, config)
    # Save detailed per-strategy outcomes
    save_jsonl(
        [
            {
                "task_id": s.task_id,
                "group_id": s.group_id,
                "reward": r,
                "milestone": m,
                "strategy": s.strategy,
            }
            for s, r, m in rewarded
        ],
        round_dir / "rewards.jsonl",
    )

    # 5) GRPO update
    metrics = await planner.grpo_update([(s, r) for s, r, _ in rewarded])

    # 6) Archive (Phase 2)
    if archive is not None and config.archive_enabled:
        archive.append_batch(
            [{"task_id": s.task_id, "strategy": s.strategy, "milestone": m}
             for s, _, m in rewarded]
        )

    # 7) Aggregate metrics + checkpoint
    milestones = [m for _, _, m in rewarded]
    pass_rate = sum(1 for m in milestones if m == 7) / max(len(milestones), 1)
    avg_milestone = sum(milestones) / max(len(milestones), 1)
    metrics.update({
        "round": round_idx,
        "n_strategies": len(rewarded),
        "n_tasks": len(tasks),
        "pass_rate": pass_rate,
        "avg_milestone": avg_milestone,
        "milestone_histogram": {i: milestones.count(i) for i in range(8)},
        "gen_seconds": gen_seconds,
        "exec_seconds": exec_seconds,
        "wall_seconds": int(time.monotonic() - t0),
    })
    logger.info(
        f"Round {round_idx} done: pass_rate={pass_rate:.3f} avg_milestone={avg_milestone:.2f} "
        f"degenerate={metrics['degenerate']}/{metrics['total_groups']}"
    )
    await planner.save_checkpoint(round_idx, metrics)
    save_json(metrics, round_dir / "metrics.json")
    return metrics


async def train(config: Config) -> None:
    """Main entry: run config.num_rounds rounds of iterative GRPO."""
    config.ensure_dirs()
    setup_logging(config.log_path)

    logger.info(f"=== Policy Loop Training ===")
    logger.info(f"Run ID: {config.run_id}  Output: {config.output_dir}")
    logger.info(
        f"Planner: {config.tinker_model} (LoRA rank {config.tinker_rank})"
    )
    logger.info(f"Executor: {config.executor_model} at {config.executor_base_url}")
    logger.info(
        f"GRPO: K={config.group_size}, batch={config.batch_size}, "
        f"rounds={config.num_rounds}, lr={config.learning_rate}"
    )

    # Save config snapshot
    save_json(asdict(config), config.output_dir / "config.json")

    all_task_ids = parse_tasks_file(config.tasks_file)
    if not all_task_ids:
        raise RuntimeError(f"No tasks in {config.tasks_file}")
    logger.info(f"Task pool: {len(all_task_ids)} tasks")

    archive = (
        Archive(config.archive_path, seed=42)
        if config.archive_enabled
        else None
    )

    planner = Planner(config)
    await planner.init()

    rng = random.Random(42)
    round_metrics: list[dict] = []
    for round_idx in range(config.num_rounds):
        metrics = await run_round(
            round_idx, planner, archive, config, all_task_ids, rng,
        )
        round_metrics.append(metrics)

    # Summary
    save_json(round_metrics, config.output_dir / "all_metrics.json")
    logger.info("=== Training complete ===")
    for m in round_metrics:
        logger.info(
            f"Round {m['round']}: pass_rate={m['pass_rate']:.3f} "
            f"avg_milestone={m['avg_milestone']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative offline GRPO training")
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--tasks-file", type=Path, default=None)
    parser.add_argument("--archive", action="store_true", help="Enable Phase 2 archive")
    parser.add_argument("--executor-parallel", type=int, default=None)
    args = parser.parse_args()

    config = Config()
    if args.num_rounds is not None:
        config.num_rounds = args.num_rounds
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.group_size is not None:
        config.group_size = args.group_size
    if args.tasks_file is not None:
        config.tasks_file = args.tasks_file
    if args.executor_parallel is not None:
        config.executor_parallel = args.executor_parallel
    if args.archive:
        config.archive_enabled = True

    asyncio.run(train(config))


if __name__ == "__main__":
    main()
