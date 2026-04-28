"""Per-round execution for the iterative GRPO training loop."""

from __future__ import annotations

import logging
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime

from .archive import Archive
from .config import Config
from .executor import ExecutionResult, execute_strategies
from .planner import Planner, StrategyToExecute, Task
from .reward import compute_reward, detect_milestone, score_reflection_batch
from .utils import get_task_description, save_json, save_jsonl

logger = logging.getLogger("dual_loops.train")


def build_tasks(task_ids: list[str], config: Config) -> list[Task]:
    """Materialize Task objects with descriptions only."""
    return [
        Task(task_id=task_id, description=get_task_description(task_id, config.data_dir))
        for task_id in task_ids
    ]


def score_milestones(
    results: list[ExecutionResult],
    config: Config,
) -> list[tuple[StrategyToExecute, int]]:
    """Score every execution result as (strategy, milestone)."""
    scored: list[tuple[StrategyToExecute, int]] = []
    for result in results:
        if result.trajectory_path is None:
            scored.append((result.strategy, 0))
            continue
        milestone_result = detect_milestone(
            result.trajectory_path,
            result.agent_id,
            config.server,
            config.cybergym_api_key,
            traj_format="openhands",
            verify_fix=True,
        )
        logger.debug(
            f"{result.strategy.task_id} [g{result.strategy.group_id}] "
            f"milestone={milestone_result.milestone} — {milestone_result.reasoning}"
        )
        scored.append((result.strategy, milestone_result.milestone))
    return scored


def _rollout_quality_metrics(
    rewarded: list[tuple[StrategyToExecute, int]],
    results: list[ExecutionResult],
) -> dict:
    milestones = [milestone for _, milestone in rewarded]
    n = max(len(milestones), 1)
    task_ids = sorted({strategy.task_id for strategy, _ in rewarded})
    task_success = {
        task_id: any(
            strategy.task_id == task_id and milestone == 7
            for strategy, milestone in rewarded
        )
        for task_id in task_ids
    }
    rollout_pass_rate = sum(1 for milestone in milestones if milestone == 7) / n
    return {
        "n_strategies": len(rewarded),
        "n_tasks": len(task_ids),
        "pass_rate": rollout_pass_rate,
        "rollout_pass_rate": rollout_pass_rate,
        "task_pass_at_n": (
            sum(1 for passed in task_success.values() if passed)
            / max(len(task_success), 1)
        ),
        "task_success": task_success,
        "avg_milestone": sum(milestones) / n,
        "milestone_histogram": {i: milestones.count(i) for i in range(8)},
        "n_cancelled": sum(1 for result in results if result.cancelled),
        "n_with_trajectory": sum(1 for result in results if result.has_trajectory),
        "mean_wall_seconds": (
            sum(result.wall_seconds for result in results) / max(len(results), 1)
        ),
    }


async def run_validation_round(
    label: str,
    planner: Planner,
    archive: Archive | None,
    config: Config,
    task_ids: list[str],
) -> dict:
    """Evaluate the current planner on a fixed task set without training."""
    eval_dir = config.output_dir / "validation" / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    tasks = build_tasks(task_ids, config)

    old_group_size = config.group_size
    old_archive_enabled = config.archive_enabled
    old_archive = planner.archive
    samples_per_task = (
        config.validation_samples_per_task
        or config.validation_group_size
        or config.group_size
    )
    try:
        config.group_size = samples_per_task
        if not config.validation_use_archive:
            config.archive_enabled = False
            planner.bind_archive(None)

        strategies_pkl = eval_dir / "strategies.pkl"
        if strategies_pkl.exists():
            with open(strategies_pkl, "rb") as f:
                strategies = pickle.load(f)
            gen_seconds = 0
            logger.info(
                f"Validation {label}: loaded {len(strategies)} strategies "
                f"from {strategies_pkl.name}"
            )
        else:
            t_gen = time.monotonic()
            strategies = await planner.generate_strategies(tasks)
            gen_seconds = int(time.monotonic() - t_gen)
            with open(strategies_pkl, "wb") as f:
                pickle.dump(strategies, f)
            save_json(
                [
                    {
                        "task_id": strategy.task_id,
                        "group_id": strategy.group_id,
                        "strategy": strategy.strategy,
                        "thinking": strategy.thinking,
                        "n_tokens": len(strategy.tokens),
                    }
                    for strategy in strategies
                ],
                eval_dir / "strategies.json",
            )
    finally:
        config.group_size = old_group_size
        config.archive_enabled = old_archive_enabled
        planner.bind_archive(old_archive)

    t_exec = time.monotonic()
    results = execute_strategies(strategies, config, eval_dir)
    exec_seconds = int(time.monotonic() - t_exec)
    scored = score_milestones(results, config)

    save_jsonl(
        [
            {
                "task_id": strategy.task_id,
                "group_id": strategy.group_id,
                "milestone": milestone,
                "cancelled": results[i].cancelled,
                "trajectory_path": str(results[i].trajectory_path)
                if results[i].trajectory_path
                else None,
                "strategy": strategy.strategy,
            }
            for i, (strategy, milestone) in enumerate(scored)
        ],
        eval_dir / "rewards.jsonl",
    )
    metrics = _rollout_quality_metrics(scored, results)
    metrics.update({
        "label": label,
        "task_ids": task_ids,
        "samples_per_task": samples_per_task,
        "group_size": samples_per_task,
        "archive_enabled": old_archive_enabled and config.validation_use_archive,
        "gen_seconds": gen_seconds,
        "exec_seconds": exec_seconds,
        "wall_seconds": int(time.monotonic() - t0),
    })
    save_json(metrics, eval_dir / "metrics.json")
    logger.info(
        f"Validation {label}: rollout_pass_rate={metrics['rollout_pass_rate']:.3f} "
        f"task_pass_at_{samples_per_task}={metrics['task_pass_at_n']:.3f} "
        f"avg_milestone={metrics['avg_milestone']:.2f} "
        f"cancelled={metrics['n_cancelled']}/{metrics['n_strategies']}"
    )
    return metrics


async def run_round(
    round_idx: int,
    planner: Planner,
    archive: Archive | None,
    config: Config,
    all_task_ids: list[str],
    rng: random.Random,
) -> dict:
    """Run one full GRPO round."""
    round_dir = config.output_dir / f"round_{round_idx:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    logger.info(f"=== ROUND {round_idx + 1}/{config.num_rounds} ===")

    if config.batch_size >= len(all_task_ids):
        batch_ids = list(all_task_ids)
        rng.shuffle(batch_ids)
        logger.info(f"Using full task pool ({len(batch_ids)} tasks) for round {round_idx}")
    else:
        batch_ids = rng.sample(all_task_ids, config.batch_size)
        logger.info(f"Sampled {len(batch_ids)} tasks for round {round_idx}")
    tasks = build_tasks(batch_ids, config)

    strategies_pkl = round_dir / "strategies.pkl"
    if strategies_pkl.exists():
        with open(strategies_pkl, "rb") as f:
            strategies = pickle.load(f)
        gen_seconds = 0
        logger.info(f"Resuming: loaded {len(strategies)} strategies from {strategies_pkl.name}")
    else:
        t_gen = time.monotonic()
        strategies = await planner.generate_strategies(tasks)
        gen_seconds = int(time.monotonic() - t_gen)
        with open(strategies_pkl, "wb") as f:
            pickle.dump(strategies, f)
        save_json(
            [
                {
                    "task_id": strategy.task_id,
                    "group_id": strategy.group_id,
                    "strategy": strategy.strategy,
                    "thinking": strategy.thinking,
                    "n_tokens": len(strategy.tokens),
                }
                for strategy in strategies
            ],
            round_dir / "strategies.json",
        )
        logger.info(f"Generation: {len(strategies)} strategies in {gen_seconds}s")

    t_exec = time.monotonic()
    results = execute_strategies(strategies, config, round_dir)
    exec_seconds = int(time.monotonic() - t_exec)
    logger.info(f"Execution: {len(results)} rollouts in {exec_seconds}s")

    scored = score_milestones(results, config)

    t_adh = time.monotonic()
    reflection_mode = "skipped"
    if config.lambda_adherence > 0.0:
        reflection_mode = "reward"
    elif config.judge_archive_only:
        reflection_mode = "archive_only"

    reward_adherences = [1.0] * len(results)
    judge_adherences: list[float | None] = [None] * len(results)
    insights = [""] * len(results)
    if reflection_mode != "skipped":
        try:
            pairs = await score_reflection_batch(
                results,
                base_url=config.judge_base_url,
                model=config.judge_model,
                concurrency=config.judge_parallel,
                max_traj_chars=config.judge_max_traj_chars,
                max_tokens=config.reflection_max_tokens,
                api_key=config.judge_api_key,
                insight_max_tokens=config.insight_max_tokens,
            )
        except RuntimeError as e:
            if reflection_mode == "archive_only":
                logger.warning(
                    "Reflection: archive_only judge failed for the whole round; "
                    "degrading to skipped and continuing with milestone-only reward: "
                    f"{e}"
                )
                reflection_mode = "skipped"
            else:
                raise
        else:
            judge_adherences = [adherence for adherence, _ in pairs]
            insights = [insight for _, insight in pairs]
            if reflection_mode == "reward":
                reward_adherences = list(judge_adherences)
    adh_seconds = int(time.monotonic() - t_adh)

    rewarded = [
        (
            strategy,
            compute_reward(
                milestone=milestone,
                adherence=reward_adherences[i],
                lambda_adherence=config.lambda_adherence,
                thinking_length=strategy.n_thinking_tokens,
                strategy_length=strategy.n_strategy_tokens,
                gamma_thinking=config.gamma_thinking,
                gamma_strategy=config.gamma_strategy,
                thinking_ref_tokens=config.thinking_ref_tokens,
                strategy_ref_tokens=config.strategy_ref_tokens,
                reward_compression=config.reward_compression,
            ),
            milestone,
        )
        for i, (strategy, milestone) in enumerate(scored)
    ]
    mean_judge_adherence = (
        sum(adherence for adherence in judge_adherences if adherence is not None)
        / max(sum(1 for adherence in judge_adherences if adherence is not None), 1)
        if reflection_mode != "skipped"
        else None
    )
    mean_reward_adherence = sum(reward_adherences) / max(len(reward_adherences), 1)
    n_with_insight = sum(1 for insight in insights if insight)

    if reflection_mode == "skipped":
        logger.info(
            "Reflection: SKIPPED "
            "(λ_adherence=0.0 and judge_archive_only=false); "
            "reward reduces to milestone + length terms."
        )
    elif reflection_mode == "archive_only":
        logger.info(
            f"Reflection: mode=archive_only mean_judge_adherence={mean_judge_adherence:.3f}, "
            f"insights={n_with_insight}/{len(insights)} in {adh_seconds}s; "
            f"reward ignores adherence (λ={config.lambda_adherence})."
        )
    else:
        logger.info(
            f"Reflection: mode=reward mean_judge_adherence={mean_judge_adherence:.3f}, "
            f"insights={n_with_insight}/{len(insights)} in {adh_seconds}s "
            f"(λ={config.lambda_adherence})"
        )

    save_jsonl(
        [
            {
                "task_id": strategy.task_id,
                "group_id": strategy.group_id,
                "reward": reward,
                "milestone": milestone,
                "adherence": judge_adherences[i],
                "judge_adherence": judge_adherences[i],
                "reward_adherence": reward_adherences[i],
                "reflection_mode": reflection_mode,
                "insight": insights[i],
                "n_thinking_tokens": strategy.n_thinking_tokens,
                "n_strategy_tokens": strategy.n_strategy_tokens,
                "strategy": strategy.strategy,
            }
            for i, (strategy, reward, milestone) in enumerate(rewarded)
        ],
        round_dir / "rewards.jsonl",
    )

    cancelled_mask = [getattr(results[i], "cancelled", False) for i in range(len(rewarded))]
    n_cancelled = sum(cancelled_mask)
    if n_cancelled:
        logger.info(
            f"GRPO group stats: keeping {n_cancelled}/{len(cancelled_mask)} "
            f"APRIL-cancelled rollouts as low-reward samples"
        )
    metrics = await planner.grpo_update(
        [(strategy, reward) for strategy, reward, _ in rewarded],
        round_idx=round_idx,
        cancelled_mask=cancelled_mask,
        milestones=[milestone for _, _, milestone in rewarded],
    )

    if archive is not None and config.archive_enabled:
        archive.append_batch(
            [
                {
                    "task_id": strategy.task_id,
                    "round": round_idx,
                    "group_id": strategy.group_id,
                    "strategy": strategy.strategy,
                    "milestone": milestone,
                    "adherence": judge_adherences[i],
                    "reflection_mode": reflection_mode,
                    "insight": insights[i],
                    "n_thinking_tokens": strategy.n_thinking_tokens,
                    "n_strategy_tokens": strategy.n_strategy_tokens,
                    "trajectory_path": (
                        str(results[i].trajectory_path) if results[i].trajectory_path else None
                    ),
                    "run_id": config.run_id,
                    "timestamp": datetime.now().isoformat(),
                }
                for i, (strategy, _, milestone) in enumerate(rewarded)
            ]
        )

    milestones = [milestone for _, _, milestone in rewarded]
    pass_rate = sum(1 for milestone in milestones if milestone == 7) / max(len(milestones), 1)
    avg_milestone = sum(milestones) / max(len(milestones), 1)
    n_samples_with_priors = sum(1 for strategy, _, _ in rewarded if strategy.priors_shown)
    frac_with_priors = n_samples_with_priors / max(len(rewarded), 1)
    mean_priors_per_sample = (
        sum(len(strategy.priors_shown) for strategy, _, _ in rewarded) / max(len(rewarded), 1)
    )
    by_task_prior_sets: dict[str, set] = defaultdict(set)
    for strategy, _, _ in rewarded:
        signature = tuple((prior["strategy"], prior["milestone"]) for prior in strategy.priors_shown)
        by_task_prior_sets[strategy.task_id].add(signature)
    distinct_priors_counts = [len(values) for values in by_task_prior_sets.values()]
    mean_distinct_priors_per_task = (
        sum(distinct_priors_counts) / max(len(distinct_priors_counts), 1)
    )
    archive_size = archive.size() if archive is not None else 0
    think_lens = [strategy.n_thinking_tokens for strategy, _, _ in rewarded]
    strat_lens = [strategy.n_strategy_tokens for strategy, _, _ in rewarded]
    metrics.update(
        {
            "round": round_idx,
            "n_strategies": len(rewarded),
            "n_tasks": len(tasks),
            "pass_rate": pass_rate,
            "train_batch_pass_rate_pre_update": pass_rate,
            "avg_milestone": avg_milestone,
            "train_batch_avg_milestone_pre_update": avg_milestone,
            "milestone_histogram": {i: milestones.count(i) for i in range(8)},
            "reflection_mode": reflection_mode,
            "mean_adherence": mean_judge_adherence,
            "mean_reward_adherence": mean_reward_adherence,
            "n_judged_rollouts": sum(1 for adherence in judge_adherences if adherence is not None),
            "frac_with_priors": frac_with_priors,
            "mean_priors_per_sample": mean_priors_per_sample,
            "mean_distinct_priors_per_task": mean_distinct_priors_per_task,
            "archive_size": archive_size,
            "mean_thinking_tokens": sum(think_lens) / max(len(think_lens), 1),
            "mean_strategy_tokens": sum(strat_lens) / max(len(strat_lens), 1),
            "learning_rate": config.learning_rate,
            "advantage_normalization": config.advantage_normalization,
            "advantage_std_floor": config.advantage_std_floor,
            "reward_compression": config.reward_compression,
            "gen_seconds": gen_seconds,
            "exec_seconds": exec_seconds,
            "adh_seconds": adh_seconds,
            "wall_seconds": int(time.monotonic() - t0),
        }
    )
    logger.info(
        f"Round {round_idx} done: pass_rate={pass_rate:.3f} "
        f"avg_milestone={avg_milestone:.2f} "
        f"degenerate={metrics['degenerate']}/{metrics['total_groups']}"
    )
    await planner.save_checkpoint(round_idx, metrics)
    save_json(metrics, round_dir / "metrics.json")
    return metrics


__all__ = ["build_tasks", "run_round", "run_validation_round", "score_milestones"]
