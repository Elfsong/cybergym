"""Main iterative offline GRPO training loop.

Usage:
    uv run python -m dual_loops.train
    uv run python -m dual_loops.train --num-rounds 3 --batch-size 10

Each round:
    1. Sample batch_size tasks
    2. Generate K strategies per task (Tinker, on-policy)
    3. Execute all K*N strategies via MiniMax (parallel subprocesses)
    4. Score each: milestone → reward
    5. Compute GRPO advantages (per task group), gradient step
    6. Save checkpoint + metrics
    7. Append (strategy, milestone, adherence, insight) records to the experience archive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
import random
import time
from dataclasses import asdict
from datetime import datetime
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

logger = logging.getLogger("dual_loops.train")


def build_tasks(task_ids: list[str], config: Config, archive: Archive | None) -> list[Task]:
    """Materialize Task objects with descriptions only. Archive retrieval moved to
    `Planner.generate_strategies`, which draws priors per-sample (one tournament per
    sample, not per task). The `archive` argument is kept in the signature for
    backward compat but unused here.
    """
    del archive  # per-sample retrieval now happens inside the planner
    return [
        Task(task_id=tid, description=get_task_description(tid, config.data_dir))
        for tid in task_ids
    ]


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
            lambda_adherence=config.lambda_adherence,
            thinking_length=r.strategy.n_thinking_tokens,
            strategy_length=r.strategy.n_strategy_tokens,
            gamma_thinking=config.gamma_thinking,
            gamma_strategy=config.gamma_strategy,
            thinking_ref_tokens=config.thinking_ref_tokens,
            strategy_ref_tokens=config.strategy_ref_tokens,
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

    # 1) Select tasks: if batch_size >= pool, use all tasks (shuffled);
    #    otherwise sample without replacement.
    if config.batch_size >= len(all_task_ids):
        batch_ids = list(all_task_ids)
        rng.shuffle(batch_ids)
        logger.info(f"Using full task pool ({len(batch_ids)} tasks) for round {round_idx}")
    else:
        batch_ids = rng.sample(all_task_ids, config.batch_size)
        logger.info(f"Sampled {len(batch_ids)} tasks for round {round_idx}")
    tasks = build_tasks(batch_ids, config, archive)

    # 2) Generate K strategies per task (on-policy) — or reload from pickle on resume
    strategies_pkl = round_dir / "strategies.pkl"
    if strategies_pkl.exists():
        with open(strategies_pkl, "rb") as f:
            strategies = pickle.load(f)
        gen_seconds = 0
        logger.info(
            f"Resuming: loaded {len(strategies)} strategies from {strategies_pkl.name}"
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
                    "task_id": s.task_id,
                    "group_id": s.group_id,
                    "strategy": s.strategy,
                    "thinking": s.thinking,
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

    # 4a) Score milestone (initial pass uses adherence=1.0; reflection below supplies the real value)
    rewarded = score_results(results, config)

    # 4b) Reflection judge → (adherence, insight) per rollout → composite reward
    from dual_loops.adherence import score_reflection_batch
    t_adh = time.monotonic()
    pairs = await score_reflection_batch(
        results,
        base_url=config.adherence_judge_base_url,
        model=config.adherence_judge_model,
        concurrency=config.adherence_concurrency,
        max_traj_chars=config.adherence_max_traj_chars,
        max_tokens=config.reflection_max_tokens,
    )
    adherences = [a for a, _ in pairs]
    insights = [ins for _, ins in pairs]
    adh_seconds = int(time.monotonic() - t_adh)
    # Recompose reward with the gated / composite formula:
    #   r = a · r_milestone + λ · a + γ_t · f_think + γ_s · f_strat
    rewarded = [
        (s, compute_reward(
            milestone=m,
            adherence=adherences[i],
            lambda_adherence=config.lambda_adherence,
            thinking_length=s.n_thinking_tokens,
            strategy_length=s.n_strategy_tokens,
            gamma_thinking=config.gamma_thinking,
            gamma_strategy=config.gamma_strategy,
            thinking_ref_tokens=config.thinking_ref_tokens,
            strategy_ref_tokens=config.strategy_ref_tokens,
        ), m)
        for i, (s, _, m) in enumerate(rewarded)
    ]
    mean_adherence = sum(adherences) / max(len(adherences), 1)
    n_with_insight = sum(1 for ins in insights if ins)
    logger.info(
        f"Reflection: mean_adherence={mean_adherence:.3f}, insights={n_with_insight}/{len(insights)} "
        f"in {adh_seconds}s (λ={config.lambda_adherence})"
    )

    # Save detailed per-strategy outcomes
    save_jsonl(
        [
            {
                "task_id": s.task_id,
                "group_id": s.group_id,
                "reward": r,
                "milestone": m,
                "adherence": adherences[i],
                "insight": insights[i],
                "n_thinking_tokens": s.n_thinking_tokens,
                "n_strategy_tokens": s.n_strategy_tokens,
                "strategy": s.strategy,
            }
            for i, (s, r, m) in enumerate(rewarded)
        ],
        round_dir / "rewards.jsonl",
    )

    # 5) GRPO update (pass round_idx for per-round shuffle seed)
    metrics = await planner.grpo_update(
        [(s, r) for s, r, _ in rewarded],
        round_idx=round_idx,
    )

    # 6) Archive append — the v3 record carries strategy, milestone, adherence, insight, plus metadata
    if archive is not None and config.archive_enabled:
        archive.append_batch([
            {
                "task_id":           s.task_id,
                "round":             round_idx,
                "group_id":          s.group_id,
                "strategy":          s.strategy,
                "milestone":         m,
                "adherence":         adherences[i],
                "insight":           insights[i],
                "n_thinking_tokens": s.n_thinking_tokens,
                "n_strategy_tokens": s.n_strategy_tokens,
                "trajectory_path":   (str(results[i].trajectory_path)
                                      if results[i].trajectory_path else None),
                "run_id":            config.run_id,
                "timestamp":         datetime.now().isoformat(),
            }
            for i, (s, _, m) in enumerate(rewarded)
        ])

    # 7) Aggregate metrics + checkpoint
    milestones = [m for _, _, m in rewarded]
    pass_rate = sum(1 for m in milestones if m == 7) / max(len(milestones), 1)
    avg_milestone = sum(milestones) / max(len(milestones), 1)
    # Prior coverage is now sample-level: did THIS rollout see ≥1 prior?
    n_samples_with_priors = sum(1 for s, _, _ in rewarded if s.priors_shown)
    frac_with_priors = n_samples_with_priors / max(len(rewarded), 1)
    mean_priors_per_sample = (
        sum(len(s.priors_shown) for s, _, _ in rewarded) / max(len(rewarded), 1)
    )
    # Distinct prior-sets per task (unique priors_shown signatures within a K-group).
    # priors_shown is a list of dicts {strategy, milestone, insight}; we hash by
    # (strategy, milestone) since two draws with identical priors are structurally
    # the same for the planner regardless of the insight text.
    from collections import defaultdict
    by_task_prior_sets: dict[str, set] = defaultdict(set)
    for s, _, _ in rewarded:
        sig = tuple((p["strategy"], p["milestone"])
                    if isinstance(p, dict) else tuple(p)
                    for p in s.priors_shown)
        by_task_prior_sets[s.task_id].add(sig)
    distinct_priors_counts = [len(v) for v in by_task_prior_sets.values()]
    mean_distinct_priors_per_task = (
        sum(distinct_priors_counts) / max(len(distinct_priors_counts), 1)
    )
    archive_size = archive.size() if archive is not None else 0
    think_lens = [s.n_thinking_tokens for s, _, _ in rewarded]
    strat_lens = [s.n_strategy_tokens for s, _, _ in rewarded]
    metrics.update({
        "round": round_idx,
        "n_strategies": len(rewarded),
        "n_tasks": len(tasks),
        "pass_rate": pass_rate,
        "avg_milestone": avg_milestone,
        "milestone_histogram": {i: milestones.count(i) for i in range(8)},
        "mean_adherence": mean_adherence,
        "frac_with_priors": frac_with_priors,
        "mean_priors_per_sample": mean_priors_per_sample,
        "mean_distinct_priors_per_task": mean_distinct_priors_per_task,
        "archive_size": archive_size,
        "mean_thinking_tokens": sum(think_lens) / max(len(think_lens), 1),
        "mean_strategy_tokens": sum(strat_lens) / max(len(strat_lens), 1),
        "gen_seconds": gen_seconds,
        "exec_seconds": exec_seconds,
        "adh_seconds": adh_seconds,
        "wall_seconds": int(time.monotonic() - t0),
    })
    logger.info(
        f"Round {round_idx} done: pass_rate={pass_rate:.3f} avg_milestone={avg_milestone:.2f} "
        f"degenerate={metrics['degenerate']}/{metrics['total_groups']}"
    )
    await planner.save_checkpoint(round_idx, metrics)
    save_json(metrics, round_dir / "metrics.json")
    return metrics


def _find_last_completed_round(run_dir: Path) -> int:
    """Return the index of the highest completed round in a run dir, or -1."""
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return -1
    last = -1
    for d in ckpt_root.iterdir():
        if not d.is_dir() or not d.name.startswith("round_"):
            continue
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            # Only count rounds where Tinker state was persisted (resumable)
            if m.get("tinker_checkpoint"):
                last = max(last, int(d.name.split("_")[1]))
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
    return last


async def train(config: Config, resume_from: Path | None = None) -> None:
    """Main entry: run config.num_rounds rounds of iterative GRPO.

    If resume_from is provided, load the last completed round's Tinker state and
    continue from the next round. resume_from must point to a prior run_dir
    (e.g. dual_loops_runs/<run_id>).
    """
    # If resuming, reuse the existing run's output dir so logs/checkpoints extend.
    # If no rounds have completed yet, start_round stays at 0 — mid-round resume
    # is then driven by round_000/strategies.pkl + round_000/executions.jsonl.
    start_round = 0
    if resume_from is not None:
        if not resume_from.exists():
            raise RuntimeError(f"Resume path does not exist: {resume_from}")
        config.run_id = resume_from.name
        last_round = _find_last_completed_round(resume_from)
        start_round = max(last_round + 1, 0)

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
        f"mini_batch_size={config.mini_batch_size}, rounds={config.num_rounds}, "
        f"lr={config.learning_rate}"
    )
    logger.info(
        f"Archive: {'ON' if config.archive_enabled else 'OFF'} | "
        f"Reflection judge: {config.adherence_judge_model} @ {config.adherence_judge_base_url} | "
        f"λ={config.lambda_adherence}, γ_t={config.gamma_thinking}, γ_s={config.gamma_strategy}"
    )
    if start_round > 0:
        logger.info(f"RESUMING from round {start_round} (last completed: {start_round - 1})")

    # Save config snapshot (append suffix if resuming to preserve original)
    cfg_name = "config.json" if start_round == 0 else f"config_resumed_from_{start_round}.json"
    save_json(asdict(config), config.output_dir / cfg_name)

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
    planner.bind_archive(archive)   # enables per-sample retrieval in generate_strategies

    # Restore Tinker state if resuming
    if start_round > 0:
        ok = await planner.load_checkpoint(start_round - 1)
        if not ok:
            raise RuntimeError(
                f"Failed to restore Tinker state from round {start_round - 1}; "
                f"cannot resume safely."
            )

    # RNG is reseeded deterministically each round (round_idx offset), so
    # resuming with the same seed reproduces the same task sampling pattern.
    rng = random.Random(42)
    # Advance rng to match the state after skipped rounds (same sequence)
    for _ in range(start_round):
        if config.batch_size >= len(all_task_ids):
            _ = list(all_task_ids); rng.shuffle(_)
        else:
            rng.sample(all_task_ids, config.batch_size)

    round_metrics: list[dict] = []
    for round_idx in range(start_round, config.num_rounds):
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


def _load_dotenv() -> None:
    """Load .env from project root if it exists."""
    import os
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not os.environ.get(key):
            os.environ[key] = val


def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Iterative offline GRPO training")
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--tasks-file", type=Path, default=None)
    parser.add_argument("--no-archive", action="store_true",
                        help="Ablation: disable the experience archive (archive retrieval returns empty; "
                             "archive append is skipped). Archive is ON by default.")
    parser.add_argument("--lambda-adherence", type=float, default=None,
                        help="Coefficient on adherence-only bonus in composite reward "
                             "(reward = a · r_milestone + λ · a + …). Default 0.5.")
    parser.add_argument("--gamma-thinking", type=float, default=None,
                        help="Reward weight on normalized thinking length (γ_t · min(n_think/ref, 1)). "
                             "0 disables (default).")
    parser.add_argument("--gamma-strategy", type=float, default=None,
                        help="Reward weight on normalized strategy length (γ_s · min(n_strat/ref, 1)). "
                             "0 disables (default).")
    parser.add_argument("--thinking-ref-tokens", type=int, default=None,
                        help="Saturation threshold for f_think normalization (default 3000).")
    parser.add_argument("--strategy-ref-tokens", type=int, default=None,
                        help="Saturation threshold for f_strat normalization (default 500).")
    parser.add_argument("--executor-parallel", type=int, default=None)
    parser.add_argument("--executor-model", type=str, default=None)
    parser.add_argument("--executor-base-url", type=str, default=None)
    parser.add_argument("--executor-timeout", type=int, default=None)
    parser.add_argument("--tinker-api-key", type=str, default=None)
    parser.add_argument("--cybergym-api-key", type=str, default=None)
    parser.add_argument("--mini-batch-size", type=int, default=None,
                        help="Task groups per GRPO mini-batch. Substeps per round are "
                             "derived as ⌈batch_size / mini_batch_size⌉. Default 8.")
    parser.add_argument("--strategy-temperature", type=float, default=None,
                        help="Sampling temperature for strategy generation (default 1.0)")
    parser.add_argument("--strategy-top-p", type=float, default=None,
                        help="Nucleus sampling top_p for strategy generation (default 0.95)")
    parser.add_argument("--train-root", type=Path, default=None,
                        help="Root dir for training outputs (default /data/cybergym_data/cybergym-train-data)")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Path to a prior run dir (<train_root>/<run_id>) to resume from")
    args = parser.parse_args()

    config = Config()
    if args.train_root is not None:
        config.train_root = args.train_root
    if args.num_rounds is not None:
        config.num_rounds = args.num_rounds
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.group_size is not None:
        config.group_size = args.group_size
    if args.mini_batch_size is not None:
        config.mini_batch_size = args.mini_batch_size
    if args.strategy_temperature is not None:
        config.strategy_temperature = args.strategy_temperature
    if args.strategy_top_p is not None:
        config.strategy_top_p = args.strategy_top_p
    if args.tasks_file is not None:
        config.tasks_file = args.tasks_file
    if args.executor_parallel is not None:
        config.executor_parallel = args.executor_parallel
    if args.executor_model is not None:
        config.executor_model = args.executor_model
    if args.executor_base_url is not None:
        config.executor_base_url = args.executor_base_url
    if args.executor_timeout is not None:
        config.executor_timeout = args.executor_timeout
    if args.tinker_api_key is not None:
        config.tinker_api_key = args.tinker_api_key
    if args.cybergym_api_key is not None:
        config.cybergym_api_key = args.cybergym_api_key
    if args.no_archive:
        config.archive_enabled = False
    if args.lambda_adherence is not None:
        config.lambda_adherence = args.lambda_adherence
    if args.gamma_thinking is not None:
        config.gamma_thinking = args.gamma_thinking
    if args.gamma_strategy is not None:
        config.gamma_strategy = args.gamma_strategy
    if args.thinking_ref_tokens is not None:
        config.thinking_ref_tokens = args.thinking_ref_tokens
    if args.strategy_ref_tokens is not None:
        config.strategy_ref_tokens = args.strategy_ref_tokens

    asyncio.run(train(config, resume_from=args.resume_from))


if __name__ == "__main__":
    main()
