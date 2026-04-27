"""Main iterative offline GRPO training entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

from .archive import Archive
from .config import Config
from .planner import Planner
from .rounds import run_round
from .utils import parse_tasks_file, save_json, setup_logging

logger = logging.getLogger("dual_loops.train")


def _find_last_completed_round(run_dir: Path) -> int:
    """Return the index of the highest completed round in a run dir, or -1."""
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return -1
    last = -1
    for directory in ckpt_root.iterdir():
        if not directory.is_dir() or not directory.name.startswith("round_"):
            continue
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
        if metrics.get("tinker_checkpoint"):
            last = max(last, int(directory.name.split("_")[1]))
    return last


def _load_existing_round_metrics(run_dir: Path, end_round_exclusive: int) -> list[dict]:
    """Load previously-saved per-round metrics up to `end_round_exclusive`."""
    metrics: list[dict] = []
    for round_idx in range(max(end_round_exclusive, 0)):
        metrics_path = run_dir / f"round_{round_idx:03d}" / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path) as f:
                metrics.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            logger.warning(f"Could not load prior metrics from {metrics_path}")
    return metrics


async def train(config: Config, resume_from: Path | None = None) -> None:
    """Run `config.num_rounds` rounds of iterative GRPO training."""
    start_round = 0
    if resume_from is not None:
        if not resume_from.exists():
            raise RuntimeError(f"Resume path does not exist: {resume_from}")
        config.run_id = resume_from.name
        last_round = _find_last_completed_round(resume_from)
        start_round = max(last_round + 1, 0)

    config.ensure_dirs()
    setup_logging(config.log_path)

    logger.info("=== Policy Loop Training ===")
    logger.info(f"Run ID: {config.run_id}  Output: {config.output_dir}")
    logger.info(f"Planner: {config.planner_model} (LoRA rank {config.planner_rank})")
    logger.info(f"Executor: {config.executor_model} at {config.executor_base_url}")
    logger.info(
        f"GRPO: K={config.group_size}, batch={config.batch_size}, "
        f"mini_batch_size={config.mini_batch_size}, rounds={config.num_rounds}, "
        f"lr={config.learning_rate}"
    )
    logger.info(
        f"Archive: {'ON' if config.archive_enabled else 'OFF'} | "
        f"Reflection judge: {config.judge_model} @ {config.judge_base_url} | "
        f"λ={config.lambda_adherence}, judge_archive_only={config.judge_archive_only}, "
        f"γ_t={config.gamma_thinking}, γ_s={config.gamma_strategy}"
    )
    if config.judge_archive_only and config.lambda_adherence > 0.0:
        logger.info(
            "judge_archive_only is redundant when lambda_adherence > 0; "
            "reflection will still enter reward."
        )
    if start_round > 0:
        logger.info(f"RESUMING from round {start_round} (last completed: {start_round - 1})")

    cfg_name = "config.json" if start_round == 0 else f"config_resumed_from_{start_round}.json"
    save_json(asdict(config), config.output_dir / cfg_name)

    all_task_ids = parse_tasks_file(config.tasks_file)
    if not all_task_ids:
        raise RuntimeError(f"No tasks in {config.tasks_file}")
    logger.info(f"Task pool: {len(all_task_ids)} tasks")

    seed_paths: tuple[Path, ...] = ()
    if (
        config.archive_enabled
        and config.global_archive_path
        and config.global_archive_path.exists()
    ):
        seed_paths = (config.global_archive_path,)
    archive = (
        Archive(config.archive_path, seed=config.seed, seed_paths=seed_paths)
        if config.archive_enabled
        else None
    )

    planner = Planner(config)
    await planner.init()
    planner.bind_archive(archive)

    if start_round > 0:
        ok = await planner.load_checkpoint(start_round - 1)
        if not ok:
            raise RuntimeError(
                f"Failed to restore Tinker state from round {start_round - 1}; "
                f"cannot resume safely."
            )

    rng = random.Random(config.seed)
    logger.info(f"RNG seed: {config.seed}")
    for _ in range(start_round):
        if config.batch_size >= len(all_task_ids):
            batch_ids = list(all_task_ids)
            rng.shuffle(batch_ids)
        else:
            rng.sample(all_task_ids, config.batch_size)

    round_metrics = _load_existing_round_metrics(config.output_dir, start_round)
    for round_idx in range(start_round, config.num_rounds):
        metrics = await run_round(round_idx, planner, archive, config, all_task_ids, rng)
        round_metrics.append(metrics)

    save_json(round_metrics, config.output_dir / "all_metrics.json")
    logger.info("=== Training complete ===")
    for metrics in round_metrics:
        logger.info(
            f"Round {metrics['round']}: pass_rate={metrics['pass_rate']:.3f} "
            f"avg_milestone={metrics['avg_milestone']:.2f}"
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
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not os.environ.get(key):
            os.environ[key] = value


def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Iterative offline GRPO training")
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--tasks-file", type=Path, default=None)
    parser.add_argument(
        "--archive",
        dest="archive_enabled",
        action="store_true",
        help="Enable experience archive retrieval + append for this run. Current config default is OFF.",
    )
    parser.add_argument(
        "--no-archive",
        dest="archive_enabled",
        action="store_false",
        help="Disable the experience archive for this run.",
    )
    parser.set_defaults(archive_enabled=None)
    parser.add_argument(
        "--lambda-adherence",
        type=float,
        default=None,
        help="Coefficient on adherence-only bonus in composite reward "
        "(reward = a · r_milestone + λ · a + …). Default 0.0.",
    )
    parser.add_argument(
        "--judge-archive-only",
        action="store_true",
        help="Run the reflection judge and store adherence/insight in rewards.jsonl + archive, "
        "but do not feed adherence into reward.",
    )
    parser.add_argument(
        "--gamma-thinking",
        type=float,
        default=None,
        help="Reward weight on normalized thinking length (γ_t · min(n_think/ref, 1)). 0 disables.",
    )
    parser.add_argument(
        "--gamma-strategy",
        type=float,
        default=None,
        help="Reward weight on shorter strategies (γ_s · max(0, 1 - n_strat/ref)). 0 disables.",
    )
    parser.add_argument("--thinking-ref-tokens", type=int, default=None)
    parser.add_argument("--strategy-ref-tokens", type=int, default=None)
    parser.add_argument("--insight-max-tokens", type=int, default=None)
    parser.add_argument("--planner-parallel", type=int, default=None)
    parser.add_argument("--judge-parallel", type=int, default=None)
    parser.add_argument("--executor-parallel", type=int, default=None)
    parser.add_argument("--executor-model", type=str, default=None)
    parser.add_argument("--executor-base-url", type=str, default=None)
    parser.add_argument("--executor-api-key", type=str, default=None)
    parser.add_argument("--executor-timeout", type=int, default=None)
    parser.add_argument("--planner-base-url", type=str, default=None)
    parser.add_argument("--planner-api-key", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--cybergym-api-key", type=str, default=None)
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=None,
        help="Task groups per GRPO mini-batch. Substeps per round are derived as ⌈batch_size / mini_batch_size⌉.",
    )
    parser.add_argument("--strategy-temperature", type=float, default=None)
    parser.add_argument("--strategy-top-p", type=float, default=None)
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
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
    if args.planner_parallel is not None:
        config.planner_parallel = args.planner_parallel
    if args.judge_parallel is not None:
        config.judge_parallel = args.judge_parallel
    if args.executor_parallel is not None:
        config.executor_parallel = args.executor_parallel
    if args.executor_model is not None:
        config.executor_model = args.executor_model
    if args.executor_base_url is not None:
        config.executor_base_url = args.executor_base_url
    if args.executor_api_key is not None:
        config.executor_api_key = args.executor_api_key
    if args.executor_timeout is not None:
        config.executor_timeout = args.executor_timeout
    if args.planner_base_url is not None:
        config.planner_base_url = args.planner_base_url
    if args.planner_api_key is not None:
        config.planner_api_key = args.planner_api_key
    if args.judge_model is not None:
        config.judge_model = args.judge_model
    if args.judge_base_url is not None:
        config.judge_base_url = args.judge_base_url
    if args.judge_api_key is not None:
        config.judge_api_key = args.judge_api_key
    if args.cybergym_api_key is not None:
        config.cybergym_api_key = args.cybergym_api_key
    if args.archive_enabled is not None:
        config.archive_enabled = args.archive_enabled
    if args.lambda_adherence is not None:
        config.lambda_adherence = args.lambda_adherence
    if args.judge_archive_only:
        config.judge_archive_only = True
    if args.gamma_thinking is not None:
        config.gamma_thinking = args.gamma_thinking
    if args.gamma_strategy is not None:
        config.gamma_strategy = args.gamma_strategy
    if args.thinking_ref_tokens is not None:
        config.thinking_ref_tokens = args.thinking_ref_tokens
    if args.strategy_ref_tokens is not None:
        config.strategy_ref_tokens = args.strategy_ref_tokens
    if args.insight_max_tokens is not None:
        config.insight_max_tokens = args.insight_max_tokens

    asyncio.run(train(config, resume_from=args.resume_from))


if __name__ == "__main__":
    main()
