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
from .rounds import run_round, run_validation_round
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


def _load_existing_validation_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "validation_metrics.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning(f"Could not load prior validation metrics from {path}")
        return []
    return data if isinstance(data, list) else []


def _load_json_task_ids(path: Path) -> list[str]:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning(f"Could not load task IDs from {path}")
        return []
    return [str(task_id) for task_id in data] if isinstance(data, list) else []


def _select_task_batch(
    all_task_ids: list[str],
    batch_size: int,
    rng: random.Random,
) -> list[str]:
    if batch_size >= len(all_task_ids):
        task_ids = list(all_task_ids)
        rng.shuffle(task_ids)
        return task_ids
    return rng.sample(all_task_ids, batch_size)


def _load_or_select_fixed_train_task_ids(
    config: Config,
    all_task_ids: list[str],
) -> list[str]:
    path = config.output_dir / "train_task_ids.json"
    if path.exists():
        task_ids = _load_json_task_ids(path)
        if task_ids:
            return task_ids
    task_ids = _select_task_batch(
        all_task_ids,
        config.batch_size,
        random.Random(config.seed),
    )
    save_json(task_ids, path)
    return task_ids


def _select_validation_task_ids(config: Config, all_task_ids: list[str]) -> list[str]:
    existing_path = config.output_dir / "validation_task_ids.json"
    if existing_path.exists():
        task_ids = _load_json_task_ids(existing_path)
        if task_ids:
            return task_ids
    if config.validation_batch_size <= 0:
        return []
    if config.validation_tasks_file is not None:
        task_ids = parse_tasks_file(config.validation_tasks_file)
        return task_ids[: config.validation_batch_size]
    rng = random.Random(config.validation_seed)
    n = min(config.validation_batch_size, len(all_task_ids))
    return rng.sample(all_task_ids, n)


def _metric_value(metrics: dict, metric: str) -> float | None:
    value = metrics.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_fraction(value: float, name: str) -> float:
    """Accept either 0..1 fractions or 0..100 percentages."""
    if value > 1.0:
        value = value / 100.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, or 0 and 100 percent")
    return value


def _load_best_validation(run_dir: Path) -> dict | None:
    path = run_dir / "best_validation.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning(f"Could not load best validation from {path}")
        return None
    return data if isinstance(data, dict) else None


def _update_best_validation(
    config: Config,
    round_idx: int,
    eval_metrics: dict,
    train_metrics: dict,
) -> dict | None:
    metric = config.early_stop_metric
    current = _metric_value(eval_metrics, metric)
    if current is None:
        logger.warning(f"Validation metric {metric!r} missing; cannot update best")
        return _load_best_validation(config.output_dir)

    previous = _load_best_validation(config.output_dir)
    previous_value = (
        float(previous["value"])
        if previous and previous.get("metric") == metric and "value" in previous
        else None
    )
    if (
        previous_value is not None
        and current <= previous_value + config.early_stop_min_delta
    ):
        return previous

    best = {
        "metric": metric,
        "value": current,
        "round": round_idx,
        "label": eval_metrics.get("label"),
        "tinker_checkpoint": train_metrics.get("tinker_checkpoint"),
        "checkpoint_dir": str(config.checkpoint_dir / f"round_{round_idx:03d}"),
        "rollout_pass_rate": eval_metrics.get("rollout_pass_rate"),
        "task_pass_at_n": eval_metrics.get("task_pass_at_n"),
        "avg_milestone": eval_metrics.get("avg_milestone"),
    }
    save_json(best, config.output_dir / "best_validation.json")
    logger.info(
        f"New best validation: {metric}={current:.4f} at round {round_idx}"
    )
    return best


def _validation_stop_reason(validation_metrics: list[dict], config: Config) -> str | None:
    if not config.early_stop_on_validation:
        return None

    metric = config.early_stop_metric
    baseline = next(
        (
            _metric_value(metrics, metric)
            for metrics in validation_metrics
            if metrics.get("round") == -1
        ),
        None,
    )
    round_values = [
        (int(metrics["round"]), value)
        for metrics in validation_metrics
        if metrics.get("round", -1) >= 0
        for value in [_metric_value(metrics, metric)]
        if value is not None
    ]
    if not round_values:
        return None

    latest_round, latest_value = round_values[-1]
    if (
        baseline is not None
        and latest_value + config.early_stop_baseline_tolerance < baseline
    ):
        return (
            f"{metric}={latest_value:.4f} at round {latest_round} fell below "
            f"pretrain baseline {baseline:.4f}"
        )

    patience = max(config.early_stop_patience, 0)
    if patience <= 0:
        return None

    best = float("-inf")
    stale = 0
    best_round = -1
    for round_idx, value in round_values:
        if value > best + config.early_stop_min_delta:
            best = value
            best_round = round_idx
            stale = 0
        else:
            stale += 1

    if stale >= patience:
        return (
            f"{metric} has not improved for {stale} validation rounds "
            f"(best {best:.4f} at round {best_round}, latest "
            f"{latest_value:.4f} at round {latest_round})"
        )
    return None


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
        f"lr={config.learning_rate}, adv_norm={config.advantage_normalization}, "
        f"reward_compression={config.reward_compression}, "
        f"skip_update={config.skip_grpo_update}, "
        f"skip_uniform_milestone_groups={config.skip_uniform_milestone_groups}, "
        f"fixed_train_batch={config.fixed_train_batch}"
    )
    logger.info(
        f"Archive: {'ON' if config.archive_enabled else 'OFF'} | "
        f"Reflection judge: {config.judge_model} @ {config.judge_base_url} | "
        f"λ={config.lambda_adherence}, judge_archive_only={config.judge_archive_only}, "
        f"γ_t={config.gamma_thinking}, γ_s={config.gamma_strategy}"
    )
    logger.info(
        f"APRIL: round_max_wall={config.executor_round_max_wall_seconds}s, "
        f"completion_threshold={config.executor_completion_threshold:.0%}, "
        f"per_task_completion_fraction="
        f"{config.executor_min_rollout_fraction_per_task:.1%}, "
        f"round_min_wall={config.executor_round_min_wall_seconds}s"
    )
    logger.info(
        f"Stability guards: min_nonzero_milestone_rate="
        f"{config.min_nonzero_milestone_rate_for_update}, "
        f"min_progress_task_rate={config.min_progress_task_rate_for_update}, "
        f"early_stop_on_validation={config.early_stop_on_validation}, "
        f"early_stop_metric={config.early_stop_metric}, "
        f"patience={config.early_stop_patience}"
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
    if not config.fixed_train_batch and (config.output_dir / "train_task_ids.json").exists():
        config.fixed_train_batch = True
        logger.info("Detected existing train_task_ids.json; continuing fixed-train-batch mode")
        save_json(asdict(config), config.output_dir / cfg_name)
    fixed_train_task_ids: list[str] | None = None
    if config.fixed_train_batch:
        fixed_train_task_ids = _load_or_select_fixed_train_task_ids(config, all_task_ids)
        logger.info(
            f"Fixed train batch: reusing {len(fixed_train_task_ids)} task IDs "
            "for every round"
        )
    validation_task_ids = _select_validation_task_ids(config, all_task_ids)
    if validation_task_ids:
        save_json(validation_task_ids, config.output_dir / "validation_task_ids.json")
        logger.info(
            f"Fixed validation: {len(validation_task_ids)} tasks, "
            f"samples_per_task="
            f"{config.validation_samples_per_task or config.validation_group_size or config.group_size}, "
            f"every={config.validation_every}, "
            f"archive={'ON' if config.validation_use_archive else 'OFF'}"
        )

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
    if not config.fixed_train_batch:
        for _ in range(start_round):
            _select_task_batch(all_task_ids, config.batch_size, rng)

    round_metrics = _load_existing_round_metrics(config.output_dir, start_round)
    validation_metrics = _load_existing_validation_metrics(config.output_dir)
    if validation_task_ids and start_round == 0:
        metrics = await run_validation_round(
            "pretrain", planner, archive, config, validation_task_ids
        )
        validation_metrics.append({"round": -1, **metrics})
        save_json(validation_metrics, config.output_dir / "validation_metrics.json")

    for round_idx in range(start_round, config.num_rounds):
        metrics = await run_round(
            round_idx,
            planner,
            archive,
            config,
            all_task_ids,
            rng,
            batch_ids=fixed_train_task_ids,
        )
        stop_reason: str | None = None
        if (
            validation_task_ids
            and config.validation_every > 0
            and (round_idx + 1) % config.validation_every == 0
        ):
            eval_metrics = await run_validation_round(
                f"round_{round_idx:03d}",
                planner,
                archive,
                config,
                validation_task_ids,
            )
            metrics["fixed_eval"] = eval_metrics
            save_json(metrics, config.output_dir / f"round_{round_idx:03d}" / "metrics.json")
            validation_metrics.append({"round": round_idx, **eval_metrics})
            save_json(validation_metrics, config.output_dir / "validation_metrics.json")
            _update_best_validation(config, round_idx, eval_metrics, metrics)
            stop_reason = _validation_stop_reason(validation_metrics, config)
            if stop_reason:
                early_stop = {
                    "triggered": True,
                    "round": round_idx,
                    "reason": stop_reason,
                    "metric": config.early_stop_metric,
                }
                metrics["early_stop"] = early_stop
                save_json(metrics, config.output_dir / f"round_{round_idx:03d}" / "metrics.json")
                save_json(early_stop, config.output_dir / "early_stop.json")
                logger.warning(f"Early stopping triggered: {stop_reason}")
        round_metrics.append(metrics)
        if stop_reason:
            break

    save_json(round_metrics, config.output_dir / "all_metrics.json")
    logger.info("=== Training complete ===")
    for metrics in round_metrics:
        eval_metrics = metrics.get("fixed_eval") or {}
        fixed_eval = (
            f" fixed_eval_task_pass_at_n={eval_metrics['task_pass_at_n']:.3f}"
            if "task_pass_at_n" in eval_metrics
            else ""
        )
        logger.info(
            f"Round {metrics['round']}: pass_rate={metrics['pass_rate']:.3f} "
            f"avg_milestone={metrics['avg_milestone']:.2f}{fixed_eval}"
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
        "--fixed-train-batch",
        action="store_true",
        help="Reuse one sampled task batch for every training round; useful for paired comparisons.",
    )
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
    parser.add_argument(
        "--max-strategy-tokens",
        type=int,
        default=None,
        help="Override planner generation token cap. Use smaller values such as 512 for smoke tests.",
    )
    parser.add_argument("--insight-max-tokens", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--skip-grpo-update",
        action="store_true",
        help="Execute and score rollouts, but skip forward_backward/optim_step.",
    )
    parser.add_argument(
        "--advantage-normalization",
        choices=("mean_only", "mean_std", "clipped_std"),
        default=None,
    )
    parser.add_argument("--advantage-std-floor", type=float, default=None)
    parser.add_argument(
        "--reward-compression",
        choices=("none", "log1p", "sqrt"),
        default=None,
    )
    parser.add_argument(
        "--skip-uniform-milestone-groups",
        dest="skip_uniform_milestone_groups",
        action="store_true",
        help="Skip task groups whose rollouts all reached the same milestone.",
    )
    parser.add_argument(
        "--train-uniform-milestone-groups",
        dest="skip_uniform_milestone_groups",
        action="store_false",
        help="Allow length/reward tie-breakers to train even when all milestones match.",
    )
    parser.set_defaults(skip_uniform_milestone_groups=None)
    parser.add_argument(
        "--min-nonzero-milestone-rate-for-update",
        type=float,
        default=None,
        help="Skip GRPO update when the rollout fraction with milestone > 0 is below this value. Set 0 to disable.",
    )
    parser.add_argument(
        "--min-progress-task-rate-for-update",
        type=float,
        default=None,
        help="Skip GRPO update when the task fraction with any milestone > 0 is below this value. Set 0 to disable.",
    )
    parser.add_argument("--planner-parallel", type=int, default=None)
    parser.add_argument("--judge-parallel", type=int, default=None)
    parser.add_argument("--executor-parallel", type=int, default=None)
    parser.add_argument("--executor-model", type=str, default=None)
    parser.add_argument("--executor-base-url", type=str, default=None)
    parser.add_argument("--executor-api-key", type=str, default=None)
    parser.add_argument("--executor-timeout", type=int, default=None)
    parser.add_argument("--executor-round-max-wall-seconds", type=int, default=None)
    parser.add_argument(
        "--executor-completion-threshold",
        type=float,
        default=None,
        help="Fraction or percent of tasks that must meet the per-task rollout threshold before APRIL early-stops.",
    )
    parser.add_argument(
        "--executor-min-rollout-fraction-per-task",
        type=float,
        default=None,
        help="Fraction or percent of each task's group_size rollouts that must complete before that task counts as complete.",
    )
    parser.add_argument("--executor-round-min-wall-seconds", type=int, default=None)
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
    parser.add_argument("--validation-tasks-file", type=Path, default=None)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument(
        "--validation-samples-per-task",
        type=int,
        default=None,
        help="Number of sampled rollouts per validation task.",
    )
    parser.add_argument(
        "--validation-group-size",
        type=int,
        default=None,
        help="Deprecated alias for --validation-samples-per-task.",
    )
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--validation-every", type=int, default=None)
    parser.add_argument(
        "--validation-use-archive",
        action="store_true",
        help="Use archive retrieval during fixed validation. Default isolates policy-only eval.",
    )
    parser.add_argument(
        "--early-stop-on-validation",
        dest="early_stop_on_validation",
        action="store_true",
        help="Stop after validation when the configured validation metric degrades.",
    )
    parser.add_argument(
        "--no-early-stop-on-validation",
        dest="early_stop_on_validation",
        action="store_false",
        help="Disable validation-based early stopping.",
    )
    parser.set_defaults(early_stop_on_validation=None)
    parser.add_argument("--early-stop-metric", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--early-stop-baseline-tolerance", type=float, default=None)
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
    if args.fixed_train_batch:
        config.fixed_train_batch = True
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
    if args.executor_round_max_wall_seconds is not None:
        config.executor_round_max_wall_seconds = args.executor_round_max_wall_seconds
    if args.executor_completion_threshold is not None:
        config.executor_completion_threshold = _normalize_fraction(
            args.executor_completion_threshold,
            "--executor-completion-threshold",
        )
    if args.executor_min_rollout_fraction_per_task is not None:
        config.executor_min_rollout_fraction_per_task = _normalize_fraction(
            args.executor_min_rollout_fraction_per_task,
            "--executor-min-rollout-fraction-per-task",
        )
    if args.executor_round_min_wall_seconds is not None:
        config.executor_round_min_wall_seconds = args.executor_round_min_wall_seconds
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
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.skip_grpo_update:
        config.skip_grpo_update = True
    if args.advantage_normalization is not None:
        config.advantage_normalization = args.advantage_normalization
    if args.advantage_std_floor is not None:
        config.advantage_std_floor = args.advantage_std_floor
    if args.reward_compression is not None:
        config.reward_compression = args.reward_compression
    if args.skip_uniform_milestone_groups is not None:
        config.skip_uniform_milestone_groups = args.skip_uniform_milestone_groups
    if args.min_nonzero_milestone_rate_for_update is not None:
        config.min_nonzero_milestone_rate_for_update = (
            args.min_nonzero_milestone_rate_for_update
        )
    if args.min_progress_task_rate_for_update is not None:
        config.min_progress_task_rate_for_update = (
            args.min_progress_task_rate_for_update
        )
    if args.thinking_ref_tokens is not None:
        config.thinking_ref_tokens = args.thinking_ref_tokens
    if args.strategy_ref_tokens is not None:
        config.strategy_ref_tokens = args.strategy_ref_tokens
    if args.max_strategy_tokens is not None:
        config.max_strategy_tokens = args.max_strategy_tokens
    if args.insight_max_tokens is not None:
        config.insight_max_tokens = args.insight_max_tokens
    if args.validation_tasks_file is not None:
        config.validation_tasks_file = args.validation_tasks_file
    if args.validation_batch_size is not None:
        config.validation_batch_size = args.validation_batch_size
    if args.validation_samples_per_task is not None:
        config.validation_samples_per_task = args.validation_samples_per_task
    if args.validation_group_size is not None:
        if args.validation_samples_per_task is not None:
            raise ValueError(
                "Use only one of --validation-samples-per-task or "
                "--validation-group-size."
            )
        config.validation_group_size = args.validation_group_size
    if args.validation_seed is not None:
        config.validation_seed = args.validation_seed
    if args.validation_every is not None:
        config.validation_every = args.validation_every
    if args.validation_use_archive:
        config.validation_use_archive = True
    if args.early_stop_on_validation is not None:
        config.early_stop_on_validation = args.early_stop_on_validation
    if args.early_stop_metric is not None:
        config.early_stop_metric = args.early_stop_metric
    if args.early_stop_patience is not None:
        config.early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        config.early_stop_min_delta = args.early_stop_min_delta
    if args.early_stop_baseline_tolerance is not None:
        config.early_stop_baseline_tolerance = args.early_stop_baseline_tolerance

    asyncio.run(train(config, resume_from=args.resume_from))


if __name__ == "__main__":
    main()
