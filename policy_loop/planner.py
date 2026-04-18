"""Tinker-based trainable planner.

The planner is Qwen3.5-27B with LoRA (rank 32), trained via GRPO using the
Tinker service. Each round:
  1. save_weights_and_get_sampling_client → get on-policy sampler
  2. generate_strategies → K candidate strategies per task
  3. (external: execute + score via MiniMax)
  4. grpo_update → compute advantages, forward/backward, optim step
  5. save_checkpoint
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tinker
    import torch
    from tinker import TensorData
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer, get_text_content
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False
    tinker = None  # type: ignore
    torch = None  # type: ignore
    model_info = None  # type: ignore

from .config import Config
from .prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE, format_archive_block

logger = logging.getLogger(__name__)


# ==========================================================================
# Data classes
# ==========================================================================

@dataclass
class Task:
    """One CyberGym task with the vulnerability description."""
    task_id: str
    description: str
    # Legacy field — no longer populated by build_tasks. Retrieval has moved
    # into generate_strategies (one draw per sample, not once per task) so
    # that each of the K prompts for this task sees a different subset of
    # priors. Kept on the dataclass only for forward-compat with any code
    # still reading the attribute.
    prior_strategies: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class StrategyToExecute:
    """A generated strategy, with bookkeeping needed for GRPO update.

    `prompt` is the Tinker ModelInput used to sample this strategy — we keep
    it (not just its tokens) because `prompt.append(...)` is needed to build
    the Datum model_input for GRPO (see demo.py).
    """
    task_id: str
    strategy: str                     # decoded text (thinking stripped for executor)
    thinking: str = ""                # thinking content (preserved for analysis)
    tokens: list[int] = field(default_factory=list)  # full token IDs including thinking (for Datum)
    logprobs: list[float] = field(default_factory=list)  # per-token log-probabilities under sampling policy
    prompt: object = None             # Tinker ModelInput (keep original, not just tokens)
    prompt_length: int = 0            # cached prompt.length for convenience
    group_id: int = 0                 # index within the K-group (0..K-1)
    n_thinking_tokens: int = 0        # token-level length of the thinking span (pre-</think>)
    n_strategy_tokens: int = 0        # token-level length of the strategy span (post-</think>)
    priors_shown: list = field(default_factory=list)  # list[dict{strategy, milestone, insight}] shown in this sample's prompt


def _split_thinking(text: str) -> tuple[str, str]:
    """Split thinking content from strategy. Returns (strategy, thinking)."""
    marker = "</think>"
    idx = text.find(marker)
    if idx >= 0:
        thinking = text[:idx].strip()
        strategy = text[idx + len(marker):].strip()
        return strategy, thinking
    # No </think> tag — check for "Thinking Process:" plain-text pattern
    # (fallback for truncated outputs that never reached </think>)
    return text, ""


def _find_subsequence(tokens: list[int], needle: list[int]) -> int:
    """Return start index of `needle` in `tokens`, or -1 if not found.
    O(N*M) linear search; N is at most max_strategy_tokens (16k) and M is 2-3.
    """
    if not needle or len(needle) > len(tokens):
        return -1
    for i in range(len(tokens) - len(needle) + 1):
        if tokens[i : i + len(needle)] == needle:
            return i
    return -1


def _split_token_spans(
    tokens: list[int],
    close_think_tokens: list[int] | None,
) -> tuple[int, int]:
    """Return (n_thinking_tokens, n_strategy_tokens) from a full response token list.

    If </think> is not present (either thinking mode disabled or the sequence was
    truncated before emitting it), everything counts as strategy.
    """
    if not close_think_tokens:
        return 0, len(tokens)
    idx = _find_subsequence(tokens, close_think_tokens)
    if idx < 0:
        return 0, len(tokens)
    n_think = idx
    n_strat = len(tokens) - idx - len(close_think_tokens)
    return n_think, max(n_strat, 0)


# ==========================================================================
# Planner
# ==========================================================================

class Planner:
    """Tinker LoRA training client for the Qwen3.5-27B planner."""

    def __init__(self, config: Config):
        if not TINKER_AVAILABLE:
            raise RuntimeError(
                "Tinker SDK not installed. Run: uv add tinker tinker-cookbook"
            )
        self.config = config
        self.service_client = None
        self.training_client = None
        self.tokenizer = None
        self.renderer = None
        self.sampling_params = None
        self.adam_params = None
        self.archive = None                # bound by run_round when archive_enabled

    def bind_archive(self, archive) -> None:
        """Attach an Archive instance so generate_strategies can do per-sample
        tournament draws. Pass None (or leave unset) to disable archive retrieval."""
        self.archive = archive

    async def init(self) -> None:
        """Create the Tinker LoRA training client and renderers."""
        kwargs = {}
        if self.config.tinker_api_key:
            kwargs["api_key"] = self.config.tinker_api_key
        self.service_client = tinker.ServiceClient(**kwargs)
        self.training_client = (
            await self.service_client.create_lora_training_client_async(
                base_model=self.config.tinker_model,
                rank=self.config.tinker_rank,
            )
        )
        self.tokenizer = self.training_client.get_tokenizer()
        renderer_name = model_info.get_recommended_renderer_name(self.config.tinker_model)
        self.renderer = get_renderer(renderer_name, self.tokenizer)
        # Pre-tokenize </think> marker for O(1) lookup during generation
        try:
            self._close_think_tokens = self.tokenizer.encode("</think>", add_special_tokens=False)
        except TypeError:
            # tokenizer may not accept add_special_tokens; fall back
            self._close_think_tokens = self.tokenizer.encode("</think>")
        self.sampling_params = tinker.SamplingParams(
            max_tokens=self.config.max_strategy_tokens,
            temperature=self.config.strategy_temperature,
            top_p=self.config.strategy_top_p,
            stop=self.renderer.get_stop_sequences(),
        )
        # Reference params (used when lr_schedule="constant"); cosine schedule rebuilds
        # the params per substep in grpo_update.
        self.adam_params = self._adam_params_at_lr(self.config.learning_rate)
        logger.info(
            f"Planner ready: {self.config.tinker_model} LoRA rank={self.config.tinker_rank}"
        )

    def _build_planner_prompt(self, task: Task, priors: list[tuple[str, int]]):
        """Build the chat-formatted prompt for one (task, priors) pair.
        Returns the Tinker ModelInput."""
        archive_block = format_archive_block(priors)
        user_content = PLANNER_USER_TEMPLATE.format(
            task_id=task.task_id,
            description=task.description,
            archive_block=archive_block,
        )
        convo = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.renderer.build_generation_prompt(convo)

    def _retrieve_priors_for_sample(self, task_id: str) -> list[tuple[str, int]]:
        """One tournament draw per call (RNG on the Archive produces different
        subsets across calls). Returns [] when the archive is not bound/enabled."""
        if self.archive is None or not self.config.archive_enabled:
            return []
        return self.archive.retrieve(
            task_id,
            n=self.config.archive_n,
            tournament_size=self.config.archive_tournament_size,
            min_milestone=self.config.archive_min_milestone,
        )

    async def generate_strategies(
        self,
        tasks: list[Task],
    ) -> list[StrategyToExecute]:
        """Generate K strategies per task with per-sample tournament retrieval.

        For every (task, k) pair we draw an independent tournament from the
        archive and build a distinct prompt, then dispatch N·K `num_samples=1`
        sampling calls in parallel. When the archive is empty or disabled,
        every draw returns [], so all K prompts for a task are identical and
        the behavior reduces to the previous `num_samples=K` form.
        """
        K = self.config.group_size
        assert self.training_client is not None and self.renderer is not None

        # Save current weights → get on-policy sampling client
        sampling_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )

        # One job per (task, group_id) — independent retrieve + independent prompt
        jobs: list[tuple[Task, int, object, list]] = []
        for task in tasks:
            for g in range(K):
                priors = self._retrieve_priors_for_sample(task.task_id)
                prompt = self._build_planner_prompt(task, priors)
                jobs.append((task, g, prompt, priors))

        coros = [
            sampling_client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=self.sampling_params,
            )
            for _, _, prompt, _ in jobs
        ]
        sample_results = await asyncio.gather(*coros)

        strategies: list[StrategyToExecute] = []
        for (task, g, prompt, priors), sample_result in zip(jobs, sample_results):
            if not sample_result.sequences:
                continue
            seq = sample_result.sequences[0]
            if not seq.tokens:
                continue
            try:
                parsed_msg, _ = self.renderer.parse_response(seq.tokens)
                full_text = get_text_content(parsed_msg) or ""
            except Exception as e:
                logger.warning(f"Failed to parse sample tokens for {task.task_id}: {e}")
                full_text = self.tokenizer.decode(seq.tokens)

            strategy, thinking = _split_thinking(full_text)
            n_think, n_strat = _split_token_spans(
                list(seq.tokens), getattr(self, "_close_think_tokens", None),
            )

            strategies.append(StrategyToExecute(
                task_id=task.task_id,
                strategy=strategy,
                thinking=thinking,
                tokens=list(seq.tokens),
                logprobs=list(seq.logprobs),
                prompt=prompt,
                prompt_length=prompt.length,
                group_id=g,
                n_thinking_tokens=n_think,
                n_strategy_tokens=n_strat,
                priors_shown=priors,
            ))

        # Diagnostic: how many distinct prior-sets were shown across K samples for a task?
        # (Sanity check that the per-sample draw actually produces variety.)
        from collections import defaultdict
        by_task_hashes: dict[str, set] = defaultdict(set)
        for s in strategies:
            sig = tuple((p["strategy"], p["milestone"])
                        if isinstance(p, dict) else tuple(p)
                        for p in s.priors_shown)
            by_task_hashes[s.task_id].add(sig)
        distinct_per_task = [len(v) for v in by_task_hashes.values()]
        if distinct_per_task:
            mean_distinct = sum(distinct_per_task) / len(distinct_per_task)
            logger.info(
                f"Generated {len(strategies)} strategies across {len(tasks)} tasks (K={K}); "
                f"mean distinct prior-sets per task = {mean_distinct:.2f} / {K}"
            )
        else:
            logger.info(
                f"Generated {len(strategies)} strategies across {len(tasks)} tasks (K={K})"
            )
        return strategies

    def _build_task_datums(
        self,
        strategies_with_rewards: list[tuple[StrategyToExecute, float]],
        eps: float,
    ) -> tuple[dict[str, list], dict]:
        """Compute per-task GRPO advantages and build datums grouped by task_id.

        Returns (task_datums, summary) where
          task_datums maps task_id -> list[tinker.Datum]  (degenerate groups omitted)
          summary has {used, degenerate, total_groups, frac_degenerate, mean_reward}.
        """
        from collections import defaultdict

        groups: dict[str, list[tuple[StrategyToExecute, float]]] = defaultdict(list)
        for strat, reward in strategies_with_rewards:
            groups[strat.task_id].append((strat, reward))

        task_datums: dict[str, list] = {}
        n_degenerate = 0
        n_used = 0
        rewards_all: list[float] = []

        for tid, group in groups.items():
            rewards = [r for _, r in group]
            rewards_all.extend(rewards)
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = var_r ** 0.5

            if std_r < eps:
                n_degenerate += 1
                continue

            advantages = [(r - mean_r) / (std_r + eps) for r in rewards]
            group_datums = []

            for (strat, _), adv in zip(group, advantages):
                if len(strat.tokens) < 2 or strat.prompt is None:
                    continue

                # model_input = prompt + strategy[:-1] (next-token prediction setup)
                model_input = strat.prompt.append(
                    tinker.EncodedTextChunk(tokens=list(strat.tokens[:-1]))
                )
                ob_len = strat.prompt_length - 1
                tokens = list(strat.tokens)
                padded_advantages = [0.0] * ob_len + [adv] * (model_input.length - ob_len)
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + list(strat.logprobs)

                datum = tinker.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                group_datums.append(datum)
                n_used += 1

            if group_datums:
                task_datums[tid] = group_datums

        summary = {
            "used": n_used,
            "degenerate": n_degenerate,
            "total_groups": len(groups),
            "frac_degenerate": n_degenerate / max(len(groups), 1),
            "mean_reward": sum(rewards_all) / max(len(rewards_all), 1),
        }
        return task_datums, summary

    @staticmethod
    def _split_task_ids(task_ids: list[str], num_splits: int) -> list[list[str]]:
        """Split task_ids into `num_splits` approximately equal sublists.

        Sizes differ by at most 1; mirrors tinker_cookbook.utils.misc_utils.split_list.
        """
        n = len(task_ids)
        num_splits = max(1, min(num_splits, n))
        # Base size + remainder distributed over first `rem` splits
        base, rem = divmod(n, num_splits)
        splits: list[list[str]] = []
        i = 0
        for s in range(num_splits):
            size = base + (1 if s < rem else 0)
            splits.append(task_ids[i : i + size])
            i += size
        return splits

    def _adam_params_at_lr(self, lr: float):
        """Build a fresh AdamParams with the given learning rate and the configured
        AdamW / grad-clip settings."""
        return tinker.AdamParams(
            learning_rate=lr,
            beta1=self.config.adam_beta1,
            beta2=self.config.adam_beta2,
            weight_decay=self.config.adam_weight_decay,
            grad_clip_norm=self.config.grad_clip_norm,
        )

    def _lr_at_step(self, global_step: int) -> float:
        """LR schedule evaluated at a zero-indexed global substep counter.

        Supports ``"constant"`` and ``"cosine"``. Cosine: linear warmup for the
        first ``lr_warmup_ratio * total_steps`` substeps, then cosine decay from
        ``learning_rate`` to ``learning_rate * lr_min_ratio`` over the remainder.
        """
        cfg = self.config
        peak = cfg.learning_rate
        total = max(cfg.num_rounds * max(cfg.num_substeps, 1), 1)
        if cfg.lr_schedule == "constant":
            return peak
        warmup = max(int(total * cfg.lr_warmup_ratio), 0)
        if global_step < warmup:
            return peak * (global_step + 1) / max(warmup, 1)
        progress = (global_step - warmup) / max(total - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        floor = peak * cfg.lr_min_ratio
        return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))

    async def grpo_update(
        self,
        strategies_with_rewards: list[tuple[StrategyToExecute, float]],
        *,
        round_idx: int = 0,
        eps: float = 1e-8,
    ) -> dict:
        """Compute GRPO advantages per-task-group, build Datums, run `num_substeps`
        pipelined forward_backward + optim_step updates (mini-batch iterative GRPO).

        Each substep uses a disjoint subset of task groups (GRPO groups stay intact
        within a substep). `num_substeps=1` recovers the single-update behavior.

        Returns summary metrics plus per-substep info.
        """
        import random
        assert self.training_client is not None and self.adam_params is not None

        # 1. Build per-task datums (degenerate groups dropped inside)
        task_datums, metrics = self._build_task_datums(strategies_with_rewards, eps)

        if not task_datums:
            logger.warning("No datums to train on this round (all groups degenerate)")
            metrics.update({"num_substeps": 0, "substep_datum_counts": []})
            return metrics

        # 2. Shuffle task_ids deterministically per-round, split into substeps
        task_ids = list(task_datums.keys())
        rng = random.Random(42 + round_idx)
        rng.shuffle(task_ids)

        num_substeps = max(1, self.config.num_substeps)
        task_splits = self._split_task_ids(task_ids, num_substeps)
        datum_batches: list[list] = [
            [d for tid in split for d in task_datums[tid]]
            for split in task_splits
        ]
        # Drop any empty batches (shouldn't happen since task_datums is non-empty)
        datum_batches = [b for b in datum_batches if b]
        actual_substeps = len(datum_batches)

        # 3. Pipelined fwd_bwd + optim_step (pattern from tinker_cookbook.rl.train.train_step)
        substep_metrics: list[dict] = []

        # Global substep index drives the LR schedule; round_idx * configured-S is the
        # base, and within-round substep i increments it. Using configured num_substeps
        # (not `actual_substeps`) keeps the schedule reproducible across runs even when
        # a round happens to drop degenerate groups down to fewer substeps.
        base_step = round_idx * max(self.config.num_substeps, 1)

        def _adam(i: int):
            return self._adam_params_at_lr(self._lr_at_step(base_step + i))

        # Enqueue first batch
        fwd_bwd_future = await self.training_client.forward_backward_async(
            datum_batches[0], loss_fn="importance_sampling",
        )
        optim_future = await self.training_client.optim_step_async(_adam(0))

        for i in range(actual_substeps):
            # Enqueue next batch before awaiting current (keeps pipeline full)
            if i + 1 < actual_substeps:
                next_fwd = await self.training_client.forward_backward_async(
                    datum_batches[i + 1], loss_fn="importance_sampling",
                )
                next_opt = await self.training_client.optim_step_async(_adam(i + 1))
            else:
                next_fwd = None
                next_opt = None

            # Consume current
            await fwd_bwd_future.result_async()
            opt_result = await optim_future.result_async()
            step_info = {
                "substep": i,
                "n_datums": len(datum_batches[i]),
                "lr": self._lr_at_step(base_step + i),
            }
            if getattr(opt_result, "metrics", None):
                step_info["optim_metrics"] = dict(opt_result.metrics)
            substep_metrics.append(step_info)

            if next_fwd is not None:
                fwd_bwd_future = next_fwd
                optim_future = next_opt

        metrics.update({
            "num_substeps": actual_substeps,
            "substep_datum_counts": [len(b) for b in datum_batches],
            "substep_metrics": substep_metrics,
        })

        logger.info(
            f"GRPO update: used={metrics['used']}, "
            f"degenerate={metrics['degenerate']}/{metrics['total_groups']}, "
            f"substeps={actual_substeps}, mean_reward={metrics['mean_reward']:.3f}"
        )
        return metrics

    async def save_checkpoint(self, round_idx: int, metrics: dict) -> Path:
        """Save LoRA weights + per-round metrics."""
        ckpt_dir = self.config.checkpoint_dir / f"round_{round_idx:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Tinker: save_state_async(name, ttl_seconds) persists LoRA weights
        # on the Tinker service side. Returns a checkpoint reference.
        try:
            ckpt_name = f"round_{round_idx:03d}"
            future = await self.training_client.save_state_async(
                name=ckpt_name, ttl_seconds=604800,  # 7 days
            )
            checkpoint = await future.result_async()
            # SaveWeightsResponse has a .path field (tinker:// URI) used for resume
            metrics["tinker_checkpoint"] = checkpoint.path
            logger.info(f"Saved Tinker state: {ckpt_name} → {checkpoint.path}")
        except Exception as e:
            logger.warning(f"Could not save Tinker state: {e}")

        # Always save metrics locally
        metrics_path = ckpt_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved checkpoint to {ckpt_dir}")
        return ckpt_dir

    async def load_checkpoint(self, round_idx: int) -> bool:
        """Resume from a previous round's checkpoint.

        Reads the tinker_checkpoint path from saved metrics and restores state.
        Returns True if loaded successfully.
        """
        ckpt_dir = self.config.checkpoint_dir / f"round_{round_idx:03d}"
        metrics_path = ckpt_dir / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"No metrics at {metrics_path}; cannot resume")
            return False
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            ckpt_ref = metrics.get("tinker_checkpoint")
            if not ckpt_ref:
                logger.warning(f"No tinker_checkpoint in {metrics_path}")
                return False
            logger.info(f"Resuming from Tinker checkpoint: {ckpt_ref}")
            # Re-create training client from saved state (async API)
            self.training_client = await (
                self.service_client.create_training_client_from_state_with_optimizer_async(
                    ckpt_ref
                )
            )
            # Rebuild tokenizer + renderer (new training client)
            self.tokenizer = self.training_client.get_tokenizer()
            renderer_name = model_info.get_recommended_renderer_name(
                self.config.tinker_model
            )
            from tinker_cookbook.renderers import get_renderer
            self.renderer = get_renderer(renderer_name, self.tokenizer)
            logger.info(f"Resumed from round {round_idx} successfully")
            return True
        except Exception as e:
            logger.exception(f"Could not load checkpoint: {e}")
            return False
