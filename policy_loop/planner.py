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
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tinker
    import torch
    from tinker import TensorData
    from tinker_cookbook.renderers import get_renderer, get_text_content
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False
    tinker = None  # type: ignore
    torch = None  # type: ignore

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
    # Optional: retrieved prior strategies for this task (Phase 2 archive)
    prior_strategies: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class StrategyToExecute:
    """A generated strategy, with bookkeeping needed for GRPO update.

    `prompt` is the Tinker ModelInput used to sample this strategy — we keep
    it (not just its tokens) because `prompt.append(...)` is needed to build
    the Datum model_input for GRPO (see demo.py).
    """
    task_id: str
    strategy: str                     # decoded text
    tokens: list[int]                 # strategy token IDs (for Datum)
    logprobs: list[float]             # per-token log-probabilities under sampling policy
    prompt: object = None             # Tinker ModelInput (keep original, not just tokens)
    prompt_length: int = 0            # cached prompt.length for convenience
    group_id: int = 0                 # index within the K-group (0..K-1)


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

    async def init(self) -> None:
        """Create the Tinker LoRA training client and renderers."""
        self.service_client = tinker.ServiceClient()
        self.training_client = (
            await self.service_client.create_lora_training_client_async(
                base_model=self.config.tinker_model,
                rank=self.config.tinker_rank,
            )
        )
        self.tokenizer = self.training_client.get_tokenizer()
        self.renderer = get_renderer("qwen3", self.tokenizer)
        self.sampling_params = tinker.SamplingParams(
            max_tokens=self.config.max_strategy_tokens,
            temperature=self.config.strategy_temperature,
            top_p=self.config.strategy_top_p,
            stop=self.renderer.get_stop_sequences(),
        )
        self.adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=self.config.adam_beta1,
            beta2=self.config.adam_beta2,
        )
        logger.info(
            f"Planner ready: {self.config.tinker_model} LoRA rank={self.config.tinker_rank}"
        )

    def _build_planner_prompt(self, task: Task):
        """Build the chat-formatted prompt for one task. Returns the Tinker ModelInput."""
        archive_block = format_archive_block(task.prior_strategies)
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

    async def generate_strategies(
        self,
        tasks: list[Task],
    ) -> list[StrategyToExecute]:
        """Generate K strategies per task (all in parallel). Returns K*N strategies."""
        K = self.config.group_size
        assert self.training_client is not None and self.renderer is not None

        # Save current weights → get on-policy sampling client
        sampling_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )

        # Build prompts and sample concurrently
        prompts = [self._build_planner_prompt(t) for t in tasks]
        coros = [
            sampling_client.sample_async(
                prompt=prompt,
                num_samples=K,
                sampling_params=self.sampling_params,
            )
            for prompt in prompts
        ]
        sample_results = await asyncio.gather(*coros)

        strategies: list[StrategyToExecute] = []
        for task, prompt, sample_result in zip(tasks, prompts, sample_results):
            for g, seq in enumerate(sample_result.sequences):
                if not seq.tokens:
                    continue
                # Decode the strategy text from the response tokens
                try:
                    parsed_msg, _ = self.renderer.parse_response(seq.tokens)
                    text = get_text_content(parsed_msg) or ""
                except Exception as e:
                    logger.warning(f"Failed to parse sample tokens for {task.task_id}: {e}")
                    text = self.tokenizer.decode(seq.tokens)

                strategies.append(StrategyToExecute(
                    task_id=task.task_id,
                    strategy=text,
                    tokens=list(seq.tokens),
                    logprobs=list(seq.logprobs),
                    prompt=prompt,                    # keep original ModelInput
                    prompt_length=prompt.length,
                    group_id=g,
                ))
        logger.info(
            f"Generated {len(strategies)} strategies across {len(tasks)} tasks (K={K})"
        )
        return strategies

    async def grpo_update(
        self,
        strategies_with_rewards: list[tuple[StrategyToExecute, float]],
        eps: float = 1e-8,
    ) -> dict:
        """Compute GRPO advantages per-task-group, build Datums, do gradient step.

        Returns metrics: {used, degenerate, total_groups, frac_degenerate, mean_reward}.
        """
        from collections import defaultdict

        assert self.training_client is not None and self.adam_params is not None

        # Group by task_id
        groups: dict[str, list[tuple[StrategyToExecute, float]]] = defaultdict(list)
        for strat, reward in strategies_with_rewards:
            groups[strat.task_id].append((strat, reward))

        datums: list = []
        n_degenerate = 0
        n_used = 0
        rewards_all: list[float] = []
        for _tid, group in groups.items():
            rewards = [r for _, r in group]
            rewards_all.extend(rewards)
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = var_r ** 0.5

            if std_r < eps:
                n_degenerate += 1
                continue

            advantages = [(r - mean_r) / (std_r + eps) for r in rewards]

            for (strat, _), adv in zip(group, advantages):
                if len(strat.tokens) < 2 or strat.prompt is None:
                    continue

                # Build model_input as in demo.py: prompt + strategy[:-1]
                # (next-token prediction — we predict tokens[1:] from prompt+tokens[:-1])
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
                datums.append(datum)
                n_used += 1

        metrics = {
            "used": n_used,
            "degenerate": n_degenerate,
            "total_groups": len(groups),
            "frac_degenerate": n_degenerate / max(len(groups), 1),
            "mean_reward": sum(rewards_all) / max(len(rewards_all), 1),
        }

        if not datums:
            logger.warning("No datums to train on this round (all groups degenerate)")
            return metrics

        # Forward-backward + optim step (matches demo.py importance_sampling loss)
        fwd_bwd_future = await self.training_client.forward_backward_async(
            datums, loss_fn="importance_sampling",
        )
        optim_future = await self.training_client.optim_step_async(self.adam_params)
        await fwd_bwd_future.result_async()
        await optim_future.result_async()

        logger.info(
            f"GRPO update: used={n_used}, degenerate={n_degenerate}/{len(groups)}, "
            f"mean_reward={metrics['mean_reward']:.3f}"
        )
        return metrics

    async def save_checkpoint(self, round_idx: int, metrics: dict) -> Path:
        """Save LoRA weights + per-round metrics."""
        ckpt_dir = self.config.checkpoint_dir / f"round_{round_idx:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Tinker: save weights to a path the service can later reload.
        # The API shape varies; we try common patterns.
        try:
            await self.training_client.save_weights_async(str(ckpt_dir / "lora_weights"))
        except AttributeError:
            # Older/newer API: save_state_async or similar
            try:
                await self.training_client.save_state_async(str(ckpt_dir / "lora_state"))
            except Exception as e:
                logger.warning(f"Could not save Tinker weights: {e}")

        # Always save metrics
        metrics_path = ckpt_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved checkpoint to {ckpt_dir}")
        return ckpt_dir

    async def load_checkpoint(self, round_idx: int) -> None:
        """Resume from a previous round's checkpoint."""
        ckpt_dir = self.config.checkpoint_dir / f"round_{round_idx:03d}"
        weights_path = ckpt_dir / "lora_weights"
        try:
            await self.training_client.load_weights_async(str(weights_path))
            logger.info(f"Loaded checkpoint from {ckpt_dir}")
        except AttributeError:
            logger.warning("Tinker client has no load_weights_async; skipping resume")
