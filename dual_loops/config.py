"""Configuration for the policy loop."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

PROJECT_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    # =========================================================================
    # Role 1 — Planner  (Tinker managed LoRA service)
    # =========================================================================
    # Uses Tinker's SDK rather than an OpenAI-compatible HTTP endpoint, but
    # still exposes the same (model, base_url, api_key) triplet as the other
    # two roles — Tinker accepts an override base_url in the ServiceClient
    # constructor (it also honors TINKER_BASE_URL env var and falls back to
    # https://tinker.thinkingmachines.dev/services/tinker-prod when empty).
    planner_model:    str = "Qwen/Qwen3.5-27B"
    planner_rank:     int = 32
    planner_base_url: str = ""      # Empty string → Tinker SDK default
                                    # (https://tinker.thinkingmachines.dev/
                                    #  services/tinker-prod). Override for
                                    # dev/staging Tinker endpoints.
    planner_api_key:  str = field(default_factory=lambda: os.getenv("TINKER_API_KEY", ""))
    planner_parallel: int = 64      # Max concurrent sample_async calls during
                                    # strategy generation (K*B coroutines
                                    # gathered per round; cap prevents the
                                    # naive 768-way asyncio.gather at K=16, B=48).

    # =========================================================================
    # Role 2 — Executor  (OpenHands scaffold → OpenAI-compatible chat endpoint)
    # =========================================================================
    # Defaults target the local 8xA100 vLLM. The model/base_url/api_key triplet
    # is written out explicitly: there is no env-var fallback chain, because
    # inferring "which backend is this" from which env var happens to be set
    # is brittle. To switch to DashScope:
    #
    #   --executor-model     openai/qwen3.6-plus
    #   --executor-base-url  https://dashscope.aliyuncs.com/compatible-mode/v1
    #   --executor-api-key   "$DASHSCOPE_API_KEY"
    #
    # For local vLLM the api_key field is a placeholder ("EMPTY") because
    # vLLM does not validate it — but LiteLLM requires the header to be
    # present, so we must set something.
    executor_model:    str = "openai/Qwen/Qwen3.5-27B"
    executor_base_url: str = "http://localhost:8001/v1"
    executor_api_key:  str = "EMPTY"
    executor_parallel: int = 64   # 32 -> 48 -> 64: round 1 (run 2b7eb258) produced
                                  # 12/48 tasks at K_min=5 with 70% APRIL cancel rate.
                                  # Pairing with batch_size=32 keeps total rollouts at
                                  # 32×8=256, matched by 64-parallel × 40min ÷ 712s
                                  # median = 216 expected completions. Earlier 200-task
                                  # eval with parallel=64 saw vLLM queue blow-up (688
                                  # steps >60s cumulative); to monitor on this run check
                                  # P99 LLM call latency and median rollout wall — if
                                  # median >900s (vs current 712s), back off to 48.
    executor_timeout:  int = 2400

    # APRIL-style early-stop: cap a round's executor phase by wall-clock and by
    # per-task completion fraction; cancel the long tail to keep round time
    # bounded by the median rather than P99. Set
    # executor_round_max_wall_seconds=0 to disable (sync wait-for-all behavior).
    # When stop fires, in-flight subprocesses are SIGTERM'd via their pgrp and
    # cancelled rollouts are recorded as ExecutionResult(trajectory_path=None,
    # status=CANCELLED) so downstream scoring sees them as failures.
    executor_round_max_wall_seconds: int = 2400       # hard cap per round
    executor_completion_threshold: float = 0.80       # fraction of tasks that need
                                                      # ≥ K_min completed rollouts
                                                      # to allow early termination
    executor_min_rollouts_per_task: int = 5           # K_min (vs group_size=8);
                                                      # below this a task is skipped
                                                      # by the APRIL stop check
    executor_round_min_wall_seconds: int = 600        # don't stop early in the first
                                                      # 10 min even if threshold met
                                                      # (avoids false-positive stops
                                                      # from fast-pass clusters)

    # Docker container launch rate limit. With 32 parallel rollouts each
    # spawning an OpenHands runtime image + per-task CyberGym sandbox, dockerd
    # saturates and ~82% of rollouts die at runtime-init with no trajectory
    # ("CRASH"). 5s per launch keeps the daemon happy and shifts the bottleneck
    # back to the agent loop where it belongs. Set to 0 to disable.
    executor_docker_stagger_seconds: float = 5.0
    executor_max_iter: int = 72
    executor_max_output_tokens: int = 4096
    executor_temperature: float = 0.7  # paper says OpenHands default = 0.7
    executor_difficulty:  str = "level1"

    # --- GRPO training ---
    group_size: int = 8               # K rollouts per task (intra-group GRPO)
    batch_size: int = 32             # tasks per round (task groups per round).
                                     # 48 -> 32 paired with executor_parallel=64:
                                     # 32×8=256 rollouts vs 64-parallel × 40min ÷
                                     # 712s = ~216 completions, so most rollouts
                                     # finish within budget instead of being
                                     # APRIL-cancelled.
    mini_batch_size: int = 8         # task groups per GRPO mini-batch.
                                     # Substeps per round are derived: S = ceil(batch_size / mini_batch_size).
    grad_accum: int = 4
    learning_rate: float = 5e-6           # peak LR; c4f76f38 stabilized PPO loss vs 1e-5.
    skip_grpo_update: bool = False        # Execute and score rollouts, but skip
                                          # forward_backward/optim_step. Useful
                                          # for no-op controls.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.01        # AdamW weight decay (Tinker default is 0.0)
    grad_clip_norm: float = 1.0            # global grad-norm clip (Tinker default is 0.0 = disabled)
    lr_schedule: str = "cosine"            # {"constant", "cosine"}
    lr_min_ratio: float = 0.1              # cosine floor: min_lr = learning_rate * lr_min_ratio
    lr_warmup_ratio: float = 0.10          # linear warmup over first lr_warmup_ratio * total_steps steps
    kl_beta: float = 0.01                  # reserved (not currently wired into Tinker loss)
    num_rounds: int = 12
    max_strategy_tokens: int = 2048     # observed p95 ≈ 600 tokens (clean strategies);
                                        # tighter cap limits runaway "safety-refusal loop"
                                        # outputs and keeps Tinker sampling bounded.
    strategy_temperature: float = 1.0   # higher temp gives intra-group strategy diversity
    strategy_top_p: float = 0.95

    # Master seed for reproducibility. Default training uses one RNG advanced
    # across rounds, so resuming round R reuses the original task samples.
    # `fixed_train_batch=True` reuses one task subset across all rounds for
    # paired comparisons that remove task-sample noise from train-batch metrics.
    # None disables seeding (legacy nondeterministic behavior).
    seed: int = 42
    fixed_train_batch: bool = False

    # --- Loss function (PPO-clip via Tinker) ---
    # Tinker supports {"importance_sampling", "ppo", "cispo", "dro"}.
    # "ppo" adds ratio clipping; required once sub-steps per round push the
    # policy off the sampling distribution.
    loss_fn_name: str = "ppo"
    ppo_clip_low_threshold: float = 0.2    # ε_low, passed to loss_fn_config
    ppo_clip_high_threshold: float = 0.2   # ε_high, passed to loss_fn_config

    # --- Advantage normalization (GRPO) ---
    # "mean_std":    (r - μ) / (σ + eps)  — classic, noisy in small groups
    # "mean_only":   r - μ                 — Dr.GRPO; removes σ-driven variance
    # "clipped_std": (r - μ) / max(σ, floor) — compromise
    advantage_normalization: str = "clipped_std"
    advantage_std_floor: float = 0.3
    skip_uniform_milestone_groups: bool = False
                                     # Ignore task groups whose rollout
                                     # milestones are all identical. This keeps
                                     # length tie-breakers from creating policy
                                     # gradients when there is no task-progress
                                     # signal. Default stays OFF to match the
                                     # full-datum c4f76f38 stabilizer run.

    # --- Reward (milestone 0-7 → reward value) ---
    milestone_rewards: tuple = (0.0, 0.5, 1.5, 2.5, 4.0, 5.5, 8.0, 12.0)
    # Compression applied to r_milestone BEFORE the adherence multiplier.
    # "none" | "log1p" (→ 0..2.56) | "sqrt" (→ 0..3.46). Reduces milestone=7
    # outlier dominance of intra-group advantages.
    reward_compression: str = "log1p"
    lambda_adherence: float = 0.0    # adherence-bonus weight in the composite reward.
                                     # With the current milestone-only default (0.0),
                                     # the judge affects reward only if explicitly
                                     # re-enabled. See `judge_archive_only` below for
                                     # the cheaper mode that records adherence/insight
                                     # to the archive without feeding them into reward.
    gamma_thinking: float = 0.0      # reward weight on f_think = min(n_think/ref, 1)
    gamma_strategy: float = 0.1      # reward weight on f_strat. NOTE: f_strat is now
                                     # max(0, 1 - n_strat/ref) — REWARDS SHORT strategies
                                     # (was the saturating-up form rewarding long). With
                                     # γ=0.1 and r_milestone ∈ [0..12], the strategy term
                                     # is at most 0.1 — a tiebreaker between equal-
                                     # milestone rollouts, not a primary signal. Counter
                                     # to the verbosity / safety-refusal-loop tail
                                     # observed in run 2b7eb258 (~5% of strategies hit
                                     # max_strategy_tokens=4096).
    thinking_ref_tokens: int = 3000  # saturation threshold for f_think (≈ observed p70)
    strategy_ref_tokens: int = 500   # f_strat zero-out threshold (rewards taper from 1
                                     # at n_strat=0 down to 0 at this cap). Observed
                                     # round-1 distribution: median 400, p90 470, p95
                                     # 520 — so median-length strategy gets f≈0.2,
                                     # very-short (200 tok) gets f≈0.6.

    # --- Experience archive (always part of the architecture; flag present for ablations) ---
    archive_enabled: bool = False    # disabled to isolate milestone learning from
                                     # prior-strategy injection. With λ=0 + archive
                                     # OFF, the planner has only the task description
                                     # + GRPO updates as signal — no retrieval prior.
    archive_n: int = 3               # top-n strategies in context
    archive_tournament_size: int = 4
    archive_min_milestone: int = 3   # only retrieve strategies that submitted

    # --- Fixed validation eval (disabled by default) ---
    # Validation pass_rate is checkpoint eval on the same task subset each time.
    # It is distinct from a round's training-batch pass_rate, which is measured
    # before that round's GRPO update and on a fresh sampled batch.
    validation_tasks_file: Path | None = None
    validation_batch_size: int = 0       # 0 disables sampled validation unless
                                         # validation_tasks_file is provided.
    validation_samples_per_task: int = 0 # 0 => reuse group_size.
    validation_group_size: int = 0       # Back-compat alias for
                                         # validation_samples_per_task.
    validation_seed: int = 314159
    validation_every: int = 1
    validation_use_archive: bool = False

    # =========================================================================
    # Role 3 — Judge  (frozen base model → OpenAI-compatible chat endpoint)
    # =========================================================================
    # Scores each rollout's adherence + emits the insight stored in the
    # archive. Defaults co-host on the same local vLLM as the executor (prompt
    # caching benefits; one vLLM instance serves both roles). The model MUST
    # NOT be the LoRA-adapted planner: self-judging introduces non-stationary
    # reward signal and self-reinforcement bias. Override the triplet to point
    # at a different vLLM or DashScope, same pattern as the executor section.
    judge_model:    str = "Qwen/Qwen3.5-27B"
    judge_base_url: str = "http://localhost:8001/v1"
    judge_api_key:  str = "EMPTY"
    judge_parallel: int = 64            # Max concurrent judge chat completions;
                                        # sibling of planner_parallel and
                                        # executor_parallel.
    judge_archive_only: bool = False    # Run the reflection judge even when
                                        # lambda_adherence == 0, but ONLY to
                                        # populate archive / rewards metadata.
                                        # Reward remains milestone + length terms.
    judge_max_traj_chars:   int = 16000 # Hard cap passed to summarize_trajectory
                                        # before the summary is handed to the judge.
    reflection_max_tokens:  int = 8192  # max_tokens for the judge's chat call;
                                        # Qwen3.5-27B emits ~5k tokens of thinking
                                        # before the final XML tags.
    insight_max_tokens:     int = 500   # Target length of the <insight> payload
                                        # itself (not the whole LLM response).
                                        # Baked into the prompt + post-hoc char
                                        # truncate safety net.

    # --- Paths ---
    data_dir: Path = Path("/data/cybergym_data/cybergym-benchmark-data/data")
    train_root: Path = Path("/data/cybergym_data/cybergym-train-data")
    # Shared experience pool seeded into every new run's Archive so training
    # doesn't cold-start. Set to None to disable. Per-run archives still write
    # only to their own output_dir; promote merged runs into this path manually.
    global_archive_path: Path | None = Path(
        "/data/cybergym_data/cybergym-train-data/_global/archive.jsonl"
    )
    tasks_file: Path = PROJECT_DIR / "TASKS_TRAIN"
    server: str = "http://172.17.0.1:8666"
    cybergym_api_key: str = field(default_factory=lambda: os.getenv(
        "CYBERGYM_API_KEY", "cybergym-030a0cd7-5908-4862-8ab9-91f2bfc7b56d"
    ))
    run_id: str = field(default_factory=lambda: uuid4().hex[:8])

    @property
    def output_dir(self) -> Path:
        return self.train_root / self.run_id

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def archive_path(self) -> Path:
        return self.output_dir / "archive.jsonl"

    @property
    def log_path(self) -> Path:
        return self.output_dir / "train.log"

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
