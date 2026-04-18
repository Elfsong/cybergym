"""Configuration for the policy loop."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

PROJECT_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    # --- Tinker planner ---
    tinker_model: str = "Qwen/Qwen3.5-27B"
    tinker_rank: int = 32
    tinker_api_key: str = field(default_factory=lambda: os.getenv("TINKER_API_KEY", ""))

    # --- Executor (MiniMax via vLLM + OpenHands) ---
    executor_model: str = "openai/MiniMaxAI/MiniMax-M2.5"
    executor_base_url: str = "http://localhost:8000/v1"
    executor_parallel: int = 64
    executor_timeout: int = 1800
    executor_max_iter: int = 72
    executor_max_output_tokens: int = 8192
    executor_difficulty: str = "level1"

    # --- GRPO training ---
    group_size: int = 8
    batch_size: int = 100            # tasks per round
    num_substeps: int = 1            # mini-batch gradient updates per round (1 = single update)
    grad_accum: int = 4
    learning_rate: float = 2e-5           # peak LR (also the constant LR when lr_schedule="constant")
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.01        # AdamW weight decay (Tinker default is 0.0)
    grad_clip_norm: float = 1.0            # global grad-norm clip (Tinker default is 0.0 = disabled)
    lr_schedule: str = "cosine"            # {"constant", "cosine"}
    lr_min_ratio: float = 0.1              # cosine floor: min_lr = learning_rate * lr_min_ratio
    lr_warmup_ratio: float = 0.05          # linear warmup over first lr_warmup_ratio * total_steps steps
    kl_beta: float = 0.01                  # reserved (not currently wired into Tinker loss)
    num_rounds: int = 10
    max_strategy_tokens: int = 16384
    strategy_temperature: float = 0.7
    strategy_top_p: float = 0.95

    # --- Reward (milestone 0-7 → reward value) ---
    milestone_rewards: tuple = (0.0, 0.5, 1.5, 2.5, 4.0, 5.5, 8.0, 12.0)
    lambda_adherence: float = 0.0    # Phase 1: disabled
    gamma_thinking: float = 0.0      # reward weight on f_think = min(n_think/ref, 1)
    gamma_strategy: float = 0.0      # reward weight on f_strat = min(n_strat/ref, 1)
    thinking_ref_tokens: int = 3000  # saturation threshold for f_think (≈ observed p70)
    strategy_ref_tokens: int = 500   # saturation threshold for f_strat (≈ observed p90)

    # --- Archive (Phase 2) ---
    archive_enabled: bool = False
    archive_n: int = 3               # top-n strategies in context
    archive_tournament_size: int = 4
    archive_min_milestone: int = 3   # only retrieve strategies that submitted

    # --- Phase 2: reflection judge (adherence + insight) ---
    phase2_enabled: bool = False     # master switch; implies archive_enabled
    adherence_judge_model: str = "Qwen/Qwen3.5-27B"
    adherence_judge_base_url: str = "http://localhost:8001/v1"
    adherence_max_traj_chars: int = 8000
    adherence_concurrency: int = 64
    reflection_max_tokens: int = 8192   # Qwen3.5-27B has a long CoT ("Thinking Process:") before it
                                        # emits the final <adherence>/<insight> tags. We keep thinking
                                        # mode ON (better judgment quality) and give the budget to fit it.

    # --- Paths ---
    data_dir: Path = Path("/data/cybergym_data/cybergym-benchmark-data/data")
    train_root: Path = Path("/data/cybergym_data/cybergym-train-data")
    tasks_file: Path = PROJECT_DIR / "TASKS"
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
