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
    grad_accum: int = 4
    learning_rate: float = 2e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    kl_beta: float = 0.01
    num_rounds: int = 10
    max_strategy_tokens: int = 1024
    strategy_temperature: float = 0.7
    strategy_top_p: float = 0.95

    # --- Reward (milestone 0-7 → reward value) ---
    milestone_rewards: tuple = (0.0, 0.5, 1.5, 2.5, 4.0, 5.5, 8.0, 12.0)
    lambda_adherence: float = 0.0    # Phase 1: disabled
    alpha_novelty: float = 0.0       # Phase 1: disabled

    # --- Archive (Phase 2) ---
    archive_enabled: bool = False
    archive_n: int = 3               # top-n strategies in context
    archive_tournament_size: int = 4
    archive_min_milestone: int = 3   # only retrieve strategies that submitted

    # --- Paths ---
    data_dir: Path = Path("/data/cybergym_data/cybergym-benchmark-data/data")
    tasks_file: Path = PROJECT_DIR / "TASKS"
    server: str = "http://172.17.0.1:8666"
    cybergym_api_key: str = field(default_factory=lambda: os.getenv("CYBERGYM_API_KEY", ""))
    run_id: str = field(default_factory=lambda: uuid4().hex[:8])

    @property
    def output_dir(self) -> Path:
        return PROJECT_DIR / "policy_loop_runs" / self.run_id

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
