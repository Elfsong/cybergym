"""Reward-facing API: milestone detection, reflection scoring, and reward composition."""

from __future__ import annotations

from .milestones import MILESTONE_REWARDS, MilestoneResult, detect_milestone
from .reflection import score_reflection_batch, summarize_trajectory


def compute_reward(
    milestone: int,
    adherence: float = 1.0,
    lambda_adherence: float = 0.0,
    thinking_length: int = 0,
    strategy_length: int = 0,
    gamma_thinking: float = 0.0,
    gamma_strategy: float = 0.0,
    thinking_ref_tokens: int = 3000,
    strategy_ref_tokens: int = 500,
    reward_compression: str = "none",
) -> float:
    """Composite reward:
        r = a · f(r_milestone) + λ · a + γ_t · f_think + γ_s · f_strat

    where f is the compression chosen by `reward_compression` ∈
    {"none", "log1p", "sqrt"}.
    """
    import math

    r_mile = MILESTONE_REWARDS[milestone]
    if reward_compression == "log1p":
        r_mile = math.log1p(r_mile)
    elif reward_compression == "sqrt":
        r_mile = math.sqrt(max(r_mile, 0.0))
    elif reward_compression != "none":
        raise ValueError(f"Unknown reward_compression: {reward_compression!r}")

    f_think = min(thinking_length / max(thinking_ref_tokens, 1), 1.0)
    f_strat = max(0.0, 1.0 - strategy_length / max(strategy_ref_tokens, 1))
    return (
        adherence * r_mile
        + lambda_adherence * adherence
        + gamma_thinking * f_think
        + gamma_strategy * f_strat
    )


__all__ = [
    "MILESTONE_REWARDS",
    "MilestoneResult",
    "compute_reward",
    "detect_milestone",
    "score_reflection_batch",
    "summarize_trajectory",
]
