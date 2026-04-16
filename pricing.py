"""Centralized LLM pricing — all scripts import from here.

Prices live in pricing.json (USD per million tokens).
"""

from __future__ import annotations

import json
from pathlib import Path

_PRICING_FILE = Path(__file__).parent / "pricing.json"
_CACHE: dict | None = None


def _load() -> dict[str, dict]:
    global _CACHE
    if _CACHE is None:
        with open(_PRICING_FILE) as f:
            _CACHE = json.load(f)["models"]
    return _CACHE


def _match(model: str) -> dict | None:
    """Find pricing entry for a model string (supports partial match)."""
    pricing = _load()
    # Exact match (with or without "openai/" prefix)
    raw = model.removeprefix("openai/").removeprefix("MiniMaxAI/")
    if raw in pricing:
        return pricing[raw]
    # Partial match — check if any key appears in the model string
    for key, prices in pricing.items():
        if key in model:
            return prices
    return None


def compute_cost(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Compute cost in USD from token counts and model name."""
    prices = _match(model)
    if prices is None:
        return 0.0
    return (
        prompt_tokens * prices.get("input", 0)
        + completion_tokens * prices.get("output", 0)
        + cache_read_tokens * prices.get("cache_read", 0)
        + cache_write_tokens * prices.get("cache_write", 0)
    ) / 1_000_000
