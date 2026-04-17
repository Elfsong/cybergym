"""Experience archive with tournament selection retrieval (Phase 2)."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class Archive:
    """Append-only JSONL store of (strategy, milestone, task_id) records.

    Retrieval uses tournament selection:
        - Filter to same task_id with milestone >= min_milestone.
        - Sample `tournament_size` candidates, pick the highest-milestone one.
        - Repeat `n` times (without replacement).
    """

    def __init__(self, path: Path, seed: int | None = None):
        self.path = path
        self._index: dict[str, list[dict]] = defaultdict(list)
        self._rng = random.Random(seed)
        if path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._index[rec["task_id"]].append(rec)
                except (json.JSONDecodeError, KeyError):
                    continue
        n = sum(len(v) for v in self._index.values())
        logger.info(f"Archive loaded: {n} records across {len(self._index)} tasks")

    def append(self, strategy: str, milestone: int, task_id: str) -> None:
        """Add a new (strategy, milestone) pair for this task."""
        rec = {"task_id": task_id, "strategy": strategy, "milestone": milestone}
        self._index[task_id].append(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec))
            f.write("\n")

    def append_batch(self, records: list[dict]) -> None:
        """Batch append; records = [{"task_id", "strategy", "milestone"}, ...]"""
        for r in records:
            self.append(r["strategy"], r["milestone"], r["task_id"])

    def retrieve(
        self,
        task_id: str,
        n: int = 3,
        tournament_size: int = 4,
        min_milestone: int = 3,
    ) -> list[tuple[str, int]]:
        """Tournament selection: return up to n (strategy, milestone) pairs.

        - Filter the task's records to milestone >= min_milestone.
        - For each of n slots: sample t candidates, pick the highest-milestone one,
          remove it from the pool (without replacement).
        - If the pool is empty before filling n slots, return what we have.
        """
        pool = [r for r in self._index.get(task_id, []) if r["milestone"] >= min_milestone]
        if not pool:
            return []

        selected: list[dict] = []
        remaining = pool.copy()
        for _ in range(n):
            if not remaining:
                break
            t = min(tournament_size, len(remaining))
            candidates = self._rng.sample(remaining, t)
            winner = max(candidates, key=lambda r: r["milestone"])
            selected.append(winner)
            remaining.remove(winner)

        return [(r["strategy"], r["milestone"]) for r in selected]

    def size(self) -> int:
        return sum(len(v) for v in self._index.values())
