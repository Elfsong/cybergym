"""Experience archive with tournament selection retrieval."""

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

    def __init__(
        self,
        path: Path,
        seed: int | None = None,
        seed_paths: tuple[Path, ...] | list[Path] = (),
    ):
        """`seed_paths` are read-only archives (e.g. a promoted global
        experience pool) loaded before the per-run archive so new runs
        start hot instead of cold. The per-run archive at `path` is the
        only one appended to — seeds stay untouched.
        """
        self.path = path
        self._index: dict[str, list[dict]] = defaultdict(list)
        self._rng = random.Random(seed)
        own_resolved = path.resolve() if path.exists() else None
        for sp in seed_paths or ():
            if sp is None or not Path(sp).exists():
                continue
            if own_resolved is not None and Path(sp).resolve() == own_resolved:
                continue
            self._load_from(Path(sp), label=f"seed:{sp}")
        if path.exists():
            self._load_from(path, label="own")

    def _load_from(self, path: Path, label: str = "") -> None:
        added = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._index[rec["task_id"]].append(rec)
                    added += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        total = sum(len(v) for v in self._index.values())
        logger.info(
            f"Archive loaded {added} records from {path}"
            + (f" ({label})" if label else "")
            + f"; total now {total} across {len(self._index)} tasks"
        )

    # v2+ optional fields (backward compatible — records missing these are fine)
    _V2_FIELDS = (
        "round", "group_id", "adherence", "insight",
        "n_thinking_tokens", "n_strategy_tokens",
        "trajectory_path", "run_id", "timestamp",
    )

    def append(self, record: dict) -> None:
        """Add one record. Required: task_id, strategy, milestone.
        Optional v2 fields: round, group_id, adherence, trajectory_path,
        run_id, timestamp (pass-through to JSONL).
        """
        rec: dict = {
            "task_id":   record["task_id"],
            "strategy":  record["strategy"],
            "milestone": record["milestone"],
        }
        for k in self._V2_FIELDS:
            if k in record:
                rec[k] = record[k]
        self._index[rec["task_id"]].append(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec))
            f.write("\n")

    def append_batch(self, records: list[dict]) -> None:
        """Batch append; each record must have task_id/strategy/milestone plus
        any optional v2 fields."""
        for r in records:
            self.append(r)

    def retrieve(
        self,
        task_id: str,
        n: int = 3,
        tournament_size: int = 4,
        min_milestone: int = 3,
    ) -> list[dict]:
        """Tournament selection: return up to n prior records for this task.

        Each returned record is a dict with keys {"strategy", "milestone", "insight"};
        "insight" is an empty string when the source record predates reflection (v1/v2
        archives) or when the reflection judge failed to produce one.

        Selection algorithm:
        - Filter the task's records to milestone >= min_milestone.
        - For each of n slots: sample t candidates, pick the highest-milestone one,
          remove it from the pool (without replacement).
        - If the filtered pool is empty (no record at the quality threshold), fall
          back to the all-records pool. Without this fallback, hard tasks where
          every rollout fails to submit (m<3) end up with empty retrieval forever
          and the planner gets cold-start prompts every round, locking those
          tasks into a positive-feedback failure loop. We accept the slight
          quality dilution to break that loop.
        """
        all_records = self._index.get(task_id, [])
        pool = [r for r in all_records if r["milestone"] >= min_milestone]
        fallback = False
        if not pool:
            if not all_records:
                return []
            # No record meets the bar; use whatever we have so the planner sees
            # *something* about this task instead of cold-start.
            pool = all_records
            fallback = True

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

        return [
            {
                "strategy":  r["strategy"],
                "milestone": r["milestone"],
                "insight":   r.get("insight", "") or "",
                # mark fallback so callers/metrics can track quality dilution
                **({"_fallback": True} if fallback else {}),
            }
            for r in selected
        ]

    def size(self) -> int:
        return sum(len(v) for v in self._index.values())
