"""Deterministic 50-task subsample of TASKS_EVAL (seed=42).

Mirrors the convention used for PAGENT + qwen3.6-plus (which used a 100-task
seed-42 sample); writes the resulting 50-task list to TASKS_EVAL_50_seed42.
"""
import random
from pathlib import Path

ROOT = Path(__file__).parent
src = (ROOT / "TASKS_EVAL").read_text().splitlines()
src = [l.strip() for l in src if l.strip() and not l.startswith("#")]
assert len(src) == 200, f"expected 200 tasks in TASKS_EVAL, got {len(src)}"

rng = random.Random(42)
sample = sorted(rng.sample(src, 50))
out = ROOT / "TASKS_EVAL_50_seed42"
out.write_text("\n".join(sample) + "\n")
print(f"wrote {len(sample)} tasks → {out}")
