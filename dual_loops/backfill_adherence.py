"""Re-judge archive.jsonl with the fixed ADHERENCE_JUDGE_PROMPT.

Past archive (rounds 0-6) was generated with the buggy prompt that caused the
judge to echo the example insight text verbatim (68% `"A SHORT ACTIONABLE
TAKEAWAY."` + 32% empty). With the prompt fixed, we can re-run the judge on
every archived rollout and back-fill real insights, so later training rounds
and inference can retrieve useful priors.

Concurrency is kept moderate (default 16) to share the vLLM server with the
running GRPO training rounds. The script is safe to run while training is
live: it writes to a staging file, then atomically replaces archive.jsonl
after merging with any entries appended during the backfill.

Usage:
    python -m dual_loops.backfill_adherence \\
        --archive /data/cybergym_data/cybergym-train-data/96e38ba3/archive.jsonl \\
        --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import shutil
import time
from pathlib import Path

import openai

from dual_loops.prompts import ADHERENCE_JUDGE_PROMPT
from dual_loops.reward import (
    _PARSE_FAILED_ADH,
    _parse_reflection,
    summarize_trajectory,
)

logger = logging.getLogger("backfill")


def _build_user_content(strategy: str, trajectory_summary: str) -> str:
    # str.replace is safe with curly braces in the payload (str.format is not);
    # the submit-response preview in summaries is literal JSON like
    # `{"task_id": "...", "exit_code": 1}` which would blow up .format().
    return (
        ADHERENCE_JUDGE_PROMPT
        .replace("{strategy}", strategy)
        .replace("{trajectory_summary}", trajectory_summary)
    )


async def _judge_one(
    client: openai.AsyncOpenAI,
    model: str,
    strategy: str,
    traj_path: str,
    max_traj_chars: int,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> tuple[float, str]:
    if not traj_path or not Path(traj_path).is_file():
        return _PARSE_FAILED_ADH, ""
    summary = summarize_trajectory(traj_path, max_chars=max_traj_chars)
    user_content = _build_user_content(strategy, summary)
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning(f"judge call failed: {e}")
            return _PARSE_FAILED_ADH, ""
    content = (resp.choices[0].message.content or "") if resp.choices else ""
    return _parse_reflection(content)


async def rejudge_records(
    records: list[dict],
    base_url: str,
    model: str,
    concurrency: int,
    max_traj_chars: int,
    max_tokens: int,
    api_key: str = "EMPTY",
    flush_fn=None,
) -> list[tuple[float, str]]:
    """Re-judge each record. If `flush_fn` is provided, it's called every
    100 completed records with the partial results list (so callers can
    persist progress mid-run).
    """
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    out: list[tuple[float, str] | None] = [None] * len(records)
    progress = {"done": 0}

    async def one(i: int, r: dict) -> None:
        res = await _judge_one(
            client, model, r.get("strategy", ""),
            r.get("trajectory_path", ""), max_traj_chars, max_tokens, sem,
        )
        out[i] = res
        progress["done"] += 1
        if progress["done"] % 50 == 0:
            logger.info(f"  judged {progress['done']}/{len(records)}")
        if flush_fn is not None and progress["done"] % 100 == 0:
            try:
                flush_fn(out)
            except Exception as e:
                logger.warning(f"flush_fn raised: {e}")

    await asyncio.gather(*[one(i, r) for i, r in enumerate(records)])
    try:
        await client.close()
    except Exception:
        pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path,
                   default=Path("/data/cybergym_data/cybergym-train-data/96e38ba3/archive.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--base-url", default="http://localhost:8001/v1")
    p.add_argument("--concurrency", type=int, default=16,
                   help="concurrent judge calls (keep low while GRPO is running)")
    p.add_argument("--max-traj-chars", type=int, default=16000)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--only-placeholder", action="store_true",
                   help="only re-judge records whose current insight is the placeholder/empty")
    p.add_argument("--update-adherence", action="store_true",
                   help="also replace the stored adherence score (default: keep original)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.archive.is_file():
        raise SystemExit(f"archive not found: {args.archive}")

    # Snapshot current archive (so we can diff new rows that appeared during backfill)
    ts = int(time.time())
    backup = args.archive.with_suffix(f".jsonl.bak.{ts}")
    shutil.copy(args.archive, backup)
    logger.info(f"backup: {backup}")

    with open(args.archive) as f:
        records = [json.loads(l) for l in f if l.strip()]
    snapshot_n = len(records)
    logger.info(f"snapshot: {snapshot_n} records")

    placeholder_markers = (
        "a short actionable takeaway",
        "your_actionable_takeaway_text",
        "your actionable takeaway",
    )

    def needs_rejudge(r: dict) -> bool:
        if not args.only_placeholder:
            return True
        ins = (r.get("insight") or "").strip().lower()
        if not ins:
            return True
        for m in placeholder_markers:
            if ins.startswith(m) and len(ins) < 60:
                return True
        return False

    target_idx = [i for i, r in enumerate(records) if needs_rejudge(r)]
    logger.info(f"records needing re-judge: {len(target_idx)}/{snapshot_n}")

    if not target_idx:
        logger.info("nothing to do")
        return

    target_records = [records[i] for i in target_idx]

    def flush_partial(partial: list):
        """Write archive.jsonl with whatever is done so far (NaN→keep old)."""
        import math as _math
        tmp = args.archive.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for idx, rec in enumerate(records):
                if idx in set(target_idx):
                    pos = target_idx.index(idx)
                    if pos < len(partial) and partial[pos] is not None:
                        adh, ins = partial[pos]
                        if ins and not (_math.isnan(adh) and not ins):
                            rec = dict(rec)
                            rec["insight"] = ins
                            if args.update_adherence and not _math.isnan(adh):
                                rec["adherence"] = adh
                f.write(json.dumps(rec) + "\n")
        tmp.replace(args.archive)
        n_done = sum(1 for p in partial if p is not None)
        logger.info(f"  flushed partial archive ({n_done}/{len(partial)} done)")

    t0 = time.monotonic()
    results = asyncio.run(rejudge_records(
        target_records, args.base_url, args.model,
        args.concurrency, args.max_traj_chars, args.max_tokens,
        flush_fn=flush_partial,
    ))
    logger.info(f"re-judged {len(results)} in {int(time.monotonic() - t0)}s")

    n_updated_ins = 0
    n_updated_adh = 0
    for i, (adh, ins) in zip(target_idx, results):
        if ins:
            records[i]["insight"] = ins
            n_updated_ins += 1
        if args.update_adherence and not math.isnan(adh):
            if records[i].get("adherence") != adh:
                records[i]["adherence"] = adh
                n_updated_adh += 1
    logger.info(f"insight updated on {n_updated_ins} records; "
                f"adherence updated on {n_updated_adh}")

    # Merge: re-read archive (it may have grown while we were running)
    with open(args.archive) as f:
        current = [json.loads(l) for l in f if l.strip()]
    if len(current) > snapshot_n:
        new_rows = current[snapshot_n:]
        logger.info(f"merging {len(new_rows)} rows that arrived during backfill")
        records = records + new_rows

    # Atomic write
    tmp = args.archive.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(args.archive)
    logger.info(f"wrote {len(records)} records to {args.archive}")


if __name__ == "__main__":
    main()
