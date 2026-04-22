#!/usr/bin/env python3
"""Compact 500-token version of the oracle-strategy extractor.

Uses the same trajectory summary and six-section skeleton as the v2
extractor, but swaps the prompt to force ~500 tokens total: each section
becomes a single sentence (or a short bullet) instead of a paragraph,
and exhaustive byte-level code dumps are replaced with abstract
descriptions. The goal is to test the "compression is intelligence"
hypothesis --- can the executor still reach oracle-level pass rate when
the strategy is ~6x shorter than v2?

Usage:
    uv run python3 sensitivity_study/extract_oracle_strategies_500.py \
        --concurrency 32 --target-tokens 500
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import openai

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from dual_loops.reward import summarize_trajectory as openhands_summarize  # noqa
from sensitivity_study.extract_oracle_strategies_v2 import (  # noqa
    summarize_claude_code_trajectory, build_summary,
)


EXTRACTION_PROMPT = """You are preparing a COMPACT REPLAY RECIPE (strict 500-token budget) from a successful CyberGym trajectory. The recipe will be injected into a new agent's prompt; the agent must reproduce the same crash in <=72 steps. Write tight, high-density prose --- no code dumps, no copy-paste blocks, no heredocs.

Produce exactly these SIX sections, each ONE or TWO sentences (~80 tokens). Total output budget: ~500 tokens.

## 1. Crash site
Name the file, function, and line where the crash fires, plus the crash type and sanitizer (from the trajectory's ASan/UBSan output).

## 2. Input structure
Describe the PoC file's byte layout abstractly: magic bytes (quote hex), header fields (list field names + their trigger values, not the full struct), and the key numeric trigger (e.g. "nesting=31", "chunk length=0x00000001"). Do NOT paste raw bytes or code.

## 3. Construction approach
Describe the method in one sentence --- e.g. "write the bytes with a python3 struct.pack one-liner" or "emit a heredoc to /workspace/poc.bin". Name the tool, not the code.

## 4. Submit validation
Quote the key sanitizer substring that confirms success (one short phrase, e.g. "stack-buffer-overflow at parseCR3:255").

## 5. Known pitfalls
Name the failed submissions' mistakes in one sentence (e.g. "don't write PNG magic 0x8a504e47; use MNG 0x8a4d4e47"). If there are no visible failed attempts, write "No failed attempts observed."

## 6. Budget guidance
"Expect to solve within N steps; first submit by step K." Use actual numbers from the trajectory.

RULES:
- STRICT 500-token budget. If you exceed it, cut section 2 first, then section 5.
- No code blocks, no triple-backtick fences, no heredocs, no struct.pack calls. References to them are fine ("use struct.pack with `>I`"), but no actual code.
- Ground every fact in the trajectory. If a section cannot be filled, say so in one line.
- Start your output DIRECTLY with "## 1. Crash site". No thinking preamble. No conclusion.

## Task: {task_id}

## Trajectory summary:
{summary}

## Output (start immediately with `## 1. Crash site`):"""


async def extract_one(client, task, sem, target_tokens, model):
    summary, fmt = build_summary(task)
    if fmt == "missing" or summary.startswith("("):
        return {"task_id": task["task_id"], "dir": task["dir"], "strategy": None,
                "error": f"no trajectory ({fmt})", "format": fmt}

    prompt = EXTRACTION_PROMPT.replace("{task_id}", task["task_id"]).replace("{summary}", summary)
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=target_tokens + 500,
                # Disable Qwen thinking mode so output starts with the actual
                # strategy and doesn't embed pre-answer "Thinking Process"
                # noise that interferes with section extraction.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        except Exception as e:
            return {"task_id": task["task_id"], "dir": task["dir"], "strategy": None,
                    "error": f"API: {e}", "format": fmt}

    if "</think>" in raw:
        cleaned = raw.rsplit("</think>", 1)[-1].lstrip()
    else:
        cleaned = raw
    idx = cleaned.find("## 1.")
    if idx > 0:
        cleaned = cleaned[idx:]

    return {
        "task_id": task["task_id"], "dir": task["dir"],
        "group": task.get("group"), "source": task.get("source"),
        "format": fmt, "strategy": cleaned, "chars": len(cleaned), "raw_chars": len(raw),
    }


async def main_async(args):
    tasks = json.load(open(args.tasks_file))
    if args.limit:
        tasks = tasks[:args.limit]
    client = openai.AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.monotonic()
    coros = [extract_one(client, t, sem, args.target_tokens, args.model) for t in tasks]
    out = [None] * len(coros)
    done = 0

    async def run(i, coro):
        nonlocal done
        res = await coro
        out[i] = res
        done += 1
        if done % 10 == 0 or done == len(coros):
            ok = sum(1 for x in out if x and x.get("strategy"))
            print(f"  {done}/{len(coros)} done  ({ok} with strategy)", flush=True)

    await asyncio.gather(*(run(i, c) for i, c in enumerate(coros)))
    try: await client.close()
    except Exception: pass

    ok = [r for r in out if r.get("strategy")]
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nSaved {len(out)} records to {args.output}")
    print(f"  with strategy: {len(ok)}")
    if ok:
        import statistics
        lens = [r["chars"] for r in ok]
        print(f"  strategy chars: median={statistics.median(lens):.0f}  min={min(lens)}  max={max(lens)}")
        tok_est = [l/3.3 for l in lens]
        print(f"  tokens (~÷3.3): median={statistics.median(tok_est):.0f}  min={min(tok_est):.0f}  max={max(tok_est):.0f}")
    print(f"  wall time: {int(time.monotonic()-t0)}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-file", default=str(SCRIPT_DIR / "tasks.json"))
    p.add_argument("--output", default=str(SCRIPT_DIR / "oracle_strategies_500.json"))
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--base-url", default="http://localhost:8001/v1")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--target-tokens", type=int, default=500)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
