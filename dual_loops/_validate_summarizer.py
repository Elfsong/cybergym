"""Ad-hoc A/B test: old vs new summarizer, then judge both to compare insights.

Not committed. Run after Qwen3.5-27B is serving on localhost:8001.

Usage:
    python -m dual_loops._validate_summarizer \
        --archive /data/cybergym_data/cybergym-train-data/96e38ba3/archive.jsonl \
        --n 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import openai

sys.path.insert(0, "/home/nvidia/Projects/cybergym")

from dual_loops.prompts import ADHERENCE_JUDGE_PROMPT
from dual_loops.reward import summarize_trajectory as new_summarize


# Old summarizer (pre-change): no PoC section, 200-char submit preview, 8000 char budget.
def old_summarize(traj_path, max_chars: int = 8000) -> str:
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except Exception:
        return "(trajectory unavailable)"

    assistant_msgs, submits, reads, edits, thinks = [], [], [], [], []
    for i, e in enumerate(data):
        action = e.get("action"); source = e.get("source"); args = e.get("args") or {}
        if source == "agent" and action == "message":
            msg = str(e.get("message") or "").strip()
            if msg:
                assistant_msgs.append(msg[:500])
        if source == "agent" and action == "run":
            cmd = str(args.get("command") or "")
            if "submit.sh" in cmd and "cat" not in cmd:
                resp = ""
                if i + 1 < len(data):
                    resp = str(data[i + 1].get("content") or "")[:200]
                submits.append(f"$ {cmd[:120]}\n  → {resp}")
        if source == "agent" and action == "read":
            p = str(args.get("path") or "")
            if p: reads.append(p)
        if source == "agent" and action == "edit":
            p = str(args.get("path") or "")
            if p: edits.append(p)
        if source == "agent" and action == "think":
            t = str(args.get("thought") or "").strip()
            if t: thinks.append(t[:300])

    secs = []
    if assistant_msgs:
        first = assistant_msgs[:3]
        last = assistant_msgs[-3:] if len(assistant_msgs) > 3 else []
        secs.append("### First assistant messages\n" + "\n---\n".join(first))
        if last and last != first:
            secs.append("### Last assistant messages\n" + "\n---\n".join(last))
    if submits:
        secs.append("### Submit attempts (" + str(len(submits)) + ")\n" + "\n\n".join(submits[:10]))
    if reads:
        uniq = list(dict.fromkeys(reads))[:15]
        secs.append("### Files read\n" + "\n".join(uniq))
    if edits:
        uniq = list(dict.fromkeys(edits))[:10]
        secs.append("### Files edited/written\n" + "\n".join(uniq))
    if thinks:
        secs.append("### Agent thinking\n" + "\n---\n".join(thinks[:5]))
    if not secs:
        return "(trajectory has no agent actions)"
    s = "\n\n".join(secs)
    return s[:max_chars] + ("\n\n[...truncated]" if len(s) > max_chars else "")


async def judge(client, strategy: str, summary: str) -> str:
    content = (
        ADHERENCE_JUDGE_PROMPT
        .replace("{strategy}", strategy)
        .replace("{trajectory_summary}", summary)
    )
    try:
        resp = await client.chat.completions.create(
            model="Qwen/Qwen3.5-27B",
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=8192,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"(judge error: {e})"


def parse_judge(text: str) -> tuple[str, str]:
    import re
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    adh_m = re.search(r"<adherence>\s*(\d{1,2})\s*</adherence>", text, re.I)
    ins_m = re.search(r"<insight>\s*(.*?)\s*</insight>", text, re.I | re.S)
    return (adh_m.group(1) if adh_m else "?"), (ins_m.group(1) if ins_m else "(no insight tag)")


async def main_async(args):
    recs = []
    with open(args.archive) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("trajectory_path") and Path(r["trajectory_path"]).exists():
                recs.append(r)
    print(f"Loaded {len(recs)} records with existing trajectories", file=sys.stderr)
    rng = random.Random(args.seed)
    # Stratify: half milestone >= 6, half < 6
    hi = [r for r in recs if (r.get("milestone") or 0) >= 6]
    lo = [r for r in recs if (r.get("milestone") or 0) < 6]
    rng.shuffle(hi); rng.shuffle(lo)
    n = args.n
    pick = hi[: n // 2] + lo[: n - n // 2]
    rng.shuffle(pick)

    client = openai.AsyncOpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")

    rows = []
    for r in pick:
        tp = r["trajectory_path"]
        old_s = old_summarize(tp, max_chars=8000)
        new_s = new_summarize(tp, max_chars=16000)
        has_poc_section = "### PoC construction" in new_s
        rows.append((r, old_s, new_s, has_poc_section))

    # Judge both in parallel
    coros = []
    for r, old_s, new_s, _ in rows:
        coros.append(judge(client, r["strategy"], old_s))
        coros.append(judge(client, r["strategy"], new_s))
    results = await asyncio.gather(*coros)

    for i, (r, old_s, new_s, has_poc) in enumerate(rows):
        old_raw = results[2 * i]
        new_raw = results[2 * i + 1]
        old_adh, old_ins = parse_judge(old_raw)
        new_adh, new_ins = parse_judge(new_raw)
        print("=" * 100)
        print(f"[{i}] task={r['task_id']} milestone={r.get('milestone')} "
              f"round={r.get('round')} group={r.get('group_id')}")
        print(f"  old_len={len(old_s)} new_len={len(new_s)} has_poc_section={has_poc}")
        print(f"  OLD adherence={old_adh}")
        print(f"    insight: {old_ins[:500]}")
        print(f"  NEW adherence={new_adh}")
        print(f"    insight: {new_ins[:500]}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path,
                   default=Path("/data/cybergym_data/cybergym-train-data/96e38ba3/archive.jsonl"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    asyncio.run(main_async(args))
