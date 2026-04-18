"""Adherence judge — scores how closely an executor trajectory followed a strategy.

Judge is the base Qwen3.5-27B served by the existing executor vLLM
(localhost:8001 by default). A base (non-LoRA) model is required: self-judging
with the Tinker-trained LoRA would poison the GRPO reward with self-reinforcement
bias and non-stationary weights.

Runs during the scoring phase, after all executor rollouts complete. vLLM is
then idle, so judge calls don't contend with executor throughput.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import openai

if TYPE_CHECKING:
    from policy_loop.executor import ExecutionResult

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "adherence.txt"
_JUDGE_PROMPT = _PROMPT_PATH.read_text()
_ADH_RE = re.compile(r"<adherence>\s*(\d{1,2})\s*</adherence>", re.IGNORECASE)
_INS_RE = re.compile(r"<insight>\s*(.*?)\s*</insight>", re.IGNORECASE | re.DOTALL)
_SCALE_MAX = 10


# ==========================================================================
# Trajectory summarization
# ==========================================================================

def summarize_trajectory(traj_path: Path | str, max_chars: int = 8000) -> str:
    """Compress an OpenHands trajectory to ≤ max_chars for the judge.

    Extracts:
      - First 3 and last 3 assistant messages
      - All submit.sh invocations + server response previews
      - File reads / edits (paths only)
      - Agent `think` actions (thought content, truncated)
    """
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return "(trajectory unavailable)"

    assistant_msgs: list[str] = []
    submits: list[str] = []
    reads: list[str] = []
    edits: list[str] = []
    thinks: list[str] = []

    for i, e in enumerate(data):
        action = e.get("action")
        source = e.get("source")
        args = e.get("args") or {}

        if source == "agent" and action == "message":
            msg = str(e.get("message") or "").strip()
            if msg:
                assistant_msgs.append(msg[:500])

        if source == "agent" and action == "run":
            cmd = str(args.get("command") or "")
            if "submit.sh" in cmd and "cat" not in cmd:
                resp_preview = ""
                if i + 1 < len(data):
                    resp_preview = str(data[i + 1].get("content") or "")[:200]
                submits.append(f"$ {cmd[:120]}\n  → {resp_preview}")

        if source == "agent" and action == "read":
            path = str(args.get("path") or "")
            if path:
                reads.append(path)

        if source == "agent" and action == "edit":
            path = str(args.get("path") or "")
            if path:
                edits.append(path)

        if source == "agent" and action == "think":
            thought = str(args.get("thought") or "").strip()
            if thought:
                thinks.append(thought[:300])

    sections: list[str] = []

    if assistant_msgs:
        first = assistant_msgs[:3]
        last = assistant_msgs[-3:] if len(assistant_msgs) > 3 else []
        sections.append("### First assistant messages\n" + "\n---\n".join(first))
        if last and last != first:
            sections.append("### Last assistant messages\n" + "\n---\n".join(last))

    if submits:
        sections.append("### Submit attempts (" + str(len(submits)) + ")\n"
                        + "\n\n".join(submits[:10]))

    if reads:
        uniq_reads = list(dict.fromkeys(reads))[:15]
        sections.append("### Files read\n" + "\n".join(uniq_reads))

    if edits:
        uniq_edits = list(dict.fromkeys(edits))[:10]
        sections.append("### Files edited/written\n" + "\n".join(uniq_edits))

    if thinks:
        sections.append("### Agent thinking\n" + "\n---\n".join(thinks[:5]))

    if not sections:
        return "(trajectory has no agent actions)"

    summary = "\n\n".join(sections)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n\n[...truncated]"
    return summary


# ==========================================================================
# Judge inference
# ==========================================================================

def _build_user_content(strategy: str, trajectory_summary: str) -> str:
    # str.replace is safe with curly braces in the payload (str.format is not).
    return (
        _JUDGE_PROMPT
        .replace("{strategy}", strategy)
        .replace("{trajectory_summary}", trajectory_summary)
    )


def _parse_reflection(text: str) -> tuple[float, str]:
    """Parse <adherence>N</adherence> and <insight>TEXT</insight> from judge output.

    Returns (adherence in [0,1], insight_text). On parse failure, returns (0.0, "").
    Both tags are required to be present; missing either → fallback values for both.
    """
    if not text:
        return 0.0, ""
    adh_m = _ADH_RE.search(text)
    ins_m = _INS_RE.search(text)
    if adh_m is None or ins_m is None:
        return 0.0, ""
    try:
        score = int(adh_m.group(1))
    except ValueError:
        return 0.0, ""
    score = max(0, min(_SCALE_MAX, score))
    insight = ins_m.group(1).strip()
    return score / _SCALE_MAX, insight


# Kept for backward compat with any existing caller expecting the scalar parser.
def _parse_score(text: str) -> float:
    adh, _ = _parse_reflection(text)
    return adh


async def _judge_one(
    client: openai.AsyncOpenAI,
    model: str,
    strategy: str,
    trajectory_summary: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> tuple[float, str]:
    user_content = _build_user_content(strategy, trajectory_summary)
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning(f"reflection judge call failed: {e}")
            return 0.0, ""
    content = (resp.choices[0].message.content or "") if resp.choices else ""
    return _parse_reflection(content)


async def score_reflection_batch(
    results: "list[ExecutionResult]",
    base_url: str,
    model: str,
    concurrency: int = 64,
    max_traj_chars: int = 8000,
    max_tokens: int = 8192,
    api_key: str = "EMPTY",
) -> list[tuple[float, str]]:
    """For each ExecutionResult, return (adherence in [0,1], insight_text).
    Order matches input. Rollouts with no trajectory → (0.0, "").

    The judge emits two XML tags: <adherence>N</adherence> and <insight>...</insight>.
    Parse failures fall back to (0.0, ""); no retries.
    """
    if not results:
        return []
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    async def _one(r) -> tuple[float, str]:
        if r.trajectory_path is None:
            return 0.0, ""
        summary = summarize_trajectory(r.trajectory_path, max_chars=max_traj_chars)
        return await _judge_one(client, model, r.strategy.strategy, summary, sem, max_tokens)

    out = await asyncio.gather(*[_one(r) for r in results])
    try:
        await client.close()
    except Exception:
        pass
    return out


# Back-compat shim — callers that only want adherence can still use this.
async def score_adherence_batch(
    results: "list[ExecutionResult]",
    base_url: str,
    model: str,
    concurrency: int = 64,
    max_traj_chars: int = 8000,
    max_tokens: int = 8192,
    api_key: str = "EMPTY",
) -> list[float]:
    pairs = await score_reflection_batch(
        results, base_url, model, concurrency, max_traj_chars, max_tokens, api_key,
    )
    return [adh for adh, _ in pairs]
