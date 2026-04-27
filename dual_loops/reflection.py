"""Trajectory summarization and reflection-judge scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

import openai

from .prompts import ADHERENCE_JUDGE_PROMPT

if TYPE_CHECKING:
    from .executor import ExecutionResult

logger = logging.getLogger(__name__)

_ADH_RE = re.compile(r"<adherence>\s*(\d{1,2})\s*</adherence>", re.IGNORECASE)
_INS_RE = re.compile(r"<insight>\s*(.*?)\s*</insight>", re.IGNORECASE | re.DOTALL)
_ADH_SCALE_MAX = 10
_SUBMIT_PATH_RE = re.compile(r"submit\.sh\s+(\S+)")
_REDIRECT_TARGET_RE = re.compile(r"(?:>>?|\btee\b\s+(?:-a\s+)?)\s*([^\s;&|]+)")
_PARSE_FAILED_ADH = math.nan
_PARSE_FAILED_INSIGHT_SENTINEL = ""


async def _aretry(
    coro_factory,
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    label: str = "call",
):
    """Await a coroutine factory with exponential backoff."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if i == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** i))
            logger.warning(
                f"{label} attempt {i+1}/{attempts} failed: {e}; retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _extract_submit_targets(data: list) -> set[str]:
    targets: set[str] = set()
    for entry in data:
        if entry.get("source") != "agent" or entry.get("action") != "run":
            continue
        cmd = str((entry.get("args") or {}).get("command") or "")
        if "submit.sh" not in cmd or cmd.lstrip().startswith("cat"):
            continue
        for match in _SUBMIT_PATH_RE.finditer(cmd):
            token = match.group(1).strip(" ;&|")
            if token and "submit.sh" not in token:
                targets.add(token)
    return targets


def _cmd_writes_to_poc(cmd: str, submit_targets: set[str]) -> bool:
    if "submit.sh" in cmd:
        return False
    redirects = _REDIRECT_TARGET_RE.findall(cmd)
    if not redirects:
        return False
    redirect_targets = [target.lower() for target in redirects]
    if any(target in submit_targets for target in redirects):
        return True
    if any("poc" in target for target in redirect_targets):
        return True
    if submit_targets and any(target in cmd for target in submit_targets):
        return True
    return False


def summarize_trajectory(traj_path: Path | str, max_chars: int = 16000) -> str:
    """Compress an OpenHands trajectory for the reflection judge."""
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return "(trajectory unavailable)"

    submit_targets = _extract_submit_targets(data)
    assistant_msgs: list[str] = []
    submits: list[str] = []
    reads: list[str] = []
    edits: list[str] = []
    thinks: list[str] = []
    poc_artifacts: list[str] = []

    for i, entry in enumerate(data):
        action = entry.get("action")
        source = entry.get("source")
        args = entry.get("args") or {}

        if source == "agent" and action == "message":
            message = str(entry.get("message") or "").strip()
            if message:
                assistant_msgs.append(message[:500])

        if source == "agent" and action == "run":
            cmd = str(args.get("command") or "")
            if "submit.sh" in cmd and not cmd.lstrip().startswith("cat"):
                resp_preview = ""
                if i + 1 < len(data):
                    resp_preview = str(data[i + 1].get("content") or "")[:1500]
                submits.append(f"$ {cmd[:120]}\n  → {resp_preview}")
            elif _cmd_writes_to_poc(cmd, submit_targets):
                poc_artifacts.append(f"$ {cmd[:800]}")

        if source == "agent" and action == "run_ipython":
            code = str(args.get("code") or "")
            mentions_poc = (
                any(target in code for target in submit_targets)
                or "poc" in code.lower()
                or "submit.sh" in code
            )
            if mentions_poc:
                snippet = code[:800]
                obs_preview = ""
                if i + 1 < len(data):
                    obs_preview = str(data[i + 1].get("content") or "")[:400]
                block = f"$ python3 <<EOF\n{snippet}\nEOF"
                if obs_preview:
                    block += f"\n  → {obs_preview}"
                poc_artifacts.append(block)

        if source == "agent" and action == "read":
            path = str(args.get("path") or "")
            if path:
                reads.append(path)

        if source == "agent" and action in ("edit", "write"):
            path = str(args.get("path") or "")
            if path:
                edits.append(path)
            if path and (path in submit_targets or "poc" in path.lower()):
                content = args.get("content") or args.get("file_text") or ""
                if content:
                    poc_artifacts.append(f"{path}:\n  {repr(content[:400])}")

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
    if poc_artifacts:
        sections.append(
            "### PoC construction (what bytes the agent fed the fuzzer)\n"
            + "\n\n".join(poc_artifacts[:5])
        )
    if submits:
        sections.append(
            "### Submit attempts (" + str(len(submits)) + ")\n" + "\n\n".join(submits[:10])
        )
    if reads:
        sections.append("### Files read\n" + "\n".join(list(dict.fromkeys(reads))[:15]))
    if edits:
        sections.append("### Files edited/written\n" + "\n".join(list(dict.fromkeys(edits))[:10]))
    if thinks:
        sections.append("### Agent thinking\n" + "\n---\n".join(thinks[:5]))
    if not sections:
        return "(trajectory has no agent actions)"

    summary = "\n\n".join(sections)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n\n[...truncated]"
    return summary


def _build_adherence_user_content(
    strategy: str,
    trajectory_summary: str,
    insight_max_tokens: int = 500,
) -> str:
    return (
        ADHERENCE_JUDGE_PROMPT
        .replace("{strategy}", strategy)
        .replace("{trajectory_summary}", trajectory_summary)
        .replace("{insight_max_tokens}", str(insight_max_tokens))
    )


def _truncate_insight(text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return text
    cap_chars = max_tokens * 4
    if len(text) <= cap_chars:
        return text
    cut = text.rfind(" ", 0, cap_chars)
    if cut < cap_chars * 3 // 4:
        cut = cap_chars
    return text[:cut].rstrip() + " [...truncated]"


def _parse_reflection(text: str) -> tuple[float, str]:
    """Parse <adherence> and <insight> tags from judge output."""
    if not text:
        return _PARSE_FAILED_ADH, _PARSE_FAILED_INSIGHT_SENTINEL
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    adh_match = _ADH_RE.search(text)
    insight_match = _INS_RE.search(text)
    if adh_match is None or insight_match is None:
        return _PARSE_FAILED_ADH, _PARSE_FAILED_INSIGHT_SENTINEL
    try:
        score = int(adh_match.group(1))
    except ValueError:
        return _PARSE_FAILED_ADH, _PARSE_FAILED_INSIGHT_SENTINEL
    score = max(0, min(_ADH_SCALE_MAX, score))
    insight = insight_match.group(1).strip()

    placeholder_markers = (
        "a short actionable takeaway",
        "your_actionable_takeaway_text",
        "your actionable takeaway",
    )
    prompt_echo_markers = (
        "1-3 sentences actionable takeaway",
        "no preamble",
        "adherence rubric",
        "strict adherence",
        "derived from the trajectory",
        "your own content",
        "replace integer_0_to_10",
        "replace your_actionable",
        "no extra text",
        "no forbidden phrases",
        "insight must name",
        "evaluate adherence",
        "**adherence",
        "**insight",
        "insight:** ",
        "adherence:** ",
        "## adherence",
        "## insight",
    )
    low = insight.lower()
    is_placeholder = not insight or low in placeholder_markers or any(
        low.startswith(marker) and len(insight) < 60 for marker in placeholder_markers
    )
    is_prompt_echo = any(marker in low for marker in prompt_echo_markers)
    is_analysis_dump = (
        insight.startswith(("**", "## ", "1.", "1)", "- ", "* ", "`."))
        or "analyze the" in low[:80]
    )
    if is_placeholder or is_prompt_echo or is_analysis_dump:
        insight = ""
    return score / _ADH_SCALE_MAX, insight


async def _judge_one(
    client: openai.AsyncOpenAI,
    model: str,
    strategy: str,
    trajectory_summary: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    insight_max_tokens: int = 500,
) -> tuple[float, str]:
    user_content = _build_adherence_user_content(
        strategy,
        trajectory_summary,
        insight_max_tokens=insight_max_tokens,
    )
    async with semaphore:
        try:
            response = await _aretry(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                ),
                attempts=3,
                base_delay=2.0,
                label="reflection-judge",
            )
        except Exception as e:
            logger.warning(f"reflection judge call failed after retries: {e}")
            return _PARSE_FAILED_ADH, _PARSE_FAILED_INSIGHT_SENTINEL
    content = (response.choices[0].message.content or "") if response.choices else ""
    adherence, insight = _parse_reflection(content)
    return adherence, _truncate_insight(insight, insight_max_tokens)


async def score_reflection_batch(
    results: list["ExecutionResult"],
    base_url: str,
    model: str,
    concurrency: int = 64,
    max_traj_chars: int = 16000,
    max_tokens: int = 8192,
    api_key: str = "EMPTY",
    insight_max_tokens: int = 500,
) -> list[tuple[float, str]]:
    """Return (adherence, insight) pairs for each execution result."""
    if not results:
        return []
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    async def _one(result: "ExecutionResult") -> tuple[float, str]:
        if result.trajectory_path is None:
            return _PARSE_FAILED_ADH, _PARSE_FAILED_INSIGHT_SENTINEL
        summary = summarize_trajectory(result.trajectory_path, max_chars=max_traj_chars)
        return await _judge_one(
            client,
            model,
            result.strategy.strategy,
            summary,
            sem,
            max_tokens,
            insight_max_tokens=insight_max_tokens,
        )

    out = await asyncio.gather(*[_one(result) for result in results])
    try:
        await client.close()
    except Exception:
        pass

    n_with_trajectory = sum(1 for result in results if result.trajectory_path is not None)
    valid = [adherence for adherence, _ in out if not math.isnan(adherence)]
    n_fallback = sum(1 for adherence, _ in out if math.isnan(adherence))
    if n_with_trajectory > 0 and not valid:
        raise RuntimeError(
            "reflection judge produced no parseable adherence scores for any "
            "trajectory; aborting instead of training on a broken reward signal"
        )
    imputed = sum(valid) / len(valid) if valid else 1.0

    patched: list[tuple[float, str]] = []
    for adherence, insight in out:
        if math.isnan(adherence):
            patched.append((imputed, insight))
        else:
            patched.append((adherence, insight))

    if out and n_fallback / len(out) > 0.25:
        logger.warning(
            f"reflection judge fell back on {n_fallback}/{len(out)} rollouts "
            f"({100*n_fallback/len(out):.0f}%) — imputing with batch mean "
            f"{imputed:.3f}; adherence signal for this round is unreliable."
        )
    elif n_fallback:
        logger.info(
            f"reflection judge fell back on {n_fallback}/{len(out)} rollouts; "
            f"imputed with batch mean {imputed:.3f}."
        )
    return patched


__all__ = ["score_reflection_batch", "summarize_trajectory"]
