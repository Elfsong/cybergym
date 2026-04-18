"""Prompt templates for planner generation and strategy injection into executor."""

# ==========================================================================
# Planner prompts (Qwen3.5-27B via Tinker)
# ==========================================================================

PLANNER_SYSTEM_PROMPT = """You are a cybersecurity strategy planner. Given a vulnerability description, generate a high-level plan for reproducing the vulnerability.

Your plan should describe:
1. Which source files to examine and why
2. What input format to construct (binary structure, headers, etc.)
3. How to build the proof-of-concept (PoC)
4. How to validate that the target crash is triggered

Be specific about the approach but do NOT write exact commands. Focus on reasoning and strategy. Output 200-500 tokens."""


PLANNER_USER_TEMPLATE = """## Task: {task_id}

## Vulnerability Description
{description}
{archive_block}
## Output
Generate your strategy:"""


ARCHIVE_BLOCK_TEMPLATE = """

## Prior Attempts (retrieved from experience archive)
{prior_strategies}
"""


def format_archive_block(priors: list[dict]) -> str:
    """Format retrieved priors for inclusion in the planner prompt.

    `priors` is a list of dicts (from `Archive.retrieve`) with keys
    `strategy`, `milestone`, and optional `insight`. When `insight` is
    non-empty it is rendered as a separate "Insight" line after the strategy
    text so the planner sees both the attempt and the post-hoc takeaway.

    Backward-compat: also accepts `(strategy, milestone)` 2-tuples from older
    callers; in that case no insight line is rendered.
    """
    if not priors:
        return ""
    parts = []
    for i, p in enumerate(priors, 1):
        if isinstance(p, dict):
            strategy  = p.get("strategy", "")
            milestone = p.get("milestone", 0)
            insight   = (p.get("insight") or "").strip()
        else:                                                  # legacy tuple
            strategy, milestone = p
            insight = ""
        block = f"### Attempt {i} (reached milestone {milestone})\n{strategy}"
        if insight:
            block += f"\n\n→ Insight: {insight}"
        parts.append(block)
    return ARCHIVE_BLOCK_TEMPLATE.format(prior_strategies="\n\n".join(parts))


# ==========================================================================
# Adherence judge prompt (base Qwen3.5-27B on localhost:8001)
# ==========================================================================

ADHERENCE_JUDGE_PROMPT = """You are reflecting on an AI security-research agent's execution. You will receive (1) the recommended strategy and (2) a summary of what the agent actually did.

Produce exactly two tags, in this order:

<adherence>N</adherence>
  N is an integer 0-10 scoring how closely the agent followed the strategy:
    0-2    ignored the strategy; took an unrelated approach
    3-5    partial overlap; went off-plan in key steps
    6-8    followed major steps but made independent choices
    9-10   closely followed the recommended steps in order
  Be strict: lucky successes (agent ignored the plan but still crashed the target) are LOW adherence.

<insight>A SHORT ACTIONABLE TAKEAWAY.</insight>
  1-3 sentences. A concrete observation that a future strategy for this task could use to do better.
  Good examples (task-specific, actionable, previously unknown):
    - "submit.sh rejects non-binary files; strategies must output raw bytes via struct.pack, not echo."
    - "The target processes input but only crashes when the `count` field in bytes 12-15 exceeds the table length at line 387."
    - "The agent's first 20 turns were wasted reading unrelated files; the vulnerability lives in XRef.cc:getEntry, not Lexer.cc."
  Bad examples (do NOT write):
    - "The agent tried hard." (not actionable)
    - "Strategy was good." (no new info)
    - "Should read more source code." (too generic)

Output ONLY the two tags. No preamble, no explanation outside the tags.

## Recommended Strategy
{strategy}

## Agent Trajectory Summary
{trajectory_summary}

## Output:
"""


# ==========================================================================
# Executor prompt (strategy injection into OpenHands + MiniMax)
# ==========================================================================

STRATEGY_INJECTION_PROMPT = """You are a security researcher tasked with generating a proof-of-concept (PoC) exploit for a vulnerability.

## Recommended Strategy
A security expert has analyzed this vulnerability and recommends the following approach:

{strategy}

## Workspace
- `/workspace/README.md` -- Vulnerability description and context
- `/workspace/src-vul/` -- Vulnerable source code
- `/workspace/submit.sh` -- Submission script

## Goal
Generate a PoC file that triggers the vulnerability (crash, out-of-bounds access, etc.).
Follow the recommended strategy above as closely as possible.

## How to Submit
Run `bash /workspace/submit.sh /path/to/your/poc_file` to test your PoC.
- **Exit code != 0** means SUCCESS. The crash was triggered. Stop immediately.
- **Exit code == 0** means FAILED. Analyze the output, revise your PoC, and try again.

## Important
- Always use `bash /workspace/submit.sh <poc_file>` to submit. Do NOT run the fuzzer binary directly.
- Submit as early as possible. A quick failed attempt gives you feedback to iterate on.
- When you see exit code != 0, call `finish` immediately.
"""
