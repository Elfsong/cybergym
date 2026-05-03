"""Retry launcher for the 7 milestone=0 tasks from the main run.

Uses parallel=4 (instead of 8) to lighten docker daemon contention that
caused those tasks to lose their sandbox runtime in the bursty first batch
of the main run, and a fresh state DB so the state machine treats them as
uncovered. Output goes to the same phase4_qwen36_maxpreview/ tree so the
repair script can rescore them alongside the others.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CYBERGYM_ROOT = Path(__file__).parent


def _load_dotenv(p: Path) -> None:
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")


def main() -> None:
    _load_dotenv(CYBERGYM_ROOT / ".env")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY missing", file=sys.stderr)
        sys.exit(2)
    # OpenHands' get_api_key uses LLM_API_KEY for non-OPENAI/Anthropic/Gemini
    # prefixes; mirror DASHSCOPE_API_KEY into both env vars.
    os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
    os.environ["LLM_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]

    cmd = [
        "uv", "run", "python",
        str(CYBERGYM_ROOT / "baselines/pagent/runner/orchestrator.py"),
        "--phase", "4",
        "--tasks", str(CYBERGYM_ROOT / "TASKS_EVAL_retry7"),
        "--parallel", "4",
        "--model", "qwen3.6-max-preview",
        "--base-url", "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "--out-root", "/lp-dev/cybergym_data/pagent/phase4_qwen36_maxpreview",
        "--state-db", "/lp-dev/cybergym_data/pagent/state_qwen36_maxpreview_retry.sqlite",
    ]
    cmd.extend(sys.argv[1:])
    os.chdir(CYBERGYM_ROOT)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
