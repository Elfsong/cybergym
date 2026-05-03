"""Launch PAGENT phase-4 orchestrator for qwen3.6-max-preview against DashScope.

Loads .env through the same parser as qwen_api_demo.py (bash `source .env`
turned out to mis-parse this file's format and pass an empty/wrong key to the
OpenAI client, producing 401 from Aliyun). After loading, mirrors
DASHSCOPE_API_KEY -> OPENAI_API_KEY so OpenHands' OpenAI-compat client can
authenticate, then execvp's the orchestrator with the inherited environment.
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
        print("ERROR: DASHSCOPE_API_KEY missing after loading .env", file=sys.stderr)
        sys.exit(2)
    # OpenHands' run.py routes the env-var lookup by model-prefix:
    # "qwen3.6-..." doesn't match any of OPENAI_PREFIXES (gpt-/o3/o4),
    # ANTHROPIC_PREFIXES, or GEMINI_PREFIXES, so it falls through to
    # LLM_API_KEY (and renders "EMPTY" if unset, producing a 401 from
    # DashScope). Mirror the key into both OPENAI_API_KEY and LLM_API_KEY
    # so either lookup path resolves.
    os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
    os.environ["LLM_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]

    cmd = [
        "uv", "run", "python",
        str(CYBERGYM_ROOT / "baselines/pagent/runner/orchestrator.py"),
        "--phase", "4",
        "--tasks", str(CYBERGYM_ROOT / "TASKS_EVAL_50_seed42"),
        "--parallel", "8",
        "--model", "qwen3.6-max-preview",
        "--base-url", "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "--out-root", "/lp-dev/cybergym_data/pagent/phase4_qwen36_maxpreview",
        "--state-db", "/lp-dev/cybergym_data/pagent/state_qwen36_maxpreview.sqlite",
    ]
    # Pass through extra CLI args (e.g. --retry-failed, --dry-run)
    cmd.extend(sys.argv[1:])
    os.chdir(CYBERGYM_ROOT)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
