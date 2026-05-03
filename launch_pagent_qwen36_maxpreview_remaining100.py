"""Launch PAGENT phase-4 on the remaining 100 tasks of the 200-task held-out set.

After this run completes, PAGENT + Qwen3.6-Max-Preview will have full 200-task
coverage, directly comparable to OpenHands + Qwen3.6-Max-Preview's 200-task row
in tab:main. Output goes to the same /lp-dev/cybergym_data/pagent/phase4_qwen36_maxpreview/
tree (no task_id collision: this set is the disjoint complement of the 100
already evaluated under seed-42).
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
    os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
    os.environ["LLM_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]

    cmd = [
        "uv", "run", "python",
        str(CYBERGYM_ROOT / "baselines/pagent/runner/orchestrator.py"),
        "--phase", "4",
        "--tasks", str(CYBERGYM_ROOT / "TASKS_EVAL_remaining100"),
        "--parallel", "8",
        "--model", "qwen3.6-max-preview",
        "--base-url", "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "--out-root", "/lp-dev/cybergym_data/pagent/phase4_qwen36_maxpreview",
        "--state-db", "/lp-dev/cybergym_data/pagent/state_qwen36_maxpreview_remaining100.sqlite",
    ]
    cmd.extend(sys.argv[1:])
    os.chdir(CYBERGYM_ROOT)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
