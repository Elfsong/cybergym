"""Shared utilities for the policy loop."""

from __future__ import annotations

import gzip
import json
import logging
import tarfile
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_tasks_file(path: Path) -> list[str]:
    """Read task IDs from a TASKS file (one per line, # comments/blanks skipped)."""
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tasks.append(line)
    return tasks


def get_task_description(task_id: str, data_dir: Path) -> str:
    """Fetch the vulnerability description for a CyberGym task.

    For Level 1, description.txt is provided alongside the tarball. We read
    it directly from the benchmark data directory.
    """
    kind, tid = task_id.split(":", 1)
    if kind == "arvo":
        task_root = data_dir / "arvo" / tid
    elif kind == "oss-fuzz":
        task_root = data_dir / "oss-fuzz" / tid
    elif kind == "oss-fuzz-latest":
        task_root = data_dir / "oss-fuzz-latest" / tid
    else:
        return ""

    desc_path = task_root / "description.txt"
    if desc_path.exists():
        return desc_path.read_text(errors="ignore").strip()
    return f"(no description available for {task_id})"


def setup_logging(log_path: Path, level: int = logging.INFO) -> None:
    """Configure root logger with stream + file handlers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path),
    ]
    for h in handlers:
        h.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    # Clear previous handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)


def save_json(obj, path: Path, indent: int = 2) -> None:
    """Dump object to JSON (handles dataclasses via asdict)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import is_dataclass, asdict

    def _serialize(x):
        if is_dataclass(x):
            return asdict(x)
        if isinstance(x, Path):
            return str(x)
        return str(x)

    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_serialize)


def save_jsonl(records: list, path: Path) -> None:
    """Append-friendly: write each record on its own line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str))
            f.write("\n")
