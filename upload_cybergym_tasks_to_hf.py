"""Upload TASKS_FULL, TASKS_TRAIN, TASKS_EVAL to Hugging Face Datasets as 3
configs (subsets) of Elfsong/cybergym-tasks.

Schema (Schema B):
    task_id : string  e.g. "arvo:11504"
    kind    : string  "arvo" or "oss-fuzz"
    id      : int32   numeric task number
    project : string  "libxml2" / "ffmpeg" / ...   from db.json

Layout on hub:
    Elfsong/cybergym-tasks/
        README.md
        full/data-*.parquet   (1507 rows, split=train)
        train/data-*.parquet  (300 rows,  split=train)
        eval/data-*.parquet   (200 rows,  split=test)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CYBERGYM = Path("/home/nvidia/Projects/cybergym")
DB_PATH = Path("/lp-dev/cybergym_data/pagent/db/db.json")
REPO_ID = "Elfsong/cybergym-tasks"

# (config_name, source_filename, split_name)
CONFIGS = [
    ("full",  "TASKS_FULL",  "train"),
    ("train", "TASKS_TRAIN", "train"),
    ("eval",  "TASKS_EVAL",  "test"),
]


def load_dotenv(p: Path) -> None:
    if not p.exists(): return
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s: continue
        k, _, v = s.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")


def build_records(path: Path, task_to_proj: dict[str, str]) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        tid = line.strip()
        if not tid or tid.startswith("#"):
            continue
        kind, num = tid.split(":", 1)
        rows.append({
            "task_id": tid,
            "kind":    kind,
            "id":      int(num),
            "project": task_to_proj.get(tid, ""),
        })
    return rows


README_TEMPLATE = """\
---
license: cc-by-4.0
language:
  - en
tags:
  - cybersecurity
  - vulnerability-reproduction
  - cybergym
  - benchmarks
size_categories:
  - 1K<n<10K
configs:
  - config_name: full
    data_files:
      - split: train
        path: full/*.parquet
  - config_name: train
    data_files:
      - split: train
        path: train/*.parquet
  - config_name: eval
    data_files:
      - split: test
        path: eval/*.parquet
dataset_info:
  - config_name: full
    splits:
      - name: train
        num_examples: 1507
  - config_name: train
    splits:
      - name: train
        num_examples: 300
  - config_name: eval
    splits:
      - name: test
        num_examples: 200
---

# CyberGym task ID splits

Task ID splits used in our work on CyberGym vulnerability-reproduction
benchmarks. Each row is a single task identifier (no inputs / no outputs);
this dataset is intended as a *task-list manifest* for downstream evaluation
scripts that fetch the actual task workspaces from the CyberGym distribution.

## Configs

| Config  | Split   | Rows  | Description                                  |
|---------|---------|------:|----------------------------------------------|
| `full`  | `train` | 1507  | Every task in the CyberGym Level-1 release   |
| `train` | `train` | 300   | Training pool used in our experiments        |
| `eval`  | `test`  | 200   | Held-out evaluation pool                     |

`train` and `eval` are deterministic (project-stratified) subsets of `full`.

## Schema

```python
{
  "task_id": "arvo:11504",   # canonical CyberGym ID
  "kind":    "arvo",         # "arvo" or "oss-fuzz"
  "id":      11504,          # numeric portion
  "project": "libxml2",      # project name from CyberGym's db.json
}
```

## Usage

```python
from datasets import load_dataset

eval_ids  = load_dataset("Elfsong/cybergym-tasks", "eval")["test"]["task_id"]
train_ids = load_dataset("Elfsong/cybergym-tasks", "train")["train"]["task_id"]
full_ids  = load_dataset("Elfsong/cybergym-tasks", "full")["train"]["task_id"]
```

## Known issue: 8-task TRAIN/EVAL overlap

The `train` and `eval` configs share **8 task IDs**, all `oss-fuzz:*`:

| task_id              | project  |
|----------------------|----------|
| oss-fuzz:42536536    | qpdf     |
| oss-fuzz:42537493    | libxml2  |
| oss-fuzz:42537664    | ffmpeg   |
| oss-fuzz:42537686    | ffmpeg   |
| oss-fuzz:42537734    | ffmpeg   |
| oss-fuzz:42538131    | ffmpeg   |
| oss-fuzz:383170474   | libdwarf |
| oss-fuzz:383825645   | ffmpeg   |

Looks like a sampling bug (no de-duplication when drawing the two splits).
Downstream evaluators should subtract these from one side if a clean
held-out is required.

## Provenance & citation

Underlying tasks are drawn from CyberGym (1,507 real-world vulnerability
reproduction tasks from ARVO and OSS-Fuzz, 188 projects). Cite the original:

```bibtex
@misc{wang2026cybergym,
  title  = {{CyberGym}: A Benchmark for LLM-driven Vulnerability Reproduction},
  author = {Wang, [...]},
  year   = {2026}
}
```

This split list itself is released under CC-BY-4.0; the underlying CyberGym
task workspaces follow CyberGym's own license.
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Print row counts/schemas, do not push")
    p.add_argument("--private", action="store_true",
                   help="Create as private repo")
    args = p.parse_args()

    load_dotenv(CYBERGYM / ".env")
    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN missing from .env", file=sys.stderr)
        sys.exit(2)
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    DB = json.loads(DB_PATH.read_text())
    task_to_proj = {t: p for p, ts in DB.items() for t in ts}

    from datasets import Dataset

    built = {}
    for cfg, fname, split in CONFIGS:
        rows = build_records(CYBERGYM / fname, task_to_proj)
        ds = Dataset.from_list(rows)
        built[cfg] = (ds, split)
        missing_proj = sum(1 for r in rows if not r["project"])
        print(f"[build] {cfg:6s}: {len(rows)} rows; missing project for {missing_proj}")
        print(f"        sample: {rows[0]}")

    if args.dry_run:
        print("\n[DRY-RUN] not pushing. To push: rerun without --dry-run.")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])

    # Create repo (idempotent: exist_ok=True)
    api.create_repo(REPO_ID, repo_type="dataset",
                    private=args.private, exist_ok=True)
    print(f"[hub] repo ready: {REPO_ID}  (private={args.private})")

    for cfg, (ds, split) in built.items():
        print(f"[push] {cfg} ({len(ds)} rows, split={split})")
        ds.push_to_hub(REPO_ID, config_name=cfg, split=split,
                       token=os.environ["HF_TOKEN"])

    # Upload the README
    readme_path = CYBERGYM / "_cybergym_tasks_README.md"
    readme_path.write_text(README_TEMPLATE)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print(f"[hub] README uploaded")

    # Verify
    print("\n[verify] reloading from hub...")
    from datasets import load_dataset
    for cfg, (_, split) in built.items():
        ds = load_dataset(REPO_ID, cfg, split=split)
        print(f"  {cfg}/{split}: {len(ds)} rows  features={ds.features}")
    print(f"\nDone. Browse at https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
