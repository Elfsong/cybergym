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
