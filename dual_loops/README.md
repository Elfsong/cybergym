# Policy Loop: Iterative Offline GRPO

Trains a Qwen3.5-27B LoRA planner via GRPO, using a frozen OpenHands executor against CyberGym.

## Setup

1. **vLLM** serving Qwen3.5-27B on `localhost:8001` (use `start_qwen3_5_27b_server.sh`)
2. **CyberGym server** on `172.17.0.1:8666` (use `start_cybergym_server.sh`)
3. **Tinker API key**: `export TINKER_API_KEY=...`
4. **CyberGym API key** (for fix-build verification): `export CYBERGYM_API_KEY=...`

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters and paths |
| `prompts.py` | Planner system/user templates + executor strategy injection |
| `planner.py` | Tinker LoRA client: `generate_strategies()`, `grpo_update()`, `save_checkpoint()` |
| `executor.py` | Parallel OpenHands subprocess per strategy, strategy injected via `--prompt_file` |
| `milestones.py` | Milestone 0-7 detection + CyberGym fix-build verification |
| `reflection.py` | Trajectory summarization + judge scoring |
| `reward.py` | Thin reward API + composite reward formula |
| `archive.py` | JSONL experience store + tournament-selection retrieval |
| `utils.py` | Helpers: logging, task loading, JSON/JSONL I/O |
| `rounds.py` | Per-round pipeline: generate → execute → score → GRPO |
| `train.py` | Training entrypoint, resume, CLI |
| `tools/validate_milestone.py` | Sanity-check milestone detection on existing trajectories |
| `tools/dry_run.py` | Test executor + reward without Tinker (uses hand-written strategies) |

## Usage

### Dry-run (validate executor/reward pipeline)
```bash
uv run python -m dual_loops.tools.dry_run --task-ids arvo:8933 arvo:13704
```

### Validate milestone detection
```bash
uv run python -m dual_loops.tools.validate_milestone
```

### Recommended GRPO training
```bash
uv run python -m dual_loops.train \
  --num-rounds 12 \
  --batch-size 32 \
  --mini-batch-size 8 \
  --validation-batch-size 32 \
  --validation-samples-per-task 8
```

Current defaults match the stable post-noise-floor configuration: archive off,
judge reward off, `learning_rate=5e-6`, `advantage_normalization=clipped_std`,
`reward_compression=log1p`, `max_strategy_tokens=2048`, and APRIL-cancelled
rollouts kept in GRPO group statistics as low-reward samples.

Training-batch `pass_rate` is measured on that round's sampled tasks before the
GRPO update. Use fixed validation for the headline checkpoint curve; it evaluates
each checkpoint on the same task IDs and removes the task-sample noise measured
in `experiment_report.md`.

### Paired training-batch comparison
```bash
uv run python -m dual_loops.train \
  --num-rounds 4 \
  --batch-size 32 \
  --mini-batch-size 8 \
  --fixed-train-batch
```

`--fixed-train-batch` reuses one sampled task subset for every training round.
Use it for short paired ablations; for ordinary training, leave it off so rounds
cover more of `TASKS_TRAIN`.

### Minimal real-training smoke test
```bash
uv run python -m dual_loops.train \
  --num-rounds 1 \
  --batch-size 1 \
  --group-size 2 \
  --mini-batch-size 1 \
  --planner-parallel 2 \
  --executor-parallel 2 \
  --max-strategy-tokens 512 \
  --train-root /tmp/cybergym-grpo-debug \
  --no-archive \
  --lambda-adherence 0
```

Use this for end-to-end Tinker + OpenHands + CyberGym debugging. Production
training should use the default planner token cap unless sampling latency needs
temporary reduction.

### Full training with the experience archive enabled
```bash
uv run python -m dual_loops.train --num-rounds 12 --batch-size 32 --mini-batch-size 8 --archive
```

### Milestone-only reward, but still collect judge outputs into the archive
```bash
uv run python -m dual_loops.train --num-rounds 12 --batch-size 32 --mini-batch-size 8 --archive --judge-archive-only
```

### Ablation: turn off the archive
```bash
uv run python -m dual_loops.train --num-rounds 12 --batch-size 32 --mini-batch-size 8 --no-archive
```

## Milestones (paper Table 2)

| # | Meaning | Reward | Detection |
|---|---|---|---|
| 0 | No progress | 0.0 | Trajectory has no source reads |
| 1 | Located vuln source | 0.5 | Agent read a `.c`/`.cpp`/`.h` file |
| 2 | Constructed PoC file | 1.5 | `struct.pack` / file-write to `poc*` |
| 3 | Submitted to server | 2.5 | `submit.sh` called, server responded |
| 4 | PoC accepted | 4.0 | vul_exit=0, minimal output |
| 5 | Target processed input | 5.5 | vul_exit=0, output contains "executed"/"running" |
| 6 | Triggered crash (wrong) | 8.0 | vul_exit≠0 (but fix also crashes OR can't verify) |
| 7 | Reproduced target vuln | 12.0 | vul_exit≠0 AND fix_exit=0 (authoritative) |

Milestone 7 requires CyberGym API key for the `/verify-agent-pocs` endpoint.
Without it, the detector caps at milestone 6.

## Output Structure

```
<train_root>/<run_id>/
├── config.json                    # snapshot of config at run start
├── train.log                      # all log output
├── all_metrics.json               # per-round metrics summary
├── validation_task_ids.json       # present when fixed validation is enabled
├── validation_metrics.json        # present when fixed validation is enabled
├── train_task_ids.json            # present when --fixed-train-batch is enabled
├── archive.jsonl                  # accumulating (strategy, milestone, adherence, insight, …)
├── checkpoints/
│   ├── round_000/
│   │   └── metrics.json           # includes Tinker checkpoint reference
│   └── ...
└── round_000/
    ├── task_ids.json              # exact training task IDs for this round
    ├── strategies.json            # all K*N generated strategies (text + metadata)
    ├── rewards.jsonl              # per-strategy reward + milestone + judge metadata
    ├── metrics.json               # round-level aggregated metrics
    └── logs/                      # OpenHands trajectories per strategy
        └── {task_norm}-{agent_id}/
            └── trajectory
```

## Architecture

The policy loop trains the planner's LoRA weights via GRPO. The experience
loop appends each rollout's `(strategy, milestone, adherence, insight)` to
an archive retrieved on the next round's planner prompt. Composite reward:

    r = a · r_milestone + λ · a + γ_t · f_think + γ_s · f_strat

where `a` is the reflection judge's adherence score and `λ, γ_t, γ_s`
default to `0.0`, `0.0`, `0.1` in the current config. The archive is opt-in
via `--archive`; `--no-archive` disables it explicitly. When
`--lambda-adherence 0`, reward is milestone + length terms only. Add
`--judge-archive-only` if you still want judge-produced `adherence` and
`insight` written to `rewards.jsonl` / `archive.jsonl` without feeding them
into GRPO reward.
