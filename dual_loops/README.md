# Policy Loop: Iterative Offline GRPO

Trains a Qwen3.5-27B LoRA planner via GRPO, using MiniMax-M2.5 (self-hosted vLLM) as the frozen executor.

## Setup

1. **vLLM** serving MiniMax on `localhost:8000` (use `start_minimax_2_5_server.sh`)
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

### Full training
```bash
uv run python -m dual_loops.train --num-rounds 6 --batch-size 48 --mini-batch-size 8
```

### Full training with the experience archive enabled
```bash
uv run python -m dual_loops.train --num-rounds 6 --batch-size 48 --mini-batch-size 8 --archive
```

### Milestone-only reward, but still collect judge outputs into the archive
```bash
uv run python -m dual_loops.train --num-rounds 6 --batch-size 48 --mini-batch-size 8 --archive --judge-archive-only
```

### Ablation: turn off the archive
```bash
uv run python -m dual_loops.train --num-rounds 6 --batch-size 48 --mini-batch-size 8 --no-archive
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
dual_loops_runs/{run_id}/
├── config.json                    # snapshot of config at run start
├── train.log                      # all log output
├── all_metrics.json               # per-round metrics summary
├── archive.jsonl                  # accumulating (strategy, milestone, adherence, insight, …)
├── checkpoints/
│   ├── round_000/
│   │   ├── lora_weights           # Tinker checkpoint
│   │   └── metrics.json
│   └── ...
└── round_000/
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
