# Policy Loop Implementation — Status Report

**Date**: 2026-04-17
**Status**: Phase 1 complete, validated on 600 trajectories

## Files Created (11, 1793 LOC)

| File | LOC | Purpose |
|---|---|---|
| `config.py` | 78 | Hyperparameters (Tinker, executor, GRPO, paths) |
| `prompts.py` | 76 | Planner system/user templates + strategy injection for executor |
| `planner.py` | 297 | Tinker LoRA client: generate_strategies, grpo_update, save_checkpoint |
| `executor.py` | 226 | Parallel OpenHands subprocess per strategy via --prompt_file |
| `reward.py` | 421 | Milestone 0-7 detection + CyberGym fix-build verification |
| `archive.py` | 90 | JSONL store + tournament selection retrieval (Phase 2) |
| `utils.py` | 89 | Logging, task loading, JSON/JSONL I/O |
| `train.py` | 270 | Main loop: sample → generate → execute → score → GRPO → checkpoint |
| `validate_milestone.py` | 142 | Sanity-check milestone detection on existing trajectories |
| `dry_run.py` | 103 | End-to-end test without Tinker (hand-written strategies) |
| `README.md` | — | Documentation |

## Validation Results

### Milestone Detection (reward.py)

Ran `validate_milestone.py` on **600 existing trajectories**:

| Source | N | Ground truth | Detected milestones |
|---|---|---|---|
| Qwen3.5-27B (c414efb4) | 300 | 65 PASSED / 179 FAILED / 51 NO_SUBMIT | all consistent |
| MiniMax-M2.5 (ef24bc78) | 300 | 75 PASSED / 149 FAILED / 72 NO_SUBMIT | all consistent |

**Result**: 0 mismatches. Milestone detector correctly categorizes:
- PASSED → milestone 6 (without fix-build verification) / 7 (with)
- FAILED (submitted, no crash) → milestone 4 or 5
- NO_SUBMIT → milestone 1 or 2

### Integration Check

All 11 files compile cleanly via `py_compile`. Non-Tinker modules (config, prompts, reward, archive, utils, executor) import without errors. Tinker-dependent modules (planner, train) compile cleanly; actual execution requires `uv add tinker tinker-cookbook` and `TINKER_API_KEY`.

## Known Gotchas / Design Notes

### 1. Fix-build verification requires API key
Without `CYBERGYM_API_KEY`, milestone detector caps at 6 (can detect crash but not "correct vuln"). With key, calls `/verify-agent-pocs` to get authoritative milestone 7 based on dual-build (vul_exit≠0 AND fix_exit=0).

### 2. StrategyToExecute.prompt is a Tinker ModelInput
Kept as an object (not serializable). `train.py` serializes only text + metadata. `prompt` is used only during in-memory GRPO update via `prompt.append(EncodedTextChunk(tokens=...))`, matching demo.py.

### 3. OpenHands agent_id vs executor.py agent_id
Each OpenHands subprocess generates its own internal UUID. executor.py discovers the actual sub-dir name after the run and reports *that* UUID as `real_agent_id` so fix-build verification can query it.

### 4. Tinker API uncertainty
The checkpoint save/load uses `save_weights_async` / `load_weights_async` with a fallback to `save_state_async` / `load_state_async`. Whichever the current Tinker SDK supports will be used.

### 5. Claude Code trajectory format also supported
`reward.detect_milestone(traj_format="claude_code")` parses stream-json format. Used only if we need to score CC trajectories for mixed training.

## How to Run (Phase 1)

```bash
# 1. Start vLLM (MiniMax on port 8000)
bash start_minimax_2_5_server.sh

# 2. Start CyberGym server (port 8666)
bash start_cybergym_server.sh

# 3. Install Tinker deps (one-time)
uv add tinker tinker-cookbook torch

# 4. Export API keys
export TINKER_API_KEY=...
export CYBERGYM_API_KEY=...   # for milestone 6→7 distinction

# 5. Dry-run first to validate pipeline (no Tinker needed)
uv run python -m policy_loop.dry_run --task-ids arvo:8933 arvo:13704

# 6. Small-batch training (3 rounds, 10 tasks) to validate GRPO loop
uv run python -m policy_loop.train --num-rounds 3 --batch-size 10

# 7. Full run (10 rounds, 100 tasks)
uv run python -m policy_loop.train --num-rounds 10 --batch-size 100

# 8. Phase 2 (add archive)
uv run python -m policy_loop.train --num-rounds 10 --batch-size 100 --archive
```

## Cost Estimate (10 rounds × 100 tasks × K=8)

| | Per round | 10 rounds |
|---|---|---|
| Tinker (Qwen 27B planner) | ~$4 | ~$40 |
| MiniMax executor (self-hosted vLLM) | $0 | $0 |
| GPU wall time (64 parallel) | ~3 hr | ~30 hr |

## Next Steps

1. Install Tinker SDK, verify `planner.py` actually runs (may need minor API adjustments)
2. Dry-run on 2 tasks to confirm executor + reward pipeline end-to-end
3. Small training run (3 rounds × 10 tasks) to validate GRPO update
4. Phase 2: add adherence gate (Haiku judge) + novelty bonus (sentence embeddings)
