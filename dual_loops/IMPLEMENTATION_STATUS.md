# Policy Loop — Implementation Status

Current status (last updated 2026-04-18). See [`PIPELINE.md`](PIPELINE.md) for the end-to-end flow and [`README.md`](README.md) for the usage reference.

## Architecture

One integrated training loop; the experience archive and reflection judge are part of the default configuration, not optional add-ons.

```
dual_loops/
├── config.py       — all hyperparameters
├── prompts.py      — planner system/user templates + executor strategy injection
├── planner.py      — Tinker LoRA client: generate_strategies (per-sample retrieve),
│                     grpo_update (mini-batch + cosine LR), save/load checkpoint
├── executor.py     — parallel OpenHands subprocess per strategy,
│                     executions.jsonl checkpoint for resume
├── reward.py       — milestone 0-7 detection + CyberGym fix-build verification +
│                     composite reward
├── archive.py      — JSONL store + tournament-selection retrieval (returns dicts
│                     with strategy / milestone / insight)
├── utils.py        — logging, task loading, I/O
├── train.py        — orchestrator; round = retrieve → generate → execute → score →
│                     reflect → reward → GRPO → archive append → checkpoint
└── dry_run.py      — 2-task sanity check that exercises executor + reward
                      without touching Tinker
```

Reflection judge + archive live under [`experience_loop/`](../experience_loop/) (new package). See that directory's PLAN.md for its own design notes.

## Validated

- Milestone detector on 600 existing trajectories (300 Qwen + 300 MiniMax): 0 mismatches against ground-truth PASS/FAIL/NO_SUBMIT labels.
- Reflection judge on live vLLM: base Qwen3.5-27B emits the expected `<adherence>N</adherence>` and `<insight>...</insight>` tags with max_tokens=8192; parse-failure fallback `(0.0, "")` smoke-tested.
- Archive v3 schema roundtrip: records with `insight` field persist and are retrieved as dicts by the planner.
- Per-sample tournament retrieval: on a 10-entry eligible pool, K=8 independent draws produce 7 distinct prior subsets (validated by the test at `dual_loops/planner.py::generate_strategies`).
- Cosine LR + AdamW schedule: verified analytically against expected curve (peak 2e-5 at end of warmup, floor 2e-6 at T·S).
- Checkpoint / resume: three-level (strategies pickle, executions.jsonl with fsync, Tinker LoRA state); successfully resumes mid-round.

## Known gotchas

- **`CYBERGYM_API_KEY` is required** for the milestone 6 vs 7 distinction (fix-build verification via `/verify-agent-pocs`). Without it the detector caps at milestone 6.
- **`StrategyToExecute.prompt`** is a live Tinker ModelInput object. It is pickled for resume but must not be JSON-serialized; only text and metadata go into `strategies.json`.
- **OpenHands agent_id vs our agent_id**: each OpenHands subprocess generates its own UUID inside the sandbox. `executor.py` discovers the actual sub-dir name after the run and reports it as `real_agent_id` so fix-build verification can find the PoCs.
- **KL penalty** is in the config (`kl_beta`) but not currently wired: the Tinker `importance_sampling` loss does not accept a KL term. Stability comes from LoRA + cosine LR + grad-clip instead.

## Running

```bash
# 1. Executor / judge vLLM (port 8001, Qwen3.5-27B)
bash start_qwen3_5_27b_server.sh

# 2. CyberGym evaluation server (port 8666)
bash start_cybergym_server.sh

# 3. Environment
export TINKER_API_KEY=...
export CYBERGYM_API_KEY=...

# 4. Train (archive + reflection on by default)
uv run python -m dual_loops.train \
  --num-rounds 6 --batch-size 48 --num-substeps 6 \
  --group-size 8 --strategy-temperature 1.0 \
  --executor-model openai/Qwen/Qwen3.5-27B \
  --executor-base-url http://localhost:8001/v1 \
  --executor-timeout 1800 --executor-parallel 64

# Ablation: disable archive (falls back to unguided planner-executor loop)
uv run python -m dual_loops.train [...] --no-archive
```

Cost and wall-time breakdown: see paper Appendix J.
