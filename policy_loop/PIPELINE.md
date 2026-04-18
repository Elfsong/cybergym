# Policy Loop — Training Pipeline

End-to-end data flow of the iterative offline GRPO loop. For setup, hyperparameters,
and milestone definitions see [README.md](README.md).

---

## High-level

```
                  ┌──────────────────────────────────────────┐
                  │          Tinker LoRA Planner             │
                  │       (Qwen3.5-27B, rank 32)             │
                  └───────────┬──────────────┬───────────────┘
                              │ strategies   │ grad step
                              ▼              ▲
           ┌────────────────────────┐   ┌────────────────────┐
           │   OpenHands Executor   │   │  GRPO update       │
           │  (vLLM + subprocess)   │   │  (25 substeps)     │
           │   N tasks × K group    │   │                    │
           └───────────┬────────────┘   └─────────▲──────────┘
                       │ trajectory              │ (strategy,
                       ▼                         │  reward)
           ┌────────────────────────┐            │
           │ Milestone / Reward     ├────────────┘
           │ (CyberGym API 0..7)    │
           └────────────────────────┘
```

One **round** = one full traversal of the diagram. The loop runs `num_rounds`
rounds (default 10).

---

## Components

| Module | Responsibility |
|---|---|
| `train.py` | Orchestration: the `train()` entry + per-round `run_round()` |
| `planner.py` | Tinker LoRA client: `generate_strategies()`, `grpo_update()`, save/load state |
| `executor.py` | Parallel OpenHands subprocesses, each running one strategy |
| `reward.py` | Milestone 0–7 detection + CyberGym `/verify-agent-pocs` for milestone 7 |
| `archive.py` | experience store + tournament retrieval |
| `prompts.py` | Planner system/user templates + strategy-injection template for the executor |
| `utils.py` | Task-file parsing, JSON/JSONL I/O, logging setup |

---

## Entry point — `train()`

```
_load_dotenv()
parse CLI → Config
train(config, resume_from=…)
  ├─ if resume_from: reuse run_id, discover last-completed round
  ├─ ensure_dirs + setup_logging
  ├─ save config.json snapshot
  ├─ parse_tasks_file → all_task_ids
  ├─ Planner.init()  (create LoRA training client, tokenizer, renderer)
  ├─ if start_round > 0: Planner.load_checkpoint(start_round - 1)
  ├─ rng = Random(42); fast-forward to match skipped rounds
  └─ for round_idx in [start_round, num_rounds):
        run_round(…)
```

Nothing in `train()` talks to the executor directly — all heavy lifting is
inside `run_round`.

---

## One round — `run_round()`

Seven stages, in order. Data flow between them is all in-memory except where
noted (`round_000/…`).

### 1. Task sampling

```
if batch_size >= |pool|:  batch_ids = shuffle(pool)
else:                     batch_ids = rng.sample(pool, batch_size)
tasks = build_tasks(batch_ids, config, archive)   # adds description + prior strategies
```

The RNG is seeded deterministically (seed=42 + round-idx offset), so task
sampling is reproducible across resumes.

### 2. Strategy generation (on-policy)

```
strategies = planner.generate_strategies(tasks)
             └─ save_weights_and_get_sampling_client → on-policy sampler
             └─ for each task: sample K strategies in parallel
             └─ parse renderer output, split </think> → (thinking, strategy)
             └─ return K·|tasks| StrategyToExecute objects
```

Each `StrategyToExecute` carries `tokens`, `logprobs`, and the original
Tinker `ModelInput` prompt — all needed later by `grpo_update`.

**Persisted**: `round_XXX/strategies.pkl` (full objects) and
`round_XXX/strategies.json` (human-readable text + metadata).

### 3. Execution (the long stage)

```
results = execute_strategies(strategies, config, round_dir)
```

- Each strategy is written to a temp prompt file, then
  `examples/agents/openhands/run.py` is spawned as a subprocess with
  `--prompt_file` injecting the strategy.
- Up to `executor_parallel` (default 64) subprocesses in flight via a
  `ThreadPoolExecutor`.
- Hard ceiling per rollout: `executor_timeout` (default 1800s) + 300s grace;
  on timeout the whole process group is SIGKILLed.
- Agent may write a partial trajectory from OpenHands event files — recovered
  by `_recover_trajectory` even if the subprocess died early.
- Completed rollouts are streamed to `round_XXX/executions.jsonl` (see
  [Resume](#resume--checkpointing) below).

Each completed rollout → an `ExecutionResult(strategy, agent_id,
trajectory_path, wall_seconds, …)`.

### 4. Scoring

```
rewarded = score_results(results, config)      # list[(strategy, reward, milestone)]
```

For each result with a trajectory:

1. `detect_milestone(traj, agent_id, …)` parses the OpenHands events and
   classifies progress on the 0–7 scale.
2. If the agent submitted a PoC (milestone ≥ 4), the CyberGym server is
   queried (`verify_fix=True`) to distinguish milestone 6 (crash on vulnerable
   build only) from milestone 7 (crash on vul AND fix builds clean).
3. `compute_reward(milestone, adherence=1.0, …)` → scalar reward.

Rollouts with no trajectory are scored milestone 0 / reward 0.

**Persisted**: `round_XXX/rewards.jsonl` (per-rollout reward + milestone).

### 5. GRPO update

```
metrics = planner.grpo_update([(s, r) for s, r, _ in rewarded], round_idx=r)
```

Inside `grpo_update`:

1. **Group by task**, compute `(reward - mean) / (std + ε)` → per-rollout advantage.
2. **Drop degenerate groups** (std ≈ 0, i.e. everyone got the same reward).
3. For each surviving rollout, build a `tinker.Datum`:
   - `model_input = prompt + tokens[:-1]` (next-token prediction alignment)
   - `loss_fn_inputs = { target_tokens, logprobs, advantages }`
4. **Shuffle task groups** with a per-round seed, then **split into `num_substeps`
   disjoint mini-batches** (GRPO groups stay intact within a substep).
5. Pipelined forward/backward + optim steps — enqueue substep *i+1* before
   awaiting substep *i*, so Tinker stays fully utilized.
6. Loss is `importance_sampling` (Tinker-side PPO-style clipped objective).

Returns per-substep metrics and aggregate group stats.

### 6. Archive append

```
if archive is not None and config.archive_enabled:
    archive.append_batch([(task_id, strategy, milestone) for …])
```

Writes to `round_XXX/../archive.jsonl`; retrieved at stage 1 of future rounds
via tournament selection.

### 7. Checkpoint

```
planner.save_checkpoint(round_idx, metrics)
  ├─ training_client.save_state_async(name="round_XXX", ttl=7d)
  │      → metrics["tinker_checkpoint"] = "tinker://…"
  └─ write metrics.json
save_json(metrics, round_XXX/metrics.json)
```

Metrics include: `pass_rate` (milestone 7 fraction), `avg_milestone`,
`milestone_histogram`, `num_substeps`, `substep_metrics` (with per-step
`optim_metrics.loss`), `mean_reward`, `degenerate`/`total_groups`, timings.

---

## Resume & checkpointing

Three checkpoint layers, each guarding a different failure window:

| Layer | Artifact | Granularity | What it saves |
|---|---|---|---|
| **Tinker state** | `tinker://…` ref in `metrics.json` | end of round | LoRA weights + optimizer state |
| **Strategies** | `round_XXX/strategies.pkl` | after stage 2 | Generated strategies (tokens, logprobs, ModelInput prompt) |
| **Executions** | `round_XXX/executions.jsonl` | per rollout | Appended atomically as each OK rollout completes |

### Resume semantics

```bash
uv run python -m policy_loop.train \
  --resume-from /data/cybergym_data/cybergym-train-data/<run_id> \
  <other flags>
```

1. `config.run_id` is set to the resume dir's name (same output dir).
2. `_find_last_completed_round()` returns the highest round with a saved
   Tinker checkpoint. `start_round = last + 1` (or 0 if none completed yet).
3. If `start_round > 0`: `planner.load_checkpoint(start_round - 1)` restores
   LoRA + optimizer state from Tinker.
4. Task-sampling RNG is fast-forwarded through the skipped rounds so the
   sampled task batches are identical to the original run.
5. Inside `run_round`:
   - `strategies.pkl` exists → load and skip regeneration (saves ~36 min).
   - `executions.jsonl` exists → pre-populate `results[idx]` from entries
     whose trajectory file is still on disk AND `(task_id, group_id)`
     matches; only submit the remaining indices to the pool.

### What does NOT resume

- **NO_TRAJ rollouts** are not written to `executions.jsonl` on purpose, so
  they're retried automatically. Useful when vLLM or the cybergym server
  flapped and the agent bailed.
- **Mid-`grpo_update` crashes** — the 25 substeps run in one call, and
  Tinker state is only persisted at the end of the round. If you crash
  between "all rollouts done" and "checkpoint written", the whole round's
  gradient steps are lost. In practice this window is ~5 minutes so the
  risk is small.

### Killing an in-flight run cleanly

`ThreadPoolExecutor` subprocesses use `start_new_session=True`, so SIGTERM
on the parent does **not** reach the 64 OpenHands children. To kill cleanly:

```bash
pkill -f policy_loop.train           # kills parent + direct children
pkill -f examples/agents/openhands   # kills orphaned OpenHands subprocs
docker ps --filter name=openhands-runtime- -q | xargs -r docker rm -f
```

`executor.py` also runs a best-effort Docker cleanup (`_cleanup_docker_containers`)
at the end of each successful round.

---

## On-disk layout

```
<train_root>/<run_id>/
├── config.json                      # full Config snapshot
├── config_resumed_from_<N>.json     # snapshot(s) added on each resume
├── train.log                        # all-round append-only log
├── all_metrics.json                 # per-round summaries (written at end of run)
├── archive.jsonl                    # experience store (strategy / milestone / adherence / insight / …)
├── checkpoints/
│   └── round_XXX/
│       └── metrics.json             # includes tinker_checkpoint ref
└── round_XXX/
    ├── strategies.pkl               # resume: full StrategyToExecute objects
    ├── strategies.json              # human-readable text + metadata
    ├── executions.jsonl             # resume: one line per OK rollout
    ├── rewards.jsonl                # (strategy, reward, milestone) per rollout
    ├── metrics.json                 # round-level aggregated metrics
    ├── logs/
    │   └── {task_norm}-{agent_id}/
    │       ├── trajectory           # OpenHands JSON trajectory
    │       └── file/sessions/…      # raw event files (recovery source)
    └── tmp/                         # transient, cleaned at end of round
```

---

## Determinism & seeds

| Where | Seed |
|---|---|
| Task sampling (`run_round`) | `random.Random(42)`, advanced per round |
| GRPO substep shuffle (`grpo_update`) | `random.Random(42 + round_idx)` |
| Planner sampling | stochastic (`temperature`, `top_p`) — **not reproducible** |
| Executor rollouts | stochastic (vLLM sampling) — **not reproducible** |

Resuming a round does **not** reproduce the same strategies/rollouts unless
`strategies.pkl` / `executions.jsonl` exist — those are the only way to
preserve on-policy state across crashes.

---

## Typical single-round timing

For `batch_size=300, group_size=8, executor_parallel=64, num_substeps=25`
(2400 rollouts per round):

| Stage | Time |
|---|---|
| Strategy generation | ~35 min |
| Execution (mean ~24 min/rollout) | ~15 h |
| Scoring + milestone detect | ~10 min |
| GRPO update (25 substeps) | ~5 min |
| **Round total** | **~15.5 h** |

5 rounds ≈ 3 days. Execution is the dominant cost by ~2 orders of magnitude;
resume via `executions.jsonl` exists specifically to avoid re-running it.
