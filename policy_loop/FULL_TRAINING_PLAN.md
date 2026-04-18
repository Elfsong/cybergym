# Full GRPO Training Plan — Mini-Batch Iterative Offline GRPO

## User-Specified Design

Each round:
- **Rollout phase**: 300 tasks × K=8 strategies = 2400 trajectories
- **Score phase**: Compute per-task GRPO advantages (300 groups of 8)
- **Update phase**: Shuffle task groups, split into 25 mini-batches of 12 tasks (96 datums each), do one `forward_backward` + `optim_step` per mini-batch → **25 gradient updates per round**

10 rounds total = **250 gradient updates**.

## Feasibility Evaluation

### ✅ Algorithmic correctness

1. **GRPO advantages are computed per-task BEFORE mini-batching** — mini-batch split doesn't affect normalization scale.
2. **Tinker's `importance_sampling` loss handles off-policy data** — stores `logprobs` of sampling policy in each datum; computes IS ratio vs current training policy at each `forward_backward`.
3. **Tinker cookbook officially supports this pattern** — `tinker_cookbook/rl/train.py::train_step()` takes `num_substeps` param for exactly this purpose, with pipelined `forward_backward` + `optim_step` for GPU efficiency.

### ⚠️ Risks

1. **High substep count (25)**: Cookbook defaults are `num_substeps=1-2`. Typical PPO uses 3-10 epochs. 25 substeps on same rollouts = aggressive off-policy usage. Risk of importance-sampling ratio blowing up late in the round.
   - **Mitigation**: Tinker's IS loss likely clips; LoRA rank-32 updates are small per step. Monitor `frac_degenerate` and mean IS ratio (if exposed).

2. **Sampling policy drift**: After 25 updates, the policy that generated the rollouts is ~25 gradient steps stale. Less of a concern with LoRA + small lr (2e-5).

3. **Cost sensitivity**: If training collapses at round 3, we've burned ~$130.

### ✅ Infrastructure check

| Component | Status | Capacity |
|-----------|--------|----------|
| Tinker API | OK (key in .env) | No known concurrency limit |
| Qwen3.5-27B vLLM (port 8001) | OK, MAX_NUM_SEQS=64 | Supports up to 64 parallel executor rollouts |
| CyberGym server | OK | Unknown peak load; current parallel=16 works |
| 8× A100 80GB | OK, all occupied by vLLM workers | — |

## Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Rounds | 10 | 10 full passes through task pool |
| Batch size | 300 (all tasks) | User-specified; no sampling variance |
| Group size K | 8 | GRPO variance reduction; validation showed degenerate at K=4 |
| Mini-batch tasks | 12 | 300/12 = 25 substeps per round |
| **num_substeps** | **25** | Derived from (batch_size / mini_batch_tasks) |
| max_strategy_tokens | 16384 | Thinking + strategy from validation (~3000 avg) |
| Learning rate | 2e-5 | Conservative for multi-step updates |
| LoRA rank | 32 | Cookbook default |
| Executor model | Qwen3.5-27B | Port 8001, 8×A100 |
| Executor parallel | 48 | Leaves headroom on MAX_NUM_SEQS=64 |
| Executor timeout | 1800s | Full 30min as per baseline |
| Archive (Phase 2) | Disabled | Enable after baseline training curve observed |

## Cost Estimate

Per-round:
- **Sampling**: 2400 × (500 prompt + 3000 output) tokens
  - Input: 2400 × 500 = 1.2M × $1.24/M = $1.49
  - Output: 2400 × 3000 = 7.2M × $3.73/M = $26.86
  - **Total: ~$28.35**
- **Training**: 2400 × 0.5 (assume 50% non-degenerate) × 3500 tokens = 4.2M tokens
  - Each datum processed once total (split across 25 substeps)
  - **Total: 4.2M × $3.73/M = $15.66**
- **Per round: ~$44**
- **10 rounds: ~$440**

## Wall Time Estimate

Per-round:
- Generation (Tinker sampling 2400 strategies): ~15-30 min (scales with concurrent API load)
- Execution (2400 rollouts / 48 parallel × 10min avg): **~8.3 hours**
- Scoring + 25 substeps: ~10-15 min
- **Total per round: ~9 hours**

**10 rounds: ~90 hours (~3.8 days)**

Wall time dominated by executor, not Tinker.

## Code Changes Required

### 1. `policy_loop/config.py`

Add field:
```python
num_substeps: int = 1   # gradient steps per round (1 = single update on full batch)
```

### 2. `policy_loop/planner.py` — `grpo_update()`

Replace single-step update with mini-batch loop:

```python
async def grpo_update(self, strategies_with_rewards, eps=1e-8):
    # ... compute per-task advantages (UNCHANGED)
    # ... build all datums (UNCHANGED, ~1200 datums for batch=300, K=8, 50% non-degen)

    # NEW: split datums into substeps, shuffle task groups between substeps
    import random
    rng = random.Random(42)  # seed per-round based on round_idx for reproducibility
    task_to_datums = defaultdict(list)
    for datum, task_id in datums_with_task:
        task_to_datums[task_id].append(datum)
    task_ids = list(task_to_datums.keys())
    rng.shuffle(task_ids)

    num_substeps = self.config.num_substeps
    tasks_per_substep = max(1, len(task_ids) // num_substeps)

    substep_metrics = []
    for s in range(num_substeps):
        substep_tasks = task_ids[s * tasks_per_substep : (s+1) * tasks_per_substep]
        substep_datums = [d for t in substep_tasks for d in task_to_datums[t]]
        if not substep_datums:
            continue
        fwd_bwd = await self.training_client.forward_backward_async(
            substep_datums, loss_fn="importance_sampling",
        )
        opt = await self.training_client.optim_step_async(self.adam_params)
        await fwd_bwd.result_async()
        opt_result = await opt.result_async()
        substep_metrics.append({
            "substep": s,
            "n_tasks": len(substep_tasks),
            "n_datums": len(substep_datums),
        })

    metrics["substeps"] = substep_metrics
    return metrics
```

### 3. `policy_loop/train.py`

Fix `run_round()` to use all tasks when `batch_size >= task_pool_size`:
```python
if config.batch_size >= len(all_task_ids):
    batch_ids = list(all_task_ids)
    rng.shuffle(batch_ids)
else:
    batch_ids = rng.sample(all_task_ids, config.batch_size)
```

Add CLI flag:
```python
parser.add_argument("--num-substeps", type=int, default=None)
if args.num_substeps is not None:
    config.num_substeps = args.num_substeps
```

### 4. Optional: Resume capability

Add `--resume-from <run_id>` flag to load the last saved Tinker checkpoint. Currently planner.py has `load_checkpoint()` stubbed; wire it up to CLI.

## Launch Command

```bash
nohup uv run python -m policy_loop.train \
  --num-rounds 10 \
  --batch-size 300 \
  --group-size 8 \
  --num-substeps 25 \
  --executor-model "openai/Qwen/Qwen3.5-27B" \
  --executor-base-url "http://localhost:8001/v1" \
  --executor-timeout 1800 \
  --executor-parallel 48 \
  > /tmp/grpo_full_training.log 2>&1 &
```

## Monitoring (Unattended)

```bash
# Progress
tail -f /tmp/grpo_full_training.log | grep -E "ROUND|Generation:|Execution:|Substep|pass_rate|ERROR"

# Per-round metrics
jq '{round, pass_rate, avg_milestone, frac_degenerate, wall_seconds}' \
   policy_loop_runs/<run_id>/round_*/metrics.json

# Watch substep drift
jq '.substep_metrics[]' policy_loop_runs/<run_id>/round_*/metrics.json
```

## Failure Modes & Safeguards

| Scenario | Detection | Response |
|----------|-----------|----------|
| Tinker API quota exceeded | HTTP 429/error in log | Auto-retry already in Tinker SDK |
| vLLM crashes | Executor NO_TRAJ spike | Manual restart; training continues with skipped rollouts |
| CyberGym server down | Reward scoring fails | Defaults to milestone 0; visible in metrics |
| OOM on A100 | vLLM crash | Lower executor_parallel to 32 |
| Training divergence (reward decreases) | avg_milestone trends down | Ctrl+C, lower lr or num_substeps |

## Decision Points

1. **Run now or wait for Claude Code eval?** Claude eval uses Tinker? No — it's Claude API. Executor vLLM is separate. Tinker API is the only shared resource. Running both concurrently risked slowing Tinker generation in v2 validation (27min vs 7min). **Recommend: wait for Claude eval to finish (~1h).**

2. **num_substeps=25 vs smaller?** Can start with 10 substeps (30 tasks each, more conservative) and increase if training is stable. **Recommend: start with 25 as proposed; monitor degenerate rate; if unstable, rerun with 10.**

3. **Resume support?** With 4-day runtime, a crash could lose significant work. **Recommend: implement resume before launch** (low effort, critical safety).

## Deliverables Checklist (before launch)

- [ ] Code change: `config.py` — add `num_substeps` field
- [ ] Code change: `planner.py` — mini-batch `grpo_update()` with task-group shuffling
- [ ] Code change: `train.py` — batch_size=pool_size handling, CLI flags, resume support
- [ ] Smoke test: 2 rounds × batch=30 × num_substeps=3 on current setup (~1 hour, verify substep updates work)
- [ ] Claude Code eval finished
- [ ] Document launch procedure in README
- [ ] Set up log rotation (run.log will grow large over 4 days)

## Total Estimate

- **Implementation**: ~2-3 hours (code + smoke test)
- **Training wall time**: ~90 hours (~3.8 days)
- **Tinker cost**: ~$440
- **Output**: Trained LoRA checkpoint on Tinker + 24000 trajectories + 250-point training curve
