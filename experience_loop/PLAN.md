# Experience Loop — Implementation Plan

Phase 2 extension of the GRPO policy loop. Adds three mechanisms on top of Phase 1:

1. **Archive**: each round's rollouts populate a store of `(task, strategy, milestone, adherence, insight)` records; future rounds retrieve top-k priors for the same task and inject them into the planner prompt.
2. **Reflection judge**: after every execution, a frozen base Qwen3.5-27B (served by the existing executor vLLM on port 8001) inspects `(strategy, trajectory)` and emits in a single call both an adherence score and a short actionable **insight** string.
3. **Per-sample archive retrieval**: within a round, each of the K samples for a task performs an independent tournament draw; K prompts for the same task see different prior subsets, giving a hard guarantee of intra-group input variance.

For Phase 1 (bare GRPO with milestone reward only) see [`policy_loop/`](../policy_loop/).

---

## 1. Goal

Transform the planner from a stateless strategy sampler into an agent that **learns from its own rollout history** and is credited only for outcomes the executor actually drove via the plan:

- Planner conditions on retrieved priors from the archive (per-sample, for diversity).
- Reflection produces adherence (gates milestone credit) and insight (task-specific knowledge for future priors).
- Reward: `r = a · r_milestone + λ · a + γ_t · f_think + γ_s · f_strat`.

Hypothesis: versus Phase 1, Phase 2 (a) accumulates high-milestone strategies for hard tasks, (b) suppresses noise from "lucky" executor rollouts that ignored the plan, and (c) reuses task-specific insights across rounds.

---

## 2. Current Status — Gap Analysis

| Subsystem | Scaffolded? | Functional? | Gaps |
|---|---|---|---|
| `Archive` (JSONL store + tournament retrieval) | ✅ | ✅ | — |
| Planner prompt with `{archive_block}` + `format_archive_block()` | ✅ | ✅ | insight not yet shown in block |
| `archive.append_batch()` in `run_round` | ✅ | ✅ | — |
| CLI `--archive` flag + Config knobs | ✅ | ✅ | — |
| Archive schema v3 (round / group_id / adherence / insight / lengths / trajectory_path / run_id / timestamp) | ✅ | ✅ | — |
| **Reflection judge** (base Qwen via vLLM) — adherence + insight | ✅ | ✅ | calibration: manual 50-sample study pending |
| `compute_reward(milestone, adherence, λ, γ_t, γ_s, …)` | ✅ | ✅ | — |
| `γ_t`, `γ_s` length reward components | ✅ | ✅ | off by default; opt-in |
| **Per-sample archive retrieval** (K independent draws / task) | ❌ | ❌ | needed so K samples see different priors |
| Evaluation harness (archive on/off ablation) | ❌ | ❌ | pending Phase 1 baseline finish |

Only remaining implementation gap: **per-sample retrieval** (paper §4.2 already describes it).

---

## 3. Components

### 3.1 Archive schema v3

One record per rollout:

```json
{
  "task_id": "arvo:8933",
  "round": 3,
  "group_id": 4,
  "strategy": "...",
  "milestone": 7,
  "adherence": 0.78,
  "insight": "submit.sh rejects non-binary files; strategies must produce raw bytes via struct.pack.",
  "n_thinking_tokens": 2500,
  "n_strategy_tokens": 450,
  "trajectory_path": "...",
  "run_id": "abc12345",
  "timestamp": "2026-04-18T..."
}
```

Backward compat: `Archive._load()` tolerates missing fields (v1 and v2 records load fine).

### 3.2 Reflection Judge (`experience_loop/adherence.py`)

**Inputs**: `(strategy: str, trajectory: OpenHands JSON)`.
**Outputs**: `(adherence ∈ [0, 1], insight: str)` from a single LLM call.

**Model**: **base Qwen3.5-27B** on the existing vLLM at `http://localhost:8001/v1`.
- Cannot use the LoRA-adapted planner (self-judging → non-stationary reward, self-reinforcement bias).
- Frozen base model is the minimal safe choice.
- Free at compute cost; runs during the scoring phase when the executor is idle.

**Prompt output format**:
```
<adherence>N</adherence>          (integer 0-10, four-band rubric)
<insight>text</insight>            (1-3 sentences, actionable, task-specific)
```

**Trajectory summarization**: compress ≤ 4000 chars before sending to judge. Preserves first/last assistant messages, all `submit.sh` invocations, file reads/edits, and `think` actions.

**max_tokens = 8192**: Qwen3.5-27B emits ~5000 tokens of `Thinking Process:` chain-of-thought before the two XML tags. Thinking mode ON improves judgment quality; we absorb the ~75 min/round wall-clock cost.

**Parse failure**: fall back to `(0.0, "")`. No retry.

### 3.3 Per-sample archive retrieval (to implement)

Currently: `run_round` → `build_tasks` retrieves once per task → all K samples see the same priors.

Change: retrieve **K times per task** (one per sample). Each call's tournament RNG draws a different subset from the eligible pool.

**Implementation** (~30 lines):
- `Planner` gets an `archive` reference (via `bind_archive`).
- `Task` drops its `prior_strategies` field (retrieval moves to `generate_strategies`).
- `generate_strategies` iterates `(task, k)` pairs, calling `archive.retrieve()` fresh for each, building K different prompts per task, dispatching `num_samples=1` calls in parallel.
- `StrategyToExecute` adds `priors_shown: list[(str, int)]` for auditability.

**Result**: given a task whose eligible pool has `|P| > n=3`, the K=8 samples see tournament-drawn 3-subsets of P — almost always different. Degenerate groups on non-cold-start tasks become rare.

**Cold start (round 1)**: archive empty → all `retrieve()` return `[]` → K prompts identical → behavior identical to Phase 1 (correct).

**Performance**: sampling calls scale from N to N·K (e.g. 300 → 2400). Tinker handles these in parallel; expected overhead ≤ 1-3 minutes of generation wall time.

### 3.4 Cold-start

Round 1 has an empty archive. Behavior is no-op fallback to Phase 1:
- All `retrieve()` return `[]`.
- `format_archive_block([])` returns `""`.
- Planner prompt has no archive block.
- Reflection still runs, so `adherence` and `insight` are populated from round 1 onward.

Starting in round 2, tasks with any round-1 rollout at milestone ≥ 3 receive priors; coverage grows monotonically. Tasks where all K round-1 rollouts failed stay cold until some later round produces a qualifying record.

---

## 4. Reward

```
r = a · r_milestone + λ · a + γ_t · f_think + γ_s · f_strat
```

- `a ∈ [0, 1]` — adherence from reflection judge
- `r_milestone` — 0–12 via convex schedule (Table in paper)
- `λ = 0.5` (default, activated by `--phase2`) — bonus for producing followable plans
- `f_think = min(n_think / 3000, 1)`, `f_strat = min(n_strat / 500, 1)` — bounded length signals
- `γ_t = γ_s = 0` by default (opt-in via CLI)

No novelty term. Intra-group diversity comes from (a) per-sample retrieval, (b) the length-dependent terms, and (c) the stochastic sampling temperature.

---

## 5. Integration into the Loop

### `policy_loop/train.py::run_round`

After `execute_strategies`:

```python
rewarded = score_results(results, config)                           # flat milestone

if config.phase2_enabled:
    pairs = await score_reflection_batch(results, ...)              # (adherence, insight)
    adherences = [a for a, _ in pairs]
    insights   = [ins for _, ins in pairs]
    rewarded = [
        (s, compute_reward(
            milestone=m,
            adherence=adherences[i],
            lambda_adherence=config.lambda_adherence,
            thinking_length=s.n_thinking_tokens,
            strategy_length=s.n_strategy_tokens,
            gamma_thinking=config.gamma_thinking,
            gamma_strategy=config.gamma_strategy,
            thinking_ref_tokens=config.thinking_ref_tokens,
            strategy_ref_tokens=config.strategy_ref_tokens,
        ), m)
        for i, (s, _, m) in enumerate(rewarded)
    ]
```

### `policy_loop/archive.py::append_batch`

Accepts all v3 fields; pass-through to JSONL.

### Archive append site in `run_round`

```python
archive.append_batch([
    {"task_id": s.task_id, "round": round_idx, "group_id": s.group_id,
     "strategy": s.strategy, "milestone": m,
     "adherence": adherences[i], "insight": insights[i],
     "n_thinking_tokens": s.n_thinking_tokens, "n_strategy_tokens": s.n_strategy_tokens,
     "trajectory_path": str(results[i].trajectory_path) if results[i].trajectory_path else None,
     "run_id": config.run_id, "timestamp": datetime.now().isoformat()}
    for i, (s, _, m) in enumerate(rewarded)
])
```

### CLI flags (already wired)

```
--phase2                       # enables archive + reflection + sets λ=0.5
--lambda-adherence 0.5
--gamma-thinking 0.0
--gamma-strategy 0.0
--thinking-ref-tokens 3000
--strategy-ref-tokens 500
```

---

## 6. Evaluation Plan

### A. Phase 1 vs Phase 2 (matched)

- **Run A**: Phase 1 (no archive, flat milestone). Baseline = `0080dd4b`.
- **Run B**: Phase 2 (`--phase2`, per-sample retrieval enabled).
- 5 rounds × 300 tasks × K=8.
- Track: pass rate per round, avg milestone, `frac_with_priors`, mean adherence, `adherence × milestone` joint distribution, `frac_degenerate`.

### B. Adherence gate ablation

Phase 2 with `λ = 0` (archive on, adherence-only bonus off). Isolates "does rewarding followability help beyond gating?"

### C. Retrieval strategy ablation

On a frozen archive snapshot from B:
1. Tournament (current default).
2. Top-n deterministic.
3. Random-n.

### D. `archive_min_milestone` sweep

`min_milestone ∈ {0, 3, 5, 7}`. Measures how aggressively to filter priors.

---

## 7. Implementation Order (remaining work)

1. **Per-sample retrieval** (~30 lines) — §3.3 above. Smoke test: one round × 20 tasks × K=4, verify K prompts differ when archive has content.
2. **Judge calibration** — manually label 50 `(strategy, trajectory, reference adherence)` triples from `0080dd4b` round 0 logs once they finish; iterate the judge prompt until human-machine agreement exceeds 80%.
3. **Ablation A** — Phase 2 vs Phase 1 on matched config.
4. **Ablations B / C / D** — only if A shows Phase 2 ≥ Phase 1.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Reflection judge miscalibrated vs human adherence | §7 step 2: hand-label 50 samples before wiring into reward |
| `λ · a` shifts reward scale and regresses pass rate | Start `λ = 0.5`; halve or zero if round 1 regresses vs baseline |
| Planner over-imitates retrieved priors → K-group variance collapses | Per-sample retrieval (§3.3) + `γ_t` opt-in as residual signal |
| Judge cannot parse its own strategy (recursive semantics) | Judge sees strategy as opaque external text; no self-reference issue |
| Trajectory summary drops critical info | Log `(strategy, summary, adherence, insight)` triples first round; inspect |
| Insight field grows unbounded and bloats archive.jsonl | No cap; per-rollout insight ~100-500 chars, 2400/round → ~1 MB/round — acceptable |

---

## 9. Cost Summary

| Component | Per round |
|---|---|
| Executor (vLLM, self-hosted) | $0 |
| Planner (Tinker) | ~$28 |
| Reflection judge (vLLM, self-hosted) | **$0** (reuses port-8001 server) |
| Archive / retrieval | $0 |
| **Extra vs Phase 1** | **$0** |

Extra wall time per round: ~75 min (reflection phase). Per 5-round run: ~6 hours extra.

---

## 10. Deliverables

```
experience_loop/
├── PLAN.md                        # this doc
├── adherence.py                   # reflection judge: score_reflection_batch()
├── prompts/
│   └── adherence.txt              # adherence + insight prompt
└── tests/
    └── test_adherence.py          # 50 hand-labeled samples (TODO)
```

Modifications to existing files already in place:

- `policy_loop/archive.py::append_batch` — accepts v3 schema
- `policy_loop/train.py::run_round` — wires reflection + emits v3 fields + composite reward
- `policy_loop/config.py` — `phase2_enabled`, `lambda_adherence`, `gamma_*`, `adherence_judge_*`, `reflection_max_tokens`
- `policy_loop/train.py::main()` — `--phase2`, `--lambda-adherence`, `--gamma-thinking`, `--gamma-strategy`

Per-sample retrieval remaining:

- `policy_loop/planner.py` — `bind_archive`, reshape `generate_strategies` to N·K independent retrieve + sample calls
- `policy_loop/train.py::run_round` — `planner.bind_archive(archive)` after planner init
- `policy_loop/train.py::build_tasks` — drop prior_strategies population (move to planner)
