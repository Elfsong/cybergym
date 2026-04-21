# Experience Loop — Design Notes

This directory hosts the memory-side of Mastermind. The policy loop (`../policy_loop/`) trains the planner's LoRA weights; the experience loop accumulates `(strategy, milestone, adherence, insight)` records in an archive retrieved on the next round's planner prompt. The two loops are coupled — this directory is the memory substrate.

## Mechanisms

1. **Archive** — append-only JSONL store of per-rollout records. Each record carries `task_id`, `strategy`, `milestone`, `adherence`, `insight`, `n_thinking_tokens`, `n_strategy_tokens`, `trajectory_path`, `run_id`, `round`, `group_id`, `timestamp` (see `policy_loop/archive.py::_V2_FIELDS`).

2. **Tournament retrieval** — `Archive.retrieve(task_id)` filters records to the same task with `milestone >= archive_min_milestone` (3 by default), then repeatedly samples a size-`t` tournament (4) and picks the highest-milestone winner without replacement, until `n` slots (3) are filled. Returns `list[dict{strategy, milestone, insight}]`.

3. **Per-sample retrieval** — within a single training round, each of the `K` samples for a task performs an independent tournament draw. When `|eligible_pool| > n`, the `K` prompts almost always see different prior subsets; this is a hard guarantee of intra-group input variance.

4. **Reflection judge** — after each trajectory completes, a frozen base Qwen3.5-27B (served on the same vLLM instance as the executor) inspects `(strategy, trajectory_summary)` and emits in a single call both an integer adherence score `a ∈ [0, 1]` and a short actionable `insight` string. Adherence gates the milestone reward; insight is stored in the archive and surfaced in the prompt on the next encounter.

## Reward

```
r = a · r_milestone + λ · a + γ_t · f_think + γ_s · f_strat
```

Defaults: `λ = 0.5`, `γ_t = γ_s = 0`. The length terms are optional and normally off; they exist as ablation knobs and as a residual variance source when milestones coincide across a K-group.

## What lives in this directory

```
experience_loop/
├── PLAN.md                        # this doc
├── __init__.py
├── adherence.py                   # score_reflection_batch: (adherence, insight) via vLLM judge
└── prompts/
    └── adherence.txt              # judge prompt (0-10 rubric + insight requirements)
```

The archive itself (`archive.py`) lives under `policy_loop/` because it is referenced from the main training loop.

## Cold start

Round 1 starts with an empty archive, so `retrieve()` returns `[]` for every task and the planner prompt contains only the task description. From round 2 onwards, the archive fills round-over-round (records from a round become retrievable in the next round). Tasks that were never sampled in earlier rounds remain cold until they first appear; this is the same behavior as any round-1 task, so no special-casing is needed.

## Retrieval-quality parameters

| Knob | Default | Notes |
|---|---|---|
| `archive_n` | 3 | Number of priors shown per prompt. Going above ~5 makes the prompt long enough that the planner starts to dilute its attention. |
| `archive_tournament_size` | 4 | Selection pressure. `t=1` is uniform random; `t=|P|` is deterministic top-n. |
| `archive_min_milestone` | 3 | Quality filter. Milestone 3 = submitted a PoC, which is where server-side feedback starts. Lowering to 2 admits "PoC-constructed but never submitted" records (noisier); raising to 5 keeps only near-successes (sparser). |

## Reflection-judge parameters

| Knob | Default | Notes |
|---|---|---|
| `adherence_judge_model` | `Qwen/Qwen3.5-27B` | Base model (no LoRA). Self-judging with the trained LoRA would make the reward non-stationary and introduce self-reinforcement bias, so the judge must be frozen. |
| `adherence_judge_base_url` | `http://localhost:8001/v1` | The same vLLM instance as the executor; runs during the scoring phase when the executor is idle. |
| `reflection_max_tokens` | 8192 | Qwen3.5-27B's "Thinking Process" consumes ~5k tokens before it emits the final XML tags; 8192 leaves headroom. |
| `adherence_max_traj_chars` | 8000 | The trajectory summary fed to the judge. Full trajectories are too long; the summarizer keeps first/last assistant messages, all `submit.sh` calls + responses, file reads / edits, and `think` actions. |
| `judge_parallel` | 64 | Async semaphore bounding concurrent adherence-judge calls (matches vLLM's `max_num_seqs`). Sibling of `planner_parallel` / `executor_parallel`. |

## Ablations we care about

| Ablation | How to run | What it isolates |
|---|---|---|
| No archive | `--no-archive` | Isolates the archive's retrieval contribution; reflection judge still runs (adherence still gates milestone reward). |
| No adherence bonus | `--lambda-adherence 0.0` | Keeps the gate `a · r_milestone` but drops the standalone bonus. |
| Per-task (not per-sample) retrieval | internal flag in planner | Isolates the intra-group-variance contribution of independent draws. |
| No reflection | (not exposed yet — would force `a = 1.0`, skip judge calls) | Isolates adherence gating altogether. |

## Notes on stability

- The reflection judge is pure inference and never part of the gradient; changing `adherence_judge_model` does not retrain anything.
- KL penalty is not currently wired (Tinker's `importance_sampling` loss does not accept one). Stability comes from LoRA low-rank updates, a cosine LR schedule peaking at 2e-5, AdamW weight decay 0.01, and a global grad-norm clip at 1.0.
- Planner weights are snapshotted as `π_old` at the start of each round; mini-batch GRPO substeps inside a round use importance sampling against this snapshot.
