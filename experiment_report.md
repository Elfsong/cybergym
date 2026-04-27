# GRPO Training Experiment Report — Mastermind Planner LoRA

Run dates: 2026-04-27.
Setup: Tinker SDK + Qwen3.5-27B + LoRA rank=32 (planner) + OpenHands + vLLM TP=8 (executor) on local 8×A100 + CyberGym verify-agent-pocs (scoring).

## TL;DR

Across **5 consecutive 12-round GRPO training runs** with progressively more aggressive scaffolding fixes, **`pass_rate` (fraction of rollouts hitting milestone-7 verified PoC) dropped monotonically from R1 baseline through R3-R4 in every run**. The base (un-tuned) policy was the best policy at every checkpoint. Codex (gpt-5.4) diagnosed this as **survivor-biased GRPO**: `planner.py:_build_task_datums` was filtering APRIL-cancelled rollouts before computing per-task mean/std, so any strategy that led to slow (cancelled) executor behavior contributed zero gradient. Slow-strategy bias compounded each round. **Fix landed in commit `43c3c53`**: cancelled rollouts now stay in-group with their (low, near-zero) reward, so slow-strategy gets pushed to negative advantage. Run `1586c566` (started 2026-04-27 02:41 UTC+8) is the first run with the fix.

## Setup Common to All Runs

```
planner          Qwen/Qwen3.5-27B + LoRA rank=32 (Tinker cloud)
executor         openai/Qwen/Qwen3.5-27B  → local vLLM TP=8 :8001
                  --enable-prefix-caching --max-num-seqs 72 --max-model-len 65536
judge            same vLLM (only used when λ_adherence > 0)
loss             PPO clip ε=0.2 (lo=hi)
adv normalization "mean_only" (Dr.GRPO)
lr               1e-5 cosine, warmup 10%, floor 0.1
group K          8 rollouts/task
seed             42  (per-round RNG = seed + round_idx → deterministic task sample)
APRIL early-stop max_wall=2400s, threshold=0.80, K_min=5, min_wall=600s
docker_stagger   5s between Popen calls (dockerd contention floor)
cybergym         server://172.17.0.1:8666, level1 difficulty
```

Reward schedule (`reward.py:compute_reward`):
```
r = adherence · r_milestone[m] + λ · adherence + γ_t · f_think + γ_s · f_strat
r_milestone = (0, 0.5, 1.5, 2.5, 4.0, 5.5, 8.0, 12.0)
f_think     = min(n_think_tokens / 3000, 1)         rewards LONG thinking
f_strat     = max(0, 1 − n_strat_tokens / 500)      rewards SHORT strategy (since 5ef3282)
```

## Run History

| run_id | start (UTC+8) | parallel | batch | adherence λ | archive | γ_strat | reward fix? | rounds done | terminal pass_rate | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `90b2ebc4` | 13:11 | 32 | 48 | 0.5 | on | 0 | – | 1 | 0.06 | killed: pre-launch APRIL UnboundLocalError + 82% CRASH |
| `2b7eb258` | 13:19 | 32 | 48 | 0.5 | on | 0 | – | 2 | 0.042 | killed: R1=0.06, R2=0.042 — clear regression |
| `3b22894a` | 17:17 | 64 | 32 | 0.5 | on | 0 | – | 3 | 0.109 | killed: bigger throughput, but R1=0.141→R3=0.109 still monotonic ↓ |
| `e4a0ce10` | 21:13 | 64 | 32 | 0.0 (skip) | OFF | 0.1 | – | 4 | 0.082 | killed: ablation (no judge, no archive) — ↓ continued |
| `1586c566` | 02:41 (next day) | 64 | 32 | 0.0 (skip) | OFF | 0.1 | **YES** (43c3c53) | running | – | first run with survivor-bias fix |

## The Problem — Survivor-Biased GRPO

### Observed pattern (run `e4a0ce10`, the cleanest ablation)

| metric | R1 (baseline) | R2 (1 step) | R3 (2 steps) | R4 (3 steps) |
|---|---|---|---|---|
| `pass_rate` (m=7 / total) | 0.141 | 0.121 | 0.082 | 0.082 |
| `avg_milestone` | 2.21 | 1.84 | 1.91 | 1.58 |
| `surviving_groups` (≥1 valid) | 19 / 32 | 16 / 32 | 15 / 32 | 14 / 32 |
| `used` datums fed to GRPO | 147 | 121 | 97 | 104 |
| `loss / per_datum` | 0.4161 | 0.4286 | 0.7527 | 0.9164 |
| `adv std` | 1.52 | 1.22 | 2.12 | 2.11 |
| `mean_reward` (over `used`) | 5.253 | 5.364 | 5.597 | **4.919** |
| `degenerate` groups (reward all equal) | 0 / 19 | 0 / 16 | 0 / 15 | 0 / 14 |

### Diagnostic signature

Three signals had to be read together:
1. `pass_rate ↓` — fraction of full-success rollouts is dropping.
2. `mean_reward ↑` (R1→R3) — but reward averaged over the survivors goes **up**.
3. `used datums ↓` — fewer rollouts feed each gradient step.

Interpretation: **only the fast-completing rollouts shape the gradient, and the policy is being pushed toward whatever style produces the fastest-finishing rollouts, regardless of whether those rollouts succeed**. The slow but potentially-successful rollouts get APRIL-cancelled, are filtered before per-group mean/std, and contribute zero gradient. The reward-of-survivors metric inflates because the SIGTERM tail is silently censored.

### Positive feedback loop

```
GRPO step                              ┐
  ↓                                    │
planner generates wider/slower         │
strategies (whichever direction        │
the reward landscape pulled)           │
  ↓                                    │
executor takes longer per rollout      │
  ↓                                    │
more rollouts hit APRIL wall budget    │ ← runs away
and get SIGTERM'd (cancelled=True)     │
  ↓                                    │
those cancelled rollouts are FILTERED  │
out of GRPO group stats — zero         │
gradient signal for "slow strategy"    │
  ↓                                    │
only fast survivors update the planner │
  ↓                                    │
GRPO step                              ┘
```

Confirmed by R4 of `e4a0ce10`: `mean_reward` finally dropped (5.597 → 4.919) — even the survivors were now failing to reach their previous milestone level. The bias mechanism had run its course.

## Diagnosis Credit

Codex (gpt-5.4) called the survivor bias on first read, when given just the table above + the hypothesis list. Quoted recommendation:

> "Most likely root cause: **survivor-biased GRPO updates**, not just 'too much LR.' The tell is the combination of pass_rate down, avg_milestone down, mean_reward up, and used datums collapsing. That usually means the policy is getting worse on the full task distribution, but your optimizer only sees the subset of rollouts that still finish under APRIL. Timeouts/cancels are effectively disappearing instead of becoming negative evidence."

Earlier hypotheses ruled out by Codex's logic:
- `lr=1e-5 too high` — would slow the drift but not fix the biased target
- `batch_size=32 too small` — bigger batch makes APRIL censoring **worse**
- `γ_strategy=0.1 hurts` — magnitude too small to be the main driver
- `reward_compression="log1p"` — treats variance amplification, not data selection

## The Fix (commit `43c3c53`)

**File:** `dual_loops/planner.py`, function `_build_task_datums`

**Before:**
```python
for (strat, reward), is_cancelled in zip(strategies_with_rewards, cancelled_mask):
    if is_cancelled:
        n_cancelled_dropped += 1
        continue                              # ← cancelled rollouts dropped
    groups[strat.task_id].append((strat, reward))
```

**After:**
```python
for (strat, reward), is_cancelled in zip(strategies_with_rewards, cancelled_mask):
    if is_cancelled:
        n_cancelled_kept += 1
    groups[strat.task_id].append((strat, reward))   # ← all rollouts kept
```

The cancelled rollouts have:
- `strat.prompt` and `strat.tokens` — present (planner generated them successfully)
- `milestone = 0` (no trajectory → `score_milestones` defaults to 0)
- `reward = 0 + γ_strat · f_strat` = 0..0.1 (just the strategy-length term)

So a slow-strategy that always APRIL-cancels gets reward ≈ 0..0.1, while a successful strategy in the same group gets reward 0.5+ (m=1) up to 12.0 (m=7). Group mean is meaningful, slow gets negative advantage, PPO update pushes the policy away from slow strategies.

Side metric renamed in `summary` dict:
- `n_cancelled_dropped` → `n_cancelled_kept`

## Other Stabilizers Not Yet Applied

If R1-R4 of run `1586c566` still show monotonic decline, the next-priority interventions (Codex's tier 2):

1. **`advantage_normalization`: `mean_only` → `clipped_std`** — divide by `max(σ, 0.3)` to dampen the m=7 winner-take-all effect on advantages.
2. **`reward_compression`: `none` → `log1p`** — maps `r_milestone[7] = 12` → `2.56`, narrows the m=7 outlier blast radius.
3. **`learning_rate`: `1e-5` → `5e-6`** — smaller PPO step.

The order matters: `clipped_std` is a stat-level fix, `log1p` is a reward-level fix, `lr` is an optimizer-level fix. If the survivor-bias fix doesn't land R1-R4 reward stable, the next likely culprit is the m=7 outlier amplifying advantages via mean_only.

## Other Bugs Fixed Along The Way

| commit | issue |
|---|---|
| `1ecbb14` | `_run_single` raised `UnboundLocalError` on pre-launch APRIL aborts (variable `cancelled_by_stop` not yet bound). Fixed by hoisting init to before `try:` and classifying pre-launch aborts as cancelled. |
| `f7ecc1f` | 32-parallel rollouts overwhelmed dockerd (~82% CRASH at runtime-init). Fixed by 5s `_docker_rate_limit` between Popen calls + APRIL watchdog thread. |
| `c27337e` | APRIL early-stop max_wall 1800→2400, threshold 0.85→0.80. |
| `5ef3282` | `f_strat` flipped from saturating-up (rewards LONG) to linear-down (rewards SHORT) to counter the verbosity / safety-refusal-loop tail. |
| `ae69049` | Adherence judge skipped when λ=0 (judge had 60-80% imputation rate). Archive disabled to isolate GRPO from prior-strategy injection. |
| `43c3c53` | **Survivor-bias fix described above.** |

## Currently Running

Run `1586c566` started 2026-04-27 02:41 UTC+8. R1 baseline expected to match prior runs (`pass_rate ≈ 0.141`, `avg_milestone ≈ 2.21`) since same base policy + seed. R2 is the first round whose result is informative — if it matches R2 of `e4a0ce10` (0.121, dropping), survivor-bias was not the only issue. If it stabilizes or improves, the fix worked.

R1 metrics ETA: ~03:55 UTC+8 (32 min generation + 40 min execution + 3 min GRPO step).
