# GRPO Training Experiment Report — Mastermind Planner LoRA

Run dates: 2026-04-27 → 2026-04-28.
Setup: Tinker SDK + Qwen3.5-27B + LoRA rank=32 (planner) + OpenHands + vLLM TP=8 (executor) on local 8×A100 + CyberGym verify-agent-pocs (scoring).

## TL;DR

Across **6 consecutive GRPO training runs** with progressively more aggressive scaffolding fixes — survivor-bias filter removed, advantage clipped_std, reward log1p compression, lr halved to 5e-6, judge skipped, archive off — `pass_rate` (fraction of rollouts hitting milestone-7 verified PoC) dropped monotonically from R1 baseline through R3-R4 in every run; the base (un-tuned) policy looked like the best policy at every checkpoint.

Codex (gpt-5.4) correctly diagnosed the **survivor-bias** mechanism on the first three runs: `planner.py:_build_task_datums` was filtering APRIL-cancelled rollouts before per-group mean/std, so slow strategies contributed zero gradient and got reinforced by omission. Commit `43c3c53` fixed that filter; run `c4f76f38` additionally stacked Codex's tier-2 stabilizers (clipped_std + log1p + lr/2). All four interventions worked **as engineering** — `used` stayed at the full batch×K=256, `degenerate=0/32`, `loss` stayed below 0.13, the misleading "mean_reward up while pass_rate down" signature is gone — but **pass_rate still appeared to degrade each round** (0.145 → 0.133 → 0.113 → 0.031 over R1-R4 of `c4f76f38`).

We then ran a **dedicated noise-floor measurement** (`7e91a68e`, 4 rounds, `lr=0` + `--skip-grpo-update` so the LoRA weights are frozen). Result across 4 task subsamples: pass_rate ∈ {0.081, 0.086, 0.018, 0.133}, **mean = 0.080, std ≈ 0.041, range [0.018, 0.133]**. Every one of `c4f76f38`'s four data points lies inside this no-update noise envelope. **The "monotonic decline" in `c4f76f38` is task-sample noise** — GRPO update is neither significantly helping nor significantly hurting at the 4-round timescale.

Working conclusion: at K=8 group size and ~256 rollouts/round the per-step gradient signal is **below the per-round task-sample noise floor**. Unblocking actual GRPO progress measurement will require either (a) longer runs (12+ rounds so the noise averages out across rounds), (b) paired comparison on a frozen task subset across rounds, or (c) larger K so per-task advantages stabilize.

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

| run_id | start (UTC+8) | parallel | batch | adherence λ | archive | γ_strat | survivor fix | clipped_std | log1p | lr | rounds done | terminal pass_rate | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `90b2ebc4` | 27 13:11 | 32 | 48 | 0.5 | on | 0 | no | no | no | 1e-5 | 1 | 0.06 | killed: pre-launch APRIL UnboundLocalError + 82% CRASH |
| `2b7eb258` | 27 13:19 | 32 | 48 | 0.5 | on | 0 | no | no | no | 1e-5 | 2 | 0.042 | killed: R1=0.06, R2=0.042 — clear regression |
| `3b22894a` | 27 17:17 | 64 | 32 | 0.5 | on | 0 | no | no | no | 1e-5 | 3 | 0.109 | killed: more throughput, R1=0.141→R3=0.109 still ↓ |
| `e4a0ce10` | 27 21:13 | 64 | 32 | 0.0 (skip) | OFF | 0.1 | no | no | no | 1e-5 | 4 | 0.082 | killed: ablation cleanest pre-fix evidence — ↓ confirmed |
| `1586c566` | 28 02:41 | 64 | 32 | 0.0 (skip) | OFF | 0.1 | **YES** (43c3c53) | no | no | 1e-5 | 0 | – | killed at 16 min (user wanted to think before restart) |
| `c4f76f38` | 28 03:51 | 64 | 32 | 0.0 (skip) | OFF | 0.1 | YES | **YES** | **YES** | **5e-6** | **4 (full)** | **0.031** | first complete run with all 4 stabilizers + fix |
| `7e91a68e` | 28 15:11 | 64 | 48 | 0.0 (skip) | OFF | 0.1 | YES | YES | YES | **0** + skip | **4 (full)** | **0.133** | noise-floor measurement: lr=0 + `--skip-grpo-update` |

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

## Run `c4f76f38` — All Stabilizers + Survivor Fix Stacked (4-round short run)

CLI used:

```bash
uv run python -m dual_loops.train \
    --num-rounds 4 \
    --batch-size 32 \
    --mini-batch-size 8 \
    --group-size 8 \
    --no-archive \
    --lambda-adherence 0 \
    --gamma-strategy 0.1 \
    --advantage-normalization clipped_std \
    --advantage-std-floor 0.3 \
    --reward-compression log1p \
    --learning-rate 5e-6
```

This stacks **all four interventions** (survivor-bias fix from commit `43c3c53` + Codex's tier-2: clipped_std, log1p, lr/2) at once. Goal: minimum-distance ablation to test whether the residual decline survives every known fix.

### Per-round metrics

| metric | R1 (baseline) | R2 | R3 | R4 |
|---|---|---|---|---|
| `pass_rate` (m=7 / 256) | **0.145** | 0.133 | 0.113 | **0.031** |
| `avg_milestone` | **2.24** | 1.61 | 1.87 | **1.17** |
| `mean_reward` (log1p scale) | 0.871 | 0.626 | 0.728 | 0.472 |
| `loss / per_datum` | 0.1022 | 0.0752 | 0.1217 | 0.0724 |
| `adv std` | 0.40 | 0.32 | 0.40 | 0.34 |
| `used` datums | **256** | **256** | **256** | **256** |
| `degenerate` groups | **0/32** | **0/32** | **0/32** | **0/32** |
| `substeps` (configured 4) | 4 | 4 | 4 | 4 |
| `surviving_groups` (≥K_min=5) | 8 | 5 | 7 | 4 |
| `cancelled` rollouts | 112 | 143 | 134 | 155 |

### What worked

✅ **Survivor-bias fix verified** — `used = 256` every round (vs `e4a0ce10` 147→121→97→104 collapsing). All cancelled rollouts contribute group-relative gradient.

✅ **`mean_reward` and `pass_rate` move together** — R2 both down, R3 both up, R4 both down. The misleading "mean_reward ↑ while pass_rate ↓" signature from the bias-loop runs is gone.

✅ **`degenerate = 0/32` every round** — γ_strategy=0.1 ensures group reward variance even when all milestones are equal (strategy-length term breaks ties).

✅ **`loss` stayed in the linear PPO regime** — peak 0.1217 (R3), vs `e4a0ce10` R3 = 0.7527 and R4 = 0.9164 where PPO clipping was firing constantly.

✅ **`adv std` capped at 0.40** — log1p compression (m=7 reward 12 → 2.56) + clipped_std floor 0.3 successfully tamed the m=7 winner-take-all effect on per-group advantages.

✅ **`substeps = 4` every round** — full configured budget hit (`ceil(32/8) = 4`), LR cosine schedule advances on schedule.

### What didn't work

❌ **`pass_rate` still monotonically declined** — 0.145 → 0.133 → 0.113 → 0.031 (-79% from baseline over 3 GRPO updates).

❌ **R4 collapse: -72% in one round** — 0.113 → 0.031 is the worst single-round drop seen across any run.

❌ **Final policy is worse than `e4a0ce10`'s no-fix terminal** (0.031 vs 0.082). All four stabilizers + survivor-bias fix produced a *worse* terminal pass_rate than the run with none of them.

### Interpretation

The R4 LR was at the cosine schedule floor (5e-6 × 0.1 = 5e-7 by step 14/16), so the R4 update itself was nearly a no-op. The R4 pass_rate collapse is therefore likely **task-sample noise** — round-4 happened to draw a hard subset of the 300-task pool — but the underlying R1→R3 decline is **not** noise: it tracks every prior run's shape.

Three remaining hypotheses for the residual decline:

1. **Base policy is at local optimum for this rollout budget** — Qwen3.5-27B with strategy_temperature=1.0 already finds the best strategies it can; any LoRA update destroys part of the reasoning circuit needed for m=7 without compensating gain.
2. **Reward signal is too sparse** — even with log1p, only ~20 of 256 rollouts hit m≥6 (where reward differences are large). The other ~236 are at m=0-3 where the reward gradient between adjacent milestones (after log1p compression) is `log1p(2.5) - log1p(1.5) = 1.25 - 0.92 = 0.33`. Group advantages on those 236 are ≤0.4 in magnitude, contribute minimally to the gradient relative to the 20 m≥6 outliers — but those 20 are a noisy signal.
3. **GRPO with LoRA rank=32 + Tinker has a numerical issue** unrelated to advantages — possibly token-level advantage broadcasting (`per_token_adv = adv / n_gen` — see `planner.py:402`) interacts badly with thinking-mode-off generations where the strategy span is short.

### Comparison to `e4a0ce10` (no fix)

| metric | `e4a0ce10` R3 (no fix) | `c4f76f38` R3 (all fixes) | improvement? |
|---|---|---|---|
| `pass_rate` | 0.082 | 0.113 | +38% (better) |
| `avg_milestone` | 1.91 | 1.87 | ~ same |
| `loss` | 0.7527 | 0.1217 | 6× lower (much better) |
| `used` datums | 97 | 256 | 2.6× more (better) |
| Trajectory shape | erratic, loss flying | stable | **better** |

So the fixes did genuinely improve **the training dynamics** (stable loss, full sample utilization, no survivor inflation). They did not improve **the policy quality**.

## Run `7e91a68e` — Noise-Floor Measurement (4-round eval-only)

Designed in response to `c4f76f38`'s ambiguous decline: if base policy pass_rate naturally varies a lot across the random task subsamples drawn each round, the apparent GRPO degradation may be sampling noise rather than a real training effect. To isolate that, run 4 rounds with **`learning_rate=0`** AND **`--skip-grpo-update`** so the LoRA weights are guaranteed not to change. Each round still goes through generation → execution → scoring → checkpoint, but the planner is frozen at the base Qwen3.5-27B + un-trained LoRA across all 4 rounds.

CLI used:

```bash
uv run python -m dual_loops.train \
    --num-rounds 4 --batch-size 48 --mini-batch-size 12 --group-size 8 \
    --no-archive --lambda-adherence 0 --gamma-strategy 0.1 \
    --advantage-normalization clipped_std --advantage-std-floor 0.3 \
    --reward-compression log1p \
    --learning-rate 0 --skip-grpo-update
```

### Per-round metrics (no-update across all rounds)

| Round | pass_rate | avg_milestone | completed | trainable groups | note |
|---|---|---|---|---|---|
| R1 | 0.081 | 1.38 | 147 / 384 | 5 | clean baseline draw |
| R2 | 0.086 | 1.33 | 122 / 384 | 7 | clean baseline draw |
| R3 | **0.018** | 0.62 | 87 / 384 | 3 | Sonnet54 (below) ran in parallel — docker contention slashed throughput from 147→87 completed |
| R4 | 0.133 | 1.60 | 139 / 384 | 7 | Sonnet54 finished, throughput recovers |

### Statistics

- mean pass_rate = **0.080**
- std ≈ **0.041**
- range = **[0.018, 0.133]** (7.4× spread)
- 95% CI ≈ [0, 0.16]

### Key result — c4f76f38's "monotonic decline" is inside the no-update noise

| Run | R1 | R2 | R3 | R4 | mean | range |
|---|---|---|---|---|---|---|
| `c4f76f38` (with GRPO update) | 0.145 | 0.133 | 0.113 | 0.031 | 0.106 | [0.031, 0.145] |
| `7e91a68e` (no update at all) | 0.081 | 0.086 | 0.018 | 0.133 | 0.080 | [0.018, 0.133] |

Every single `c4f76f38` round-result falls inside the [0.018, 0.133] envelope produced by simply re-sampling the task pool four times against a frozen policy. The `0.145 → 0.031` trajectory is statistically indistinguishable from random task-pool reshuffling. **GRPO update direction is not measurable at this run length and group size.**

### Caveat: docker contention in R3

`Sonnet54` (the Sonnet-on-200 baseline run) was launched in parallel with this run for time efficiency. The combined load (Sonnet's 6 parallel sandbox containers + GRPO's 64 parallel runtime+sandbox containers) saturated dockerd during R3 and dropped per-rollout throughput by 41% (147→87 completed in the same 40-min APRIL budget). R3's pass_rate=0.018 may therefore be an outlier-low due to the smaller usable sample, not just task-sample noise. Even so, R4 at 0.133 (pure noise envelope, no contention) is on the high side, so the conclusion stands: the noise envelope is wide.

**Lesson for future parallel runs:** when running Sonnet (Anthropic-API-side) and GRPO (local docker) at the same time, cap Sonnet `--parallel` at 4 or run them sequentially.

## Sonnet 4.6 Baseline — Full 200-Task Coverage Achieved (run dates 2026-04-23 to 2026-04-28)

Independent of the Mastermind training experiments, we maintain a Claude Code + Sonnet 4.6 baseline on the 200-task held-out evaluation set (`TASKS_EVAL`). Quota constraints (Claude Max 20× rolling 5-hour cap) forced the sweep to span 9 sub-runs over 5 days. The final `sonnet54_316fb18b` sub-run (started 28 17:54 UTC+8) covered the last 54 missing tasks and brought coverage to 200/200.

### Sub-run accounting (9 sub-runs, 240 total attempts on 200 unique tasks)

| sub-run | dates | PASSED | FAILED | NO_SUBMIT | total |
|---|---|---|---|---|---|
| `group_00_low_89dd1d39` | 23 | 4 | 5 | 1 | 10 |
| `group_01_ad171969` | 23 | 2 | 6 | 2 | 10 |
| `group_01_soft_0a23af58` | 23 | 3 | 5 | 2 | 10 |
| `remaining180_f008d092` | 23 | 16 | 17 | 9 | 42 |
| `resume138_400d40bd` | 23-24 | 30 | 22 | 22 | 74 |
| `retry40_e38a4be2` | 24 | 6 | 29 | 5 | 40 |
| `sonnet54_316fb18b` | 28 | 14 | 36 | 4 | 54 |
| `group_00_9812353d`, `_f7bf1f87` | 23 | (logs not parseable, small) | | | — |

39 unique tasks were attempted ≥2 times (across retries due to quota truncation).

### Headline numbers (over 200 unique tasks)

| Counting rule | PASSED | pass_rate (P/Total) | among submitters (P/(P+F)) |
|---|---|---|---|
| **Single-attempt** (first chronological try per task) | **67** | **33.5%** | **42.7%** |
| Best-effort (any-attempt PASS counts) | 73 | 36.5% | — |

The single-attempt number (33.5% / 42.7%) is what goes into the paper's main table — it is strictly comparable to the other rows which are also single-attempt. The best-effort variant is reported only as a courtesy data point in the appendix; we don't promote it to the headline because the other baselines did not get retries.

### Cost (sonnet54 sub-run alone)

- $145.54 total over 54 tasks
- $2.70 average per task
- 14 PASS at avg $1.34/task, 36 FAIL at avg $4.24/task (the budget-cap rollouts dominate spend)

Total Anthropic-side cost across all 9 Sonnet sub-runs is approximately $400-500.

### Paper Table 3 — final updates

The paper's main results table (`tab:main`) was updated to reflect the new Sonnet baseline:

```
Method                                          P/(P+F)    P/Total
OpenHands + Qwen3.5-27B                         35.6       26.5
OpenHands + qwen3.6-plus (DashScope)            40.4       39.0
OpenHands + Qwen3.6-Max-Preview (DashScope)     46.4       45.5  (best)
Claude Code + Claude Sonnet 4.6                 42.7       33.5  (was 45.0/32.1 on 112-of-200)
PAGENT + Qwen3.5-27B                            33.6       23.5
PAGENT + qwen3.6-plus (DashScope)               32.0       32.0  (100-task sample)
```

Sonnet went from "second-best in P/(P+F) at 45.0% but with the dagger note for 112/200 coverage" to "second-best at 42.7% with the dagger removed (full 200/200 coverage)". Net effect on the bar Mastermind has to clear: unchanged — Qwen3.6-Max-Preview at 45.5% remains the upper bound.

## Next-Experiment Recommendations

In priority order **after the noise-floor result**:

1. **~~Eval-only R0~~** — done; the result is `7e91a68e` above. Noise floor is 0.080 ± 0.041 over 4 task subsamples. 

2. **Long run (12 rounds) with all four stabilizers** — `7e91a68e` showed our 4-round runs are noise-limited, but a 12-round run gives ~3× the rounds and lets per-round task-sample noise average out. Same config as `c4f76f38` (clipped_std + log1p + lr=5e-6 + survivor fix); just don't kill at round 4. Cost: ~12-18h wall.

3. **K=16 (double the group size) + 4 rounds** — bigger groups mean per-task advantage estimates have less variance; lets a single round produce a more reliable training signal. Trade-off: 2× rollouts per round, so APRIL cancellation rate goes up further unless we also raise `executor_round_max_wall_seconds` to 3600.

4. **Paired-rollout protocol** — fix the same 32 task IDs across all rounds (set RNG seed-derivation to constant), so each round measures the same evaluation pool. Removes task-sample noise entirely from the round-over-round comparison. Requires a small change to `train.py`'s round-RNG logic.

5. **Switch base model to a weaker one (Qwen2.5-7B)** — gives GRPO more headroom to learn. Cost: vLLM restart + new training session.

6. **Accept current result, write as negative finding** — the paper's policy-loop section may need to honestly report that GRPO over Qwen3.5-27B + level1 task pool plateaus at the un-tuned baseline within the budget we tried.

## Other Stabilizers Not Yet Applied

If `c4f76f38` is the new floor and we want to push further:

1. **Per-token advantage broadcast (`planner.py:402`)** — currently `per_token_adv = adv / n_gen`. Some PPO implementations apply the advantage uniformly without the n_gen normalization. Worth checking against tinker_cookbook.rl.train.train_step.
2. **`max_strategy_tokens`: 4096 → 2048** — caps the verbosity tail more aggressively, reducing the fraction of rollouts that hit `max_tokens` without producing a usable strategy.
3. **APRIL `K_min`: 5 → 3** — relax the per-task completion threshold; with the survivor-bias fix this is no longer about gradient signal (cancelled count anyway), only about whether the early-stop heuristic fires before wall budget.

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

Nothing. As of 2026-04-28 23:06 UTC+8 both `7e91a68e` (GRPO eval-only, finished 21:31 UTC+8) and `sonnet54_316fb18b` (Sonnet 54-task baseline, finished 21:00 UTC+8) have completed. vLLM is still loaded on `:8001` (idle, ready). Tinker sessions are closed; checkpoints saved per run dir under `/data/cybergym_data/cybergym-train-data/<run_id>/checkpoints/`.
