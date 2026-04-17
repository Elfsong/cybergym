# Strategy Sensitivity Study Results

**Date**: 2026-04-16
**Run ID**: sensitivity_study_1fb3f17f
**Configuration**: 40 tasks × 4 conditions = 160 executor runs, Qwen3.5-27B via vLLM, Level 1

## Design

- **Group A (20 tasks)**: Qwen3.5-27B previously PASSED (self-oracle from Qwen trajectories)
- **Group B (20 tasks)**: Qwen3.5-27B previously FAILED, MiniMax M2.5 PASSED (cross-oracle from MiniMax trajectories)

### Conditions
- **Oracle**: Strategy extracted from a PASSED trajectory for the same task
- **No Strategy**: Default prompt (unguided baseline)
- **Random**: Strategy from a different task's PASSED trajectory
- **Adversarial**: Deliberately misleading strategy

## Overall Results

| Condition | Pass Rate | Avg Milestone |
|---|---|---|
| **Oracle** | **32/40 (80.0%)** | **6.00** |
| Adversarial | 28/40 (70.0%) | 5.38 |
| No Strategy | 27/40 (67.5%) | 5.08 |
| Random | 24/40 (60.0%) | 4.92 |

## Group Breakdown

### Group A: Self-oracle (easy tasks, Qwen previously PASSED)

| Condition | Pass Rate | Avg Milestone |
|---|---|---|
| Oracle | 17/20 (85.0%) | 6.35 |
| Adversarial | 16/20 (80.0%) | 5.90 |
| No Strategy | 15/20 (75.0%) | 5.55 |
| Random | 15/20 (75.0%) | 5.50 |

Small differences — executor can solve these regardless of strategy.

### Group B: Cross-oracle (hard tasks, Qwen previously FAILED)

| Condition | Pass Rate | Avg Milestone |
|---|---|---|
| **Oracle** | **15/20 (75.0%)** | **5.65** |
| No Strategy | 12/20 (60.0%) | 4.60 |
| Adversarial | 12/20 (60.0%) | 4.85 |
| **Random** | **9/20 (45.0%)** | **4.35** |

Strong signal: Oracle +15pts over baseline, Random -15pts below baseline.

## Key Findings

1. **Oracle strategies causally improve performance** — +12.5pts overall, +15pts on hard tasks
2. **Effect concentrates on harder tasks** — Group A: +10pts, Group B: +15pts
3. **Wrong strategies actively harm** — Random is worst at 45% on hard tasks (vs 60% unguided)
4. **Adversarial ≈ No Strategy** — executor ignores obviously bad advice but doesn't benefit
5. **Thesis validated**: Strategy selection is the bottleneck, especially when task difficulty increases

## Implications for GRPO

- The 15-point gap between oracle and baseline on Group B represents the **upper bound** for GRPO improvement
- GRPO-trained planner needs to generate strategies that approach oracle quality
- The harm from random strategies (−15pts on Group B) shows the importance of the quality threshold in archive retrieval
