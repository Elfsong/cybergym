#!/usr/bin/env bash
# Sequential 5-group launcher for Claude Code + Claude Opus 4.7 on the
# 100-task seed-42 sample. Each group is 20 tasks at parallel=4 with
# timeout=2400s / effort=low; a 5-hour gap between group starts keeps each
# group inside its own Claude Max rolling 5-hour window.
#
# Wall-clock plan (relative to T=0 = script start):
#   T+0h    launch group 0 (20 tasks, ~50-90 min wall)
#   T+5h    launch group 1
#   T+10h   launch group 2
#   T+15h   launch group 3
#   T+20h   launch group 4
#   ~T+22h  group 4 finishes, all done
#
# Each group writes to its own out-dir so partial results are recoverable
# even if a later group fails.

set -euo pipefail

ROOT=/home/nvidia/Projects/cybergym
LOG_ROOT=/lp-dev/cybergym_data/pagent

cd "$ROOT"

GROUPS=(0 1 2 3 4)
GAP_SECONDS=$((5 * 3600))   # 5 hours between group starts

t0=$(date +%s)
echo "[plan] Master scheduler started at $(date -Iseconds), expected total ~22h"

for gi in "${GROUPS[@]}"; do
    target=$((t0 + gi * GAP_SECONDS))
    now=$(date +%s)
    if (( now < target )); then
        wait_s=$((target - now))
        echo "[plan] sleeping ${wait_s}s until group ${gi} launch at $(date -Iseconds -d @${target})"
        sleep "${wait_s}"
    fi

    out_dir="/data/cybergym_data/cybergym-eval-data/eval_cc_opus47_100_g${gi}"
    log="${LOG_ROOT}/run_cc_opus47_100_g${gi}.log"

    echo "[launch] group ${gi} → ${out_dir}  log: ${log}"
    nohup uv run python run_eval_claude_code_tasks.py \
        --model claude-opus-4-7 \
        --tasks-file "TASKS_EVAL_100_seed42_g${gi}" \
        --parallel 4 \
        --out-dir "${out_dir}" \
        --max-iter 72 \
        --timeout 2400 \
        --effort low \
        > "${log}" 2>&1 &
    pid=$!
    disown
    echo "[launch] group ${gi} PID=${pid}"
done

echo "[plan] all 5 groups dispatched (last one launched at $(date -Iseconds))"
