#!/usr/bin/env bash
# Wait for the running PAGENT + Qwen3.6-Max-Preview remaining-100 orchestrator
# to finish, then exec the user's dual_loops.train run.
#
# Watches PID 4128060 (orchestrator) and the run.log "=== Done:" sentinel as a
# belt-and-suspenders check. Once the run completes, sources .env via the same
# parser the launchers use (so DASHSCOPE/TINKER/HF tokens are exported), then
# launches the training command in the background with nohup + disown so this
# wrapper can exit cleanly.

set -euo pipefail

ROOT=/home/nvidia/Projects/cybergym
WATCH_PID=4128060
WATCH_LOG=/lp-dev/cybergym_data/pagent/run_qwen36_maxpreview_remaining100.log
TRAIN_LOG=/lp-dev/cybergym_data/pagent/dual_loops_train_postpagent.log

cd "$ROOT"

echo "[wait] watching orchestrator PID=$WATCH_PID and log $WATCH_LOG"
while kill -0 "$WATCH_PID" 2>/dev/null; do
    sleep 30
done
echo "[wait] orchestrator exited; double-checking log sentinel"

# Belt-and-suspenders: confirm the orchestrator declared done
if ! grep -q "=== Done: 100 ok" "$WATCH_LOG" 2>/dev/null; then
    echo "[wait] WARNING: PID gone but no '=== Done: 100 ok' in log; aborting" >&2
    echo "[wait]   tail of log:"
    tail -20 "$WATCH_LOG" >&2
    exit 1
fi
echo "[wait] orchestrator finished cleanly"

# Load .env via the same Python parser the launchers use (bash `source` does
# not robustly handle every line shape we've seen in this .env file).
eval "$(uv run python -c '
import os, shlex
from pathlib import Path
for line in Path("/home/nvidia/Projects/cybergym/.env").read_text().splitlines():
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s: continue
    k, _, v = s.partition("=")
    print(f"export {k.strip()}={shlex.quote(v.strip().strip(chr(34)).strip(chr(39)))}")
')"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
    echo "[wait] ERROR: TINKER_API_KEY missing after .env load" >&2
    exit 2
fi

echo "[launch] starting dual_loops.train -> $TRAIN_LOG"
nohup uv run python -m dual_loops.train \
    --num-rounds 12 \
    --batch-size 32 \
    --mini-batch-size 8 \
    --group-size 8 \
    --validation-batch-size 32 \
    --validation-samples-per-task 8 \
    --validation-every 1 \
    --no-archive \
    --lambda-adherence 0 \
    --gamma-strategy 0.1 \
    --advantage-normalization clipped_std \
    --advantage-std-floor 0.3 \
    --reward-compression log1p \
    --learning-rate 5e-6 \
    --max-strategy-tokens 2048 \
    --planner-parallel 64 \
    --executor-parallel 64 \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
disown
echo "[launch] dual_loops.train started, PID=$TRAIN_PID"
echo "[launch] log: $TRAIN_LOG"
