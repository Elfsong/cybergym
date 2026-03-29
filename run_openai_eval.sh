#!/bin/bash
set -e

cd /home/nus_cisco_wp1/Projects/cybergym

# Load API keys
source /home/nus_cisco_wp1/Projects/cybergym/.env
export OPENAI_API_KEY
export PATH="$HOME/.local/bin:$PATH"

# Configuration
MODEL="gpt-5.4-mini"
CYBERGYM_DATA_DIR=./cybergym_data/data
OUT_DIR=./eval_gpt_5_4_mini
SERVER_IP=172.17.0.1
SERVER_PORT=8666
DIFFICULTY=level1
TIMEOUT=1200
MAX_ITER=100

# 100 randomly selected tasks
TASKS=(
    "arvo:47101"
    "arvo:3938"
    "arvo:24993"
    "arvo:1065"
    "arvo:10400"
    "arvo:368"
    "oss-fuzz:42535201"
    "oss-fuzz:42535468"
    "oss-fuzz:370689421"
    "oss-fuzz:385167047"
)

mkdir -p "$OUT_DIR"

for TASK_ID in "${TASKS[@]}"; do
    echo "=========================================="
    echo "[$(date)] Running task: $TASK_ID"
    echo "=========================================="
    ~/.local/bin/uv run python3 examples/agents/openhands/run.py \
        --model "$MODEL" \
        --log_dir "$OUT_DIR/logs" \
        --tmp_dir "$OUT_DIR/tmp" \
        --data_dir "$CYBERGYM_DATA_DIR" \
        --task_id "$TASK_ID" \
        --server "http://$SERVER_IP:$SERVER_PORT" \
        --timeout "$TIMEOUT" \
        --max_iter "$MAX_ITER" \
        --silent false \
        --difficulty "$DIFFICULTY" \
    || echo "Task $TASK_ID failed, continuing..."
    echo ""
done

echo "All tasks completed. Results in $OUT_DIR/logs/"
