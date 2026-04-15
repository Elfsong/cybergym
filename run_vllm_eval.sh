#!/bin/bash
set -e

cd /home/nvidia/Projects/cybergym

# Parse arguments
VERBOSE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose) VERBOSE=true; shift ;;
        -h|--help) echo "Usage: $0 [-v|--verbose]"; echo "  -v  Verbose mode: show full agent interaction"; echo "  Default: concise mode (time, steps, PoC result only)"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# vLLM serves locally — no external API key needed
export LLM_API_KEY="EMPTY"
export PATH="$HOME/.local/bin:$(pwd)/.venv/bin:$PATH"

# Configuration
# MODEL="openai/Qwen/Qwen3.5-122B-A10B/thinking"
MODEL="openai/MiniMaxAI/MiniMax-M2.5"
BASE_URL="http://localhost:8000/v1"
CYBERGYM_DATA_DIR=/data/cybergym_data/cybergym-benchmark-data/data
OUT_DIR=./eval_minimax_m2_5
SERVER_IP=172.17.0.1
SERVER_PORT=8666
DIFFICULTY=level1
TIMEOUT=1800
MAX_ITER=64
MAX_OUTPUT_TOKENS=65536
PARALLEL=8

if [ "$VERBOSE" = true ]; then
    SILENT=false
else
    SILENT=true
fi

TASKS_FILE="$(dirname "$0")/TASKS"
if [[ ! -f "$TASKS_FILE" ]]; then
    echo "Error: TASKS file not found: $TASKS_FILE" >&2
    exit 1
fi
mapfile -t TASKS < <(grep -v '^\s*$\|^\s*#' "$TASKS_FILE")

mkdir -p "$OUT_DIR"

TOTAL=${#TASKS[@]}

echo "Running $TOTAL tasks (parallel: $PARALLEL, mode: $([ "$VERBOSE" = true ] && echo 'verbose' || echo 'concise'))"
echo "Model: $MODEL via vLLM ($BASE_URL)"
echo "==========================================================="

# Use Python to manage parallel execution and result tallying
python3 "$(dirname "$0")/run_eval_tasks.py" \
    "$PARALLEL" "$TOTAL" "$VERBOSE" "$MODEL" "$BASE_URL" \
    "$OUT_DIR" "$CYBERGYM_DATA_DIR" "$SERVER_IP" "$SERVER_PORT" \
    "$TIMEOUT" "$MAX_ITER" "$MAX_OUTPUT_TOKENS" "$SILENT" "$DIFFICULTY" \
    "${TASKS[@]}"
