#!/bin/bash
# Launch vLLM server for Qwen3.5-27B on a subset of GPUs
#
# Usage:
#   bash start_qwen3_5_27b_server.sh
#   bash start_qwen3_5_27b_server.sh --max-model-len 65536
#
# Adjust CUDA_VISIBLE_DEVICES if running standalone.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH="${CYBERGYM_VENV:-.venv-mastermind}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

export HF_HOME="${HF_HOME:-/mnt/bn/tiktok-mm-5/aiic/users/mz.du/cybergym_data/.cache/huggingface}"

MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen3.5-27B}"
PORT="${QWEN_VLLM_PORT:-8001}"
TP="${QWEN_VLLM_TP:-8}"
MAX_MODEL_LEN="${QWEN_VLLM_MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${QWEN_VLLM_MAX_NUM_SEQS:-72}"
GPU_MEMORY_UTILIZATION="${QWEN_VLLM_GPU_MEMORY_UTILIZATION:-0.95}"

vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype auto \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    "$@"
