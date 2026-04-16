#!/bin/bash
# Launch vLLM server for Qwen3.5-27B on a subset of GPUs
#
# Usage:
#   bash start_qwen3_5_27b_server.sh
#   bash start_qwen3_5_27b_server.sh --max-model-len 65536
#
# Adjust CUDA_VISIBLE_DEVICES if running standalone.

source /home/nvidia/Projects/cybergym/.venv/bin/activate

export HF_HOME="/data/hf_home"

MODEL="Qwen/Qwen3.5-27B"
PORT=8001
TP=8
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.95

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
