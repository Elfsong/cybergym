
#!/bin/bash
# Launch vLLM server for Qwen3.5-122B-A10B on 8xA100 80GB
#
# Usage:
#   bash start_vllm_server.sh                  # default: 32K context, thinking mode
#   bash start_vllm_server.sh --max-model-len 65536   # override context length
#
# Prerequisites:
#   uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

export HF_HOME="/data/hf_home"

MODEL="Qwen/Qwen3.5-122B-A10B"
PORT=8000
TP=8
MAX_MODEL_LEN=65536     # conservative for 8xA100; bump to 65536 or 131072 if fits
MAX_NUM_SEQS=4         # max concurrent requests (matches parallel CyberGym containers)
GPU_MEMORY_UTILIZATION=0.92

vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --language-model-only \
    --dtype auto \
    --trust-remote-code \
    "$@"
