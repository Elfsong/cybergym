
#!/bin/bash
# Launch vLLM server on 8xA100 80GB
#
# Usage:
#   bash start_vllm_server.sh                  # default: 32K context, thinking mode
#   bash start_vllm_server.sh --max-model-len 65536   # override context length
#
# Prerequisites:
#   uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

export HF_HOME="/data/hf_home"

# MODEL="Qwen/Qwen3.5-122B-A10B"
MODEL="MiniMaxAI/MiniMax-M2.5"
PORT=8000
TP=4
DP=2
MAX_MODEL_LEN=131072     # reduced for 4-GPU TP to fit in memory
MAX_NUM_SEQS=16         # max concurrent requests (matches parallel CyberGym containers)
GPU_MEMORY_UTILIZATION=0.94

vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-expert-parallel \
    --reasoning-parser minimax_m2 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --dtype auto \
    --trust-remote-code \
    "$@"
