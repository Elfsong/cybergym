# How to Run CyberGym Evaluation

## 1. Start Servers

```bash
cd ~/Projects/cybergym
nohup bash start_minimax_2_5_server.sh > vllm_server.log 2>&1 &
nohup bash start_qwen3_5_27b_server.sh > vllm_server.log 2>&1 &

nohup bash start_cybergym_server.sh > cybergym_server.log 2>&1 &

# Verify
curl http://localhost:8000/v1/models
curl http://localhost:8666/query-poc
```

## 2. Run Eval

```bash
cd ~/Projects/cybergym && nohup uv run run_eval_minimax_2_5_tasks.py &>/dev/null &
cd ~/Projects/cybergym && nohup uv run run_eval_qwen3_5_27b_tasks.py &>/dev/null &
```

Each run creates an isolated output directory: `eval_minimax_m2_5_<uuid>/` with a `run.log` inside.

## 3. Monitor

```bash
uv run monitor.py --log-dir /data/cybergym_data/cybergym-eval-data/eval_minimax_m2_5_<uuid>/logs
```

## 4. View Results

```bash
uv run trajectory_viewer.py --logs_dir /data/cybergym_data/cybergym-eval-data/eval_minimax_m2_5_<uuid>/logs
```

## 5. Stop

```bash
./stop_eval.sh              # kill all processes + Docker containers
./stop_eval.sh --dry-run    # preview only
```

## vLLM Pressure Check

```bash
curl -s http://localhost:8000/metrics | grep -E 'vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total)' | grep -v '#'
```
