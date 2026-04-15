# How to Run CyberGym Evaluation

## Prerequisites

- Docker running
- 8x A100 80GB GPUs (for vLLM serving)
- `uv` installed (`~/.local/bin/uv`)
- Python venv at `.venv/`
- Benchmark data at `/data/cybergym_data/cybergym-benchmark-data/data`
- Server binaries at `/data/cybergym_data/cybergym-server-data`

## Step 1: Start the vLLM Server (background)

```bash
nohup bash start_vllm_server.sh > vllm_server.log 2>&1 &
```

Serves `MiniMaxAI/MiniMax-M2.5` on port `8000` with TP=4, DP=2. Wait until the log shows the server is ready before proceeding.

Check readiness:

```bash
curl http://localhost:8000/v1/models
```

## Step 2: Start the CyberGym Server (background)

```bash
nohup bash start_cybergym_server.sh > cybergym_server.log 2>&1 &
```

Runs the CyberGym task server on port `8666`. Requires Docker.

Check readiness:

```bash
curl http://localhost:8666/health
```

## Step 3: Run the Evaluation (background)

```bash
nohup uv run python3 run_eval_tasks.py > /data/cybergym_data/cybergym-eval-data/eval_minimax_m2_5/run.log 2>&1 &
```

Results go to `/data/cybergym_data/cybergym-eval-data/eval_minimax_m2_5/logs/`.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `openai/MiniMaxAI/MiniMax-M2.5` | Model name |
| `--base-url` | `http://localhost:8000/v1` | vLLM API endpoint |
| `--out-dir` | `/data/cybergym_data/cybergym-eval-data/eval_minimax_m2_5` | Output directory |
| `--data-dir` | `/data/cybergym_data/cybergym-benchmark-data/data` | Benchmark data directory |
| `--tasks-file` | `TASKS` (in script dir) | Task list file |
| `--parallel` | `36` | Number of parallel tasks |
| `--max-iter` | `72` | Max agent iterations per task |
| `--max-output-tokens` | `8192` | Max LLM output tokens per call |
| `--timeout` | `1800` | Task timeout in seconds |
| `--difficulty` | `level1` | Task difficulty filter |
| `-v` / `--verbose` | off | Show full agent interaction output |

Example with custom settings:

```bash
nohup uv run python3 run_eval_tasks.py --parallel 24 --max-iter 96 --tasks-file FULL_TASKS > run.log 2>&1 &
```

## Task Files

- `TASKS` — current evaluation task list (300 tasks)
- `FULL_TASKS` — all available tasks (1507 tasks)

One task ID per line. Supports `#` comments and blank lines.

## Step 4: Monitor Progress (foreground)

```bash
uv run python3 monitor.py
```

Opens a TUI dashboard showing real-time task status, cost, and progress.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--log-dir` | `<out-dir>/logs` | Log directory to monitor |
| `--out-dir` | `/data/cybergym_data/cybergym-eval-data/eval_minimax_m2_5` | Output directory |
| `--tasks` | `TASKS` (in script dir) | Task list file |
| `--max-iter` | `72` | Max iterations per task |
| `--refresh` | `30.0` | Refresh interval in seconds |

Keybindings in the TUI:

| Key | Action |
|-----|--------|
| `Enter` | View trajectory (detailed interaction log) |
| `q` | Quit / Back (in trajectory view) |
| `Escape` | Back (in trajectory view) |
| `r` | Refresh / Reload |
| `g` / `G` | Scroll to top / bottom (in trajectory view) |
| `p` | Sort by passed first |
| `s` | Sort by steps |
| `c` | Sort by cost |
| `n` | Sort by name |
| `t` | Sort by status |

## Quick Start (all-in-one)

```bash
# Start servers in background
nohup bash start_vllm_server.sh > vllm_server.log 2>&1 &
nohup bash start_cybergym_server.sh > cybergym_server.log 2>&1 &

# Wait for servers to be ready, then run eval
cd ~/Projects/cybergym && nohup uv run python3 run_eval_tasks.py &>/dev/null &  

# Monitor
uv run python3 monitor.py
```

## Checking vLLM Pressure

```bash
curl -s http://localhost:8000/metrics | grep -E 'vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total)' | grep -v '#'
```

Key indicators:
- `num_requests_waiting > 0` sustained — reduce `--parallel`
- `kv_cache_usage_perc > 0.8` — reduce `--parallel` or `--max-output-tokens`
- `num_preemptions_total` increasing — reduce `--parallel` immediately
