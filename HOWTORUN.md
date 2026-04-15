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
nohup bash run_vllm_eval.sh > eval.log 2>&1 &
```

Runs all tasks with 8-way parallelism. Results go to `./eval_minimax_m2_5/logs/`.

Options:

- `-v` / `--verbose` — show full agent interaction output

## Step 4: Monitor Progress (foreground)

```bash
uv run monitor.py
```

Opens a TUI dashboard showing real-time task status, cost, and progress.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--log-dir` | auto-detected from script | Log directory to monitor |
| `--script` | `run_vllm_eval.sh` | Script to extract task list from |
| `--max-iter` | `64` | Max iterations per task |
| `--refresh` | `30.0` | Refresh interval in seconds |

Keybindings in the TUI:

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Refresh |
| `p` | Sort by passed first |
| `s` | Sort by steps |
| `c` | Sort by cost |
| `n` | Sort by name |
| `t` | Sort by status |

## Quick Start (all-in-one)

```bash
# Terminal 1 — start servers in background
nohup bash start_vllm_server.sh > vllm_server.log 2>&1 &
nohup bash start_cybergym_server.sh > cybergym_server.log 2>&1 &

# Wait for servers to be ready, then run eval
nohup bash run_vllm_eval.sh > eval.log 2>&1 &

# Terminal 2 — monitor
python3 monitor.py
```
