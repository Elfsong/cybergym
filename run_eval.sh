#!/bin/bash
set -e

cd /home/nus_cisco_wp1/Projects/cybergym

# Load environment
source .env
export GEMINI_API_KEY
export PATH="$HOME/.local/bin:$PATH"

# ── Preflight checks ──────────────────────────────────────────────

if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY is not set. Check your .env file."
    exit 1
fi

# ── Install dependencies ──────────────────────────────────────────

echo "Syncing dependencies..."
uv pip install -e '.[server,dev]'

# ── Server configuration ─────────────────────────────────────────

SERVER_HOST=0.0.0.0
SERVER_PORT=8666
SERVER_LOG_DIR=./server_poc
SERVER_DB_PATH=./server_poc/poc.db
BINARY_DIR=./cybergym-server-data

mkdir -p "$SERVER_LOG_DIR"

# ── Start server in background ────────────────────────────────────

SERVER_CMD="uv run python3 -m cybergym.server \
    --host $SERVER_HOST \
    --port $SERVER_PORT \
    --log_dir $SERVER_LOG_DIR \
    --db_path $SERVER_DB_PATH"

if [ -d "$BINARY_DIR" ]; then
    echo "Binary directory found, using binary-only mode."
    SERVER_CMD="$SERVER_CMD --binary_dir $BINARY_DIR"
fi

echo "Starting CyberGym server on $SERVER_HOST:$SERVER_PORT ..."
$SERVER_CMD &
SERVER_PID=$!

# Shut down server on exit
cleanup() {
    echo ""
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$SERVER_PORT/docs" > /dev/null 2>&1; then
        echo "Server is ready."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died."
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:$SERVER_PORT/docs" > /dev/null 2>&1; then
    echo "ERROR: Server failed to start within 30 seconds."
    exit 1
fi

# ── Eval configuration ────────────────────────────────────────────

MODEL="gemini-3-flash-preview"
CYBERGYM_DATA_DIR=./cybergym_data/data
OUT_DIR=./eval_gemini_3_flash_preview
SERVER_IP=172.17.0.1
DIFFICULTY=level1
TIMEOUT=1200
MAX_ITER=100

TASKS=(
    "arvo:10341"  "arvo:10486"  "arvo:11033"  "arvo:11429"  "arvo:11657"
    "arvo:14232"  "arvo:14537"  "arvo:14821"  "arvo:18356"  "arvo:18562"
    "arvo:19100"  "arvo:19910"  "arvo:20050"  "arvo:20823"  "arvo:21342"
    "arvo:21579"  "arvo:21936"  "arvo:21984"  "arvo:22430"  "arvo:22560"
    "arvo:23433"  "arvo:23653"  "arvo:25341"  "arvo:25473"  "arvo:25561"
    "arvo:26325"  "arvo:26327"  "arvo:26952"  "arvo:27279"  "arvo:27812"
    "arvo:28191"  "arvo:28392"  "arvo:28462"  "arvo:28666"  "arvo:29827"
    "arvo:30099"  "arvo:30921"  "arvo:30999"  "arvo:31179"  "arvo:32356"
    "arvo:32785"  "arvo:32807"  "arvo:34116"  "arvo:3498"   "arvo:35172"
    "arvo:35410"  "arvo:3560"   "arvo:3818"   "arvo:42464"  "arvo:42957"
    "arvo:43414"  "arvo:44503"  "arvo:4451"   "arvo:45222"  "arvo:46082"
    "arvo:46615"  "arvo:46847"  "arvo:49638"  "arvo:50099"  "arvo:50414"
    "arvo:50629"  "arvo:51011"  "arvo:52465"  "arvo:5494"   "arvo:56156"
    "arvo:56726"  "arvo:57037"  "arvo:59056"  "arvo:59393"  "arvo:59418"
    "arvo:60037"  "arvo:60532"  "arvo:60557"  "arvo:61011"  "arvo:61111"
    "arvo:62388"  "arvo:62707"  "arvo:63622"  "arvo:63746"  "arvo:64849"
    "arvo:65209"  "arvo:65518"  "arvo:6581"   "arvo:66196"  "arvo:66426"
    "arvo:6857"   "arvo:7024"   "arvo:7538"   "arvo:8283"   "arvo:8615"
    "oss-fuzz:377642312"  "oss-fuzz:386128938"  "oss-fuzz:42535437"
    "oss-fuzz:42535447"   "oss-fuzz:42536646"   "oss-fuzz:42537014"
    "oss-fuzz:42537493"   "oss-fuzz:42537562"   "oss-fuzz:42537948"
    "oss-fuzz:42537998"
)

# ── Run evaluation ────────────────────────────────────────────────

mkdir -p "$OUT_DIR"

echo ""
echo "=========================================="
echo "Running ${#TASKS[@]} tasks with model $MODEL"
echo "=========================================="
echo ""

for TASK_ID in "${TASKS[@]}"; do
    echo "=========================================="
    echo "[$(date)] Running task: $TASK_ID"
    echo "=========================================="
    uv run python3 examples/agents/openhands/run.py \
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

echo "All ${#TASKS[@]} tasks completed. Results in $OUT_DIR/logs/"
