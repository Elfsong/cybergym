#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH="${CYBERGYM_VENV:-.venv-mastermind}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Ensure Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Install server dependencies
echo "Syncing server dependencies..."
uv pip install -e '.[server]'

# Configuration
HOST="${CYBERGYM_SERVER_HOST:-0.0.0.0}"
PORT="${CYBERGYM_SERVER_PORT:-8666}"
LOG_DIR="${CYBERGYM_SERVER_LOG_DIR:-./server_poc}"
DB_PATH="${CYBERGYM_SERVER_DB_PATH:-${LOG_DIR}/poc.db}"
BINARY_DIR="${CYBERGYM_SERVER_DATA_DIR:-/mnt/bn/tiktok-mm-5/aiic/users/mz.du/cybergym-server-data}"

mkdir -p "$LOG_DIR"

# Kill existing server on the same port
if command -v fuser > /dev/null 2>&1; then
    EXISTING_PID=$(fuser "$PORT/tcp" 2>/dev/null || true)
    if [ -n "$EXISTING_PID" ]; then
        echo "Killing existing process on port $PORT (PID $EXISTING_PID)..."
        kill "$EXISTING_PID" 2>/dev/null || true
        sleep 1
    fi
fi

# Build server command
CMD="uv run python3 -m cybergym.server \
    --host $HOST \
    --port $PORT \
    --log_dir $LOG_DIR \
    --db_path $DB_PATH"

# Use binary-only mode if the binary directory exists
if [ -d "$BINARY_DIR" ]; then
    echo "Binary directory found, using binary-only mode."
    CMD="$CMD --binary_dir $BINARY_DIR"
fi

echo "Starting CyberGym server on $HOST:$PORT ..."
exec $CMD
