#!/bin/bash
set -e

cd /home/nus_cisco_wp1/Projects/cybergym

# Ensure Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Install server dependencies
echo "Syncing server dependencies..."
uv pip install -e '.[server]'

# Configuration
HOST=0.0.0.0
PORT=8666
LOG_DIR=./server_poc
DB_PATH=./server_poc/poc.db
BINARY_DIR=./cybergym-server-data

mkdir -p "$LOG_DIR"

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
