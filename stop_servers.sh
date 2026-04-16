#!/bin/bash
# Stop vLLM server (port 8000) and CyberGym server (port 8666)

echo "Stopping servers..."

for PORT in 8000 8666; do
    PIDS=$(sudo fuser "$PORT/tcp" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "Killing process(es) on port $PORT (PID: $PIDS)..."
        sudo fuser -k "$PORT/tcp" 2>/dev/null
    else
        echo "No process found on port $PORT."
    fi
done

echo "Done."
