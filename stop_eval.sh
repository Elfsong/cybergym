#!/bin/bash
# Stop all running CyberGym eval processes and clean up Docker containers.
#
# Usage:
#   ./stop_eval.sh          # kill everything
#   ./stop_eval.sh --dry-run  # show what would be killed

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# 1. Kill run_eval_*_tasks.py main processes
echo "=== Eval runner processes ==="
RUNNER_PIDS=$(pgrep -f 'run_eval.*tasks\.py' 2>/dev/null || true)
if [[ -n "$RUNNER_PIDS" ]]; then
    echo "$RUNNER_PIDS" | while read pid; do
        echo "  kill $pid: $(ps -p "$pid" -o args= 2>/dev/null | head -c 120)"
    done
    if ! $DRY_RUN; then
        echo "$RUNNER_PIDS" | xargs kill -9 2>/dev/null || true
        echo "  Killed $(echo "$RUNNER_PIDS" | wc -w) runner process(es)"
    fi
else
    echo "  None found"
fi

# 2. Kill run.py (openhands launcher) subprocesses
echo ""
echo "=== OpenHands launcher processes (run.py) ==="
LAUNCHER_PIDS=$(pgrep -f 'run\.py.*--task_id' 2>/dev/null || true)
if [[ -n "$LAUNCHER_PIDS" ]]; then
    echo "  Found $(echo "$LAUNCHER_PIDS" | wc -w) launcher process(es)"
    if ! $DRY_RUN; then
        echo "$LAUNCHER_PIDS" | xargs kill -9 2>/dev/null || true
        echo "  Killed"
    fi
else
    echo "  None found"
fi

# 3. Kill openhands.core.main agent processes
echo ""
echo "=== OpenHands agent processes ==="
AGENT_PIDS=$(pgrep -f 'openhands\.core\.main' 2>/dev/null || true)
if [[ -n "$AGENT_PIDS" ]]; then
    echo "  Found $(echo "$AGENT_PIDS" | wc -w) agent process(es)"
    if ! $DRY_RUN; then
        echo "$AGENT_PIDS" | xargs kill -9 2>/dev/null || true
        echo "  Killed"
    fi
else
    echo "  None found"
fi

# 4. Remove all openhands Docker containers
echo ""
echo "=== OpenHands Docker containers ==="
CONTAINERS=$(docker ps -a --filter "name=openhands-runtime" -q 2>/dev/null || true)
if [[ -n "$CONTAINERS" ]]; then
    COUNT=$(echo "$CONTAINERS" | wc -w)
    echo "  Found $COUNT container(s)"
    if ! $DRY_RUN; then
        echo "$CONTAINERS" | xargs docker rm -f 2>/dev/null || true
        echo "  Removed"
    fi
else
    echo "  None found"
fi

# 5. Summary
echo ""
if $DRY_RUN; then
    echo "DRY RUN complete. Re-run without --dry-run to execute."
else
    echo "Cleanup complete."
fi
