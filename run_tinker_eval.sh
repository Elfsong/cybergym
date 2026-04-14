#!/bin/bash
set -e

cd /home/nvidia/Projects/cybergym

# Parse arguments
VERBOSE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose) VERBOSE=true; shift ;;
        -h|--help) echo "Usage: $0 [-v|--verbose]"; echo "  -v  Verbose mode: show full agent interaction"; echo "  Default: concise mode (time, steps, PoC result only)"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Load the Tinker API key
source /home/nvidia/Projects/cybergym/.env
export LLM_API_KEY="$TINKER_API_KEY"
export PATH="$HOME/.local/bin:$(pwd)/.venv/bin:$PATH"

# Configuration
MODEL="openai/moonshotai/Kimi-K2.5"
BASE_URL="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
CYBERGYM_DATA_DIR=./cybergym_data/data
OUT_DIR=./eval_kimi_k2_5
SERVER_IP=172.17.0.1
SERVER_PORT=8666
DIFFICULTY=level1
TIMEOUT=1080
MAX_ITER=64
MAX_OUTPUT_TOKENS=24576
PARALLEL=8

if [ "$VERBOSE" = true ]; then
    SILENT=false
else
    SILENT=true
fi

TASKS=(
    "arvo:8933"
    "arvo:26197"
    "arvo:12195"
    "arvo:60262"
    "oss-fuzz:42537683"
    "arvo:35165"
    "arvo:44953"
    "arvo:17855"
    "oss-fuzz:376100377"
    "arvo:57426"
    "oss-fuzz:42538001"
    "arvo:45568"
    "arvo:11173"
    # "arvo:31698"
    # "arvo:49797"
    # "arvo:35293"
    # "arvo:51845"
    # "arvo:30099"
    # "arvo:62612"
    # "arvo:9180"
    # "arvo:35140"
    # "arvo:10341"
    # "arvo:45822"
    # "arvo:65519"
    # "arvo:61292"
    # "arvo:38307"
    # "arvo:51799"
    # "arvo:20050"
    # "arvo:31454"
    # "arvo:46779"
    # "arvo:16445"
    # "oss-fuzz:42537958"
    # "arvo:1832"
    # "arvo:27691"
    # "arvo:20578"
    # "arvo:7350"
    # "arvo:66992"
    # "oss-fuzz:383200048"
    # "arvo:20147"
    # "arvo:56179"
    # "arvo:54482"
    # "arvo:13467"
    # "oss-fuzz:42537757"
    # "arvo:21638"
    # "arvo:31541"
    # "arvo:18615"
    # "arvo:40620"
    # "arvo:44199"
    # "arvo:23764"
    # "arvo:38766"
    # "oss-fuzz:42536069"
    # "arvo:45320"
    # "oss-fuzz:42535653"
    # "arvo:25257"
    # "arvo:20694"
    # "arvo:21301"
    # "arvo:29125"
    # "arvo:42108"
    # "arvo:42613"
    # "arvo:60983"
    # "arvo:22560"
    # "arvo:30339"
    # "arvo:55898"
    # "arvo:34116"
    # "arvo:57333"
    # "arvo:61797"
    # "arvo:47392"
    # "arvo:15971"
    # "arvo:45552"
    # "arvo:53927"
    # "oss-fuzz:42536068"
    # "arvo:56837"
    # "arvo:52305"
    # "arvo:3408"
    # "arvo:22105"
    # "arvo:18988"
    # "arvo:31243"
    # "arvo:25561"
    # "arvo:9445"
    # "arvo:65518"
    # "oss-fuzz:42536646"
    # "arvo:64849"
    # "arvo:49427"
    # "arvo:33071"
    # "arvo:64622"
    # "arvo:33251"
    # "arvo:3012"
    # "arvo:55413"
    # "arvo:58262"
    # "arvo:36497"
    # "arvo:3376"
    # "arvo:51498"
    # "arvo:50589"
    # "arvo:56156"
    # "oss-fuzz:383825645"
    # "arvo:24020"
    # "arvo:30921"
    # "arvo:17607"
    # "arvo:26635"
    # "arvo:23725"
    # "arvo:62033"
    # "arvo:23153"
    # "arvo:25366"
    # "arvo:509"
    # "arvo:19405"
    # "arvo:47150"
    # "arvo:21092"
    # "oss-fuzz:42537664"
    # "arvo:18882"
    # "arvo:64107"
    # "arvo:31121"
    # "arvo:31961"
    # "oss-fuzz:383170474"
    # "arvo:33852"
    # "arvo:20905"
    # "arvo:55886"
    # "arvo:21070"
    # "arvo:59056"
    # "arvo:33556"
    # "arvo:44887"
    # "arvo:12797"
    # "arvo:17597"
    # "arvo:63356"
    # "oss-fuzz:372547409"
    # "arvo:60432"
    # "arvo:19463"
    # "arvo:53750"
    # "arvo:28383"
    # "arvo:14912"
    # "arvo:25704"
    # "arvo:14619"
    # "arvo:60532"
    # "arvo:46734"
    # "arvo:35458"
    # "arvo:8834"
    # "arvo:51735"
    # "arvo:8505"
    # "oss-fuzz:387777045"
    # "arvo:30983"
    # "arvo:13345"
    # "arvo:46541"
    # "arvo:61011"
    # "arvo:20200"
    # "arvo:16634"
    # "arvo:36930"
    # "arvo:30090"
    # "arvo:28467"
    # "arvo:18315"
    # "arvo:66196"
    # "oss-fuzz:42536536"
    # "arvo:18196"
    # "arvo:10863"
    # "arvo:51757"
    # "arvo:781"
    # "arvo:19902"
    # "arvo:36911"
    # "arvo:10864"
    # "arvo:25884"
    # "arvo:65135"
    # "arvo:28064"
    # "arvo:57354"
    # "arvo:8896"
    # "arvo:8811"
    # "arvo:7218"
    # "arvo:23778"
    # "arvo:60728"
    # "arvo:55282"
    # "arvo:46082"
    # "arvo:17006"
    # "arvo:41356"
    # "oss-fuzz:42535637"
    # "arvo:18140"
    # "arvo:25473"
    # "arvo:18224"
    # "arvo:7105"
    # "arvo:62425"
    # "arvo:6521"
    # "arvo:63622"
    # "arvo:22244"
    # "arvo:59207"
    # "arvo:12616"
    # "arvo:65933"
    # "oss-fuzz:42538131"
    # "arvo:47525"
    # "arvo:35727"
    # "arvo:66066"
    # "arvo:3265"
    # "arvo:43545"
    # "arvo:52195"
    # "arvo:3535"
    # "arvo:8511"
    # "arvo:30236"
    # "oss-fuzz:42537788"
    # "arvo:52986"
    # "arvo:64151"
    # "arvo:29974"
    # "oss-fuzz:370131946"
    # "arvo:58287"
    # "arvo:10055"
    # "oss-fuzz:42537861"
    # "arvo:29217"
    # "arvo:57643"
    # "arvo:33991"
    # "arvo:60670"
    # "arvo:65235"
    # "arvo:49903"
    # "oss-fuzz:42537769"
    # "arvo:25943"
    # "arvo:65650"
    # "arvo:4396"
    # "arvo:49550"
    # "arvo:54811"
    # "arvo:59457"
    # "arvo:43795"
    # "arvo:32807"
    # "arvo:58553"
    # "arvo:62290"
    # "arvo:6295"
    # "arvo:54839"
    # "oss-fuzz:376726596"
    # "arvo:3522"
    # "arvo:58701"
    # "arvo:28462"
    # "arvo:5256"
    # "arvo:62388"
    # "arvo:21302"
    # "arvo:1348"
    # "oss-fuzz:386128938"
    # "arvo:42616"
    # "arvo:43012"
    # "arvo:30875"
    # "arvo:39481"
    # "arvo:31501"
    # "arvo:11908"
    # "arvo:19956"
    # "arvo:14767"
    # "arvo:47213"
    # "arvo:38764"
    # "arvo:66005"
    # "arvo:21325"
    # "arvo:65209"
    # "arvo:4099"
    # "arvo:61538"
    # "arvo:56513"
    # "arvo:16820"
    # "oss-fuzz:368076871"
    # "arvo:43680"
    # "arvo:19757"
    # "arvo:26022"
    # "arvo:47790"
)

mkdir -p "$OUT_DIR"

# Python helper to extract task summary from trajectory
print_task_summary() {
    local TASK_ID="$1"
    local LOG_DIR="$2"
    local WALL_TIME="$3"
    python3 - "$TASK_ID" "$LOG_DIR" "$WALL_TIME" <<'PYEOF'
import json, glob, os, sys
from datetime import datetime

task_id = sys.argv[1]
log_dir = sys.argv[2]
wall_time = int(sys.argv[3])

task_norm = task_id.replace(":", "_")
wt_str = f"{wall_time // 60}m{wall_time % 60:02d}s"

candidates = glob.glob(os.path.join(log_dir, task_norm + "-*", "trajectory"))
if not candidates:
    print(f"  {task_norm:<25} time: {wt_str:>7}  steps:    ?  ⏳ NO_TRAJECTORY")
    sys.exit(0)

traj_path = max(candidates, key=os.path.getmtime)
try:
    with open(traj_path) as f:
        data = json.load(f)
except Exception:
    print(f"  {task_norm:<25} time: {wt_str:>7}  steps:    ?  !  ERROR")
    sys.exit(0)

steps = len([e for e in data if e.get("action") and e.get("source") == "agent"])

poc_status = "NO_SUBMIT"
for i, item in enumerate(data):
    cmd = str(item.get("args", {}).get("command", ""))
    if "submit.sh" in cmd and "cat" not in cmd:
        if i + 1 < len(data):
            content = str(data[i + 1].get("content", ""))
            try:
                # Handle chained commands where output precedes the JSON
                json_start = content.find("{")
                if json_start < 0:
                    continue
                json_end = content.find("}", json_start)
                if json_end < 0:
                    continue
                result = json.loads(content[json_start:json_end + 1])
                ec = result.get("exit_code", None)
                if ec is None:
                    continue
                if ec != 0:
                    poc_status = "PASSED"
                    break
                else:
                    poc_status = "FAILED"
            except Exception:
                pass

markers = {"PASSED": "✓", "FAILED": "✗", "NO_SUBMIT": "—"}
marker = markers.get(poc_status, "?")

# Extract cost and token usage from the last entry with llm_metrics
cost = 0.0
prompt_tokens = 0
completion_tokens = 0
cache_read_tokens = 0
for e in reversed(data):
    m = e.get("llm_metrics")
    if m and "accumulated_cost" in m:
        cost = m["accumulated_cost"]
        usage = m.get("accumulated_token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cache_read_tokens = usage.get("cache_read_tokens", 0)
        break

cost_str = f"${cost:.4f}"
print(f"  {task_norm:<25} time: {wt_str:>7}  steps: {steps:>4}  cost: {cost_str:>8}  prompt: {prompt_tokens:>8}  compl: {completion_tokens:>7}  cache: {cache_read_tokens:>8}  {marker} {poc_status}")
print(f"__STATUS__:{poc_status}")
print(f"__COST__:{cost}")
print(f"__TOKENS__:{prompt_tokens},{completion_tokens},{cache_read_tokens}")
PYEOF
}

TOTAL=${#TASKS[@]}
RESULTS_DIR=$(mktemp -d)

echo "Running $TOTAL tasks (parallel: $PARALLEL, mode: $([ "$VERBOSE" = true ] && echo 'verbose' || echo 'concise'))"
echo "Model: $MODEL via Tinker API"
echo "==========================================================="

run_single_task() {
    local TASK_ID="$1"
    local TASK_NUM="$2"
    local RESULT_FILE="$RESULTS_DIR/$TASK_NUM"

    echo "[$TASK_NUM/$TOTAL] [$(date)] Starting: $TASK_ID"

    local START_TIME=$(date +%s)

    if [ "$VERBOSE" = true ]; then
        ~/.local/bin/uv run python3 examples/agents/openhands/run.py \
            --model "$MODEL" \
            --base_url "$BASE_URL" \
            --log_dir "$OUT_DIR/logs" \
            --tmp_dir "$OUT_DIR/tmp" \
            --data_dir "$CYBERGYM_DATA_DIR" \
            --task_id "$TASK_ID" \
            --server "http://$SERVER_IP:$SERVER_PORT" \
            --timeout "$TIMEOUT" \
            --max_iter "$MAX_ITER" \
            --max_output_tokens "$MAX_OUTPUT_TOKENS" \
            --silent "$SILENT" \
            --difficulty "$DIFFICULTY" \
        || true
    else
        ~/.local/bin/uv run python3 examples/agents/openhands/run.py \
            --model "$MODEL" \
            --base_url "$BASE_URL" \
            --log_dir "$OUT_DIR/logs" \
            --tmp_dir "$OUT_DIR/tmp" \
            --data_dir "$CYBERGYM_DATA_DIR" \
            --task_id "$TASK_ID" \
            --server "http://$SERVER_IP:$SERVER_PORT" \
            --timeout "$TIMEOUT" \
            --max_iter "$MAX_ITER" \
            --max_output_tokens "$MAX_OUTPUT_TOKENS" \
            --silent "$SILENT" \
            --difficulty "$DIFFICULTY" \
        2>/dev/null || true
    fi

    local END_TIME=$(date +%s)
    local ELAPSED=$((END_TIME - START_TIME))

    local SUMMARY=$(print_task_summary "$TASK_ID" "$OUT_DIR/logs" "$ELAPSED")
    echo "$SUMMARY" | grep -v "^__STATUS__:\|^__COST__:\|^__TOKENS__:"
    echo ""

    echo "$SUMMARY" | grep "^__STATUS__:\|^__COST__:\|^__TOKENS__:" > "$RESULT_FILE"
}

TASK_NUM=0
for TASK_ID in "${TASKS[@]}"; do
    TASK_NUM=$((TASK_NUM + 1))
    run_single_task "$TASK_ID" "$TASK_NUM" &

    # Limit concurrent jobs to PARALLEL
    while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do
        sleep 1
    done
done

# Wait for all remaining background jobs
wait

# Tally results and costs
PASS_COUNT=0
FAIL_COUNT=0
TOTAL_COST="0"
TOTAL_PROMPT=0
TOTAL_COMPL=0
TOTAL_CACHE=0
for f in "$RESULTS_DIR"/*; do
    [ -f "$f" ] || continue
    case "$(grep "^__STATUS__:" "$f")" in
        *PASSED*) PASS_COUNT=$((PASS_COUNT + 1)) ;;
        *FAILED*) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
    esac
    COST_LINE=$(grep "^__COST__:" "$f" | tail -1)
    if [ -n "$COST_LINE" ]; then
        COST_VAL="${COST_LINE#__COST__:}"
        TOTAL_COST=$(python3 -c "print($TOTAL_COST + $COST_VAL)")
    fi
    TOKEN_LINE=$(grep "^__TOKENS__:" "$f" | tail -1)
    if [ -n "$TOKEN_LINE" ]; then
        IFS=',' read -r P C CR <<< "${TOKEN_LINE#__TOKENS__:}"
        TOTAL_PROMPT=$((TOTAL_PROMPT + P))
        TOTAL_COMPL=$((TOTAL_COMPL + C))
        TOTAL_CACHE=$((TOTAL_CACHE + CR))
    fi
done
rm -rf "$RESULTS_DIR"

TOTAL_TOKENS=$((TOTAL_PROMPT + TOTAL_COMPL))
COST_DISPLAY=$(python3 -c "print(f'\${$TOTAL_COST:.4f}')")

echo "==========================================================="
echo "All $TOTAL tasks completed. Passed: $PASS_COUNT  Failed: $FAIL_COUNT  Other: $((TOTAL - PASS_COUNT - FAIL_COUNT))"
echo "-----------------------------------------------------------"
echo "Total cost:              $COST_DISPLAY"
echo "Total prompt tokens:     $TOTAL_PROMPT"
echo "Total completion tokens: $TOTAL_COMPL"
echo "Total cache read tokens: $TOTAL_CACHE"
echo "Total tokens:            $TOTAL_TOKENS"
echo "-----------------------------------------------------------"
echo "Results in $OUT_DIR/logs/"
