import glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

args = sys.argv[1:]
parallel = int(args[0])
total = int(args[1])
verbose = args[2] == "true"
model, base_url = args[3], args[4]
out_dir = args[5]
data_dir = args[6]
server_ip, server_port = args[7], args[8]
timeout_s, max_iter, max_output_tokens = args[9], args[10], args[11]
silent, difficulty = args[12], args[13]
tasks = args[14:]

log_dir = f"{out_dir}/logs"


def summarize_task(task_id, wall_time):
    """Parse trajectory to extract status, cost, and token usage."""
    task_norm = task_id.replace(":", "_")
    wt_str = f"{wall_time // 60}m{wall_time % 60:02d}s"

    candidates = glob.glob(os.path.join(log_dir, task_norm + "-*", "trajectory"))
    if not candidates:
        print(f"  {task_norm:<25} time: {wt_str:>7}  steps:    ?  NO_TRAJECTORY", flush=True)
        return "OTHER", 0.0, 0, 0, 0

    traj_path = max(candidates, key=os.path.getmtime)
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except Exception:
        print(f"  {task_norm:<25} time: {wt_str:>7}  steps:    ?  !  ERROR", flush=True)
        return "OTHER", 0.0, 0, 0, 0

    steps = len([e for e in data if e.get("action") and e.get("source") == "agent"])

    poc_status = "NO_SUBMIT"
    for i, item in enumerate(data):
        cmd = str(item.get("args", {}).get("command", ""))
        if "submit.sh" in cmd and "cat" not in cmd:
            if i + 1 < len(data):
                content = str(data[i + 1].get("content", ""))
                try:
                    json_start = content.find("{")
                    if json_start < 0:
                        continue
                    json_end = content.find("}", json_start)
                    if json_end < 0:
                        continue
                    result = json.loads(content[json_start : json_end + 1])
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

    markers = {"PASSED": "\u2713", "FAILED": "\u2717", "NO_SUBMIT": "\u2014"}
    marker = markers.get(poc_status, "?")

    cost = 0.0
    prompt_tokens = completion_tokens = cache_read_tokens = 0
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
    print(
        f"  {task_norm:<25} time: {wt_str:>7}  steps: {steps:>4}  cost: {cost_str:>8}"
        f"  prompt: {prompt_tokens:>8}  compl: {completion_tokens:>7}"
        f"  cache: {cache_read_tokens:>8}  {marker} {poc_status}",
        flush=True,
    )
    return poc_status, cost, prompt_tokens, completion_tokens, cache_read_tokens


def run_task(task_num, task_id):
    print(f"[{task_num}/{total}] [{datetime.now():%Y-%m-%d %H:%M:%S}] Starting: {task_id}", flush=True)
    start = time.monotonic()
    cmd = [
        os.path.expanduser("~/.local/bin/uv"), "run", "python3",
        "examples/agents/openhands/run.py",
        "--model", model,
        "--base_url", base_url,
        "--log_dir", log_dir,
        "--tmp_dir", f"{out_dir}/tmp",
        "--data_dir", data_dir,
        "--task_id", task_id,
        "--server", f"http://{server_ip}:{server_port}",
        "--timeout", timeout_s,
        "--max_iter", max_iter,
        "--max_output_tokens", max_output_tokens,
        "--silent", silent,
        "--difficulty", difficulty,
    ]
    stderr = None if verbose else subprocess.DEVNULL
    try:
        subprocess.run(cmd, stderr=stderr, timeout=int(timeout_s) + 300)
    except Exception as e:
        print(f"  [{task_num}/{total}] {task_id}: process error: {e}", flush=True)
    elapsed = int(time.monotonic() - start)
    return task_num, task_id, elapsed


# Run all tasks with bounded parallelism
results = []
with ThreadPoolExecutor(max_workers=parallel) as pool:
    futures = {
        pool.submit(run_task, i + 1, tid): tid
        for i, tid in enumerate(tasks)
    }
    for fut in as_completed(futures):
        task_num, task_id, elapsed = fut.result()
        result = summarize_task(task_id, elapsed)
        results.append(result)

# Tally results
pass_count = sum(1 for r in results if r[0] == "PASSED")
fail_count = sum(1 for r in results if r[0] == "FAILED")
total_cost = sum(r[1] for r in results)
total_prompt = sum(r[2] for r in results)
total_compl = sum(r[3] for r in results)
total_cache = sum(r[4] for r in results)
total_tokens = total_prompt + total_compl
other_count = len(results) - pass_count - fail_count

print("===========================================================")
print(f"All {total} tasks completed. Passed: {pass_count}  Failed: {fail_count}  Other: {other_count}")
print("-----------------------------------------------------------")
print(f"Total cost:              ${total_cost:.4f}")
print(f"Total prompt tokens:     {total_prompt}")
print(f"Total completion tokens: {total_compl}")
print(f"Total cache read tokens: {total_cache}")
print(f"Total tokens:            {total_tokens}")
print("-----------------------------------------------------------")
print(f"Results in {log_dir}/")
