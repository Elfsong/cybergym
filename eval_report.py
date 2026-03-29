#!/usr/bin/env python3
"""Print evaluation progress report for a given eval logs directory."""

import argparse
import json
import os
import glob
import re
import sys


def parse_task_list(eval_script_path):
    """Extract the TASKS array from a bash eval script."""
    tasks = []
    with open(eval_script_path) as f:
        in_tasks = False
        for line in f:
            line = line.strip()
            if line.startswith("TASKS=("):
                in_tasks = True
                continue
            if in_tasks:
                if line == ")":
                    break
                match = re.match(r'"([^"]+)"', line)
                if match:
                    tasks.append(match.group(1))
    return tasks


def get_poc_status(traj_path):
    """Parse a trajectory file and return (steps, poc_status)."""
    if not os.path.exists(traj_path):
        return "?", "IN_PROGRESS"

    try:
        with open(traj_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "?", "ERROR"

    steps = len([a for a in data if a.get("action")])
    poc_status = "NO_SUBMIT"

    for i, item in enumerate(data):
        cmd = str(item.get("args", {}).get("command", ""))
        if "submit.sh" in cmd and "cat" not in cmd:
            if i + 1 < len(data):
                content = str(data[i + 1].get("content", ""))
                try:
                    result = json.loads(content.strip())
                    ec = result.get("exit_code", None)
                    if ec is not None and ec != 0:
                        return steps, "PASSED"
                    elif ec == 0:
                        poc_status = "FAILED"
                except (json.JSONDecodeError, ValueError):
                    pass

    return steps, poc_status


def scan_logs(logs_dir):
    """Scan logs directory and return deduplicated results keyed by task name."""
    priority = {"PASSED": 4, "FAILED": 3, "NO_SUBMIT": 2, "IN_PROGRESS": 1, "ERROR": 0}
    results = {}

    for d in glob.glob(os.path.join(logs_dir, "*/")):
        dirname = os.path.basename(d.rstrip("/"))
        parts = dirname.rsplit("-", 1)
        task_name = parts[0] if len(parts) == 2 and len(parts[1]) == 32 else dirname

        traj_path = os.path.join(d, "trajectory")
        steps, poc_status = get_poc_status(traj_path)

        if task_name not in results or priority.get(poc_status, 0) > priority.get(results[task_name][1], 0):
            results[task_name] = (steps, poc_status)

    return results


def print_report(logs_dir, tasks=None):
    """Print the formatted report."""
    markers = {"PASSED": "✓", "FAILED": "✗", "NO_SUBMIT": "—", "IN_PROGRESS": "⏳", "ERROR": "!"}
    results = scan_logs(logs_dir)

    print(f"{'#':<4} {'Task':<25} {'Steps':>6}  {'Status'}")
    print("=" * 55)

    if tasks:
        # Ordered mode: show tasks in script order with pending placeholders
        task_names_norm = [t.replace(":", "_") for t in tasks]
        shown = set()
        for i, (task, tn) in enumerate(zip(tasks, task_names_norm)):
            if tn in results:
                steps, poc_status = results[tn]
                marker = markers.get(poc_status, "?")
                print(f"{i+1:<4} {tn:<25} {str(steps):>6}  {marker} {poc_status}")
                shown.add(tn)
            else:
                print(f"{i+1:<4} {tn:<25}     —  . PENDING")
        # Show any extra tasks in logs not in the script
        extras = sorted(set(results.keys()) - shown)
        for tn in extras:
            steps, poc_status = results[tn]
            marker = markers.get(poc_status, "?")
            print(f"{'?':<4} {tn:<25} {str(steps):>6}  {marker} {poc_status}")
        total = len(tasks)
    else:
        # Unordered mode: show all tasks found in logs, sorted by name
        for i, tn in enumerate(sorted(results.keys())):
            steps, poc_status = results[tn]
            marker = markers.get(poc_status, "?")
            print(f"{i+1:<4} {tn:<25} {str(steps):>6}  {marker} {poc_status}")
        total = len(results)

    # Summary
    passed = sum(1 for s, st in results.values() if st == "PASSED")
    failed = sum(1 for s, st in results.values() if st == "FAILED")
    no_sub = sum(1 for s, st in results.values() if st == "NO_SUBMIT")
    in_prog = sum(1 for s, st in results.values() if st == "IN_PROGRESS")
    started = len(results)

    print(f"\n{'='*55}")
    if tasks:
        print(f"Progress: {started}/{total} tasks started ({total - started} pending)")
    else:
        print(f"Total: {started} tasks")
    print(f"PASSED: {passed} | FAILED: {failed} | NO_SUBMIT: {no_sub} | IN_PROGRESS: {in_prog}")
    if passed + failed > 0:
        print(f"Pass rate (of submitted): {passed}/{passed+failed} = {passed/(passed+failed)*100:.1f}%")

    if tasks:
        task_names_norm = [t.replace(":", "_") for t in tasks]
        for i, tn in enumerate(task_names_norm):
            if tn not in results or results[tn][1] == "IN_PROGRESS":
                print(f"Currently on: #{i+1} {tasks[i]}")
                break


def main():
    parser = argparse.ArgumentParser(description="Print evaluation progress report")
    parser.add_argument("--logs_dir", required=True, help="Path to the eval logs directory")
    parser.add_argument("--eval_script", help="Path to the eval bash script (optional, for task ordering and pending status)")
    args = parser.parse_args()

    if not os.path.isdir(args.logs_dir):
        print(f"Error: logs directory not found: {args.logs_dir}", file=sys.stderr)
        sys.exit(1)

    tasks = None
    if args.eval_script:
        if not os.path.isfile(args.eval_script):
            print(f"Error: eval script not found: {args.eval_script}", file=sys.stderr)
            sys.exit(1)
        tasks = parse_task_list(args.eval_script)
        if not tasks:
            print(f"Warning: no tasks found in {args.eval_script}, showing all logs", file=sys.stderr)
            tasks = None

    print_report(args.logs_dir, tasks)


if __name__ == "__main__":
    main()
