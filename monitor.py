#!/usr/bin/env python3
"""CyberGym Task Monitor — a fancy TUI for watching eval progress in real-time."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ProgressBar,
    RichLog,
    Static,
)


@dataclass
class TaskState:
    task_id: str
    dir_name: str
    status: str = "PENDING"  # PENDING, RUNNING, PASSED, FAILED, NO_SUBMIT, ERROR, TIMEOUT
    step: int = 0
    max_iter: int = 64
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_tokens: int = 0
    start_time: str = ""
    last_action: str = ""
    wall_seconds: int = 0
    submit_count: int = 0


def parse_trajectory(traj_path: Path) -> dict:
    """Parse a trajectory file and extract progress info."""
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except Exception:
        return {}

    agent_steps = [e for e in data if e.get("source") == "agent" and e.get("action")]
    step_count = len(agent_steps)

    # Last action description
    last_action = ""
    if agent_steps:
        last = agent_steps[-1]
        last_action = (last.get("message") or last.get("action", ""))[:80]

    # PoC status
    poc_status = "RUNNING"
    submit_count = 0
    for i, item in enumerate(data):
        cmd = str(item.get("args", {}).get("command", ""))
        if "submit.sh" in cmd and "cat" not in cmd:
            submit_count += 1
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

    # Cost / tokens from last llm_metrics
    cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    cache_tokens = 0
    for e in reversed(data):
        m = e.get("llm_metrics")
        if m and "accumulated_cost" in m:
            cost = m["accumulated_cost"]
            usage = m.get("accumulated_token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cache_tokens = usage.get("cache_read_tokens", 0)
            break

    # Start time
    start_time = ""
    if data:
        start_time = data[0].get("timestamp", "")[:19]

    return {
        "steps": step_count,
        "status": poc_status,
        "cost": cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_tokens": cache_tokens,
        "last_action": last_action,
        "start_time": start_time,
        "submit_count": submit_count,
    }


def scan_log_dir(log_dir: Path, task_list: list[str], max_iter: int) -> list[TaskState]:
    """Scan the log directory and build task states."""
    # Build map: task_id_normalized -> dir
    existing_dirs = {}
    if log_dir.exists():
        for d in log_dir.iterdir():
            if d.is_dir():
                # dir name format: arvo_8933-<uuid>
                task_norm = d.name.rsplit("-", 1)[0] if "-" in d.name else d.name
                existing_dirs[task_norm] = d

    states = []
    for task_id in task_list:
        task_norm = task_id.replace(":", "_")
        state = TaskState(task_id=task_id, dir_name=task_norm, max_iter=max_iter)

        if task_norm not in existing_dirs:
            state.status = "PENDING"
            states.append(state)
            continue

        task_dir = existing_dirs[task_norm]
        traj_path = task_dir / "trajectory"

        # Check args.json for start info
        args_path = task_dir / "args.json"
        if args_path.exists():
            try:
                with open(args_path) as f:
                    args_data = json.load(f)
                state.max_iter = args_data.get("agent_args", {}).get("max_iter", max_iter)
            except Exception:
                pass

        if not traj_path.exists():
            # Dir exists but no trajectory yet — task is starting up
            state.status = "STARTING"
            states.append(state)
            continue

        info = parse_trajectory(traj_path)
        if not info:
            state.status = "ERROR"
            states.append(state)
            continue

        state.step = info["steps"]
        state.status = info["status"]
        state.cost = info["cost"]
        state.prompt_tokens = info["prompt_tokens"]
        state.completion_tokens = info["completion_tokens"]
        state.cache_tokens = info["cache_tokens"]
        state.last_action = info["last_action"]
        state.start_time = info["start_time"]
        state.submit_count = info["submit_count"]

        # Estimate wall time from trajectory timestamps
        try:
            with open(traj_path) as f:
                data = json.load(f)
            if len(data) >= 2:
                from datetime import datetime
                t0 = datetime.fromisoformat(data[0]["timestamp"])
                t1 = datetime.fromisoformat(data[-1]["timestamp"])
                state.wall_seconds = int((t1 - t0).total_seconds())
        except Exception:
            pass

        states.append(state)
    return states


STATUS_STYLES = {
    "PASSED": "[bold green]PASSED[/]",
    "FAILED": "[bold red]FAILED[/]",
    "RUNNING": "[bold cyan]RUNNING[/]",
    "PENDING": "[dim]PENDING[/]",
    "STARTING": "[bold yellow]STARTING[/]",
    "NO_SUBMIT": "[yellow]NO_SUBMIT[/]",
    "ERROR": "[bold red]ERROR[/]",
    "TIMEOUT": "[bold magenta]TIMEOUT[/]",
}

STATUS_ICONS = {
    "PASSED": "✓",
    "FAILED": "✗",
    "RUNNING": "⟳",
    "PENDING": "○",
    "STARTING": "◑",
    "NO_SUBMIT": "—",
    "ERROR": "!",
    "TIMEOUT": "⏱",
}


class StatsBar(Static):
    """Top statistics bar."""

    def update_stats(
        self,
        total: int,
        passed: int,
        failed: int,
        running: int,
        pending: int,
        total_cost: float,
        total_steps: int,
    ):
        parts = [
            f"[bold]Total:[/] {total}",
            f"[bold green]Passed:[/] {passed}",
            f"[bold red]Failed:[/] {failed}",
            f"[bold cyan]Running:[/] {running}",
            f"[dim]Pending:[/] {pending}",
            f"[bold yellow]Cost:[/] ${total_cost:.4f}",
            f"[bold]Steps:[/] {total_steps}",
        ]
        pct = (passed / total * 100) if total > 0 else 0
        parts.append(f"[bold green]Pass Rate:[/] {pct:.1f}%")
        self.update("  │  ".join(parts))


class CyberGymMonitor(App):
    CSS = """
    Screen {
        background: $surface;
    }
    #stats-bar {
        dock: top;
        height: 3;
        padding: 1 2;
        background: $panel;
        border-bottom: solid $accent;
    }
    #progress-container {
        dock: top;
        height: 3;
        padding: 0 2;
        background: $panel;
    }
    #progress-label {
        width: 20;
        content-align: left middle;
    }
    #main-table {
        height: 1fr;
    }
    DataTable {
        height: 1fr;
    }
    #detail-panel {
        dock: bottom;
        height: 8;
        border-top: solid $accent;
        padding: 0 2;
        background: $panel;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("p", "sort_passed", "Sort: Passed first"),
        Binding("s", "sort_steps", "Sort: Steps"),
        Binding("c", "sort_cost", "Sort: Cost"),
        Binding("n", "sort_name", "Sort: Name"),
        Binding("t", "sort_status", "Sort: Status"),
    ]

    TITLE = "CyberGym Task Monitor"
    SUB_TITLE = ""

    sort_key: reactive[str] = reactive("status")
    sort_reverse: reactive[bool] = reactive(False)

    def __init__(self, log_dir: Path, task_list: list[str], max_iter: int = 64, refresh_interval: float = 3.0):
        super().__init__()
        self.log_dir = log_dir
        self.task_list = task_list
        self.max_iter = max_iter
        self.refresh_interval = refresh_interval
        self.states: list[TaskState] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatsBar(id="stats-bar")
        with Horizontal(id="progress-container"):
            yield Label("Completion: ", id="progress-label")
            yield ProgressBar(total=len(self.task_list), show_eta=False, id="progress-bar")
        yield DataTable(id="main-table")
        yield Static(id="detail-panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(
            "",        # status icon
            "Task ID",
            "Status",
            "Step",
            "Progress",
            "Submits",
            "Cost",
            "Prompt Tok",
            "Compl Tok",
            "Cache Tok",
            "Wall Time",
            "Last Action",
        )
        self.do_refresh()
        self.set_interval(self.refresh_interval, self.do_refresh)

    def do_refresh(self) -> None:
        self.states = scan_log_dir(self.log_dir, self.task_list, self.max_iter)
        self._update_table()
        self._update_stats()

    def _sorted_states(self) -> list[TaskState]:
        status_order = {
            "RUNNING": 0, "STARTING": 1, "PASSED": 2, "FAILED": 3,
            "NO_SUBMIT": 4, "ERROR": 5, "TIMEOUT": 6, "PENDING": 7,
        }
        if self.sort_key == "status":
            return sorted(self.states, key=lambda s: status_order.get(s.status, 99))
        elif self.sort_key == "steps":
            return sorted(self.states, key=lambda s: s.step, reverse=True)
        elif self.sort_key == "cost":
            return sorted(self.states, key=lambda s: s.cost, reverse=True)
        elif self.sort_key == "name":
            return sorted(self.states, key=lambda s: s.task_id)
        elif self.sort_key == "passed":
            return sorted(
                self.states,
                key=lambda s: (0 if s.status == "PASSED" else 1, status_order.get(s.status, 99)),
            )
        return self.states

    def _update_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear()

        for s in self._sorted_states():
            icon = STATUS_ICONS.get(s.status, "?")
            pct = (s.step / s.max_iter * 100) if s.max_iter > 0 else 0
            progress_bar = self._make_progress_bar(s.step, s.max_iter, s.status)
            wall = f"{s.wall_seconds // 60}m{s.wall_seconds % 60:02d}s" if s.wall_seconds > 0 else "—"
            cost_str = f"${s.cost:.4f}" if s.cost > 0 else "—"
            prompt_str = f"{s.prompt_tokens:,}" if s.prompt_tokens > 0 else "—"
            compl_str = f"{s.completion_tokens:,}" if s.completion_tokens > 0 else "—"
            cache_str = f"{s.cache_tokens:,}" if s.cache_tokens > 0 else "—"

            table.add_row(
                icon,
                s.task_id,
                s.status,
                f"{s.step}/{s.max_iter}",
                progress_bar,
                str(s.submit_count) if s.submit_count > 0 else "—",
                cost_str,
                prompt_str,
                compl_str,
                cache_str,
                wall,
                s.last_action[:60] if s.last_action else "—",
                key=s.task_id,
            )

    def _make_progress_bar(self, current: int, total: int, status: str) -> str:
        width = 20
        if total == 0:
            return " " * width
        filled = int(current / total * width)
        filled = min(filled, width)

        if status == "PASSED":
            char, empty = "█", "░"
        elif status == "FAILED":
            char, empty = "█", "░"
        elif status == "RUNNING":
            char, empty = "▓", "░"
        else:
            char, empty = "░", " "

        bar = char * filled + empty * (width - filled)
        pct = current / total * 100
        return f"{bar} {pct:5.1f}%"

    def _update_stats(self) -> None:
        total = len(self.states)
        passed = sum(1 for s in self.states if s.status == "PASSED")
        failed = sum(1 for s in self.states if s.status == "FAILED")
        running = sum(1 for s in self.states if s.status in ("RUNNING", "STARTING"))
        pending = sum(1 for s in self.states if s.status == "PENDING")
        total_cost = sum(s.cost for s in self.states)
        total_steps = sum(s.step for s in self.states)

        stats = self.query_one(StatsBar)
        stats.update_stats(total, passed, failed, running, pending, total_cost, total_steps)

        # Update progress bar
        completed = sum(1 for s in self.states if s.status in ("PASSED", "FAILED", "NO_SUBMIT", "ERROR", "TIMEOUT"))
        pbar = self.query_one("#progress-bar", ProgressBar)
        pbar.update(progress=completed)

        self.sub_title = f"Log: {self.log_dir} | Refresh: {self.refresh_interval}s | Sort: {self.sort_key}"

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None:
            return
        task_id = str(event.row_key.value)
        state = next((s for s in self.states if s.task_id == task_id), None)
        if state is None:
            return

        detail = self.query_one("#detail-panel", Static)
        lines = [
            f"[bold]{state.task_id}[/] — {STATUS_STYLES.get(state.status, state.status)}",
            f"  Step: {state.step}/{state.max_iter}  |  Submits: {state.submit_count}  |  Wall: {state.wall_seconds // 60}m{state.wall_seconds % 60:02d}s  |  Started: {state.start_time or '—'}",
            f"  Cost: ${state.cost:.4f}  |  Prompt: {state.prompt_tokens:,}  |  Completion: {state.completion_tokens:,}  |  Cache: {state.cache_tokens:,}",
            f"  Last: {state.last_action or '—'}",
        ]
        detail.update("\n".join(lines))

    def action_refresh(self) -> None:
        self.do_refresh()

    def action_sort_passed(self) -> None:
        self.sort_key = "passed"
        self._update_table()

    def action_sort_steps(self) -> None:
        self.sort_key = "steps"
        self._update_table()

    def action_sort_cost(self) -> None:
        self.sort_key = "cost"
        self._update_table()

    def action_sort_name(self) -> None:
        self.sort_key = "name"
        self._update_table()

    def action_sort_status(self) -> None:
        self.sort_key = "status"
        self._update_table()


def parse_task_list_from_script(script_path: Path) -> list[str]:
    """Extract TASKS array from the bash script."""
    tasks = []
    in_tasks = False
    with open(script_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("TASKS=("):
                in_tasks = True
                continue
            if in_tasks:
                if stripped == ")":
                    break
                # Extract quoted string
                if '"' in stripped:
                    task = stripped.strip('" ')
                    if task:
                        tasks.append(task)
    return tasks


def parse_out_dir_from_script(script_path: Path) -> str | None:
    """Extract OUT_DIR from the bash script."""
    import re
    with open(script_path) as f:
        for line in f:
            m = re.match(r'^OUT_DIR\s*=\s*(.+)', line.strip())
            if m:
                return m.group(1).strip().strip('"').strip("'")
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CyberGym Task Monitor")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory to monitor")
    parser.add_argument("--script", type=str, default=None, help="Path to run_vllm_eval.sh to extract task list")
    parser.add_argument("--max-iter", type=int, default=64, help="Max iterations per task")
    parser.add_argument("--refresh", type=float, default=30.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    project_dir = Path(__file__).parent

    # Get task list
    script_path = Path(args.script) if args.script else project_dir / "run_vllm_eval.sh"
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        sys.exit(1)

    task_list = parse_task_list_from_script(script_path)
    if not task_list:
        print("No tasks found in script.")
        sys.exit(1)

    # Get log dir
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        out_dir = parse_out_dir_from_script(script_path)
        if out_dir:
            p = Path(out_dir)
            if not p.is_absolute():
                p = project_dir / p
            log_dir = p / "logs"
        else:
            log_dir = project_dir / "eval_minimax_m2_5" / "logs"

    print(f"Monitoring {len(task_list)} tasks in {log_dir}")

    app = CyberGymMonitor(
        log_dir=log_dir,
        task_list=task_list,
        max_iter=args.max_iter,
        refresh_interval=args.refresh,
    )
    app.run()


if __name__ == "__main__":
    main()
