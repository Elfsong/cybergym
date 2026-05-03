"""Execute strategies via OpenHands + MiniMax (vLLM) in parallel subprocesses.

Each strategy is injected into the executor's prompt and runs in its own
OpenHands subprocess. Reuses the existing examples/agents/openhands/run.py
with --prompt_file to inject the strategy-augmented prompt.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .config import PROJECT_DIR, Config
from .planner import StrategyToExecute
from .prompts import STRATEGY_INJECTION_PROMPT

logger = logging.getLogger(__name__)

OPENHANDS_RUN = PROJECT_DIR / "examples" / "agents" / "openhands" / "run.py"
UV_BIN = os.path.expanduser("~/.local/bin/uv")

# APRIL-style scheduler shared state. The execute_strategies() driver flips
# _STOP_EVENT when its early-stop heuristic fires; in-flight workers in
# _run_single() poll this between proc.wait() ticks and SIGTERM their child
# subprocess if it's set. _ACTIVE_PROCS lets the driver also fan-out a
# pkill from the outside in case workers are mid-poll.
_ACTIVE_PROCS: dict[int, "subprocess.Popen"] = {}
_ACTIVE_PROCS_LOCK = threading.Lock()
_STOP_EVENT = threading.Event()

# Global Docker container launch rate limiter. With 32 parallel rollouts each
# pulling an OpenHands runtime image and starting a per-task CyberGym
# container, the dockerd daemon hits its concurrent-build budget and rollouts
# CRASH (no trajectory; observed 82% CRASH rate without rate limiting).
# Serialize new container launches: at most one Popen every
# config.executor_docker_stagger_seconds across the entire pool.
_DOCKER_LAUNCH_LOCK = threading.Lock()
_LAST_DOCKER_LAUNCH_TS = 0.0
_OPENHANDS_RUNTIME_RE = re.compile(
    r"runtime ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-[0-9a-f]{16})"
)


@dataclass
class ExecutionResult:
    """Outcome of running a single strategy through the executor."""
    strategy: StrategyToExecute
    agent_id: str
    trajectory_path: Path | None
    wall_seconds: int
    subprocess_returncode: int | None
    log_dir: Path
    cancelled: bool = False  # APRIL early-stop preempted this rollout; the
                             # planner keeps it in GRPO group stats as a
                             # low-reward sample so slow strategies become
                             # negative evidence instead of disappearing.

    @property
    def has_trajectory(self) -> bool:
        return self.trajectory_path is not None and self.trajectory_path.exists()


def _find_trajectory(log_dir: Path, task_norm: str, agent_id: str) -> Path | None:
    """Find the OpenHands trajectory file after a run."""
    # Primary: {log_dir}/{task_norm}-{agent_id}/trajectory
    expected = log_dir / f"{task_norm}-{agent_id}" / "trajectory"
    if expected.exists():
        return expected
    # Fallback: any trajectory in any matching dir
    candidates = glob.glob(str(log_dir / f"{task_norm}-*" / "trajectory"))
    if candidates:
        return Path(max(candidates, key=os.path.getmtime))
    return None


def _recover_trajectory(task_norm: str, agent_id: str, log_dir: Path) -> None:
    """Rebuild trajectory from event files if OpenHands was killed early."""
    task_dir = log_dir / f"{task_norm}-{agent_id}"
    if not task_dir.exists():
        return
    traj_path = task_dir / "trajectory"
    if traj_path.exists():
        return
    event_dirs = list(task_dir.glob("file/sessions/*/events"))
    if not event_dirs:
        return
    best = max(event_dirs, key=lambda d: len(list(d.glob("*.json"))))
    events = sorted(best.glob("*.json"), key=lambda p: int(p.stem))
    if not events:
        return
    import json
    trajectory = []
    for ef in events:
        try:
            with open(ef) as f:
                trajectory.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    if trajectory:
        with open(traj_path, "w") as f:
            json.dump(trajectory, f)


def _find_openhands_runtime_containers(log_dir: Path) -> set[str]:
    """Return OpenHands runtime container names referenced by this run's logs."""
    names: set[str] = set()
    if not log_dir.exists():
        return names
    for path in log_dir.rglob("*.log"):
        if not path.is_file():
            continue
        try:
            with open(path, errors="ignore") as f:
                for line in f:
                    match = _OPENHANDS_RUNTIME_RE.search(line)
                    if match:
                        names.add(f"openhands-runtime-{match.group(1)}")
        except OSError:
            continue
    return names


def _cleanup_docker_containers(log_dir: Path | None = None) -> int:
    """Force-remove only OpenHands containers referenced by this run's logs."""
    if log_dir is None:
        logger.debug("Skipping Docker cleanup without run log_dir; refusing global prefix sweep")
        return 0
    container_names = _find_openhands_runtime_containers(log_dir)
    if not container_names:
        return 0
    try:
        import docker
        client = docker.from_env()
        removed = 0
        for name in sorted(container_names):
            try:
                container = client.containers.get(name)
            except docker.errors.NotFound:
                continue
            try:
                container.remove(force=True)
                removed += 1
            except Exception:
                pass
        return removed
    except Exception:
        return 0


def _docker_rate_limit(min_interval_s: float) -> None:
    """Block until at least `min_interval_s` has elapsed since the last
    container launch. Serializes Docker startup across the worker pool so
    dockerd doesn't get overwhelmed. Called immediately before Popen so the
    delay sits between sibling launches, not at task arrival.
    """
    global _LAST_DOCKER_LAUNCH_TS
    if min_interval_s <= 0:
        return
    with _DOCKER_LAUNCH_LOCK:
        now = time.monotonic()
        wait = (_LAST_DOCKER_LAUNCH_TS + min_interval_s) - now
        if wait > 0:
            # While waiting, periodically check stop event so we don't
            # block APRIL cancellation behind a long Docker queue.
            t_end = now + wait
            while time.monotonic() < t_end:
                if _STOP_EVENT.is_set():
                    raise RuntimeError("aborted by APRIL stop event before Docker launch")
                time.sleep(min(0.5, t_end - time.monotonic()))
        _LAST_DOCKER_LAUNCH_TS = time.monotonic()


def _run_single(
    idx: int,
    total: int,
    strategy: StrategyToExecute,
    config: Config,
    log_dir: Path,
    tmp_dir: Path,
) -> ExecutionResult:
    """Run one strategy through OpenHands + MiniMax."""

    agent_id = uuid4().hex
    task_norm = strategy.task_id.replace(":", "_")
    sub_dir = f"{task_norm}-{agent_id}"

    # Write the strategy-injected prompt to a temp file (under tmp_dir, not /tmp).
    # Scale recon/first-submit cues proportionally to the 72-turn defaults, matching
    # the scaling used in examples/agents/openhands/run.py so both code paths stay
    # coherent with the OpenHands per-turn reminder injector.
    max_iter = int(config.executor_max_iter)
    timeout = int(config.executor_timeout)
    recon_turns = max(1, round(max_iter * 10 / 72))
    first_submit_turn = max(1, round(max_iter * 15 / 72))
    prompt_text = (
        STRATEGY_INJECTION_PROMPT
        .replace("{MAX_ITER}", str(max_iter))
        .replace("{TIMEOUT}", str(timeout))
        .replace("{RECON_TURNS}", str(recon_turns))
        .replace("{FIRST_SUBMIT_TURN}", str(first_submit_turn))
        .replace("{strategy}", strategy.strategy)
    )
    prompt_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix=f"strategy_{task_norm}_",
        dir=str(tmp_dir),
    )
    prompt_file.write(prompt_text)
    prompt_file.close()

    logger.info(f"[{idx+1}/{total}] [{datetime.now():%H:%M:%S}] Executing {strategy.task_id} (agent {agent_id[:8]})")
    start = time.monotonic()
    returncode: int | None = None

    # Use Popen for explicit kill on timeout (subprocess.run doesn't kill children)
    proc = None
    # Hoisted out of the try block: _docker_rate_limit() can raise the
    # "aborted by APRIL stop event before Docker launch" RuntimeError before
    # the inner cancelled_by_stop assignment is reached, and the result
    # constructor at the bottom of the function reads this variable
    # unconditionally. Without this initialization that path crashes with
    # UnboundLocalError, which the outer execute_strategies loop then logs
    # as a "Strategy N raised" — misclassifying the rollout as a generic
    # exception (cancelled=False) and letting it pollute GRPO group stats.
    cancelled_by_stop = False
    try:
        cmd = [
            UV_BIN, "run", "python3", str(OPENHANDS_RUN),
            "--model", config.executor_model,
            "--base_url", config.executor_base_url,
            "--log_dir", str(log_dir),
            "--tmp_dir", str(tmp_dir),
            "--data_dir", str(config.data_dir),
            "--task_id", strategy.task_id,
            "--server", config.server,
            "--timeout", str(config.executor_timeout),
            "--max_iter", str(config.executor_max_iter),
            "--max_output_tokens", str(config.executor_max_output_tokens),
            "--temperature", str(config.executor_temperature),
            "--silent", "true",
            "--difficulty", config.executor_difficulty,
            "--prompt_file", prompt_file.name,
        ]
        env = {
            **os.environ,
            # OpenHands/LiteLLM reads LLM_API_KEY from env. Use the resolved
            # executor_api_key (fallback chain: EXECUTOR_API_KEY > DASHSCOPE_API_KEY
            # > LLM_API_KEY > "EMPTY") so the same code path works for both
            # local vLLM (unvalidated, "EMPTY") and DashScope (real key).
            "LLM_API_KEY": config.executor_api_key or "EMPTY",
            "TMPDIR": str(tmp_dir),  # Force subprocesses to use round tmp, not /tmp
        }
        # Rate-limit Docker container starts (each OpenHands subprocess
        # launches its own runtime container + the per-task CyberGym sandbox).
        # Without this, 32 simultaneous launches saturate dockerd and most
        # rollouts CRASH at the runtime-init stage with no trajectory.
        _docker_rate_limit(config.executor_docker_stagger_seconds)
        proc = subprocess.Popen(
            cmd,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            cwd=str(PROJECT_DIR),
            env=env,
            start_new_session=True,  # Own process group for clean kill
        )
        with _ACTIVE_PROCS_LOCK:
            _ACTIVE_PROCS[idx] = proc
        # Stop-aware wait. Poll proc.wait() in short slices so the APRIL
        # stop event can preempt long subprocesses.
        deadline = time.monotonic() + config.executor_timeout + 300
        cancelled_by_stop = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, config.executor_timeout + 300)
            try:
                proc.wait(timeout=min(5.0, remaining))
                returncode = proc.returncode
                break
            except subprocess.TimeoutExpired:
                if _STOP_EVENT.is_set():
                    cancelled_by_stop = True
                    logger.info(f"[{idx+1}/{total}] {strategy.task_id} APRIL-cancelled (stop event)")
                    try:
                        os.killpg(os.getpgid(proc.pid), 15)  # SIGTERM first
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(os.getpgid(proc.pid), 9)
                        except (ProcessLookupError, PermissionError):
                            pass
                        proc.wait(timeout=5)
                    returncode = proc.returncode
                    break  # cancelled_by_stop=True is captured in the result below
                # not stopping; loop back and keep waiting
                continue
    except subprocess.TimeoutExpired:
        logger.warning(f"[{idx+1}/{total}] {strategy.task_id} timed out — killing process group")
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), 9)  # SIGKILL entire group
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait(timeout=10)
    except Exception as e:
        logger.warning(f"[{idx+1}/{total}] {strategy.task_id} error: {e}")
        # If the round-level stop event was set, classify the rollout as an
        # APRIL cancellation regardless of whether the failure point was
        # before or after Docker launch. Without this, pre-launch aborts
        # ("aborted by APRIL stop event before Docker launch" raised by
        # _docker_rate_limit) reach the result constructor with
        # cancelled=False and pollute GRPO per-task mean/std as zero-reward
        # failures rather than being filtered out.
        if _STOP_EVENT.is_set():
            cancelled_by_stop = True
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, PermissionError):
                pass
    finally:
        with _ACTIVE_PROCS_LOCK:
            _ACTIVE_PROCS.pop(idx, None)
        try:
            os.unlink(prompt_file.name)
        except OSError:
            pass

    # Recover partial trajectory if the subprocess died mid-run
    _recover_trajectory(task_norm, agent_id, log_dir)
    traj_path = _find_trajectory(log_dir, task_norm, agent_id)
    elapsed = int(time.monotonic() - start)

    # The OpenHands sub-dir uses its OWN internal agent_id (different UUID).
    # Find the actual sub-dir it created; we report the *discovered* agent_id
    # from the directory name so downstream verification can query it.
    real_agent_id = agent_id
    if traj_path is not None:
        parent = traj_path.parent
        # Sub-dir name: "{task_norm}-{uuid32hex}"
        if parent.name.startswith(f"{task_norm}-"):
            real_agent_id = parent.name[len(task_norm) + 1 :]

    result = ExecutionResult(
        strategy=strategy,
        agent_id=real_agent_id,
        trajectory_path=traj_path,
        wall_seconds=elapsed,
        subprocess_returncode=returncode,
        log_dir=log_dir / (traj_path.parent.name if traj_path else sub_dir),
        cancelled=cancelled_by_stop,
    )
    # Three-state status so diagnostics can distinguish the two no-trajectory modes:
    #   OK      — trajectory file found (subprocess produced a full or partial trace)
    #   TIMEOUT — no trajectory AND ran close to the wall-clock budget (agent still
    #             looping when SIGKILL'd; typical for hard tasks with huge source trees)
    #   CRASH   — no trajectory AND died quickly (Docker startup race, port conflict, etc.)
    if traj_path is not None:
        status = "OK"
    elif elapsed >= config.executor_timeout - 5:
        status = "TIMEOUT"
    else:
        status = "CRASH"
    logger.info(
        f"[{idx+1}/{total}] {strategy.task_id} ({strategy.group_id}) "
        f"{status} in {elapsed//60}m{elapsed%60:02d}s"
    )
    return result


def _load_exec_checkpoint(
    ckpt_path: Path,
    strategies: list[StrategyToExecute],
    default_log_dir: Path,
) -> dict[int, ExecutionResult]:
    """Load previously-completed ExecutionResults keyed by submission index.

    Only entries whose trajectory file still exists on disk are kept; TIMEOUT
    / CRASH / missing-file entries are dropped so they get retried on resume. Entries
    whose (task_id, group_id) no longer match the current strategies list are
    also discarded (guards against a stale checkpoint from a different run).
    """
    if not ckpt_path.exists():
        return {}
    done: dict[int, ExecutionResult] = {}
    with open(ckpt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = entry.get("idx")
            if not isinstance(idx, int) or idx < 0 or idx >= len(strategies):
                continue
            traj = entry.get("trajectory_path")
            if not traj or not Path(traj).exists():
                continue
            strat = strategies[idx]
            if entry.get("task_id") != strat.task_id or entry.get("group_id") != strat.group_id:
                logger.warning(
                    f"Checkpoint idx={idx} mismatch "
                    f"({entry.get('task_id')}:{entry.get('group_id')} vs "
                    f"{strat.task_id}:{strat.group_id}); discarding"
                )
                continue
            done[idx] = ExecutionResult(
                strategy=strat,
                agent_id=entry.get("agent_id", ""),
                trajectory_path=Path(traj),
                wall_seconds=entry.get("wall_seconds", 0),
                subprocess_returncode=entry.get("subprocess_returncode"),
                log_dir=Path(entry.get("log_dir", str(default_log_dir))),
            )
    return done


def _append_exec_checkpoint(
    ckpt_path: Path,
    idx: int,
    result: ExecutionResult,
    lock: threading.Lock,
) -> None:
    """Append one completed OK result to the checkpoint jsonl (thread-safe)."""
    entry = {
        "idx": idx,
        "task_id": result.strategy.task_id,
        "group_id": result.strategy.group_id,
        "agent_id": result.agent_id,
        "trajectory_path": str(result.trajectory_path) if result.trajectory_path else None,
        "wall_seconds": result.wall_seconds,
        "subprocess_returncode": result.subprocess_returncode,
        "log_dir": str(result.log_dir),
    }
    line = json.dumps(entry) + "\n"
    with lock:
        with open(ckpt_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def execute_strategies(
    strategies: list[StrategyToExecute],
    config: Config,
    round_dir: Path,
) -> list[ExecutionResult]:
    """Execute all strategies in parallel. Returns results in the same order as input.

    Each strategy gets its own workspace (by agent_id) under round_dir/logs.
    If round_dir/executions.jsonl exists, completed rollouts (with a trajectory
    file still on disk) are loaded and skipped.
    """
    log_dir = round_dir / "logs"
    tmp_dir = round_dir / "tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = round_dir / "executions.jsonl"
    ckpt_lock = threading.Lock()
    done = _load_exec_checkpoint(ckpt_path, strategies, log_dir)

    results: list[ExecutionResult | None] = [None] * len(strategies)
    for idx, r in done.items():
        results[idx] = r

    pending = [i for i in range(len(strategies)) if results[i] is None]
    if done:
        logger.info(
            f"Resuming: {len(done)}/{len(strategies)} rollouts already done, "
            f"{len(pending)} to execute"
        )
    if not pending:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return [r for r in results if r is not None]

    # APRIL early-stop bookkeeping. Per-task completed-rollout counter; the
    # round can stop once a target fraction of tasks have completed enough of
    # their K rollouts, OR when the wall-clock cap fires.
    from collections import defaultdict
    task_completed: dict[str, int] = defaultdict(int)
    for r in done.values():
        if r.trajectory_path is not None and r.trajectory_path.exists():
            task_completed[r.strategy.task_id] += 1
    n_tasks = len({s.task_id for s in strategies})
    round_start = time.monotonic()
    round_max_wall = config.executor_round_max_wall_seconds
    round_min_wall = config.executor_round_min_wall_seconds
    completion_threshold = config.executor_completion_threshold
    per_task_completion_fraction = config.executor_min_rollout_fraction_per_task
    per_task_min_completed_rollouts = min(
        max(1, math.ceil(config.group_size * per_task_completion_fraction)),
        max(config.group_size, 1),
    )
    april_enabled = round_max_wall > 0
    stop_reason_box: list[str] = []  # mutable via closures
    stop_lock = threading.Lock()

    _STOP_EVENT.clear()  # fresh round

    def _trigger_stop(reason: str) -> None:
        """Atomic: flip stop event, fan out SIGTERM, clean this run's containers.
        Idempotent — safe to call from both the as_completed loop and the
        watchdog thread without racing.
        """
        with stop_lock:
            if _STOP_EVENT.is_set():
                return
            _STOP_EVENT.set()
            stop_reason_box.append(reason)
        logger.warning(f"APRIL stop: {reason}")
        # Fan-out SIGTERM to all in-flight subprocesses concurrently. Workers
        # also poll _STOP_EVENT, but the explicit signal here gets termination
        # going immediately rather than waiting for each worker's next 5s
        # poll tick.
        with _ACTIVE_PROCS_LOCK:
            procs = list(_ACTIVE_PROCS.values())
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), 15)
            except (ProcessLookupError, PermissionError):
                pass
        # Docker cleanup is scoped to containers whose runtime IDs appear in
        # this round's logs. A prefix-wide sweep can kill unrelated OpenHands
        # jobs on a shared host.
        try:
            n_orphans = _cleanup_docker_containers(log_dir)
            if n_orphans:
                logger.warning(f"APRIL stop: removed {n_orphans} Docker containers from this run")
        except Exception as e:
            logger.warning(f"APRIL stop: docker cleanup failed: {e}")

    def _april_watchdog() -> None:
        """Background thread that checks stop conditions on a wall clock,
        independent of how often futures complete. Without this, a round
        full of slow rollouts can hold the as_completed loop quiet for an
        hour and APRIL never gets a chance to fire.
        """
        if not april_enabled:
            return
        while not _STOP_EVENT.is_set():
            elapsed = time.monotonic() - round_start
            tasks_with_kmin = sum(
                1
                for c in task_completed.values()
                if c >= per_task_min_completed_rollouts
            )
            frac_ok = tasks_with_kmin / max(n_tasks, 1)
            if elapsed >= round_max_wall:
                _trigger_stop(
                    f"wall budget hit ({int(elapsed)}s ≥ {round_max_wall}s); "
                    f"{tasks_with_kmin}/{n_tasks} tasks have "
                    f"≥{per_task_min_completed_rollouts}/{config.group_size} "
                    f"rollouts ({per_task_completion_fraction:.0%}) (watchdog)"
                )
                return
            if frac_ok >= completion_threshold and elapsed >= round_min_wall:
                _trigger_stop(
                    f"completion threshold met "
                    f"({tasks_with_kmin}/{n_tasks}={frac_ok:.0%} ≥ {completion_threshold:.0%}) "
                    f"with per-task threshold "
                    f"≥{per_task_min_completed_rollouts}/{config.group_size} "
                    f"({per_task_completion_fraction:.0%}) "
                    f"after {int(elapsed)}s (watchdog)"
                )
                return
            # Sleep in short slices so we exit quickly when stop fires from
            # the as_completed path (a completion-driven trigger races us).
            for _ in range(20):
                if _STOP_EVENT.is_set():
                    return
                time.sleep(0.5)

    watchdog_thread = threading.Thread(
        target=_april_watchdog, name="april-watchdog", daemon=True,
    )
    if april_enabled:
        watchdog_thread.start()

    with ThreadPoolExecutor(max_workers=config.executor_parallel) as pool:
        futures = {
            pool.submit(
                _run_single, i, len(strategies), strategies[i], config, log_dir, tmp_dir,
            ): i
            for i in pending
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                logger.exception(f"Strategy {idx} raised: {e}")
                result = ExecutionResult(
                    strategy=strategies[idx],
                    agent_id="",
                    trajectory_path=None,
                    wall_seconds=0,
                    subprocess_returncode=None,
                    log_dir=log_dir,
                )
            results[idx] = result
            if result.trajectory_path is not None and result.trajectory_path.exists():
                _append_exec_checkpoint(ckpt_path, idx, result, ckpt_lock)
                task_completed[strategies[idx].task_id] += 1

            # Same conditions as the watchdog, evaluated on every completion.
            # The watchdog drives most stops (especially when completions are
            # rare), but this path catches the "fast-finish then idle" case
            # before the next watchdog tick.
            if april_enabled and not _STOP_EVENT.is_set():
                elapsed = time.monotonic() - round_start
                tasks_with_kmin = sum(
                    1
                    for c in task_completed.values()
                    if c >= per_task_min_completed_rollouts
                )
                frac_ok = tasks_with_kmin / max(n_tasks, 1)
                if elapsed >= round_max_wall:
                    _trigger_stop(
                        f"wall budget hit ({int(elapsed)}s ≥ {round_max_wall}s); "
                        f"{tasks_with_kmin}/{n_tasks} tasks have "
                        f"≥{per_task_min_completed_rollouts}/{config.group_size} "
                        f"rollouts ({per_task_completion_fraction:.0%})"
                    )
                elif frac_ok >= completion_threshold and elapsed >= round_min_wall:
                    _trigger_stop(
                        f"completion threshold met "
                        f"({tasks_with_kmin}/{n_tasks}={frac_ok:.0%} ≥ {completion_threshold:.0%}) "
                        f"with per-task threshold "
                        f"≥{per_task_min_completed_rollouts}/{config.group_size} "
                        f"({per_task_completion_fraction:.0%}) "
                        f"after {int(elapsed)}s; cancelling tail"
                    )

    # Fill any still-None slots with a CANCELLED marker so callers (scoring,
    # reward) see a uniform list and downstream GRPO can drop those samples.
    n_cancelled_slots = 0
    for i in range(len(strategies)):
        if results[i] is None:
            n_cancelled_slots += 1
            results[i] = ExecutionResult(
                strategy=strategies[i],
                agent_id="",
                trajectory_path=None,
                wall_seconds=int(time.monotonic() - round_start),
                subprocess_returncode=None,
                log_dir=log_dir,
                cancelled=True,
            )
    n_cancelled_total = sum(1 for r in results if r.cancelled)
    if n_cancelled_total:
        logger.warning(
            f"APRIL: {n_cancelled_total} rollouts cancelled "
            f"({n_cancelled_slots} never-started + {n_cancelled_total - n_cancelled_slots} mid-flight) — "
            f"will be kept in GRPO group stats as low-reward samples"
        )

    # Tidy up tmp dir (logs are kept; tmp is transient)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Force-remove only orphaned OpenHands containers referenced by this run's logs.
    n_orphans = _cleanup_docker_containers(log_dir)
    if n_orphans > 0:
        logger.warning(f"Cleaned up {n_orphans} OpenHands Docker containers from this run")

    # Log disk usage for monitoring
    try:
        root_usage = shutil.disk_usage("/")
        data_usage = shutil.disk_usage("/data")
        tmp_size = sum(
            f.stat().st_size for f in Path("/tmp").iterdir()
            if f.is_file() or f.is_symlink()
        )
        logger.info(
            f"Disk: / {root_usage.used/1024**3:.1f}G/{root_usage.total/1024**3:.0f}G, "
            f"/data {data_usage.used/1024**3:.0f}G/{data_usage.total/1024**3:.0f}G, "
            f"/tmp ~{tmp_size/1024**2:.0f}MB"
        )
    except Exception:
        pass

    return [r for r in results if r is not None]
