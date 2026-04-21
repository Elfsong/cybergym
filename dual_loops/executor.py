"""Execute strategies via OpenHands + MiniMax (vLLM) in parallel subprocesses.

Each strategy is injected into the executor's prompt and runs in its own
OpenHands subprocess. Reuses the existing examples/agents/openhands/run.py
with --prompt_file to inject the strategy-augmented prompt.
"""

from __future__ import annotations

import glob
import json
import logging
import os
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


@dataclass
class ExecutionResult:
    """Outcome of running a single strategy through the executor."""
    strategy: StrategyToExecute
    agent_id: str
    trajectory_path: Path | None
    wall_seconds: int
    subprocess_returncode: int | None
    log_dir: Path

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


def _cleanup_docker_containers(prefix: str = "openhands-runtime-") -> int:
    """Force-remove any Docker containers matching prefix. Returns count removed."""
    try:
        import docker
        client = docker.from_env()
        removed = 0
        for c in client.containers.list(all=True):
            if c.name.startswith(prefix):
                try:
                    c.remove(force=True)
                    removed += 1
                except Exception:
                    pass
        return removed
    except Exception:
        return 0


def _run_single(
    idx: int,
    total: int,
    strategy: StrategyToExecute,
    config: Config,
    log_dir: Path,
    tmp_dir: Path,
    stagger: float = 0.5,
) -> ExecutionResult:
    """Run one strategy through OpenHands + MiniMax."""
    if stagger > 0:
        time.sleep(idx * stagger)

    agent_id = uuid4().hex
    task_norm = strategy.task_id.replace(":", "_")
    sub_dir = f"{task_norm}-{agent_id}"

    # Write the strategy-injected prompt to a temp file (under tmp_dir, not /tmp)
    prompt_text = STRATEGY_INJECTION_PROMPT.format(strategy=strategy.strategy)
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
            "LLM_API_KEY": os.environ.get("LLM_API_KEY", "EMPTY"),
            "TMPDIR": str(tmp_dir),  # Force subprocesses to use round tmp, not /tmp
        }
        proc = subprocess.Popen(
            cmd,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            cwd=str(PROJECT_DIR),
            env=env,
            start_new_session=True,  # Own process group for clean kill
        )
        proc.wait(timeout=config.executor_timeout + 300)
        returncode = proc.returncode
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
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, PermissionError):
                pass
    finally:
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

    # Tidy up tmp dir (logs are kept; tmp is transient)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Force-remove any orphaned OpenHands Docker containers
    n_orphans = _cleanup_docker_containers()
    if n_orphans > 0:
        logger.warning(f"Cleaned up {n_orphans} orphaned Docker containers")

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
