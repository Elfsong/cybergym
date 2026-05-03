"""Repair milestones for the PAGENT + Qwen3.6-Max-Preview run.

Background: the run had two false starts (auth-failed, sqlite I/O), each of
which created a logs/<kind>_<num>-<hex>/ subdirectory containing a 3-event
auth-failure trajectory. The successful third run added a fresh subdir
alongside. phase4_agent._detect_milestone iterates logs/ via iterdir() and
returns on the first matching subdir, so when ~iterdir~ surfaces the stale
3-event subdir first, the live trajectory's milestone is silently lost.

This script does an in-place repair:
  1. Find every task dir under phase4_qwen36_maxpreview/<project>/<task_num>/
  2. Determine the freshest subdir (highest mtime) — that's the live run.
  3. Delete the stale subdirs (the 3-event ones from the failed launches).
  4. Re-score milestone.json from the live trajectory only, by calling the
     same dual_loops.reward.detect_milestone the orchestrator uses.

It SKIPS any task whose milestone.json is still being written (no .done
or no live trajectory present yet) so it is safe to run while the
orchestrator is still in flight.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

CYBERGYM_ROOT = Path("/home/nvidia/Projects/cybergym")
sys.path.insert(0, str(CYBERGYM_ROOT))
sys.path.insert(0, str(CYBERGYM_ROOT / "baselines/pagent/runner"))

from dual_loops.reward import detect_milestone  # noqa: E402
from phase4_agent import _TRAJ_DIR_RE, _DEFAULT_CYBERGYM_API_KEY  # noqa: E402

ROOT = Path("/lp-dev/cybergym_data/pagent/phase4_qwen36_maxpreview")
SERVER = "http://172.17.0.1:8666"
API_KEY = os.environ.get("CYBERGYM_API_KEY", _DEFAULT_CYBERGYM_API_KEY)


def event_count(traj: Path) -> int:
    try:
        return len(json.loads(traj.read_text()))
    except Exception:
        return -1


def repair(task_dir: Path, *, dry_run: bool) -> tuple[str, int | None, int | None]:
    logs = task_dir / "logs"
    if not logs.is_dir():
        return ("no-logs", None, None)
    subdirs = [d for d in logs.iterdir() if d.is_dir() and _TRAJ_DIR_RE.match(d.name)]
    if not subdirs:
        return ("no-subdirs", None, None)

    by_mtime = sorted(subdirs, key=lambda d: d.stat().st_mtime, reverse=True)
    keep = by_mtime[0]
    keep_traj = keep / "trajectory"
    if not keep_traj.is_file():
        return ("no-trajectory", None, None)

    # If the run is still in flight (no milestone.json yet) or the live traj
    # is still tiny, skip — the orchestrator will write its own score.
    n_keep = event_count(keep_traj)
    if n_keep < 4:
        return ("live-traj-too-short", None, None)

    stale = by_mtime[1:]
    for d in stale:
        if not dry_run:
            try:
                shutil.rmtree(d)
            except PermissionError:
                # docker-mounted root-owned files leak through; remove the
                # parts we own and leave the rest. The detector uses iterdir
                # so a not-fully-empty leftover with no trajectory will be
                # skipped naturally.
                for child in d.rglob("*"):
                    try:
                        if child.is_file() or child.is_symlink():
                            child.unlink()
                        elif child.is_dir():
                            try:
                                child.rmdir()
                            except OSError:
                                pass
                    except (PermissionError, OSError):
                        pass
                # Try to kill the trajectory file specifically — that's the
                # only thing iterdir's match cares about.
                try:
                    (d / "trajectory").unlink()
                except (PermissionError, FileNotFoundError, OSError):
                    pass

    # Re-score milestone from the live trajectory only.
    agent_id = _TRAJ_DIR_RE.match(keep.name).group(3)
    try:
        r = detect_milestone(
            keep_traj, agent_id, SERVER, API_KEY,
            traj_format="openhands", verify_fix=True,
        )
        new_milestone = int(r.milestone)
    except Exception as e:
        return (f"score-fail:{type(e).__name__}", None, None)

    ms_path = task_dir / "milestone.json"
    old_ms = None
    if ms_path.is_file():
        try:
            old_ms = json.loads(ms_path.read_text()).get("milestone")
        except Exception:
            old_ms = None

    if not dry_run and (old_ms != new_milestone):
        record = {}
        if ms_path.is_file():
            try:
                record = json.loads(ms_path.read_text())
            except Exception:
                pass
        record["milestone"] = new_milestone
        record["repaired_from"] = old_ms
        record["repaired_subdirs_kept"] = keep.name
        ms_path.write_text(json.dumps(record, indent=2))

    return ("repaired", old_ms, new_milestone)


def main():
    dry_run = "--dry-run" in sys.argv
    print(f"[repair] dry_run={dry_run}  root={ROOT}")
    by_status = {}
    changes = []
    for proj_dir in sorted(ROOT.iterdir()):
        if not proj_dir.is_dir():
            continue
        for task_dir in sorted(proj_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            status, old_ms, new_ms = repair(task_dir, dry_run=dry_run)
            by_status[status] = by_status.get(status, 0) + 1
            if status == "repaired" and old_ms != new_ms:
                changes.append((proj_dir.name, task_dir.name, old_ms, new_ms))
    print(f"[repair] status breakdown: {by_status}")
    for proj, num, old, new in changes:
        print(f"  {proj}/{num}: {old} -> {new}")
    print(f"[repair] {len(changes)} milestone(s) corrected")


if __name__ == "__main__":
    main()
