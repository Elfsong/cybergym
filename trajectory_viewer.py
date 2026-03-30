#!/usr/bin/env python3
"""CyberGym Trajectory Viewer — serves an interactive visualization of eval runs."""

# Quickstart usage:
# uv run python3 trajectory_viewer.py --logs_dir eval_gemini_3_flash_preview/logs       
# uv run python3 trajectory_viewer.py --logs_dir eval_qwen3_5_35b_a3b/logs

import argparse
import json
import os
import re
import sys
from datetime import datetime
from functools import lru_cache
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_task_summary(logs_dir: str, folder: str) -> dict:
    """Extract a compact summary for one task (no full steps)."""
    traj_path = os.path.join(logs_dir, folder, "trajectory")
    args_path = os.path.join(logs_dir, folder, "args.json")

    # Folder naming: {task_name}-{32-char-uuid}
    parts = folder.rsplit("-", 1)
    task_name = parts[0] if len(parts) == 2 and len(parts[1]) == 32 else folder
    agent_id = parts[1] if len(parts) == 2 else ""

    args = {}
    if os.path.isfile(args_path):
        with open(args_path) as f:
            args = json.load(f)

    task_id = args.get("task", {}).get("task_id", task_name.replace("_", ":", 1))
    difficulty = args.get("task", {}).get("difficulty", "unknown")
    model = args.get("agent_args", {}).get("llm", {}).get("model", "unknown")
    source = "oss-fuzz" if task_name.startswith("oss-fuzz") else "arvo"

    if not os.path.isfile(traj_path):
        return dict(
            folder=folder, task_name=task_name, task_id=task_id,
            agent_id=agent_id, source=source, difficulty=difficulty,
            model=model, status="IN_PROGRESS", num_steps=0,
            total_entries=0, cost=0, tokens={}, submit_attempts=0,
            start_time=None, end_time=None, duration_s=0, action_counts={},
        )

    try:
        with open(traj_path) as f:
            traj = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(
            folder=folder, task_name=task_name, task_id=task_id,
            agent_id=agent_id, source=source, difficulty=difficulty,
            model=model, status="ERROR", num_steps=0,
            total_entries=0, cost=0, tokens={}, submit_attempts=0,
            start_time=None, end_time=None, duration_s=0, action_counts={},
        )

    # Walk trajectory once to collect everything
    poc_status = "NO_SUBMIT"
    final_cost = 0.0
    final_tokens: dict = {}
    submit_attempts = 0
    action_counts: dict[str, int] = {}
    agent_steps = 0
    timestamps: list[str] = []

    for i, entry in enumerate(traj):
        ts = entry.get("timestamp")
        if ts:
            timestamps.append(ts)

        is_action = "action" in entry
        src = entry.get("source", "")
        action = entry.get("action", entry.get("observation", ""))

        if is_action and src == "agent":
            agent_steps += 1
            action_counts[action] = action_counts.get(action, 0) + 1

        # Cost tracking
        llm = entry.get("llm_metrics")
        if llm:
            c = llm.get("accumulated_cost", 0)
            if c > final_cost:
                final_cost = c
                final_tokens = llm.get("accumulated_token_usage", {})

        # Submit detection
        cmd = str(entry.get("args", {}).get("command", "")) if entry.get("args") else ""
        if "submit.sh" in cmd and "cat" not in cmd:
            submit_attempts += 1
            if i + 1 < len(traj):
                content = str(traj[i + 1].get("content", ""))
                try:
                    j = content.find("{")
                    if j >= 0:
                        result = json.loads(content[j:].split("\n")[0].strip())
                        ec = result.get("exit_code")
                        if ec is not None and ec != 0:
                            poc_status = "PASSED"
                        elif ec == 0 and poc_status != "PASSED":
                            poc_status = "FAILED"
                except (json.JSONDecodeError, ValueError):
                    pass

    # Timing
    duration_s = 0.0
    if len(timestamps) >= 2:
        try:
            t0 = datetime.fromisoformat(timestamps[0])
            t1 = datetime.fromisoformat(timestamps[-1])
            duration_s = (t1 - t0).total_seconds()
        except (ValueError, TypeError):
            pass

    return dict(
        folder=folder,
        task_name=task_name,
        task_id=task_id,
        agent_id=agent_id,
        source=source,
        difficulty=difficulty,
        model=model,
        status=poc_status,
        num_steps=agent_steps,
        total_entries=len(traj),
        cost=round(final_cost, 6),
        tokens=dict(
            prompt=final_tokens.get("prompt_tokens", 0),
            completion=final_tokens.get("completion_tokens", 0),
            cache_read=final_tokens.get("cache_read_tokens", 0),
        ),
        submit_attempts=submit_attempts,
        start_time=timestamps[0] if timestamps else None,
        end_time=timestamps[-1] if timestamps else None,
        duration_s=round(duration_s, 1),
        action_counts=action_counts,
    )


def extract_steps(logs_dir: str, folder: str) -> list[dict]:
    """Extract compact step list for a single task (for detail view)."""
    traj_path = os.path.join(logs_dir, folder, "trajectory")
    if not os.path.isfile(traj_path):
        return []
    try:
        with open(traj_path) as f:
            traj = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    steps = []
    for i, entry in enumerate(traj):
        action = entry.get("action", entry.get("observation", "unknown"))
        src = entry.get("source", "unknown")
        msg = entry.get("message", "") or ""
        is_action = "action" in entry
        is_obs = "observation" in entry
        extras = entry.get("extras") or {}
        entry_args = entry.get("args") or {}
        content = entry.get("content", "") or ""
        llm = entry.get("llm_metrics")

        step: dict = dict(
            id=entry.get("id", i), ts=entry.get("timestamp", ""),
            src=src, type=action, msg=msg[:500],
            is_action=is_action, is_obs=is_obs,
        )

        if llm:
            step["cost"] = llm.get("accumulated_cost", 0)
            tok = llm.get("accumulated_token_usage", {})
            step["prompt_tok"] = tok.get("prompt_tokens", 0)
            step["comp_tok"] = tok.get("completion_tokens", 0)
            step["cache_tok"] = tok.get("cache_read_tokens", 0)

        # Run actions / observations
        if action == "run" and is_action:
            step["command"] = entry_args.get("command", "") or ""
        elif action == "run" and is_obs:
            step["command"] = extras.get("command", "") or ""
            meta = extras.get("metadata") or {}
            step["exit_code"] = meta.get("exit_code", extras.get("exit_code"))
            step["output"] = content

        if action == "run_ipython" and is_action:
            step["command"] = entry_args.get("code", "") or ""
        elif action == "run_ipython" and is_obs:
            step["output"] = content

        if action == "read":
            step["path"] = entry_args.get("path", extras.get("path", ""))
            if is_obs:
                step["output"] = content

        if action == "edit":
            step["path"] = entry_args.get("path", "")
            step["edit_cmd"] = entry_args.get("command", "")
            if entry_args.get("file_text"):
                step["output"] = entry_args["file_text"]

        if action in ("finish", "condensation", "error"):
            step["msg"] = msg

        if action == "message":
            full = entry_args.get("content", "") or content or msg
            if full:
                step["msg_full"] = str(full)

        if action == "recall":
            if is_action:
                full = entry_args.get("query", "") or msg
                step["msg_full"] = str(full)
            elif is_obs:
                parts = []
                for key in ("repo_instructions", "additional_agent_instructions", "microagent_knowledge"):
                    val = str(extras.get(key, "") or "")
                    if val and val not in ("", "[]"):
                        parts.append(f"[{key}]\n{val}")
                if parts:
                    step["msg_full"] = "\n\n".join(parts)
                else:
                    step["msg_full"] = str(content or msg)

        if action == "browse" and is_obs:
            step["output"] = content

        # LLM thinking
        tcm = entry.get("tool_call_metadata") or {}
        mr = tcm.get("model_response") or {}
        choices = mr.get("choices") or []
        if choices:
            thinking = (choices[0].get("message") or {}).get("content", "")
            if thinking:
                step["thinking"] = thinking

        # Submit detection
        cmd_str = entry_args.get("command", "") or ""
        if "submit.sh" in cmd_str and "cat" not in cmd_str:
            step["is_submit"] = True
            if i + 1 < len(traj):
                nxt = str(traj[i + 1].get("content", ""))
                try:
                    j = nxt.find("{")
                    if j >= 0:
                        result = json.loads(nxt[j:].split("\n")[0].strip())
                        step["submit_exit_code"] = result.get("exit_code")
                except (json.JSONDecodeError, ValueError):
                    pass

        steps.append(step)

    return steps


def compute_stats(summaries: list[dict]) -> dict:
    total = len(summaries)
    passed = sum(1 for t in summaries if t["status"] == "PASSED")
    failed = sum(1 for t in summaries if t["status"] == "FAILED")
    no_submit = sum(1 for t in summaries if t["status"] == "NO_SUBMIT")
    error = sum(1 for t in summaries if t["status"] == "ERROR")
    costs = [t["cost"] for t in summaries]
    steps_list = [t["num_steps"] for t in summaries]
    durations = [t["duration_s"] for t in summaries if t["duration_s"] > 0]

    return dict(
        total=total,
        passed=passed, failed=failed, no_submit=no_submit, error=error,
        pass_rate=round(passed / total * 100, 1) if total else 0,
        pass_rate_submitted=round(passed / (passed + failed) * 100, 1) if (passed + failed) else 0,
        total_cost=round(sum(costs), 2),
        avg_cost=round(sum(costs) / total, 4) if total else 0,
        min_cost=round(min(costs), 4) if costs else 0,
        max_cost=round(max(costs), 4) if costs else 0,
        avg_steps=round(sum(steps_list) / total, 1) if total else 0,
        min_steps=min(steps_list) if steps_list else 0,
        max_steps=max(steps_list) if steps_list else 0,
        avg_duration=round(sum(durations) / len(durations), 1) if durations else 0,
        total_tokens_prompt=sum(t["tokens"].get("prompt", 0) for t in summaries),
        total_tokens_completion=sum(t["tokens"].get("completion", 0) for t in summaries),
        total_tokens_cache=sum(t["tokens"].get("cache_read", 0) for t in summaries),
        source_counts=dict(
            arvo=sum(1 for t in summaries if t["source"] == "arvo"),
            **{"oss-fuzz": sum(1 for t in summaries if t["source"] == "oss-fuzz")},
        ),
        status_by_source={
            src: dict(
                passed=sum(1 for t in summaries if t["source"] == src and t["status"] == "PASSED"),
                failed=sum(1 for t in summaries if t["source"] == src and t["status"] == "FAILED"),
                no_submit=sum(1 for t in summaries if t["source"] == src and t["status"] == "NO_SUBMIT"),
            )
            for src in ("arvo", "oss-fuzz")
        },
    )


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CyberGym Trajectory Viewer</title>
<style>
:root{--bg:#0d1117;--sf:#161b22;--sf2:#21262d;--bd:#30363d;--tx:#e6edf3;--tx2:#8b949e;--tx3:#6e7681;--ac:#58a6ff;--gn:#3fb950;--rd:#f85149;--yl:#d29922;--og:#db6d28;--pp:#bc8cff;--pk:#f778ba;--cy:#39d2c0}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:14px;line-height:1.5}
a{color:var(--ac);text-decoration:none}a:hover{text-decoration:underline}

.hdr{background:var(--sf);border-bottom:1px solid var(--bd);padding:14px 24px;display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:100}
.hdr h1{font-size:18px;font-weight:600}
.hdr .badge{background:var(--ac);color:var(--bg);padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600}
.hdr .nav{margin-left:auto;display:flex;gap:6px}
.hdr .nav button{background:var(--sf2);color:var(--tx);border:1px solid var(--bd);padding:5px 14px;border-radius:6px;cursor:pointer;font-size:13px}
.hdr .nav button:hover{background:var(--bd)}
.hdr .nav button.on{background:var(--ac);color:var(--bg);border-color:var(--ac)}
.ctr{max-width:1400px;margin:0 auto;padding:24px}

/* stat cards */
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin-bottom:22px}
.sc{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:14px}
.sc .lb{font-size:11px;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.sc .vl{font-size:26px;font-weight:700}
.sc .sb{font-size:11px;color:var(--tx3);margin-top:1px}

/* chart box */
.cr{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:22px}
.cb{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:14px}
.cb h3{font-size:13px;color:var(--tx2);margin-bottom:10px}

/* donut */
.dc{display:flex;align-items:center;gap:20px}
.dc canvas{width:170px!important;height:170px!important;flex-shrink:0}
.dl{display:flex;flex-direction:column;gap:5px}
.dl .it{display:flex;align-items:center;gap:7px;font-size:12px}
.dl .dt{width:10px;height:10px;border-radius:50%;flex-shrink:0}

/* bars */
.bc{display:flex;flex-direction:column;gap:5px}
.br{display:flex;align-items:center;gap:7px}
.bl{width:100px;font-size:11px;color:var(--tx2);text-align:right;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bt{flex:1;height:18px;background:var(--sf2);border-radius:3px;overflow:hidden;position:relative}
.bf{height:100%;border-radius:3px;transition:width .3s}
.bv{position:absolute;right:5px;top:50%;transform:translateY(-50%);font-size:10px;color:var(--tx)}

/* stacked */
.hs{display:flex;height:28px;border-radius:5px;overflow:hidden;margin:6px 0}
.hs .sg2{display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:var(--bg);min-width:22px}

/* table */
.tc{display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap;align-items:center}
.tc input,.tc select{background:var(--sf);border:1px solid var(--bd);color:var(--tx);padding:5px 10px;border-radius:6px;font-size:13px}
.tc input{width:240px}
.tt{width:100%;border-collapse:collapse;font-size:13px}
.tt th{background:var(--sf);border:1px solid var(--bd);padding:7px 10px;text-align:left;position:sticky;top:52px;z-index:10;cursor:pointer;user-select:none;white-space:nowrap}
.tt th:hover{background:var(--sf2)}
.tt td{border:1px solid var(--bd);padding:5px 10px}
.tt tr:hover td{background:var(--sf)}
.tt tr.ck{cursor:pointer}

/* badges */
.bd{padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;display:inline-block}
.bd.passed{background:rgba(63,185,80,.15);color:var(--gn)}
.bd.failed{background:rgba(248,81,73,.15);color:var(--rd)}
.bd.no-submit{background:rgba(210,153,34,.15);color:var(--yl)}
.bd.error{background:rgba(219,109,40,.15);color:var(--og)}
.bd.in-progress{background:rgba(88,166,255,.15);color:var(--ac)}

/* detail */
.dp{display:none}.dp.on{display:block}
.dh{display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap}
.dh .bb{background:var(--sf2);border:1px solid var(--bd);color:var(--tx);padding:5px 12px;border-radius:6px;cursor:pointer;font-size:13px}
.dh .bb:hover{background:var(--bd)}
.dm{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.dm .mi{background:var(--sf);border:1px solid var(--bd);border-radius:6px;padding:6px 12px;font-size:12px}
.dm .ml{color:var(--tx3)}.dm .mv{font-weight:600}

.ct{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:14px;margin-bottom:14px}
.ct h3{font-size:13px;color:var(--tx2);margin-bottom:6px}
svg.sp{width:100%;height:60px}
svg.sp .ln{fill:none;stroke:var(--ac);stroke-width:1.5}
svg.sp .ar{fill:rgba(88,166,255,.1)}

.fc{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
.fc button{background:var(--sf2);border:1px solid var(--bd);color:var(--tx);padding:3px 9px;border-radius:4px;cursor:pointer;font-size:12px}
.fc button:hover{background:var(--bd)}
.fc button.on{background:var(--ac);color:var(--bg);border-color:var(--ac)}

.st{background:var(--sf);border:1px solid var(--bd);border-radius:6px;margin-bottom:6px;overflow:hidden}
.st.ssub{border-color:var(--yl)}.st.spass{border-color:var(--gn)}.st.sfail{border-color:var(--rd)}
.sh{display:flex;align-items:center;gap:6px;padding:7px 10px;cursor:pointer;user-select:none}
.sh:hover{background:var(--sf2)}
.si{background:var(--sf2);color:var(--tx3);padding:1px 5px;border-radius:3px;font-size:10px;font-family:monospace;min-width:26px;text-align:center}
.stp{padding:1px 7px;border-radius:3px;font-size:10px;font-weight:600;font-family:monospace}
.stp.run{background:rgba(88,166,255,.15);color:var(--ac)}
.stp.read{background:rgba(188,140,255,.15);color:var(--pp)}
.stp.edit{background:rgba(247,120,186,.15);color:var(--pk)}
.stp.message{background:rgba(139,148,158,.15);color:var(--tx2)}
.stp.recall{background:rgba(139,148,158,.1);color:var(--tx3)}
.stp.finish{background:rgba(63,185,80,.15);color:var(--gn)}
.stp.error{background:rgba(248,81,73,.15);color:var(--rd)}
.stp.condensation{background:rgba(57,210,192,.15);color:var(--cy)}
.stp.browse{background:rgba(210,153,34,.15);color:var(--yl)}
.stp.run_ipython{background:rgba(88,166,255,.25);color:var(--ac)}
.ss{font-size:11px;color:var(--tx3)}
.sm{font-size:12px;color:var(--tx2);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.scs{font-size:10px;color:var(--tx3);font-family:monospace;white-space:nowrap}
.sb2{display:none;padding:10px;border-top:1px solid var(--bd)}
.st.ex .sb2{display:block}
.sb2 pre{background:var(--bg);border:1px solid var(--bd);border-radius:4px;padding:8px;overflow-x:auto;font-size:12px;font-family:'SFMono-Regular',Consolas,monospace;white-space:pre-wrap;word-break:break-all;max-height:400px;overflow-y:auto}
.sb2 .th{background:rgba(57,210,192,.05);border:1px solid rgba(57,210,192,.2);border-radius:4px;padding:8px;margin-bottom:6px;font-size:12px;color:var(--cy);font-style:italic}
.sb2 .lbl{font-size:10px;color:var(--tx3);text-transform:uppercase;margin-bottom:3px;margin-top:6px}
.sbg{padding:1px 5px;border-radius:3px;font-size:10px;font-weight:700}
.sbg.p{background:var(--gn);color:var(--bg)}.sbg.f{background:var(--rd);color:#fff}

.ac{display:flex;flex-wrap:wrap;gap:4px}
.ap{display:flex;align-items:center;gap:3px;background:var(--sf2);border-radius:3px;padding:1px 6px;font-size:10px}
.ap .cn{font-weight:700}

.loading{text-align:center;padding:40px;color:var(--tx2)}

@media(max-width:900px){.cr{grid-template-columns:1fr}.sg{grid-template-columns:repeat(2,1fr)}}
::-webkit-scrollbar{width:7px;height:7px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--bd);border-radius:4px}
</style>
</head>
<body>

<div class="hdr">
  <h1>CyberGym Trajectory Viewer</h1>
  <span class="badge" id="model-badge"></span>
  <div class="nav">
    <button class="on" onclick="showView('dashboard',this)">Dashboard</button>
    <button onclick="showView('tasks',this)">Tasks</button>
    <button id="refresh-btn" onclick="doRefresh()" title="Rescan logs directory">&#8635;</button>
  </div>
</div>

<div class="ctr">
  <div id="view-dashboard"><div class="loading">Loading dashboard...</div></div>
  <div id="view-tasks" style="display:none">
    <div id="task-list-panel">
      <div class="tc">
        <input type="text" id="search-input" placeholder="Search tasks..." oninput="filterTasks()">
        <select id="status-filter" onchange="filterTasks()"><option value="">All Statuses</option><option value="PASSED">Passed</option><option value="FAILED">Failed</option><option value="NO_SUBMIT">No Submit</option><option value="ERROR">Error</option></select>
        <select id="source-filter" onchange="filterTasks()"><option value="">All Sources</option><option value="arvo">Arvo</option><option value="oss-fuzz">OSS-Fuzz</option></select>
        <select id="sort-select" onchange="sortAndRender()"><option value="name-asc">Name ↑</option><option value="name-desc">Name ↓</option><option value="cost-desc">Cost ↓</option><option value="cost-asc">Cost ↑</option><option value="steps-desc">Steps ↓</option><option value="steps-asc">Steps ↑</option><option value="duration-desc">Duration ↓</option><option value="submits-desc">Submits ↓</option></select>
      </div>
      <table class="tt"><thead><tr><th>#</th><th>Task</th><th>Source</th><th>Difficulty</th><th>Status</th><th>Steps</th><th>Submits</th><th>Cost</th><th>Duration</th><th>Actions</th></tr></thead><tbody id="task-tbody"></tbody></table>
    </div>
    <div id="detail-panel" class="dp">
      <div class="dh">
        <button class="bb" onclick="hideDetail()">&#8592; Back</button>
        <h2 id="detail-title"></h2>
        <span id="detail-badge"></span>
      </div>
      <div class="dm" id="detail-meta"></div>
      <div class="ct"><h3>Cost Accumulation</h3><svg id="detail-sparkline" class="sp" viewBox="0 0 1000 60"></svg></div>
      <div class="fc" id="traj-controls"></div>
      <div id="traj-steps"></div>
    </div>
  </div>
</div>

<script>
const SC={PASSED:'#3fb950',FAILED:'#f85149',NO_SUBMIT:'#d29922',ERROR:'#db6d28','IN_PROGRESS':'#58a6ff'};
let TASKS=[],STATS={},stepCache={},curFolder=null;

function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function fmtDur(s){return s>0?(s>=60?Math.round(s/60)+'m':Math.round(s)+'s'):'—'}

async function api(path){const r=await fetch(path);return r.json()}

// ---- Views ----
function showView(v,btn){
  document.getElementById('view-dashboard').style.display=v==='dashboard'?'block':'none';
  document.getElementById('view-tasks').style.display=v==='tasks'?'block':'none';
  document.querySelectorAll('.hdr .nav button').forEach(b=>b.classList.remove('on'));
  if(btn)btn.classList.add('on');
  if(v==='tasks')hideDetail();
}

// ---- Init ----
async function init(){
  const [tasks,stats]=await Promise.all([api('/api/tasks'),api('/api/stats')]);
  TASKS=tasks;STATS=stats;stepCache={};
  document.getElementById('model-badge').textContent=TASKS.length?TASKS[0].model:'';
  renderDashboard();
  filterTasks();
}

async function doRefresh(){
  const btn=document.getElementById('refresh-btn');
  btn.disabled=true;btn.style.opacity='.5';
  try{await api('/api/refresh');await init()}
  catch(e){btn.textContent='\u2717';setTimeout(()=>btn.textContent='\u21BB',2000)}
  finally{btn.style.opacity='1';btn.disabled=false}
}

// ---- Dashboard ----
function renderDashboard(){
  const S=STATS,el=document.getElementById('view-dashboard');
  el.innerHTML=`
  <div class="sg">
    <div class="sc"><div class="lb">Total Tasks</div><div class="vl" style="color:var(--ac)">${S.total}</div><div class="sb">${S.source_counts.arvo} arvo, ${S.source_counts['oss-fuzz']} oss-fuzz</div></div>
    <div class="sc"><div class="lb">Passed</div><div class="vl" style="color:var(--gn)">${S.passed}</div><div class="sb">${S.pass_rate}% overall · ${S.pass_rate_submitted}% of submitted</div></div>
    <div class="sc"><div class="lb">Failed / No Submit</div><div class="vl" style="color:var(--rd)">${S.failed} <span style="color:var(--yl)">/ ${S.no_submit}</span></div><div class="sb">${S.error} errors</div></div>
    <div class="sc"><div class="lb">Total Cost</div><div class="vl">$${S.total_cost}</div><div class="sb">avg $${S.avg_cost} · range $${S.min_cost}–$${S.max_cost}</div></div>
    <div class="sc"><div class="lb">Avg Steps</div><div class="vl">${S.avg_steps}</div><div class="sb">range ${S.min_steps}–${S.max_steps}</div></div>
    <div class="sc"><div class="lb">Avg Duration</div><div class="vl">${Math.round(S.avg_duration/60)}m</div><div class="sb">${Math.round(S.avg_duration)}s avg</div></div>
  </div>
  <div class="cr">
    <div class="cb"><h3>Status Distribution</h3><div class="dc"><canvas id="donut" width="170" height="170"></canvas><div class="dl" id="donut-leg"></div></div></div>
    <div class="cb"><h3>Source Breakdown</h3><div id="src-chart"></div></div>
  </div>
  <div class="cr">
    <div class="cb"><h3>Cost Distribution (USD)</h3><div id="cost-hist"></div></div>
    <div class="cb"><h3>Steps Distribution</h3><div id="steps-hist"></div></div>
  </div>
  <div class="cr">
    <div class="cb"><h3>Token Usage</h3><div id="tok-chart"></div></div>
    <div class="cb"><h3>Cost vs Steps</h3><svg id="scatter" viewBox="0 0 500 220" style="width:100%;height:220px"></svg></div>
  </div>
  <div class="cb" style="margin-bottom:22px"><h3>All Tasks — Cost &amp; Status</h3><div id="task-bars" style="max-height:600px;overflow-y:auto"></div></div>`;
  drawDonut();renderSrcChart();
  renderHist('cost-hist',TASKS.map(t=>t.cost),'$',10);
  renderHist('steps-hist',TASKS.map(t=>t.num_steps),'',10);
  renderTokens();renderScatter();renderTaskBars();
}

function drawDonut(){
  const c=document.getElementById('donut'),ctx=c.getContext('2d'),S=STATS;
  const data=[{l:'Passed',v:S.passed,c:'#3fb950'},{l:'Failed',v:S.failed,c:'#f85149'},{l:'No Submit',v:S.no_submit,c:'#d29922'},{l:'Error',v:S.error,c:'#db6d28'}].filter(d=>d.v>0);
  const cx=85,cy=85,R=78,r=48,tot=data.reduce((s,d)=>s+d.v,0);
  let a=-Math.PI/2;ctx.clearRect(0,0,170,170);
  data.forEach(d=>{const sl=(d.v/tot)*Math.PI*2;ctx.beginPath();ctx.arc(cx,cy,R,a,a+sl);ctx.arc(cx,cy,r,a+sl,a,true);ctx.closePath();ctx.fillStyle=d.c;ctx.fill();a+=sl});
  ctx.fillStyle='#e6edf3';ctx.font='bold 22px -apple-system,sans-serif';ctx.textAlign='center';ctx.fillText(tot,cx,cy+3);
  ctx.font='10px -apple-system,sans-serif';ctx.fillStyle='#8b949e';ctx.fillText('tasks',cx,cy+16);
  document.getElementById('donut-leg').innerHTML=data.map(d=>`<div class="it"><div class="dt" style="background:${d.c}"></div>${d.l}: ${d.v} (${(d.v/tot*100).toFixed(0)}%)</div>`).join('');
}

function renderSrcChart(){
  const el=document.getElementById('src-chart'),S=STATS;let h='';
  ['arvo','oss-fuzz'].forEach(src=>{
    const d=S.status_by_source[src],tot=d.passed+d.failed+d.no_submit;if(!tot)return;
    h+=`<div style="margin-bottom:10px"><div style="font-size:12px;font-weight:600;margin-bottom:3px">${src} (${tot})</div><div class="hs">`;
    if(d.passed)h+=`<div class="sg2" style="flex:${d.passed};background:#3fb950">${d.passed}</div>`;
    if(d.failed)h+=`<div class="sg2" style="flex:${d.failed};background:#f85149">${d.failed}</div>`;
    if(d.no_submit)h+=`<div class="sg2" style="flex:${d.no_submit};background:#d29922">${d.no_submit}</div>`;
    h+=`</div><div style="display:flex;gap:10px;font-size:10px;color:var(--tx3)"><span style="color:#3fb950">● ${d.passed} passed</span><span style="color:#f85149">● ${d.failed} failed</span><span style="color:#d29922">● ${d.no_submit} no submit</span></div></div>`;
  });el.innerHTML=h;
}

function renderHist(id,vals,pfx,n){
  const el=document.getElementById(id);if(!vals.length){el.innerHTML='No data';return}
  const mn=Math.min(...vals),mx=Math.max(...vals),rng=mx-mn||1,bs=rng/n,bins=Array(n).fill(0);
  vals.forEach(v=>{let b=Math.floor((v-mn)/bs);if(b>=n)b=n-1;bins[b]++});const mb=Math.max(...bins);
  let h='<div class="bc">';
  bins.forEach((c,i)=>{const lo=(mn+i*bs).toFixed(2),hi=(mn+(i+1)*bs).toFixed(2),p=mb?Math.round(c/mb*100):0;
    h+=`<div class="br"><div class="bl">${pfx}${parseFloat(lo).toFixed(mn>10?0:2)}–${parseFloat(hi).toFixed(mn>10?0:2)}</div><div class="bt"><div class="bf" style="width:${p}%;background:var(--ac)"></div><div class="bv">${c}</div></div></div>`;
  });h+='</div>';el.innerHTML=h;
}

function renderTokens(){
  const el=document.getElementById('tok-chart'),S=STATS;
  const it=[{l:'Prompt',v:S.total_tokens_prompt,c:'var(--ac)'},{l:'Completion',v:S.total_tokens_completion,c:'var(--gn)'},{l:'Cache Read',v:S.total_tokens_cache,c:'var(--pp)'}];
  const mx=Math.max(...it.map(i=>i.v));
  let h='<div class="bc">';
  it.forEach(i=>{const p=mx?Math.round(i.v/mx*100):0;h+=`<div class="br"><div class="bl">${i.l}</div><div class="bt"><div class="bf" style="width:${p}%;background:${i.c}"></div><div class="bv">${(i.v/1e6).toFixed(1)}M</div></div></div>`});
  const cr=S.total_tokens_prompt>0?(S.total_tokens_cache/S.total_tokens_prompt*100).toFixed(1):0;
  h+=`<div style="font-size:11px;color:var(--tx3);margin-top:6px">Cache hit rate: ${cr}%</div></div>`;el.innerHTML=h;
}

function renderScatter(){
  const svg=document.getElementById('scatter'),P={l:45,r:15,t:15,b:30},w=500-P.l-P.r,h=220-P.t-P.b;
  const ss=TASKS.map(t=>t.num_steps),cs=TASKS.map(t=>t.cost),mxS=Math.max(...ss)||1,mxC=Math.max(...cs)||1;
  let o='';
  o+=`<line x1="${P.l}" y1="${P.t}" x2="${P.l}" y2="${P.t+h}" stroke="#30363d"/>`;
  o+=`<line x1="${P.l}" y1="${P.t+h}" x2="${P.l+w}" y2="${P.t+h}" stroke="#30363d"/>`;
  o+=`<text x="${P.l+w/2}" y="${P.t+h+25}" fill="#8b949e" font-size="11" text-anchor="middle">Steps</text>`;
  o+=`<text x="12" y="${P.t+h/2}" fill="#8b949e" font-size="11" text-anchor="middle" transform="rotate(-90,12,${P.t+h/2})">Cost ($)</text>`;
  for(let i=0;i<=4;i++){const y=P.t+h-i*(h/4);o+=`<text x="${P.l-4}" y="${y+4}" fill="#6e7681" font-size="10" text-anchor="end">${(mxC*i/4).toFixed(2)}</text><line x1="${P.l}" y1="${y}" x2="${P.l+w}" y2="${y}" stroke="#21262d" stroke-width=".5"/>`}
  for(let i=0;i<=4;i++){const x=P.l+i*(w/4);o+=`<text x="${x}" y="${P.t+h+14}" fill="#6e7681" font-size="10" text-anchor="middle">${Math.round(mxS*i/4)}</text>`}
  TASKS.forEach(t=>{const x=P.l+(t.num_steps/mxS)*w,y=P.t+h-(t.cost/mxC)*h,c=SC[t.status]||'#8b949e';
    o+=`<circle cx="${x}" cy="${y}" r="4" fill="${c}" opacity=".8" style="cursor:pointer" onclick="showView('tasks',document.querySelectorAll('.hdr .nav button')[1]);openTask('${t.folder}')"><title>${t.task_name}\nSteps: ${t.num_steps}\nCost: $${t.cost}</title></circle>`});
  svg.innerHTML=o;
}

function renderTaskBars(){
  const el=document.getElementById('task-bars'),sorted=[...TASKS].sort((a,b)=>b.cost-a.cost),mx=Math.max(...sorted.map(t=>t.cost))||1;
  let h='<div class="bc">';
  sorted.forEach(t=>{const p=Math.round(t.cost/mx*100),c=SC[t.status]||'#8b949e',nm=t.task_name.length>22?t.task_name.slice(0,20)+'…':t.task_name;
    h+=`<div class="br" style="cursor:pointer" onclick="showView('tasks',document.querySelectorAll('.hdr .nav button')[1]);openTask('${t.folder}')"><div class="bl" title="${esc(t.task_name)}">${esc(nm)}</div><div class="bt"><div class="bf" style="width:${p}%;background:${c}"></div><div class="bv">$${t.cost.toFixed(3)}</div></div></div>`});
  h+='</div>';el.innerHTML=h;
}

// ---- Tasks Table ----
let filteredTasks=[];
function filterTasks(){
  const q=document.getElementById('search-input').value.toLowerCase(),st=document.getElementById('status-filter').value,sr=document.getElementById('source-filter').value;
  filteredTasks=TASKS.filter(t=>{if(q&&!t.task_name.toLowerCase().includes(q)&&!t.task_id.toLowerCase().includes(q))return false;if(st&&t.status!==st)return false;if(sr&&t.source!==sr)return false;return true});
  sortAndRender();
}
function sortAndRender(){
  const[key,dir]=document.getElementById('sort-select').value.split('-'),m=dir==='desc'?-1:1;
  filteredTasks.sort((a,b)=>{if(key==='name')return m*a.task_name.localeCompare(b.task_name);if(key==='cost')return m*(a.cost-b.cost);if(key==='steps')return m*(a.num_steps-b.num_steps);if(key==='duration')return m*(a.duration_s-b.duration_s);if(key==='submits')return m*(a.submit_attempts-b.submit_attempts);return 0});
  renderTable();
}
function renderTable(){
  const tb=document.getElementById('task-tbody');let h='';
  filteredTasks.forEach((t,i)=>{const sc=t.status.toLowerCase().replace('_','-'),dur=fmtDur(t.duration_s);
    h+=`<tr class="ck" onclick="openTask('${t.folder}')"><td>${i+1}</td><td><strong>${esc(t.task_name)}</strong></td><td>${t.source}</td><td>${t.difficulty}</td><td><span class="bd ${sc}">${t.status}</span></td><td>${t.num_steps}</td><td>${t.submit_attempts}</td><td>$${t.cost.toFixed(4)}</td><td>${dur}</td><td><div class="ac">${Object.entries(t.action_counts||{}).map(([k,v])=>`<span class="ap"><span class="stp ${k}" style="padding:0 3px;font-size:9px">${k}</span><span class="cn">${v}</span></span>`).join('')}</div></td></tr>`});
  tb.innerHTML=h;
}

// ---- Detail ----
async function openTask(folder){
  curFolder=folder;
  const task=TASKS.find(t=>t.folder===folder);if(!task)return;
  document.getElementById('task-list-panel').style.display='none';
  const dp=document.getElementById('detail-panel');dp.classList.add('on');
  document.getElementById('detail-title').textContent=task.task_name;
  document.getElementById('detail-badge').innerHTML=`<span class="bd ${task.status.toLowerCase().replace('_','-')}">${task.status}</span>`;
  document.getElementById('detail-meta').innerHTML=`
    <div class="mi"><span class="ml">Task ID: </span><span class="mv">${esc(task.task_id)}</span></div>
    <div class="mi"><span class="ml">Difficulty: </span><span class="mv">${task.difficulty}</span></div>
    <div class="mi"><span class="ml">Steps: </span><span class="mv">${task.num_steps}</span></div>
    <div class="mi"><span class="ml">Submits: </span><span class="mv">${task.submit_attempts}</span></div>
    <div class="mi"><span class="ml">Cost: </span><span class="mv">$${task.cost.toFixed(4)}</span></div>
    <div class="mi"><span class="ml">Duration: </span><span class="mv">${fmtDur(task.duration_s)}</span></div>
    <div class="mi"><span class="ml">Prompt Tokens: </span><span class="mv">${(task.tokens.prompt||0).toLocaleString()}</span></div>
    <div class="mi"><span class="ml">Completion Tokens: </span><span class="mv">${(task.tokens.completion||0).toLocaleString()}</span></div>`;

  // Load steps via API
  document.getElementById('traj-steps').innerHTML='<div class="loading">Loading trajectory...</div>';
  let steps;
  if(stepCache[folder]){steps=stepCache[folder]}
  else{steps=await api('/api/steps/'+encodeURIComponent(folder));stepCache[folder]=steps}

  drawSparkline(steps);
  const types=[...new Set(steps.map(s=>s.type))];
  let ch=`<button class="on" onclick="doFilter('all',this)">All (${steps.length})</button><button onclick="doFilter('agent',this)">Agent Only</button>`;
  types.forEach(t=>{const n=steps.filter(s=>s.type===t).length;ch+=`<button onclick="doFilter('${t}',this)">${t} (${n})</button>`});
  document.getElementById('traj-controls').innerHTML=ch;
  renderSteps(steps,'all');
}

function doFilter(f,btn){
  document.querySelectorAll('#traj-controls button').forEach(b=>b.classList.remove('on'));
  if(btn)btn.classList.add('on');
  const steps=stepCache[curFolder];if(!steps)return;
  renderSteps(steps,f);
}

function hideDetail(){document.getElementById('detail-panel').classList.remove('on');document.getElementById('task-list-panel').style.display='block'}

function renderSteps(steps,filter){
  let fl=steps;
  if(filter==='agent')fl=steps.filter(s=>s.src==='agent');
  else if(filter!=='all')fl=steps.filter(s=>s.type===filter);
  const el=document.getElementById('traj-steps');let h='';
  fl.forEach(s=>{
    let ec='';if(s.is_submit){ec=s.submit_exit_code&&s.submit_exit_code!==0?'spass':'ssub'}
    let cs=s.cost!==undefined?`$${s.cost.toFixed(4)}`:'';
    let sb='';if(s.is_submit&&s.submit_exit_code!==undefined){sb=s.submit_exit_code!==0?'<span class="sbg p">CRASH</span>':'<span class="sbg f">NO CRASH</span>'}
    let bd='';
    if(s.thinking)bd+=`<div class="th"><div class="lbl">LLM Thinking</div>${esc(s.thinking)}</div>`;
    if(s.command){bd+=`<div class="lbl">Command</div><pre>${esc(s.command)}</pre>`}
    if(s.path){bd+=`<div class="lbl">Path</div><pre>${esc(s.path)}</pre>`}
    if(s.output){bd+=`<div class="lbl">Output</div><pre>${esc(s.output)}</pre>`}
    if(s.exit_code!==undefined&&s.exit_code!==null)bd+=`<div class="lbl">Exit Code: ${s.exit_code}</div>`;
    if(s.msg_full)bd+=`<div class="lbl">Message</div><pre>${esc(s.msg_full)}</pre>`;
    else if(s.type==='finish'||s.type==='condensation'||s.type==='error')bd+=`<div class="lbl">Message</div><pre>${esc(s.msg)}</pre>`;
    h+=`<div class="st ${ec}" onclick="this.classList.toggle('ex')"><div class="sh"><span class="si">${s.id}</span><span class="stp ${s.type}">${s.type}</span><span class="ss">${s.src}</span><span class="sm">${esc((s.msg||'').split('\\n')[0])}</span>${sb}<span class="scs">${cs}</span></div>${bd?`<div class="sb2">${bd}</div>`:''}</div>`;
  });el.innerHTML=h;
}

function drawSparkline(steps){
  const svg=document.getElementById('detail-sparkline'),cs=steps.filter(s=>s.cost!==undefined);
  if(!cs.length){svg.innerHTML='';return}
  const mx=Math.max(...cs.map(s=>s.cost))||1,w=1000,h=60,p=2,xs=(w-p*2)/(cs.length-1||1);
  let pts=cs.map((s,i)=>[p+i*xs,h-p-(s.cost/mx)*(h-p*2)]);
  let pd=pts.map((pt,i)=>(i===0?'M':'L')+pt[0]+','+pt[1]).join(' ');
  let ad=pd+` L${pts[pts.length-1][0]},${h} L${pts[0][0]},${h} Z`;
  let o=`<path class="ar" d="${ad}"/><path class="ln" d="${pd}"/>`;
  steps.forEach(s=>{if(s.is_submit&&s.cost!==undefined){const ci=cs.indexOf(s);if(ci>=0){const x=p+ci*xs,cl=s.submit_exit_code&&s.submit_exit_code!==0?'#3fb950':'#f85149';o+=`<line x1="${x}" y1="0" x2="${x}" y2="${h}" stroke="${cl}" stroke-width="1.5" opacity=".6"/>`}}});
  o+=`<text x="2" y="10" fill="#6e7681" font-size="9">$0</text><text x="${w-4}" y="10" fill="#8b949e" font-size="9" text-anchor="end">$${mx.toFixed(3)}</text>`;
  svg.innerHTML=o;
}

init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class ViewerHandler(BaseHTTPRequestHandler):
    """Serves the trajectory viewer UI and JSON API."""

    logs_dir: str = ""
    _summaries: list[dict] = []
    _stats: dict = {}

    def log_message(self, fmt, *args):
        # Quieter logging
        sys.stderr.write(f"  {args[0]} {args[1]}\n")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/":
            self._send_html(INDEX_HTML)
        elif path == "/api/stats":
            self._send_json(self.__class__._stats)
        elif path == "/api/tasks":
            self._send_json(self.__class__._summaries)
        elif path == "/api/refresh":
            logs_dir = self.__class__.logs_dir
            folders = sorted(
                d for d in os.listdir(logs_dir)
                if os.path.isdir(os.path.join(logs_dir, d))
            )
            summaries = [extract_task_summary(logs_dir, f) for f in folders]
            stats = compute_stats(summaries)
            self.__class__._summaries = summaries
            self.__class__._stats = stats
            print(f"  Refreshed: {len(summaries)} tasks, {stats['passed']} passed")
            self._send_json({"ok": True, "total": len(summaries)})
        elif path.startswith("/api/steps/"):
            folder = path[len("/api/steps/"):]
            # Validate folder exists
            if folder in {s["folder"] for s in self.__class__._summaries}:
                steps = extract_steps(self.__class__.logs_dir, folder)
                self._send_json(steps)
            else:
                self._send_json({"error": "not found"}, 404)
        else:
            self._send_json({"error": "not found"}, 404)

    def _send_html(self, body: str):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(
        description="CyberGym Trajectory Viewer — visualize and explore eval trajectories"
    )
    parser.add_argument("--logs_dir", required=True, help="Path to eval logs directory")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8667, help="Port to serve on (default: 8667)")
    args = parser.parse_args()

    if not os.path.isdir(args.logs_dir):
        print(f"Error: {args.logs_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Scan tasks on startup
    print(f"Scanning {args.logs_dir} ...")
    folders = sorted(
        d for d in os.listdir(args.logs_dir)
        if os.path.isdir(os.path.join(args.logs_dir, d))
    )
    summaries = [extract_task_summary(args.logs_dir, f) for f in folders]
    stats = compute_stats(summaries)

    print(f"  {len(summaries)} tasks: {stats['passed']} passed, "
          f"{stats['failed']} failed, {stats['no_submit']} no_submit, "
          f"{stats['error']} error")
    print(f"  Total cost: ${stats['total_cost']}")

    # Attach data to handler class
    ViewerHandler.logs_dir = args.logs_dir
    ViewerHandler._summaries = summaries
    ViewerHandler._stats = stats

    server = HTTPServer((args.host, args.port), ViewerHandler)
    print(f"\nTrajectory Viewer running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
