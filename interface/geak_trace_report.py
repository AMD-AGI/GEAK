#!/usr/bin/env python3
"""Render a GEAK execution trace as an interactive HTML report (plus Markdown).

Reads ``geak_trace.json`` (see ``geak_trace_collector.py``) and produces three
synchronized views over the same run:

  * a **phase tree** grouping agents into Setup / Round N / Finalize -- derived
    from journal LABELS, so it is shown as INFERRED, never as recorded truth;
  * a **timeline** of agent spans, positioned as an offset from the run origin
    so parallel work reads as overlapping bars (spans are never summed);
  * a **delegation graph** of the edges the journal actually proves --
    workflow -> agent invocation, and agent -> workflow return.

Selecting any node drills down to that agent's API calls, and each call opens to
the four things a call actually consists of: the new input since the previous
response, the reasoning marker, the text output, and the tool actions with their
matched results. Consecutive calls link forward and back, so a chain of calls
reads as the connected sequence it was.

Honesty rules enforced here: timings are labelled estimated, inferred groupings
are labelled inferred, unavailable data renders as an explicit marker rather
than a blank, and an input excerpt is never presented as the full API request.
"""

import argparse
import html
import json
import os
import re
import sys

SCHEMA_IN = "geak.trace/1"

# A round tag like "r3" in "eng r3_d1:compute", "verify r1_d0" or "reprofile r2".
# No trailing \b: the tag is routinely followed by "_" (as in "r1_d0"), which is
# a word character, so requiring a boundary there would silently drop every
# engineer and verifier out of its round.
_ROUND_RE = re.compile(r"\br(\d+)")


def esc(text):
    """HTML-escape for text nodes and attributes."""
    return html.escape("" if text is None else str(text), quote=True)


def embed_json(obj):
    """Serialize for a <script> block without allowing a tag breakout."""
    raw = json.dumps(obj, ensure_ascii=False, default=str)
    return (raw.replace("<", "\\u003c").replace(">", "\\u003e")
               .replace("&", "\\u0026").replace("\u2028", "\\u2028")
               .replace("\u2029", "\\u2029"))


def fmt_usd(value):
    if value is None:
        return "—"
    return "$%.4f" % value


def fmt_dur(ms):
    if ms is None:
        return "—"
    s = ms / 1000.0
    if s < 60:
        return "%.1fs" % s
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return ("%dh%02dm%02ds" % (h, m, s)) if h else ("%dm%02ds" % (m, s))


def fmt_int(n):
    return "—" if n is None else "{:,}".format(n)


def phase_of(agent, first_round_seen):
    """Infer a phase bucket from the journal label.

    The journal's own ``phase`` field is a single constant for the whole run, so
    the only available grouping signal is the label text. That makes this an
    INFERRED grouping, and it is labelled as such everywhere it is shown.
    """
    label = (agent.get("label") or "").strip()
    m = _ROUND_RE.search(label)
    if m:
        return "Round %s" % m.group(1), int(m.group(1))
    lowered = label.lower()
    if not first_round_seen:
        return "Setup", -1
    if any(k in lowered for k in ("report", "validate", "experience", "final")):
        return "Finalize", 10 ** 6
    return "Between rounds", 10 ** 5


def group_phases(agents):
    """Bucket agents into ordered phases, preserving journal order within each.

    The journal's ``phase`` field is the RECORDED phase and is used whenever it
    actually distinguishes anything -- a real E2E run carries a full taxonomy
    there (Setup, WarmStart, Profile, Strategize, ConfigSweep, ... Report,
    Validate). Only when every agent shares one constant phase (as on the kernel
    lane, where it is just the lane name) does this fall back to parsing round
    tags out of labels, and that fallback is marked INFERRED.

    Grouping is by CONTIGUOUS run in journal order, so a phase that recurs later
    in the run becomes a second group rather than being folded into the first.
    """
    # Preference: the workflow's OWN recorded per-agent phase (survives nesting,
    # which collapses a nested lane's phases in the parent journal), then the
    # journal phase when it distinguishes anything, then label inference.
    tl_phases = {(a.get("timeline_phase") or "") for a in agents
                 if a.get("timeline_phase")}
    use_timeline = len(tl_phases) > 1
    distinct = {(a.get("journal_phase") or "") for a in agents}
    use_journal = (not use_timeline) and len(distinct) > 1

    groups, seen_round = [], False
    for agent in agents:
        if use_timeline:
            name = agent.get("timeline_phase") or "(no phase recorded)"
            provenance = "workflow_timeline"
        elif use_journal:
            name = agent.get("journal_phase") or "(no phase recorded)"
            provenance = "journal"
        else:
            name, rank = phase_of(agent, seen_round)
            if 0 <= rank < 10 ** 5:
                seen_round = True
            provenance = "inferred"
        if groups and groups[-1]["name"] == name:
            groups[-1]["agents"].append(agent["agent_id"])
        else:
            groups.append({"name": name, "provenance": provenance,
                           "agents": [agent["agent_id"]]})
    return groups


def build_view(trace):
    """Reshape the trace into exactly what the page needs, with totals."""
    agents = trace.get("agents") or []
    run = trace.get("run") or {}
    origin = run.get("origin_ts_ms")

    nodes = []
    for agent in agents:
        first, last = agent.get("first_ts_ms"), agent.get("last_ts_ms")
        nodes.append({
            "id": agent["agent_id"],
            "ordinal": agent.get("ordinal"),
            "label": agent.get("label") or agent.get("description") or agent["agent_id"],
            "description": agent.get("description"),
            "status": agent.get("status"),
            "result_status": agent.get("result_status"),
            "result_preview": agent.get("result_preview"),
            "result_truncated": agent.get("result_truncated"),
            "spawn_depth": agent.get("spawn_depth"),
            "transcript_status": agent.get("transcript_status"),
            "timeline_phase": agent.get("timeline_phase"),
            "timeline_sub_phase": agent.get("timeline_sub_phase"),
            "phase_provenance": agent.get("phase_provenance"),
            "start_off": (None if (first is None or origin is None) else first - origin),
            "end_off": (None if (last is None or origin is None) else last - origin),
            "dur": (None if (first is None or last is None) else last - first),
            "totals": agent.get("totals") or {},
            "calls": agent.get("calls") or [],
        })

    # Phase grouping is the one place this renderer depends on the WORDING of
    # workflow labels, so it is also the one place a future label change could
    # quietly misfile agents. Surface that instead of letting it pass silently:
    # a bucket that only exists because nothing matched is a signal, not a fact.
    phases = group_phases(agents)
    warnings = list(trace.get("warnings") or [])
    inferred = [p for p in phases if p.get("provenance") == "inferred"]
    unparsed = next((p for p in phases if p["name"] == "Between rounds"), None)
    if unparsed:
        warnings.append(
            "%d agent label(s) did not match any known phase pattern and were "
            "grouped as 'Between rounds' (labels: %s). If workflow label wording "
            "changed, the phase grouping needs updating -- the per-agent and "
            "per-call data below is unaffected."
            % (len(unparsed["agents"]),
               ", ".join(sorted({(next((a.get("label") for a in agents
                                        if a["agent_id"] == aid), aid) or aid)
                                 for aid in unparsed["agents"]})[:6])))
    if agents and inferred and len(phases) == 1:
        warnings.append(
            "Every agent fell into a single phase bucket (%s); the label-derived "
            "phase grouping may no longer match this workflow's labels."
            % phases[0]["name"])
    if agents and not inferred:
        warnings.append(
            "Phases below are the workflow's OWN recorded journal phases, not "
            "inferred from labels.")

    totals = {
        "agents": len(nodes),
        "returned": sum(1 for n in nodes if n["result_status"] == "returned_to_workflow"),
        "calls": sum(len(n["calls"]) for n in nodes),
        "actions": sum((n["totals"].get("actions") or 0) for n in nodes),
        "cost_usd": sum((n["totals"].get("cost_usd") or 0.0) for n in nodes),
        "output_tokens": sum((n["totals"].get("output_tokens") or 0) for n in nodes),
        "input_tokens": sum((n["totals"].get("input_tokens") or 0) for n in nodes),
        "cache_read": sum((n["totals"].get("cache_read_input_tokens") or 0) for n in nodes),
        "usage_unknown_calls": sum((n["totals"].get("usage_unknown_calls") or 0)
                                   for n in nodes),
    }
    if totals["usage_unknown_calls"]:
        warnings.append(
            "%d API call(s) have NO usage recorded; their tokens and cost are "
            "UNKNOWN and are excluded from the totals above rather than counted "
            "as zero." % totals["usage_unknown_calls"])
    record_status = run.get("record_status")
    if record_status and str(record_status).lower() not in ("completed", "running"):
        warnings.append(
            "Workflow outcome was '%s'. A complete CAPTURE is not evidence of a "
            "successful run." % record_status)
    return {
        "run": run,
        "phases": phases,
        "agents": nodes,
        "edges": trace.get("edges") or [],
        "linkage": (trace.get("run") or {}).get("linkage"),
        "warnings": warnings,
        "totals": totals,
    }


_CSS = """
:root{--bg:#0f1117;--panel:#171a23;--line:#272c3a;--fg:#e6e8ef;--dim:#9aa3b8;
--accent:#6ea8fe;--ok:#4ec9a5;--warn:#e0b050;--err:#e06c75;--chip:#222736;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:13px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:14px 18px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0 0 4px;font-size:16px;font-weight:650}
.sub{color:var(--dim);font-size:12px}
.kpis{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.kpi{background:var(--chip);border:1px solid var(--line);border-radius:7px;padding:6px 10px}
.kpi b{display:block;font-size:14px}
.kpi span{color:var(--dim);font-size:11px}
.note{margin:10px 18px 0;padding:8px 11px;border-left:3px solid var(--warn);
background:#1d1a12;color:#e8dcc0;border-radius:0 6px 6px 0;font-size:12px}
.note ul{margin:5px 0 0 16px;padding:0}
.tabs{display:flex;gap:4px;padding:10px 18px 0}
.tab{padding:6px 13px;border:1px solid var(--line);border-bottom:none;
border-radius:7px 7px 0 0;background:var(--panel);color:var(--dim);cursor:pointer}
.tab.on{color:var(--fg);background:var(--bg);border-color:var(--accent)}
.wrap{display:grid;grid-template-columns:minmax(300px,380px) 1fr;gap:14px;
padding:14px 18px 40px;align-items:start}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:9px;
padding:12px;max-height:78vh;overflow:auto}
.view{display:none}.view.on{display:block}
.phase{margin-bottom:12px}
.phase>.ph{font-weight:650;color:var(--accent);margin-bottom:5px}
.inferred{font-size:10px;color:var(--warn);border:1px solid var(--warn);
border-radius:4px;padding:0 4px;margin-left:6px;vertical-align:1px}
.node{padding:5px 8px;border-radius:6px;cursor:pointer;display:flex;
justify-content:space-between;gap:8px;border:1px solid transparent}
.node:hover{background:#1e2330}
.node.sel{background:#1e2942;border-color:var(--accent)}
.node .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.node .mt{color:var(--dim);font-size:11px;white-space:nowrap}
.tl{position:relative;height:22px;background:#12151d;border-radius:4px;margin:3px 0}
.bar{position:absolute;top:3px;height:16px;background:linear-gradient(90deg,#3d6fc4,#6ea8fe);
border-radius:3px;min-width:2px;cursor:pointer}
.bar.sel{outline:2px solid var(--fg)}
.tlrow{display:grid;grid-template-columns:190px 1fr 92px;gap:8px;align-items:center;
padding:1px 0;font-size:12px}
.tlrow .lb{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:pointer}
.tlrow .tm{color:var(--dim);text-align:right;font-variant-numeric:tabular-nums}
h2{font-size:14px;margin:0 0 8px}
h3{font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:var(--dim);
margin:16px 0 6px}
.meta{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.chip{background:var(--chip);border:1px solid var(--line);border-radius:5px;
padding:2px 8px;font-size:11px}
.chip.ok{border-color:var(--ok);color:var(--ok)}
.chip.warnc{border-color:var(--warn);color:var(--warn)}
.chip.errc{border-color:var(--err);color:var(--err)}
pre{background:#0c0e14;border:1px solid var(--line);border-radius:6px;padding:9px;
overflow:auto;max-height:340px;white-space:pre-wrap;word-break:break-word;
font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0}
details{border:1px solid var(--line);border-radius:7px;margin-bottom:7px;background:#141823}
summary{cursor:pointer;padding:7px 10px;display:flex;gap:8px;align-items:center;
flex-wrap:wrap;font-size:12px}
summary::-webkit-details-marker{display:none}
summary:hover{background:#1a1f2c}
.dbody{padding:0 10px 10px}
.sec{margin-top:9px}
.sec>.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.5px;
color:var(--accent);margin-bottom:3px}
.marker{color:var(--dim);font-style:italic;font-size:12px;
border:1px dashed var(--line);border-radius:5px;padding:6px 9px}
.tool{border-left:2px solid var(--accent);padding-left:9px;margin:7px 0}
.tool .tn{font-weight:650;font-size:12px}
.nav{display:flex;gap:8px;margin-top:10px}
.nav button{background:var(--chip);color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px}
.nav button:disabled{opacity:.35;cursor:default}
.nav button:hover:not(:disabled){border-color:var(--accent)}
.gnode{display:flex;align-items:center;gap:8px;padding:4px 0;font-size:12px}
.gline{color:var(--dim);font-family:ui-monospace,monospace;white-space:pre}
.ret{color:var(--ok)}
.empty{color:var(--dim);padding:20px;text-align:center}
.trunc{color:var(--warn);font-size:11px;margin-top:3px}
"""

_JS = r"""
const D = window.__TRACE__;
const byId = {}; D.agents.forEach(a => byId[a.id] = a);
let sel = null, selCall = null;

const esc = s => String(s==null?'':s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const dur = ms => ms==null ? '—' : (ms<60000 ? (ms/1000).toFixed(1)+'s'
  : Math.floor(ms/60000)+'m'+String(Math.round(ms%60000/1000)).padStart(2,'0')+'s');
const usd = v => v==null ? '—' : '$'+Number(v).toFixed(4);
const num = v => v==null ? '—' : Number(v).toLocaleString();

function tab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.v===name));
  document.querySelectorAll('.view').forEach(v=>v.classList.toggle('on',v.dataset.v===name));
}

function select(id, callIdx){
  sel = id; selCall = (callIdx==null ? null : callIdx);
  document.querySelectorAll('.node').forEach(n=>n.classList.toggle('sel',n.dataset.id===id));
  document.querySelectorAll('.bar').forEach(b=>b.classList.toggle('sel',b.dataset.id===id));
  renderDetail();
  if(callIdx!=null){
    const el = document.getElementById('call-'+callIdx);
    if(el){ el.open = true; el.scrollIntoView({block:'nearest'}); }
  }
}

function inputSection(call){
  if(call.input.kind === 'none_recorded')
    return '<div class="marker">No new transcript input recorded before this '
      + 'response (a retry or continuation).</div>';
  return call.input.blocks.map(b=>{
    if(b.kind === 'tool_result'){
      return '<div class="tool"><div class="tn">tool result &larr; '
        + esc(b.tool_use_id||'?') + (b.is_error?' <span class="chip errc">error</span>':'')
        + '</div><pre>' + esc(b.text) + '</pre>'
        + (b.truncated?'<div class="trunc">truncated — original '+num(b.bytes_total)+' bytes</div>':'')
        + '</div>';
    }
    return '<pre>' + esc(b.text) + '</pre>'
      + (b.truncated?'<div class="trunc">truncated — original '+num(b.bytes_total)+' bytes</div>':'');
  }).join('');
}

function reasoningSection(r){
  if(r.state === 'text') return '<pre>'+esc(r.text)+'</pre>';
  if(r.state === 'recorded_unreadable')
    return '<div class="marker">Reasoning block recorded ('+r.blocks
      + '); text unavailable in the transcript.</div>';
  return '<div class="marker">Not captured.</div>';
}

function outputSection(call){
  let h = '';
  if(call.output_text && call.output_text.trim()){
    h += '<pre>'+esc(call.output_text)+'</pre>';
    if(call.output_truncated)
      h += '<div class="trunc">truncated — original '+num(call.output_bytes_total)+' bytes</div>';
  } else if(!call.actions.length){
    h += '<div class="marker">No text output recorded.</div>';
  }
  call.actions.forEach(a=>{
    const st = a.result.status;
    const cls = st==='ok'?'ok':(st==='error'?'errc':'warnc');
    h += '<div class="tool"><div class="tn">action &rarr; '+esc(a.name)
      + ' <span class="chip '+cls+'">'+esc(st)+'</span></div>'
      + '<pre>'+esc(a.args_preview)+'</pre>'
      + (a.args_truncated?'<div class="trunc">args truncated — original '+num(a.args_bytes_total)+' bytes</div>':'');
    if(st==='missing'){
      h += '<div class="marker">'+esc(a.result.note)+'</div>';
    } else {
      h += '<div class="sec"><div class="lbl">result</div><pre>'+esc(a.result.preview)+'</pre>'
        + (a.result.truncated?'<div class="trunc">truncated — original '+num(a.result.bytes_total)+' bytes</div>':'')
        + '</div>';
    }
    h += '</div>';
  });
  return h;
}

function callBlock(a, call, i){
  const kindChip = {actions:'warnc', text:'ok', mixed:'ok', incomplete:'errc'}[call.output_kind]||'';
  return '<details id="call-'+i+'"><summary>'
    + '<b>#'+(i+1)+'</b>'
    + '<span class="chip">'+esc(call.model||'—')+'</span>'
    + '<span class="chip '+kindChip+'">'+esc(call.output_kind)+'</span>'
    + '<span class="chip">'+esc(call.stop_reason||'—')+'</span>'
    + '<span class="chip">'+num(call.usage.output_tokens)+' out-tok</span>'
    + '<span class="chip'+(call.usage_known===false?' warnc':'')+'">'
    + (call.usage_known===false ? 'usage unknown' : usd(call.cost_usd))+'</span>'
    + (call.actions.length?'<span class="chip">'+call.actions.length+' action(s)</span>':'')
    + '</summary><div class="dbody">'
    + '<div class="sec"><div class="lbl">Input prompt — new input since previous response</div>'
    + inputSection(call)
    + '<div class="trunc">Transcript excerpt only. Not the full API request: system '
    + 'prompt, tool definitions and inherited context are not recorded.</div></div>'
    + '<div class="sec"><div class="lbl">Reasoning</div>'+reasoningSection(call.reasoning)+'</div>'
    + '<div class="sec"><div class="lbl">Output — response and actions taken</div>'
    + outputSection(call)+'</div>'
    + '<div class="nav">'
    + '<button '+(i===0?'disabled':'')+' onclick="select(\''+a.id+'\','+(i-1)+')">&larr; previous call</button>'
    + '<button '+(i===a.calls.length-1?'disabled':'')+' onclick="select(\''+a.id+'\','+(i+1)+')">next call &rarr;</button>'
    + '</div></div></details>';
}

function renderDetail(){
  const el = document.getElementById('detail');
  if(!sel || !byId[sel]){
    el.innerHTML = '<div class="empty">Select an agent in the tree, timeline or '
      + 'graph to see its API calls.</div>';
    return;
  }
  const a = byId[sel], t = a.totals||{};
  let h = '<h2>'+esc(a.label)+'</h2><div class="meta">'
    + '<span class="chip">agent '+esc(a.id)+'</span>'
    + '<span class="chip">journal #'+a.ordinal+'</span>'
    + '<span class="chip '+(a.result_status==='returned_to_workflow'?'ok':'warnc')+'">'
    + esc(a.result_status)+'</span>'
    + '<span class="chip">'+(a.calls.length)+' API calls</span>'
    + '<span class="chip">'+usd(t.cost_usd)+'</span>'
    + '<span class="chip">'+dur(a.dur)+' est.</span>'
    + '<span class="chip">depth '+esc(a.spawn_depth)+'</span></div>';
  if(a.description) h += '<div class="sub">'+esc(a.description)+'</div>';

  h += '<h3>Conclusion returned to the workflow</h3>';
  if(a.result_status === 'returned_to_workflow'){
    h += '<pre>'+esc(a.result_preview)+'</pre>'
      + (a.result_truncated?'<div class="trunc">truncated preview</div>':'');
  } else {
    h += '<div class="marker">No result recorded yet — the agent is still '
      + 'running, or the run ended before it returned.</div>';
  }

  h += '<h3>API calls — input, reasoning, output, actions</h3>';
  if(!a.calls.length){
    h += '<div class="marker">No transcript calls recorded ('
      + esc(a.transcript_status) + ').</div>';
  } else {
    a.calls.forEach((c,i)=>{ h += callBlock(a,c,i); });
  }
  el.innerHTML = h;
}

function renderTree(){
  let h = '';
  D.phases.forEach(p=>{
    const badge = p.provenance === 'journal'
      ? '<span class="chip ok" style="margin-left:6px">recorded phase</span>'
      : '<span class="inferred">inferred from labels</span>';
    h += '<div class="phase"><div class="ph">'+esc(p.name)+badge+'</div>';
    p.agents.forEach(id=>{
      const a = byId[id]; if(!a) return;
      h += '<div class="node" data-id="'+esc(id)+'" onclick="select(\''+id+'\')">'
        + '<span class="nm">'+esc(a.label)+'</span>'
        + '<span class="mt">'+a.calls.length+' calls · '+usd((a.totals||{}).cost_usd)+'</span>'
        + '</div>';
    });
    h += '</div>';
  });
  document.getElementById('tree').innerHTML = h || '<div class="empty">No agents recorded.</div>';
}

function renderTimeline(){
  const span = D.agents.reduce((m,a)=>Math.max(m, a.end_off==null?0:a.end_off), 0) || 1;
  let h = '<div class="sub">Offsets from run origin. Bars may overlap — agents run '
    + 'concurrently, so durations must never be summed. Times are estimated from '
    + 'transcript timestamps.</div>';
  D.agents.forEach(a=>{
    const s = a.start_off==null?null:(a.start_off/span*100);
    const w = (a.start_off==null||a.end_off==null)?null:Math.max(0.4,(a.end_off-a.start_off)/span*100);
    h += '<div class="tlrow"><span class="lb" onclick="select(\''+a.id+'\')">'
      + esc(a.label)+'</span><span class="tl">'
      + (s==null?'<span class="marker" style="font-size:11px">no timing recorded</span>'
        : '<span class="bar" data-id="'+esc(a.id)+'" style="left:'+s+'%;width:'+w+'%" '
          + 'title="'+esc(a.label)+'" onclick="select(\''+a.id+'\')"></span>')
      + '</span><span class="tm">'+dur(a.dur)+'</span></div>';
  });
  document.getElementById('timeline').innerHTML = h;
}

function renderGraph(){
  const run = D.run.run_id || 'workflow';
  let h = '<div class="sub">Only edges the journal proves are drawn: the workflow '
    + 'invoking each agent, and each agent returning its result. Role ordering '
    + '(e.g. tech_lead before engineer) is a workflow data dependency and is NOT '
    + 'drawn as an agent-to-agent spawn edge.</div>';
  h += '<div class="gnode" style="margin-top:8px"><b>'+esc(run)+'</b>'
    + '<span class="chip">workflow orchestrator</span></div>';
  const recorded = (D.edges||[]).filter(e =>
    e.type==='result_supplied_to_dispatch' || e.type==='agent_spawn');
  D.agents.forEach((a,i)=>{
    const last = i===D.agents.length-1;
    h += '<div class="gnode"><span class="gline">'+(last?' └─':' ├─')+'▶ </span>'
      + '<span class="lb" style="cursor:pointer" onclick="select(\''+a.id+'\')">'
      + esc(a.label)+'</span>'
      + '<span class="chip">'+a.calls.length+' calls</span>'
      + (a.result_status==='returned_to_workflow'
          ? '<span class="chip ok">◀ returned result</span>'
          : '<span class="chip warnc">no result</span>')
      + '</div>';
  });
  // Recorded linkage: transfer and spawn edges, shown as a SEPARATE relationship
  // from the workflow's orchestration, with their references inspectable.
  const link = D.linkage;
  h += '<h3>Recorded linkage</h3>';
  if(!link || !link.present){
    h += '<div class="marker">'+esc((link&&link.note) ||
      'No recorded linkage events for this run.')+'</div>';
  } else if(!recorded.length){
    h += '<div class="marker">Linkage events were recorded but none resolved to an '
      + 'edge in this trace.</div>';
  } else {
    if(link.complete===false)
      h += '<div class="marker">Linkage coverage is INCOMPLETE — some events were '
        + 'malformed or unresolved; missing edges mean unrecorded, not absent.</div>';
    recorded.forEach(e=>{
      const from=(e.from||'').replace('agent:',''), to=(e.to||'').replace('agent:','');
      const nm=id=>{const a=byId[id]; return a?a.label:id;};
      if(e.type==='result_supplied_to_dispatch'){
        h += '<div class="gnode"><span class="chip ok">result supplied</span>'
          + '<span class="lb" style="cursor:pointer" onclick="select(\''+from+'\')">'
          + esc(nm(from))+'</span><span class="gline"> ──▶ </span>'
          + '<span class="lb" style="cursor:pointer" onclick="select(\''+to+'\')">'
          + esc(nm(to))+'</span></div>'
          + '<div class="tool"><div>from <code>'+esc(e.producer_result_ref)+'</code>'
          + ' into <code>'+esc(e.consumer_input_ref)+'</code></div>'
          + '<div class="trunc">forwarding: '+esc(e.forwarding)
          + (e.forwarding==='transformed' && !e.transformation_known
              ? ' (transformation not described — unknown)' : '')
          + (e.transformation? ' — '+esc(e.transformation):'')+'</div></div>';
      } else {
        const st=e.return_status||'unmatched';
        h += '<div class="gnode"><span class="chip '
          + (st==='returned'?'ok':(st==='unmatched'?'warnc':'errc'))+'">spawn</span>'
          + '<span class="lb" style="cursor:pointer" onclick="select(\''+from+'\')">'
          + esc(nm(from))+'</span><span class="gline"> ──▶ </span>'
          + '<span class="lb" style="cursor:pointer" onclick="select(\''+to+'\')">'
          + esc(nm(to))+'</span>'
          + '<span class="chip">return: '+esc(st)+'</span></div>'
          + '<div class="tool"><div class="trunc">via tool call '
          + esc(e.spawn_tool_call_id||'—')+'; attempts: '
          + ((e.attempts||[]).map(a=>esc(a.attempt_id)+'='+esc(a.status)).join(', ')
             || 'none recorded')+'</div></div>';
      }
    });
  }
  document.getElementById('graph').innerHTML = h;
}

renderTree(); renderTimeline(); renderGraph(); renderDetail();
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>tab(t.dataset.v));
"""


def render_html(view, title="GEAK run — execution trace"):
    run = view["run"]
    t = view["totals"]
    warn_items = "".join("<li>%s</li>" % esc(w) for w in view["warnings"])
    warn_block = ("<div class='note'><b>Scope and provenance</b><ul>%s</ul></div>"
                  % warn_items) if warn_items else ""

    kpis = [
        ("Agents", "%d" % t["agents"], "%d returned a result" % t["returned"]),
        ("API calls", fmt_int(t["calls"]), "%s tool actions" % fmt_int(t["actions"])),
        ("Cost", fmt_usd(t["cost_usd"]), "same basis as the ledger"),
        ("Output tokens", fmt_int(t["output_tokens"]), "%s cache-read in" % fmt_int(t["cache_read"])),
        ("Elapsed", fmt_dur(run.get("elapsed_ms_est")), "estimated, offset from origin"),
        ("Capture", esc(run.get("status") or "—"),
         esc(run.get("status_reason") or "")),
        ("Workflow outcome", esc(run.get("record_status") or "unknown"),
         "from the run record"),
        ("Unknown usage", fmt_int(t["usage_unknown_calls"]),
         "call(s) with no usage recorded"),
    ]
    kpi_html = "".join("<div class='kpi'><b>%s</b><span>%s</span><span>%s</span></div>"
                       % (v, k, extra) for k, v, extra in kpis)

    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%(title)s</title><style>%(css)s</style></head><body>
<header>
  <h1>%(title)s</h1>
  <div class="sub">run <b>%(run_id)s</b> · %(status)s · collected from the workflow
  journal, agent metadata and per-agent transcripts</div>
  <div class="kpis">%(kpis)s</div>
</header>
%(warn)s
<div class="tabs">
  <div class="tab on" data-v="tree">Phase tree</div>
  <div class="tab" data-v="timeline">Timeline</div>
  <div class="tab" data-v="graph">Delegation graph</div>
</div>
<div class="wrap">
  <div class="pane">
    <div class="view on" data-v="tree"><div id="tree"></div></div>
    <div class="view" data-v="timeline"><div id="timeline"></div></div>
    <div class="view" data-v="graph"><div id="graph"></div></div>
  </div>
  <div class="pane"><div id="detail"></div></div>
</div>
<script>window.__TRACE__ = %(data)s;</script>
<script>%(js)s</script>
</body></html>
""" % {
        "title": esc(title), "css": _CSS, "js": _JS,
        "run_id": esc(run.get("run_id") or "—"),
        "status": esc(run.get("status") or "—"),
        "kpis": kpi_html, "warn": warn_block,
        "data": embed_json(view),
    }


def render_markdown(view):
    """Markdown twin. Same counts, ordering and phase semantics as the HTML."""
    run, t = view["run"], view["totals"]
    by_id = {a["id"]: a for a in view["agents"]}
    out = []
    add = out.append

    add("# GEAK run — execution trace\n")
    add("Run `%s` · status **%s**\n" % (run.get("run_id"), run.get("status")))
    add("| Metric | Value |")
    add("| --- | --- |")
    add("| Agents | %d (%d returned a result) |" % (t["agents"], t["returned"]))
    add("| API calls | %s |" % fmt_int(t["calls"]))
    add("| Tool actions | %s |" % fmt_int(t["actions"]))
    add("| Cost | %s |" % fmt_usd(t["cost_usd"]))
    add("| Output tokens | %s |" % fmt_int(t["output_tokens"]))
    add("| Elapsed (estimated) | %s |" % fmt_dur(run.get("elapsed_ms_est")))
    add("| Capture status | %s |" % (run.get("status") or "—"))
    add("| Workflow outcome (run record) | %s |" % (run.get("record_status") or "unknown"))
    add("| Calls with unknown usage | %s |" % fmt_int(t["usage_unknown_calls"]))

    if view["warnings"]:
        add("\n## Scope and provenance\n")
        for w in view["warnings"]:
            add("- %s" % w)

    recorded = any(p.get("provenance") in ("journal", "workflow_timeline")
                   for p in view["phases"])
    add("\n## Phase tree (%s)\n"
        % ("recorded journal phases" if recorded
           else "phases inferred from journal labels"))
    add("```")
    for phase in view["phases"]:
        add("%s" % phase["name"])
        ids = phase["agents"]
        for i, aid in enumerate(ids):
            a = by_id.get(aid)
            if not a:
                continue
            branch = "└──" if i == len(ids) - 1 else "├──"
            add("  %s %-30s %3d calls  %9s  %8s  %s"
                % (branch, a["label"], len(a["calls"]),
                   fmt_usd((a["totals"] or {}).get("cost_usd")),
                   fmt_dur(a["dur"]),
                   "returned" if a["result_status"] == "returned_to_workflow" else "no result"))
    add("```")

    add("\n## Timeline (offsets from run origin; spans may overlap — never sum them)\n")
    add("| Agent | Start +Δ | End +Δ | Duration (est.) |")
    add("| --- | ---: | ---: | ---: |")
    for a in view["agents"]:
        add("| %s | %s | %s | %s |"
            % (a["label"], fmt_dur(a["start_off"]), fmt_dur(a["end_off"]), fmt_dur(a["dur"])))

    add("\n## Delegation graph (proven edges only)\n")
    add("Workflow → agent invocations, and agent → workflow returns. Role ordering is a "
        "workflow data dependency, not an agent-to-agent spawn edge.\n")
    add("| # | Agent | Calls | Returned to workflow |")
    add("| ---: | --- | ---: | --- |")
    for a in view["agents"]:
        add("| %d | %s | %d | %s |"
            % (a["ordinal"], a["label"], len(a["calls"]),
               "yes" if a["result_status"] == "returned_to_workflow" else "no"))

    link = (view.get("run") or {}).get("linkage") or {}
    recorded = [e for e in view["edges"]
                if e.get("type") in ("result_supplied_to_dispatch", "agent_spawn")]
    add("\n## Recorded linkage\n")
    if not link.get("present"):
        add("_%s_\n" % (link.get("note")
                         or "No recorded linkage events for this run."))
    elif not recorded:
        add("_Linkage events were recorded but none resolved to an edge._\n")
    else:
        if link.get("complete") is False:
            add("> Coverage INCOMPLETE — some events were malformed or unresolved; "
                "missing edges mean unrecorded, not absent.\n")
        add("| Kind | From | To | Reference | Detail |")
        add("| --- | --- | --- | --- | --- |")
        names = {a["id"]: a["label"] for a in view["agents"]}
        for e in recorded:
            f = names.get((e.get("from") or "").replace("agent:", ""), e.get("from"))
            t = names.get((e.get("to") or "").replace("agent:", ""), e.get("to"))
            if e["type"] == "result_supplied_to_dispatch":
                add("| result supplied | %s | %s | `%s` -> `%s` | forwarding: %s |"
                    % (f, t, e.get("producer_result_ref"), e.get("consumer_input_ref"),
                       e.get("forwarding")))
            else:
                att = ", ".join("%s=%s" % (a.get("attempt_id"), a.get("status"))
                                for a in (e.get("attempts") or [])) or "none recorded"
                add("| spawn | %s | %s | tool `%s` | return: %s; attempts: %s |"
                    % (f, t, e.get("spawn_tool_call_id"), e.get("return_status"), att))

    add("\n## Per-agent API calls\n")
    add("Full input excerpts, reasoning markers, outputs and tool results are in the "
        "HTML report and in `geak_trace.json`; this table is the index.\n")
    for a in view["agents"]:
        add("\n### %d. %s\n" % (a["ordinal"], a["label"]))
        if not a["calls"]:
            add("_No transcript calls recorded (%s)._\n" % a["transcript_status"])
            continue
        add("| # | Model | Output kind | Stop | Out tok | Cost | Actions |")
        add("| ---: | --- | --- | --- | ---: | ---: | ---: |")
        for i, c in enumerate(a["calls"]):
            add("| %d | %s | %s | %s | %s | %s | %d |"
                % (i + 1, c.get("model") or "—", c.get("output_kind"),
                   c.get("stop_reason") or "—",
                   fmt_int(c["usage"]["output_tokens"]), fmt_usd(c.get("cost_usd")),
                   len(c.get("actions") or [])))
    return "\n".join(out) + "\n"


def write_reports(trace, out_dir, basename="geak_execution_trace"):
    view = build_view(trace)
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, basename + ".html")
    md_path = os.path.join(out_dir, basename + ".md")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render_html(view))
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(view))
    return {"html": html_path, "md": md_path, "totals": view["totals"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True, help="Path to geak_trace.json")
    ap.add_argument("--out-dir", required=True, help="Directory for the reports")
    ap.add_argument("--basename", default="geak_execution_trace")
    args = ap.parse_args(argv)

    with open(args.trace, "r", encoding="utf-8") as fh:
        trace = json.load(fh)
    if trace.get("schema") != SCHEMA_IN:
        sys.stderr.write("geak_trace_report: warning: unexpected schema %r\n"
                         % trace.get("schema"))
    res = write_reports(trace, args.out_dir, args.basename)
    sys.stderr.write("geak_trace_report: %s agents, %s calls -> %s\n"
                     % (res["totals"]["agents"], res["totals"]["calls"], res["html"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
