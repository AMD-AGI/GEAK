#!/usr/bin/env python3
"""Render a GEAK run as a clickable role-execution tree (HTML + Markdown).

The per-call ledger (``reports/trace/llm_calls.jsonl``, written by
``e2e_workflow/scripts/llm_ledger.py``) answers *what each API call cost*. This
module turns that flat list into the shape a human reads a run in: a tree that
starts at the Director, nests the TechLead under it, the engineers and verifiers
under the TechLead, and the helper scopes beside them — one node per agent call.
Clicking a node reveals that agent's cost (split into cache-write / cache-read /
uncached-context / router / output), its wall time, its tokens, and the prompt
and output (thinking + response) it exchanged.

Why a heuristic tree. The transcripts carry no literal "agent X spawned agent Y"
edge — ``parentUuid`` only links messages WITHIN one agent's own chain. So the
hierarchy is reconstructed from two things the logs DO carry: each agent's role
(``director`` outranks ``tech_lead`` outranks ``engineer`` …) and its execution
order (first-call timestamp). An agent nests under the most recent
higher-ranked agent still open — the same way the run actually delegated. Unknown
roles fall back to leaves in execution order, so a workflow this file has never
seen still renders, just flatter.

Both run styles share this: kernel-lane and E2E differ only in their role names,
and the rank table below covers both. The renderer reads only ``llm_calls.jsonl``,
so it has no dependency on any external report tool.

Usage:
  python3 geak_call_tree_html.py --calls <path/to/llm_calls.jsonl> \
      --out-dir <dir> [--model <name>]
Writes ``<dir>/geak_run_report_<model>.html`` and ``.md``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys


# Lower rank = closer to the root of the delegation tree. Both workflows' roles
# live here; anything absent is a leaf (rank LEAF) placed in execution order.
ROLE_RANK = {
    "director": 0, "e2e_director": 0,
    "tech_lead": 1, "techlead": 1, "system_architect": 1,
    "senior_engineer": 2, "profiler": 2, "profile_engineer": 2,
    "benchmark_engineer": 2, "config_tuner": 2, "kernel_extractor": 2,
    "tuning_specialist": 2, "integrator": 2, "e2e_integrator": 2,
    "engineer": 3, "verify": 3, "verifier": 3, "file_writer": 3,
}
LEAF = 4

# Buckets, in the order a report reads them, with display labels.
COST_BUCKETS = [
    ("cache_write", "cache-write"),
    ("cache_read", "cache-read"),
    ("uncached_input", "uncached-context"),
    ("router", "router"),
    ("output", "output"),
]


# --------------------------------------------------------------------------- #
# Loading + aggregation
# --------------------------------------------------------------------------- #
def read_calls(path):
    """Load ``llm_calls.jsonl`` into a list of call rows (skipping bad lines)."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _agent_key(row):
    """One conversation = one agent node. The timeline label is the stable id
    when present; otherwise role + sub_phase keeps distinct scopes apart."""
    label = row.get("agent_label") or ""
    return (label, row.get("role") or "", row.get("sub_phase") or "")


def agentize(rows):
    """Fold call rows into agent nodes, preserving first-seen (execution) order.

    Each node aggregates its calls' tokens and cost buckets, keeps the first
    prompt and the concatenated output/thinking, and records its first timestamp
    so the tree can be ordered and nested by execution order.
    """
    order, nodes = [], {}
    for r in rows:
        key = _agent_key(r)
        node = nodes.get(key)
        if node is None:
            node = {
                "key": key,
                "role": r.get("role") or "(driver)",
                "sub_phase": r.get("sub_phase") or "",
                "label": r.get("agent_label") or "",
                "phase": r.get("phase") or "",
                "attribution": r.get("attribution") or "",
                "models": set(),
                "calls": 0,
                "ts_ms": r.get("ts_ms"),
                "first_ts": r.get("ts"),
                "llm_ms": 0.0,
                "tokens": {"input": 0, "cache_read": 0, "cache_write_5m": 0,
                           "cache_write_1h": 0, "output": 0},
                "cost": {k: 0.0 for k, _ in COST_BUCKETS},
                "cost_usd": 0.0,
                "prompt": r.get("prompt") or "",
                "outputs": [],
                "thinkings": [],
            }
            nodes[key] = node
            order.append(key)
        node["calls"] += 1
        if r.get("model"):
            node["models"].add(r["model"])
        node["llm_ms"] += _num(r.get("duration_ms"))
        node["tokens"]["input"] += int(_num(r.get("input_tokens")))
        node["tokens"]["cache_read"] += int(_num(r.get("cache_read_input_tokens")))
        node["tokens"]["cache_write_5m"] += int(_num(r.get("cache_write_5m_tokens")))
        node["tokens"]["cache_write_1h"] += int(_num(r.get("cache_write_1h_tokens")))
        node["tokens"]["output"] += int(_num(r.get("output_tokens")))
        node["cost_usd"] += _num(r.get("cost_usd"))
        bd = r.get("cost_breakdown") or {}
        for k, _ in COST_BUCKETS:
            node["cost"][k] += _num(bd.get(k))
        ts = r.get("ts_ms")
        if ts is not None and (node["ts_ms"] is None or ts < node["ts_ms"]):
            node["ts_ms"], node["first_ts"] = ts, r.get("ts")
        if r.get("output"):
            node["outputs"].append(r["output"])
        if r.get("thinking"):
            node["thinkings"].append(r["thinking"])
    return [nodes[k] for k in order]


def build_tree(nodes):
    """Nest agent nodes into a delegation tree by role rank + execution order.

    A stack holds the currently-open ancestors. Each node (in timestamp order)
    pops every open node of equal-or-lower precedence, then attaches to whatever
    higher-ranked node remains — the agent that was still running when it
    started, i.e. the one that delegated to it.
    """
    ordered = sorted(enumerate(nodes),
                     key=lambda it: (it[1]["ts_ms"] is None, it[1]["ts_ms"] or 0, it[0]))
    root = {"role": "Run", "children": [], "_root": True}
    stack = [(root, -1)]
    for _, node in ordered:
        node["children"] = []
        rank = ROLE_RANK.get(node["role"], LEAF)
        while len(stack) > 1 and stack[-1][1] >= rank:
            stack.pop()
        stack[-1][0]["children"].append(node)
        stack.append((node, rank))
    return root


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _n(x):
    return "{:,}".format(int(x))


def _hms(ms):
    if not ms:
        return "0s"
    s = ms / 1000.0
    if s < 60:
        return "%.1fs" % s
    m, s = divmod(int(s), 60)
    if m < 60:
        return "%dm%02ds" % (m, s)
    h, m = divmod(m, 60)
    return "%dh%02dm%02ds" % (h, m, s)


def _usd(x):
    return "$%.4f" % x if x < 1 else "$%.2f" % x


def node_title(node):
    role = node["role"]
    sub = node["sub_phase"]
    return "%s%s" % (role, (":" + sub) if sub else "")


def node_detail(node):
    """A flat dict of everything the detail panel shows for one node."""
    tk = node["tokens"]
    cw = tk["cache_write_5m"] + tk["cache_write_1h"]
    total_in = tk["input"] + tk["cache_read"] + cw
    return {
        "title": node_title(node),
        "role": node["role"], "sub_phase": node["sub_phase"],
        "phase": node["phase"], "label": node["label"],
        "attribution": node["attribution"],
        "models": sorted(node["models"]),
        "calls": node["calls"],
        "llm_ms": node["llm_ms"],
        "tokens": {"uncached_input": tk["input"], "cache_read": tk["cache_read"],
                   "cache_write": cw, "output": tk["output"], "total_input": total_in},
        "cost": {k: node["cost"][k] for k, _ in COST_BUCKETS},
        "cost_usd": node["cost_usd"],
        "prompt": node["prompt"],
        "output": "\n\n".join(node["outputs"]),
        "thinking": "\n\n".join(node["thinkings"]),
    }


def run_totals(nodes):
    """Whole-run aggregates plus a per-model split, for the report header."""
    total = {"calls": 0, "llm_ms": 0.0, "cost_usd": 0.0,
             "cost": {k: 0.0 for k, _ in COST_BUCKETS},
             "tokens": {"uncached_input": 0, "cache_read": 0, "cache_write": 0, "output": 0}}
    per_model = {}
    for n in nodes:
        d = node_detail(n)
        total["calls"] += d["calls"]
        total["llm_ms"] += d["llm_ms"]
        total["cost_usd"] += d["cost_usd"]
        for k, _ in COST_BUCKETS:
            total["cost"][k] += d["cost"][k]
        for k in total["tokens"]:
            total["tokens"][k] += d["tokens"][k]
        for m in (d["models"] or ["(unknown)"]):
            pm = per_model.setdefault(m, {"calls": 0, "cost_usd": 0.0})
            pm["calls"] += d["calls"]
            pm["cost_usd"] += d["cost_usd"]
    return total, per_model


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def render_markdown(nodes, root, model):
    total, per_model = run_totals(nodes)
    out = ["# GEAK run report — %s" % model, ""]
    out += ["## Run totals", ""]
    out += ["- **API calls**: %s" % _n(total["calls"])]
    out += ["- **LLM wall time**: %s" % _hms(total["llm_ms"])]
    out += ["- **Cost**: %s" % _usd(total["cost_usd"])]
    out += ["  - " + ", ".join("%s %s" % (lbl, _usd(total["cost"][k]))
                               for k, lbl in COST_BUCKETS)]
    out += ["- **Tokens**: uncached-input %s · cache-read %s · cache-write %s · output %s"
            % (_n(total["tokens"]["uncached_input"]), _n(total["tokens"]["cache_read"]),
               _n(total["tokens"]["cache_write"]), _n(total["tokens"]["output"]))]
    out += ["", "### By model", "",
            "| model | calls | cost |", "|---|---|---|"]
    for m, pm in sorted(per_model.items(), key=lambda kv: -kv[1]["cost_usd"]):
        out.append("| %s | %s | %s |" % (m, _n(pm["calls"]), _usd(pm["cost_usd"])))

    out += ["", "## Execution tree", ""]

    def walk(node, depth):
        if not node.get("_root"):
            d = node_detail(node)
            out.append("%s- **%s** — %s calls · %s · %s"
                       % ("  " * depth, d["title"], _n(d["calls"]),
                          _hms(d["llm_ms"]), _usd(d["cost_usd"])))
        for c in node.get("children", []):
            walk(c, depth + (0 if node.get("_root") else 1))

    walk(root, 0)

    out += ["", "## Agent details", ""]
    for n in sorted(nodes, key=lambda x: (x["ts_ms"] is None, x["ts_ms"] or 0)):
        d = node_detail(n)
        out += ["### %s" % d["title"]]
        if d["label"]:
            out.append("- label: `%s`" % d["label"])
        out += ["- models: %s" % (", ".join(d["models"]) or "—"),
                "- phase: %s" % (d["phase"] or "—"),
                "- calls: %s · time: %s" % (_n(d["calls"]), _hms(d["llm_ms"])),
                "- cost: %s (%s)" % (_usd(d["cost_usd"]),
                                     ", ".join("%s %s" % (lbl, _usd(d["cost"][k]))
                                               for k, lbl in COST_BUCKETS)),
                "- tokens: uncached-input %s · cache-read %s · cache-write %s · output %s"
                % (_n(d["tokens"]["uncached_input"]), _n(d["tokens"]["cache_read"]),
                   _n(d["tokens"]["cache_write"]), _n(d["tokens"]["output"])), ""]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# HTML (self-contained: embedded JSON + vanilla JS, no external deps)
# --------------------------------------------------------------------------- #
def _tree_json(node):
    """A compact nested structure the page's JS walks to draw the tree."""
    out = {"root": bool(node.get("_root"))}
    if not node.get("_root"):
        d = node_detail(node)
        out.update({
            "title": d["title"], "calls": d["calls"], "llm_ms": d["llm_ms"],
            "cost_usd": d["cost_usd"], "detail": d,
        })
    out["children"] = [_tree_json(c) for c in node.get("children", [])]
    return out


def render_html(nodes, root, model):
    total, per_model = run_totals(nodes)
    payload = {
        "model": model,
        "total": total,
        "per_model": [{"model": m, **pm} for m, pm in
                      sorted(per_model.items(), key=lambda kv: -kv[1]["cost_usd"])],
        "tree": _tree_json(root),
        "buckets": [{"key": k, "label": lbl} for k, lbl in COST_BUCKETS],
    }
    data = json.dumps(payload).replace("</", "<\\/")
    title = html.escape("GEAK run report — %s" % model)
    return _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; --bg:#fbfbfa; --fg:#1a1a1a; --muted:#666;
    --line:#e2e2df; --card:#fff; --accent:#7a3ffb; --sel:#efe9ff; }
  @media (prefers-color-scheme: dark){ :root:not([data-theme=light]){
    --bg:#1a1a1c; --fg:#e8e8e6; --muted:#9a9a97; --line:#33333a; --card:#232327;
    --accent:#a888ff; --sel:#2c2540; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:20px 24px; border-bottom:1px solid var(--line); }
  h1 { font-size:18px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; }
  .totals { display:flex; flex-wrap:wrap; gap:16px; padding:14px 24px; border-bottom:1px solid var(--line); }
  .chip { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 12px; }
  .chip b { display:block; font-size:16px; }
  .chip span { color:var(--muted); font-size:12px; }
  .wrap { display:flex; gap:0; align-items:stretch; min-height:60vh; }
  .tree { flex:1 1 55%; padding:12px 8px 40px 16px; overflow:auto; border-right:1px solid var(--line); }
  .panel { flex:1 1 45%; padding:16px 20px 40px; overflow:auto; position:sticky; top:0; max-height:100vh; }
  @media (max-width:760px){ .wrap{flex-direction:column;} .tree{border-right:none;border-bottom:1px solid var(--line);} .panel{position:static;max-height:none;} }
  ul.t { list-style:none; margin:0; padding-left:16px; }
  ul.t.root { padding-left:0; }
  li.node > .row { display:flex; align-items:center; gap:6px; padding:3px 6px; border-radius:6px; cursor:pointer; }
  li.node > .row:hover { background:var(--sel); }
  li.node > .row.sel { background:var(--sel); outline:1px solid var(--accent); }
  .tog { width:14px; text-align:center; color:var(--muted); cursor:pointer; user-select:none; flex:0 0 14px; }
  .tt { font-weight:600; }
  .meta { color:var(--muted); font-size:12px; margin-left:auto; white-space:nowrap; }
  .cost { color:var(--accent); font-variant-numeric:tabular-nums; }
  .hidden { display:none; }
  .panel h2 { font-size:15px; margin:0 0 10px; }
  .kv { display:grid; grid-template-columns:auto 1fr; gap:2px 14px; margin:0 0 14px; }
  .kv dt { color:var(--muted); }
  .kv dd { margin:0; font-variant-numeric:tabular-nums; }
  .bars { margin:6px 0 14px; }
  .bar { display:flex; align-items:center; gap:8px; margin:2px 0; }
  .bar .lab { width:130px; color:var(--muted); font-size:12px; }
  .bar .track { flex:1; background:var(--line); border-radius:4px; height:12px; overflow:hidden; }
  .bar .fill { height:100%; background:var(--accent); }
  .bar .val { width:88px; text-align:right; font-variant-numeric:tabular-nums; font-size:12px; }
  details { margin:8px 0; border:1px solid var(--line); border-radius:8px; padding:8px 10px; background:var(--card); }
  details summary { cursor:pointer; color:var(--muted); }
  pre { white-space:pre-wrap; word-break:break-word; margin:8px 0 0; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; max-height:340px; overflow:auto; }
  .empty { color:var(--muted); padding:24px; }
</style></head>
<body>
<header><h1>__TITLE__</h1>
<div class="sub">Role-execution tree — click any agent to see its cost, time, tokens, prompt and output. Cost is derived from transcript token buckets, not an SDK total.</div></header>
<div class="totals" id="totals"></div>
<div class="wrap">
  <div class="tree"><ul class="t root" id="tree"></ul></div>
  <div class="panel" id="panel"><div class="empty">Select an agent on the left.</div></div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
  var D = JSON.parse(document.getElementById('data').textContent);
  var BUCKETS = D.buckets;
  function n(x){ return (x||0).toLocaleString(); }
  function usd(x){ x=x||0; return x<1 ? '$'+x.toFixed(4) : '$'+x.toFixed(2); }
  function hms(ms){ ms=ms||0; var s=ms/1000; if(s<60) return s.toFixed(1)+'s';
    var m=Math.floor(s/60); s=Math.floor(s%60); if(m<60) return m+'m'+String(s).padStart(2,'0')+'s';
    var h=Math.floor(m/60); m=m%60; return h+'h'+String(m).padStart(2,'0')+'m'; }
  function esc(t){ var d=document.createElement('div'); d.textContent=(t==null?'':String(t)); return d.innerHTML; }

  // Header totals
  var tt=D.total, tot=document.getElementById('totals');
  function chip(v,l){ var d=document.createElement('div'); d.className='chip'; d.innerHTML='<b>'+v+'</b><span>'+l+'</span>'; return d; }
  tot.appendChild(chip(n(tt.calls),'API calls'));
  tot.appendChild(chip(hms(tt.llm_ms),'LLM wall time'));
  tot.appendChild(chip(usd(tt.cost_usd),'total cost'));
  tot.appendChild(chip(n(tt.tokens.cache_read),'cache-read tokens'));
  tot.appendChild(chip(n(tt.tokens.output),'output tokens'));
  D.per_model.forEach(function(pm){ tot.appendChild(chip(usd(pm.cost_usd), pm.model+' ('+n(pm.calls)+')')); });

  // Tree
  var sel=null;
  function detailPanel(d){
    var maxc=0; BUCKETS.forEach(function(b){ maxc=Math.max(maxc, d.cost[b.key]||0); });
    var bars=BUCKETS.map(function(b){
      var v=d.cost[b.key]||0, w=maxc>0?(100*v/maxc):0;
      return '<div class="bar"><div class="lab">'+esc(b.label)+'</div><div class="track"><div class="fill" style="width:'+w.toFixed(1)+'%"></div></div><div class="val">'+usd(v)+'</div></div>';
    }).join('');
    var tk=d.tokens;
    var h=''
      +'<h2>'+esc(d.title)+'</h2>'
      +'<dl class="kv">'
      +'<dt>role</dt><dd>'+esc(d.role)+(d.sub_phase?(' · '+esc(d.sub_phase)):'')+'</dd>'
      +'<dt>phase</dt><dd>'+esc(d.phase||'—')+'</dd>'
      +'<dt>models</dt><dd>'+esc((d.models||[]).join(', ')||'—')+'</dd>'
      +'<dt>calls</dt><dd>'+n(d.calls)+'</dd>'
      +'<dt>LLM time</dt><dd>'+hms(d.llm_ms)+'</dd>'
      +'<dt>total cost</dt><dd>'+usd(d.cost_usd)+'</dd>'
      +'</dl>'
      +'<div class="bars"><div class="sub" style="color:var(--muted);font-size:12px;margin-bottom:4px">Cost by bucket</div>'+bars+'</div>'
      +'<dl class="kv">'
      +'<dt>uncached-input tok</dt><dd>'+n(tk.uncached_input)+'</dd>'
      +'<dt>cache-read tok</dt><dd>'+n(tk.cache_read)+'</dd>'
      +'<dt>cache-write tok</dt><dd>'+n(tk.cache_write)+'</dd>'
      +'<dt>output tok</dt><dd>'+n(tk.output)+'</dd>'
      +'<dt>total input tok</dt><dd>'+n(tk.total_input)+'</dd>'
      +'</dl>';
    if(d.attribution) h+='<div class="sub" style="color:var(--muted);font-size:12px">attribution: '+esc(d.attribution)+'</div>';
    h+='<details><summary>Input prompt</summary><pre>'+esc(d.prompt||'(none captured)')+'</pre></details>';
    h+='<details><summary>Thinking</summary><pre>'+esc(d.thinking||'(none captured)')+'</pre></details>';
    h+='<details><summary>Output (response)</summary><pre>'+esc(d.output||'(none captured)')+'</pre></details>';
    return h;
  }
  function show(d, rowEl){
    document.getElementById('panel').innerHTML = detailPanel(d);
    if(sel) sel.classList.remove('sel');
    sel=rowEl; if(sel) sel.classList.add('sel');
  }
  function drawNode(node, ul){
    var li=document.createElement('li'); li.className='node';
    var row=document.createElement('div'); row.className='row';
    var kids=node.children||[];
    var tog=document.createElement('span'); tog.className='tog'; tog.textContent=kids.length?'▾':'·';
    row.appendChild(tog);
    var tt=document.createElement('span'); tt.className='tt'; tt.textContent=node.title; row.appendChild(tt);
    var meta=document.createElement('span'); meta.className='meta';
    meta.innerHTML=n(node.calls)+' calls · '+hms(node.llm_ms)+' · <span class="cost">'+usd(node.cost_usd)+'</span>';
    row.appendChild(meta);
    row.addEventListener('click', function(e){ if(e.target===tog) return; show(node.detail, row); });
    li.appendChild(row);
    if(kids.length){
      var sub=document.createElement('ul'); sub.className='t';
      kids.forEach(function(c){ drawNode(c, sub); });
      li.appendChild(sub);
      tog.addEventListener('click', function(){ var h=sub.classList.toggle('hidden'); tog.textContent=h?'▸':'▾'; });
    }
    ul.appendChild(li);
  }
  var treeEl=document.getElementById('tree');
  var top=D.tree.children||[];
  if(!top.length){ document.querySelector('.tree').innerHTML='<div class="empty">No agent calls found in this ledger.</div>'; }
  top.forEach(function(c){ drawNode(c, treeEl); });
})();
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def render(rows, model="run"):
    """Return (html_str, md_str) for a list of call rows."""
    nodes = agentize(rows)
    root = build_tree(nodes)
    return render_html(nodes, root, model), render_markdown(nodes, root, model)


def write(calls_path, out_dir, model="run"):
    """Read ``llm_calls.jsonl`` and write the HTML + MD report. Returns paths."""
    rows = read_calls(calls_path)
    html_str, md_str = render(rows, model)
    os.makedirs(out_dir, exist_ok=True)
    base = "geak_run_report_%s" % model
    html_path = os.path.join(out_dir, base + ".html")
    md_path = os.path.join(out_dir, base + ".md")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html_str)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_str)
    return html_path, md_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render a GEAK run as a role-execution tree (HTML + MD).")
    ap.add_argument("--calls", required=True, help="path to reports/trace/llm_calls.jsonl")
    ap.add_argument("--out-dir", required=True, help="directory to write the report into")
    ap.add_argument("--model", default="run", help="model/run name used in the filename + header")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.calls):
        print("geak_call_tree_html: no such file: %s" % args.calls, file=sys.stderr)
        return 2
    html_path, md_path = write(args.calls, args.out_dir, args.model)
    print("geak_call_tree_html: wrote %s and %s" % (html_path, md_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
