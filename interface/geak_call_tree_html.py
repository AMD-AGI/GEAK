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
from datetime import datetime, timezone


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

# Rows whose transcript is not a workflow sub-agent's (a launching session, or an
# explicit glob over loose files) have no workflow run to group under.
OUTSIDE_WORKFLOW = "(outside a workflow)"

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


def _call_window(row):
    """(inferred start, last observed) in epoch ms, or (None, None).

    Neither end is a measured request time — the transcript records none. The END is the last
    timestamp at which this response was seen in the transcript (a streamed response is flushed
    more than once; the earlier flushes are prefixes of the same reply, so the last one is when
    it finished). The START is inferred by stepping back ``duration_ms``, the gap since the
    previous record — which brackets the call but also contains whatever ran between the two.

    So a span built from these is an OBSERVED span with an inferred left edge: an upper bound on
    the model time and a lower bound on nothing. Everything derived from it is labelled
    ``inferred`` for that reason, and it is still the right quantity for "how long did this take"
    — it includes the compiling and benchmarking between calls, which is real elapsed time."""
    ts = row.get("ts_ms")
    if ts is None:
        return None, None
    end = _num(row.get("last_seen_ms") if row.get("last_seen_ms") is not None else ts)
    start = _num(ts) - (_num(row.get("duration_ms")) if row.get("duration_ms") is not None else 0.0)
    return start, max(start, end)


def _span(rows):
    """(first request, last response) over *rows*, or (None, None)."""
    starts, ends = [], []
    for r in rows:
        s, e = _call_window(r)
        if s is not None:
            starts.append(s)
            ends.append(e)
    return (min(starts), max(ends)) if starts else (None, None)


def _iso(ms):
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def breakdown_rows(rows, field, missing):
    """Calls, agents, cost, tokens, billed span and ELAPSED time per value of *field*.

    Elapsed is first request to last response — the clock a reader means by "how
    long did this phase take", including the compiling and benchmarking between
    calls. Billed span is the sum of per-call durations, time spent waiting on the
    model. Groups overlap in time whenever work ran in parallel, so elapsed figures
    are never summed. Rows come back in the order their group started."""
    groups = {}
    for r in rows:
        groups.setdefault(r.get(field) or missing, []).append(r)
    out = []
    for name, grp in groups.items():
        start, end = _span(grp)
        out.append({
            "name": name,
            "calls": len(grp),
            "agents": len({_agent_key(r) for r in grp}),
            "cost_usd": sum(_num(r.get("cost_usd")) for r in grp),
            "billed_ms": sum(_num(r.get("duration_ms")) for r in grp),
            "elapsed_ms": (end - start) if start is not None else None,
            "started": _iso(start), "ended": _iso(end), "_start": start,
            "input_tokens": sum(int(_num(r.get("input_tokens"))) + int(_num(r.get("cache_read_input_tokens")))
                                + int(_num(r.get("cache_write_5m_tokens"))) + int(_num(r.get("cache_write_1h_tokens")))
                                for r in grp),
            "output_tokens": sum(int(_num(r.get("output_tokens"))) for r in grp),
        })
    out.sort(key=lambda d: (d["_start"] is None, d["_start"] or 0))
    for d in out:
        d.pop("_start")
    return out


def _agent_key(row):
    """One conversation = one agent node.

    ``group_id`` (written by the ledger) is the authoritative identity: it ties a
    row to the exact agent attempt that produced it, so retries and the same role
    in different transcripts stay SEPARATE nodes instead of collapsing into one.
    When it is absent (older ledgers, hand-built rows) fall back to the transcript
    plus the role/label, which still keeps agents in different transcripts apart.
    """
    gid = row.get("group_id")
    if gid:
        return ("gid", gid)
    return (row.get("transcript") or row.get("source") or "",
            row.get("agent_label") or "",
            row.get("role") or "", row.get("sub_phase") or "")


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
                "api_calls": [],   # the individual provider responses under this agent
                "workflow_run": r.get("workflow_run") or "",
                "t_start": None,   # first request sent (response time minus its duration)
                "t_end": None,     # last response received
            }
            nodes[key] = node
            order.append(key)
        start, end = _call_window(r)
        if start is not None:
            node["t_start"] = start if node["t_start"] is None else min(node["t_start"], start)
            node["t_end"] = end if node["t_end"] is None else max(node["t_end"], end)
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
        # Keep every API response distinct — an agent attempt is not one call.
        cw = int(_num(r.get("cache_write_5m_tokens"))) + int(_num(r.get("cache_write_1h_tokens")))
        # duration is kept as None when the transcript never gave us one, so the
        # report can say "unknown" instead of coercing a missing span to 0s.
        dur = r.get("duration_ms")
        node["api_calls"].append({
            "ts": r.get("ts"),
            "model": r.get("model") or "",
            "duration_ms": None if dur is None else _num(dur),
            "stop_reason": r.get("stop_reason") or "",
            "message_id": r.get("message_id") or "",
            "tokens": {"uncached_input": int(_num(r.get("input_tokens"))),
                       "cache_read": int(_num(r.get("cache_read_input_tokens"))),
                       "cache_write": cw,
                       "output": int(_num(r.get("output_tokens")))},
            "cost_usd": _num(r.get("cost_usd")),
            "cost": {k: _num((r.get("cost_breakdown") or {}).get(k)) for k, _ in COST_BUCKETS},
            "output": r.get("output") or "",
            "thinking": r.get("thinking") or "",
        })
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


def node_label_note(node):
    """The workflow's own label for the agent, when it says more than the title.

    Several agents share a title (every ``engineer:memory``, every
    ``kernel_extractor:extract_op``); the label the script gave each one —
    ``eng r2_d0:memory``, ``extract_op fused_moe_a16w4_decode`` — is what tells
    them apart. Empty when it would only repeat the title."""
    label = (node.get("label") or "").strip()
    title = node_title(node)
    if not label or label in (title, title + ":", "(driver)"):
        return ""
    return label


def node_detail(node):
    """A flat dict of everything the detail panel shows for one node."""
    tk = node["tokens"]
    cw = tk["cache_write_5m"] + tk["cache_write_1h"]
    total_in = tk["input"] + tk["cache_read"] + cw
    t0, t1 = node.get("t_start"), node.get("t_end")
    return {
        "title": node_title(node),
        "label_note": node_label_note(node),
        "role": node["role"], "sub_phase": node["sub_phase"],
        "phase": node["phase"], "label": node["label"],
        "attribution": node["attribution"],
        "workflow_run": node.get("workflow_run") or "",
        "models": sorted(node["models"]),
        "calls": node["calls"],
        "llm_ms": node["llm_ms"],
        "elapsed_ms": (t1 - t0) if t0 is not None else None,
        "started": _iso(t0), "ended": _iso(t1),
        "tokens": {"uncached_input": tk["input"], "cache_read": tk["cache_read"],
                   "cache_write": cw, "output": tk["output"], "total_input": total_in},
        "cost": {k: node["cost"][k] for k, _ in COST_BUCKETS},
        # input-token cost is the SUM of the three input leaves, shown as a
        # subtotal — not a separate additive charge (would double-count).
        "input_cost_subtotal": (node["cost"]["cache_write"] + node["cost"]["cache_read"]
                                + node["cost"]["uncached_input"]),
        "cost_usd": node["cost_usd"],
        "api_calls": node["api_calls"],
        "prompt": node["prompt"],
        "output": "\n\n".join(node["outputs"]),
        "thinking": "\n\n".join(node["thinkings"]),
    }


TERMINAL_STOPS = {"end_turn", "tool_use", "stop_sequence", "max_tokens"}


def completeness(rows, meta=None):
    """Three DISTINCT signals, not one 'complete' flag (per review):

    * workflow_completion — did the run's own telemetry come through whole?
      Taken from the ledger meta (``complete`` / ``warnings``); unknown if no
      meta was supplied.
    * capture_completeness — how many API responses have a terminal stop record.
      A response with no terminal ``stop_reason`` was captured mid-flight; its
      usage/output may be partial.
    * cost_coverage — a fixed caveat: dollars are child-scope, estimated from a
      rate card, and exclude parent driver/resume/monitor scope. Never an invoice.
    """
    total = len(rows)
    incomplete = [r for r in rows if (r.get("stop_reason") or "") not in TERMINAL_STOPS]
    meta = meta or {}
    warnings = list(meta.get("warnings") or [])
    return {
        "workflow_completion": {
            "complete": meta.get("complete") if meta else None,
            "attribution_mode": meta.get("attribution_mode"),
            # How transcripts were selected: 'explicit' / 'run-scoped' /
            # 'run-scoped-inferred' / 'partial' / 'substring-fallback' (see
            # geak_report.run). A substring-fallback, partial, or inferred scope
            # means the billed numbers may be over- or under-attributed; surface
            # it, never hide it. ``transcript_scope_anchor`` (e.g.
            # 'exp_root-ancestor') records HOW an inferred whole-run identity was
            # established, so a containment-only match reads as unproven.
            "transcript_scope": meta.get("transcript_scope"),
            "transcript_scope_anchor": meta.get("transcript_scope_anchor"),
            "warnings": warnings,
        },
        "capture_completeness": {
            "api_calls": total,
            "incomplete_output": len(incomplete),
            "complete": len(incomplete) == 0,
        },
        "cost_coverage": "child-scope, estimated from a fixed rate card "
                         "(not an SDK total, not an invoice); excludes parent "
                         "driver/resume/monitor scope.",
    }


def run_totals(nodes):
    """Whole-run aggregates plus a per-model split, for the report header."""
    total = {"calls": 0, "llm_ms": 0.0, "cost_usd": 0.0,
             "cost": {k: 0.0 for k, _ in COST_BUCKETS},
             "tokens": {"uncached_input": 0, "cache_read": 0, "cache_write": 0, "output": 0}}
    starts = [n["t_start"] for n in nodes if n.get("t_start") is not None]
    ends = [n["t_end"] for n in nodes if n.get("t_end") is not None]
    t0, t1 = (min(starts), max(ends)) if starts else (None, None)
    # Wall clock of the whole run as the ledger saw it: first request to last
    # response. Idle stretches with no LLM call at either end are outside it.
    total["elapsed_ms"] = (t1 - t0) if t0 is not None else None
    total["started"], total["ended"] = _iso(t0), _iso(t1)
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
        # Per-model totals come from the INDIVIDUAL provider responses, so a node
        # that switched models (routing/fallback) splits its cost across the
        # models that actually served it. Summing whole-node cost into every model
        # present would double-count; this keeps Σ(per-model) == run total.
        for c in n["api_calls"]:
            m = c.get("model") or "(unknown)"
            pm = per_model.setdefault(m, {"calls": 0, "cost_usd": 0.0})
            pm["calls"] += 1
            pm["cost_usd"] += _num(c.get("cost_usd"))
    return total, per_model


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _md_breakdown(title, first_col, groups, note):
    out = ["", "## %s" % title, "",
           "| %s | started | elapsed (inferred) | billed span | calls | agents | input tok | output tok | cost |" % first_col,
           "|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            g["name"], g["started"] or "—",
            _hms(g["elapsed_ms"]) if g["elapsed_ms"] is not None else "—",
            _hms(g["billed_ms"]), _n(g["calls"]), _n(g["agents"]),
            _n(g["input_tokens"]), _n(g["output_tokens"]), _usd(g["cost_usd"])))
    return out + ["", note]


def render_markdown(nodes, root, model, comp=None, phases=None, invocations=None):
    total, per_model = run_totals(nodes)
    out = ["# GEAK run report — %s" % model, ""]
    if comp:
        cap = comp["capture_completeness"]
        wf = comp["workflow_completion"]
        _scope = wf.get("transcript_scope")
        _anchor = wf.get("transcript_scope_anchor")
        _scope_note = {
            "explicit": "caller-named transcripts",
            "run-scoped": "scoped to this run's own transcripts",
            "run-scoped-inferred": "INFERRED — enclosing run matched by exp_root "
                                   "containment only; ownership not proven",
            "partial": "PARTIAL — some lanes could not be established",
            "substring-fallback": "FALLBACK — path-substring discovery; may include concurrent sessions",
        }.get(_scope, _scope)
        if _scope == "run-scoped-inferred" and _anchor in ("live-journal", "record+journal"):
            _scope_note = ("INFERRED — an invocation with no workflow record was tied to this "
                           "eval dir by its journal; ownership not proven")
        out += ["## Completeness", "",
                "- **workflow telemetry**: %s%s" % (
                    {True: "complete", False: "incomplete", None: "unknown"}[wf["complete"]],
                    (" — " + "; ".join(wf["warnings"])) if wf["warnings"] else ""),
                "- **transcript scope**: %s%s%s" % (
                    _scope or "unknown",
                    (" (%s)" % _scope_note) if _scope_note and _scope_note != _scope else "",
                    (" [anchor: %s]" % _anchor) if _anchor else ""),]
        out += [
                "- **capture**: %s/%s API responses have a terminal stop%s"
                % (cap["api_calls"] - cap["incomplete_output"], cap["api_calls"],
                   "" if cap["complete"] else " (%d captured mid-flight — usage/output may be partial)" % cap["incomplete_output"]),
                "- **cost coverage**: %s" % comp["cost_coverage"], ""]
    out += ["## Run totals", ""]
    out += ["- **API calls**: %s" % _n(total["calls"])]
    if total.get("elapsed_ms") is not None:
        out += ["- **Elapsed (first request → last response)**: %s (%s → %s)"
                % (_hms(total["elapsed_ms"]), total["started"], total["ended"])]
    out += ["- **Billed span (Σ per-call, not wall-time)**: %s" % _hms(total["llm_ms"])]
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

    if phases:
        out += _md_breakdown(
            "Time, cost and tokens by phase", "phase", phases,
            "*elapsed\\** is an OBSERVED span: from a start inferred by stepping back the gap "
            "since the previous transcript record, to the last time the final response was seen. "
            "The transcript records no request time, so the left edge is inferred, and the span "
            "includes the compiling and benchmarking between calls. *billed span* is the sum of "
            "those same inferred per-call gaps, i.e. time plausibly spent waiting on the model. "
            "Phases overlap whenever work ran in parallel, so elapsed times do not add up "
            "to the run's.")
    if invocations and (len(invocations) > 1 or invocations[0]["name"] != OUTSIDE_WORKFLOW):
        out += _md_breakdown(
            "Workflow invocations counted", "invocation", invocations,
            "Every workflow invocation that worked in this run's eval dir is billed to it — "
            "a resumed or re-entered run is several invocations.")

    out += ["", "## Role view (organizational)", "",
            "> Nesting is by role rank + execution order, not a literal spawn tree "
            "(transcripts carry no cross-agent spawn edge). Each node is one agent "
            "attempt; counts/time/cost are that agent's **own** API calls, exclusive "
            "of children. \"Billed span\" is Σ per-call observed durations, not true "
            "request wall-time.", ""]

    def walk(node, depth):
        if not node.get("_root"):
            d = node_detail(node)
            out.append("%s- **%s**%s — %s API calls · %s billed · %s (own)"
                       % ("  " * depth, d["title"],
                          (" `%s`" % d["label_note"]) if d["label_note"] else "",
                          _n(d["calls"]), _hms(d["llm_ms"]), _usd(d["cost_usd"])))
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
                "- phase: %s%s" % (d["phase"] or "—",
                                   (" · workflow run: %s" % d["workflow_run"]) if d["workflow_run"] else ""),
                "- calls: %s · billed: %s · elapsed: %s" % (
                    _n(d["calls"]), _hms(d["llm_ms"]),
                    _hms(d["elapsed_ms"]) if d["elapsed_ms"] is not None else "—"),
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


def render_html(nodes, root, model, comp=None, phases=None, invocations=None):
    total, per_model = run_totals(nodes)
    show_inv = bool(invocations) and (len(invocations) > 1
                                      or invocations[0]["name"] != OUTSIDE_WORKFLOW)
    payload = {
        "model": model,
        "total": total,
        "per_model": [{"model": m, **pm} for m, pm in
                      sorted(per_model.items(), key=lambda kv: -kv[1]["cost_usd"])],
        "tree": _tree_json(root),
        "buckets": [{"key": k, "label": lbl} for k, lbl in COST_BUCKETS],
        "completeness": comp,
        "phases": phases or [],
        "invocations": invocations if show_inv else [],
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
  table.api { border-collapse:collapse; width:100%; font-size:12px; margin-top:6px; }
  table.api th, table.api td { border-bottom:1px solid var(--line); padding:3px 8px; text-align:left; white-space:nowrap; }
  table.api th { color:var(--muted); font-weight:600; }
  tr.apirow { cursor:pointer; }
  tr.apirow:hover td { background:var(--sel); }
  tr.apidetail { display:none; }
  tr.apidetail.open { display:table-row; }
  tr.apidetail td { white-space:normal; }
  .empty { color:var(--muted); padding:24px; }
  .sums { padding:6px 24px 14px; border-bottom:1px solid var(--line); }
  .sums h2 { font-size:15px; margin:12px 0 4px; }
  .sums .note { color:var(--muted); font-size:12px; margin:4px 0 0; }
  .scroll { overflow-x:auto; }
  table.sum { border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }
  table.sum th, table.sum td { border-bottom:1px solid var(--line); padding:3px 10px; white-space:nowrap; }
  table.sum th { color:var(--muted); font-weight:600; text-align:left; }
  table.sum td.r, table.sum th.r { text-align:right; }
  .lbl { color:var(--muted); font-size:12px; font-weight:400; }
</style></head>
<body>
<header><h1>__TITLE__</h1>
<div class="sub">Role view (organizational) — nesting is by role rank + execution order, <b>not</b> a literal spawn tree; transcripts carry no cross-agent spawn edge. Each node is one <b>agent attempt</b>; click it to expand its individual <b>API calls</b> and see cost / time / tokens / prompt / output. Node totals are that agent's <b>own</b> calls (exclusive of children). Cost is derived from transcript token buckets against a fixed rate card, not an SDK total.</div></header>
<div class="totals" id="totals"></div>
<div class="sums" id="sums"></div>
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
  if(tt.elapsed_ms!=null) tot.appendChild(chip(hms(tt.elapsed_ms),'elapsed · '+esc(tt.started)+' → '+esc(tt.ended)));
  tot.appendChild(chip(hms(tt.llm_ms),'billed span (Σ per-call, not wall-time)'));
  tot.appendChild(chip(usd(tt.cost_usd),'total cost'));
  tot.appendChild(chip(n(tt.tokens.cache_read),'cache-read tokens'));
  tot.appendChild(chip(n(tt.tokens.output),'output tokens'));
  D.per_model.forEach(function(pm){ tot.appendChild(chip(usd(pm.cost_usd), pm.model+' ('+n(pm.calls)+')')); });

  // Completeness banner: three distinct signals, not one flag.
  var C=D.completeness;
  if(C){
    var cap=C.capture_completeness, wf=C.workflow_completion;
    var b=document.createElement('div'); b.className='totals'; b.style.borderTop='none';
    var wfTxt = wf.complete===true?'complete':(wf.complete===false?'incomplete':'unknown');
    var capTxt = (cap.api_calls-cap.incomplete_output)+'/'+cap.api_calls+' terminal'
      + (cap.complete?'':(' · '+cap.incomplete_output+' mid-flight'));
    b.appendChild(chip(wfTxt,'workflow telemetry'+((wf.warnings&&wf.warnings.length)?' ⚠':'')));
    if(wf.transcript_scope){
      var scp=wf.transcript_scope;
      var scLbl={'explicit':'caller-named','run-scoped':'this run only',
                 'run-scoped-inferred':'INFERRED ⚠','partial':'PARTIAL ⚠',
                 'substring-fallback':'FALLBACK ⚠'}[scp]||scp;
      if(wf.transcript_scope_anchor){ scLbl += ' · anchor: '+wf.transcript_scope_anchor; }
      b.appendChild(chip(scp, 'transcript scope · '+scLbl));
    }
    b.appendChild(chip(capTxt, cap.complete?'capture complete':'capture partial'));
    var cc=document.createElement('div'); cc.className='chip'; cc.style.maxWidth='420px';
    cc.innerHTML='<b>cost coverage</b><span>'+esc(C.cost_coverage)+'</span>';
    b.appendChild(cc);
    tot.parentNode.insertBefore(b, tot.nextSibling);
  }

  // Time / cost / tokens by phase, and by workflow invocation.
  function sumTable(title, col, rows, note){
    if(!rows||!rows.length) return '';
    var body=rows.map(function(g){
      return '<tr><td>'+esc(g.name)+'</td><td>'+esc(g.started||'—')+'</td>'
        +'<td class="r">'+(g.elapsed_ms==null?'—':hms(g.elapsed_ms))+'</td>'
        +'<td class="r">'+hms(g.billed_ms)+'</td><td class="r">'+n(g.calls)+'</td>'
        +'<td class="r">'+n(g.agents)+'</td><td class="r">'+n(g.input_tokens)+'</td>'
        +'<td class="r">'+n(g.output_tokens)+'</td><td class="r cost">'+usd(g.cost_usd)+'</td></tr>';
    }).join('');
    return '<h2>'+esc(title)+'</h2><div class="scroll"><table class="sum"><thead><tr><th>'+esc(col)
      +'</th><th>started</th><th class="r" title="observed transcript span, left edge inferred">elapsed*</th><th class="r">billed span</th><th class="r">calls</th>'
      +'<th class="r">agents</th><th class="r">input tok</th><th class="r">output tok</th><th class="r">cost</th>'
      +'</tr></thead><tbody>'+body+'</tbody></table></div><div class="note">'+esc(note)+'</div>';
  }
  document.getElementById('sums').innerHTML =
    sumTable('Time, cost and tokens by phase','phase',D.phases,
      'elapsed* = an observed span: from a start inferred by stepping back the gap since the previous transcript '
      +'record, to the last time the final response was seen. The transcript records no request time, so the left '
      +'edge is inferred, and the span includes the compiling and benchmarking between calls; '
      +'billed span = the sum of those same inferred per-call gaps. Phases overlap when work ran in parallel, so elapsed times do not add up to the run\'s.')
    + sumTable('Workflow invocations counted','invocation',D.invocations,
      'Every workflow invocation that worked in this run\'s eval dir is billed to it: a resumed or re-entered run is several invocations.');
  if(!document.getElementById('sums').innerHTML) document.getElementById('sums').style.display='none';

  // Tree
  var sel=null;
  function detailPanel(d){
    var maxc=0; BUCKETS.forEach(function(b){ maxc=Math.max(maxc, d.cost[b.key]||0); });
    var bars=BUCKETS.map(function(b){
      var v=d.cost[b.key]||0, w=maxc>0?(100*v/maxc):0;
      return '<div class="bar"><div class="lab">'+esc(b.label)+'</div><div class="track"><div class="fill" style="width:'+w.toFixed(1)+'%"></div></div><div class="val">'+usd(v)+'</div></div>';
    }).join('');
    var tk=d.tokens;
    var api=d.api_calls||[];
    var h=''
      +'<h2>'+esc(d.title)+(d.label_note?(' <span class="lbl">'+esc(d.label_note)+'</span>'):'')+'</h2>'
      +'<dl class="kv">'
      +'<dt>role</dt><dd>'+esc(d.role)+(d.sub_phase?(' · '+esc(d.sub_phase)):'')+'</dd>'
      +'<dt>phase</dt><dd>'+esc(d.phase||'—')+'</dd>'
      +(d.workflow_run?('<dt>workflow run</dt><dd>'+esc(d.workflow_run)+'</dd>'):'')
      +'<dt>models</dt><dd>'+esc((d.models||[]).join(', ')||'—')+'</dd>'
      +'<dt>API calls</dt><dd>'+n(d.calls)+' (this agent, exclusive of children)</dd>'
      +'<dt title="observed transcript span, left edge inferred">elapsed*</dt><dd>'+(d.elapsed_ms==null?'—':hms(d.elapsed_ms))+(d.started?(' <span style="color:var(--muted)">('+esc(d.started)+' → '+esc(d.ended)+')</span>'):'')+'</dd>'
      +'<dt>billed span</dt><dd>'+hms(d.llm_ms)+' <span style="color:var(--muted)">(Σ per-call observed durations, not true request wall-time)</span></dd>'
      +'<dt>own cost</dt><dd>'+usd(d.cost_usd)+'</dd>'
      +'</dl>'
      +'<div class="bars"><div class="sub" style="color:var(--muted);font-size:12px;margin-bottom:4px">Cost by bucket (disjoint leaves)</div>'+bars+'</div>'
      +'<dl class="kv">'
      +'<dt>input cost (subtotal)</dt><dd>'+usd(d.input_cost_subtotal)+' <span style="color:var(--muted)">= cache-write + cache-read + uncached; not additive</span></dd>'
      +'<dt>uncached-input tok</dt><dd>'+n(tk.uncached_input)+'</dd>'
      +'<dt>cache-read tok</dt><dd>'+n(tk.cache_read)+'</dd>'
      +'<dt>cache-write tok</dt><dd>'+n(tk.cache_write)+'</dd>'
      +'<dt>output tok</dt><dd>'+n(tk.output)+' <span style="color:var(--muted)">(incl. thinking + tool args)</span></dd>'
      +'<dt>total input tok</dt><dd>'+n(tk.total_input)+'</dd>'
      +'</dl>';
    if(d.attribution) h+='<div class="sub" style="color:var(--muted);font-size:12px">attribution: '+esc(d.attribution)+'</div>';
    // Per-API-call table: an agent attempt expands to its provider responses,
    // and EACH response row opens to its own input buckets / output / thinking —
    // so the requested per-call drill-down reaches individual responses, not just
    // the agent-level concatenation.
    if(api.length){
      var span = function(ms){ return ms==null ? '—' : hms(ms); };
      var rows=api.map(function(c,i){
        var ct=c.tokens||{}, cc=c.cost||{};
        var costLine = BUCKETS.map(function(b){ return esc(b.label)+' '+usd(cc[b.key]||0); }).join(' · ');
        var detail=''
          +'<tr class="apidetail"><td></td><td colspan="5">'
          +'<div class="sub" style="color:var(--muted);font-size:12px">'
          +'msg '+esc(c.message_id||'—')+' · tokens: uncached-input '+n(ct.uncached_input)
          +' · cache-read '+n(ct.cache_read)+' · cache-write '+n(ct.cache_write)+' · output '+n(ct.output)+'</div>'
          +'<div class="sub" style="color:var(--muted);font-size:12px">cost: '+costLine+'</div>'
          +'<details><summary>thinking</summary><pre>'+esc(c.thinking||'(none captured)')+'</pre></details>'
          +'<details><summary>output</summary><pre>'+esc(c.output||'(none captured)')+'</pre></details>'
          +'</td></tr>';
        return '<tr class="apirow" data-i="'+i+'"><td>'+(i+1)+'</td><td>'+esc(c.model.replace('claude-',''))+'</td><td>'+span(c.duration_ms)+'</td>'
          +'<td style="text-align:right">'+n(ct.output)+'</td>'
          +'<td style="text-align:right">'+usd(c.cost_usd)+'</td>'
          +'<td>'+esc(c.stop_reason||'')+'</td></tr>'+detail;
      }).join('');
      h+='<details open><summary>API calls ('+api.length+') — click a row to open its input/output/thinking</summary>'
        +'<div style="overflow-x:auto"><table class="api"><thead><tr><th>#</th><th>model</th><th>span</th><th>out tok</th><th>cost</th><th>stop</th></tr></thead><tbody>'
        +rows+'</tbody></table></div></details>';
    }
    h+='<details><summary>Input prompt <span style="color:var(--muted)">(transcript snippet — not the full wire prompt/system/tool defs)</span></summary><pre>'+esc(d.prompt||'(none captured)')+'</pre></details>';
    h+='<details><summary>Thinking <span style="color:var(--muted)">(when captured; may be redacted/unavailable)</span></summary><pre>'+esc(d.thinking||'(none captured)')+'</pre></details>';
    h+='<details><summary>Output (response)</summary><pre>'+esc(d.output||'(none captured)')+'</pre></details>';
    return h;
  }
  function show(d, rowEl){
    var panel=document.getElementById('panel');
    panel.innerHTML = detailPanel(d);
    if(sel) sel.classList.remove('sel');
    sel=rowEl; if(sel) sel.classList.add('sel');
    // Each API-call row toggles its own detail row (input buckets / output / thinking).
    panel.querySelectorAll('tr.apirow').forEach(function(tr){
      tr.addEventListener('click', function(){
        var det=tr.nextElementSibling;
        if(det && det.classList.contains('apidetail')) det.classList.toggle('open');
      });
    });
  }
  function drawNode(node, ul){
    var li=document.createElement('li'); li.className='node';
    var row=document.createElement('div'); row.className='row';
    var kids=node.children||[];
    var tog=document.createElement('span'); tog.className='tog'; tog.textContent=kids.length?'▾':'·';
    row.appendChild(tog);
    var tt=document.createElement('span'); tt.className='tt'; tt.textContent=node.title; row.appendChild(tt);
    if(node.detail && node.detail.label_note){ var lb=document.createElement('span'); lb.className='lbl'; lb.textContent=node.detail.label_note; row.appendChild(lb); }
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
def render(rows, model="run", meta=None):
    """Return (html_str, md_str) for a list of call rows."""
    nodes = agentize(rows)
    root = build_tree(nodes)
    comp = completeness(rows, meta)
    phases = breakdown_rows(rows, "phase", "(no phase)")
    invocations = breakdown_rows(rows, "workflow_run", OUTSIDE_WORKFLOW)
    return (render_html(nodes, root, model, comp, phases, invocations),
            render_markdown(nodes, root, model, comp, phases, invocations))


def _read_ledger_meta(calls_path):
    """Best-effort: the ledger's token_stats.json (sibling of llm_calls.jsonl)
    carries the run's completeness meta. Absent/unreadable -> None."""
    stats = os.path.join(os.path.dirname(calls_path), "token_stats.json")
    try:
        with open(stats, "r", encoding="utf-8") as fh:
            return (json.load(fh) or {}).get("meta")
    except (OSError, ValueError):
        return None


def write(calls_path, out_dir, model="run"):
    """Read ``llm_calls.jsonl`` and write the HTML + MD report. Returns paths."""
    rows = read_calls(calls_path)
    html_str, md_str = render(rows, model, _read_ledger_meta(calls_path))
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
