#!/usr/bin/env python3
"""Live execution tracker for a GEAK workflow run (e2e or kernel lane).

Reads the three artifacts the Claude Code runtime writes *while a workflow is
running* and joins them into one durable trace:

  <workflow_dir>/journal.jsonl        orchestration events (authoritative ORDER,
                                      agent identity, and the result each agent
                                      returned to the workflow)
  <workflow_dir>/agent-<id>.meta.json per-agent metadata (label, spawn depth)
  <workflow_dir>/agent-<id>.jsonl     per-agent transcript (API calls, tool
                                      actions, tool results, token usage)

What this module will and will not claim
----------------------------------------
The journal proves *who ran* and *what each agent returned to the workflow*. In
the runs observed so far it carries NO timestamps, every ``started.phase`` is the
same workflow label, and ``meta.spawnDepth`` is a level, never a parent id. So:

* orchestration edges (workflow -> agent) and return edges (agent -> workflow)
  are emitted as PROVEN, sourced from journal records;
* a role ordering such as tech_lead-before-engineer is a DATA DEPENDENCY of the
  workflow script, not evidence that one agent spawned another, so no such edge
  is invented here;
* wall-clock timing is derived from transcript record timestamps and is marked
  ``estimated`` -- it is not provider latency, and cumulative elapsed is an
  offset from the run origin, never a sum of overlapping spans;
* an input excerpt is the new transcript input since the previous response, NOT
  the full API request: the system prompt, tool definitions and inherited
  context are not recoverable from a transcript and are never implied here.

Everything unknown is recorded as an explicit marker rather than a blank.

Collection is read-only and side-effect free with respect to the run: it never
influences model choice, prompts, budgets or optimization decisions, and a
failure to collect is reported, never raised into the caller's workflow.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

SCHEMA = "geak.trace/1"

# Per-preview byte cap. Previews are excerpts for reading, never a statement
# about billed tokens; every truncation is flagged with the original length.
PREVIEW_BYTES = 4096
ARGS_PREVIEW_BYTES = 2048
RESULT_PREVIEW_BYTES = 2048

# Marker vocabulary. Callers/renderers branch on these instead of on "".
NOT_CAPTURED = "not_captured"
RECORDED_UNREADABLE = "recorded_unreadable"
TEXT = "text"

_SECRET_PATTERNS = [
    (re.compile(r"\b(sk-[A-Za-z0-9_\-]{12,})"), "sk-***REDACTED***"),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{16,})"), "gh*_***REDACTED***"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token)"
                r"(\s*[:=]\s*)(['\"]?)([^\s'\"]{6,})"), r"\1\2\3***REDACTED***"),
    (re.compile(r"\b(AKIA[0-9A-Z]{16})\b"), "AKIA***REDACTED***"),
]


def redact(text):
    """Best-effort credential scrubbing before an excerpt is persisted."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pat, repl in _SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out


def preview(text, cap=PREVIEW_BYTES):
    """Return ``(shown_text, truncated, original_bytes)`` for a bounded excerpt.

    Truncation is on a UTF-8 byte budget (so the cap means what it says even for
    non-ASCII) and is then repaired to a codepoint boundary.
    """
    if text is None:
        return "", False, 0
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False, default=str)
        except Exception:
            text = str(text)
    raw = text.encode("utf-8", "replace")
    total = len(raw)
    if total <= cap:
        return redact(text), False, total
    return redact(raw[:cap].decode("utf-8", "ignore")), True, total


def iter_jsonl_tolerant(path):
    """Yield parsed records from a JSONL file being appended to concurrently.

    A trailing partial line (the writer mid-flush) and any single malformed line
    are skipped rather than aborting the read, so polling a live file is safe.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    # Partial trailing write, or a corrupt line; either way the
                    # next poll re-reads the file and picks it up once complete.
                    continue
    except (OSError, IOError):
        return


def _iso_to_ms(s):
    if not s or not isinstance(s, str):
        return None
    try:
        txt = s.replace("Z", "+00:00")
        import datetime as _dt
        return int(_dt.datetime.fromisoformat(txt).timestamp() * 1000)
    except Exception:
        return None


def read_journal(workflow_dir):
    """Parse journal.jsonl into ordered started/result events.

    Returns ``(started, results, launched, ordinals)`` where ``started`` is an
    ordered list of dicts and ``results`` maps agent_id -> returned result. The
    ordinal is the journal's own event order, which IS authoritative even though
    the journal carries no timestamps.
    """
    path = os.path.join(workflow_dir, "journal.jsonl")
    started, results, launched, ordinals = [], {}, False, {}
    for rec in iter_jsonl_tolerant(path):
        kind = rec.get("type")
        if kind == "launched":
            launched = True
        elif kind == "started":
            aid = rec.get("agentId")
            if not aid:
                continue
            if aid not in ordinals:
                ordinals[aid] = len(started)
                started.append({
                    "agent_id": aid,
                    "journal_key": rec.get("key"),
                    "label": rec.get("label"),
                    "journal_phase": rec.get("phase"),
                })
        elif kind == "result":
            aid = rec.get("agentId")
            if aid:
                results[aid] = rec.get("result")
    return started, results, launched, ordinals


def read_run_record(workflow_dir):
    """Read this run's ``wf_<runId>.json`` lifecycle record, if it exists.

    The journal contains only launched/started/result -- it has NO terminal
    event -- so agent quiescence can never establish that a workflow finished.
    Between one agent returning and the next being dispatched, every observed
    agent has returned, which is indistinguishable from completion. The run
    record is the authoritative lifecycle signal, so completion is read from
    there or not claimed at all.
    """
    workflow_dir = os.path.abspath(workflow_dir)
    run_id = os.path.basename(workflow_dir)
    session_dir = os.path.dirname(os.path.dirname(os.path.dirname(workflow_dir)))
    # A mirror keeps its own copy beside the sources, so a rebuild from the
    # mirror still knows the run's real lifecycle status instead of degrading
    # to "unknown" just because the Claude home is gone.
    candidates = [os.path.join(session_dir, "workflows", "%s.json" % run_id),
                  os.path.join(workflow_dir, "run_record.json")]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return (data if isinstance(data, dict) else {}), path
        except Exception:
            continue
    return None, candidates[0]


#: Run-record statuses that mean the workflow is over. Anything else (including
#: an absent record) leaves the trace non-terminal.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "canceled",
                                "error", "aborted", "timeout"})


def read_agent_timeline(workflow_dir):
    """Per-agent phases as the WORKFLOW itself recorded them, or None.

    A nested lane's phases are collapsed in the parent journal to a single
    "> <lane>" group, so the journal cannot describe a kernel run's real
    pipeline. The workflow records them itself though: kernel_lane builds a
    ``geak.agent_timeline/1`` timeline at dispatch, with the phase, label, role
    and sub_phase of every agent, and it is carried in the run record's
    ``result.llm_timeline``. That is a recorded source, not an inference.

    Preference order: the run record (always present once the run returns), then
    a persisted ``agent_timeline.json`` beside the run's other trace artifacts.
    """
    record, _ = read_run_record(workflow_dir)
    for holder in ((record or {}).get("result"), record or {}):
        if isinstance(holder, dict):
            tl = holder.get("llm_timeline")
            if isinstance(tl, dict) and isinstance(tl.get("events"), list):
                return tl, "run_record.result.llm_timeline"
    eval_dir = eval_dir_of_run(workflow_dir)
    for cand in ([os.path.join(eval_dir, "reports", "trace", "agent_timeline.json")]
                 if eval_dir else []) + [
                 os.path.join(workflow_dir, "agent_timeline.json")]:
        try:
            with open(cand, "r", encoding="utf-8") as fh:
                tl = json.load(fh)
            if isinstance(tl, dict) and isinstance(tl.get("events"), list):
                return tl, cand
        except Exception:
            continue
    return None, None


def _timeline_index(timeline):
    """Map label -> ordered timeline events, so repeats join by dispatch order."""
    by_label = {}
    for ev in (timeline or {}).get("events") or []:
        if not isinstance(ev, dict):
            continue
        by_label.setdefault(ev.get("label"), []).append(ev)
    for evs in by_label.values():
        evs.sort(key=lambda e: e.get("seq") if isinstance(e.get("seq"), int) else 0)
    return by_label


def read_meta(workflow_dir, agent_id):
    path = os.path.join(workflow_dir, "agent-%s.meta.json" % agent_id)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _blocks(message):
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in (content or []) if isinstance(b, dict)]


def _reasoning_state(blocks):
    """Classify reasoning honestly.

    A ``thinking`` block whose text is empty (signature only) means the reasoning
    WAS recorded but its text is unavailable to us -- materially different from
    no reasoning block at all. Cryptographic signatures are never serialized.
    """
    n = 0
    texts = []
    for b in blocks:
        if b.get("type") in ("thinking", "redacted_thinking"):
            n += 1
            t = b.get("thinking")
            if isinstance(t, str) and t.strip():
                texts.append(t)
    if not n:
        return {"blocks": 0, "state": NOT_CAPTURED, "text": ""}
    if texts:
        shown, trunc, total = preview("\n".join(texts))
        return {"blocks": n, "state": TEXT, "text": shown,
                "truncated": trunc, "bytes_total": total}
    return {"blocks": n, "state": RECORDED_UNREADABLE, "text": ""}


def _input_blocks_from_user(rec):
    """Extract the renderable parts of one user record.

    Tool results are the mechanical feedback of the previous call's actions;
    plain text is a human/orchestrator instruction. Both are 'new input', and
    each keeps its own provenance so the UI can join a result to its action.
    """
    out = []
    msg = rec.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        if content.strip():
            shown, trunc, total = preview(content)
            out.append({"kind": "text", "text": shown,
                        "truncated": trunc, "bytes_total": total,
                        "source_uuid": rec.get("uuid")})
        return out
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and isinstance(b.get("text"), str):
            if b["text"].strip():
                shown, trunc, total = preview(b["text"])
                out.append({"kind": "text", "text": shown,
                            "truncated": trunc, "bytes_total": total,
                            "source_uuid": rec.get("uuid")})
        elif b.get("type") == "tool_result":
            body = b.get("content")
            if isinstance(body, list):
                parts, omitted = [], []
                for sub in body:
                    if not isinstance(sub, dict):
                        continue
                    if sub.get("type") == "text":
                        parts.append(sub.get("text") or "")
                    else:
                        # Images/binary are referenced, never inlined as base64.
                        omitted.append(sub.get("type") or "unknown")
                body = "\n".join(parts)
                if omitted:
                    body += "\n[omitted non-text blocks: %s]" % ", ".join(omitted)
            shown, trunc, total = preview(body, RESULT_PREVIEW_BYTES)
            out.append({"kind": "tool_result",
                        "tool_use_id": b.get("tool_use_id"),
                        "is_error": bool(b.get("is_error")),
                        "text": shown, "truncated": trunc,
                        "bytes_total": total,
                        "source_uuid": rec.get("uuid")})
    return out


def build_agent_calls(transcript_path, rates=None, cost_fns=None):
    """Reconstruct one agent's API calls from its transcript.

    Streaming means one logical response can appear as several records sharing a
    ``message.id``; content is merged by ``(kind, apiBlockIndex + position)``
    keeping the longest text per block, and usage is taken from the record with
    the largest ``output_tokens`` (a partial flush can only undercount). The
    input window for a call is every user record seen since the PREVIOUS
    DISTINCT message id -- frozen when the id first appears, so later streamed
    records for that same id merge content without resetting the window.
    """
    calls, order = {}, []
    pending_input = []      # user records seen since the previous distinct id
    pending_uuids = []
    content_by_id = {}
    prev_ts_ms = None

    for rec in iter_jsonl_tolerant(transcript_path):
        rtype = rec.get("type")
        if rtype == "user":
            pending_input.extend(_input_blocks_from_user(rec))
            if rec.get("uuid"):
                pending_uuids.append(rec.get("uuid"))
            continue
        if rtype != "assistant":
            continue

        msg = rec.get("message") or {}
        mid = msg.get("id") or rec.get("requestId") or rec.get("uuid")
        if not mid:
            continue
        ts_ms = _iso_to_ms(rec.get("timestamp"))
        blocks = _blocks(msg)
        base = rec.get("apiBlockIndex") or 0

        store = content_by_id.setdefault(mid, {})
        for j, b in enumerate(blocks):
            btype = b.get("type")
            if btype == "text" and isinstance(b.get("text"), str):
                key = ("resp", base + j)
                if len(b["text"]) >= len(store.get(key, "")):
                    store[key] = b["text"]
            elif btype == "tool_use":
                key = ("tool", base + j)
                store[key] = {
                    "tool_use_id": b.get("id"),
                    "name": b.get("name"),
                    "input": b.get("input"),
                }
            elif btype in ("thinking", "redacted_thinking"):
                # Merged by block identity like text, not by counting blocks: a
                # block can be flushed empty (signature only) and then re-flushed
                # carrying readable text, and a count-based rule would keep the
                # empty one. Longest-wins preserves whatever text was recorded.
                key = ("think", base + j)
                txt = b.get("thinking") if isinstance(b.get("thinking"), str) else ""
                prev = store.get(key)
                prev_txt = prev.get("text") if isinstance(prev, dict) else ""
                if prev is None or len(txt) >= len(prev_txt or ""):
                    store[key] = {"text": txt, "redacted": btype == "redacted_thinking"}

        if mid not in calls:
            # First sighting of this response: freeze its input window.
            usage = msg.get("usage") or {}
            cache = usage.get("cache_creation") or {}
            calls[mid] = {
                "call_id": mid,
                "message_id": msg.get("id"),
                "request_id": rec.get("requestId"),
                "model": msg.get("model"),
                "ts": rec.get("timestamp"),
                "ts_ms": ts_ms,
                "duration_ms_est": ((ts_ms - prev_ts_ms)
                                    if (ts_ms is not None and prev_ts_ms is not None
                                        and ts_ms >= prev_ts_ms) else None),
                "stop_reason": msg.get("stop_reason"),
                "usage": {
                    "input_tokens": int(usage.get("input_tokens") or 0),
                    "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
                    "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
                    "cache_write_5m_tokens": int(cache.get("ephemeral_5m_input_tokens") or 0),
                    "cache_write_1h_tokens": int(cache.get("ephemeral_1h_input_tokens") or 0),
                    "output_tokens": int(usage.get("output_tokens") or 0),
                    "service_tier": usage.get("service_tier"),
                },
                "input": {
                    "kind": ("new_input_since_previous_response"
                             if pending_input else "none_recorded"),
                    "note": ("Transcript excerpt of new input since the previous "
                             "response. NOT the full API request: system prompt, "
                             "tool definitions and inherited context are not "
                             "recorded in the transcript."),
                    "blocks": list(pending_input),
                    "source_uuids": list(pending_uuids),
                },
                "_usage_out": int(usage.get("output_tokens") or 0),
                "_usage_seen": bool(usage),
            }
            order.append(mid)
            pending_input, pending_uuids = [], []
        else:
            usage = msg.get("usage") or {}
            out_tok = int(usage.get("output_tokens") or 0)
            if out_tok > calls[mid]["_usage_out"]:
                cache = usage.get("cache_creation") or {}
                calls[mid]["usage"].update({
                    "input_tokens": int(usage.get("input_tokens") or 0),
                    "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
                    "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
                    "cache_write_5m_tokens": int(cache.get("ephemeral_5m_input_tokens") or 0),
                    "cache_write_1h_tokens": int(cache.get("ephemeral_1h_input_tokens") or 0),
                    "output_tokens": out_tok,
                })
                calls[mid]["_usage_out"] = out_tok
            if calls[mid].get("stop_reason") is None:
                calls[mid]["stop_reason"] = msg.get("stop_reason")
            if usage:
                calls[mid]["_usage_seen"] = True

        if ts_ms is not None:
            prev_ts_ms = ts_ms

    # Materialize merged content, then join tool results to their actions.
    results_by_tool_id = _collect_tool_results(transcript_path)
    out = []
    for mid in order:
        call = calls[mid]
        store = content_by_id.get(mid, {})
        ordered = sorted(store.items(), key=lambda kv: kv[0][1])
        texts = [v for (k, _), v in ordered if k == "resp" and isinstance(v, str)]
        joined = "\n".join(t for t in texts if t)
        shown, trunc, total = preview(joined)
        call["output_text"] = shown
        call["output_truncated"] = trunc
        call["output_bytes_total"] = total

        think_blocks = [v for (k, _), v in ordered if k == "think"]
        think_texts = [v["text"] for v in think_blocks if (v.get("text") or "").strip()]
        if not think_blocks:
            call["reasoning"] = {"blocks": 0, "state": NOT_CAPTURED, "text": ""}
        elif think_texts:
            shown, trunc, total = preview("\n".join(think_texts))
            call["reasoning"] = {"blocks": len(think_blocks), "state": TEXT,
                                 "text": shown, "truncated": trunc,
                                 "bytes_total": total}
        else:
            call["reasoning"] = {"blocks": len(think_blocks),
                                 "state": RECORDED_UNREADABLE, "text": ""}

        # Usage coverage: a record with no usage block is UNKNOWN, not zero. It
        # must not silently contribute 0 tokens / $0 to a total that is presented
        # as the run's spend.
        call["usage_known"] = bool(call.pop("_usage_seen", False))

        actions = []
        for (kind, _), v in ordered:
            if kind != "tool":
                continue
            args_shown, args_trunc, args_total = preview(v.get("input"), ARGS_PREVIEW_BYTES)
            tid = v.get("tool_use_id")
            res = results_by_tool_id.get(tid)
            actions.append({
                "tool_use_id": tid,
                "name": v.get("name"),
                "args_preview": args_shown,
                "args_truncated": args_trunc,
                "args_bytes_total": args_total,
                "result": res if res else {"status": "missing",
                                           "note": "No tool_result recorded for this "
                                                   "action (run may be live, "
                                                   "cancelled, or interrupted)."},
            })
        call["actions"] = actions

        has_text = bool(joined.strip())
        if has_text and actions:
            call["output_kind"] = "mixed"
        elif actions:
            call["output_kind"] = "actions"
        elif has_text:
            call["output_kind"] = "text"
        else:
            call["output_kind"] = "incomplete"

        if rates is not None and cost_fns is not None and call.get("usage_known", True):
            call["cost_usd"], call["cost_breakdown"] = _cost_for(call, rates, cost_fns)
        else:
            # No usage block recorded: the cost is UNKNOWN, not zero. Pricing it
            # as $0 would understate a total that is presented as the run spend.
            call["cost_usd"], call["cost_breakdown"] = None, None

        call.pop("_usage_out", None)
        out.append(call)

    for i, call in enumerate(out):
        call["index"] = i
    return out


def _collect_tool_results(transcript_path):
    """Map tool_use_id -> bounded result preview, joined across the transcript."""
    found = {}
    for rec in iter_jsonl_tolerant(transcript_path):
        if rec.get("type") != "user":
            continue
        for blk in _input_blocks_from_user(rec):
            if blk.get("kind") != "tool_result":
                continue
            tid = blk.get("tool_use_id")
            if not tid or tid in found:
                continue
            found[tid] = {
                "status": "error" if blk.get("is_error") else "ok",
                "preview": blk.get("text", ""),
                "truncated": blk.get("truncated", False),
                "bytes_total": blk.get("bytes_total", 0),
                "source_uuid": blk.get("source_uuid"),
            }
    return found


def _cost_for(call, rates, cost_fns):
    """Price one call with the ledger's own functions, so accounting never forks."""
    cost_of, cost_breakdown = cost_fns
    row = dict(call["usage"])
    row["model"] = call.get("model")
    if not row.get("cache_write_5m_tokens") and not row.get("cache_write_1h_tokens"):
        row["cache_write_5m_tokens"] = row.get("cache_creation_input_tokens", 0)
    try:
        return cost_of(row, rates), cost_breakdown(row, rates)
    except Exception:
        return None, None


def _load_cost_support(rates_path=None):
    """Import the ledger's rate table and pricing functions if importable.

    ``rates_path`` is merged exactly as ``llm_ledger`` merges ``--rates`` so a
    priced trace and the ledger report beside it cannot disagree.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(os.path.dirname(here), "e2e_workflow", "scripts")
    if cand not in sys.path:
        sys.path.insert(0, cand)
    try:
        import llm_ledger  # noqa: F401
        rates = dict(llm_ledger.DEFAULT_RATES)
        if rates_path:
            with open(rates_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                merged = dict(llm_ledger.DEFAULT_RATES["_default"])
                merged.update(loaded.get("_default") or {})
                rates = dict(loaded)
                rates["_default"] = merged
        return rates, (llm_ledger.cost_of, llm_ledger.cost_breakdown)
    except Exception:
        return None, None


def build_trace(workflow_dir, run_status=None, price=True, rates_path=None,
                resolver_info=None):
    """Join journal + meta + transcripts into one trace document."""
    workflow_dir = os.path.abspath(workflow_dir)
    started, results, launched, _ = read_journal(workflow_dir)
    rates, cost_fns = (_load_cost_support(rates_path) if price else (None, None))

    warnings = []
    if not launched and not started:
        warnings.append("No journal records read yet; the run may be starting, "
                        "or this directory is not a workflow run.")

    agents, edges = [], []
    _rec_early, _ = read_run_record(workflow_dir)
    run_node = "run:%s" % ((_rec_early or {}).get("runId")
                           or os.path.basename(workflow_dir))
    origin_ms, end_ms = None, None
    journal_has_ts = False

    for ordinal, ev in enumerate(started):
        aid = ev["agent_id"]
        meta = read_meta(workflow_dir, aid)
        tpath = os.path.join(workflow_dir, "agent-%s.jsonl" % aid)
        has_transcript = os.path.exists(tpath)
        calls = build_agent_calls(tpath, rates, cost_fns) if has_transcript else []
        # Retained generations hold content a later rewrite replaced. Reading only
        # the current file would silently drop calls the mirror deliberately kept.
        import glob as _g
        gens = sorted(_g.glob(tpath + ".gen*"))
        if gens:
            by_id = {c.get("call_id"): c for c in calls}
            for gen in gens:
                for call in build_agent_calls(gen, rates, cost_fns):
                    cid = call.get("call_id")
                    by_id[cid] = _merge_call(by_id.get(cid), call)
            calls = sorted(by_id.values(),
                           key=lambda c: (c.get("ts_ms") is None, c.get("ts_ms") or 0))
            for i, c in enumerate(calls):
                c["index"] = i
            has_transcript = True
            warnings.append(
                "agent %s: reconciled %d retained source generation(s) with the "
                "current transcript." % (aid, len(gens)))

        ts_list = [c["ts_ms"] for c in calls if c.get("ts_ms") is not None]
        first_ms = min(ts_list) if ts_list else None
        last_ms = max(ts_list) if ts_list else None
        if first_ms is not None:
            origin_ms = first_ms if origin_ms is None else min(origin_ms, first_ms)
        if last_ms is not None:
            end_ms = last_ms if end_ms is None else max(end_ms, last_ms)

        returned = aid in results
        totals = _totals_of(calls)
        if rates is None:
            totals["cost_usd"] = None

        result_preview, result_trunc, result_total = (None, False, 0)
        if returned:
            result_preview, result_trunc, result_total = preview(
                results.get(aid), PREVIEW_BYTES)

        agents.append({
            "agent_id": aid,
            "ordinal": ordinal,
            "journal_key": ev.get("journal_key"),
            "label": ev.get("label"),
            "journal_phase": ev.get("journal_phase"),
            "description": meta.get("description"),
            "agent_type": meta.get("agentType"),
            "spawn_depth": meta.get("spawnDepth"),
            "request_shape": meta.get("requestShape"),
            "transcript": tpath if has_transcript else None,
            "transcript_status": "present" if has_transcript else "missing",
            "status": ("completed" if returned
                       else ("running_or_incomplete" if has_transcript else "started_no_transcript")),
            "result_status": "returned_to_workflow" if returned else "pending_or_absent",
            "result": results.get(aid) if returned else None,
            "result_preview": result_preview,
            "result_truncated": result_trunc,
            "result_bytes_total": result_total,
            "first_ts_ms": first_ms,
            "last_ts_ms": last_ms,
            "timing_provenance": "transcript_timestamps_estimated",
            "totals": totals,
            "calls": calls,
        })

        edges.append({
            "type": "orchestration",
            "from": run_node,
            "to": "agent:%s" % aid,
            "ordinal": ordinal,
            "provenance": "journal.started",
            "proven": True,
        })
        if returned:
            edges.append({
                "type": "return",
                "from": "agent:%s" % aid,
                "to": run_node,
                "provenance": "journal.result",
                "proven": True,
            })
        if not has_transcript:
            warnings.append("agent %s (%s): started with no transcript file yet."
                            % (aid, ev.get("label")))

    if started and not journal_has_ts:
        warnings.append(
            "Journal records carry no timestamps; all times shown are derived "
            "from transcript record timestamps and are ESTIMATES (they include "
            "queueing/idle gaps and are not provider latency).")
    if any(a["spawn_depth"] == 1 for a in agents) and agents:
        warnings.append(
            "No parent-agent id is recorded for any agent; the proven structure "
            "is workflow -> agent invocations and their returns. Role ordering "
            "(e.g. tech_lead before engineer) is a workflow data dependency, "
            "not evidence of agent-to-agent spawning.")

    # Recorded linkage (result-transfer / spawn) if the run emitted any. Absent
    # is the normal case and is reported as "not recorded", never as "none exist".
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import geak_trace_events as _events
    except Exception:
        _events = None

    # Real phases, when the workflow recorded them. Joined by label; repeated
    # labels are consumed in dispatch order rather than guessed between.
    timeline, tl_source = read_agent_timeline(workflow_dir)
    if timeline:
        tl_index = _timeline_index(timeline)
        used, joined, unjoined = {}, 0, []
        for agent in agents:
            evs = tl_index.get(agent.get("label")) or []
            i = used.get(agent.get("label"), 0)
            if i < len(evs):
                ev = evs[i]
                used[agent.get("label")] = i + 1
                agent["timeline_phase"] = ev.get("phase") or None
                agent["timeline_role"] = ev.get("role") or None
                agent["timeline_sub_phase"] = ev.get("sub_phase") or None
                agent["timeline_seq"] = ev.get("seq")
                agent["phase_provenance"] = "workflow_timeline"
                # The timeline records label/seq/attempt but NO runtime agent id,
                # so when a label occurs more than once the match is by position,
                # not by an established join. Say so rather than presenting a
                # candidate phase as settled.
                if len(evs) > 1:
                    agent["timeline_attribution_ambiguous"] = True
                joined += 1
            else:
                unjoined.append(agent.get("label"))
        warnings.append(
            "Phases for %d of %d agent(s) come from the workflow's OWN recorded "
            "timeline (%s), not from labels." % (joined, len(agents), tl_source))
        if unjoined:
            warnings.append(
                "%d agent(s) had no timeline entry and fall back to label-derived "
                "grouping: %s" % (len(unjoined), ", ".join(sorted(set(unjoined))[:6])))

    n_done = sum(1 for a in agents if a["status"] == "completed")

    # Lifecycle: completion comes from the run record, never from the fact that
    # every agent observed SO FAR has returned -- in a sequential workflow that
    # is also true in the gap before the next dispatch.
    record, record_path = read_run_record(workflow_dir)
    record_status = (record or {}).get("status")
    terminal = isinstance(record_status, str) and record_status.lower() in _TERMINAL_STATUSES
    if run_status:
        status, status_reason = run_status, "caller-supplied"
    elif terminal:
        status = "complete"
        status_reason = "run record status=%s" % record_status
    elif record is not None:
        status = "live"
        status_reason = "run record status=%s (not terminal)" % (record_status or "unset")
    else:
        status = "unknown"
        status_reason = ("no run record at %s; completion cannot be established "
                         "from the journal alone" % record_path)
    if not terminal and agents and n_done == len(agents):
        warnings.append(
            "All %d observed agent(s) have returned, but this is NOT evidence the "
            "workflow finished: the journal has no terminal event, and a sequential "
            "workflow looks identical between a return and the next dispatch."
            % len(agents))

    # Resolver provenance belongs IN the trace: how this run was identified, and
    # whether anything was ambiguous, is coverage information a reader needs.
    if resolver_info:
        if resolver_info.get("ambiguous"):
            warnings.append(
                "Run identity was ambiguous: %d records owned this directory; "
                "the observer attached to %s."
                % (resolver_info["ambiguous"], resolver_info.get("run_id")))
        if resolver_info.get("skipped_started_before_observer"):
            warnings.append(
                "%d owning record(s) were skipped as pre-existing runs that started "
                "before this observer." % resolver_info["skipped_started_before_observer"])

    # Identity must survive a rebuild from a mirror. The mirror's folder name is
    # not a run id, and treating it as one silently renames the run, breaks the
    # graph's run node, and defeats the cross-run retention guard when a rebuilt
    # trace is reconciled with its original capture.
    _dir_name = os.path.basename(os.path.abspath(workflow_dir))
    _true_run_id = (record or {}).get("runId") or _dir_name
    if _true_run_id != _dir_name:
        warnings.append(
            "Rebuilt from a directory named %r; run identity restored from the "
            "mirrored run record as %s." % (_dir_name, _true_run_id))

    trace_out = {
        "schema": SCHEMA,
        "run": {
            "run_id": _true_run_id,
            "source_dir_name": _dir_name,
            "resolver": (dict(resolver_info) if resolver_info else None),
            "workflow_dir": workflow_dir,
            "status": status,
            "status_reason": status_reason,
            "status_provenance": ("run_record" if record is not None else "absent"),
            "record_status": record_status,
            "launched": launched,
            "agents_started": len(agents),
            "agents_returned": n_done,
            "origin_ts_ms": origin_ms,
            "end_ts_ms": end_ms,
            "elapsed_ms_est": ((end_ms - origin_ms)
                               if (origin_ms is not None and end_ms is not None) else None),
            "collected_at_ms": int(time.time() * 1000),
            "timing_note": ("Cumulative elapsed is an offset from the run origin; "
                            "spans may overlap and must not be summed."),
        },
        "agents": agents,
        "edges": edges,
        "warnings": warnings,
    }
    if _events is not None:
        try:
            trace_out = _events.attach(
                trace_out, os.path.join(workflow_dir, "linkage_events.jsonl"))
        except Exception as exc:
            trace_out["warnings"].append("linkage attach failed: %s" % exc)
    return trace_out


def write_trace(trace, out_path):
    """Atomically publish the trace so a reader never sees a half-written file."""
    out_path = os.path.abspath(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError:
            pass
    # Unique per writer: a shared ".tmp" would let two observers (a restart, or
    # two runs under one exp_root) clobber each other mid-write.
    tmp = "%s.tmp.%d.%d" % (out_path, os.getpid(), int(time.time() * 1000) % 100000)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(trace, fh, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, out_path)
    return out_path


try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import geak_trace_reconcile as _rc
except Exception:  # pragma: no cover - reconciliation is required in practice
    _rc = None


def _richness(call):
    """Ordering key for "how much of this call did we actually capture".

    Used to stop a shorter re-read of the SAME call id from replacing a richer
    earlier capture -- a truncated or mid-flush record is not new information.
    """
    usage = call.get("usage") or {}
    return (len(call.get("output_text") or ""),
            int(usage.get("output_tokens") or 0),
            len(call.get("actions") or []),
            len((call.get("input") or {}).get("blocks") or []),
            len((call.get("reasoning") or {}).get("text") or ""))


def _merge_call(prev, new):
    """Field-wise monotonic merge of two captures of the SAME call id.

    Each observable is taken from whichever capture actually has it, so a
    shortened re-read can never delete text, actions, usage or reasoning that
    were already recorded.
    """
    if prev is None:
        return new
    if new is None:
        return prev
    merged = dict(new)
    if len(prev.get("output_text") or "") > len(new.get("output_text") or ""):
        merged["output_text"] = prev.get("output_text")
        merged["output_truncated"] = prev.get("output_truncated")
        merged["output_bytes_total"] = prev.get("output_bytes_total")
    # Actions merge by tool_use_id. Comparing list LENGTHS loses a previously
    # recorded tool_result when the counts happen to match, and loses the action
    # entirely when a later read omits it. Absence in a later read is not
    # evidence the event never happened.
    prev_acts = {a.get("tool_use_id") or ("#%d" % i): a
                 for i, a in enumerate(prev.get("actions") or [])}
    new_acts = {a.get("tool_use_id") or ("#%d" % i): a
                for i, a in enumerate(new.get("actions") or [])}
    if prev_acts or new_acts:
        out_acts, order = {}, []
        for key in list(new_acts) + [k for k in prev_acts if k not in new_acts]:
            order.append(key)
            pa, na = prev_acts.get(key), new_acts.get(key)
            if pa is None:
                out_acts[key] = na
                continue
            if na is None:
                out_acts[key] = dict(pa, retained_from_earlier_capture=True)
                continue
            act = dict(na)
            pr, nr = (pa.get("result") or {}), (act.get("result") or {})
            pres, nres = pr.get("status"), nr.get("status")
            # A recorded ok/error result is never replaced by "missing" -- and a
            # result that KEEPS its status must not silently shrink either: a
            # shorter re-read of the same result is not new information.
            if nres == "missing" and pres in ("ok", "error"):
                act["result"] = pr
                act["retained_from_earlier_capture"] = True
            elif len(pr.get("preview") or "") > len(nr.get("preview") or ""):
                act["result"] = pr
                act["retained_from_earlier_capture"] = True
            if len(pa.get("args_preview") or "") > len(act.get("args_preview") or ""):
                act["args_preview"] = pa.get("args_preview")
                act["args_truncated"] = pa.get("args_truncated")
                act["args_bytes_total"] = pa.get("args_bytes_total")
            out_acts[key] = act
        merged["actions"] = [out_acts[k] for k in order]
    # Inputs reconcile PER BLOCK by source identity. Aggregate byte totals let
    # one block's growth mask another block's loss, which is exactly the defect
    # this replaces.
    p_in, n_in = prev.get("input") or {}, new.get("input") or {}
    if _rc is not None and (p_in.get("blocks") or n_in.get("blocks")):
        cid = new.get("call_id")
        store, seen, order = _rc.Reconciled(), set(), []
        for i, blk in enumerate(n_in.get("blocks") or []):
            key = _rc.block_key(cid, blk, i)
            seen.add(key)
            if key not in store.items:
                order.append(key)
            store.absorb(key, blk, merge=_rc.merge_block)
        for i, blk in enumerate(p_in.get("blocks") or []):
            key = _rc.block_key(cid, blk, i)
            if key in store.items:
                store.absorb(key, blk, merge=_rc.merge_block)
            else:
                order.append(key)
                store.absorb(key, blk, seen_now=False)
        retained = store.retain_missing(seen)
        usable = dict(store.usable())
        blocks = [usable[k] for k in order if k in usable]
        base = dict(n_in if n_in.get("blocks") else p_in)
        base["blocks"] = blocks
        merged["input"] = base
        if retained or any(b.get("retained_from_earlier_capture") for b in blocks):
            merged["retained_input_from_earlier_capture"] = True
    pr, nr = prev.get("reasoning") or {}, new.get("reasoning") or {}
    if len(pr.get("text") or "") > len(nr.get("text") or "") or \
            (pr.get("blocks") or 0) > (nr.get("blocks") or 0):
        merged["reasoning"] = pr
    pu, nu = prev.get("usage") or {}, new.get("usage") or {}
    if int(pu.get("output_tokens") or 0) > int(nu.get("output_tokens") or 0):
        merged["usage"] = pu
        merged["usage_known"] = prev.get("usage_known", True)
        merged["cost_usd"] = prev.get("cost_usd")
        merged["cost_breakdown"] = prev.get("cost_breakdown")
    # NOTE: no overall-score override here. A single lexicographic comparison
    # would discard this field-wise reconciliation whenever one dimension (e.g.
    # longer text) improved while another (e.g. a tool action) regressed.
    if (len(merged.get("actions") or []) > len(new.get("actions") or [])
            or merged.get("retained_input_from_earlier_capture")
            or any(a.get("retained_from_earlier_capture")
                   for a in (merged.get("actions") or []))
            or _richness(prev) > _richness(new)):
        merged["capture_note"] = ("Fields retained from an earlier capture of this "
                                  "call; a later read of the source omitted them.")
    # Recompute the observable kind from what actually survived the merge.
    has_text = bool((merged.get("output_text") or "").strip())
    acts = merged.get("actions") or []
    merged["output_kind"] = ("mixed" if (has_text and acts) else
                             "actions" if acts else
                             "text" if has_text else "incomplete")
    return merged


def merge_traces(previous, current):
    """Fold a freshly-read trace onto what was already captured.

    A polling collector re-reads the run's files every pass, so a source that
    shrinks, rotates or is pruned between passes reads as a SUCCESSFUL EMPTY
    read -- and would otherwise overwrite calls that were captured while the
    source still existed. That would defeat the whole point of tracking during
    the run and rendering afterwards.

    So retention is by stable identity: agents by ``agent_id`` and calls by
    ``call_id``. For each agent the richer capture wins, and an agent whose
    transcript has disappeared keeps its previously captured calls, flagged so
    the report can say the source is gone rather than pretending it was empty.
    """
    if not previous or not isinstance(previous, dict):
        return current
    # Identity guard: persisted state is only this run's history if it IS this
    # run. Merging by path alone would import an unrelated run's agents and calls.
    prev_id = (previous.get("run") or {}).get("run_id")
    cur_id = (current.get("run") or {}).get("run_id")
    if prev_id and cur_id and prev_id != cur_id:
        current.setdefault("warnings", []).append(
            "Ignored a previous trace at this output path: it belongs to run %s, "
            "not %s. Retention applies only within a run." % (prev_id, cur_id))
        return current
    prev_agents = {a.get("agent_id"): a for a in (previous.get("agents") or [])}
    # NOTE: no early return when there are no previous AGENTS. A previous trace
    # can still carry edges and linkage that must be reconciled, and bailing out
    # here silently dropped them.

    retained_agents, retained_calls = 0, 0
    for agent in (current.get("agents") or []) if prev_agents else []:
        old = prev_agents.get(agent.get("agent_id"))
        if not old:
            continue
        old_calls = old.get("calls") or []
        new_calls = agent.get("calls") or []
        if not old_calls:
            continue
        by_id = {c.get("call_id"): c for c in old_calls}
        downgraded = 0
        for call in new_calls:
            cid = call.get("call_id")
            before = by_id.get(cid)
            merged_call = _merge_call(before, call)
            # ALWAYS install the reconciled call -- never gate it on a score.
            by_id[cid] = merged_call
            if before is not None and merged_call.get("capture_note"):
                downgraded += 1
        # ALWAYS install the reconciled calls. Gating installation on a count
        # or a note is the same defect as gating on a score: a reconciliation
        # that preserved a block or a tool result would be silently discarded.
        merged_calls = sorted(by_id.values(),
                              key=lambda c: (c.get("ts_ms") is None, c.get("ts_ms") or 0))
        for i, call in enumerate(merged_calls):
            call["index"] = i
        agent["calls"] = merged_calls
        agent["totals"] = _totals_of(merged_calls)
        ts = [c.get("ts_ms") for c in merged_calls if c.get("ts_ms") is not None]
        if ts:
            agent["first_ts_ms"] = min(ts)
            agent["last_ts_ms"] = max(ts)
        gained = len(by_id) - len(new_calls)
        if gained or downgraded:
            retained_calls += gained + downgraded
            retained_agents += 1
            agent["capture_note"] = (
                "Includes %d call(s) retained and %d call(s) whose earlier capture "
                "supplied fields a later read omitted." % (gained, downgraded))
            if agent.get("transcript_status") == "missing":
                agent["transcript_status"] = "missing_source_retained_capture"

    if retained_agents:
        current.setdefault("warnings", []).append(
            "Source degradation detected: %d call(s) across %d agent(s) are retained "
            "from earlier passes because their transcripts shrank or disappeared. "
            "This is retained history, not a fresh read."
            % (retained_calls, retained_agents))
        current["run"]["capture_retention"] = {
            "retained_calls": retained_calls, "retained_agents": retained_agents}
    # Never let a later pass report fewer agents than were already observed.
    known = {a.get("agent_id") for a in current.get("agents") or []}
    lost = [a for aid, a in prev_agents.items() if aid not in known]
    if lost:
        current["agents"] = (current.get("agents") or []) + lost
        current["agents"].sort(key=lambda a: (a.get("ordinal") is None, a.get("ordinal") or 0))
        current.setdefault("warnings", []).append(
            "%d agent(s) previously observed are absent from the current read and "
            "were retained from earlier passes." % len(lost))

    # Journal SHRINK (losing only a result event) must not strand the agent in
    # pending while its proven return edge survives -- the graph and the agent
    # details would then contradict each other.
    restored_returns = 0
    for agent in current.get("agents") or []:
        old = prev_agents.get(agent.get("agent_id"))
        if not old:
            continue
        if (old.get("result_status") == "returned_to_workflow"
                and agent.get("result_status") != "returned_to_workflow"):
            for field in ("result", "result_preview", "result_truncated",
                          "result_bytes_total", "result_status", "status"):
                if field in old:
                    agent[field] = old[field]
            agent["capture_note"] = (
                (agent.get("capture_note") or "")
                + " Return event retained from an earlier capture; the journal has "
                  "since shrunk.").strip()
            restored_returns += 1
    if restored_returns:
        current.setdefault("warnings", []).append(
            "%d agent return event(s) retained from earlier passes because the "
            "journal shrank; the graph and agent details are reconciled from the "
            "retained capture." % restored_returns)

    # Journal loss must not silently drop the proven graph or zero the headers
    # while retained agents are still being reported.
    # Edge identity includes the EVENT id: two distinct transfers between the
    # same pair of agents are different edges, and keying on endpoints alone
    # collapses them when the source loses one.
    def _ekey(e):
        return (e.get("type"), e.get("from"), e.get("to"),
                e.get("event_id") or e.get("spawn_event_id"))
    cur_edges = {_ekey(e): e for e in (current.get("edges") or [])}
    # Identities the CURRENT read reports as contradicted must stay unusable.
    linkage_now = (current.get("run") or {}).get("linkage") or {}
    invalidated = set()
    for item in (linkage_now.get("invalidated") or []):
        if isinstance(item, (list, tuple)):
            invalidated.add(tuple(item))
    added = 0
    for edge in previous.get("edges") or []:
        key = _ekey(edge)
        prior_here = cur_edges.get(key)
        if key in invalidated:
            # This relationship has been contradicted by later evidence. It does
            # not become true again just because an older capture still asserts
            # it; the disputed record is kept for inspection, not as proof.
            continue
        if prior_here is None:
            cur_edges[key] = edge
            added += 1
        else:
            # Attempts reconcile individually: preserving them only when the new
            # list is EMPTY discards history on a partial re-read.
            if _rc is not None and (edge.get("attempts") or prior_here.get("attempts")):
                atts, outcome, _summary = _rc.reconcile_attempts(
                    edge.get("attempts"), prior_here.get("attempts"),
                    prior_here.get("spawn_event_id") or key[3])
                if len(atts) > len(prior_here.get("attempts") or []):
                    prior_here["retained_from_earlier_capture"] = True
                    added += 1
                prior_here["attempts"] = atts
                prior_here["return_status"] = outcome
    if added:
        current["edges"] = list(cur_edges.values())
        current.setdefault("warnings", []).append(
            "%d proven edge(s) retained from earlier passes; the journal has since "
            "shrunk or been removed." % added)

    # Recompute run counters/timing from the RECONCILED capture, so the header
    # cannot disagree with the agents and edges actually being shown.
    agents_now = current.get("agents") or []
    run = current.setdefault("run", {})
    run["agents_started"] = len(agents_now)
    run["agents_returned"] = sum(
        1 for a in agents_now if a.get("result_status") == "returned_to_workflow")
    firsts = [a.get("first_ts_ms") for a in agents_now if a.get("first_ts_ms") is not None]
    lasts = [a.get("last_ts_ms") for a in agents_now if a.get("last_ts_ms") is not None]
    if firsts:
        run["origin_ts_ms"] = min(firsts)
    if lasts:
        run["end_ts_ms"] = max(lasts)
    if firsts and lasts:
        run["elapsed_ms_est"] = max(lasts) - min(firsts)
    return current


def _totals_of(calls):
    return {
        "calls": len(calls),
        "input_tokens": sum(c["usage"]["input_tokens"] for c in calls),
        "cache_read_input_tokens": sum(c["usage"]["cache_read_input_tokens"] for c in calls),
        "cache_creation_input_tokens": sum(c["usage"]["cache_creation_input_tokens"] for c in calls),
        "output_tokens": sum(c["usage"]["output_tokens"] for c in calls),
        "cost_usd": sum(c.get("cost_usd") or 0.0 for c in calls),
        "actions": sum(len(c.get("actions") or []) for c in calls),
        "usage_unknown_calls": sum(1 for c in calls if not c.get("usage_known", True)),
    }


def load_trace(path):
    """Read a previously published trace, or None."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def collect_once(workflow_dir, out_path, run_status=None, rates_path=None,
                 status_path=None, previous=None, resolver_info=None,
                 mirror_dir=None):
    trace = build_trace(workflow_dir, run_status=run_status, rates_path=rates_path,
                        resolver_info=resolver_info)
    # Fold onto whatever is already on disk (or held in memory) so a pruned
    # source cannot erase captured history.
    trace = merge_traces(previous if previous is not None else load_trace(out_path),
                         trace)
    if mirror_dir:
        # Durability: copy the run-owned sources beside the trace every pass, so
        # the report can be rebuilt after the container-local originals go away.
        manifest = mirror_sources(workflow_dir, mirror_dir)
        trace["run"]["mirror"] = {
            "dest": manifest.get("dest"), "complete": manifest.get("complete"),
            "bytes_copied": manifest.get("bytes_copied"),
            "retained": manifest.get("retained"),
            "skipped_budget": manifest.get("skipped_budget"),
            "errors": manifest.get("errors"),
        }
        if manifest.get("retained"):
            trace.setdefault("warnings", []).append(
                "%d mirrored source file(s) are retained copies: the original "
                "shrank or disappeared, so the mirror is now the only record."
                % manifest["retained"])
        if not manifest.get("complete"):
            trace.setdefault("warnings", []).append(
                "Source mirror is INCOMPLETE (%d skipped for budget, %d errors); "
                "a rebuild from the mirror may not cover the whole run."
                % (manifest.get("skipped_budget", 0), manifest.get("errors", 0)))
    write_trace(trace, out_path)
    if status_path:
        write_status(status_path, trace["run"]["status"],
                     trace["run"].get("status_reason"),
                     run_id=trace["run"].get("run_id"),
                     agents=trace["run"].get("agents_started"))
    return trace


def write_status(status_path, state, reason=None, **extra):
    """Publish a small, durable observer-status file.

    The launch hooks run the observer detached with stdio discarded, so this is
    the only way a resolution failure, a crash or a timeout stays visible to a
    human instead of vanishing.
    """
    doc = {"state": state, "reason": reason, "ts_ms": int(time.time() * 1000)}
    doc.update({k: v for k, v in extra.items() if v is not None})
    try:
        parent = os.path.dirname(os.path.abspath(status_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = "%s.tmp.%d" % (status_path, os.getpid())
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, status_path)
    except Exception:
        pass
    return doc


#: Ceiling on bytes copied per mirror pass. Agent transcripts for a long run
#: reach hundreds of megabytes; the budget stops a runaway transcript filling the
#: run's output volume. Override with ``GEAK_TRACE_MIRROR_MAX_MB``.
DEFAULT_MIRROR_MAX_BYTES = 4 * 1024 * 1024 * 1024

#: Bytes compared at the append boundary to confirm a file really is the same
#: file, grown, rather than a different file that happens to be longer.
_APPEND_PROBE = 4096

#: The run-owned artifacts. Everything the trace is built from, so a rebuild can
#: run entirely off the mirror once the container-local originals are gone.
_MIRROR_PATTERNS = ("journal.jsonl", "agent-*.jsonl", "agent-*.meta.json",
                    "linkage_events.jsonl", "agent_timeline.json")


def _sha256_of(path, length=None):
    """SHA-256 of a file, or of its first ``length`` bytes. None if unreadable."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            remaining = length
            while True:
                chunk = fh.read(1024 * 1024 if remaining is None
                                else min(1024 * 1024, remaining))
                if not chunk:
                    break
                h.update(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break
        if length is not None and remaining and remaining > 0:
            return None  # source shorter than requested prefix
        return h.hexdigest()
    except OSError:
        return None


def _next_generation(dest):
    """Path for the next retained generation of ``dest``.

    Each superseded version gets its OWN name. Reusing one ``.superseded`` path
    destroys the previous retained generation on the next rewrite, which loses
    exactly the history the mirror exists to keep.
    """
    n = 1
    while os.path.exists("%s.gen%d" % (dest, n)):
        n += 1
    return "%s.gen%d" % (dest, n)


def _mirror_one(src, dest, max_bytes_left):
    """Copy one source file durably, incrementally, and VERIFIED BY CONTENT.

    Append eligibility is decided by hashing the whole already-mirrored prefix
    against the source, never by a sampled boundary or by equal length: a change
    earlier in the file, or a same-size replacement, must not be reported as
    unchanged. A source that shrank or vanished never truncates the mirror, and
    a rewrite retains the previous content as its own numbered generation.

    Returns ``(action, bytes_copied, info)``.
    """
    try:
        src_size = os.path.getsize(src)
    except OSError:
        return (("source_missing_mirror_retained" if os.path.exists(dest)
                 else "source_missing"), 0, {})
    dest_size = os.path.getsize(dest) if os.path.exists(dest) else None

    if dest_size is None:
        if src_size > max_bytes_left:
            return "skipped_budget", 0, {"needed": src_size}
        with open(src, "rb") as fh_in, open(dest + ".part", "wb") as fh_out:
            shutil.copyfileobj(fh_in, fh_out)
        os.replace(dest + ".part", dest)
        return "copied", src_size, {"sha256": _sha256_of(dest)}

    dest_hash = _sha256_of(dest)

    if src_size == dest_size:
        # Equal length is NOT equal content: a same-size replacement would be
        # invisible. Compare the bytes.
        if _sha256_of(src) == dest_hash:
            return "unchanged", 0, {"sha256": dest_hash}
        if src_size > max_bytes_left:
            return "skipped_budget", 0, {"needed": src_size}
        gen = _next_generation(dest)
        shutil.copy2(dest, gen)
        with open(src, "rb") as a, open(dest + ".part", "wb") as b:
            shutil.copyfileobj(a, b)
        os.replace(dest + ".part", dest)
        return ("rewritten_same_size_previous_kept", src_size,
                {"generation": os.path.basename(gen), "sha256": _sha256_of(dest)})

    if src_size < dest_size:
        # Pruned, rotated or truncated at the source. Keep what we already have;
        # if the shorter source is NOT a prefix of it, retain it as a generation
        # too, because it is newly observed content we have not stored.
        if _sha256_of(src, src_size) != _sha256_of(dest, src_size):
            if src_size > max_bytes_left:
                return ("source_shrank_mirror_retained", 0,
                        {"divergent": True, "skipped_budget": True})
            gen = _next_generation(dest)
            with open(src, "rb") as a, open(gen, "wb") as b:
                shutil.copyfileobj(a, b)
            return ("source_shrank_divergent_both_kept", src_size,
                    {"generation": os.path.basename(gen)})
        return "source_shrank_mirror_retained", 0, {}

    # src_size > dest_size: only an append if the WHOLE mirrored prefix matches.
    if _sha256_of(src, dest_size) == dest_hash:
        delta = src_size - dest_size
        if delta > max_bytes_left:
            return "skipped_budget", 0, {"needed": delta}
        try:
            with open(src, "rb") as fh_in, open(dest, "r+b") as fh_out:
                fh_in.seek(dest_size)
                fh_out.seek(dest_size)
                shutil.copyfileobj(fh_in, fh_out)
        except OSError:
            return "error", 0, {}
        return "appended", delta, {"sha256": _sha256_of(dest)}

    # Longer, but the prefix differs: a rewrite, not an append. Budget must be
    # checked against the ACTUAL copy size (the whole replacement), not a delta.
    if src_size > max_bytes_left:
        return "skipped_budget", 0, {"needed": src_size}
    gen = _next_generation(dest)
    shutil.copy2(dest, gen)
    with open(src, "rb") as a, open(dest + ".part", "wb") as b:
        shutil.copyfileobj(a, b)
    os.replace(dest + ".part", dest)
    return ("rewritten_previous_kept", src_size,
            {"generation": os.path.basename(gen), "sha256": _sha256_of(dest)})


def mirror_sources(workflow_dir, dest_dir, max_bytes=None):
    """Mirror this run's own source artifacts to a durable directory.

    The runtime writes journal/meta/transcripts inside the Claude home, which is
    container-local and subject to pruning. Copying them beside the run's other
    artifacts is what makes the trace genuinely durable: the mirror keeps the
    same layout, so ``build_trace`` can be pointed straight at it to rebuild a
    report after the originals are gone.

    Incremental and restart-safe; returns a manifest describing exactly what was
    copied, retained or skipped, so coverage is auditable rather than assumed.
    """
    import glob as _glob
    if max_bytes is None:
        try:
            max_bytes = int(float(os.environ.get("GEAK_TRACE_MIRROR_MAX_MB",
                                                 "0")) * 1024 * 1024) or None
        except ValueError:
            max_bytes = None
    budget = max_bytes or DEFAULT_MIRROR_MAX_BYTES

    manifest = {"dest": os.path.abspath(dest_dir), "source": os.path.abspath(workflow_dir),
                "files": {}, "bytes_copied": 0, "retained": 0, "skipped_budget": 0,
                "errors": 0, "ts_ms": int(time.time() * 1000)}
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as exc:
        manifest["error"] = "cannot create mirror dir: %s" % exc
        return manifest

    names = []
    for pattern in _MIRROR_PATTERNS:
        names.extend(os.path.basename(p)
                     for p in _glob.glob(os.path.join(workflow_dir, pattern)))
    # Mirrored-but-now-absent files must still be reported, not silently dropped.
    try:
        names.extend(f for f in os.listdir(dest_dir)
                     if f.endswith((".jsonl", ".meta.json")))
    except OSError:
        pass

    names = [n for n in names if ".gen" not in n and not n.endswith(".part")]
    for name in sorted(set(names)):
        action, copied, extra = _mirror_one(os.path.join(workflow_dir, name),
                                            os.path.join(dest_dir, name),
                                            budget - manifest["bytes_copied"])
        entry = {"action": action, "bytes": copied}
        entry.update(extra or {})
        manifest["files"][name] = entry
        if entry.get("generation"):
            manifest.setdefault("generations", []).append(
                {"file": name, "kept_as": entry["generation"]})
        manifest["bytes_copied"] += copied
        if action.endswith("retained"):
            manifest["retained"] += 1
        elif action == "skipped_budget":
            manifest["skipped_budget"] += 1
        elif action == "error":
            manifest["errors"] += 1

    # The lifecycle record lives outside the workflow dir; copy it in so the
    # mirror is self-contained and a rebuild keeps the real run status.
    # A file-backed timeline lives outside the workflow dir; copy it in so a
    # rebuild keeps its phases when the originals are gone.
    tl, tl_src = read_agent_timeline(workflow_dir)
    if tl is not None and tl_src and tl_src != "run_record.result.llm_timeline":
        try:
            tmp = os.path.join(dest_dir, "agent_timeline.json.tmp.%d" % os.getpid())
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(tl, fh, ensure_ascii=False, indent=1, default=str)
            os.replace(tmp, os.path.join(dest_dir, "agent_timeline.json"))
            manifest["files"]["agent_timeline.json"] = {
                "action": "copied", "bytes": 0, "from": tl_src}
        except OSError:
            manifest["errors"] += 1

    record, record_path = read_run_record(workflow_dir)
    if record is not None:
        try:
            tmp = os.path.join(dest_dir, "run_record.json.tmp.%d" % os.getpid())
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, indent=1, default=str)
            os.replace(tmp, os.path.join(dest_dir, "run_record.json"))
            manifest["files"]["run_record.json"] = {"action": "copied", "bytes": 0,
                                                    "from": record_path}
        except OSError:
            manifest["errors"] += 1
    else:
        manifest["run_record"] = "absent"

    manifest["complete"] = (manifest["skipped_budget"] == 0 and manifest["errors"] == 0)
    try:
        tmp = os.path.join(dest_dir, "mirror_manifest.json.tmp.%d" % os.getpid())
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, os.path.join(dest_dir, "mirror_manifest.json"))
    except OSError:
        pass
    return manifest


def render_from_trace(trace, out_dir, basename=None):
    """Render the HTML/Markdown views FROM AN ALREADY-COLLECTED TRACE.

    The point of rendering from the trace rather than re-reading the run is that
    the trace is what was actually captured while the workflow was alive. By the
    time a report is produced the transcripts may have been rotated, pruned or
    moved, and the workflow directory may not resolve at all; the tracked JSON
    still renders completely.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import geak_trace_report as trace_report
    except Exception as exc:
        sys.stderr.write("geak_trace_collector: renderer unavailable: %s\n" % exc)
        return None
    try:
        return trace_report.write_reports(
            trace, out_dir, basename or "geak_execution_trace")
    except Exception as exc:
        sys.stderr.write("geak_trace_collector: render failed: %s\n" % exc)
        return None


def eval_dir_of_run(workflow_dir):
    """The run's own eval_dir, if its workflow record names one.

    Known only once the run reports it, which is why the final render -- not the
    launch -- is where the report directory can be filled in.
    """
    record, _ = read_run_record(workflow_dir)
    if not isinstance(record, dict):
        return None
    for holder in (record.get("result"), record.get("args")):
        if isinstance(holder, dict):
            value = holder.get("eval_dir")
            if isinstance(value, str) and value.strip():
                return value.strip().rstrip("/")
    return None


def publish_final(trace, workflow_dir, out_path, render=True):
    """Write the end-of-run views from the tracked data.

    Renders beside the tracked JSON, and additionally into ``<eval_dir>/report/``
    when the run record names one, so the report lands where a run's other
    artifacts already live. Both are rendered from the same captured trace.
    """
    if not render:
        return {}
    written = {}
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    run_id = (trace.get("run") or {}).get("run_id") or "run"
    # Per-run basename: the output dir is shared (an exp_root can hold several
    # runs), so a constant basename would have a later run silently overwrite an
    # earlier run's report -- including a partial run whose only report is here.
    beside = render_from_trace(trace, out_dir,
                               basename="geak_execution_trace_%s" % run_id)
    if beside:
        written["beside_trace"] = beside
    eval_dir = eval_dir_of_run(workflow_dir)
    if eval_dir and os.path.isdir(eval_dir):
        report_dir = os.path.join(eval_dir, "report")
        in_report = render_from_trace(trace, report_dir)
        if in_report:
            try:
                write_trace(trace, os.path.join(report_dir, "geak_trace.json"))
            except Exception:
                pass
            written["report_dir"] = in_report
    return written


def watch(workflow_dir, out_path, interval=20.0, max_seconds=None,
          rates_path=None, status_path=None, render=True, resolver_info=None,
          mirror_dir=None):
    """Poll until the RUN RECORD reports a terminal status, or the deadline hits.

    Each pass fully re-reads and atomically republishes, so a restart or a
    resumed run recovers without incremental state to corrupt, and the last
    valid snapshot survives a failed pass.

    Crucially the loop does NOT stop because every agent observed so far has
    returned: in a sequential workflow that is equally true in the gap before
    the next dispatch, and stopping there would abandon the rest of the run.
    A deadline or an error exits with a non-terminal status and a reason, never
    a fabricated ``complete``.
    """
    started_at = time.time()
    last, fails = load_trace(out_path), 0
    while True:
        try:
            last = collect_once(workflow_dir, out_path, rates_path=rates_path,
                                status_path=status_path, previous=last,
                                resolver_info=resolver_info, mirror_dir=mirror_dir)
            fails = 0
        except Exception as exc:  # never fatal to the run being observed
            fails += 1
            sys.stderr.write("geak_trace_collector: pass failed: %s\n" % exc)
            if status_path:
                write_status(status_path, "error",
                             "collection pass failed (%d consecutive): %s" % (fails, exc))
        if last and last["run"]["status"] == "complete":
            # End of run: build the views from what was tracked.
            published = publish_final(last, workflow_dir, out_path, render)
            if status_path:
                write_status(status_path, "complete",
                             last["run"].get("status_reason"),
                             rendered=bool(published) or None,
                             run_id=last["run"].get("run_id"))
            return last
        if max_seconds is not None and (time.time() - started_at) >= max_seconds:
            # Deadline, not completion. Say so rather than implying the run ended.
            if last is not None:
                last["run"]["status"] = "partial"
                last["run"]["status_reason"] = (
                    "observer deadline reached after %.0fs; the workflow may still "
                    "be running" % (time.time() - started_at))
                try:
                    write_trace(last, out_path)
                except Exception:
                    pass
            # Still render: a partial trace is the data that WAS captured, and a
            # report built from it is more useful than none.
            if last is not None:
                publish_final(last, workflow_dir, out_path, render)
            if status_path:
                write_status(status_path, "partial",
                             "observer deadline reached; run not known to be complete")
            return last
        time.sleep(interval)


def args_fingerprint(args):
    """Stable fingerprint of a workflow's invocation arguments.

    This is the identity join the runtime actually supplies: the ``wf_*.json``
    record carries the SAME ``args`` object the launch hook was invoked with
    (kernel_path, gpu_ids, budget, deadline_epoch, ...), so matching it is a
    property of the recorded data rather than a guess about timing. Two distinct
    launches differ in at least ``deadline_epoch``; two launches that are
    byte-identical in every argument are genuinely indistinguishable and are
    reported as ambiguous rather than picked.
    """
    try:
        return hashlib.sha256(
            json.dumps(args, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")).hexdigest()
    except Exception:
        return None


def _record_start_ms(record):
    """Epoch-ms this run started, from whichever field the record carries."""
    for key in ("startTime", "timestamp"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return int(value if value > 1e11 else value * 1000)
        if isinstance(value, str) and value:
            got = _iso_to_ms(value)
            if got is not None:
                return got
    return None


def resolve_workflow_dir(exp_root=None, eval_dir=None, script_dir=None,
                         session_id=None, require_live=False, run_id=None,
                         allow_ambiguous=False, prospective=False,
                         identity_args=None):
    """Find the run's ``subagents/workflows/<runId>/`` directory at launch.

    The Claude Code runtime writes a ``wf_*.json`` record whose ``args`` block
    (``exp_root``, ``workflow_dir``, ``kernel_path``) is populated from the
    moment the workflow starts -- before any ``eval_dir`` exists. That makes it
    the correct launch-time key: match on a directory the caller already knows,
    take the record's ``runId``, and derive the transcript directory from the
    record's own location. No substring matching is used; a directory matches
    only if it is equal to, or a parent of, a directory the record names.

    Returns ``(workflow_dir, info)``; ``workflow_dir`` is None when the record
    is not on disk yet, which is a normal state in the first seconds of a run.
    """
    info = {"matched_on": None, "run_id": None, "session_id": None,
            "record": None, "candidates": 0}
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import claude_trace_mirror as mirror
    except Exception as exc:
        info["error"] = "claude_trace_mirror unavailable: %s" % exc
        return None, info

    if not (eval_dir or exp_root):
        info["error"] = "no eval_dir or exp_root given; ownership cannot be established"
        return None, info

    # OWNERSHIP, not ranking. ``find_record`` ranks candidates and will happily
    # return a sibling whose shared exp_root merely CONTAINS the requested
    # directory, which is how an unrelated run gets adopted when the requested
    # one has no record yet. ``_owns`` is the real contract: a record owns the
    # requested eval_dir only if ITS OWN eval_dir equals it (same field, exact),
    # and likewise for exp_root. Eligibility is filtered BEFORE selection, so a
    # newer ineligible record cannot hide an older valid one.
    try:
        homes = mirror.candidate_homes()
        records = list(mirror.iter_records(homes))
    except Exception as exc:
        info["error"] = "record scan failed: %s" % exc
        return None, info

    identity_fp = args_fingerprint(identity_args) if identity_args is not None else None
    info["identity_fingerprint"] = identity_fp
    eligible, skipped_terminal, near_misses = [], 0, []
    for rec_path, record in records:
        if not mirror._owns(record, eval_dir, exp_root):
            continue
        if session_id and rec_path.parent.parent.name != session_id:
            continue
        args = record.get("args") if isinstance(record.get("args"), dict) else {}
        if script_dir and args.get("workflow_dir"):
            if os.path.abspath(str(args["workflow_dir"])).rstrip("/") != \
                    os.path.abspath(script_dir).rstrip("/"):
                continue
        if run_id and record.get("runId") != run_id:
            continue
        if identity_fp is not None:
            rec_fp = args_fingerprint(record.get("args"))
            if rec_fp != identity_fp:
                near_misses.append(record.get("runId"))
                continue
        if require_live and str(record.get("status") or "").lower() in _TERMINAL_STATUSES:
            # A finished run is not this launch's run. Skipping it here (rather
            # than after picking a winner) is what lets a still-registering new
            # run be waited for instead of silently adopting the old one.
            skipped_terminal += 1
            continue
        if not record.get("runId"):
            continue
        eligible.append((rec_path, record))

    info["candidates"] = len(eligible)
    info["skipped_terminal"] = skipped_terminal
    info["args_mismatch_records"] = [r for r in near_misses if r][:10]
    if not eligible:
        info["error"] = (
            "no record OWNS this directory (same-field exact match)%s"
            % (("; %d owning record(s) skipped as already terminal" % skipped_terminal)
               if skipped_terminal else "")
            + (("; %d owning record(s) rejected on args fingerprint" % len(near_misses))
               if near_misses else ""))
        return None, info

    def _stamp(pair):
        rec = pair[1]
        return str(rec.get("timestamp") or rec.get("startTime") or "")

    # Invocation identity. A time window cannot establish it: any slack admits a
    # recent neighbour, observer start is not workflow start, and a record with
    # no usable timestamp would be admitted regardless -- so no time filter is
    # used at all. Nor is "the only owning record" a substitute: the sole record
    # under a directory may simply be an unrelated run. Prospective attachment
    # therefore requires a supported identity, or it stays unresolved.
    if prospective and not run_id and identity_fp is None:
        info["ambiguous"] = len(eligible)
        info["error"] = (
            "prospective attachment requires a supported invocation identity "
            "(--run-id, or --identity-args matching the run record's own args). "
            "%d owning record(s) found, but none is PROVEN to be this launch, so "
            "attachment is left unresolved rather than guessed." % len(eligible))
        info["integration_gap"] = (
            "The launch hook supplied no runtime invocation identity. Pass "
            "--identity-args with the workflow's own args object, or --run-id.")
        return None, info
    if prospective and not run_id and len(eligible) > 1:
        info["ambiguous"] = len(eligible)
        info["error"] = (
            "%d records match this identity; the launches are indistinguishable. "
            "Attachment left unresolved rather than guessed." % len(eligible))
        return None, info

    eligible.sort(key=_stamp, reverse=True)
    if len(eligible) > 1:
        info["ambiguous"] = len(eligible)
        if not allow_ambiguous:
            # Two concurrent runs both own this root. Picking the newer one would
            # silently track the wrong workflow; refusing is the honest answer.
            info["error"] = (
                "ambiguous: %d live records own this directory; cannot establish "
                "which one this launch is. Pass a narrower identity (eval_dir or "
                "session) to disambiguate." % len(eligible))
            return None, info
    rec_path, record = eligible[0]

    # Provenance keeps the field distinction: which requested field this record
    # owns, so an exp_root-owned attachment is not reported as an eval_dir identity.
    owned_fields = []
    if eval_dir and mirror._owns(record, eval_dir, None):
        owned_fields.append("eval_dir")
    if exp_root and mirror._owns(record, None, exp_root):
        owned_fields.append("exp_root")
    hit = "owns:" + ",".join(owned_fields or ["unknown"])
    if run_id:
        info["identity"] = "explicit-run-id"
    elif identity_fp is not None:
        info["identity"] = "args-fingerprint"
    else:
        info["identity"] = "retrospective"
    info["owned_fields"] = owned_fields
    info["record_status"] = record.get("status")

    run_id = record.get("runId")
    if not run_id:
        info["error"] = "matched record has no runId"
        return None, info
    session_dir = rec_path.parent.parent
    wf_dir = os.path.join(str(session_dir), "subagents", "workflows", str(run_id))
    info.update({"matched_on": hit, "run_id": run_id,
                 "session_id": session_dir.name, "record": str(rec_path)})
    return (wf_dir if os.path.isdir(wf_dir) else None), info


def _await_workflow_dir(exp_root, eval_dir, script_dir, timeout_s, interval,
                        require_live=True, run_id=None, prospective=True,
                        identity_args=None):
    """Poll for THIS launch's run record; a workflow takes a moment to register.

    ``require_live`` defaults True here because this is the prospective path: a
    tracker starting with a workflow must wait for that workflow's own record,
    never adopt an already-completed run that happens to share the directory.
    """
    deadline = time.time() + max(0.0, timeout_s)
    info = {}
    while True:
        wf_dir, info = resolve_workflow_dir(exp_root=exp_root, eval_dir=eval_dir,
                                            script_dir=script_dir,
                                            require_live=require_live,
                                            run_id=run_id, prospective=prospective,
                                            identity_args=identity_args)
        if wf_dir:
            return wf_dir, info
        if time.time() >= deadline:
            return None, info
        time.sleep(interval)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workflow-dir",
                    help="Directory holding journal.jsonl and agent-*.jsonl. "
                         "Omit to resolve it from --exp-root/--eval-dir.")
    ap.add_argument("--exp-root", help="Run exp_root, known at launch time")
    ap.add_argument("--eval-dir", help="Run eval_dir, once it exists")
    ap.add_argument("--script-dir", help="Workflow script dir, to disambiguate")
    ap.add_argument("--resolve-timeout", type=float, default=120.0,
                    help="Seconds to wait for the run record to appear")
    ap.add_argument("--out", help="Exact path to write the trace JSON")
    ap.add_argument("--out-dir",
                    help="Directory to write geak_trace_<runId>.json into. Preferred "
                         "over --out for live tracking: the filename is per-run, so "
                         "two runs under one exp_root cannot overwrite each other.")
    ap.add_argument("--rates", help="Rate table JSON, merged as llm_ledger --rates does")
    ap.add_argument("--prospective", dest="prospective", action="store_true", default=True,
                    help="Only attach to a run that started with this observer "
                         "(default; the launch-hook case)")
    ap.add_argument("--any-run", dest="prospective", action="store_false",
                    help="Attach to an existing run regardless of start time "
                         "(for reporting on a past run)")
    ap.add_argument("--mirror-dir",
                    help="Durably mirror this run's journal/meta/transcripts here "
                         "(default: <out-dir>/geak_trace_sources_<runId>)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false", default=True,
                    help="Do not mirror the run-owned sources")
    ap.add_argument("--identity-args",
                    help="JSON of the workflow's own invocation args, as the launch "
                         "hook received them. Matched against the run record's args "
                         "block -- a runtime-supplied join, not a timing guess.")
    ap.add_argument("--run-id",
                    help="Explicit workflow runId to attach to. This is the only "
                         "PROOF of invocation identity; without it attachment is "
                         "refused when more than one record owns the directory.")
    ap.add_argument("--render", dest="render", action="store_true", default=True,
                    help="Render HTML+Markdown from the tracked data when the run "
                         "ends (default)")
    ap.add_argument("--no-render", dest="render", action="store_false",
                    help="Track only; do not render")
    ap.add_argument("--watch", action="store_true",
                    help="Poll until the run completes instead of collecting once")
    ap.add_argument("--interval", type=float, default=20.0,
                    help="Seconds between polls when --watch (default 20)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Safety deadline for --watch")
    args = ap.parse_args(argv)
    if not (args.out or args.out_dir):
        ap.error("give --out or --out-dir")

    identity = None
    if args.identity_args:
        try:
            identity = json.loads(args.identity_args)
        except ValueError as exc:
            sys.stderr.write("geak_trace_collector: --identity-args is not JSON (%s); "
                             "prospective attachment will be unresolved\n" % exc)

    wf_dir, resolved_info = args.workflow_dir, None
    if not wf_dir:
        if not (args.exp_root or args.eval_dir):
            ap.error("give --workflow-dir, or --exp-root/--eval-dir to resolve it")
        # The observer starts with the workflow, so any run that began before it
        # (minus a small registration slack) is a neighbour, not this launch.
        # --any-run is retrospective: it must also lift require_live, or an
        # exactly-owned COMPLETED record is rejected and the flag does nothing.
        wf_dir, info = _await_workflow_dir(args.exp_root, args.eval_dir,
                                           args.script_dir,
                                           args.resolve_timeout, args.interval,
                                           require_live=args.prospective,
                                           run_id=args.run_id,
                                           prospective=args.prospective,
                                           identity_args=identity)
        if not wf_dir:
            # Not an error: a run that never registered leaves nothing to track,
            # and the observer must never fail the workflow it is observing. It
            # IS recorded durably, so the failure is visible despite stdio being
            # discarded by the launch hook.
            reason = info.get("error") or "run record not found before timeout"
            sys.stderr.write("geak_trace_collector: no workflow run resolved (%s)\n"
                             % reason)
            if args.out_dir:
                write_status(os.path.join(args.out_dir, "geak_trace.status.json"),
                             "unresolved", reason)
            elif args.out:
                write_status(args.out + ".status.json", "unresolved", reason)
            return 0
        resolved_info = info
        sys.stderr.write("geak_trace_collector: tracking run %s (matched %s)\n"
                         % (info.get("run_id"), info.get("matched_on")))

    run_id = os.path.basename(os.path.abspath(wf_dir))
    out_path = args.out or os.path.join(args.out_dir, "geak_trace_%s.json" % run_id)
    status_path = out_path + ".status.json"
    mirror_dir = None
    if args.mirror:
        mirror_dir = args.mirror_dir or os.path.join(
            os.path.dirname(os.path.abspath(out_path)),
            "geak_trace_sources_%s" % run_id)

    if args.watch:
        trace = watch(wf_dir, out_path, interval=args.interval,
                      max_seconds=args.max_seconds, rates_path=args.rates,
                      status_path=status_path, render=args.render,
                      resolver_info=resolved_info, mirror_dir=mirror_dir)
    else:
        trace = collect_once(wf_dir, out_path, rates_path=args.rates,
                             status_path=status_path, mirror_dir=mirror_dir)
        if args.render and trace:
            publish_final(trace, wf_dir, out_path, True)
    if trace:
        run = trace["run"]
        sys.stderr.write(
            "geak_trace_collector: %s agents=%d returned=%d status=%s -> %s\n"
            % (run["run_id"], run["agents_started"], run["agents_returned"],
               run["status"], out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
