#!/usr/bin/env python3
"""Per-API-call token + wall-clock ledger for a GEAK run.

GEAK measures what it achieves and nothing about what it spends. This closes that: one row
for every single LLM request the run made, however small, joined to the phase that made it,
plus how long each call and each phase took.

WHERE THE NUMBERS COME FROM
  Tokens are NOT estimated. Claude Code writes a transcript for every session and sub-agent
  under ~/.claude/projects/<slug>/*.jsonl, and every assistant record in it carries the exact
  usage the API billed:
      input_tokens                 text sent fresh, full price
      cache_creation_input_tokens  text stored so later calls can re-send it cheaply (~25% surcharge)
      cache_read_input_tokens      text re-sent that was already stored (one tenth price)
      output_tokens                text generated
  One assistant record == one API call. Two details bite and are handled: the same response is
  sometimes written twice with an identical message.id (deduplicated here), and cache_creation
  splits into 5-minute and 1-hour buckets that are priced differently.

  Per-call duration is the gap between an assistant record and the record before it in the same
  conversation — how long that request took to come back.

  Phase comes from the workflow itself. The JS workflows record a timeline (which agent ran, in
  which phase, when, for how long) and persist it to reports/trace/agent_timeline.json. That is
  the authoritative source, because prompt text alone is ambiguous: `bakeoff` runs in both
  HeadKernel and Milestone, and `setup` runs in both Setup and Benchmark. With no timeline the
  ledger still builds, attributing by prompt text and time window, and marks itself "inferred".

USAGE
  python3 llm_ledger.py --eval-dir <dir>                     # normal: called at end of a run
  python3 llm_ledger.py --eval-dir <dir> --transcripts 'X/*.jsonl'   # explicit transcripts
  python3 llm_ledger.py --eval-dir <dir> --rates my_rates.json       # override prices

OUTPUT (all under <eval_dir>/reports/trace/)
  llm_calls.jsonl    one row per API call — the raw ledger
  agent_calls.jsonl  one row per agent invocation, with attempts and outcome
  token_stats.json   every table below, machine-readable
  token_stats.md     the same tables, rendered

NEVER RAISES on bad input. A missing or unreadable transcript produces a ledger marked
incomplete with a stated reason — this runs at the end of a multi-hour GPU run and must not be
able to fail it.

Stdlib only.
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict
from datetime import datetime, timezone

SCHEMA = "geak.llm_ledger/1"

# --------------------------------------------------------------------------- #
# Prices, in dollars per million tokens. ONE home for every rate, so a wrong
# price is a one-line correction that never touches the arithmetic below.
# Basis: the $5 in / $25 out list price used throughout PROJECTS/research, with
# the published cache multipliers (read 0.1x, 5-minute write 1.25x, 1-hour 2x).
# Override wholesale with --rates <file.json>, or per-model by adding a key.
# Every table also reports raw tokens, so the dollar columns can be ignored
# entirely if the rate is wrong for your contract.
# --------------------------------------------------------------------------- #
DEFAULT_RATES = {
    "_default": {
        "input": 5.00,
        "output": 25.00,
        "cache_read": 0.50,
        "cache_write_5m": 6.25,
        "cache_write_1h": 10.00,
    },
}

# Every role prompt in both workflows opens with this exact line (see roleAgent()
# in e2e_workflow.js / kernel_lane.js / kernel_workflow.js), which is what lets a
# transcript be tied back to the agent that produced it.
ROLE_RE = re.compile(r"You are the ([A-Za-z0-9_.\-]+)\.\s*PHASE=([A-Za-z0-9_.\-]+)\.")
# A couple of one-off agents (the workflow_return persister) are plain file writers with no role
# header. Without this they would not start a new conversation, and their tokens would be silently
# added to whichever agent happened to run before them — a small error, but a wrong one, and it
# would land on a different agent every run.
BARE_RE = re.compile(r"You are a file writer\.")

UNATTRIBUTED = "(unattributed)"
DRIVER = "(driver)"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _iso_to_ms(s):
    """'2026-08-10T16:31:37.388Z' -> epoch ms. None on anything unparseable."""
    if not s or not isinstance(s, str):
        return None
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _ms_to_iso(ms):
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _hms(ms):
    """Duration as 0h00m — the unit a reader of a 14-hour run actually wants."""
    if not ms or ms < 0:
        return "0h00m"
    s = int(ms // 1000)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


def _secs(ms):
    if ms is None:
        return ""
    return "%.1fs" % (ms / 1000.0) if ms < 60000 else "%dm%02ds" % (ms // 60000, (ms % 60000) // 1000)


def _pctile(values, p):
    """Nearest-rank percentile. No numpy in this tree."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, int(round((p / 100.0) * len(vals) + 0.5)) - 1))
    return vals[k]


def _n(x):
    return "{:,}".format(int(x or 0))


def _text_of(message):
    """Message content is either a plain string or a list of typed blocks."""
    if message is None:
        return ""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                out.append(b["text"])
        return "\n".join(out)
    return ""


def read_jsonl(path):
    """Yield objects from a JSONL file, skipping anything unparseable.

    Deliberately forgiving: a transcript being written concurrently can end in a
    half-line, and one bad line must not cost us the other thousand.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


# --------------------------------------------------------------------------- #
# Transcript discovery
# --------------------------------------------------------------------------- #
def transcript_roots():
    """Where Claude Code keeps transcripts. CLAUDE_CONFIG_DIR wins when set."""
    roots = []
    env = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    if env:
        roots.append(env)
    roots.append(os.path.expanduser("~/.claude"))
    seen, out = set(), []
    for r in roots:
        r = os.path.abspath(r)
        if r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def _mentions(path, needle, limit_bytes=64 * 1024 * 1024):
    """Does this file mention the needle? Chunked so a huge transcript is cheap.

    Overlap between chunks is the needle length minus one, so a match that
    straddles a chunk boundary is still found.
    """
    if not needle:
        return True
    nb = needle.encode("utf-8", "replace")
    keep = len(nb) - 1
    try:
        with open(path, "rb") as fh:
            prev = b""
            read = 0
            while read < limit_bytes:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    return False
                read += len(chunk)
                if nb in (prev + chunk):
                    return True
                prev = chunk[-keep:] if keep > 0 else b""
    except OSError:
        return False
    return False


def discover_transcripts(eval_dir, explicit_globs=None, roots=None):
    """Find the transcripts belonging to this run.

    The filter is the eval dir string. Every role prompt carries EVAL_DIR, so a
    transcript that mentions it belongs to this run — and because the kernel
    layer's eval dirs live UNDER the e2e one (<eval_dir>/kernels/_exp/team_*),
    the same substring catches the nested runs too. That makes discovery
    independent of how the CLI happens to lay sub-agent transcripts out, which
    is an internal detail we should not depend on.
    """
    if explicit_globs:
        # The caller named the files, so take them as given — and take NOTHING
        # else. Falling back to a full scan when a named glob happens to match
        # nothing would quietly pull in unrelated sessions and bill them to this
        # run, which is worse than reporting an empty ledger.
        paths = []
        for g in explicit_globs:
            paths.extend(glob.glob(g, recursive=True))
        return sorted({os.path.abspath(p) for p in paths if os.path.isfile(p)})

    paths = []
    for root in (roots if roots is not None else transcript_roots()):
        paths.extend(glob.glob(os.path.join(root, "projects", "**", "*.jsonl"), recursive=True))
    uniq = sorted({os.path.abspath(p) for p in paths if os.path.isfile(p)})
    return [p for p in uniq if _mentions(p, eval_dir)]


# --------------------------------------------------------------------------- #
# Transcript -> conversation groups -> API-call rows
# --------------------------------------------------------------------------- #
def split_conversations(records):
    """Split a transcript's records into conversations, one per agent.

    A conversation starts at a user record whose text is a role prompt ("You are
    the X. PHASE=Y."). Anything before the first such record is the driver turn —
    the top-level session that invoked the Workflow tool.

    Splitting on the prompt rather than on file boundaries or the isSidechain
    flag keeps this working whether the CLI writes each sub-agent to its own file
    or inlines them into the parent transcript.
    """
    groups, cur = [], {"role": DRIVER, "subphase": "", "records": [], "prompt": ""}
    for rec in records:
        if rec.get("type") == "user":
            text = _text_of(rec.get("message"))
            m = ROLE_RE.search(text)
            role, sub = (m.group(1), m.group(2)) if m else (
                ("file_writer", "persist") if BARE_RE.search(text) else (None, None))
            if role:
                if cur["records"]:
                    groups.append(cur)
                # Keep a slice of the prompt: it is what tells an e2e `director`
                # apart from a kernel-layer `director` (the latter's paths sit
                # under <eval>/kernels/_exp/).
                cur = {"role": role, "subphase": sub, "records": [], "prompt": text[:8000]}
        cur["records"].append(rec)
    if cur["records"]:
        groups.append(cur)
    return [g for g in groups if any(r.get("type") == "assistant" for r in g["records"])]


def calls_of(group, source):
    """One row per API call in this conversation, deduplicated.

    Dedupe key is message.id: the same response is sometimes flushed to the
    transcript twice. We keep the copy with the largest output_tokens (a partial
    flush can only undercount) and the EARLIEST timestamp (when it landed).
    """
    by_id, order, prev_ts = {}, [], None
    for rec in group["records"]:
        ts = _iso_to_ms(rec.get("timestamp"))
        if rec.get("type") != "assistant":
            if ts is not None:
                prev_ts = ts
            continue
        msg = rec.get("message") or {}
        usage = msg.get("usage") or {}
        key = msg.get("id") or rec.get("requestId") or rec.get("uuid")
        cache = usage.get("cache_creation") or {}
        row = {
            "ts_ms": ts,
            "duration_ms": (ts - prev_ts) if (ts is not None and prev_ts is not None and ts >= prev_ts) else None,
            "message_id": msg.get("id"),
            "request_id": rec.get("requestId"),
            "model": msg.get("model"),
            "stop_reason": msg.get("stop_reason"),
            "service_tier": usage.get("service_tier"),
            "input_tokens": int(usage.get("input_tokens") or 0),
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
            "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
            "cache_write_5m_tokens": int(cache.get("ephemeral_5m_input_tokens") or 0),
            "cache_write_1h_tokens": int(cache.get("ephemeral_1h_input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "role": group["role"],
            "sub_phase": group["subphase"],
            "source": source,
        }
        # The split fields are advisory; when absent, everything stored counts as
        # a 5-minute write, which is the default TTL.
        if not row["cache_write_5m_tokens"] and not row["cache_write_1h_tokens"]:
            row["cache_write_5m_tokens"] = row["cache_creation_input_tokens"]
        prior = by_id.get(key)
        if prior is None:
            by_id[key] = row
            order.append(key)
        else:
            if row["output_tokens"] > prior["output_tokens"]:
                keep_ts, keep_dur = prior["ts_ms"], prior["duration_ms"]
                by_id[key] = row
                by_id[key]["ts_ms"], by_id[key]["duration_ms"] = keep_ts, keep_dur
        if ts is not None:
            prev_ts = ts
    return [by_id[k] for k in order]


def total_input(row):
    return (row["input_tokens"] + row["cache_read_input_tokens"]
            + row["cache_creation_input_tokens"])


def cost_of(row, rates):
    r = rates.get(row.get("model") or "", rates["_default"])
    return (row["input_tokens"] * r["input"]
            + row["cache_read_input_tokens"] * r["cache_read"]
            + row["cache_write_5m_tokens"] * r["cache_write_5m"]
            + row["cache_write_1h_tokens"] * r["cache_write_1h"]
            + row["output_tokens"] * r["output"]) / 1e6


def list_cost_of(row, rates):
    """What the same traffic would cost with no reuse discount at all.

    Every input token at the fresh rate. This is the honest counterfactual: the
    surcharge for storing text is only ever paid BECAUSE reuse is on, so pricing
    it into the comparison would overstate the saving.
    """
    r = rates.get(row.get("model") or "", rates["_default"])
    return (total_input(row) * r["input"] + row["output_tokens"] * r["output"]) / 1e6


# --------------------------------------------------------------------------- #
# Timeline (what the workflow recorded about itself)
# --------------------------------------------------------------------------- #
def load_timeline(eval_dir):
    """Read reports/trace/agent_timeline.json plus any nested kernel ones.

    The timeline carries NO timestamps by design — Date.now() is unavailable in
    workflow scripts — so it is a strictly ordered list of agent attempts and
    nothing more. Order plus label is all we need: the transcripts supply every
    time, and the timeline supplies the one thing they cannot know, the phase.

    The e2e workflow merges its nested kernel timelines before writing, because a
    nested workflow's return value comes straight back into the parent JS with no
    model in between. The glob is belt-and-braces for a kernel run launched alone.
    """
    events, sources = [], []
    candidates = [os.path.join(eval_dir, "reports", "trace", "agent_timeline.json")]
    candidates += sorted(glob.glob(os.path.join(
        eval_dir, "kernels", "_exp", "*", "reports", "trace", "agent_timeline.json")))
    seen_nodes = set()
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        sources.append(path)
        stack = [doc]
        while stack:
            node = stack.pop(0)
            if not isinstance(node, dict):
                continue
            stack.extend(n for n in (node.get("nested") or []) if isinstance(n, dict))
            wf = node.get("workflow") or "?"
            evs = node.get("events") or []
            # A nested timeline can be reachable both through the parent's merge
            # and through the glob; identify a node by its shape and skip repeats.
            fp = (wf, len(evs), evs[0].get("label") if evs else None,
                  evs[-1].get("label") if evs else None)
            if fp in seen_nodes:
                continue
            seen_nodes.add(fp)
            for e in evs:
                # role/sub_phase are the join key. The workflow records them from the
                # prompt; `label` is a free-form display string and is NOT parseable
                # into an identity ('architect:strategize' for role system_architect,
                # 'bakeoff <op name>', 'eng r1_d0:memory'). Older timelines predate the
                # fields, so fall back to the label's leading role:sub_phase if it
                # happens to look like one.
                role, sub = e.get("role") or "", e.get("sub_phase") or ""
                if not role:
                    bits = (e.get("label") or "").split(":")
                    if len(bits) >= 2 and " " not in bits[0]:
                        role, sub = bits[0], bits[1].split(" ")[0]
                events.append({
                    "workflow": wf,
                    "tree": "root" if wf == "e2e_workflow" else "kernel",
                    "phase": e.get("phase") or UNATTRIBUTED,
                    "label": e.get("label") or "agent",
                    "key": "%s:%s" % (role, sub) if sub else role,
                    "attempt": e.get("attempt") or 1,
                    "ok": bool(e.get("ok")),
                    "seq": e.get("seq"),
                })
    return {"events": events, "sources": sources}


def _conv_key(role, subphase):
    return "%s:%s" % (role, subphase) if subphase else role


def attribute(groups, timeline):
    """Tie each conversation to the agent attempt that launched it.

    A timeline label is `role:sub_phase[:suffix]` — exactly what the prompt gives
    us — so the label narrows a conversation to a handful of candidates. The
    remaining ambiguity (the same role running in two different phases, which is
    the whole reason the timeline exists) is resolved POSITIONALLY: within one
    label, the workflow's Nth recorded attempt is the Nth conversation in time
    order. That works without clocks, which the workflow does not have.

    Conversations are first split by which workflow they belong to. Most roles
    appear in only one of them; `director` appears in both, and is placed by
    whether the prompt refers to a kernel-layer eval dir.

    When a label has more recorded attempts than conversations — a retry that
    hung, or an API error before any response — the surplus attempts are left
    unmatched and reported separately with zero calls. Which attempt is left over
    follows from the positional rule and is not knowable from the data; what
    matters is that the attempt is not silently dropped.
    """
    events = timeline.get("events") or []
    if not events:
        for g in groups:
            g["label"] = _conv_key(g["role"], g["subphase"])
            if g["role"] == DRIVER:
                g["phase"], g["attribution"] = DRIVER, "driver"
            else:
                # No timeline: group by the agent's own identity rather than dumping it
                # in a nameless bucket. `~` marks "grouped by role, phase not recorded",
                # so it can never be mistaken for a real workflow phase name.
                g["phase"] = "~%s" % g["label"]
                g["attribution"] = "inferred"
        return "inferred"

    # label -> ordered attempts, per workflow
    by_wf = defaultdict(lambda: defaultdict(list))
    labels_of_wf = defaultdict(set)
    for e in events:
        key = e.get("key") or ""
        if not key:
            continue
        by_wf[e["workflow"]][key].append(e)
        labels_of_wf[e["workflow"]].add(key)

    ordered = sorted(groups, key=lambda g: (g["t0_ms"] is None, g["t0_ms"] or 0))
    cursor = defaultdict(int)
    for g in ordered:
        g["label"] = _conv_key(g["role"], g["subphase"])
        g["attribution"] = "inferred"
        if g["role"] == DRIVER:
            g["phase"], g["attribution"] = DRIVER, "driver"
            continue
        key = g["label"]
        owners = [wf for wf in by_wf if key in labels_of_wf[wf]]
        if len(owners) > 1:
            # Only `director` collides in practice. The kernel layer's eval dirs
            # live under <e2e eval>/kernels/_exp/, so the prompt says which it is.
            nested = "/kernels/_exp/" in (g.get("prompt") or "")
            pick = [w for w in owners if (w != "e2e_workflow") == nested]
            owners = pick or owners
        for wf in owners:
            slots = by_wf[wf].get(key) or []
            i = cursor[(wf, key)]
            if i < len(slots):
                e = slots[i]
                cursor[(wf, key)] = i + 1
                e["matched"] = True
                g["phase"] = e["phase"] if e["tree"] == "root" else "kernel/" + e["phase"]
                g["label"] = e["label"]
                g["workflow"] = e["workflow"]
                g["attempt"] = e["attempt"]
                g["attribution"] = "timeline"
                break
        else:
            # No timeline slot: group by the agent's own identity rather than dumping
            # it in a nameless bucket. `~` marks "grouped by role, phase not recorded"
            # so it can never be mistaken for a workflow phase name.
            g["phase"] = "~%s" % g["label"]
    return "timeline"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
TOKEN_FIELDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                "cache_write_5m_tokens", "cache_write_1h_tokens", "output_tokens")


def _blank():
    d = {f: 0 for f in TOKEN_FIELDS}
    d.update(calls=0, cost=0.0, list_cost=0.0, durations=[])
    return d


def _add(acc, row, rates):
    for f in TOKEN_FIELDS:
        acc[f] += row[f]
    acc["calls"] += 1
    acc["cost"] += cost_of(row, rates)
    acc["list_cost"] += list_cost_of(row, rates)
    if row["duration_ms"] is not None:
        acc["durations"].append(row["duration_ms"])


def _finish(acc):
    d = acc.pop("durations")
    acc["total_input"] = acc["input_tokens"] + acc["cache_read_input_tokens"] + acc["cache_creation_input_tokens"]
    acc["llm_ms"] = sum(d)
    acc["median_call_ms"] = _pctile(d, 50)
    acc["p95_call_ms"] = _pctile(d, 95)
    acc["slowest_call_ms"] = max(d) if d else None
    return acc


def aggregate(rows, groups, timeline, rates):
    by_phase, by_role, by_agent = defaultdict(_blank), defaultdict(_blank), defaultdict(_blank)
    total = _blank()
    for row in rows:
        _add(total, row, rates)
        _add(by_phase[row["phase"]], row, rates)
        _add(by_role[row["role"]], row, rates)
        _add(by_agent[(row["phase"], row["agent_label"])], row, rates)
    total = _finish(total)

    # Phase span: first to last API call attributed to that phase. Every clock in
    # this report comes from the transcripts, because the workflow has none.
    #
    # Spans OVERLAP by construction and that is not a bug: a nested kernel phase
    # (kernel/Optimize) runs INSIDE an e2e phase (Milestone), so both are ticking
    # at once. That is why the shares below are of the whole run rather than of
    # each other, and why nested phases are prefixed `kernel/`.
    # A call's timestamp is when the RESPONSE landed, so a span measured between
    # response times would start after the first request was already in flight and
    # could come out shorter than the time spent inside the model. Start each span
    # at the first request instead (response time minus that call's duration), so
    # "% of phase in LLM" can never exceed 100%.
    def _span(sel):
        starts = [r["ts_ms"] - (r["duration_ms"] or 0) for r in rows
                  if sel(r) and r["ts_ms"] is not None]
        ends = [r["ts_ms"] for r in rows if sel(r) and r["ts_ms"] is not None]
        if not ends:
            return None, None
        return min(starts), max(ends)

    for ph, acc in by_phase.items():
        _finish(acc)
        t0, t1 = _span(lambda r, ph=ph: r["phase"] == ph)
        acc["wall_ms"] = (t1 - t0) if t0 is not None else None
        acc["started_at"] = _ms_to_iso(t0)
    for acc in by_role.values():
        _finish(acc)
    for acc in by_agent.values():
        _finish(acc)

    run_t0, run_t1 = _span(lambda r: True)
    total["wall_ms"] = (run_t1 - run_t0) if (run_t0 is not None and run_t1 is not None) else None
    total["started_at"] = _ms_to_iso(run_t0)
    total["ended_at"] = _ms_to_iso(run_t1)
    # "agents" = how many agent attempts happened. The workflow's own count is
    # authoritative — it also sees the attempts that hung or errored without ever
    # producing a transcript, which conversations alone would miss entirely.
    events = timeline.get("events") or []
    total["agents"] = len(events) or len(groups)
    total["agent_attempts_failed"] = sum(1 for e in events if not e.get("ok"))
    total["conversations"] = len(groups)
    return {
        "total": total,
        "by_phase": {k: v for k, v in by_phase.items()},
        "by_role": {k: v for k, v in by_role.items()},
        "by_agent": {"%s\t%s" % k: v for k, v in by_agent.items()},
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _table(headers, aligns, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(":--" if a == "l" else "--:" for a in aligns) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _share(part, whole):
    return "%.1f%%" % (100.0 * part / whole) if whole else "—"


def render_md(agg, meta):
    t = agg["total"]
    saved = (1 - t["cost"] / t["list_cost"]) * 100 if t["list_cost"] else 0.0
    reuse = _share(t["cache_read_input_tokens"], t["total_input"])
    ratio = ("%d:1" % round(t["total_input"] / t["output_tokens"])) if t["output_tokens"] else "—"

    L = ["# LLM token + time ledger", ""]
    L.append("Every API call this run made, however small, with what it cost and how long it took.")
    L.append("")
    L.append("- run: `%s`" % meta.get("eval_dir", "?"))
    L.append("- window: %s → %s (%s)" % (t.get("started_at") or "?", t.get("ended_at") or "?", _hms(t.get("wall_ms"))))
    L.append("- phase attribution: **%s**%s" % (
        meta.get("attribution_mode", "?"),
        "" if meta.get("attribution_mode") == "timeline"
        else " — no workflow timeline was found, so phases are a best guess from prompt text and timing"))
    L.append("- transcripts read: %d" % len(meta.get("transcripts", [])))
    if meta.get("calls_excluded_outside_window"):
        L.append("- excluded: %d call(s) before this run began. Finding a transcript by its "
                 "eval-dir path is not the same as dating it — a session that launched the run "
                 "also holds whatever else it did that day."
                 % meta["calls_excluded_outside_window"])
    if meta.get("warnings"):
        L.append("- **incomplete**: " + "; ".join(meta["warnings"]))
    L.append("")

    L.append("## Run totals")
    L.append("")
    L += _table(
        ["API calls", "agents", "wall", "in (total)", "out", "in:out", "billed", "no-reuse", "saved", "re-sent cheaply"],
        ["r"] * 10,
        [[_n(t["calls"]), _n(t["agents"]), _hms(t.get("wall_ms")), _n(t["total_input"]), _n(t["output_tokens"]),
          ratio, "$%.2f" % t["cost"], "$%.2f" % t["list_cost"], "%.1f%%" % saved, reuse]])
    L.append("")
    L.append("*billed* is what the reuse discount actually cost; *no-reuse* is the same traffic with every "
             "input token at full price. *re-sent cheaply* is the share of input that was already stored — "
             "high is good, but it is not the same as efficient: it can also mean cheaply re-sending "
             "something that should not be sent at all.")
    L.append("")

    L.append("## Tokens by phase")
    L.append("")
    rows = []
    for ph, a in sorted(agg["by_phase"].items(), key=lambda kv: -kv[1]["cost"]):
        rows.append([ph, _n(a["calls"]), _n(a["input_tokens"]), _n(a["cache_read_input_tokens"]),
                     _n(a["cache_creation_input_tokens"]), _n(a["output_tokens"]),
                     "$%.2f" % a["cost"], _share(a["cost"], t["cost"])])
    L += _table(["phase", "calls", "fresh in", "re-sent", "stored", "out", "$", "% of $"],
                ["l", "r", "r", "r", "r", "r", "r", "r"], rows)
    L.append("")

    L.append("## Time by phase")
    L.append("")
    rows = []
    for ph, a in sorted(agg["by_phase"].items(), key=lambda kv: -(kv[1].get("wall_ms") or 0)):
        rows.append([ph, _hms(a.get("wall_ms")), _share(a.get("wall_ms") or 0, t.get("wall_ms") or 0),
                     _hms(a["llm_ms"]), _share(a["llm_ms"], a.get("wall_ms") or 0),
                     _secs(a["median_call_ms"]), _secs(a["p95_call_ms"]), _secs(a["slowest_call_ms"])])
    L += _table(["phase", "wall", "% wall", "in LLM", "% of phase in LLM", "median call", "p95 call", "slowest"],
                ["l", "r", "r", "r", "r", "r", "r", "r"], rows)
    L.append("")
    L.append("*wall* is elapsed time including the compiling and benchmarking between calls; *in LLM* is the "
             "part spent waiting on the model. A phase with a low *% of phase in LLM* is not token-bound, so "
             "token savings will barely move it.")
    L.append("")

    L.append("## By role")
    L.append("")
    rows = []
    for role, a in sorted(agg["by_role"].items(), key=lambda kv: -kv[1]["cost"]):
        rows.append([role, _n(a["calls"]), _n(a["total_input"]), _n(a["output_tokens"]),
                     "$%.2f" % a["cost"], _secs(a["median_call_ms"])])
    L += _table(["role", "calls", "in", "out", "$", "median call"], ["l", "r", "r", "r", "r", "r"], rows)
    L.append("")

    L.append("## Ten most expensive agents")
    L.append("")
    rows = []
    for key, a in sorted(agg["by_agent"].items(), key=lambda kv: -kv[1]["cost"])[:10]:
        phase, label = key.split("\t", 1)
        rows.append([label, phase, _n(a["calls"]), _n(a["total_input"]), _n(a["output_tokens"]), "$%.2f" % a["cost"]])
    L += _table(["agent", "phase", "calls", "in", "out", "$"], ["l", "l", "r", "r", "r", "r"], rows)
    L.append("")
    L.append("---")
    L.append("")
    L.append("Prices used, per million tokens: " + ", ".join(
        "%s $%.2f" % (k, v) for k, v in sorted(meta.get("rates", {}).get("_default", {}).items())) +
        ". Raw token counts are measured; the dollar columns are those counts times these rates.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_window(groups, since_ms=None, until_ms=None):
    """When did this run actually happen?

    Mentioning the eval dir is necessary to find a transcript but not sufficient
    to date it. The session that LAUNCHES a run keeps one long transcript, and if
    a human drove it interactively that transcript also holds every unrelated
    thing they did that day — all of it mentioning the eval-dir path. Counted
    naively it dwarfs the run (observed: 248 of 386 calls, and 95M of 115M input
    tokens, from before the run even started).

    The run starts at its FIRST ROLE AGENT. Everything GEAK does goes through
    roleAgent(), so the earliest such call is the earliest moment any of this
    could be GEAK's. Driver-session chatter before that point is somebody else's
    day. --since/--until override when a caller knows better.
    """
    if since_ms is None:
        starts = [g["t0_ms"] for g in groups
                  if g["role"] != DRIVER and g["t0_ms"] is not None]
        since_ms = min(starts) if starts else None
    return since_ms, until_ms


def build(eval_dir, explicit_globs=None, rates=None, roots=None,
          since_ms=None, until_ms=None):
    """Build the whole ledger. Returns (rows, agent_rows, agg, meta)."""
    rates = rates or DEFAULT_RATES
    warnings = []
    timeline = load_timeline(eval_dir)
    if not timeline["sources"]:
        warnings.append("no agent_timeline.json — phases inferred, not recorded")

    transcripts = discover_transcripts(eval_dir, explicit_globs, roots)
    if not transcripts:
        warnings.append("no transcripts found for this eval dir — token counts are empty")

    groups = []
    for path in transcripts:
        recs = list(read_jsonl(path))
        for g in split_conversations(recs):
            g["calls"] = calls_of(g, os.path.basename(path))
            if not g["calls"]:
                continue
            ts = [c["ts_ms"] for c in g["calls"] if c["ts_ms"] is not None]
            g["t0_ms"], g["t1_ms"] = (min(ts), max(ts)) if ts else (None, None)
            g["transcript"] = path
            groups.append(g)

    # Drop everything outside the run's own window BEFORE attributing, so a
    # launching session's unrelated history cannot be billed to this run.
    win_t0, win_t1 = run_window(groups, since_ms, until_ms)
    dropped = 0
    if win_t0 is not None or win_t1 is not None:
        for g in groups:
            keep = [c for c in g["calls"]
                    if c["ts_ms"] is None
                    or ((win_t0 is None or c["ts_ms"] >= win_t0)
                        and (win_t1 is None or c["ts_ms"] <= win_t1))]
            dropped += len(g["calls"]) - len(keep)
            g["calls"] = keep
            ts = [c["ts_ms"] for c in keep if c["ts_ms"] is not None]
            g["t0_ms"], g["t1_ms"] = (min(ts), max(ts)) if ts else (None, None)
        groups = [g for g in groups if g["calls"]]
    if dropped:
        warnings.append("%d call(s) outside the run window were excluded "
                        "(a launching session's earlier, unrelated work)" % dropped)

    mode = attribute(groups, timeline)

    rows = []
    for g in groups:
        for c in g["calls"]:
            c["phase"] = g["phase"]
            c["agent_label"] = g["label"]
            c["attribution"] = g["attribution"]
            c["transcript"] = os.path.basename(g["transcript"])
            c["total_input_tokens"] = total_input(c)
            c["cost_usd"] = round(cost_of(c, rates), 6)
            c["ts"] = _ms_to_iso(c["ts_ms"])
            rows.append(c)
    rows.sort(key=lambda r: (r["ts_ms"] is None, r["ts_ms"] or 0))

    # One row per agent invocation. An invocation's span and duration come from
    # its own conversation; an attempt the workflow recorded but that produced no
    # conversation at all (it hung, or the API errored before a first response)
    # still gets a row, with zero calls — otherwise a retry storm would be
    # invisible in exactly the run where it mattered most.
    agent_rows = []
    for g in sorted(groups, key=lambda x: (x["t0_ms"] is None, x["t0_ms"] or 0)):
        durs = [c["duration_ms"] for c in g["calls"] if c["duration_ms"] is not None]
        agent_rows.append({
            "workflow": g.get("workflow", ""), "phase": g["phase"], "label": g["label"],
            "attempt": g.get("attempt", 1), "ok": True, "attribution": g["attribution"],
            "started_at": _ms_to_iso(g["t0_ms"]), "ended_at": _ms_to_iso(g["t1_ms"]),
            "span_ms": (g["t1_ms"] - g["t0_ms"]) if (g["t0_ms"] is not None and g["t1_ms"] is not None) else None,
            "llm_ms": sum(durs),
            "api_calls": len(g["calls"]),
            "input_tokens": sum(c["total_input_tokens"] for c in g["calls"]),
            "output_tokens": sum(c["output_tokens"] for c in g["calls"]),
            "cost_usd": round(sum(c["cost_usd"] for c in g["calls"]), 6),
        })
    for e in (timeline.get("events") or []):
        if e.get("matched"):
            continue
        agent_rows.append({
            "workflow": e["workflow"],
            "phase": e["phase"] if e["tree"] == "root" else "kernel/" + e["phase"],
            "label": e["label"], "attempt": e["attempt"], "ok": e["ok"],
            "attribution": "timeline-only (no transcript: hung, or failed before answering)",
            "started_at": None, "ended_at": None, "span_ms": None, "llm_ms": 0,
            "api_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        })

    agg = aggregate(rows, groups, timeline, rates)
    meta = {
        "schema": SCHEMA, "eval_dir": eval_dir, "attribution_mode": mode,
        "transcripts": transcripts, "timeline_sources": timeline["sources"],
        "window_start": _ms_to_iso(win_t0), "window_end": _ms_to_iso(win_t1),
        "calls_excluded_outside_window": dropped,
        "warnings": warnings, "rates": rates, "complete": not warnings,
        "generated_at": _ms_to_iso(int(datetime.now(tz=timezone.utc).timestamp() * 1000)),
    }
    return rows, agent_rows, agg, meta


def write_outputs(eval_dir, rows, agent_rows, agg, meta):
    out_dir = os.path.join(eval_dir, "reports", "trace")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "llm_calls.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    with open(os.path.join(out_dir, "agent_calls.jsonl"), "w", encoding="utf-8") as fh:
        for a in agent_rows:
            fh.write(json.dumps(a, sort_keys=True) + "\n")
    with open(os.path.join(out_dir, "token_stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "stats": agg}, fh, indent=2, sort_keys=True)
    md_path = os.path.join(out_dir, "token_stats.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_md(agg, meta))
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-API-call token + time ledger for a GEAK run.")
    ap.add_argument("--eval-dir", required=True, help="the run's eval dir; output lands in <eval-dir>/reports/trace/")
    ap.add_argument("--transcripts", action="append", default=None,
                    help="explicit transcript glob (repeatable); default is to discover them")
    ap.add_argument("--rates", default=None, help="JSON file overriding the per-million-token prices")
    ap.add_argument("--since", default=None,
                    help="ISO time; ignore calls before it. Default: the run's first role agent.")
    ap.add_argument("--until", default=None, help="ISO time; ignore calls after it")
    ap.add_argument("--quiet", action="store_true", help="do not print the summary to stdout")
    args = ap.parse_args(argv)

    rates = dict(DEFAULT_RATES)
    if args.rates:
        try:
            with open(args.rates, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                merged = dict(DEFAULT_RATES["_default"])
                merged.update(loaded.get("_default") or {})
                rates = dict(loaded)
                rates["_default"] = merged
        except (OSError, ValueError) as exc:
            print("llm_ledger: --rates ignored (%s: %s)" % (type(exc).__name__, exc), file=sys.stderr)

    try:
        rows, agent_rows, agg, meta = build(
            args.eval_dir, args.transcripts, rates,
            since_ms=_iso_to_ms(args.since), until_ms=_iso_to_ms(args.until))
        out_dir = write_outputs(args.eval_dir, rows, agent_rows, agg, meta)
    except Exception as exc:  # never fail the run that called us
        print("llm_ledger: FAILED (%s: %s) — run is unaffected" % (type(exc).__name__, exc), file=sys.stderr)
        return 0

    if not args.quiet:
        t = agg["total"]
        print("llm_ledger: %s API calls, %s in / %s out, $%.2f, wall %s -> %s"
              % (_n(t["calls"]), _n(t["total_input"]), _n(t["output_tokens"]),
                 t["cost"], _hms(t.get("wall_ms")), out_dir))
        for w in meta["warnings"]:
            print("llm_ledger: incomplete — %s" % w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
