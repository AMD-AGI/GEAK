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
# Two kernel-lane agents do NOT use that header: the optimization engineers
# (kernel_lane.js:683) and the round-winner commit step (kernel_lane.js:817). Between them the
# engineers are the largest block of spend in a run — 948 calls / $108 on the first measured
# Qwen3-14B run — so without these patterns the biggest cost centre in GEAK reads as "(driver)",
# i.e. as though it were not GEAK's spend at all. Anchored at the start of the message so a
# tool result that happens to quote a prompt cannot open a spurious conversation.
ENGINEER_RE = re.compile(r"\A\s*You are Engineer (\S+) \(specialty=([A-Za-z0-9_.\-]+)\) for round")
COMMIT_RE = re.compile(r"\A\s*You are the TechLead committing round")
# Claude Code (>= 2.1.2xx) no longer hands a workflow agent its prompt verbatim. The first user
# turn relays the launching user's request, and the second wraps the script-computed task in a
# one-line frame ("[Workflow harness — computed task] ... The computed task text follows:") with
# EVERY line of the task indented two spaces, so that a frame-like line inside it cannot be forged.
# The anchored patterns above never see column zero of the prompt through that frame, which is how
# the kernel engineers — the largest block of spend in a run — came to read as "(driver)" again:
# $242.61 of $457.08 on the 2026-09-24 gpt-oss-120b run.
#: The harness frame's opening line. Group 1 is the MARKER -- the text inside the
#: brackets, e.g. " — computed task" or " — user request" -- which is the only part
#: that says what the record is. The prose after the bracket is explanation, and it
#: discusses the other kind of frame by name ("the computed task text that follows
#: in the next turn" appears in the USER REQUEST header), so classifying on the
#: whole line read a relayed request as a computed task.
HARNESS_FRAME_RE = re.compile(r"\A\s*\[Workflow harness\b([^\]\n]*)\][^\n]*\n")

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


def _harness_frame(text):
    """``(kind, task_text)`` for one user record.

    *kind* is what the record IS, stated rather than inferred:

    - ``"computed-task"`` — the harness frame the runtime writes when it hands a
      sub-agent its task. This is the only text that says what the agent was
      launched to do.
    - ``"harness"`` — some other ``[Workflow harness …]`` frame.
    - ``""`` — an ordinary user record. Inside a sub-agent transcript that is a
      relayed turn: a tool result, a quoted prompt, a forwarded instruction. It
      may LOOK like a role header and must not be mistaken for one.

    The frame is one header line, then the task with every line indented by exactly two spaces
    (see HARNESS_FRAME_RE). Only that indent is removed, so the task's own indentation survives.
    Callers used to distinguish the two cases by ``text is not raw``, which is an identity trick,
    not a fact about the record — an unframed record that needed no dedent is indistinguishable
    from a framed one under that test.
    """
    m = HARNESS_FRAME_RE.match(text or "")
    if not m:
        return "", text
    body = "\n".join(line[2:] if line.startswith("  ") else line
                      for line in text[m.end():].split("\n"))
    # The marker inside the brackets, never the explanatory prose after it.
    marker = (m.group(1) or "").lower()
    return ("computed-task" if "computed task" in marker else "harness"), body


def _unwrap_harness(text):
    """The task text inside a Claude Code workflow frame, dedented; *text* unchanged otherwise."""
    return _harness_frame(text)[1]


def _content_parts(message):
    """Split an assistant message into (response_text, thinking_text).

    An assistant turn is a list of typed blocks: ``text`` blocks are the reply
    the user (or the next tool) sees, ``thinking``/``redacted_thinking`` blocks
    are the model's reasoning. The ledger records only the input prompt today;
    capturing both output halves here is what lets a report show what a call
    produced, not just what it was asked. ``tool_use`` blocks are left out —
    their inputs are captured as tools elsewhere and can be large/binary.
    """
    if not isinstance(message, dict):
        return "", ""
    content = message.get("content")
    if isinstance(content, str):
        return content, ""
    resp, think = [], []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and isinstance(b.get("text"), str):
                resp.append(b["text"])
            elif t in ("thinking", "redacted_thinking"):
                # The reasoning text lives under "thinking"; redacted blocks have
                # no readable text but are still worth marking as present.
                if isinstance(b.get("thinking"), str):
                    think.append(b["thinking"])
                elif t == "redacted_thinking":
                    think.append("[redacted]")
    return "\n".join(resp), "\n".join(think)


def _content_blocks(message):
    """Yield ``(kind, position, text)`` for each content block of an assistant
    message; ``kind`` is ``"resp"`` (a ``text`` block) or ``"think"`` (a
    ``thinking``/``redacted_thinking`` block). ``position`` is the block's index
    within this message's content list — combined with the record's
    ``apiBlockIndex`` it identifies which block a record is (re-)flushing, so the
    same response streamed across several records merges by block instead of the
    largest-usage record silently dropping the others' text.
    """
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [("resp", 0, content)] if content else []
    out = []
    if isinstance(content, list):
        for j, b in enumerate(content):
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and isinstance(b.get("text"), str):
                out.append(("resp", j, b["text"]))
            elif t in ("thinking", "redacted_thinking"):
                if isinstance(b.get("thinking"), str):
                    out.append(("think", j, b["thinking"]))
                elif t == "redacted_thinking":
                    out.append(("think", j, "[redacted]"))
    return out


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


def agent_meta(path):
    """The runtime's own record of a workflow sub-agent, or None.

    Claude Code writes ``agent-<id>.meta.json`` beside every workflow sub-agent transcript, with
    the exact label the workflow script gave the agent (``description``: ``eng r2_d0:algorithm``,
    ``director:setup``) and the phase it ran in (``workflowPhase``). Unlike the prompt text this
    does not depend on how the harness frames the prompt, and unlike agent_timeline.json it exists
    for a run that was killed before it could write its timeline.
    """
    if not path.endswith(".jsonl"):
        return None
    try:
        with open(path[:-len(".jsonl")] + ".meta.json", "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    desc = doc.get("description") if isinstance(doc.get("description"), str) else ""
    phase = doc.get("workflowPhase") if isinstance(doc.get("workflowPhase"), str) else ""
    if not (desc.strip() or phase.strip()):
        return None
    return {"description": desc.strip(), "phase": phase.strip()}


def workflow_run_of(path):
    """The workflow run a transcript belongs to: ``<runId>`` for
    ``.../subagents/workflows/<runId>/agent-*.jsonl``, None for any other file."""
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 4 and parts[-3] == "workflows" and parts[-4] == "subagents":
        return parts[-2]
    return None


# A workflow label is `role:sub_phase` (``director:setup``) or a verb plus arguments
# (``eng r2_d0:algorithm``, ``commit r2``, ``extract_op <op>``). The kernel lane's two verbs for
# agents that also have a prompt pattern above map to the role that pattern would have given.
_LABEL_VERB_ROLE = {"eng": ("engineer", None), "deep": ("engineer", None), "commit": ("tech_lead", "commit")}
_SUB_RE = re.compile(r"[A-Za-z0-9_.\-]+")


def role_of_label(label):
    """(role, sub_phase) from a workflow label; ('', '') when there is nothing to read."""
    head, _, rest = (label or "").strip().partition(" ")
    if not head:
        return "", ""
    if head in _LABEL_VERB_ROLE:
        role, sub = _LABEL_VERB_ROLE[head]
        if sub is None:
            m = _SUB_RE.match(rest.rpartition(":")[2]) if ":" in rest else None
            sub = m.group(0) if m else ""
        return role, sub
    role, _, sub = head.partition(":")
    m = _SUB_RE.match(sub)
    return role, (m.group(0) if m else "")


def apply_agent_meta(groups):
    """Name every conversation that has runtime metadata by the workflow's own label and phase.

    The label names a conversation the prompt did not: a label is a free-form display string
    (``extract_op fused_moe_a16w4_decode``), so reading a role out of it is lossy, and the
    agent's own brief -- which states its role outright -- is the better source when there is
    one. The label is always kept as ``label`` either way. The phase wins outright — it is recorded per agent by the runtime, where the timeline
    join is positional — with one exception: a nested kernel lane's metadata phase is only the
    lane marker (``▸ kernel-lane #4``), so a reliable timeline match that already placed the agent
    in a specific ``kernel/<phase>`` keeps it. A conversation no prompt pattern recognised stops
    reading as the driver: inside a sub-agent transcript it is that agent, not the launching session.
    """
    for g in groups:
        meta = g.get("agent_meta")
        if not meta:
            continue
        desc = meta["description"]
        if g["role"] == DRIVER and desc:
            role, sub = role_of_label(desc)
            if role:
                g["role"], g["subphase"] = role, sub
        if desc:
            g["label"] = desc
        phase = meta["phase"].lstrip("▸").strip()
        keep_timeline = (g.get("attribution") == "timeline" and meta["phase"].startswith("▸")
                         and str(g.get("phase") or "").startswith("kernel/"))
        if phase and not keep_timeline:
            g["phase"] = phase
            if g.get("attribution") != "timeline":
                g["attribution"] = "agent-meta"


# --------------------------------------------------------------------------- #
# Transcript -> conversation groups -> API-call rows
# --------------------------------------------------------------------------- #
def split_conversations(records, single_agent=False):
    """Split a transcript's records into conversations, one per agent.

    A conversation starts at a user record whose text is a role prompt ("You are
    the X. PHASE=Y."). Anything before the first such record is the driver turn —
    the top-level session that invoked the Workflow tool.

    Splitting on the prompt rather than on file boundaries or the isSidechain
    flag keeps this working whether the CLI writes each sub-agent to its own file
    or inlines them into the parent transcript.

    ``single_agent`` says the file is known to hold exactly ONE agent (a workflow
    sub-agent transcript with its own ``.meta.json``). The first role header then
    names the whole file and later headers are ignored: a tool result that quotes
    another role's prompt must not split one agent into two.
    """
    groups, cur = [], {"role": DRIVER, "subphase": "", "records": [], "prompt": ""}
    named = False
    saw_task = False
    task_text = None   # single_agent: the computed task, kept as the prompt if no header names it
    # single_agent: what each KIND of record proposed the agent is. The computed task is the
    # agent's own brief; everything else is text that reached it, and a relayed turn quoting
    # "You are the profiler. PHASE=baseline." describes whoever was quoted, not this agent.
    proposed = {}
    for rec in records:
        if rec.get("type") == "user" and not (single_agent and named):
            raw = _text_of(rec.get("message"))
            kind, text = _harness_frame(raw)
            if single_agent and kind == "computed-task":
                # Seen, whether or not it names a role. An agent launched with a
                # free-form brief has a computed task that matches no role regex;
                # letting a later quoted turn name it anyway produced a `profiler`
                # label for a `deep_explore` agent.
                saw_task = True
                if task_text is None:
                    task_text = text
            m = ROLE_RE.search(text)
            me = None if m else ENGINEER_RE.search(text)
            if m:
                role, sub = m.group(1), m.group(2)
            elif me:
                # sub_phase is the specialty, so the report separates the memory lane from the
                # compute lane rather than merging every engineer into one bucket.
                role, sub = "engineer", me.group(2)
            elif COMMIT_RE.search(text):
                role, sub = "tech_lead", "commit"
            elif BARE_RE.search(text):
                role, sub = "file_writer", "persist"
            else:
                role, sub = None, None
            if role and single_agent:
                # Do not name from the first match. Let every kind of record have its say and
                # decide once, after the file, so the computed task cannot lose a race to a
                # relayed turn that merely appeared earlier.
                proposed.setdefault(kind or "relayed", (role, sub, text))
            elif role:
                if cur["records"]:
                    groups.append(cur)
                # Keep a slice of the prompt: it is what tells an e2e `director`
                # apart from a kernel-layer `director` (the latter's paths sit
                # under <eval>/kernels/_exp/).
                cur = {"role": role, "subphase": sub, "records": [], "prompt": text[:8000]}
        cur["records"].append(rec)
    if single_agent:
        # Precedence is by what the record is, not when it arrived. A computed task
        # that named no role still SPEAKS for the agent: it is the brief the agent
        # was launched with, so a relayed turn does not get to name it in the gap.
        # The agent then keeps its own task text and falls to explicit metadata.
        pick = proposed.get("computed-task")
        if pick is None and not saw_task:
            pick = proposed.get("harness") or proposed.get("relayed")
        if pick:
            role, sub, text = pick
            cur.update(role=role, subphase=sub, prompt=text[:8000])
            named = True
            others = {k: v[:2] for k, v in proposed.items() if v[:2] != (role, sub)}
            if others:
                # Both readings are kept. Overwriting one with the other produced a group whose
                # role and prompt described different agents, with nothing left to say so.
                cur["role_conflict"] = sorted("%s:%s (%s)" % (r, sp, k)
                                              for k, (r, sp) in others.items())
        elif proposed:
            # Nothing was allowed to name this agent, but readings were offered.
            # Dropping them silently loses the fact that the run was ambiguous.
            cur["role_conflict"] = sorted("%s:%s (%s)" % (r, sp, k)
                                          for k, (r, sp, _t) in proposed.items())
        cur["role_source"] = (
            next(iter(k or "relayed" for k in proposed if proposed[k] is pick), "")
            if pick else ("computed-task" if task_text else ""))
        if not named and task_text:
            cur["prompt"] = task_text[:8000]
    if cur["records"]:
        groups.append(cur)
    return [g for g in groups if any(r.get("type") == "assistant" for r in g["records"])]


def calls_of(group, source):
    """One row per API call in this conversation, deduplicated.

    Dedupe key is message.id: the same response is sometimes flushed to the
    transcript twice. USAGE (token counts) is taken from the copy with the
    largest output_tokens — a partial flush can only undercount — at the EARLIEST
    timestamp. CONTENT (output/thinking) is merged SEPARATELY, by block: the same
    response can emit a thinking block, a text block, and a later tool_use block
    as three records that share one id, and the usage-bearing record is not a full
    copy of the earlier blocks' text. We collect every block seen for the id,
    keyed by ``(kind, apiBlockIndex + position)``, keeping the LONGEST text per
    block (a cumulative flush grows in place, so longest = most complete, and this
    never concatenates a block onto its own earlier prefix). See ``_content_blocks``.
    """
    by_id, order, prev_ts = {}, [], None
    content_by_id = {}  # id -> {(kind, idx): longest_text}
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
        resp_text, think_text = _content_parts(msg)
        # Merge content by block identity, independent of the usage-record choice.
        base = rec.get("apiBlockIndex")
        base = 0 if base is None else base
        blocks = content_by_id.setdefault(key, {})
        for kind, pos, text in _content_blocks(msg):
            if not text:
                continue
            bk = (kind, base + pos)
            if len(text) >= len(blocks.get(bk, "")):
                blocks[bk] = text
        row = {
            "ts_ms": ts,
            "last_seen_ms": ts,
            # NOT a measured request duration -- the transcript records no request time. This is
            # the gap since the previous record in the file, which brackets the call but also
            # contains whatever happened between the two (tool work, compiling, benchmarking,
            # a pause). Named in the row so no reader has to guess how it was arrived at.
            "duration_ms": (ts - prev_ts) if (ts is not None and prev_ts is not None and ts >= prev_ts) else None,
            "duration_source": "inter-record-gap",
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
            "output": resp_text,
            "thinking": think_text,
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
            # Keep BOTH ends of what was observed. The first flush is when the response first
            # appeared; the last is when it stopped growing. Carrying only the first and pairing
            # it with the final token count timed a complete response by its earliest fragment.
            last = max((t for t in (prior.get("last_seen_ms"), ts) if t is not None), default=None)
            if row["output_tokens"] > prior["output_tokens"]:
                keep_ts, keep_dur = prior["ts_ms"], prior["duration_ms"]
                by_id[key] = row
                by_id[key]["ts_ms"], by_id[key]["duration_ms"] = keep_ts, keep_dur
            by_id[key]["last_seen_ms"] = last
        if ts is not None:
            prev_ts = ts
    # Reassemble each kept row's output/thinking from ALL of its blocks (the
    # usage-winning record alone can miss earlier text/thinking blocks).
    for key, row in by_id.items():
        blocks = content_by_id.get(key) or {}
        ordered = sorted(blocks.items(), key=lambda it: it[0][1])
        row["output"] = "\n".join(t for (kind, _), t in ordered if kind == "resp")
        row["thinking"] = "\n".join(t for (kind, _), t in ordered if kind == "think")
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


def cost_breakdown(row, rates):
    """The same ``cost_of`` total, split into the buckets a report shows.

    Keys mirror the question "where did this call's dollars go?":
      - ``uncached_input``  fresh context billed at the full input rate. Anthropic
                            already nets cache out of ``input_tokens``, so this IS
                            the "uncached-context" line — no further subtraction.
      - ``cache_read``      re-sent text served from cache (the tenth-price bucket).
      - ``cache_write``     text stored this call (5-minute + 1-hour writes summed).
      - ``output``          generated text (thinking + response bill the same).
      - ``router``          routing-time LLM cost. 0.0 for the static deterministic
                            router (no routing-time call); a labelled hook for a
                            future dynamic router that would spend to choose a model.
    The five values sum to ``cost_of(row, rates)`` by construction; a test pins that.
    """
    r = rates.get(row.get("model") or "", rates["_default"])
    return {
        "uncached_input": row["input_tokens"] * r["input"] / 1e6,
        "cache_read": row["cache_read_input_tokens"] * r["cache_read"] / 1e6,
        "cache_write": (row["cache_write_5m_tokens"] * r["cache_write_5m"]
                        + row["cache_write_1h_tokens"] * r["cache_write_1h"]) / 1e6,
        "output": row["output_tokens"] * r["output"] / 1e6,
        "router": 0.0,
    }


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
            # and through the glob. Identify a node by its stable `instance` (the
            # producing run's eval dir) so the SAME run is deduped while two DISTINCT
            # runs that happen to share a shape are BOTH kept -- a bare shape
            # fingerprint silently dropped the second of two same-shape lanes. Older
            # timelines predate `instance`; fall back to the shape fingerprint for them.
            inst = node.get("instance")
            fp = ("i", wf, inst) if inst else (
                "s", wf, len(evs), evs[0].get("label") if evs else None,
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

    # A (wf, key) bucket with >= 2 independent dispatches (>= 2 attempt-1 events) is
    # positionally ambiguous: the timeline is recorded in DISPATCH order, but the Nth
    # dispatch need not be the Nth conversation in first-response order -- concurrent
    # same-key agents can answer out of order. Such a match still gets a best-effort
    # phase/label, but is labelled `inferred`, never claimed as exact `timeline`. A lone
    # dispatch (with or without sequential retries -> attempts 1,2,3) stays unambiguous.
    ambiguous = {}
    for wf, keys in by_wf.items():
        for key, slots in keys.items():
            ambiguous[(wf, key)] = sum(1 for e in slots if (e.get("attempt") or 1) == 1) > 1

    ordered = sorted(groups, key=lambda g: (g["t0_ms"] is None, g["t0_ms"] or 0))

    def _owners_for(g, key):
        owners = [wf for wf in by_wf if key in labels_of_wf[wf]]
        if len(owners) > 1:
            # Only `director` collides in practice. The kernel layer's eval dirs
            # live under <e2e eval>/kernels/_exp/, so the prompt says which it is.
            nested = "/kernels/_exp/" in (g.get("prompt") or "")
            pick = [w for w in owners if (w != "e2e_workflow") == nested]
            owners = pick or owners
        return owners

    # How many conversations will land in each (wf, key) bucket. When this differs from
    # the number of recorded dispatches (len(slots)) the positional cursor is mapping an
    # INCOMPLETE set: a surviving transcript can be slotted onto an attempt it did not
    # produce -- e.g. attempt 1 hung with no transcript, attempt 2 answered, so the lone
    # transcript falls on slot 0 (attempt 1). Such a match still gets a best-effort
    # phase/label, but its attempt identity and recorded outcome are a guess, so it is
    # `inferred`, never claimed as exact `timeline`.
    conv_per_bucket = defaultdict(int)
    for g in ordered:
        if g["role"] == DRIVER:
            continue
        key = _conv_key(g["role"], g["subphase"])
        for wf in _owners_for(g, key):
            conv_per_bucket[(wf, key)] += 1
            break
    incomplete = {}
    for wf, keys in by_wf.items():
        for key, slots in keys.items():
            incomplete[(wf, key)] = conv_per_bucket.get((wf, key), 0) != len(slots)

    cursor = defaultdict(int)
    for g in ordered:
        g["label"] = _conv_key(g["role"], g["subphase"])
        g["attribution"] = "inferred"
        if g["role"] == DRIVER:
            g["phase"], g["attribution"] = DRIVER, "driver"
            continue
        key = g["label"]
        for wf in _owners_for(g, key):
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
                reliable = not ambiguous.get((wf, key)) and not incomplete.get((wf, key))
                g["attribution"] = "timeline" if reliable else "inferred"
                # Carry the recorded outcome ONLY when we reliably know which attempt this
                # transcript is; otherwise leave it unknown (None) rather than promote a
                # guessed attempt's ok to fact. The timeline keeps its own per-attempt
                # outcomes (surfaced via the unmatched/`agent_attempts_failed` path).
                g["ok"] = e["ok"] if reliable else None
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


def row_end_ms(row):
    """When this call was LAST observed, not when it was first seen.

    A response that streams is flushed repeatedly as it grows, and each flush is
    its own record. ``ts_ms`` is the first flush -- kept as the call's timestamp
    so first-seen ordering and the run window are unchanged -- and
    ``last_seen_ms`` is the final one. Measuring a span to ``ts_ms`` stops it at
    the first flush, which is why the HTML tree and the ledger's own totals
    disagreed about the same run: one had been taught about the last flush and
    the other had not.

    This is the last flush OBSERVED in the transcript. An interrupted stream
    never writes a final one, so it is not a claim that the response completed.
    """
    end, ts = row.get("last_seen_ms"), row.get("ts_ms")
    if end is None:
        return ts
    if ts is None:
        return end
    return max(ts, end)


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
        ends = [e for e in (row_end_ms(r) for r in rows if sel(r)) if e is not None]
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
    # How many agents ran is two populations that are counted independently and cannot be
    # joined: `load_timeline` carries no identity that a conversation can be matched on, so
    # how far the two overlap is simply unknown. Report both, plus the unknown, and never
    # present a derived single number as a count.
    #
    #   agents_with_transcripts: conversations found in the scoped transcripts.
    #   agents_dispatched:       dispatch attempts the workflow itself recorded, including the
    #                            ones that hung or errored before writing any transcript.
    #
    # Neither contains the other: a timeline written by only the LAST invocation of a resumed
    # or re-entered run records that invocation's attempts alone, while transcripts from every
    # invocation are in scope; and an attempt that died early is in the timeline only.
    events = timeline.get("events") or []
    total["agents_with_transcripts"] = len(groups)
    total["agents_dispatched"] = len(events)
    # A LOWER BOUND on the agents that ran, not a count -- it assumes the smaller population is
    # wholly contained in the larger, which nothing here establishes. `agents_exact` says whether
    # it may be read as a count: only when one population is empty is there nothing to overlap.
    total["agents"] = max(len(events), len(groups))
    # Exact only where the populations were actually RECORDED and there is nothing
    # to overlap. With no timeline at all, an empty dispatch population is not a
    # recorded zero -- it is an unread one, and calling the transcript count exact
    # on that basis presented "what we happened to find" as "what ran".
    total["agents_timeline_recorded"] = bool(timeline.get("sources"))
    total["agents_exact"] = bool(
        total["agents_timeline_recorded"] and not (events and groups))
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
    L.append("- observed window: %s → %s (%s) — from the first request (start inferred by "
             "stepping back that call's gap) to the last flush seen in the transcripts"
             % (t.get("started_at") or "?", t.get("ended_at") or "?", _hms(t.get("wall_ms"))))
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
        ["API calls", "agents" if t.get("agents_exact") else "agents (>=)",
         "wall", "in (total)", "out", "in:out", "billed", "no-reuse", "saved", "re-sent cheaply"],
        ["r"] * 10,
        [[_n(t["calls"]), _n(t["agents"]), _hms(t.get("wall_ms")), _n(t["total_input"]), _n(t["output_tokens"]),
          ratio, "$%.2f" % t["cost"], "$%.2f" % t["list_cost"], "%.1f%%" % saved, reuse]])
    L.append("")
    if not t.get("agents_exact"):
        L.append("- agents: %d conversation(s) in the scoped transcripts, %d dispatch attempt(s) "
                 "recorded by the workflow (%d of them failed). The two are counted separately "
                 "and cannot be joined — no shared identity is recorded — so how far they overlap "
                 "is unknown and the figure above is a lower bound, not a count."
                 % (t.get("agents_with_transcripts", 0), t.get("agents_dispatched", 0),
                    t.get("agent_attempts_failed", 0)))
        if not t.get("agents_timeline_recorded"):
            L.append("- no `agent_timeline.json` was read for this run, so the dispatch "
                     "population is NOT RECORDED rather than recorded as zero. The figure "
                     "above rests on transcripts alone and cannot be read as a total.")
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
def run_window(groups, since_ms=None, until_ms=None, trust_scope=False):
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

    ``trust_scope`` says the transcripts were already established as this run's
    OWN files (an owned-workflow scope, e.g. its ``subagents/workflows/<runId>/``
    dir) rather than found by an eval-dir substring across every session. In that
    case there is no foreign launching session to fence out, and inferring a
    lower bound would wrongly DROP the run's own early un-role-headed overhead —
    a clock/storage helper or the driver's own first turn that ran before the
    first role agent. So the inferred lower bound is skipped; an explicit
    ``--since`` still applies, because a caller that names a bound means it.
    """
    if since_ms is None and not trust_scope:
        starts = [g["t0_ms"] for g in groups
                  if g["role"] != DRIVER and g["t0_ms"] is not None]
        since_ms = min(starts) if starts else None
    return since_ms, until_ms


def build(eval_dir, explicit_globs=None, rates=None, roots=None,
          since_ms=None, until_ms=None, scope=None, scope_warnings=(),
          owned_scope=False, scope_anchor=None):
    """Build the whole ledger. Returns (rows, agent_rows, agg, meta).

    ``scope`` records HOW the transcripts were selected (``run-scoped`` /
    ``run-scoped-inferred`` / ``partial`` / ``substring-fallback`` / ``explicit``),
    surfaced in the meta so the report can show it. ``scope_anchor`` records how a
    whole-run anchor was ESTABLISHED (e.g. ``exp_root-ancestor`` for an inferred,
    unproven containment match) so the persisted meta — not just a caller's return
    dict — reveals that the scope was inferred. ``scope_warnings`` are coverage
    caveats the caller already knows (e.g. a nested lane whose transcripts could
    not be resolved); they join the ledger's own warnings and mark the run
    incomplete. ``owned_scope`` says the globs are this run's established own files,
    which lifts the inferred run-window lower bound (see ``run_window``)."""
    rates = rates or DEFAULT_RATES
    warnings = list(scope_warnings or [])
    timeline = load_timeline(eval_dir)
    if not timeline["sources"]:
        warnings.append("no agent_timeline.json — phases inferred, not recorded")

    transcripts = discover_transcripts(eval_dir, explicit_globs, roots)
    if not transcripts:
        warnings.append("no transcripts found for this eval dir — token counts are empty")

    groups = []
    for path in transcripts:
        recs = list(read_jsonl(path))
        base = os.path.basename(path)
        meta_rec = agent_meta(path)
        for ci, g in enumerate(split_conversations(recs, single_agent=meta_rec is not None)):
            g["calls"] = calls_of(g, base)
            if not g["calls"]:
                continue
            ts = [c["ts_ms"] for c in g["calls"] if c["ts_ms"] is not None]
            ends = [e for e in (row_end_ms(c) for c in g["calls"]) if e is not None]
            g["t0_ms"] = min(ts) if ts else None
            g["t1_ms"] = max(ends) if ends else None
            g["transcript"] = path
            g["agent_meta"] = meta_rec
            g["workflow_run"] = workflow_run_of(path)
            # Stable per-conversation identity: one agent attempt = one group.
            # Two attempts that share a role/label — retries, or the same role in
            # separate transcripts — stay distinct nodes because the id carries
            # the transcript + the conversation's ordinal within it.
            g["group_id"] = "%s#%d" % (base, ci)
            groups.append(g)

    # Drop everything outside the run's own window BEFORE attributing, so a
    # launching session's unrelated history cannot be billed to this run.
    win_t0, win_t1 = run_window(groups, since_ms, until_ms, trust_scope=owned_scope)
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
            ends = [e for e in (row_end_ms(c) for c in keep) if e is not None]
            g["t0_ms"] = min(ts) if ts else None
            g["t1_ms"] = max(ends) if ends else None
        groups = [g for g in groups if g["calls"]]
    if dropped:
        warnings.append("%d call(s) outside the run window were excluded "
                        "(a launching session's earlier, unrelated work)" % dropped)

    mode = attribute(groups, timeline)
    apply_agent_meta(groups)

    rows = []
    for g in groups:
        for c in g["calls"]:
            c["phase"] = g["phase"]
            # calls_of stamped the role before the metadata pass could name the agent.
            c["role"], c["sub_phase"] = g["role"], g["subphase"]
            c["agent_label"] = g["label"]
            c["attribution"] = g["attribution"]
            c["transcript"] = os.path.basename(g["transcript"])
            c["workflow_run"] = g.get("workflow_run")
            c["group_id"] = g["group_id"]
            c["prompt"] = g.get("prompt", "")
            # How this agent came by its name, and what else claimed it. Both were
            # computed and then thrown away with the temporary group: nothing that
            # was saved said whether a name was read off the agent's own brief or
            # off a turn relayed into it.
            c["role_source"] = g.get("role_source", "")
            if g.get("role_conflict"):
                c["role_conflict"] = list(g["role_conflict"])
            c["total_input_tokens"] = total_input(c)
            c["cost_usd"] = round(cost_of(c, rates), 6)
            c["cost_breakdown"] = {k: round(v, 6) for k, v in cost_breakdown(c, rates).items()}
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
            "workflow": g.get("workflow", ""), "workflow_run": g.get("workflow_run"),
            "phase": g["phase"], "label": g["label"],
            # `ok` carries the reliably-matched timeline outcome; None when the join is a
            # guess (ambiguous/incomplete mapping). Groups from a run with no timeline at
            # all default True -- a transcript exists, and there is no recorded outcome to
            # contradict. Never hardcode True over a known-uncertain mapping.
            "attempt": g.get("attempt", 1), "ok": g.get("ok", True), "attribution": g["attribution"],
            "role_source": g.get("role_source", ""),
            "role_conflict": list(g.get("role_conflict") or []),
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
        "transcript_scope": scope,
        "transcript_scope_anchor": scope_anchor,
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
    ap.add_argument("--scope", default=None,
                    help="how transcripts were selected (run-scoped/run-scoped-inferred/"
                         "partial/substring-fallback/explicit); recorded in the meta")
    ap.add_argument("--scope-anchor", default=None,
                    help="how a whole-run anchor was established (e.g. exp_root-ancestor "
                         "for an inferred containment match); recorded in the meta")
    ap.add_argument("--scope-warning", action="append", default=None, dest="scope_warnings",
                    help="a coverage caveat to record (repeatable); marks the run incomplete")
    ap.add_argument("--owned-scope", action="store_true",
                    help="the globs are this run's OWN files; lift the inferred run-window "
                         "lower bound so the run's early un-role-headed calls are kept")
    ap.add_argument("--claude-home", action="append", default=[], dest="claude_homes",
                    help="extra Claude home to discover transcripts in (repeatable), searched "
                         "after $CLAUDE_CONFIG_DIR and ~/.claude")
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
        roots = None
        if args.claude_homes:
            roots = transcript_roots() + [os.path.abspath(h) for h in args.claude_homes
                                          if os.path.isdir(h)]
        rows, agent_rows, agg, meta = build(
            args.eval_dir, args.transcripts, rates, roots=roots,
            since_ms=_iso_to_ms(args.since), until_ms=_iso_to_ms(args.until),
            scope=args.scope, scope_warnings=args.scope_warnings or (),
            owned_scope=args.owned_scope, scope_anchor=args.scope_anchor)
        out_dir = write_outputs(args.eval_dir, rows, agent_rows, agg, meta)
    except Exception as exc:  # never fail the run that called us
        print("llm_ledger: FAILED (%s: %s) — run is unaffected" % (type(exc).__name__, exc), file=sys.stderr)
        return 0

    if not args.quiet:
        t = agg["total"]
        print("llm_ledger: %s API calls, %s in / %s out, $%.2f, wall %s -> %s"
              % (_n(t["calls"]), _n(t["total_input"]), _n(t["output_tokens"]),
                 t["cost"], _hms(t.get("wall_ms")), out_dir))
        if meta.get("transcript_scope"):
            print("llm_ledger: transcript scope = %s%s"
                  % (meta["transcript_scope"],
                     "" if meta.get("complete") else " (INCOMPLETE — see warnings)"))
        for w in meta["warnings"]:
            print("llm_ledger: incomplete — %s" % w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
