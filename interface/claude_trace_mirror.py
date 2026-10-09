#!/usr/bin/env python3
"""Mirror Claude Code's LLM ledger into the run's own output directory.

GEAK issues almost no LLM calls itself: ``run_e2e.py`` hands one prompt to
Claude Code, which runs ``e2e_workflow/e2e_workflow.js`` and records the whole
call tree under its OWN home — ``$CLAUDE_CONFIG_DIR`` if the launching shell set
it, ``~/.claude`` otherwise. That home is chosen by the environment, not by the
run, and on a container it is routinely an overlay that dies with the container.
A run's entire cost record therefore has a shorter lifetime than the run's
output directory, which sits on durable storage with none of it in there.

This module closes that gap: it copies the ledger for THIS run into
``<eval_dir>/llm_trace/`` as the run goes, so the telemetry inherits the
durability of the artifacts it describes.

The mirror is a Claude home in miniature, not a free-form copy. Run discovery
here globs ``projects/*/*/workflows/wf_*.json`` and derives everything else
RELATIVE to the record it found — per-agent transcripts at
``<record>/../../subagents/workflows/<runId>/``, the orchestrator conversation at
``<session_dir>.jsonl``. The slug and session names are wildcards and are never
parsed; only the depth matters. So the mirror reproduces that shape verbatim and
reads back as a home:
``python3 interface/geak_report.py --eval-dir <eval_dir> --claude-home <eval_dir>/llm_trace``.
``--claude-home`` appends a search root, it does not select a run; the eval dir
does that. The record is copied byte for byte precisely so the eval dir it names
still selects it from the mirror.

Everything here is best effort. An 18-hour optimization run must never die over
telemetry, so no function in this module raises to its caller: failures are
recorded in the manifest and the run carries on.
"""
from __future__ import annotations

import glob as _glob
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Name of the mirror directory created inside a run's ``eval_dir``.
MIRROR_DIRNAME = "llm_trace"

#: Name of the mirror's self-describing sidecar.
MANIFEST_NAME = "manifest.json"

#: The consumer's discovery glob. Reproduced here so the mirror's layout is
#: pinned to the contract rather than to a description of it.
WORKFLOW_GLOB = "projects/*/*/workflows/wf_*.json"

#: Default ceiling on bytes copied per mirror pass. Agent transcripts for a long
#: run reach hundreds of megabytes; the budget keeps a runaway transcript from
#: filling the run's output volume. Override with ``GEAK_TRACE_MIRROR_MAX_MB``.
DEFAULT_MAX_BYTES = 4 * 1024 * 1024 * 1024

#: Default minimum seconds between mid-run mirror passes
#: (``GEAK_TRACE_MIRROR_INTERVAL_S``).
DEFAULT_INTERVAL_S = 900.0

#: Default wall-clock ceiling on a single mirror pass
#: (``GEAK_TRACE_MIRROR_DEADLINE_S``). A pass may run inside a SIGTERM grace
#: period, so it must be able to give up rather than delay ``result.json``.
DEFAULT_DEADLINE_S = 120.0


# --------------------------------------------------------------------------- #
# Locating the ledger
# --------------------------------------------------------------------------- #
def candidate_homes(extra: Iterable[Path] = ()) -> list[Path]:
    """List the Claude Code homes that may hold this run's record.

    Deliberately the same precedence the reader uses (``CLAUDE_CONFIG_DIR``
    first, then ``~/.claude``) so the mirror searches where the report tool
    would look.

    Args:
        extra: Additional roots to search after the standard two.

    Returns:
        Existing directories, in search order, without duplicates.
    """
    seen: dict[str, Path] = {}
    configured = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    ordered: list[Path] = [Path(configured)] if configured else []
    ordered.append(Path.home() / ".claude")
    ordered.extend(Path(p) for p in extra)
    for home in ordered:
        try:
            resolved = home.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_dir():
            seen.setdefault(str(resolved), resolved)
    return list(seen.values())


# The two fields a record can name a run directory under. Provenance matters:
# an ``exp_root`` that happens to equal another run's ``eval_dir`` is NOT that
# run's identity — a child lane declaring ``exp_root == parent's eval_dir`` must
# never be mistaken for the parent. So paths are carried tagged with their field.
_FIELD_EVAL = "eval_dir"
_FIELD_EXP = "exp_root"


def record_paths_typed(record: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(field, directory)`` pairs a workflow record names.

    An orchestrator-driven run is identified by ``eval_dir``; a standalone one by
    ``exp_root``. Either can appear under ``args`` or under ``result``. The field
    each directory came from is preserved so an exact ``exp_root`` match cannot
    be confused with — or outrank — an exact ``eval_dir`` identity.

    Args:
        record: A parsed ``wf_*.json`` record.

    Returns:
        Distinct ``(field, directory)`` pairs, ``eval_dir`` entries before
        ``exp_root`` entries.
    """
    found: list[tuple[str, str]] = []
    for field in (_FIELD_EVAL, _FIELD_EXP):
        for holder in (record.get("args"), record.get("result")):
            if not isinstance(holder, dict):
                continue
            value = holder.get(field)
            if isinstance(value, str) and value.strip():
                pair = (field, value.strip().rstrip("/"))
                if pair not in found:
                    found.append(pair)
    return found


def record_paths(record: dict[str, Any]) -> list[str]:
    """Return the run directories a workflow record names, most specific first.

    A thin, field-erased view of :func:`record_paths_typed` for callers (e.g.
    the mirror manifest) that only need the distinct directory strings.

    Args:
        record: A parsed ``wf_*.json`` record.

    Returns:
        Distinct directory strings, ``eval_dir`` before ``exp_root``.
    """
    found: list[str] = []
    for _field, directory in record_paths_typed(record):
        if directory not in found:
            found.append(directory)
    return found


def iter_records(homes: Iterable[Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield every readable workflow record beneath *homes*.

    Args:
        homes: Claude Code home directories.

    Yields:
        ``(record_path, record)`` pairs. Unreadable or non-object files are
        skipped silently — a half-written record is normal mid-run.
    """
    for home in homes:
        try:
            paths = sorted(home.glob(WORKFLOW_GLOB))
        except OSError:
            continue
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(record, dict):
                yield path, record


def _matches(
    record: dict[str, Any], wanted: str, want_field: str | None = None
) -> bool:
    """Report whether *record* names *wanted* (or a parent/child of it).

    Args:
        record: A parsed workflow record.
        wanted: A run directory, already stripped of a trailing slash.
        want_field: The field *wanted* was requested as (``eval_dir`` or
            ``exp_root``), used only to rank same-field vs other-field exact
            matches; ``None`` disables the distinction.

    Returns:
        ``True`` when one of the record's directories relates to *wanted*.
    """
    return _match_rank(record, wanted, want_field) is not None


# Ownership is not binary, and it is field-aware. An EXACT directory match may
# identify a run, but ONLY when it comes from the same field the caller asked by:
# a record whose ``exp_root`` equals the requested ``eval_dir`` is a child lane
# announcing its experiment root, not the owner of that eval-dir. A mere shared
# experiment-root ("both live under /exp") is weaker still. These ranks order a
# record's best relation to *wanted*, smallest = most specific, so neither a
# same-root sibling NOR a cross-field exact near-miss can outrank the run whose
# own field IS *wanted*.
_RANK_EXACT_SAME = 0    # exact match, from the SAME field the caller requested
_RANK_EXACT_OTHER = 1   # exact match, but from the OTHER field (e.g. exp_root vs eval_dir)
_RANK_CHILD = 2         # the record names a directory INSIDE *wanted* (a nested lane)
_RANK_ANCESTOR = 3      # the record names an ancestor of *wanted* (e.g. its exp_root)


def _match_rank(
    record: dict[str, Any], wanted: str, want_field: str | None = None
) -> int | None:
    """The most specific relation *record* has to *wanted*, or ``None``.

    ``None`` means no directory the record names relates to *wanted* at all.
    A smaller number is a stronger claim to *own* *wanted* (see the rank
    constants); this is what lets ``find_record`` prefer a same-field exact match
    over a newer record that only shares an experiment root OR names *wanted*
    through a different field. When *want_field* is ``None`` an exact match on
    either field is treated as same-field (rank 0), preserving the older
    field-agnostic behaviour for callers that do not care about provenance.
    """
    best: int | None = None
    for field, found in record_paths_typed(record):
        if found == wanted:
            rank = (_RANK_EXACT_SAME
                    if want_field is None or field == want_field
                    else _RANK_EXACT_OTHER)
        elif found.startswith(wanted + "/"):
            rank = _RANK_CHILD
        elif wanted.startswith(found + "/"):
            rank = _RANK_ANCESTOR
        else:
            continue
        if best is None or rank < best:
            best = rank
    return best


def find_record(
    homes: Iterable[Path],
    *,
    eval_dir: str | None = None,
    exp_root: str | None = None,
    session_id: str | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    """Find this run's workflow record.

    Selection is an identity match on the directories the record names — the
    same check the report tool's ``--eval-dir`` performs — never a guess by
    mtime. ``session_id`` only narrows an already-matching set; a record is
    never chosen on the session id alone, because a session can drive more than
    one run.

    Args:
        homes: Claude Code homes to search.
        eval_dir: This run's eval dir, if known. Tried first.
        exp_root: This run's experiment root. Tried when *eval_dir* misses.
        session_id: The SDK session id, used only as a tie-breaker.

    Returns:
        The best ``(path, record)`` match, or ``None``.
    """
    candidates = list(iter_records(homes))
    for wanted, want_field in ((eval_dir, _FIELD_EVAL), (exp_root, _FIELD_EXP)):
        cleaned = (wanted or "").strip().rstrip("/")
        if not cleaned:
            continue
        hits = [(p, r) for p, r in candidates if _matches(r, cleaned, want_field)]
        if not hits:
            continue
        if session_id:
            narrowed = [(p, r) for p, r in hits if p.parent.parent.name == session_id]
            if narrowed:
                hits = narrowed
        # Rank by ownership specificity FIRST — a SAME-FIELD exact match beats a
        # record that names *cleaned* only through the other field (e.g. a child
        # lane whose exp_root equals this eval-dir) or that merely shares an
        # ancestor experiment root — and only then by newest recorded timestamp.
        # Without the specificity key a newer sibling that happens to share
        # exp_root, or a child announcing this dir as its exp_root, would outrank
        # the run whose own eval-dir is exactly *cleaned*. (mtime is never used; a
        # recorded timestamp is.)
        hits.sort(key=lambda pr: (_match_rank(pr[1], cleaned, want_field),
                                  _neg_ts(pr[1].get("timestamp"))))
        return hits[0]
    return None


def _neg_ts(timestamp: Any) -> tuple[int, str]:
    """A sort key that orders newer timestamps first under an ascending sort.

    ISO-8601 stamps are lexically ordered, so inverting each character's code
    point yields a descending order without parsing. A leading present/absent
    flag (0 = present, 1 = missing) guarantees missing stamps sort LAST among
    equal ranks: a single-character sentinel could not, since an inverted real
    stamp can itself reach the top of the code-point range.
    """
    s = str(timestamp or "")
    if not s:
        return (1, "")                    # no stamp -> last among equal ranks
    return (0, "".join(chr(0x10FFFF - ord(c)) for c in s))


def _owns(record: dict[str, Any], eval_dir: str | None, exp_root: str | None) -> bool:
    """Whether *record* names *eval_dir* or *exp_root* as a SAME-FIELD exact match.

    Ownership is field-aware: a record owns the requested ``eval_dir`` only when
    ITS OWN ``eval_dir`` equals it (rank 0), and the requested ``exp_root`` only
    when its own ``exp_root`` equals it. A child lane whose ``exp_root`` merely
    equals the requested ``eval_dir`` is an OTHER-FIELD match (rank 1) and does
    NOT own it — so it can never be adopted as the parent's top-level identity.
    A shared experiment root (an ANCESTOR match) is weaker still and never owns.
    This is the guard that stops a sibling or child being selected for a run
    whose own top-level record is absent.
    """
    for wanted, want_field in ((eval_dir, _FIELD_EVAL), (exp_root, _FIELD_EXP)):
        cleaned = (wanted or "").strip().rstrip("/")
        if cleaned and _match_rank(record, cleaned, want_field) == _RANK_EXACT_SAME:
            return True
    return False


def _resolve_one(
    homes: list[Path], eval_dir: str | None, exp_root: str | None,
) -> tuple[str, bool] | None:
    """Resolve one requested instance to its own ``agent-*.jsonl`` glob.

    Returns ``(glob, usability)`` when a record EXACTLY owns the instance and
    carries a ``runId``; ``None`` when no owning record exists. ``usability`` is
    the :func:`transcript_usability` split of what is actually on disk, so the
    caller can flag a resolved-but-empty lane rather than trust a nonempty glob
    STRING -- or a zero-byte file -- as proof of coverage.
    """
    rec = find_record(homes, eval_dir=eval_dir, exp_root=exp_root)
    if not rec:
        return None
    record_path, record = rec
    # find_record ranks exact over ancestor, but may still return an ancestor-only
    # match when nothing exact exists (a sibling sharing exp_root). Require exact
    # ownership here so such a near-miss is treated as unresolved, not adopted.
    if not _owns(record, eval_dir, exp_root):
        return None
    return _glob_for_record(record_path, record)


def is_transcript_name(name: str) -> bool:
    """Whether *name* is an agent transcript, not a journal or a metadata file."""
    return name.startswith("agent-") and name.endswith(".jsonl")


def classify_transcript(path: Path | str) -> str:
    """One transcript's usability: ``usable``, ``empty`` or ``unreadable``.

    Only a file with BYTES in it is evidence that an invocation's calls were
    captured. A zero-byte ``agent-*.jsonl`` is a file the runtime created for an
    agent that never flushed: it proves the agent existed, NOT that it spent
    nothing. Treating its existence as coverage is how a known hole shipped as a
    complete report.
    """
    try:
        size = Path(path).stat().st_size
    except OSError:
        return "unreadable"
    return "usable" if size > 0 else "empty"


def transcript_usability(pattern: str) -> dict[str, list[str]]:
    """Split the transcripts *pattern* names into usable / empty / unreadable.

    This is the ONE reader for the two halves that have to agree: the scope
    resolver, which decides whether a run's coverage is complete, and the mirror,
    which copies the files and reports coverage for the same invocations. While
    the resolver tested only ``glob.glob(...)`` -- a FILENAME test -- a zero-byte
    transcript satisfied report scope while the mirror called the same invocation
    unusable, and the report shipped ``complete: true`` with no warnings over a
    hole the mirror had already found.

    Returns:
        ``{"usable", "empty", "unreadable"}``, each a sorted list of paths.
    """
    out: dict[str, list[str]] = {"usable": [], "empty": [], "unreadable": []}
    for path in sorted(_glob.glob(pattern)):
        if not is_transcript_name(os.path.basename(path)):
            continue
        out[classify_transcript(path)].append(path)
    return out


def _site_usability(site: dict[str, Any]) -> dict[str, list[str]]:
    """The usability split an ``_owned_sites`` entry carries, as one dict."""
    return {
        "usable": ["<kept>"] if site.get("files_exist") else [],
        "empty": list(site.get("transcripts_empty") or ()),
        "unreadable": list(site.get("transcripts_unreadable") or ()),
    }


def _hole_note(usability: dict[str, list[str]]) -> str:
    """Name the unusable transcripts, so a warning says WHICH file is a hole."""
    bits = []
    for state in ("empty", "unreadable"):
        names = [os.path.basename(p) for p in usability.get(state) or ()]
        if names:
            bits.append("%d %s (%s)" % (len(names), state, ", ".join(names)))
    return "; ".join(bits)


def _glob_for_record(
    record_path: Path, record: dict[str, Any]
) -> tuple[str, dict[str, list[str]]] | None:
    """Build the ``agent-*.jsonl`` glob for *record*'s own run dir.

    Returns ``(glob, usability)`` -- the usability split from
    :func:`transcript_usability`, so the caller can tell a resolved-but-empty
    lane from a covered one instead of trusting a nonempty glob STRING, or a
    zero-byte file, as proof of coverage. ``None`` when the record has no
    ``runId``.
    """
    run_id = str(record.get("runId") or "")
    if not run_id:
        return None
    session_dir = record_path.parent.parent
    g = str(session_dir / "subagents" / "workflows" / run_id / "agent-*.jsonl")
    return g, transcript_usability(g)



# Sentinel: more than one enclosing run could anchor a mid-run eval-dir, so we
# refuse to guess rather than bill the wrong run's transcripts.
_ANCHOR_AMBIGUOUS = "ambiguous"

# How an invocation was tied to the eval-dir: by its own workflow record, or -- when it has
# none, because it is still running or was killed before returning -- by its journal.
EVIDENCE_RECORD = "record"
EVIDENCE_JOURNAL = "journal"


def owned_invocations(
    homes: Iterable[Path], eval_dir: str | None,
) -> list[dict[str, str]]:
    """Every workflow invocation that worked IN *eval_dir*, as its own run directory.

    One eval-dir is routinely served by more than one invocation: a run resumed after its
    session died, or re-entered with ``phases: final`` to finish what a killed run left. Each
    gets its own ``runId`` and ``subagents/workflows/<runId>/`` dir, and all of them are spend on
    this run. Picking one -- the newest record -- dropped an 18-hour original and kept a 53-minute
    finalize (2,270 calls down to 103, 2026-09-25).

    Ownership is positive either way; a mere mention of the path never counts:

    - a workflow record whose OWN ``eval_dir`` (args or result) is exactly *eval_dir*;
    - failing a record, a journal row in which an agent RETURNED *eval_dir* as its own:
      ``{"type": "result", "key": ..., "agentId": ..., "result": {"eval_dir": "<eval_dir>"}}``.
      The row is parsed, not matched as text, and only a DIRECT ``result.eval_dir`` counts -- a
      path under any other key (``comparison_target``, a diagnostic subject) is what that row is
      ABOUT, not who wrote it, and reading the two alike billed unrelated invocations here.

    Transcript sets never overlap (each runId owns its directory), so the union cannot double
    count. The same runId found under two homes (a live home and a mirror of it) is taken once,
    from the first home. Directories without a USABLE
    ``agent-*.jsonl`` (one with bytes in it) are skipped here; the unfiltered
    inventory keeps them, so the hole can be reported.

    Returns:
        ``[{"run_id", "glob", "evidence"}]`` in discovery order; empty when nothing owns it.
    """
    return [
        {key: site[key] for key in ("run_id", "glob", "evidence")}
        for site in _owned_sites(homes, eval_dir)
        if site.get("files_exist")
    ]


#: The one journal row that carries an invocation's OWN identity. The runtime
#: writes ``{"type": "result", "key": ..., "agentId": ..., "result": {...}}`` for
#: each agent that returns; ``result.eval_dir`` is what THAT agent declared it was
#: working in. Nothing else in the journal is an ownership statement.
_JOURNAL_RESULT = "result"

#: How well a journal-owned invocation can be attributed. The supported native
#: schema carries BOTH producer fields (``key`` and ``agentId``); a row that
#: declares an ``eval_dir`` without them is still a claim, but it is one nobody
#: signed, so it is adopted down an explicit path and labelled rather than
#: passed off as a provenanced hit with two empty strings in it.
OWNER_DECLARED = "declared"
OWNER_UNATTRIBUTED = "unattributed"


def _eval_dir_mention_re(wanted: str) -> "re.Pattern[str]":
    """Matches the *text* ``"eval_dir": "<wanted>"`` anywhere in a journal.

    Deliberately NOT an ownership test. It is kept only to tell a journal that
    says nothing about *wanted* apart from a mention that could not be read as a
    declaration -- which is reportable as incomplete evidence.
    """
    return re.compile(r'"eval_dir"\s*:\s*"' + re.escape(wanted) + r'/?"')


def _journal_claims(journal: Path, wanted: str) -> tuple[list[dict[str, str]], bool]:
    """What this journal DECLARES, and whether it only mentions *wanted*.

    Ownership is read from a supported row shape, not from the file's text. A
    text search cannot tell a declaration from a reference: a debug row's
    ``{"comparison_target": {"eval_dir": "<wanted>"}}`` serializes to exactly the
    bytes an owner's row does, and matching it billed an unrelated invocation to
    this run. So each line is parsed, and only a ``type="result"`` row whose
    ``result`` is a dict with a DIRECT ``eval_dir`` string counts. An
    ``eval_dir`` nested under any other key is somebody else's subject, not this
    invocation's identity.

    The supported native schema is ``{"type", "key", "agentId", "result"}`` and
    BOTH producer fields are required to be non-empty strings; they are what makes
    a hit auditable back to the agent that wrote it. They are validated here
    rather than read optimistically: ``str(row.get("key") or "")`` turned a
    missing or wrongly-typed field into ``""`` and the site then travelled into
    the serialized scope as an owner with no owner, which reads as provenance
    that was never recorded. A row that declares an ``eval_dir`` without them is
    NOT dropped -- an older or hand-written journal may genuinely predate the
    fields, and discarding its claim would undercount real spend -- it is
    adopted on the explicit ``OWNER_UNATTRIBUTED`` path, which the caller
    reports so the missing provenance is visible instead of silent.

    Returns:
        ``(claims, mention_only)``. Each claim is ``{"eval_dir", "key",
        "agent_id", "provenance"}`` -- the declared dir, the agent that declared
        it, and whether that attribution was actually present, kept so a hit can
        say WHICH agent proves it rather than just that the file matched. *mention_only* is True when *wanted* appears as ``"eval_dir"``
        text but no supported row declares it: evidence that is present but not
        readable as ownership, which the caller reports rather than acts on.
    """
    try:
        text = journal.read_text(errors="replace")
    except OSError:
        return [], False
    claims: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue    # a torn last line of a live journal, not a claim
        if not isinstance(row, dict) or row.get("type") != _JOURNAL_RESULT:
            continue
        result = row.get("result")
        if not isinstance(result, dict):
            continue
        declared = result.get("eval_dir")
        if not isinstance(declared, str) or not declared.strip():
            continue
        key, agent_id = row.get("key"), row.get("agentId")
        key = key.strip() if isinstance(key, str) else ""
        agent_id = agent_id.strip() if isinstance(agent_id, str) else ""
        claims.append({
            "eval_dir": declared.strip().rstrip("/"),
            "key": key,
            "agent_id": agent_id,
            # Both, or neither counts: half an attribution names a producer that
            # cannot be looked up, which is not better evidence than none.
            "provenance": (
                OWNER_DECLARED if (key and agent_id) else OWNER_UNATTRIBUTED),
        })
    if any(c["eval_dir"] == wanted for c in claims):
        return claims, False
    return claims, bool(_eval_dir_mention_re(wanted).search(text))


def _owned_sites(
    homes: Iterable[Path], eval_dir: str | None,
    conflicts: list[tuple[str, str]] | None = None,
    mentions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """:func:`owned_invocations`, keeping each hit's provenance for the mirror.

    Ownership is decided exactly as :func:`owned_invocations` documents -- this is
    that function's body, not a second rule. It additionally carries the two paths
    the copier needs and the reader does not: the ``wf_*.json`` backing the hit
    (``None`` for a journal-owned invocation, which has no record yet) and the
    session dir its transcripts live under, plus, for a journal-owned hit, the
    ``key``/``agentId`` of the row that declared it. The public shape stays three
    keys so a scope dict, which is serialized into the report, does not grow copier
    detail.

    This is the run's FULL inventory and is deliberately unfiltered: a site whose
    transcripts are absent -- or present but zero-byte/unreadable -- is returned
    with ``files_exist=False``, its unusable files named in ``transcripts_empty``
    and ``transcripts_unreadable``, rather than dropped. :func:`owned_invocations` filters to usable sites for its callers;
    dropping the hole here instead would leave the survivors to certify the scope
    as complete, which is the undercount this exists to prevent.

    Args:
        homes: Claude home dirs to search.
        eval_dir: The run directory whose owners are wanted.
        conflicts: Out-param. ``(run_id, other_eval_dir)`` for each journal that
            declares *eval_dir* while its own record assigns it elsewhere; such a
            site is excluded, never silently adopted.
        mentions: Out-param. ``run_id`` for each journal that contains *eval_dir*
            as ``"eval_dir"`` text but declares it in no supported row -- evidence
            present but unreadable as ownership, reported rather than acted on.
    """
    wanted = (eval_dir or "").strip().rstrip("/")
    if not wanted:
        return []
    homes = list(homes)
    found: dict[str, dict[str, Any]] = {}
    # Every eval-dir each runId is on record for. A journal may not override this:
    # a record positively assigning an invocation to another run is stronger
    # evidence than that invocation's journal mentioning this one.
    on_record: dict[str, set[str]] = {}
    for record_path, record in iter_records(homes):
        run_id = str(record.get("runId") or "")
        if run_id:
            on_record.setdefault(run_id, set()).update(
                directory for field, directory in record_paths_typed(record)
                if field == _FIELD_EVAL)
        if _match_rank(record, wanted, _FIELD_EVAL) != _RANK_EXACT_SAME:
            continue
        res = _glob_for_record(record_path, record)
        if res and run_id not in found:
            # Retained even when no agent-*.jsonl exists. The record PROVES this
            # invocation is part of the run; absent transcripts are a hole in the
            # evidence, and a hole has to be reported, not dropped -- dropping it
            # lets the remaining subset certify the whole scope as complete.
            found[run_id] = {
                "run_id": run_id, "glob": res[0], "evidence": EVIDENCE_RECORD,
                # Usability, not mere existence, and from the SAME reader the
                # mirror uses -- a zero-byte transcript is a hole in both halves
                # or the two disagree about the very same invocation.
                "files_exist": bool(res[1]["usable"]),
                "transcripts_empty": list(res[1]["empty"]),
                "transcripts_unreadable": list(res[1]["unreadable"]),
                # <session>/workflows/wf_*.json -> <session>
                "record_path": record_path, "session_dir": record_path.parent.parent,
            }
    for home in homes:
        try:
            journals = sorted(home.glob("projects/*/*/subagents/workflows/*/journal.jsonl"))
        except OSError:
            continue
        for journal in journals:
            run_id = journal.parent.name
            if run_id in found:
                continue
            claims, mention_only = _journal_claims(journal, wanted)
            mine = [c for c in claims if c["eval_dir"] == wanted]
            if not mine:
                if mention_only and mentions is not None:
                    # The path is in the file, but not as anything that declares
                    # ownership. Silence here would read as "no evidence"; this is
                    # evidence we refuse to act on, which is a different state.
                    mentions.append(run_id)
                continue
            known = on_record.get(run_id)
            if known and wanted not in known:
                # The runtime already wrote this invocation's record, and it names a
                # DIFFERENT run. Adopting it here would silently overrule that record
                # and then warn that the invocation has no record at all. Refuse, and
                # let the caller say the evidence disagrees.
                if conflicts is not None:
                    conflicts.append((run_id, sorted(known)[0]))
                continue
            g = str(journal.parent / "agent-*.jsonl")
            _journal_usability = transcript_usability(g)
            found[run_id] = {
                "run_id": run_id, "glob": g, "evidence": EVIDENCE_JOURNAL,
                # Kept on the SAME terms as a record-backed site: a declared owner
                # with no transcripts yet is a hole in the evidence, and dropping it
                # from the inventory is what let the surviving subset certify the
                # whole scope as complete. The public wrapper filters; this does not.
                "files_exist": bool(_journal_usability["usable"]),
                "transcripts_empty": list(_journal_usability["empty"]),
                "transcripts_unreadable": list(_journal_usability["unreadable"]),
                # Which agent's returned result proves this, so a hit can be
                # audited back to its row instead of to "the file matched".
                "owner_key": mine[0]["key"], "owner_agent": mine[0]["agent_id"],
                "owner_provenance": mine[0]["provenance"],
                "declared_eval_dirs": sorted({c["eval_dir"] for c in claims}),
                # <session>/subagents/workflows/<runId>/journal.jsonl -> <session>
                "record_path": None, "session_dir": journal.parents[3],
            }
    return list(found.values())


def _anchor_top_by_exp_root(
    homes: list[Path], eval_dir: str | None,
) -> tuple[str, bool] | str | None:
    """Anchor a top-level run whose OWN ``eval_dir`` is not yet on record.

    A dispatcher (e.g. ``kernel_workflow`` in optimize/author pass-through) emits
    the run report from INSIDE itself, before it returns — so its record has an
    ``args.exp_root`` but no ``result.eval_dir`` yet, and the report's
    ``--eval-dir`` (the lane's generated dir) is not named by any record. The
    dispatcher is still identifiable, though: its ``exp_root`` is a STRICT
    ancestor of that lane eval-dir (``<exp_root>/team_task_*/task``). This finds
    that enclosing dispatcher deliberately:

    - Only a SAME-FIELD ``exp_root`` that is a *proper* prefix of *eval_dir*
      qualifies. Equality does NOT (that is a child lane announcing its own
      experiment root, not an enclosing run — see the missing-parent test), and
      an ``eval_dir`` field never anchors here.
    - A candidate whose OWN ``result.eval_dir`` is already on record and is
      SOMETHING OTHER than the requested target is a completed SIBLING under the
      shared ``exp_root``, not the enclosing dispatcher of *eval_dir*. Its known
      identity contradicts the request, so it is rejected outright — a unique
      visible ancestor is NOT proof of ownership when the target's own record is
      absent (else one finished sibling would silently "own", and mis-bill, an
      absent run). Only records that are genuinely args-only for the target
      (no contradicting ``eval_dir``) may anchor, and even then the caller marks
      the result INFERRED, never a proven whole-run identity.
    - When several candidates nest, the MOST SPECIFIC (longest) ``exp_root``
      wins — the immediate dispatcher, not a grandparent.
    - If two DISTINCT runs tie at that most-specific depth, resolution is
      ambiguous and we refuse (return ``_ANCHOR_AMBIGUOUS``) rather than pick one.

    Returns ``(glob, usability)`` for a unique anchor, ``_ANCHOR_AMBIGUOUS`` on
    a genuine tie, or ``None`` when nothing encloses *eval_dir*.
    """
    cleaned = (eval_dir or "").strip().rstrip("/")
    if not cleaned:
        return None
    # (exp_root_len, record_path, record) for every record whose own exp_root is a
    # strict ancestor of the requested eval-dir AND whose own known eval_dir does
    # not contradict the request.
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for record_path, record in iter_records(homes):
        if not str(record.get("runId") or ""):
            continue
        typed = record_paths_typed(record)
        # A record that KNOWS its own eval_dir, and it is not the target, is a
        # sibling/unrelated run — never the enclosing dispatcher of *eval_dir*.
        own_evals = [d for f, d in typed if f == _FIELD_EVAL]
        if own_evals and cleaned not in own_evals:
            continue
        for field, directory in typed:
            if field == _FIELD_EXP and cleaned.startswith(directory + "/"):
                candidates.append((len(directory), record_path, record))
                break
    if not candidates:
        return None
    best_len = max(c[0] for c in candidates)
    finalists = [c for c in candidates if c[0] == best_len]
    distinct_runs = {str(c[2].get("runId")) for c in finalists}
    if len(distinct_runs) > 1:
        return _ANCHOR_AMBIGUOUS
    _, record_path, record = finalists[0]
    return _glob_for_record(record_path, record)


def nested_lane_dirs(eval_dir: Path | str) -> list[str]:
    """The eval-dirs of this run's nested lanes, from its persisted timeline.

    Each ``nested[]`` entry of ``reports/trace/agent_timeline.json`` carries the
    lane's own ``instance`` -- authoritative, because that is how the dispatcher
    accounts for its lanes. Nesting is never guessed from the filesystem.

    This exists so scope resolution and mirroring discover the SAME instances.
    They used to differ: the resolver was given the lanes and the copier was not,
    so the report counted lane calls the mirror had never copied, and a rebuild
    from that mirror lost them with nothing to say they were missing.

    Returns:
        Lane eval-dirs in timeline order, de-duplicated; empty when the timeline
        is absent or records no nesting. Never raises.
    """
    timeline = Path(eval_dir) / "reports" / "trace" / "agent_timeline.json"
    try:
        data = json.loads(timeline.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[str] = []
    seen: set[str] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for child in (node.get("nested") or []):
            if not isinstance(child, dict):
                continue
            inst = child.get("instance")
            if isinstance(inst, str) and inst and inst not in seen:
                seen.add(inst)
                out.append(inst)
            walk(child)

    walk(data)
    return out


def resolve_run_scope(
    homes: Iterable[Path],
    *,
    eval_dir: str | None = None,
    exp_root: str | None = None,
    nested_eval_dirs: Iterable[str] = (),
) -> dict[str, Any]:
    """Resolve the agent transcripts THIS run owns, with explicit coverage.

    The ledger's default discovery finds transcripts by eval-dir path SUBSTRING
    across every session under the Claude home, then bounds them by a day-window.
    That over-attributes: any OTHER session (a human debugging the run, a second
    workflow, the report driver itself) whose transcript merely mentions the
    eval-dir path is billed to the run, and its late calls blow the window open.
    Scoping instead to each instance's OWN ``subagents/workflows/<runId>/`` dir
    removes that class of contamination by construction — a concurrent session
    never writes into another run's workflow subdir.

    Resolution is by identity, never mtime: the top-level run is matched EXACTLY
    from *eval_dir* / *exp_root*; each nested lane is matched EXACTLY from its own
    eval-dir, passed in via *nested_eval_dirs* (the ``instance`` field of the
    timeline's ``nested[]`` entries — authoritative, because that is how the
    dispatcher accounts for its lanes; we do not guess nesting from the
    filesystem). A single-lane pass-through whose worker shares the parent
    ``runId`` dir resolves to the same glob and is de-duplicated.

    Globs ``agent-*.jsonl`` ONLY — never ``agent-*`` — so the sibling
    ``agent-<id>.output`` symlinks (which point at the full transcript) are not
    pulled in and double-counted.

    Coverage is reported, not assumed. The returned dict carries:

    - ``globs``:     the owned, file-backed glob patterns (empty on fallback)
    - ``scope``:     ``run-scoped`` (top + every lane resolved with files),
                     ``partial`` (top resolved, but a lane is missing/empty), or
                     ``unresolved`` (no trustworthy top-level identity)
    - ``complete``:  ``True`` only when nothing is missing
    - ``requested`` / ``resolved`` / ``missing``: the instance labels in each state
    - ``warnings``:  one human-readable line per missing/empty instance
    - ``invocations``: every workflow invocation counted for the top-level eval-dir
                     (see :func:`owned_invocations`), with the evidence for each

    A whole-run scope is NEVER claimed without a resolved, file-backed top-level
    identity: when the top-level record is absent (or resolves to an empty dir),
    ``globs`` is ``[]`` and ``scope`` is ``unresolved`` so the caller falls back
    to substring discovery rather than billing a lane-only slice as the whole run.
    A lane that cannot be established downgrades the run to ``partial`` — it is
    never silently dropped while still calling the result complete.

    KNOWN LIMITATION: an LLM call the dispatcher makes DIRECTLY in the parent
    session context (outside ``subagents/``) is not in this set and is undercounted.
    For the shipped modes the dispatcher only forwards, so this is ~nil; a future
    mode that reasons in-parent would need its parent-scoped calls added back.
    """
    homes = list(homes)
    # (kind, eval_dir, exp_root, label) — top-level first, then each nested lane.
    requested: list[tuple[str, str | None, str | None, str]] = []
    top_label = (eval_dir or exp_root or "").strip().rstrip("/")
    if top_label:
        requested.append(("top", eval_dir, exp_root, top_label))
    for d in (nested_eval_dirs or []):
        lbl = (d or "").strip().rstrip("/")
        if lbl:
            requested.append(("lane", d, None, lbl))

    globs: list[str] = []
    seen: set[str] = set()
    resolved: list[str] = []
    missing: list[str] = []
    inferred: list[str] = []
    warnings: list[str] = []
    top_ok = False
    top_anchor = "eval_dir"     # how the whole-run anchor was established
    invocations: list[dict[str, str]] = []

    conflicts: list[tuple[str, str]] = []
    mentions: list[str] = []

    for kind, ev, ex, label in requested:
        # The SAME ownership rule for every instance. A nested lane is as much a
        # multi-invocation affair as the top is -- a lane that was killed and
        # re-entered owns both invocations -- and resolving lanes by "newest
        # record wins" silently billed one of them and called the result whole.
        sites = _owned_sites(homes, ev, conflicts, mentions)
        usable = [s for s in sites if s.get("files_exist")]
        if kind == "top":
            # Provenance travels with the invocation. "This invocation is ours on
            # journal evidence" is not auditable on its own; "agent <id> returned
            # this eval-dir under key <k>" is, and the report serializes it.
            invocations = []
            for s in usable:
                inv = {key: s[key] for key in ("run_id", "glob", "evidence")}
                for key in ("owner_key", "owner_agent", "owner_provenance"):
                    if s.get(key):
                        inv[key] = s[key]
                invocations.append(inv)
        if sites:
            # Every invocation that owns this eval-dir, not the one newest record: a resumed or
            # re-entered run is several invocations, and each one's spend is this run's spend.
            for inv in usable:
                if inv["glob"] not in seen:
                    seen.add(inv["glob"])
                    globs.append(inv["glob"])
            holed = sorted(
                (s["run_id"], _hole_note(_site_usability(s)))
                for s in sites
                if s.get("files_exist")
                and (s.get("transcripts_empty") or s.get("transcripts_unreadable")))
            if holed:
                # A sibling that DID flush does not cover for one that did not.
                # Keeping the usable calls is right -- they are real spend -- but
                # the invocation is partially captured, and saying so is the whole
                # point: a zero-byte transcript is an unknown, not a zero.
                if label not in missing:
                    missing.append(label)
                warnings.append(
                    "%s instance %r has partially captured workflow invocation(s) %s — "
                    "their other transcripts were kept, but the unusable ones cannot be "
                    "counted, so this scope is incomplete" % (
                        kind, label,
                        ", ".join("%s [%s]" % (rid, note) for rid, note in holed)))
            unattributed = sorted(
                s["run_id"] for s in usable
                if s.get("evidence") == EVIDENCE_JOURNAL
                and s.get("owner_provenance") != OWNER_DECLARED)
            if unattributed:
                # Adopted, but nobody signed the claim. Said out loud rather than
                # serialized as an owner with two empty strings for an owner.
                warnings.append(
                    "workflow invocation(s) %s declare this eval-dir in a journal row that "
                    "carries no usable ``key``/``agentId`` — ownership is adopted but "
                    "UNATTRIBUTED: it cannot be audited back to the agent that wrote it"
                    % ", ".join(unattributed))
            gone = sorted(
                "%s [%s]" % (s["run_id"], _hole_note(_site_usability(s)) or "no files")
                for s in sites if not s.get("files_exist"))
            if gone:
                # A record PROVES these belong to the run, so their absent
                # transcripts are a known hole. Counting only what survived and
                # calling that complete is exactly the undercount to avoid.
                if label not in missing:
                    missing.append(label)
                warnings.append(
                    "%s instance %r owns workflow invocation(s) %s whose agent-*.jsonl "
                    "transcripts are absent or unusable — their spend cannot be counted, so this "
                    "scope is incomplete" % (kind, label, ", ".join(gone)))
            if not usable:
                continue
            journal_only = [s["run_id"] for s in usable if s["evidence"] == EVIDENCE_JOURNAL]
            if journal_only:
                if kind == "top":
                    top_anchor = (
                        "live-journal" if len(journal_only) == len(usable) else "record+journal")
                inferred.append(label)
                warnings.append(
                    "workflow invocation(s) %s have no workflow record (still running, or ended "
                    "before returning) and were tied to this eval-dir by their journal naming it "
                    "as their eval_dir -- ownership INFERRED, not proven" % ", ".join(journal_only))
            if kind == "top":
                top_ok = True
            if not gone and not holed:
                resolved.append(label)
            continue
        res = _resolve_one(homes, ev, ex)
        if res is None and kind == "top":
            # The run's OWN eval-dir may not be on record yet: a dispatcher emits
            # its report before it returns, so ``result.eval_dir`` is unwritten
            # and only ``args.exp_root`` (a STRICT ancestor of the lane eval-dir)
            # identifies it. Anchor on that enclosing run deliberately — a unique
            # most-specific exp_root only; a genuine tie stays unresolved.
            # NOTE: there is deliberately no looser journal fallback here. A run
            # still in flight is picked up ABOVE, by the one strict rule: its
            # journal must name this eval-dir in an ``"eval_dir": "<path>"``
            # field. The rule this replaced accepted any whole-path mention, so
            # a journal that merely referenced ``<eval>/final_report.md`` — or a
            # descendant, or an unrelated longer path ending in ``<eval>`` —
            # became an owned, `owned=True` scope. Matching a mention is not
            # evidence of ownership, and nothing downstream could tell the two
            # apart once the scope said ``run-scoped-inferred``.
            anchored = _anchor_top_by_exp_root(homes, ev)
            if anchored == _ANCHOR_AMBIGUOUS:
                warnings.append(
                    "top-level run identity %r matches more than one enclosing "
                    "exp_root run — refusing to guess; falling back" % label)
            elif anchored is not None:
                # Containment is EVIDENCE, not proof: the enclosing run's exp_root
                # nests the target, but no record positively ties it to this
                # report's eval-dir (the target's own record is absent). Scope to
                # it so we avoid the substring-discovery contamination, but mark
                # the result INFERRED/incomplete — never a proven whole-run
                # identity — and say so, so the report cannot read as authoritative.
                res = anchored
                top_anchor = "exp_root-ancestor"
                inferred.append(label)
                warnings.append(
                    "top-level run identity %r resolved only by exp_root "
                    "containment of an enclosing run (the run's own eval_dir is "
                    "not on record) — scope INFERRED, ownership not proven" % label)
        if res is None:
            missing.append(label)
            warnings.append(
                "%s instance %r has no owning workflow record — excluded from scope"
                % (kind, label))
            continue
        g, usability = res
        if not usability["usable"]:
            missing.append(label)
            note = _hole_note(usability)
            warnings.append(
                "%s instance %r resolved to %s but no usable agent-*.jsonl files exist%s"
                % (kind, label, g, " — %s" % note if note else ""))
            continue
        if usability["empty"] or usability["unreadable"]:
            # Scope to it, count what flushed, and still say the capture has a
            # hole. Letting the usable siblings certify the instance is exactly
            # how a report reached "complete, no warnings" over a known gap.
            missing.append(label)
            warnings.append(
                "%s instance %r is partially captured: %s — those calls cannot be "
                "counted, so this scope is incomplete" % (kind, label, _hole_note(usability)))
            partial_capture = True
        else:
            partial_capture = False
        if kind == "top":
            top_ok = True
        if g not in seen:
            seen.add(g)
            globs.append(g)
        if not partial_capture:
            resolved.append(label)

    for run_id, elsewhere in sorted(set(conflicts)):
        # Evidence that disagrees is not evidence to pick from.
        warnings.append(
            "workflow invocation %s names this eval-dir in its journal but its workflow "
            "record assigns it to %r — refusing to override the record; excluded from scope"
            % (run_id, elsewhere))

    for run_id in sorted(set(mentions)):
        # Reported, not resolved. The path is in the file but not as a declaration,
        # so we can neither claim the invocation nor say the evidence is absent.
        warnings.append(
            "workflow invocation %s mentions this eval-dir but declares no ``result.eval_dir`` "
            "for it — a mention is not a claim of ownership; excluded from scope, and its "
            "spend (if any) is therefore uncounted" % run_id)

    req_labels = [r[3] for r in requested]
    if not top_ok:
        # No trustworthy whole-run anchor: return no globs so the caller falls
        # back to substring discovery instead of billing a lane-only slice.
        if top_label:
            warnings.append(
                "top-level run identity %r unresolved — cannot claim a whole-run "
                "scope; falling back to substring discovery" % top_label)
        return {
            "globs": [],
            "scope": "unresolved",
            "complete": False,
            "requested": req_labels,
            "resolved": resolved,
            "missing": missing,
            "inferred": inferred,
            "warnings": warnings,
            "top_anchor": None,
            "invocations": [],
        }

    # An inferred (exp_root-containment) top is resolved enough to scope, but its
    # ownership is unproven — so it is never ``complete`` and never a clean
    # ``run-scoped``. It surfaces as ``run-scoped-inferred`` so the caller still
    # uses the owned globs (avoiding substring contamination) while the report
    # states plainly that the whole-run identity was inferred, not established.
    complete = not missing and not inferred
    if complete:
        scope = "run-scoped"
    elif inferred and not missing:
        scope = "run-scoped-inferred"
    else:
        scope = "partial"
    return {
        "globs": globs,
        "scope": scope,
        "complete": complete,
        "requested": req_labels,
        "resolved": resolved,
        "missing": missing,
        "inferred": inferred,
        "warnings": warnings,
        "top_anchor": top_anchor,
        "invocations": invocations,
    }


def run_transcript_globs(
    homes: Iterable[Path],
    *,
    eval_dir: str | None = None,
    exp_root: str | None = None,
    nested_eval_dirs: Iterable[str] = (),
) -> list[str]:
    """Backward-compatible thin wrapper: the owned globs only (no coverage).

    Prefer :func:`resolve_run_scope`, which also reports partial coverage and the
    reason a slice is missing. Retained so existing callers keep resolving to the
    run's own ``subagents/workflows/<runId>/`` dirs.
    """
    return resolve_run_scope(
        homes, eval_dir=eval_dir, exp_root=exp_root,
        nested_eval_dirs=nested_eval_dirs)["globs"]


# --------------------------------------------------------------------------- #
# Copying
# --------------------------------------------------------------------------- #
def _sources(record_path: Path, run_id: str) -> list[tuple[Path, Path]]:
    """List the files to mirror, most important first.

    Priority matters: under a byte budget the record must survive even if the
    transcripts cannot, because a record alone still yields the phase and agent
    tree.

    Observed sessions put per-agent transcripts in two shapes: under
    ``subagents/workflows/<runId>/`` when a Workflow tool ran, and flat in
    ``subagents/`` when none did. Both are taken; other runs' workflow subdirs
    are not, because they belong to their own mirrors.

    Args:
        record_path: Path of the ``wf_*.json`` record inside a Claude home.
        run_id: The record's ``runId``.

    Returns:
        ``(source, relative_destination)`` pairs. The destination is relative to
        the mirror root and reproduces the home's own layout.
    """
    return _sources_at(record_path.parent.parent, run_id, record_path=record_path)


def _sources_at(
    session_dir: Path, run_id: str, *, record_path: Path | None = None,
) -> list[tuple[Path, Path]]:
    """:func:`_sources` for ONE invocation, addressed by its session dir.

    Split out so an invocation owned by its live journal can be copied on the same
    terms as a record-backed one. Such an invocation has no ``wf_*.json`` at all --
    the runtime writes the record only when the workflow returns -- so the record
    is optional here rather than the starting point.

    Args:
        session_dir: ``<home>/projects/<slug>/<session>``.
        run_id: The invocation whose ``subagents/workflows/<runId>/`` to take.
        record_path: Its ``wf_*.json`` when one exists; copied first so the record
            survives a byte budget that the transcripts do not.

    Returns:
        ``(source, relative_destination)`` pairs, most important first.
    """
    # <home>/projects/<slug>/<session> -> <home>. The mirror reproduces the
    # full relative path from the home down, because the consumer's glob is
    # anchored at ``projects/`` and counts directory levels.
    home = session_dir.parent.parent.parent

    def rel(path: Path) -> Path:
        return path.relative_to(home)

    out: list[tuple[Path, Path]] = []
    # Listed even when it is already gone: transcripts are live files, and a
    # record that vanishes between listing and copy must surface as an error in
    # the manifest, not as a silently shorter source list.
    if record_path is not None:
        out.append((record_path, rel(record_path)))

    convo = session_dir.with_suffix(".jsonl")
    if convo.is_file():
        out.append((convo, rel(convo)))

    subagents = session_dir / "subagents"
    found: list[Path] = []
    # This run's own transcripts. A session can drive several runs; the others'
    # transcripts belong to their own mirrors, not to this one.
    if run_id:
        try:
            found.extend(p for p in (subagents / "workflows" / run_id).rglob("*") if p.is_file())
        except OSError:
            pass
    # A session that ran no Workflow tool has flat ``subagents/agent-*.jsonl``
    # and no ``workflows/`` dir at all. Take those too rather than branching on
    # a layout we do not control.
    try:
        found.extend(p for p in subagents.glob("*") if p.is_file())
    except OSError:
        pass
    out.extend((path, rel(path)) for path in sorted(set(found)))
    return out


def _is_current(src: Path, dest: Path) -> bool:
    """Report whether *dest* is already an up-to-date copy of *src*.

    Lets a repeated pass re-copy only what grew, which is what makes a mid-run
    mirror cheap enough to run on a timer.

    Args:
        src: Source file.
        dest: Mirrored file.

    Returns:
        ``True`` when size and mtime match.
    """
    try:
        a, b = src.stat(), dest.stat()
    except OSError:
        return False
    return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)


def mirror(
    record_path: Path,
    record: dict[str, Any],
    dest_root: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    deadline_s: float = DEFAULT_DEADLINE_S,
) -> dict[str, Any]:
    """Copy one run's ledger into *dest_root*.

    Copies in place and skips files that have not changed, so a repeated pass
    costs only what grew. That is what makes a mid-run mirror affordable: a
    full stage-and-swap of hundreds of megabytes every quarter hour would not
    be, and the transcripts this reads are append-only.

    An oversized file is skipped and named in the manifest rather than
    truncated. A truncated ``wf_*.json`` fails ``json.load`` in the consumer,
    and a truncated transcript silently understates a token total — a missing
    file is at least honestly missing.

    Args:
        record_path: The ``wf_*.json`` to mirror, inside a Claude home.
        record: Its parsed content.
        dest_root: The mirror root, normally ``<eval_dir>/llm_trace``.
        max_bytes: Ceiling on bytes copied in this pass. Files past the ceiling
            are listed in the manifest rather than silently dropped.
        deadline_s: Wall-clock ceiling. This runs inside ``_emit``, which may be
            executing under a SIGTERM grace period, so it must be able to give
            up: ``result.json`` is never delayed by more than this.

    Returns:
        The manifest written alongside the copy.
    """
    return mirror_invocations(
        [{
            "run_id": str(record.get("runId") or ""),
            "evidence": EVIDENCE_RECORD,
            "record_path": record_path,
            # <session>/workflows/wf_*.json -> <session>
            "session_dir": record_path.parent.parent,
        }],
        dest_root,
        max_bytes=max_bytes,
        deadline_s=deadline_s,
        recorded_paths=record_paths(record),
    )


def mirror_invocations(
    sites: Iterable[dict[str, Any]],
    dest_root: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    deadline_s: float = DEFAULT_DEADLINE_S,
    recorded_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Copy EVERY invocation that owns this run into *dest_root*, under one manifest.

    A run's spend is the union of the invocations that worked in its eval dir --
    a resumed run, or one re-entered to finish what a killed one left, is several.
    :func:`resolve_run_scope` has counted them all since 824db57; the mirror still
    copied the single invocation :func:`find_record` returned. So a scope that
    resolved to two invocations live rebuilt from the mirror as one, and called
    itself complete: the original's transcripts were lost with the live home, which
    is the one thing a durable mirror exists to prevent.

    The budget and the deadline are shared across the whole set rather than granted
    per invocation, so the ceiling the caller asked for still holds. Invocations are
    copied in the order given, so the caller decides what survives a tight budget.

    A session that drove two invocations shares one conversation file and one flat
    ``subagents/`` dir; those are copied once. The per-invocation
    ``subagents/workflows/<runId>/`` trees never overlap.

    An invocation whose transcripts are missing, empty or unreadable is RETAINED in
    the manifest with a status saying so, and named in ``coverage``. Dropping it
    would let a nonempty subset certify the whole known scope as complete -- the
    same silent-undercount this function exists to fix, one layer down.

    Args:
        sites: ``{"run_id", "session_dir", "record_path", "evidence"}`` per owning
            invocation, as :func:`_owned_sites` returns them.
        dest_root: The mirror root, normally ``<eval_dir>/llm_trace``.
        max_bytes: Ceiling on bytes copied in this pass, across all invocations.
        deadline_s: Wall-clock ceiling, across all invocations.
        recorded_paths: The primary record's declared paths, for the manifest.

    Returns:
        The manifest written alongside the copy.
    """
    sites = list(sites)
    primary = sites[0] if sites else None
    manifest: dict[str, Any] = {
        "run_id": str(primary.get("run_id") or "") if primary else "",
        # <home>/projects/<slug>/<session> -> <home>
        "source_home": str(primary["session_dir"].parents[2]) if primary else "",
        "source_record": str(primary.get("record_path") or "") if primary else "",
        "recorded_paths": list(recorded_paths or []),
        "mirrored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [],
        "skipped": [],
        "errors": [],
        "bytes_copied": 0,
        "max_bytes": int(max_bytes),
        "deadline_hit": False,
        "invocations": [],
    }

    budget = int(max_bytes)
    started = time.monotonic()
    seen: set[str] = set()
    for site in sites:
        run_id = str(site.get("run_id") or "")
        record_path = site.get("record_path")
        entry: dict[str, Any] = {
            "run_id": run_id,
            "evidence": str(site.get("evidence") or ""),
            "source_record": str(record_path) if record_path else "",
            "files": [],
            "skipped": [],
            "errors": [],
            "bytes_copied": 0,
            "status": "ok",
        }
        manifest["invocations"].append(entry)
        try:
            pairs = _sources_at(site["session_dir"], run_id, record_path=record_path)
        except (OSError, ValueError, KeyError) as exc:
            entry["status"] = "unreadable"
            err = {"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"}
            entry["errors"].append(err)
            manifest["errors"].append(err)
            continue
        # Counted before de-duplication: the shared files a second invocation
        # skips are not evidence that IT was captured.
        own = "subagents/workflows/%s/" % run_id if run_id else None
        mine = [(src, rel) for src, rel in pairs if own and own in rel.as_posix()]
        # Only an agent-*.jsonl with bytes in it is evidence that this invocation's
        # calls were captured. Counting every file under the workflow dir counted
        # the journal and the metadata as transcripts, so an invocation with a
        # journal and no agent transcript at all reported "1 transcript, complete";
        # and a zero-byte transcript -- a file created for an agent that never
        # flushed -- counted the same as a full one.
        # The SAME classifier the scope resolver reads, so the manifest and the
        # report cannot disagree about whether an invocation was captured.
        split = {"usable": [], "empty": [], "unreadable": []}
        for src, rel in mine:
            if not is_transcript_name(rel.name):
                continue
            split[classify_transcript(src)].append(rel.as_posix())
        usable, empty, unreadable = (
            split["usable"], split["empty"], split["unreadable"])
        entry["files_seen"] = len(mine)
        entry["transcripts"] = len(usable)
        entry["transcripts_empty"] = sorted(empty)
        entry["transcripts_unreadable"] = sorted(unreadable)
        for src, rel in pairs:
            key = rel.as_posix()
            if key in seen:
                continue
            seen.add(key)
            if deadline_s > 0 and time.monotonic() - started >= deadline_s:
                manifest["deadline_hit"] = True
                rec = {"path": key, "reason": "deadline"}
                entry["skipped"].append(rec)
                manifest["skipped"].append(rec)
                continue
            dest = dest_root / rel
            try:
                size = src.stat().st_size
            except OSError as exc:
                rec = {"path": str(src), "error": f"{type(exc).__name__}: {exc}"}
                entry["errors"].append(rec)
                manifest["errors"].append(rec)
                continue
            if _is_current(src, dest):
                rec = {"path": key, "bytes": size, "copied": False}
                entry["files"].append(rec)
                manifest["files"].append(rec)
                continue
            if size > budget:
                rec = {"path": key, "bytes": size, "reason": "max_bytes"}
                entry["skipped"].append(rec)
                manifest["skipped"].append(rec)
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            except OSError as exc:
                rec = {"path": str(src), "error": f"{type(exc).__name__}: {exc}"}
                entry["errors"].append(rec)
                manifest["errors"].append(rec)
                continue
            budget -= size
            entry["bytes_copied"] += size
            manifest["bytes_copied"] += size
            rec = {"path": key, "bytes": size, "copied": True}
            entry["files"].append(rec)
            manifest["files"].append(rec)
        if entry["status"] == "ok":
            if not entry.get("transcripts"):
                # Distinguish "nothing was written" from "something was written and
                # cannot be read". Neither is proof the invocation spent nothing --
                # missing evidence is missing, not zero -- so both leave the run
                # incomplete rather than letting the captured subset certify it.
                entry["status"] = (
                    "unusable_transcripts"
                    if (entry.get("transcripts_empty") or entry.get("transcripts_unreadable"))
                    else "no_transcripts")
            elif entry["transcripts_empty"] or entry["transcripts_unreadable"]:
                # Some of this invocation's agents flushed and some did not. The
                # calls that WERE captured stay counted, but a usable sibling is
                # not evidence about the file next to it: the invocation is
                # partially captured, and the run's coverage is not complete.
                entry["status"] = "partial_transcripts"
            elif entry["skipped"]:
                entry["status"] = "partial"
            elif entry["errors"]:
                entry["status"] = "errors"

    incomplete = [e["run_id"] for e in manifest["invocations"] if e["status"] != "ok"]
    manifest["coverage"] = {
        "invocations": len(sites),
        "captured": len(sites) - len(incomplete),
        "complete": bool(sites) and not incomplete,
        "incomplete": incomplete,
    }

    _write_manifest(dest_root, manifest)
    return manifest


def _write_manifest(dest_root: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest atomically, swallowing any IO failure.

    Args:
        dest_root: The mirror root.
        manifest: The manifest to serialize.
    """
    target = dest_root / MANIFEST_NAME
    tmp = target.with_suffix(".json.tmp")
    try:
        dest_root.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Durability warning
# --------------------------------------------------------------------------- #
def warn_if_volatile(home: Path, exp_root: Path) -> str | None:
    """Return a warning when the ledger's filesystem differs from the run's.

    The test is a device-id comparison and nothing cleverer. ``exp_root`` is by
    construction this run's durable output location, so a ledger on a different
    device is a ledger with a different lifetime — which is exactly the failure
    this module exists for. Sniffing ``/proc/mounts`` for overlayfs or matching
    on ``/root`` produces false positives on a legitimately persistent home.

    Args:
        home: The resolved Claude Code home.
        exp_root: This run's experiment root.

    Returns:
        The warning text, or ``None`` when the two share a filesystem or the
        comparison cannot be made.
    """
    try:
        home_dev = os.stat(home).st_dev
        root_dev = os.stat(exp_root).st_dev
    except OSError:
        return None
    if home_dev == root_dev:
        return None
    return (
        f"Claude Code's LLM ledger is on a different filesystem from this run's "
        f"output (home={home} dev={home_dev}, exp_root={exp_root} dev={root_dev}).\n"
        f"         If that filesystem is a container overlay it dies with the container, "
        f"taking the run's entire token/cost record with it.\n"
        f"         This run will mirror the ledger into <eval_dir>/{MIRROR_DIRNAME}/ as it "
        f"goes, so the copy survives even if the original does not.\n"
        f"         To keep the originals too, set CLAUDE_CONFIG_DIR to a durable path in the "
        f"launching shell BEFORE starting Claude Code — it is read at session start only, "
        f"so exporting it later has no effect."
    )


# --------------------------------------------------------------------------- #
# Rendering (best effort)
# --------------------------------------------------------------------------- #
def _model_name(eval_dir: Path) -> str:
    """The model this run optimized, named the way the run itself named it.

    Three sources in falling order of authority, because a report filename must
    never be the reason a run's telemetry step fails:

    1. ``kb_identity.json`` -> ``dims.model``. This is the canonical identity the
       run registers itself under in the knowledge base, so it is the same string
       used to match this run against previous ones.
    2. ``env_report.json`` -> ``model``, which is a filesystem path; its basename
       is the model directory name.
    3. The trailing part of the eval dir name, which is ``e2e_<model>_<stamp>``.

    Returns:
        A filename-safe name, or ``"run"`` when nothing on disk identifies the
        model. Never raises.
    """
    candidates: list[str] = []
    try:
        kb = json.loads((eval_dir / "kb_identity.json").read_text(encoding="utf-8"))
        candidates.append(str((kb.get("dims") or {}).get("model") or ""))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    try:
        env = json.loads((eval_dir / "env_report.json").read_text(encoding="utf-8"))
        candidates.append(Path(str(env.get("model") or "")).name)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    name = eval_dir.name
    if name.startswith("e2e_"):
        # e2e_<model>_<date>_<time>_<pid>_<n> -- strip the four trailing stamp fields.
        candidates.append("_".join(name[4:].split("_")[:-4]))
    for candidate in candidates:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-.")
        if safe:
            return safe
    return "run"


SKILL_RELPATH = Path("e2e_workflow") / "knowledge" / "analysis_skills" / "run-report" / "SKILL.md"


def install_skill(out_dir: Path) -> dict[str, Any]:
    """Drop the report-building skill beside the reports it describes.

    A report is only as useful as the instructions for rebuilding and reading
    it, and those instructions must travel with the artifacts — a run archived
    to shared storage months later has no checkout beside it.

    Args:
        out_dir: The run's ``reports`` directory.

    Returns:
        A status dict. Never raises.
    """
    src = Path(__file__).resolve().parent.parent / SKILL_RELPATH
    try:
        if not src.is_file():
            return {"status": "skipped", "reason": f"skill not found at {src}"}
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "SKILL.md"
        shutil.copy2(src, dest)
    except OSError as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "path": str(dest)}


def render_run_report(eval_dir: Path, homes: Iterable[Path] = ()) -> dict[str, Any]:
    """Re-render the run's one report page with GEAK's own driver, ``interface/geak_report.py``.

    The workflow already renders ``<eval_dir>/report/geak_run_report_<model>.html`` as its last
    step, but from INSIDE the run, before the runtime has written the run's workflow record.
    Rendered again here, after the workflow returned, the same page can resolve the run's scope
    from its record. Everything runs in-process from this checkout: GEAK depends on no external
    renderer.

    Args:
        eval_dir: The run directory; the page lands in its ``report/``.
        homes: Extra Claude homes to search (e.g. this run's mirror), after the standard ones.

    Returns:
        A status dict. Never raises.
    """
    try:
        import sys
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        import geak_report
        res = geak_report.run(eval_dir=str(eval_dir), extra_homes=[str(h) for h in homes])
    except Exception as exc:  # never let telemetry kill a run
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    out = {"status": res.get("status", "error")}
    for key in ("html", "md", "transcript_scope", "reason"):
        if res.get(key):
            out[key] = res[key]
    return out


# --------------------------------------------------------------------------- #
# Entry point used by the runner
# --------------------------------------------------------------------------- #
def _max_bytes() -> int:
    """Resolve the per-pass byte budget from the environment.

    Returns:
        The budget in bytes, falling back to :data:`DEFAULT_MAX_BYTES`.
    """
    raw = os.environ.get("GEAK_TRACE_MIRROR_MAX_MB", "").strip()
    try:
        value = int(float(raw))
    except ValueError:
        return DEFAULT_MAX_BYTES
    return value * 1024 * 1024 if value > 0 else DEFAULT_MAX_BYTES


def mirror_run_trace(
    eval_dir: Path | str,
    *,
    exp_root: Path | str | None = None,
    session_id: str | None = None,
    homes: Iterable[Path] | None = None,
    nested_eval_dirs: Iterable[str] | None = None,
    render: bool = False,
) -> dict[str, Any]:
    """Mirror this run's Claude ledger into ``<eval_dir>/llm_trace``.

    The single call the runner makes. Safe to call repeatedly: unchanged files
    are not re-copied.

    Args:
        eval_dir: This run's eval dir — the mirror's parent.
        exp_root: This run's experiment root, used to find the record when
            ``eval_dir`` is not what the record wrote down.
        session_id: The SDK session id, used only to disambiguate.
        homes: Override the searched homes (tests).
        nested_eval_dirs: This run's lane eval-dirs. ``None`` reads them from the
            run's own timeline via :func:`nested_lane_dirs`, so the copier covers
            the same instances the resolver does without the caller having to
            know about lanes; pass a list to override, ``()`` to mirror the top
            invocations only.
        render: Also re-render the run's report page (``report/``) with GEAK's
            own driver, and drop the run-report skill beside it.

    Returns:
        A status dict carrying ``path`` on success. Never raises.
    """
    try:
        eval_path = Path(eval_dir)
        search = list(homes) if homes is not None else candidate_homes()
        # The record this run declares stays the primary: it keeps the manifest's
        # top-level identity, and it is first in line for the byte budget. Every
        # OTHER invocation that owns this eval dir is then mirrored too, so what
        # the report counts and what the mirror preserves are the same set.
        sites: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        record: dict[str, Any] | None = None
        hit = find_record(
            search,
            eval_dir=str(eval_path),
            exp_root=str(exp_root) if exp_root else None,
            session_id=session_id,
        )
        if hit is not None:
            record_path, record = hit
            run_id = str(record.get("runId") or "")
            seen_ids.add(run_id)
            sites.append({
                "run_id": run_id,
                "evidence": EVIDENCE_RECORD,
                "record_path": record_path,
                "session_dir": record_path.parent.parent,
            })
        # Top-level owners, then every nested lane's owners. Mirroring only the
        # top eval-dir left a lane's agent-*.jsonl out of the mirror entirely
        # while the manifest still said coverage was complete -- the resolver had
        # already learned to union the lanes, and the copier had not, so the calls
        # the report counted from live sources were simply absent once the mirror
        # was the only source left.
        lanes = (list(nested_eval_dirs) if nested_eval_dirs is not None
                 else nested_lane_dirs(eval_path))
        for instance in [str(eval_path)] + [ln for ln in lanes if ln]:
            for site in _owned_sites(search, instance):
                run_id = str(site.get("run_id") or "")
                if run_id in seen_ids:
                    continue
                seen_ids.add(run_id)
                sites.append(site)
        if not sites:
            return {"status": "no_record", "homes": [str(h) for h in search]}
        dest = eval_path / MIRROR_DIRNAME
        manifest = mirror_invocations(
            sites, dest, max_bytes=_max_bytes(),
            recorded_paths=record_paths(record) if record else None,
        )
        result: dict[str, Any] = {
            "status": "ok",
            "path": str(dest),
            "run_id": manifest["run_id"],
            "files": len(manifest["files"]),
            "bytes_copied": manifest["bytes_copied"],
            "invocations": len(manifest["invocations"]),
            "coverage": manifest["coverage"],
        }
        if render:
            # Render against the mirror that was just written, as well as the
            # live homes: the page must be reproducible from the durable copy
            # alone, and passing it here is what proves the copy is sufficient.
            report = render_run_report(eval_path, homes=[dest, *search])
            skill = install_skill(eval_path / "report")
            manifest["report"] = report
            manifest["skill"] = skill
            _write_manifest(dest, manifest)
            result["report"] = report
            result["skill"] = skill
        return result
    except Exception as exc:  # never let telemetry kill a run
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
