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

Layout is dictated by the consumer, not chosen freely. Hyperloom's
``dump_geak_call_report`` discovers runs with the glob
``projects/*/*/workflows/wf_*.json`` and then derives everything else RELATIVE
to the record it found — per-agent transcripts at
``<record>/../../subagents/workflows/<runId>/``, the orchestrator conversation at
``<session_dir>.jsonl``. The slug and session names are wildcards and are never
parsed; only the depth matters, and there is no flat-directory escape hatch. So
the mirror reproduces that shape verbatim and is readable with
``--claude-home <eval_dir>/llm_trace --eval-dir <eval_dir>``. Both halves are
needed: ``--claude-home`` appends a search root, it does not select a run, and
the reader refuses to run without a selector. The record is copied byte for byte
precisely so the eval dir it names still selects it from the mirror.

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
import subprocess
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

    A Hyperloom-driven run is identified by ``eval_dir``; a standalone one by
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

    Returns ``(glob, files_exist)`` when a record EXACTLY owns the instance and
    carries a ``runId``; ``None`` when no owning record exists. ``files_exist``
    reflects ``glob.glob`` on disk, so the caller can flag a resolved-but-empty
    lane rather than trust a nonempty glob STRING as proof of coverage.
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


def _glob_for_record(
    record_path: Path, record: dict[str, Any]
) -> tuple[str, bool] | None:
    """Build the ``agent-*.jsonl`` glob for *record*'s own run dir.

    Returns ``(glob, files_exist)`` or ``None`` when the record has no ``runId``.
    """
    run_id = str(record.get("runId") or "")
    if not run_id:
        return None
    session_dir = record_path.parent.parent
    g = str(session_dir / "subagents" / "workflows" / run_id / "agent-*.jsonl")
    return g, bool(_glob.glob(g))



def _anchor_top_by_live_journal(
    homes: list[Path], eval_dir: str | None,
) -> tuple[str, bool] | str | None:
    """Anchor a run whose workflow record does not exist YET.

    ``_anchor_top_by_exp_root`` assumes a record is on disk carrying
    ``args.exp_root``. For a report emitted from inside the run that assumption
    fails outright: the runtime CREATES ``workflows/wf_<runId>.json`` when the
    workflow returns, so during the run there is no record to rank, anchor or
    own anything — every record-based path is unresolvable by construction.

    The run's own ``subagents/workflows/<runId>/journal.jsonl`` is written from
    the first agent onward, and the agents record the eval-dir they work in. So
    a journal that NAMES the requested eval-dir identifies the run positively,
    from live evidence, while the run is still going.

    This is a stronger signal than exp_root containment (which is mere
    ancestry), but it is still not the run's own record, so the caller marks the
    result INFERRED and never ``complete``.

    - Only an EXACT eval-dir path occurrence counts; a prefix of a longer path
      (``/x/eval`` inside ``/x/eval-2``) does not, or a sibling run whose dir
      merely starts the same way would answer for this one.
    - A run dir with no ``agent-*.jsonl`` is not adopted: a journal alone
      carries no billable calls, and claiming it would report an empty ledger as
      a scoped one.
    - If two DISTINCT run dirs name the same eval-dir (a nested dispatcher and
      its lane, each with its own runId), that is a genuine tie: refuse rather
      than bill one run's transcripts as the other's.
    """
    wanted = (eval_dir or "").strip().rstrip("/")
    if not wanted:
        return None
    hits: dict[str, tuple[str, bool]] = {}
    for home in homes:
        for journal in home.glob("projects/*/*/subagents/workflows/*/journal.jsonl"):
            run_dir = journal.parent
            try:
                text = journal.read_text(errors="replace")
            except OSError:
                continue
            if not _names_path(text, wanted):
                continue
            g = str(run_dir / "agent-*.jsonl")
            if not _glob.glob(g):
                continue
            hits[run_dir.name] = (g, True)
    if not hits:
        return None
    if len(hits) > 1:
        return _ANCHOR_AMBIGUOUS
    return next(iter(hits.values()))


def _names_path(text: str, wanted: str) -> bool:
    """Whether *text* contains *wanted* as a whole path, not as a prefix.

    A bare substring test would let ``/runs/eval`` match ``/runs/eval-2``, so an
    occurrence only counts when what follows it cannot extend the path — i.e.
    end of text, a separator, or any character that cannot appear mid-path.
    """
    start = 0
    while True:
        i = text.find(wanted, start)
        if i < 0:
            return False
        nxt = text[i + len(wanted):i + len(wanted) + 1]
        if nxt == "" or nxt in '"\'\\ \t\r\n,}]:' or nxt == "/":
            return True
        start = i + 1

# Sentinel: more than one enclosing run could anchor a mid-run eval-dir, so we
# refuse to guess rather than bill the wrong run's transcripts.
_ANCHOR_AMBIGUOUS = "ambiguous"


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

    Returns ``(glob, files_exist)`` for a unique anchor, ``_ANCHOR_AMBIGUOUS`` on
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

    for kind, ev, ex, label in requested:
        res = _resolve_one(homes, ev, ex)
        if res is None and kind == "top":
            # The run's OWN eval-dir may not be on record yet: a dispatcher emits
            # its report before it returns, so ``result.eval_dir`` is unwritten
            # and only ``args.exp_root`` (a STRICT ancestor of the lane eval-dir)
            # identifies it. Anchor on that enclosing run deliberately — a unique
            # most-specific exp_root only; a genuine tie stays unresolved.
            anchored = _anchor_top_by_exp_root(homes, ev)
            if anchored is None:
                # No record anchored it — which is the NORMAL state for a report
                # emitted from inside its own run, because the runtime creates
                # the workflow record only when the workflow returns. Fall back
                # to the live journal, which exists from the first agent on.
                anchored = _anchor_top_by_live_journal(homes, ev)
                if anchored is not None and anchored != _ANCHOR_AMBIGUOUS:
                    res = anchored
                    top_anchor = "live-journal"
                    inferred.append(label)
                    warnings.append(
                        "top-level run identity %r resolved from the live "
                        "journal naming this eval-dir (the run's own workflow "
                        "record is not written until the run returns) — scope "
                        "INFERRED, ownership not proven" % label)
                    anchored = None
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
        g, files_exist = res
        if not files_exist:
            missing.append(label)
            warnings.append(
                "%s instance %r resolved to %s but no agent-*.jsonl files exist"
                % (kind, label, g))
            continue
        if kind == "top":
            top_ok = True
        if g not in seen:
            seen.add(g)
            globs.append(g)
        resolved.append(label)

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
    session_dir = record_path.parent.parent
    # <home>/projects/<slug>/<session> -> <home>. The mirror reproduces the
    # full relative path from the home down, because the consumer's glob is
    # anchored at ``projects/`` and counts directory levels.
    home = session_dir.parent.parent.parent

    def rel(path: Path) -> Path:
        return path.relative_to(home)

    out: list[tuple[Path, Path]] = [(record_path, rel(record_path))]

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
    run_id = str(record.get("runId") or "")
    # <home>/projects/<slug>/<session>/workflows/wf_*.json -> <home>
    source_home = record_path.parents[4]
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "source_home": str(source_home),
        "source_record": str(record_path),
        "recorded_paths": record_paths(record),
        "mirrored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [],
        "skipped": [],
        "errors": [],
        "bytes_copied": 0,
        "max_bytes": int(max_bytes),
        "deadline_hit": False,
    }

    budget = int(max_bytes)
    started = time.monotonic()
    for src, rel in _sources(record_path, run_id):
        if deadline_s > 0 and time.monotonic() - started >= deadline_s:
            manifest["deadline_hit"] = True
            manifest["skipped"].append({"path": str(rel), "reason": "deadline"})
            continue
        dest = dest_root / rel
        try:
            size = src.stat().st_size
        except OSError as exc:
            manifest["errors"].append({"path": str(src), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if _is_current(src, dest):
            manifest["files"].append({"path": str(rel), "bytes": size, "copied": False})
            continue
        if size > budget:
            manifest["skipped"].append({"path": str(rel), "bytes": size, "reason": "max_bytes"})
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        except OSError as exc:
            manifest["errors"].append({"path": str(src), "error": f"{type(exc).__name__}: {exc}"})
            continue
        budget -= size
        manifest["bytes_copied"] += size
        manifest["files"].append({"path": str(rel), "bytes": size, "copied": True})

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
def _report_command(mirror_root: Path, out_dir: Path, eval_dir: Path) -> list[str] | None:
    """Build the command that renders a report from the mirror, if available.

    GEAK cannot import Hyperloom, so the renderer is reached by subprocess when
    a checkout happens to be present and skipped entirely when it is not. The
    raw mirror is the durable artifact; the rendered report is a convenience.

    ``--claude-home`` *appends* a search root rather than replacing the default
    ones, and the tool refuses to run without a selector, so ``--eval-dir`` is
    not optional here: without it the renderer exits 2 having written nothing.
    The mirrored record keeps the eval dir it was written with -- the copy is
    verbatim, deliberately -- so this run's own eval dir is what selects it.

    ``--include-text`` is likewise required rather than a nicety: it is what
    writes ``geak_calls.jsonl``, and the HTML report that runs next reads only
    that file.

    Args:
        mirror_root: The mirror directory to read.
        out_dir: Where the report should be written.
        eval_dir: This run's eval dir, which selects its record.

    Returns:
        An argv list, or ``None`` when no renderer can be located.
    """
    tail = [
        "--claude-home", str(mirror_root),
        "--eval-dir", str(eval_dir),
        "--include-text",
        "--output-dir", str(out_dir),
    ]
    override = os.environ.get("GEAK_LLM_REPORT_CMD", "").strip()
    if override:
        return override.split() + tail
    return _hyperloom_command(REPORT_MODULE, tail)


HTML_MODULE = "hyperloom.inference_optimizer.tools.render_geak_html_report"
REPORT_MODULE = "hyperloom.inference_optimizer.tools.dump_geak_call_report"


def _hyperloom_command(module: str, tail: list[str]) -> list[str] | None:
    """Build an argv that runs a Hyperloom tool out of ``HYPERLOOM_SRC``.

    GEAK cannot import Hyperloom, so its tools are reached by subprocess when a
    checkout happens to be present and skipped entirely when it is not.

    Args:
        module: Dotted module path to run as ``__main__``.
        tail: Arguments appended after the module.

    Returns:
        An argv list, or ``None`` when no checkout can be located.
    """
    src = os.environ.get("HYPERLOOM_SRC", "").strip()
    if not src:
        return None
    relative = Path(module.replace(".", "/")).with_suffix(".py")
    if not (Path(src) / relative).is_file():
        return None
    return [
        "python3",
        "-c",
        "import sys,runpy; sys.path.insert(0, sys.argv.pop(1)); "
        f"runpy.run_module({module!r}, run_name='__main__')",
        src,
        *tail,
    ]


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


def report_basename(eval_dir: Path) -> str:
    """Name the HTML report after the harness that ran it and the model it ran on.

    A reports directory accumulates: several models are optimized into sibling
    dirs, archives get flattened together, and a file called ``geak_report.html``
    tells the reader nothing about which run it belongs to. The prefix says who
    drove the run -- ``hl_`` when Hyperloom invoked GEAK as its KERNEL_AGENT
    phase, ``geak_`` when GEAK ran standalone -- because the two answer different
    questions and their numbers are not comparable.

    Hyperloom announces itself by exporting ``GEAK_INVOKED_BY=hyperloom`` in the
    child environment. Absence means standalone: an older Hyperloom that does not
    set it produces a ``geak_``-prefixed report, which understates the context but
    never misattributes a standalone run to Hyperloom.
    """
    prefix = "hl" if os.environ.get("GEAK_INVOKED_BY", "").strip().lower() == "hyperloom" else "geak"
    return f"{prefix}_run_report_{_model_name(eval_dir)}.html"


def _html_command(out_dir: Path, eval_dir: Path) -> list[str] | None:
    """Build the command that turns the rendered ledger into the HTML report."""
    override = os.environ.get("GEAK_HTML_REPORT_CMD", "").strip()
    tail = ["--reports-dir", str(out_dir), "-o", str(out_dir / report_basename(eval_dir))]
    if override:
        return override.split() + tail
    return _hyperloom_command(HTML_MODULE, tail)


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


def render_report(
    mirror_root: Path, out_dir: Path, eval_dir: Path, *, timeout_s: float = 600.0
) -> dict[str, Any]:
    """Render a per-call report from the mirror, if a renderer is reachable.

    Args:
        mirror_root: The mirror directory to read.
        out_dir: Where the report should be written.
        eval_dir: This run's eval dir, which selects its record.
        timeout_s: Ceiling on the renderer's runtime.

    Returns:
        A status dict; ``{"status": "skipped"}`` when no renderer was found.
        Never raises — a failed render leaves the raw mirror untouched.
    """
    argv = _report_command(mirror_root, out_dir, eval_dir)
    if not argv:
        return {"status": "skipped", "reason": "no renderer (set HYPERLOOM_SRC or GEAK_LLM_REPORT_CMD)"}
    return _run(argv, timeout_s, {"output_dir": str(out_dir)})


def render_html_report(out_dir: Path, eval_dir: Path, *, timeout_s: float = 600.0) -> dict[str, Any]:
    """Turn the rendered ledger into the structured HTML report, if possible.

    Runs after :func:`render_report` has written ``geak_calls.jsonl`` and after
    the outcome report has written ``geak_outcome.json``, because the HTML joins
    the two: what each phase cost, beside what it measured. With only the ledger
    present it still renders, and says in its coverage banner that the outcome
    half is missing rather than implying the phases bought nothing.

    Args:
        out_dir: The run's ``reports`` directory — both input and output.
        eval_dir: The run directory, read only to name the output file.
        timeout_s: Ceiling on the renderer's runtime.

    Returns:
        A status dict; ``{"status": "skipped"}`` when no renderer was found.
        Never raises.
    """
    argv = _html_command(out_dir, eval_dir)
    if not argv:
        return {"status": "skipped", "reason": "no renderer (set HYPERLOOM_SRC or GEAK_HTML_REPORT_CMD)"}
    return _run(argv, timeout_s, {"output": str(out_dir / report_basename(eval_dir))})


def _run(argv: list[str], timeout_s: float, extra: dict[str, Any]) -> dict[str, Any]:
    """Run a renderer subprocess, converting every failure into a status dict."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"status": "error", "returncode": proc.returncode, "stderr": (proc.stderr or "")[-2000:]}
    return {"status": "ok", **extra}


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
        render: Also attempt a rendered report and the HTML report beside the
            raw copy.

    Returns:
        A status dict carrying ``path`` on success. Never raises.
    """
    try:
        eval_path = Path(eval_dir)
        search = list(homes) if homes is not None else candidate_homes()
        hit = find_record(
            search,
            eval_dir=str(eval_path),
            exp_root=str(exp_root) if exp_root else None,
            session_id=session_id,
        )
        if hit is None:
            return {"status": "no_record", "homes": [str(h) for h in search]}
        record_path, record = hit
        dest = eval_path / MIRROR_DIRNAME
        manifest = mirror(record_path, record, dest, max_bytes=_max_bytes())
        result: dict[str, Any] = {
            "status": "ok",
            "path": str(dest),
            "run_id": manifest["run_id"],
            "files": len(manifest["files"]),
            "bytes_copied": manifest["bytes_copied"],
        }
        if render:
            reports_dir = eval_path / "reports"
            report = render_report(dest, reports_dir, eval_path)
            skill = install_skill(reports_dir)
            html = render_html_report(reports_dir, eval_path)
            manifest["report"] = report
            manifest["skill"] = skill
            manifest["html"] = html
            _write_manifest(dest, manifest)
            result["report"] = report
            result["skill"] = skill
            result["html"] = html
        return result
    except Exception as exc:  # never let telemetry kill a run
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
