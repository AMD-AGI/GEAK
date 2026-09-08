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
the mirror reproduces that shape verbatim and is readable unchanged with
``--claude-home <eval_dir>/llm_trace``.

Everything here is best effort. An 18-hour optimization run must never die over
telemetry, so no function in this module raises to its caller: failures are
recorded in the manifest and the run carries on.
"""
from __future__ import annotations

import json
import os
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


def record_paths(record: dict[str, Any]) -> list[str]:
    """Return the run directories a workflow record names, most specific first.

    A Hyperloom-driven run is identified by ``eval_dir``; a standalone one by
    ``exp_root``. Either can appear under ``args`` or under ``result``.

    Args:
        record: A parsed ``wf_*.json`` record.

    Returns:
        Distinct directory strings, ``eval_dir`` before ``exp_root``.
    """
    found: list[str] = []
    for key in ("eval_dir", "exp_root"):
        for holder in (record.get("args"), record.get("result")):
            if not isinstance(holder, dict):
                continue
            value = holder.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = value.strip().rstrip("/")
                if cleaned not in found:
                    found.append(cleaned)
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


def _matches(record: dict[str, Any], wanted: str) -> bool:
    """Report whether *record* names *wanted* (or a parent/child of it).

    Args:
        record: A parsed workflow record.
        wanted: A run directory, already stripped of a trailing slash.

    Returns:
        ``True`` when one of the record's directories relates to *wanted*.
    """
    return any(
        found == wanted or found.startswith(wanted + "/") or wanted.startswith(found + "/")
        for found in record_paths(record)
    )


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
    for wanted in (eval_dir, exp_root):
        cleaned = (wanted or "").strip().rstrip("/")
        if not cleaned:
            continue
        hits = [(p, r) for p, r in candidates if _matches(r, cleaned)]
        if not hits:
            continue
        if session_id:
            narrowed = [(p, r) for p, r in hits if p.parent.parent.name == session_id]
            if narrowed:
                hits = narrowed
        # Among identity matches, the newest recorded timestamp is the live run.
        hits.sort(key=lambda pr: str(pr[1].get("timestamp") or ""), reverse=True)
        return hits[0]
    return None


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
def _report_command(mirror_root: Path, out_dir: Path) -> list[str] | None:
    """Build the command that renders a report from the mirror, if available.

    GEAK cannot import Hyperloom, so the renderer is reached by subprocess when
    a checkout happens to be present and skipped entirely when it is not. The
    raw mirror is the durable artifact; the rendered report is a convenience.

    Args:
        mirror_root: The mirror directory to read.
        out_dir: Where the report should be written.

    Returns:
        An argv list, or ``None`` when no renderer can be located.
    """
    tail = ["--claude-home", str(mirror_root), "--output-dir", str(out_dir)]
    override = os.environ.get("GEAK_LLM_REPORT_CMD", "").strip()
    if override:
        return override.split() + tail
    src = os.environ.get("HYPERLOOM_SRC", "").strip()
    if not src:
        return None
    tool = Path(src) / "hyperloom" / "inference_optimizer" / "tools" / "dump_geak_call_report.py"
    if not tool.is_file():
        return None
    return [
        "python3",
        "-c",
        "import sys,runpy; sys.path.insert(0, sys.argv.pop(1)); "
        "runpy.run_module('hyperloom.inference_optimizer.tools.dump_geak_call_report', "
        "run_name='__main__')",
        src,
        *tail,
    ]


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


def render_report(mirror_root: Path, out_dir: Path, *, timeout_s: float = 600.0) -> dict[str, Any]:
    """Render a per-call report from the mirror, if a renderer is reachable.

    Args:
        mirror_root: The mirror directory to read.
        out_dir: Where the report should be written.
        timeout_s: Ceiling on the renderer's runtime.

    Returns:
        A status dict; ``{"status": "skipped"}`` when no renderer was found.
        Never raises — a failed render leaves the raw mirror untouched.
    """
    argv = _report_command(mirror_root, out_dir)
    if not argv:
        return {"status": "skipped", "reason": "no renderer (set HYPERLOOM_SRC or GEAK_LLM_REPORT_CMD)"}
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"status": "error", "returncode": proc.returncode, "stderr": (proc.stderr or "")[-2000:]}
    return {"status": "ok", "output_dir": str(out_dir)}


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
        render: Also attempt a rendered report beside the raw copy.

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
            report = render_report(dest, reports_dir)
            skill = install_skill(reports_dir)
            manifest["report"] = report
            manifest["skill"] = skill
            _write_manifest(dest, manifest)
            result["report"] = report
            result["skill"] = skill
        return result
    except Exception as exc:  # never let telemetry kill a run
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
