#!/usr/bin/env python3
"""One entry point that turns a GEAK run into its report — for both run styles.

A run leaves behind Claude transcripts; this driver turns them into the
role-execution-tree report (HTML + Markdown) in two steps that already exist as
separate tools:

  1. ``e2e_workflow/scripts/llm_ledger.py`` reads the transcripts and writes the
     per-call ledger ``reports/trace/llm_calls.jsonl`` (tokens, wall time, and —
     since the ledger was extended — per-call output/thinking and the cost-bucket
     split).
  2. ``interface/geak_call_tree_html.py`` folds that ledger into the clickable
     role tree and renders ``report/geak_run_report_<model>.{html,md}``.

The two run styles differ only in where the transcripts live, which is the one
thing this driver hides:
  * E2E workflow — pass ``--eval-dir <run>``; the ledger discovers the run's own
    transcripts under it.
  * Kernel lane — pass ``--transcripts '<glob>'`` (repeatable) pointing at the
    session's ``subagents/**/agent-*.jsonl``; ``--eval-dir`` is then just the
    scratch dir the ledger writes into (defaults to a temp dir).

With ``--persist`` the whole set is copied into the shared layout
``<persist-root>/<model>/{geak_run_ledger,geak_llm_artifacts,report}/`` so a run's
telemetry, per-call JSON, and report sit together. (``claude_llm_artifacts`` — the
raw mirrored transcripts — is populated by ``claude_trace_mirror``, out of scope
here; the folder is created so the layout is complete.)

Usage:
  # E2E run
  python3 interface/geak_report.py --eval-dir /path/to/e2e_<model>_<stamp>
  # kernel lane
  python3 interface/geak_report.py \
      --transcripts '/path/session/subagents/**/agent-*.jsonl' \
      --model Qwen3-14B-FP8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "e2e_workflow", "scripts"))

import geak_call_tree_html as tree  # noqa: E402
import llm_ledger  # noqa: E402

PERSIST_ROOT_DEFAULT = "/mnt/dcgpuval/aditysin/shared_nfs/geak-outs-m1513"
PERSIST_SUBDIRS = ("geak_run_ledger", "claude_llm_artifacts",
                   "geak_llm_artifacts", "report")


def _model_name(eval_dir, override=None):
    """Best-effort model identity, reusing the mirror's resolver when importable."""
    if override:
        return override
    try:
        import claude_trace_mirror
        from pathlib import Path
        return claude_trace_mirror._model_name(Path(eval_dir))
    except Exception:
        base = os.path.basename(os.path.normpath(eval_dir))
        if base.startswith("e2e_"):
            return "_".join(base[4:].split("_")[:-4]) or "run"
        return base or "run"


def _run_ledger(eval_dir, transcripts, rates_path):
    """Invoke the in-repo ledger; returns the path to llm_calls.jsonl (or None)."""
    argv = ["--eval-dir", eval_dir, "--quiet"]
    for glob in (transcripts or []):
        argv += ["--transcripts", glob]
    if rates_path:
        argv += ["--rates", rates_path]
    llm_ledger.main(argv)
    calls = os.path.join(eval_dir, "reports", "trace", "llm_calls.jsonl")
    return calls if os.path.isfile(calls) else None


def _write_per_call_artifacts(calls_path, out_dir):
    """One JSON per call — full prompt, output, thinking, and cost buckets —
    so the raw Claude exchange is persisted, not just the report's summary."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for i, row in enumerate(tree.read_calls(calls_path)):
        art = {
            "call_index": i,
            "role": row.get("role"), "sub_phase": row.get("sub_phase"),
            "agent_label": row.get("agent_label"), "phase": row.get("phase"),
            "model": row.get("model"), "ts": row.get("ts"),
            "duration_ms": row.get("duration_ms"),
            "tokens": {
                "input": row.get("input_tokens"),
                "cache_read": row.get("cache_read_input_tokens"),
                "cache_write_5m": row.get("cache_write_5m_tokens"),
                "cache_write_1h": row.get("cache_write_1h_tokens"),
                "output": row.get("output_tokens"),
            },
            "cost_usd": row.get("cost_usd"),
            "cost_breakdown": row.get("cost_breakdown"),
            "prompt": row.get("prompt"), "output": row.get("output"),
            "thinking": row.get("thinking"),
        }
        with open(os.path.join(out_dir, "call_%05d.json" % i), "w",
                  encoding="utf-8") as fh:
            json.dump(art, fh, indent=2)
        n += 1
    return n


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_id(calls_path):
    """A stable identity for THIS run, derived from the set of API responses it
    captured (message ids). The same run regenerated hashes to the same id — so
    its export replaces itself in place — while a different run (different calls)
    lands in its own directory and stays individually recoverable. Two runs of
    the same model can therefore never overwrite or mix each other's artifacts."""
    ids = sorted({(r.get("message_id") or r.get("group_id") or "")
                  for r in tree.read_calls(calls_path)} - {""})
    if not ids:
        ids = [os.path.abspath(calls_path)]
    digest = hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:12]
    return "run-%s" % digest


def _write_manifest(root, dirs, model, run_id, n_art):
    """A manifest so a shared export is self-describing: what the dollars mean,
    what the prompt text is (a snippet, not the wire prompt), and a hash per file
    so the archive is verifiable. It enumerates ONLY this run's files (the run
    lives in its own directory) and records the run identity, so a shared export
    can never be read as covering more than the one run it belongs to. Costs are
    estimated from a fixed rate card, not an invoice."""
    files = {}
    for name, d in dirs.items():
        for fn in sorted(os.listdir(d)):
            fp = os.path.join(d, fn)
            if os.path.isfile(fp):
                files["%s/%s" % (name, fn)] = {"sha256": _sha256(fp),
                                               "bytes": os.path.getsize(fp)}
    manifest = {
        "model": model,
        "run_id": run_id,
        "layout": list(PERSIST_SUBDIRS),
        "per_call_artifacts": n_art,
        "basis": {
            "cost": "ESTIMATED from token buckets against a fixed rate card "
                    "(DEFAULT_RATES in e2e_workflow/scripts/llm_ledger.py); not an "
                    "SDK total and not a provider invoice. Excludes parent "
                    "driver/resume/monitor scope not present in the transcripts.",
            "tokens": "Merged per message.id from the transcripts by the canonical "
                      "ledger (keeps the complete/final usage record).",
            "prompt": "A transcript snippet of the role prompt — NOT the full wire "
                      "prompt (system/tool definitions are not included).",
            "billed_span": "Sum of per-call observed durations, not true API "
                           "request wall-time.",
            "output": "Captured assistant text; output tokens also include thinking "
                      "and tool arguments. Redacted/unavailable fields are labelled, "
                      "not inferred as zero.",
        },
        "files": files,
    }
    with open(os.path.join(root, "_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _persist(eval_dir, calls_path, report_dir, model, persist_root):
    """Copy ledger + per-call JSON + report into the shared per-model/per-run
    layout ``<root>/<model>/<run_id>/{...}``. Each run owns its directory, and
    different runs of the same model never collide.

    Regeneration is ATOMIC and NON-DESTRUCTIVE. The old code cleared ``root`` with
    ``rmtree`` *before* copying, so any failure mid-copy (disk full, a source file
    vanishing, an interrupt) left the run with a half-deleted export and no way
    back. Instead the new export is fully written into a sibling stage directory
    and swapped in with a single ``os.replace`` only once complete; the previous
    valid export is retired to the side first and kept until the swap succeeds, so
    at no point is there no valid export, and a staging failure leaves the prior
    one exactly as it was."""
    run_id = _run_id(calls_path)
    model_root = os.path.join(persist_root, model)
    root = os.path.join(model_root, run_id)
    os.makedirs(model_root, exist_ok=True)
    stage = os.path.join(model_root, "%s.stage-%d" % (run_id, os.getpid()))
    shutil.rmtree(stage, ignore_errors=True)      # clear an orphaned prior stage
    try:
        dirs = {name: os.path.join(stage, name) for name in PERSIST_SUBDIRS}
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        # ledger: everything under reports/trace/
        trace_dir = os.path.dirname(calls_path)
        for fn in os.listdir(trace_dir):
            src = os.path.join(trace_dir, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dirs["geak_run_ledger"], fn))
        # per-call raw artifacts
        n_art = _write_per_call_artifacts(calls_path, dirs["geak_llm_artifacts"])
        # report
        for fn in os.listdir(report_dir):
            src = os.path.join(report_dir, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dirs["report"], fn))
        _write_manifest(stage, dirs, model, run_id, n_art)
    except BaseException:
        # Staging failed: drop the partial stage and leave the prior export intact.
        shutil.rmtree(stage, ignore_errors=True)
        raise
    # Promote. os.replace cannot drop a non-empty directory, so retire the old
    # export to the side FIRST (only now that the new one is fully staged), swap
    # the stage in, then delete the retired copy. If the swap itself fails, roll
    # the retired copy back so a valid export always remains.
    retired = None
    if os.path.isdir(root):
        retired = os.path.join(model_root, "%s.old-%d" % (run_id, os.getpid()))
        shutil.rmtree(retired, ignore_errors=True)
        os.replace(root, retired)
    try:
        os.replace(stage, root)
    except BaseException:
        if retired is not None and not os.path.isdir(root):
            os.replace(retired, root)             # roll back to the prior export
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if retired is not None:
        shutil.rmtree(retired, ignore_errors=True)
    return root, n_art


def run(eval_dir=None, transcripts=None, model=None, rates_path=None,
        out_dir=None, persist=False, persist_root=PERSIST_ROOT_DEFAULT):
    """Build the report; optionally persist to the shared layout. Returns a dict."""
    tmp = None
    if not eval_dir:
        tmp = tempfile.mkdtemp(prefix="geak_report_")
        eval_dir = tmp
    try:
        calls = _run_ledger(eval_dir, transcripts, rates_path)
        if not calls:
            return {"status": "no-calls",
                    "reason": "ledger produced no llm_calls.jsonl (no transcripts?)"}
        # A ledger can write an EMPTY llm_calls.jsonl (e.g. an explicit transcript
        # glob that matches nothing) — that is a captured-nothing run, not a
        # healthy one. Report it as such rather than emitting a zero-call "ok".
        row_count = sum(1 for _ in tree.read_calls(calls))
        if row_count == 0:
            meta = tree._read_ledger_meta(calls) or {}
            return {"status": "no-capture", "calls": calls,
                    "reason": "ledger captured 0 API calls (no matching transcripts / "
                              "nothing in the run window)",
                    "warnings": list(meta.get("warnings") or [])}
        name = _model_name(eval_dir, model)
        report_dir = out_dir or os.path.join(eval_dir, "report")
        html_path, md_path = tree.write(calls, report_dir, name)
        result = {"status": "ok", "model": name, "calls": calls,
                  "html": html_path, "md": md_path, "report_dir": report_dir}
        if persist:
            root, n_art = _persist(eval_dir, calls, report_dir, name, persist_root)
            result["persisted_to"] = root
            result["artifacts"] = n_art
        return result
    finally:
        # Keep a caller-supplied eval_dir; only clean the temp we made — but not
        # if the report was written inside it.
        if tmp and (out_dir and not out_dir.startswith(tmp)):
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the GEAK role-execution-tree report for a run (E2E or kernel).")
    ap.add_argument("--eval-dir", default=None,
                    help="E2E run dir (ledger discovers its transcripts); "
                         "for kernel runs, a scratch dir for ledger output (optional)")
    ap.add_argument("--transcripts", action="append", default=None,
                    help="explicit transcript glob (repeatable) — use for kernel-lane runs")
    ap.add_argument("--model", default=None, help="override the model/run name")
    ap.add_argument("--rates", default=None, help="JSON file overriding per-million prices")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the report (default <eval-dir>/report)")
    ap.add_argument("--persist", action="store_true",
                    help="also copy ledger + per-call JSON + report into the shared layout")
    ap.add_argument("--persist-root", default=PERSIST_ROOT_DEFAULT,
                    help="root of the shared per-model layout")
    args = ap.parse_args(argv)

    if not args.eval_dir and not args.transcripts:
        ap.error("give --eval-dir (E2E) or --transcripts (kernel)")

    res = run(eval_dir=args.eval_dir, transcripts=args.transcripts, model=args.model,
              rates_path=args.rates, out_dir=args.out_dir, persist=args.persist,
              persist_root=args.persist_root)
    if res["status"] != "ok":
        print("geak_report: %s — %s" % (res["status"], res.get("reason", "")), file=sys.stderr)
        return 1
    print("geak_report: wrote %s and %s" % (res["html"], res["md"]))
    if res.get("persisted_to"):
        print("geak_report: persisted %s (%s per-call artifacts) to %s"
              % (res["model"], res.get("artifacts", 0), res["persisted_to"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
