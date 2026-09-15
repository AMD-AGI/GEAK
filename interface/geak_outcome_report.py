"""Throughput outcome and per-phase attribution for a GEAK e2e run.

The per-LLM-call report answers *what a run cost*.  This module answers the
other half — *what the run bought, and which phase bought it* — and it does so
from the run's own measured artifacts only:

    baseline/bench_summary.json      the round-0 serving baseline
    config/sweep_results.json        the ConfigSweep accepted stack
    kernels/*/opbench_result.json    each HeadKernel task's isolated result
    kernels/*/_capture_overlay/integrate_result.json
                                    the end-to-end A/B for a kernel the run
                                    authored, when one reached validation
    tuning/tuning_result.json        the TuningSkillset A/B

Nothing here is modelled, extrapolated or defaulted.  A stage whose artifact is
missing is reported as absent, never as zero, because "we did not measure it"
and "it contributed nothing" are different claims.

Stdlib only, no GPU, no network: it runs at the end of every e2e run and must
never be the reason one fails.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPORTS_DIRNAME = "reports"
OUTCOME_JSON = "geak_outcome.json"
OUTCOME_MD = "geak_outcome.md"

# Phase labels as the call-tree reporter names them, so the spend table and the
# gain table can be joined on a common key.
PHASE_BASELINE = "P1 Setup+Baseline"
PHASE_SWEEP = "P4 ConfigSweep"
PHASE_TUNING = "P6 TuningSkillset"
PHASE_HEADKERNEL = "HeadKernel"


def _load(path: Path) -> Any | None:
    """Read JSON, or return None for anything unreadable."""
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _num(value: Any) -> float | None:
    """Coerce to float, or None. Bools are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _pct(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return round((after / before - 1.0) * 100.0, 4)


def _speedup(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return round(after / before, 6)


def baseline_stage(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "baseline" / "bench_summary.json"
    data = _load(path)
    if not isinstance(data, dict):
        return {"phase": PHASE_BASELINE, "present": False, "source": str(path)}
    tput = _num(data.get("throughput_tok_s_median"))
    return {
        "phase": PHASE_BASELINE,
        "present": tput is not None,
        "source": str(path),
        "kind": "reference",
        "measurement_mode": data.get("measurement_mode"),
        "after_tok_s": tput,
        "ttft_ms": _num(data.get("ttft_ms_median")),
        "tpot_ms": _num(data.get("tpot_ms_median")),
    }


def sweep_stage(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config" / "sweep_results.json"
    data = _load(path)
    if not isinstance(data, dict):
        return {"phase": PHASE_SWEEP, "present": False, "source": str(path)}
    before = _num(data.get("baseline_throughput_tok_s"))
    after = _num(data.get("best_throughput_tok_s"))
    trials = data.get("trials")
    measurement = data.get("measurement") if isinstance(data.get("measurement"), dict) else {}
    return {
        "phase": PHASE_SWEEP,
        "present": before is not None and after is not None,
        "source": str(path),
        "kind": "delta",
        "measurement_mode": measurement.get("mode"),
        "before_tok_s": before,
        "after_tok_s": after,
        "speedup": _speedup(before, after) or _num(data.get("throughput_speedup_vs_baseline")),
        "delta_pct": _pct(before, after),
        "trials": len(trials) if isinstance(trials, list) else None,
        "accepted_flags": data.get("accepted_flags"),
        "accepted_env": data.get("accepted_env"),
        "gsm8k_exact_match": _num(data.get("baseline_gsm8k_exact_match")),
    }


def integration_result(task_dir: Path) -> dict[str, Any] | None:
    """The end-to-end A/B for a kernel this run authored, if it got that far.

    ``opbench_result.json`` and this file answer different questions and are
    routinely in tension. Opbench races the available *library backends* against
    each other, so when the incumbent stays fastest it records a 1.0000x speedup
    and a 0.00% ceiling. That says nothing about the kernel the run itself
    wrote: that candidate is validated separately, served end to end against the
    reference, and its result lands here. Reading only opbench therefore reports
    a task that produced a real measured win as having produced nothing.

    Returns ``None`` when no candidate reached validation, which is the common
    case and is not a failure.
    """
    data = _load(task_dir / "_capture_overlay" / "integrate_result.json")
    if not isinstance(data, dict):
        return None
    gsm8k = data.get("gsm8k") if isinstance(data.get("gsm8k"), dict) else {}
    return {
        "source": str(task_dir / "_capture_overlay" / "integrate_result.json"),
        "candidate": data.get("cand_tag"),
        "isolated_speedup": _num(data.get("isolated_speedup")),
        "e2e_delta_pct": _num(data.get("e2e_delta_pct")),
        "e2e_throughput_tok_s": _num(data.get("e2e_throughput_tok_s")),
        "amdahl_ceiling_pct": _num(data.get("amdahl_ceiling_pct")),
        "gate": data.get("gate"),
        "ab_complete": data.get("ab_complete"),
        "provenance_ok": data.get("provenance_ok"),
        "output_parity": data.get("output_parity"),
        "gsm8k_ref": _num(gsm8k.get("ref")),
        "gsm8k_cand": _num(gsm8k.get("cand")),
        "accepted_overlay": data.get("accepted_overlay"),
        "reason": data.get("reason"),
    }


def headkernel_stages(run_dir: Path) -> list[dict[str, Any]]:
    """One entry per kernel task directory, ordered by name.

    ``amdahl_ceiling_e2e_pct`` is the ceiling the task's own harness computed
    from its measured share of GPU time; it is the honest upper bound on what
    the task could have contributed end-to-end, not a claim that it did.

    ``integration`` carries the separate end-to-end A/B of a kernel the run
    wrote, when there was one -- see :func:`integration_result` for why that is
    not the same measurement as the opbench fields beside it.
    """
    kernels_dir = run_dir / "kernels"
    out: list[dict[str, Any]] = []
    try:
        entries = sorted(p for p in kernels_dir.iterdir() if p.is_dir())
    except OSError:
        return out
    for task_dir in entries:
        if task_dir.name.startswith("_"):
            continue
        path = task_dir / "opbench_result.json"
        data = _load(path)
        integration = integration_result(task_dir)
        if not isinstance(data, dict):
            out.append({
                "phase": PHASE_HEADKERNEL,
                "task": task_dir.name,
                "present": False,
                "source": str(path),
                "kind": "kernel",
                "integration": integration,
            })
            continue
        out.append({
            "phase": PHASE_HEADKERNEL,
            "task": task_dir.name,
            "present": True,
            "source": str(path),
            "kind": "kernel",
            "winner_backend": data.get("winner_backend"),
            "baseline_backend": data.get("baseline_backend"),
            "winner_ms": _num(data.get("winner_ms")),
            "baseline_ms": _num(data.get("baseline_ms")),
            "isolated_speedup": _num(data.get("isolated_speedup")),
            "pct_gpu_time": _num(data.get("pct_gpu_time")),
            "amdahl_ceiling_e2e_pct": _num(data.get("amdahl_ceiling_e2e_pct")),
            "winner_editable": data.get("winner_editable"),
            "measured": data.get("measured"),
            "integration": integration,
        })
    return out


def tuning_stage(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "tuning" / "tuning_result.json"
    data = _load(path)
    if not isinstance(data, dict):
        return {"phase": PHASE_TUNING, "present": False, "source": str(path)}
    before = _num(data.get("pre_tune_throughput_tok_s"))
    after = _num(data.get("post_tune_throughput_tok_s"))
    ops = data.get("ops_tuned")
    return {
        "phase": PHASE_TUNING,
        "present": before is not None and after is not None,
        "source": str(path),
        "kind": "delta",
        "measurement_mode": "isolated_server_ab" if data.get("ab_interleaved") else None,
        "before_tok_s": before,
        "after_tok_s": after,
        "speedup": _speedup(before, after) or _num(data.get("tuning_speedup")),
        "delta_pct": _pct(before, after) if before else _num(data.get("tuning_delta_pct")),
        "noise_floor_pct": _num(data.get("noise_floor_pct")),
        "gate": data.get("gate"),
        "correctness_gate": data.get("correctness_gate"),
        "ab_complete": data.get("ab_complete"),
        "engagement_verified": data.get("engagement_verified"),
        "ops_tuned": [op.get("op") for op in ops if isinstance(op, dict)] if isinstance(ops, list) else None,
    }


def _delta_stages(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in stages if s.get("kind") == "delta" and s.get("present")]


def summarize(stages: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    """Roll the measured stages up, and say plainly where the chain does not join.

    Each phase measures its own before/after in its own server session.  One
    phase's ``after`` is therefore not guaranteed to equal the next phase's
    ``before``; when it does not, the compounded figure is an estimate and is
    labelled as one.  We never silently paper over the seam.
    """
    deltas = _delta_stages(stages)
    compounded = 1.0
    for stage in deltas:
        compounded *= stage.get("speedup") or 1.0

    seams: list[dict[str, Any]] = []
    for prev, nxt in zip(deltas, deltas[1:]):
        a, b = prev.get("after_tok_s"), nxt.get("before_tok_s")
        if a and b and abs(a - b) / a > 0.01:
            seams.append({
                "from_phase": prev["phase"],
                "to_phase": nxt["phase"],
                "handoff_from_tok_s": a,
                "handoff_to_tok_s": b,
                "gap_pct": round((b / a - 1.0) * 100.0, 4),
            })

    first = deltas[0].get("before_tok_s") if deltas else None
    last = deltas[-1].get("after_tok_s") if deltas else None
    return {
        "reference_baseline_tok_s": baseline.get("after_tok_s"),
        "first_measured_before_tok_s": first,
        "last_measured_after_tok_s": last,
        "observed_speedup_first_to_last": _speedup(first, last),
        "observed_delta_pct_first_to_last": _pct(first, last),
        "compounded_speedup": round(compounded, 6) if deltas else None,
        "compounded_is_estimate": bool(seams),
        "handoff_seams": seams,
        "stages_measured": len(deltas),
    }


def spend_by_phase(reports_dir: Path) -> dict[str, dict[str, Any]] | None:
    """Per-phase LLM spend from ``geak_calls.jsonl``, when that report exists.

    Returns None rather than zeros when the file is absent — a run whose ledger
    was never harvested has unknown spend, not zero spend.
    """
    path = reports_dir / "geak_calls.jsonl"
    if not path.is_file():
        return None
    out: dict[str, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                phase = str(row.get("phase") or "unknown")
                acc = out.setdefault(phase, {"calls": 0, "isl": 0, "osl": 0, "usd": 0.0, "tool_calls": 0})
                acc["calls"] += 1
                acc["isl"] += int(_num(row.get("isl")) or 0)
                acc["osl"] += int(_num(row.get("osl")) or 0)
                acc["usd"] += _num(row.get("usd")) or 0.0
                tools = row.get("tools")
                acc["tool_calls"] += len(tools) if isinstance(tools, list) else 0
    except OSError:
        return None
    for acc in out.values():
        acc["usd"] = round(acc["usd"], 6)
    return out or None


def collect(run_dir: Path, *, reports_dir: Path | None = None) -> dict[str, Any]:
    """Build the whole outcome record for one run directory."""
    run_dir = Path(run_dir)
    reports = Path(reports_dir) if reports_dir is not None else run_dir / REPORTS_DIRNAME
    baseline = baseline_stage(run_dir)
    stages = [baseline, sweep_stage(run_dir), *headkernel_stages(run_dir), tuning_stage(run_dir)]
    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "stages": stages,
        "summary": summarize(stages, baseline),
        "spend_by_phase": spend_by_phase(reports),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    num = _num(value)
    if num is None:
        return "—"
    return f"{num:,.{digits}f}"


def render_markdown(record: dict[str, Any]) -> str:
    """Human-readable outcome table. Absent measurements print as em dashes."""
    summary = record.get("summary") or {}
    spend = record.get("spend_by_phase")
    lines = [
        f"# GEAK outcome — {record.get('run_id')}",
        "",
        "Every number below is read from the run's own measured artifacts. "
        "`—` means the artifact was absent, not that the value was zero.",
        "",
        "## Throughput",
        "",
        "| Phase | Measurement | Before tok/s | After tok/s | Speedup | Delta | Gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for stage in record.get("stages") or []:
        if stage.get("kind") == "kernel":
            continue
        name = stage.get("phase", "?")
        if not stage.get("present"):
            lines.append(f"| {name} | *artifact absent* | — | — | — | — | — |")
            continue
        delta = stage.get("delta_pct")
        lines.append(
            f"| {name} | {stage.get('measurement_mode') or '—'} | {_fmt(stage.get('before_tok_s'))} | "
            f"{_fmt(stage.get('after_tok_s'))} | {_fmt(stage.get('speedup'), 4)}x | "
            f"{'—' if delta is None else f'{delta:+.2f}%'} | {stage.get('gate') or '—'} |"
        )

    kernels = [s for s in (record.get("stages") or []) if s.get("kind") == "kernel"]
    if kernels:
        lines += [
            "",
            "## HeadKernel tasks",
            "",
            "`isolated speedup` is the task's own op-level A/B; `Amdahl ceiling` is the most "
            "end-to-end gain that op's share of GPU time could have produced.",
            "",
            "| Task | Winner | Baseline ms | Winner ms | Isolated speedup | % GPU time | Amdahl ceiling e2e |",
            "|---|---|---|---|---|---|---|",
        ]
        for stage in kernels:
            if not stage.get("present"):
                lines.append(f"| {stage.get('task')} | *no opbench_result.json* | — | — | — | — | — |")
                continue
            lines.append(
                f"| {stage.get('task')} | {stage.get('winner_backend') or '—'} | "
                f"{_fmt(stage.get('baseline_ms'), 5)} | {_fmt(stage.get('winner_ms'), 5)} | "
                f"{_fmt(stage.get('isolated_speedup'), 4)}x | {_fmt(stage.get('pct_gpu_time'))}% | "
                f"{_fmt(stage.get('amdahl_ceiling_e2e_pct'))}% |"
            )

    if spend:
        lines += ["", "## LLM spend by phase", "", "| Phase | Calls | ISL | OSL | Tool calls | USD |", "|---|---|---|---|---|---|"]
        for phase, acc in sorted(spend.items(), key=lambda kv: -kv[1]["usd"]):
            lines.append(
                f"| {phase} | {acc['calls']:,} | {acc['isl']:,} | {acc['osl']:,} | "
                f"{acc['tool_calls']:,} | ${acc['usd']:,.2f} |"
            )
    else:
        lines += ["", "## LLM spend by phase", "", "No `geak_calls.jsonl` in this run's reports directory, so spend is unknown."]

    lines += ["", "## Summary", ""]
    for key in (
        "reference_baseline_tok_s",
        "first_measured_before_tok_s",
        "last_measured_after_tok_s",
        "observed_speedup_first_to_last",
        "observed_delta_pct_first_to_last",
        "compounded_speedup",
        "compounded_is_estimate",
        "stages_measured",
    ):
        lines.append(f"- `{key}`: {summary.get(key)}")
    for seam in summary.get("handoff_seams") or []:
        lines.append(
            f"- seam {seam['from_phase']} -> {seam['to_phase']}: "
            f"{seam['handoff_from_tok_s']:,.2f} handed off as {seam['handoff_to_tok_s']:,.2f} tok/s "
            f"({seam['gap_pct']:+.2f}%) — separate server sessions, so the compounded figure is an estimate."
        )
    return "\n".join(lines) + "\n"


def write(run_dir: Path, *, reports_dir: Path | None = None) -> dict[str, Any]:
    """Collect and persist the outcome report. Never raises."""
    run_dir = Path(run_dir)
    reports = Path(reports_dir) if reports_dir is not None else run_dir / REPORTS_DIRNAME
    try:
        record = collect(run_dir, reports_dir=reports)
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    try:
        reports.mkdir(parents=True, exist_ok=True)
        json_path = reports / OUTCOME_JSON
        tmp = json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, json_path)
        md_path = reports / OUTCOME_MD
        md_path.write_text(render_markdown(record), encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "ok",
        "json": str(json_path),
        "markdown": str(md_path),
        "stages_measured": record["summary"]["stages_measured"],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Throughput outcome report for a GEAK e2e run.")
    parser.add_argument("run_dir", help="the run directory (eval_dir), e.g. .../e2e_<model>_<stamp>")
    parser.add_argument("--reports-dir", default=None, help="override the output directory")
    parser.add_argument("--stdout", action="store_true", help="print the markdown instead of writing files")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    reports = Path(args.reports_dir) if args.reports_dir else None
    if args.stdout:
        print(render_markdown(collect(run_dir, reports_dir=reports)), end="")
        return 0
    result = write(run_dir, reports_dir=reports)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
