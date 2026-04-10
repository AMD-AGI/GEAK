"""Result selection, reporting, and finalization across optimization rounds.

After all rounds complete (or the step limit is hit), these functions
pick the best result, rewrite the final report with verified speedup
data, and record the outcome to cross-session memory.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_reported_speedup(total_speedup: str | None) -> float | None:
    """Parse either ``1.06x`` or ``6%`` style speedup strings."""
    if total_speedup is None:
        return None
    text = str(total_speedup).strip()
    if not text:
        return None

    pct_match = re.search(r"([\d.]+)\s*%", text)
    if pct_match:
        return 1.0 + float(pct_match.group(1)) / 100.0

    mult_match = re.search(r"([\d.]+)\s*x", text, re.IGNORECASE)
    if mult_match:
        return float(mult_match.group(1))

    raw_match = re.search(r"([\d.]+)", text)
    if raw_match:
        return float(raw_match.group(1))
    return None


def _get_benchmark_section(round_eval: dict[str, Any]) -> dict[str, Any]:
    """Return the benchmark results section, checking both key names."""
    for key in ("full_benchmark", "benchmark"):
        section = round_eval.get(key)
        if isinstance(section, dict):
            return section
    return {}


def extract_verified_speedup(round_eval: dict[str, Any]) -> float | None:
    """Return the verified speedup from the benchmark section."""
    section = _get_benchmark_section(round_eval)
    verified = section.get("verified_speedup")
    if isinstance(verified, (int, float)):
        return float(verified)
    return None


def round_eval_label(round_eval: dict[str, Any]) -> str:
    """Return a stable round label like ``round_2`` for a round evaluation."""
    round_val = round_eval.get("round")
    if isinstance(round_val, int):
        return f"round_{round_val}"
    text = str(round_val).strip()
    if not text:
        return ""
    return text if text.startswith("round_") else f"round_{text}"


def round_eval_candidate_ms(round_eval: dict[str, Any]) -> float | None:
    """Return the candidate latency from the benchmark section, if available."""
    section = _get_benchmark_section(round_eval)
    candidate = section.get("candidate_ms")
    if isinstance(candidate, (int, float)) and candidate > 0:
        return float(candidate)
    return None


def select_best_verified_round_evaluation(output_dir: Path) -> Any:
    """Pick the best verified round deterministically from ``round_*_evaluation.json``.

    Selection order:
    1. Highest FULL_BENCHMARK verified speedup
    2. Lowest FULL_BENCHMARK candidate latency
    3. Stable lexical patch/task/round tie-breakers
    """
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for eval_path in sorted(output_dir.glob("round_*_evaluation.json")):
        try:
            round_eval = json.loads(eval_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        verified = extract_verified_speedup(round_eval)
        if verified is None:
            continue

        candidate_ms = round_eval_candidate_ms(round_eval)
        best_patch = str(round_eval.get("best_patch") or "")
        best_task = str(round_eval.get("best_task") or "")
        label = round_eval_label(round_eval) or eval_path.stem.replace("_evaluation", "")
        sort_key = (
            -float(verified),
            float(candidate_ms) if candidate_ms is not None else float("inf"),
            best_patch,
            best_task,
            label,
        )
        candidates.append((sort_key, round_eval))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def format_patch_label(patch_path: str | None) -> str:
    if not patch_path:
        return "unknown"
    patch = Path(patch_path)
    stem = patch.stem if patch.suffix else patch.name
    parent = patch.parent.name
    return f"{parent}/{stem}" if parent else stem


def rewrite_summary_with_verified_selection(
    summary: str,
    round_eval: dict[str, Any],
    *,
    verified_speedup_raw: float,
    verified_speedup: float,
) -> str:
    """Return one canonical verified final-selection summary block."""
    best_patch_label = format_patch_label(round_eval.get("best_patch"))
    best_task = round_eval.get("best_task") or Path(round_eval.get("best_patch", "")).parent.name or "unknown"
    bench_section = _get_benchmark_section(round_eval)
    baseline_ms = bench_section.get("baseline_ms")
    candidate_ms = bench_section.get("candidate_ms")
    baseline_speedup = bench_section.get("baseline_reported_speedup")
    candidate_speedup = bench_section.get("candidate_reported_speedup")

    lines = [
        "## Verified Final Selection",
        f"- Best task: {best_task}",
        f"- Best patch: {best_patch_label}",
    ]
    if verified_speedup_raw > 1.0:
        lines.append(f"- Verified FULL_BENCHMARK speedup: {verified_speedup_raw:.4f}x")
    else:
        lines.append(
            "- Verified FULL_BENCHMARK result: "
            f"no improvement ({verified_speedup_raw:.4f}x raw, clamped to {verified_speedup:.4f}x)"
        )
    if isinstance(baseline_ms, (int, float)) and isinstance(candidate_ms, (int, float)):
        lines.append(f"- Full benchmark geomean: {baseline_ms:.6f} ms -> {candidate_ms:.6f} ms")
    elif isinstance(baseline_speedup, (int, float)) and isinstance(candidate_speedup, (int, float)):
        lines.append(f"- Full benchmark reported speedup: {baseline_speedup:.4f}x -> {candidate_speedup:.4f}x")
    return "\n".join(lines)


def record_final_outcome(ctx: dict[str, Any], report: dict[str, Any]) -> None:
    """Record the final outcome using verified speedup when available."""
    try:
        from minisweagent.memory.cross_session_memory import (  # pylint: disable=import-error,no-name-in-module
            classify_kernel_category,
        )
        from minisweagent.memory.integration import (  # pylint: disable=import-error,no-name-in-module
            record_optimization_outcome,
        )

        speedup_val = report.get("verified_speedup")
        if not isinstance(speedup_val, (int, float)):
            parsed = parse_reported_speedup(report.get("total_speedup"))
            speedup_val = parsed if parsed is not None else 1.0

        speedup_val = float(speedup_val)
        success = bool(report.get("verified_improvement", speedup_val > 1.0))
        if not success and speedup_val < 1.0:
            speedup_val = 1.0

        _bm = ctx.get("baseline_metrics") or {}
        _kpath = ctx.get("kernel_path", "")
        _kcat = classify_kernel_category(_kpath) if _kpath else "unknown"
        record_optimization_outcome(
            kernel_path=_kpath,
            kernel_category=_kcat,
            bottleneck_type=_bm.get("bottleneck", "unknown"),
            strategy_name=(report.get("summary") or "")[:100],
            speedup_achieved=speedup_val,
            success=success,
            failure_reason=None if success else "no_improvement",
            profiling_metrics=_bm,
            patch_file=report.get("best_patch"),
        )
    except Exception as _rec_exc:
        logger.debug("Final memory outcome recording failed: %s", _rec_exc)


def merge_round_evaluation_into_final_report(
    ctx: dict[str, Any],
    output_dir: Path,
    report: dict[str, Any],
    round_eval: dict[str, Any],
) -> dict[str, Any]:
    """Rewrite final_report.json so it reflects verified final evaluation."""
    final_report_path = output_dir / "final_report.json"
    merged = dict(report)
    if final_report_path.exists():
        try:
            merged = json.loads(final_report_path.read_text())
        except (json.JSONDecodeError, OSError):
            merged = dict(report)

    existing_summary = str(merged.get("summary", ""))
    merged["round_evaluation"] = round_eval
    if round_eval.get("best_patch"):
        merged["best_patch"] = round_eval["best_patch"]
    if round_eval.get("best_task"):
        merged["best_task"] = round_eval["best_task"]
    label = round_eval_label(round_eval)
    if label:
        merged["best_round"] = label

    verified_speedup_raw = extract_verified_speedup(round_eval)
    if verified_speedup_raw is not None:
        # FULL_BENCHMARK verified_speedup is the canonical result — it's what
        # anyone gets by independently running the harness on the patched kernel.
        # The agent's benchmark_speedup may be higher due to non-reproducible
        # warm state (CUDA graphs, JIT caches) but that's not independently
        # verifiable, so we don't use it.
        verified_speedup = max(1.0, float(verified_speedup_raw))
        merged["best_speedup"] = round(float(verified_speedup_raw), 6)
        merged["best_speedup_verified"] = round(float(verified_speedup_raw), 6)
        merged["verified_speedup_raw"] = round(float(verified_speedup_raw), 6)
        merged["verified_speedup"] = round(verified_speedup, 6)
        merged["verified_improvement"] = verified_speedup_raw > 1.0
        merged["total_speedup"] = f"{verified_speedup:.4f}x"
        merged["verification_note"] = (
            f"Verified FULL_BENCHMARK speedup {verified_speedup_raw:.4f}x."
            if verified_speedup_raw > 1.0
            else (
                "No verified FULL_BENCHMARK improvement "
                f"({verified_speedup_raw:.4f}x raw); clamped to {verified_speedup:.4f}x."
            )
        )
        best_patch_path = str(merged.get("best_patch") or round_eval.get("best_patch") or "")
        if best_patch_path and Path(best_patch_path).is_file():
            merged["best_patch_size_bytes"] = Path(best_patch_path).stat().st_size
        bench_section = _get_benchmark_section(round_eval)
        if bench_section:
            baseline_ms = bench_section.get("baseline_ms")
            candidate_ms = bench_section.get("candidate_ms")
            patch_sz = merged.get("best_patch_size_bytes")
            if isinstance(baseline_ms, (int, float)) and isinstance(candidate_ms, (int, float)):
                analysis = (
                    f"Verified FULL_BENCHMARK: baseline={float(baseline_ms):.4f}ms, "
                    f"candidate={float(candidate_ms):.4f}ms. "
                    f"Speedup={float(verified_speedup_raw):.4f}x."
                )
                if isinstance(patch_sz, int) and patch_sz >= 0:
                    analysis += f" Patch={patch_sz}B."
                merged["best_patch_analysis"] = analysis
        canonical_summary = rewrite_summary_with_verified_selection(
            existing_summary,
            round_eval,
            verified_speedup_raw=float(verified_speedup_raw),
            verified_speedup=verified_speedup,
        )
        if existing_summary.strip() and existing_summary.strip() != canonical_summary.strip():
            merged["agent_summary"] = existing_summary
        merged["summary"] = canonical_summary

    final_report_path.write_text(json.dumps(merged, indent=2, default=str))
    record_final_outcome(ctx, merged)
    return merged


def post_round_evaluate(
    ctx: dict[str, Any],
    round_num: int,
    output_dir: Path,
) -> Any:
    """Run post-round evaluation and update ctx with best-patch tracking.

    Shared by both homogeneous and heterogeneous orchestrators.  After
    each round completes:

    1. Calls ``evaluate_round_best`` to run FULL_BENCHMARK + PROFILE on
       the best candidate from this round.
    2. Stores the result in ``ctx[f"round_{round_num}_eval"]``.
    3. Updates ``ctx["starting_patch"]`` and ``ctx["_best_global_speedup"]``
       if this round produced the best result so far.

    Returns a ``RoundEvaluation``, or ``None`` if no candidates existed.
    """
    from minisweagent.run.postprocess.evaluation import evaluate_round_best

    results_dir = output_dir / "results" / f"round_{round_num}"
    round_eval = evaluate_round_best(ctx, round_num, results_dir)
    if round_eval is None:
        return None

    ctx[f"round_{round_num}_eval"] = round_eval
    if round_eval.best_patch:
        fb = round_eval.full_benchmark
        # Only count rounds with an independently verified FULL_BENCHMARK result.
        # Falling back to the agent's self-reported benchmark_speedup risks
        # promoting a hallucinated or inflated speedup as the global best.
        current = fb.verified_speedup if fb and fb.verified_speedup is not None else None
        if current is None:
            agent_speedup = round_eval.benchmark_speedup
            note = (
                f"No FULL_BENCHMARK verified speedup available; "
                f"agent reported {agent_speedup:.4f}x (not used for global best selection)"
            )
            logger.warning("Round %d: %s", round_num, note)
            eval_path = output_dir / f"round_{round_num}_evaluation.json"
            try:
                eval_dict = json.loads(eval_path.read_text())
                eval_dict["verified_speedup_skipped"] = note
                eval_path.write_text(json.dumps(eval_dict, indent=2, default=str))
            except (json.JSONDecodeError, OSError):
                pass
        elif current >= ctx.get("_best_global_speedup", 0):
            ctx["starting_patch"] = round_eval.best_patch
            ctx["_best_global_speedup"] = current
    return round_eval


def _dict_to_final_report(d: dict[str, Any]) -> Any:
    """Convert an internal report dict to a FinalReport at the boundary."""
    from minisweagent.run.pipeline_types import FinalReport

    verified_raw = d.get("verified_speedup_raw", d.get("verified_speedup"))
    best_speedup = d.get("best_speedup")
    if best_speedup is None:
        best_speedup = parse_reported_speedup(d.get("total_speedup"))
    return FinalReport(
        status=str(d.get("status", "complete")),
        summary=str(d.get("summary", "")),
        best_patch=d.get("best_patch"),
        best_round=d.get("best_round"),
        best_task=d.get("best_task"),
        best_speedup=float(best_speedup) if best_speedup is not None else None,
        verified_speedup_unclamped=float(verified_raw) if verified_raw is not None else None,
    )


def finalize_run(
    ctx: dict[str, Any],
    output_dir: Path,
    *,
    finalize_result: dict[str, Any] | None = None,
    round_eval: Any = None,
) -> Any:
    """Finalize the optimization run.

    Shared by both orchestrator modes.  Returns a ``FinalReport``.

    If *finalize_result* is provided (the LLM explicitly called the
    finalize tool), merges it with *round_eval* for verified data.
    Otherwise auto-finalizes by scanning all rounds for the best result.

    When the LLM calls finalize on the last round, we cross-check against
    all previous rounds' verified evaluations and use the best one overall,
    not just the current round's result.
    """
    if finalize_result is not None:
        best_eval = select_best_verified_round_evaluation(output_dir)
        if best_eval is not None:
            merged = merge_round_evaluation_into_final_report(
                ctx,
                output_dir,
                finalize_result,
                best_eval,
            )
            return _dict_to_final_report(merged)
        if round_eval is not None:
            round_eval_dict = round_eval.to_dict() if hasattr(round_eval, "to_dict") else round_eval
            merged = merge_round_evaluation_into_final_report(
                ctx,
                output_dir,
                finalize_result,
                round_eval_dict,
            )
            return _dict_to_final_report(merged)
        # No verified round evaluations found — write final_report.json
        # with the LLM's summary so the report always exists on disk.
        record_final_outcome(ctx, finalize_result)
        output_dir = Path(ctx.get("output_dir", "."))
        report_path = output_dir / "final_report.json"
        if not report_path.exists():
            finalize_result.setdefault("status", "complete")
            finalize_result.setdefault("best_speedup", 1.0)
            finalize_result.setdefault("best_speedup_verified", 1.0)
            report_path.write_text(json.dumps(finalize_result, indent=2, default=str))
            logger.info("Wrote final_report.json (no verified round evaluations)")
        return _dict_to_final_report(finalize_result)
    report_dict = auto_finalize(ctx)
    return _dict_to_final_report(report_dict)


def auto_finalize(
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Auto-select the best result across all rounds when step limit is hit.

    Scans every ``results/round_N/*/best_results.json`` and picks the task
    with the highest speedup, then writes ``final_report.json``.
    """
    from minisweagent.agents.heterogeneous.task_generator import _scan_previous_results

    output_dir = Path(ctx["output_dir"])
    results_base = output_dir / "results"

    best_overall: dict[str, Any] | None = None
    best_speedup: float = 0.0
    best_round: str = ""
    best_task: str = ""
    round_summaries: list[str] = []

    if results_base.is_dir():
        for round_dir in sorted(results_base.iterdir()):
            if not round_dir.is_dir() or round_dir.name.startswith("."):
                continue

            summary = _scan_previous_results(round_dir)
            if summary:
                round_summaries.append(f"## {round_dir.name}\n{summary}")

            for task_dir in sorted(round_dir.iterdir()):
                if not task_dir.is_dir() or task_dir.name in ("worktrees",):
                    continue
                br_file = task_dir / "best_results.json"
                if not br_file.exists():
                    continue
                try:
                    br = json.loads(br_file.read_text())
                    speedup = float(br.get("best_patch_speedup", 0))
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_overall = br
                        best_round = round_dir.name
                        best_task = task_dir.name
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

    if best_overall and best_speedup > 1.0:
        summary_text = (
            f"Best result: {best_task} ({best_round}) with "
            f"{best_speedup:.2f}x speedup. "
            f"Patch: {best_overall.get('best_patch_file', 'N/A')}"
        )
    elif best_overall:
        summary_text = (
            f"No measurable improvement across all rounds. "
            f"Best candidate: {best_task} ({best_round}), "
            f"speedup {best_speedup:.2f}x. "
            f"Analysis: {best_overall.get('llm_selection_analysis', 'N/A')[:500]}"
        )
    else:
        summary_text = "No results found across any round."

    _patch_file = best_overall.get("best_patch_file") if best_overall else None
    _patch_sz = -1
    if _patch_file and Path(_patch_file).is_file():
        _patch_sz = Path(_patch_file).stat().st_size
        if _patch_sz == 0:
            best_speedup = 1.0
            logger.warning("Auto-finalize: empty patch (0 bytes), clamping speedup to 1.0")

    report: dict[str, Any] = {
        "status": "auto_finalized",
        "summary": summary_text,
        "best_round": best_round,
        "best_task": best_task,
        "best_speedup": best_speedup,
        "best_speedup_verified": best_speedup,
        "best_patch": _patch_file,
        "best_patch_size_bytes": _patch_sz,
        "best_patch_analysis": best_overall.get("llm_selection_analysis") if best_overall else None,
        "round_summaries": round_summaries,
    }

    best_verified_round_eval = select_best_verified_round_evaluation(output_dir)
    report_path = output_dir / "final_report.json"
    if best_verified_round_eval is not None:
        report["speedup_source"] = "FULL_BENCHMARK verified result"
        merged = merge_round_evaluation_into_final_report(
            ctx,
            output_dir,
            report,
            best_verified_round_eval,
        )
        logger.info("Auto-finalized: %s", merged.get("verification_note", summary_text))
        logger.info("Report written to: %s", report_path)
        return merged

    report["speedup_source"] = (
        "agent-reported benchmark (no FULL_BENCHMARK verified result available — "
        "the orchestrator will run FULL_BENCHMARK automatically after this round; "
        "do not use this speedup for final selection)"
    )
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Auto-finalized: %s", summary_text)
    logger.info("Report written to: %s", report_path)

    if not best_overall:
        return report

    try:
        from minisweagent.memory.cross_session_memory import (  # pylint: disable=import-error,no-name-in-module
            classify_kernel_category,
        )
        from minisweagent.memory.integration import (  # pylint: disable=import-error,no-name-in-module
            record_optimization_outcome,
        )

        _kpath = ctx.get("kernel_path", "")
        _kcat = classify_kernel_category(_kpath) if _kpath else "unknown"
        _bm = ctx.get("baseline_metrics") or {}
        record_optimization_outcome(
            kernel_path=_kpath,
            kernel_category=_kcat,
            bottleneck_type=_bm.get("bottleneck", "unknown"),
            strategy_name=summary_text[:100],
            speedup_achieved=best_speedup,
            success=best_speedup > 1.0,
            failure_reason=None if best_speedup > 1.0 else "no_improvement",
            profiling_metrics=_bm,
            patch_file=_patch_file,
        )
    except Exception as _rec_exc:
        logger.debug("Auto-finalize memory recording failed: %s", _rec_exc)

    return report
