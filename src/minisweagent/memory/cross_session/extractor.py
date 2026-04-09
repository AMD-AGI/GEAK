"""Extract an ExperienceRecord from optimization run artifacts.

Called at session end by record_optimization_outcome() with kwargs from
results.py -- kernel_path, kernel_category, bottleneck_type, strategy_name,
speedup_achieved, success, failure_reason, profiling_metrics, patch_file.

Optionally enriches by reading:
  - final_report.json (best_patch_analysis, round summaries)
  - _working_memory/events/*.jsonl (what_worked, what_failed, trajectory)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from minisweagent.memory.cross_session.schemas import ExperienceRecord

logger = logging.getLogger(__name__)


def extract_experience(**kwargs: Any) -> ExperienceRecord:
    """Build an ExperienceRecord from the kwargs passed to record_optimization_outcome."""

    kernel_path = str(kwargs.get("kernel_path", ""))
    kernel_name = Path(kernel_path).stem if kernel_path else ""
    profiling_metrics = kwargs.get("profiling_metrics") or {}

    baseline_latency_ms = 0.0
    if profiling_metrics.get("benchmark_duration_us"):
        baseline_latency_ms = float(profiling_metrics["benchmark_duration_us"]) / 1000.0
    elif profiling_metrics.get("duration_us"):
        baseline_latency_ms = float(profiling_metrics["duration_us"]) / 1000.0

    speedup = float(kwargs.get("speedup_achieved", 1.0) or 1.0)
    best_latency_ms = baseline_latency_ms / speedup if speedup > 0 and baseline_latency_ms > 0 else 0.0

    patch_file = str(kwargs.get("patch_file", "") or "")
    report_dir = Path(patch_file).parent if patch_file else None

    what_worked, what_failed, dead_ends, trajectory = _extract_notebook_insights(report_dir)
    key_insight, best_change_category = _extract_report_insights(report_dir)
    hardware = _extract_hardware(profiling_metrics)
    language = _infer_language(kernel_path)
    patch_content, code_changes_summary = _extract_patch_content(patch_file, report_dir)

    strategy_name = str(kwargs.get("strategy_name", "") or "")
    if not best_change_category and strategy_name:
        best_change_category = _classify_strategy(strategy_name)

    return ExperienceRecord(
        kernel_path=kernel_path,
        kernel_name=kernel_name,
        kernel_category=str(kwargs.get("kernel_category", "unknown") or "unknown"),
        kernel_language=language,
        repo_url="",
        bottleneck_type=str(kwargs.get("bottleneck_type", "unknown") or "unknown"),
        baseline_latency_ms=baseline_latency_ms,
        top_kernels=profiling_metrics.get("top_kernels", []) if isinstance(profiling_metrics, dict) else [],
        hardware=hardware,
        profiling_metrics=_extract_numeric_metrics(profiling_metrics),
        best_speedup=speedup,
        best_latency_ms=best_latency_ms,
        success=bool(kwargs.get("success", False)),
        best_strategy=strategy_name[:200],
        best_change_category=best_change_category,
        what_worked=what_worked,
        what_failed=what_failed,
        dead_ends=dead_ends,
        key_insight=key_insight[:500],
        trajectory_sketch=trajectory[:500],
        patch_content=patch_content,
        code_changes_summary=code_changes_summary,
        patch_file=patch_file,
        final_report_path=str(report_dir / "final_report.json") if report_dir else "",
        notebook_dir=str(report_dir / "_working_memory") if report_dir else "",
    )


def _extract_patch_content(patch_file: str, report_dir: Path | None) -> tuple[str, str]:
    """Read the actual patch/diff content and generate a code changes summary.

    Returns (patch_content, code_changes_summary).
    patch_content: the raw diff text (truncated to 5000 chars)
    code_changes_summary: human-readable summary of key code changes
    """
    patch_text = ""

    # Try the patch_file path first
    if patch_file:
        p = Path(patch_file)
        if p.exists() and p.stat().st_size > 0:
            try:
                patch_text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # Fallback: search for patches in the report directory
    if not patch_text and report_dir:
        for pattern in ("best_patch.diff", "best_patch.patch", "results/*/best_patch.diff"):
            import glob
            matches = glob.glob(str(report_dir / pattern))
            if not matches and report_dir.parent:
                matches = glob.glob(str(report_dir.parent / pattern))
            for m in sorted(matches, reverse=True):
                try:
                    candidate = Path(m).read_text(encoding="utf-8", errors="replace")
                    if candidate.strip():
                        patch_text = candidate
                        break
                except Exception:
                    continue
            if patch_text:
                break

    # Also check for modified kernel.py in worktrees
    if not patch_text and report_dir:
        for wt_kernel in sorted(report_dir.rglob("worktrees/*/kernel.py")):
            try:
                patch_text = f"# Modified kernel from {wt_kernel.parent.name}\n" + wt_kernel.read_text(
                    encoding="utf-8", errors="replace"
                )
                break
            except Exception:
                continue

    # Truncate large patches
    max_len = 5000
    if len(patch_text) > max_len:
        patch_text = patch_text[:max_len] + f"\n... (truncated, {len(patch_text)} chars total)"

    # Generate code changes summary from the diff
    summary = _summarize_patch(patch_text) if patch_text else ""

    return patch_text, summary


def _summarize_patch(patch_text: str) -> str:
    """Generate a human-readable summary of code changes from a diff."""
    if not patch_text:
        return ""

    lines = patch_text.splitlines()
    added = [l[1:].strip() for l in lines if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:].strip() for l in lines if l.startswith("-") and not l.startswith("---")]

    # Filter to meaningful code lines (skip blank, comments, imports)
    added_code = [l for l in added if l and not l.startswith("#") and len(l) > 10]
    removed_code = [l for l in removed if l and not l.startswith("#") and len(l) > 10]

    parts = []
    if added_code:
        parts.append(f"Added {len(added_code)} code lines")
        # Show the most distinctive added lines (likely the optimization)
        key_adds = [l for l in added_code if any(
            kw in l.lower() for kw in (
                "block", "tile", "warp", "tl.", "triton", "autotune",
                "config", "shared", "cache", "fuse", "vectori", "mfma",
                "atomic", "reduce", "parallel", "unroll", "pipeline",
            )
        )]
        if key_adds:
            parts.append("Key additions: " + "; ".join(key_adds[:3]))
    if removed_code:
        parts.append(f"Removed {len(removed_code)} code lines")

    return ". ".join(parts)[:500] if parts else ""


def _extract_numeric_metrics(profiling_metrics: dict) -> dict[str, Any]:
    """Extract only numeric metrics for fingerprinting (strip non-numeric fields)."""
    if not isinstance(profiling_metrics, dict):
        return {}
    metrics = profiling_metrics.get("metrics", profiling_metrics)
    if not isinstance(metrics, dict):
        return {}
    return {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float)) and v is not None
    }


def _extract_hardware(profiling_metrics: dict) -> str:
    """Try to extract GPU hardware info from profiling metrics."""
    if not isinstance(profiling_metrics, dict):
        return ""
    gpu_info = profiling_metrics.get("gpu_info", {})
    if isinstance(gpu_info, dict):
        model = gpu_info.get("model", "")
        if model:
            return str(model)
    return ""


def _infer_language(kernel_path: str) -> str:
    """Infer kernel language from file path and extension."""
    p = kernel_path.lower()
    if any(k in p for k in ("triton", ".py")):
        if "hip" in p or "rocm" in p:
            return "hip"
        return "triton"
    if any(k in p for k in (".hip", ".cpp", "hip")):
        return "hip"
    if any(k in p for k in (".cu", "cuda")):
        return "cuda"
    return "unknown"


def _classify_strategy(strategy_text: str) -> str:
    """Classify strategy text into a change category."""
    text = strategy_text.lower()
    if any(k in text for k in ("algorithm", "rewrite", "restructur", "new kernel")):
        return "algorithmic"
    if any(k in text for k in ("fuse", "fusion", "merge", "combine")):
        return "fusion"
    if any(k in text for k in ("tune", "block_size", "num_warps", "tile", "autotune")):
        return "tuning"
    return "wrapper"


def _extract_notebook_insights(report_dir: Path | None) -> tuple[list[str], list[str], list[str], str]:
    """Read working notebook events to extract what_worked/what_failed/dead_ends/trajectory."""
    worked: list[str] = []
    failed: list[str] = []
    dead_ends: list[str] = []
    trajectory_parts: list[str] = []

    if not report_dir:
        return worked, failed, dead_ends, ""

    notebook_dir = report_dir / "_working_memory" / "events"
    if not notebook_dir.is_dir():
        notebook_dir = report_dir.parent / "_working_memory" / "events"
    if not notebook_dir.is_dir():
        return worked, failed, dead_ends, ""

    try:
        events = []
        for jsonl_path in sorted(notebook_dir.glob("*.jsonl")):
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        events.sort(key=lambda e: e.get("ts", 0))

        for event in events:
            kind = event.get("kind", "")
            strategy = str(event.get("strategy") or "").strip()
            speedup = event.get("overall_speedup")
            tag = str(event.get("tag") or "")
            msg = str(event.get("message") or "")

            if kind == "result" and strategy:
                if speedup is not None:
                    sp = float(speedup)
                    entry = f"{strategy}: {sp:.2f}x"
                    if sp > 1.01:
                        worked.append(entry)
                        trajectory_parts.append(f"{strategy}({sp:.2f}x)")
                    elif sp < 0.99:
                        failed.append(entry)
                    else:
                        failed.append(f"{strategy}: no gain")
                elif tag == "FAIL":
                    reason = msg[:80] if msg else "failed"
                    failed.append(f"{strategy}: {reason}")

        # Dead ends: strategies tried 3+ times without improvement
        strategy_attempts: dict[str, int] = {}
        strategy_wins: dict[str, int] = {}
        for event in events:
            strategy = str(event.get("strategy") or "").strip()
            if not strategy or event.get("kind") != "result":
                continue
            cat = strategy.split("(")[0].strip()
            strategy_attempts[cat] = strategy_attempts.get(cat, 0) + 1
            sp = event.get("overall_speedup")
            if sp is not None and float(sp) > 1.01:
                strategy_wins[cat] = strategy_wins.get(cat, 0) + 1

        for cat, attempts in strategy_attempts.items():
            wins = strategy_wins.get(cat, 0)
            if attempts >= 3 and wins == 0:
                dead_ends.append(f"{cat}: 0/{attempts} improved")

    except Exception as exc:
        logger.debug("Notebook insight extraction failed: %s", exc)

    trajectory = " -> ".join(trajectory_parts[:10]) if trajectory_parts else ""

    return (
        _dedup(worked)[:10],
        _dedup(failed)[:10],
        _dedup(dead_ends)[:5],
        trajectory,
    )


def _extract_report_insights(report_dir: Path | None) -> tuple[str, str]:
    """Read final_report.json for key insight and best change category."""
    if not report_dir:
        return "", ""

    report_path = report_dir / "final_report.json"
    if not report_path.exists():
        for parent in [report_dir.parent, report_dir.parent.parent]:
            candidate = parent / "final_report.json"
            if candidate.exists():
                report_path = candidate
                break

    if not report_path.exists():
        return "", ""

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        insight = str(report.get("summary", "") or "")
        analysis = str(report.get("best_patch_analysis", "") or "")

        change_cat = ""
        if analysis:
            analysis_lower = analysis.lower()
            if "algorithm" in analysis_lower or "rewrite" in analysis_lower:
                change_cat = "algorithmic"
            elif "fuse" in analysis_lower or "fusion" in analysis_lower:
                change_cat = "fusion"
            elif "tune" in analysis_lower or "tile" in analysis_lower or "block" in analysis_lower:
                change_cat = "tuning"

        return insight[:500], change_cat
    except Exception:
        return "", ""


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
