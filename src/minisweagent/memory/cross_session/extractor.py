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
    report_dir = _resolve_report_dir(patch_file, kernel_path)

    what_worked, what_failed, dead_ends, trajectory = _extract_notebook_insights(report_dir)
    all_task_outcomes = _extract_all_task_outcomes(report_dir)
    what_worked.extend(all_task_outcomes.get("worked", []))
    what_failed.extend(all_task_outcomes.get("failed", []))
    dead_ends.extend(all_task_outcomes.get("dead_ends", []))
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


def _resolve_report_dir(patch_file: str, kernel_path: str) -> Path | None:
    """Find the GEAK logs directory containing final_report.json and working notebook.

    Searches upward from patch_file, then looks for _logs directories near kernel_path.
    """
    candidates: list[Path] = []

    if patch_file:
        p = Path(patch_file)
        # Walk up from patch file looking for final_report.json or _working_memory
        for parent in [p.parent] + list(p.parents)[:6]:
            if (parent / "final_report.json").exists() or (parent / "_working_memory").exists():
                candidates.append(parent)
                break
            # Check for _logs sibling
            for sibling in parent.parent.iterdir() if parent.parent else []:
                if sibling.name.endswith("_logs") and sibling.is_dir():
                    if (sibling / "final_report.json").exists() or (sibling / "_working_memory").exists():
                        candidates.append(sibling)
                        break
            if candidates:
                break

    if kernel_path and not candidates:
        kp = Path(kernel_path)
        for parent in [kp.parent] + list(kp.parents)[:6]:
            for child in parent.iterdir() if parent.exists() else []:
                if child.name.endswith("_logs") and child.is_dir():
                    candidates.append(child)
                    break
            if candidates:
                break

    return candidates[0] if candidates else (Path(patch_file).parent if patch_file else None)


def _extract_patch_content(patch_file: str, report_dir: Path | None) -> tuple[str, str]:
    """Read the actual patch/diff content and generate a code changes summary.

    Searches multiple locations because GEAK's heterogeneous orchestrator
    writes patches in various structures:
      - patch_file (from final_report.json best_patch)
      - results/round_N/best_patch.diff (orchestrator-created)
      - results/round_N/<task_name>/patch_N.patch (per-task patches)
      - results/round_N/worktrees/slot_N/kernel.py (modified kernels)

    Returns (patch_content, code_changes_summary).
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

    # Fallback: search report_dir and parent directories
    search_roots = []
    if report_dir:
        search_roots.append(report_dir)
        if report_dir.parent:
            search_roots.append(report_dir.parent)

    if not patch_text:
        import glob
        for root in search_roots:
            # Try all patch patterns in priority order
            for pattern in (
                "best_patch.diff",
                "best_patch.patch",
                "results/*/best_patch.diff",
                "results/*/best_patch.patch",
                "results/*/*/patch_*.patch",  # per-task patches
                "results/*/*/patch_*.diff",
            ):
                matches = glob.glob(str(root / pattern))
                # Sort by modification time (newest first) to get the best patch
                matches.sort(key=lambda m: Path(m).stat().st_mtime, reverse=True)
                for m in matches:
                    try:
                        candidate = Path(m).read_text(encoding="utf-8", errors="replace")
                        if candidate.strip() and len(candidate) > 20:
                            patch_text = candidate
                            break
                    except Exception:
                        continue
                if patch_text:
                    break
            if patch_text:
                break

    # Fallback: find best modified kernel.py in worktrees (diff against original)
    if not patch_text:
        for root in search_roots:
            wt_kernels = sorted(root.rglob("worktrees/*/kernel.py"))
            if not wt_kernels:
                continue
            # Find the original kernel.py (should be at repo root)
            orig = root / "kernel.py"
            if not orig.exists():
                for parent_check in [root.parent, root.parent.parent]:
                    candidate_orig = parent_check / "kernel.py"
                    if candidate_orig.exists():
                        orig = candidate_orig
                        break

            if orig.exists():
                import subprocess
                for wt_kernel in wt_kernels:
                    try:
                        diff_result = subprocess.run(
                            ["diff", "-u", str(orig), str(wt_kernel)],
                            capture_output=True, text=True, timeout=5,
                        )
                        if diff_result.stdout.strip() and len(diff_result.stdout) > 20:
                            patch_text = diff_result.stdout
                            break
                    except Exception:
                        continue
            else:
                # No original to diff against -- just take the modified kernel
                for wt_kernel in wt_kernels:
                    try:
                        patch_text = f"# Modified kernel from {wt_kernel.parent.name}\n" + wt_kernel.read_text(
                            encoding="utf-8", errors="replace"
                        )
                        break
                    except Exception:
                        continue
            if patch_text:
                break

    # Truncate large patches but keep enough to be useful
    max_len = 8000
    if len(patch_text) > max_len:
        patch_text = patch_text[:max_len] + f"\n... (truncated, {len(patch_text)} chars total)"

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

    # Also extract round evaluation insights (verified vs task-local discrepancies)
    _extract_round_eval_insights(report_dir, worked, failed, dead_ends, trajectory_parts)

    trajectory = " -> ".join(trajectory_parts[:10]) if trajectory_parts else ""

    return (
        _dedup(worked)[:15],
        _dedup(failed)[:15],
        _dedup(dead_ends)[:8],
        trajectory,
    )


def _extract_all_task_outcomes(report_dir: Path | None) -> dict[str, list[str]]:
    """Scan ALL round/task directories for outcomes, not just the best task."""
    result: dict[str, list[str]] = {"worked": [], "failed": [], "dead_ends": []}
    if not report_dir:
        return result

    results_dir = report_dir / "results"
    if not results_dir.is_dir():
        return result

    seen_strategies: set[str] = set()

    for round_dir in sorted(results_dir.glob("round_*")):
        if not round_dir.is_dir():
            continue
        round_name = round_dir.name

        for task_dir in sorted(round_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name == "worktrees":
                continue

            br_file = task_dir / "best_results.json"
            if not br_file.exists():
                continue

            try:
                br = json.loads(br_file.read_text())
                speedup = float(br.get("best_patch_speedup", 0))
                task_name = task_dir.name
                analysis = str(br.get("llm_selection_analysis", ""))[:150]

                if task_name in seen_strategies:
                    continue
                seen_strategies.add(task_name)

                if speedup > 1.02:
                    result["worked"].append(f"{round_name}/{task_name}: {speedup:.3f}x")
                elif speedup < 0.98 and speedup > 0:
                    result["failed"].append(f"{round_name}/{task_name}: {speedup:.3f}x regression")
                    if analysis:
                        result["dead_ends"].append(f"{task_name}: {analysis[:100]}")
            except Exception:
                continue

    return result


def _extract_round_eval_insights(
    report_dir: Path | None,
    worked: list[str],
    failed: list[str],
    dead_ends: list[str],
    trajectory_parts: list[str],
) -> None:
    """Extract insights from round_N_evaluation.json files.

    Captures verified speedups, task-local vs verified discrepancies,
    and per-shape breakdowns.
    """
    if not report_dir:
        return

    for eval_path in sorted(report_dir.glob("round_*_evaluation.json")):
        try:
            d = json.loads(eval_path.read_text())
            round_num = d.get("round", "?")
            task = d.get("best_task", "?")
            bm_speedup = d.get("benchmark_speedup", 0)
            fb = d.get("full_benchmark", {})
            verified = fb.get("verified_speedup") if isinstance(fb, dict) else None

            if verified is not None and bm_speedup > 0:
                if verified > 1.01:
                    worked.append(f"R{round_num} {task}: verified {verified:.2f}x")
                    trajectory_parts.append(f"R{round_num}:{task}({verified:.2f}x)")
                elif verified < 0.99:
                    discrepancy = f"task-local={bm_speedup:.2f}x but verified={verified:.2f}x"
                    failed.append(f"R{round_num} {task}: REGRESSED ({discrepancy})")
                    if bm_speedup > 1.1 and verified < 0.95:
                        dead_ends.append(
                            f"{task}: task-local showed {bm_speedup:.2f}x gain but "
                            f"verified at {verified:.2f}x (regression on full shapes)"
                        )

            # Per-shape data
            per_shape = d.get("per_shape_speedups", {})
            if per_shape:
                slow_shapes = [s for s, v in per_shape.items() if isinstance(v, dict) and v.get("speedup", 1) < 0.9]
                fast_shapes = [s for s, v in per_shape.items() if isinstance(v, dict) and v.get("speedup", 1) > 1.5]
                if slow_shapes:
                    failed.append(f"R{round_num}: shapes {slow_shapes[:3]} regressed")
                if fast_shapes and slow_shapes:
                    dead_ends.append(
                        f"R{round_num} {task}: helped {len(fast_shapes)} shapes but hurt {len(slow_shapes)} -- "
                        f"needs shape-specialized dispatch"
                    )
        except Exception:
            continue


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
