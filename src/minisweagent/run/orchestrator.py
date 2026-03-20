"""Orchestrator: LLM-driven agent that generates, dispatches, and manages
optimisation tasks across available GPUs.

The orchestrator sits between the preprocessor (which produces profiling
artefacts) and the per-task sub-agents (``geak --task``).

By default it runs in **homogeneous** mode: each round dispatches a
single task to ``geak --task <file> --num-parallel N``, so all agents
work on the same optimization task.

With ``--heterogeneous``, it runs in **heterogeneous** mode as an LLM
agent whose tools are:

* **generate_tasks** – invoke the task-generator to create task files
* **dispatch_tasks** – run tasks in parallel via ``ParallelAgent`` pool
* **collect_results** – read back results from completed tasks
* **finalize** – signal that optimisation is complete

All heavy lifting is done by the *existing* modules; the orchestrator
is a thin decision layer.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from minisweagent.run.postprocess.benchmark_parsing import (
    extract_latency_ms as _extract_latency_ms,
)
from minisweagent.run.postprocess.benchmark_parsing import (
    extract_reported_speedup as _extract_reported_speedup,
)
from minisweagent.run.postprocess.benchmark_parsing import (
    parse_shape_count as _parse_shape_count,
)
from minisweagent.run.postprocess.benchmark_parsing import (
    extract_benchmark_config_lines as _extract_benchmark_config_lines,
)
from minisweagent.run.postprocess.benchmark_parsing import (
    parse_total_kernel_time_ms as _parse_total_kernel_time_ms,
)
from minisweagent.debug_runtime import emit_debug_log, model_tools_snapshot
from minisweagent.run.utils.generated_artifacts import (
    apply_patch_with_generated_helper_fallback,
)
from minisweagent.run.utils.git_safe_env import get_git_safe_env
from minisweagent.run.pipeline_helpers import DEFAULT_HETEROGENEOUS, DEFAULT_PIPELINE_OUTPUT_DIR

# ── System prompt for the orchestrator LLM ───────────────────────────

_SYSTEM_PROMPT = """\
You are the GEAK orchestrator – an expert at planning and coordinating
GPU kernel optimisation.

You have been given the results of a preprocessing pipeline:
* Profiling data with per-kernel bottleneck analysis
* Baseline metrics (duration, throughput, bottleneck classification)
* A COMMANDMENT.md that specifies the rules every sub-agent must follow

You also have access to **bash** (execute shell commands),
**str_replace_editor** (view / edit files), **profile_kernel** (GPU
profiling), and **strategy_manager**.  Use these only when you need to
inspect artefacts, debug a failure, or gather information the
orchestration tools above cannot provide.

## IMPORTANT: Phased Execution

The orchestration runs in TWO phases:

### Phase 1: Exploration (current phase)
During exploration, you should ONLY:
- Read and understand the kernel source code
- Review profiling data and baseline metrics
- Analyze the COMMANDMENT.md
- Plan your optimization strategy

Do NOT call generate_tasks, dispatch_tasks, collect_results, or finalize
during exploration. Simply respond with "Ready to begin optimization rounds"
when you have finished exploring.

### Phase 2: Round Loop
The system will explicitly tell you "Begin round N" to start each round.
WAIT for this instruction before calling any orchestration tools.

Within each round you MUST call these tools in order:
1. **generate_tasks** – produce optimisation task files for this round.
2. **dispatch_tasks** – run those tasks in parallel across available GPUs.
3. **collect_results** – review what each task achieved.

After collect_results, respond with your evaluation and WAIT for the next
round instruction. The system will automatically run validation (FULL_BENCHMARK
and PROFILE) on the best kernel from each round.

Only call **finalize** when the system tells you it is the FINAL round.
The finalize call should include:
- summary: A comprehensive summary of optimizations achieved
- best_patch: Path to the best patch file
- total_speedup: The verified speedup (e.g., "1.06x" or "6%")

Rules:
- Do NOT modify preprocessor artefacts (test harness, test command,
  discovery, profiling, COMMANDMENT.md).
- Do NOT run tasks yourself; always dispatch via **dispatch_tasks**.
- Do NOT call finalize until explicitly told it is the FINAL round.
- After **collect_results**, review each sub-agent's output against
  its original task intent:
  1. Did it actually optimise the *kernel*, or did it modify something
     else (e.g. test harness, benchmark framework)?  Reject the latter.
  2. Did it report a before/after performance comparison using baseline
     metrics?  If not, note that the result is unverified.
  3. Did it violate the COMMANDMENT?  Reject if so.
  4. Did the correctness tests pass?  Reject if tests failed.
  Mark rejected results as "rejected" and explain why.
- For cross-round decisions, treat the system-provided FULL_BENCHMARK
  evaluation as canonical. Raw task-local speedups are provisional and
  may be noisy or invalidated by later verification.
"""

_INSTANCE_TEMPLATE = """\
## Preprocessor Context

Kernel: {kernel_path}
Repo root: {repo_root}
Test command: {test_command}
Available GPUs: {gpu_ids}
Output directory: {output_dir}

### Codebase Context (repo structure and key files)
{codebase_context}

### Baseline Metrics
{baseline_metrics_summary}

### Profiling Summary
{profiling_summary}

### COMMANDMENT (rules for sub-agents)
{commandment_excerpt}

{memory_context}

---

Begin by reading the kernel source and profiling data to understand the
optimisation landscape.  Then follow the round instructions.
"""


# ── Helper: build DiscoveryResult from discovery dict ────────────────
# Uses the canonical DiscoveryResult.from_dict() classmethod.


# ── Tool implementations ─────────────────────────────────────────────


def _tool_generate_tasks(
    ctx: dict[str, Any],
    round_num: int = 1,
    previous_results_dir: str | None = None,
    **_extra,
) -> str:
    """Generate optimisation tasks for a given round.

    If the task-generation sub-agent hits its step/cost limit, we treat
    that as convergence (no more tasks) rather than propagating the error.
    """
    from minisweagent.agents.heterogeneous.task_generator import generate_tasks as _gen

    output_dir = Path(ctx["output_dir"]) / "tasks" / f"round_{round_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    taskgen_model = ctx["model_factory"]() if ctx.get("model_factory") else ctx["model"]

    kwargs: dict[str, Any] = {
        "discovery_result": ctx["discovery_result"],
        "base_task_context": "",
        "agent_class": ctx["agent_class"],
        "model": taskgen_model,
        "num_gpus": len(ctx.get("gpu_ids", [0])),
    }

    pp_dir = Path(ctx["preprocess_dir"])
    profiling_path = pp_dir / "profile.json"
    if profiling_path.exists():
        kwargs["profiling_path"] = profiling_path
    commandment_path = pp_dir / "COMMANDMENT.md"
    if commandment_path.exists():
        kwargs["commandment_path"] = commandment_path
    baseline_path = pp_dir / "baseline_metrics.json"
    if baseline_path.exists():
        kwargs["baseline_metrics_path"] = baseline_path
    discovery_path = pp_dir / "discovery.json"
    if discovery_path.exists():
        kwargs["discovery_path"] = discovery_path
    codebase_ctx_path = pp_dir / "CODEBASE_CONTEXT.md"
    if codebase_ctx_path.exists():
        kwargs["codebase_context_path"] = codebase_ctx_path
    if previous_results_dir:
        kwargs["previous_results_dir"] = Path(previous_results_dir)

    # region agent log
    emit_debug_log(
        "orchestrator.py:_tool_generate_tasks:before_gen",
        "Invoking task generator with orchestrator model",
        {
            "round_num": round_num,
            "previous_results_dir": str(previous_results_dir) if previous_results_dir else None,
            "shared_model": model_tools_snapshot(ctx["model"]),
            "taskgen_model": model_tools_snapshot(taskgen_model),
        },
        hypothesis_id="H2",
    )
    # endregion

    try:
        tasks = _gen(**kwargs)
    except RuntimeError as exc:
        if "LimitsExceeded" in str(exc):
            logger.warning(
                "Task generator hit limits (round %d), treating as convergence: %s",
                round_num,
                str(exc)[:200],
            )
            return json.dumps({
                "tasks": [],
                "message": "Task generator hit step/cost limit – treating as convergence.",
            })
        raise

    # region agent log
    emit_debug_log(
        "orchestrator.py:_tool_generate_tasks:after_gen",
        "Task generator returned control to orchestrator",
        {
            "round_num": round_num,
            "task_count": len(tasks) if tasks else 0,
            "tasks": [
                {
                    "label": getattr(task, "label", None),
                    "agent_class": getattr(getattr(task, "agent_class", None), "__name__", None),
                }
                for task in (tasks or [])
            ],
            "shared_model_after": model_tools_snapshot(ctx["model"]),
            "taskgen_model_after": model_tools_snapshot(taskgen_model),
        },
        hypothesis_id="H2",
    )
    # endregion

    if not tasks:
        return json.dumps({"tasks": [], "message": "No tasks generated – converged."})

    from minisweagent.agents.agent_spec import _agent_class_to_type
    from minisweagent.run.task_file import write_task_file

    _AGENT_CLASS_TO_TYPE = _agent_class_to_type()

    task_files: list[str] = []
    for i, t in enumerate(tasks):
        fname = f"{i * 5:02d}_{t.label}.md"
        fpath = output_dir / fname
        pp_dir = Path(ctx.get("preprocess_dir") or ctx.get("output_dir", "."))
        metadata = {
            "label": t.label,
            "priority": t.priority,
            "agent_type": _AGENT_CLASS_TO_TYPE.get(t.agent_class, "strategy_agent"),
            "kernel_language": t.kernel_language,
            "kernel_path": ctx.get("kernel_path"),
            "repo_root": ctx.get("repo_root"),
            "test_command": ctx.get("test_command"),
            "harness_path": ctx.get("harness_path"),
            "commandment": str(pp_dir / "COMMANDMENT.md"),
            "baseline_metrics": str(pp_dir / "baseline_metrics.json"),
            "profiling": str(pp_dir / "profile.json"),
            "codebase_context": str(pp_dir / "CODEBASE_CONTEXT.md"),
            "benchmark_baseline": str(pp_dir / "benchmark_baseline.txt"),
            "num_gpus": t.num_gpus,
            "round": round_num,
        }
        write_task_file(fpath, metadata, t.task, relative_to=fpath.parent)
        task_files.append(str(fpath))

    return json.dumps({"tasks": task_files, "count": len(task_files)})


def _dispatch_stage_name(priority: int) -> str:
    """Map task priority to a staged dispatch band."""
    if priority <= 5:
        return "kernel_body"
    if priority <= 8:
        return "tuning_fallback"
    return "wrapper_fallback"


def _group_task_files_by_dispatch_stage(task_files: list[Path]) -> list[tuple[str, list[Path]]]:
    """Group task files into staged dispatch bands in priority order."""
    from minisweagent.run.task_file import read_task_file

    stage_order = ("kernel_body", "tuning_fallback", "wrapper_fallback")
    grouped: dict[str, list[tuple[int, Path]]] = {name: [] for name in stage_order}
    for task_file in task_files:
        try:
            meta, _ = read_task_file(task_file)
        except Exception:
            meta = {}
        try:
            priority = int(meta.get("priority", 10))
        except (TypeError, ValueError):
            priority = 10
        grouped[_dispatch_stage_name(priority)].append((priority, task_file))

    stages: list[tuple[str, list[Path]]] = []
    for stage_name in stage_order:
        entries = sorted(grouped[stage_name], key=lambda item: (item[0], item[1].name))
        if entries:
            stages.append((stage_name, [path for _, path in entries]))
    return stages


def _stage_found_improvement(results_dir: Path, task_files: list[Path]) -> bool:
    """Return True if any dispatched task produced a >1.0x candidate."""
    from minisweagent.run.task_file import read_task_file

    for task_file in task_files:
        try:
            meta, _ = read_task_file(task_file)
        except Exception:
            meta = {}
        label = str(meta.get("label") or task_file.stem)
        best_results_path = results_dir / label / "best_results.json"
        if not best_results_path.is_file():
            continue
        try:
            payload = json.loads(best_results_path.read_text())
            if float(payload.get("best_patch_speedup", 0) or 0) > 1.0:
                return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return False


def _tool_dispatch_tasks(
    ctx: dict[str, Any],
    task_files: list[str] | str | None = None,
    **_extra,
) -> str:
    """Dispatch task files to GPUs for parallel execution.

    The LLM may truncate the ``task_files`` list when there are many tasks.
    To guard against this, after parsing the provided list we scan the
    round directory for any ``.md`` task files that were not included and
    append them automatically.

    If ``task_files`` is omitted entirely, auto-discover from the most
    recent round's task directory.
    """
    from minisweagent.run.dispatch import run_task_batch

    if isinstance(task_files, str):
        task_files = json.loads(task_files)
    if task_files is None:
        task_files = []

    gpu_ids = ctx.get("gpu_ids", [0])
    base_dir = Path(ctx["output_dir"])

    # If no task files provided, auto-discover from the latest round dir
    if not task_files:
        tasks_base = base_dir / "tasks"
        if tasks_base.is_dir():
            round_dirs = sorted(
                (d for d in tasks_base.iterdir() if d.is_dir() and d.name.startswith("round_")),
                key=lambda d: d.name,
            )
            if round_dirs:
                task_files = sorted(str(f) for f in round_dirs[-1].glob("*.md"))
                logger.info(
                    "Auto-discovered %d task files from %s",
                    len(task_files), round_dirs[-1].name,
                )
        if not task_files:
            return json.dumps({"error": "No task files provided and none auto-discovered."})

    # Derive round directory from the first task file path
    round_dir = "round_1"
    task_dir: Path | None = None
    if task_files:
        first_parent = Path(task_files[0]).parent
        if first_parent.name.startswith("round_"):
            round_dir = first_parent.name
            task_dir = first_parent

    # Auto-discover any task files the LLM may have omitted
    if task_dir and task_dir.is_dir():
        provided = {str(Path(f).resolve()) for f in task_files}
        for md in sorted(task_dir.glob("*.md")):
            if str(md.resolve()) not in provided:
                task_files.append(str(md))
                logger.info("Auto-included missing task file: %s", md.name)

    results_dir = base_dir / "results" / round_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    task_paths = [Path(f) for f in task_files]

    # Inject starting_patch from prior rounds into task files
    sp = ctx.get("starting_patch")
    if sp:
        from minisweagent.run.task_file import read_task_file, write_task_file
        for tf in task_paths:
            meta, body = read_task_file(tf)
            if not meta.get("starting_patch"):
                meta["starting_patch"] = sp
                write_task_file(tf, meta, body, relative_to=tf.parent)

    stage_gating_enabled = os.environ.get("GEAK_STAGE_PRIORITY_GATING", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if not stage_gating_enabled:
        results = run_task_batch(
            task_files=task_paths,
            gpu_ids=gpu_ids,
            output_dir=results_dir,
            model_factory=ctx["model_factory"],
        )
        return json.dumps(results, default=str)

    grouped_stages = _group_task_files_by_dispatch_stage(task_paths)
    aggregate: dict[str, Any] = {
        "completed": 0,
        "failed": 0,
        "results": [],
        "results_dir": str(results_dir),
        "stages": [],
    }

    for stage_name, stage_files in grouped_stages:
        logger.info(
            "Dispatch stage %s with %d task(s)",
            stage_name,
            len(stage_files),
        )
        stage_result = run_task_batch(
            task_files=stage_files,
            gpu_ids=gpu_ids,
            output_dir=results_dir,
            model_factory=ctx["model_factory"],
        )
        aggregate["completed"] += int(stage_result.get("completed", 0) or 0)
        aggregate["failed"] += int(stage_result.get("failed", 0) or 0)
        aggregate["results"].extend(stage_result.get("results", []))

        improvement_found = _stage_found_improvement(results_dir, stage_files)
        aggregate["stages"].append(
            {
                "stage": stage_name,
                "task_count": len(stage_files),
                "improvement_found": improvement_found,
            }
        )
        if improvement_found:
            break

    return json.dumps(aggregate, default=str)


def _tool_collect_results(
    ctx: dict[str, Any],
    results_dir: str | None = None,
    **_extra,
) -> str:
    """Read results from a completed round and return a summary.

    If ``results_dir`` is omitted, auto-derive from the most recent
    round's results directory.
    """
    from minisweagent.agents.heterogeneous.task_generator import _scan_previous_results

    if not results_dir:
        base = Path(ctx["output_dir"]) / "results"
        if base.is_dir():
            round_dirs = sorted(
                (d for d in base.iterdir() if d.is_dir() and d.name.startswith("round_")),
                key=lambda d: d.name,
            )
            if round_dirs:
                results_dir = str(round_dirs[-1])
                logger.info("Auto-discovered results dir: %s", results_dir)
        if not results_dir:
            return json.dumps({"error": "No results_dir provided and none auto-discovered."})

    results_path = Path(results_dir)
    if not results_path.is_dir():
        return json.dumps({"error": f"Results directory not found: {results_dir}"})

    summary = _scan_previous_results(results_path)
    return summary if summary else "No results found."


def _tool_finalize(
    ctx: dict[str, Any],
    summary: str,
    best_patch: str | None = None,
    total_speedup: str | None = None,
    **_extra,
) -> str:
    """Signal optimisation is complete.  Write final report.
    
    If best_patch or total_speedup are not provided, attempts to auto-detect
    them from the results directory.
    """
    output_dir = Path(ctx["output_dir"])
    
    # Auto-detect best patch and speedup if not provided
    if best_patch is None or total_speedup is None:
        best_speedup_val = 0.0
        best_patch_file = None
        
        results_dir = output_dir / "results"
        if results_dir.is_dir():
            for round_dir in sorted(results_dir.iterdir()):
                if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
                    continue
                for task_dir in sorted(round_dir.iterdir()):
                    if not task_dir.is_dir() or task_dir.name == "worktrees":
                        continue
                    br_file = task_dir / "best_results.json"
                    if br_file.exists():
                        try:
                            br = json.loads(br_file.read_text())
                            speedup = float(br.get("best_patch_speedup", 0))
                            if speedup > best_speedup_val:
                                best_speedup_val = speedup
                                best_patch_file = br.get("best_patch_file")
                        except (json.JSONDecodeError, ValueError, TypeError):
                            continue
        
        if best_patch is None and best_patch_file:
            best_patch = best_patch_file
        if total_speedup is None and best_speedup_val > 0:
            total_speedup = f"{best_speedup_val:.4f}x"
    
    report = {
        "status": "complete",
        "summary": summary,
        "best_patch": best_patch,
        "total_speedup": total_speedup,
    }

    report_path = output_dir / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    best_verified_round_eval = _select_best_verified_round_evaluation(output_dir)
    if best_verified_round_eval is not None:
        merged = _merge_round_evaluation_into_final_report(
            ctx,
            output_dir,
            report,
            best_verified_round_eval,
        )
        return json.dumps(merged, indent=2, default=str)

    return json.dumps(report)


def _parse_reported_speedup(total_speedup: str | None) -> float | None:
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


def _extract_verified_speedup(round_eval: dict[str, Any]) -> float | None:
    """Return only FULL_BENCHMARK verified speedup."""
    full_benchmark = round_eval.get("full_benchmark", {})
    if isinstance(full_benchmark, dict):
        verified = full_benchmark.get("verified_speedup")
        if isinstance(verified, (int, float)):
            return float(verified)
    return None


def _round_eval_label(round_eval: dict[str, Any]) -> str:
    """Return a stable round label like ``round_2`` for a round evaluation."""
    round_val = round_eval.get("round")
    if isinstance(round_val, int):
        return f"round_{round_val}"
    text = str(round_val).strip()
    if not text:
        return ""
    return text if text.startswith("round_") else f"round_{text}"


def _round_eval_candidate_ms(round_eval: dict[str, Any]) -> float | None:
    """Return the candidate latency from FULL_BENCHMARK, if available."""
    full_benchmark = round_eval.get("full_benchmark", {})
    if isinstance(full_benchmark, dict):
        candidate = full_benchmark.get("candidate_ms")
        if isinstance(candidate, (int, float)) and candidate > 0:
            return float(candidate)
    return None


def _select_best_verified_round_evaluation(output_dir: Path) -> dict[str, Any] | None:
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

        verified = _extract_verified_speedup(round_eval)
        if verified is None:
            continue

        candidate_ms = _round_eval_candidate_ms(round_eval)
        best_patch = str(round_eval.get("best_patch") or "")
        best_task = str(round_eval.get("best_task") or "")
        round_label = _round_eval_label(round_eval) or eval_path.stem.replace("_evaluation", "")
        sort_key = (
            -float(verified),
            float(candidate_ms) if candidate_ms is not None else float("inf"),
            best_patch,
            best_task,
            round_label,
        )
        candidates.append((sort_key, round_eval))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _format_patch_label(patch_path: str | None) -> str:
    if not patch_path:
        return "unknown"
    patch = Path(patch_path)
    stem = patch.stem if patch.suffix else patch.name
    parent = patch.parent.name
    return f"{parent}/{stem}" if parent else stem


def _rewrite_summary_with_verified_selection(
    summary: str,
    round_eval: dict[str, Any],
    *,
    verified_speedup_raw: float,
    verified_speedup: float,
) -> str:
    """Return one canonical verified final-selection summary block."""
    best_patch_label = _format_patch_label(round_eval.get("best_patch"))
    best_task = round_eval.get("best_task") or Path(round_eval.get("best_patch", "")).parent.name or "unknown"
    full_benchmark = round_eval.get("full_benchmark", {})
    baseline_ms = full_benchmark.get("baseline_ms") if isinstance(full_benchmark, dict) else None
    candidate_ms = full_benchmark.get("candidate_ms") if isinstance(full_benchmark, dict) else None
    baseline_speedup = (
        full_benchmark.get("baseline_reported_speedup")
        if isinstance(full_benchmark, dict)
        else None
    )
    candidate_speedup = (
        full_benchmark.get("candidate_reported_speedup")
        if isinstance(full_benchmark, dict)
        else None
    )

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
        lines.append(
            f"- Full benchmark geomean: {baseline_ms:.6f} ms -> {candidate_ms:.6f} ms"
        )
    elif isinstance(baseline_speedup, (int, float)) and isinstance(candidate_speedup, (int, float)):
        lines.append(
            "- Full benchmark reported speedup: "
            f"{baseline_speedup:.4f}x -> {candidate_speedup:.4f}x"
        )
    return "\n".join(lines)


def _record_final_outcome(ctx: dict[str, Any], report: dict[str, Any]) -> None:
    """Record the final outcome using verified speedup when available."""
    try:
        from minisweagent.memory.cross_session_memory import classify_kernel_category
        from minisweagent.memory.integration import record_optimization_outcome

        speedup_val = report.get("verified_speedup")
        if not isinstance(speedup_val, (int, float)):
            parsed = _parse_reported_speedup(report.get("total_speedup"))
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


def _merge_round_evaluation_into_final_report(
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
    round_label = _round_eval_label(round_eval)
    if round_label:
        merged["best_round"] = round_label

    verified_speedup_raw = _extract_verified_speedup(round_eval)
    if verified_speedup_raw is not None:
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
        full_benchmark = round_eval.get("full_benchmark", {})
        if isinstance(full_benchmark, dict):
            baseline_ms = full_benchmark.get("baseline_ms")
            candidate_ms = full_benchmark.get("candidate_ms")
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
        canonical_summary = _rewrite_summary_with_verified_selection(
            existing_summary,
            round_eval,
            verified_speedup_raw=float(verified_speedup_raw),
            verified_speedup=verified_speedup,
        )
        if existing_summary.strip() and existing_summary.strip() != canonical_summary.strip():
            merged["agent_summary"] = existing_summary
        merged["summary"] = canonical_summary

    final_report_path.write_text(json.dumps(merged, indent=2, default=str))
    _record_final_outcome(ctx, merged)
    return merged



def _auto_finalize(
    ctx: dict[str, Any],
    _print,
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

    report = {
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

    best_verified_round_eval = _select_best_verified_round_evaluation(output_dir)
    report_path = output_dir / "final_report.json"
    if best_verified_round_eval is not None:
        merged = _merge_round_evaluation_into_final_report(
            ctx,
            output_dir,
            report,
            best_verified_round_eval,
        )
        _print(f"Auto-finalized: {merged.get('verification_note', summary_text)}")
        _print(f"Report written to: {report_path}")
        return merged

    report_path.write_text(json.dumps(report, indent=2))
    _print(f"Auto-finalized: {summary_text}")
    _print(f"Report written to: {report_path}")

    if not best_overall:
        return report

    try:
        from minisweagent.memory.cross_session_memory import classify_kernel_category
        from minisweagent.memory.integration import record_optimization_outcome
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


# ── Per-round evaluation (extracted to run/postprocess/evaluation.py) ──

from minisweagent.run.postprocess.evaluation import (
    PatchApplyError,
    build_eval_env as _build_eval_env,
    build_eval_script as _build_eval_script,
    cleanup_eval_worktree as _cleanup_eval_worktree,
    evaluate_round_best as _evaluate_round_best,
    setup_eval_worktree as _setup_eval_worktree,
)


# ── Homogeneous orchestrator ─────────────────────────────────────────


def _run_homogeneous_orchestrator(
    preprocess_ctx: dict[str, Any],
    gpu_ids: list[int],
    output_dir: Path,
    max_rounds: int,
    start_round: int,
    _print,
    console,
    model_factory=None,
) -> dict[str, Any]:
    """Run the orchestrator in homogeneous mode.

    Each round writes a single task file and dispatches it in-process
    via ``run_task_batch``, so all agents work on the same optimization
    task.  Per-round evaluation and early stopping are reused from the
    heterogeneous path.
    """
    from minisweagent.run.task_file import write_task_file

    pp_dir = output_dir
    starting_patch = preprocess_ctx.get("starting_patch")

    start_label = (
        f"rounds {start_round}-{max_rounds}" if start_round > 1
        else f"{max_rounds} rounds"
    )
    _print(
        f"[bold cyan]--- Orchestrator (homogeneous) starting ({start_label}, {len(gpu_ids)} GPUs) ---[/bold cyan]"
        if console
        else f"--- Orchestrator (homogeneous) starting ({start_label}, {len(gpu_ids)} GPUs) ---"
    )

    ctx: dict[str, Any] = {**preprocess_ctx, "output_dir": str(output_dir)}

    for round_num in range(start_round, max_rounds + 1):
        is_last = round_num == max_rounds
        round_header = (
            f"--- Homogeneous round {round_num}/{max_rounds}"
            f"{' (final)' if is_last else ''} ---"
        )
        _print(
            f"[bold cyan]{round_header}[/bold cyan]"
            if console else round_header
        )

        task_dir = output_dir / "tasks" / f"round_{round_num}"
        task_dir.mkdir(parents=True, exist_ok=True)

        task_file = task_dir / "00_optimize.md"
        metadata: dict[str, Any] = {
            "label": "kernel_optimization",
            "priority": 10,
            "agent_type": "strategy_agent",
            "kernel_path": preprocess_ctx.get("kernel_path"),
            "repo_root": preprocess_ctx.get("repo_root"),
            "test_command": preprocess_ctx.get("test_command"),
            "harness_path": preprocess_ctx.get("harness_path"),
            "commandment": str(pp_dir / "COMMANDMENT.md"),
            "baseline_metrics": str(pp_dir / "baseline_metrics.json"),
            "profiling": str(pp_dir / "profile.json"),
            "codebase_context": str(pp_dir / "CODEBASE_CONTEXT.md"),
            "benchmark_baseline": str(pp_dir / "benchmark_baseline.txt"),
            "round": round_num,
            "starting_patch": starting_patch,
        }

        task_body = (
            "Optimize this GPU kernel for maximum performance.\n\n"
            "Follow the workflow described in the pipeline instructions.\n"
            "Use the discovered tests and benchmarks for correctness and performance validation.\n"
            "Report final speedup when done."
        )

        write_task_file(task_file, metadata, task_body, relative_to=task_file.parent)
        _print(f"  Task file: {task_file}")

        results_dir = output_dir / "results" / f"round_{round_num}"
        results_dir.mkdir(parents=True, exist_ok=True)

        n_agents = len(gpu_ids)
        _print(f"  Dispatching: run_task_batch({task_file.name}, {n_agents} agents on {n_agents} GPUs)")
        try:
            from minisweagent.run.dispatch import run_task_batch

            run_task_batch(
                task_files=[task_file] * n_agents,
                gpu_ids=gpu_ids,
                output_dir=results_dir,
                model_factory=model_factory,
                console=console,
            )
        except Exception as exc:
            _print(f"  [yellow]Round {round_num} dispatch failed: {exc}[/yellow]")

        round_eval = _evaluate_round_best(
            ctx, round_num, results_dir, _print,
        )
        if round_eval:
            ctx[f"round_{round_num}_eval"] = round_eval
            if round_eval.get("best_patch"):
                current_speedup_val = (
                    round_eval.get("full_benchmark", {}).get("verified_speedup")
                    or round_eval.get("benchmark_speedup", 0)
                )
                best_global_speedup = ctx.get("_best_global_speedup", 0)
                if current_speedup_val >= best_global_speedup:
                    starting_patch = round_eval["best_patch"]
                    ctx["starting_patch"] = starting_patch
                    ctx["_best_global_speedup"] = current_speedup_val

        early_stop_threshold = float(os.getenv("GEAK_EARLY_STOP_THRESHOLD", "0.005"))
        if round_eval and round_num >= 2:
            current_speedup = (
                round_eval.get("full_benchmark", {}).get("verified_speedup")
                or round_eval.get("benchmark_speedup", 1.0)
            )
            prior_speedups = []
            for r in range(1, round_num):
                rev = ctx.get(f"round_{r}_eval", {})
                s = (
                    rev.get("full_benchmark", {}).get("verified_speedup")
                    or rev.get("benchmark_speedup", 1.0)
                )
                prior_speedups.append(s)
            best_prior = max(prior_speedups) if prior_speedups else 1.0
            if current_speedup <= best_prior * (1 + early_stop_threshold):
                _print(
                    f"  Early stopping: round {round_num} ({current_speedup:.4f}x) "
                    f"did not improve over prior best ({best_prior:.4f}x)"
                )
                break

    return _auto_finalize(ctx, _print)


# ── Orchestrator runner (heterogeneous) ──────────────────────────────


_ORCHESTRATOR_SWE_TOOLS = {"bash", "str_replace_editor", "profile_kernel", "strategy_manager"}

_ORCHESTRATOR_ONLY_TOOLS: list[dict] = [
    {
        "name": "generate_tasks",
        "description": (
            "Generate optimisation task files for a round.  Returns a JSON "
            "object with a 'tasks' list of file paths.  An empty list means "
            "convergence – no more optimisations to try."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "round_num": {
                    "type": "integer",
                    "description": "Round number (1-based).",
                },
                "previous_results_dir": {
                    "type": "string",
                    "description": "Path to previous round's results directory (optional for round 1).",
                },
            },
            "required": ["round_num"],
        },
    },
    {
        "name": "dispatch_tasks",
        "description": (
            "Dispatch a list of task files to available GPUs for parallel "
            "execution.  Returns a JSON summary of results.  If task_files "
            "is omitted, auto-discovers from the latest round's task directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of task file paths to dispatch (auto-discovered if omitted).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "collect_results",
        "description": (
            "Read results from a completed round's output directory.  "
            "Returns a Markdown summary of patches, test outputs, and logs.  "
            "If results_dir is omitted, auto-discovers the latest round."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "results_dir": {
                    "type": "string",
                    "description": "Path to the results directory to scan (auto-discovered if omitted).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "finalize",
        "description": (
            "Signal that optimisation is complete.  Provide a summary of "
            "what was achieved, the best patch, and total speedup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Human-readable summary of the optimisation.",
                },
                "best_patch": {
                    "type": "string",
                    "description": "Path or identifier of the best patch.",
                },
                "total_speedup": {
                    "type": "string",
                    "description": "Total speedup achieved (e.g. '15%').",
                },
            },
            "required": ["summary"],
        },
    },
]


def _build_tools_schema(toolruntime) -> list[dict]:
    """Merge ToolRuntime schemas (allowlisted) with orchestrator-specific tools."""
    swe_tools = [
        t for t in toolruntime.get_tools_schema()
        if t["name"] in _ORCHESTRATOR_SWE_TOOLS
    ]
    return swe_tools + _ORCHESTRATOR_ONLY_TOOLS


def _dispatch_tool_call(
    ctx: dict[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    phase: str = "",
) -> str:
    """Route a tool call to the appropriate implementation.

    Exceptions are caught and returned as JSON error payloads so the
    orchestrator LLM can decide how to proceed (retry, skip, or finalize).
    """
    # Block orchestration tools during exploration phase
    ORCHESTRATION_TOOLS = {"generate_tasks", "dispatch_tasks", "collect_results", "finalize"}
    if phase == "explore" and tool_name in ORCHESTRATION_TOOLS:
        return json.dumps({
            "error": f"Cannot call {tool_name} during exploration phase. "
            "Please read and understand the kernel first, then respond with "
            "'Ready to begin optimization rounds' to proceed to the round loop."
        })

    try:
        if tool_name == "generate_tasks":
            return _tool_generate_tasks(ctx, **tool_args)
        if tool_name == "dispatch_tasks":
            return _tool_dispatch_tasks(ctx, **tool_args)
        if tool_name == "collect_results":
            return _tool_collect_results(ctx, **tool_args)
        if tool_name == "finalize":
            return _tool_finalize(ctx, **tool_args)
        # Delegate to ToolRuntime (bash, str_replace_editor, profile_kernel, ...)
        result = ctx["toolruntime"].dispatch({"name": tool_name, "arguments": tool_args})
        return json.dumps(result, default=str) if isinstance(result, dict) else str(result)
    except Exception as exc:
        from minisweagent.utils.log import logger

        logger.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
        return json.dumps({"error": f"{tool_name} failed: {exc}"})


def run_orchestrator(
    preprocess_ctx: dict[str, Any],
    gpu_ids: list[int],
    model,
    model_factory,
    *,
    output_dir: Path | None = None,
    max_rounds: int | None = None,
    start_round: int = 1,
    heterogeneous: bool = DEFAULT_HETEROGENEOUS,
    console=None,
) -> dict[str, Any]:
    """Run the orchestrator agent loop.

    Parameters
    ----------
    preprocess_ctx:
        Context dict returned by ``run_preprocessor()``.
    gpu_ids:
        List of GPU device IDs available for task execution.
    model:
        LLM model instance for the orchestrator.
    model_factory:
        Callable returning a new model instance (for sub-agents).
    output_dir:
        Override output directory (defaults to preprocess_ctx source).
    max_rounds:
        Maximum optimisation rounds (default: from GEAK_MAX_ROUNDS env or 5).
        Each round = generate_tasks → dispatch_tasks → collect_results.
    start_round:
        Round number to start from (1-based, default 1).  When > 1 the
        exploration phase is skipped and prior round evaluations are loaded
        from ``round_<N>_evaluation.json`` files on disk.
    heterogeneous:
        If True, use LLM-generated diverse tasks per round (original behavior).
        If False (default), use homogeneous mode where all agents get the same task.
    console:
        Optional Rich console for progress messages.
    """
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    _out = output_dir or Path(preprocess_ctx.get("output_dir", DEFAULT_PIPELINE_OUTPUT_DIR))
    _out = Path(_out)
    _out.mkdir(parents=True, exist_ok=True)

    max_rounds = max_rounds or int(os.getenv("GEAK_MAX_ROUNDS", "5"))

    def _print(msg: str) -> None:
        if console:
            console.print(msg)
        else:
            print(msg, file=sys.stderr)

    if not heterogeneous:
        return _run_homogeneous_orchestrator(
            preprocess_ctx, gpu_ids, _out, max_rounds, start_round, _print, console,
            model_factory=model_factory,
        )

    # Build DiscoveryResult from preprocessor's discovery dict
    from minisweagent.run.preprocess.discovery_types import DiscoveryResult

    disc_dict = preprocess_ctx.get("discovery") or {}
    kernel_path = preprocess_ctx.get("kernel_path", "")
    discovery_result = DiscoveryResult.from_dict(disc_dict, kernel_path)

    # Determine the preprocessor artefacts directory
    preprocess_dir = _out
    for candidate in ("resolved.json", "discovery.json", "profile.json"):
        p = _out / candidate
        if p.exists():
            preprocess_dir = _out
            break
    else:
        preprocess_dir = _out

    from minisweagent.tools.tools_runtime import ToolRuntime

    toolruntime = ToolRuntime(tool_profile="full", use_strategy_manager=True)

    # Build orchestrator context shared across tool calls
    ctx: dict[str, Any] = {
        **preprocess_ctx,
        "discovery_result": discovery_result,
        "output_dir": str(_out),
        "preprocess_dir": str(preprocess_dir),
        "gpu_ids": gpu_ids,
        "model": model,
        "model_factory": model_factory,
        "agent_class": StrategyInteractiveAgent,
        "toolruntime": toolruntime,
    }

    # Set the orchestrator's tools on the model so it can use tool calling.
    # Copy the original tools so nested callers (e.g. task_generator) that also
    # mutate model_impl.tools don't corrupt our saved reference.
    tools_schema = _build_tools_schema(toolruntime)
    model_impl = getattr(model, "_impl", model)
    _orig = getattr(model_impl, "tools", None)
    original_tools = list(_orig) if isinstance(_orig, list) else _orig
    model_impl.tools = tools_schema

    # Prepare summaries for the instance prompt
    bm = preprocess_ctx.get("baseline_metrics") or {}
    bm_summary = json.dumps(bm, indent=2, default=str) if bm else "Not available"

    prof = preprocess_ctx.get("profiling") or {}
    prof_summary = json.dumps(prof, indent=2, default=str)[:2000] if prof else "Not available"

    cmd = preprocess_ctx.get("commandment") or ""
    cmd_excerpt = cmd[:1500] + ("..." if len(cmd) > 1500 else "") if cmd else "Not available"

    codebase_ctx = ""
    _codebase_ctx_path = preprocess_dir / "CODEBASE_CONTEXT.md"
    if _codebase_ctx_path.exists():
        codebase_ctx = _codebase_ctx_path.read_text().strip()

    _memory_context = ""
    try:
        from minisweagent.memory.integration import assemble_memory_context
        _bm = preprocess_ctx.get("baseline_metrics") or {}
        _memory_context = assemble_memory_context(
            kernel_path=str(preprocess_ctx.get("kernel_path", "")),
            bottleneck_type=_bm.get("bottleneck"),
            profiling_metrics=_bm,
        )
        if _memory_context:
            _memory_context = "### Optimization Memory (from past runs)\n" + _memory_context
    except Exception as _mem_exc:
        logger.debug("Memory context assembly failed: %s", _mem_exc)

    _working_mem = None
    try:
        from minisweagent.memory.integration import is_working_memory_enabled

        if is_working_memory_enabled():
            from minisweagent.memory.cross_session_memory import classify_kernel_category
            from minisweagent.memory.working_memory import WorkingMemory

            _kpath = str(preprocess_ctx.get("kernel_path", ""))
            _wm_notebook_dir = str(_out / "_working_memory")
            _working_mem = WorkingMemory(
                kernel_category=classify_kernel_category(_kpath) if _kpath else "unknown",
                max_steps=int(os.getenv("GEAK_AGENT_STEP_LIMIT", "100")),
                notebook_dir=_wm_notebook_dir,
                notebook_writer_id="orchestrator",
            )
            _bm_dict = preprocess_ctx.get("baseline_metrics") or {}
            if _bm_dict.get("bottleneck"):
                _working_mem.bottleneck_type = str(_bm_dict["bottleneck"])
            if _bm_dict.get("benchmark_duration_us"):
                _working_mem.baseline_latency_ms = float(_bm_dict["benchmark_duration_us"]) / 1000.0
            elif _bm_dict.get("duration_us"):
                _working_mem.baseline_latency_ms = float(_bm_dict["duration_us"]) / 1000.0
            _working_mem.sync_notebook_baseline()
            ctx["working_memory"] = _working_mem
    except Exception as _wm_exc:
        logger.debug("WorkingMemory init failed: %s", _wm_exc)

    # Build messages
    instance_msg = _INSTANCE_TEMPLATE.format(
        kernel_path=str(preprocess_ctx.get("kernel_path", "N/A")),
        repo_root=str(preprocess_ctx.get("repo_root", "N/A")),
        test_command=str(preprocess_ctx.get("test_command", "N/A")),
        gpu_ids=str(gpu_ids),
        output_dir=str(_out),
        codebase_context=codebase_ctx or "Not available",
        baseline_metrics_summary=bm_summary,
        profiling_summary=prof_summary,
        commandment_excerpt=cmd_excerpt,
        memory_context=_memory_context,
    )

    start_label = (
        f"rounds {start_round}-{max_rounds}" if start_round > 1
        else f"{max_rounds} rounds"
    )
    _print(
        f"[bold cyan]--- Orchestrator starting ({start_label}, {len(gpu_ids)} GPUs) ---[/bold cyan]"
        if console
        else f"--- Orchestrator starting ({start_label}, {len(gpu_ids)} GPUs) ---"
    )

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": instance_msg},
    ]

    # When resuming from a later round, load prior round evaluations
    # into ctx and messages so the LLM has full context.
    if start_round > 1:
        for prev_round in range(1, start_round):
            eval_path = _out / f"round_{prev_round}_evaluation.json"
            if eval_path.exists():
                try:
                    round_eval = json.loads(eval_path.read_text())
                    ctx[f"round_{prev_round}_eval"] = round_eval
                    eval_summary = json.dumps(round_eval, indent=2, default=str)[:2000]
                    messages.append({
                        "role": "user",
                        "content": (
                            f"## Round {prev_round} Evaluation (prior run)\n\n"
                            f"The best kernel from round {prev_round} was evaluated "
                            f"with FULL_BENCHMARK and PROFILE:\n```\n{eval_summary}\n```\n"
                            "Use this data to inform your strategy."
                        ),
                    })
                    _print(f"  Loaded prior evaluation: {eval_path.name}")
                except (json.JSONDecodeError, OSError) as exc:
                    _print(f"  Warning: could not load {eval_path.name}: {exc}")

    try:
        if start_round <= 1:
            # Phase 1: Exploration -- let the LLM read files and understand the kernel
            _print(
                "[bold cyan]--- Exploration phase ---[/bold cyan]"
                if console
                else "--- Exploration phase ---"
            )
            finalize_result = _run_llm_steps(
                model, messages, ctx, _print, console, phase="explore",
            )
            if finalize_result is not None:
                return finalize_result

        # Phase 2: Round loop
        for round_num in range(start_round, max_rounds + 1):
            is_last = round_num == max_rounds
            round_header = (
                f"--- Round {round_num}/{max_rounds}"
                f"{' (final round)' if is_last else ''} ---"
            )
            _print(
                f"[bold cyan]{round_header}[/bold cyan]"
                if console
                else round_header
            )

            # Inject a user-level instruction for this round
            if is_last:
                round_instruction = (
                    f"Begin round {round_num} (FINAL round). "
                    "Call generate_tasks, dispatch_tasks, collect_results, "
                    "then call **finalize** with a full summary of the best "
                    "results across all rounds."
                )
            else:
                round_instruction = (
                    f"Begin round {round_num}/{max_rounds}. "
                    "Call generate_tasks, dispatch_tasks, collect_results. "
                    "Then evaluate the results and respond with your analysis. "
                    "Focus on strategies not yet tried or that build on "
                    "previous successes. For later-round decisions, prefer the "
                    "system-provided FULL_BENCHMARK verified outcomes over raw "
                    "task-local speedup claims."
                )
            messages.append({"role": "user", "content": round_instruction})

            finalize_result = _run_llm_steps(
                model, messages, ctx, _print, console,
                phase=f"round_{round_num}",
            )

            # Per-round evaluation: FULL_BENCHMARK + PROFILE on best kernel
            round_results_dir = _out / "results" / f"round_{round_num}"
            round_eval = _evaluate_round_best(
                ctx, round_num, round_results_dir, _print,
            )
            if round_eval:
                ctx[f"round_{round_num}_eval"] = round_eval
                if _working_mem:
                    _working_mem.record_round_evaluation(round_eval)
                # Feed evaluation into next round's context
                eval_summary = json.dumps(round_eval, indent=2, default=str)[:2000]
                messages.append({
                    "role": "user",
                    "content": (
                        f"## Round {round_num} Evaluation\n\n"
                        f"The best kernel from round {round_num} was evaluated "
                        f"with FULL_BENCHMARK and PROFILE:\n```\n{eval_summary}\n```\n"
                        "Use this data to inform your next-round strategy. "
                        "Treat the FULL_BENCHMARK result as canonical and use "
                        "task-local speedups only as supporting evidence."
                    ),
                })
            # Warm-start: update starting_patch for next round
            if round_eval and round_eval.get("best_patch"):
                current_speedup_val = (
                    round_eval.get("full_benchmark", {}).get("verified_speedup")
                    or round_eval.get("benchmark_speedup", 0)
                )
                best_global_speedup = ctx.get("_best_global_speedup", 0)
                if current_speedup_val >= best_global_speedup:
                    ctx["starting_patch"] = round_eval["best_patch"]
                    ctx["_best_global_speedup"] = current_speedup_val

            if finalize_result is not None:
                if round_eval:
                    finalize_result = _merge_round_evaluation_into_final_report(
                        ctx,
                        _out,
                        finalize_result,
                        round_eval,
                    )
                else:
                    _record_final_outcome(ctx, finalize_result)
                return finalize_result
    finally:
        if original_tools is not None:
            model_impl.tools = original_tools
        elif hasattr(model_impl, "tools"):
            model_impl.tools = []

    _print(
        "[yellow]Orchestrator completed all rounds without calling finalize – auto-selecting best result...[/yellow]"
        if console
        else "Orchestrator completed all rounds without calling finalize – auto-selecting best result..."
    )

    return _auto_finalize(ctx, _print)


def _run_llm_steps(
    model,
    messages: list[dict],
    ctx: dict[str, Any],
    _print,
    console,
    *,
    phase: str,
) -> dict[str, Any] | None:
    """Run LLM tool-call steps until the LLM responds with text or calls ``finalize``.

    Returns a finalize report dict if the LLM called ``finalize``,
    otherwise ``None`` (the LLM responded with text, signalling it is
    ready for the next phase).
    """
    max_steps = int(os.getenv("GEAK_ORCHESTRATOR_STEP_LIMIT", "200"))
    step = 0
    _wm = ctx.get("working_memory")

    while step < max_steps:
        step += 1
        _print(
            f"[dim]{phase} step {step}[/dim]"
            if console
            else f"{phase} step {step}"
        )

        if _wm and phase != "explore":
            _wm.update_step(step, 0.0)
            _wm_text = _wm.format_for_injection()
            if _wm_text and not any("[Working Memory" in m.get("content", "") for m in messages[-3:]):
                messages.append({"role": "user", "content": f"[Working Memory Update]\n{_wm_text}"})

        response = model.query(messages)

        content_text = response.get("content", "") if isinstance(response, dict) else ""
        tool_call = response.get("tools") if isinstance(response, dict) else None

        if not tool_call:
            if phase.startswith("round_") and any(
                name in content_text for name in ("dispatch_tasks", "collect_results", "finalize")
            ):
                # region agent log
                emit_debug_log(
                    "orchestrator.py:_run_llm_steps:no_tool_call",
                    "Orchestrator produced text mentioning missing orchestration tools",
                    {
                        "phase": phase,
                        "step": step,
                        "content_preview": content_text[:300],
                        "model_tools": model_tools_snapshot(model),
                    },
                    hypothesis_id="H3",
                )
                # endregion
            if content_text:
                _print(f"  Orchestrator: {content_text[:300]}")
            messages.append({"role": "assistant", "content": content_text})
            return None

        tool_name = tool_call.get("function", {}).get("name", "")
        tool_args = tool_call.get("function", {}).get("arguments", {})
        tool_id = tool_call.get("id", f"call_{phase}_{step}")

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {}

        _print(f"  Tool: {tool_name}({json.dumps(tool_args)[:200]})")

        messages.append(
            {
                "role": "assistant",
                "content": content_text,
                "tool_calls": tool_call,
            }
        )

        result_str = _dispatch_tool_call(ctx, tool_name, tool_args, phase=phase)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result_str,
            }
        )

        _print(f"  Result: {result_str[:300]}")

        if _wm:
            try:
                from minisweagent.memory.working_memory import extract_insight_from_tool_result
                insight = extract_insight_from_tool_result(tool_name, result_str, 0)
                if insight:
                    insight.step = step
                    _wm.insights.append(insight)
                    if len(_wm.insights) > 5:
                        _wm.insights = _wm.insights[-5:]
                    if "speedup" in (insight.message or "").lower():
                        import re as _re
                        _sp = _re.search(r'(\d+\.\d+)x', insight.message)
                        if _sp:
                            _wm.update_speedup(float(_sp.group(1)))
            except Exception:
                pass

        if tool_name == "finalize":
            try:
                report = json.loads(result_str)
            except json.JSONDecodeError:
                report = {"summary": result_str}
            _print(
                "[bold green]Orchestrator: Optimisation finalised.[/bold green]"
                if console
                else "Orchestrator: Optimisation finalised."
            )
            return report

    logger.warning(
        "Orchestrator hit step limit (%d) for phase %s -- proceeding to next phase",
        max_steps,
        phase,
    )
    _print(f"  Step limit ({max_steps}) reached for {phase}, moving on...")
    return None


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    """CLI: ``geak-orchestrate --preprocess-dir <dir> [--gpu-ids 0,1] [--max-rounds 3]``."""
    import argparse

    from minisweagent.run.pipeline_helpers import DEFAULT_HETEROGENEOUS

    parser = argparse.ArgumentParser(
        description="GEAK orchestrator: LLM-driven task generation, dispatch, and iteration loop",
    )
    parser.add_argument(
        "--preprocess-dir",
        required=True,
        help="Directory containing preprocessor artefacts (resolved.json, discovery.json, profile.json, ...)",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU device IDs (default: all detected GPUs, or 0 as fallback)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum optimisation rounds (default: GEAK_MAX_ROUNDS env or 5)",
    )
    parser.add_argument(
        "--start-round",
        type=int,
        default=1,
        help="Round to resume from (1-based, default: 1). "
             "Skips exploration and loads prior round evaluations from disk.",
    )
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        default=DEFAULT_HETEROGENEOUS,
        help="Use LLM-generated diverse tasks per round. Default: homogeneous (all agents get the same task).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: from GEAK_MODEL env or geak.yaml)",
    )
    from minisweagent.run.pipeline_helpers import add_agent_filter_args, apply_agent_filter_env

    add_agent_filter_args(parser)
    args = parser.parse_args()
    apply_agent_filter_env(args)

    pp_dir = Path(args.preprocess_dir).resolve()
    if not pp_dir.is_dir():
        print(f"ERROR: preprocess directory not found: {args.preprocess_dir}", file=sys.stderr)
        sys.exit(1)

    # Reconstruct preprocessor context from artefact files
    ctx: dict[str, Any] = {}

    resolved_path = pp_dir / "resolved.json"
    if resolved_path.exists():
        ctx["resolved"] = json.loads(resolved_path.read_text())
        ctx["kernel_path"] = ctx["resolved"].get("local_file_path", "")
        _repo_root = ctx["resolved"].get("local_repo_path", "")
        ctx["repo_root"] = str(Path(_repo_root).resolve()) if _repo_root else ""

    disc_path = pp_dir / "discovery.json"
    if disc_path.exists():
        ctx["discovery"] = json.loads(disc_path.read_text())
        focused = ctx["discovery"].get("focused_test") or {}
        if focused.get("focused_command"):
            ctx["test_command"] = focused["focused_command"]
        else:
            tests = ctx["discovery"].get("tests", [])
            ctx["test_command"] = tests[0]["command"] if tests else None
        if focused.get("focused_test_file"):
            ctx["harness_path"] = focused["focused_test_file"]

    # Prefer the UnitTestAgent's generated harness over the discovery
    # focused test -- the generated harness supports --benchmark etc.
    harness_txt = pp_dir / "harness_path.txt"
    if harness_txt.exists():
        ctx["harness_path"] = harness_txt.read_text().strip()

    prof_path = pp_dir / "profile.json"
    if prof_path.exists():
        ctx["profiling"] = json.loads(prof_path.read_text())

    bm_path = pp_dir / "baseline_metrics.json"
    if bm_path.exists():
        ctx["baseline_metrics"] = json.loads(bm_path.read_text())

    cmd_path = pp_dir / "COMMANDMENT.md"
    if cmd_path.exists():
        ctx["commandment"] = cmd_path.read_text()

    # Create model
    from minisweagent.run.pipeline_helpers import geak_model_factory, load_geak_model

    model = load_geak_model(args.model)
    _model_factory = geak_model_factory(args.model)
    if args.gpu_ids:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]
    else:
        from minisweagent.agents.agent_spec import detect_available_gpus

        gpu_ids = detect_available_gpus()

    try:
        from rich.console import Console

        console = Console()
    except ImportError:
        console = None

    # Print effective configuration at startup
    _sep = "=" * 60
    print(_sep)
    print("  GEAK-v3 Orchestrator Configuration")
    print(_sep)
    print(f"  preprocess_dir:       {pp_dir}")
    print(f"  gpu_ids:              {gpu_ids}")
    print(f"  max_rounds:           {args.max_rounds}")
    print(f"  start_round:          {args.start_round}")
    print(f"  heterogeneous:        {args.heterogeneous}")
    print(f"  model (cli):          {args.model}")
    print(f"  model (resolved):     {model.config if hasattr(model, 'config') else model}")
    print("-" * 60)
    print(f"  GEAK_MODEL:                 {os.environ.get('GEAK_MODEL', '<not set>')}")
    print(f"  GEAK_MODEL_ENSEMBLE:        {os.environ.get('GEAK_MODEL_ENSEMBLE', '<not set>')}")
    print(f"  GEAK_EXCLUDED_AGENTS:       {os.environ.get('GEAK_EXCLUDED_AGENTS', '<not set>')}")
    print(f"  GEAK_EARLY_STOP_THRESHOLD:  {os.environ.get('GEAK_EARLY_STOP_THRESHOLD', '<not set>')}")
    print(f"  GEAK_AGENT_STEP_LIMIT:      {os.environ.get('GEAK_AGENT_STEP_LIMIT', '<not set>')}")
    print(f"  GEAK_TASKGEN_STEP_LIMIT:    {os.environ.get('GEAK_TASKGEN_STEP_LIMIT', '<not set>')}")
    print(f"  GEAK_ORCHESTRATOR_STEP_LIMIT: {os.environ.get('GEAK_ORCHESTRATOR_STEP_LIMIT', '<not set>')}")
    print(f"  GEAK_BENCHMARK_ITERATIONS:  {os.environ.get('GEAK_BENCHMARK_ITERATIONS', '<not set>')}")
    print(f"  AITER_ROOT:                 {os.environ.get('AITER_ROOT', '<not set>')}")
    print("-" * 60)
    print(f"  kernel_path:          {ctx.get('kernel_path', '<unknown>')}")
    print(f"  harness_path:         {ctx.get('harness_path', '<unknown>')}")
    print(f"  test_command:         {ctx.get('test_command', '<unknown>')}")
    print(f"  repo_root:            {ctx.get('repo_root', '<unknown>')}")
    print(_sep)
    print(flush=True)

    report = run_orchestrator(
        preprocess_ctx=ctx,
        gpu_ids=gpu_ids,
        model=model,
        model_factory=_model_factory,
        output_dir=pp_dir,
        max_rounds=args.max_rounds,
        start_round=args.start_round,
        heterogeneous=args.heterogeneous,
        console=console,
    )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
