"""Orchestrator tool implementations and dispatch router.

Each tool function receives a shared ``ctx`` dict (built by
``run_heterogeneous_orchestrator``) and returns a JSON string.
``dispatch_tool_call`` routes LLM tool calls to the correct function.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from minisweagent.debug_runtime import emit_debug_log

logger = logging.getLogger(__name__)


# ── generate_tasks ────────────────────────────────────────────────────


def tool_generate_tasks(
    ctx: dict[str, Any],
    round_num: int = 1,
    previous_results_dir: str | None = None,
    **_extra,
) -> str:
    """Generate optimisation tasks for a given round.

    Returns a JSON string with a ``tasks`` key listing the created task file paths.
    """
    from minisweagent.agents.heterogeneous.task_generator import generate_tasks as _gen

    output_dir = Path(ctx["output_dir"]) / "tasks" / f"round_{round_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    taskgen_model = ctx["model_factory"]() if ctx.get("model_factory") else ctx["model"]

    kernel_meta = ctx.get("kernel_meta") or {}
    kwargs: dict[str, Any] = {
        "base_task_context": "",
        "agent_class": ctx["agent_class"],
        "model": taskgen_model,
        "kernel_path": kernel_meta.get("kernel_path", str(ctx.get("kernel_path", ""))),
        "kernel_name": kernel_meta.get("kernel_name", ""),
        "kernel_type": kernel_meta.get("kernel_type", "unknown"),
        "kernel_language": kernel_meta.get("kernel_language", "python"),
        "function_names": kernel_meta.get("function_names", []),
        "workspace_path": kernel_meta.get("workspace_path", str(ctx.get("repo_root", ""))),
        "num_gpus": len(ctx.get("gpu_ids", [0])),
    }

    pp_dir = Path(ctx["preprocess_dir"])
    for attr, filename in [
        ("profiling_path", "profile.json"),
        ("commandment_path", "COMMANDMENT.md"),
        ("baseline_metrics_path", "baseline_metrics.json"),
        ("discovery_path", "discovery.json"),
        ("codebase_context_path", "CODEBASE_CONTEXT.md"),
    ]:
        p = pp_dir / filename
        if p.exists():
            kwargs[attr] = p

    if previous_results_dir:
        kwargs["previous_results_dir"] = Path(previous_results_dir)
    elif round_num > 1:
        prev_dir = Path(ctx["output_dir"]) / "results" / f"round_{round_num - 1}"
        if prev_dir.is_dir():
            kwargs["previous_results_dir"] = prev_dir

    tasks_parent = Path(ctx["output_dir"]) / "tasks"
    if tasks_parent.is_dir():
        kwargs["previous_tasks_dir"] = tasks_parent
        kwargs["current_round"] = round_num

    # Collect round evaluations from orchestrator context for memory injection
    _round_evals = []
    for _r in range(1, round_num):
        _eval_key = f"round_{_r}_eval"
        if _eval_key in ctx:
            _rev = ctx[_eval_key]
            if hasattr(_rev, "to_dict"):
                _rev = _rev.to_dict()
            elif not isinstance(_rev, dict):
                import dataclasses as _dc

                if _dc.is_dataclass(_rev) and not isinstance(_rev, type):
                    _rev = _dc.asdict(_rev)
            _round_evals.append(_rev)
        else:
            _eval_path = Path(ctx["output_dir"]) / f"round_{_r}_evaluation.json"
            if _eval_path.exists():
                try:
                    _round_evals.append(json.loads(_eval_path.read_text()))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.debug("tool_generate_tasks: could not load prior round eval %s: %s", _eval_path.name, exc)
    if _round_evals:
        kwargs["round_evaluations"] = _round_evals

    emit_debug_log(
        "heterogeneous_orchestrator:tool_generate_tasks:before_gen",
        "Invoking task generator with orchestrator model",
        {
            "round_num": round_num,
            "previous_results_dir": str(kwargs.get("previous_results_dir")),
        },
        hypothesis_id="H0",
    )

    try:
        tasks = _gen(**kwargs)
    except Exception as gen_exc:
        if "LimitsExceeded" in type(gen_exc).__name__ or "LimitsExceeded" in str(gen_exc):
            logger.warning(
                "Task generator hit limits (round %d), treating as convergence: %s",
                round_num,
                gen_exc,
            )
            return json.dumps({"tasks": [], "convergence": True, "reason": str(gen_exc)})
        raise

    emit_debug_log(
        "heterogeneous_orchestrator:tool_generate_tasks:after_gen",
        "Task generator completed",
        {
            "round_num": round_num,
            "task_count": len(tasks),
        },
        hypothesis_id="H0",
    )

    from minisweagent.agents.heterogeneous.task_generator import write_task_files

    task_file_paths = write_task_files(
        tasks,
        output_dir,
        kernel_path=str(ctx.get("kernel_path", "")),
        repo_root=str(ctx.get("repo_root", "")),
        commandment=str(kwargs.get("commandment_path", "")),
        baseline_metrics=str(kwargs.get("baseline_metrics_path", "")),
        profiling=str(kwargs.get("profiling_path", "")),
        codebase_context=str(kwargs.get("codebase_context_path", "")),
        benchmark_baseline=str(pp_dir / "benchmark_baseline.txt")
        if (pp_dir / "benchmark_baseline.txt").exists()
        else "",
        test_command=str(ctx.get("test_command", "")),
        starting_patch=str(ctx.get("starting_patch", "")),
        harness_path=str(ctx.get("harness_path", "")),
        round_num=round_num,
    )

    result: dict[str, Any] = {
        "tasks": [str(f) for f in task_file_paths],
        "round": round_num,
        "output_dir": str(output_dir),
        "task_summaries": [
            {
                "file": str(f),
                "label": tasks[i].label,
                "priority": tasks[i].priority,
                "num_gpus": tasks[i].num_gpus,
                "round": round_num,
            }
            for i, f in enumerate(task_file_paths)
        ],
    }
    return json.dumps(result, default=str)


# ── dispatch_tasks ────────────────────────────────────────────────────


def _dispatch_stage_name(priority: int) -> str:
    if priority <= 5:
        return "high"
    if priority <= 10:
        return "medium"
    return "low"


def _group_task_files_by_dispatch_stage(task_files: list[Path]) -> list[tuple[str, list[Path]]]:
    """Group tasks by priority tier for staged dispatch."""
    from minisweagent.run.task_file import read_task_file

    buckets: dict[str, list[Path]] = {}
    for tf in task_files:
        meta, _ = read_task_file(tf)
        pri = int(meta.get("priority", 10))
        stage = _dispatch_stage_name(pri)
        buckets.setdefault(stage, []).append(tf)
    order = ["high", "medium", "low"]
    return [(s, buckets[s]) for s in order if s in buckets]


def _stage_found_improvement(results_dir: Path, task_files: list[Path]) -> bool:
    """Return True if any task in the stage produced speedup > 1.0."""
    for task_file in task_files:
        meta, _ = __import__("minisweagent.run.task_file", fromlist=["read_task_file"]).read_task_file(task_file)
        label = str(meta.get("label") or task_file.stem)
        best_results_path = results_dir / label / "best_results.json"
        if not best_results_path.is_file():
            continue
        try:
            payload = json.loads(best_results_path.read_text())
            if float(payload.get("best_patch_speedup", 0) or 0) > 1.0:
                return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return False


def tool_dispatch_tasks(
    ctx: dict[str, Any],
    task_files: list[str] | None = None,
    **_extra,
) -> str:
    """Dispatch task files to GPUs for parallel execution.

    Returns a JSON summary of completed results.
    """
    from minisweagent.run.dispatch import run_task_batch

    output_dir = Path(ctx["output_dir"])
    gpu_ids = ctx.get("gpu_ids", [0])

    if not task_files:
        tasks_base = output_dir / "tasks"
        if tasks_base.is_dir():
            round_dirs = sorted(
                (d for d in tasks_base.iterdir() if d.is_dir() and d.name.startswith("round_")),
                key=lambda d: d.name,
            )
            if round_dirs:
                task_files = sorted(str(f) for f in round_dirs[-1].glob("*.md"))

    if not task_files:
        return json.dumps({"error": "No task files found"})

    task_paths = [Path(f) for f in task_files]
    logger.info("[bold yellow]Dispatching %d task(s)[/bold yellow] across GPU(s) %s.", len(task_paths), gpu_ids)
    _dispatch_t0 = time.monotonic()
    stages = _group_task_files_by_dispatch_stage(task_paths)
    round_match = None
    for tf in task_paths[:1]:
        for part in tf.parts:
            if part.startswith("round_"):
                round_match = part
                break
    results_base = output_dir / "results" / (round_match or "round_1")

    all_results: list[dict] = []
    for stage_name, stage_tasks in stages:
        logger.info("tool_dispatch_tasks: running stage '%s' (%d tasks).", stage_name, len(stage_tasks))
        stage_result = run_task_batch(
            task_files=stage_tasks,
            gpu_ids=gpu_ids,
            output_dir=results_base,
            model_factory=ctx.get("model_factory"),
        )
        all_results.append(
            {
                "stage": stage_name,
                "tasks": len(stage_tasks),
                "result": stage_result if isinstance(stage_result, dict) else str(stage_result),
            }
        )
        if _stage_found_improvement(results_base, stage_tasks):
            logger.info(
                "tool_dispatch_tasks: improvement found in stage '%s'; skipping lower-priority stages.", stage_name
            )
            for remaining_stage, _remaining_tasks in stages:
                if remaining_stage == stage_name:
                    continue
                if _dispatch_stage_name(0) == remaining_stage:
                    continue
            break

    _dispatch_elapsed = time.monotonic() - _dispatch_t0
    logger.info(
        "[bold green]Dispatch completed[/bold green] in %.1fs (%d stages).", _dispatch_elapsed, len(all_results)
    )

    return json.dumps(
        {
            "status": "completed",
            "results_dir": str(results_base),
            "stages": all_results,
        },
        default=str,
    )


# ── collect_results ───────────────────────────────────────────────────


def tool_collect_results(
    ctx: dict[str, Any],
    results_dir: str | None = None,
    **_extra,
) -> str:
    """Read and summarize results from completed tasks."""
    output_dir = Path(ctx["output_dir"])
    if results_dir:
        base = Path(results_dir)
    else:
        base = output_dir / "results"
        if base.is_dir():
            round_dirs = sorted(
                (d for d in base.iterdir() if d.is_dir() and d.name.startswith("round_")),
                key=lambda d: d.name,
            )
            base = round_dirs[-1] if round_dirs else base

    from minisweagent.agents.heterogeneous.result_scanning import scan_previous_results

    logger.debug("tool_collect_results: scanning %s", base)
    summary = scan_previous_results(base)
    if not summary:
        logger.info("tool_collect_results: no results found in %s.", base)
    return summary if summary else "No results found."


# ── finalize ──────────────────────────────────────────────────────────


def tool_finalize(
    ctx: dict[str, Any],
    summary: str,
    best_patch: str | None = None,
    total_speedup: str | None = None,
    **_extra,
) -> str:
    """Signal optimisation is complete.  Write the LLM's final report.

    The actual best-patch selection and FULL_BENCHMARK verification is
    done by ``post_round_evaluate`` + ``finalize_run`` in the orchestrator
    after this tool returns.  This tool just records the LLM's summary.
    """
    output_dir = Path(ctx["output_dir"])
    report = {
        "status": "complete",
        "summary": summary,
        "best_patch": best_patch,
        "total_speedup": total_speedup,
    }
    report_path = output_dir / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("tool_finalize: wrote LLM final report to %s (speedup=%s).", report_path.name, total_speedup)
    return json.dumps(report, default=str)


# ── Dispatch router ───────────────────────────────────────────────────


def dispatch_tool_call(
    ctx: dict[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    phase: str = "",
) -> str:
    """Route a tool call to the appropriate implementation.

    Exceptions are caught and returned as JSON error payloads so the
    orchestrator LLM can decide how to proceed.
    """
    ORCHESTRATION_TOOLS = {"generate_tasks", "dispatch_tasks", "collect_results", "finalize"}
    if phase == "explore" and tool_name in ORCHESTRATION_TOOLS:
        logger.debug("dispatch_tool_call: blocked %s during explore phase.", tool_name)
        return json.dumps(
            {
                "error": f"Cannot call {tool_name} during exploration phase. "
                "Please read and understand the kernel first, then respond with "
                "'Ready to begin optimization rounds' to proceed to the round loop."
            }
        )

    try:
        if tool_name == "generate_tasks":
            return tool_generate_tasks(ctx, **tool_args)
        if tool_name == "dispatch_tasks":
            return tool_dispatch_tasks(ctx, **tool_args)
        if tool_name == "collect_results":
            return tool_collect_results(ctx, **tool_args)
        if tool_name == "finalize":
            return tool_finalize(ctx, **tool_args)
        result = ctx["toolruntime"].dispatch({"name": tool_name, "arguments": tool_args})
        return json.dumps(result, default=str) if isinstance(result, dict) else str(result)
    except Exception as exc:
        logger.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
        return json.dumps({"error": f"{tool_name} failed: {exc}"})
