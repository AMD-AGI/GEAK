"""Orchestrator: LLM-driven agent that generates, dispatches, and manages
optimisation tasks across available GPUs.

The orchestrator sits between the preprocessor (which produces profiling
artefacts) and the per-task sub-agents (``geak --from-task``).  It is
itself an LLM agent whose tools are:

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
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── System prompt for the orchestrator LLM ───────────────────────────

_SYSTEM_PROMPT = """\
You are the GEAK orchestrator – an expert at planning and coordinating
GPU kernel optimisation.

You have been given the results of a preprocessing pipeline:
* Profiling data with per-kernel bottleneck analysis
* Baseline metrics (duration, throughput, bottleneck classification)
* A COMMANDMENT.md that specifies the rules every sub-agent must follow

Your job is to drive an iterative optimisation loop:

1. Call **generate_tasks** to produce a set of optimisation task files.
   Each task targets a specific strategy (kernel fusion, memory coalescing,
   vectorisation, OpenEvolve parameter tuning, etc.).
2. Call **dispatch_tasks** to run those tasks in parallel across available
   GPUs.  Strategy tasks each use 1 GPU; OpenEvolve tasks may use 2+.
3. Call **collect_results** to review what each task achieved.
4. Decide whether to iterate (generate new tasks building on successes,
   or trying strategies not yet explored) or to finalise.
5. When no further improvement is possible, call **finalize** with a
   summary of the best results.

Rules:
- Do NOT modify preprocessor artefacts (test harness, test command,
  discovery, profiling, COMMANDMENT.md).
- Do NOT run tasks yourself; always dispatch via **dispatch_tasks**.
- After **collect_results**, review each sub-agent's output against
  its original task intent:
  1. Did it actually optimise the *kernel*, or did it modify something
     else (e.g. test harness, benchmark framework)?  Reject the latter.
  2. Did it report a before/after performance comparison using baseline
     metrics?  If not, note that the result is unverified.
  3. Did it violate the COMMANDMENT?  Reject if so.
  4. Did the correctness tests pass?  Reject if tests failed.
  Mark rejected results as "rejected" and explain why.
- When the last round showed no improvement, finalise with a summary
  that lists each task, its result status, and measurable gains.
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

---

Use the tools below to optimise the kernel.  Start by generating tasks.
"""


# ── Helper: build DiscoveryResult from discovery dict ────────────────


def _build_discovery_result(disc_dict: dict, kernel_path: str):
    """Convert an automated_test_discovery JSON dict into a DiscoveryResult."""
    from minisweagent.tools.discovery import (
        BenchmarkInfo,
        DiscoveryResult,
        KernelInfo,
        TestInfo,
    )

    kp = Path(kernel_path)
    kernel_info = disc_dict.get("kernel") or {}
    kernels = []
    if kernel_info.get("file"):
        kernels.append(
            KernelInfo(
                file_path=Path(kernel_info["file"]),
                kernel_name=kernel_info.get("name", kp.stem),
                kernel_type=kernel_info.get("type", "unknown"),
                kernel_language="python" if kp.suffix == ".py" else "cpp",
                function_names=kernel_info.get("functions", []),
            )
        )
    tests = [
        TestInfo(
            file_path=Path(t["file"]),
            test_type=t.get("type", "script"),
            command=t.get("command", ""),
            confidence=t.get("confidence", 0.5),
        )
        for t in (disc_dict.get("tests") or [])
    ]
    benchmarks = [
        BenchmarkInfo(
            file_path=Path(b["file"]),
            bench_type=b.get("type", "script"),
            command=b.get("command", ""),
            confidence=b.get("confidence", 0.5),
        )
        for b in (disc_dict.get("benchmarks") or [])
    ]
    workspace = Path(disc_dict.get("workspace", kp.parent))
    return DiscoveryResult(
        kernels=kernels,
        tests=tests,
        benchmarks=benchmarks,
        workspace_path=workspace,
    )


# ── Tool implementations ─────────────────────────────────────────────


def _tool_generate_tasks(
    ctx: dict[str, Any],
    round_num: int = 1,
    previous_results_dir: str | None = None,
) -> str:
    """Generate optimisation tasks for a given round."""
    from minisweagent.run.task_generator import generate_tasks as _gen

    output_dir = Path(ctx["output_dir"]) / "tasks" / f"round_{round_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "discovery_result": ctx["discovery_result"],
        "base_task_context": "",
        "agent_class": ctx["agent_class"],
        "model": ctx["model"],
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

    tasks = _gen(**kwargs)

    if not tasks:
        return json.dumps({"tasks": [], "message": "No tasks generated – converged."})

    from minisweagent.agents.openevolve_worker import OpenEvolveWorker
    from minisweagent.agents.swe_agent import SweAgent
    from minisweagent.run.task_file import write_task_file

    _AGENT_CLASS_TO_TYPE = {OpenEvolveWorker: "openevolve", SweAgent: "swe_agent"}

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
            "commandment": str(pp_dir / "COMMANDMENT.md"),
            "baseline_metrics": str(pp_dir / "baseline_metrics.json"),
            "profiling": str(pp_dir / "profile.json"),
            "codebase_context": str(pp_dir / "CODEBASE_CONTEXT.md"),
            "round": round_num,
        }
        write_task_file(fpath, metadata, t.task)
        task_files.append(str(fpath))

    return json.dumps({"tasks": task_files, "count": len(task_files)})


def _tool_dispatch_tasks(
    ctx: dict[str, Any],
    task_files: list[str],
) -> str:
    """Dispatch task files to GPUs for parallel execution."""
    from minisweagent.run.dispatch import run_task_batch

    gpu_ids = ctx.get("gpu_ids", [0])
    base_dir = Path(ctx["output_dir"])

    # Derive round from the task file path (e.g. .../tasks/round_1/00_foo.md)
    round_dir = "round_1"
    if task_files:
        parent_name = Path(task_files[0]).parent.name
        if parent_name.startswith("round_"):
            round_dir = parent_name

    results_dir = base_dir / "results" / round_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    results = run_task_batch(
        task_files=[Path(f) for f in task_files],
        gpu_ids=gpu_ids,
        output_dir=results_dir,
        model_factory=ctx["model_factory"],
    )

    return json.dumps(results, default=str)


def _tool_collect_results(
    ctx: dict[str, Any],
    results_dir: str,
) -> str:
    """Read results from a completed round and return a summary."""
    from minisweagent.run.task_generator import _scan_previous_results

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
) -> str:
    """Signal optimisation is complete.  Write final report."""
    report = {
        "status": "complete",
        "summary": summary,
        "best_patch": best_patch,
        "total_speedup": total_speedup,
    }

    report_path = Path(ctx["output_dir"]) / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    return json.dumps(report)


def _auto_finalize(
    ctx: dict[str, Any],
    _print,
) -> dict[str, Any]:
    """Auto-select the best result across all rounds when step limit is hit.

    Scans every ``results/round_N/*/best_results.json`` and picks the task
    with the highest speedup, then writes ``final_report.json``.
    """
    from minisweagent.run.task_generator import _scan_previous_results

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

    report = {
        "status": "auto_finalized",
        "summary": summary_text,
        "best_round": best_round,
        "best_task": best_task,
        "best_speedup": best_speedup,
        "best_patch": best_overall.get("best_patch_file") if best_overall else None,
        "best_patch_analysis": best_overall.get("llm_selection_analysis") if best_overall else None,
        "round_summaries": round_summaries,
    }

    report_path = output_dir / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _print(f"Auto-finalized: {summary_text}")
    _print(f"Report written to: {report_path}")

    return report


# ── Orchestrator runner ──────────────────────────────────────────────


def _build_tools_schema() -> list[dict]:
    """Return the JSON-schema tool descriptors for the orchestrator LLM."""
    return [
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
                "execution.  Returns a JSON summary of results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task file paths to dispatch.",
                    },
                },
                "required": ["task_files"],
            },
        },
        {
            "name": "collect_results",
            "description": (
                "Read results from a completed round's output directory.  "
                "Returns a Markdown summary of patches, test outputs, and logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "results_dir": {
                        "type": "string",
                        "description": "Path to the results directory to scan.",
                    },
                },
                "required": ["results_dir"],
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


def _dispatch_tool_call(
    ctx: dict[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
) -> str:
    """Route a tool call to the appropriate implementation.

    Exceptions are caught and returned as JSON error payloads so the
    orchestrator LLM can decide how to proceed (retry, skip, or finalize).
    """
    try:
        if tool_name == "generate_tasks":
            return _tool_generate_tasks(ctx, **tool_args)
        if tool_name == "dispatch_tasks":
            return _tool_dispatch_tasks(ctx, **tool_args)
        if tool_name == "collect_results":
            return _tool_collect_results(ctx, **tool_args)
        if tool_name == "finalize":
            return _tool_finalize(ctx, **tool_args)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
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
        Max orchestrator steps (default: from GEAK_MAX_STEPS env or 30).
    console:
        Optional Rich console for progress messages.
    """
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    _out = output_dir or Path(preprocess_ctx.get("output_dir", "geak_output"))
    _out = Path(_out)
    _out.mkdir(parents=True, exist_ok=True)

    max_steps = max_rounds or int(os.getenv("GEAK_MAX_STEPS", "30"))

    def _print(msg: str) -> None:
        if console:
            console.print(msg)
        else:
            print(msg, file=sys.stderr)

    # Build DiscoveryResult from preprocessor's discovery dict
    disc_dict = preprocess_ctx.get("discovery") or {}
    kernel_path = preprocess_ctx.get("kernel_path", "")
    discovery_result = _build_discovery_result(disc_dict, kernel_path)

    # Determine the preprocessor artefacts directory
    preprocess_dir = _out
    for candidate in ("resolved.json", "discovery.json", "profile.json"):
        p = _out / candidate
        if p.exists():
            preprocess_dir = _out
            break
    else:
        preprocess_dir = _out

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
    }

    # Set the orchestrator's tools on the model so it can use tool calling.
    # Copy the original tools so nested callers (e.g. task_generator) that also
    # mutate model_impl.tools don't corrupt our saved reference.
    tools_schema = _build_tools_schema()
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
    )

    _print(
        f"[bold cyan]--- Orchestrator starting (max {max_steps} steps, {len(gpu_ids)} GPUs) ---[/bold cyan]"
        if console
        else f"--- Orchestrator starting (max {max_steps} steps, {len(gpu_ids)} GPUs) ---"
    )

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": instance_msg},
    ]

    final_report: dict[str, Any] = {}

    try:
        for step in range(1, max_steps + 1):
            _print(
                f"[dim]Orchestrator step {step}/{max_steps}[/dim]"
                if console
                else f"Orchestrator step {step}/{max_steps}"
            )

            response = model.query(messages)

            # AMD model returns {"content": "...", "tools": {...}, "extra": {...}}
            content_text = response.get("content", "") if isinstance(response, dict) else ""
            tool_call = response.get("tools") if isinstance(response, dict) else None

            if not tool_call:
                if content_text:
                    _print(f"  Orchestrator: {content_text[:300]}")
                messages.append({"role": "assistant", "content": content_text})
                break

            tool_name = tool_call.get("function", {}).get("name", "")
            tool_args = tool_call.get("function", {}).get("arguments", {})
            tool_id = tool_call.get("id", f"call_{step}")

            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}

            _print(f"  Tool: {tool_name}({json.dumps(tool_args)[:200]})")

            # Add assistant message with tool_calls for the model's message history
            messages.append(
                {
                    "role": "assistant",
                    "content": content_text,
                    "tool_calls": tool_call,
                }
            )

            result_str = _dispatch_tool_call(ctx, tool_name, tool_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_str,
                }
            )

            _print(f"  Result: {result_str[:300]}")

            if tool_name == "finalize":
                try:
                    final_report = json.loads(result_str)
                except json.JSONDecodeError:
                    final_report = {"summary": result_str}
                _print(
                    "[bold green]Orchestrator: Optimisation finalised.[/bold green]"
                    if console
                    else "Orchestrator: Optimisation finalised."
                )
                return final_report
    finally:
        # Restore the model's original tools
        if original_tools is not None:
            model_impl.tools = original_tools
        elif hasattr(model_impl, "tools"):
            model_impl.tools = []

    _print(
        "[yellow]Orchestrator reached step limit without finalising – auto-selecting best result...[/yellow]"
        if console
        else "Orchestrator reached step limit without finalising – auto-selecting best result..."
    )

    return _auto_finalize(ctx, _print)


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    """CLI: ``geak-orchestrate --preprocess-dir <dir> [--gpu-ids 0,1] [--max-rounds 3]``."""
    import argparse

    import yaml

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
        default="0",
        help="Comma-separated GPU device IDs (default: 0)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum orchestrator steps (default: GEAK_MAX_STEPS env or 30)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: from GEAK_MODEL env or geak.yaml)",
    )
    args = parser.parse_args()

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
    from minisweagent.config import get_config_path
    from minisweagent.models import get_model

    geak_cfg = get_config_path("geak")
    model_config: dict[str, Any] = {}
    if geak_cfg.exists():
        full_cfg = yaml.safe_load(geak_cfg.read_text()) or {}
        model_config = full_cfg.get("model", {})

    model = get_model(args.model, config=model_config)
    gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]

    try:
        from rich.console import Console

        console = Console()
    except ImportError:
        console = None

    report = run_orchestrator(
        preprocess_ctx=ctx,
        gpu_ids=gpu_ids,
        model=model,
        model_factory=lambda: get_model(args.model, config=model_config),
        output_dir=pp_dir,
        max_rounds=args.max_rounds,
        console=console,
    )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
