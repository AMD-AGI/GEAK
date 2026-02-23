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
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _parse_median_latency_ms(output: str) -> float | None:
    """Extract median latency (ms) from harness benchmark output."""
    m = re.search(
        r"median\s+latency[\w\s]*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms",
        output,
        re.IGNORECASE,
    )
    return float(m.group(1)) if m else None


def _parse_shape_count(output: str) -> int | None:
    """Extract shape count from harness benchmark output."""
    m = re.search(r"(\d+)\s+shapes", output, re.IGNORECASE)
    return int(m.group(1)) if m else None


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

Within each round you MUST call these tools in order:
1. **generate_tasks** – produce optimisation task files for this round.
2. **dispatch_tasks** – run those tasks in parallel across available GPUs.
3. **collect_results** – review what each task achieved.

After collecting results, respond with your evaluation.  If the system
tells you there are more rounds, start the next round.  When all rounds
are done (or no further improvement is possible), call **finalize** with
a summary.

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
    from minisweagent.run.task_generator import generate_tasks as _gen

    output_dir = Path(ctx["output_dir"]) / "tasks" / f"round_{round_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "discovery_result": ctx["discovery_result"],
        "base_task_context": "",
        "agent_class": ctx["agent_class"],
        "model": ctx["model"],
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

    results = run_task_batch(
        task_files=[Path(f) for f in task_files],
        gpu_ids=gpu_ids,
        output_dir=results_dir,
        model_factory=ctx["model_factory"],
    )

    return json.dumps(results, default=str)


def _tool_collect_results(
    ctx: dict[str, Any],
    results_dir: str | None = None,
    **_extra,
) -> str:
    """Read results from a completed round and return a summary.

    If ``results_dir`` is omitted, auto-derive from the most recent
    round's results directory.
    """
    from minisweagent.run.task_generator import _scan_previous_results

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


# ── Per-round evaluation ─────────────────────────────────────────────


def _setup_eval_worktree(repo_root: str, patch_file: str, output_dir: Path) -> Path:
    """Create a temporary worktree and apply the best patch.

    Returns the worktree path.  The caller is responsible for cleanup
    via ``_cleanup_eval_worktree``.
    """
    eval_dir = output_dir / "_eval_worktree"
    if eval_dir.exists():
        shutil.rmtree(eval_dir, ignore_errors=True)

    repo = Path(repo_root).resolve()
    is_git = (repo / ".git").exists() or (repo / ".git").is_file()

    if is_git:
        wt_name = f"eval_{os.getpid()}"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(eval_dir)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        )
    else:
        shutil.copytree(str(repo), str(eval_dir), dirs_exist_ok=True)

    patch_path = Path(patch_file)
    if patch_path.exists() and patch_path.stat().st_size > 0:
        apply_result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path)],
            cwd=str(eval_dir),
            capture_output=True,
            text=True,
        )
        if apply_result.returncode != 0:
            logger.warning(
                "git apply failed (rc=%d): %s",
                apply_result.returncode,
                apply_result.stderr[:500],
            )
    return eval_dir


def _cleanup_eval_worktree(repo_root: str, eval_dir: Path) -> None:
    """Remove the temporary evaluation worktree."""
    repo = Path(repo_root).resolve()
    is_git = (repo / ".git").exists() or (repo / ".git").is_file()
    if is_git:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(eval_dir)],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
    if eval_dir.exists():
        shutil.rmtree(eval_dir, ignore_errors=True)


def _build_eval_env(
    work_dir: Path,
    repo_root: str,
    harness_path: str,
    gpu_id: int,
) -> dict[str, str]:
    """Build the GEAK_* environment dict for evaluation subprocesses."""
    env = os.environ.copy()
    env["GEAK_WORK_DIR"] = str(work_dir)
    env["GEAK_REPO_ROOT"] = repo_root
    env["GEAK_HARNESS"] = harness_path
    env["GEAK_GPU_DEVICE"] = str(gpu_id)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{work_dir}:{repo_root}:{env.get('PYTHONPATH', '')}"
    alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in alloc_conf:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


def _build_eval_script(
    commandment_path: str,
    sections: list[str],
) -> str | None:
    """Build a shell script from one or more COMMANDMENT sections.

    Returns the path to the written script, or None if no commands.
    """
    from minisweagent.run.dispatch import _read_commandment_section

    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    has_commands = False
    for sec in sections:
        body = _read_commandment_section(commandment_path, sec)
        if body:
            lines.append(f"# --- {sec} ---")
            lines.append(body)
            has_commands = True
    if not has_commands:
        return None
    script_dir = Path(commandment_path).parent
    script_path = script_dir / "_geak_eval_cmd.sh"
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    return str(script_path)


def _evaluate_round_best(
    ctx: dict[str, Any],
    round_num: int,
    results_dir: Path,
    _print,
) -> dict[str, Any] | None:
    """Evaluate the single best kernel from a round with FULL_BENCHMARK + PROFILE.

    Creates a temporary worktree, applies the best patch, sets all GEAK_*
    env vars, runs SETUP + FULL_BENCHMARK, then profiles with PYTHONPATH
    pointing at the patched worktree.

    Returns a round evaluation dict, or None if no valid candidates exist.
    """
    output_dir = Path(ctx["output_dir"])
    pp_dir = Path(ctx.get("preprocess_dir", ctx.get("output_dir", ".")))

    best_patch_file: str | None = None
    best_speedup: float = 0.0
    best_task: str = ""

    if not results_dir.is_dir():
        return None

    for task_dir in sorted(results_dir.iterdir()):
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
                best_patch_file = br.get("best_patch_file")
                best_task = task_dir.name
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    if not best_patch_file or best_speedup <= 0:
        _print(f"  Round {round_num}: no valid candidates for evaluation")
        return None

    _print(f"  Round {round_num} best: {best_task} ({best_speedup:.2f}x)")

    round_eval: dict[str, Any] = {
        "round": round_num,
        "best_patch": best_patch_file,
        "best_task": best_task,
        "benchmark_speedup": best_speedup,
    }

    commandment_path = pp_dir / "COMMANDMENT.md"
    if not commandment_path.exists():
        eval_path = output_dir / f"round_{round_num}_evaluation.json"
        eval_path.write_text(json.dumps(round_eval, indent=2, default=str))
        return round_eval

    repo_root = ctx.get("repo_root", "")
    harness_path = ctx.get("harness_path", "")
    gpu_id = ctx.get("gpu_ids", [0])[0]

    eval_worktree: Path | None = None
    try:
        # Create worktree and apply patch
        eval_worktree = _setup_eval_worktree(repo_root, best_patch_file, output_dir)
        eval_env = _build_eval_env(eval_worktree, repo_root, harness_path, gpu_id)
        _print(f"  Eval worktree: {eval_worktree}")

        # Load baselines for comparison
        full_benchmark_baseline_path = pp_dir / "full_benchmark_baseline.txt"
        full_benchmark_baseline = (
            full_benchmark_baseline_path.read_text().strip()
            if full_benchmark_baseline_path.exists()
            else None
        )
        benchmark_baseline_path = pp_dir / "benchmark_baseline.txt"
        benchmark_baseline = (
            benchmark_baseline_path.read_text().strip()
            if benchmark_baseline_path.exists()
            else None
        )

        # --- FULL_BENCHMARK ---
        fb_script = _build_eval_script(str(commandment_path), ["SETUP", "FULL_BENCHMARK"])
        if fb_script:
            _print(f"  Running FULL_BENCHMARK on best kernel from round {round_num}...")
            try:
                fb_result = subprocess.run(
                    ["bash", fb_script],
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    cwd=str(eval_worktree),
                    env=eval_env,
                )
                fb_stdout = fb_result.stdout
                round_eval["full_benchmark"] = {
                    "stdout": fb_stdout[:5000],
                    "returncode": fb_result.returncode,
                    "success": fb_result.returncode == 0,
                }
                if full_benchmark_baseline:
                    round_eval["full_benchmark"]["baseline"] = full_benchmark_baseline[:2000]

                # Programmatic speedup verification
                if fb_result.returncode == 0:
                    candidate_ms = _parse_median_latency_ms(fb_stdout)
                    baseline_ref = full_benchmark_baseline or benchmark_baseline
                    baseline_ms = _parse_median_latency_ms(baseline_ref) if baseline_ref else None
                    if candidate_ms and baseline_ms and baseline_ms > 0:
                        verified_speedup = baseline_ms / candidate_ms
                        round_eval["full_benchmark"]["verified_speedup"] = round(verified_speedup, 4)
                        round_eval["full_benchmark"]["candidate_ms"] = candidate_ms
                        round_eval["full_benchmark"]["baseline_ms"] = baseline_ms
                        _print(
                            f"  FULL_BENCHMARK verified speedup: {verified_speedup:.4f}x "
                            f"({baseline_ms:.4f} ms -> {candidate_ms:.4f} ms)"
                        )
                    # Shape count validation
                    candidate_shapes = _parse_shape_count(fb_stdout)
                    baseline_shapes = _parse_shape_count(baseline_ref) if baseline_ref else None
                    if candidate_shapes and baseline_shapes and candidate_shapes != baseline_shapes:
                        logger.warning(
                            "Shape count mismatch: baseline=%d, candidate=%d",
                            baseline_shapes, candidate_shapes,
                        )
                        round_eval["full_benchmark"]["shape_count_warning"] = (
                            f"baseline={baseline_shapes}, candidate={candidate_shapes}"
                        )

                _print(f"  FULL_BENCHMARK: {'PASS' if fb_result.returncode == 0 else 'FAIL'}")
            except Exception as exc:
                _print(f"  FULL_BENCHMARK failed: {exc}")
                round_eval["full_benchmark"] = {"error": str(exc)}

        # --- PROFILE ---
        _print(f"  Running PROFILE on best kernel from round {round_num}...")
        baseline_metrics_path = pp_dir / "baseline_metrics.json"
        baseline_metrics = None
        if baseline_metrics_path.exists():
            try:
                baseline_metrics = json.loads(baseline_metrics_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        try:
            from minisweagent.run.pipeline_helpers import _ensure_mcp_importable

            _ensure_mcp_importable()
            from profiler_mcp.server import profile_kernel

            _profile_fn = getattr(profile_kernel, "fn", profile_kernel)
            if harness_path:
                profile_result = _profile_fn(
                    command=f"python {harness_path} --profile",
                    backend="metrix",
                    num_replays=3,
                    quick=True,
                    gpu_devices=str(gpu_id),
                    env={"PYTHONPATH": f"{eval_worktree}:{repo_root}"},
                )

                if baseline_metrics and profile_result:
                    from minisweagent.baseline_metrics import build_baseline_metrics

                    optimized_metrics = build_baseline_metrics(
                        profile_result, include_all=True
                    )
                    profile_comparison: dict[str, Any] = {}
                    for key in ("duration_us", "bottleneck"):
                        if key in baseline_metrics and key in optimized_metrics:
                            base_val = baseline_metrics[key]
                            opt_val = optimized_metrics[key]
                            if key == "duration_us" and isinstance(base_val, (int, float)):
                                change_pct = ((opt_val - base_val) / base_val * 100) if base_val else 0
                                profile_comparison[key] = {
                                    "baseline": base_val,
                                    "optimized": opt_val,
                                    "change_pct": round(change_pct, 1),
                                }
                            else:
                                profile_comparison[key] = {
                                    "baseline": base_val,
                                    "optimized": opt_val,
                                }

                    opt_bn = optimized_metrics.get("bottleneck", "unknown")
                    base_bn = baseline_metrics.get("bottleneck", "unknown")
                    if base_bn != opt_bn:
                        profile_comparison["bottleneck_shift"] = f"{base_bn} -> {opt_bn}"

                    round_eval["profile_comparison"] = profile_comparison
                    _print(f"  PROFILE comparison: {json.dumps(profile_comparison)[:300]}")
                else:
                    _print("  PROFILE: completed (no baseline for comparison)")
        except Exception as exc:
            _print(f"  PROFILE failed: {exc}")
            round_eval["profile_comparison"] = {"error": str(exc)}

    finally:
        if eval_worktree:
            _cleanup_eval_worktree(repo_root, eval_worktree)

    eval_path = output_dir / f"round_{round_num}_evaluation.json"
    eval_path.write_text(json.dumps(round_eval, indent=2, default=str))
    _print(f"  Round evaluation written to: {eval_path}")

    return round_eval


# ── Orchestrator runner ──────────────────────────────────────────────


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
    console:
        Optional Rich console for progress messages.
    """
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    _out = output_dir or Path(preprocess_ctx.get("output_dir", "geak_output"))
    _out = Path(_out)
    _out.mkdir(parents=True, exist_ok=True)

    max_rounds = max_rounds or int(os.getenv("GEAK_MAX_ROUNDS", "5"))

    def _print(msg: str) -> None:
        if console:
            console.print(msg)
        else:
            print(msg, file=sys.stderr)

    # Build DiscoveryResult from preprocessor's discovery dict
    from minisweagent.tools.discovery_types import DiscoveryResult

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
        f"[bold cyan]--- Orchestrator starting ({max_rounds} rounds, {len(gpu_ids)} GPUs) ---[/bold cyan]"
        if console
        else f"--- Orchestrator starting ({max_rounds} rounds, {len(gpu_ids)} GPUs) ---"
    )

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": instance_msg},
    ]

    try:
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
        for round_num in range(1, max_rounds + 1):
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
                    "previous successes."
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
                # Feed evaluation into next round's context
                eval_summary = json.dumps(round_eval, indent=2, default=str)[:2000]
                messages.append({
                    "role": "user",
                    "content": (
                        f"## Round {round_num} Evaluation\n\n"
                        f"The best kernel from round {round_num} was evaluated "
                        f"with FULL_BENCHMARK and PROFILE:\n```\n{eval_summary}\n```\n"
                        "Use this data to inform your next-round strategy."
                    ),
                })

            if finalize_result is not None:
                # Use last round eval as final report
                if round_eval:
                    finalize_result["round_evaluation"] = round_eval
                    final_eval_path = _out / "final_report.json"
                    if final_eval_path.exists():
                        try:
                            existing = json.loads(final_eval_path.read_text())
                            existing["round_evaluation"] = round_eval
                            final_eval_path.write_text(json.dumps(existing, indent=2, default=str))
                        except (json.JSONDecodeError, OSError):
                            pass
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
    while step < max_steps:
        step += 1
        _print(
            f"[dim]{phase} step {step}[/dim]"
            if console
            else f"{phase} step {step}"
        )

        response = model.query(messages)

        content_text = response.get("content", "") if isinstance(response, dict) else ""
        tool_call = response.get("tools") if isinstance(response, dict) else None

        if not tool_call:
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
        help="Maximum optimisation rounds (default: GEAK_MAX_ROUNDS env or 5)",
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
        model_factory=_model_factory,
        output_dir=pp_dir,
        max_rounds=args.max_rounds,
        console=console,
    )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
