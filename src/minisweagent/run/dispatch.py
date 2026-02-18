"""Dispatch helpers: run task files via ParallelAgent pool mode.

This module provides ``run_task_batch()`` which converts a list of task
file paths into ``AgentTask`` objects and feeds them into the existing
``ParallelAgent.run_parallel(tasks=...)`` pool mode.  The orchestrator
calls this; so does the ``run-tasks`` CLI indirectly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _task_file_to_agent_task(task_file: Path):
    """Read a task markdown file and convert it to an AgentTask."""
    from minisweagent.agents.agent_spec import AgentTask
    from minisweagent.run.task_file import read_task_file

    meta, body = read_task_file(task_file)

    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    agent_class = StrategyInteractiveAgent

    cfg: dict = {
        "save_patch": True,
        "step_limit": 0,
        "cost_limit": 0.0,
        "mode": "yolo",
    }

    if meta.get("test_command"):
        cfg["test_command"] = meta["test_command"]

    # Prepend pipeline context so the sub-agent has all necessary information.
    # IMPORTANT: Paths from metadata use the ORIGINAL repo root.  The parallel
    # agent's _replace_paths() rewrites them to the worktree path before the
    # agent sees the task text.  We must include these paths verbatim here.
    context_lines: list[str] = [
        "## Pipeline Context (auto-injected from task metadata)",
        "",
    ]

    # Critical: include kernel_path and repo_root so the agent knows where to
    # edit and so _replace_paths can rewrite them to the worktree.
    if meta.get("kernel_path"):
        context_lines.append(f"KERNEL FILE TO EDIT: {meta['kernel_path']}")
    if meta.get("repo_root"):
        context_lines.append(f"REPO ROOT: {meta['repo_root']}")
    if meta.get("test_command"):
        context_lines.append(f"TEST COMMAND: {meta['test_command']}")
    context_lines.append("")

    context_lines.append(
        "IMPORTANT: Only edit files within your REPO ROOT directory. "
        "Do NOT search or modify files outside of it. "
        "The KERNEL FILE TO EDIT path above is the exact file you should optimize."
    )
    context_lines.append("")

    commandment_path = meta.get("commandment")
    if commandment_path and Path(commandment_path).exists():
        cmd_text = Path(commandment_path).read_text().strip()
        context_lines.append("## COMMANDMENT (evaluation contract -- you MUST follow these rules)")
        context_lines.append(cmd_text)
        context_lines.append("")

    baseline_path = meta.get("baseline_metrics")
    if baseline_path and Path(baseline_path).exists():
        import json as _json

        bm = _json.loads(Path(baseline_path).read_text())
        dur = bm.get("duration_us", "unknown")
        bn = bm.get("bottleneck", "unknown")
        context_lines.append("## Baseline Performance (your optimization must improve on these)")
        context_lines.append(f"Total duration: {dur} us")
        context_lines.append(f"Bottleneck: {bn}")
        top = bm.get("top_kernels", [])
        if top:
            context_lines.append("Top kernels by duration:")
            for k in top[:5]:
                context_lines.append(
                    f"  - {k.get('name', '?')}: {k.get('duration_us', '?')} us ({k.get('pct_of_total', '?')}%)"
                )
        context_lines.append("")

    prof_path = meta.get("profiling")
    if prof_path and Path(prof_path).exists():
        context_lines.append(f"PROFILING DATA: {prof_path}")
        context_lines.append("(Read this file for detailed per-kernel profiling metrics)")
        context_lines.append("")

    body = "\n".join(context_lines) + "\n" + body

    return AgentTask(
        agent_class=agent_class,
        task=body,
        label=meta.get("label", task_file.stem),
        priority=int(meta.get("priority", 10)),
        kernel_language=meta.get("kernel_language", "python"),
        config=cfg,
    )


def run_task_batch(
    task_files: list[Path],
    gpu_ids: list[int],
    output_dir: Path,
    model_factory,
    *,
    console=None,
) -> dict[str, Any]:
    """Run a batch of task files via ParallelAgent pool mode.

    Parameters
    ----------
    task_files:
        List of task markdown file paths.
    gpu_ids:
        GPU device IDs to use.
    output_dir:
        Base output directory for results.
    model_factory:
        Callable returning a new model instance.
    console:
        Optional Rich console.

    Returns
    -------
    dict with 'completed', 'failed', and 'results' keys.
    """
    from minisweagent.agents.parallel_agent import ParallelAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.run.task_file import read_task_file

    if not task_files:
        return {"completed": 0, "failed": 0, "results": []}

    tasks = [_task_file_to_agent_task(f) for f in task_files]

    # Determine repo_path from first task's metadata
    meta_0, _ = read_task_file(task_files[0])
    repo_root = meta_0.get("repo_root")
    repo_path = Path(repo_root) if repo_root else Path.cwd()

    is_git = False
    if repo_path.is_dir():
        is_git = (repo_path / ".git").exists() or (repo_path / ".git").is_file()

    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    agent_config: dict[str, Any] = {
        "save_patch": True,
        "mode": "yolo",
    }

    def env_factory():
        return LocalEnvironment(**{"cwd": str(repo_path.resolve()), "timeout": 3600})

    try:
        raw_results = ParallelAgent.run_parallel(
            num_parallel=len(gpu_ids),
            repo_path=repo_path,
            is_git_repo=is_git,
            task_content="",
            agent_class=tasks[0].agent_class if tasks else type(None),
            agent_config=agent_config,
            model_factory=model_factory,
            env_factory=env_factory,
            base_patch_dir=results_dir,
            output=None,
            gpu_ids=gpu_ids,
            console=console,
            tasks=tasks,
        )
    except Exception as exc:
        logger.error("Task batch execution failed: %s", exc, exc_info=True)
        return {
            "completed": 0,
            "failed": len(tasks),
            "error": str(exc),
            "results": [],
        }

    completed = 0
    failed = 0
    summaries = []

    for entry in raw_results:
        agent_idx, _agent, exit_status, result = entry
        label = tasks[agent_idx].label if agent_idx < len(tasks) else f"task_{agent_idx}"
        success = exit_status not in ("error", "Error", None)
        if success:
            completed += 1
        else:
            failed += 1

        # Count patches written to the task's result directory
        task_result_dir = results_dir / label
        patch_count = len(list(task_result_dir.glob("*.patch"))) if task_result_dir.is_dir() else 0

        summaries.append(
            {
                "index": agent_idx,
                "label": label,
                "exit": str(exit_status),
                "patches": patch_count,
            }
        )

    return {
        "completed": completed,
        "failed": failed,
        "results": summaries,
        "results_dir": str(results_dir),
    }


def run_from_task(
    task_file: Path,
    gpu_id: int = 0,
    output_dir: Path | None = None,
    model_factory=None,
    *,
    console=None,
) -> dict[str, Any]:
    """Run a single task file. Python-callable wrapper around geak --from-task.

    Shares the same underlying code as the CLI ``--from-task`` path but
    returns a results dict instead of printing to console.
    """
    # Default: tasks/round_N/00_label.md -> results/round_N/
    if output_dir:
        out = output_dir
    else:
        round_name = task_file.parent.name
        out = task_file.parent.parent.parent / "results" / round_name
    return run_task_batch(
        task_files=[task_file],
        gpu_ids=[gpu_id],
        output_dir=out,
        model_factory=model_factory,
        console=console,
    )
