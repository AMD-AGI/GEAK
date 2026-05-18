"""Write a ``DispatchPlan`` as ``.md`` task files for traceability.

The ``.md`` files are the canonical intermediate format between planning
and execution.  ``run_task_batch`` reads them back and converts each into
an ``AgentTask`` for ``ParallelAgent.run_parallel``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from minisweagent.run.dispatch_plan import DispatchPlan, DispatchPlanItem
from minisweagent.run.task_file import write_task_file

logger = logging.getLogger(__name__)


def write_dispatch_plan_as_task_files(
    plan: DispatchPlan,
    output_dir: Path,
    *,
    commandment: str | None = None,
    baseline_metrics: str | None = None,
    profiling: str | None = None,
    codebase_context: str | None = None,
    benchmark_baseline: str | None = None,
    starting_patch: str | None = None,
    harness_path: str | None = None,
    kernel_path: str | None = None,
    repo_root: str | None = None,
    test_command: str | None = None,
    round_num: int = 1,
) -> list[Path]:
    """Write each ``DispatchPlanItem`` as a ``.md`` task file.

    Returns the list of written file paths, sorted by priority.
    """
    task_dir = output_dir / "tasks" / f"round_{round_num}"
    task_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for item in plan.items:
        filename = f"{item.priority:02d}_{item.label}.md"
        task_path = task_dir / filename

        metadata: dict[str, Any] = {
            "label": item.label,
            "priority": item.priority,
            "agent_type": item.agent_type if hasattr(item, "agent_type") else "strategy_agent",
            "agent_name": item.agent_name,
            "kind": item.kind,
            "kernel_language": item.kernel_language,
            "num_gpus": item.num_gpus,
            "round": round_num,
        }

        if kernel_path:
            metadata["kernel_path"] = kernel_path
        if repo_root:
            metadata["repo_root"] = repo_root
        if commandment:
            metadata["commandment"] = commandment
        if baseline_metrics:
            metadata["baseline_metrics"] = baseline_metrics
        if profiling:
            metadata["profiling"] = profiling
        if codebase_context:
            metadata["codebase_context"] = codebase_context
        if benchmark_baseline:
            metadata["benchmark_baseline"] = benchmark_baseline
        if starting_patch:
            metadata["starting_patch"] = starting_patch
        if harness_path:
            metadata["harness_path"] = harness_path
        if test_command:
            metadata["test_command"] = test_command

        body = f"# {item.label}\n\n{item.task}\n"
        write_task_file(task_path, metadata, body)
        written.append(task_path)

        logger.debug("Wrote task file: %s (kind=%s, priority=%d)", task_path, item.kind, item.priority)

    logger.info(
        "write_dispatch_plan_as_task_files: wrote %d task files to %s",
        len(written),
        task_dir,
    )
    return sorted(written)
