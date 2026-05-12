"""Single GPU-pool execution path for every pipeline mode.

Every mode (fixed, planned, mixed) funnels through:

    PipelineContext -> (materialise AgentTask list) -> execute(ctx, tasks) -> run_pool(...)

The underlying ``run_pool`` scheduler in ``run/utils/parallel_helpers.py``
handles worktree + env + agent.run + log file + trajectory save for M
tasks on N GPU slots with overflow queueing, priority ordering,
WorkingMemory init, patch auto-extract, and progress reporting.  This
module is the thin front door that pipeline code calls into.

Two producers build task lists today:

  - ``build_fixed_tasks(N, agent_class, body, ...)`` — N identical
    tasks sharing the same body, used by the ``fixed`` mode dispatcher.
  - The planner (``planned`` mode's task generator) builds its own
    AgentTask list from LLM-generated per-task prompts.

Both producers yield ``list[AgentTask]`` which ``run_pool`` treats
identically; the only difference is the task body.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from minisweagent.agents.agent_spec import AgentTask
from minisweagent.run.utils.parallel_helpers import (
    redirect_output_to_file,
    run_pool as _run_pool_impl,
)
from minisweagent.run.unified import PipelineContext

logger = logging.getLogger(__name__)


def build_fixed_tasks(
    num_parallel: int,
    agent_class: type,
    task_body: str,
    *,
    base_label: str = "parallel",
    priority: int = 10,
    kernel_language: str = "python",
    num_gpus_per_task: int = 1,
) -> list[AgentTask]:
    """Materialize ``num_parallel`` identical ``AgentTask`` objects.

    ``run_pipeline(mode="fixed")`` asks for a list of identical tasks and
    hands them to the pool scheduler.  The scheduler then treats these
    tasks exactly like planner-generated (``planned``) tasks — same
    worktrees, same logs, same priority order, same GPU acquire/release.

    Labels follow the existing convention (``parallel_0``, ``parallel_1``,
    ...) so patch directories written by the pool remain byte-compatible
    with the legacy identical-copies runner.
    """
    if num_parallel < 1:
        raise ValueError(f"num_parallel must be >= 1, got {num_parallel}")

    tasks: list[AgentTask] = []
    for i in range(num_parallel):
        tasks.append(
            AgentTask(
                agent_class=agent_class,
                task=task_body,
                label=f"{base_label}_{i}",
                priority=priority,
                kernel_language=kernel_language,
                num_gpus=num_gpus_per_task,
            )
        )
    logger.debug(
        "build_fixed_tasks: produced %d tasks (label prefix=%s, agent=%s)",
        num_parallel,
        base_label,
        agent_class.__name__,
    )
    return tasks


# Back-compat alias; remove in next release.
build_homogeneous_tasks = build_fixed_tasks


def execute(
    ctx: PipelineContext,
    tasks: list[AgentTask],
    *,
    agent_config: dict[str, Any] | None = None,
    repo_path: Path | None = None,
    base_patch_dir: Path | None = None,
    is_git_repo: bool | None = None,
    env_factory: Callable[[], Any] | None = None,
    output: Path | None = None,
    save_traj_fn: Callable | None = None,
    console: Any = None,
    redirect_output_fn: Callable = redirect_output_to_file,
) -> list[tuple[int, Any, Any, Any]]:
    """Run ``tasks`` through the shared pool scheduler.

    All required context comes from ``ctx``; kwargs are narrow overrides
    for pieces that today are not yet on ``PipelineContext`` (e.g. the
    environment factory).  Later commits will move these onto the context
    object so the signature becomes ``execute(ctx, tasks)``.
    """
    if not tasks:
        logger.warning("execute: no tasks provided, nothing to run")
        return []

    resolved_repo = repo_path or ctx.repo
    if resolved_repo is None:
        raise ValueError("execute requires a repo path (pass repo_path= or set ctx.repo)")
    resolved_repo = Path(resolved_repo).resolve()

    if base_patch_dir is None:
        if ctx.output_dir is None:
            raise ValueError("execute requires base_patch_dir or ctx.output_dir")
        base_patch_dir = Path(ctx.output_dir).resolve()

    if is_git_repo is None:
        is_git_repo = (resolved_repo / ".git").exists()

    if env_factory is None:
        raise ValueError("execute requires env_factory; PipelineContext does not yet carry one")

    if ctx.model_factory is not None:
        model_factory: Callable[[], Any] = ctx.model_factory
    elif ctx.model is not None:
        _model = ctx.model
        model_factory = lambda: _model  # noqa: E731  — simple capture is the whole point
    else:
        raise ValueError("execute requires a model factory (ctx.model_factory or ctx.model)")

    resolved_cfg = dict(agent_config or {})

    base_task_content = tasks[0].task or ctx.user_prompt

    logger.info(
        "pool_runner.execute: %d tasks, %d GPUs, repo=%s, patch_dir=%s",
        len(tasks),
        len(ctx.gpu_ids),
        resolved_repo,
        base_patch_dir,
    )

    return _run_pool_impl(
        tasks=tasks,
        gpu_ids=list(ctx.gpu_ids),
        repo_path=resolved_repo,
        is_git_repo=is_git_repo,
        base_task_content=base_task_content,
        agent_config=resolved_cfg,
        model_factory=model_factory,
        env_factory=env_factory,
        base_patch_dir=base_patch_dir,
        output=output,
        redirect_output_fn=redirect_output_fn,
        save_traj_fn=save_traj_fn,
        console=console,
    )


__all__ = [
    "build_fixed_tasks",
    "build_homogeneous_tasks",
    "execute",
]
