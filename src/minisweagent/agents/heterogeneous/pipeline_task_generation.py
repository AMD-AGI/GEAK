"""Orchestrator task lists for ``fixed`` / ``planned`` / ``mixed`` pipeline modes.

Single module for the branching that used to live inline in
``tools.tool_generate_tasks``.  Behavior is unchanged: same inputs → same
``list[AgentTask]`` as before extraction.
"""

from __future__ import annotations

from typing import Any

from minisweagent.agents.agent_spec import AgentTask


def orchestrator_tasks_for_mode(
    *,
    task_generation: str,
    ctx: dict[str, Any],
    kwargs_for_planned: dict[str, Any],
) -> list[AgentTask]:
    """Return tasks for the current pipeline mode (fixed, planned, or mixed).

    Parameters
    ----------
    task_generation:
        ``ctx["task_generation"]`` — ``fixed``, ``planned``, ``mixed``, or
        default ``planned``.
    ctx:
        Orchestrator context (``agent_class``, ``gpu_ids``,
        ``parallel_worker_count``, ``fixed_parallel_task_body``, ``kernel_meta``,
        …).
    kwargs_for_planned:
        Keyword arguments passed to :func:`task_generator.generate_tasks` for
        planned / mixed-planned portions. Must include ``base_task_context``
        (truncated user text), matching the previous ``tool_generate_tasks``
        behavior.
    """
    from minisweagent.agents.heterogeneous.task_generator import (
        generate_tasks as generate_planned_tasks,
        generate_identical_parallel_tasks,
    )

    tg = str(task_generation or "planned").strip().lower()
    n_workers = max(
        1,
        int(ctx.get("parallel_worker_count") or len(ctx.get("gpu_ids") or [0]) or 1),
    )
    kernel_meta = ctx.get("kernel_meta") or {}
    kernel_language = str(kernel_meta.get("kernel_language") or "python")
    base_instr = str(kwargs_for_planned.get("base_task_context") or "")
    fixed_base = str(ctx.get("fixed_parallel_task_body") or base_instr)
    agent_class = ctx["agent_class"]

    if tg == "fixed":
        return generate_identical_parallel_tasks(
            base_task_context=fixed_base,
            agent_class=agent_class,
            num_tasks=n_workers,
            kernel_language=kernel_language,
        )

    if tg == "mixed":
        n_fixed = n_workers // 2
        n_planned = n_workers - n_fixed
        combined: list[AgentTask] = []

        if n_fixed:
            combined.extend(
                generate_identical_parallel_tasks(
                    base_task_context=fixed_base,
                    agent_class=agent_class,
                    num_tasks=n_fixed,
                    kernel_language=kernel_language,
                    priority=5,
                    label_prefix="fixed-parallel",
                )
            )

        if n_planned:
            planned_tasks = generate_planned_tasks(**kwargs_for_planned)
            planned_tasks = sorted(planned_tasks, key=lambda t: t.priority)[:n_planned]
            while len(planned_tasks) < n_planned:
                planned_tasks.append(
                    AgentTask(
                        agent_class=agent_class,
                        task=base_instr,
                        label=f"planned-pad-{len(planned_tasks)}",
                        priority=20,
                        kernel_language=kernel_language,
                        num_gpus=1,
                    )
                )
            combined.extend(planned_tasks)

        if not combined:
            return generate_identical_parallel_tasks(
                base_task_context=fixed_base,
                agent_class=agent_class,
                num_tasks=1,
                kernel_language=kernel_language,
            )
        return combined

    return generate_planned_tasks(**kwargs_for_planned)
