"""Planned-mode dispatch: LLM-generated diverse tasks across GPUs.

This package implements "planned" mode from the execution plan: a
planner LLM emits N distinct optimization strategies and dispatches
one per GPU slot through the pool scheduler in ``parallel_agent.py``.
Every worker runs the same ``OptimizationAgent`` class — only the task
body differs.

Key modules:
- ``orchestrator``        -- LLM-driven multi-round optimization loop.
- ``tools``               -- Orchestrator tool implementations (generate, dispatch, collect, finalize).
- ``prompts``             -- System and instance prompt templates.
- ``schemas``             -- Tool JSON schemas for the LLM.
- ``task_generator``      -- LLM-driven task generation from discovery artifacts.
- ``workload_guidance``   -- Backend-specific strategy recommendation builders.
- ``result_scanning``     -- Prior-round result and task scanning utilities.

Historical note: this package was named ``heterogeneous`` when the
codebase had distinct agent classes per dispatch style.  With the
unified ``OptimizationAgent``, the directory name is retained as a
compatibility shim; new code should reference these modules through
``run_orchestrator`` (in ``run/orchestrator.py``) rather than importing
from here directly.
"""
