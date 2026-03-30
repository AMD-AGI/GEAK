"""Heterogeneous execution: LLM-generated diverse tasks dispatched across GPUs.

In heterogeneous mode the orchestrator asks an LLM to generate multiple
distinct optimization tasks (different strategies, different kernel regions)
and dispatches them across available GPU slots via the pool scheduler in
``parallel_agent.py``.

Key modules:
- ``orchestrator``        -- LLM-driven multi-round optimization loop.
- ``tools``               -- Orchestrator tool implementations (generate, dispatch, collect, finalize).
- ``prompts``             -- System and instance prompt templates.
- ``schemas``             -- Tool JSON schemas for the LLM.
- ``task_generator``      -- LLM-driven task generation from discovery artifacts.
- ``workload_guidance``   -- Backend-specific strategy recommendation builders.
- ``result_scanning``     -- Prior-round result and task scanning utilities.
"""
