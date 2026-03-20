"""Heterogeneous execution: LLM-generated diverse tasks dispatched across GPUs.

In heterogeneous mode the orchestrator asks an LLM to generate multiple
distinct optimization tasks (different strategies, different kernel regions)
and dispatches them across available GPU slots via the pool scheduler in
``parallel_agent.py``.

Key modules:
- ``task_generator`` -- LLM-driven task generation from discovery artifacts.
- ``task_planner``  -- rule-based task planning from DiscoveryResult.
"""
