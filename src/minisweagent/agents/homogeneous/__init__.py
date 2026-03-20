"""Homogeneous execution: identical task replicated across GPUs.

In homogeneous mode every GPU slot runs the same optimization task
independently.  The orchestrator writes a single task file and
``run_task_batch`` replicates it N times.  After all agents finish,
``SelectPatchAgent`` picks the best result.

The execution logic lives in ``parallel_agent.py`` (at the agents/ root).
"""
