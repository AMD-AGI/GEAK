"""Fixed-mode helpers and direct ParallelAgent entry (tests / rare use).

Production ``geak -t`` fixed mode goes through ``cli`` → ``run_pipeline``
→ ``run_orchestrator(..., mode="fixed")`` — not through this package.

``run_fixed_mode`` / ``run_homogeneous_agent`` remain for unit tests and
for callers that want the thin ParallelAgent path without the LLM
orchestrator shell.  Shared utilities such as ``parse_gpu_ids`` live in
``run.utils.gpu_ids``; this module re-exports them for backward-compatible
imports.

Historical note: the ``homogeneous`` directory name predates the unified
``OptimizationAgent``; prefer ``run_pipeline(..., mode=\"fixed\")`` for
new code.
"""

from minisweagent.run.utils.gpu_ids import parse_gpu_ids

__all__ = ["parse_gpu_ids"]
