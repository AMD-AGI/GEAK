"""Preprocess subagents — one-shot LLM tasks run during preprocessing.

Contract: every class here extends ``SubagentBase`` and overrides
``run()`` exactly once (CI gate
``scripts/refactor_ci/check_subagent_base_contract.py`` enforces).

Current members:

  - ``HarnessBuilder``       — adapts user test files into a
                                universal-contract harness (Harness phase)
  - ``KernelAnalysisAgent``  — produces the [A]-[D] analysis rubric
                                markdown (Explore phase)

Not migrated (deliberately per user direction — "keep preprocess
subagents separate like it already exists"; they stay on
``DefaultAgent`` in ``run/preprocess/`` and ``agents/``):

  - ``UnitTestAgent``        — generates test skeleton
  - ``ShapeFixerAgent``      — fixes kernel shape mismatches
"""

from minisweagent.pipeline_workers.preprocess.harness_builder import HarnessBuilder
from minisweagent.pipeline_workers.preprocess.kernel_analysis import KernelAnalysisAgent

__all__ = ["HarnessBuilder", "KernelAnalysisAgent"]
