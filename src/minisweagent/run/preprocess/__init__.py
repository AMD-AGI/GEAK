"""Preprocessing pipeline: owned stage for kernel resolution and harness setup.

The ``geak-preprocess`` CLI runs the modules in this package sequentially:

1. ``resolve_kernel_url`` -- clone repo, locate kernel file and function.
2. ``codebase_context``   -- generate CODEBASE_CONTEXT.md for the LLM.
3. ``run_harness``        -- execute harness for correctness and benchmarking.
4. ``kernel_profile``     -- profile the kernel via profiler-mcp.
5. ``baseline``           -- build baseline_metrics.json from profiler output.
6. ``commandment``        -- generate COMMANDMENT.md (evaluation contract).
7. ``testcase_cache``     -- cache preprocessor results across runs.
8. ``harness_utils``      -- local harness/runtime bootstrap helpers.

Harness-generation helpers that support this pipeline also live here:

- ``unit_test_agent``     -- generates the fixed harness when discovery is insufficient
- ``shape_fixer_agent``   -- verifies harness shapes against benchmark/test sources

The main entry points are ``run_preprocessor()`` and ``preprocessor.main()``.
"""

from minisweagent.run.preprocess.orchestrator import run_preprocessor_via_orchestrator
from minisweagent.run.preprocess.preprocessor import run_preprocessor

__all__ = ["run_preprocessor", "run_preprocessor_via_orchestrator"]
