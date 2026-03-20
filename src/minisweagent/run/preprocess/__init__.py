"""Preprocessing pipeline: kernel resolution, discovery, profiling, baselining.

The ``geak-preprocess`` CLI runs the modules in this package sequentially:

1. ``resolve_kernel_url`` -- clone repo, locate kernel file and function.
2. ``codebase_context``   -- generate CODEBASE_CONTEXT.md for the LLM.
3. ``run_harness``        -- execute harness for correctness and benchmarking.
4. ``kernel_profile``     -- profile the kernel via profiler-mcp.
5. ``baseline``           -- build baseline_metrics.json from profiler output.
6. ``commandment``        -- generate COMMANDMENT.md (evaluation contract).
7. ``testcase_cache``     -- cache preprocessor results across runs.

The main entry point is ``preprocessor.main()``.
"""
