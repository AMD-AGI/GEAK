Optimize a controlled TileLang GEMM kernel derived from the official tilelang-benchmark CDNA GEMM benchmark.

This example contains:
- kernel.py: the target kernel to optimize.
- test_tilelang_gemm_harness.py: the fixed GEAK-compatible correctness/profile/benchmark harness.
- official_reference_kernel.py: the measured reference schedule from the official benchmark search space.

The target kernel.py was created by taking official_reference_kernel.py and intentionally making both schedule-level and kernel-body changes worse. It uses worse official schedule-space constants and adds an unnecessary shared-memory staging path between the accumulator fragment and the final global output store. This demonstrates that GEAK can optimize a TileLang kernel derived from an official benchmark without relying on a one-line parameter recovery.

Your task:
1. Compare kernel.py with official_reference_kernel.py.
2. Edit only kernel.py.
3. Optimize the TileLang kernel body and schedule constants only when the change passes correctness and improves the full benchmark.

Constraints:
- Do not edit test_tilelang_gemm_harness.py or official_reference_kernel.py.
- Do not add files.
- Do not change benchmark shapes, reference math, timing logic, imports, environment handling, or output parsing.
- Preserve C = A @ B.T exactly.
- Keep the implementation TileLang based.
- The final optimized kernel may remove unnecessary output staging if correctness and benchmark results justify it.
