---
type: Optimization Pattern
title: Host-side HIP/CUDA-graph replay
description: Capture the kernel launch(es) into a graph and replay it, eliminating per-call host dispatch on launch-bound ops.
tags: [domain-any, bottleneck-launch, lever-host-side, no-rebuild, gfx942]
bottleneck: launch/occupancy
lever_class: host-side
median_speedup: 1.2x-2.05x
timestamp: 2026-06-22T00:00:00Z
---

# When to use
The op is **launch-overhead-bound**: GPU util < 30%, or kernel runtime ≈ host dispatch
(~20–25 µs on this stack), so the GPU body is already fully hidden behind launch latency.
Tiny elementwise/index ops and small-shape GEMMs are the usual candidates. Confirm with
[bottleneck-first classification](/methodology/bottleneck-first-classification.md).

# Mechanism
Capture the launch (or launch sequence) once into a HIP graph and `replay` it on each
call. This removes Python/Torch dispatch, kernel arg marshalling, and `launch_metadata`
work from the hot path. No kernel code changes; bit-exact.

# Evidence
- [write_req_to_token_pool](/cases/write-req-to-token-pool.md) — **2.05×** (16.5→~15 µs fast path; pure host win)
- [_topk_forward](/cases/topk-forward.md) — **1.90×** (GPU ~9 µs hidden under ~24 µs dispatch)
- [gemm_a8w8_blockscale](/cases/gemm-a8w8-blockscale.md) — graph is a component of the 1.82× win
- [_fwd_grouped_kernel_stage1](/cases/fwd-grouped-stage1.md) — **1.18×**
- [chunk_scaled_dot_kkt](/cases/chunk-scaled-dot-kkt.md) — graph carries the c2 (small) shape

# Caveats
- A graph-replay gain that exists **only because the benchmark reuses the same tensors
  across its timing loop** is over-fit — re-validate against fresh-tensor / real serving
  before trusting it. See [benchmark over-fit](/anti-patterns/benchmark-overfit.md).
- No help once the kernel is GPU-bound; replay the launch, not the compute.

# Citations
1. KernelForge/results/write_req_to_token_pool_triton/tasks/cli/*/workspace/optimization_report.md
2. KernelForge/results/_topk_forward/tasks/cli/*/workspace/optimization_report.md
