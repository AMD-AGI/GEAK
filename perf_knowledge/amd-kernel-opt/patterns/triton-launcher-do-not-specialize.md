---
type: Optimization Pattern
title: Triton do_not_specialize + thin cached launcher
description: Remove per-call Triton host overhead (arg specialization, launch_metadata) for launch-bound kernels.
tags: [domain-any, bottleneck-launch, lever-host-side, no-rebuild, gfx942]
bottleneck: launch
lever_class: host-side
median_speedup: 1.02x-2.05x
timestamp: 2026-06-22T00:00:00Z
---

# When to use
A Triton kernel whose wall-clock is dominated by **host dispatch**, not GPU work
(see [bottleneck-first](/methodology/bottleneck-first-classification.md)). Symptom:
GPU body changes (BLOCK_SIZE, vectorization) do nothing to wall-clock.

# Mechanism
- `do_not_specialize` on pointer / scalar args the JIT would otherwise re-specialize on,
  avoiding recompilation and per-call specialization cost.
- A **thin cached launcher** that reuses the compiled kernel handle and skips
  `launch_metadata` computation on the hot path.
- Combine with [HIP-graph replay](/patterns/host-graph-replay.md) for the largest cut.

# Evidence
- [write_req_to_token_pool](/cases/write-req-to-token-pool.md) — host-side levers (do_not_specialize + cached launcher + skip launch_metadata) carry the **2.05×**
- [_topk_forward](/cases/topk-forward.md) — `do_not_specialize` on pointer args (v6) was the only transferable win (~1.02×); graph replay rejected as over-fit

# Caveats
- The win is bounded by the irreducible launch floor (~60% of runtime in the topk case);
  do not expect GPU-body-scale speedups.
- Verify the launcher cache keys correctly so it does not serve a stale kernel.

# Citations
1. KernelForge/results/write_req_to_token_pool_triton/tasks/cli/*/workspace/optimization_report.md
2. KernelForge/results/_topk_forward/tasks/cli/*/workspace/optimization_report.md
