---
type: Optimization Pattern
title: Per-shape kernel-family dispatch
description: Pick the fastest kernel family (CK vs ASM vs Triton) per shape/M regime at the host, instead of one kernel for all shapes.
tags: [domain-gemm, domain-moe, bottleneck-memory, bottleneck-compute, lever-host-side, no-rebuild, gfx942]
bottleneck: shape-dependent
lever_class: host-side / backend-swap
median_speedup: 1.19x-1.37x
timestamp: 2026-06-22T00:00:00Z
---

# When to use
No single kernel wins across all shapes: small-M (decode) and large-M (prefill) stress
different bottlenecks, and CK / ASM / Triton families cross over. Common in fp8/int
block-scale GEMM and MoE.

# Mechanism
Build a host-side dispatch table keyed on M (or token count) routing each regime to its
best family; bit-exact, no rebuild. Boundaries should be env-overridable for retuning.
Pairs naturally with [block_m to routing sparsity](/patterns/block-m-routing-sparsity.md)
and [CK V3→V1](/patterns/ck-pipeline-v1-occupancy.md).

# Evidence
- [gemm_a8w8_blockscale](/cases/gemm-a8w8-blockscale.md) — route CK for M≤8192, ASM at M=17920 → **1.23×** geomean (1.00–1.54× per shape)
- [moe_gemm_fp8_blockscale](/cases/moe-gemm-fp8-blockscale.md) — 2-stage CK + per-regime block_m + NT-off@1024 → **1.19×**
- [fused_moe (fp8 w8a8)](/cases/fused-moe-fp8-blockscale.md) — adaptive GROUP_M / EVEN_K path per shape

# Caveats
- Tune on small/medium/large; a table fit to one shape regresses the others.
- The winning family may be arch-gated (gfx950) — confirm reachable on the deploy target.

# Citations
1. spare_kernels/arena_tasks/triton2triton/gemm_a8w8_blockscale/RESULTS.md
2. spare_kernels/arena_tasks/hip2hip/moe_gemm_fp8_blockscale/RESULTS.md
