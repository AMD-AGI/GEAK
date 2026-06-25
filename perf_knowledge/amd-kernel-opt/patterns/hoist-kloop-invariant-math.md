---
type: Optimization Pattern
title: Hoist K-loop-invariant address/divide/mask math
description: Move per-iteration pointer arithmetic, integer divides, and mask computation that don't change across the K-loop out of it.
tags: [domain-moe, domain-gemm, bottleneck-compute, lever-kernel-body, gfx942]
bottleneck: compute / VALU
lever_class: kernel-body
median_speedup: 1.05x-1.36x
timestamp: 2026-06-22T00:00:00Z
---

# When to use
A GEMM/MoE inner K-loop recomputes values that are loop-invariant: integer divides for
scale/zp pointer indexing (block-scale / int4), boundary masks, or strided pointer math.
VALU/scalar work that the compiler did not hoist.

# Mechanism
- Hoist the per-iteration integer divide for scale/zp pointers out of the K-loop when
  `BLOCK_SIZE_K % group_size == 0` (the group index is then loop-invariant).
- **EVEN_K mask hoist / mask-collapse**: when `block_k` divides K, drop the per-iter
  bounds mask entirely.
- Fold loop-invariant pointer arithmetic and contiguity hints.

# Evidence
- [fused_moe (fp8 w8a8)](/cases/fused-moe-fp8-blockscale.md) — EVEN_K mask hoist is the biggest component of **1.36×**
- [fused_moe_gptq_awq](/cases/fused-moe-gptq-awq.md) — hoist int divide for scale/zp pointers → **1.10–1.21×** (biggest on has_zp path)

# Caveats
- The guard (`BLOCK_SIZE_K % group_size == 0`, `K % block_k == 0`) is a correctness
  precondition — keep the masked path for the tail.
- On a memory-latency-bound kernel the VALU saving may be hidden; measure.

# Citations
1. KernelForge/results/fused_moe_kernel/tasks/cli/*/workspace/optimization_report.md
2. spare_kernels/k01_fused_moe/OPT_NOTES.md
3. spare_kernels/triton_fused_moe_gptq_awq_kimi/reference_solution/OPT_NOTES.md
