---
type: Kernel Case Study
title: fused_moe_kernel (vLLM fp8 w8a8 block-scale MoE GEMM)
description: Skinny-M fp8 block-scale MoE GEMM on MI300X sped up 1.36x geomean by hoisting the always-true K-mask out of the load loop plus L2/XCD pid swizzling.
tags: [domain-moe, bottleneck-compute, lever-kernel-body, gfx942]
speedup: 1.36x geomean
correctness: PASS (all 3 cases; guards preserve general/large-M paths)
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
- Kernel: Triton `fused_moe_kernel`, fp8 (e4m3fnuz) w8a8 block-scale MoE GEMM.
- HW: AMD Instinct MI300X (gfx942 / CDNA3).
- Fixed launch config (harness-locked, only body tunable): BLOCK_SIZE_M=16,
  BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=8, num_stages=2,
  group_n=group_k=128, K=7168, N=512, E=256, topk=8.
- Skinny-M (M-tile=16): each program loads a [128,128] B tile reused only 16x →
  low arithmetic intensity. Profiling shows stall/latency-bound (Address-Stall
  ~52%, IPC 0.79, HBM ~18% of peak, MFMA ~11%), not pure mem/compute.

| case | baseline (ms) |
|------|---------------|
| c2  (B=2,  M=2048)  | 1.2549  |
| c32 (B=32, M=32768) | 10.4803 |
| c64 (B=64, M=65536) | 20.1986 |

# What changed (the win)
Four kernel-internal, guarded optimizations stacked (v4):
1. **EVEN_K mask hoist (biggest win)** — K=7168 is an exact multiple of
   BLOCK_SIZE_K=128, so the per-iter `offs_k < K - k*BLOCK_SIZE_K` mask is always
   true. `EVEN_K = (K % BLOCK_SIZE_K)==0` constexpr → unmasked B loads, only the
   token-row mask on A. Removes 56 predicated compares/loads; masked fallback when
   K not divisible.
2. **Scalar b_scale + folded scaling** — group_n=128=BLOCK_SIZE_N, so b_scale is a
   128-wide vector of identical values per N-block. Load as scalar, fold into
   a_scale (`(a_scale*b_scale)[:,None]`), dropping a wide load + one broadcast
   multiply per K-iter (×56). Guarded by constexpr `group_n%BLOCK_SIZE_N==0`.
3. **Adaptive L2-swizzle GROUP_M** — set GROUP_M=4 only when num_pid_m>=4096 so
   same-expert m-blocks reuse B n-tiles in L2; stays GROUP_M=1 for the small c2
   grid (avoids concentrating work on fewer CUs).
4. **XCD-aware pid remap** — HW maps workgroup→XCD by pid%8, scattering same-expert
   (shared-B) tiles across all 8 XCDs. Remap `pid=(pid%8)*(num_pids//8)+pid//8` so
   each of the 8 MI300X XCDs owns a contiguous logical range → contiguous experts
   stay warm in per-XCD L2. Guarded by `num_pids%8==0`.

Deployed at `source/triton_fused_moe_kernel.py`.

# Result
| case | baseline (ms) | final v4 (ms) | speedup |
|------|---------------|---------------|---------|
| c2  (B=2,  M=2048)  | 1.2549  | ~0.738 | ~1.70x |
| c32 (B=32, M=32768) | 10.4803 | ~8.55  | ~1.23x |
| c64 (B=64, M=65536) | 20.1986 | ~16.97 | ~1.19x |

- Geomean ≈ **1.36x**, correctness PASS on all cases.
- Largest gain on the small c2 case (load-path overhead dominates there).
- Note: measurement env had intermittent contention spikes; v3→v4 decision made on
  medians of 4 runs each. No SNR/bit-exact claim stated beyond PASS in the report.

# What was tried and reverted
- **num_stages=3 / num_stages=1** via `tl.range` — regressed or no gain; default
  depth-2 already overlaps loads, extra stages cost occupancy/registers.
- **Larger GROUP_M {8,16}** with XCD remap active — still regressed; GROUP_M=4 optimal.
- **int32 addressing** — within noise on benchmark AND removes the int64 large-token
  overflow safety the original added → rejected to preserve real-training correctness.
- **tl.max_contiguous/multiple_of hints on offs_k** — no gain; Triton already infers
  contiguity for tl.arange.

# Patterns
- [Hoist K-loop-invariant math](/patterns/hoist-kloop-invariant-math.md)
- [L2-locality pid remap](/patterns/l2-locality-pid-remap.md)

# Citations
1. KernelForge/results/fused_moe_kernel/tasks/cli/fc9085b3-05f4-43e7-9d0d-eb508c744016/workspace/optimization_report.md
