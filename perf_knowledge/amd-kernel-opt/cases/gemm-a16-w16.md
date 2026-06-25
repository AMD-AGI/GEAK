---
type: Kernel Case Study
title: _gemm_a16_w16_kernel (bf16 GEMM)
description: A bf16 Triton GEMM on MI300X sped up ~1.5x by in-kernel super-grouping of pid order (GROUP_M=8 coupled with remap_xcd) for L2 reuse, plus dropping a forced .cg on the B load.
tags: [domain-gemm, bottleneck-memory, lever-kernel-body, gfx942]
speedup: "1.5x geomean (per-shape 1.38 / 1.58 / 1.59x)"
correctness: PASS on all shapes (kernel-internal changes, transfers 1:1 to real training)
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
Original Triton `@triton.jit` bf16 GEMM. Launch config is FROZEN by the harness
(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE_M=1, NUM_KSPLIT=1, num_warps=4,
num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=".cg").
Only the kernel BODY is tunable. Shapes: M ∈ {2048, 32768, 65536}, N=5120, K=2880
(K not divisible by BLOCK_K=128 → EVEN_K=False → masked-load path every K iter).

Stable baseline latency (c64 first run was a JIT warmup artifact; re-measured):

| case | M | baseline ms |
|------|-------|------|
| c2 | 2048 | 0.31 |
| c32 | 32768 | 4.97 |
| c64 | 65536 | 9.99 |

Bottleneck: MEMORY / L2-locality. With forced GROUP_SIZE_M=1 (row-major pid),
the B operand is re-streamed ~num_pid_m times.

# What changed (the win)
- **In-kernel super-grouping (GROUP_M=8):** re-order pid mapping so super-blocks of
  M-tiles keep B-tiles warm in L2. This is the dominant ~1.5x lever.
- **COUPLED with the existing `remap_xcd`:** grouping and XCD remap are
  interdependent — each alone reverts to baseline; only together give the 1.5x.
- **Drop forced `.cg` on the B load** (use default → B also caches in L1): +~4%,
  consistent. (A kept default; A with `.cg` regressed since A is reused and needs L1.)
- **K-peel / tail split** (22 full unmasked iters + 1 masked tail): perf-neutral
  but kept for cleaner pipelining.
- GROUP_M swept {4,6,8,10,12,16,32}; all within ~1% noise — picked balanced 8.

Best artifact: `optimized_versions/v3_group8_Bdefault.py` (in place at source path).

# Result
| case | M | baseline ms | best ms | speedup |
|------|-------|------|------|------|
| c2 | 2048 | 0.31 | 0.225 | 1.38x |
| c32 | 32768 | 4.97 | 3.145 | 1.58x |
| c64 | 65536 | 9.99 | 6.28 | 1.59x |

Overall ≈ 1.5x geomean (campaign Director recorded 1.52x), correctness PASS on all
cases. After grouping, L2 hit ≈86% and the kernel becomes issue/LDS-stall bound.
All changes are kernel-internal (pid remap + cache policy + loop structure), so they
transfer 1:1 to real training with no benchmark-identity dependence. Campaign20
classed this kernel "salvaged" (a `/model`+`/effort` abort lost an earlier run).

# What was tried and reverted
- **Drop input_precision="ieee" (use default):** REGRESSION vs ieee. Reverted.
- **Bypass remap_xcd (keep grouping):** back to BASELINE — confirmed coupling. Reverted.
- **Contiguity hints (max_contiguous/multiple_of on offs_k):** NEUTRAL, compiler
  already vectorized. Reverted.
- **2D super-tile grouping (GROUP_M=8 × GROUP_N=5):** NEUTRAL/slightly worse vs 1D;
  B already resident in 256MB MALL so per-XCD L2 windowing adds nothing. Reverted.
- **`.cg` on the A load:** WORSE (c64 8.03 ms) — A is reused, needs L1. Reverted.
- **K-loop split alone (Attempt 1):** neutral, kept only as clean base, not a win.

# Patterns
- [L2-locality PID remap (super-grouping)](/patterns/l2-locality-pid-remap.md)

# Citations
1. KernelForge/results/_gemm_a16_w16_kernel/tasks/cli/05781910-8b53-4430-b112-ff574690b402/workspace/optimization_report.md
2. head_kernels/campaign20/FINAL_REPORT.md
