---
type: Kernel Case Study
title: fused_moe_kernel_gptq_awq (int4 W4A16 AWQ/GPTQ MoE GEMM)
description: An int4 W4A16 fused-MoE GEMM whose biggest win came from launch-config tuning (num_warps=1/nonkdim=16/kpack=2) for 1.63x; kernel-body hoisting of per-iter int divides added ~1.05-1.21x in a separate pass.
tags: [domain-moe, bottleneck-compute, lever-config, gfx942]
speedup: 1.63x geomean (campaign20 launch-config); ~1.05-1.21x (kernel-body passes)
correctness: PASS (gate rtol/atol=2e-2; dequant order preserved, fp32 accum)
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
- Backend: Triton (`@triton.jit`) int4 W4A16 AWQ/GPTQ fused-MoE GEMM on MI300X (gfx942, CDNA3), Triton 3.6, torch 2.9.
- Workload (KernelForge harness): K=7168, N=512, E=384, top_k=8, group_size=32, BLOCK_M/N/K=64/64/32, GROUP_SIZE_M=1, SPLIT_K=1, has_zp=False, compute_type=bf16.
- Baseline latency (KernelForge): c2(M=2048) 4.00 ms / c32(M=32768) 47.14 ms / c64(M=65536) 96.97 ms.
- k01 harness uses a different pinned config (BLOCK_M=16, GROUP_SIZE_M=8, num_warps=4, num_stages=2); per-case baselines below.

# What changed (the win)
Two independent efforts on the same kernel:
- **campaign20 (headline 1.63x) — launch-config lever.** HIP launch-proxy compile opts that are NOT among the harness-pinned meta args: `num_warps=1`, `matrix_instr_nonkdim=16`, `kpack=2`. Backend unchanged (Triton). Accepted, re-validated by Director vs the true baseline.
- **KernelForge (~1.05-1.10x) — launch-param + body.** The v0→v1 gain was `num_stages=2` (not num_warps; warps=4 stays best, 8/16 much worse). Plus a `b_scale` broadcast: load `[1,BLOCK_N]` scale row instead of redundant `[BLOCK_K,BLOCK_N]` (legal since `group_size % BLOCK_K == 0`). Pinned single `Config({}, num_warps=4, num_stages=2)` to kill autotune variance.
- **k01 (~1.05-1.21x) — kernel-body lever (main win).** Hoisted the per-iteration integer divide `(offs_k + BLOCK_K*k)//group_size` out of the K-loop (112 iters): since `BLOCK_K % group_size == 0`, the scale/zp group base advances by a compile-time constant, so it is computed once at k=0 and bumped by a constant (like `a_ptrs`/`b_ptrs`). Also hoisted loop-invariant scale/zp pointer arithmetic, added `tl.multiple_of`/`tl.max_contiguous` on `offs_k`, and collapsed the always-true A-mask. Guarded by constexpr `K_DIV_GROUP` with fallback for generality.

# Result
| effort | case | speedup |
|--------|------|---------|
| campaign20 (launch-config) | c2 / c32 / c64 | 1.665 / 1.613 / 1.612 (geomean **1.63x**) |
| KernelForge (num_stages=2 + bscale) | c2 / c32 / c64 | 1.00 / 1.055 / 1.096 |
| k01 (divide hoist) | gemm1_M64 / gemm2_M64 / gemm1_M2048 / gemm2_M2048 | 1.099 / 1.067 / 1.070 / 1.058 |
| k01 has_zp path | gemm1_gateup_M64_zp | **1.214x** (two divides/iter removed) |

- Correctness: PASS on all cases; dequant arithmetic order `((b.to(fp32)-zp)*scale).to(compute_type)` with fp32 accumulation preserved → within the 2e-2 golden gate. Not claimed bit-exact (bf16 compute), but numerically matched.
- Bottleneck note: KernelForge PMC showed VALU:MFMA ≈ 22:1, but revised analysis concluded the kernel is **memory-latency bound** (~97% stalled; A is a scattered token gather), so VALU micro-opts have little headroom. The divide hoist still wins because integer division is the most expensive scalar op and helps the latency-bound small-M cases.

# What was tried and reverted
- **Dequant in bf16 instead of fp32** — reverted in both KernelForge (Triton auto-promotes to fp32, strictly worse) and k01 (risks the 2e-2 gate for no measurable gain; dequant not the bottleneck).
- **Fuse dequant to FMA `b*scale+bias` (bias=-zp*scale)** (KernelForge attempt 5) — no change; compiler already fuses the original form. Reverted.
- **Loop-invariant hoist + drop K-mask in KernelForge** (attempt 6) — neutral; LLVM already does this LICM under its config. (k01 got a real win from the same idea because its config keeps the divide live and its harness measures it differently.)
- **Override GROUP_M grid grouping (sweep 1/2/4/8)** — no value beat GME=1 on dominant c32/c64; reverted.
- **num_warps 8/16** — far worse (c64 114/160 ms). **num_stages>2** OORs on ROCm.
- **`tl.multiple_of`/`max_contiguous` on `offs_bn`** (k01) — neutral (`% N` wrap defeats contiguity; N is strided axis). Reverted.
- **De-dup `[BK,BN]` scale load via restructure** (k01) — rejected; duplicate rows hit identical cached addresses, no real memory win.

# Patterns
- [Launch-config autotune](/patterns/launch-config-autotune.md)
- [Hoist K-loop invariant math](/patterns/hoist-kloop-invariant-math.md)

# Citations
1. KernelForge/results/fused_moe_kernel_gptq_awq/tasks/cli/8f9a28eb-2960-47d3-81f8-bb8f598ce863/workspace/optimization_report.md
2. spare_kernels/k01_fused_moe/OPT_NOTES.md
3. head_kernels/campaign20/FINAL_REPORT.md
