---
type: Kernel Case Study
title: moe_gemm_fp8_blockscale (MiniMax-M2.5 fp8 block-scale fused MoE)
description: A fused fp8 block-scale MoE GEMM sped up 1.19x geomean by swapping the 1-stage ASM dispatch for aiter's 2-stage CK path with per-regime block_m and NT-load off at token=1024.
tags: [domain-moe, bottleneck-memory, lever-backend-swap, gfx942]
speedup: 1.19x geomean (per-shape 1.04-1.40x)
correctness: PASS, SNR 32.7 dB (baseline ASM was only 23-27 dB, below the 25 dB bar at token>=256)
kept: kept-deployed (host-only dispatch change, no aiter rebuild)
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
Original production path: aiter's **1-stage ASM** `fmoe_fp8_blockscale_g1u1` (`asm_baseline.py`). aiter's stock heuristic routes `token>32` to this 1-stage ASM for these MiniMax shapes (model_dim=3072). Golden: `moe_harness.torch_moe_blockscale` (fp32). Pass bar: SNR >= 25 dB.

Baseline ASM median latency:

| token | ASM µs |
|-------|--------|
| 64    | 437    |
| 256   | 500    |
| 1024  | 684    |
| 2048  | 1158.84 |
| 4096  | 2106 (worst case) |
| 32768 | 10813.38 |
| 65536 | 16112.61 |

The ASM was the dominant vLLM MoE path (~52% device time) and was actually *below* the 25 dB bar at token>=256 (SNR 23-27 dB).

# What changed (the win)
Pure **host-side dispatch swap** — no tile-internal change, no aiter rebuild:
- **1-stage ASM -> 2-stage CK.** Force `aiter.fused_moe(per_1x128)` to route to the 2-stage CK path (`ck_moe_stage1` + `ck_moe_stage2_fwd`) instead of the 1-stage ASM. Achieved by removing the per_1x128/fp8/g1u1 key from `fused_moe_1stage_dict[gfx]` so `run_1stage=False`. Identical quant + shuffled-weight + block-scale layout (reuses `moe_harness.prepare`).
- **Per-regime block_m** scaling with tokens-per-expert: 16 / 16 / 32-64 / 64 for token 64/256/1024/4096 (and 64 @2048, 32 @32768/65536). bm128 unsupported by moe_sorting.
- **NT-load OFF at token=1024 (the decisive edge).** The stock NT heuristic (`tpe<64` -> on) hurts at mid tokens where each expert's weight is re-read across token-blocks and stays cache-hot; NT-off flipped token1024 from 0.99x (loss) to 1.04x. NT-on is kept for small tokens (64/256, read-once); NT-off for large tokens.

Bonus: CK's fp32 accumulation lifts SNR from the ASM's 23-27 dB to 32.7 dB — both faster AND more accurate.

# Result
Verified speedup vs ASM baseline (same-GPU interleaved, 6 rounds, trimmed):

| token | speedup |
|-------|---------|
| 64    | 1.23x   |
| 256   | 1.14x   |
| 1024  | 1.04x (NT-off rescued this) |
| 2048  | 1.18x   |
| 4096  | 1.40x (carries the win) |
| 32768 | 1.36x   |
| 65536 | 1.04x   |

**Geomean ~= 1.19x** (per-shape 1.04-1.40x). Correctness PASS, **SNR 32.7 dB** (+6-9 dB vs ASM). Not bit-exact (different backend); accuracy is strictly better than baseline. Host-only deliverable (`asm_baseline.py` dispatch change + `.patch`); transfers 1:1 to the vLLM serving path with `run_1stage` forced off — no identity-keyed caches, no benchmark over-fit.

# What was tried and reverted
- **`doweight_stage1=True`** (fold routed-weight multiply into stage1): REJECTED — SNR 1.31 dB at token=32768 (does not match block-scale golden), token=2048 path failed to build, no perf gain. Kept `doweight_stage1=False`.
- **NT-on globally**: worse everywhere (c64 24207 vs 15514 µs; c2 1010 vs 924) — kept aiter's NT-off default for large tokens.
- **bm64 at small tokens (64/256)**: regresses ~0.82x (under-occupied / padding). **bm32 at token4096**: loses to bm64. **bm128**: unsupported by moe_sorting.
- **split-K (ksplit)**: no effect — token*topk > E so get_ksplit returns 0 (consistent with splitK regressing on few-K-tile GEMMs).
- **V1-stage1 pipeline (sibling task-3 lever)**: neutral-to-small here (+1.9% @token4096 via clone); NOT worth the clone dependency, but compounds for free (~1.20x) if the task-3 patch is already installed in aiter. Recorded as optional headroom.

# Patterns
- [Backend dispatch swap](/patterns/backend-dispatch-swap.md)
- [Per-shape kernel dispatch](/patterns/per-shape-kernel-dispatch.md)
- [Non-temporal load regression](/anti-patterns/non-temporal-load-regression.md) (anti-pattern: stock NT heuristic regressed token=1024 until disabled)

# Citations
1. KernelForge/results/moe_gemm_fp8_blockscale/tasks/cli/dd1e3dc4-32bb-4ad9-bd50-c3fa41e0b8dc/workspace/optimization_report.md
2. spare_kernels/arena_tasks/hip2hip/moe_gemm_fp8_blockscale/RESULTS.md
