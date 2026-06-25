---
type: Kernel Case Study
title: aiter CK ck_moe_stage1 block-scale grouped GEMM (gate/up + SiLU)
description: aiter CK block-scale MoE stage-1 GEMM sped up by host-side occupancy/padding levers (V3->V1 pipeline + block_m=16 for sparse routing), bit-exact, 1.08-1.11x geomean on gfx942.
tags: [domain-moe, bottleneck-occupancy, lever-host-side, gfx942]
speedup: 1.08-1.11x geomean (1.077x integrable host-side patch; 1.114x in-kernel .cuh override)
correctness: PASS, bit-exact (err_ratio 0.0000, cos_diff ~1e-10)
kept: kept-deployed (host-side aiter patch gated to q_type==per_1x128; in-kernel override also viable)
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
Stock aiter CK `ck_moe_stage1` block-scale grouped GEMM (gate/up + SiLU), MI300X gfx942.
The block_m=64 prefill instance `256x64x128x128` defaults to BlockGemmPipelineVersion **v3**
(2-LDS double-buffer = 64 KB LDS -> 1 block/CU). Kernel is occupancy/latency-bound, NOT
HBM-saturated (~3.0 TB/s of ~4.8 achievable at large M).

Measured baseline (scorer regime, token 2048/32768/65536):

| case | token | latency (ms) |
|------|-------|--------------|
| c2   | 2048  | 0.5499 |
| c32  | 32768 | 4.6319 |
| c64  | 65536 | 8.9017 |
| **geomean** | | **2.8303** |

Distiller harness regime (token 256..17920): baseline geomean 0.867 ms.

# What changed (the win)
Two host-side occupancy/padding levers (no byte reduction — the CK V3 tile is constraint-locked):
- **V3 -> V1 pipeline for block_m=64 at large tokens.** Force `BlockGemmPipelineVersion::v1`
  (single 32 KB LDS buffer -> 2 blocks/CU). Lifts achieved HBM ~3.0 -> ~3.9 TB/s; ~1.16x at
  token >= 16384. In the .cuh path this is a single targeted edit to the stage-1 `DeviceOpInstance`
  `PipelineVer` arg. In the integrable path the V1 `256x64x128x128` instance is added to the codegen list.
- **block_m=16 for token <= 256 (sparse routing).** With ~8 tokens/expert a 64-row routing block is
  ~87% padding -> wasted weight reads. bm16 gives ~1.196x at token=256 (beat bm32's 1.172x).
- Regime dispatch: bm16 (<=256) / bm32 (257-768) / bm64-V1 (769-8192) / bm32 (9216-12288) / bm64-V1 (>12288).

Deliverable (integrable): 3-file patch to the aiter tree — `aiter/fused_moe.py` +
`csrc/ck_gemm_moe_2stages_codegen/{gen_instances.py,gemm_moe_ck2stages_common.py}`, all GATED to
`q_type==per_1x128`. Verified by building the patched aiter in an isolated clone.

# Result
| view | regime | baseline geomean | optimized geomean | speedup | correctness |
|------|--------|------------------|-------------------|---------|-------------|
| in-kernel .cuh (V1 only) | token 2048/32768/65536 | 2.8303 ms | 2.5480 ms | **1.111x** (stable 1.10-1.11x over 3 runs) | bit-exact |
| in-kernel .cuh, attempt-1 | token 2048/32768/65536 | 2.8303 ms | 2.5407 ms | 1.114x | bit-exact |
| integrable patch (synth, V1+bm16) | token 256-17920 | 0.867 ms | 0.805 ms | **1.077x** | bit-exact |

Per-shape (integrable synth): token256 1.196x, t2048 1.035x, t4096 1.047x, t8192 1.020x,
t11264 1.011x, t16384 1.156x, t17920 1.168x. token16 ~1.00 (decode, launch-bound).
All variants bit-exact (err_ratio 0.0000, cos_diff ~1e-10). ~1.08x is near the no-new-kernel ceiling.

# What was tried and reverted
- **Nswizzle (N-tile L2-locality remap) -> GPU memory access fault.** Setting `Nswizzle=true` compiles
  but the chunk-of-8 block remap (`p_sorted_expert_ids[blockIdx.x/NBlock]`) does not match this blockscale
  MoE's grid/sorted-id layout -> OOB. REVERTED.
- **Non-temporal / streaming (NT) weight loads -> REGRESS** (0.87 -> 1.01-1.04 ms, large cases worst).
  Weights are re-read across an expert's token-blocks and stay hot in L2/Infinity Cache; NT bypasses that
  reuse. Refutes the profile's "HBM saturated, optimize bytes" claim (corroborated by all 4 harnesses).
- **Tile / MFMA / scheduler sweeps -> dead end.** CK V3 blockscale tile is hard-LOCKED on gfx942
  (KPerBlock!=128, NPerBlock!=128, MPerBlock=128 all assert/blow the 64 KB LDS limit; Interwave unsupported).
- GUFusion already on; BK1/AK1 already 16B-vectorized; HotLoopScheduler lives in a header outside the scored file.
- Recorded headroom: ako's stock-V3 bm32 was better at token=11264 (1.036 vs 1.011) but didn't transfer.

# Patterns
- [CK pipeline-v1 occupancy](/patterns/ck-pipeline-v1-occupancy.md)
- [block_m routing sparsity](/patterns/block-m-routing-sparsity.md)
- [Non-temporal load regression (anti-pattern)](/anti-patterns/non-temporal-load-regression.md)

# Citations
1. KernelForge/results/moe_stage1/tasks/cli/b23d3841-3546-403f-9df5-bf1b21baaf8b/workspace/optimization_report.md
2. spare_kernels/arena_tasks/hip2hip/moe_stage1/RESULTS.md
