---
type: Kernel Case Study
title: gemm_a8w8_blockscale (MiniMax-M2.5 fp8 attn-proj GEMM)
description: fp8 a8w8 block-scaled attn-projection GEMM on MI300X sped up via host-side HIP/CUDA-graph replay plus per-shape kernel-family dispatch (CK vs ASM), up to 1.82x, bit-exact.
tags: [domain-gemm, bottleneck-memory, lever-host-side, gfx942]
speedup: 1.82x (campaign20 graph); 1.37x (KernelForge fast-kernel dispatch); 1.23x (CK-vs-ASM dispatch)
correctness: PASS — bit-exact (err_ratio=0, cos_diff ~2e-9), no rebuild
kept: kept-deployed (campaign20 accepted; spare CK swap gated, left to user consent)
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
MiniMax-M2.5 fp8 attn-projection GEMM, N∈{3072,4096}, K=3072, on AMD MI300X (gfx942/CDNA3). Three independent campaigns hit the same kernel from different angles, so baselines differ:

- **campaign20 (pos11)**: Triton baseline, geomean over concurrency {2,32,64}, seqlen=1024. Launch-bound.
- **KernelForge (Triton-only)**: original Triton, default 128x128x128 / warps=4 / stages=2 / GROUP_SIZE_M=1. Per-shape (N=4096,K=3072): c2 M=2048 = 0.2035 ms; c32 M=32768 = 1.7457 ms; c64 M=65536 = 3.5703 ms; **geomean 1.0825 ms**.
- **spare (CK-vs-ASM)**: stock ASM blockscale entry-point (`gemm_a8w8_blockscale` C++ heuristic, default 128x128). 8 shapes M∈{2048..17920}; baseline geomean 0.280 ms. ASM sits at only 12–32% fp8 peak — the `.co` binaries were authored for MI308, never tuned for MI300X's 304 CUs.

# What changed (the win)
The decisive lever differs per campaign; all are host-side / dispatch, compute backend unchanged:

- **campaign20 — 1.82x**: HIP/CUDA-graph capture+replay of the same Triton kernel (collapses per-call host/dispatch floor on this launch-bound GEMM) + per-shape config + dequant fold. Bit-exact, no rebuild.
- **KernelForge — 1.37x**: pure Triton config + a bit-exact streamlined `_gemm_a8w8_blockscale_fast_kernel` (split-K machinery removed) dispatched on runtime M: fast kernel for M ≤ 12288, tuned 256x128x128/warps=8/stages=2/GROUP_SIZE_M=4 library path for larger M. Single-variable progression: 256x128/nw8 tile (1.09x) → kpack=1 + cache_modifier=None + waves_per_eu=1 (1.30x, the largest single jump) → fast-kernel M-dispatch (1.37x).
- **spare — 1.23x**: per-shape **kernel-family dispatch (no rebuild)** — route the already-compiled CK kernel `gemm_a8w8_blockscale_bpreshuffle_ck` for M ≤ 8192 (1.3–1.5x faster than ASM, consumes identical shuffled-weight (16,16) + transposed x_scale inputs), keep ASM (96x128 for o_proj N=3072) only at M=17920 where ASM still wins. CK bpreshuffle is the enabling backend on gfx942. Two harnesses (ako, kda) discovered it independently.

# Result
| campaign | speedup | per-shape range | correctness |
|---|---|---|---|
| campaign20 (graph) | **1.82x** geomean | c2 2.22 / c32 1.65 / c64 1.64 | bit-exact, accepted |
| KernelForge (Triton) | **1.37x** geomean | c2 ~1.33 / c32 ~1.34 / c64 ~1.43 (0.79 ms vs 1.0825) | bit-exact (err=0, cos_diff=0) |
| spare (CK dispatch) | **1.23x** geomean | 1.00x–1.54x; e.g. qkv_M2048 1.54x, o_M2048 1.52x, o_M17920 1.07x | bit-exact (err=0, cos_diff ~2e-9) |

All bit-exact, no JIT rebuild. spare deliverable is a gated patch to `aiter/ops/gemm_op_a8w8.py` (falls through to stock heuristic outside the 8 shapes, so production cannot regress); left uninstalled pending user consent.

# What was tried and reverted
- **Split-K / NUM_KSPLIT > 1**: regresses everywhere. K=3072 = only 24 K-tiles; grid already CU-saturated at splitK=1, so atomic-accum / fp32-reduce overhead dominates (confirmed in both KernelForge Attempt 3 and spare §2 across all 4 harnesses).
- **ns=3 at 256x128**: overflows LDS (inf) — kept stages=2.
- **ASM tile + splitK autotune (spare's roofline hypothesis — WRONG)**: full sweep of 6 ASM tiles × splitK{1,2,3,4,6,8,12} moved geomean ≤2.5%; the C++ heuristic already picks near-optimal (128x128 best/tied on 5/8 shapes). Only ASM micro-win: 96x128 for N=3072 o_proj (~3–6% at large M). geak/expert harnesses followed this hint, missed CK, and stayed flat (~1.00x).
- **Triton blockscale path (as ASM replacement, spare)**: 1.3–1.7x slower, and its preshuffle convention ((N·16,K/16)) is incompatible with `shuffle_weight(layout=(16,16))` — not viable.
- **cktile bpreshuffle (spare)**: 1.5–3x slower, not used.
- **nkdim=32 (KernelForge)**: slower than 16.
- Real 2–3x headroom requires a new MI300X-tuned ASM/CK kernel + rebuild — out of scope.

# Patterns
- [Host-side graph replay](/patterns/host-graph-replay.md)
- [Per-shape kernel dispatch](/patterns/per-shape-kernel-dispatch.md)
- [Backend dispatch swap](/patterns/backend-dispatch-swap.md)

# Citations
1. KernelForge/results/gemm_a8w8_blockscale/tasks/cli/fadbcae5-40f2-42e5-9783-b5ae6c532d32/workspace/optimization_report.md
2. spare_kernels/arena_tasks/triton2triton/gemm_a8w8_blockscale/RESULTS.md
3. head_kernels/campaign20/FINAL_REPORT.md
