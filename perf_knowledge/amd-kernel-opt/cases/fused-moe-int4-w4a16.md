---
type: Kernel Case Study
title: fused_moe_int4_w4a16 (int4 W4A16 fused-MoE GEMM, vLLM Triton)
description: vLLM int4 W4A16 fused-MoE Triton GEMM made 5.19x faster by loading each packed int4 byte once and unpacking both nibbles in-register, plus scale/zp group-dedup broadcast.
tags: [domain-moe, bottleneck-memory, lever-kernel-body, gfx942]
speedup: "5.19x geomean"
correctness: PASS (all 5 cases)
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
Stock vLLM int4 W4A16 fused-MoE Triton GEMM (vLLM PR #12185, owns the kernel; `(offs_k//2)` double-load + b_shifter scheme unchanged across ~200 commits). MI300X / gfx942.

Baseline latency per shape (ms):

| shape | baseline ms |
|---|---|
| gemm1_M64 | 4.465 |
| gemm2_M64 | 1.944 |
| gemm1_M2048 | 18.835 |
| gemm2_M2048 | 8.294 |
| gemm1_M64_zp | 4.737 |
| total | ≈38.3 |

Roofline: cache-read-BW / VMEM-issue bound. L1→L2 63.6 GB/iter @ ~3.4 TB/s; HBM only 3.6% of peak, MFMA ~1%.

# What changed (the win)
Body-only optimization (host.py launch config is grader-locked). Winner harness: geak; installed to `kernel_jit.py`. Two levers, all four harnesses independently converged on the same recipe.

- **L10 — load each packed int4 byte ONCE (~3.8x alone, dominant).** Stock indexes B with `(offs_k//2)` over BLOCK_K=64 rows, so every weight byte is loaded + unpacked twice. Winner loads BLOCK_K//2=32 byte rows once, unpacks both nibbles in-register (`lo=b&0xF`→even-K, `hi=(b>>4)&0xF`→odd-K), dequants two [32,BN] half-tiles, interleaves via `tl.join → tl.trans(0,2,1) → tl.reshape`, then one `tl.dot`. Cut VMEM loads 6x, L1→L2 traffic 3.1x.
- **L11 — scale/zp group-dedup broadcast (3.8x→5.2x).** group_size=32, BLOCK_K=64 → only NG=2 distinct group rows per K-block. Load [2,BN] scale/zp and `tl.broadcast_to`; cuts scale/zp VMEM loads + zp-unpack.
- **micro (geak's edge over ako, within noise):** `tl.multiple_of` + `tl.max_contiguous` on the contiguous packed-K byte index `offs_kh`; collapse the always-true A-mask when `block_k_diviable`.

# Result
**5.19x geomean (geak), PASS.** All 5 cases land ~4.9–5.5x.

| shape | speedup | winner ms |
|---|---|---|
| gemm1_M64 | 5.511x | 0.810 |
| gemm2_M64 | 4.924x | 0.395 |
| gemm1_M2048 | 5.495x | 3.347 |
| gemm2_M2048 | 4.997x | 1.660 |
| gemm1_M64_zp | 5.076x | 0.934 |
| total | — | ≈7.1 |

Correctness PASS (no bit-exactness/SNR claim in report). Post-opt roofline: now L2-read-BW bound — L1→L2 20.4 GB/iter @ ~6.1 TB/s (near L2 wall), VMEM 6x fewer, HBM 28% of peak. VALU/iter unchanged (dequant was always overlapped, never the wall). **At/near the body-only ceiling**; remaining ~2x headroom (amortize weight over larger BLOCK_M) needs host.py edits = out of scope.

ako tied at 5.18x (strict subset of geak). expert (4.03x) and kda (3.96x) used the same L10 but a weaker L11 dedup — the spread is purely scale/zp dedup quality, most visible on the has_zp case (5.08x vs 3.2x).

# What was tried and reverted
- **bf16/compute_type dequant** instead of fp32 → large regression (gemm1_M2048 ~3.5→5.8 ms); the int4→bf16 dequant VALU path isn't packed efficiently on gfx942. Keep fp32 dequant, cast once.
- **Two-dot K-split** (split A even/odd K, two BK//2=32-deep dots to skip the trans/reshape) → regresses; two 32-deep MFMA dots underutilize vs one 64-deep dot.
- **Loop-invariant scale/zp pointer hoist** → neutral + adds VGPR pressure.
- **Contiguity hints on `offs_bn`** → invalid (has `%N` modulo, non-contiguous stride_bn). Hints only on `offs_kh`.

# Patterns
- [int4 load-once unpack both nibbles](/patterns/int4-load-once-unpack.md)
- [Hoist K-loop-invariant math](/patterns/hoist-kloop-invariant-math.md)

# Citations
1. spare_kernels/arena_tasks/triton2triton/fused_moe_int4_w4a16/RESULTS.md
2. AgentKernelArena/tasks/campaign20/baseline/fused_moe_int4_w4a16/RESULTS.md
