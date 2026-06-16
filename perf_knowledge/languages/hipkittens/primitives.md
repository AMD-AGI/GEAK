---
title: HipKittens — tile primitives, swizzling & wave scheduling
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3, fp6]
regimes: [both]
status: experimental
updated: 2026-06-08
sources:
  - https://arxiv.org/html/2511.08083v1
  - https://github.com/HazyResearch/HipKittens
---

# HipKittens primitives

## TL;DR
HK gives you **tiles** (register or shared), **bulk ops** (`mma`/`exp`/`add`/load/store), **swizzles**
that are bank-conflict-free per layout (solved, not hand-derived), and two **wave-scheduling patterns**
(8-wave ping-pong, 4-wave interleave) that replace NVIDIA-style producer/consumer specialization. Plus
**pinned register tiles** to bypass HIPCC's AGPR restriction, and an **XCD grid swizzle** for chiplet L2
reuse.

## Tile types
| dim | values |
|---|---|
| location | register tile · shared (LDS) tile |
| dtype | FP32, BF16, FP16, FP8, FP6 |
| rows × cols | multiples of the matrix-core shape |
| layout | row major · col major |

- **Register tiles default to the smallest MFMA shape `16×16×32`** for maximal scheduling control; can be
  parameterized by MFMA shape for edge cases.
- **Pinned register tiles** expose the *same interface* as compiler-managed tiles but let the developer
  own register placement — bypassing HIPCC so **AGPRs can be fed directly to matrix instructions** (HIPCC
  otherwise inserts redundant `v_accvgpr_read` AGPR→VGPR moves before every MFMA).

## Bulk operators
- **Memory:** `load` / `store` across the hierarchy (HBM ↔ LDS ↔ registers).
- **Compute:** PyTorch/NumPy-style `mma`, `exp`, `add`, etc. — thin wrappers over raw CDNA asm/HIP, no
  abstraction overhead.
- **Async global load:** AMD supports async HBM→LDS loads that bypass registers (TMA-like, "buffer load to
  lds"); HK swizzles the **HBM address**, not the shared-memory address.

## Swizzling (the hard part on AMD)
- AMD matrix layouts are **not compositional** (NVIDIA derives all layouts from a single 16×16 core
  matrix) → an explosion of distinct tile layouts. LDS access **phases are non-sequential and differ per
  instruction**: `ds_read_b128` runs 4 phases over 64 banks; `ds_read_b96` runs 8 phases over 32 banks.
  Phases are undocumented, so HK uses a **solver** to find conflict-free swizzles (paper Table 5).
- A *single* swizzle can't serve all layouts; HK ships bank-conflict-free swizzles for **commonly
  co-occurring** layouts. Example (Fig 4): a `16×32` BF16 swizzle "swaps the first 8 columns with the last
  8 starting from the 8th row," killing 2-way bank conflicts and enabling conflict-free column-major reads
  via `ds_read_b64_tr_b16`.

## Wave scheduling — two patterns (replaces wave specialization)
Wave specialization (producer/consumer) **underperforms on CDNA3/CDNA4**: registers are statically split
across all waves, so producer waves burn registers without producing output, shrinking output tile size /
arithmetic intensity (only ~80% of peak BF16 GEMM). The two performant AMD patterns:

| pattern | layout | idea | code size | example perf (FP8 GEMM) |
|---|---|---|---|---|
| **8-wave ping-pong** (balanced) | 8 waves/block, 2 per SIMD | two waves per SIMD alternate: one issues a cluster of memory instrs, the other a cluster of compute, then swap via a conditional barrier; long runs of identical instructions over **large** tiles | compact (**48 LoC**) | **3222 TFLOPS** |
| **4-wave interleave** (imbalanced) | 1 wave/SIMD | finely staggered compute+memory within each wave; needs **small** base tiles | large (**183 LoC**) | **3327 TFLOPS** |

(MHA backward, paper Table 3: 8-wave = 331 LoC / 894 TFLOPS; 4-wave = 989 LoC / 1091 TFLOPS — interleave
buys ~22% on backward at ~3× the code.) **8-wave ping-pong is the default** — it already reaches SOTA for
GEMM and attention-forward on MI355X.

## XCD-aware grid swizzle (chiplet scheduling)
MI355X = 256 CUs across **8 XCDs** (32 CUs each); each XCD has a private 4 MB L2, all share an LLC before
HBM (miss ~300 ns L2, ~500 ns LLC). Blocks are assigned round-robin to XCDs. HK's Algorithm 1:
1. **XCD grouping** — chunks of `C` consecutive block IDs land on the same XCD.
2. **Hierarchical windowed traversal** — process in vertical windows of height `W` to fold block space into
   rectangles for L2 reuse.
   Tuning: L2 bw ≈ 3× LLC; L2 tiles of `8×4` or `4×8` best on MI355X. Worst case = output width in tiles
   coprime with XCD count (e.g. 57 tiles / 8 XCDs). Gives up to ~15–19% over naïve row-major (see
   [perf_findings.md](perf_findings.md) Table 4).

## What AMD lacks vs NVIDIA (and HK's substitute)
| NVIDIA feature | AMD status | HK substitute |
|---|---|---|
| TMA (dedicated copy HW) | none (async buffer-load-to-lds only) | HBM-address swizzle + async load |
| `wgmma`/`tcgen05` (async matmul) | none — only MFMA | tile `mma` over MFMA, deep pipeline w/ small shapes |
| `mbarrier` HW sync | none | shared-memory atomics (negligible overhead) |
| register reallocation (producer→consumer) | none (static split) | 8-wave ping-pong / 4-wave interleave |
| AGPR as MFMA input via compiler | HIPCC refuses | **pinned register tiles** |
| larger SRAM (B200 +40%) | smaller LDS | small MFMA shapes + 2× larger register file |

Terminology map: warp→**wave** (32→64), SM→CU, SMEM→**LDS**, tensor core→**matrix core**,
WGMMA/WMMA/TCGEN05→**MFMA**, TMA→buffer-load-to-lds, CUDA/NVCC→HIP/HIPCC.

## Sources
- HK paper §3 (tiles/ops), §4 (wave scheduling, register pinning, swizzling, XCD), Tables 3–5:
  https://arxiv.org/html/2511.08083v1
- Code: https://github.com/HazyResearch/HipKittens
- AGPR/`v_accvgpr_read` & LDS bank/phase facts corroborated by perf_knowledge glossary + Matrix Core CDNA blog.
- overview: [overview.md](overview.md) · numbers: [perf_findings.md](perf_findings.md)
