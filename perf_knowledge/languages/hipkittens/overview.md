---
title: HipKittens — tile-primitive C++ kernels for CDNA (MI3xx)
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3, fp8_e5m2, fp6]
regimes: [both]
status: experimental
updated: 2026-06-08
sources:
  - https://arxiv.org/abs/2511.08083
  - https://arxiv.org/html/2511.08083v1
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk
  - https://github.com/HazyResearch/HipKittens
---

# HipKittens (HK)

## TL;DR
HipKittens is a **minimal C++ embedded tile-primitive library** for writing fast AMD Matrix-Core kernels
without dropping to raw assembly — the AMD entry in Stanford HazyResearch's "Kittens" family
(ThunderKittens/NVIDIA, ThunderMittens/Apple). Its thesis: the **tile abstraction is portable** but the
**backend (swizzling, register scheduling, wave scheduling) must be AMD-specific**. On MI355X (CDNA4) HK
reports SOTA or near-SOTA across GEMM, attention fwd/bwd, and memory-bound kernels — beating AMD's own
hand-tuned **AITER assembly** and hipBLASLt on several shapes — while keeping kernels short (attention fwd
~500 LoC, GEMM hot loop <100 LoC). It is a **research artifact** (arXiv 2511.08083v1, Nov 2025), not a
shipping production dependency, but it is the strongest public evidence that competing DSLs leave perf on
the table on AMD (see [perf_findings.md](perf_findings.md)).

## Concepts
- **Tile = the unit of data and compute.** Register or shared tiles parametrized by `dtype` (FP32, BF16,
  FP16, FP8, FP6), `rows`, `cols` (multiples of the matrix-core shape), `layout` (row/col major). Bulk ops
  (`mma`, `exp`, `add`, …) are PyTorch/NumPy-flavored and wrap raw CDNA asm/HIP with no overhead. See
  [primitives.md](primitives.md).
- **Interface portable, implementation not.** Tile types and ops translate from NVIDIA TK to AMD; what
  changes is memory access (swizzling) and register/wave scheduling — AMD's matrix layouts are not
  compositional (NVIDIA builds everything from a 16×16 core matrix), causing an "explosion of layouts."
- **AMD lacks NVIDIA's wave-specialization enablers.** No TMA, no `wgmma`/`tcgen05` (async matmul
  accepting shared/tensor-mem operands), no `mbarrier` HW sync, **no register reallocation**. HK
  compensates with a **2× larger register file**, small MFMA shapes (`16×16×32`) for deep pipelines, and
  **shared-memory atomics** in place of mbarriers (negligible overhead).
- **Two AMD-native scheduling patterns** replace producer/consumer wave specialization: **8-wave
  ping-pong** and **4-wave interleave** (see [primitives.md](primitives.md) §scheduling). Wave
  specialization *underperforms* on CDNA3/CDNA4 because AMD statically divides registers across all
  waves — producer waves consume registers without contributing output (~80% of peak BF16 GEMM).
- **Register pinning bypasses HIPCC.** HIPCC won't accept AGPRs as matrix-instruction inputs, forcing
  redundant `v_accvgpr_read` moves. HK's **pinned register tiles** (same interface as compiler-managed)
  let the developer pin registers and use AGPRs directly as MFMA inputs — the key to its SOTA backward
  attention.

## The levers (when authoring with HK)
- Pick the **scheduling pattern**: 8-wave ping-pong (compact, large tiles, fewer LoC — the default win)
  vs 4-wave interleave (more code, sometimes a few % faster on backward/imbalanced kernels).
- Use **pinned register tiles** for matmul-heavy + vector-heavy kernels (attention backward) to dodge the
  AGPR `v_accvgpr_read` penalty.
- Choose **MFMA shape**: register tiles default to `16×16×32` for max scheduling control; parameterize for
  edge cases.
- **XCD-aware grid swizzle** (chiplet scheduling) for L2/LLC reuse on the 8-XCD MI355X — see
  [perf_findings.md](perf_findings.md).
- **HBM-address swizzling** for conflict-free async HBM→LDS loads (AMD swizzles the *global* address, not
  the shared-memory address).

## Pitfalls
- **Research artifact, not a maintained library.** Pin a commit; APIs are unstable; no AMD support
  contract. For production prefer aiter/CK/hipBLASLt and use HK as a perf reference / source of ideas.
- **CDNA3/CDNA4 only** (gfx942/gfx950). Benchmarks are MI355X-centric.
- **Pinned register tiles are sharp** — you are bypassing the compiler's register allocator; mistakes are
  silent correctness or occupancy cliffs. Validate parity.
- HK's reported wins are **per-shape**; do not assume a blanket speedup over AITER/hipBLASLt — re-measure.

## Verify
- Repo: `github.com/HazyResearch/HipKittens` (pin a commit). Build with HIPCC for `--offload-arch=gfx950`
  (or gfx942).
- Reproduce the paper's GEMM/attention micro-benchmarks on your MI3xx and compare to your AITER/hipBLASLt
  baseline before adopting any pattern.
- ISA-check that pinned tiles actually feed AGPRs to MFMA (no spurious `v_accvgpr_read`).

## Sources
- HipKittens paper: arXiv 2511.08083v1 "HipKittens: Fast and Furious AMD Kernels" (Hu, Wadsworth,
  Siddens, Winata, Fu, et al.), https://arxiv.org/html/2511.08083v1 ; abstract https://arxiv.org/abs/2511.08083
- HazyResearch blog "AMD GPUs go brrr": https://hazyresearch.stanford.edu/blog/2025-11-09-hk
- Code: https://github.com/HazyResearch/HipKittens
- AGPR/HIPCC limitation & CDNA wave-specialization analysis: paper §3–§4 (see also glossary AGPR entry).
