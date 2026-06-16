---
title: TileLang pitfalls & anti-patterns on AMD
kind: language
gens: [gfx90a, gfx942]
dtypes: [fp16, bf16]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
  - https://github.com/tile-ai/tilelang
  - https://arxiv.org/abs/2511.08083
---

# TileLang pitfalls (AMD)

## TL;DR
TileLang is fast to write and competitive on CDNA3 attention, but it is not a CDNA4 peak path and its
reported numbers are vendor/project-specific. The traps: assuming MI350/CDNA4 maturity, treating
autotuned configs as portable, expecting asm-level GEMM, and missing the AMD-specific feature gaps
HipKittens documents.

## The pitfalls
1. **CDNA3-only maturity.** TileLang's AMD validation is MI250 (Auto MatrixCore) and MI300X (Async Copy).
   **MI350/CDNA4 is unproven** — don't ship TileLang as your gfx950 path without measuring.
2. **Missing AMD-specific abstractions** (HipKittens, arXiv 2511.08083): no first-class handling of
   flexible tile sizing under register pressure, thread-block scheduling, or **cache-aware XCD grid
   ordering**; it underuses **32×32×16 MFMA**, **`buffer_load_dwordx4`**, and **XCD swizzle** relative to
   peak AMD asm. → Expect below-ceiling perf on memory-bound and chiplet-cache-sensitive kernels.
3. **Backend dependence.** Some paths call into **CUTLASS/CK** — performance and portability inherit those
   backends' constraints.
4. **Autotuned configs are not portable.** A winning config is **shape-specific** (b/h/s/d) and
   **build-specific** (ROCm/TileLang version). Re-tune per serving shape and after every upgrade; never
   freeze and ship a single config.
5. **Vendor-reported numbers.** 1.53× Triton (FA), 1.98× Triton / ~AITER-parity (MLA), 257 TFLOPs single
   attention — all measured at specific shapes/versions. Re-measure on your stack.
6. **GEMM is not the strong suit.** ~0.94–1.05× Triton and below AITER asm for dense GEMM — use it for
   attention iteration, not as your matmul backend.
7. **LDS budget (64 KB on gfx942).** Deep `num_stages` or large `block_M×block_N` can overrun LDS; a
   hand-set config (vs the autotuner) can silently spill.
8. **Feature lag.** Some intrinsics / Tensor-Core acceleration are flagged "future" in AMD's blog —
   coverage trails the NVIDIA path; check the example before assuming a primitive is AMD-validated.

## Verify
- Confirm the target arch is actually validated (MI250/MI300X) and re-tune for it.
- Re-measure every quoted speedup at your shape (median of ≥3 warm repeats); compare against Triton and,
  if present, AITER.
- Greedy temp=0 parity vs a reference for correctness.

## Sources
- TileLang FlashAttention on MI300X (validated arches, autotune, vendor numbers): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- tile-ai/tilelang (AMD support matrix, FlashMLA example): https://github.com/tile-ai/tilelang
- HipKittens (arXiv 2511.08083 — missing AMD abstractions, 32x32x16/buffer_load_dwordx4/XCD-swizzle gaps, CUTLASS/CK deps, 257 TFLOPs): https://arxiv.org/abs/2511.08083
