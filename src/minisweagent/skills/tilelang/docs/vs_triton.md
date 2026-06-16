---
title: TileLang vs Triton (and asm) on MI300X
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

# TileLang vs Triton (and asm)

## TL;DR
On MI300X (CDNA3), TileLang generally **edges out Triton** on attention while staying far easier to write
than asm: FA fwd ~**1.53× Triton** / ~**2.7× PyTorch**; FlashMLA ~**parity with AITER asm** and ~**1.98×
Triton** (vendor/project-reported). For GEMM the gap is small (~0.94–1.05× Triton). The ceiling still
belongs to hand-asm (AITER); HipKittens reports a tile DSL recovering most of that gap with less
brittleness, and critiques TileLang for missing AMD-specific abstractions. Pick TileLang for fast
attention iteration; Triton for portability/ecosystem; asm or AITER for the last few percent.

## Core comparison (MI300X, fp16, vendor/project-reported)
| workload | TileLang vs Triton | TileLang vs asm/AITER | source |
|---|---|---|---|
| FlashAttention fwd (b1,h8,s4096,d128) | **1.53×** (0.36 vs 0.55 ms) | — | ROCm blog 2026-01-20, ROCm 7.0.1 |
| FlashMLA (bf16/fp16, batch 64/128) | **1.98×** | **~parity** ("performance parity with aiter-asm in most cases") | tilelang repo example_mla_amd |
| dense GEMM | ~0.94–1.05× | below AITER asm | tilelang/HipKittens |
| single attention kernel | — | 257 TFLOPs (HipKittens-measured) — below peak | arXiv 2511.08083 |

## Why TileLang wins on attention vs Triton
- **Autotuner** sweeps 108 FA configs in ~1 s (see [autotune.md](autotune.md)); Triton's AMD autotune
  space is narrower and buffer-loads were not Triton's default on AMD as of late 2025 (HipKittens).
- **Auto LDS swizzle** for AMD's bank-conflict rules without code changes.
- **Flexible tile sizes** (no WGMMA/TMA constraint → `block_m` need not be a multiple of 64).
- HipKittens on Triton: developers "often resort to inline assembly" to recover AMD performance; Triton
  "struggles with register lifetime tracking and lowering to the most performant intrinsics."

## Why asm/AITER still wins the ceiling
- AITER's hot paths are **raw asm** hand-scheduled by experts — see [../asm_mfma/raw_asm.md](../asm_mfma/raw_asm.md).
- TileLang lacks abstractions for register-pressure-aware tile sizing, thread-block scheduling, and
  **cache-aware (XCD) grid ordering**, and uses **32×32×16 MFMA / `buffer_load_dwordx4` / XCD swizzle**
  less aggressively than peak AMD kernels (HipKittens) — so it is CDNA3-validated, not a CDNA4 peak path.
- Some TileLang paths still call into **CUTLASS/CK** backends.

## Pitfalls
- All TileLang-vs-X numbers above are **vendor/project-reported** at specific shapes/versions — re-measure
  on your stack before relying on them.
- TileLang's AMD maturity is **CDNA3 (MI250/MI300X)**; treat MI350/CDNA4 as unproven.
- A TileLang win at one shape does not generalize — it is autotuned per shape.

## Verify
- Bench TileLang, Triton, and (if available) the AITER kernel at your exact shape; median of ≥3 warm
  repeats; note spread.
- Greedy temp=0 parity vs a reference for correctness before comparing speed.

## Sources
- TileLang FlashAttention on MI300X (1.53× Triton, 2.7× PyTorch, latency table): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- tile-ai/tilelang (FlashMLA AMD ~AITER-asm parity, 1.98× Triton): https://github.com/tile-ai/tilelang
- HipKittens (arXiv 2511.08083 — 257 TFLOPs, Triton inline-asm resort, TileLang missing AMD abstractions, GEMM gap): https://arxiv.org/abs/2511.08083
