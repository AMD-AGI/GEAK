---
title: Mojo — Modular's portable GPU kernel language
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [both]
status: experimental
updated: 2026-06-08
sources:
  - https://www.modular.com/blog/achieving-state-of-the-art-performance-on-amd-mi355----in-just-14-days
  - https://arxiv.org/html/2511.08083v1
  - https://docs.modular.com/mojo/
---

# Mojo

## TL;DR
Mojo is **Modular's Python-superset systems language** for writing portable GPU kernels, paired with the
**MAX** inference framework and **Mammoth** serving system. Its differentiator is **portability by
design**: it does *not* hardcode GPU knowledge — hardware specifics live in libraries as **parameterized
kernels** (no "magic constants" for SIMD width / tile shape), so Modular claims ~99.9% of the stack is
architecture-agnostic and a new GPU is "a few kernels." On AMD this paid off: Modular brought MI355X
(CDNA4) up and reached a **matmul kernel 3% faster than hipBLASLt in ~1 day / 14 days total** (vendor-
reported, Oct 2025). The open question for AMD is *attention*: independent measurement (HipKittens paper)
found Mojo's MHA at **~50% of peak** from LDS bank conflicts. See [status_amd.md](status_amd.md).

## Concepts
- **Python-superset, MLIR-backed.** Mojo compiles through MLIR; kernels are written in a Pythonic syntax
  with systems-level control (explicit types, SIMD, ownership). Targets NVIDIA, AMD (CDNA), and early
  Apple silicon from one source.
- **Library-directed, not compiler-hardcoded.** Offloading, scheduling, and instruction selection are
  expressed through abstractions and **parametric kernels** retuned per hardware, rather than baked into
  the compiler — the stated reason AMD enablement was fast.
- **MAX / Mammoth stack.** Mojo (language) → MAX (inference framework, op library) → Mammoth (distributed
  serving). The portability claim is for the whole stack: Modular reported MAX outperforming AMD's
  optimized vLLM fork by up to **2.2×** across workloads on MI355X while staying portable.
- **Authoring backend, not a library to call.** In perf_knowledge terms Mojo is an *authoring language* (a column
  in the backend taxonomy alongside triton/hip/ck/asm/tilelang), used to write a kernel you then bench
  against aiter/hipBLASLt — not a tuned library you dispatch to.

## The levers
- **Parameterize MFMA/tile shape, SIMD width, stages** rather than hardcoding — the same kernel retunes
  for gfx942 vs gfx950.
- **Lean on MAX op library** for the parts you don't author; author only the hot kernel.
- For dense GEMM on MI355X, Modular's published kernel is ~500 LoC and reached SOTA — a viable reference
  pattern; for attention, expect to fight LDS bank conflicts (the known weak spot).

## Pitfalls
- **Closed/young ecosystem.** Mojo is Modular-controlled; AMD support is recent (MI355X enabled Sep–Oct
  2025) and evolving. Pin a toolchain version; APIs move.
- **Attention not yet at peak on AMD** (~50% MHA per HK, Nov 2025) — do not assume Mojo's GEMM win
  generalizes to FMHA. Re-measure per kernel.
- **Vendor-reported numbers** (Modular blog) are single-source; treat as vendor-labeled until reproduced.
- Not part of the standard ROCm distribution — adopting it adds a non-AMD toolchain dependency to a
  serving stack.

## Verify
- Mojo docs / MAX: https://docs.modular.com/ ; reproduce the M=N=K=8192 GEMM benchmark on your MI3xx vs
  hipBLASLt before trusting the 3% claim on your stack.
- For attention, bench Mojo MHA vs AITER/CK/Triton FA at your shapes; expect the ~50% gap unless the LDS
  swizzle has since been fixed.

## Sources
- Modular blog "Achieving SOTA on AMD MI355 in 14 days" (matmul 3% > hipBLASLt; library-directed design;
  2.2× vs vLLM fork; Oct 2025): https://www.modular.com/blog/achieving-state-of-the-art-performance-on-amd-mi355----in-just-14-days
- Independent Mojo MHA ~50% peak finding: HipKittens paper https://arxiv.org/html/2511.08083v1
- Mojo language docs: https://docs.modular.com/mojo/
- detail: [status_amd.md](status_amd.md)
