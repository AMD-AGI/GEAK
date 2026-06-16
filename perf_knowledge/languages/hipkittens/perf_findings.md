---
title: HipKittens — measured perf & honest DSL comparison (MI355X)
kind: language
gens: [gfx950, gfx942]
dtypes: [bf16, fp8_e4m3]
regimes: [both]
status: experimental
updated: 2026-06-08
sources:
  - https://arxiv.org/html/2511.08083v1
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk
---

# HipKittens perf findings

## TL;DR
On **MI355X (CDNA4, gfx950)** HK reports SOTA or near-SOTA: it **matches AITER/hipBLASLt on BF16/FP8
GEMM**, **beats AITER assembly** on several attention forward/backward shapes, and is **1.2–10×** over
baselines on underserved workloads (some attention shapes, GQA backward, memory-bound). The paper's most
useful contribution to perf_knowledge is its **honest, measured indictment of the competing AMD DSLs**: Triton
underperforms even a vanilla BF16 GEMM, Mojo MHA ~50% of peak from bank conflicts, TileLang is CDNA3-only
and only "competitive with PyTorch." **All numbers below are vendor/author-reported** (Stanford HazyResearch,
Nov 2025) on MI355X unless noted — not independently re-measured in perf_knowledge.

## HK measured numbers (author-reported, MI355X gfx950, arXiv 2511.08083v1, 2025-11)
| workload | config | HK | best baseline | note |
|---|---|---|---|---|
| BF16 GEMM | M=N=K=8192, 256×256 tile, 8-wave | **1610 TFLOPS** | hipBLASLt 1561; CUTLASS 1570 (B200 ref) | Table 2; matches/edges hipBLASLt |
| FP8 GEMM | 8-wave ping-pong | **3222 TFLOPS** (48 LoC) | 4-wave 3327 (183 LoC) | Table 3; interleave +3% at ~4× code |
| MHA non-causal bwd | seq 4096 | HK 855, HK+pinned **1024** | AITER 1018 | Table 1; pinned tiles reach AITER asm |
| MHA non-causal bwd | seq 8192 | HK 909, HK+pinned **1091** | AITER 1169 | Table 1 |
| GQA non-causal bwd | — | 8-wave 1.8×, 4-wave **2.3×** over baseline | AITER 272–384, PyTorch SDPA 259 TFLOPS | AITER GQA bwd weak |
| GEMM vs Triton | BF16/FP8 | **1.3–3.0×** faster | ROCm Triton | — |
| attention fwd | various | beats AITER 1.0–2.1×, SDPA 1.3–4.5×, CK 1.0–1.4×, Triton 1.2–4.5× | — | competitive w/ FlashAttention-3 |
| memory-bound (fused dropout-residual-layernorm, rotary) | — | beats AITER & torch.compile **1.1–2.2×** | — | — |
| XCD grid swizzle | M=N=K=9216 | row-major 1113 → best (W5/C25) **1145 TFLOPS** / 18.3 TB/s | — | Table 4; L2 55%→75% |

Headline: **1.2–10×** over baselines on underserved workloads; competitive with well-tuned NVIDIA
Blackwell kernels.

## The honest cross-DSL findings (why this file matters for perf_knowledge)
These are the load-bearing "competing DSLs leave perf on the table on AMD" claims, all from the HK paper:

- **AMD Triton (ROCm)** "struggles with register lifetime tracking," may fail to reclaim registers or
  lower vectorized loads, and **underperforms even on a vanilla BF16 GEMM** — HK beats it 1.3–3.0× on
  GEMM, up to 4.5× on attention. Corroborates perf_knowledge `languages/triton_amd/` and the dense_gemm triton
  card (Triton loses to tuned hipBLASLt/aiter on plain GEMM).
- **Mojo** MHA forward "suffers from bank conflicts" → **~50% of peak**, e.g. **430 TFLOPs** at
  B=16,H=16,N=2048,D=128 on MI355X. (See [../mojo/status_amd.md](../mojo/status_amd.md) — Mojo's own
  matmul reaches SOTA, but its attention had not yet closed the swizzle gap as of Nov 2025.)
- **TileLang** "is currently limited to CDNA3," its MHA kernel is only "competitive with PyTorch" (among
  the slowest baselines), lacks some matrix-core shapes, and depends on CUTLASS/CK backends. A single
  TileLang attention kernel is reported at **257 TFLOPs on MI300X**.
- **AITER (hand-tuned assembly)** is strong on forward/GEMM but **weak on GQA backward**: only
  **272/384 TFLOPS at seq 8192** (causal/non-causal), which HK beats 1.8–2.5×.
- **PyTorch SDPA** Llama GQA backward = **259 TFLOPS** (24% of HK SOTA); AITER GQA backward = 30%.

## Reading these honestly (sourcing caveat)
- All HK numbers are **author/vendor-reported**, single-source (the paper), MI355X-centric. Per perf_knowledge
  sourcing rules, treat as *vendor-labeled* until re-measured on-box. Where HK contradicts a turnkey lib
  on *your* shape, re-bench.
- The **cross-DSL findings are the durable takeaway**, not HK's absolute TFLOPS: they explain *why* the
  perf_knowledge backend landscape ranks aiter/hipBLASLt/CK/asm above Triton/Mojo/TileLang for production AMD
  kernels today.
- HK is a research artifact — do not deploy it as a serving dependency without your own parity + perf gate.

## Sources
- HipKittens paper (Tables 1–5, all numbers): https://arxiv.org/html/2511.08083v1
- Blog (Mojo ~50% peak, TileLang CDNA3-only, Triton vanilla-GEMM underperformance):
  https://hazyresearch.stanford.edu/blog/2025-11-09-hk
- Cross-links: [overview.md](overview.md) · [primitives.md](primitives.md) ·
  [../mojo/status_amd.md](../mojo/status_amd.md) · dense_gemm triton card
  (`operators/dense_gemm/backends/triton.md`).
