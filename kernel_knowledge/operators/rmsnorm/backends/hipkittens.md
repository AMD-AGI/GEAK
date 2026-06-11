---
title: rmsnorm on HipKittens — SOTA card
kind: sota_card
operator: rmsnorm
backend: hipkittens
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode, training]
status: sota
updated: 2026-06-09
sources:
  - https://arxiv.org/abs/2511.08083
  - https://arxiv.org/html/2511.08083v1
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk
  - https://github.com/HazyResearch/HipKittens
---

# rmsnorm × HipKittens

## TL;DR
On **memory-bound normalization** kernels (RMSNorm / LayerNorm, incl. fused dropout-residual-layernorm),
HipKittens (HK) beats AITER and PyTorch (torch.compile) **1.1–2.2×** on MI355X — these are workloads where
no tuned hand-assembly exists, so a clean tile kernel that saturates HBM bandwidth wins. Status:
**competitive-to-SOTA on mem-bound** norm. For production, AITER fused RMSNorm (+ vLLM Inductor fusion
passes) remains the default; use HK as the perf reference / when you need a custom fused norm. Headline
gfx950 (MI355X); validated gfx942.

## SOTA implementation(s)
RMSNorm/LayerNorm are **HBM-bandwidth-bound**, not MFMA-bound: the win is conflict-free vectorized loads
and a tight reduce. HK expresses the norm with **vector/register tiles** + fused `sum`/`rsqrt`/`mul`, and
uses **HBM-address swizzling** for conflict-free async HBM→LDS loads — pushing toward roofline bandwidth
(the GEMM XCD work shows HK reaching ~18.3 TB/s on MI355X). See [[languages/hipkittens]] and
arXiv 2511.08083 (memory-bound results).

| impl | source | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| HK fused norm (RMSNorm/LayerNorm, dropout-residual-layernorm) | arXiv 2511.08083v1; `github.com/HazyResearch/HipKittens` | gfx950 (gfx942 validated); bf16/fp16 | **beats AITER & torch.compile 1.1–2.2× @ MI355X, Hazy Nov 2025 (academic/author-reported)** | mem-bound norm / custom fused norm |
| (ref) AITER fused RMSNorm (+vLLM Inductor) | landscape §4; vLLM fusions | gfx942/gfx950; bf16/fp16 | production default; RMSNorm+quant fusion 1–6% e2e | shipping norm path |

## Config space / knobs (backend-specific)
- **Schedule**: less critical than for GEMM (mem-bound) — focus on **vectorized load width** and
  **HBM-address swizzle** for conflict-free loads.
- **Tile / block** sizing to saturate bandwidth; **fusion** (residual-add, dropout) inside one tile pass.
- 8-wave vs 4-wave matters little here; occupancy to cover HBM latency is the lever.

## Numerics / parity
- BF16/FP16 in, **FP32 reduction** for the RMS/mean → parity with AITER/torch RMSNorm within FP32-reduce
  tolerance; if fused with FP8 quant, gate FP8 accuracy downstream.

## Integration (how it gets used at serving time)
- HK is positioned as **an aiter backend** per the landscape. Authoring path: build with HIPCC for
  gfx950/gfx942 (pinned HK commit), wire the fused norm via the aiter norm dispatch seam or a model
  call-site rebind, then **e2e-gate** against [[operators/rmsnorm/backends/aiter]] — for a mem-bound op the
  e2e win is small, so confirm it survives end-to-end, not just microbench.

## Pitfalls & anti-patterns
- **Mem-bound → modest e2e**: a 1.1–2.2× kernel speedup on a bandwidth-bound op rarely moves e2e much; the
  vLLM Inductor fusion path (RMSNorm+quant) often captures most of the win already.
- **Academic maturity**: arXiv 2511.08083v1, Nov 2025; unstable APIs, no support contract — pin a commit.
- **gfx950 headline**; gfx942 validated, numbers differ. **Author-reported, single-source, per-shape** — re-bench.

## How to verify (bench + oracle)
```bash
hipcc --offload-arch=gfx950 ...   # pinned HK commit, RMSNorm/LayerNorm micro-bench
# bench HK vs AITER fused RMSNorm at target (rows×hidden); report achieved HBM GB/s vs roofline
# parity vs torch RMSNorm (FP32 reduce); wire via aiter norm seam and e2e-gate
```

## Alternatives / cross-links
[[operators/rmsnorm/backends/aiter]] (production + e2e gate) · [[operators/rmsnorm/backends/triton]] ·
[[operators/rmsnorm/backends/hip]] · [[operators/layernorm/backends/aiter]] ·
[[operators/fused_add_rmsnorm/overview]] · [[operators/fused_norm_quant/overview]] ·
[[operators/rope/backends/hipkittens]] (same mem-bound class) ·
[[languages/hipkittens]] · [[optimization/mfma_scheduling]].

## Sources
- HipKittens — arXiv 2511.08083 (Hazy Research, Nov 2025): https://arxiv.org/abs/2511.08083 ; HTML https://arxiv.org/html/2511.08083v1
- HazyResearch blog: https://hazyresearch.stanford.edu/blog/2025-11-09-hk ; code https://github.com/HazyResearch/HipKittens
- AMD SOTA landscape §4 norm (HK beats AITER/PyTorch 1.1–2.2× mem-bound): [[landscape/amd_sota_2026]]
