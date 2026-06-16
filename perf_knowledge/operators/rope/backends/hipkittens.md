---
title: rope on HipKittens — SOTA card
kind: sota_card
operator: rope
backend: hipkittens
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-09
sources:
  - https://arxiv.org/abs/2511.08083
  - https://arxiv.org/html/2511.08083v1
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk
  - https://github.com/HazyResearch/HipKittens
---

# rope × HipKittens

## TL;DR
Rotary position embedding (RoPE) is **memory-bound**, in the same class as norm where HipKittens (HK) beats
AITER and torch.compile **1.1–2.2×** on MI355X (the HK paper measures "rotary" alongside
dropout-residual-layernorm). Status: **competitive** — a clean HK tile kernel saturates HBM bandwidth where
no tuned assembly exists, but for production the AITER RoPE (+KV-cache) fusion and the vLLM ROCm RoPE+KV
fusion pass remain the default. Headline gfx950 (MI355X); validated gfx942.

## SOTA implementation(s)
RoPE is elementwise rotate-pairs over Q/K, bandwidth-bound. HK uses **vector/register tiles** with fused
`sin`/`cos` rotate and **HBM-address swizzling** for conflict-free async loads, approaching roofline
bandwidth. See [[languages/hipkittens]] and arXiv 2511.08083 (memory-bound "rotary" results).

| impl | source | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| HK rotary (RoPE) | arXiv 2511.08083v1; `github.com/HazyResearch/HipKittens` | gfx950 (gfx942 validated); bf16/fp16 | **beats AITER & torch.compile 1.1–2.2× @ MI355X, Hazy Nov 2025 (academic/author-reported)** — same mem-bound class as norm | mem-bound rotary / custom fused RoPE |
| (ref) AITER RoPE + KV-cache fusion | landscape §4; vLLM fusions | gfx942/gfx950; bf16/fp16 | production default; ROCm RoPE+KV-cache fusion pass (O1+, auto) | shipping RoPE path |

## Config space / knobs (backend-specific)
- **Vectorized load width** + **HBM-address swizzle** for conflict-free loads (the lever for a mem-bound op).
- **Tile/block** sizing for bandwidth; **fusion** of RoPE with KV-cache write in one tile pass.
- 8-wave vs 4-wave scheduling is near-irrelevant here; occupancy to hide HBM latency matters.

## Numerics / parity
- BF16/FP16 in; sin/cos tables (FP32) → parity with AITER/torch RoPE within tolerance. Watch interleaved vs
  half-rotated (GPT-NeoX vs GPT-J) layout — must match the model's convention; validate against reference.

## Integration (how it gets used at serving time)
- HK is positioned as **an aiter backend** per the landscape. Authoring path: build with HIPCC for
  gfx950/gfx942 (pinned HK commit), wire the RoPE (or fused RoPE+KV) kernel via the aiter dispatch seam or a
  model call-site rebind, then **e2e-gate** against [[operators/rope/backends/aiter]] — mem-bound, so the
  e2e win is small; confirm it survives end-to-end.

## Pitfalls & anti-patterns
- **Mem-bound → modest e2e**: the vLLM ROCm RoPE+KV-cache fusion pass already captures much of the win.
- **Layout convention**: interleaved vs half-rotated RoPE must match the model — mismatches are silent
  correctness bugs.
- **Academic maturity**: arXiv 2511.08083v1, Nov 2025; unstable APIs, no support contract — pin a commit.
- **gfx950 headline**; gfx942 validated, numbers differ. **Author-reported, single-source, per-shape** — re-bench.

## How to verify (bench + oracle)
```bash
hipcc --offload-arch=gfx950 ...   # pinned HK commit, RoPE micro-bench
# bench HK vs AITER RoPE at target (tokens×heads×dim); report achieved HBM GB/s vs roofline
# parity vs torch RoPE (match interleaved/half-rotated convention); wire via aiter seam and e2e-gate
```

## Alternatives / cross-links
[[operators/rope/backends/aiter]] (production + KV fusion + e2e gate) · [[operators/rope/backends/triton]] ·
[[operators/rope/backends/hip]] · [[operators/mrope/overview]] · [[operators/paged_kv_copy/overview]] ·
[[operators/rmsnorm/backends/hipkittens]] (same mem-bound class) ·
[[languages/hipkittens]] · [[optimization/mfma_scheduling]].

## Sources
- HipKittens — arXiv 2511.08083 (Hazy Research, Nov 2025): https://arxiv.org/abs/2511.08083 ; HTML https://arxiv.org/html/2511.08083v1
- HazyResearch blog: https://hazyresearch.stanford.edu/blog/2025-11-09-hk ; code https://github.com/HazyResearch/HipKittens
- AMD SOTA landscape §4 norm/rope (HK beats AITER/PyTorch 1.1–2.2× mem-bound): [[landscape/amd_sota_2026]]
