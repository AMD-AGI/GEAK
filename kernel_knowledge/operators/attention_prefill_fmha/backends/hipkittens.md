---
title: attention_prefill_fmha on HipKittens — SOTA card
kind: sota_card
operator: attention_prefill_fmha
backend: hipkittens
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, training]
status: sota
updated: 2026-06-09
sources:
  - https://arxiv.org/abs/2511.08083
  - https://arxiv.org/html/2511.08083v1
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk
  - https://github.com/HazyResearch/HipKittens
---

# attention_prefill_fmha × HipKittens

## TL;DR
HipKittens (HK) is the **academic SOTA for attention forward (prefill FMHA)** on CDNA4: in **~500 LoC** its
fwd kernel beats AMD's hand-written **AITER assembly 1.0–2.1×**, **CK 1.0–1.4×**, ROCm **Triton 1.2–4.5×**,
and PyTorch **SDPA 1.3–4.5×** on MI355X — i.e. it beats hand-asm AITER on average, while being a readable
tile kernel competitive with FlashAttention-3 on NVIDIA. Use HK as the forward perf reference; production
prefill is still AITER MHA asm/CK (ROCM_AITER_FA). Headline numbers gfx950 (MI355X); validated on gfx942.
**Status is SOTA for forward**; backward lives in [[operators/gqa_mqa_attention/backends/hipkittens]].

## SOTA implementation(s)
HK builds FlashAttention from tile primitives: register/shared tiles for Q/K/V, `mma` for QKᵀ and PV,
fused `exp`/row-max/row-sum for the online softmax, all scheduled with an AMD-native pattern (8-wave
ping-pong / 4-wave interleave) instead of NVIDIA wave specialization — plus **HBM-address swizzling** for
conflict-free async loads. The whole forward is **~500 LoC**. See [[languages/hipkittens]] and
arXiv 2511.08083 (forward results).

| impl | source | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| HK FMHA forward (~500 LoC) | arXiv 2511.08083v1; `github.com/HazyResearch/HipKittens` | gfx950 (gfx942 validated); bf16/fp16; non-causal & causal | **beats AITER 1.0–2.1×, CK 1.0–1.4×, Triton 1.2–4.5×, SDPA 1.3–4.5× @ MI355X, Hazy Nov 2025 (academic/author-reported)** | prefill fwd perf reference; beats hand-asm AITER on avg |
| (ref) AITER MHA asm/CK (ROCM_AITER_FA) | vLLM ROCm blog, Feb 2026 | gfx942/gfx950; bf16/fp16 | 2.7–4.4× TPS vs ROCM_ATTN; production default | shipping prefill backend |

## Config space / knobs (backend-specific)
- **Schedule**: 8-wave ping-pong (default, compact) vs 4-wave interleave (full per-wave register budget).
- **Pinned register tiles** for the matmul+vector mix (more decisive for backward; available for fwd).
- **MFMA shape** `16×16×32` default; **head dim** d=64/128 (d=64 is an HK strength — see GQA card).
- **Causality**: causal vs non-causal masking; **HBM-address swizzle** for conflict-free Q/K/V loads.

## Numerics / parity
- BF16/FP16 in, **FP32 softmax accumulate** (online row-max/row-sum) → FlashAttention-class parity vs
  reference; verify against torch SDPA / AITER on (B,H,N,D) with a max-abs/rel tolerance.

## Integration (how it gets used at serving time)
- HK is now positioned as **an aiter backend** per the landscape. Authoring path: build with HIPCC for
  gfx950/gfx942 from a pinned HK commit, wire the fwd kernel via the aiter attention dispatch seam (the same
  seam ROCM_AITER_FA dispatches through) or a model-level call-site rebind, then **e2e-gate** against
  [[operators/attention_prefill_fmha/backends/aiter]] (the isolated win must hold at e2e TTFT).

## Pitfalls & anti-patterns
- **Academic maturity**: arXiv 2511.08083v1, Nov 2025; unstable APIs, no support contract — pin a commit,
  bring your own parity + perf gate; production prefill = AITER MHA asm/CK.
- **gfx950 headline**; gfx942 validated, numbers differ — re-measure.
- **Forward only here**: HK's backward story (and where AITER still wins, e.g. **MHA non-causal bwd
  seq=8192: AITER 1169 > HK 1091**) is in [[operators/gqa_mqa_attention/backends/hipkittens]].
- **Author-reported, single-source, per-shape** — the 1.0–2.1× over AITER spans shapes; some shapes are
  near 1.0×. Re-bench.

## How to verify (bench + oracle)
```bash
hipcc --offload-arch=gfx950 ...   # pinned HK commit, FMHA fwd micro-bench
# bench HK fwd vs AITER FA / CK / SDPA over (B,H,N,D), causal & non-causal
# parity: max-abs/rel vs torch SDPA; then wire via aiter attn seam and e2e-gate TTFT
```

## Alternatives / cross-links
[[operators/attention_prefill_fmha/backends/aiter]] (production FA + e2e gate) ·
[[operators/attention_prefill_fmha/backends/ck]] · [[operators/attention_prefill_fmha/backends/triton]] ·
[[operators/attention_prefill_fmha/backends/asm]] ·
[[operators/gqa_mqa_attention/backends/hipkittens]] (backward, GQA, d=64) ·
[[languages/hipkittens]] · [[optimization/mfma_scheduling]].

## Sources
- HipKittens — arXiv 2511.08083 (Hazy Research, Nov 2025): https://arxiv.org/abs/2511.08083 ; HTML https://arxiv.org/html/2511.08083v1
- HazyResearch blog: https://hazyresearch.stanford.edu/blog/2025-11-09-hk ; code https://github.com/HazyResearch/HipKittens
- Beyond Porting: vLLM attention backends on AMD ROCm (AITER FA 2.7–4.4×, Feb 27 2026): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- AMD SOTA landscape §2 attention (HK fwd beats AITER 1.0–2.1× etc.): [[landscape/amd_sota_2026]]
