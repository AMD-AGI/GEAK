---
title: gqa_mqa_attention on HipKittens — SOTA card
kind: sota_card
operator: gqa_mqa_attention
backend: hipkittens
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [training, prefill]
status: sota
updated: 2026-06-09
sources:
  - https://arxiv.org/abs/2511.08083
  - https://arxiv.org/html/2511.08083v1
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk
  - https://github.com/HazyResearch/HipKittens
---

# gqa_mqa_attention × HipKittens

## TL;DR
HipKittens (HK) is the **academic SOTA for GQA/MQA attention, especially the backward pass**, on CDNA4:
GQA non-causal **backward** runs **1.8× (8-wave) / 2.3× (4-wave)** over baselines on MI355X — because the
incumbents are weak here (AITER GQA bwd reaches only **30%** of SoTA, PyTorch SDPA **24%**). HK also wins on
**head-dim d=64** shapes that the tuned asm libs underserve. Its **pinned register tiles** (feeding AGPRs
directly to MFMA) are the enabler for SOTA backward. Use HK as the GQA/bwd perf reference; honest caveat —
on **MHA non-causal bwd seq=8192 AITER (1169) still beats HK (1091)**. Headline gfx950 (MI355X); validated
gfx942. **Status: SOTA, especially backward.**

## SOTA implementation(s)
HK's backward is matmul-heavy *and* vector-heavy (dQ/dK/dV + softmax recompute). On CDNA, HIPCC refuses to
feed AGPRs to matrix instructions, forcing redundant `v_accvgpr_read` moves; HK's **pinned register tiles**
pin those registers and use AGPRs directly as MFMA inputs — the key to SOTA backward. Scheduled with 8-wave
ping-pong or 4-wave interleave (interleave's full per-wave register budget helps the register-pressured
backward). See [[languages/hipkittens]] and arXiv 2511.08083 Table 1.

| impl | source | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| HK GQA non-causal **backward** | arXiv 2511.08083v1 | gfx950 (gfx942 validated); bf16/fp16 | **8-wave 1.8× / 4-wave 2.3× over baselines @ MI355X, Hazy Nov 2025 (academic)** — vs AITER 272–384 TFLOPS (30% of SoTA), SDPA 259 (24%) | GQA training backward — clear win |
| HK MHA non-causal **bwd** + pinned tiles | arXiv 2511.08083v1 Table 1 | gfx950; bf16; seq 4096 | HK+pinned **1024 TFLOPS** ≈ AITER 1018 @ MI355X (academic) | seq=4096 bwd parity with hand-asm |
| HK MHA non-causal **bwd** + pinned tiles | arXiv 2511.08083v1 Table 1 | gfx950; bf16; seq 8192 | HK+pinned **1091** — **AITER 1169 wins** @ MI355X (academic) | NOT HK's win; AITER preferred |
| HK GQA/MQA **forward**, d=64 | arXiv 2511.08083v1 | gfx950; bf16/fp16 | beats baselines (fwd in the 1.0–2.1× vs AITER class); **d=64 a strength** | small-head GQA fwd |

## Config space / knobs (backend-specific)
- **Schedule**: 4-wave interleave often wins the backward (full register budget eases dQ/dK/dV pressure);
  8-wave ping-pong for compact forward.
- **Pinned register tiles**: the decisive lever for backward — pin registers, feed AGPRs to MFMA, avoid
  `v_accvgpr_read`. Sharp tool: bypasses the allocator; validate parity.
- **Head dim** d=64 vs 128 (d=64 underserved by tuned asm → HK win); **GQA group ratio** (KV-head sharing).
- **MFMA shape** `16×16×32`; causal vs non-causal.

## Numerics / parity
- BF16/FP16 in, **FP32 accumulate**; backward recomputes softmax → FlashAttention-class parity. Validate
  dQ/dK/dV against torch SDPA backward; **pinned tiles can silently corrupt** if mis-pinned — gate parity.

## Integration (how it gets used at serving time)
- HK is positioned as **an aiter backend** per the landscape. Authoring path: build with HIPCC for
  gfx950/gfx942 (pinned HK commit), ISA-check that pinned tiles actually feed AGPRs to MFMA (no spurious
  `v_accvgpr_read`), wire via the aiter attention seam / call-site rebind, then **e2e-gate** against
  [[operators/gqa_mqa_attention/backends/aiter]] — and only adopt where HK actually wins (GQA bwd, d=64),
  keeping AITER for MHA bwd seq=8192.

## Pitfalls & anti-patterns
- **AITER still wins** MHA non-causal bwd seq=8192 (1169 > HK 1091) — do not blanket-replace AITER bwd.
- **Pinned register tiles are sharp** — silent correctness / occupancy cliffs if mis-pinned. ISA-verify + parity-gate.
- **Academic maturity**: arXiv 2511.08083v1, Nov 2025; unstable APIs, no support contract — pin a commit.
- **gfx950 headline**; gfx942 validated, numbers differ. **Author-reported, single-source, per-shape** — re-bench.

## How to verify (bench + oracle)
```bash
hipcc --offload-arch=gfx950 ...   # pinned HK commit, GQA fwd+bwd micro-bench
AMDGCN dump: confirm pinned tiles feed AGPRs to MFMA (no spurious v_accvgpr_read)
# bench HK bwd vs AITER GQA bwd / SDPA over (B,H,Hkv,N,D); parity dQ/dK/dV vs torch SDPA backward
# e2e-gate only where HK wins (GQA bwd, d=64), keep AITER for MHA bwd seq=8192
```

## Alternatives / cross-links
[[operators/gqa_mqa_attention/backends/aiter]] (production; MHA bwd seq8192 winner) ·
[[operators/gqa_mqa_attention/backends/ck]] · [[operators/gqa_mqa_attention/backends/triton]] ·
[[operators/attention_prefill_fmha/backends/hipkittens]] (forward / MHA prefill) ·
[[languages/hipkittens]] · [[optimization/mfma_scheduling]] (pinned register tiles, AGPR/`v_accvgpr_read`).

## Sources
- HipKittens — arXiv 2511.08083 (Hazy Research, Nov 2025): https://arxiv.org/abs/2511.08083 ; HTML (Table 1) https://arxiv.org/html/2511.08083v1
- HazyResearch blog: https://hazyresearch.stanford.edu/blog/2025-11-09-hk ; code https://github.com/HazyResearch/HipKittens
- AMD SOTA landscape §2 attention (GQA bwd 1.8×/2.3×; AITER GQA bwd 30%; MHA bwd seq8192 AITER 1169 > HK 1091): [[landscape/amd_sota_2026]]
