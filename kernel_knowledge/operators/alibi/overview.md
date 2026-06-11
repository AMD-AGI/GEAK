---
title: alibi — overview
kind: operator_overview
operator: alibi
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://arxiv.org/abs/2108.12409
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# alibi  (Attention with Linear Biases — positional bias added to attention scores)

## TL;DR
ALiBi replaces RoPE-style positional encoding with a **linear bias added to the QKᵀ scores**:
`score_{ij} += −m_h · (i − j)`, where `m_h` is a per-head slope. It is **not a standalone kernel** on the
serving path — it is **folded into the attention kernel** (added to the score matrix before softmax). The
optimization story is therefore "make sure your FMHA backend supports an ALiBi bias," not "tune an alibi
op." Used by BLOOM, MPT, some Falcon/Baichuan variants; modern models use RoPE.

## Math contract
For head `h` with slope `m_h` (geometric sequence over heads), and query/key positions `i, j`:
`bias_{ij} = −m_h · (i − j)` (causal: only `j ≤ i`), added to `S_{ij} = Q_i·K_jᵀ/√d` before softmax.
Slopes `m_h = 2^(−8h/H)`-style. dtype: bias computed in fp32, added to the fp32 score accumulator. No
weights, no cache — just the head slope and the position delta.

## Shape regimes
- **prefill**: the full `[seq, seq]` causal score matrix gets the bias; computed inside FMHA tile by tile
  (`bias = −m_h·(i−j)` per score element, no materialization).
- **decode**: the single query row vs all keys; `bias = −m_h·(q_pos − j)` per key.

## Where it matters (Amdahl)
ALiBi adds a **near-zero-cost** elementwise term inside attention (a multiply-add per score). Its impact is
**enabling correct long-context extrapolation**, not perf. The only "cost" is that the attention backend
must support it — if it doesn't, you fall back to a slower path. ~0% GPU time as a separate op.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 sota (bias term in the FMHA Triton kernel) | [backends/triton.md](backends/triton.md) |
| hip | 🟢 sota (bias in vLLM/CK FMHA HIP kernel) | [backends/hip.md](backends/hip.md) |

(No standalone aiter/library "alibi" op — it's an attention-kernel feature flag.)

## Fusion neighbors
**Folded into attention** ([[attention_prefill_fmha]], [[attention_decode_paged]],
[[gqa_mqa_attention]]) — that IS the op. Mutually exclusive with [[rope]] (a model uses one or the other).
See [fusion.md](fusion.md).

## Numerics
bias in fp32 added before softmax; slope sequence per head must match the model; causal mask interplay. See
[numerics.md](numerics.md).

## How to bench
Bench the **attention kernel with ALiBi enabled** vs disabled at the model's head config; oracle = fp64
attention with the bias. There is no isolated alibi bench worth running.

## Sources
- ALiBi paper (linear bias, per-head slopes, extrapolation): https://arxiv.org/abs/2108.12409.
- ALiBi folded into the attention score (FMHA): vLLM paged-attn HIP kernel https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu.
- Attention kernel tuning on AMD: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
