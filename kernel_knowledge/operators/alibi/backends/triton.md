---
title: alibi on triton — SOTA card
kind: sota_card
operator: alibi
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://arxiv.org/abs/2108.12409
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu
---

# alibi × triton

## TL;DR
On Triton, ALiBi is a **bias argument to the FMHA kernel**, not its own kernel. The Triton attention kernel
computes `bias = −m_h·(i−j)` inline from the tile indices and the per-head slope, adds it to the fp32 score
before softmax. Use it by passing the alibi slopes to the Triton attention backend.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton FMHA with alibi-slopes arg | Triton attention kernels (vLLM `TRITON_ATTN`, aiter triton attention) | gfx942/950, bf16/fp16 | near-zero added cost (1 FMA/score) | ALiBi models on the Triton attention path |
| (no standalone alibi triton op) | — | — | — | it's an attention feature |

## Config space / knobs
- Pass `alibi_slopes[H]` to the attention call; the kernel computes the bias inline.
- All real knobs are the attention kernel's (`num_stages=1`, `schedule_hint="attention"`, tile sizes,
  wave64 softmax) — see [[languages/triton_amd/patterns]] §4.
- Compute bias in fp32; don't materialize a `[seq,seq]` bias.

## Numerics / parity
fp32 bias before softmax; correct slope sequence + sign + causal-mask order. Deterministic bias; attention
reduction order differs → greedy re-gate. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- vLLM: `--attention-backend TRITON_ATTN` with the model's alibi slopes (the layer passes them).
- Direct: the aiter/Triton FMHA kernel takes an alibi/bias arg.

## Pitfalls & anti-patterns
- ⚠ Materializing the bias tensor → `O(seq²)` HBM, defeats the inline design.
- ⚠ Wrong slope sequence / sign / non-pow2-head handling.
- ⚠ Assuming a backend supports ALiBi when it doesn't → silent wrong positional behavior or a fallback.

## How to verify
Attention-with-alibi vs fp64 oracle; confirm the backend's alibi path engaged (not ignored); greedy parity.

## Alternatives / cross-links
[hip.md](hip.md) · [[attention_prefill_fmha]] · [[attention_decode_paged]] ·
[[languages/triton_amd/patterns]] §4 · [[rope]] (mutually exclusive).

## Sources
- ALiBi (linear bias, slopes): https://arxiv.org/abs/2108.12409.
- Triton attention tuning: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
- inline bias in FMHA: https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu.
