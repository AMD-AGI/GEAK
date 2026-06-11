---
title: speculative_decode_verify — fusion
kind: technique
operator: speculative_decode_verify
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
  - https://docs.vllm.ai/en/latest/features/speculative_decoding/
---

# speculative_decode_verify — fusion

## Fuse away the small-kernel storm
The SSD blog's key warning: naive multi-round speculative sampling launches **numerous small kernels**
(nested loops over tree depth) — launch overhead dominates at decode. Fusion is about collapsing those.

| fusion | where | benefit |
|---|---|---|
| **tree-mask build** | one kernel (sgl `build_tree_kernel_efficient`) | builds the flattened tree mask + position ids in one pass instead of Python loops |
| **fused KV materialize** | sgl `triton_ops/fused_kv_materialize.py` | write all draft-tree KV into the paged cache in one kernel, not T tiny copies |
| **prefix + suffix verify attention** | one custom-mask FA | attend cached prefix densely + tree suffix masked, in one kernel ([[attention_decode_paged]]) |
| **verify + rejection-sample setup** | epilogue | produce target logits laid out for the sampler without an extra gather |
| **fp8 KV quant on tree insert** | KV materialize | fnuz quantise + insert fused (gfx942) |

## Macro-fusion: unified attention for verify
sglang's `SGLANG_AITER_UNIFIED_VERIFY` / unified-attention path handles the verify pass in one kernel for
launch-bound small batches — the spec analogue of the chunked-prefill+decode unification. aiter's
`unified_attention` carries the verify case.

## What stays separate
- Draft model forward (its own small forward / EAGLE head) — separate from verify.
- The rejection sampler / accept logic — small, but kept distinct for correctness clarity
  ([[sampling_topk_topp]]).

## Cross-links
- Verify attention: [[attention_decode_paged]] · [[gqa_mqa_attention]] (GQA custom-mask path is the buggy
  one — test it).
- Mask/tree build: [[gather_scatter]] · KV: [[paged_kv_copy]] · [[kv_cache_quant]].
- Sampling: [[sampling_topk_topp]] · [[argmax_topk]].
- Languages/backends: [[triton_amd]] · [[aiter]] · [[sglang_kernels]] · [[vllm_kernels]].

## Sources
- SSD small-kernel storm + tree decode: https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
- sglang fused tree mask + KV materialize: `sgl-project/sglang:python/sglang/srt/speculative/` (on-box).
- vLLM spec-decode structure: https://docs.vllm.ai/en/latest/features/speculative_decoding/
