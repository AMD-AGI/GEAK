---
title: embedding on vllm_kernels — SOTA card
kind: sota_card
operator: embedding
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/vocab_parallel_embedding.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
---

# embedding × vllm_kernels

## TL;DR
This is **the live serving path**. vLLM (and SGLang, near-identical) implement token embedding as a
**pure-PyTorch `VocabParallelEmbedding`**: `F.embedding` for the gather, a `@torch.compile`-fused mask for
the Megatron vocab split, and a `tensor_model_parallel_all_reduce` to reconstruct full embeddings across
TP ranks. There is no custom HIP kernel in `csrc/rocm/` for embedding — it rides ATen's ROCm gather. SOTA
here means "the correct, fused, near-free reference," not a tuned kernel.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `VocabParallelEmbedding.forward` (`F.embedding` + masked gather + all-reduce) | `vllm-project/vllm@HEAD:vllm/model_executor/layers/vocab_parallel_embedding.py` | gfx942/950, bf16/fp16 | ≪1% GPU time; bandwidth-bound gather + one all-reduce (no isolated gate worth chasing) | **always** — the production path |
| `ParallelLMHead` (subclass, weight only) | same file | — | `forward` raises; weight consumed by [[lm_head_logits]] | when weights are tied |

## Config space / knobs
- **TP layout**: vocab split along `V` (Megatron); shard size `V/tp`, `valid_offset`/`vocab_mask` per rank.
  No tile knobs — the gather is library.
- **`@torch.compile`**: `get_masked_input_and_mask` is decorated so remap+mask+zero-fill fuse into one
  kernel. Engage by running vLLM's compiled path (default on V1 for many models).
- **all-reduce backend**: the post-gather sum rides the same collective path as the rest of the model
  (RCCL or `custom_all_reduce`/`quick_all_reduce`) → tune there, not here. See [[allreduce]].
- **Tied weights**: `tie_weights()` sets `lm_head.weight = embed_tokens.weight` (GGUF exception returns
  the module). One `[V,d]` allocation shared with [[lm_head_logits]].

## Numerics / parity
Bit-exact gather; vocab-parallel sum is one-hot-per-token (FP add of zeros, order-stable). The only gate is
the **mask boundary** at shard edges — verify TP>1 reproduces single-GPU bit-for-bit (see
[../numerics.md](../numerics.md)).

## Integration (rebind seam)
Call site: `VocabParallelEmbedding.forward` in the model's `embed_tokens`. No env flag toggles a faster
kernel (there is none); the relevant levers are the **compiled-graph fusion** and the **all-reduce
backend**. To verify engagement: rocprofv3 shows a fused gather/mask kernel + an all-reduce at TP>1, both
≪1% of GPU time.

## Pitfalls & anti-patterns
- Treating embedding as a tunable hotspot — it is not; spend the budget on [[lm_head_logits]] / [[dense_gemm]].
- Vocab-parallel mask off-by-one at shard boundaries → wrong embedding for a few tokens (quality
  regression, not a crash) — the one real bug class. Gate at TP>1.
- Expecting an aiter embedding kernel — **there is none** (aiter ships only `rotary_embedding`).

## How to verify
rocprofv3 Top-N → gather + all-reduce ≪1% GPU time; oracle `W[ids]` bit-exact; TP>1 boundary id sweep
matches single-GPU.

## Alternatives / cross-links
[triton.md](triton.md) (Inductor codegen) · [hip.md](hip.md) · [../overview.md](../overview.md) ·
[[allreduce]] · [[lm_head_logits]] · backend [[vllm_kernels]].

## Sources
- vLLM `VocabParallelEmbedding`/`ParallelLMHead`/`tie_weights` (`F.embedding`, masked gather, all-reduce):
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
- SGLang equivalent: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/vocab_parallel_embedding.py
- ROCm platform (no custom embedding kernel in csrc/rocm): https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
