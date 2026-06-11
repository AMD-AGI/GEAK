---
title: embedding — overview
kind: operator_overview
operator: embedding
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/vocab_parallel_embedding.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# embedding  (token-id → hidden vector gather)

## TL;DR
Token embedding is a **pure-gather, memory-bound** op (`H = W[ids]`): one `[V,d]` weight row per token, no
math. On AMD it is **not a hand-written kernel** — both vLLM and SGLang implement it as
`F.embedding` (= `torch.embedding` → a `hipMemcpy`/index_select gather), wrapped in a vocab-parallel
mask + all-reduce that is `@torch.compile`-fused. The only "optimization" levers are (1) letting
Inductor fuse the mask/zero-fill pointwise ops into one kernel and (2) the all-reduce that the vocab
split forces. It is **Amdahl-negligible** (≪1% of GPU time) — do not spend kernel-authoring effort here;
the sibling op that *does* matter is [[lm_head_logits]] (the same `[V,d]` weight read as a GEMM).

## Math contract
- **Lookup**: `H[t, :] = W[ids[t], :]`, `W ∈ [V, d]`, `ids ∈ ℤ^{T}`, out `H ∈ [T, d]`. dtype: weight
  bf16/fp16 (fp32 master in training), out matches weight; **no accumulation** (pure copy).
- **Vocab-parallel** (TP>1, Megatron split along V): rank `r` holds rows `[r·V/tp, (r+1)·V/tp)`.
  Per-rank: remap `ids` into local index space, **mask** out-of-range ids to 0, gather, **zero** masked
  rows, then `all_reduce(SUM)` so each rank sees the full embedding (exactly one rank contributes a
  non-zero row per token). See [[allreduce]].
- **Tied weights**: when `tie_word_embeddings`, the *same* `[V,d]` tensor backs both this op and
  [[lm_head_logits]] (`ParallelLMHead.weight = embed_tokens.weight`). One allocation, two access
  patterns (row-gather here, GEMM there).

## Shape regimes
- **prefill**: `T = chunk×batch` (1k–16k token ids) → a `[T,d]` gather, `d`=hidden (4k–8k). Coalesced
  per-row reads of `d·dtype` bytes; bandwidth-bound, trivially fast.
- **decode**: `T = running batch` (1..256) → tiny gather, latency = a few µs; dominated by launch
  overhead, not bandwidth.
- Large vocab (128k–256k, e.g. Llama-3 128k, Gemma 256k) inflates the **weight footprint**
  (256k×8k×2B = 4 GB), not the gather cost — the row count read is `T`, independent of `V`.

## Where it matters (Amdahl)
Embedding is **≪1% of GPU time** on every dense LLM (one gather + one all-reduce per forward). There is
no e2e win to chase in the lookup itself. The *only* embedding-adjacent cost worth a look is the
**vocab-parallel all-reduce** at TP>1 (it rides the same NCCL/RCCL path as everything else → see
[[allreduce]]); SGLang/vLLM keep it as a single fused `masked_fill_ + all_reduce`.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟡 competitive (only via Inductor codegen; no standalone tuned kernel) | [backends/triton.md](backends/triton.md) |
| hip | 🟡 competitive (`torch.embedding` = rocPRIM gather; hand-HIP only for exotic fusion) | [backends/hip.md](backends/hip.md) |
| vllm_kernels | 🟢 sota (the live path: `F.embedding` + vocab-parallel mask/all-reduce) | [backends/vllm_kernels.md](backends/vllm_kernels.md) |
| aiter | ⚪ na | no token-embedding kernel in aiter (only `rotary_embedding`); not authored |

## Fusion neighbors
- **mask + remap + zero-fill** pointwise ops fuse into the gather under `@torch.compile` (vLLM marks
  `get_masked_input_and_mask` with `@torch.compile`). → see [fusion.md](fusion.md).
- Downstream the embedding feeds the first [[rmsnorm]] / [[rope]]; in practice these stay separate (the
  gather is too cheap to be worth fusing into the norm).
- Tied-weight sharing with [[lm_head_logits]] is a memory fusion, not a compute fusion.

## Numerics
Pure copy → **bit-exact**, no accumulation, no tolerance question. The only correctness trap is the
vocab-parallel **mask/all-reduce** (a missed mask double-counts a row). See [numerics.md](numerics.md).

## How to bench
Isolated: time `F.embedding(ids, W)` for the served `(T, d, dtype)` plus, at TP>1, the masked-gather +
all-reduce block; oracle = `W[ids]` reference, bit-exact. e2e: it will not move a token/s gate — measure
only to *rule it out* as a hotspot (rocprofv3 Top-N; `embedding`/`index_select`/`masked_fill` rows ≪1%).

## Sources
- vLLM `VocabParallelEmbedding`/`ParallelLMHead` (`F.embedding`, masked gather, all-reduce, tie_weights):
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
- SGLang vocab-parallel embedding (same Megatron split):
  https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/vocab_parallel_embedding.py
- aiter has no token-embedding kernel (on-box grep: only `rotary_embedding`/`pos_encoding`):
  `/sgl-workspace/aiter` = `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0`.
- Memory-bound op guidance (coalesced reads, ≥1024 grid): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
