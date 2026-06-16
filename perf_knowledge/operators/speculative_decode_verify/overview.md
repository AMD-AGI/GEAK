---
title: speculative_decode_verify — overview
kind: operator_overview
operator: speculative_decode_verify
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html
  - https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
  - https://docs.vllm.ai/en/latest/features/speculative_decoding/
  - https://github.com/sgl-project/sglang/issues/16027
---

# speculative_decode_verify  (draft-token verify / tree attention)

## TL;DR
Speculative decoding has a small **draft** model propose `k` future tokens (or a **tree** of candidates),
which the **target** model verifies in **one parallel forward pass** — turning `k` memory-bound
single-token decodes into one slightly-larger batched pass. The two kernels that matter here are
(1) **tree/verify attention**: the target attends over the draft tree with a **custom (tree) mask** so
parent/child causal relations hold while siblings don't see each other, and (2) the **rejection sampler /
verify** that accepts the longest matching prefix. On MI300X spec-decode gives **up to ~3× out-of-the-box,
2.31× measured (vLLM)** and **3.6× combined with FP8** (AMD vendor). It is **decode-regime, custom-mask
attention** — the same FA kernel as [[attention_decode_paged]] plus a tree mask. The single most important
fact: the AMD path is **maturing** — tree decode + custom masks have hit real CDNA3 bugs (bf16 MFMA
type-confusion in rowsum; GQA row-mapping under custom masks), so **accuracy-gate the acceptance rate**.

## Math contract
- **Draft**: small model (or EAGLE head: takes prev token + base hidden state) auto-regressively builds a
  **tree** of `T` candidate tokens with a parent/child structure.
- **Verify (tree attention)**: target runs one forward over the `T` flattened draft tokens with a **tree
  mask** — token `j` attends to key `i` iff `i` is an **ancestor** of `j` in the tree (plus the cached
  prefix). Optimized tree attention splits the work into **prefix** (cached KV) + **suffix** (the tree),
  per xFormers / SpecInfer / Medusa.
- **Accept (rejection sampling)**: compare target vs draft distributions along each tree path; accept the
  longest verified prefix, sample one bonus token. Greedy = exact-match accept.
- Mask vocabulary (sglang `build_tree_kernel_efficient`): `TreeMaskMode.FULL_MASK` (general tree) /
  `QLEN_ONLY` (chain/linear draft) / partial-packed tree mask.

## Shape regimes
- **Decode only.** Query length = tree size `T` (e.g. 8–64 tokens) instead of 1; KV = cached prefix +
  tree. The verify pass is a **small-M, custom-mask FA** over paged KV — launch-bound, batch×T queries.
- Draft tree shapes: EAGLE (multi-candidate tree), Medusa (multiple heads), n-gram, MTP (multi-token
  prediction). Linear (chain) draft = `QLEN_ONLY` mask; branching tree = `FULL_MASK`.

## Where it matters (Amdahl)
Spec-decode directly attacks decode being **memory-bandwidth bound** (GPU idle waiting on weights). The
verify attention itself is cheap; the wins come from **acceptance rate** (more accepted tokens/pass) and
**not regressing** the target. A broken tree mask silently drops acceptance (looks like a perf loss, is
actually a correctness bug). Combined with FP8 the effects multiply (AMD: 3.6× on Llama-3.1-405B).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 (custom-mask verify FA + sgl tree-mask builder; editable) | [backends/triton.md](backends/triton.md) |
| aiter | 🟡 (unified-attention verify; EAGLE+AITER had integration bugs) | [backends/aiter.md](backends/aiter.md) |
| sglang_kernels | 🟢 (EAGLE/Medusa/ngram/MTP workers, `build_tree_kernel_efficient`, fused KV materialize) | [backends/sglang_kernels.md](backends/sglang_kernels.md) |
| vllm_kernels | 🟢 (V1 spec-decode: EAGLE/Medusa/ngram, rejection sampler) | [backends/vllm_kernels.md](backends/vllm_kernels.md) |

## Fusion neighbors
Tree-mask build ([[gather_scatter]]), fused KV materialize for the tree, custom-mask FA
([[attention_decode_paged]]), rejection sampling ([[sampling_topk_topp]] / [[argmax_topk]]), fp8 KV. See
[fusion.md](fusion.md).

## Numerics
Custom mask correctness + rejection-sampling fidelity (must preserve the target distribution); bf16 MFMA
dtype match. See [numerics.md](numerics.md).

## How to bench
Decode with a draft model / EAGLE head; measure **accepted tokens/step** and tok/s vs no-spec; oracle =
the same target greedy without spec (output must be identical for greedy spec-decode). See
[tuning.md](tuning.md).

## Sources
- AMD spec-decode on MI300X (up to 3×, 2.31× vLLM, 3.6× +FP8): https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html
- AMD SSD (tree decode + custom-mask attention; CDNA3 bf16 rowsum + GQA row-mapping bugs): https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
- vLLM spec-decode (EAGLE/Medusa/ngram, rejection sampler): https://docs.vllm.ai/en/latest/features/speculative_decoding/
- sglang EAGLE+AITER integration bug (`max_split_per_batch`): https://github.com/sgl-project/sglang/issues/16027
- sglang on-box tree builder: `sgl-project/sglang:python/sglang/srt/speculative/eagle_utils.py` (`build_tree_kernel_efficient`, `TreeMaskMode`).
