---
title: context_parallel_attention — numerics
kind: technique
operator: context_parallel_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/22223
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
---

# context_parallel_attention — numerics

## CP must be bit-comparable to single-GPU attention
CP shards the *computation* of the same attention — done right, it produces the same result (within fp
reduction-order noise) as a single-GPU run. The numerics live entirely in the **LSE-merge**:

1. **LSE-merge associativity.** Combining partial outputs from `cp` ranks uses online-softmax:
   `merge((O_a, m_a, l_a), (O_b, m_b, l_b))` rescales by `exp(m_a - m), exp(m_b - m)`. This **must be in
   fp32** and the running max `m` tracked exactly. A bf16 merge or a dropped rescale corrupts the output —
   the larger `cp`, the more merges, the more error compounds.
2. **Merge order.** Floating-point merge is not perfectly associative; ring (incremental, one block at a
   time) and all-to-all (single full-seq local FA, no cross-rank merge) take different reduction paths →
   slightly different bits. Both must match the reference within tolerance; pick one and gate it.
3. **Zigzag re-gather.** Causal zigzag assigns non-contiguous tokens per rank; the output must be
   **un-permuted** back to sequence order. An index bug here silently shuffles tokens (a correctness bug,
   not precision).

## FP8 KV in CP
fp8 KV cache (fnuz on gfx942) transferred around the ring must keep the dialect consistent across ranks —
a wrong-dialect read on any rank is off ~2×. Accuracy-gate fp8 CP.

## Parity gate
- Oracle: single-GPU full attention at the same (shorter, fits-in-HBM) seq, or a CP-disabled run.
- Gate: CP output vs oracle, fp32, max-abs and mean-abs error within FA tolerance; **token-exact** greedy
  for short prompts where single-GPU fits.
- Sweep `cp ∈ {2,4,8}` — error should not grow materially with `cp` (if it does, the LSE-merge isn't fp32).
- Long greedy bf16 token flips across ring vs all-to-all are benign (reduction-order equivalence-class) —
  distinguish from a merge/zigzag bug by comparing logits and the un-permutation.

## Sources
- SGLang zigzag ring attention (token assignment / re-gather): https://github.com/sgl-project/sglang/issues/22223
- vLLM ROCm extend/chunked-context (LSE merge across chunks): https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
