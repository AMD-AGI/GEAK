---
title: speculative_decode_verify — numerics
kind: technique
operator: speculative_decode_verify
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
  - https://docs.vllm.ai/en/latest/features/speculative_decoding/
  - https://github.com/sgl-project/sglang/issues/16027
---

# speculative_decode_verify — numerics

## Two correctness contracts
1. **Distribution preservation (the spec-decode guarantee).** Correct speculative decoding with rejection
   sampling produces a token stream **distributed identically to the target model alone** — speed only,
   no quality change. **Greedy** spec-decode must be **token-exact** vs greedy non-spec. If output differs,
   the verify/rejection-sampler is wrong (not "acceptable noise").
2. **Tree-mask correctness.** Token `j` must attend exactly to its ancestors + the cached prefix. A wrong
   mask either (a) lets a token see a sibling → wrong logits → corrupted verification, or (b) hides an
   ancestor → also wrong. Both silently **drop acceptance rate** (looks like a perf regression).

## AMD CDNA3-specific failures (documented, real)
The AMD SSD bring-up hit two bugs that are exactly the kind to watch for:
- **bf16/fp16 MFMA type-confusion in the rowsum kernel** → bfloat16 inputs interpreted with the wrong MFMA
  intrinsic → **draft acceptance rate collapsed**. Verify the MFMA intrinsic matches the dtype.
- **GQA row-mapping bug under custom masks on CDNA3** → corrupted attention outputs for grouped-query
  attention when tree decode used custom masks. Test tree attention specifically with GQA/MQA heads.
These corrupt outputs *only* on the tree/custom-mask path, so a plain-decode parity test won't catch them.

## fp8 verify
fp8 target + spec multiplies speed but adds the fnuz-dialect risk (off-by-one bias → 2×) on gfx942.
Accuracy-gate the fp8 verify; FP8 + spec is where AMD reports 3.6×, but only if numerics hold.

## Parity gate (do all three)
1. **Greedy token-exactness**: spec greedy output == non-spec greedy output, ≥10 prompts. Any divergence =
   bug. This is the strongest, cheapest gate.
2. **Acceptance-rate sanity**: measure accepted tokens/step; a sudden drop after a backend/kernel change =
   tree-mask or MFMA bug, not "tuning."
3. **Sampling fidelity** (temperature > 0): the accepted-token distribution must match the target's
   (statistical test over many samples) — the rejection sampler must be exact.

## Cross-backend note
Reduction order differs across AITER/Triton/CK verify kernels → benign bf16 token flips on long
*sampled* decode, but **greedy must stay exact** — that's the line between benign and bug.

## Sources
- AMD SSD CDNA3 bugs (bf16 rowsum MFMA, GQA custom-mask row-mapping): https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html
- Spec-decode distribution guarantee: https://docs.vllm.ai/en/latest/features/speculative_decoding/
- EAGLE+AITER integration (`max_split_per_batch`): https://github.com/sgl-project/sglang/issues/16027
