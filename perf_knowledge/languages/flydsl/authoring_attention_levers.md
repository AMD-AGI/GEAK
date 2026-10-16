---
title: FlyDSL fused-attention design
kind: language
gens: [gfx942]
dtypes: [bf16, fp16]
regimes: [prefill, training, both]
updated: 2026-10-16
---

# FlyDSL fused-attention design

Attention chains several matmuls through softmax or its derivative. Long-lived
operands, accumulators, layout conversions and reductions can dominate the cost.
Use the complete forward/backward contract and all output/state updates when
comparing a fusion or decomposition.

| Design choice | Compare explicitly |
|---|---|
| Fusion boundary | Saved intermediates/launches versus registers, synchronization and recomputation; splitting does not have a universal work-cost threshold |
| MFMA orientation | Producer output ownership versus the next consumer's operand layout; a transpose may require shuffles, LDS or reloads |
| Query-outer versus KV-outer | Which outputs require cross-block reduction, workspace volume, atomics and contention |
| Wave/output partition | Per-wave live operands, accumulation work and communication; more or fewer waves can each help |
| Grid order | Irregular/causal work distribution and cache locality; launch-index order does not guarantee execution order |
| Epilogue | Required output dtype and rounding; include a separate cast if needed by the contract |

Derive staging counts and fragment indexing from the chosen MFMA instruction and
tile shape. For a wave64 `16×16×16` design, quantities such as `BM/16`, `HD/16`
and `BN/NW` are meaningful only when its divisibility and ownership assumptions
hold. Validate tails and asymmetric head dimensions. Do not copy lane-fragment
maps between instruction variants.

Inspect register movement, scratch accesses and synchronization when testing a
longer live range. A lower register count, fewer barriers or exact instruction-issue
total does not alone prove the proposed mechanism or full numerical correctness.
Use target-specific resource definitions; retain a correct slower realization when
its remaining cost offers a useful next experiment.

For debugging, compare all required outputs and useful intermediate values against
the declared reference/tolerance. No generic cosine or FP8 error threshold replaces
that contract. Verify that the chosen geometry and implementation actually ran,
then compare unprofiled full-operation timing. Retained source variants can support
later ablations without keeping permanent runtime switches for every attempt.

## Sources

- [Shared FlyDSL constraints](authoring_optimization.md)
- [MFMA instruction/layout reference](../asm_mfma/mfma_intrinsics.md)
- [Detailed fused-attention recipe](../../expert_skills/skills/flydsl_fused_attention_backward/skill.md)
- [Historical MLA backward example](../../operators/mla_attention/backends/flydsl.md)
- [HipKittens paper](https://arxiv.org/abs/2511.08083)
