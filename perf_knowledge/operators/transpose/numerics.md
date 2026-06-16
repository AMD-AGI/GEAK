---
title: transpose — numerics
kind: technique
operator: transpose
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# transpose — numerics

## Byte-exact, no tolerance
Transpose is a **pure element relocation** — no arithmetic, no accumulation, no rounding. The output is
**bit-identical** to a reference for every dtype (fp32/bf16/fp16/fp8/int8). Oracle is
`torch.equal(out, ref.t().contiguous())`, not `allclose`. If a "transpose" produces a numeric delta, the
implementation is wrong (mis-strided / partial-tile bug), not a precision artifact.

## The two real correctness traps
1. **Partial-tile masking.** When `M`/`N` aren't multiples of the tile, the boundary tiles must mask both
   the load and the store. A common bug writes garbage (or reads OOB) on the ragged edge — caught only by
   testing non-power-of-2 shapes (e.g. 4097×513). Always include odd dims in the oracle.
2. **fp8 dialect is irrelevant here** — because no value is interpreted, a transpose moves fp8 bytes
   verbatim regardless of FNUZ (gfx942) vs OCP (gfx950). The FNUZ-vs-OCP 2× hazard only bites the
   *consumer* that reads the transposed fp8 as numbers (see [[operators/transpose/fusion.md]]: if you
   fuse a cast into the transpose, the cast — not the move — carries the dialect risk).

## CDNA4 `ds_read_tr` transpose-on-read
The hardware transpose (gfx950) operates at **16-bit element granularity** (`tr_b16`). It relocates 16-bit
lanes exactly; for bf16/fp16 this is byte-exact. For 8-bit dtypes the 16-bit transpose granularity means
you transpose **pairs** — verify the element layout matches the consumer (an off-by-pair layout is a
correctness bug, again caught by an exact oracle, not a tolerance).

## Sources
- Element-move semantics, dtype preservation: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- fp8 FNUZ/OCP dialect hazard (consumer-side only): [[hardware/shared/dtype_numerics.md]], [[languages/triton_amd/pitfalls.md]].
