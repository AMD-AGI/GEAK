---
id: flydsl_prefill_moe_stage2_fp8partial
title: Halve grouped-MoE stage-2 down-proj reduce HBM traffic by storing the top-k partials in fp8 (compute
  stays bf16; symmetric scale on store, unscale on reduce)
kind: expert_skill
authors:
- zhengy
scope: kernel
match:
  operator:
  - grouped_gemm_moe
  - fused_moe_grouped_gemm
  arch_class:
  - '*'
  gens:
  - gfx950
  dtypes:
  - mxfp4
  - fp4_e2m1
  - fp8_e4m3
  - fp8_e4m3_fnuz
  regimes:
  - prefill
  from_backend: flydsl
  to_backend: flydsl
  profile_signature:
    op_name_regex: ''
    min_pct_gpu: 0.0
expects:
  isolated_speedup_min: 1.15
  isolated_scope: 'stage-2 SEGMENT = the down-proj GEMM kernel PLUS the separate top-k reduce kernel,
    summed. Do NOT score the GEMM kernel alone: ~75% of the win is the halved partial READ, which happens
    in the reduce kernel, so a GEMM-only measurement tops out at a measured 1.053x and fails this gate
    even when the recipe is correctly reproduced. A per-kernel timing filter matched on the GEMM name
    will not match the reduce kernel name -- verify your filter catches both.'
  parity: relaxed
validation:
  status: validated
  last_verified: '2026-08-05'
  gpu: gfx950 / MI355X
  model: a8w4 (fp8-act / fp4-wt per_1x32) grouped-MoE prefill M=16384, real routing imbalance 3.44x
  measured:
    isolated: 1.2099041
    e2e_pct: ''
    parity: pass
  measured_detail:
    scope: 'FlyDSL 0.2.2 stage-2 segment = down GEMM + separate reduce kernel, summed;
      rocprofv3 medians over 3 paired runs with 8 calls/run'
    segment: '1.206495 -> 0.997182 ms = 1.209904x (-17.35%)'
    reduce_kernel: '0.320969 -> 0.156617 ms = 2.049388x'
    down_gemm: '0.885526 -> 0.840565 ms = 1.053489x'
    parity: 'median logits_diff 0.00106758 -> 0.00141795 and cos_sim 0.998934 -> 0.998583;
      within the aiter real-routing gate threshold 0.01'
    bundling_check: 'optimized arm emitted cshuffle_pf8 plus infp8 reducer; baseline emitted bf16 partials'
  artifact: skills/flydsl_prefill_moe_stage2_fp8partial/validation_flydsl_0_2_2.yaml
role: advisory_prior
supersedes: []
---

## When to use
Trigger on the **problem signature, not a specific model**: a **grouped-GEMM MoE stage-2 (down-proj + top-k
reduce)** at **prefill / large-M** on gfx950 that is **HBM-traffic-bound on the per-(token, top-k slot) partial
tensor**. Down-proj writes `top_k` partial rows per token and the reduce sums them, so that partial is the
largest stage-2 HBM stream — written once and read once. Applies where the down path runs the **non-accumulating**
variant (each expert-slot's partial stored standalone, then summed) with a full-width (bf16) final output.

If the workflow has not already distinguished MoE phase/stage, resolve it before applying this skill. Do not
match on `grouped_gemm_moe` alone. This recipe is for the **prefill stage-2 down/reduce** path: look for the
stage-2 GEMM (`mfma_moe2` / down-proj naming), materialized top-k partial buffer, and a separate reduce kernel
in the scored segment. If the evidence instead points at decode stage-1 gate/up (`mfma_moe1`, sorted-block
leader mapping, no top-k reduce), use the stage-1 blkmap recipe instead. If phase or stage remains ambiguous,
treat this skill as not applicable and let the normal workflow exploration classify the bottleneck first.

## FlyDSL portability
This recipe requires FlyDSL `>=0.2.2`. The measured evidence below was produced on 0.2.2, but the
transferable content is the fp8-partial store/load mechanism, not private API names from that release.

On a newer FlyDSL version, inspect the current AITER/FlyDSL implementation, map the same partial format,
scale, reducer, and cache-identity invariants onto the available APIs, then run fresh compile/parity/A/B.
A version difference alone is not a reason to skip the skill; revalidate it on-box, and 0.2.2 performance numbers must not be
reused as evidence for another version.

The partial scale is part of the generated program ABI. Pass one scale as an explicit **compile argument**
to both producer and reducer, derive the reciprocal from that same value, and include the scale in both
JIT **cache identity** tuples/module names. A fixed producer constant combined with a reducer-only
environment override is a silent correctness bug. After porting, compile with a clean cache and require
both producer and reducer signatures (`cshuffle_pf8` and `infp8`) before full-logits parity or timing.

The **0.2.2 compatibility smoke** and strict rocprof validation on gfx950 used an isolated FlyDSL 0.2.2
wheel. Across three paired runs, the stage-2 segment median was
`1.206495 -> 0.997182 ms = 1.210x`; the GEMM was `0.885526 -> 0.840565 ms = 1.053x` and the reducer
was `0.320969 -> 0.156617 ms = 2.049x`. Median `logits_diff` was
`0.00106758 -> 0.00141795` and cosine was `0.998934 -> 0.998583`, within the relaxed gate.

## Mechanism
With `top_k` experts per token, stage-2 materializes `top_k` partial rows per token and then reduces them. At
bf16 that partial is **2 bytes/elem** written by the GEMM and read back by the reduce — pure traffic that
dominates stage-2 at prefill. Storing the partial as **fp8 (1 byte)** halves **both** the write and the read,
cutting the reduce's HBM traffic ~2x.

Precision comes from where the fp8 lives: keep the **MFMA and accumulation at full width** (bf16 datapath,
f32 accumulate) — **only the global partial store and its matching load are fp8.** Down-proj partials have a
narrow, stable dynamic range, so a single **symmetric scale `s`** applied before the fp8 store, with `1/s`
applied right after the fp8->f32 unpack in the reduce (before summation), keeps the values centered in the fp8
representable range; `s` cancels exactly in the f32 sum. The only lossy step is the fp8 round-trip of the
**stored partials** — small and bounded (cos_sim 0.9986, logits_diff ~+0.0004 vs the bf16-partial baseline). It
is **not** a re-quantization of weights/activations and **not** a new kernel: the same GEMM+reduce with an fp8
store/load epilogue-prologue variant.

## Procedure
1. **Confirm the partial is the stage-2 HBM bottleneck** (traffic ~= `2 x top_k x tokens x N x dtype_bytes`; the
   reduce is bandwidth-bound). Engage only on the non-accumulating down path with a full-width output.
2. **Store side (down-proj GEMM epilogue).** Keep the CShuffle / accumulate datapath at bf16; at the **final
   global store** of each per-(token, slot) partial, multiply by scale `s` and convert to fp8 (e4m3). Allocate
   the partial buffer as fp8 (half the bytes).
3. **Reduce side.** Load fp8 partials, convert fp8->f32, multiply by `1/s`, **then** sum the `top_k` slots and
   write the full-width result. The unscale must happen before summation so `s` cancels exactly.
4. **Choose `s` from the partial magnitude histogram** so pre-store values sit mid-range in e4m3 — avoid
   saturation at the top and flush-to-zero at the bottom. `s` is data-dependent: re-derive it per model/quant;
   a fixed constant is valid only for the distribution it was calibrated on.
5. **Make the scale a single source of truth.** The GEMM store side and reducer load side must use the same
   `s` and `1/s`. Do not let an environment variable or late runtime override change only the reducer scale
   or only the store scale; that creates a silent precision bug. Prefer carrying the scale through the same
   config tag / generated kernel parameters used to select the fp8-partial arm, and log the effective scale
   on both sides during validation.
6. **Match the fp8 flavour to the arch** (gfx950 = OCP e4m3, not fnuz).
7. **Apply a runtime-signature gate before timing.** A valid optimized arm must show the partial buffer is fp8
   on the GEMM side and the reducer reads the fp8 partial variant (for example a reduce kernel containing
   `infp8`, while the bf16 baseline contains `inbf16`). If the reducer name stays bf16 or only the reduce side
   changes, reject the timing as a plumbing/config failure.
8. **Validate FULL-logits parity vs the bf16-partial baseline at prefill M** (this path is lossy, so it must
   stay within the accepted relaxed tolerance), then confirm with rocprof the reduce-kernel traffic cut (~2x).
9. **Score the A/B on the segment: down GEMM + reduce kernel, summed** (see `expects.isolated_scope`). Report
   the two kernels' before/after separately as well, so the write-side and read-side shares stay visible.

## Knobs & pitfalls
- **The store scale `s` is a calibrated constant, not a free knob.** Too large -> fp8 saturation (Inf / clamp);
  too small -> underflow to zero. Both **silently** degrade accuracy. Re-derive `s` from the actual partial
  distribution for any new model / shape.
- **Scale consistency is part of correctness.** A fixed reference value such as `0.0007` is only valid when
  both producer and reducer receive the same value. A reducer-only environment override can pass compilation
  and still corrupt the decoded scale; treat mismatched effective scales as correctness failure even if timing
  improves.
- **Only valid on the non-accumulating down path** with a tile-N aligned to the fp8 store width and a full-width
  output; the accumulating / split-K path is **not** covered — leave it bf16.
- **It is a partial data-format change, not weight/activation re-quantization** — no model quant recalibration
  is needed, and it is the same kernel (fp8 store/load variant), not a replacement operator.
- **fp8 e4m3 flavour must match the GPU** (OCP on gfx950).
- **Measure the whole segment, or you will measure ~nothing.** The write-side saving lands in the down GEMM
  but the read-side saving — the larger share — lands in the *separate* reduce kernel, which has a different
  kernel name. A per-kernel timing filter matched on the GEMM's name silently drops the reduce and reports
  only the GEMM's `1.053x`, making a correctly reproduced recipe look like a failure. Time both kernels.
- **Reject arm identity ambiguity.** The optimized arm must have a positive runtime signature on both sides:
  fp8 partial output from the GEMM and `infp8` input in the reducer. Missing either signature means the skill
  did not actually run, regardless of reported speed.

## Do-no-harm notes
- **Lossy by construction** (fp8 partial round-trip) -> this is a **relaxed-parity** optimization, never
  bit-exact. Keep it OFF for any model whose down-proj partial range has not been calibrated: a wrong scale is a
  **silent** precision loss, not a crash. Gate acceptance on full-logits parity within tolerance vs the
  bf16-partial baseline.
- Because acceptance depends on a model-calibrated scale **and** a parity check, it must never be applied blindly
  across shapes/models — an advisory prior gated by the workflow's on-box parity + A/B, never a default.
- The default path stores bf16 partials and is byte-identical to baseline -> no regression when not triggered.

## Sources
- `validation_flydsl_0_2_2.yaml` records the FlyDSL 0.2.2 gfx950 validation: three paired rocprofv3 runs,
  stage-2 segment median `1.206495 -> 0.997182 ms = 1.210x`.
- Runtime evidence showed `cshuffle_pf8` partial stores and an `infp8` reducer; relaxed full-logits parity
  stayed at `logits_diff ~0.001418`, `cos_sim ~0.998583`.
- The implementation remains an opt-in fp8-partial store/load variant; the default bf16-partial path is
  unchanged.
