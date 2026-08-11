---
name: re-store-the-streamed-weight-operand-in-the-narrowest-dtype--moe-grouped-gemm-gfx950-compute-bound
description: Re-store the dominant streamed weight operand in fp4 the MFMA reads natively: 31.35x -> 39.95x cumulative in one direction, 42.24x with the wider-K instruction
keywords: [dequant, mfma, moe, operand-reuse, fp8, tile-shape, compute-bound, roofline, dtype-dialect]
kernels: [fused_moe_kernel]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: compute-bound
key: narrowing the B-weight storage dtype to native fp4 (e2m1) on a fused-MoE grouped GEMM on gfx950, where the weight operand dominates streamed traffic
lifecycle: active
type: lever
confidence: ★★
effect: moving B-weight storage to fp4 (e2m1), consumed natively by the MFMA, was worth a single-direction jump to a 39.95x ledger cumulative after 11 passes of knob work had crawled the pass geomean 18.84 -> 31.35x; matrix_instr_nonkdim=16 on the large grid (2x-K v_mfma_f32_16x16x128_f8f6f4) then moved the cumulative to 41.57x, and the whole chain ended at 42.24x cumulative (42.59x best pass over 35 passes); per case at the end, 29.93x at batch 2, 50.01x at batch 32 and 50.24x at batch 64 - the smallest case gains least because it is not streaming enough weight to be roofline-limited - with the empirical roofline moving 0.020 (latency-bound) -> 0.510 (compute-bound); the same shrink applied to the ACTIVATION operand closed three ways and paid nothing (A->fp4 measured cos 0.9883 against a >=0.99 gate, worth ~+13% / ~45x if the gate ever loosens; A->fp6 e3m2/e2m3 is rejected by the triton 3.6.0 frontend with no gfx950 fp6 MFMA lowering; outlier-preserving mixed-precision A->fp4 had a ~+5% ceiling at an unsafe margin on iid-Gaussian activations)
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.57h / 35 passes, 2026-08-11
last_seen: 2026-08-11
---
# Re-store the streamed weight operand in the narrowest dtype the matrix core consumes natively
- lever: When the profile says the kernel is latency- or bandwidth-limited on one large streamed operand, the cheapest large lever is often the operand's STORAGE dtype rather than the loop - pick the narrowest format the part's matrix core reads natively, and re-check the MFMA instruction shape afterwards, since the narrower operand can unlock a wider-K instruction that is a second, separate gain.
- apply: Store the dominant streamed operand in the narrow native format with a per-block scale folded into the epilogue; keep the minority operand wide. Check that it is a cbsz/blgp native consumption and not a software unpack, or the win is eaten by VALU.
- stack: total 42.24x cumulative on the ledger = two directions compounded on top of 11 passes of knob work (31.35x pass geomean before them) - 1. B-weight storage to native fp4 - the single-direction jump to 39.95x cumulative, the bulk of the win; 2. matrix_instr_nonkdim=16 unlocking the 2x-K f8f6f4 MFMA on the large grid - 41.57x cumulative on top of (1), only available once (1) narrowed the operand; note: attribution is incremental in landing order, and the residual to 42.24x came from later knob passes.
- pitfall: the same narrowing was proposed for the activation operand -> it is the minority of streamed traffic, so it spent the whole accuracy margin for a few percent -> measure each operand's traffic share before proposing it, and expect the frontend to reject formats with no native MFMA lowering.
- verify: Empirical roofline before/after and the emitted MFMA opcode - a native consumption shows the wide-K f8f6f4 instruction and no unpack VALU storm.
- caution: Also verify the accuracy gate on the shrunken operand at the per-case level; the activation-side shrink here failed a cos>=0.99 gate at 0.9883.
- source: chuschen 16h time-budget campaign run, 15.57h / 35 passes, 2026-08-11
