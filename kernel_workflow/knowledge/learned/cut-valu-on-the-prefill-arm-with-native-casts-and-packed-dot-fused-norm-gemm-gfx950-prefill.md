---
key: the instruction-issue-bound prefill arm of a bf16 fused norm+GEMM path on gfx950/MI355, written in TileLang against a C++ half-precision wrapper type
type: lever
confidence: ★★
effect: cumulative 1.87x to 2.18x in one round (the largest single step of the campaign), isolated prefill body -38% at 7211 tokens; carried on both prefill cases (hidden 7168 and 4096) and flat on the two decode cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: cut-valu-on-the-prefill-arm-with-native-casts-and-packed-dot-fused-norm-gemm-gfx950-prefill
description: On an issue-bound bf16 prefill norm+GEMM arm, native scalar casts, packed bf16 dot and a scalar accumulator cut VALU: 1.87x to 2.18x cumulative
keywords: ['valu-bound', 'bf16', 'packed-valu', 'dtype-emulation', 'prefill', 'tilelang', 'gfx950', 'unroll', 'size-gating', 'dead-list']
kernels: ['mhc_pre_big_fuse']
platforms: ['gfx950']
kernel_class: fused_norm_gemm
regime: prefill
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-14
roofline: the arm moves from issue-bound to ~91% of the packed-dot issue roof
levers: ['compute.valu-reduction', 'compute.packed-dot']
---
# Cut VALU on the prefill arm with native casts and packed dot
- lever: On a bf16 arm whose pipe is issue-bound rather than byte-bound, four cheap edits compound: cast through the compiler's native scalar bf16 type instead of the library wrapper (whose constructor expands into a software round-to-nearest-even), feed a packed two-lane bf16 dot against a packed mirror of the weight operand, keep the row sum-of-squares in a scalar accumulator, and re-pick the unroll factor.
- apply: Mirror the streamed operand once in packed layout so the packed dot has an operand to consume; the scalar accumulator is what removes a long tail of select instructions the vector form emitted per element.
- verify: Diff the ISA per region before and after so the cut is attributed to the instruction it removed, then re-time the isolated arm against the frozen baseline; a cut that does not move runtime means the arm is already at a traffic roof rather than an issue roof.
- pitfall: A size gate that selected the fused arm only above a large hidden size was leaving a whole shape unserved -> the threshold had been fitted against the slower pre-cut body -> re-fitting it after the body sped up was worth another 4.3% of the session, and two unroll/width knobs a do-not-retest list called two-sided optima moved again once 42 registers were freed, for +1.15% geomean.
- caution: Also verify the knob under the production configuration rather than the kernel default: one round measured a knob at its default split count instead of the shipped one and reported a clean 18% win for what is a 15% loss, and a second round nearly repeated it with a global env knob feeding two shapes with different defaults.
- source: run mi355x_vllm_tilelang_mhc_fused_post_pre-bmk7-12h, rounds 4 and 9, director validation accepted 2026-08-14
