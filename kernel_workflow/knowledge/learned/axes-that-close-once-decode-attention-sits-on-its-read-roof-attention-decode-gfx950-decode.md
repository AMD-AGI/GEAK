---
key: a HIP C++ paged decode attention on gfx950/CDNA4 already reading at ~90% of nameplate HBM peak, where the residual is a fixed per-call head rather than bytes
type: anti-pattern
confidence: ★★
effect: four families measured closed over eight late rounds: fractional LLC residency 13 sweep points all negative and the case the physics demanded was 2.6% worse; a variant that genuinely reached 7 waves/SIMD at zero scratch was -0.18% (5/5 pairs negative); the backend codegen-flag axis produced one byte-identical-ISA null and one -0.026% loser; an ISA byte census put measured DRAM traffic at 1.000x of theory on all 7 cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-17
name: axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode
description: Closed axes on a paged decode already at its read roof: LLC residency, more occupancy, backend codegen flags and byte reduction all ~1.00x or worse.
keywords: ['anti-pattern', 'closed-axis', 'attention-decode', 'paged-attention', 'roofline', 'occupancy', 'codegen', 'l2-residency', 'isa-inspection', 'decode', 'gfx950']
kernels: ['paged_attention_ll4mi_QKV_mfma16_kernel', 'paged_attention_ll4mi_reduce_kernel']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
verified_on: 2026-08-15
origin_kernels: ['mi355x_vllm_hip_paged_attention_decode']
---
# Axes that close once decode attention sits on its read roof
- lever: Once an affine fit of time against partition count shows the marginal byte cost near the read roof, treat LLC residency, further occupancy, backend codegen flags and byte-count reduction as the low-yield families and spend the round on the fitted intercept, the per-call head, instead.
- apply: The cheap triage is the fit itself plus an ISA byte census: measured DRAM traffic over the theoretical minimum at 1.000x says there is no byte multiplier left, and the intercept's share (about 23% of the shortest-context case, about 13% at the longest) sizes the only remaining pot.
- verify: Price each closed family with one arm rather than a sweep: an occupancy arm that reaches the target wave count at zero scratch predicts the sign of the family, and a codegen flag needs an ISA diff next to the resource dump because a flag that is accepted and emits byte-identical code is a proven null the resource dump cannot see.
- pitfall: Higher occupancy closed from the far side: the extra waves arrived, the slope improved 0.93%, and the arm still lost because it consumed the prologue-pipelining slot that had already been banked -> price a new arm against the incumbent's structure, not against the original baseline.
A zero-work probe of the second dispatch was read as bounding its removal -> it keeps a full-machine dispatch ramp, so it bounds optimizing that kernel and never fusing it away -> use a separate bound for fusion.
Two independently certified sub-1% wins were discarded by a round-delta promotion gate ~10x coarser than the measured run-to-run bar -> judge small wins by paired-sign agreement and effect over pair spread.
- caution: Also verify additivity rather than assuming it when stacking these late small wins: two disjoint-file directions predicted to sum to +0.47% measured +0.40%, so independent mechanisms composed at about 85%.
- source: run mi355x_vllm_hip_paged_attention_decode-bmk7-12h, 2026-08-11..17, gfx950/MI355X, 13 rounds / 25 direction-slots, directions r5_d0, r7_d0, r10_d2, r13_d0; director-validated geomean 1.365x, correctness 7/7
