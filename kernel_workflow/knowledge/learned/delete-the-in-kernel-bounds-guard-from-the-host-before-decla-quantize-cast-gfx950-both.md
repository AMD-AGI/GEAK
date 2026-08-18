---
key: grid-stride elementwise / quantize bodies on gfx950 where an in-kernel bounds guard blocks a scheduling transform, and buffer addressing already bounds-checks in hardware
type: lever
confidence: ★★
effect: the assigned load/fmax phase-split alone is a -4.7% loss; deleting the guard by construction flips the sign, carrying ~90% of round 9's win on the fused small case against ~10% for the assigned mechanism; extended to the two multi-block cases it measured +1.529% [95% CI +0.97, +2.09], 3.0x the control spread, sign replicated 7/7 across three sessions, with the byte-identical third case moving 0.000% as constructed
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both
description: A bounds guard can be what makes a latency-hiding transform lose; deleting it from the host with exact tiles flipped the sign and carried ~90% of a round's win
keywords: ['grid-stride', 'host-runtime', 'launch-shape', 'software-pipelining', 'vgpr', 'isa-inspection', 'quantize-cast', 'raw-hip', 'measurement-rig', 'gfx950']
kernels: ['scaled_quant_kernel']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: both
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-14
levers: ['host.launch-shape', 'compute.software-pipelining']
---
# Delete the in-kernel bounds guard from the host before declaring a scheduling transform unprofitable
- lever: when a latency-hiding or phase-split transform measures as a loss, ask what the surrounding bounds guard forces the compiler to materialise before writing the transform off; then delete the guard by constructing the launch so its predicate is compile-time dead
- apply: pass a true extent plus a compile-time trip count from the host and specialise an exact-tile instantiation; buffer addressing returns 0 on an out-of-range load and drops an out-of-range store, so the guard can go with no device edit and no tail launch. For shape generality ship a peeled full-tile body plus a guarded remainder tail; the safe host predicate is grid x block == vector count, which is stricter than plain divisibility once the grid is capped
- verify: diff register and scratch counts plus the ISA between the guarded and guardless instantiations, and route the incumbent's own parameters through the new host path as an identity control arm so the host rewrite itself is priced at zero on shapes that should not move
- pitfall: the phase-split used two fewer VGPRs and zero scratch yet still lost → the guard forced a zero-fill/phi on the hoisted loads → control flow, not register pressure, was the blocker; the unsplit form's higher VGPR count was the tell that it had already been software-pipelined
19 reps read as valid but had tested unchanged source → applying a patch from a scratch directory walked up to the enclosing repository, exited 0 and printed nothing → gate every arm on a checksum of the built source rather than on the apply command's exit status
identical code read 0.9% apart twenty minutes later on the same box, several times the effect being hunted → cross-session drift → compare only within-session arms, shuffle arm order, discard the first post-build run, and disable implicit rebuilds after one explicit build
- caution: also verify whether the guard was paying a knob-dependent share: its cost scales with per-thread trip count, so an unguarded body can move the optimum of knobs already swept against the guarded one — two grid points flipped sign between the two bodies here
- source: run kernel_20_geak_0811_2h_bmk7_long lane, 2026-08-14, rounds 9-11 with a pooled n=16 shuffled-arm rig, TechLead report + director validation (accepted)
