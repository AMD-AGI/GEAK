---
key: register/occupancy dialing via the waves-per-EU compile request on a chunked linear-attention (delta-rule / KKT) kernel on gfx950, all batch cases
type: lever
confidence: ★★
effect: As a probe it is free and decisive: requesting 4/5/6/8 waves moved the compiled register budget deterministically (128/96/80/64) with no source change, and in a few minutes falsified an occupancy hypothesis that had been funded as a whole direction (the wider-warp form crossed the register threshold cleanly with zero spills and was still 9.7% slower on the largest case, 14% on the mid case). As a shipping value it flipped: retired as probe-only on one code shape, it became the elected pin (+2%) on the restructured shape in a later round. End state director-verified 25.55x geomean (3.21x / 47.2x / 110.2x from smallest to largest batch case).
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 9
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-12
name: use-the-waves-per-eu-request-as-a-zero-edit-register-dial-an-linear-attention-gfx950-mixed
description: Sweep waves-per-EU as a zero-edit register dial on linear attention: falsified a funded occupancy direction fast, then shipped as a +2% pin after a rewrite
keywords: ['waves-per-eu', 'vgpr', 'occupancy', 'isa-check', 'config-sweep', 'linear-attention', 'measurement-method']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: mixed
layer: learned
lifecycle: active
---
# Use the waves-per-EU request as a zero-edit register dial, and re-run the ladder after every structural edit
- lever: Before funding a body rewrite whose goal is to save registers, sweep the waves-per-EU request as a compile-time dial and read the resulting register/spill counts and times; and treat any occupancy verdict as scoped to the code shape it was measured on, re-running the same short ladder after each structural edit.
- apply: Sweep the request across its useful range on the unmodified source, read the register and spill counts from the compiled kernel's assembly metadata rather than from a profiler's per-dispatch columns, and record time alongside. Keep the ladder in the round's scripts so it can be replayed on the restructured source at no thinking cost.
- verify: The dial engaged when the reported register budget actually changes across the sweep. A win is real when the fastest rung is also the one with the fewest spills; results here ordered monotonically in spill count and non-monotonically in occupancy, so a rung that gains residency while gaining spills is the hypothesis failing, not noise.
- pitfall: a verdict of 'probe-only, nothing to ship' went stale within the run -> the occupancy verdict was scoped to the code shape it was measured on, and a later structural rewrite changed the register/store balance -> re-run the ladder after every structural edit; the same dial then shipped as the elected pin (+2%).
- caution: Also verify whether the kernel is actually residency-limited before buying more co-resident waves — on a store-queue-bound case more waves made it worse — and also re-check the elected value after any edit that changes the program count or the store structure, since the previous round's verdict can invert.
- source: run kb_on_0810 2026-08-10
