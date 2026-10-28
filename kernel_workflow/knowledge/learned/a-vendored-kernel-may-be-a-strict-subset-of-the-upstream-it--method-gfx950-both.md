---
key: recovering tuning axes that a task's vendored or flattened kernel copy deleted from its upstream original, before declaring any axis closed
type: lever
confidence: ★★
effect: The task's vendored stage-1 had dropped num_waves, k_wave, prefetch_hidden and the 64/128 tile widths and hard-asserted a single tile size; recovering ONE of those axes was worth weighted 1.1294 -> 1.1502 across the decode buckets on its own, and the campaign that started from the diff reached 1.3205 (director 3-run median; 1.3182/1.3205/1.3284, span 2.5% -- read it as ~1.32x, not to 4 digits) against a prior ceiling priced at ~7% over 1.1294.
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: flydsl 0.3.4.1 / aiter@4faf8713 / ROCm 7.2
last_seen: 2026-10-12
name: a-vendored-kernel-may-be-a-strict-subset-of-the-upstream-it--method-gfx950-both
description: Before calling an axis closed, diff the vendored kernel's signature against upstream: an assert that deletes a knob reads exactly like the knob being inert.
keywords: ['method', 'vendored-source', 'upstream-diff', 'api-surface', 'closed-axis', 'anti-pattern', 'tuning-knobs', 'campaign-brief', 'flydsl', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: both
language: flydsl
decision_refs: []
layer: learned
lifecycle: active
origin_kernels: ['glm52_mxfp4_fused_moe']
cost: L0
verified_on: 2026-10-12
---
# A vendored kernel may be a strict subset of the upstream it was flattened from
- lever: Diff the vendored/flattened kernel's function signature against the same function in the upstream library and enumerate the parameters the candidate never passes; those are axes nobody has measured, not axes that were measured and closed.
- apply: Put the signature diff in the campaign brief as the FIRST section, naming each recovered knob. A compat checker that validates the symbols a candidate already uses cannot surface a knob the candidate does not use.
- stack: Pairs with authorising a new kernel rather than editing the given one; an edit-this-file loop will not reach for a parameter the file does not mention.
- verify: Resolve a handful of upstream citations by line before trusting the diff, and confirm which tree you are in -- a box carrying two checkouts of the same library makes every citation fail silently in the wrong one.
- pitfall: Some deletions are real contracts, not simplifications -> a scale-layout parameter documented as a GEMM1/GEMM2 contract must stay consistent across both stages -> read the upstream docstring before widening a whitelist.
Line numbers do NOT survive flattening -> a citation that will not resolve means the wrong tree or the wrong copy, not a typo.
- caution: Closed-axis lists and cost ceilings are scoped to the skeleton they were measured in; carried into the next brief as properties of the problem they delete the structural search space, which is how the ~7% ceiling above came to be off by ~10pp (predicted +7%, delivered +16.9%).
- source: run wwd7t6625 / team_volck_A_20261012_082749, 2026-10-12, gfx950/MI355X; diff of the task's fused_moe_seam.py against aiter@4faf8713 ops/flydsl/kernels/mxfp4_gemm1.py
