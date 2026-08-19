---
key: bf16 MoE-router top-k select + softmax, small row counts, dispatch-floored · gfx950 · Triton 3.6
type: lever
confidence: ★★
effect: 2.12x cumulative isolated geomean vs frozen baseline; per-case 2.33x at 2 rows, 2.19x at 32 rows, 1.86x at 64 rows — distributions non-overlapping
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: dispatch-floored-router-select-spend-the-budget-on-the-host--topk-router-gfx950-both
description: Tiny router select whose wall time is flat across a 32x row spread is host-marshaling floored: cached launch closure + steady-state gives 1.9-2.3x per case.
keywords: ['launch-overhead', 'host-runtime', 'dispatch-bound', 'triton', 'small-batch', 'top-k', 'moe-router', 'register-math']
kernels: ['_topk_forward']
platforms: ['gfx950']
kernel_class: topk_router
regime: both
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
origin_kernels: ['_topk_forward']
---
# Dispatch-floored router select: spend the budget on the host lane first
- lever: When per-case time barely moves across a 32x spread in row count, the floor is host per-call marshaling, not the GPU body — attack the launch path before any tile tuning.
- apply: Cache a direct compiled-kernel run closure keyed on data-pointer identity only, plus a trusted steady-state path that skips the grid-runner/fingerprint walk; then stack the device-side items below.
- stack: total 2.12x isolated (weighted, verified) = five directions compounded, incremental in landing order
  - 1. host cached launch closure + trusted steady-state — carries the two small cases (2.33x / 2.19x); the largest case unchanged
  - 2. iterative argmax replacing a full bitonic sort for k=4 over 128 lanes — ~1.11x on the largest case, bit-exact
  - 3. partition-parallel halves + register-local 4-way merge — ~1.185x on the largest case; the win is ILP overlap, not reduced tree depth
  - 4. branchless one-shot bitmask pack (4 where+reduce_or+store collapsed to 1) — ~+2% on the small cases
  - 5. softmax as register math (split into scalar columns, drop the fp32 [M,4] intermediate and 2 cross-lane reduces) — +2.7% cumulative, VGPR 28->24
- verify: Re-time every case against the frozen baseline and confirm the small cases move while the largest one does not — that split is the signature that the host lane, not the body, was the floor.
- pitfall: Flat wall time across all row counts read as memory-bound and sent the first rounds into tile tuning → the cost was per-call host marshaling → a host-side per-call trace separated the two in one round.
- caution: The frozen golden reference may import the same shared helper module as the edit path, so the built-in correctness gate moves with your change and cannot catch a selection bug — also verify any shared-helper edit against an independent oracle harness.
- source: run _topk_forward-ch16h, 2026-08-12 (16h per-kernel time-budget campaign, 50 passes)
