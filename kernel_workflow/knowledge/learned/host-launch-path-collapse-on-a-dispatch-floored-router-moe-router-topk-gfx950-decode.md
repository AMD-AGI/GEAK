---
key: Small bf16 top-k routing/select op on gfx950 under Triton, where per-call host marshaling dominates the GPU body at low row counts
type: lever
confidence: ★★
effect: 2.12x cumulative vs frozen baseline; per-case 2.33x / 2.19x / 1.86x at rising row counts. Host lane alone: 1.65x (cached-closure) then 1.77x (steady state) on the two low-row cases, ~1.0x on the largest case; algorithm+compute lanes added only ~1.07x and ~1.02x on top.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-11
name: host-launch-path-collapse-on-a-dispatch-floored-router-moe-router-topk-gfx950-decode
description: On a tiny dispatch-floored Triton router, collapsing the host launch path (cached compiled-kernel closure + trusted steady state) carries most of the win.
keywords: ['launch-overhead', 'host-runtime', 'dispatch-bound', 'triton', 'topk-router', 'decode']
kernels: ['_topk_forward']
platforms: ['gfx950']
kernel_class: moe_router_topk
regime: decode
layer: learned
lifecycle: active
---
# Host launch-path collapse on a dispatch-floored router
- lever: When baseline wall time is flat across a 32x spread in rows, the op is launch/dispatch-floored: attack the host path first (cache the compiled-kernel run closure keyed on pointer-only identity, and take a trusted steady-state path that skips grid-runner and fingerprint walks) before touching the kernel body.
- apply: L2 wrapper rewrite: memoize the compiled callable once, identity on data pointers only, and bypass the per-call re-derivation of grid/signature after the first launch; body left bit-identical.
- stack: total 2.12x = host lane (dominant, low-row cases) then algorithm partition-parallel select (~1.07x, largest case only, ILP overlap not tree depth) then branchless one-shot pack + register-math softmax (~1.02x and ~+2.7%); attribution is incremental in landing order.
- verify: Confirm the flat-vs-rows signature first, then A/B each case separately against the frozen baseline: the win should appear on the low-row cases and be ~1.0x on the case whose body already fills the GPU, which is what proves it was the host path.
- pitfall: A shared helper module imported by BOTH the edit path and the frozen golden reference makes the correctness gate blind (both sides change together) -> validate any edit to such a shared file against an independent reference harness instead.
- caution: Also verify that the pointer-only identity really is sufficient for the wrapper's argument set; a shape or flag that varies between calls would be silently cached away.
- source: 16h per-kernel time-budget campaign, lane chuschen16h, 50 passes, 2026-08-11
