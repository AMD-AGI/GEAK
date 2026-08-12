---
key: Graph capture as a launch-overhead lever for a single small routing kernel on gfx950, where there is no launch sequence to amortize
type: anti-pattern
confidence: ★★
effect: 1.00x (clean measured negative, no patch banked): captured replay lands at ~1.4-1.7x the cost of the direct launch it replaces; no gain on any case, and the low-row cases where launch overhead dominates are exactly the ones that regress. Two capture layers (launcher and wrapper) both measured.
confirms_cited: 1
confirms_blind: 0
losses: 1
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: graph-capture-does-not-pay-for-one-small-dispatch-moe-router-topk-gfx950-decode
description: Anti-pattern: graph capture/replay around a single tiny kernel is slower than a direct launch; replay dispatch alone exceeds the whole launch it replaces.
keywords: ['graph-capture', 'launch-overhead', 'dispatch-bound', 'anti-pattern', 'triton', 'decode']
kernels: ['_topk_forward']
platforms: ['gfx950']
kernel_class: moe_router_topk
regime: decode
layer: learned
lifecycle: active
---
# Graph capture does not pay for one small dispatch
- lever: Treat graph capture as a lever for a SEQUENCE of launches; for a single small kernel the replay path's own command-processor dispatch is on the same order as the launch being removed, so budget one round to measure it and move on.
- apply: L2: capture the wrapper's single launch and replay it in steady state; the campaign tried this at both the low-level launcher and the wrapper layer.
- verify: Time replay against direct launch on the smallest case, where dispatch is the largest share; if replay is not clearly under the direct path there, it never will be on larger cases.
- pitfall: A capture layer can be inert (silently not engaging) and read as 1.00x -> distinguish 'replay was slower' from 'replay never ran' by checking the reverted workspace is bit-identical and by timing replay directly, not through the geomean.
- caution: Also verify whether several launches can be batched into one capture before concluding the direction is closed; the negative here is about a lone dispatch, not about capture in general.
- source: 16h per-kernel time-budget campaign, lane chuschen16h, direction verdict dead_end, 2026-08-11
