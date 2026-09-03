---
key: GPU-side tuning of a tiny dispatch-bound index-scatter on gfx950, where the copy is already coalesced and the operand set stays cache-resident
type: anti-pattern
confidence: ★★
effect: 0.973x vectorized scan, 1.000x launch-meta sweep, 0.998x larger BLOCK_SIZE, ~1.00x native submit shim and ~1.00x for a doorbell/persistent variant; flat across all three batch cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 7
toolchain: unknown
last_seen: 2026-08-12
name: gpu-side-knobs-are-a-closed-axis-once-submit-dominates-memory-movement-gfx950-launch-bound
description: Closed axis: on a dispatch-bound tiny copy, four GPU-side and three extra host-submit directions all returned ~1.00x; only the raw launch path moved
keywords: ['anti-pattern', 'launch-overhead', 'dispatch-bound', 'tiny-kernel', 'memory-movement', 'block-size', 'num-warps', 'host-submit']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: memory_movement
regime: launch-bound
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-07-30
origin_kernels: ['write_req_to_token_pool_triton']
---
# GPU-side knobs are a closed axis once submit dominates
- lever: Cheap first check on a tiny op: compare its measured window against an empty launch bracket; when the two are close, occupancy/tile/warp knobs have little room and the round is better spent on the submit path.
- apply: Directions that returned ~1.00x here: widening BLOCK_SIZE, a num_warps/num_stages sweep, vectorizing the serial per-program scan, a native pybind submit shim, and a persistent/doorbell resident kernel.
- verify: Score each as a true negative only against the same frozen baseline in the same session; an unchanged number across every batch case is the signature of a closed axis rather than a bad implementation.
- pitfall: The vectorized masked scan measured 0.973x -> it added arithmetic while the serial loads were already cache-resident, so bandwidth was never the constraint -> what paid was removing the dependency chain, not widening the loads.
- caution: The axis read as closed at small batch on this arch; also verify at larger batch, where a per-program dependent chain grows and can return to the critical path.
- source: run write_req_to_token_pool_triton-ch16h, 2026-07-30, ledger of 7 scored directions
