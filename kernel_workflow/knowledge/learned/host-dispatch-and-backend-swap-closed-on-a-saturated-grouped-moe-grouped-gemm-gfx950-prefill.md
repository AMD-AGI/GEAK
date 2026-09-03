---
key: host-side dispatch, work-ordering and vendor-backend substitution on a compute-saturated fp8 grouped GEMM, gfx950/MI355X under ROCm/Triton
type: anti-pattern
confidence: ★★
effect: ~1.00x or worse per-case on four host-side directions: wrapper-level graph capture was slower on all three cases (+9.7% / +0.85% / +0.42% time); persistent/split-N lost -3.0% at 2 tiles per program and -7.3% at 4; a joint GROUP_SIZE_M x NUM_XCD sweep peaked at +0.23%, under its own 0.4% noise floor; the best vendor dense fp8 block-scale GEMM prices as a 1.26x loss against the incumbent on the full grouped problem
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill
description: gfx950 grouped GEMM: graph capture, persistent/split-N, GSMxXCD sweeps and a vendor dense-GEMM swap all priced at ~1.00x or a loss.
keywords: ['moe-grouped-gemm', 'gfx950', 'launch-overhead', 'hip-graph', 'xcd-remap', 'persistent-kernel', 'aiter', 'anti-pattern', 'paired-ab-rig']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: prefill
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
origin_kernels: ['fused_moe_kernel']
---
# Host dispatch and backend swap closed on a saturated grouped GEMM
- lever: Price the host-side axis before opening it: compare enqueue cost against the SHORTEST case's GPU time, and price a vendor kernel on the grouped problem you actually have rather than on the dense shape its benchmark reports.
- apply: Use an in-process paired rig that imports the harness's own input builders to measure the enqueue floor; for a vendor entry, check its activation contract first (activation-baked / fused-gate-by-construction entries can silently no-op at runtime) and compare achieved fraction of peak on matching shapes.
- verify: Plant a known-null arm in every sweep and md5 the compiled artifact before spending GPU time; a sweep whose best point sits inside its own noise floor is a closed axis, not a small win.
- pitfall: Graph capture was expected to remove enqueue and measured slower on every case -> on this stack a graph node's GPU-side dispatch costs more than a live dispatch -> the enqueue was already fully hidden inside even the shortest case's GPU time.
A vendor entry appeared to run and changed nothing -> the runtime-activation entry silently no-ops -> probe the outputs, not the return code.
- caution: Also verify whether a pid remap has bandwidth headroom to convert: ungated it cost -2.2% on the largest case, yet re-tested once the kernel became memory-bound and gated to a wide-tile predicate the same remap was half of a +3.13% win, so bank a refuted null with its re-test trigger and the contract it was refuted under.
- source: run fused_moe_kernel-own16h, 2026-08-12, rounds 2/5/6 nulls on a director-validated run
