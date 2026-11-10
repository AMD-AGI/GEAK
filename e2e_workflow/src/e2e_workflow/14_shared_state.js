// Shared state for the head + kernel tracks (carried across phase invocations via args.state);
// declared before the HeadKernel block that uses it (temporal dead zone).
let dispatched = 0;                        // counts ONLY kernel-optimization tasks (the budget)
let milestone = 0;
let noImprove = 0;
// kbSeedKernels: stored kernels Module A replayed and this run's own A/B accepted.
const acceptedKernels = (ST.accepted_kernels || []).slice().concat(kbSeedKernels);
const acceptedHeads = (ST.accepted_heads || []).slice();
// Verified-isolated wins whose e2e A/B did not complete. Not rejections: Finalize finishes them
// (Fix C) and the return surfaces them as pending_integrations.
const pendingIntegrations = (ST.pending_integrations || []).slice();
const flaggedHeads = (ST.flagged_heads || []).concat(PRE_FLAGGED_HEADS);   // heads that could NOT be optimized (loudly surfaced, never silently skipped)
let headDispatched = 0;
const history = ST.history || { insights: [], ledger: [], milestones: [], bottleneck_now: '', suggest_next: '' };

// A fused op (op_kind='moe', set by the op-identity guard OR the Architect) is extracted AS the fused op,
// never decomposed into a standalone dense GEMM — so dense-GEMM synth is off for it.
function gemmSynthFor(h) { return (h && h.op_kind === 'moe') ? 'false' : GEMM_SYNTH; }

