// ---------------------------------------------------------------------------
// Shared state carried across the head + kernel tracks (and across phase invocations via args.state).
// MUST be declared BEFORE the HeadKernel block that uses them (else temporal-dead-zone ReferenceError).
// ---------------------------------------------------------------------------
let dispatched = 0;                        // counts ONLY kernel-optimization tasks (the budget)
let milestone = 0;
let noImprove = 0;
// kbSeedKernels is empty unless Module A REPLAYED a stored kernel and its fresh A/B accepted it —
// these are this run's own measurements of a recovered patch, not the store's claims about it.
const acceptedKernels = (ST.accepted_kernels || []).slice().concat(kbSeedKernels);
const acceptedHeads = (ST.accepted_heads || []).slice();
// Verified-isolated wins whose e2e A/B did NOT complete (integrate agent timed
// out / hung / degraded to null mid-gate). These are NOT rejections — keep them
// so Finalize can finish the best one's A/B (Fix C) and so a real isolated win
// is surfaced (return.pending_integrations) instead of being silently dropped.
const pendingIntegrations = (ST.pending_integrations || []).slice();
const flaggedHeads = (ST.flagged_heads || []).concat(PRE_FLAGGED_HEADS);   // heads that could NOT be optimized (loudly surfaced, never silently skipped)
let headDispatched = 0;
const history = ST.history || { insights: [], ledger: [], milestones: [], bottleneck_now: '', suggest_next: '' };

// A fused op (op_kind='moe', set by the op-identity guard OR the Architect) is extracted AS the fused op,
// never decomposed into a standalone dense GEMM — so dense-GEMM synth is off for it.
function gemmSynthFor(h) { return (h && h.op_kind === 'moe') ? 'false' : GEMM_SYNTH; }

