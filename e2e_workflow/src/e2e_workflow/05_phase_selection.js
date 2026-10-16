// ---------------------------------------------------------------------------
// Phase-scoped driving (robustness): a long single background run can be orphaned if the host session
// context compacts mid-run. To avoid that, the orchestration can be driven phase-by-phase: invoke with
// args.phases = subset of {setup,config,tune,head,kernel,final} (default 'all' = run everything in one go).
// Cross-phase state flows through the RETURN value (the script has no fs); pass the prior return back as
// args.state for the next phase. Each phase only RUNS if requested; otherwise it loads carried state.
// ---------------------------------------------------------------------------
const PHASES = String(A.phases || 'all').split(',').map(s => s.trim()).filter(Boolean);
const RUN_ALL = PHASES.includes('all');
// Fast mode SKIPS ConfigSweep ('config'), the tuning skillset ('tune') and the editable-kernel Milestone
// ('kernel') so all optimization comes from HeadKernel within the wall-clock budget. ('tune' joins the
// skip set for the same reason 'config' does: fast mode's contract is "best head-kernel wins in N hours",
// and the tuning loop's tuner sweeps + its own pre/post A/B do not fit that budget.) Default mode:
// FAST_SKIP is null → want() is the original `RUN_ALL || PHASES.includes(p)`, unchanged.
const FAST_SKIP = FAST_MODE ? new Set(['config', 'tune', 'kernel']) : null;
// Deep mode concentrates its (20h) HeadKernel budget on cross-backend co-opt: skip the editable-kernel
// Milestone ('kernel') but KEEP ConfigSweep ('config' — cheap and it stabilizes the baseline) and KEEP
// the tuning skillset ('tune' — same rationale, and a tuned stack is a strictly better starting point
// for the co-opt lanes). Null in every other mode, so normal + fast are unchanged.
const DEEP_SKIP = DEEP_MODE ? new Set(['kernel']) : null;
const want = (p) => (RUN_ALL || PHASES.includes(p)) && !(FAST_SKIP && FAST_SKIP.has(p)) && !(DEEP_SKIP && DEEP_SKIP.has(p));
const ST = A.state || {};   // carried state from a prior phase invocation
// Hoisted: tuningIntegrateInputs() is reachable from runIntegrateBothLegs, which WarmStart drives
// long before the TuningSkillset phase body — declaring `tuning` there left it in the temporal dead
// zone on the first integrate leg. The tuning phase later reassigns it.
let tuning = ST.tuning || null;
if (FAST_MODE) log(`[fast-mode] ON: skipping ConfigSweep + Milestone; HeadKernel-only; budget ${Math.round(FAST_BUDGET_MS / 60000)}min (stop new heads at ${Math.round(FAST_HEAD_DEADLINE_MS / 60000)}min, per-head workflow cap ${Math.round(FAST_HEAD_WF_MS / 60000)}min).`);

