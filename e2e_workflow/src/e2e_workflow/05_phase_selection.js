// Phase-scoped driving (so a long run survives host-session compaction): args.phases = subset of
// {setup,config,tune,head,kernel,final}, default 'all'. State crosses phases via the return value (no
// fs): pass the prior return back as args.state. Unrequested phases load carried state.
const PHASES = String(A.phases || 'all').split(',').map(s => s.trim()).filter(Boolean);
const RUN_ALL = PHASES.includes('all');
// Fast mode skips ConfigSweep, TuningSkillset and Milestone: everything comes from HeadKernel within
// the wall-clock budget.
const FAST_SKIP = FAST_MODE ? new Set(['config', 'tune', 'kernel']) : null;
// Deep mode skips only Milestone; ConfigSweep and TuningSkillset are cheap and give co-opt a better
// starting stack.
const DEEP_SKIP = DEEP_MODE ? new Set(['kernel']) : null;
// AgentX prepares UTs once, then optimizes them against the same serving baseline.
const AGENTX_SKIP = KERNEL_EXTRACT_ON ? new Set(['profile', 'config', 'tune']) : null;
const want = (p) => (RUN_ALL || PHASES.includes(p)) && !(FAST_SKIP && FAST_SKIP.has(p)) && !(DEEP_SKIP && DEEP_SKIP.has(p)) && !(AGENTX_SKIP && AGENTX_SKIP.has(p));
const ST = A.state || {};   // carried state from a prior phase invocation
// Hoisted: WarmStart's integrate legs reach tuningIntegrateInputs() before the TuningSkillset phase
// runs, so `tuning` must already be declared; that phase reassigns it.
let tuning = ST.tuning || null;
if (FAST_MODE) log(`[fast-mode] ON: skipping ConfigSweep + Milestone; HeadKernel-only; budget ${Math.round(FAST_BUDGET_MS / 60000)}min (stop new heads at ${Math.round(FAST_HEAD_DEADLINE_MS / 60000)}min, per-head workflow cap ${Math.round(FAST_HEAD_WF_MS / 60000)}min).`);

