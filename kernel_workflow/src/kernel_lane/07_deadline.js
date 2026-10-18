// ---------------------------------------------------------------------------
// WALL-CLOCK DEADLINE (opt-in; absent => byte-identical to a build without it).
//   DEADLINE_EPOCH  unix seconds after which NO new optimization round may start.
//   NO_STOP_S       while more than this many seconds remain, the TechLead may not end the run on
//                   its own convergence gates; it gets exactly one forced re-plan.
// In CODE and not in the caller's prose because the prose version was measured: a "fixed 4h window
// per kernel" comparison whose window had no implementation anywhere realised 1.55h to 15.22h, a
// 9.81x spread, and the comparison it existed to make was void. The loop below is
// `while (dispatched < BUDGET && noImprove < MAX_NO_IMPROVE)` — no time term. Emphatic prompt text
// is not a constraint, it is one more input to a judgement call.
// A workflow script has no clock (Date.now() is unavailable — it would break resume), so the time is
// read by a tiny low-effort agent. If that read FAILS the run is NOT killed: an unreadable clock
// returns Infinity, degrading to the old budget-only behaviour rather than truncating a healthy run.
const DEADLINE_EPOCH = (() => {
  const v = parseInt(A.deadline_epoch != null ? A.deadline_epoch : 0, 10);
  return Number.isFinite(v) && v > 0 ? v : 0;
})();
// Opt-in process cutoffs retain the original deadline and reserve later stages.
const SEARCH_RESERVE_S = A.search_reserve_s == null ? 0 : Number(A.search_reserve_s);
const FINAL_VALIDATION_RESERVE_S = Number(A.final_validation_reserve_s || 0);
const FINAL_CLEANUP_RESERVE_S = Number(A.final_cleanup_reserve_s || 0);
const PROCESS_BOUNDED = A.search_reserve_s != null;
if (PROCESS_BOUNDED) {
  if (!DEADLINE_EPOCH || !Number.isFinite(SEARCH_RESERVE_S) || SEARCH_RESERVE_S <= 0
      || !Number.isFinite(FINAL_VALIDATION_RESERVE_S) || FINAL_VALIDATION_RESERVE_S < 0
      || !Number.isFinite(FINAL_CLEANUP_RESERVE_S) || FINAL_CLEANUP_RESERVE_S < 0
      || FINAL_CLEANUP_RESERVE_S > FINAL_VALIDATION_RESERVE_S
      || FINAL_VALIDATION_RESERVE_S > SEARCH_RESERVE_S
      || typeof setProcessDeadline !== 'function' || !agent.processDeadlineActive) {
    throw new Error('search reserve requires the process-bounded runtime and ordered final reserves');
  }
  const optimizerReserve = Math.max(0, parseInt(A.optimizer_reserve_s != null ? A.optimizer_reserve_s : 600, 10));
  // Importing the optimizer's committed result must have time before Report/Validate.
  if (String(A.optimizer || 'greedy') !== 'greedy'
      && (!Number.isFinite(optimizerReserve) || optimizerReserve <= FINAL_VALIDATION_RESERVE_S + 60)) {
    throw new Error('optimizer reserve must exceed the final validation reserve plus 60 seconds for result import');
  }
  setProcessDeadline(DEADLINE_EPOCH - SEARCH_RESERVE_S);
}
const NO_STOP_S = Math.max(0, parseInt(A.no_stop_s != null ? A.no_stop_s : 900, 10));
// Run-level cap on how many TechLead stops may be refused (forcedReplans is not reset per round).
// Not 1: the refusal must scale with the window, or a long budget is handed back early.
// Not unbounded: a role that truly cannot name a direction would spend the window planning.
const MAX_FORCED_REPLANS = Math.max(1, parseInt(A.max_forced_replans != null ? A.max_forced_replans : 6, 10));
const CLOCK_SCHEMA = { type: 'object', properties: { epoch: { type: 'number' } },
                       required: ['epoch'], additionalProperties: true };
let deadlineHit = false;
let forcedReplans = 0;

async function secondsLeft(tag) {
  if (PROCESS_BOUNDED) return setProcessDeadline.secondsLeft();
  if (!DEADLINE_EPOCH) return Infinity;
  const r = await agentT(
    `Run EXACTLY this command and nothing else:
\`\`\`bash
date +%s
\`\`\`
Return {"epoch": <the integer it printed>}. Do NOT modify any file and do NOT run anything else.`,
    { phase: 'Optimize', label: `clock ${tag}`, effort: 'low', schema: CLOCK_SCHEMA });
  if (!r || !Number.isFinite(r.epoch)) {
    log(`  [deadline] clock read failed at ${tag} — treating time as unlimited for this check.`);
    return Infinity;
  }
  return DEADLINE_EPOCH - r.epoch;
}

// Expert-skills injection. PURELY ADDITIVE: '' when OFF or the role is not a skills consumer, so both
// call sites (roleAgent, and the inline Optimize prompt) are byte-identical to the pre-feature build in
// those cases. When ON, appends an advisory pointer telling the agent to Read the fragment + query the
// skills index (scripts have no fs access).
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op (operator/dtype/regime, and ` +
    `from_backend->to_backend for migration skills), whose \`scope\` is \`kernel\` (\`tuning\` entries ` +
    `belong to the e2e tuning phase and match every operator), and whose validation_status is ` +
    `\`validated\`, and ` +
    `treat each as a HIGH-PRIOR candidate to reproduce — advisory only, never overriding your isolated ` +
    `A/B vs the oracle, never reducing a result below the measured baseline.`;
}
