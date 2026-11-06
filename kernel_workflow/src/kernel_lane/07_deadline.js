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
const OUTER_DEADLINE_EPOCH = (() => {
  const v = parseInt(A.deadline_epoch != null ? A.deadline_epoch : 0, 10);
  return Number.isFinite(v) && v > 0 ? v : 0;
})();
let DEADLINE_EPOCH = OUTER_DEADLINE_EPOCH;
// Opt-in process cutoffs retain the original deadline and reserve later stages.
const SEARCH_RESERVE_S = A.search_reserve_s == null ? 0 : Number(A.search_reserve_s);
const FINAL_VALIDATION_RESERVE_S = Number(A.final_validation_reserve_s || 0);
const FINAL_CLEANUP_RESERVE_S = Number(A.final_cleanup_reserve_s || 0);
const PROCESS_BOUNDED = A.search_reserve_s != null;
const SEARCH_DURATION_S = A.search_duration_s ?? 0;
const PREPARATION_ALLOWANCE_S = A.preparation_allowance_s ?? 0;
const RELATIVE_SEARCH = A.search_duration_s != null || A.preparation_allowance_s != null;
let preparationStartedEpoch = null, preparationCutoffEpoch = null, searchStartedEpoch = null;
if (RELATIVE_SEARCH && (!PROCESS_BOUNDED
    || !Number.isSafeInteger(SEARCH_DURATION_S) || SEARCH_DURATION_S <= 0
    || !Number.isSafeInteger(PREPARATION_ALLOWANCE_S) || PREPARATION_ALLOWANCE_S <= 0)) {
  throw new Error('preparation_allowance_s and search_duration_s must both be positive safe integers with a process-bounded outer deadline');
}
if (PROCESS_BOUNDED) {
  if (!DEADLINE_EPOCH || !Number.isFinite(SEARCH_RESERVE_S) || SEARCH_RESERVE_S <= 0
      || !Number.isFinite(FINAL_VALIDATION_RESERVE_S) || FINAL_VALIDATION_RESERVE_S < 0
      || !Number.isFinite(FINAL_CLEANUP_RESERVE_S) || FINAL_CLEANUP_RESERVE_S < 0
      || FINAL_CLEANUP_RESERVE_S > FINAL_VALIDATION_RESERVE_S
      || FINAL_VALIDATION_RESERVE_S > SEARCH_RESERVE_S
      || typeof setProcessDeadline !== 'function' || !agent.processDeadlineActive) {
    throw new Error('search reserve requires the process-bounded runtime and ordered final reserves');
  }
  if (RELATIVE_SEARCH) {
    // At entry the existing runtime deadline is its immutable outer maximum.
    const now = OUTER_DEADLINE_EPOCH - FINAL_CLEANUP_RESERVE_S - setProcessDeadline.secondsLeft();
    preparationStartedEpoch = now;
    preparationCutoffEpoch = Math.min(now + PREPARATION_ALLOWANCE_S,
      OUTER_DEADLINE_EPOCH - SEARCH_RESERVE_S - SEARCH_DURATION_S);
    if (!(preparationCutoffEpoch > now + 7)) {
      throw new Error('GEAK_NATIVE_DELIVERY_INCOMPLETE: search not admitted; full search and closure do not fit within the outer deadline');
    }
    setProcessDeadline(preparationCutoffEpoch);
  } else setProcessDeadline(DEADLINE_EPOCH - SEARCH_RESERVE_S);
}
// DELIVERY FACTS (opt-in; absent => unchanged prompts and control flow). Engineering and the round's
// verify, integrate and commit share the search cutoff. Planner and engineer prompts get observed facts
// only: no derived reserve, no prior and no bound. A role wall is host time from dispatch to return,
// including lease, model and semaphore waits. It counts as completed when the role-returned verdict
// matches COMPLETED_ROLE; every other return is listed as not completed. Both are recorded before the
// host's external-measurement check, EVO retention and post-commit source-tree check, so they are not
// host-validated completion or accepted delivery, and they exclude validation, retention and recovery.
if (A.delivery_facts != null && typeof A.delivery_facts !== 'boolean') throw new Error('delivery_facts must be a boolean');
const DELIVERY_FACTS = A.delivery_facts === true;
if (DELIVERY_FACTS && !PROCESS_BOUNDED) throw new Error('delivery_facts requires the search reserve');
let SEARCH_CUTOFF_EPOCH = DEADLINE_EPOCH - SEARCH_RESERVE_S;
const LANE_START_EPOCH = DELIVERY_FACTS
  ? Math.floor((RELATIVE_SEARCH ? preparationCutoffEpoch : SEARCH_CUTOFF_EPOCH) - setProcessDeadline.secondsLeft()) : 0;
const COMPLETED_ROLE = {
  verify: r => says(r.status, 'verified') || says(r.status, 'regression'),
  integrate: r => r.attempted === true && (says(r.conclusion, 'improved') || says(r.conclusion, 'no_improvement')),
  commit: r => r.committed === true,
};
const ROLE_WALLS_S = { verify: [], integrate: [], commit: [] }, NOT_COMPLETED = [];
async function observeRole(kind, label, call) {
  if (!DELIVERY_FACTS) return call();
  const started = setProcessDeadline.secondsLeft();
  let result = null;
  try { return (result = await call()); } finally {
    const left = setProcessDeadline.secondsLeft(), wall_s = Math.round(started - left);
    if (result && COMPLETED_ROLE[kind](result)) ROLE_WALLS_S[kind].push(wall_s);
    else NOT_COMPLETED.push({ kind, label, wall_s, outcome: result
      ? String(result.status ?? result.conclusion ?? result.committed)
      : left <= 7 ? 'no result at the search cutoff' : 'no result (timed out or failed)' });
  }
}
async function deliveryFacts() {
  let rows = '';
  try { rows = await hostRun('sh', ['-c', 'if [ -r "${GEAK_GPU_USE_LOG:-}" ]; then cat -- "$GEAK_GPU_USE_LOG"; fi']) || ''; }
  catch (error) { rows = ''; }
  const now = SEARCH_CUTOFF_EPOCH - setProcessDeadline.secondsLeft();   // after the host read
  const waits = rows.split('\n').map(line => { try { return JSON.parse(line); } catch (error) { return null; } })
    .filter(row => row && row.t >= LANE_START_EPOCH && Number.isFinite(row.wait_s)).map(row => row.wait_s);
  return { as_of_epoch: Math.floor(now), search_cutoff_epoch: SEARCH_CUTOFF_EPOCH,
    seconds_to_search_cutoff: Math.floor(SEARCH_CUTOFF_EPOCH - now),
    // ENGINEER_RESERVE_S is defined below; this runs only during Optimize, after it is set.
    ...(ENGINEER_RESERVE_S ? { engineer_cutoff_epoch: ENGINEER_CUTOFF_EPOCH,
      seconds_to_engineer_cutoff: Math.floor(ENGINEER_CUTOFF_EPOCH - now),
      shared_cutoff: 'Planner and engineer calls end at engineer_cutoff_epoch; this round\'s verify, integrate ' +
        'and commit continue until search_cutoff_epoch; a candidate is verified only after its engineer ' +
        'returns or is stopped. Report and Validate have later cutoffs.' }
      : { shared_cutoff: 'Engineering and this round\'s verify, integrate and commit all end at search_cutoff_epoch; ' +
        'a candidate is verified only after its engineer returns. Report and Validate have later cutoffs.' }),
    observed_role_walls_s: { meaning: 'observed role wall from dispatch to the role\'s return, includes ' +
        'lease/model/semaphore waits; excludes host validation, retention and recovery; not a service cost',
      ...Object.fromEntries(Object.entries(ROLE_WALLS_S).map(([kind, walls]) => [kind, walls.length
        ? { count: walls.length, walls_s: [...walls].sort((a, b) => a - b) }
        : 'unknown: no completed call observed in this run'])) },
    completed_means: 'role-returned verdict, not host-validated completion or accepted delivery: verify ' +
      'returned verified|regression; integrate attempted with improved|no_improvement; commit committed=true',
    not_completed: NOT_COMPLETED.length ? NOT_COMPLETED : 'none observed',
    gpu_lease_waits: waits.length ? { meaning: 'GEAK_GPU_USE_LOG rows with t at or after this lane start: one row ' +
        'per successful lease acquisition after lock/idle settling, one per device for set leases; pending or ' +
        'failed acquisitions are absent; overlapping waits, not hold time; assumes a lane-dedicated log and ' +
        'comparable clocks; time filtering does not identify the command or role',
      rows: waits.length, positive: waits.filter(w => w > 0).length,
      sum_s: waits.reduce((a, b) => a + b, 0), max_s: Math.max(...waits) }
      : 'unknown: no GEAK_GPU_USE_LOG rows readable for this run',
    complete_ut_wall: 'unknown: no per-command hold time is recorded' };
}
// Common assigned-pool facts at each dispatch; observed delivery facts remain opt-in.
const evaluationServiceFacts = () => ({
  assigned_gpu_ids: GPU_LIST, lock_argument: GPU_POOL, mode: GPU_MODE,
  required_gpus_per_evaluation: REQUIRED_GPU_COUNT,
  meaning: 'Configured assigned evaluation pool, not an observation of live availability or throughput. ' +
    (GPU_LIST.length === 1 ? 'All GPU evaluations in this lane serialize through the one assigned device lock. ' :
      'Evaluations acquire the existing device/group locks. ') +
    'Model concurrency is not GPU evaluation concurrency; directions meet at the existing round verification/integration barrier.',
  clocks: 'Role timeouts are computed before runtime semaphore acquisition; queue, model and GPU lease waits consume wall time. ' +
    'Delivery facts report qualified observations only; missing waits and complete evaluation service time remain unknown.'
});
const deliveryFactsInput = async () => ({EVALUATION_SERVICE: evaluationServiceFacts(),
  ...(DELIVERY_FACTS ? {DELIVERY_FACTS: await deliveryFacts()} : {})});
// ENGINEER RESERVE (opt-in; absent => unchanged). A declared budget allocation, not a cost estimate.
// Planner and engineer calls carry the earlier per-call deadline search cutoff - engineer_reserve_s;
// verify, integrate and commit keep the shared search cutoff, and Report/Validate keep their reserves.
// The shared process deadline is never moved for this. No round or replan starts after the cutoff.
const ENGINEER_RESERVE_S = A.engineer_reserve_s == null ? 0 : A.engineer_reserve_s;
if (A.engineer_reserve_s != null && (!PROCESS_BOUNDED || !Number.isSafeInteger(ENGINEER_RESERVE_S)
    || ENGINEER_RESERVE_S < 0)) {
  throw new Error('engineer_reserve_s must be a nonnegative integer and requires the search reserve');
}
let ENGINEER_CUTOFF_EPOCH = ENGINEER_RESERVE_S ? SEARCH_CUTOFF_EPOCH - ENGINEER_RESERVE_S : null;
const ENGINEER_CALL = ENGINEER_RESERVE_S ? { processDeadlineEpoch: ENGINEER_CUTOFF_EPOCH } : {};
// Called once, after baseline/EVO initialization and before the first search plan.
function beginSearchWindow() {
  if (!RELATIVE_SEARCH) return null;
  const now = preparationCutoffEpoch - setProcessDeadline.secondsLeft();
  if (searchStartedEpoch !== null || now >= preparationCutoffEpoch
      || now + SEARCH_DURATION_S + SEARCH_RESERVE_S > OUTER_DEADLINE_EPOCH) {
    throw nativeDeliveryError('search not admitted; preparation exhausted the full-search fit');
  }
  searchStartedEpoch = now;
  SEARCH_CUTOFF_EPOCH = now + SEARCH_DURATION_S;
  DEADLINE_EPOCH = SEARCH_CUTOFF_EPOCH + SEARCH_RESERVE_S;
  ENGINEER_CUTOFF_EPOCH = ENGINEER_RESERVE_S ? SEARCH_CUTOFF_EPOCH - ENGINEER_RESERVE_S : null;
  if (ENGINEER_RESERVE_S) ENGINEER_CALL.processDeadlineEpoch = ENGINEER_CUTOFF_EPOCH;
  const timing = {outer_deadline_epoch: OUTER_DEADLINE_EPOCH,
    preparation_started_epoch: preparationStartedEpoch, preparation_cutoff_epoch: preparationCutoffEpoch,
    search_window_state: 'started', search_started_epoch: now, search_duration_s: SEARCH_DURATION_S,
    full_deadline_epoch: DEADLINE_EPOCH, search_cutoff_epoch: SEARCH_CUTOFF_EPOCH,
    report_cutoff_epoch: DEADLINE_EPOCH - FINAL_VALIDATION_RESERVE_S,
    validation_cutoff_epoch: DEADLINE_EPOCH - FINAL_CLEANUP_RESERVE_S};
  setProcessDeadline(SEARCH_CUTOFF_EPOCH, timing);
  return timing;
}
// True when no agent could start before the engineer cutoff (the runtime's 7 s floor).
const engineerCutoffReached = (left) => ENGINEER_RESERVE_S > 0 && left - ENGINEER_RESERVE_S <= 7;
// Optional verification time for unverified retained sources. Custody, selection of already qualified
// sources and host export do not depend on this allocation. No cutoff moves.
const DELIVERY_RESERVE_S = A.delivery_reserve_s == null ? 0 : A.delivery_reserve_s;
if (A.delivery_reserve_s != null && (!PROCESS_BOUNDED || EXTERNAL_OBJECTIVE || !Number.isSafeInteger(DELIVERY_RESERVE_S)
    || DELIVERY_RESERVE_S < 0 || DELIVERY_RESERVE_S >= SEARCH_RESERVE_S - FINAL_VALIDATION_RESERVE_S)) {
  throw new Error('delivery_reserve_s must be a nonnegative integer inside the Report allowance of a process-bounded, agent-measured lane');
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
