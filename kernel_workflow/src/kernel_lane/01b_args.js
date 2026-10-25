const OPTIMIZER = String(A.optimizer || 'greedy');
if (!['greedy', 'evo'].includes(OPTIMIZER)) {
  throw new Error(`unknown optimizer: ${OPTIMIZER}; use greedy or evo`);
}
const retiredOptimizerArg = ['optimizer_command', 'optimizer_task_ref', 'optimizer_reserve_s']
  .find(key => Object.prototype.hasOwnProperty.call(A, key));
if (retiredOptimizerArg) {
  throw new Error(`${retiredOptimizerArg} is retired; use optimizer=greedy or optimizer=evo`);
}
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
// Minimum verified geomean improvement over the cumulative best for a round winner to be COMMITTED
// into the canonical workspace (default 2%). Kept as a knob rather than a hard-coded constant so the
// gate is tunable per run (e.g. raise it on a noisy box, lower it to bank small compounding wins).
const MIN_IMPROVE = (() => {
  const v = parseFloat(A.min_improve != null ? A.min_improve : 0.02);
  return Number.isFinite(v) && v >= 0 ? v : 0.02;
})();
// Minimum verified speedup for a candidate to enter the round's candidate list (default 1.0 = only a
// candidate that beats the baseline is worth looking at). A knob for the same reason MIN_IMPROVE is:
// a transcription (plain Triton -> Gluon/TileLang/HIP) lands BELOW the comparator by construction, and
// at 1.0 its recovery phase is invisible -- no patch saved, no verify, `winner` null every round. The
// COMMIT gate is separate and still requires beating `cumulative` by MIN_IMPROVE, so a sub-baseline
// candidate can be TRACKED but never BANKED.
const CANDIDATE_FLOOR = (() => {
  const v = parseFloat(A.candidate_floor != null ? A.candidate_floor : 1.0);
  return Number.isFinite(v) && v > 0 ? v : 1.0;
})();
// Rendered into the Optimize prompt, where `${1.0}` would stringify to "1" and silently reword a
// prompt that has always said "geomean>1.0". Keeps the default run byte-identical.
const CANDIDATE_FLOOR_TXT = Number.isInteger(CANDIDATE_FLOOR)
  ? CANDIDATE_FLOOR.toFixed(1) : String(CANDIDATE_FLOOR);
// How far BELOW the best candidate ever seen a round may land and still count as "the search is
// advancing" (default +MIN_IMPROVE = the historical test). A knob because a climb that starts under
// the comparator advances for many rounds without ever clearing `cumulative`, and scoring those as
// stalls ends it at MAX_NO_IMPROVE however much budget was given -- and no static counter substitutes,
// since it would have to pre-guess how many rounds the climb takes. Negative admits a round that gives
// ground (a layout experiment that does not pay is information, not a stall).
const PROGRESS_DELTA = (() => {
  const v = parseFloat(A.progress_delta != null ? A.progress_delta : MIN_IMPROVE);
  return Number.isFinite(v) && v > -1 ? v : MIN_IMPROVE;
})();
// Budget cost of ONE `deep_explore` direction. The deep-explore engineer does far more than a single
// specialist — broad rewrite authority, its own multi-iteration measure→profile→rewrite loop — so it
// is charged more than 1 against the direction budget (default 2). It also always runs in a DEDICATED
// round (no other directions that round), enforced below.
const DEEP_COST = (() => {
  const v = parseInt(A.deep_cost != null ? A.deep_cost : 2, 10);
  return Number.isFinite(v) && v >= 1 ? v : 2;
})();
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map(s => s.trim()).filter(Boolean);
// Every GPU consumer gets the WHOLE pool rather than a pinned lane. gpu_lock.sh
// resolves a comma spec by flocking whichever lane is free AND idle at acquire
// time, so placement follows what work actually costs instead of an index fixed
// before the cost is known. Measured task costs span 23x on campaign20, and a
// pinned round ends with the slowest LANE, not the slowest task -- at 4-way that
// idles ~43% of lane time. Pinning is also fragile on a shared box: GPU_LIST[0]
// has no fallback when lane 0 has a foreign tenant.
// gpu_mode='pin' restores the pre-scheduler behavior (direction i pinned to GPU_LIST[i % n]).
// It exists ONLY so the scheduler can be A/B'd as a single-variable change against the arm it
// replaced; leaving it out would make the "before" arm unreachable and force the comparison to be
// made across two different scripts, where any other drift would be indistinguishable from the
// scheduler's effect. Default 'pool'.
const GPU_MODE = String(A.gpu_mode || 'pool') === 'pin' ? 'pin' : 'pool';
const GPU_POOL = GPU_MODE === 'pin' ? GPU_LIST[0] : GPU_LIST.join(',');
const TASK = A.task || '';
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const KERNEL_NAME_HINT = KERNEL_PATH_ORIG.replace(/\/+$/, '').split('/').pop();

// --- author mode: when there is NO existing source, write a fresh from-scratch SEED first, then optimize it.
// mode=optimize (default) keeps the exact original behavior (backward compatible). mode=author seeds
// the workspace from an op task dir (immutable oracle + frozen online kernel in baseline_src/), the
// author_engineer writes a passing seed, then the SAME optimize loop runs — always timing against the
// frozen online kernel, never against the seed's own language. KERNEL_KNOWLEDGE_DIR is the AMD authoring
// knowledge base — REFERENCE ONLY (facts/how-to, never decisions; the author always measures regardless). Default:
// sibling perf_knowledge/ so standalone runs use it too; empty if WORKFLOW_DIR is unset (no behavior change).
const MODE = String(A.mode != null ? A.mode : 'optimize').trim() || 'optimize';
const TARGET_LANGUAGE = String(A.target_language != null ? A.target_language : 'triton').trim() || 'triton';
const OP_SPEC = A.op_spec || {};
// When the op will run on the CUDA/HIP-graph-captured decode path (e2e sets op_spec.cuda_graph_safe=true),
// the isolated oracle alone CANNOT catch a kernel that passes iso but host-syncs or lazily-compiles under
// graph capture — the "wins isolated, crashes serving" class (cuda_graph_capture_unsafe / NO_BINARY_FOR_GPU).
// This turns on an OPTIONAL capture+replay smoke in the verify step so that failure is caught at the cheap
// isolated stage. Unset (standalone single-kernel runs / non-graph ops) => byte-identical to before.
const REQUIRE_GRAPH_CAPTURE = !!(OP_SPEC && OP_SPEC.cuda_graph_safe === true);
// WORKLOAD ALIGNMENT (optional). When the caller supplies the real-workload shape/dtype case
// distribution, the benchmark harness benchmarks EXACTLY those (shape, dtype) cases, weights each
// by its total time contribution in the workload (weight = count * baseline_latency), and the
// optimization target becomes the time-weighted ratio-of-sums instead of an unweighted geomean.
//   workload_spec_path : path to a workload-v1 json (produced by parse_profile.py --workload-out,
//                        or hand-written). The benchmark_engineer reads it (JS can't touch FS).
//   op_spec.workload   : inline cases, same shape as a workload-v1 "kernels[].cases" list (or the
//                        full object). Takes precedence; weight_source becomes "caller".
// Both unset => unweighted behavior, byte-identical to before. Correctness ALWAYS stays on the
// frozen immutable oracle (parity vs baseline_src/, + a recorded reference_io.pt where the task dir
// came from e2e's kernel_extractor); this only shapes the PERFORMANCE measurement.
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const WORKLOAD_SPEC = (OP_SPEC && OP_SPEC.workload) || A.workload || null;
const HAS_WORKLOAD = !!(WORKLOAD_SPEC_PATH ||
  (Array.isArray(WORKLOAD_SPEC) && WORKLOAD_SPEC.length) ||
  (WORKLOAD_SPEC && Array.isArray(WORKLOAD_SPEC.kernels) && WORKLOAD_SPEC.kernels.length));
const EXTERNAL_OBJECTIVE = A.objective && A.objective.kind === 'external_primary' ? A.objective : null;
if (EXTERNAL_OBJECTIVE && (Object.keys(EXTERNAL_OBJECTIVE).sort().join(',') !== 'kind,primary_regime'
    || typeof EXTERNAL_OBJECTIVE.primary_regime !== 'string' || !EXTERNAL_OBJECTIVE.primary_regime.trim())) {
  throw new Error('external_primary requires exactly kind and primary_regime');
}
const externalArgv = (value) => Array.isArray(value) && value.length
  && value.every(v => typeof v === 'string' && v) ? value : null;
if (EXTERNAL_OBJECTIVE && (!externalArgv(A.external_result_command)
    || !externalArgv(A.external_measurement_command) || typeof runProcess !== 'function')) {
  throw new Error('external_primary requires the existing owner measurement and result-reader argv plus runProcess');
}
const EXTERNAL_FINAL_COMMANDS = A.external_final_measurement_command != null
  || A.external_final_result_command != null;
if (EXTERNAL_OBJECTIVE && EXTERNAL_FINAL_COMMANDS
    && (!externalArgv(A.external_final_measurement_command)
        || !externalArgv(A.external_final_result_command))) {
  throw new Error('final external measurements require both admitted command prefixes');
}
let frozenFinalEvaluation = null;
let frozenFinalPatch = null;
const finalMeasurementArgv = () => [
  ...(EXTERNAL_FINAL_COMMANDS ? A.external_final_measurement_command : A.external_measurement_command),
  ...(frozenFinalEvaluation ? ['--candidate-id', frozenFinalEvaluation.candidate_id,
    '--evaluation-id', frozenFinalEvaluation.evaluation_id] : []),
];
const externalMetricInstructions = (final = false) => EXTERNAL_OBJECTIVE ? `
External measurement contract: ${JSON.stringify(EXTERNAL_OBJECTIVE)}.
${final ? `The workflow host has launched the final evaluation using ${JSON.stringify(finalMeasurementArgv())}.
Review its retained evidence for ${JSON.stringify(frozenFinalEvaluation)}; do not launch another evaluation.
Preserve applicable unresolved validation objections in validation_status and explain how
the remaining defect applies to this final result.`
  : `Use the existing admitted owner command ${JSON.stringify(A.external_measurement_command)}
from the candidate workspace according to COMMANDMENT.`} One evaluation runs the
unchanged full UT in three fresh processes with native min_runs=2. The measurement
command owns GPU serialization: invoke it directly, without wrapping it
in gpu_lock.sh again. Use the stock GPU lock for separate profiling/diagnostics.
This operation replaces the role's separate correctness/benchmark repetitions and any
extra numerical or graph coverage; preserve the copied UT exactly. Do not invoke
the UT directly except for unscored diagnostics within TASK's admitted scope.
During search, such diagnostics use a separate private workspace and the declared
stock GPU lock; never run them in the canonical workspace or treat their outputs
as correctness or performance scores.
Do not rebuild the UT's case folds, require gain for ORIGINAL qualification, or
repeat valid no-gain results as failures. Retain both native phase folds.
Return the operation's exact IDs as external_evaluation:{candidate_id,evaluation_id}
(under best for an integrated candidate). The workflow's trusted owner reader
checks source, settlement and raw results. A scalar or OBSERVATIONS.json relayed
by a model is not accepted as measurement. Report missing/failed operations and
preserve artifacts. GEAK development and final reports use the checked native
primary for the declared primary_regime; existing improvement thresholds and
stopping rules apply. Patch export also requires the native reproducible-gain gate.
Keep the checked unweighted geomean of all per-case speedups as a secondary
diagnostic, not the ranking objective. The reader retains it under the historical
field development_geomean_speedup; legacy geomean fields may be omitted.
` : '';
const EXTERNAL_METRIC_INSTRUCTIONS = externalMetricInstructions();
const externalFields = (o) => EXTERNAL_OBJECTIVE && o ? Object.fromEntries(
  ['objective', 'primary_regime', 'primary_speedup', 'native_reproducible_gain',
   'native_reproducible_regression', 'phase_speedups', 'external_evaluation',
   'external_measurement_valid', 'receipt_ref', 'timing_basis', 'development_geomean_speedup',
   'final_failure', 'director_review_status', 'director_reported_evaluation', 'patch_sha256',
  ].map(k => [k, o[k]])) : {};
const nativeGainEligible = (o) => !EXTERNAL_OBJECTIVE || !!(o && o.external_measurement_valid === true
  && o.native_reproducible_gain === true && o.native_reproducible_regression === false);

async function captureFinalEvaluation(workspace) {
  if (!EXTERNAL_OBJECTIVE) return;
  const argv = EXTERNAL_FINAL_COMMANDS ? A.external_final_measurement_command : A.external_measurement_command;
  const result = await runProcess({cmd: argv[0], args: [...argv.slice(1),
    '--workspace', workspace, '--capture-only'], timeoutMs: 30000, graceMs: 2000, settleMs: 5000});
  if (result.status !== 'completed' || result.code !== 0 || result.strays_killed
      || !result.group_cleared || result.stdio_held) throw new Error('final source capture failed');
  const captured = JSON.parse(result.stdout);
  if (captured.schema !== 'geak-native-measurement-identity-v1' || captured.capture_only !== true
      || ![captured.candidate_id, captured.source_id].every(v => typeof v === 'string' && /^[a-f0-9]{64}$/.test(v))
      || !Array.isArray(captured.source_paths) || !captured.source_paths.length
      || !captured.source_paths.every(v => typeof v === 'string' && v)
      || typeof captured.evaluation_id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$/.test(captured.evaluation_id)) {
    throw new Error('invalid final source capture');
  }
  // Diff exactly the captured contract sources, including uncommitted changes,
  // from the initial ORIGINAL commit. The final reader checks this against the
  // immutable ORIGINAL files and the captured source identity.
  const baseline = ((await git(workspace, 'rev-list', '--max-parents=0', 'HEAD')) || '').trim();
  frozenFinalPatch = /^[0-9a-f]{40,64}$/.test(baseline)
    ? await git(workspace, '--literal-pathspecs', 'diff', '--binary', '--no-color', '--no-ext-diff',
      '--src-prefix=a/', '--dst-prefix=b/', baseline, '--', ...captured.source_paths) : null;
  await writeFrozenFinalPatch();
  frozenFinalEvaluation = Object.freeze(captured);
}

async function writeFrozenFinalPatch() {
  const path = `${EVAL_DIR}/host_final_patch.diff`;
  if (frozenFinalPatch === null || await writeFile(path, frozenFinalPatch) === null) {
    throw new Error('could not write the captured final source patch');
  }
  return path;
}

async function measureFinalEvaluation(workspace) {
  if (!EXTERNAL_OBJECTIVE) return null;
  try {
    const argv = finalMeasurementArgv();
    const result = await runProcess({cmd: argv[0], args: [...argv.slice(1), '--workspace', workspace],
      timeoutMs: AGENT_TIMEOUT_MS > 0 ? AGENT_TIMEOUT_MS : 3600000, graceMs: 2000, settleMs: 5000});
    if (result.status !== 'completed' || result.code !== 0 || result.strays_killed
        || !result.group_cleared || result.stdio_held) throw new Error('final evaluator did not complete cleanly');
    return null;
  } catch (error) {
    log(`Final measurement failed: ${String(error)}`);
    return {class: 'final_measurement_failed', reason: String(error)};
  }
}

// Optional prospective source history. Keep exact nanosecond values in the
// existing Store evidence bytes; JavaScript numbers must not round them.
const CAPTURE_CANONICAL_SELECTIONS = A.capture_canonical_selections === true;
if (CAPTURE_CANONICAL_SELECTIONS && (!EXTERNAL_OBJECTIVE || String(A.optimizer || 'greedy') !== 'greedy')) {
  throw new Error('canonical selection capture requires the external_primary GEAK lane');
}
async function captureCanonicalSelection(history, kind, round, winner = null, stoppedBy = null) {
  if (!CAPTURE_CANONICAL_SELECTIONS) return;
  const row = {kind, round, stopped_by: stoppedBy, status: 'unresolved'};
  if (!history.canonical_selections) history.canonical_selections = [];
  const previous = [...history.canonical_selections].reverse().find(item => item.capture_ref);
  const transition = {kind, round, stopped_by: stoppedBy,
    sequence: history.canonical_selections.length + 1,
    previous_capture_ref: previous ? previous.capture_ref || null : null};
  Object.assign(row, transition);
  history.canonical_selections.push(row);
  try {
    const argv = A.external_measurement_command;
    const selected = winner && winner.external_evaluation;
    const result = await runProcess({cmd: argv[0], args: [...argv.slice(1),
      '--workspace', CANONICAL, '--capture-selection', '--selection-kind', kind,
      '--selection-round', String(round), '--selection-sequence', String(transition.sequence),
      ...(transition.previous_capture_ref ? ['--previous-selection', transition.previous_capture_ref.evidence_id] : []),
      ...(stoppedBy ? ['--selection-stop-reason', stoppedBy] : []), ...(selected ? [
        '--selected-candidate-id', selected.candidate_id, '--selected-evaluation-id', selected.evaluation_id] : [])],
      timeoutMs: 30000, graceMs: 2000, settleMs: 5000});
    if (result.status !== 'completed' || result.code !== 0 || result.strays_killed
        || !result.group_cleared || result.stdio_held) throw new Error('canonical source capture failed');
    const capture = JSON.parse(result.stdout);
    const returned = capture.transition;
    const prior = returned && returned.previous_capture_ref;
    const expectedPrior = transition.previous_capture_ref;
    if (capture.schema !== 'geak-canonical-source-capture-v2'
        || !returned || !['kind', 'round', 'sequence', 'stopped_by'].every(key => returned[key] === transition[key])
        || (expectedPrior ? !prior || prior.evidence_id !== expectedPrior.evidence_id
          || prior.artifact !== expectedPrior.artifact : prior !== null)
        || ![capture.candidate_id, capture.source_id, capture.capture_ref && capture.capture_ref.evidence_id]
          .every(value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value))
        || capture.capture_ref.artifact !== 'canonical-selection.json') throw new Error('invalid canonical capture');
    Object.assign(row, {capture_ref: capture.capture_ref, candidate_id: capture.candidate_id,
      source_id: capture.source_id, clock_ref: capture.clock_ref,
      development_status: capture.development_evidence.status});
    if (capture.canonical_commit.status !== 'committed') throw new Error('canonical Git state is unresolved');
    Object.assign(row, {status: 'captured', git_commit: capture.canonical_commit.git_commit});
  } catch (error) {
    row.reason = String(error);
  }
  // This is observational. Failure cannot stop authentic GEAK search. Attempted
  // sequence numbers still advance, so a gap cannot become a complete chain.
}

async function checkedExternalMeasurement(value, workspace, patch = null, final = false) {
  if (!EXTERNAL_OBJECTIVE || (!value && !final)) return value;
  const invalid = { ...value, primary_speedup: null, external_measurement_valid: false,
    status: 'invalid_measurement', correctness: 'not_checked', validation_status: 'invalid_measurement',
    native_reproducible_gain: false, native_reproducible_regression: false,
    development_geomean_speedup: null };
  const ref = value && value.external_evaluation;
  if (final && (!ref || !frozenFinalEvaluation || ref.candidate_id !== frozenFinalEvaluation.candidate_id
      || ref.evaluation_id !== frozenFinalEvaluation.evaluation_id)) {
    return {...invalid, final_failure: {class: 'final_identity_mismatch',
      expected: frozenFinalEvaluation, returned: ref || null}};
  }
  if (!ref || typeof ref.candidate_id !== 'string' || typeof ref.evaluation_id !== 'string') return invalid;
  try {
    const argv = final && EXTERNAL_FINAL_COMMANDS
      ? A.external_final_result_command : A.external_result_command;
    const result = await runProcess({cmd: argv[0], args: [...argv.slice(1),
      '--candidate-id', ref.candidate_id, '--evaluation-id', ref.evaluation_id,
      '--workspace', workspace, ...(patch ? ['--patch', patch] : [])],
      timeoutMs: 30000, graceMs: 2000, settleMs: 5000});
    if (result.status !== 'completed' || result.code !== 0 || result.strays_killed
        || !result.group_cleared || result.stdio_held) return invalid;
    const read = JSON.parse(result.stdout);
    if (read.external_measurement_valid !== true || !read.objective
        || read.objective.kind !== 'external_primary'
        || read.primary_regime !== EXTERNAL_OBJECTIVE.primary_regime
        || read.objective.primary_regime !== EXTERNAL_OBJECTIVE.primary_regime
        || !Number.isFinite(read.primary_speedup) || read.primary_speedup <= 0
        || !Number.isFinite(read.development_geomean_speedup) || read.development_geomean_speedup <= 0
        || read.external_evaluation.candidate_id !== ref.candidate_id
        || read.external_evaluation.evaluation_id !== ref.evaluation_id) return invalid;
    return { ...value, ...read };
  } catch (error) {
    log(`External measurement reader failed: ${String(error)}`);
    return invalid;
  }
}
// Checked external fields only; invalid or foreign-regime readings score null.
const checkedExternal = (o, key) => o && o.external_measurement_valid === true
  && o.primary_regime === EXTERNAL_OBJECTIVE.primary_regime
  && Number.isFinite(o[key]) ? o[key] : null;
// The declared native primary is shared by external development and report/export.
const nativePrimary = (o) => EXTERNAL_OBJECTIVE ? checkedExternal(o, 'primary_speedup') : primSpeedup(o);
// PRIMARY-metric selector: prefer the time-weighted number when a workload spec is in play and the
// agent reported one; otherwise fall back to the geomean (unweighted runs => unchanged behavior).
// External mode uses the checked task primary, matching its declared final objective.
// The checked all-case geomean remains a secondary diagnostic.
const primSpeedup = (o) => {
  if (!o) return EXTERNAL_OBJECTIVE ? null : 0;
  if (EXTERNAL_OBJECTIVE) return checkedExternal(o, 'primary_speedup');
  const w = o.verified_weighted != null ? o.verified_weighted
          : (o.speedup_weighted != null ? o.speedup_weighted : null);
  if (HAS_WORKLOAD && Number.isFinite(w)) return w;
  const g = o.verified_geomean != null ? o.verified_geomean : o.speedup_geomean;
  return Number.isFinite(g) ? g : 0;
};
// Gate fields like `correctness` / `status` are free strings in the schemas, so an agent legitimately
// answers "PASS - 15/15 draws" and a `=== 'pass'` test silently drops a genuinely verified candidate.
// Match the leading word instead: "PASS - ..." / "passed" gate open, "FAIL"/"did not pass" stay shut.
const says = (v, w) => String(v == null ? '' : v).trim().toLowerCase().startsWith(w);
let KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge' : '')).replace(/\/+$/, '');
// Expert skills = human-authored, validated kernel recipes (perf_knowledge/expert_skills/). ADVISORY
// priors only: a matched `validated` skill is a HIGH-PRIOR author/optimize candidate the planning/author
// roles reproduce, then gate by the isolated A/B vs the oracle — it NEVER overrides measurement. Default
// OFF (opt-in: pass use_expert_skills="true"). When OFF (the default) NOTHING is injected -> byte-identical
// to a build without this feature. When invoked by the e2e layer the flag + dir are passed down.
const USE_EXPERT_SKILLS_REQUESTED = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
let USE_EXPERT_SKILLS = USE_EXPERT_SKILLS_REQUESTED;
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
// Only planning + authoring roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['tech_lead', 'author_engineer', 'engineer', 'deep_engineer']);
// Expected-identity isolation must apply before Setup (no CDNA skills/KB on a declared R9700 run).
// Detected identity is OR'd in after director setup — do not read GFX here (TDZ).
let SKILLS_ISOLATE = EXPECTED_GFX === 'gfx1201';

// --- Deep Research Agent (DRA) -------------------------------------------------------------------
// OPT-IN: a v4-native research phase that runs AFTER Profile and BEFORE the optimize loop (so the
// COMMANDMENT + baseline profile + analysis exist). The `researcher` persona extracts facts and a
// ranked set of research QUESTIONS; the script fans those out in PARALLEL (each question = one
// hang-guarded agent using native WebSearch/WebFetch), then a synthesis pass writes a ranked
// directions portfolio (deep_search.md / deep_search_brief.md / deep_search.json) into EVAL_DIR that
// the TechLead's plan_round seeds from. DEFAULT OFF: when dra_enabled is not "true" NOTHING runs and
// behavior is byte-identical to a build without this feature (existing runs unchanged).
const DRA_ENABLED = String(A.dra_enabled != null ? A.dra_enabled : 'false') === 'true';
const DRA_MAX_QUESTIONS = (() => {
  const v = parseInt(A.dra_max_questions != null ? A.dra_max_questions : 8, 10);
  return Number.isFinite(v) && v >= 1 ? v : 8;
})();
// Optional Stage 5/6 blindspot-critique pass (one more parallel research wave). Default OFF (budget).
const DRA_BLINDSPOT = String(A.dra_blindspot != null ? A.dra_blindspot : 'false') === 'true';
const DRA_MAX_BLINDSPOTS = (() => {
  const v = parseInt(A.dra_max_blindspots != null ? A.dra_max_blindspots : 4, 10);
  return Number.isFinite(v) && v >= 1 ? v : 4;
})();
