export const meta = {
  name: 'kernel-workflow',
  description: 'Multi-agent GPU kernel / e2e-model optimization (Director/TechLead/specialist Engineers) with budget-controlled rounds, independent verification, and integration. Optimizes a kernel or vLLM/SGLang model for inference speed on AMD Instinct MI-series GPUs (MI300X/300A/308X/325X on CDNA3 gfx942, MI350X/355X on CDNA4 gfx950 — the target card is auto-detected on-box).',
  whenToUse: 'Optimize the inference speed of a kernel directory (single kernel, fused kernels) or an end-to-end vLLM/SGLang model. Pass args.kernel_path (required), args.budget, args.gpu_ids, args.task.',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + canonical workspace' },
    { title: 'Author', detail: 'author_engineer writes a fresh optimize-loop seed (only when mode=author); speedup denominator stays the frozen online kernel' },
    { title: 'Analyze', detail: 'tech_lead analyzes kernel + writes roadmap' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + baseline' },
    { title: 'Profile', detail: 'profile_engineer classifies the bottleneck' },
    { title: 'Optimize', detail: 'budget loop: tech_lead plans, specialist OR deep_explore engineers optimize, reprofile' },
    { title: 'Verify', detail: 'each candidate patch independently re-benchmarked' },
    { title: 'Merge', detail: 'integrator combines the round winners' },
    { title: 'Report', detail: 'tech_lead writes the final report + patch' },
    { title: 'Validate', detail: 'director independently validates vs the true baseline' },
  ],
};

// ---------------------------------------------------------------------------
// Args + defaults. (The script cannot touch the filesystem or read its own
// path; agents do all FS work, and every path is supplied/derived from args —
// nothing about the install location is hard-coded.)
// ---------------------------------------------------------------------------
const A = args || {};
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/model directory)');

// WORKFLOW_DIR = the directory that holds this script + roles/ + knowledge/ + scripts/.
// A JS workflow script can't read its own path, so the caller passes it (it is just the
// dirname of the scriptPath used to launch the workflow).
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_workflow.js, roles/, knowledge/, scripts/ (i.e. the dirname of this script).');
}
// EXP_ROOT = where timestamped run dirs are written. Default: a sibling "exp/" next to the
// kernel_workflow dir (…/<parent>/kernel_workflow -> …/<parent>/exp). Override with args.exp_root.
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

const KERNEL_PATH_ORIG = A.kernel_path;
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
// Minimum verified geomean improvement over the cumulative best for a round winner to be COMMITTED
// into the canonical workspace (default 2%). Kept as a knob rather than a hard-coded constant so the
// gate is tunable per run (e.g. raise it on a noisy box, lower it to bank small compounding wins).
const MIN_IMPROVE = (() => {
  const v = parseFloat(A.min_improve != null ? A.min_improve : 0.02);
  return Number.isFinite(v) && v >= 0 ? v : 0.02;
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
// frozen reference_io.pt oracle; this only shapes the PERFORMANCE measurement.
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const WORKLOAD_SPEC = (OP_SPEC && OP_SPEC.workload) || A.workload || null;
const HAS_WORKLOAD = !!(WORKLOAD_SPEC_PATH ||
  (Array.isArray(WORKLOAD_SPEC) && WORKLOAD_SPEC.length) ||
  (WORKLOAD_SPEC && Array.isArray(WORKLOAD_SPEC.kernels) && WORKLOAD_SPEC.kernels.length));
// PRIMARY-metric selector: prefer the time-weighted number when a workload spec is in play and the
// agent reported one; otherwise fall back to the geomean (unweighted runs => unchanged behavior).
const primSpeedup = (o) => {
  if (!o) return 0;
  const w = o.verified_weighted != null ? o.verified_weighted
          : (o.speedup_weighted != null ? o.speedup_weighted : null);
  if (HAS_WORKLOAD && Number.isFinite(w)) return w;
  const g = o.verified_geomean != null ? o.verified_geomean : o.speedup_geomean;
  return Number.isFinite(g) ? g : 0;
};
const KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge' : '')).replace(/\/+$/, '');
// Expert skills = human-authored, validated kernel recipes (perf_knowledge/expert_skills/). ADVISORY
// priors only: a matched `validated` skill is a HIGH-PRIOR author/optimize candidate the planning/author
// roles reproduce, then gate by the isolated A/B vs the oracle — it NEVER overrides measurement. Default
// OFF (opt-in: pass use_expert_skills="true"). When OFF (the default) NOTHING is injected -> byte-identical
// to a build without this feature. When invoked by the e2e layer the flag + dir are passed down.
const USE_EXPERT_SKILLS = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
// Only planning + authoring roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['tech_lead', 'author_engineer', 'engineer', 'deep_engineer']);

// ---------------------------------------------------------------------------
// DEEP-MODE continuation + cross-backend / e2e-feedback hooks. ALL OPTIONAL.
// When none are passed (every normal/fast e2e run, and every standalone run) these are '' / the
// current defaults and are NEVER threaded into a prompt — so behavior is byte-identical to the
// pre-feature build. They are set ONLY by e2e_workflow's deep_mode head scheduler.
//   STATE_DIR        a STABLE dir for THIS (kernel,backend) ACROSS deep waves. When set the run
//                    RESUMES: director seeds the canonical from STATE_DIR/best (the cumulative-best
//                    code) and returns prior_state (cumulative + history) so re-invocation CONTINUES
//                    instead of restarting (no lost experience, no re-explored directions). The frozen
//                    oracle baseline (immutable unittest.py/meta.json) stays the reference, so speedups
//                    remain comparable to the TRUE baseline across waves. update_memory writes STATE.json
//                    + syncs best/ each round.
//   SHARED_KB        cross-backend blackboard file (read by plan+engineers, appended by update_memory).
//   GLOBAL_KB        run-global cross-KERNEL technique blackboard (deep): techniques that generalize
//                    across head ops/backends. Optional; unset (default/fast) => byte-identical prompts.
//   E2E_FEEDBACK     path to the latest end-to-end A/B result + problems from e2e_workflow (engaged?,
//                    cudagraph behavior, mem footprint, decode regression, e2e delta) — steers planning.
//   HARNESS_ADDENDUM path to an e2e-refined harness addendum (timing-weight / cudagraph-capture / hard
//                    constraint gates). The IMMUTABLE oracle is NEVER touched; this only refines what the
//                    perf bench emphasizes so the isolated target aligns with e2e.
//   MAX_NO_IMPROVE   consecutive non-improving rounds before stopping (default 2 = current behavior).
const STATE_DIR = String(A.state_dir || '').replace(/\/+$/, '');
const SHARED_KB = String(A.shared_kb || '').trim();
const GLOBAL_KB = String(A.global_kb || '').trim();   // run-global cross-KERNEL technique blackboard (deep)
const E2E_FEEDBACK = String(A.e2e_feedback || '').trim();
const HARNESS_ADDENDUM = String(A.harness_addendum || '').trim();
// P2 (deep continuation): on a RESUMED wave (STATE_DIR holds prior work) the cold re-derivation of
// Analyze + baseline Profile is largely redundant. INCREMENTAL_RESUME tells those two ADVISORY agents to
// load the prior roadmap/profile and return it with only delta updates instead of re-deriving from
// scratch, so each burst spends its budget on optimization ROUNDS, not re-analysis. Benchmark is NEVER
// incremental (it re-pins a fresh in-window baseline every wave \u2014 the matched-A/B correctness rail).
// Unset (default/fast / first deep burst) => spreading {} adds nothing => byte-identical prompts.
const INCREMENTAL = !!STATE_DIR && String(A.incremental_analyze || '') === 'true';
const RESUME_INPUT = INCREMENTAL ? { INCREMENTAL_RESUME: '1' } : {};
const MAX_NO_IMPROVE = Math.max(1, parseInt(A.max_no_improve != null ? A.max_no_improve : 2, 10));
// Conditional inputs: spreading {} adds NOTHING to a prompt (byte-identical) when a hook is unset.
const KB_INPUTS = {
  ...(SHARED_KB ? { SHARED_KB } : {}),
  ...(GLOBAL_KB ? { GLOBAL_KB } : {}),
  ...(E2E_FEEDBACK ? { E2E_FEEDBACK } : {}),
  ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
};

// ---------------------------------------------------------------------------
// Reusable JSON-schema fragments.
// ---------------------------------------------------------------------------
const perCase = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      baseline_ms: { type: 'number' },
      optimized_ms: { type: 'number' },
      speedup: { type: 'number' },
      // Workload-alignment fields (present only when a WORKLOAD_SPEC drives the harness; absent
      // on a normal unweighted run). weight = this case's baseline time SHARE in the real workload;
      // it is the coefficient of the time-weighted metric Σweight / Σ(weight/speedup). count is
      // optional/informational (regime-attributed cases have no per-call count).
      weight: { type: 'number' },
      count: { type: 'number' },
      dims: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
      dtypes: { type: 'array', items: { type: 'string' } },
      weight_source: { type: 'string' }, // trace | regime | regime_floor | prior | caller
    },
    required: ['name', 'speedup'],
  },
};
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });

const SETUP_SCHEMA = obj({
  eval_dir: { type: 'string' }, workspace: { type: 'string' }, baseline_dir: { type: 'string' },
  kernel_name: { type: 'string' }, source_files: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // Frozen-baseline verdict (BOTH modes). The unittest's timing + random-value parity baseline MUST be
  // the real online kernel — the immutable baseline_src/ dir OR an importable meta.baseline_callable —
  // never kernel_src/ (the candidate's own scaffold). The director sets baseline_frozen=true after it
  // copies baseline_src/ + confirms meta.baseline_callable; the script aborts the run if neither holds.
  baseline_frozen: { type: 'boolean' }, baseline_callable: { type: 'string' },
  // DEEP-MODE resume only: populated by the director ONLY when STATE_DIR was provided AND a prior best
  // exists there. Lets a continued wave restore its cumulative speedup + insight/ledger history so it
  // does not re-explore dead directions. Absent (undefined) on a fresh run -> no behavior change.
  resumed: { type: 'boolean' },
  prior_state: obj({
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
  }, []),
}, ['eval_dir', 'workspace', 'kernel_name']);

const AUTHOR_SCHEMA = obj({
  authored: { type: 'boolean' }, target_language: { type: 'string' }, correctness: { type: 'string' },
  baseline_ms: { type: 'number' }, kernel_src_path: { type: 'string' }, entry_point: { type: 'string' },
  build: { type: 'boolean' }, notes: { type: 'string' },
}, ['authored', 'correctness']);

const ANALYZE_SCHEMA = obj({
  kernel_type: { type: 'string' }, kernel_file: { type: 'string' }, entry_point: { type: 'string' },
  modifiable_files: { type: 'array', items: { type: 'string' } },
  bottleneck_guess: { type: 'string' }, roadmap_summary: { type: 'string' },
  candidate_directions: { type: 'array', items: { type: 'object', additionalProperties: true } },
  // perf_knowledge resolution (REFERENCE ONLY): the operator/language this kernel maps to in the
  // AMD perf_knowledge base, plus the most relevant card paths, so engineers read focused context
  // instead of re-navigating the whole base. Empty string / [] / null when no card applies.
  kk_operator: { type: ['string', 'null'] }, kk_language: { type: ['string', 'null'] },
  kk_refs: { type: 'array', items: { type: 'string' } },
}, ['kernel_type', 'roadmap_summary']);

const BENCH_SCHEMA = obj({
  commandment_path: { type: 'string' }, correctness_cmd: { type: 'string' },
  benchmark_cmd: { type: 'string' }, profile_cmd: { type: 'string' }, parse_hint: { type: 'string' },
  baseline_per_case: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_geomean_ms: { type: 'number' }, num_test_cases: { type: 'number' },
  // Workload-aligned outputs: present when a WORKLOAD_SPEC drove case selection + weights.
  // baseline_weighted_total_ms = the baseline time the weights represent (Σ weight_i in time units).
  // The metric is Σ weight_i / Σ (weight_i/speedup_i). workload_aligned flags weights are real (not 1).
  workload_aligned: { type: 'boolean' },
  baseline_weighted_total_ms: { type: 'number' },
  weights_provenance: { type: 'string' }, // e.g. "trace" | "regime" | "regime_floor" | "prior" | "caller" | "mixed"
  reliable: { type: 'boolean' }, notes: { type: 'string' },
}, ['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']);

const PROFILE_SCHEMA = obj({
  bottleneck: { type: 'string' }, profiler_used: { type: 'string' }, dispatch_count: { type: 'number' },
  // The accelerator detected on-box (e.g. "MI300X / gfx942 / CDNA3, 304 CU, ~5.3 TB/s"), so the
  // roofline ceiling + grid-sizing advice downstream use the real card instead of an assumed MI300X.
  device: { type: 'string' },
  key_metrics: { type: 'object', additionalProperties: true },
  top_kernels: { type: 'array', items: { type: 'object', additionalProperties: true } },
  top_opportunities: { type: 'array', items: { type: 'string' } },
  summary_path: { type: 'string' }, shift_note: { type: 'string' },
}, ['bottleneck', 'top_opportunities']);

const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'deep_explore'] },
      focus_files: { type: 'array', items: { type: 'string' } },
      expected_speedup: { type: 'number' }, prompt: { type: 'string' },
      kk_refs: { type: 'array', items: { type: 'string' } }, // optional: perf_knowledge card paths for THIS direction (REFERENCE ONLY)
    }, ['id', 'title', 'specialty', 'prompt']),
  },
}, ['stop', 'directions']);

const ENG_SCHEMA = obj({
  engineer_id: { type: 'string' }, specialty: { type: 'string' }, task: { type: 'string' },
  strategy: { type: 'string' }, speedup_geomean: { type: 'number' }, speedup_arithmetic: { type: 'number' },
  // Time-weighted ratio-of-sums vs the TRUE baseline (PRIMARY metric when workload_aligned).
  // = Σ weight_i / Σ (weight_i / speedup_i). Omitted on unweighted runs.
  speedup_weighted: { type: 'number' },
  per_case: perCase, status: { type: 'string' }, patch_file: { type: 'string' },
  strategies_tried: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
}, ['status', 'speedup_geomean']);

const VERIFY_SCHEMA = obj({
  status: { type: 'string' }, correctness: { type: 'string' },
  verified_geomean: { type: 'number' }, verified_arithmetic: { type: 'number' },
  verified_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  per_case: perCase, variance_note: { type: 'string' }, notes: { type: 'string' },
  graph_safe: { type: 'string' },
}, ['status', 'verified_geomean']);

const INTEGRATE_SCHEMA = obj({
  attempted: { type: 'boolean' },
  combos_tried: { type: 'array', items: { type: 'object', additionalProperties: true } },
  best: { type: 'object', additionalProperties: true },
  improved_over_best_individual: { type: 'boolean' },
  conclusion: { type: 'string' }, notes: { type: 'string' },
}, ['attempted', 'conclusion']);

const MEMORY_SCHEMA = obj({
  insights: { type: 'array', items: { type: 'string' } },
  ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);

const COMMIT_SCHEMA = obj({
  committed: { type: 'boolean' }, current_best_diff: { type: 'string' }, note: { type: 'string' },
}, ['committed']);

const REPORT_SCHEMA = obj({
  final_speedup_geomean: { type: 'number' }, final_speedup_arithmetic: { type: 'number' },
  final_speedup_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  rounds: { type: 'number' }, budget_used: { type: 'number' },
  report_path: { type: 'string' }, final_patch: { type: 'string' }, per_case: perCase,
}, ['final_speedup_geomean', 'report_path', 'final_patch']);

const VALIDATE_SCHEMA = obj({
  kernel_name: { type: 'string' },
  director_verified_speedup_geomean: { type: 'number' },
  director_verified_speedup_arithmetic: { type: 'number' },
  director_verified_speedup_weighted: { type: 'number' }, // PRIMARY when workload_aligned
  tech_lead_reported_speedup_geomean: { type: 'number' },
  validation_status: { type: 'string' }, correctness: { type: 'string' },
  per_case: perCase, applied_to_original: { type: 'string' },
  arbitration_note: { type: 'string' }, final_patch: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status']);

// ---------------------------------------------------------------------------
// Prompt helpers. Every agent reads its role file from WORKFLOW_DIR and the
// relevant knowledge files itself; the script only passes paths + JSON inputs.
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// --- Hung-agent guard ------------------------------------------------------
// An agent LLM call that HANGS (no response, no terminal error) blocks a
// parallel()/pipeline() round-barrier forever (observed: engineer agents frozen
// mid-turn wedged the whole optimize round for >30min). The harness resolves
// terminal API errors to null but NOT an indefinite hang. So bound every agent()
// call: if it has not returned after AGENT_TIMEOUT_MS, resolve it to null (which
// every .filter(Boolean)/null-check downstream already tolerates) and let the
// round proceed. VERY generous default (60min): a true hang never returns, so this only fires on a
// hang, NEVER on a legitimately-long agent. Inner agents include benchmark/profile/verify that build
// (hipcc/ninja) and run benches — minutes, well under 60min — plus the LLM-heavy optimize engineers
// (the ones observed hanging). A too-short bound would kill legit long agents (e.g. a slow rocprof or
// build), so keep it large. Cache keys (prompt, opts) are unchanged so resume still works. Falls back
// to raw agent() if setTimeout is unavailable. args.agent_timeout_ms=0 disables.
// API-FAULT TOLERANCE: a transient API failure (gateway 4xx/5xx, rate-limit, dropped connection, the
// model API going down mid-run) must NOT crash the whole workflow. agentT retries the call up to
// AGENT_RETRIES times on a thrown API/agent error, then resolves to null (every .filter(Boolean)/
// null-check downstream — incl. the Director validate + final report — already degrades on null rather
// than exiting). A timeout (hang) resolves null immediately and is NOT retried (a real hang would just
// burn another full timeout window). args.agent_retries tunes the count. If the failure is PERSISTENT
// (e.g. an auth/header requirement the client doesn't send), retries are exhausted then the run
// degrades — re-run with Workflow({resumeFromRunId}) once the client/API is fixed; cached agent results
// make resume cheap.
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 3600000, 10);
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
async function agentT(p, o) {
  const label = (o && o.label) ? o.label : 'agent';
  for (let attempt = 1; attempt <= AGENT_RETRIES; attempt++) {
    try {
      if (typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) return await agent(p, o);
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(AGENT_TIMEOUT_MS / 60000)}min with no return — resolving null so the round proceeds.`);
          resolve(null);
        }, AGENT_TIMEOUT_MS);
      });
      // A timeout resolves null (returned as-is, no retry). An API/agent error rejects -> caught below.
      return await Promise.race([
        agent(p, o).then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
    } catch (e) {
      const msg = String(e && e.message ? e.message : e).slice(0, 200);
      if (attempt < AGENT_RETRIES) {
        log(`  [api-fault guard] ${label} attempt ${attempt}/${AGENT_RETRIES} hit an API/agent error (${msg}) — retrying so a transient outage doesn't kill the run.`);
        continue;
      }
      log(`  [api-fault guard] ${label} still failing after ${AGENT_RETRIES} attempts (${msg}) — resolving null so the workflow degrades gracefully instead of exiting.`);
      return null;
    }
  }
  return null;
}

// Expert-skills injection. PURELY ADDITIVE: '' when OFF or the role is not a skills consumer, so
// roleAgent is byte-identical to the pre-feature build in those cases. When ON, appends an advisory
// pointer telling the agent to Read the fragment + query the skills index (scripts have no fs access).
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op (operator/dtype/regime, and ` +
    `from_backend->to_backend for migration skills) and whose validation_status is \`validated\`, and ` +
    `treat each as a HIGH-PRIOR candidate to reproduce — advisory only, never overriding your isolated ` +
    `A/B vs the oracle, never reducing a result below the measured baseline.`;
}

function roleAgent(role, phase, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role);
}

// ===========================================================================
// PHASE: Setup
// ===========================================================================
phase('Setup');
const setup = await agentT(
  roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC,
    ...(STATE_DIR ? { STATE_DIR } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
const KERNEL_NAME = setup.kernel_name;
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
log(`Setup done. EVAL_DIR=${EVAL_DIR}`);

// ---------------------------------------------------------------------------
// Enforce a FROZEN REAL-ONLINE BASELINE in BOTH modes (author AND same-language
// optimize). The immutable unittest times + parity-checks the candidate against
// baseline_src/ / meta.baseline_callable (the live online kernel); if neither
// exists it would silently fall back to timing kernel_src/ against itself — the
// "optimized-HIP vs naive-HIP = fake 15.7×" bug this harness exists to prevent.
// The script has no FS access, so we trust the director's structured verdict
// (it copied baseline_src/ + confirmed the callable). Missing -> abort/re-extract.
// ---------------------------------------------------------------------------
const hasBaseline = setup.baseline_frozen === true ||
  (typeof setup.baseline_callable === 'string' && setup.baseline_callable.trim().length > 0);
if (!hasBaseline) {
  const reason = `no frozen baseline (baseline_src/ or meta.baseline_callable) for ${KERNEL_NAME} — ` +
    `re-extract; refusing to time the candidate against kernel_src/ (fake-win risk)`;
  log(`Setup ABORT: ${reason}`);
  return {
    mode: MODE, authored: false, target_language: TARGET_LANGUAGE,
    eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
    final_geomean: 0, final_patch: '', validation_status: 'no_baseline', reason,
  };
}

// ===========================================================================
// PHASE: Author (mode=author only) — write a fresh from-scratch impl as the
// optimize loop's CODE SEED. On success, HEAD of CANONICAL becomes that seed
// (what the optimize loop diffs its edits against) and the rest of the pipeline
// (Analyze/Benchmark/Profile/optimize loop) runs UNCHANGED on it. The SPEEDUP
// denominator is NEVER the seed — it is the frozen REAL ONLINE kernel in
// baseline_src/ (meta.baseline_callable), regardless of TARGET_LANGUAGE.
// On failure (no correct seed), abort early with a structured result so the
// e2e caller drops this language.
// ===========================================================================
if (MODE === 'author') {
  phase('Author');
  const authored = await agentT(
    roleAgent('author_engineer', 'author', 'Write the simplest correct baseline in the target language.', {
      TARGET_LANGUAGE, OP_SPEC, WORKSPACE: CANONICAL, TASK_DIR: KERNEL_PATH_ORIG,
      GPU_ID: GPU_LIST[0], SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, KERNEL_KNOWLEDGE_DIR,
    }),
    { phase: 'Author', label: `author:${TARGET_LANGUAGE}`, schema: AUTHOR_SCHEMA });
  if (!authored || !authored.authored || authored.correctness !== 'pass') {
    log(`Author mode FAILED for ${TARGET_LANGUAGE}: ${authored ? authored.notes || authored.correctness : 'no result'}. Aborting (no seed to optimize).`);
    return {
      mode: 'author', authored: false, target_language: TARGET_LANGUAGE,
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
      final_geomean: 0, final_patch: '', validation_status: 'author_failed',
      reason: authored ? authored.notes || 'author produced no correct baseline' : 'author returned nothing',
    };
  }
  log(`Author mode: ${TARGET_LANGUAGE} seed written (correct, seed ${authored.baseline_ms || '?'} ms; denominator = frozen online kernel). Optimizing it now.`);
}

// ===========================================================================
// PHASE: Analyze + Roadmap (TechLead)
// ===========================================================================
phase('Analyze');
const analysis = await agentT(
  roleAgent('tech_lead', 'analyze', 'Analyze the kernel and write the roadmap.', {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_KNOWLEDGE_DIR,
    ...RESUME_INPUT,
  }),
  { phase: 'Analyze', label: 'tech_lead:analyze', schema: ANALYZE_SCHEMA });
log(`Analyze done. kernel_type=${analysis ? analysis.kernel_type : '?'}`);

// perf_knowledge pointers resolved by the TechLead in analyze (REFERENCE ONLY; threaded to the
// planner + engineers so they read focused op/language cards instead of the whole base). Empty when
// no operator card applies (e.g. point-cloud HIP ops) or KERNEL_KNOWLEDGE_DIR is unset → no change.
const KK_OPERATOR = (analysis && analysis.kk_operator) || '';
const KK_LANGUAGE = (analysis && analysis.kk_language) || '';
const KK_REFS = (analysis && Array.isArray(analysis.kk_refs)) ? analysis.kk_refs : [];

// ===========================================================================
// PHASE: Benchmark setup (Benchmark Engineer)
// ===========================================================================
phase('Benchmark');
const bench = await agentT(
  roleAgent('benchmark_engineer', 'setup', 'Build the COMMANDMENT and record a reliable baseline.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_LIST[0],
    ANALYSIS: analysis,
    ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
    ...(WORKLOAD_SPEC_PATH ? { WORKLOAD_SPEC_PATH } : {}),
    ...(WORKLOAD_SPEC ? { WORKLOAD_SPEC } : {}),
  }),
  { phase: 'Benchmark', label: 'benchmark_engineer', schema: BENCH_SCHEMA });
if (!bench || !bench.baseline_per_case) throw new Error('Benchmark setup failed: no baseline recorded');
const BASELINE_PER_CASE = bench.baseline_per_case;
const BASELINE_GEOMEAN_MS = bench.baseline_geomean_ms;
log(`Benchmark done. ${bench.num_test_cases || BASELINE_PER_CASE.length} cases, baseline geomean ${BASELINE_GEOMEAN_MS} ms, reliable=${bench.reliable}`);

// ===========================================================================
// PHASE: Baseline profile (Profile Engineer)
// ===========================================================================
phase('Profile');
let profileSummary = await agentT(
  roleAgent('profile_engineer', 'baseline', 'Profile the baseline and classify the bottleneck.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_LIST[0], ROUND: 0,
    COMMANDMENT,
    ...RESUME_INPUT,
  }),
  { phase: 'Profile', label: 'profile_engineer:baseline', schema: PROFILE_SCHEMA });
log(`Baseline bottleneck: ${profileSummary ? profileSummary.bottleneck : '?'} (dispatch_count=${profileSummary ? profileSummary.dispatch_count : '?'})`);

// ===========================================================================
// PHASE: Optimization loop (budget-controlled)
// ===========================================================================
let dispatched = 0;          // counts ONLY optimization-direction engineers (the budget)
let round = 0;
let cumulative = 1.0;        // best verified geomean speedup vs the TRUE baseline
let noImprove = 0;
let bestPerCase = BASELINE_PER_CASE;
let finalWinner = null;      // {geomean, arithmetic, per_case, patch, source}
const history = { insights: [], ledger: [], rounds: [], bottleneck_now: profileSummary ? profileSummary.bottleneck : 'unknown', suggest_next: '' };

// DEEP-MODE resume: restore cumulative speedup + insight/ledger history from the prior wave so this
// continuation builds ON the cumulative best (canonical was already seeded from STATE_DIR/best by the
// director) and does not re-explore dead directions. No-op on a fresh run (prior_state undefined).
if (setup.resumed && setup.prior_state) {
  const ps = setup.prior_state;
  if (Number.isFinite(ps.cumulative) && ps.cumulative > cumulative) cumulative = ps.cumulative;
  if (Array.isArray(ps.insights)) history.insights = ps.insights;
  if (Array.isArray(ps.ledger)) history.ledger = ps.ledger;
  if (ps.bottleneck_now) history.bottleneck_now = ps.bottleneck_now;
  if (Array.isArray(ps.best_per_case) && ps.best_per_case.length) bestPerCase = ps.best_per_case;
  log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
}

while (dispatched < BUDGET && noImprove < MAX_NO_IMPROVE) {
  round++;
  const remaining = BUDGET - dispatched;
  phase('Optimize');

  // --- (a) Plan the round (TechLead) ------------------------------------
  const plan = await agentT(
    roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).', {
      EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
      BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
      CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
      KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
      ...KB_INPUTS,
    }),
    { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });

  if (!plan || plan.stop || !plan.directions || plan.directions.length === 0) {
    log(`Round ${round}: TechLead chose to stop. ${plan ? plan.reasoning || '' : ''}`);
    break;
  }

  let directions = plan.directions.slice(0, remaining).map((d, i) => ({
    ...d,
    idx: i,
    id: d.id || `r${round}_d${i}`,
    gpu_id: GPU_LIST[i % GPU_LIST.length],
    out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
  }));
  // deep_explore is a DEDICATED-ROUND, heavyweight mandate: if the plan includes one, run ONLY it this
  // round (its broad ground-up rewrite touches many files and can't be merged with specialist patches),
  // and charge DEEP_COST against the budget. Otherwise each specialist direction costs 1.
  const deepDir = directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) directions = [deepDir];
  const roundCost = directions.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  dispatched += roundCost;
  log(`Round ${round}: ${directions.length} direction(s) [${directions.map(d => d.specialty).join(', ')}], cost ${roundCost}, budget ${dispatched}/${BUDGET}`);

  // --- (b,c) Optimize -> Verify, pipelined per direction ----------------
  const results = await pipeline(
    directions,
    (d) => {
      const isDeep = d.specialty === 'deep_explore';
      // deep_explore reads its own role (broad authority + own iteration loop); specialists read engineer.md.
      const readLine = isDeep
        ? `Then Read ${WORKFLOW_DIR}/roles/deep_engineer.md and ALL knowledge files under ${WORKFLOW_DIR}/knowledge/ ` +
          `(you have broad authority — combine algorithm + memory + compute + host_runtime levers in one ` +
          `coherent rewrite), and follow them. You MAY edit ANY modifiable source (kernel + Python wrapper ` +
          `+ C++ binding), not just focus_files. Run your OWN multi-iteration measure→(self-)profile→rewrite ` +
          `loop and push to the TARGET; keep the best correct version.`
        : `Then Read ${WORKFLOW_DIR}/roles/engineer.md and ${WORKFLOW_DIR}/knowledge/self_monitoring.md and the ` +
          `knowledge files for your specialty, and follow them.`;
      return agentT(
      `You are Engineer ${d.id} (specialty=${d.specialty}) for round ${round}.
First create YOUR private workspace, then optimize.
\`\`\`bash
# Fresh, ISOLATED workspace via tar-copy that EXCLUDES build artifacts (.git/build/__pycache__/.torch_ext/
# *.so/*.o) — no 'rm' anywhere. Each engineer's out_dir is unique per (round,engineer), so the workspace
# is clean on creation; the tar excludes mean no stale build cache is ever inherited (torch .torch_ext
# stores ABSOLUTE paths, so excluding it forces each workspace to build its own fresh).
mkdir -p ${d.out_dir}/workspace
( cd ${CANONICAL} && tar --exclude=./.git --exclude='*/.git' --exclude=./build --exclude='*/build' \\
    --exclude=./__pycache__ --exclude='*/__pycache__' --exclude=./.torch_ext --exclude='*/.torch_ext' \\
    --exclude='*.so' --exclude='*.o' -cf - . ) | ( cd ${d.out_dir}/workspace && tar -xf - )
\`\`\`
${readLine} If KK_OPERATOR is non-empty, also consult the operator/language SOTA cards under
KERNEL_KNOWLEDGE_DIR per your role's "operator/language SOTA knowledge (REFERENCE ONLY)" section
(facts/how-to only; measure everything; never go below baseline).
Save best_patch.diff via \`cd <KERNEL_PATH> && git diff > ${d.out_dir}/best_patch.diff\` when geomean>1.0.

## Inputs
${cfg({
        SPECIALTY: d.specialty,
        DIRECTION: { id: d.id, title: d.title, focus_files: d.focus_files || [], expected_speedup: d.expected_speedup, prompt: d.prompt },
        ...(isDeep ? { TARGET: d.expected_speedup ? `reach ${d.expected_speedup}x (or ~90% of the roofline ceiling), whichever is the harder bar` : 'reach ~90% of the roofline ceiling' } : {}),
        KERNEL_PATH: `${d.out_dir}/workspace`,
        OUTPUT_DIR: d.out_dir,
        CANONICAL, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
        codebase_context: `${EVAL_DIR}/codebase_context.md`,
        profiling_summary: profileSummary ? profileSummary.summary_path : '',
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        ...KB_INPUTS,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.`,
      { phase: 'Optimize', label: `${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`, schema: ENG_SCHEMA }
    ).then((eng) => ({ d, eng }));
    },

    (prev) => {
      const { d, eng } = prev;
      const patch = `${d.out_dir}/best_patch.diff`;
      if (!eng || eng.status === 'failed' || !(primSpeedup(eng) > 1.0)) {
        return { d, eng, ver: null };
      }
      return agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
        }),
        { phase: 'Verify', label: `verify ${d.id}`, schema: VERIFY_SCHEMA }
      ).then((ver) => ({ d, eng, ver, patch }));
    }
  );

  const clean = results.filter(Boolean);
  const verified = clean.filter(r => r.ver && r.ver.status === 'verified' &&
    r.ver.correctness === 'pass' && primSpeedup(r.ver) > 1.0);

  // --- (d) Build candidate list; integrate if >=2 verified --------------
  // `geomean` here is the PRIMARY metric used for sorting/gating/cumulative: the time-weighted
  // ratio-of-sums when a workload spec is active, else the unweighted geomean (unchanged behavior).
  // The raw unweighted geomean is retained separately for the report.
  let candidates = verified.map(r => ({
    source: `engineer ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
    geomean: primSpeedup(r.ver), geomean_unweighted: r.ver.verified_geomean,
    weighted: r.ver.verified_weighted != null ? r.ver.verified_weighted : null,
    arithmetic: r.ver.verified_arithmetic || r.ver.verified_geomean,
    per_case: r.ver.per_case || [], patch: r.patch,
  }));

  let integrate = null;
  if (verified.length >= 2) {
    phase('Merge');
    integrate = await agentT(
      roleAgent('integrator', 'integrate', 'Combine this round\'s verified patches into one best implementation.', {
        CANONICAL, INTEGRATE_DIR: `${EVAL_DIR}/round_${round}/integrate`,
        GPU_ID: GPU_LIST[0], SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        BEST_INDIVIDUAL: Math.max(...candidates.map(c => c.geomean)),
        PATCHES: verified.map(r => ({ id: r.d.id, specialty: r.d.specialty, title: r.d.title,
          strategy: r.eng ? r.eng.strategy : '', verified_geomean: r.ver.verified_geomean,
          files: r.d.focus_files || [], patch: r.patch })),
        INSIGHTS: history.insights,
      }),
      { phase: 'Merge', label: `integrate r${round}`, schema: INTEGRATE_SCHEMA });
    const integPrim = integrate && integrate.best ? primSpeedup({
      verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean,
    }) : 0;
    if (integrate && integrate.conclusion === 'improved' && integrate.best &&
      integPrim > Math.max(...candidates.map(c => c.geomean))) {
      candidates.push({
        source: 'integrated', id: `r${round}_integrated`, title: 'integrated', specialty: 'integrate',
        geomean: integPrim, geomean_unweighted: integrate.best.geomean,
        weighted: integrate.best.weighted != null ? integrate.best.weighted : null,
        arithmetic: integrate.best.arithmetic || integrate.best.geomean,
        per_case: integrate.best.per_case || [], patch: integrate.best.patch_file,
      });
    }
  }

  candidates.sort((a, b) => b.geomean - a.geomean);
  const winner = candidates[0] || null;
  const improved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));

  // --- (e) Commit the winner into the canonical workspace ---------------
  if (improved) {
    await agentT(
      `You are the TechLead committing round ${round}'s winning patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
# Try a plain apply first, then a 3-way apply (auto-reconciles context-line drift against the blobs)
# before falling back to a manual reconstruction. --3way resolves most "patch does not apply" cases
# that are just context offsets, so the manual path is only hit on a genuine semantic conflict.
git apply ${winner.patch} || git apply --3way ${winner.patch}
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "round ${round} winner: ${winner.source} (${winner.geomean.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH \`git apply\` and \`git apply --3way\` fail, inspect the patch and apply it manually (edit the
files to match the patch's intent), then \`add -A\` + commit. The applied source is NOT guaranteed to
match the patch verbatim after a hand-merge, so after committing, RE-RUN the COMMANDMENT correctness
check (cd ${CANONICAL} && the COMMANDMENT CORRECTNESS cmd via gpu_lock); only report committed=true if
it still passes. (When a clean \`git apply\`/\`--3way\` succeeds, correctness was already verified and a
re-check is not required.) Return JSON {committed, current_best_diff, note}.`,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA });
    cumulative = winner.geomean;
    bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
    finalWinner = winner;
    noImprove = 0;

    // --- (f) Re-profile the new best ------------------------------------
    profileSummary = await agentT(
      roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_LIST[0], ROUND: round,
        COMMANDMENT, PREVIOUS_METRICS: profileSummary,
      }),
      { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
  } else {
    noImprove++;
  }

  // --- update cross-round memory (insight blackboard + hypothesis ledger)
  const mem = await agentT(
    roleAgent('tech_lead', 'update_memory', 'Distill durable insights + update the hypothesis ledger.', {
      EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR,
      ROUND_RESULTS: clean.map(r => ({ id: r.d.id, title: r.d.title, specialty: r.d.specialty,
        expected: r.d.expected_speedup, claimed: r.eng ? r.eng.speedup_geomean : 0,
        verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
        notes: r.eng ? r.eng.notes : '' })),
      INTEGRATE: integrate, WINNER: winner ? { source: winner.source, geomean: winner.geomean } : null,
      IMPROVED: improved, REPROFILE_SHIFT: profileSummary ? profileSummary.shift_note : '',
      PRIOR_HISTORY: history,
      ...(STATE_DIR ? { STATE_DIR, CANONICAL, CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase } : {}),
      ...(SHARED_KB ? { SHARED_KB, TARGET_LANGUAGE } : {}),
    }),
    { phase: 'Optimize', label: `tech_lead:memory r${round}`, schema: MEMORY_SCHEMA });
  if (mem) {
    if (mem.insights) history.insights = mem.insights;
    if (mem.ledger) history.ledger = history.ledger.concat(mem.ledger);
    if (mem.bottleneck_now) history.bottleneck_now = mem.bottleneck_now;
    if (mem.suggest_next) history.suggest_next = mem.suggest_next;
  }
  history.rounds.push({
    round,
    directions: directions.map(d => ({ id: d.id, title: d.title, specialty: d.specialty })),
    results: clean.map(r => ({ id: r.d.id, claimed: r.eng ? r.eng.speedup_geomean : 0,
      verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none') })),
    integrate: integrate ? { conclusion: integrate.conclusion, geomean: integrate.best ? integrate.best.geomean : 0 } : null,
    winner: winner ? { source: winner.source, geomean: winner.geomean } : null,
    improved, cumulative,
  });
  log(`Round ${round} done. winner=${winner ? winner.source + ' ' + winner.geomean.toFixed(2) + 'x' : 'none'}, cumulative=${cumulative.toFixed(2)}x, noImprove=${noImprove}`);
}

// ===========================================================================
// PHASE: Final report (TechLead)
// ===========================================================================
phase('Report');
const report = await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.', {
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
  }),
  { phase: 'Report', label: 'tech_lead:report', schema: REPORT_SCHEMA });

// ===========================================================================
// PHASE: Director validation + arbitration
// ===========================================================================
phase('Validate');
const validation = await agentT(
  roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
    KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_LIST[0],
    APPLY_TO_ORIGINAL, COMMANDMENT,
    FINAL_PATCH: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
    TECH_LEAD_REPORTED_GEOMEAN: report ? report.final_speedup_geomean : cumulative,
    ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
        ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
  }),
  { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });

const finalGeomean = validation ? validation.director_verified_speedup_geomean : cumulative;
// PRIMARY headline: the time-weighted speedup when workload-aligned, else the geomean (unchanged).
const finalWeighted = validation && validation.director_verified_speedup_weighted != null
  ? validation.director_verified_speedup_weighted : null;
const finalPrimary = HAS_WORKLOAD && Number.isFinite(finalWeighted) ? finalWeighted : finalGeomean;
log(`COMPLETE. ${KERNEL_NAME}: verified ${HAS_WORKLOAD ? 'time-weighted' : 'geomean'} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
    `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
    ` (status ${validation ? validation.validation_status : '?'}). Results in ${EVAL_DIR}`);

return {
  mode: MODE,
  target_language: MODE === 'author' ? TARGET_LANGUAGE : undefined,
  authored: MODE === 'author' ? true : undefined,
  eval_dir: EVAL_DIR,
  kernel_name: KERNEL_NAME,
  workload_aligned: HAS_WORKLOAD,
  final_speedup: finalPrimary,                 // PRIMARY metric (weighted when workload-aligned)
  final_weighted: finalWeighted,
  final_geomean: finalGeomean,
  final_arithmetic: validation ? validation.director_verified_speedup_arithmetic : null,
  tech_lead_reported_geomean: report ? report.final_speedup_geomean : cumulative,
  validation_status: validation ? validation.validation_status : 'unknown',
  rounds: report ? report.rounds : round,
  budget_used: dispatched,
  budget_total: BUDGET,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
};
