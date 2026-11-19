export const meta = {
  name: 'gluon-authoring',
  description: 'Expert workflow gluon_authoring: takes a kernel to Gluon on AMD CDNA (gfx942/gfx950) through the gluon_authoring method as a code-driven stage machine (entry -> budget -> transcribe -> recover -> climb -> close). Recover and climb use the kernel workflow\'s search (tech_lead plans parallel directions, budget counts directions, an integrate step stacks the winners) with the Gluon method and its evidence gates inside each direction; GEAK setup, frozen oracle, verify and director validation decide acceptance.',
  whenToUse: 'Launch this script directly to port a tuned Triton kernel to Gluon, or to climb an existing Gluon kernel, on gfx942/gfx950. Pass args.kernel_path, args.workflow_dir (the kernel_workflow dir: shared roles, GPU lock, kernel tools), args.gpu_ids, args.budget (directions, as in kernel-workflow), optionally args.deadline_epoch, args.entry_mode (auto|port|incumbent|fresh) and args.target_arch. See this directory\'s README.md.',
  phases: [
    { title: 'Setup', detail: 'director (kernel_workflow role): isolated eval dir + workspace, GPU identity' },
    { title: 'Analyze', detail: 'tech_lead PHASE=analyze (kernel_workflow role): roadmap, codebase context, perf_knowledge cards' },
    { title: 'Benchmark', detail: 'benchmark_engineer (kernel_workflow role): COMMANDMENT + frozen baseline' },
    { title: 'Profile', detail: 'gluon_profiler: gluon_profile.py baseline profile, the four dials (trace, PMC, ATT, IR)' },
    { title: 'Entry', detail: 'gluon_lead: entry mode, champion bundle, champion_gate.py (G1), track workspace' },
    { title: 'Budget', detail: 'gluon_lead: hw_budget.py, bound class and floor (no GPU)' },
    { title: 'WarmStart', detail: 'experience KB (language gluon): adopt a verified stored patch (incumbent) or read it as reference' },
    { title: 'Transcribe', detail: 'gluon_engineer: faithful anchor + equivalence (port) / first Gluon kernel (fresh)' },
    { title: 'Recover', detail: 'tech_lead plans one direction per debt suspect; gluon_engineers repay in parallel; parity_gate.py (G2) on the round winner gates the climb' },
    { title: 'Climb', detail: 'tech_lead plans 2-3 directions (Gluon layers); gluon_engineers in parallel with G3/G4 (+ATT) evidence; verify_engineer measures each' },
    { title: 'Integrate', detail: 'gluon_engineer PHASE=integrate: integrator.md strategy on the track tags, permuted screen; verify; tech_lead update_memory' },
    { title: 'Close', detail: 'gluon_lead: outcome, final report, closure self-review' },
    { title: 'Validate', detail: 'director (kernel_workflow role): independent validation vs the true baseline; update_experience, citation ledger, experience store' },
  ],
};

// The stage machine is in this file; the method (what to do inside a stage) is the gluon_authoring
// skill pack, which the roles read by path. Acceptance stays GEAK's: verify_engineer measures every
// candidate against the frozen baseline, the commit gate below needs MIN_IMPROVE over the accepted
// best, and director validates the delivered patch. A workflow script has no filesystem and no clock,
// so every artifact is produced by an agent and every gate is decided here from structured returns.
const A = args || {};
const TOKENS = typeof budget !== 'undefined' ? budget : null;
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel task directory)');
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the kernel_workflow directory ' +
    '(roles/, scripts/gpu_lock.sh, scripts/kernel_tools/).');
}
const GEAK_ROOT = WORKFLOW_DIR.replace(/\/[^/]*$/, '');
const PKG_DIR = String(A.expert_workflow_dir ||
  `${GEAK_ROOT}/perf_knowledge/expert_skills/expert_workflows/gluon_authoring`).replace(/\/+$/, '');
const SKILL = String(A.skill_dir || `${GEAK_ROOT}/perf_knowledge/expert_skills/skills/gluon_authoring`).replace(/\/+$/, '');
const KT = `${WORKFLOW_DIR}/scripts/kernel_tools`;
const EXP_ROOT = String(A.exp_root || `${GEAK_ROOT}/exp`).replace(/\/+$/, '');
const KERNEL_PATH_ORIG = String(A.kernel_path).replace(/\/+$/, '');
const KERNEL_NAME_HINT = KERNEL_PATH_ORIG.split('/').pop();
const TASK = String(A.task || '');
const EVAL_DIR_OVERRIDE = String(A.eval_dir || '');
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');

const num = (v, d) => { const n = Number(v); return v == null || v === '' || !Number.isFinite(n) ? d : n; };
const GPU_LIST = String(A.gpu_ids != null ? A.gpu_ids : '0').split(',').map(s => s.trim()).filter(Boolean);
if (!GPU_LIST.length || GPU_LIST.some(id => !/^\d+$/.test(id))) throw new Error('args.gpu_ids must be a comma list of GPU ids');
if (new Set(GPU_LIST).size !== GPU_LIST.length) throw new Error('args.gpu_ids contains duplicate GPU ids');
// Every GPU command takes one GPU from this pool through gpu_lock.sh; parallel directions queue on it.
const GPU_POOL = GPU_LIST.join(',');
if (num(A.gpus_per_job, 1) !== 1) throw new Error('gluon_authoring runs single-GPU kernels only (gpus_per_job must be 1)');

// The search knobs are kernel_lane.js's, with its defaults (tests/test_main_alignment.js checks them):
// the budget counts directions (transcribe/author 1, each recover or climb direction 1, a deep_explore
// direction DEEP_COST in a round of its own); the integrate step costs nothing.
const BUDGET = Math.max(1, Math.floor(num(A.budget, 6)));
const DEEP_COST = Math.max(1, Math.floor(num(A.deep_cost, 2)));
// Optional cap on directions per round (0 = none: tech_lead paces 2-3 within the remaining budget).
const MAX_DIRECTIONS = Math.max(0, Math.floor(num(A.max_directions, 0)));
// With a deadline, a tech_lead stop is refused while more than NO_STOP_S remain (MAX_FORCED_REPLANS times).
const NO_STOP_S = Math.max(0, num(A.no_stop_s, 900));
const MAX_FORCED_REPLANS = Math.max(1, Math.floor(num(A.max_forced_replans, 6)));
// A continuation of an earlier run on the same kernel (whose kept line is this run's baseline):
// that run's insight log / final report, read by the planner and the engineers like HISTORY.
const PRIOR_LOG = String(A.prior_log || '').trim();

// The rest of kernel_lane.js's knowledge loop, same switches and defaults (tests/test_main_alignment.js):
// perf_knowledge cards resolved by tech_lead analyze; the experience KB read at the start (warm start)
// and written at the end (update_experience, the citation ledger, the experience store). Every KB read
// and write is keyed by language 'gluon', so a Triton page never seeds a Gluon run.
const LANGUAGE = 'gluon';
const KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir || `${GEAK_ROOT}/perf_knowledge`).replace(/\/+$/, '');
const LEARNED_DIR = `${WORKFLOW_DIR}/knowledge/learned`;
const WARM_START = String(A.warm_start != null ? A.warm_start : 'on').trim().toLowerCase() || 'on';
const WARM_START_ON = WARM_START !== 'off' && WARM_START !== 'false' && WARM_START !== 'none';
const WARM_START_MIN_SPEEDUP = Number.isFinite(parseFloat(A.warm_start_min_speedup))
  ? parseFloat(A.warm_start_min_speedup) : 1.05;
const WARM_START_MATCH = ['exact', 'normalized', 'fuzzy'].includes(String(A.warm_start_match || '').trim())
  ? String(A.warm_start_match).trim() : 'fuzzy';
const UPDATE_EXPERIENCE = String(A.update_experience != null ? A.update_experience : 'on').trim().toLowerCase() || 'on';
const UPDATE_EXPERIENCE_ON = UPDATE_EXPERIENCE !== 'off' && UPDATE_EXPERIENCE !== 'false' && UPDATE_EXPERIENCE !== 'none';
// A contended box measured its neighbours, and a held-out kernel is the instrument that measures the
// KB itself: neither is distilled (kernel_lane.js).
const BOX_QUIET = String(A.box_quiet != null ? A.box_quiet : 'true') === 'true';
const HELD_OUT = String(A.held_out != null ? A.held_out : 'false') === 'true';
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir || `${GEAK_ROOT}/kb_artifacts`).replace(/\/+$/, '');
const EXPERIENCE_STORE = `${WORKFLOW_DIR}/scripts/experience_store.py`;
const KB_MODE = String(A.kb_mode || 'store').trim().toLowerCase() === 'local' ? 'local' : 'store';
const KB_STORE_DIR = String(A.kb_store_dir ||
  (KB_ARTIFACTS_DIR ? KB_ARTIFACTS_DIR.replace(/\/[^/]*$/, '') + '/kb_store_local' : '')).replace(/\/+$/, '');
const KB_FRAMEWORK_VERSION = String(A.kb_framework_version || '').trim();
const KB_VERSION_FLAG = KB_FRAMEWORK_VERSION ? ` --framework-version ${JSON.stringify(KB_FRAMEWORK_VERSION)}` : '';
const KB_ROOT_OK = KB_MODE === 'store' ? !!KB_STORE_DIR : !!KB_ARTIFACTS_DIR;
const KB_WRITE_OK = KB_ROOT_OK && !!KB_ARTIFACTS_DIR;
const KB_REMOTE = String(A.kb_remote || 'auto').trim().toLowerCase() === 'off' ? 'off' : 'auto';
// kernel_lane.js's credential prelude, verbatim: the commands read the KB service credentials themselves.
const KB_ENV_PRELUDE =
  'export KB_STORE_URL="${GEAK_KB_STORE_URL:-${KB_STORE_URL:-https://global.primus-safe.amd.com/knowledge-base}}"; ' +
  'export KB_STORE_TOKEN="${GEAK_KB_STORE_TOKEN:-${KB_STORE_TOKEN:-$(cat ~/.geak_kb_token 2>/dev/null)}}"; ' +
  'if [ -z "${SSL_CERT_FILE:-}" ]; then for _ca in "${KB_CA_BUNDLE:-}" ' +
  '/shared_nfs/hyperloom/ca/amd-ca-combined.pem "$HOME/amd-extra-ca-bundle.pem"; do ' +
  '[ -n "$_ca" ] && [ -r "$_ca" ] && { export SSL_CERT_FILE="$_ca" REQUESTS_CA_BUNDLE="$_ca" ' +
  'CURL_CA_BUNDLE="$_ca" NODE_EXTRA_CA_CERTS="$_ca"; break; }; done; fi; ';
const STORAGE_RECLAIM = String(A.storage_reclaim != null ? A.storage_reclaim : 'on') !== 'off';
// kernel_workflow/knowledge/learned/ for the planner, with kernel_lane.js's budget on how much of a
// round it may steer: at most KB_DIR_CAP directions cite cards, one direction is profile-only.
const USE_LEARNED_READ = String(A.use_learned_kb != null ? A.use_learned_kb : 'true') === 'true';
const KB_DIR_CAP = Math.max(0, Math.floor(num(A.kb_dir_cap, 1)));
const KB_COLD_DIRECTION = String(A.kb_cold_direction != null ? A.kb_cold_direction : 'true') === 'true';
const ENTRY_HINT = String(A.entry_mode || 'auto').trim().toLowerCase();
if (!['auto', 'port', 'incumbent', 'fresh'].includes(ENTRY_HINT)) {
  throw new Error(`args.entry_mode must be auto, port, incumbent or fresh, got '${ENTRY_HINT}'`);
}
const TARGET_ARCH = String(A.target_arch || '').trim().toLowerCase();
const SUPPORTED_GFX = ['gfx942', 'gfx950'];
if (TARGET_ARCH && !SUPPORTED_GFX.includes(TARGET_ARCH)) {
  throw new Error(`args.target_arch must be gfx942 or gfx950, got '${TARGET_ARCH}'`);
}
const MIN_IMPROVE = Math.max(0, num(A.min_improve, 0.02));
const PARITY_THRESHOLD = num(A.parity_threshold, 0.95);
const MAX_RECOVER_ROUNDS = Math.max(1, Math.floor(num(A.max_recover_rounds, Math.ceil(BUDGET / 2))));
// How many times a round with incomplete evidence is sent back before it is rejected.
const EVIDENCE_REASKS = Math.max(0, Math.floor(num(A.evidence_reasks, 1)));
const EVIDENCE_AUDIT = String(A.evidence_audit != null ? A.evidence_audit : 'true') === 'true';
const DEADLINE_EPOCH = Math.max(0, Math.floor(num(A.deadline_epoch, 0)));
// Seconds kept free at the end for the last verify, Close and Validate.
const DEADLINE_RESERVE_S = Math.max(0, num(A.deadline_reserve_s, 1800));
const ROUND_TIME_CAP_S = Math.max(600, num(A.round_time_cap_s, 3600));
const TOKEN_RESERVE = Math.max(0, num(A.token_reserve, 150000));
const AGENT_TIMEOUT_MS = Math.max(0, num(A.agent_timeout_ms, 3600000));
const ENGINEER_TIMEOUT_MS = Math.max(0, num(A.engineer_timeout_ms, 7200000));

// ---------------------------------------------------------------------------
// The method's contract, as data: required references per stage (relative to SKILL) and the layers.
// ---------------------------------------------------------------------------
const STAGE_REFS = {
  entry: [
    ['skill.md', 'the whole file: entry modes, gates, Roles'],
    ['references/method/entry.md', '## 4. Which entry mode; ## Stage-Entry: the champion assertion; ## 6. The champion bundle contract'],
  ],
  budget: [
    ['references/method/budget.md', '## 1. Your hardware budget; ## Roofline method; ## Record'],
    ['references/workloads/index.md', 'the archetype row hw_budget.py prints'],
  ],
  profile: [
    ['references/method/profile.md', '## 3.1 Required evidence — the four dials, every round'],
  ],
  transcribe: [
    ['references/method/transcribe.md', 'the whole stage: recover, apply, compile, ## 5. Verify — the equivalence gate'],
    ['references/tile-programming/layout-recipes.md', 'only for a layout you have to name'],
  ],
  author: [
    ['references/gluon/index.md', 'where to start, gfx950 vs gfx942'],
    ['references/gluon/gfx950-minimal-examples.md', 'the skeleton closest to this operator'],
  ],
  recover: [
    ['references/method/recover.md', '## Stage-Recover; ## 2. Attribute; ## 3. Repay by hand; ## 4. The parity gate'],
    ['references/tile-programming/pipeline.md', 'the hand-written pipeline order (lost_pipeline)'],
  ],
  climb: [
    ['references/method/climb.md', '## Layer Backbone; ## 2. Reversed-intuition traps; ## 3. The round loop'],
    ['references/method/profile.md', '## 3.1 Required evidence — the four dials, every round'],
    ['references/method/benchmark-hygiene.md', '## The binding rule: GEAK\'s frozen task contract owns acceptance'],
  ],
  integrate: [
    ['references/method/climb.md', '## Layer Backbone (the gating order: which layers compose); ## 3. The round loop'],
    ['references/method/benchmark-hygiene.md', '## The binding rule: GEAK\'s frozen task contract owns acceptance'],
  ],
  close: [
    ['references/method/close.md', '## Stop conditions; ## Closure challenge; ## final_report.json (schema)'],
  ],
  audit: [
    ['references/method/index.md', '### The four gates'],
  ],
};
const LAYER_REFS = {
  '1.5': ['references/tile-programming/scheduling-model.md', 'references/tile-programming/warp-pipeline.md'],
  '2': ['references/tile-programming/memory-path.md', 'references/gluon/memory-reference.md'],
  '3': ['references/tile-programming/layout-recipes.md', 'references/gluon/smem-lds-reference.md'],
  '4': ['references/tile-programming/pipeline.md', 'references/gluon/pipeline-reference.md'],
  '4+': ['references/tile-programming/instruction-scheduling.md', 'references/gluon/inline-asm-reference.md'],
  '5': ['references/tile-programming/slicing.md'],
  '6': ['references/workloads/gemm.md'],
  '7': ['references/tile-programming/low-precision.md', 'references/gluon/matrix-reference.md'],
};
const LAYERS = Object.keys(LAYER_REFS);
// Layers whose claim is about issue order or overlap: they owe instruction-level (ATT) evidence.
const PIPELINE_LAYERS = new Set(['1.5', '4', '4+']);
// A debt suspect is repaid at one layer (recover.md ## 3).
const SUSPECT_LAYER = { lost_pipeline: '4', lost_layout: '3', lost_RA: '5' };
// The gating order of climb.md ## Layer Backbone, applied to the directions of one round (they all
// start from the same kept line): a layer and a layer it gates never share a round (1.5 before 4 and 5,
// 3 before 4, 4 before 4+), and the two layer-1.5 scheduling models are never two arms. A coupled pair
// whose halves do not pay alone is one `paired` direction instead.
const GATES = { '1.5': ['1.5', '4', '4+', '5'], '3': ['4'], '4': ['4+'] };
const gates = (a, b) => (GATES[a] || []).includes(b);

const refPath = (rel) => `${SKILL}/${rel}`;
function requiredRefs(stage, layer) {
  const base = (STAGE_REFS[stage] || []).map(([rel]) => rel);
  const extra = layer && LAYER_REFS[layer] ? LAYER_REFS[layer] : [];
  return [...new Set([...base, ...extra])];
}
// The block that opens every method-stage prompt, before the role file is even named.
function readingBlock(stage, layer) {
  const rows = (STAGE_REFS[stage] || []).map(([rel, what], i) => `${i + 1}. ${refPath(rel)} — ${what}`);
  if (layer && LAYER_REFS[layer]) {
    for (const rel of LAYER_REFS[layer]) rows.push(`${rows.length + 1}. ${refPath(rel)} — layer ${layer}`);
  }
  if (!rows.length) return '';
  // Climb and recover pick their layer from evidence, so every layer's files are listed; the ones for
  // the layer the round works on are required.
  const perLayer = (stage === 'climb' || stage === 'recover') && !layer
    ? `For the layer this round works on, also read (required for that layer):\n` +
      LAYERS.map(l => `- layer ${l}: ${LAYER_REFS[l].map(refPath).join(', ')}`).join('\n') + '\n'
    : '';
  return `## Required reading — open these before running any command\n${rows.join('\n')}\n${perLayer}` +
    `Read the named sections (grep the heading, or \`python3 ${SKILL}/scripts/toolctl.py context query ` +
    `--markdown-heading "<heading>"\`), not whole trees. List every path you opened in \`references_read\`.\n`;
}

const cfg = (o) => Object.entries(o).filter(([, v]) => v !== undefined)
  .map(([k, v]) => `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

function rolePrompt(role, roleFile, phase, intro, inputs, opts) {
  const o = opts || {};
  const reading = o.stage ? readingBlock(o.stage, o.layer) : '';
  return `You are the ${role}. PHASE=${phase}.
${reading ? `\n${reading}\n` : ''}Then Read ${roleFile} and follow its instructions for PHASE=${phase}.
Do all filesystem/shell work yourself (Bash/Read/Write). Every GPU command runs under
\`bash ${WORKFLOW_DIR}/scripts/gpu_lock.sh <GPU_ID> <cmd>\`; never set HIP_VISIBLE_DEVICES yourself. ${intro}

## Inputs
${cfg(inputs)}
${o.tail ? `\n${o.tail}\n` : ''}
Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
}
const geakRole = (role) => `${WORKFLOW_DIR}/roles/${role}.md`;
const pkgRole = (role) => `${PKG_DIR}/roles/${role}.md`;

// Hang guard: a call that never returns resolves null so the run can still close. A timed-out agent
// is not killed, which is why the engineer's track is re-pinned to a tag at the start of every round.
async function agentT(prompt, opts, timeoutMs) {
  const ms = timeoutMs != null ? timeoutMs : AGENT_TIMEOUT_MS;
  try {
    if (typeof setTimeout !== 'function' || !(ms > 0)) return await agent(prompt, opts);
    let to;
    const guard = new Promise((resolve) => {
      to = setTimeout(() => { log(`  [hung-agent guard] ${opts.label} gave no return in ${Math.round(ms / 60000)} min.`); resolve(null); }, ms);
    });
    return await Promise.race([agent(prompt, opts).then((r) => { clearTimeout(to); return r; },
      (e) => { clearTimeout(to); throw e; }), guard]);
  } catch (e) {
    log(`  [agent error] ${opts.label}: ${String(e && e.message || e).slice(0, 200)}`);
    return null;
  }
}

// No clock in a workflow script: a tiny agent reads it. An unreadable clock never truncates a run.
const CLOCK_SCHEMA = { type: 'object', properties: { epoch: { type: 'number' } }, required: ['epoch'] };
async function secondsLeft(tag) {
  if (!DEADLINE_EPOCH) return { left: Infinity, now: 0 };
  const r = await agentT('Run exactly `date +%s` and nothing else. Return {"epoch": <the integer it printed>}.',
    { phase: 'Climb', label: `clock ${tag}`, effort: 'low', schema: CLOCK_SCHEMA }, 300000);
  if (!r || !Number.isFinite(r.epoch)) {
    log(`  [deadline] clock read failed at ${tag}; treating time as unlimited for this check.`);
    return { left: Infinity, now: 0 };
  }
  return { left: DEADLINE_EPOCH - r.epoch, now: r.epoch };
}

// ---------------------------------------------------------------------------
// Schemas. Required lists stay short on purpose: completeness is checked in code (missingEvidence),
// so an incomplete return is re-asked by name instead of being forced into a plausible-looking shape.
// ---------------------------------------------------------------------------
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const anyObj = { type: 'object', additionalProperties: true };
const strs = { type: 'array', items: { type: 'string' } };
const SETUP_SCHEMA = obj({
  eval_dir: { type: 'string' }, workspace: { type: 'string' }, kernel_name: { type: 'string' },
  device_gfx: { type: 'string' }, device_target: { type: 'string' }, physical_cu_count: { type: 'number' },
  baseline_frozen: { type: 'boolean' }, baseline_callable: { type: 'string' }, notes: { type: 'string' },
}, ['eval_dir', 'workspace', 'kernel_name', 'device_gfx']);
const BENCH_SCHEMA = obj({
  commandment_path: { type: 'string' }, baseline_per_case: { type: 'array', items: anyObj },
  baseline_geomean_ms: { type: 'number' }, reliable: { type: 'boolean' }, notes: { type: 'string' },
}, ['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']);
const PROFILE_SCHEMA = obj({
  bottleneck: { type: 'string' }, binding: { type: 'string' }, command: { type: 'string' },
  exit_code: { type: ['number', 'null'] }, profile_dir: { type: 'string' }, profile_json: { type: 'string' },
  layers: anyObj, degraded_reason: { type: 'string' }, case: { type: 'string' },
  summary_path: { type: 'string' }, key_metrics: anyObj, top_opportunities: strs, references_read: strs,
}, ['bottleneck', 'command', 'exit_code', 'profile_dir', 'layers', 'references_read']);
const GATE_RUN = obj({
  command: { type: 'string' }, exit_code: { type: ['number', 'null'] }, output_path: { type: 'string' },
  flags: strs, not_applicable_reason: { type: 'string' },
});
const ENTRY_SCHEMA = obj({
  entry_mode: { type: 'string', enum: ['port', 'incumbent', 'fresh'] },
  source_language: { type: 'string' }, baseline_language: { type: 'string' },
  source_is_anchor: { type: 'boolean' },
  champion_source: { type: 'string' }, champion_bundle: { type: 'string' }, champion_patch: { type: 'string' },
  champion_gate: GATE_RUN, track_ready: { type: 'boolean' },
  references_read: strs, caveats: strs, notes: { type: 'string' },
}, ['entry_mode', 'champion_gate', 'track_ready', 'references_read']);
const BUDGET_SCHEMA = obj({
  command: { type: 'string' }, exit_code: { type: ['number', 'null'] }, budget_json: { type: 'string' },
  sku: { type: 'string' }, workload: { type: 'string' }, bound_class: { type: 'string' },
  bound_direction: { type: 'string' }, floor_ms: { type: ['number', 'null'] },
  multiple_over_floor: { type: ['number', 'null'] }, archetype_layers: strs, references_read: strs,
}, ['command', 'exit_code', 'references_read']);
const ROUND_SCHEMA = obj({
  stage: { type: 'string', enum: ['transcribe', 'author', 'recover', 'climb'] },
  layer: { type: 'string' }, suspect: { type: 'string' }, hypothesis: { type: 'string' },
  change_summary: { type: 'string' }, no_change: { type: 'boolean' }, anchor_provided: { type: 'boolean' },
  references_read: strs, track_tag: { type: 'string' }, patch_file: { type: 'string' },
  correctness: { type: 'string', enum: ['pass', 'fail', 'not_run'] },
  profile_dir: { type: 'string' }, profile_command: { type: 'string' },
  g3_occupancy: obj({
    command: { type: 'string' }, exit_code: { type: ['number', 'null'] }, ir_dir: { type: 'string' },
    output_path: { type: 'string' }, vgpr_wg_per_cu: { type: ['number', 'null'] },
    lds_wg_per_cu: { type: ['number', 'null'] }, binding: { type: 'string' }, missing_reason: { type: 'string' },
  }),
  g4_screen: obj({
    command: { type: 'string' }, exit_code: { type: ['number', 'null'] }, json_path: { type: 'string' },
    delta_pct: { type: ['number', 'null'] }, noise_band_pct: { type: ['number', 'null'] },
    resolved: { type: 'boolean' }, valid: { type: 'boolean' },
  }),
  att: obj({
    status: { type: 'string', enum: ['captured', 'unavailable', 'not_required'] }, tool: { type: 'string' },
    command: { type: 'string' }, exit_code: { type: ['number', 'null'] }, output_dir: { type: 'string' },
    state: { type: 'string' }, reason: { type: 'string' },
  }),
  parity_gate: obj({
    command: { type: 'string' }, exit_code: { type: ['number', 'null'] }, ratio: { type: ['number', 'null'] },
    suspects: { type: 'array', items: anyObj }, json_path: { type: 'string' },
    mode: { type: 'string', enum: ['gate', 'diagnostic'] },
  }),
  equivalence: obj({
    layout_diff: { type: 'string' }, numeric: { type: 'string' }, determinism: { type: 'string' },
    asm_parity: { type: 'string' }, output_path: { type: 'string' },
  }),
  edge_shapes: obj({
    command: { type: 'string' }, exit_code: { type: ['number', 'null'] }, json_path: { type: 'string' },
    baseline: { type: 'string' }, shared_limitation: { type: 'string' },
  }),
  pair_screen: obj({
    prerequisite_json: { type: 'string' }, prerequisite_delta_pct: { type: ['number', 'null'] },
    pair_json: { type: 'string' }, pair_delta_pct: { type: ['number', 'null'] },
    outcome: { type: 'string', enum: ['pair', 'prerequisite', 'neither'] },
  }),
  inline_asm: { type: 'boolean' }, disassembly_ref: { type: 'string' },
  enabling_step: { type: 'boolean' }, claim: { type: 'string' }, lesson: { type: 'string' },
  resweep_request: { type: 'string' }, structure_suspect: { type: 'string' }, notes: { type: 'string' },
}, ['stage', 'references_read', 'patch_file', 'correctness']);
const AUDIT_SCHEMA = obj({ ok: { type: 'boolean' }, findings: strs, checked: strs }, ['ok', 'findings']);
// kernel_lane.js PLAN_SCHEMA, plus the Gluon fields of a direction (layer, suspect, paired).
const SPECIALTIES = ['algorithm', 'memory', 'compute', 'host_runtime', 'deep_explore'];
const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: SPECIALTIES },
      focus_files: strs, expected_speedup: { type: 'number' }, prompt: { type: 'string' },
      kk_refs: strs, learned_refs: strs,
      layer: { type: 'string' }, suspect: { type: 'string' }, paired: { type: 'boolean' },
    }, ['id', 'title', 'specialty', 'prompt']),
  },
}, ['stop', 'directions']);
// kernel_lane.js MEMORY_SCHEMA (tech_lead PHASE=update_memory).
const MEMORY_SCHEMA = obj({
  insights: strs, ledger: { type: 'array', items: anyObj },
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);
// kernel_lane.js INTEGRATE_SCHEMA, plus what the track and verify need from an integrated candidate.
const INTEGRATE_SCHEMA = obj({
  attempted: { type: 'boolean' }, combos_tried: { type: 'array', items: anyObj }, best: anyObj,
  improved_over_best_individual: { type: 'boolean' }, conclusion: { type: 'string' }, notes: { type: 'string' },
  references_read: strs, track_tag: { type: 'string' }, patch_file: { type: 'string' },
  correctness: { type: 'string', enum: ['pass', 'fail', 'not_run'] },
  g4_screen: ROUND_SCHEMA.properties.g4_screen, parity_gate: ROUND_SCHEMA.properties.parity_gate,
}, ['attempted', 'conclusion']);
// kernel_lane.js ANALYZE_SCHEMA (tech_lead PHASE=analyze).
const ANALYZE_SCHEMA = obj({
  kernel_type: { type: 'string' }, kernel_file: { type: 'string' }, entry_point: { type: 'string' },
  modifiable_files: strs, bottleneck_guess: { type: 'string' }, roadmap_summary: { type: 'string' },
  candidate_directions: { type: 'array', items: anyObj },
  kk_operator: { type: ['string', 'null'] }, kk_language: { type: ['string', 'null'] }, kk_refs: strs,
}, ['kernel_type', 'roadmap_summary']);
// kernel_lane.js experience-store schemas (experience_store.py JSON, verbatim).
const WARMSTART_RESOLVE_SCHEMA = obj({
  read_reason: { type: 'string' }, slug: { type: 'string' }, requested_slug: { type: 'string' },
  match_tier: { type: 'string' }, canonical_id: { type: 'string' }, read_plane: { type: 'string' },
  other_language_pages: strs, filtered: anyObj,
  candidates: { type: 'array', items: obj({
    rank: { type: 'number' }, slug: { type: 'string' }, speedup: { type: 'number' }, exp_dir: { type: 'string' },
    session_id: { type: 'string' }, patch_path: { type: 'string' }, prose_path: { type: 'string' },
    direction: { type: 'string' },
  }, ['rank', 'patch_path']) },
}, ['read_reason']);
const WARMSTART_WRITE_SCHEMA = obj({
  written: { type: 'boolean' }, reason: { type: 'string' }, slug: { type: 'string' }, dir: { type: 'string' },
  speedup: { type: 'number' }, remote: anyObj,
}, ['written']);
const UPDATE_EXPERIENCE_SCHEMA = obj({
  action: { type: 'string' }, card_path: { type: 'string' }, key: { type: 'string' }, note: { type: 'string' },
}, []);
const ADOPT_SCHEMA = obj({ committed: { type: 'boolean' }, patch_file: { type: 'string' }, note: { type: 'string' } }, ['committed']);
const VERIFY_SCHEMA = obj({
  status: { type: 'string' }, correctness: { type: 'string' }, verified_geomean: { type: 'number' },
  verified_weighted: { type: 'number' }, per_case: { type: 'array', items: anyObj }, notes: { type: 'string' },
}, ['status', 'verified_geomean']);
const CLOSE_SCHEMA = obj({
  report_path: { type: 'string' }, final_report_json: { type: 'string' }, final_patch: { type: 'string' },
  closure_review_path: { type: 'string' }, strongest_counter_evidence: { type: 'string' },
  deferred: strs, references_read: strs, notes: { type: 'string' },
}, ['report_path', 'references_read']);
const VALIDATE_SCHEMA = obj({
  director_verified_speedup_geomean: { type: 'number' }, director_verified_speedup_weighted: { type: 'number' },
  validation_status: { type: 'string' }, correctness: { type: 'string' }, arbitration_note: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status']);

const says = (v, w) => String(v == null ? '' : v).toLowerCase() === w;
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const has = (v) => typeof v === 'string' && v.trim().length > 0;
// The primary verified metric: time-weighted when the oracle is workload-weighted, else the geomean.
const primary = (v) => (v && isNum(v.verified_weighted) && v.verified_weighted > 0) ? v.verified_weighted
  : (v && isNum(v.verified_geomean) ? v.verified_geomean : 0);

// ---------------------------------------------------------------------------
// Evidence completeness, per stage. Returns the names of what is missing; empty means complete.
// ---------------------------------------------------------------------------
function readsCover(stage, layer, read) {
  const got = (Array.isArray(read) ? read : []).map(String);
  return requiredRefs(stage, layer).filter(rel => !got.some(p => p.endsWith(rel)))
    .map(rel => `references_read: ${refPath(rel)}`);
}
function g3Missing(g) {
  if (!g || typeof g !== 'object') return ['g3_occupancy (probe.py measure on this candidate\'s IR)'];
  const out = [];
  if (!/probe\.py/.test(g.command || '') || !/\bmeasure\b/.test(g.command || '')) out.push('g3_occupancy.command (probe.py measure --dir <ir> --arch <gfx>)');
  if (g.exit_code !== 0) out.push('g3_occupancy.exit_code 0');
  if (!has(g.ir_dir)) out.push('g3_occupancy.ir_dir');
  for (const k of ['vgpr_wg_per_cu', 'lds_wg_per_cu']) {
    if (!(k in g) || (g[k] === null && !has(g.missing_reason))) out.push(`g3_occupancy.${k} (or missing_reason)`);
  }
  if (!has(g.binding)) out.push('g3_occupancy.binding');
  return out;
}
function g4Missing(g) {
  if (!g || typeof g !== 'object') return ['g4_screen (ab_bench.py --permute)'];
  const out = [];
  if (!/ab_bench\.py/.test(g.command || '') || !/--permute/.test(g.command || '')) out.push('g4_screen.command (ab_bench.py ... --permute --json)');
  if (!isNum(g.exit_code)) out.push('g4_screen.exit_code');
  if (!has(g.json_path)) out.push('g4_screen.json_path');
  if (typeof g.resolved !== 'boolean') out.push('g4_screen.resolved');
  return out;
}
// The ATT a round cites: gluon_profile.py with its att layer (the pack's entry), or the shared
// capture.sh / rocprofv3_safe.sh --att directly. A capture whose collection state is not `ok`
// (rocprofv3_safe.json: partial, kernel identity unverified, ...) is evidence only with a reason.
const ATT_CMD = 'gluon_profile.py (att layer on), capture.sh without --no-att, or rocprofv3_safe.sh --att';
// The round's own gluon_profile.py call collects ATT unless its --layers leaves att out.
const profileHasAtt = (cmd) => {
  const layersFlag = /--layers[ =](\S+)/.exec(cmd || '');
  return /gluon_profile\.py/.test(cmd || '') && (!layersFlag || /\batt\b/.test(layersFlag[1]));
};
function attMissing(a, r) {
  if (!a || typeof a !== 'object' || a.status === 'not_required') return [`att (${ATT_CMD}; or status unavailable with a reason)`];
  if (a.status === 'captured') {
    const out = [];
    const cmd = a.command || '';
    // ATT that came out of the round's re-profile may cite it by where it landed: an output_dir
    // inside profile_dir, with profile_command the gluon_profile.py call (att layer on).
    const fromRoundProfile = r && has(r.profile_dir) && has(a.output_dir) &&
      String(a.output_dir).startsWith(String(r.profile_dir).replace(/\/+$/, '')) && profileHasAtt(r.profile_command);
    // A cited gluon_profile.py call speaks for itself (its --layers may leave att out).
    const viaProfile = /gluon_profile\.py/.test(cmd) ? profileHasAtt(cmd) : fromRoundProfile;
    const viaCapture = /capture\.sh/.test(cmd) && !/--no-att|--lite|--isa-only/.test(cmd);
    const viaSafe = /rocprofv3_safe\.sh/.test(cmd) && /--att/.test(cmd);
    if (!viaProfile && !viaCapture && !viaSafe) out.push(`att.command (${ATT_CMD})`);
    if (!has(a.output_dir)) out.push('att.output_dir');
    if (a.state !== 'ok' && (a.reason || '').trim().length < 20) {
      out.push('att.state (ok from rocprofv3_safe.json / profile.json layers.att; otherwise the reason it is not)');
    }
    return out;
  }
  if (a.status === 'unavailable') return (a.reason || '').trim().length >= 20 ? [] : ['att.reason (why ATT is unavailable, e.g. capture.json att_state=decoder_absent)'];
  return ['att.status'];
}
function missingEvidence(stage, r, ctx) {
  if (!r) return ['the whole return'];
  const out = [];
  const layer = stage === 'recover' ? (SUSPECT_LAYER[r.suspect] || r.layer) : r.layer;
  if (r.stage !== stage) out.push(`stage=${stage}`);
  out.push(...readsCover(stage, stage === 'climb' || stage === 'recover' ? layer : null, r.references_read));
  if (!r.no_change && !r.anchor_provided && !has(r.patch_file)) out.push('patch_file');
  if (!r.no_change && !has(r.track_tag)) out.push('track_tag');
  if (stage === 'transcribe') {
    const e = r.equivalence || {};
    if (!['PASS', 'RECONCILED', 'FAIL'].includes(String(e.layout_diff || '').toUpperCase())) out.push('equivalence.layout_diff (ttgir_bridge.py verify: PASS|RECONCILED|FAIL)');
    for (const k of ['numeric', 'determinism', 'asm_parity']) if (!has(e[k])) out.push(`equivalence.${k}`);
    out.push(...g3Missing(r.g3_occupancy));
  }
  if (stage === 'author') out.push(...g3Missing(r.g3_occupancy));
  if (stage === 'transcribe' || stage === 'author') out.push(...edgeMissing(r.edge_shapes, true));
  else if (!r.no_change && r.edge_shapes) out.push(...edgeMissing(r.edge_shapes, false));
  if (stage === 'recover') {
    const p = r.parity_gate;
    if (!p || !/parity_gate\.py/.test(p.command || '') || !isNum(p.exit_code)) out.push('parity_gate (parity_gate.py command and exit_code)');
    else if (p.mode !== 'gate') out.push('parity_gate.mode=gate (a recover round runs it as the gate)');
    if (!r.no_change) {
      if (!SUSPECT_LAYER[r.suspect]) out.push('suspect (lost_pipeline|lost_layout|lost_RA)');
      out.push(...g3Missing(r.g3_occupancy), ...g4Missing(r.g4_screen));
      if (PIPELINE_LAYERS.has(layer)) out.push(...attMissing(r.att, r));
    }
  }
  if (stage === 'climb') {
    if (!LAYERS.includes(String(r.layer))) out.push(`layer (one of ${LAYERS.join(', ')})`);
    if (!has(r.profile_dir)) out.push('profile_dir (this round\'s re-profile)');
    if (!/gluon_profile\.py/.test(r.profile_command || '')) out.push('profile_command (the gluon_profile.py call behind profile_dir, with --compare LAST_PROFILE)');
    out.push(...g3Missing(r.g3_occupancy), ...g4Missing(r.g4_screen));
    if (PIPELINE_LAYERS.has(String(r.layer))) out.push(...attMissing(r.att, r));
    if (r.inline_asm && !has(r.disassembly_ref)) out.push('disassembly_ref (inline asm is accepted on the disassembly)');
    if (ctx.entry === 'incumbent' && r.parity_gate && r.parity_gate.mode === 'gate') out.push('parity_gate.mode=diagnostic (an incumbent has no parity gate)');
  }
  // A paired direction (a prerequisite + the lever it makes legal) shows that the pair, not one half, is the win.
  if (!r.no_change && ((ctx.direction && ctx.direction.paired) || r.paired === true)) out.push(...pairMissing(r.pair_screen));
  return out;
}
// Boundary-shape correctness (edge_shapes.py): required where masks are authored (transcribe,
// author), checked wherever it is reported. Exit 1 = the candidate fails a shape its baseline
// passes; exit 3 = the baseline fails it too (a shared limitation, carried as a caveat).
function edgeMissing(e, required) {
  if (!e || typeof e !== 'object') return required ? ['edge_shapes (edge_shapes.py on the candidate and its baseline)'] : [];
  if (!/edge_shapes\.py/.test(e.command || '') || !isNum(e.exit_code)) return ['edge_shapes.command (edge_shapes.py) and exit_code'];
  if (e.exit_code === 1) return ['edge_shapes: exit 1 -- the candidate fails boundary shapes its baseline passes (fix the masks)'];
  if (![0, 3].includes(e.exit_code)) return [`edge_shapes.exit_code ${e.exit_code} (0 clean | 3 shared limitation)`];
  return [];
}
function pairMissing(s) {
  if (!s || typeof s !== 'object') return ['pair_screen (ab_bench.py --permute: the prerequisite alone and the pair, each vs TRACK_START)'];
  const out = [];
  for (const k of ['prerequisite_json', 'pair_json']) if (!has(s[k])) out.push(`pair_screen.${k}`);
  for (const k of ['prerequisite_delta_pct', 'pair_delta_pct']) if (!isNum(s[k])) out.push(`pair_screen.${k}`);
  if (!['pair', 'prerequisite', 'neither'].includes(s.outcome)) {
    out.push('pair_screen.outcome (pair | prerequisite | neither: which half the returned candidate is)');
  }
  return out;
}
// An integrated candidate owes the track tag and patch, its correctness, the permuted screen against the
// best individual and, below the parity gate, G2 on itself. A return that improved nothing owes nothing.
function integrateMissing(stage, x) {
  if (!x) return ['the whole return'];
  if (x.conclusion !== 'improved') return [];
  const out = readsCover('integrate', null, x.references_read);
  if (!has(x.track_tag)) out.push('track_tag');
  if (!has(x.patch_file)) out.push('patch_file');
  if (x.correctness !== 'pass') out.push('correctness pass (COMMANDMENT correctness on the integrated candidate)');
  out.push(...g4Missing(x.g4_screen));
  if (stage === 'recover') {
    const p = x.parity_gate;
    if (!p || !/parity_gate\.py/.test(p.command || '') || !isNum(p.exit_code) || p.mode !== 'gate') out.push('parity_gate (parity_gate.py on the integrated candidate, mode=gate)');
  }
  return out;
}

// ===========================================================================
// PHASE: Setup / Benchmark — GEAK's own roles, unchanged. Profile — the pack's gluon_profiler.
// ===========================================================================
phase('Setup');
const setup = await agentT(
  rolePrompt('director', geakRole('director'), 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK, MODE: 'optimize',
    WORKFLOW_DIR, SKILL_DIR: WORKFLOW_DIR, EXPECTED_GFX: TARGET_ARCH, EXPECTED_TARGET: '',
    EXPECTED_DEVICE_NAME: '', EXPECTED_PHYSICAL_CU_COUNT: 0,
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;
const GFX = String(setup.device_gfx || '').trim().toLowerCase();
if (!SUPPORTED_GFX.includes(GFX)) throw new Error(`gluon_authoring supports gfx942/gfx950 only; detected '${GFX || 'unknown'}'`);
if (TARGET_ARCH && GFX !== TARGET_ARCH) throw new Error(`GPU architecture mismatch: target_arch ${TARGET_ARCH}, detected ${GFX}`);
const KERNEL_NAME = String(setup.kernel_name || KERNEL_NAME_HINT).replace(/_task$/, '');
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
const GLUON_DIR = `${EVAL_DIR}/gluon`;
const TRACK = `${GLUON_DIR}/track`;
if (!(setup.baseline_frozen === true || has(setup.baseline_callable))) {
  log('Setup ABORT: no frozen baseline (baseline_src/ or meta.baseline_callable).');
  return { workflow: 'gluon_authoring', eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
    outcome: 'negative_keep_baseline', blocked: 'no_frozen_baseline', final_patch: '' };
}
log(`Setup done. EVAL_DIR=${EVAL_DIR} GFX=${GFX}`);

// kernel_lane.js's tech_lead analyze, before Benchmark as there: the roadmap, codebase_context.md and
// the perf_knowledge cards (KK_*) the planner and every engineer read.
phase('Analyze');
const analysis = await agentT(
  rolePrompt('tech_lead', geakRole('tech_lead'), 'analyze', 'Analyze the kernel and write the roadmap.', {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR, KERNEL_KNOWLEDGE_DIR, GFX,
    DEVICE_TARGET: setup.device_target || '', PHYSICAL_CU_COUNT: setup.physical_cu_count || 0,
  }, { tail: '## This run\nThe gluon_authoring expert workflow keeps this kernel in Gluon ' +
    `(${SKILL}): resolve the perf_knowledge cards for the operator with kk_language gluon; your ` +
    'candidate_directions are advisory to the later plan_round. A tile, config or language change is not ' +
    'a direction here (deep_explore / resweep_request decide those later).' }),
  { phase: 'Analyze', label: 'tech_lead:analyze', schema: ANALYZE_SCHEMA });
const KK_OPERATOR = (analysis && analysis.kk_operator) || '';
const KK_LANGUAGE = (analysis && analysis.kk_language) || LANGUAGE;
const KK_REFS = (analysis && Array.isArray(analysis.kk_refs)) ? analysis.kk_refs : [];
const KK_INPUTS = { KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS };
log(`Analyze done. kernel_type=${analysis ? analysis.kernel_type : '?'} kk_operator=${KK_OPERATOR || '-'}`);

phase('Benchmark');
const bench = await agentT(
  rolePrompt('benchmark_engineer', geakRole('benchmark_engineer'), 'setup',
    'Build the COMMANDMENT and record a reliable baseline.', {
      WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
      ANALYSIS: analysis || { kernel_type: 'gluon_authoring', note: 'see the task dir config/README for the target' },
    }),
  { phase: 'Benchmark', label: 'benchmark_engineer', schema: BENCH_SCHEMA });
if (!bench || !Array.isArray(bench.baseline_per_case)) throw new Error('Benchmark setup failed: no baseline recorded');
const BASELINE_PER_CASE = bench.baseline_per_case;

// The pack's own profiling entry (gluon_profile.py) instead of profile_kernel.sh: every dial of the
// method in one call, a kernel filter and a timeout on every profiler pass, per-layer states.
phase('Profile');
const PROFILE_LAYERS = ['trace', 'pmc', 'att', 'ir'];
const profileMissing = (p) => {
  if (!p) return ['the whole return'];
  const out = readsCover('profile', null, p.references_read);
  if (!/gluon_profile\.py/.test(p.command || '') || !isNum(p.exit_code)) out.push('command (gluon_profile.py) and exit_code');
  if (!has(p.profile_dir)) out.push('profile_dir');
  const L = p.layers && typeof p.layers === 'object' ? p.layers : {};
  const notOk = PROFILE_LAYERS.filter(k => String(L[k] || '') !== 'ok');
  const unstated = PROFILE_LAYERS.filter(k => !has(L[k]));
  if (unstated.length) out.push(`layers.${unstated.join('/')} (the state profile.json records)`);
  if (notOk.length && (p.degraded_reason || '').trim().length < 20) out.push(`degraded_reason (layers ${notOk.join(', ')} are not ok)`);
  return out;
};
let profile = null;
for (let k = 0; k <= EVIDENCE_REASKS; k++) {
  const prev = profile;
  profile = await agentT(
    rolePrompt('gluon_profiler', pkgRole('gluon_profiler'), 'baseline',
      k ? 'Your previous profile return was incomplete.' : 'Profile the frozen baseline with gluon_profile.py and classify the bottleneck from the four dials.', {
        WORKSPACE: CANONICAL, EVAL_DIR, GPU_ID: GPU_POOL, COMMANDMENT, GFX, KT, SKILL, WORKFLOW_DIR,
        ...(k ? { PREVIOUS_RETURN: prev, MISSING: profileMissing(prev) } : {}),
      }, { stage: 'profile', ...(k ? { tail: 'Complete exactly the MISSING items (run the commands; do not restate). Keep everything that was already valid.' } : {}) }),
    { phase: 'Profile', label: k ? `gluon_profiler:baseline (reask ${k})` : 'gluon_profiler:baseline', schema: PROFILE_SCHEMA });
  if (!profileMissing(profile).length) break;
}
const profileGaps = profileMissing(profile);

// ===========================================================================
// PHASE: Entry — mode, champion bundle, G1, the track workspace.
// ===========================================================================
phase('Entry');
const leadInputs = { EVAL_DIR, WORKSPACE: CANONICAL, KERNEL_PATH_ORIG, TRACK, GLUON_DIR, SKILL, KT,
  WORKFLOW_DIR, GPU_ID: GPU_POOL, GFX, COMMANDMENT, BASELINE_PER_CASE, TASK };
let entry = await agentT(
  rolePrompt('gluon_lead', pkgRole('gluon_lead'), 'entry',
    'Settle the entry mode, build and assert the champion bundle, and create the track workspace.', {
      ...leadInputs, ENTRY_HINT, PROFILE_SUMMARY: profile ? profile.summary_path : '', PROFILE_DIR: profile ? profile.profile_dir : '',
    }, { stage: 'entry' }),
  { phase: 'Entry', label: 'gluon_lead:entry', schema: ENTRY_SCHEMA });
const entryMissing = (e) => {
  if (!e) return ['the whole return'];
  const out = readsCover('entry', null, e.references_read);
  const g = e.champion_gate || {};
  const mode = ENTRY_HINT !== 'auto' ? ENTRY_HINT : e.entry_mode;
  if (mode === 'fresh') { if (!has(g.not_applicable_reason)) out.push('champion_gate.not_applicable_reason'); }
  else if (!/champion_gate\.py/.test(g.command || '') || !isNum(g.exit_code)) out.push('champion_gate (champion_gate.py command and exit_code)');
  if (mode === 'port' && e.source_is_anchor && !has(e.champion_patch)) out.push('champion_patch (the anchor is the frozen baseline, so the champion must be measured)');
  if (e.track_ready !== true) out.push(`track_ready (${TRACK} materialized and tagged geak_base)`);
  return out;
};
for (let k = 0; k < EVIDENCE_REASKS && entryMissing(entry).length; k++) {
  const miss = entryMissing(entry);
  log(`Entry return incomplete (${miss.join('; ')}); asking again.`);
  entry = await agentT(
    rolePrompt('gluon_lead', pkgRole('gluon_lead'), 'entry', 'Your previous entry return was incomplete.', {
      ...leadInputs, ENTRY_HINT, PREVIOUS_RETURN: entry, MISSING: miss,
    }, { stage: 'entry', tail: 'Complete exactly the MISSING items (run the commands; do not restate). Keep everything that was already valid.' }),
    { phase: 'Entry', label: `gluon_lead:entry (reask ${k + 1})`, schema: ENTRY_SCHEMA });
}
const caveats = [];
if (profileGaps.length) caveats.push(`baseline_profile_incomplete: ${profileGaps.join('; ')}`);
if (profile && has(profile.degraded_reason)) caveats.push(`baseline_profile_degraded: ${profile.degraded_reason}`);
const history = [];
let blocked = '';
const stillMissing = entryMissing(entry);
if (stillMissing.length) blocked = `entry_incomplete: ${stillMissing.join('; ')}`;
const ENTRY = entry ? (ENTRY_HINT !== 'auto' ? ENTRY_HINT : entry.entry_mode) : ENTRY_HINT;
if (entry && ENTRY_HINT !== 'auto' && entry.entry_mode !== ENTRY_HINT) {
  caveats.push(`entry_mode ${ENTRY_HINT} set by the caller; gluon_lead detected ${entry.entry_mode}`);
}
if (!blocked && ENTRY !== 'fresh' && entry.champion_gate.exit_code !== 0) {
  blocked = `champion_gate exit ${entry.champion_gate.exit_code} (G1): edit nothing (${entry.champion_gate.output_path || 'see gate output'})`;
}
if (entry && Array.isArray(entry.champion_gate && entry.champion_gate.flags)) {
  for (const f of entry.champion_gate.flags) caveats.push(`champion_gate ${f}`);
}
if (entry && Array.isArray(entry.caveats)) caveats.push(...entry.caveats);
log(`Entry: mode=${ENTRY}${blocked ? `, BLOCKED: ${blocked}` : ''}`);

// The loop shape follows the entry (G0 by construction): a port lands below the comparator and is
// allowed to give ground; an incumbent or fresh kernel keeps kernel_lane.js's defaults.
// MAX_NO_IMPROVE: climb rounds in a row without progress. PROGRESS_DELTA: how far over the best
// candidate seen a round's winner must land to count as progress (kernel_lane.js: MIN_IMPROVE).
// ENABLING_DELTA: how far below the track best an `enabling_step` may sit and still be kept.
const MAX_NO_IMPROVE = Math.max(1, Math.floor(num(A.max_no_improve, ENTRY === 'port' ? 6 : 2)));
const PROGRESS_DELTA = num(A.progress_delta, ENTRY === 'port' ? -0.05 : MIN_IMPROVE);
const ENABLING_DELTA = num(A.enabling_delta, ENTRY === 'port' ? -0.05 : 0);

// ===========================================================================
// PHASE: Budget — hw_budget.py before any authoring (no GPU).
// ===========================================================================
let budgetRes = null;
if (!blocked) {
  phase('Budget');
  const budgetMissing = (b) => !b ? ['the whole return'] : [
    ...readsCover('budget', null, b.references_read),
    ...(/hw_budget\.py/.test(b.command || '') && b.exit_code === 0 ? [] : ['hw_budget.py command with exit_code 0']),
    ...(has(b.budget_json) ? [] : ['budget_json']),
  ];
  for (let k = 0; k <= EVIDENCE_REASKS; k++) {
    const prev = budgetRes;
    budgetRes = await agentT(
      rolePrompt('gluon_lead', pkgRole('gluon_lead'), 'budget', 'Size the prize before touching the kernel.', {
        ...leadInputs, ENTRY_MODE: ENTRY, ...(k ? { PREVIOUS_RETURN: prev, MISSING: budgetMissing(prev) } : {}),
      }, { stage: 'budget' }),
      { phase: 'Budget', label: k ? `gluon_lead:budget (reask ${k})` : 'gluon_lead:budget', schema: BUDGET_SCHEMA });
    if (!budgetMissing(budgetRes).length) break;
  }
  if (budgetMissing(budgetRes).length) caveats.push(`budget_missing: ${budgetMissing(budgetRes).join('; ')}`);
}

// The champion is measured by verify_engineer under the same COMMANDMENT when it is not the frozen
// baseline (e.g. the task ships the Gluon anchor and the plain champion only as its golden copy).
let champVer = null;
if (!blocked && ENTRY === 'port' && entry && has(entry.champion_patch)) {
  champVer = await agentT(
    rolePrompt('verify_engineer', geakRole('verify_engineer'), 'verify',
      'Independently measure the plain champion (this patch installs it) against the frozen baseline.', {
        CANONICAL, PATCH: entry.champion_patch, VERIFY_DIR: `${GLUON_DIR}/champion/verify`, EVAL_DIR,
        WORKFLOW_DIR, GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
      }),
    { phase: 'Entry', label: 'verify champion', schema: VERIFY_SCHEMA });
  if (!champVer || !says(champVer.correctness, 'pass') || !(primary(champVer) > 0)) {
    blocked = `champion measurement failed (verify status ${champVer ? champVer.status : 'none'})`;
  }
}
// Every speedup below is vs the frozen baseline; dividing by this gives vs_champion.
const CHAMP_PRIMARY = champVer ? primary(champVer) : 1;
const vsChampion = (p) => (p > 0 && CHAMP_PRIMARY > 0) ? p / CHAMP_PRIMARY : 0;

// ===========================================================================
// The search. Transcribe/Author is one direction. Recover and Climb are kernel_lane.js's round loop
// with the Gluon method inside each direction:
//   kernel_lane.js                                 here
//   tech_lead plan_round: ~2-3 orthogonal          the same role and phase, plus a Gluon addendum
//     directions within the remaining budget         (layers, paired directions, the layer gating order)
//   engineer per direction, then verify_engineer   gluon_engineer per direction (G3/G4/ATT, audit), then verify_engineer
//   integrator when >= 2 verified                  gluon_engineer PHASE=integrate (integrator.md's strategy on the
//                                                    track tags, permuted screen), then verify_engineer
//   commit the winner, profile_engineer reprofile  the track moves to the winner's tag; its gluon_profile.py profile
//   tech_lead update_memory                        the same role and phase
// Every direction works in its own copy of the track and fetches its tag back, so the track is only a
// store of tags; TRACK_START is the kept line every direction of a round starts from.
// ===========================================================================
const BASE_TAG = 'geak_base';
let trackTag = BASE_TAG;
// Port: nothing is kept until the anchor exists. Incumbent: the frozen baseline is the track's start.
let trackBest = ENTRY === 'incumbent' ? 1 : 0;
let accepted = null;            // the GEAK-accepted best: verified, correct, > accepted * (1 + MIN_IMPROVE)
let stage = ENTRY === 'port' ? 'transcribe' : ENTRY === 'fresh' ? 'author' : 'climb';
let parity = ENTRY === 'port' ? 'pending' : 'not_applicable';
let recoverRounds = 0;
let noImprove = 0;
let rejected = 0;
let stoppedBy = blocked ? 'blocked' : '';
let deadlineHit = false;
let lastProfile = profile ? (profile.profile_dir || profile.summary_path) : '';
let bestSeen = 0;               // best verified candidate of any round, kept or not (kernel_lane.js)
let dispatched = 0;             // budget spent, in directions
let round = 0;
let forcedReplans = 0;
let bestPerCase = BASELINE_PER_CASE;
const memory = { insights: [], ledger: [], bottleneck_now: '', suggest_next: '' };
const requests = [];
const STAGE_PHASE = { transcribe: 'Transcribe', author: 'Transcribe', recover: 'Recover', climb: 'Climb' };

function evidenceTail(stg, d) {
  const lines = ['## Evidence this round must return (checked in code; a gap is sent back, then rejected)'];
  if (stg === 'transcribe') lines.push('- equivalence: layout_diff (ttgir_bridge.py verify), numeric, determinism, asm_parity', '- g3_occupancy: probe.py measure on the anchor IR, BOTH limiters', '- edge_shapes: edge_shapes.py with the champion as --baseline and the anchor as a variant (exit 1 is not equivalent)');
  if (stg === 'author') lines.push('- correctness of the first Gluon kernel; g3_occupancy: probe.py measure, BOTH limiters', '- edge_shapes: edge_shapes.py on the kernel (exit 3 = boundary shapes it does not handle; say which in shared_limitation)');
  if (stg === 'recover' || stg === 'climb') lines.push('- edge_shapes (when the change touches masks, index arithmetic, tile or loop bounds): edge_shapes.py with TRACK_START as --baseline; exit 1 rejects the candidate');
  if (stg === 'recover') lines.push('- parity_gate: parity_gate.py run as the gate (mode=gate), exit code, ratio, suspects', '- for a repayment: suspect, g3_occupancy, g4_screen (ab_bench.py --permute --json), and ATT when the suspect is lost_pipeline', '- a gate-only direction (nothing left to repay or nothing attributable) sets no_change=true');
  if (stg === 'climb') lines.push(`- layer (one of ${LAYERS.join(', ')}), profile_dir + profile_command (gluon_profile.py ... --compare LAST_PROFILE)`, '- g3_occupancy: probe.py measure, BOTH limiters (a missing one as null + missing_reason)', '- g4_screen: ab_bench.py --module <adapter> --permute --json', `- att for layers ${[...PIPELINE_LAYERS].join('/')}: ${ATT_CMD} with state ok, or status=unavailable with the reason`, '- inline asm rounds: disassembly_ref');
  if (d && d.paired) lines.push('- pair_screen: ab_bench.py --permute of the prerequisite alone and of the pair, each vs TRACK_START (prerequisite_json, prerequisite_delta_pct, pair_json, pair_delta_pct)');
  if (d && d.planned) lines.push('- lesson: one line on what this direction showed, for a null or negative result too (the next plan reads it)');
  return lines.join('\n');
}

const verifyPrompt = (patch, out, anchorAsIs) => rolePrompt('verify_engineer', geakRole('verify_engineer'), 'verify', anchorAsIs
  ? 'Independently measure the anchor the canonical workspace already contains. PATCH is intentionally empty: ' +
    'skip the apply step and measure the materialized copy of CANONICAL as it is (this is not apply_failed).'
  : 'Independently re-measure this candidate patch.', {
  CANONICAL, PATCH: anchorAsIs ? '' : patch, VERIFY_DIR: `${out}/verify`, ENGINEER_WORKSPACE: `${out}/ws`, EVAL_DIR,
  WORKFLOW_DIR, GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
  ...(anchorAsIs ? { MEASURE_AS_IS: 'yes: the anchor is the workspace source; there is no patch to apply' } : {}),
});

// One direction: engineer -> evidence check (re-asked by name) -> artifact audit -> verify. Returns a
// history row and the verified primary metric `p` (0 unless verify measured it correct).
async function runDirection(d, stg, roundInputs, engTimeout) {
  const ph = STAGE_PHASE[stg];
  const inputs = { ...roundInputs, ROUND_TAG: d.id, OUTPUT_DIR: d.out,
    ...(d.planned ? { DIRECTION: { id: d.id, title: d.title, specialty: d.specialty, layer: d.layer || '',
      suspect: d.suspect || '', paired: d.paired === true, expected_speedup: d.expected_speedup, prompt: d.prompt,
      kk_refs: d.kk_refs || [], learned_refs: d.learned_refs || [] } } : {}),
    // a direction's own perf_knowledge cards replace the run's (kernel_lane.js)
    ...(d.planned && Array.isArray(d.kk_refs) && d.kk_refs.length ? { KK_REFS: d.kk_refs } : {}) };
  const intro = d.planned
    ? `Carry direction ${d.id} (${d.specialty}${d.layer ? `, layer ${d.layer}` : ''}${d.paired ? ', a prerequisite + lever pair' : ''}) ` +
      `of round ${inputs.ROUND}, stage ${stg}, in your own copy of the track.`
    : `Carry round ${inputs.ROUND} of the Gluon track: stage ${stg}, one attributable change.`;
  const engPrompt = (extra, tail) => rolePrompt('gluon_engineer', pkgRole('gluon_engineer'), stg, intro,
    { ...inputs, ...extra }, { stage: stg, tail: tail || evidenceTail(stg, d) });
  let r = await agentT(engPrompt({}), { phase: ph, label: `gluon_engineer:${stg} ${d.id}`, schema: ROUND_SCHEMA }, engTimeout);
  const ctx = { entry: ENTRY, direction: d.planned ? d : null };
  // Completeness, then (optionally) an independent artifact audit; each gap is sent back by name.
  let missing = missingEvidence(stg, r, ctx);
  let audit = null;
  for (let k = 0; ; k++) {
    if (!missing.length && EVIDENCE_AUDIT && r) {
      audit = await agentT(
        rolePrompt('evidence_auditor', pkgRole('evidence_auditor'), 'audit', 'Check that the cited evidence exists and says what the round claims.', {
          EVIDENCE: r, STAGE: stg, GFX, SKILL, KT, OUTPUT_DIR: d.out, TRACK, ROUND_TAG: d.id,
        }, { stage: 'audit' }),
        { phase: ph, label: `evidence_audit ${d.id}`, effort: 'low', schema: AUDIT_SCHEMA });
      if (audit && audit.ok === false) {
        missing = (audit.findings || []).map(f => `audit: ${f}`);
        if (!missing.length) missing = ['audit: failed without naming a finding; re-check every cited artifact'];
      }
      if (!audit) missing = [];   // a lost audit never blocks a complete round; the gap is logged
      if (!audit) log(`  [audit] ${d.id}: auditor returned nothing; structural check stands.`);
    }
    if (!missing.length || k >= EVIDENCE_REASKS) break;
    log(`${d.id}: evidence incomplete (${missing.join('; ')}); asking for exactly that.`);
    const prev = r;
    r = await agentT(engPrompt({ PREVIOUS_RETURN: prev, MISSING: missing },
      'Complete exactly the MISSING items for the candidate already tagged ROUND_TAG in your workspace (do NOT start a ' +
      'new edit, do NOT move ROUND_TAG unless an item requires it, and fetch it into TRACK again if you do). Return the ' +
      'whole object with the gaps filled.'),
    { phase: ph, label: `gluon_engineer:evidence ${d.id} (reask ${k + 1})`, schema: ROUND_SCHEMA }, engTimeout);
    missing = missingEvidence(stg, r, ctx);
  }
  const row = { round: inputs.ROUND, id: d.id, stage: stg, specialty: d.specialty || '', title: d.title || '',
    layer: r ? (stg === 'recover' ? (SUSPECT_LAYER[r.suspect] || r.layer || '') : (r.layer || '')) : (d.layer || ''),
    claim: r ? r.claim || '' : '', lesson: r ? r.lesson || '' : '', status: '', vs_baseline: null, vs_champion: null,
    kept: false, accepted: false, learned_refs: d.planned && Array.isArray(d.learned_refs) ? d.learned_refs : [] };
  if (r && has(r.resweep_request)) requests.push({ round: inputs.ROUND, id: d.id, kind: 'resweep_request', ref: r.resweep_request });
  if (r && has(r.structure_suspect)) requests.push({ round: inputs.ROUND, id: d.id, kind: 'structure_suspect', ref: r.structure_suspect });
  if (r && r.edge_shapes && r.edge_shapes.exit_code === 3 && !caveats.some(c => c.startsWith('boundary_limitation'))) {
    caveats.push(`boundary_limitation: the baseline fails boundary shapes too (${d.id}: ${r.edge_shapes.shared_limitation || r.edge_shapes.json_path || 'see edge_shapes json'})`);
  }
  if (audit && Array.isArray(audit.findings) && audit.findings.length) row.audit = audit.findings.slice(0, 5);
  if (missing.length) {
    rejected++;
    row.status = 'rejected_missing_evidence'; row.missing = missing;
    log(`${d.id} REJECTED: ${missing.join('; ')}. Track stays at ${roundInputs.TRACK_START}.`);
    return { d, r, row, ver: null, p: 0, rejected: true, measured: false, anchorAsIs: false };
  }
  // GEAK verify is the only measurement that counts; a gate-only direction has nothing new to measure.
  // An anchor the workspace already ships is measured too, as it is, whatever the engineer set (it may
  // report no_change): the frozen baseline may be the plain champion, so its ratio is never assumed 1.
  const anchorAsIs = stg === 'transcribe' && r.anchor_provided === true;
  const measured = anchorAsIs || !r.no_change;
  let ver = null;
  let p = 0;
  if (measured && (anchorAsIs || says(r.correctness, 'pass'))) {
    ver = await agentT(verifyPrompt(r.patch_file, d.out, anchorAsIs),
      { phase: ph, label: anchorAsIs ? `verify ${d.id} (anchor as is)` : `verify ${d.id}`, schema: VERIFY_SCHEMA });
    p = ver && says(ver.correctness, 'pass') ? primary(ver) : 0;
  }
  row.status = ver ? String(ver.status || '') : (measured ? 'not_verified' : 'not_measured');
  row.vs_baseline = p || null;
  row.vs_champion = p ? vsChampion(p) : null;
  return { d, r, row, ver, p, rejected: false, measured, anchorAsIs };
}

function planAddendum(stg) {
  return [
    '## This run: the gluon_authoring expert workflow (what changes in plan_round)',
    `Every direction is carried by a gluon_engineer (${pkgRole('gluon_engineer')}) with the gluon_authoring method ` +
      `(${SKILL}) and its evidence gates, from the kept line TRACK_START. Your rules hold, read this way:`,
    `- Each direction also carries \`layer\`, one of ${LAYERS.join(', ')} (${refPath('references/method/climb.md')} ## Layer Backbone). ` +
      'Read PROFILE_SUMMARY (a gluon_profile.py dir: report.md, profile.json dials, compare.md) and BUDGET_JSON ' +
      '(hw_budget.py: which layers have surface on this archetype) before choosing.',
    '- Orthogonality (rule 2): the kernel is one file, so a distinct `focus_files` set becomes a distinct `layer` and code ' +
      'region. Layers that gate each other never share a round (1.5 with 4/4+/5, two layer-1.5 scheduling models, 3 with 4, ' +
      '4 with 4+); the script defers the later one of such a pair without charging it. Coupled work goes in one paired direction.',
    '- `paired: true` declares a prerequisite + the lever it makes legal or profitable (e.g. a second LDS buffer for one ' +
      'operand + moving the barrier it frees). The engineer screens the prerequisite alone and the pair and returns the ' +
      'better one, saying which (`pair_screen.outcome`). Use it when HISTORY shows a lever measured null for want of its ' +
      'prerequisite, or when a prerequisite costs a resource (registers, LDS) the lever needs.',
    '- HISTORY.rounds lists every direction with its verified number, `claim` and `lesson`, null and rejected ones included: ' +
      'do not re-issue a lever measured null on the same body unless its lesson names what was missing, and then pair it.',
    '- A verdict holds for the body it was measured on. When a kept change moves the binding occupancy limiter, the ' +
      'register budget or the bottleneck class (compare.md), re-issue the directions HISTORY marked rival or null whose ' +
      'lesson depended on what moved; a rival of one kept line can stack on the next.',
    '- host_runtime is the launch side the frozen harness allows: grid order, workgroup grouping, XCD-aware pid remap, ' +
      'persistent CTAs (layer 6). A launch parameter is screened at a handful of points, not swept.',
    `- deep_explore is a Gluon rewrite with broad authority (re-tile, layout, pipeline structure); it costs ${DEEP_COST} and ` +
      'runs alone in its round. A tile or warp re-cut is structure and goes here, with the occupancy arithmetic as its ' +
      'reason (both limiters from probe.py, workgroups per CU before and after): the usual trigger is both limiters binding ' +
      'at one workgroup per CU while the stall sits on waits a second workgroup could cover. Also use it on a ' +
      'structure_suspect or resweep_request in HISTORY, or a plateau.',
    ...(stg === 'recover' ? [
      `- STAGE=recover (port; G2 is not cleared at ${PARITY_THRESHOLD} of the champion): each direction repays ONE ` +
      'transcription-debt `suspect` (lost_pipeline -> layer 4, lost_layout -> layer 3, lost_RA -> layer 5; ' +
      `${refPath('references/method/recover.md')}), at most one direction per suspect, no deep_explore. If the debt is not ` +
      'attributed yet, one direction attributes it and repays the largest suspect.'] : []),
    '- Knowledge (KK_REFS, learned cards, KB_REFERENCES, the analyze roadmap) is advisory and mostly written for other ' +
      'languages: take the mechanism, not the recipe. The kernel stays Gluon; a config or tile change is a resweep_request ' +
      'or a deep_explore direction. A construct that exists only on some targets (an async copy, a scaled MFMA, a ' +
      `scheduling knob) is planned only after \`${SKILL}/scripts/probe_levers.py --all\` shows it is live on ${GFX}.`,
    ...(PRIOR_LOG ? [`- PRIOR_RUN_LOG (${PRIOR_LOG}) is an earlier run on this kernel whose kept line is this run's baseline: ` +
      'read its insights, ledger and every direction\'s verdict and lesson as part of HISTORY before planning.'] : []),
    'Return the plan_round JSON with `layer` on every direction (and `suspect` / `paired` where they apply).',
  ].join('\n');
}

// tech_lead plan_round (kernel_lane.js's planner), the deadline's no-early-stop rule, then the
// script's filters: the remaining budget, a dedicated deep_explore round, the layer gating order.
async function planRound(stg, left) {
  const remaining = BUDGET - dispatched;
  const cap = MAX_DIRECTIONS ? Math.min(remaining, MAX_DIRECTIONS) : remaining;
  const planInputs = (forbidStop) => ({
    EVAL_DIR, ROUND: round, STAGE: stg, BUDGET_REMAINING: cap, DEEP_COST,
    CUMULATIVE_SPEEDUP: accepted ? accepted.primary : 1, TRACK_BEST_VS_BASELINE: trackBest,
    CHAMPION_VS_BASELINE: CHAMP_PRIMARY, BASELINE_GEOMEAN_MS: bench.baseline_geomean_ms,
    SKILL_DIR: WORKFLOW_DIR, SKILL, PROFILE_SUMMARY: lastProfile, CURRENT_BEST_PER_CASE: bestPerCase,
    ...(PRIOR_LOG ? { PRIOR_RUN_LOG: PRIOR_LOG } : {}),
    HISTORY: { insights: memory.insights, ledger: memory.ledger.slice(-20), rounds: history.slice(-12),
      bottleneck_now: memory.bottleneck_now, suggest_next: memory.suggest_next },
    GFX, DEVICE_TARGET: setup.device_target || '', PHYSICAL_CU_COUNT: setup.physical_cu_count || 0,
    ROOFLINE_STATUS: budgetRes && budgetRes.exit_code === 0 ? `tabulated-${GFX}` : `unknown-${GFX}`,
    BUDGET_JSON: budgetRes ? budgetRes.budget_json : '', ...KK_INPUTS,
    CODEBASE_CONTEXT: `${EVAL_DIR}/codebase_context.md`, ROADMAP: analysis ? analysis.roadmap_summary || '' : '',
    ...(KB_REFERENCES ? { KB_REFERENCES } : {}),
    LEARNED_KB: USE_LEARNED_READ ? 'on' : 'off',
    ...(USE_LEARNED_READ ? { LEARNED_KB_BUDGET:
      `At most ${KB_DIR_CAP} of this round's directions may draw on learned cards, and you must cite ` +
      'the cards you used in that direction\'s `learned_refs`. ' +
      (KB_COLD_DIRECTION ? 'At least ONE direction must be planned from the profile alone, with an empty ' +
        '`learned_refs` -- it is this round\'s control. ' : '') +
      'Read as much of the index as you like; the budget is on how much of the ROUND the KB steers.' } : {}),
    ...(Number.isFinite(left) ? { MINUTES_REMAINING: Math.round(left / 60) } : {}),
    ...(forbidStop ? { STOP_NOT_PERMITTED:
      `You returned stop=true with ${Math.round(left / 60)} min of the wall-clock window still unspent. This run is a ` +
      'FIXED-WINDOW measurement: plan at least one direction. Switch layer rather than re-running an exhausted one, or ' +
      'issue a dedicated deep_explore; if every direction really is exhausted, re-measure the kept line and say so.' } : {}),
  });
  const ask = (forbidStop, label) => agentT(
    rolePrompt('tech_lead', geakRole('tech_lead'), 'plan_round', forbidStop
      ? 'Your stop was refused: plan at least one direction.' : 'Decide this round\'s orthogonal directions (or stop).',
    planInputs(forbidStop), { tail: planAddendum(stg) }),
    { phase: STAGE_PHASE[stg], label, schema: PLAN_SCHEMA });
  const empty = (p) => !p || p.stop || !Array.isArray(p.directions) || !p.directions.length;
  let plan = await ask(false, `tech_lead:plan r${round}`);
  while (DEADLINE_EPOCH && left > NO_STOP_S && forcedReplans < MAX_FORCED_REPLANS && empty(plan)) {
    forcedReplans++;
    log(`Round ${round}: stop=true with ${Math.round(left / 60)} min left - refusing (#${forcedReplans}), re-planning.`);
    plan = await ask(true, `tech_lead:replan r${round}#${forcedReplans}`);
    left = (await secondsLeft(`replan-r${round}`)).left;
  }
  if (empty(plan)) return { stop: true, reasoning: plan ? plan.reasoning || '' : 'no plan returned' };
  let ds = plan.directions.filter(d => d && typeof d === 'object').slice(0, cap);
  if (stg === 'recover' && ds.some(d => d.specialty === 'deep_explore')) {
    log(`Round ${round}: deep_explore is not a recover direction; dropped.`);
    ds = ds.filter(d => d.specialty !== 'deep_explore');
  }
  // deep_explore is a dedicated round (kernel_lane.js): the others are not dispatched.
  const deep = ds.find(d => d.specialty === 'deep_explore');
  if (deep) ds = [deep];
  const kept = [];
  for (const d of ds) {
    const L = String(d.layer || '');
    const clash = kept.find(k => gates(String(k.layer || ''), L) || gates(L, String(k.layer || '')));
    if (clash) {
      log(`Round ${round}: deferred "${d.title}" (layer ${L}): it and layer ${clash.layer} ("${clash.title}") gate each other ` +
          '(climb.md ## Layer Backbone); not dispatched, not charged.');
      continue;
    }
    kept.push(d);
  }
  if (!kept.length) return { stop: true, reasoning: 'no dispatchable direction after the filters' };
  return { stop: false, reasoning: plan.reasoning || '', directions: kept.map((d, i) => ({ ...d, plan_id: d.id,
    id: `r${round}_d${i}`, out: `${GLUON_DIR}/round_${round}/d${i}`, planned: true })) };
}

function integrateTail(stg) {
  return ['## Evidence the integrated candidate must return (checked in code; a gap is sent back, then the integration is dropped)',
    '- conclusion improved|no_improvement|all_failed and combos_tried (each add: patches, method, correctness, delta_pct, relation orthogonal|rival|conflict)',
    '- for improved: track_tag (= ROUND_TAG, fetched into TRACK), patch_file (git diff geak_base ROUND_TAG), correctness, ' +
      'g4_screen: ab_bench.py --permute of the integrated candidate vs the best individual tag',
    ...(stg === 'recover' ? ['- parity_gate: parity_gate.py on the integrated candidate, mode=gate'] : []),
  ].join('\n');
}

// kernel_lane.js integrates when >= 2 directions verified. Here the candidates are this round's verified
// directions that move the kept line (or enable a later layer); the integrated candidate is verified too.
async function integrate(stg, cands, roundInputs, engTimeout) {
  const id = `r${round}_int`;
  const out = `${GLUON_DIR}/round_${round}/integrate`;
  phase('Integrate');
  const inputs = { ...roundInputs, ROUND_TAG: id, OUTPUT_DIR: out, STAGE: stg, INTEGRATOR_ROLE: geakRole('integrator'),
    BEST_INDIVIDUAL: Math.max(...cands.map(c => c.p)),
    CANDIDATES: cands.map(c => ({ id: c.d.id, tag: c.r.track_tag, specialty: c.d.specialty, layer: c.row.layer,
      title: c.d.title, paired: c.d.paired === true, verified_vs_baseline: c.p,
      change_summary: c.r.change_summary || '', patch_file: c.r.patch_file })) };
  const ask = (extra, tail, label) => agentT(
    rolePrompt('gluon_engineer', pkgRole('gluon_engineer'), 'integrate',
      `Combine this round's verified directions into one candidate on the track (stage ${stg}).`,
      { ...inputs, ...extra }, { stage: 'integrate', tail: tail || integrateTail(stg) }),
    { phase: 'Integrate', label, schema: INTEGRATE_SCHEMA }, engTimeout);
  let x = await ask({}, '', `integrate r${round}`);
  let missing = integrateMissing(stg, x);
  for (let k = 0; k < EVIDENCE_REASKS && missing.length; k++) {
    log(`${id}: integration evidence incomplete (${missing.join('; ')}); asking for exactly that.`);
    x = await ask({ PREVIOUS_RETURN: x, MISSING: missing },
      'Complete exactly the MISSING items for the integrated candidate already tagged ROUND_TAG (no new combination). ' +
      'Return the whole object with the gaps filled.', `integrate r${round} (reask ${k + 1})`);
    missing = integrateMissing(stg, x);
  }
  const row = { round, id, stage: stg, specialty: 'integrate', title: 'integrated',
    layer: [...new Set(cands.map(c => c.row.layer).filter(Boolean))].join('+'),
    claim: x ? String(x.conclusion || '') : '', lesson: x ? x.notes || '' : '', status: '', vs_baseline: null,
    vs_champion: null, kept: false, accepted: false,
    combos: x && Array.isArray(x.combos_tried) ? x.combos_tried.slice(0, 8) : [] };
  const base = { d: { id, title: 'integrated', specialty: 'integrate', planned: true }, r: x || {}, row, ver: null, p: 0,
    rejected: false, measured: true, anchorAsIs: false, integrated: true };
  if (!x || x.conclusion !== 'improved') { row.status = x ? `integrate_${x.conclusion || 'none'}` : 'integrate_no_return'; return base; }
  if (missing.length) {
    rejected++;
    row.status = 'rejected_missing_evidence'; row.missing = missing;
    log(`${id} REJECTED: ${missing.join('; ')}.`);
    return { ...base, rejected: true };
  }
  const ver = await agentT(verifyPrompt(x.patch_file, out, false),
    { phase: 'Integrate', label: `verify ${id}`, schema: VERIFY_SCHEMA });
  const p = ver && says(ver.correctness, 'pass') ? primary(ver) : 0;
  row.status = ver ? String(ver.status || '') : 'not_verified';
  row.vs_baseline = p || null;
  row.vs_champion = p ? vsChampion(p) : null;
  return { ...base, ver, p };
}

// The kept line's profile after an integrated winner (a single direction's own round profile is used as is).
async function reprofile(tag, outDir) {
  const pr = await agentT(
    rolePrompt('gluon_profiler', pkgRole('gluon_profiler'), 'reprofile', 'Profile the new kept line against the last profile.', {
      TRACK, TRACK_TAG: tag, OUTPUT_DIR: outDir || `${GLUON_DIR}/round_${round}/profile`, LAST_PROFILE: lastProfile,
      EVAL_DIR, GPU_ID: GPU_POOL, COMMANDMENT, GFX, KT, SKILL, WORKFLOW_DIR,
    }, { stage: 'profile' }),
    { phase: outDir ? 'WarmStart' : 'Integrate', label: outDir ? 'warm_start:reprofile' : `gluon_profiler:reprofile r${round}`, schema: PROFILE_SCHEMA });
  return pr && has(pr.profile_dir) ? pr.profile_dir : '';
}

// ===========================================================================
// WarmStart: kernel_lane.js's experience read, keyed by language gluon. An incumbent adopts the first
// stored patch that passes verify (its tag becomes the kept line, and this run earns only the delta
// above it); a port or fresh run reads the candidates as reference only, since its anchor must come
// from its own transcription. Every benched candidate is attested back, as in kernel_lane.js.
// ===========================================================================
let KB_REFERENCES = '';
const warm_start = { adopted: false, read_reason: WARM_START_ON ? 'read' : 'disabled', candidates: [] };
if (!stoppedBy && WARM_START_ON && KB_ROOT_OK) {
  phase('WarmStart');
  const adoptMode = ENTRY === 'incumbent';
  const localResolveCmd = KB_MODE === 'store'
    ? `resolve-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
    : `resolve --root ${JSON.stringify(KB_ARTIFACTS_DIR)} --match ${WARM_START_MATCH}`;
  const syncCmd = KB_MODE === 'store' && KB_ARTIFACTS_DIR
    ? `python3 ${JSON.stringify(EXPERIENCE_STORE)} sync-local --root ${JSON.stringify(KB_ARTIFACTS_DIR)} \\
  --store ${JSON.stringify(KB_STORE_DIR)} --kernel-name ${JSON.stringify(KERNEL_NAME)} --gfx ${GFX} \\
  > ${JSON.stringify(EVAL_DIR + '/kb_sync.json')} 2>&1 || true\n` : '';
  const commonArgs = `--kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${LANGUAGE} \\\n` +
    `  --gfx ${GFX} --top-n 3 --min-speedup ${WARM_START_MIN_SPEEDUP} \\\n` +
    `  --refs-dir ${JSON.stringify(EVAL_DIR + '/kb_references')}`;
  const resolveScript = KB_REMOTE === 'off'
    ? `${syncCmd}python3 ${JSON.stringify(EXPERIENCE_STORE)} ${localResolveCmd} \\\n  ${commonArgs}`
    : `${syncCmd}${KB_ENV_PRELUDE}
REMOTE_OUT=''
if [ -n "$KB_STORE_TOKEN" ]; then
  REMOTE_OUT=$(python3 ${JSON.stringify(EXPERIENCE_STORE)} resolve-remote --plane remote${KB_VERSION_FLAG} \\
  ${commonArgs} 2>/dev/null || true)
  if printf '%s' "$REMOTE_OUT" | python3 -c 'import json,sys; sys.exit(0 if (json.load(sys.stdin).get("candidates") or []) else 1)' 2>/dev/null; then
    printf '%s\\n' "$REMOTE_OUT"; exit 0
  fi
fi
python3 ${JSON.stringify(EXPERIENCE_STORE)} ${localResolveCmd} \\
  ${commonArgs}`;
  const resolved = await agentT(
    'You are the warm-start resolver. Run EXACTLY this script and return its single-line JSON stdout ' +
    'verbatim as StructuredOutput -- do not add, drop, reorder or reinterpret any field. An empty ' +
    'candidate list is a VALID answer meaning "nothing recorded for this kernel yet". Do not retry with ' +
    'different flags and do not invent candidates:\n```bash\n' + resolveScript + '\n```',
    { phase: 'WarmStart', label: 'warm_start:resolve', schema: WARMSTART_RESOLVE_SCHEMA }) || {};
  const cands = Array.isArray(resolved.candidates) ? resolved.candidates : [];
  warm_start.read_reason = resolved.read_reason || 'read';
  warm_start.slug = resolved.slug || '';
  warm_start.plane = resolved.read_plane === 'remote' ? 'remote' : resolved.canonical_id ? 'store'
    : KB_MODE === 'store' ? 'none' : 'local';
  warm_start.candidates = cands.map(c => ({ rank: c.rank, slug: c.slug, speedup: c.speedup,
    direction: c.direction || '', session_id: c.session_id || '', exp_dir: c.exp_dir || '', status: 'read' }));
  log(`[kb] warm start (${LANGUAGE}): ${cands.length} candidate(s), reason=${warm_start.read_reason}, mode=${adoptMode ? 'adopt' : 'reference'}`);
  if (cands.length) KB_REFERENCES = `${EVAL_DIR}/kb_references`;
  if (adoptMode) {
    for (const c of cands) {
      const rec = warm_start.candidates.find(x => x.rank === c.rank);
      const dir = `${GLUON_DIR}/warm_start/cand_${c.rank}`;
      const remapOut = `${dir}/remapped.diff`;
      const ver = await agentT(
        rolePrompt('verify_engineer', geakRole('verify_engineer'), 'verify',
          'Validate a HISTORICAL warm-start patch -- the SAME gate as any round candidate. The patch was won in ' +
          'the workspace that produced it, so FIRST run EXACTLY:\n```bash\n' +
          `mkdir -p ${JSON.stringify(dir)}\n` +
          `python3 ${JSON.stringify(EXPERIENCE_STORE)} remap --patch ${JSON.stringify(c.patch_path)} \\\n` +
          `  --out ${JSON.stringify(remapOut)} --editable ${JSON.stringify(((analysis && analysis.modifiable_files) || []).join(','))} \\\n` +
          `  --workspace ${JSON.stringify(CANONICAL)}\n` + '```\n' +
          'remapped=true: apply the REMAPPED patch. reason "no_change_needed": apply the original PATCH. Any other ' +
          'reason: return status:"apply_failed", notes "patch_outside_editable_set", and do not apply or benchmark.', {
          CANONICAL, PATCH: c.patch_path, REMAP_OUT: remapOut, VERIFY_DIR: dir, EVAL_DIR, WORKFLOW_DIR,
          GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        }),
        { phase: 'WarmStart', label: `warm_start:verify c${c.rank}`, schema: VERIFY_SCHEMA });
      const sp = ver && says(ver.correctness, 'pass') && says(ver.status, 'verified') ? primary(ver) : 0;
      if (sp > 1.0) {
        // Install on the track as tag `warm_start` (the kept line every round-1 direction starts from).
        const adopt = await agentT(
          `You are installing a validated warm-start patch on the Gluon track. Run, then report:
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
WS=${JSON.stringify(dir + '/ws')}; mkdir -p "$WS"
bash ${JSON.stringify(WORKFLOW_DIR + '/scripts/materialize_workspace.sh')} --src ${JSON.stringify(TRACK)} --dst "$WS" --shared-root ${JSON.stringify(EVAL_DIR + '/_shared')} --link-aiter
cp -a ${JSON.stringify(TRACK + '/.git')} "$WS/.git"
cd "$WS" && git reset -q --hard ${BASE_TAG}
P=${JSON.stringify(remapOut)}; [ -f "$P" ] || P=${JSON.stringify(c.patch_path)}
git apply "$P" || git apply --3way "$P"
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "warm start: ${c.slug} (${sp.toFixed(3)}x)"
git tag -f warm_start
git diff ${BASE_TAG} warm_start > ${JSON.stringify(dir + '/best_patch.diff')}
git -C ${JSON.stringify(TRACK)} fetch -q "$WS" refs/tags/warm_start:refs/tags/warm_start
\`\`\`
If both applies fail, apply by hand to match the patch's intent, commit, and re-run the COMMANDMENT correctness
command (${COMMANDMENT}); report committed=true only if it passes. Return {committed, patch_file, note}.`,
          { phase: 'WarmStart', label: `warm_start:adopt c${c.rank}`, schema: ADOPT_SCHEMA });
        if (adopt && adopt.committed) {
          trackTag = 'warm_start'; trackBest = sp; bestSeen = Math.max(bestSeen, sp);
          if (Array.isArray(ver.per_case) && ver.per_case.length) bestPerCase = ver.per_case;
          accepted = { round: 0, id: 'warm_start', primary: sp, vs_champion: vsChampion(sp),
            patch: adopt.patch_file || `${dir}/best_patch.diff`, layer: '', per_case: ver.per_case || [] };
          Object.assign(warm_start, { adopted: true, adopted_speedup: sp, slug: c.slug, direction: c.direction || '',
            exp_dir: c.exp_dir || '' });
          if (rec) Object.assign(rec, { status: 'adopted', outcome: 'validated', verified_speedup: sp });
          lastProfile = (await reprofile('warm_start', `${GLUON_DIR}/warm_start/profile`)) || lastProfile;
          log(`[kb] warm start ADOPTED ${c.slug} @ ${sp.toFixed(3)}x -- the track starts there.`);
          break;
        }
      }
      if (rec && rec.status !== 'adopted') {
        rec.status = ver && ver.status ? `rejected:${ver.status}` : 'rejected:apply_failed';
        rec.outcome = !ver ? 'not_reproduced' : says(ver.correctness, 'fail') ? 'failed'
          : says(ver.status, 'apply_failed') ? (/outside_editable_set/i.test(String(ver.notes || '')) ? 'inapplicable' : 'not_reproduced')
          : !says(ver.status, 'verified') ? 'not_reproduced' : !(sp > 1.0) ? 'failed' : 'inapplicable';
        rec.verified_speedup = sp || 0;
      }
    }
    const benched = warm_start.candidates.filter(x => x.outcome && (x.session_id || x.exp_dir));
    if (benched.length) {
      const planeFlags = warm_start.plane === 'remote' ? '--plane remote'
        : warm_start.plane === 'store' ? `--plane local --store ${JSON.stringify(KB_STORE_DIR)}` : '';
      const cmds = benched.map(x => `python3 ${JSON.stringify(EXPERIENCE_STORE)} attest ${planeFlags} ` +
        (x.exp_dir ? `--exp-dir ${JSON.stringify(x.exp_dir)} ` : '') +
        (planeFlags && x.session_id ? `--session-id ${JSON.stringify(x.session_id)} ` : '') +
        `--kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${LANGUAGE} --gfx ${GFX}${KB_VERSION_FLAG} ` +
        `--outcome ${x.outcome} --measured-speedup ${Number(x.verified_speedup) || 0} ` +
        `--note ${JSON.stringify(`gluon_authoring warm-start c${x.rank}: ${x.status}`.slice(0, 200))} ` +
        `--measured-by ${JSON.stringify('gluon_authoring:' + GFX)} --apply || true`);
      await agentT('You are the kernel knowledge-base attestor. Run EXACTLY these commands in order and return ' +
        '{"ran": <how many you ran>, "note": "<anything that failed>"}. Do NOT edit, add, drop or retry any.\n' +
        '```bash\n' + KB_ENV_PRELUDE + cmds.join('\n') + '\n```',
        { phase: 'WarmStart', label: 'kb:attest', schema: obj({ ran: { type: 'number' }, note: { type: 'string' } }, []) });
    }
  }
}

while (!stoppedBy) {
  if (dispatched >= BUDGET) { stoppedBy = 'budget'; break; }
  if (TOKENS && TOKENS.total && TOKENS.remaining() < TOKEN_RESERVE) { stoppedBy = 'token_budget'; break; }
  const clock = await secondsLeft(`r${round + 1}`);
  if (clock.left <= DEADLINE_RESERVE_S) {
    deadlineHit = true; stoppedBy = 'deadline';
    log(`Deadline: ${Math.round(clock.left / 60)} min left, ${Math.round(DEADLINE_RESERVE_S / 60)} reserved for close; no round ${round + 1}.`);
    break;
  }
  round++;
  const stg = stage;
  const roundEnd = Number.isFinite(clock.left)
    ? clock.now + Math.min(ROUND_TIME_CAP_S, clock.left - DEADLINE_RESERVE_S) : 0;
  // The hang guard tracks the round deadline too, with a grace period for the return itself.
  const engTimeout = roundEnd && ENGINEER_TIMEOUT_MS > 0
    ? Math.min(ENGINEER_TIMEOUT_MS, (roundEnd - clock.now + 900) * 1000) : ENGINEER_TIMEOUT_MS;
  phase(STAGE_PHASE[stg]);
  const roundInputs = {
    ROUND: round, STAGE: stg, ENTRY_MODE: ENTRY, TRACK, BASE_TAG, TRACK_START: trackTag,
    GPU_ID: GPU_POOL, GFX, SKILL, KT, WORKFLOW_DIR, EVAL_DIR,
    COMMANDMENT, BASELINE_PER_CASE, CHAMPION_BUNDLE: entry ? entry.champion_bundle : '',
    CHAMPION_VS_BASELINE: CHAMP_PRIMARY, PARITY_THRESHOLD, PARITY: parity,
    TRACK_BEST_VS_BASELINE: trackBest, ACCEPTED_VS_BASELINE: accepted ? accepted.primary : 1,
    BUDGET_JSON: budgetRes ? budgetRes.budget_json : '', LAST_PROFILE: lastProfile,
    HISTORY: history.slice(-12), INSIGHTS: memory.insights, ...(PRIOR_LOG ? { PRIOR_RUN_LOG: PRIOR_LOG } : {}),
    ...KK_INPUTS, CODEBASE_CONTEXT: `${EVAL_DIR}/codebase_context.md`,
    SELF_MONITORING: `${WORKFLOW_DIR}/knowledge/self_monitoring.md`, ...(KB_REFERENCES ? { KB_REFERENCES } : {}),
    ...(roundEnd ? { ROUND_DEADLINE_EPOCH: roundEnd } : {}),
  };

  // --- Transcribe / Author: one direction, the anchor or the first Gluon kernel ---------------------
  if (stg === 'transcribe' || stg === 'author') {
    dispatched += 1;
    const x = await runDirection({ id: `r${round}`, out: `${GLUON_DIR}/round_${round}`, planned: false }, stg, roundInputs, engTimeout);
    const { r, row } = x;
    history.push(row);
    if (x.rejected) { stoppedBy = `${stg}_failed`; continue; }
    const p = x.measured ? x.p : trackBest;
    if (!x.measured) row.vs_baseline = p || null;
    if (x.measured && p > 0 && (p > trackBest || (r.enabling_step === true && trackBest > 0 && p >= trackBest * (1 + ENABLING_DELTA)))) {
      trackBest = Math.max(trackBest, p); trackTag = r.track_tag; row.kept = true;
    }
    if (!x.measured) { trackBest = p; row.kept = true; }
    // The anchor the workspace already ships is the track's starting line, never a deliverable.
    if (!x.anchorAsIs && x.ver && p > (accepted ? accepted.primary : 1) * (1 + MIN_IMPROVE)) {
      accepted = { round, id: x.d.id, primary: p, vs_champion: vsChampion(p), patch: r.patch_file, layer: row.layer, per_case: x.ver.per_case || [] };
      row.accepted = true;
    }
    const eq = r.equivalence || {};
    const equivalent = stg === 'author' || (['PASS', 'RECONCILED'].includes(String(eq.layout_diff).toUpperCase()) &&
      ![eq.numeric, eq.determinism].some(v => /fail/i.test(String(v))));
    if (!equivalent || !(p > 0)) {
      stoppedBy = `${stg}_failed`;
      row.status = equivalent ? 'anchor_incorrect' : 'equivalence_failed';
      log(`Round ${round}: ${stg} did not produce a ${equivalent ? 'correct' : 'equivalent'} anchor; no recovery on a wrong anchor.`);
    } else {
      if (r.anchor_provided || !x.measured) { trackTag = has(r.track_tag) ? r.track_tag : BASE_TAG; }
      if (p > bestSeen) bestSeen = p;
      stage = ENTRY === 'port' ? 'recover' : 'climb';
    }
    log(`Round ${round} ${stg}: ${row.status} vs_baseline=${p ? p.toFixed(3) : '-'} vs_champion=${p ? vsChampion(p).toFixed(3) : '-'}` +
        `${row.kept ? ' kept' : ''}${row.accepted ? ' ACCEPTED' : ''}; budget ${dispatched}/${BUDGET}`);
    continue;
  }

  // --- Recover / Climb: plan -> directions in parallel -> integrate -> keep / accept -> memory ------
  const plan = await planRound(stg, clock.left);
  if (plan.stop) {
    stoppedBy = 'tech_lead_stop';
    log(`Round ${round}: tech_lead chose to stop. ${plan.reasoning}`);
    break;
  }
  const dirs = plan.directions;
  const cost = dirs.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  dispatched += cost;
  log(`Round ${round}: ${dirs.length} direction(s) [${dirs.map(d => `${d.specialty}${d.layer ? ` L${d.layer}` : ''}` +
      `${d.suspect ? ` ${d.suspect}` : ''}${d.paired ? ' paired' : ''}`).join(', ')}], cost ${cost}, budget ${dispatched}/${BUDGET}`);
  const results = (await parallel(dirs.map((d) => () => runDirection(d, stg, roundInputs, engTimeout)))).filter(Boolean);
  for (const x of results) history.push(x.row);

  const cands = results.filter(x => x.p > 0);
  const integrable = cands.filter(c => c.p > trackBest || c.r.enabling_step === true).sort((a, b) => b.p - a.p);
  let integ = null;
  if (integrable.length >= 2) {
    integ = await integrate(stg, integrable, roundInputs, engTimeout);
    history.push(integ.row);
    if (integ.p > Math.max(...cands.map(c => c.p))) cands.push(integ);
  }
  cands.sort((a, b) => b.p - a.p);
  const winner = cands[0] || null;
  // Track keep: the best candidate above the kept line, else an enabling step within ENABLING_DELTA of it.
  const keep = cands.find(c => c.p > trackBest) ||
    cands.find(c => c.r.enabling_step === true && trackBest > 0 && c.p >= trackBest * (1 + ENABLING_DELTA));
  if (keep) {
    trackBest = Math.max(trackBest, keep.p); trackTag = keep.r.track_tag; keep.row.kept = true;
    if (keep.ver && Array.isArray(keep.ver.per_case) && keep.ver.per_case.length) bestPerCase = keep.ver.per_case;
    lastProfile = keep.integrated ? ((await reprofile(keep.r.track_tag)) || lastProfile) : (keep.r.profile_dir || lastProfile);
  }
  // Commit gate (GEAK acceptance) and the stall rule, as kernel_lane.js computes them.
  const improved = !!(winner && winner.p > (accepted ? accepted.primary : 1) * (1 + MIN_IMPROVE));
  if (improved) {
    accepted = { round, id: winner.d.id, primary: winner.p, vs_champion: vsChampion(winner.p), patch: winner.r.patch_file,
      layer: winner.row.layer, per_case: winner.ver.per_case || [] };
    winner.row.accepted = true;
  }
  const madeProgress = !!(winner && bestSeen > 0 && winner.p > bestSeen * (1 + PROGRESS_DELTA));
  if (winner && winner.p > bestSeen) bestSeen = winner.p;

  if (stg === 'recover') {
    recoverRounds++;
    // G2 on the kept line: the gate the kept candidate ran AND the verified ratio to the champion. A
    // gate-only direction (no_change) speaks for the line when nothing new was kept.
    const gateRun = keep ? keep.r.parity_gate
      : (results.filter(x => !x.rejected && !x.measured).map(x => x.r.parity_gate).find(g => g && g.exit_code === 0) || null);
    const gateExit = gateRun ? gateRun.exit_code : null;
    const ratio = vsChampion(trackBest);
    if (gateExit === 0 && ratio >= PARITY_THRESHOLD) {
      parity = 'cleared'; stage = 'climb';
      log(`Round ${round}: parity CLEARED (parity_gate exit 0, verified ${ratio.toFixed(3)}x of the champion).`);
    } else {
      if (gateExit === 0) log(`Round ${round}: parity_gate exit 0 contradicted by verify (${ratio.toFixed(3)}x < ${PARITY_THRESHOLD}); not cleared.`);
      if (recoverRounds >= MAX_RECOVER_ROUNDS) {
        parity = 'parity_unreached'; stoppedBy = 'parity_unreached';
        log(`Parity not reached in ${recoverRounds} recover round(s): do not climb (G2).`);
      }
    }
  } else {
    if (madeProgress || improved) noImprove = 0; else noImprove++;
    if (noImprove >= MAX_NO_IMPROVE) stoppedBy = 'max_no_improve';
  }

  // Cross-round memory (kernel_lane.js: tech_lead update_memory), read by the next plan.
  const mem = await agentT(
    rolePrompt('tech_lead', geakRole('tech_lead'), 'update_memory', 'Distill durable insights + update the hypothesis ledger.', {
      EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR,
      ROUND_RESULTS: results.map(x => ({ id: x.d.id, title: x.d.title, specialty: x.d.specialty, layer: x.row.layer,
        expected: x.d.expected_speedup, claimed_screen_pct: x.r && x.r.g4_screen ? x.r.g4_screen.delta_pct : null,
        verified: x.p, status: x.row.status, claim: x.row.claim, lesson: x.row.lesson, notes: x.r ? x.r.notes || '' : '' })),
      INTEGRATE: integ ? { conclusion: integ.row.claim, status: integ.row.status, verified: integ.p, combos_tried: integ.row.combos } : null,
      WINNER: winner ? { source: winner.d.id, geomean: winner.p } : null, IMPROVED: improved,
      REPROFILE_SHIFT: keep ? lastProfile : '', PRIOR_HISTORY: { insights: memory.insights, ledger: memory.ledger },
    }),
    { phase: STAGE_PHASE[stg], label: `tech_lead:memory r${round}`, schema: MEMORY_SCHEMA });
  if (mem) {
    if (Array.isArray(mem.insights)) memory.insights = mem.insights;
    if (Array.isArray(mem.ledger)) memory.ledger = memory.ledger.concat(mem.ledger);
    if (has(mem.bottleneck_now)) memory.bottleneck_now = mem.bottleneck_now;
    if (has(mem.suggest_next)) memory.suggest_next = mem.suggest_next;
  }
  // kernel_lane.js's storage reclaim after every round, on the Gluon round dirs (the track keeps every tag).
  if (STORAGE_RECLAIM) {
    const reclaim = `bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${GLUON_DIR} --keep-round ${round}` +
      `; bytes=$(du -sb ${EVAL_DIR} 2>/dev/null | awk '{print $1}');` +
      ` hard=\${GEAK_EVAL_HARD_BYTES:-68719476736};` +
      ` if [ -n "$bytes" ] && [ "$bytes" -gt "$hard" ]; then` +
      `   bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${GLUON_DIR} --keep-round ${round} --force-heavy;` +
      ` fi; echo STORAGE_RECLAIM_DONE round=${round}`;
    await agentT(`Storage reclaim after round ${round}. Run EXACTLY this bash, then return {ok:true, note:"reclaimed"}.\n` +
      '```bash\n' + reclaim + '\n```',
      { phase: STAGE_PHASE[stg], label: `storage:reclaim r${round}`, effort: 'low',
        schema: obj({ ok: { type: 'boolean' }, note: { type: 'string' } }, ['ok']) });
  }
  log(`Round ${round} ${stg} done: winner=${winner ? `${winner.d.id} ${winner.p.toFixed(3)}x` : 'none'}, ` +
      `track=${trackTag} ${trackBest ? trackBest.toFixed(3) : '-'}x, accepted=${accepted ? accepted.primary.toFixed(3) : '-'}x` +
      `${stg === 'climb' ? `, noImprove=${noImprove}` : ''}, budget ${dispatched}/${BUDGET}`);
}
if (!stoppedBy) stoppedBy = 'budget';
if (parity === 'pending' && ENTRY === 'port' && !blocked) parity = 'parity_unreached';
if (parity === 'parity_unreached') caveats.push('parity_unreached');
if (rejected) caveats.push(`${rejected} direction(s) rejected for missing evidence`);
const roundsUsed = new Set(history.map(h => h.round)).size;

// ===========================================================================
// PHASE: Close — the outcome is decided here; gluon_lead writes the report and closure review.
// ===========================================================================
phase('Close');
const deliverable = !!accepted && (ENTRY !== 'port' || accepted.vs_champion >= 1 + MIN_IMPROVE);
const outcome = deliverable ? 'win'
  : deadlineHit ? 'timeout'
  : ENTRY === 'incumbent' ? 'negative_keep_baseline'
  : 'negative_revert_plain';
// A port that did not beat the champion reverts to plain: when the frozen baseline is the anchor and
// the champion is faster than it, the champion itself is the deliverable (Director re-validates it).
const revertToChampion = !deliverable && ENTRY === 'port' && champVer && CHAMP_PRIMARY > 1 + MIN_IMPROVE;
const FINAL_SRC = deliverable ? accepted.patch : revertToChampion ? entry.champion_patch : '';
const closeInputs = {
  ...leadInputs, ENTRY_MODE: ENTRY, OUTCOME: outcome, BLOCKED: blocked, STOPPED_BY: stoppedBy,
  PARITY: parity, CAVEATS: caveats, HISTORY: history, REQUESTS: requests, INSIGHTS: memory.insights,
  BUDGET_USED: dispatched, BUDGET: BUDGET,
  ACCEPTED: accepted ? { round: accepted.round, id: accepted.id, vs_baseline: accepted.primary, vs_champion: accepted.vs_champion,
    patch: accepted.patch } : null,
  CHAMPION_VS_BASELINE: CHAMP_PRIMARY, FINAL_PATCH_SOURCE: FINAL_SRC, FINAL_PATCH: `${EVAL_DIR}/final_patch.diff`,
};
const closeMissing = (c) => !c ? ['the whole return'] : [
  ...readsCover('close', null, c.references_read),
  ...(has(c.report_path) ? [] : ['report_path']),
  ...(outcome !== 'win' && !has(c.closure_review_path) ? ['closure_review_path (closure_review.md for a non-win close)'] : []),
  ...(FINAL_SRC && !has(c.final_patch) ? ['final_patch'] : []),
];
let closeRes = null;
for (let k = 0; k <= EVIDENCE_REASKS; k++) {
  const prev = closeRes;
  closeRes = await agentT(
    rolePrompt('gluon_lead', pkgRole('gluon_lead'), 'close', `Close the run: outcome ${outcome} (decided by the workflow).`,
      { ...closeInputs, ...(k ? { PREVIOUS_RETURN: prev, MISSING: closeMissing(prev) } : {}) }, { stage: 'close' }),
    { phase: 'Close', label: k ? `gluon_lead:close (reask ${k})` : 'gluon_lead:close', schema: CLOSE_SCHEMA });
  if (!closeMissing(closeRes).length) break;
}

// ===========================================================================
// PHASE: Validate — GEAK's director, against the true baseline, only when a patch is delivered.
// ===========================================================================
let validation = null;
if (FINAL_SRC) {
  phase('Validate');
  validation = await agentT(
    rolePrompt('director', geakRole('director'), 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
      KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR, WORKFLOW_DIR, GPU_ID: GPU_POOL,
      APPLY_TO_ORIGINAL, COMMANDMENT, FROZEN_ORACLE: 'false',
      FINAL_PATCH: closeRes && has(closeRes.final_patch) ? closeRes.final_patch : `${EVAL_DIR}/final_patch.diff`,
      TECH_LEAD_REPORTED_GEOMEAN: deliverable ? accepted.primary : CHAMP_PRIMARY, BASELINE_TIMING: BASELINE_PER_CASE,
    }),
    { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });
}
const finalSpeedup = validation ? validation.director_verified_speedup_geomean : null;
const finalPrimary = validation ? (isNum(validation.director_verified_speedup_weighted) && validation.director_verified_speedup_weighted > 0
  ? validation.director_verified_speedup_weighted : Number(validation.director_verified_speedup_geomean)) : NaN;

// ===========================================================================
// Experience write-back, as kernel_lane.js does it at the end of a run, keyed by language gluon:
// update_experience distills one learned card from a director-accepted win; the citation ledger
// (what every cited card's direction measured) is filed for every run; the experience store records
// the delivered patch. Off with update_experience=off (A/B runs), box_quiet=false or held_out=true.
// ===========================================================================
let learned_card = null;
let kb_written = null;
const citations = [];
for (const h of history) for (const card of (h.learned_refs || [])) {
  citations.push({ card, round: h.round, direction: h.id, specialty: h.specialty,
    cited_then_verified: isNum(h.vs_baseline) ? h.vs_baseline : null, became_winner: h.accepted === true });
}
// Gates as in kernel_lane.js: the card needs a director-accepted win on a quiet, non-held-out box; the
// citation ledger is owed by every run except a held-out one; the store records any verified win.
const kbGate = !BOX_QUIET ? 'the box was contended' : HELD_OUT ? 'this kernel is held out' : '';
if (kbGate) log(`[kb] not distilling: ${kbGate}.`);
const kbAccepted = String((validation && validation.validation_status) || '').toLowerCase() === 'accepted';
const reportPath = closeRes && has(closeRes.report_path) ? closeRes.report_path : `${GLUON_DIR}/report.md`;
if (UPDATE_EXPERIENCE_ON && !kbGate && kbAccepted && deliverable && finalPrimary > 1.0) {
  learned_card = await agentT(
    rolePrompt('update_experience', geakRole('update_experience'), 'Validate',
      'Curate one distilled learned card from this run\'s verified win (ADD-only, measured evidence, ratios not ' +
      'wall-clock; record the pitfalls hit; total-then-per-direction for a stacked win).', {
        SCOPE: 'lane', LEARNED_DIR, SKILL_DIR: WORKFLOW_DIR, EVAL_DIR, PERF_KNOWLEDGE_DIR: KERNEL_KNOWLEDGE_DIR,
        CURATION_ISOLATE: 'false', REQUIRED_PLATFORMS: [],
        WINNER: { kernel: KERNEL_NAME, language: LANGUAGE, mode: 'optimize', gfx: GFX,
          device_target: setup.device_target || '', physical_cu_count: setup.physical_cu_count || 0,
          kernel_class: (analysis && analysis.kernel_type) || '', speedup: finalPrimary,
          validation_status: validation.validation_status, bottleneck: profile ? profile.bottleneck : '' },
        HISTORY: { rounds: history, insights: memory.insights, ledger: memory.ledger },
        CITATIONS: citations, PROFILE: lastProfile, REPORT_PATH: reportPath,
      }),
    { phase: 'Validate', label: 'update_experience', schema: UPDATE_EXPERIENCE_SCHEMA });
}
if (citations.length && !HELD_OUT) {
  await agentT(`Run EXACTLY this command and nothing else. Do NOT edit any file.
\`\`\`bash
cat <<'CITEJSON' | python3 ${WORKFLOW_DIR}/scripts/kb.py --kb-dir ${LEARNED_DIR} cite \\
--run-id ${JSON.stringify(KERNEL_NAME)} \\
--kernel ${JSON.stringify(KERNEL_NAME)} --citations -
${JSON.stringify(citations)}
CITEJSON
\`\`\`
Return {"filed": <the "citations" number the command printed, or 0>}.`,
    { phase: 'Validate', label: 'kb:cite', effort: 'low',
      schema: obj({ filed: { type: 'number' } }, ['filed']) });
}
if (KB_WRITE_OK && GFX && deliverable && finalPrimary > 1.0) {
  const remoteWriteOn = KB_REMOTE !== 'off' && !!KB_STORE_DIR;
  const writeCmd = remoteWriteOn
    ? `write-remote --plane both --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
    : KB_MODE === 'store' ? `write-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}` : 'write';
  const caseNames = (bestPerCase || []).map(c => c && String(c.name || '').replace(/,/g, ';')).filter(Boolean).join(',');
  const direction = String(warm_start.direction || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60);
  kb_written = await agentT(
    'You are the experience writer. Run EXACTLY this command (it applies its own gates and prints a single-line ' +
    'JSON) and return that JSON verbatim as StructuredOutput. If the command errors, return ' +
    '{"written": false, "reason": "io_error"}' + (remoteWriteOn ? ' -- do NOT retry it: every write the shared ' +
    'service accepts is permanent.' : '.') + `
\`\`\`bash
${remoteWriteOn ? KB_ENV_PRELUDE + '\n' : ''}python3 ${JSON.stringify(EXPERIENCE_STORE)} ${writeCmd} --root ${JSON.stringify(KB_ARTIFACTS_DIR)} \\
  --kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${LANGUAGE} \\
  --gfx ${GFX} --kernel-class ${JSON.stringify((analysis && analysis.kernel_type) || 'unknown')} \\
  --speedup ${finalPrimary} --baseline-wall-ms ${bench.baseline_geomean_ms} \\
  --patch ${JSON.stringify(`${EVAL_DIR}/final_patch.diff`)} --eval-dir ${JSON.stringify(EVAL_DIR)} \\
  --report ${JSON.stringify(reportPath)} --metric-kind geomean \\
  --direction ${JSON.stringify(direction)} --case-names ${JSON.stringify(caseNames)}\
${warm_start.exp_dir ? ` \\\n  --parent ${JSON.stringify(warm_start.exp_dir)}` : ''}
\`\`\``,
    { phase: 'Validate', label: 'kb:write', schema: WARMSTART_WRITE_SCHEMA });
}
log(`COMPLETE. ${KERNEL_NAME}: entry=${ENTRY} outcome=${outcome} stopped_by=${stoppedBy} parity=${parity}` +
    `${finalSpeedup != null ? ` director-verified ${Number(finalSpeedup).toFixed(3)}x` : ''}. Results in ${EVAL_DIR}`);

return {
  workflow: 'gluon_authoring', kernel_name: KERNEL_NAME, eval_dir: EVAL_DIR, gfx: GFX,
  entry_mode: ENTRY, outcome, blocked, stopped_by: stoppedBy, deadline_hit: deadlineHit, parity,
  caveats, requests, rounds: history, rounds_used: roundsUsed, budget: BUDGET, budget_used: dispatched,
  insights: memory.insights,
  champion_vs_baseline: CHAMP_PRIMARY,
  accepted: accepted ? { round: accepted.round, id: accepted.id, vs_baseline: accepted.primary, vs_champion: accepted.vs_champion,
    layer: accepted.layer, patch: accepted.patch } : null,
  final_patch: FINAL_SRC ? (closeRes && has(closeRes.final_patch) ? closeRes.final_patch : `${EVAL_DIR}/final_patch.diff`) : '',
  reverted_to_champion: !!revertToChampion,
  report_path: closeRes ? closeRes.report_path : '',
  closure_review_path: closeRes ? closeRes.closure_review_path || '' : '',
  validation_status: validation ? validation.validation_status : (FINAL_SRC ? 'not_validated' : 'no_candidate'),
  director_verified_speedup_geomean: finalSpeedup,
  analysis: analysis ? { kernel_type: analysis.kernel_type, kk_operator: KK_OPERATOR, kk_refs: KK_REFS } : null,
  warm_start, learned_card, kb_written, citations,
};
