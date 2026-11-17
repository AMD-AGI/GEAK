export const meta = {
  name: 'gluon-authoring',
  description: 'Expert workflow gluon_authoring: takes a kernel to Gluon on AMD CDNA (gfx942/gfx950) through the gluon_authoring method as a code-driven stage machine (entry -> budget -> transcribe -> recover -> climb -> close), with GEAK setup, frozen oracle, verify and director validation deciding acceptance.',
  whenToUse: 'Launch this script directly to port a tuned Triton kernel to Gluon, or to climb an existing Gluon kernel, on gfx942/gfx950. Pass args.kernel_path, args.workflow_dir (the kernel_workflow dir: shared roles, GPU lock, kernel tools), args.gpu_ids, args.budget, optionally args.deadline_epoch, args.entry_mode (auto|port|incumbent|fresh) and args.target_arch. See this directory\'s README.md.',
  phases: [
    { title: 'Setup', detail: 'director (kernel_workflow role): isolated eval dir + workspace, GPU identity' },
    { title: 'Benchmark', detail: 'benchmark_engineer (kernel_workflow role): COMMANDMENT + frozen baseline' },
    { title: 'Profile', detail: 'gluon_profiler: gluon_profile.py baseline profile, the four dials (trace, PMC, ATT, IR)' },
    { title: 'Entry', detail: 'gluon_lead: entry mode, champion bundle, champion_gate.py (G1), track workspace' },
    { title: 'Budget', detail: 'gluon_lead: hw_budget.py, bound class and floor (no GPU)' },
    { title: 'Transcribe', detail: 'gluon_engineer: faithful anchor + equivalence (port) / first Gluon kernel (fresh)' },
    { title: 'Recover', detail: 'gluon_engineer: attribute and repay the debt; parity_gate.py (G2) gates the climb' },
    { title: 'Climb', detail: 'gluon_engineer: one layer per round with G3/G4 (+ATT) evidence; verify_engineer measures' },
    { title: 'Close', detail: 'gluon_lead: outcome, final report, closure self-review' },
    { title: 'Validate', detail: 'director (kernel_workflow role): independent validation vs the true baseline' },
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
// One sequential track: every GPU command takes one GPU from this pool through gpu_lock.sh.
const GPU_POOL = GPU_LIST.join(',');
if (num(A.gpus_per_job, 1) !== 1) throw new Error('gluon_authoring runs single-GPU kernels only (gpus_per_job must be 1)');

// Rounds of the one engineer track (transcribe/author, recover and climb rounds all count).
const BUDGET = Math.max(1, Math.floor(num(A.budget, 6)));
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
  inline_asm: { type: 'boolean' }, disassembly_ref: { type: 'string' },
  enabling_step: { type: 'boolean' }, claim: { type: 'string' },
  resweep_request: { type: 'string' }, structure_suspect: { type: 'string' }, notes: { type: 'string' },
}, ['stage', 'references_read', 'patch_file', 'correctness']);
const AUDIT_SCHEMA = obj({ ok: { type: 'boolean' }, findings: strs, checked: strs }, ['ok', 'findings']);
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
function attMissing(a) {
  if (!a || typeof a !== 'object' || a.status === 'not_required') return [`att (${ATT_CMD}; or status unavailable with a reason)`];
  if (a.status === 'captured') {
    const out = [];
    const cmd = a.command || '';
    const layersFlag = /--layers[ =](\S+)/.exec(cmd);
    const viaProfile = /gluon_profile\.py/.test(cmd) && (!layersFlag || /\batt\b/.test(layersFlag[1]));
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
  if (stage === 'recover') {
    const p = r.parity_gate;
    if (!p || !/parity_gate\.py/.test(p.command || '') || !isNum(p.exit_code)) out.push('parity_gate (parity_gate.py command and exit_code)');
    else if (p.mode !== 'gate') out.push('parity_gate.mode=gate (a recover round runs it as the gate)');
    if (!r.no_change) {
      if (!SUSPECT_LAYER[r.suspect]) out.push('suspect (lost_pipeline|lost_layout|lost_RA)');
      out.push(...g3Missing(r.g3_occupancy), ...g4Missing(r.g4_screen));
      if (PIPELINE_LAYERS.has(layer)) out.push(...attMissing(r.att));
    }
  }
  if (stage === 'climb') {
    if (!LAYERS.includes(String(r.layer))) out.push(`layer (one of ${LAYERS.join(', ')})`);
    if (!has(r.profile_dir)) out.push('profile_dir (this round\'s re-profile)');
    if (!/gluon_profile\.py/.test(r.profile_command || '')) out.push('profile_command (the gluon_profile.py call behind profile_dir, with --compare LAST_PROFILE)');
    out.push(...g3Missing(r.g3_occupancy), ...g4Missing(r.g4_screen));
    if (PIPELINE_LAYERS.has(String(r.layer))) out.push(...attMissing(r.att));
    if (r.inline_asm && !has(r.disassembly_ref)) out.push('disassembly_ref (inline asm is accepted on the disassembly)');
    if (ctx.entry === 'incumbent' && r.parity_gate && r.parity_gate.mode === 'gate') out.push('parity_gate.mode=diagnostic (an incumbent has no parity gate)');
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

phase('Benchmark');
const bench = await agentT(
  rolePrompt('benchmark_engineer', geakRole('benchmark_engineer'), 'setup',
    'Build the COMMANDMENT and record a reliable baseline.', {
      WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
      ANALYSIS: { kernel_type: 'gluon_authoring', note: 'see the task dir config/README for the target' },
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
// allowed to give ground; an incumbent keeps the ordinary shape.
const MAX_NO_IMPROVE = Math.max(1, Math.floor(num(A.max_no_improve, ENTRY === 'port' ? 6 : 2)));
const PROGRESS_DELTA = num(A.progress_delta, ENTRY === 'port' ? -0.05 : 0);

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
// The track: Transcribe/Author -> Recover (port) -> Climb, one engineer round at a time.
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
const requests = [];
const STAGE_PHASE = { transcribe: 'Transcribe', author: 'Transcribe', recover: 'Recover', climb: 'Climb' };

function evidenceTail(stg) {
  const lines = ['## Evidence this round must return (checked in code; a gap is sent back, then rejected)'];
  if (stg === 'transcribe') lines.push('- equivalence: layout_diff (ttgir_bridge.py verify), numeric, determinism, asm_parity', '- g3_occupancy: probe.py measure on the anchor IR, BOTH limiters');
  if (stg === 'author') lines.push('- correctness of the first Gluon kernel; g3_occupancy: probe.py measure, BOTH limiters');
  if (stg === 'recover') lines.push('- parity_gate: parity_gate.py run as the gate (mode=gate), exit code, ratio, suspects', '- for a repayment: suspect, g3_occupancy, g4_screen (ab_bench.py --permute --json), and ATT when the suspect is lost_pipeline', '- a gate-only round (nothing left to repay or nothing attributable) sets no_change=true');
  if (stg === 'climb') lines.push(`- layer (one of ${LAYERS.join(', ')}), profile_dir + profile_command (gluon_profile.py ... --compare LAST_PROFILE)`, '- g3_occupancy: probe.py measure, BOTH limiters (a missing one as null + missing_reason)', '- g4_screen: ab_bench.py --module <adapter> --permute --json', `- att for layers ${[...PIPELINE_LAYERS].join('/')}: ${ATT_CMD} with state ok, or status=unavailable with the reason`, '- inline asm rounds: disassembly_ref');
  return lines.join('\n');
}

for (let round = 1; round <= BUDGET && !stoppedBy; round++) {
  if (TOKENS && TOKENS.total && TOKENS.remaining() < TOKEN_RESERVE) { stoppedBy = 'token_budget'; break; }
  const clock = await secondsLeft(`r${round}`);
  if (clock.left <= DEADLINE_RESERVE_S) {
    deadlineHit = true; stoppedBy = 'deadline';
    log(`Deadline: ${Math.round(clock.left / 60)} min left, ${Math.round(DEADLINE_RESERVE_S / 60)} reserved for close; no round ${round}.`);
    break;
  }
  const roundEnd = Number.isFinite(clock.left)
    ? clock.now + Math.min(ROUND_TIME_CAP_S, clock.left - DEADLINE_RESERVE_S) : 0;
  // The hang guard tracks the round deadline too, with a grace period for the return itself.
  const engTimeout = roundEnd && ENGINEER_TIMEOUT_MS > 0
    ? Math.min(ENGINEER_TIMEOUT_MS, (roundEnd - clock.now + 900) * 1000) : ENGINEER_TIMEOUT_MS;
  phase(STAGE_PHASE[stage]);
  const OUT = `${GLUON_DIR}/round_${round}`;
  const roundInputs = {
    ROUND: round, STAGE: stage, ENTRY_MODE: ENTRY, TRACK, BASE_TAG, TRACK_START: trackTag,
    ROUND_TAG: `r${round}`, OUTPUT_DIR: OUT, GPU_ID: GPU_POOL, GFX, SKILL, KT, WORKFLOW_DIR, EVAL_DIR,
    COMMANDMENT, BASELINE_PER_CASE, CHAMPION_BUNDLE: entry ? entry.champion_bundle : '',
    CHAMPION_VS_BASELINE: CHAMP_PRIMARY, PARITY_THRESHOLD, PARITY: parity,
    TRACK_BEST_VS_BASELINE: trackBest, ACCEPTED_VS_BASELINE: accepted ? accepted.primary : 1,
    BUDGET_JSON: budgetRes ? budgetRes.budget_json : '', LAST_PROFILE: lastProfile,
    HISTORY: history.slice(-8), ...(roundEnd ? { ROUND_DEADLINE_EPOCH: roundEnd } : {}),
  };
  const engPrompt = (extra, tail) => rolePrompt('gluon_engineer', pkgRole('gluon_engineer'), stage,
    `Carry round ${round} of the Gluon track: stage ${stage}, one attributable change.`,
    { ...roundInputs, ...extra }, { stage, tail: tail || evidenceTail(stage) });
  let r = await agentT(engPrompt({}), { phase: STAGE_PHASE[stage], label: `gluon_engineer:${stage} r${round}`, schema: ROUND_SCHEMA }, engTimeout);
  // Completeness, then (optionally) an independent artifact audit; each gap is sent back by name.
  let missing = missingEvidence(stage, r, { entry: ENTRY });
  let audit = null;
  for (let k = 0; ; k++) {
    if (!missing.length && EVIDENCE_AUDIT && r) {
      audit = await agentT(
        rolePrompt('evidence_auditor', pkgRole('evidence_auditor'), 'audit', 'Check that the cited evidence exists and says what the round claims.', {
          EVIDENCE: r, STAGE: stage, GFX, SKILL, KT, OUTPUT_DIR: OUT, TRACK, ROUND_TAG: `r${round}`,
        }, { stage: 'audit' }),
        { phase: STAGE_PHASE[stage], label: `evidence_audit r${round}`, effort: 'low', schema: AUDIT_SCHEMA });
      if (audit && audit.ok === false) {
        missing = (audit.findings || []).map(f => `audit: ${f}`);
        if (!missing.length) missing = ['audit: failed without naming a finding; re-check every cited artifact'];
      }
      if (!audit) missing = [];   // a lost audit never blocks a complete round; the gap is logged
      if (!audit) log(`  [audit] round ${round}: auditor returned nothing; structural check stands.`);
    }
    if (!missing.length || k >= EVIDENCE_REASKS) break;
    log(`Round ${round}: evidence incomplete (${missing.join('; ')}); asking for exactly that.`);
    const prev = r;
    r = await agentT(engPrompt({ PREVIOUS_RETURN: prev, MISSING: missing },
      'Complete exactly the MISSING items for the candidate already at TRACK (do NOT start a new edit, do NOT move ' +
      'ROUND_TAG unless an item requires it). Return the whole object with the gaps filled.'),
    { phase: STAGE_PHASE[stage], label: `gluon_engineer:evidence r${round} (reask ${k + 1})`, schema: ROUND_SCHEMA }, engTimeout);
    missing = missingEvidence(stage, r, { entry: ENTRY });
  }
  const row = { round, stage, layer: r ? (stage === 'recover' ? (SUSPECT_LAYER[r.suspect] || r.layer || '') : (r.layer || '')) : '',
    claim: r ? r.claim || '' : '', status: '', vs_baseline: null, vs_champion: null, kept: false, accepted: false };
  if (r && has(r.resweep_request)) requests.push({ round, kind: 'resweep_request', ref: r.resweep_request });
  if (r && has(r.structure_suspect)) requests.push({ round, kind: 'structure_suspect', ref: r.structure_suspect });
  if (missing.length) {
    rejected++;
    row.status = 'rejected_missing_evidence'; row.missing = missing;
    history.push(row);
    log(`Round ${round} REJECTED: ${missing.join('; ')}. Track stays at ${trackTag}.`);
    if (stage === 'climb' && ++noImprove >= MAX_NO_IMPROVE) stoppedBy = 'max_no_improve';
    if (stage === 'recover' && ++recoverRounds >= MAX_RECOVER_ROUNDS) { parity = 'parity_unreached'; stoppedBy = 'parity_unreached'; }
    if (stage === 'transcribe' || stage === 'author') { stoppedBy = `${stage}_failed`; }
    continue;
  }

  // GEAK verify is the only measurement that counts. A gate-only round, or a transcribe round over an
  // anchor that is already the frozen baseline, has no new candidate to measure.
  let ver = null;
  let p = 0;
  const measured = !r.no_change && !(stage === 'transcribe' && r.anchor_provided);
  if (measured && says(r.correctness, 'pass')) {
    ver = await agentT(
      rolePrompt('verify_engineer', geakRole('verify_engineer'), 'verify', 'Independently re-measure this candidate patch.', {
        CANONICAL, PATCH: r.patch_file, VERIFY_DIR: `${OUT}/verify`, ENGINEER_WORKSPACE: TRACK, EVAL_DIR,
        WORKFLOW_DIR, GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
      }),
      { phase: STAGE_PHASE[stage], label: `verify r${round}`, schema: VERIFY_SCHEMA });
    p = ver && says(ver.correctness, 'pass') ? primary(ver) : 0;
  } else if (!measured) {
    p = r.anchor_provided ? 1 : trackBest;
  }
  row.status = ver ? String(ver.status || '') : (measured ? 'not_verified' : 'not_measured');
  row.vs_baseline = p || null;
  row.vs_champion = p ? vsChampion(p) : null;

  // Track keep (the engineer's line, may sit below the comparator) vs commit gate (GEAK acceptance).
  const prevBest = trackBest;
  const kept = p > 0 && (p > trackBest || (r.enabling_step === true && trackBest > 0 && p >= trackBest * (1 + PROGRESS_DELTA)));
  if (measured && kept) { trackBest = Math.max(trackBest, p); trackTag = r.track_tag; row.kept = true; }
  if (!measured && (stage === 'transcribe' || stage === 'author')) { trackBest = p; row.kept = true; }
  if (ver && says(ver.correctness, 'pass') && p > (accepted ? accepted.primary : 1) * (1 + MIN_IMPROVE)) {
    accepted = { round, primary: p, vs_champion: vsChampion(p), patch: r.patch_file, layer: row.layer, per_case: ver.per_case || [] };
    row.accepted = true;
  }

  if (stage === 'transcribe' || stage === 'author') {
    const eq = r.equivalence || {};
    const equivalent = stage === 'author' || (['PASS', 'RECONCILED'].includes(String(eq.layout_diff).toUpperCase()) &&
      ![eq.numeric, eq.determinism].some(v => /fail/i.test(String(v))));
    if (!equivalent || !(p > 0)) {
      stoppedBy = `${stage}_failed`;
      row.status = equivalent ? 'anchor_incorrect' : 'equivalence_failed';
      log(`Round ${round}: ${stage} did not produce a ${equivalent ? 'correct' : 'equivalent'} anchor; no recovery on a wrong anchor.`);
    } else {
      if (r.anchor_provided || !measured) { trackTag = has(r.track_tag) ? r.track_tag : BASE_TAG; }
      stage = ENTRY === 'port' ? 'recover' : 'climb';
    }
  } else if (stage === 'recover') {
    recoverRounds++;
    // G2: the tool's exit code AND the verified ratio to the champion must both clear.
    const pg = r.parity_gate;
    const ratio = vsChampion(trackBest);
    if (pg.exit_code === 0 && ratio >= PARITY_THRESHOLD) {
      parity = 'cleared'; stage = 'climb';
      log(`Round ${round}: parity CLEARED (parity_gate exit 0, verified ${ratio.toFixed(3)}x of the champion).`);
    } else {
      if (pg.exit_code === 0) log(`Round ${round}: parity_gate exit 0 contradicted by verify (${ratio.toFixed(3)}x < ${PARITY_THRESHOLD}); not cleared.`);
      if (recoverRounds >= MAX_RECOVER_ROUNDS) {
        parity = 'parity_unreached'; stoppedBy = 'parity_unreached';
        log(`Parity not reached in ${recoverRounds} recover round(s): do not climb (G2).`);
      }
    }
    row.parity_exit = pg.exit_code;
  } else if (stage === 'climb') {
    noImprove = (row.kept && p > prevBest) ? 0 : noImprove + 1;
    if (noImprove >= MAX_NO_IMPROVE) stoppedBy = 'max_no_improve';
  }
  if (row.kept) lastProfile = r.profile_dir || lastProfile;
  if (audit && Array.isArray(audit.findings) && audit.findings.length) row.audit = audit.findings.slice(0, 5);
  history.push(row);
  log(`Round ${round} ${row.stage}${row.layer ? ` L${row.layer}` : ''}: ${row.status} ` +
      `vs_baseline=${p ? p.toFixed(3) : '-'} vs_champion=${p ? vsChampion(p).toFixed(3) : '-'}` +
      `${row.kept ? ' kept' : ''}${row.accepted ? ' ACCEPTED' : ''}`);
}
if (!stoppedBy) stoppedBy = 'budget';
if (parity === 'pending' && ENTRY === 'port' && !blocked) parity = 'parity_unreached';
if (parity === 'parity_unreached') caveats.push('parity_unreached');
if (rejected) caveats.push(`${rejected} round(s) rejected for missing evidence`);

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
  PARITY: parity, CAVEATS: caveats, HISTORY: history, REQUESTS: requests,
  ACCEPTED: accepted ? { round: accepted.round, vs_baseline: accepted.primary, vs_champion: accepted.vs_champion,
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
log(`COMPLETE. ${KERNEL_NAME}: entry=${ENTRY} outcome=${outcome} stopped_by=${stoppedBy} parity=${parity}` +
    `${finalSpeedup != null ? ` director-verified ${Number(finalSpeedup).toFixed(3)}x` : ''}. Results in ${EVAL_DIR}`);

return {
  workflow: 'gluon_authoring', kernel_name: KERNEL_NAME, eval_dir: EVAL_DIR, gfx: GFX,
  entry_mode: ENTRY, outcome, blocked, stopped_by: stoppedBy, deadline_hit: deadlineHit, parity,
  caveats, requests, rounds: history, rounds_used: history.length, budget: BUDGET,
  champion_vs_baseline: CHAMP_PRIMARY,
  accepted: accepted ? { round: accepted.round, vs_baseline: accepted.primary, vs_champion: accepted.vs_champion,
    layer: accepted.layer, patch: accepted.patch } : null,
  final_patch: FINAL_SRC ? (closeRes && has(closeRes.final_patch) ? closeRes.final_patch : `${EVAL_DIR}/final_patch.diff`) : '',
  reverted_to_champion: !!revertToChampion,
  report_path: closeRes ? closeRes.report_path : '',
  closure_review_path: closeRes ? closeRes.closure_review_path || '' : '',
  validation_status: validation ? validation.validation_status : (FINAL_SRC ? 'not_validated' : 'no_candidate'),
  director_verified_speedup_geomean: finalSpeedup,
};
