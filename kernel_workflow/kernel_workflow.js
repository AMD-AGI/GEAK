export const meta = {
  name: 'kernel-workflow',
  description: 'Single ENTRY POINT for kernel optimization on AMD GPUs: Instinct MI-series (CDNA gfx942/gfx950), the validated RDNA4 product Radeon AI PRO R9700 (gfx1201) and the RDNA3 Radeon Pro W7900 (gfx1100). The worker validates on-box identity via structured rocminfo data; generic gfx1201 is not a product identity, and gfx1100 without the exact W7900 product is refused. Dispatches on args.mode: optimize/author -> delegate one unchanged single-language lane to the kernel_lane worker (backward compatible); bakeoff -> freeze the input kernel into ONE immutable oracle + frozen baseline, discover per-language existing impls + offline-tune env backends (aiter/CK), then run one worker lane per backend language (HIP/Triton/FlyDSL/CK/...) in parallel over the GPU pool and pick the fastest verified result across ALL candidates (author/optimize lanes AND the tuned env backend) — every one scored against the SAME frozen original baseline (anti-cheating). Wraps the unchanged kernel_lane worker (one workflow() nesting level; the dispatcher is the bake-off orchestrator).',
  whenToUse: 'Optimize a kernel. Three modes, all via args.mode (there is NO natural-language mode detection — the caller picks): mode=optimize (DEFAULT) speeds up an EXISTING kernel and behaves exactly like the old single-language workflow; mode=author writes a fresh implementation from scratch, then optimizes it — use it when there is no source to edit yet, or to port the op to another language (pass args.target_language); mode=bakeoff tries several backend languages in parallel and keeps the fastest (pass args.backends, or leave empty to auto-discover). Bakeoff Freeze establishes structured product/ISA identity before Discover; optional expected_gfx/expected_target pins are validated against it. Anything else throws. Pass args.kernel_path (required), args.workflow_dir (required), args.mode, args.target_language, args.backends, args.budget, args.gpu_ids, args.gpu_mode (pool|pin, default pool), args.gpus_per_job / args.job_gpu_ids (optional: a task whose test runs on several GPUs is detected at Setup; pass these only to override; optimize/author only; gpu_ids must hold enough GPUs). args.config_aware="true" + args.family_spec (an operator family: one kernel_lane per config, in parallel over gpu_ids, then family_plan/results/report + an exact-signature implementation_catalog; kernel_path is not used on that path).',
  phases: [
    { title: 'Freeze',   detail: 'oracle_freezer: freeze the input kernel -> immutable oracle + baseline_src/ (the ONE denominator) [bakeoff] | create the family eval dir [config_aware]' },
    { title: 'Discover', detail: 'op_benchmarker: per-language existing-impl probe + measure + OFFLINE env tune (aiter/CK, shapes from the frozen oracle) -> author_plan, best_known_ms [bakeoff] | operator_family_planner: groups + strategy hints [config_aware]' },
    { title: 'Bakeoff',  detail: 'one unchanged kernel_lane worker per language (optimize if an impl exists, else author), parallel over the GPU pool [bakeoff] | one kernel_lane per family config [config_aware] | the single lane [optimize/author]' },
    { title: 'Report',   detail: 'rank ALL candidates (lanes + tuned env backend) on the SAME frozen baseline; write the comparison table + winner (+ optional apply_to_original), then curate ONE distilled card into knowledge/learned/ on a measured win [bakeoff] | family results, report and catalog [config_aware]' },
  ],
};

// ---------------------------------------------------------------------------
// Args. This is a DISPATCHER: on mode=optimize/author it forwards straight to the
// single-language worker (kernel_lane.js) unchanged; on mode=bakeoff it acts
// as the multi-language bake-off orchestrator; with config_aware on it runs one
// worker per operator-family config. The worker script is a sibling in this same
// dir, so it is called via workflow() at exactly ONE nesting level
// (dispatcher=level0 -> worker=level1). The dispatcher never nests a second level.
// ---------------------------------------------------------------------------
const A = args || {};
// Operator-family path (see "Config-aware operator family" below). Only "true"/"on" enable it; every
// other value, including absent, keeps the optimize/author/bakeoff contract byte-for-byte.
const CONFIG_AWARE = ['true', 'on'].includes(
  String(A.config_aware != null ? A.config_aware : 'false').trim().toLowerCase());
if (!CONFIG_AWARE && !A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/op dir)');
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_workflow.js + kernel_lane.js + roles/ + knowledge/ (i.e. the dirname of this script).');
}
// The single-language worker lane (unchanged behavior). Same dir as this dispatcher.
if (A.optimizer != null && typeof A.optimizer !== 'string') {
  throw new Error('optimizer must be greedy or evo (a string)');
}
const OPTIMIZER = String(A.optimizer || '').trim().toLowerCase();
if (OPTIMIZER && !['greedy', 'evo'].includes(OPTIMIZER)) {
  throw new Error('optimizer must be greedy or evo');
}
if (OPTIMIZER === 'evo' && A.kernel_lane_script) {
  throw new Error('optimizer=evo cannot override kernel_lane_script');
}
const WORKER = String(A.kernel_lane_script || `${WORKFLOW_DIR}/kernel_lane.js`);
const MODE = String(A.mode != null ? A.mode : 'optimize').trim().toLowerCase() || 'optimize';
if (OPTIMIZER === 'evo' && MODE === 'bakeoff') {
  throw new Error('optimizer=evo currently supports optimize and author lanes, not bakeoff');
}
// Same default kernel_lane.js computes for its EXP_ROOT (args.exp_root is optional).
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

// ---- Live execution tracker (read-only observer; see interface/geak_trace_collector.py) ----
// Started HERE so a direct kernel run is tracked from its first agent, not only
// at the end. Every dispatcher path passes through this point, so one call covers
// them. The observer resolves its own workflow run from the runtime's workflow
// record (no eval-dir substring matching) — needed because no eval_dir exists yet
// on the pass-through branch.
//
// Detached, read-only and strictly non-fatal: it never influences model choice,
// prompts, budgets or optimization decisions. Set GEAK_LIVE_TRACE=0 to disable.
(function startLiveTracker() {
  try {
    if (String(process.env.GEAK_LIVE_TRACE || '1') === '0') return;
    const { spawn } = require('child_process');
    const child = spawn('python3', [
      '-B', `${WORKFLOW_DIR}/../interface/geak_trace_collector.py`,
      '--exp-root', EXP_ROOT,
      // Runtime-supplied invocation identity: the wf_*.json record carries this
      // same args object, so matching it identifies THIS launch deterministically
      // instead of guessing from timing or from being the only run around.
      '--identity-args', JSON.stringify(A),
      '--script-dir', WORKFLOW_DIR,
      // Per-run filename: two runs under one exp_root must not overwrite each other.
      '--out-dir', EXP_ROOT,
      '--watch',
      '--interval', String(process.env.GEAK_LIVE_TRACE_INTERVAL_S || '30'),
      '--resolve-timeout', '600',
      '--max-seconds', String(process.env.GEAK_LIVE_TRACE_MAX_S || '172800'),
    ], { detached: true, stdio: 'ignore' });
    // spawn reports a missing executable ASYNCHRONOUSLY: try/catch cannot see it,
    // and without this listener the ENOENT is an unhandled error event.
    child.on('error', (err) => {
      try { log(`Live execution tracker failed to start (non-fatal): ${err && err.message}`); }
      catch (_) {}
    });
    child.unref();
    log(`Live execution tracker started -> ${EXP_ROOT}/geak_trace_<runId>.json`);
  } catch (e) {
    try { log(`Live execution tracker not started (non-fatal): ${e && e.message}`); } catch (_) {}
  }
})();

// ===========================================================================
// Shared helpers (config_aware, pass-through and bakeoff paths).
// ===========================================================================
// ---- GPU families: keep this block identical in kernel_workflow.js, kernel_lane.js and e2e_workflow.js ----
// Per-GPU-family policy, looked up by ISA. The Workflow sandbox has no import, so each workflow carries
// a copy (kernel_workflow/scripts/test_gpu_family_table.js keeps them identical). An ISA no row
// matches gets GPU_FAMILY_DEFAULT, the Instinct (CDNA) policy. Each workflow reads the fields it needs.
//   name        how the bake-off Discover prompt names an isolated family (gfx1201 keeps
//               upstream's 'R9700' label)
//   isolate     withhold the MI-series-calibrated inputs (perf_knowledge cards, expert skills, learned
//               KB, warm start, tuning skillset); the in-tree guide is the hardware reference
//   matrixCore  the matrix ISA the prompts name
const GPU_FAMILIES = [
  { name: 'R9700', isa: /^gfx1201$/, isolate: true, matrixCore: 'WMMA' },
  // RDNA 3.5 APUs (Strix Halo is gfx1151). Only a gfx1151 Radeon 8060S has been run.
  { name: 'RDNA 3.5', isa: /^gfx115\d$/, isolate: true, matrixCore: 'WMMA' },
  // Discrete RDNA3. The workflows accept gfx1100 only as the exact Radeon Pro W7900 (target w7900).
  { name: 'W7900', isa: /^gfx1100$/, isolate: true, matrixCore: 'WMMA' },
];
const GPU_FAMILY_DEFAULT = { name: 'default', isa: null, isolate: false, matrixCore: 'MFMA' };
const gpuFamily = (gfx) => GPU_FAMILIES.find((f) => f.isa.test(String(gfx || ''))) || GPU_FAMILY_DEFAULT;
// ---- end GPU families ----

// Schema helpers.
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const arrStr = { type: 'array', items: { type: 'string' } };
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };

// Prompt + agent helpers (self-contained; mirror kernel_lane.js / e2e_workflow.js).
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// Hung-agent + API-fault guard (same contract as kernel_lane.js:agentT).
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 3600000, 10);
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
// LLM token+time accounting (PURELY ADDITIVE; args.llm_stats="false" makes it a no-op).
// DELIBERATELY NO TIMESTAMPS: Date.now()/new Date() are unavailable in workflow scripts. Every
// duration in the report comes from the transcripts (scripts/llm_ledger.py); this records only the
// role/phase/attempt identity of each agent call so the ledger attributes tokens to the right agent.
const LLM_STATS = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
const LLM_TL = { schema: 'geak.agent_timeline/1', workflow: 'kernel_workflow', events: [], nested: [] };
const TL_ROLE_RE = /You are the ([A-Za-z0-9_.\-]+)\.\s*PHASE=([A-Za-z0-9_.\-]+)\./;
// Called AT DISPATCH (before the await) so LLM_TL.events is in dispatch order, not completion order.
// Returns the event so the caller flips `ok` once the attempt resolves (null when off).
function tlAgent(prompt, o, attempt) {
  if (!LLM_STATS) return null;
  const m = TL_ROLE_RE.exec(String(prompt || ''));
  const e = {
    seq: LLM_TL.events.length,
    phase: (o && o.phase) || '',
    label: (o && o.label) || 'agent',
    role: m ? m[1] : '',
    sub_phase: m ? m[2] : '',
    attempt: attempt,
    ok: false,
  };
  LLM_TL.events.push(e);
  return e;
}
async function agentT(p, o) {
  const label = (o && o.label) ? o.label : 'agent';
  for (let attempt = 1; attempt <= AGENT_RETRIES; attempt++) {
    const ev = tlAgent(p, o, attempt);   // record AT DISPATCH; ok=false until it resolves
    try {
      if (typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) {
        const r0 = await agent(p, o); if (ev && r0) ev.ok = true; return r0;
      }
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(AGENT_TIMEOUT_MS / 60000)}min with no return — resolving null.`);
          resolve(null);
        }, AGENT_TIMEOUT_MS);
      });
      const r = await Promise.race([
        agent(p, o).then((rr) => { clearTimeout(to); return rr; }, (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
      if (ev && r) ev.ok = true;
      return r;
    } catch (e) {
      const msg = String(e && e.message ? e.message : e).slice(0, 200);
      if (attempt < AGENT_RETRIES) {
        log(`  [api-fault guard] ${label} attempt ${attempt}/${AGENT_RETRIES} error (${msg}) — retrying.`);
        continue;
      }
      log(`  [api-fault guard] ${label} failed after ${AGENT_RETRIES} attempts (${msg}) — resolving null.`);
      return null;
    }
  }
  return null;
}

// Role prompt from an arbitrary roles/ dir (bakeoff references e2e's op_benchmarker in place).
function roleAgentFrom(dir, role, phaseName, intro, inputs, extra) {
  return `You are the ${role}. PHASE=${phaseName}.
First Read ${dir}/roles/${role}.md and follow its instructions for PHASE=${phaseName}.
Read any knowledge files it points you to under ${dir}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).` + (extra || '');
}
// This dispatcher's own roles live under WORKFLOW_DIR/roles/.
const roleAgent = (role, phaseName, intro, inputs) => roleAgentFrom(WORKFLOW_DIR, role, phaseName, intro, inputs);

// GPU semaphore: 1 GPU per lane, EXCLUSIVE. size==1 serializes lanes. (Lifted from e2e_workflow.js.)
function makeSem(ids) {
  const free = ids.slice(); const waiters = [];
  const pump = () => { while (waiters.length && waiters[0].n <= free.length) {
    const w = waiters.shift(); w.resolve(free.splice(0, w.n)); } };
  return {
    size: ids.length,
    acquire(n = 1) { if (n <= free.length) return Promise.resolve(free.splice(0, n));
      return new Promise((resolve) => { waiters.push({ n, resolve }); }); },
    release(got) { free.push(...got); pump(); },
    async with(n, fn) { const g = await this.acquire(n); try { return await fn(g); } finally { this.release(g); } },
  };
}

// Final step of every top-level run: persist the agent timeline, then render the run report (ledger ->
// clickable role-execution-tree HTML + MD twin). kernel_workflow is always a TOP-LEVEL entry point (E2E
// nests kernel_lane.js, not this dispatcher), so it owns its own ledger + report. The script has no
// filesystem access, so a tiny agent writes the timeline and runs the report driver. Entirely
// best-effort: accounting must never fail a run that produced a real speedup.
async function emitRunReport(evalDir, timeline, phaseName) {
  try {
    const tlPath = `${evalDir}/reports/trace/agent_timeline.json`;
    await agentT(
      `You are the file_writer. PHASE=persist_llm_stats.\n` +
      `Do exactly two things, in order:\n` +
      `1. Use the Write tool to create "${tlPath}" with EXACTLY the JSON below, verbatim ` +
      `(create parent directories if needed; do NOT reformat, truncate or summarize it):\n\n` +
      '```json\n' + JSON.stringify({ ...timeline, instance: evalDir }) + '\n```\n\n' +
      `2. Then run this Bash command (best-effort; if it fails, carry on and report ok=false).\n` +
      `   It runs the token/time/cost ledger AND renders the role-execution-tree report (HTML + MD):\n` +
      `   python3 -B "${WORKFLOW_DIR}/../interface/geak_report.py" --eval-dir "${evalDir}"\n\n` +
      `Then return {"written": true, "path": "${tlPath}", "ok": <true if the command exited 0 else false>}.`,
      { phase: phaseName, label: 'file_writer:persist_llm_stats',
        schema: obj({ written: { type: 'boolean' }, path: { type: 'string' }, ok: { type: 'boolean' } }, []) });
    const n = (timeline.nested || []).length;
    log(`LLM token+time+cost ledger -> ${evalDir}/reports/trace/ and run report (HTML+MD) -> ${evalDir}/report/.` +
        (timeline.events ? ` ${timeline.events.length} agent attempts recorded${n ? ` plus ${n} nested lane(s)` : ''}.` : ''));
  } catch (e) {
    log(`LLM stats/report emit failed (NON-FATAL — the run is unaffected): ${String(e)}`);
  }
}

// ===========================================================================
// CONFIG-AWARE OPERATOR FAMILY (args.config_aware = "true" | "on"; docs/config_aware_family_design.md).
// One kernel_lane.js per family_spec config, launched together over the GPU pool like bakeoff (one GPU id
// per lane, a thrown lane is caught), so nesting stays dispatcher -> kernel_lane. Validation, statuses,
// aggregate and catalog are computed here; agents only propose an advisory plan and write files. The
// catalog covers each config's own exact signature and is never deployment_ready.
// ===========================================================================
if (CONFIG_AWARE) {
  const fail = (m) => { throw new Error(m); };
  if (MODE === 'bakeoff') fail('config_aware does not support mode=bakeoff yet; set each config\'s mode in family_spec.');
  if (MODE !== 'optimize' && MODE !== 'author') fail(`unknown mode='${MODE}' with config_aware; each config carries its own mode.`);
  if (OPTIMIZER) fail(`config_aware does not support optimizer=${OPTIMIZER} yet`);
  if (String(A.gpus_per_job != null ? A.gpus_per_job : '1').trim() !== '1' || String(A.job_gpu_ids || '').trim()) {
    fail('config_aware gives each lane one GPU id; gpus_per_job > 1 and job_gpu_ids are not supported.');
  }
  if (A.apply_to_original != null && String(A.apply_to_original).trim().toLowerCase() !== 'false') {
    fail('config_aware never modifies an input kernel_path; apply_to_original must be "false".');
  }
  if (/(^|\/)kernel_workflow\.js$/.test(WORKER)) fail('config_aware lanes must run kernel_lane.js, not this dispatcher.');
  const isObj = (v) => v != null && typeof v === 'object' && !Array.isArray(v);
  const S = A.family_spec;
  if (!isObj(S)) fail('config_aware requires args.family_spec as an already-parsed JSON object (not a string or a path).');

  const isInt = Number.isInteger;
  const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  const str = (v) => (typeof v === 'string' && v.trim() ? v.trim() : null);
  const low = (v) => String(v == null ? '' : v).trim().toLowerCase();
  const clip = (v, n) => (str(v) || '').slice(0, n);
  // Key-order-independent JSON, so equal contracts compare equal however they were written.
  const canon = (v) => Array.isArray(v) ? `[${v.map(canon).join(',')}]`
    : isObj(v) ? `{${Object.keys(v).sort().map((k) => `${JSON.stringify(k)}:${canon(v[k])}`).join(',')}}`
    : JSON.stringify(v === undefined ? null : v);
  const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
  const LANE_FIELDS = ['mode', 'target_language', 'budget', 'gpu_mode', 'task',
    'use_learned_kb', 'update_experience', 'warm_start', 'apply_to_original'];
  // Values kernel_lane.js understands for each switch; anything else would silently become its default.
  const ENUMS = {
    mode: ['optimize', 'author'], gpu_mode: ['pool', 'pin'], use_learned_kb: ['true', 'false'], apply_to_original: ['false'],
    update_experience: ['on', 'off', 'true', 'false', 'none'],
    warm_start: ['on', 'off', 'true', 'false', 'none', 'reference', 'return_after_read'],
  };

  // ---- Phase 1: validate (synchronous, nothing spawned yet); every problem is reported in one error ----
  const errs = [];
  const need = (cond, msg) => { if (!cond) errs.push(msg); return !!cond; };
  const onlyFields = (o, allowed, at) =>
    Object.keys(o).forEach((k) => need(allowed.includes(k), `${at}.${k} is not a supported field`));
  need(S.schema_version === 1, `schema_version must be 1 (got ${JSON.stringify(S.schema_version)})`);
  const FAMILY_ID = str(S.family_id) || '';
  need(SAFE_ID.test(FAMILY_ID), 'family_id must be a non-empty [A-Za-z0-9._-] identifier');
  const OPERATOR = str(S.operator) || '';
  need(OPERATOR, 'operator must be a non-empty string');
  const defaults = S.defaults == null ? {} : S.defaults;
  if (need(isObj(defaults), 'defaults must be an object')) onlyFields(defaults, [...LANE_FIELDS, 'gpu_ids'], 'defaults');
  const dep = S.deployment == null ? null : S.deployment;
  need(dep == null || isObj(dep), 'deployment must be an object');
  const axes = isObj(dep) && dep.signature_axes != null ? dep.signature_axes : null;
  need(axes == null || (Array.isArray(axes) && axes.length && axes.every(str)),
    'deployment.signature_axes must be a non-empty array of axis names');
  need(S.target == null || isObj(S.target), 'target must be an object');
  const pins = [A, isObj(S.target) ? S.target : {}].map((o) => [low(o.expected_gfx), low(o.expected_target)]);
  for (const [g, t] of pins) {
    need(!!g === !!t && (!g || /^gfx[0-9a-f]+$/.test(g)), 'expected_gfx (a gfx token) and expected_target must be supplied together');
  }
  need(!pins[0][0] || !pins[1][0] || pins[0].join() === pins[1].join(),
    `expected_gfx/expected_target (${pins[0].join('/')}) conflict with family_spec.target (${pins[1].join('/')})`);
  const [LANE_GFX, LANE_TARGET] = pins[0][0] ? pins[0] : pins[1];

  const list = Array.isArray(S.configs) ? S.configs : [];
  need(list.length, 'configs must be a non-empty array');
  let rank = axes ? axes.length : null;
  const CONFIGS = list.map((c, i) => {
    if (!need(isObj(c), `configs[${i}] must be an object`)) return null;
    const id = str(c.id) || '';
    const at = SAFE_ID.test(id) ? `config '${id}'` : `configs[${i}]`;
    need(SAFE_ID.test(id), `${at}.id must be a non-empty [A-Za-z0-9._-] identifier`);
    need(str(c.kernel_path) && c.kernel_path.trim().startsWith('/'), `${at}: kernel_path must be an absolute path`);
    need(isObj(c.op_spec), `${at}: op_spec must be an object`);
    const ov = c.overrides == null ? {} : c.overrides;
    if (need(isObj(ov), `${at}: overrides must be an object`)) onlyFields(ov, LANE_FIELDS, `${at}.overrides`);
    // config.overrides -> family_spec.defaults -> kernel_lane default (target_language has no default).
    const pick = (f) => [ov, defaults].map((o) => (isObj(o) ? o[f] : undefined)).find((v) => v != null);
    const knobs = {};
    for (const [f, allowed] of Object.entries(ENUMS)) {
      const v = low(pick(f));
      if (pick(f) != null && need(allowed.includes(v), `${at}: ${f} must be one of ${allowed.join('|')} (got '${v}')`)) knobs[f] = v;
    }
    const lang = low(pick('target_language'));
    need(lang, `${at}: target_language must resolve from overrides or defaults (it selects the warm-start KB page)`);
    const b = pick('budget');
    const budget = b == null ? 6 : (/^\s*\d+\s*$/.test(String(b)) ? Number(b) : NaN);
    need(budget > 0, `${at}: budget must be a positive integer`);
    need(pick('task') == null || typeof pick('task') === 'string', `${at}: task must be a string`);
    const sig = c.signature == null ? null : c.signature;
    if (sig != null && need(Array.isArray(sig) && sig.length && sig.every((x) => isInt(x) && x >= 0),
      `${at}: signature must be a non-empty array of non-negative integers`)) {
      if (rank == null) rank = sig.length;
      need(sig.length === rank, `${at}: signature rank ${sig.length} does not match ` +
        `${axes ? 'deployment.signature_axes' : 'the other configs'} (${rank})`);
    }
    need(c.signature_ranges == null || isObj(c.signature_ranges), `${at}: signature_ranges must be an object keyed by axis index`);
    for (const [ax, r] of Object.entries(isObj(c.signature_ranges) ? c.signature_ranges : {})) {
      need(Array.isArray(sig) && Array.isArray(r) && r.length === 2 && r.every(isInt) && r[0] <= sig[ax] && sig[ax] <= r[1],
        `${at}: signature[${ax}] lies outside signature_ranges['${ax}'] (a range is [lo, hi] integers)`);
    }
    const ct = c.contract == null ? null : c.contract;
    need(ct == null || isObj(ct), `${at}: contract must be an object`);
    for (const k of ['semantics', 'graph']) need(!isObj(ct) || ct[k] == null || isObj(ct[k]), `${at}: contract.${k} must be an object`);
    need(c.incumbent == null || isObj(c.incumbent), `${at}: incumbent must be an object`);
    need(c.workload_weight == null || (num(c.workload_weight) ?? -1) >= 0, `${at}: workload_weight must be a finite number >= 0`);
    need(c.priority == null || isInt(c.priority), `${at}: priority must be an integer`);
    return {
      id, index: i, kernel_path: String(c.kernel_path || '').trim().replace(/(.)\/+$/, '$1'), op_spec: c.op_spec,
      mode: knobs.mode || 'optimize', target_language: lang, budget, gpu_mode: knobs.gpu_mode || 'pool',
      kb: Object.fromEntries(['use_learned_kb', 'update_experience', 'warm_start'].filter((k) => knobs[k]).map((k) => [k, knobs[k]])),
      task: pick('task') || '', signature: Array.isArray(sig) ? sig.slice() : null, contract: isObj(ct) ? ct : null,
      incumbent: isObj(c.incumbent) ? c.incumbent : null, workload_weight: num(c.workload_weight),
      priority: isInt(c.priority) ? c.priority : null,
    };
  });
  const ids = CONFIGS.filter(Boolean).map((c) => c.id).filter((id) => SAFE_ID.test(id));
  [...new Set(ids.filter((id, k) => ids.indexOf(id) !== k))].forEach((id) => errs.push(`duplicate config id '${id}'`));
  const GPUS = String(A.gpu_ids != null ? A.gpu_ids : (isObj(defaults) && defaults.gpu_ids != null ? defaults.gpu_ids : '0'))
    .split(',').map((s) => s.trim()).filter(Boolean);
  need(GPUS.length && new Set(GPUS).size === GPUS.length, `gpu_ids '${GPUS.join(',')}' is empty or lists a GPU twice`);
  const PINNED = String(A.eval_dir != null ? A.eval_dir : '').trim();
  need(!PINNED || PINNED.startsWith('/'), 'eval_dir must be an absolute path');
  if (errs.length) fail(`config_aware: invalid input (${errs.length} problem(s)):\n  - ${errs.join('\n  - ')}`);

  const byId = new Map(CONFIGS.map((c) => [c.id, c]));
  const BUDGET = {
    rule: 'each config spends its own lane budget; lanes neither share nor pool budget',
    per_config: Object.fromEntries(CONFIGS.map((c) => [c.id, c.budget])),
    total: CONFIGS.reduce((s, c) => s + c.budget, 0),
  };
  const CROSS_CONFIG = {
    status: 'unsupported',
    reason: 'no candidate adapter: a lane patch is relative to its own workspace, so no implementation was run on a sibling config',
  };
  // Knobs forwarded from the parent call to every lane, only when the caller set them.
  const PARENT_FORWARD = ['exp_root', 'expected_device_name', 'expected_physical_cu_count', 'gpu_perf_level',
    'kb_artifacts_dir', 'kb_mode', 'kb_store_dir', 'kb_framework_version', 'perf_knowledge_dir',
    'use_expert_skills', 'expert_skills_dir', 'llm_stats', 'profile', 'fast_mode', 'max_rounds', 'min_improve',
    'candidate_floor', 'engineer_minutes', 'deep_cost', 'max_no_improve', 'dra_enabled', 'dra_max_questions',
    'dra_blindspot', 'dra_max_blindspots', 'agent_timeout_ms', 'agent_retries'];
  // JSON is rendered here and copied verbatim, so the writer cannot change a status or a number.
  const persist = async (label, files, render) => {
    const r = await agentT('You are the file_writer. PHASE=persist_family_artifacts.\n' +
      'Use the Write tool to create each file below with EXACTLY the content between its four-backtick fences, ' +
      'verbatim (create parent directories if needed; do NOT reformat, reinterpret, truncate or summarize it).\n\n' +
      files.map((f) => `### ${f.path}\n\`\`\`\`\n${f.content}\n\`\`\`\``).join('\n\n') +
      (render ? `\n\n### ${render.path}\n${render.how}` : '') +
      '\n\nWrite nothing else. Then return {"written": [<every path you wrote>]}.',
      { phase: 'Report', label, schema: obj({ written: arrStr }, ['written']) });
    const written = r && Array.isArray(r.written) ? r.written : [];
    const missing = [...files, ...(render ? [render] : [])].map((f) => f.path).filter((p) => !written.includes(p));
    if (missing.length) log(`  ${label}: not confirmed written: ${missing.join(', ')}`);
    return !missing.length;
  };

  phase('Freeze');
  log(`config_aware: family ${FAMILY_ID} (${OPERATOR}), ${CONFIGS.length} config(s), GPUs=${GPUS.join(',')} => ` +
      `concurrency ${Math.min(CONFIGS.length, GPUS.length)}; total lane budget ${BUDGET.total}.`);
  let FAMILY_DIR = PINNED.replace(/(.)\/+$/, '$1');
  if (!FAMILY_DIR) {
    const prefix = `${EXP_ROOT}/family_${FAMILY_ID}_`;
    const made = await agentT('You are the file_writer. PHASE=family_setup.\n' +
      'Create one NEW, empty directory for an operator-family run. Run exactly this Bash command:\n' +
      `  mkdir -p "${EXP_ROOT}" && d="${prefix}$(date -u +%Y%m%dT%H%M%SZ)_$$" && mkdir "$d" && echo "$d"\n` +
      'If the final mkdir fails because the directory exists, run the command again. Create nothing else.\n' +
      'Return {"eval_dir": "<the absolute path the command printed>"}.',
      { phase: 'Freeze', label: 'file_writer:family_setup', schema: obj({ eval_dir: { type: 'string' } }, ['eval_dir']) });
    const got = (str(made && made.eval_dir) || '').replace(/\/+$/, '');
    if (!got.startsWith(prefix) || got.length === prefix.length || /(^|\/)\.\.(\/|$)/.test(got)) {
      log(`config_aware: family eval dir setup failed (got '${got}'); no lane was started.`);
      return { schema_version: 1, mode: 'config_aware', config_aware: true, family_id: FAMILY_ID,
        validation_status: 'setup_failed', eval_dir: null, configs: [],
        reason: 'could not create a family eval dir; pass args.eval_dir to pin one' };
    }
    FAMILY_DIR = got;
  }
  const P = {
    plan: `${FAMILY_DIR}/family_plan.json`, results: `${FAMILY_DIR}/family_results.json`,
    report: `${FAMILY_DIR}/family_report.md`, catalog: `${FAMILY_DIR}/implementation_catalog.json`,
  };

  // ---- Phase 2: plan (advisory; an invalid answer falls back to one group per config) ----
  phase('Discover');
  // The planner reads the learned KB: off when any config turns it off (a KB-off arm stays KB-off) or the
  // expected GPU family is isolated from the MI-series-calibrated knowledge.
  const PLANNER_KB = CONFIGS.some((c) => c.kb.use_learned_kb === 'false') || gpuFamily(LANE_GFX).isolate ? 'off' : 'on';
  const raw = await agentT(
    roleAgent('operator_family_planner', 'plan', 'Write no files and run no benchmarks.', {
      FAMILY_ID, OPERATOR, DESCRIPTION: str(S.description) || '', DEPLOYMENT: dep || {}, SIGNATURE_AXES: axes || [],
      EXPECTED_GFX: LANE_GFX || 'unknown (each lane detects the GPU on-box)', GPU_POOL_SIZE: GPUS.length, LEARNED_KB: PLANNER_KB,
      OBJECTIVE: 'per-config verified speedup against each config\'s own baseline; each config runs its own lane',
      CONFIGS: CONFIGS.map(({ index, kernel_path, gpu_mode, budget, kb, task, incumbent, ...c }) => ({ ...c, has_incumbent: !!incumbent })),
    }),
    // The role file specifies each group's fields; planError() below checks them.
    { phase: 'Discover', label: 'operator_family_planner:plan',
      schema: obj({ groups: arrObj, execution_order: arrStr, notes: { type: 'string' } }, ['groups']) });
  const planError = (() => {
    if (!isObj(raw)) return 'planner returned no object';
    if (!Array.isArray(raw.groups) || !raw.groups.length) return 'planner returned no groups';
    const owner = new Map();
    const gids = new Set();
    for (const [gi, g] of raw.groups.entries()) {
      const gid = isObj(g) ? str(g.group_id) || '' : '';
      if (!SAFE_ID.test(gid)) return `groups[${gi}].group_id is empty or not [A-Za-z0-9._-]`;
      if (gids.has(gid)) return `duplicate group_id '${gid}'`;
      gids.add(gid);
      if (!Array.isArray(g.config_ids) || !g.config_ids.length) return `group '${gid}' has no config_ids`;
      for (const id of g.config_ids) {
        if (!byId.has(id)) return `group '${gid}' names unknown config '${id}'`;
        if (owner.has(id)) return `config '${id}' is assigned to both '${owner.get(id)}' and '${gid}'`;
        owner.set(id, gid);
      }
      for (const part of ['semantics', 'graph']) {
        if (new Set(g.config_ids.map((id) => canon((byId.get(id).contract || {})[part]))).size > 1) {
          return `group '${gid}' mixes configs with different contract.${part}`;
        }
      }
    }
    const missing = CONFIGS.filter((c) => !owner.has(c.id)).map((c) => c.id);
    return missing.length ? `planner omitted config(s): ${missing.join(', ')}` : null;
  })();
  if (planError) log(`config_aware: planner output rejected (${planError}); using one group per config.`);
  const eo = !planError && raw.execution_order;
  const orderOk = Array.isArray(eo) && eo.length === CONFIGS.length && new Set(eo).size === eo.length && eo.every((id) => byId.has(id));
  const PLAN = {
    schema_version: 1, family_id: FAMILY_ID, operator: OPERATOR,
    source: planError ? 'fallback' : 'planner', fallback_reason: planError,
    groups: (planError ? CONFIGS.map((c) => ({ group_id: c.id, config_ids: [c.id] })) : raw.groups).map((g) => ({
      group_id: str(g.group_id), config_ids: g.config_ids.slice(),
      strategy: clip(g.strategy, 600), reasoning: clip(g.reasoning, 600),
      implementation_form: ['shared_kernel', 'per_signature', 'internal_buckets'].includes(g.implementation_form) ? g.implementation_form : null,
      confidence: num(g.confidence) != null && g.confidence >= 0 && g.confidence <= 1 ? g.confidence : null,
      config_hints: Object.fromEntries(g.config_ids.map((id) => [id, clip(isObj(g.config_hints) ? g.config_hints[id] : '', 400)])
        .filter(([, h]) => h)),
    })),
    // Default order: priority ascending, then input order. It only sorts the report; every lane launches together.
    execution_order: orderOk ? eo.slice() : CONFIGS.slice()
      .sort((a, b) => (a.priority ?? Infinity) - (b.priority ?? Infinity) || a.index - b.index).map((c) => c.id),
    order_source: orderOk ? 'planner' : 'default', notes: planError ? '' : clip(raw.notes, 800),
    gpu_pool: GPUS, concurrency: Math.min(CONFIGS.length, GPUS.length), budget: BUDGET, cross_config_evaluation: CROSS_CONFIG,
  };
  const groupOf = new Map(PLAN.groups.flatMap((g) => g.config_ids.map((id) => [id, g])));
  const planPersisted = await persist('file_writer:family_plan', [{ path: P.plan, content: JSON.stringify(PLAN, null, 2) }]);

  // ---- Phase 3: one kernel_lane.js per config, all launched together, one GPU id each ----
  const sigStr = (c) => (c.signature ? `[${c.signature.join(', ')}]` : '');
  const familyTask = (c, g) => {
    const sib = CONFIGS.filter((x) => x.id !== c.id).map((x) => x.id + sigStr(x));
    return [
      '[Operator-family context - advisory only; this config\'s own profile and measurements override it]',
      `family=${FAMILY_ID} operator=${OPERATOR} config=${c.id}` + (c.signature ? ` signature=${sigStr(c)}` : '') +
        (axes ? ` axes=${axes.join(',')}` : '') + ` group=${g.group_id} (${g.config_ids.length} config(s): ${g.config_ids.join(', ')})`,
      sib.length ? `siblings, optimized in separate lanes this lane cannot see: ${sib.slice(0, 12).join('; ')}` +
        (sib.length > 12 ? `; +${sib.length - 12} more` : '') : '',
      g.strategy ? `planner strategy hypothesis: ${g.strategy}` : '',
      g.reasoning ? `planner reasoning: ${g.reasoning}` : '',
      g.config_hints[c.id] ? `config hint: ${g.config_hints[c.id]}` : '',
      'Optimize ONLY this config against its own oracle and baseline. The TechLead may drop this hint after profiling.',
      c.task ? `caller task: ${c.task}` : '',
    ].filter(Boolean).join('\n');
  };
  // Family status of one lane, decided from its structured return, never by an agent.
  const TERMINAL = ['flagged', 'author_failed', 'no_baseline', 'failed'];
  const laneRecord = (c, k, gpu, evalDir, r, thrown) => {
    const R = isObj(r) ? r : {};
    const prim = [R.final_speedup, R.final_weighted, R.final_geomean].map(num).find((v) => v != null) ?? null;
    const vs = low(typeof R.validation_status === 'string' ? R.validation_status : '');
    const [status, error] = thrown != null ? ['lane_failed', thrown]
      : !isObj(r) ? ['invalid_result', `lane returned ${r === null ? 'null' : typeof r}`]
      : !vs ? ['invalid_result', 'lane result has no validation_status']
      : vs !== 'accepted' ? [TERMINAL.includes(vs) ? vs : 'not_accepted', null]
      : prim == null ? ['invalid_result', 'accepted lane result has no finite speedup']
      : prim > 1.0 && !str(R.final_patch) ? ['invalid_result', 'accepted speedup without a final_patch']
      : [prim > 1.0 ? 'optimized' : 'accepted_no_improvement', null];
    return {
      config_id: c.id, group_id: groupOf.get(c.id).group_id, execution_index: k, gpu_id: gpu,
      mode: c.mode, target_language: c.target_language, budget: c.budget, kernel_path: c.kernel_path, eval_dir: evalDir,
      signature: c.signature, contract: c.contract, incumbent: c.incumbent, workload_weight: c.workload_weight,
      status, lane_validation_status: vs || null, speedup_vs_stock: prim, speedup_vs_incumbent: null,
      incumbent_comparison: { status: 'unavailable', reason: c.incumbent
        ? 'no incumbent adapter: the incumbent was not run under this lane\'s oracle and timing basis' : 'no incumbent supplied' },
      final_speedup: num(R.final_speedup), final_weighted: num(R.final_weighted), final_geomean: num(R.final_geomean),
      metric_kind: R.workload_aligned === true ? 'weighted' : (R.workload_aligned === false ? 'geomean' : null),
      timing_basis: str(R.timing_basis), workload_aligned: typeof R.workload_aligned === 'boolean' ? R.workload_aligned : null,
      lane_eval_dir: str(R.eval_dir), report_path: str(R.report_path), final_patch: str(R.final_patch),
      warm_start: isObj(R.warm_start) ? R.warm_start : null, kb_written: isObj(R.kb_written) ? R.kb_written : null,
      learned_card: isObj(R.learned_card) ? R.learned_card : null, reason: str(R.reason), error,
    };
  };

  phase('Bakeoff');
  const famSem = makeSem(GPUS);
  const RECORDS = await Promise.all(PLAN.execution_order.map((id, k) => famSem.with(1, async ([gpu]) => {
    const c = byId.get(id);
    const evalDir = `${FAMILY_DIR}/configs/${c.id}`;
    let r = null;
    let thrown = null;
    try {
      r = await workflow({ scriptPath: WORKER }, {
        ...Object.fromEntries(PARENT_FORWARD.filter((f) => A[f] != null).map((f) => [f, A[f]])),
        ...(LANE_GFX ? { expected_gfx: LANE_GFX, expected_target: LANE_TARGET } : {}),
        ...c.kb,
        kernel_path: c.kernel_path, workflow_dir: WORKFLOW_DIR, eval_dir: evalDir,
        mode: c.mode, target_language: c.target_language, budget: c.budget,
        gpu_ids: gpu, gpu_mode: c.gpu_mode, op_spec: c.op_spec, task: familyTask(c, groupOf.get(id)),
        apply_to_original: 'false',
      });
    } catch (e) {
      thrown = String(e && e.message ? e.message : e).slice(0, 500);
      log(`config ${id} lane failed: ${thrown}`);
    }
    // Absorb the lane's timeline so this run's ledger accounts for every lane's agents.
    if (LLM_STATS && isObj(r) && r.llm_timeline) LLM_TL.nested.push(r.llm_timeline);
    const rec = laneRecord(c, k, gpu, evalDir, r, thrown);
    log(`config ${id} -> ${rec.status}` + (rec.speedup_vs_stock != null ? ` ${rec.speedup_vs_stock.toFixed(2)}x` : '') +
        ` (lane ${rec.lane_validation_status || 'n/a'})`);
    return rec;
  })));
  RECORDS.sort((a, b) => a.execution_index - b.execution_index);

  // ---- Phase 4-6: summary, aggregate and catalog are pure functions of RECORDS; the writer renders the report ----
  const isOk = (r) => r.status === 'optimized' || r.status === 'accepted_no_improvement';
  const by_status = {};
  for (const r of RECORDS) by_status[r.status] = (by_status[r.status] || 0) + 1;
  const SUMMARY = {
    total: RECORDS.length, optimized: by_status.optimized || 0, accepted_no_improvement: by_status.accepted_no_improvement || 0,
    flagged: by_status.flagged || 0, failed: RECORDS.filter((r) => !isOk(r) && r.status !== 'flagged').length, by_status,
  };
  const AGGREGATE = (() => {
    const inc = RECORDS.filter(isOk);
    const bases = [...new Set(inc.map((r) => r.timing_basis))];
    const kinds = [...new Set(inc.map((r) => r.metric_kind))];
    const base = {
      value: null, kind: null, metric: null, timing_basis: null, weighting: 'unweighted',
      rule: 'geometric mean of per-config speedup_vs_stock over accepted configs; requires one metric kind and one known timing_basis',
      included: inc.map((r) => r.config_id),
      excluded: RECORDS.filter((r) => !isOk(r)).map((r) => ({ config_id: r.config_id, status: r.status })),
      complete: inc.length === RECORDS.length,
    };
    const why = !inc.length ? 'no accepted config results'
      : bases.length !== 1 || bases[0] == null || bases[0] === 'unknown'
        ? `timing_basis not comparable across accepted configs: ${bases.map(String).join(', ')}`
      : kinds.length !== 1 || kinds[0] == null ? `metric kind not comparable across accepted configs: ${kinds.map(String).join(', ')}`
      : inc.some((r) => !(r.speedup_vs_stock > 0)) ? 'an accepted config has a non-positive speedup' : null;
    if (why) return { ...base, reason: why };
    return { ...base, value: Math.exp(inc.reduce((s, r) => s + Math.log(r.speedup_vs_stock), 0) / inc.length),
      kind: 'geomean_of_config_speedups', metric: kinds[0], timing_basis: bases[0],
      reason: base.complete ? 'all configs accepted under one timing basis'
        : `partial family: ${base.excluded.length} config(s) without an accepted result are excluded` };
  })();
  const fallbackReason = (r) => ({
    accepted_no_improvement: 'accepted but not faster than its own baseline (speedup <= 1.0)',
    flagged: 'the Director flagged the lane result',
    author_failed: 'author lane produced no correct baseline',
    no_baseline: 'lane could not establish a baseline',
    lane_failed: `lane threw: ${r.error}`,
    invalid_result: `malformed lane result: ${r.error}`,
  })[r.status] || `lane validation_status=${r.lane_validation_status}`;
  const CATALOG = {
    schema_version: 1, family_id: FAMILY_ID, operator: OPERATOR, deployment: dep, signature_axes: axes,
    coverage_policy: 'exact_per_config', deployment_ready: false, cross_config_evaluation: CROSS_CONFIG,
    implementations: RECORDS.filter((r) => r.status === 'optimized').map((r) => ({
      implementation_id: `${FAMILY_ID}:${r.config_id}`, group_id: r.group_id, config_ids: [r.config_id],
      artifact: { kind: 'lane_patch', path: r.final_patch }, backend: r.target_language, mode: r.mode, contract: r.contract,
      coverage: { exact: r.signature ? [r.signature] : [], ranges: [] }, orchestration_key: r.config_id,
      validation: 'accepted', verified_speedup: r.speedup_vs_stock, speedup_vs_stock: r.speedup_vs_stock,
      speedup_vs_incumbent: null, incumbent_comparison: r.incumbent_comparison,
      metric_kind: r.metric_kind, timing_basis: r.timing_basis,
      candidate_class: 'optimization_candidate', deployment_ready: false,
      eval_dir: r.lane_eval_dir || r.eval_dir, report_path: r.report_path,
    })),
    fallback: RECORDS.filter((r) => r.status !== 'optimized').map((r) => ({
      config_id: r.config_id, selected: r.incumbent ? 'incumbent' : 'stock', status: r.status, reason: fallbackReason(r),
    })),
  };
  const FAMILY_RESULTS = {
    schema_version: 1, family_id: FAMILY_ID, operator: OPERATOR, description: str(S.description) || '',
    eval_dir: FAMILY_DIR, summary: SUMMARY, aggregate: AGGREGATE, configs: RECORDS,
  };
  const REPORT_HOW = `Render this Markdown file yourself from ONLY the two JSON files above and ${P.plan}: ` +
    `"# Operator family report: ${FAMILY_ID}", then Summary, Configs, Plan (groups are planner hypotheses; every config ran ` +
    'its own lane), Results (a row per config: status, lane validation, speedup vs stock, metric, timing basis, GPU, patch), ' +
    'Selection (catalog implementation or fallback reason), Family aggregate (value and rule, or why not reported), ' +
    'Knowledge (lane report_path, warm_start, kb_written), Failed configs and next actions. Copy statuses and numbers exactly; ' +
    'speedup_vs_incumbent is n/a (no incumbent or sibling config was executed) and the catalog is not deployment-ready.';
  phase('Report');
  const persisted = await persist('file_writer:family_artifacts', [
    { path: P.results, content: JSON.stringify(FAMILY_RESULTS, null, 2) },
    { path: P.catalog, content: JSON.stringify(CATALOG, null, 2) },
  ], { path: P.report, how: REPORT_HOW });
  log(`config_aware COMPLETE: ${SUMMARY.optimized}/${SUMMARY.total} config(s) optimized. Results in ${FAMILY_DIR}`);
  if (LLM_STATS) await emitRunReport(FAMILY_DIR, LLM_TL, 'Report');
  return {
    schema_version: 1, mode: 'config_aware', config_aware: true, family_id: FAMILY_ID, operator: OPERATOR, eval_dir: FAMILY_DIR,
    llm_timeline: LLM_STATS ? { ...LLM_TL, instance: FAMILY_DIR } : undefined,
    plan_path: P.plan, results_path: P.results, report_path: P.report, catalog_path: P.catalog,
    plan: PLAN, configs: RECORDS, summary: SUMMARY, aggregate: AGGREGATE, catalog: CATALOG,
    artifacts_persisted: planPersisted && persisted,
    validation_status: SUMMARY.optimized > 0 ? 'ok' : 'no_optimized_config',
  };
}

// ===========================================================================
// SINGLE-LANGUAGE PASS-THROUGH (mode=optimize | author) — byte-compatible with
// the pre-dispatcher behavior. Forward EVERY arg to the worker unchanged (the
// worker ignores dispatcher-only args like `backends`). One nesting level.
// ===========================================================================
if (MODE === 'optimize' || MODE === 'author') {
  phase('Bakeoff');            // reuse a declared phase slot for the single passthrough lane
  log(`mode=${MODE}: single-language pass-through -> ${WORKER}`);
  const r = await workflow({ scriptPath: WORKER }, { ...A, workflow_dir: WORKFLOW_DIR });
  // The dispatcher makes no agent calls of its own here, so the report is rendered from the lane's
  // OWN timeline (`llm_timeline`) into the lane's `eval_dir`.
  if (LLM_STATS && r && r.eval_dir && r.llm_timeline) await emitRunReport(r.eval_dir, r.llm_timeline, 'Bakeoff');
  return r;
}

// ===========================================================================
// From here on: mode=bakeoff — the multi-language bake-off orchestrator. Pass
// args.backends to pick the languages, or leave it empty to auto-discover.
// ===========================================================================
if (MODE !== 'bakeoff') {
  throw new Error(`unknown mode='${MODE}'. Use optimize | author | bakeoff.`);
}
// Bake-off lanes each take a slice of gpu_ids, so multi-GPU jobs are only supported by the
// single-language modes for now.
if ((A.gpus_per_job != null && String(A.gpus_per_job).trim() !== '1') || String(A.job_gpu_ids || '').trim()) {
  throw new Error('mode=bakeoff does not support gpus_per_job > 1 or job_gpu_ids; use mode=optimize or mode=author.');
}

// Sibling e2e_workflow dir — we REFERENCE its op_benchmarker role + bench scripts in place (no copy).
const E2E_WF_DIR = String(A.e2e_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/e2e_workflow')).replace(/\/+$/, '');
const EXPECTED_GFX = String(A.expected_gfx || '').trim().toLowerCase();
const EXPECTED_TARGET = String(A.expected_target || '').trim().toLowerCase();
const EXPECTED_DEVICE_NAME = String(A.expected_device_name || '').trim();
const EXPECTED_PHYSICAL_CU_COUNT = Number(A.expected_physical_cu_count || 0);
if (!!EXPECTED_GFX !== !!EXPECTED_TARGET) {
  throw new Error('expected_gfx and expected_target must be supplied together');
}
if ((EXPECTED_GFX && !/^gfx[0-9a-f]+$/.test(EXPECTED_GFX)) ||
    (EXPECTED_TARGET && !['r9700', 'r9600d', 'w7900', 'unknown'].includes(EXPECTED_TARGET))) {
  throw new Error('invalid bakeoff identity: expected_gfx must be a gfx token and expected_target r9700|r9600d|w7900|unknown');
}
if (['r9700', 'r9600d'].includes(EXPECTED_TARGET) && EXPECTED_GFX !== 'gfx1201') {
  throw new Error(`expected_target=${EXPECTED_TARGET} requires expected_gfx=gfx1201`);
}
if (EXPECTED_GFX === 'gfx1200' || EXPECTED_TARGET === 'gfx1200') {
  throw new Error('gfx1200 is not supported: this workflow is hardware-validated only on R9700 / gfx1201');
}
if (EXPECTED_TARGET === 'w7900' && EXPECTED_GFX !== 'gfx1100') {
  throw new Error('expected_target=w7900 requires expected_gfx=gfx1100');
}
if (EXPECTED_GFX === 'gfx1100' && EXPECTED_TARGET !== 'w7900') {
  throw new Error(
    `expected_gfx=gfx1100 requires expected_target=w7900 (got ${EXPECTED_TARGET}): ` +
    'no other gfx1100 product is validated');
}
let GPU_FAMILY = gpuFamily(EXPECTED_GFX);
let KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge')).replace(/\/+$/, '');
if (GPU_FAMILY.isolate) KERNEL_KNOWLEDGE_DIR = '';
const KERNEL_PATH_ORIG = A.kernel_path;
const KERNEL_NAME_HINT = String(KERNEL_PATH_ORIG).replace(/\/+$/, '').split('/').pop();
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
const TASK = A.task || '';
const OP_SPEC = A.op_spec || {};
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const ENABLE_FP8 = String(A.enable_fp8 != null ? A.enable_fp8 : 'false');
const GPU_LIST = String(A.gpu_ids != null ? A.gpu_ids : '0').split(',').map(s => s.trim()).filter(Boolean);
// Forwarded verbatim to each lane worker; the dispatcher itself does not schedule. Without the
// passthrough gpu_mode would be settable only on a direct kernel_lane.js call, so the pinned "before"
// arm would be unreachable through the normal entry point and the A/B could not be run at all.
const GPU_MODE = String(A.gpu_mode || 'pool') === 'pin' ? 'pin' : 'pool';
// Explicit backend list (empty => auto-discover from the freeze + op_benchmarker probe).
const BACKENDS = (Array.isArray(A.backends) ? A.backends
  : (typeof A.backends === 'string' ? A.backends.split(',') : []))
  .map(s => String(s == null ? '' : s).trim().toLowerCase()).filter(Boolean);
// Expert-skills passthrough (advisory; OFF by default -> nothing injected).
let USE_EXPERT_SKILLS = !GPU_FAMILY.isolate && String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');

// Warm-start experience KB. Passed to each lane explicitly (the bakeoff lane invocation spreads
// specific keys, not ...A) so every language lane reads/writes its own <kernel>__<lang>__<gfx> slug.
let WARM_START = GPU_FAMILY.isolate ? 'off'
  : (String(A.warm_start != null ? A.warm_start : 'on').trim().toLowerCase() || 'on');
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kb_artifacts')).replace(/\/+$/, '');
// Plane selection, forwarded the same way and for the same reason: every bakeoff lane must read and
// write the plane the run was launched with, not each its own default.
const KB_PLANE_ARGS = {
  ...(A.kb_mode != null ? { kb_mode: String(A.kb_mode) } : {}),
  ...(A.kb_store_dir != null ? { kb_store_dir: String(A.kb_store_dir) } : {}),
  ...(A.kb_framework_version != null ? { kb_framework_version: String(A.kb_framework_version) } : {}),
};

// oracle_freezer output — the standalone counterpart of e2e's kernel_extractor.extract_op, built from a
// kernel dir instead of a live server. Same op-task-dir contract; also creates the run dir (EVAL_DIR).
const FREEZE_SCHEMA = obj({
  eval_dir: { type: 'string' },            // the isolated run dir this freeze created under EXP_ROOT
  op_kind: { type: 'string' },             // gemm|attn|elementwise|moe|other
  task_dir: { type: 'string' },            // the immutable op task dir (unittest.py + meta.json + baseline_src/)
  live_backend: { type: 'string' },        // the input kernel's language, e.g. "hip"
  candidate_backends: arrStr,
  baseline_frozen: { type: 'boolean' },
  baseline_callable: { type: 'string' },
  device_gfx: { type: 'string' },
  device_target: { type: 'string' },
  device_name: { type: 'string' },
  physical_cu_count: { type: 'number' },
  // Always "" from oracle_freezer — a freezer-built task dir records NO golden tensors (correctness is
  // live parity vs baseline_src/). Kept in the schema because an e2e kernel_extractor task dir, which
  // captures unsynthesizable real routing / paged-KV metadata, does ship a reference_io.pt and fills it.
  reference_io_sha256: { type: 'string' },
  op_spec: { type: 'object', additionalProperties: true },
  workload_path: { type: 'string' },
  smoke: { type: 'string' },               // "pass" required to proceed
  notes: { type: 'string' },
}, ['op_kind', 'task_dir', 'smoke']);

// op_benchmarker output — the role file is reused VERBATIM from e2e (referenced in place, never edited);
// the standalone-tune behavior is steered entirely by the discover prompt below. `baseline_ms` +
// `tuned_speedup` are extra fields this dispatcher asks the agent to fill (in the prompt) so a tuned env
// backend can be ranked against the SAME frozen input-kernel baseline as the author/optimize lanes.
const OPBENCH_SCHEMA = obj({
  short_name: { type: 'string' }, op_kind: { type: 'string' }, provenance_ok: { type: 'boolean' },
  winner_backend: { type: 'string' }, winner_kind: { type: 'string' },
  // null when nothing was timed -- a speedup is a measurement, and 0.0 read as
  // 'benched, not faster'. `measured` is the discriminator; gate on it, not on the number.
  isolated_speedup: { type: ['number', 'null'] }, measured: { type: 'boolean' },
  winner_editable: { type: 'boolean' },
  best_known_ms: { type: 'number' },
  baseline_ms: { type: 'number' },     // ms of the FROZEN input kernel (baseline_src/) on the same oracle
  tuned_speedup: { type: 'number' },   // best tuned env backend's speedup vs the FROZEN baseline (baseline_ms/tuned_ms)
  recommend_tier_c: { type: 'boolean' }, author_plan: arrObj, tuning_artifact: { type: 'string' },
  apply_env: { type: 'string' }, apply_flags: { type: 'string' }, code_patch: { type: 'string' },
  per_backend: arrObj, parity_note: { type: 'string' },
  gate: { type: 'string' }, harness_suspect: { type: 'boolean' }, reason: { type: 'string' },
}, ['gate', 'isolated_speedup']);

const REPORT_SCHEMA = obj({
  report_path: { type: 'string' }, applied_to_original: { type: 'string' }, note: { type: 'string' },
}, ['report_path']);

const UPDATE_EXPERIENCE_SCHEMA = obj({
  action: { type: 'string' },      // created | merged | skipped
  card_path: { type: 'string' },   // path under knowledge/learned/, or "" if nothing distilled
  key: { type: 'string' }, note: { type: 'string' },
}, []);

// Expert-skills block appended to the op_benchmarker prompt (advisory; opt-in).
const expertSkillsBlock = () => (!USE_EXPERT_SKILLS || !EXPERT_SKILLS_DIR) ? '' :
  `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also query ${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op, whose \`scope\` ` +
    `is \`kernel\` (\`tuning\` entries belong to the e2e tuning phase and match every operator), and whose ` +
    `validation_status is \`validated\`, and treat each as a HIGH-PRIOR candidate — advisory only, never ` +
    `overriding your isolated A/B vs the oracle, never reducing a result below the measured baseline.`;

// PRIMARY metric of a worker (kernel_lane.js) result: its final_speedup is already the weighted number
// when the run was workload-aligned, else the geomean. Fall back defensively.
const primSpeedup = (r) => {
  if (!r) return 0;
  const g = r.final_speedup != null ? r.final_speedup
          : (r.final_weighted != null ? r.final_weighted
             : (r.final_geomean != null ? r.final_geomean : 0));
  return Number.isFinite(g) ? g : 0;
};

// Gate fields like `smoke` are free strings in the schemas, so an agent legitimately answers
// "PASS - 15/15 parity draws, geomean 0.994" and a `=== 'pass'` test throws the whole run away.
// Match the leading word instead: "PASS - ..." / "passed" gate open, "FAIL"/"did not pass" stay shut.
const says = (v, w) => String(v == null ? '' : v).trim().toLowerCase().startsWith(w);

// ===========================================================================
// PHASE: Freeze — establish the ONE immutable oracle + frozen baseline (denominator)
// ===========================================================================
phase('Freeze');
const oracle = await agentT(
  roleAgent('oracle_freezer', 'freeze',
    'Freeze the input kernel into an immutable op task dir (no server). Create the run dir too.', {
      KERNEL_PATH: KERNEL_PATH_ORIG, EXP_ROOT, KERNEL_NAME_HINT, GPU_ID: GPU_LIST[0],
      OP_SPEC, WORKLOAD_SPEC_PATH, SKILL_DIR: WORKFLOW_DIR,
      EXPECTED_GFX, EXPECTED_TARGET, EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
      // harness_lib.py (the shared timing/correctness lib) ships with e2e_workflow; gpu_lock.sh ships here.
      HARNESS_LIB: `${E2E_WF_DIR}/scripts/harness_lib.py`,
      GPU_LOCK: `${WORKFLOW_DIR}/scripts/gpu_lock.sh`,
    }),
  { phase: 'Freeze', label: 'oracle_freezer', schema: FREEZE_SCHEMA });
if (!oracle || !says(oracle.smoke, 'pass') || !oracle.task_dir || oracle.baseline_frozen === false) {
  log(`Freeze FAILED (${oracle ? oracle.notes || oracle.smoke : 'no result'}); aborting — no comparable baseline.`);
  return { mode: MODE, validation_status: 'freeze_failed', winner: null,
    reason: oracle ? oracle.notes || 'freeze smoke did not pass' : 'oracle_freezer returned nothing' };
}
const DETECTED_GFX = String(oracle.device_gfx || '').trim().toLowerCase();
const DETECTED_TARGET = String(oracle.device_target || '').trim().toLowerCase();
const DETECTED_DEVICE_NAME = String(oracle.device_name || '').trim();
const DETECTED_PHYSICAL_CU_COUNT = Number(oracle.physical_cu_count);
if (!DETECTED_GFX || !DETECTED_TARGET) {
  throw new Error('Freeze failed: oracle_freezer did not return structured device_gfx and device_target');
}
if (DETECTED_GFX === 'gfx1200') {
  throw new Error('Detected gfx1200, which is not validated by this workflow; refusing R9700 guidance');
}
if (EXPECTED_GFX && DETECTED_GFX !== EXPECTED_GFX) {
  throw new Error(`GPU architecture mismatch: expected ${EXPECTED_GFX}, detected ${DETECTED_GFX}`);
}
if (['r9700', 'r9600d'].includes(EXPECTED_TARGET) && DETECTED_TARGET !== EXPECTED_TARGET) {
  throw new Error(`GPU product mismatch: expected ${EXPECTED_TARGET}, detected ${DETECTED_TARGET}`);
}
if (EXPECTED_TARGET === 'w7900' && DETECTED_TARGET !== 'w7900') {
  throw new Error(`GPU product mismatch: expected w7900, detected ${DETECTED_TARGET}`);
}
if (DETECTED_GFX === 'gfx1100' && DETECTED_TARGET !== 'w7900') {
  throw new Error(
    `Detected gfx1100 with device_target=${DETECTED_TARGET}: only the exact Radeon Pro W7900 is ` +
    'validated on gfx1100; refusing RDNA3 guidance');
}
if (DETECTED_TARGET === 'w7900' && DETECTED_GFX !== 'gfx1100') {
  throw new Error(`Freeze returned device_target=w7900 on ${DETECTED_GFX}: w7900 requires gfx1100`);
}
if (!Number.isFinite(DETECTED_PHYSICAL_CU_COUNT) || DETECTED_PHYSICAL_CU_COUNT <= 0) {
  throw new Error('Freeze failed: oracle_freezer did not return a positive physical_cu_count');
}
if (EXPECTED_DEVICE_NAME && DETECTED_DEVICE_NAME !== EXPECTED_DEVICE_NAME) {
  throw new Error(
    `GPU product name mismatch: expected ${EXPECTED_DEVICE_NAME}, detected ${DETECTED_DEVICE_NAME || 'unknown'}`);
}
if (EXPECTED_PHYSICAL_CU_COUNT > 0 &&
    DETECTED_PHYSICAL_CU_COUNT !== EXPECTED_PHYSICAL_CU_COUNT) {
  throw new Error(
    `GPU physical CU mismatch: expected ${EXPECTED_PHYSICAL_CU_COUNT}, ` +
    `detected ${DETECTED_PHYSICAL_CU_COUNT}`);
}
const LANE_GFX = EXPECTED_GFX || DETECTED_GFX;
const LANE_TARGET = EXPECTED_TARGET || DETECTED_TARGET;
const LANE_DEVICE_NAME = EXPECTED_DEVICE_NAME || DETECTED_DEVICE_NAME;
const LANE_PHYSICAL_CU_COUNT =
  EXPECTED_PHYSICAL_CU_COUNT || DETECTED_PHYSICAL_CU_COUNT;
if (gpuFamily(DETECTED_GFX).isolate) {
  GPU_FAMILY = gpuFamily(DETECTED_GFX);
  KERNEL_KNOWLEDGE_DIR = '';
  USE_EXPERT_SKILLS = false;
  WARM_START = 'off';
}
const EVAL_DIR = oracle.eval_dir || `${EXP_ROOT}/bakeoff_${KERNEL_NAME_HINT}`;
log(`Freeze done. op_kind=${oracle.op_kind}, task_dir=${oracle.task_dir}, live_backend=${oracle.live_backend || '?'}`);

// ===========================================================================
// PHASE: Discover — per-language existing-impl probe + measure + author_plan +
// OFFLINE per-backend tune. e2e's op_benchmarker role is reused IN PLACE and
// UNCHANGED (SKILL_DIR=E2E_WF_DIR so its knowledge/scripts resolve). Its Tier-B
// step is written for a live server; since there is none here, the prompt below
// REDIRECTS the tune to run OFFLINE on the frozen oracle's shapes (which carry the
// real M/N/K/bias/dtype) and to verify engagement in the ISOLATED unittest — so a
// standalone bake-off STILL gets the aiter/CK GEMM tune, without touching the role
// file. Only pure server-flag levers (--attention-backend swap, serving-only
// engagement probes) are skipped.
// ===========================================================================
phase('Discover');
const DISCOVER_INTRO = GPU_FAMILY.isolate
  ? (
  'STANDALONE kernel bake-off — there is NO live server (do not try to launch or capture from one).\n' +
  `${GPU_FAMILY.name} / ${LANE_GFX} ISOLATION: do NOT consult learned KB, expert skills, AITER CDNA win cards, or CK/AITER offline tune.\n` +
  'Do NOT inject MFMA / FNUZ / Instinct guidance.' +
  (LANE_GFX === 'gfx1100' ? ' On gfx1100, plan lanes on wave32 WMMA with FP16, BF16 or INT8 inputs (INT4 is not validated).' : '') +
  ' Discover existing impls by measuring the frozen oracle only.\n' +
  '(1) Tier-A DISCOVER: bench candidate backends on the immutable oracle in OP_TASK_DIR (HIP/Triton/hipBLASLt).\n' +
  '(2) SKIP Tier-B AITER/CK env tune — those inputs are CDNA-calibrated.\n' +
  '(3) Fill `baseline_ms` = the FROZEN input kernel`s ms on the oracle.\n' +
  '(4) DECIDE the author_plan as usual without CDNA SOTA cards.\n' +
  '(5) DO NOT curate e2e_workflow/knowledge/learned/. WRITE NOTHING under e2e_workflow/.'
  )
  : (
  'STANDALONE kernel bake-off — there is NO live server (do not try to launch or capture from one).\n' +
  '(1) Tier-A DISCOVER: bench every candidate backend on the immutable oracle in OP_TASK_DIR.\n' +
  '(2) Tier-B TUNE is STILL IN SCOPE — run it OFFLINE, not from a server capture. The step`s "capture ' +
  'shapes from a warm server" instruction is replaced here: take the tune shapes DIRECTLY from ' +
  'OP_TASK_DIR/meta.json (the frozen oracle carries the real M/N/K, the `bias` flag, and dtype — so you ' +
  'do NOT need a server and do NOT have to guess bias). Write them into the untuned CSV, run the gradlib ' +
  '(GEMM) / CK tuner as usual, deploy the tuned CSV (AITER_CONFIG_GEMM_BF16 / the CK env), then VERIFY ' +
  'engagement AND measure speed in the ISOLATED unittest/op_bench with AITER_LOG_TUNED_CONFIG=1 (look for ' +
  '"is tuned on cu_num" in THIS process`s output — NOT server.log). Return the best tuned backend as a ' +
  'winner_kind=env candidate with apply_env + tuning_artifact.\n' +
  '(3) Fill `baseline_ms` = the FROZEN input kernel`s ms on the oracle (baseline_src/ / meta.baseline_callable), ' +
  'and `tuned_speedup` = baseline_ms / (best tuned backend ms) — i.e. the tuned env win measured against ' +
  'the SAME frozen baseline the author lanes use, so it is directly comparable. If no tuned backend beats ' +
  'the baseline, set tuned_speedup=0.\n' +
  '(4) SKIP only pure server-flag levers (--attention-backend swap; serving-only fp8/MoE playbook probes ' +
  'that need a running server). (5) DECIDE the author_plan as usual.\n' +
  '(6) DO NOT run your step 6 (`CURATE SKILL_DIR/knowledge/learned/`). Your SKILL_DIR points at ' +
  'e2e_workflow, but this is a kernel_workflow run: e2e`s learned/ is e2e`s own memory (owned by its ' +
  'system_architect after an e2e A/B), and a kernel bake-off has no e2e-transfer evidence to put in it. ' +
  'This run`s learned sink is kernel_workflow/knowledge/learned/, curated by the TechLead`s ' +
  'update_experience step after Report. WRITE NOTHING under e2e_workflow/ — read it freely, but the ' +
  'only files you create or modify live under EVAL_DIR (plus the tuning artifacts you were asked for).'
  );
const bake = await agentT(
  roleAgentFrom(E2E_WF_DIR, 'op_benchmarker', 'bakeoff', DISCOVER_INTRO, {
    EVAL_DIR, OP_TASK_DIR: oracle.task_dir, OP_KIND: oracle.op_kind,
    CANDIDATE_BACKENDS: (BACKENDS.length ? BACKENDS : (oracle.candidate_backends || [])),
    GPU_ID: GPU_LIST[0], ENABLE_FP8, LIVE_SERVER: 'false',
    KERNEL_WF_DIR: WORKFLOW_DIR, KERNEL_BUDGET: BUDGET, SKILL_DIR: E2E_WF_DIR,
  }, expertSkillsBlock()),
  { phase: 'Discover', label: 'op_benchmarker:discover', schema: OPBENCH_SCHEMA }) || {};

// Resolve lanes. The INCUMBENT (original input) language ALWAYS competes as an in-place `optimize` lane:
// it is the floor every rewrite must beat, and the entire premise of a bake-off is "keep the fastest,
// INCLUDING simply optimizing what we already have". args.backends only ADDS rewrite candidates on top —
// it can never drop the incumbent (otherwise a rewrite could "win" while being slower than the un-tried
// in-place optimization of the original). Other languages author (or rewrite->optimize if an editable impl
// already exists on the box).
// `author_plan` is a free-form array of objects in the schema, so the agent may name the language under
// any of a few obvious keys. Naive `String(a.language)` turned a missing key into the *string* "undefined",
// which is truthy — it survived .filter(Boolean) and became a real lane literally named `undefined` (lane
// dir `bakeoff/undefined/`, winner row `lang: "undefined"`). Read the aliases; an entry that still has no
// name is an entry about THIS op, so attribute it to the incumbent language rather than inventing one.
const liveLang = String(oracle.live_backend || '').toLowerCase();
const rawLang = (a) => String((a && (a.language || a.lang || a.backend || a.target_language)) || '').trim();
const langOf = (a) => rawLang(a).toLowerCase() || liveLang;
const unnamed = (bake.author_plan || []).filter(a => !rawLang(a)).length;
if (unnamed) log(`WARNING: ${unnamed} author_plan entr(y|ies) carried no language field; ` +
                 `attributed to the incumbent language '${liveLang || '(unknown)'}'. Check Discover output.`);
const modeOf = (a) => (a && a.route === 'rewrite') ? 'optimize' : 'author';
// route per language, for the explicit-`backends` path (an existing editable impl => optimize, not author).
const planByLang = Object.fromEntries((bake.author_plan || []).map(a => [langOf(a), modeOf(a)]));
if (!liveLang) {
  // The freezer could not identify the input language, so the incumbent optimize lane cannot be guaranteed.
  // Do NOT silently proceed with rewrites only — that would let a slower rewrite "win" over a baseline we
  // never tried to optimize in-place. Surface it loudly so it is caught, then continue with what we have.
  log('WARNING: oracle.live_backend is empty — cannot guarantee the incumbent-language optimize lane; ' +
      'rewrites will still run but the original language is under-represented. Check oracle_freezer.');
}
// Incumbent FIRST (force-included regardless of args.backends / discovery), then the requested/discovered
// rewrites. Deduped on lang+mode, NOT on lang alone: "optimize the existing Triton in place" and "author a
// fresh Triton from scratch" are two genuinely different candidates and both deserve a lane.
const wanted = [];
const seen = new Set();
const want = (lang, mode) => {
  if (!lang) return;
  const k = `${lang}:${mode}`;
  if (seen.has(k)) return;
  seen.add(k);
  wanted.push({ lang, mode });
};
want(liveLang, 'optimize');
if (BACKENDS.length) BACKENDS.forEach(l => want(l, l === liveLang ? 'optimize' : (planByLang[l] || 'author')));
else (bake.author_plan || []).forEach(a => want(langOf(a), modeOf(a)));
// Lane dir/log key: plain language when that language has a single lane, `<lang>_<mode>` when it has two.
const laneCount = {};
wanted.forEach(w => { laneCount[w.lang] = (laneCount[w.lang] || 0) + 1; });
const lanes = wanted.map(w => ({
  lang: w.lang, mode: w.mode,
  key: laneCount[w.lang] > 1 ? `${w.lang}_${w.mode}` : w.lang,
}));
if (!lanes.length) {
  log('Discover produced no viable lanes; aborting.');
  return { mode: MODE, task_dir: oracle.task_dir, validation_status: 'no_lanes', winner: null };
}
// a single discovered lane just runs that one lane below (equivalent to a plain single-language optimize).
log(`baseline ${bake.best_known_ms != null ? bake.best_known_ms + ' ms' : '(unknown)'}; ` +
    `${lanes.length} lane(s) = ${lanes.map(l => `${l.lang}:${l.mode}`).join(', ')}; ` +
    `GPUs=${GPU_LIST.length} => concurrency ${Math.min(lanes.length, GPU_LIST.length)}; ` +
    `est cost ~${lanes.length}x a single ${BUDGET}-budget run.`);

// ===========================================================================
// PHASE: Bakeoff — one UNCHANGED kernel_lane worker per language, parallel over the
// GPU pool. dispatcher(level0) -> worker(level1): exactly one nesting level per lane.
// Every lane runs against oracle.task_dir's immutable unittest => the SAME frozen
// baseline denominator => directly comparable speedups (anti-cheating invariant).
// ===========================================================================
phase('Bakeoff');
const sem = makeSem(GPU_LIST);
const results = await Promise.all(lanes.map(l => sem.with(1, async ([gpu]) => {
  try {
    const r = await workflow({ scriptPath: WORKER }, {
      kernel_path: oracle.task_dir, workflow_dir: WORKFLOW_DIR,
      // This lane runs on an oracle_freezer-frozen task dir, so a GEAK_TIMING_RECEIPT is EXPECTED and
      // its absence is a real fault. The optimize/author path spreads {...A} and never sets this, so a
      // pass-through lane (and e2e, which calls this worker directly) correctly defaults to false —
      // there is no Freeze on those routes, so director must not demand a receipt they cannot produce.
      frozen_oracle: 'true',
      expected_gfx: LANE_GFX, expected_target: LANE_TARGET,
      expected_device_name: LANE_DEVICE_NAME,
      expected_physical_cu_count: LANE_PHYSICAL_CU_COUNT,
      ...(A.gpu_perf_level ? { gpu_perf_level: String(A.gpu_perf_level) } : {}),
      mode: l.mode, target_language: l.lang,
      op_spec: oracle.op_spec || OP_SPEC, workload_spec_path: oracle.workload_path || WORKLOAD_SPEC_PATH || '',
      budget: BUDGET, gpu_ids: gpu, gpu_mode: GPU_MODE, task: TASK, apply_to_original: 'false',
      exp_root: `${EVAL_DIR}/bakeoff/${l.key}`,
      use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
      perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
      // Forward the KB switch. This arg object is explicit (the optimize/author path spreads {...A},
      // this one does not), so anything omitted here silently reverts to the lane's default — a
      // caller asking for a KB-off bakeoff would have got eight KB-on lanes and no error.
      use_learned_kb: A.use_learned_kb != null ? String(A.use_learned_kb) : 'true',
      // This explicit arg object drops nothing load-bearing that the parent set: forward llm_stats
      // when supplied so a parent opt-out reaches each lane (unset stays absent -> lane default on).
      ...(A.llm_stats != null ? { llm_stats: String(A.llm_stats) } : {}),
      // Forward the Expt-3 routing switch the same way, so a routed bake-off reaches every lane's
      // kernel_lane (unset stays absent -> routing OFF -> byte-identical run).
      ...(A.routing != null ? { routing: String(A.routing) } : {}),
      // ...and the cost-ladder knobs (thresholds, caps, floors, decider), each only when set.
      ...Object.fromEntries(['route_conf_escalate', 'route_max_retries_per_lane', 'route_max_top_escalations',
        'route_max_output_tokens', 'route_floors', 'route_decider', 'route_eikos_tau', 'route_eikos_max_consults',
        'route_eikos_carrier_model', 'route_eikos_carrier_timeout_ms', 'route_eikos_probe_failed']
        .filter((k) => A[k] != null).map((k) => [k, typeof A[k] === 'object' ? JSON.stringify(A[k]) : String(A[k])])),
      // Curation is central in bake-off mode (see the UpdateExperience step below). In optimize/author
      // mode this dispatcher is a passthrough, so the lane keeps its default `on` and curates itself.
      update_experience: 'off',
      warm_start: WARM_START, kb_artifacts_dir: KB_ARTIFACTS_DIR, ...KB_PLANE_ARGS,
    });
    // Absorb this lane's own agent timeline so the dispatcher's returned timeline accounts for every
    // lane's attempts (kernel lanes never persist their own trace file, so this is the only path).
    if (LLM_STATS && r && r.llm_timeline) LLM_TL.nested.push(r.llm_timeline);
    const speedup = primSpeedup(r);
    log(`lane ${l.key}:${l.mode} -> ${speedup ? speedup.toFixed(2) + 'x' : 'no result'} (${r ? r.validation_status : 'null'})`);
    return { lane: l, r, speedup };
  } catch (e) {
    log(`lane ${l.key} failed: ${e && e.message ? e.message : e}`);
    return { lane: l, r: null, speedup: 0 };
  }
})));

// ===========================================================================
// PHASE: Report — rank ALL candidates on the SAME frozen baseline; write table + winner.
// Candidates are THREE classes, all scored against the one frozen input-kernel baseline:
//   (a) each kernel_lane lane (input-language optimize + per-language author) — a source patch;
//   (b) the tuned env backend from Discover (aiter/CK offline tune) — an env/config winner (no patch).
// ===========================================================================
phase('Report');
// (a) lane candidates
const cands = results.map(x => ({
  lang: x.lane.lang, mode: x.lane.mode, kind: 'lane', speedup: x.speedup,
  validation_status: x.r ? x.r.validation_status : 'failed',
  eval_dir: x.r ? x.r.eval_dir : '', patch: x.r ? x.r.final_patch : '', apply_env: '', tuning_artifact: '',
}));
// (b) tuned env backend candidate (only if it beat the frozen baseline)
const tunedSpeedup = Number(bake.tuned_speedup);
if (Number.isFinite(tunedSpeedup) && tunedSpeedup > 1.0 && bake.winner_backend && bake.winner_backend !== 'none') {
  cands.push({
    lang: bake.winner_backend, mode: 'env-tune', kind: 'env', speedup: tunedSpeedup,
    validation_status: 'env', eval_dir: EVAL_DIR, patch: '',
    apply_env: bake.apply_env || '', tuning_artifact: bake.tuning_artifact || '',
  });
  log(`tuned env backend ${bake.winner_backend} -> ${tunedSpeedup.toFixed(2)}x (vs frozen baseline)`);
}
// A candidate only WINS if it actually BEAT the frozen baseline (speedup > 1.0) — the SAME guard the
// tuned env candidate uses above. Slower-than-baseline lanes remain in the table (laneRows) for full
// transparency but must never win "by default"; winner=null => validation_status 'no_winner' => keep
// the original kernel. Without this, a lane that is SLOWER than baseline (e.g. the only non-failed lane
// at 0.17x) would be mislabeled the winner and mislead downstream automation reading .winner.
// ...and only if the director ACCEPTED it. Ranking on speed alone let a lane whose validation came
// back `flagged` (patch did not install, correctness failed, contended box) win the bake-off, be
// applied to the original kernel, and be curated into the KB — a number with a known reason not to
// be believed, promoted by every downstream step. Reported in review of #411. `laneRows` still
// carries every lane for transparency; only eligibility to WIN is tightened.
const ACCEPTED = (c) => String(c.validation_status || '').toLowerCase() === 'accepted';
const ranked = cands.filter(c => c.speedup > 1.0 && (c.kind !== 'lane' || ACCEPTED(c)))
  .sort((a, b) => b.speedup - a.speedup);
const rejectedByGate = cands.filter(c => c.speedup > 1.0 && c.kind === 'lane' && !ACCEPTED(c));
if (rejectedByGate.length) {
  log(`bake-off: ${rejectedByGate.length} lane(s) beat the baseline but are NOT eligible to win ` +
      `(validation_status != accepted): ${rejectedByGate.map(c => `${c.lang}=${c.validation_status}`).join(', ')}`);
}
const winner = ranked[0] || null;
const laneRows = cands;
const bestSpeedup = cands.reduce((m, c) => Math.max(m, Number(c.speedup) || 0), 0);

// Apply instruction depends on the winner class: a lane winner is a source patch (git apply); an
// env-tune winner is a config/env deploy (no source change — record apply_env + tuning_artifact).
const applyStep = (APPLY_TO_ORIGINAL !== 'true' || !winner)
  ? `Do NOT modify the original kernel (report only). Set applied_to_original to "".`
  : (winner.kind === 'env')
    ? `The winner is an ENV/CONFIG tune (backend ${winner.lang}), not a source patch. Do NOT edit kernel ` +
      `source. Record in the report the deploy: apply_env='${winner.apply_env}' and tuning_artifact=` +
      `'${winner.tuning_artifact}'. Set applied_to_original to "${winner.lang} (env: ${winner.apply_env})".`
    : (winner.patch
        ? `APPLY the winning lane patch: cd ${KERNEL_PATH_ORIG} && (git apply ${winner.patch} || git apply --3way ${winner.patch}); ` +
          `if it does not apply cleanly, apply it manually to match intent. Set applied_to_original to the winning language. Do NOT delete anything.`
        : `The winner has no patch file; report only and set applied_to_original to "".`);

const rep = await agentT(
  `You are the bake-off reporter. Write a concise markdown comparison report and, if requested, apply the winner.

## Task
1. Write ${EVAL_DIR}/bakeoff_report.md containing:
   - The op: ${oracle.op_kind} from ${KERNEL_PATH_ORIG} (task_dir ${oracle.task_dir}).
   - The FROZEN baseline (the ONE denominator every candidate shares): the input kernel itself
     (baseline_ms = ${bake.baseline_ms != null ? bake.baseline_ms : 'unknown'} ms; op_bench best_known_ms = ${bake.best_known_ms != null ? bake.best_known_ms : 'unknown'} ms).
   - A candidate table sorted fastest-first with columns: language | mode | kind (lane|env) | speedup vs frozen baseline | validation_status | eval_dir | patch/env.
   - The WINNER (fastest verified) and a one-paragraph rationale. All speedups are directly comparable
     because every candidate — the input-language optimize lane, each authored language, AND the tuned
     env backend — was scored against the SAME frozen input kernel.
     NOTE: WINNER may be null — this means NO candidate beat the frozen baseline (every speedup <= 1.0x,
     e.g. all lanes failed or were slower than the original). In that case state clearly that no
     candidate beat the baseline and the ORIGINAL kernel is RETAINED (do not present a slower-than-1.0x
     candidate as a "winner by default"); still list every candidate in the table for transparency.
2. ${applyStep}

## Inputs
${cfg({
    OP_KIND: oracle.op_kind, KERNEL_PATH: KERNEL_PATH_ORIG, TASK_DIR: oracle.task_dir,
    BASELINE_MS: bake.baseline_ms != null ? bake.baseline_ms : '',
    BEST_KNOWN_MS: bake.best_known_ms != null ? bake.best_known_ms : '',
    CANDIDATES: laneRows,
    WINNER: winner,
    APPLY_TO_ORIGINAL,
  })}

Return ONLY {report_path, applied_to_original, note} as StructuredOutput.`,
  { phase: 'Report', label: 'bakeoff:report', schema: REPORT_SCHEMA });

log(winner
  ? `Bake-off COMPLETE. winner=${winner.lang}:${winner.mode} ${winner.speedup.toFixed(2)}x. Results in ${EVAL_DIR}`
  : `Bake-off COMPLETE. NO candidate beat the frozen baseline (best ${bestSpeedup.toFixed(2)}x <= 1.0x across ${cands.length} candidate(s)) — keeping the ORIGINAL kernel. Results in ${EVAL_DIR}`);

// ---------------------------------------------------------------------------
// PHASE: UpdateExperience — bake-off curates CENTRALLY here (its lanes run with
// update_experience=off): the reusable lesson is the cross-language routing outcome, which
// no single lane can see. Sink is THIS workflow's knowledge/learned/ (see its README.md),
// never e2e's. Only on a measured win, and ADD-only — a failed step is byte-neutral.
if (winner && winner.speedup > 1.0) {
  const LEARNED_DIR = `${WORKFLOW_DIR}/knowledge/learned`;
  try {
    const ue = await agentT(
      roleAgent('update_experience', 'Report',
        'Curate one distilled learned card from this bake-off win (ADD-only, measured evidence, ' +
        'ratios not wall-clock; record the pitfalls hit).', {
          SCOPE: 'bakeoff', LEARNED_DIR, SKILL_DIR: WORKFLOW_DIR, EVAL_DIR,
          PERF_KNOWLEDGE_DIR: KERNEL_KNOWLEDGE_DIR,
          CURATION_ISOLATE: GPU_FAMILY.isolate ? 'true' : 'false',
          REQUIRED_PLATFORMS: GPU_FAMILY.isolate ? [LANE_GFX] : [],
          WINNER: winner, CANDIDATES: laneRows,
          REPORT_PATH: rep ? rep.report_path : `${EVAL_DIR}/bakeoff_report.md`,
          OP_SPEC,
        }),
      { phase: 'Report', label: 'update_experience', schema: UPDATE_EXPERIENCE_SCHEMA });
    if (ue && ue.card_path) log(`[kb] learned card ${ue.action || 'written'}: ${ue.card_path}`);
  } catch (e) {
    log(`[kb] update_experience skipped: ${e && e.message ? e.message : e}`);
  }
}

if (EVAL_DIR && LLM_STATS) await emitRunReport(EVAL_DIR, LLM_TL, 'Report');

return {
  mode: MODE,
  // `instance` = this run's eval dir: a stable identity so the parser dedupes a timeline reached
  // twice without collapsing two distinct same-shape lanes.
  llm_timeline: LLM_STATS ? { ...LLM_TL, instance: EVAL_DIR } : undefined,
  task_dir: oracle.task_dir,
  eval_dir: EVAL_DIR,
  baseline_ms: bake.baseline_ms != null ? bake.baseline_ms : (bake.best_known_ms != null ? bake.best_known_ms : null),
  candidates: laneRows,
  winner: winner && {
    lang: winner.lang, mode: winner.mode, kind: winner.kind, speedup: winner.speedup,
    eval_dir: winner.eval_dir, final_patch: winner.patch,
    apply_env: winner.apply_env, tuning_artifact: winner.tuning_artifact,
    validation_status: winner.validation_status,
  },
  report_path: rep ? rep.report_path : `${EVAL_DIR}/bakeoff_report.md`,
  applied_to_original: rep ? rep.applied_to_original : '',
  validation_status: winner ? 'ok' : 'no_winner',
};
