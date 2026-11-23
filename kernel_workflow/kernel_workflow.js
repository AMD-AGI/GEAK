export const meta = {
  name: 'kernel-workflow',
  description: 'Single ENTRY POINT for kernel optimization on AMD GPUs: Instinct MI-series (CDNA gfx942/gfx950), the validated RDNA4 product Radeon AI PRO R9700 (gfx1201) and the RDNA3 Radeon Pro W7900 (gfx1100). The worker validates on-box identity via structured rocminfo data; generic gfx1201 is not a product identity, and gfx1100 without the exact W7900 product is refused. Dispatches on args.mode: optimize/author -> delegate one unchanged single-language lane to the kernel_lane worker (backward compatible); bakeoff -> freeze the input kernel into ONE immutable oracle + frozen baseline, discover per-language existing impls + offline-tune env backends (aiter/CK), then run one worker lane per backend language (HIP/Triton/FlyDSL/CK/...) in parallel over the GPU pool and pick the fastest verified result across ALL candidates (author/optimize lanes AND the tuned env backend) — every one scored against the SAME frozen original baseline (anti-cheating). Wraps the unchanged kernel_lane worker (one workflow() nesting level; the dispatcher is the bake-off orchestrator).',
  whenToUse: 'Optimize a kernel. Three modes, all via args.mode (there is NO natural-language mode detection — the caller picks): mode=optimize (DEFAULT) speeds up an EXISTING kernel and behaves exactly like the old single-language workflow; mode=author writes a fresh implementation from scratch, then optimizes it — use it when there is no source to edit yet, or to port the op to another language (pass args.target_language); mode=bakeoff tries several backend languages in parallel and keeps the fastest (pass args.backends, or leave empty to auto-discover). Bakeoff Freeze establishes structured product/ISA identity before Discover; optional expected_gfx/expected_target pins are validated against it. Anything else throws. Pass args.kernel_path (required), args.workflow_dir (required), args.mode, args.target_language, args.backends, args.budget, args.gpu_ids, args.gpu_mode (pool|pin, default pool), args.gpus_per_job / args.job_gpu_ids (optional: a task whose test runs on several GPUs is detected at Setup; pass these only to override; optimize/author only; gpu_ids must hold enough GPUs). args.config_aware="true" + args.family_spec (an operator family: one kernel_lane per config, in parallel over gpu_ids, then family_plan/results/report + an exact-signature implementation_catalog; kernel_path is not used on that path).',
  phases: [
    { title: 'Freeze',   detail: 'oracle_freezer: freeze the input kernel -> immutable oracle + baseline_src/ (the ONE denominator) [bakeoff only]' },
    { title: 'Discover', detail: 'op_benchmarker: per-language existing-impl probe + measure + OFFLINE env tune (aiter/CK, shapes from the frozen oracle) -> author_plan, best_known_ms [bakeoff only]' },
    { title: 'Bakeoff',  detail: 'one unchanged kernel_lane worker per language (optimize if an impl exists, else author), parallel over the GPU pool [bakeoff only]' },
    { title: 'Report',   detail: 'rank ALL candidates (lanes + tuned env backend) on the SAME frozen baseline; write the comparison table + winner (+ optional apply_to_original), then curate ONE distilled card into knowledge/learned/ on a measured win [bakeoff only]' },
  ],
};

// ---------------------------------------------------------------------------
// Args. This is a DISPATCHER: on mode=optimize/author it forwards straight to the
// single-language worker (kernel_lane.js) unchanged; on mode=bakeoff it acts
// as the multi-language bake-off orchestrator. The worker script is a sibling in
// this same dir, so it is called via workflow() at exactly ONE nesting level
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

// ---- Live execution tracker (read-only observer; see interface/geak_trace_collector.py) ----
// Started HERE so a direct kernel run is tracked from its first agent, not only
// at the end. Both dispatcher lanes (pass-through and bakeoff) pass through this
// point, so one call covers them. The observer resolves its own workflow run from
// the runtime's workflow record (no eval-dir substring matching) — needed because
// neither EXP_ROOT nor eval_dir exists yet on the pass-through branch.
//
// Detached, read-only and strictly non-fatal: it never influences model choice,
// prompts, budgets or optimization decisions. Set GEAK_LIVE_TRACE=0 to disable.
(function startLiveTracker() {
  try {
    if (String(process.env.GEAK_LIVE_TRACE || '1') === '0') return;
    // Same default as kernel_lane.js computes for EXP_ROOT: args.exp_root is NOT
    // required, so bailing out when it is absent silently disabled tracking for
    // every ordinary invocation.
    const expRoot = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp'))
      .replace(/\/+$/, '');
    const { spawn } = require('child_process');
    const child = spawn('python3', [
      '-B', `${WORKFLOW_DIR}/../interface/geak_trace_collector.py`,
      '--exp-root', expRoot,
      // Runtime-supplied invocation identity: the wf_*.json record carries this
      // same args object, so matching it identifies THIS launch deterministically
      // instead of guessing from timing or from being the only run around.
      '--identity-args', JSON.stringify(A),
      '--script-dir', WORKFLOW_DIR,
      // Per-run filename: two runs under one exp_root must not overwrite each other.
      '--out-dir', expRoot,
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
    log(`Live execution tracker started -> ${expRoot}/geak_trace_<runId>.json`);
  } catch (e) {
    try { log(`Live execution tracker not started (non-fatal): ${e && e.message}`); } catch (_) {}
  }
})();

// ===========================================================================
// CONFIG-AWARE OPERATOR FAMILY (args.config_aware = "true" | "on").
// One kernel_lane.js per config in args.family_spec, launched together over the GPU pool the same way
// bakeoff launches its lanes (Promise.all + makeSem, one GPU id per lane, a thrown lane is caught), so
// nesting stays dispatcher -> kernel_lane. Validation, status mapping, aggregation and the catalog are
// pure JS; agents only propose a plan (advisory) and write files verbatim. The catalog covers each
// config's own exact signature only and is never deployment_ready. Runs before the kernel_path-based
// paths below; those stay untouched when the flag is off.
// ===========================================================================
if (CONFIG_AWARE) {
  if (MODE === 'bakeoff') {
    throw new Error('config_aware does not support mode=bakeoff yet; set each config\'s mode (optimize|author) in family_spec.');
  }
  if (MODE !== 'optimize' && MODE !== 'author') {
    throw new Error(`unknown mode='${MODE}' with config_aware; each config carries its own mode (optimize|author).`);
  }
  if (OPTIMIZER) throw new Error(`config_aware does not support optimizer=${OPTIMIZER} yet`);
  if ((A.gpus_per_job != null && String(A.gpus_per_job).trim() !== '1') || String(A.job_gpu_ids || '').trim()) {
    throw new Error('config_aware gives each config lane one GPU id; gpus_per_job > 1 and job_gpu_ids are not supported.');
  }
  if (A.apply_to_original != null && String(A.apply_to_original).trim().toLowerCase() !== 'false') {
    throw new Error('config_aware never modifies an input kernel_path; apply_to_original must be "false".');
  }
  if (/(^|\/)kernel_workflow\.js$/.test(WORKER)) {
    throw new Error('config_aware lanes must run kernel_lane.js, not the kernel_workflow.js dispatcher.');
  }

  const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
  const isObj = (v) => v != null && typeof v === 'object' && !Array.isArray(v);
  const isInt = (v) => Number.isInteger(v);
  const finite = (v) => (typeof v === 'number' && Number.isFinite(v)) ? v : null;
  const str = (v) => (typeof v === 'string' && v.trim()) ? v.trim() : null;
  const clip = (v, n) => { const t = typeof v === 'string' ? v.trim() : ''; return t.length > n ? t.slice(0, n - 3) + '...' : t; };
  // Key-order-independent JSON, so two contracts compare equal regardless of how they were written.
  const canon = (v) => Array.isArray(v) ? `[${v.map(canon).join(',')}]`
    : isObj(v) ? `{${Object.keys(v).sort().map((k) => `${JSON.stringify(k)}:${canon(v[k])}`).join(',')}}`
    : JSON.stringify(v === undefined ? null : v);
  const LANE_FIELDS = ['mode', 'target_language', 'budget', 'gpu_mode', 'task',
    'use_learned_kb', 'update_experience', 'warm_start', 'apply_to_original'];
  const DEFAULT_FIELDS = [...LANE_FIELDS, 'gpu_ids'];
  // Values kernel_lane.js understands for each switch; anything else would silently become its default.
  const ENUMS = {
    gpu_mode: ['pool', 'pin'],
    use_learned_kb: ['true', 'false'],
    update_experience: ['on', 'off', 'true', 'false', 'none'],
    warm_start: ['on', 'off', 'true', 'false', 'none', 'reference', 'return_after_read'],
  };
  const PLAN_FORMS = ['shared_kernel', 'per_signature', 'internal_buckets'];
  const CROSS_CONFIG = {
    status: 'unsupported',
    reason: 'no candidate adapter: a lane patch is relative to its own workspace, so no implementation was run on a sibling config',
  };

  // ---- Phase 1: validate + normalize (synchronous; nothing has been spawned yet) ----
  const validateFamilySpec = (spec) => {
    if (!isObj(spec)) {
      throw new Error('config_aware requires args.family_spec as an already-parsed JSON object (not a string or a path).');
    }
    const errs = [];
    if (spec.schema_version !== 1) errs.push(`schema_version must be 1 (got ${JSON.stringify(spec.schema_version)})`);
    const familyId = typeof spec.family_id === 'string' ? spec.family_id.trim() : '';
    if (!SAFE_ID.test(familyId)) errs.push('family_id must be a non-empty [A-Za-z0-9._-] identifier');
    const operator = typeof spec.operator === 'string' ? spec.operator.trim() : '';
    if (!operator) errs.push('operator must be a non-empty string');
    const defaults = spec.defaults == null ? {} : spec.defaults;
    if (!isObj(defaults)) errs.push('defaults must be an object');
    const checkFields = (o, allowed, where) => {
      if (!isObj(o)) return;
      for (const k of Object.keys(o)) {
        if (!allowed.includes(k)) errs.push(`${where}.${k} is not a supported field (allowed: ${allowed.join(', ')})`);
      }
    };
    checkFields(defaults, DEFAULT_FIELDS, 'defaults');

    const dep = spec.deployment;
    let axes = null;
    if (dep != null && !isObj(dep)) errs.push('deployment must be an object');
    if (isObj(dep)) {
      if (dep.signature_axes != null) {
        if (!Array.isArray(dep.signature_axes) || !dep.signature_axes.length ||
            !dep.signature_axes.every((s) => typeof s === 'string' && s.trim())) {
          errs.push('deployment.signature_axes must be a non-empty array of axis names');
        } else axes = dep.signature_axes.map((s) => s.trim());
      }
      for (const k of ['tensor_parallel_size', 'expert_parallel_size']) {
        if (dep[k] != null && !(isInt(dep[k]) && dep[k] > 0)) errs.push(`deployment.${k} must be a positive integer`);
      }
    }

    const target = { expected_gfx: '', expected_target: '' };
    if (spec.target != null && !isObj(spec.target)) errs.push('target must be an object');
    if (isObj(spec.target)) {
      target.expected_gfx = String(spec.target.expected_gfx || '').trim().toLowerCase();
      target.expected_target = String(spec.target.expected_target || '').trim().toLowerCase();
      if (!!target.expected_gfx !== !!target.expected_target) {
        errs.push('target.expected_gfx and target.expected_target must be supplied together');
      }
      if (target.expected_gfx && !/^gfx[0-9a-f]+$/.test(target.expected_gfx)) errs.push('target.expected_gfx must be a gfx token');
    }

    const configs = Array.isArray(spec.configs) ? spec.configs : null;
    if (!configs || !configs.length) errs.push('configs must be a non-empty array');
    const seen = new Set();
    const dispatchKeys = new Map();
    let rank = axes ? axes.length : null;
    const out = [];
    (configs || []).forEach((c, i) => {
      const at = `configs[${i}]`;
      if (!isObj(c)) { errs.push(`${at} must be an object`); return; }
      const id = typeof c.id === 'string' ? c.id.trim() : '';
      const where = id ? `config '${id}'` : at;
      if (!SAFE_ID.test(id)) errs.push(`${at}.id must be a non-empty [A-Za-z0-9._-] identifier`);
      else if (seen.has(id)) errs.push(`duplicate config id '${id}'`);
      else seen.add(id);
      const kernelPath = typeof c.kernel_path === 'string' ? c.kernel_path.trim() : '';
      if (!kernelPath.startsWith('/')) errs.push(`${where}: kernel_path must be an absolute path`);
      if (!isObj(c.op_spec)) errs.push(`${where}: op_spec must be an object`);
      const ov = c.overrides == null ? {} : c.overrides;
      if (!isObj(ov)) errs.push(`${where}: overrides must be an object`);
      checkFields(ov, LANE_FIELDS, `${where}.overrides`);

      // config.overrides -> family_spec.defaults -> kernel_lane default (target_language has no default).
      const pick = (f) => (isObj(ov) && ov[f] != null) ? ov[f]
        : ((isObj(defaults) && defaults[f] != null) ? defaults[f] : undefined);
      const mode = String(pick('mode') != null ? pick('mode') : 'optimize').trim().toLowerCase();
      if (mode !== 'optimize' && mode !== 'author') errs.push(`${where}: mode must be optimize or author (got '${mode}')`);
      const lang = pick('target_language') != null ? String(pick('target_language')).trim().toLowerCase() : '';
      if (!lang) {
        errs.push(`${where}: target_language must resolve from overrides or defaults ` +
          '(it selects the warm-start KB page, so the lane\'s triton default is not inherited)');
      }
      const rawBudget = pick('budget');
      const budget = rawBudget == null ? 6
        : (typeof rawBudget === 'number' ? rawBudget
          : (/^\s*\d+\s*$/.test(String(rawBudget)) ? parseInt(rawBudget, 10) : NaN));
      if (!(isInt(budget) && budget > 0)) errs.push(`${where}: budget must be a positive integer`);
      const ato = pick('apply_to_original');
      if (ato != null && String(ato).trim().toLowerCase() !== 'false') {
        errs.push(`${where}: apply_to_original must be false on the config_aware path`);
      }
      const knobs = {};
      for (const [f, allowed] of Object.entries(ENUMS)) {
        const v = pick(f);
        if (v == null) continue;
        const s = String(v).trim().toLowerCase();
        if (!allowed.includes(s)) errs.push(`${where}: ${f} must be one of ${allowed.join('|')} (got '${s}')`);
        else knobs[f] = s;
      }
      const task = pick('task');
      if (task != null && typeof task !== 'string') errs.push(`${where}: task must be a string`);

      let sig = null;
      if (c.signature != null) {
        if (!Array.isArray(c.signature) || !c.signature.length || !c.signature.every((x) => isInt(x) && x >= 0)) {
          errs.push(`${where}: signature must be a non-empty array of non-negative integers`);
        } else {
          sig = c.signature.slice();
          if (rank == null) rank = sig.length;
          if (sig.length !== rank) {
            errs.push(`${where}: signature rank ${sig.length} does not match ` +
              `${axes ? 'deployment.signature_axes' : 'the other configs'} (${rank})`);
          }
        }
      }
      if (c.signature_ranges != null) {
        if (!isObj(c.signature_ranges)) errs.push(`${where}: signature_ranges must be an object keyed by axis index`);
        else if (!sig) errs.push(`${where}: signature_ranges requires a valid signature`);
        else {
          for (const [ax, r] of Object.entries(c.signature_ranges)) {
            const k = /^\d+$/.test(ax) ? parseInt(ax, 10) : -1;
            if (k < 0 || k >= sig.length) errs.push(`${where}: signature_ranges key '${ax}' is not an axis index`);
            else if (!Array.isArray(r) || r.length !== 2 || !r.every(isInt) || r[0] > r[1]) {
              errs.push(`${where}: signature_ranges['${ax}'] must be [lo, hi] integers with lo <= hi`);
            } else if (sig[k] < r[0] || sig[k] > r[1]) {
              errs.push(`${where}: signature[${k}]=${sig[k]} lies outside signature_ranges['${ax}']`);
            }
          }
        }
      }
      const contract = c.contract == null ? null : c.contract;
      if (contract != null && !isObj(contract)) errs.push(`${where}: contract must be an object`);
      if (isObj(contract)) {
        for (const k of ['semantics', 'graph']) {
          if (contract[k] != null && !isObj(contract[k])) errs.push(`${where}: contract.${k} must be an object`);
        }
      }
      if (c.incumbent != null && !isObj(c.incumbent)) errs.push(`${where}: incumbent must be an object`);
      if (c.workload_weight != null && !(finite(c.workload_weight) != null && c.workload_weight >= 0)) {
        errs.push(`${where}: workload_weight must be a finite number >= 0`);
      }
      if (c.priority != null && !isInt(c.priority)) errs.push(`${where}: priority must be an integer`);
      if (c.tags != null && !(Array.isArray(c.tags) && c.tags.every((t) => typeof t === 'string'))) {
        errs.push(`${where}: tags must be an array of strings`);
      }
      // A signature + contract is a dispatch key: two configs with the same key would claim one catalog slot.
      if (sig) {
        const dk = canon([sig, isObj(contract) ? contract : {}]);
        if (dispatchKeys.has(dk)) errs.push(`${where}: same signature and contract as config '${dispatchKeys.get(dk)}'`);
        else dispatchKeys.set(dk, id);
      }
      out.push({
        id, index: i,
        kernel_path: kernelPath.length > 1 ? kernelPath.replace(/\/+$/, '') : kernelPath,
        op_spec: c.op_spec, mode, target_language: lang, budget,
        gpu_mode: knobs.gpu_mode || 'pool',
        use_learned_kb: knobs.use_learned_kb || null,
        update_experience: knobs.update_experience || null,
        warm_start: knobs.warm_start || null,
        task: typeof task === 'string' ? task : '',
        signature: sig,
        signature_ranges: isObj(c.signature_ranges) ? c.signature_ranges : null,
        contract: isObj(contract) ? contract : null,
        incumbent: isObj(c.incumbent) ? c.incumbent : null,
        workload_weight: c.workload_weight != null ? c.workload_weight : null,
        priority: c.priority != null ? c.priority : null,
        tags: Array.isArray(c.tags) ? c.tags.slice() : [],
      });
    });
    if (errs.length) throw new Error(`family_spec invalid (${errs.length} problem(s)):\n  - ${errs.join('\n  - ')}`);
    return {
      schema_version: 1, family_id: familyId, operator,
      description: typeof spec.description === 'string' ? spec.description : '',
      deployment: isObj(dep) ? dep : null, signature_axes: axes, target, defaults, configs: out,
    };
  };

  const FAM = validateFamilySpec(A.family_spec);
  const byId = new Map(FAM.configs.map((c) => [c.id, c]));
  const poolSrc = A.gpu_ids != null ? A.gpu_ids : (FAM.defaults.gpu_ids != null ? FAM.defaults.gpu_ids : '0');
  const FAMILY_GPUS = String(poolSrc).split(',').map((s) => s.trim()).filter(Boolean);
  if (!FAMILY_GPUS.length) throw new Error('config_aware needs at least one GPU id in gpu_ids');
  if (new Set(FAMILY_GPUS).size !== FAMILY_GPUS.length) {
    throw new Error(`gpu_ids lists a GPU twice (${FAMILY_GPUS.join(',')}); two lanes would share one GPU`);
  }
  const parentGfx = String(A.expected_gfx || '').trim().toLowerCase();
  const parentTarget = String(A.expected_target || '').trim().toLowerCase();
  if (!!parentGfx !== !!parentTarget) throw new Error('expected_gfx and expected_target must be supplied together');
  if (parentGfx && FAM.target.expected_gfx &&
      (parentGfx !== FAM.target.expected_gfx || parentTarget !== FAM.target.expected_target)) {
    throw new Error(`expected_gfx/expected_target (${parentGfx}/${parentTarget}) conflict with family_spec.target ` +
      `(${FAM.target.expected_gfx}/${FAM.target.expected_target})`);
  }
  const LANE_GFX_F = parentGfx || FAM.target.expected_gfx;
  const LANE_TARGET_F = parentTarget || FAM.target.expected_target;
  const EXP_ROOT_F = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');
  const pinnedDir = String(A.eval_dir != null ? A.eval_dir : '').trim();
  if (pinnedDir && !pinnedDir.startsWith('/')) throw new Error('eval_dir must be an absolute path');
  const ignored = ['kernel_path', 'target_language', 'budget', 'gpu_mode', 'task', 'op_spec', 'workload_spec_path',
    'backends', 'use_learned_kb', 'update_experience', 'warm_start'].filter((k) => A[k] != null);
  if (A.mode != null) ignored.unshift('mode');
  if (ignored.length) {
    log(`config_aware: parent ${ignored.join(', ')} ignored; each config resolves them from overrides -> family_spec.defaults.`);
  }

  // Knobs forwarded from the parent call to every lane, only when the caller set them.
  const PARENT_FORWARD = ['exp_root', 'expected_device_name', 'expected_physical_cu_count', 'gpu_perf_level',
    'kb_artifacts_dir', 'kb_mode', 'kb_store_dir', 'kb_framework_version', 'perf_knowledge_dir',
    'use_expert_skills', 'expert_skills_dir', 'llm_stats', 'profile', 'fast_mode', 'max_rounds', 'min_improve',
    'candidate_floor', 'engineer_minutes', 'deep_cost', 'max_no_improve', 'dra_enabled', 'dra_max_questions',
    'dra_blindspot', 'dra_max_blindspots', 'agent_timeout_ms', 'agent_retries'];
  const configSummary = (c) => ({
    id: c.id, mode: c.mode, target_language: c.target_language, op_spec: c.op_spec,
    signature: c.signature, contract: c.contract, tags: c.tags, priority: c.priority,
    workload_weight: c.workload_weight, has_incumbent: !!c.incumbent,
  });
  const fmtInputs = (o) => Object.entries(o).map(([k, v]) =>
    `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');
  const tryAgent = async (prompt, opts) => {
    try { return await agent(prompt, opts); } catch (e) {
      log(`  ${opts.label} failed (${String(e && e.message ? e.message : e).slice(0, 200)})`);
      return null;
    }
  };
  // Files are rendered in JS; the writer copies them verbatim and cannot change a status or a number.
  const persist = async (files, label) => {
    const body = files.map((f) => `### ${f.path}\n\`\`\`\`\n${f.content}\n\`\`\`\``).join('\n\n');
    const r = await tryAgent(
      'You are the file_writer. PHASE=persist_family_artifacts.\n' +
      'Use the Write tool to create each file below with EXACTLY the content between its four-backtick fences, ' +
      'verbatim (create parent directories if needed; do NOT reformat, reinterpret, truncate or summarize it). ' +
      'Write nothing else.\n\n' + body + '\n\nThen return {"written": [<every path you wrote>]}.',
      { phase: 'Report', label, schema: { type: 'object', properties: { written: { type: 'array', items: { type: 'string' } } },
        required: ['written'], additionalProperties: true } });
    const written = r && Array.isArray(r.written) ? r.written : [];
    const missing = files.filter((f) => !written.includes(f.path)).map((f) => f.path);
    if (missing.length) log(`  ${label}: not confirmed written: ${missing.join(', ')}`);
    return missing.length === 0;
  };

  phase('Freeze');
  log(`config_aware: family ${FAM.family_id} (${FAM.operator}), ${FAM.configs.length} config(s), ` +
      `GPUs=${FAMILY_GPUS.join(',')} => concurrency ${Math.min(FAM.configs.length, FAMILY_GPUS.length)}; ` +
      `est cost ~${FAM.configs.length} lane budget(s).`);
  let FAMILY_DIR = pinnedDir.replace(/\/+$/, '');
  if (!FAMILY_DIR) {
    const prefix = `${EXP_ROOT_F}/family_${FAM.family_id}_`;
    const made = await tryAgent(
      'You are the file_writer. PHASE=family_setup.\n' +
      'Create one NEW, empty directory for an operator-family run. Run exactly this Bash command:\n' +
      `  mkdir -p "${EXP_ROOT_F}" && d="${prefix}$(date -u +%Y%m%dT%H%M%SZ)_$$" && mkdir "$d" && echo "$d"\n` +
      'If the final mkdir fails because the directory exists, run the command again. Create nothing else.\n' +
      'Return {"eval_dir": "<the absolute path the command printed>"}.',
      { phase: 'Freeze', label: 'file_writer:family_setup', schema: { type: 'object',
        properties: { eval_dir: { type: 'string' } }, required: ['eval_dir'], additionalProperties: true } });
    const got = made && typeof made.eval_dir === 'string' ? made.eval_dir.trim().replace(/\/+$/, '') : '';
    if (!got.startsWith(prefix) || got.length === prefix.length || /(^|\/)\.\.(\/|$)/.test(got)) {
      log(`config_aware: family eval dir setup failed (got '${got}'); no lane was started.`);
      return { schema_version: 1, mode: 'config_aware', config_aware: true, family_id: FAM.family_id,
        validation_status: 'setup_failed', eval_dir: null, configs: [],
        reason: 'could not create a family eval dir; pass args.eval_dir to pin one' };
    }
    FAMILY_DIR = got;
  }
  const PATHS = {
    plan: `${FAMILY_DIR}/family_plan.json`, results: `${FAMILY_DIR}/family_results.json`,
    report: `${FAMILY_DIR}/family_report.md`, catalog: `${FAMILY_DIR}/implementation_catalog.json`,
  };

  // ---- Phase 2: plan (advisory; any invalid answer -> one group per config) ----
  const defaultOrder = () => FAM.configs.slice().sort((a, b) => {
    const pa = a.priority == null ? Infinity : a.priority;
    const pb = b.priority == null ? Infinity : b.priority;
    return pa !== pb ? (pa < pb ? -1 : 1) : a.index - b.index;
  }).map((c) => c.id);
  const validatePlan = (raw) => {
    if (!isObj(raw)) return { error: 'planner returned no object' };
    if (!Array.isArray(raw.groups) || !raw.groups.length) return { error: 'planner returned no groups' };
    const assigned = new Map();
    const groupIds = new Set();
    const groups = [];
    for (let gi = 0; gi < raw.groups.length; gi++) {
      const g = raw.groups[gi];
      if (!isObj(g)) return { error: `groups[${gi}] is not an object` };
      const gid = typeof g.group_id === 'string' ? g.group_id.trim() : '';
      if (!SAFE_ID.test(gid)) return { error: `groups[${gi}].group_id is empty or not [A-Za-z0-9._-]` };
      if (groupIds.has(gid)) return { error: `duplicate group_id '${gid}'` };
      groupIds.add(gid);
      if (!Array.isArray(g.config_ids) || !g.config_ids.length) return { error: `group '${gid}' has no config_ids` };
      for (const id of g.config_ids) {
        if (!byId.has(id)) return { error: `group '${gid}' names unknown config '${id}'` };
        if (assigned.has(id)) return { error: `config '${id}' is assigned to both '${assigned.get(id)}' and '${gid}'` };
        assigned.set(id, gid);
      }
      const members = g.config_ids.map((id) => byId.get(id));
      for (const part of ['semantics', 'graph']) {
        const kinds = new Set(members.map((c) => canon(c.contract && c.contract[part] != null ? c.contract[part] : null)));
        if (kinds.size > 1) return { error: `group '${gid}' mixes configs with different contract.${part}` };
      }
      const hints = {};
      if (isObj(g.config_hints)) {
        for (const id of g.config_ids) { const h = clip(g.config_hints[id], 400); if (h) hints[id] = h; }
      }
      groups.push({
        group_id: gid, config_ids: g.config_ids.slice(),
        strategy: clip(g.strategy, 600), reasoning: clip(g.reasoning, 600),
        implementation_form: PLAN_FORMS.includes(g.implementation_form) ? g.implementation_form : null,
        confidence: (finite(g.confidence) != null && g.confidence >= 0 && g.confidence <= 1) ? g.confidence : null,
        config_hints: hints,
      });
    }
    const missing = FAM.configs.filter((c) => !assigned.has(c.id)).map((c) => c.id);
    if (missing.length) return { error: `planner omitted config(s): ${missing.join(', ')}` };
    const eo = raw.execution_order;
    const orderOk = Array.isArray(eo) && eo.length === FAM.configs.length &&
      new Set(eo).size === eo.length && eo.every((id) => byId.has(id));
    return { groups, execution_order: orderOk ? eo.slice() : defaultOrder(),
      order_source: orderOk ? 'planner' : 'default', notes: clip(raw.notes, 800) };
  };

  phase('Discover');
  const PLANNER_SCHEMA = { type: 'object', additionalProperties: true, required: ['groups'], properties: {
    groups: { type: 'array', items: { type: 'object', additionalProperties: true, properties: {
      group_id: { type: 'string' }, config_ids: { type: 'array', items: { type: 'string' } },
      strategy: { type: 'string' }, reasoning: { type: 'string' }, implementation_form: { type: 'string' },
      confidence: { type: 'number' }, config_hints: { type: 'object', additionalProperties: { type: 'string' } },
    } } },
    execution_order: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  } };
  const rawPlan = await tryAgent(
    `You are the operator_family_planner. PHASE=plan.
First Read ${WORKFLOW_DIR}/roles/operator_family_planner.md and follow its instructions for PHASE=plan.
You may Read knowledge under ${WORKFLOW_DIR}/knowledge/ (including knowledge/learned/INDEX.md). Write no files.

## Inputs
${fmtInputs({
    FAMILY_ID: FAM.family_id, OPERATOR: FAM.operator, DESCRIPTION: FAM.description,
    DEPLOYMENT: FAM.deployment || {}, SIGNATURE_AXES: FAM.signature_axes || [],
    EXPECTED_GFX: LANE_GFX_F || 'unknown (each lane detects the GPU on-box)',
    GPU_POOL_SIZE: FAMILY_GPUS.length,
    OBJECTIVE: 'per-config verified speedup against each config\'s own baseline; each config runs its own lane',
    CONFIGS: FAM.configs.map(configSummary),
  })}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`,
    { phase: 'Discover', label: 'operator_family_planner:plan', schema: PLANNER_SCHEMA });
  const checked = validatePlan(rawPlan);
  const planBody = checked.error
    ? { groups: FAM.configs.map((c) => ({ group_id: c.id, config_ids: [c.id], strategy: '', reasoning: '',
          implementation_form: null, confidence: null, config_hints: {} })),
        execution_order: defaultOrder(), order_source: 'default', notes: '' }
    : checked;
  if (checked.error) log(`config_aware: planner output rejected (${checked.error}); using one group per config.`);
  const PLAN = {
    schema_version: 1, family_id: FAM.family_id, operator: FAM.operator,
    source: checked.error ? 'fallback' : 'planner', fallback_reason: checked.error || null,
    groups: planBody.groups, execution_order: planBody.execution_order, order_source: planBody.order_source,
    notes: planBody.notes, gpu_pool: FAMILY_GPUS,
    concurrency: Math.min(FAM.configs.length, FAMILY_GPUS.length),
    cross_config_evaluation: CROSS_CONFIG,
  };
  const groupOf = new Map();
  for (const g of PLAN.groups) for (const id of g.config_ids) groupOf.set(id, g);
  const planPersisted = await persist([{ path: PATHS.plan, content: JSON.stringify(PLAN, null, 2) }], 'file_writer:family_plan');

  // ---- Phase 3: one kernel_lane.js per config, all launched together, one GPU id each ----
  const sigStr = (c) => c.signature ? `[${c.signature.join(', ')}]` : '';
  const familyTask = (c, g) => {
    const sib = FAM.configs.filter((x) => x.id !== c.id);
    const sibStr = (x) => x.id + (x.signature ? sigStr(x)
      : (isObj(x.op_spec) && x.op_spec.shapes ? ` ${clip(JSON.stringify(x.op_spec.shapes), 80)}` : ''));
    return [
      '[Operator-family context - advisory only; this config\'s own profile and measurements override it]',
      `family=${FAM.family_id} operator=${FAM.operator} config=${c.id}` +
        (c.signature ? ` signature=${sigStr(c)}` : '') +
        (FAM.signature_axes ? ` axes=${FAM.signature_axes.join(',')}` : '') +
        ` group=${g.group_id} (${g.config_ids.length} config(s): ${g.config_ids.join(', ')})`,
      sib.length ? `siblings, optimized in separate lanes this lane cannot see: ${sib.slice(0, 12).map(sibStr).join('; ')}` +
        (sib.length > 12 ? `; +${sib.length - 12} more` : '') : '',
      g.strategy ? `planner strategy hypothesis: ${g.strategy}` : '',
      g.reasoning ? `planner reasoning: ${g.reasoning}` : '',
      g.config_hints[c.id] ? `config hint: ${g.config_hints[c.id]}` : '',
      'Optimize ONLY this config against its own oracle and baseline. The TechLead may drop this hint after profiling.',
      c.task ? `caller task: ${c.task}` : '',
    ].filter(Boolean).join('\n');
  };
  const makeLaneArgs = (c, g, gpu) => ({
    ...Object.fromEntries(PARENT_FORWARD.filter((k) => A[k] != null).map((k) => [k, A[k]])),
    ...(LANE_GFX_F ? { expected_gfx: LANE_GFX_F, expected_target: LANE_TARGET_F } : {}),
    ...(c.use_learned_kb ? { use_learned_kb: c.use_learned_kb } : {}),
    ...(c.update_experience ? { update_experience: c.update_experience } : {}),
    ...(c.warm_start ? { warm_start: c.warm_start } : {}),
    kernel_path: c.kernel_path, workflow_dir: WORKFLOW_DIR, eval_dir: `${FAMILY_DIR}/configs/${c.id}`,
    mode: c.mode, target_language: c.target_language, budget: c.budget,
    gpu_ids: gpu, gpu_mode: c.gpu_mode, op_spec: c.op_spec, task: familyTask(c, g),
    apply_to_original: 'false',
  });
  const baseRecord = (c, g, k, gpu, laneArgs) => ({
    config_id: c.id, group_id: g.group_id, execution_index: k, gpu_id: gpu,
    mode: c.mode, target_language: c.target_language, budget: c.budget,
    kernel_path: c.kernel_path, eval_dir: laneArgs.eval_dir,
    signature: c.signature, contract: c.contract, incumbent: c.incumbent, workload_weight: c.workload_weight,
    status: null, lane_validation_status: null,
    speedup_vs_stock: null, speedup_vs_incumbent: null,
    incumbent_comparison: { status: 'unavailable', reason: c.incumbent
      ? 'no incumbent adapter: the incumbent was not run under this lane\'s oracle and timing basis'
      : 'no incumbent supplied' },
    final_speedup: null, final_weighted: null, final_geomean: null, metric_kind: null,
    timing_basis: null, workload_aligned: null, lane_eval_dir: null, report_path: null, final_patch: null,
    warm_start: null, kb_written: null, learned_card: null, reason: null, error: null,
  });
  const normalizeLaneResult = (rec, r) => {
    if (!isObj(r)) return { ...rec, status: 'invalid_result', error: `lane returned ${r === null ? 'null' : typeof r}` };
    const vs = typeof r.validation_status === 'string' ? r.validation_status.trim().toLowerCase() : '';
    const prim = finite(r.final_speedup) != null ? r.final_speedup
      : (finite(r.final_weighted) != null ? r.final_weighted : finite(r.final_geomean));
    const rr = {
      ...rec, lane_validation_status: vs || null,
      final_speedup: finite(r.final_speedup), final_weighted: finite(r.final_weighted), final_geomean: finite(r.final_geomean),
      speedup_vs_stock: prim,
      metric_kind: r.workload_aligned === true ? 'weighted' : (r.workload_aligned === false ? 'geomean' : null),
      workload_aligned: typeof r.workload_aligned === 'boolean' ? r.workload_aligned : null,
      timing_basis: str(r.timing_basis), lane_eval_dir: str(r.eval_dir), report_path: str(r.report_path),
      final_patch: str(r.final_patch), warm_start: isObj(r.warm_start) ? r.warm_start : null,
      kb_written: isObj(r.kb_written) ? r.kb_written : null, learned_card: isObj(r.learned_card) ? r.learned_card : null,
      reason: str(r.reason),
    };
    if (!vs) return { ...rr, status: 'invalid_result', error: 'lane result has no validation_status' };
    if (vs === 'accepted') {
      if (prim == null) return { ...rr, status: 'invalid_result', error: 'accepted lane result has no finite speedup' };
      if (prim > 1.0 && !rr.final_patch) return { ...rr, status: 'invalid_result', error: 'accepted speedup without a final_patch' };
      return { ...rr, status: prim > 1.0 ? 'optimized' : 'accepted_no_improvement' };
    }
    if (vs === 'flagged' || vs === 'author_failed' || vs === 'no_baseline') return { ...rr, status: vs };
    return { ...rr, status: 'not_accepted' };
  };

  phase('Bakeoff');
  const famSem = makeSem(FAMILY_GPUS);
  const RECORDS = await Promise.all(PLAN.execution_order.map((id, k) => famSem.with(1, async ([gpu]) => {
    const c = byId.get(id);
    const g = groupOf.get(id);
    const laneArgs = makeLaneArgs(c, g, gpu);
    const rec = baseRecord(c, g, k, gpu, laneArgs);
    let r;
    try {
      r = await workflow({ scriptPath: WORKER }, laneArgs);
    } catch (e) {
      const msg = String(e && e.message ? e.message : e).slice(0, 500);
      log(`config ${id} lane failed: ${msg}`);
      return { ...rec, status: 'lane_failed', error: msg };
    }
    const out = normalizeLaneResult(rec, r);
    log(`config ${id} -> ${out.status}` +
        (out.speedup_vs_stock != null ? ` ${out.speedup_vs_stock.toFixed(2)}x` : '') +
        ` (lane ${out.lane_validation_status || 'n/a'})`);
    return out;
  })));
  RECORDS.sort((a, b) => a.execution_index - b.execution_index);

  // ---- Phase 4-6: aggregate, catalog, report (pure functions of RECORDS) ----
  const OK_STATUSES = ['optimized', 'accepted_no_improvement'];
  const countOf = (s) => RECORDS.filter((r) => r.status === s).length;
  const byStatus = {};
  for (const r of RECORDS) byStatus[r.status] = (byStatus[r.status] || 0) + 1;
  const SUMMARY = {
    total: RECORDS.length, optimized: countOf('optimized'),
    accepted_no_improvement: countOf('accepted_no_improvement'), flagged: countOf('flagged'),
    failed: RECORDS.filter((r) => !OK_STATUSES.includes(r.status) && r.status !== 'flagged').length,
    by_status: byStatus,
  };
  const AGGREGATE = (() => {
    const inc = RECORDS.filter((r) => OK_STATUSES.includes(r.status));
    const base = {
      value: null, kind: null, metric: null, timing_basis: null, weighting: 'unweighted',
      rule: 'geometric mean of per-config speedup_vs_stock over accepted configs; requires one metric kind and one known timing_basis',
      included: inc.map((r) => r.config_id),
      excluded: RECORDS.filter((r) => !OK_STATUSES.includes(r.status)).map((r) => ({ config_id: r.config_id, status: r.status })),
      complete: inc.length === RECORDS.length, reason: '',
    };
    if (!inc.length) return { ...base, reason: 'no accepted config results' };
    const bases = [...new Set(inc.map((r) => r.timing_basis))];
    const kinds = [...new Set(inc.map((r) => r.metric_kind))];
    if (bases.length !== 1 || bases[0] == null || bases[0] === 'unknown') {
      return { ...base, reason: `timing_basis not comparable across accepted configs: ${bases.map(String).join(', ')}` };
    }
    if (kinds.length !== 1 || kinds[0] == null) {
      return { ...base, reason: `metric kind not comparable across accepted configs: ${kinds.map(String).join(', ')}` };
    }
    if (inc.some((r) => !(r.speedup_vs_stock > 0))) return { ...base, reason: 'an accepted config has a non-positive speedup' };
    const value = Math.exp(inc.reduce((s, r) => s + Math.log(r.speedup_vs_stock), 0) / inc.length);
    return { ...base, value, kind: 'geomean_of_config_speedups', metric: kinds[0], timing_basis: bases[0],
      reason: base.complete ? 'all configs accepted under one timing basis'
        : `partial family: ${base.excluded.length} config(s) without an accepted result are excluded` };
  })();
  const FALLBACK_REASON = {
    accepted_no_improvement: 'accepted but not faster than its own baseline (speedup <= 1.0)',
    flagged: 'the Director flagged the lane result',
    author_failed: 'author lane produced no correct baseline',
    no_baseline: 'lane could not establish a baseline',
  };
  const CATALOG = {
    schema_version: 1, family_id: FAM.family_id, operator: FAM.operator,
    deployment: FAM.deployment, signature_axes: FAM.signature_axes,
    coverage_policy: 'exact_per_config', deployment_ready: false, cross_config_evaluation: CROSS_CONFIG,
    implementations: RECORDS.filter((r) => r.status === 'optimized').map((r) => ({
      implementation_id: `${FAM.family_id}:${r.config_id}`, group_id: r.group_id, config_ids: [r.config_id],
      artifact: { kind: 'lane_patch', path: r.final_patch }, backend: r.target_language, mode: r.mode,
      contract: r.contract,
      coverage: { exact: r.signature ? [r.signature] : [], ranges: [] },
      orchestration_key: r.config_id,
      validation: 'accepted', verified_speedup: r.speedup_vs_stock, speedup_vs_stock: r.speedup_vs_stock,
      speedup_vs_incumbent: null, incumbent_comparison: r.incumbent_comparison,
      metric_kind: r.metric_kind, timing_basis: r.timing_basis,
      candidate_class: 'optimization_candidate', deployment_ready: false,
      eval_dir: r.lane_eval_dir || r.eval_dir, report_path: r.report_path,
    })),
    fallback: RECORDS.filter((r) => r.status !== 'optimized').map((r) => ({
      config_id: r.config_id, selected: r.incumbent ? 'incumbent' : 'stock', status: r.status,
      reason: FALLBACK_REASON[r.status]
        || (r.status === 'lane_failed' ? `lane threw: ${r.error}`
          : r.status === 'invalid_result' ? `malformed lane result: ${r.error}`
            : `lane validation_status=${r.lane_validation_status}`),
    })),
  };
  const fx = (v) => (v == null ? 'n/a' : `${v.toFixed(3)}x`);
  const cell = (v) => String(v == null || v === '' ? '-' : v).replace(/\|/g, '\\|').replace(/\n/g, ' ');
  const REPORT_MD = [
    `# Operator family report: ${FAM.family_id}`,
    '',
    `Operator: ${FAM.operator}. Configs: ${SUMMARY.total}. Optimized: ${SUMMARY.optimized}. ` +
      `Accepted without improvement: ${SUMMARY.accepted_no_improvement}. Flagged: ${SUMMARY.flagged}. ` +
      `Failed: ${SUMMARY.failed}.`,
    FAM.description ? `\n${FAM.description}` : '',
    '',
    'Every speedup below is against that config\'s own baseline (stock), measured by its own lane. ' +
      'No incumbent and no sibling config was executed, so `speedup_vs_incumbent` is n/a and the catalog is not deployment-ready.',
    '',
    '## Configs',
    '',
    '| config | signature | mode | language | budget | kernel_path |',
    '|---|---|---|---|---|---|',
    ...FAM.configs.map((c) => `| ${cell(c.id)} | ${cell(sigStr(c))} | ${c.mode} | ${cell(c.target_language)} | ${c.budget} | ${cell(c.kernel_path)} |`),
    '',
    `## Plan (${PLAN.source}${PLAN.fallback_reason ? `: ${PLAN.fallback_reason}` : ''})`,
    '',
    'Groups are planner hypotheses. Each config still ran its own lane.',
    '',
    '| group | configs | form | confidence | strategy |',
    '|---|---|---|---|---|',
    ...PLAN.groups.map((g) => `| ${cell(g.group_id)} | ${cell(g.config_ids.join(', '))} | ${cell(g.implementation_form)} | ` +
      `${cell(g.confidence)} | ${cell(g.strategy)} |`),
    '',
    '## Results',
    '',
    '| config | status | lane validation | speedup vs stock | vs incumbent | metric | timing basis | GPU | patch |',
    '|---|---|---|---|---|---|---|---|---|',
    ...RECORDS.map((r) => `| ${cell(r.config_id)} | ${r.status} | ${cell(r.lane_validation_status)} | ${fx(r.speedup_vs_stock)} | ` +
      `n/a | ${cell(r.metric_kind)} | ${cell(r.timing_basis)} | ${cell(r.gpu_id)} | ${cell(r.final_patch)} |`),
    '',
    '## Selection',
    '',
    ...CATALOG.implementations.map((i) => `- ${i.config_ids[0]}: ${i.implementation_id} (lane patch ${i.artifact.path}), ` +
      `exact coverage ${i.coverage.exact.length ? JSON.stringify(i.coverage.exact[0]) : 'none (no signature supplied)'}`),
    ...CATALOG.fallback.map((f) => `- ${f.config_id}: ${f.selected} (${f.reason})`),
    '',
    '## Family aggregate',
    '',
    AGGREGATE.value != null
      ? `${fx(AGGREGATE.value)} (${AGGREGATE.kind}, ${AGGREGATE.metric}, timing basis ${AGGREGATE.timing_basis}, ${AGGREGATE.weighting}). ${AGGREGATE.reason}.`
      : `Not reported: ${AGGREGATE.reason}.`,
    `Rule: ${AGGREGATE.rule}.`,
    '',
    '## Profiling and knowledge',
    '',
    'Bottleneck summaries are not part of the lane return; see each lane\'s report.',
    '',
    ...RECORDS.map((r) => `- ${r.config_id}: report ${r.report_path || 'n/a'}; warm start ` +
      `${r.warm_start ? (r.warm_start.adopted ? `adopted (${r.warm_start.slug || 'slug n/a'})` : `not adopted (${r.warm_start.read_reason || 'n/a'})`) : 'n/a'}; ` +
      `KB write ${r.kb_written ? r.kb_written.slug || 'yes' : 'none'}; learned card ${r.learned_card ? r.learned_card.card_path : 'none'}`),
    '',
    '## Failed configs and next actions',
    '',
    ...(RECORDS.some((r) => !OK_STATUSES.includes(r.status))
      ? RECORDS.filter((r) => !OK_STATUSES.includes(r.status)).map((r) => `- ${r.config_id} (${r.status}): ` +
        `${r.error || r.reason || 'see the lane report'}. Re-run this config alone after checking ${r.eval_dir}.`)
      : ['None.']),
    '',
  ].join('\n');

  const FAMILY_RESULTS = {
    schema_version: 1, family_id: FAM.family_id, operator: FAM.operator, eval_dir: FAMILY_DIR,
    summary: SUMMARY, aggregate: AGGREGATE, configs: RECORDS,
  };
  phase('Report');
  const persisted = await persist([
    { path: PATHS.results, content: JSON.stringify(FAMILY_RESULTS, null, 2) },
    { path: PATHS.report, content: REPORT_MD },
    { path: PATHS.catalog, content: JSON.stringify(CATALOG, null, 2) },
  ], 'file_writer:family_artifacts');
  log(`config_aware COMPLETE: ${SUMMARY.optimized}/${SUMMARY.total} config(s) optimized. Results in ${FAMILY_DIR}`);
  return {
    schema_version: 1, mode: 'config_aware', config_aware: true,
    family_id: FAM.family_id, operator: FAM.operator, eval_dir: FAMILY_DIR,
    plan_path: PATHS.plan, results_path: PATHS.results, report_path: PATHS.report, catalog_path: PATHS.catalog,
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
  // Final step: render the run report (token/time/cost ledger -> clickable role-execution-tree
  // HTML + MD twin) from the lane's OWN timeline. In passthrough the dispatcher makes no agent calls
  // itself, so the lane result carries the timeline (`llm_timeline`) + its `eval_dir`. This is the
  // top-level entry for optimize/author runs, so it owns the report (E2E nests kernel_lane.js directly
  // and reports at its own top level, so there is no double-emit). NOTE: this branch runs BEFORE the
  // obj()/agentT()/LLM_STATS definitions below, so it is self-contained: raw agent() + inline schema +
  // an inline llm_stats check. Entirely best-effort — never fail a real result over accounting.
  const llmStatsOn = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
  if (llmStatsOn && r && r.eval_dir && r.llm_timeline) {
    try {
      const tlJson = JSON.stringify({ ...r.llm_timeline, instance: r.eval_dir });
      const tlPath = `${r.eval_dir}/reports/trace/agent_timeline.json`;
      await agent(
        `You are the file_writer. PHASE=persist_llm_stats.\n` +
        `Do exactly two things, in order:\n` +
        `1. Use the Write tool to create "${tlPath}" with EXACTLY the JSON below, verbatim ` +
        `(create parent directories if needed; do NOT reformat, truncate or summarize it):\n\n` +
        '```json\n' + tlJson + '\n```\n\n' +
        `2. Then run this Bash command (best-effort; if it fails, carry on and report ok=false).\n` +
        `   It runs the token/time/cost ledger AND renders the role-execution-tree report (HTML + MD):\n` +
        `   python3 -B "${WORKFLOW_DIR}/../interface/geak_report.py" --eval-dir "${r.eval_dir}"\n\n` +
        `Then return {"written": true, "path": "${tlPath}", "ok": <true if the command exited 0 else false>}.`,
        { phase: 'Bakeoff', label: 'file_writer:persist_llm_stats',
          schema: { type: 'object',
            properties: { written: { type: 'boolean' }, path: { type: 'string' }, ok: { type: 'boolean' } },
            required: [], additionalProperties: true } });
      log(`LLM token+time+cost ledger + run report (HTML+MD) -> ${r.eval_dir}/reports/trace/ and ${r.eval_dir}/report/.`);
    } catch (e) {
      log(`LLM stats/report emit failed (NON-FATAL — the run is unaffected): ${String(e)}`);
    }
  }
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
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');
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
const EXPERT_SKILL_ROLES = new Set(['op_benchmarker']);

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

// ---------------------------------------------------------------------------
// Schema helpers.
// ---------------------------------------------------------------------------
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const arrStr = { type: 'array', items: { type: 'string' } };
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };

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

// ---------------------------------------------------------------------------
// Prompt + agent helpers (self-contained; mirror kernel_lane.js / e2e_workflow.js).
// ---------------------------------------------------------------------------
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

function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also query ${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op, whose \`scope\` ` +
    `is \`kernel\` (\`tuning\` entries belong to the e2e tuning phase and match every operator), and whose ` +
    `validation_status is \`validated\`, and treat each as a HIGH-PRIOR candidate — advisory only, never ` +
    `overriding your isolated A/B vs the oracle, never reducing a result below the measured baseline.`;
}

// Read a role file from an arbitrary roles/ dir (used for e2e's op_benchmarker referenced in place).
function roleAgentFrom(dir, role, phaseName, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phaseName}.
First Read ${dir}/roles/${role}.md and follow its instructions for PHASE=${phaseName}.
Read any knowledge files it points you to under ${dir}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role);
}
// This dispatcher's own roles (oracle_freezer) live under WORKFLOW_DIR/roles/.
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
  }),
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

// Persist the agent timeline, then render the run report (ledger -> clickable role-execution-tree
// HTML + MD twin) as the very last step. kernel_workflow is always a TOP-LEVEL entry point (E2E nests
// kernel_lane.js, not this dispatcher), so it owns its own ledger + report. The nested lane timelines
// are already merged into LLM_TL.nested, so this one report covers the whole bake-off. The script has
// no filesystem access, so a tiny agent writes the timeline and runs the report driver. Entirely
// best-effort: accounting must never fail a run that produced a real speedup.
if (EVAL_DIR && LLM_STATS) {
  try {
    const tlJson = JSON.stringify({ ...LLM_TL, instance: EVAL_DIR });
    const tlPath = `${EVAL_DIR}/reports/trace/agent_timeline.json`;
    await agentT(
      `You are the file_writer. PHASE=persist_llm_stats.\n` +
      `Do exactly two things, in order:\n` +
      `1. Use the Write tool to create "${tlPath}" with EXACTLY the JSON below, verbatim ` +
      `(create parent directories if needed; do NOT reformat, truncate or summarize it):\n\n` +
      '```json\n' + tlJson + '\n```\n\n' +
      `2. Then run this Bash command (best-effort; if it fails, carry on and report ok=false).\n` +
      `   It runs the token/time/cost ledger AND renders the role-execution-tree report (HTML + MD):\n` +
      `   python3 -B "${WORKFLOW_DIR}/../interface/geak_report.py" --eval-dir "${EVAL_DIR}"\n\n` +
      `Then return {"written": true, "path": "${tlPath}", "ok": <true if the command exited 0 else false>}.`,
      { phase: 'Report', label: 'file_writer:persist_llm_stats',
        schema: obj({ written: { type: 'boolean' }, path: { type: 'string' }, ok: { type: 'boolean' } }, []) });
    const nestedNote = LLM_TL.nested.length ? ' plus ' + LLM_TL.nested.length + ' nested lane(s)' : '';
    log(`LLM token+time+cost ledger -> ${EVAL_DIR}/reports/trace/ and run report (HTML+MD) -> ` +
        `${EVAL_DIR}/report/. ${LLM_TL.events.length} agent attempts recorded${nestedNote}.`);
  } catch (e) {
    log(`LLM stats/report emit failed (NON-FATAL — the run is unaffected): ${String(e)}`);
  }
}

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
