export const meta = {
  name: 'kernel-workflow',
  description: 'Single ENTRY POINT for kernel optimization on AMD Instinct MI-series GPUs (CDNA gfx942/gfx950, auto-detected on-box). Dispatches on args.mode: optimize/author -> delegate one unchanged single-language lane to the kernel_lane worker (backward compatible); bakeoff -> freeze the input kernel into ONE immutable oracle + frozen baseline, discover per-language existing impls + offline-tune env backends (aiter/CK), then run one worker lane per backend language (HIP/Triton/FlyDSL/CK/...) in parallel over the GPU pool and pick the fastest verified result across ALL candidates (author/optimize lanes AND the tuned env backend) — every one scored against the SAME frozen original baseline (anti-cheating). Wraps the unchanged kernel_lane worker (one workflow() nesting level; the dispatcher is the bake-off orchestrator).',
  whenToUse: 'Optimize a kernel. Three modes, all via args.mode (there is NO natural-language mode detection — the caller picks): mode=optimize (DEFAULT) speeds up an EXISTING kernel and behaves exactly like the old single-language workflow; mode=author writes a fresh implementation from scratch, then optimizes it — use it when there is no source to edit yet, or to port the op to another language (pass args.target_language); mode=bakeoff tries several backend languages in parallel and keeps the fastest (pass args.backends, or leave empty to auto-discover — leaving it empty also lets Discover decide per-language whether to optimize an existing impl or author a new one). Anything else throws. Pass args.kernel_path (required), args.workflow_dir (required), args.mode, args.target_language, args.backends, args.budget, args.gpu_ids, args.gpu_mode (pool|pin, default pool).',
  phases: [
    { title: 'Freeze',   detail: 'oracle_freezer: freeze the input kernel -> immutable oracle + baseline_src/ (the ONE denominator) [bakeoff only]' },
    { title: 'Discover', detail: 'op_benchmarker: per-language existing-impl probe + measure + OFFLINE env tune (aiter/CK, shapes from the frozen oracle) -> author_plan, best_known_ms [bakeoff only]' },
    { title: 'Bakeoff',  detail: 'one unchanged kernel_lane worker per language (optimize if an impl exists, else author), parallel over the GPU pool [bakeoff only]' },
    { title: 'Report',   detail: 'rank ALL candidates (lanes + tuned env backend) on the SAME frozen baseline; write the comparison table + winner (+ optional apply_to_original) [bakeoff only]' },
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
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/op dir)');
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_workflow.js + kernel_lane.js + roles/ + knowledge/ (i.e. the dirname of this script).');
}
// The single-language worker lane (unchanged behavior). Same dir as this dispatcher.
const WORKER = String(A.kernel_lane_script || `${WORKFLOW_DIR}/kernel_lane.js`);
const MODE = String(A.mode != null ? A.mode : 'optimize').trim().toLowerCase() || 'optimize';

// ===========================================================================
// SINGLE-LANGUAGE PASS-THROUGH (mode=optimize | author) — byte-compatible with
// the pre-dispatcher behavior. Forward EVERY arg to the worker unchanged (the
// worker ignores dispatcher-only args like `backends`). One nesting level.
// ===========================================================================
if (MODE === 'optimize' || MODE === 'author') {
  phase('Bakeoff');            // reuse a declared phase slot for the single passthrough lane
  log(`mode=${MODE}: single-language pass-through -> ${WORKER}`);
  return await workflow({ scriptPath: WORKER }, { ...A, workflow_dir: WORKFLOW_DIR });
}

// ===========================================================================
// From here on: mode=bakeoff — the multi-language bake-off orchestrator. Pass
// args.backends to pick the languages, or leave it empty to auto-discover.
// ===========================================================================
if (MODE !== 'bakeoff') {
  throw new Error(`unknown mode='${MODE}'. Use optimize | author | bakeoff.`);
}

// Sibling e2e_workflow dir — we REFERENCE its op_benchmarker role + bench scripts in place (no copy).
const E2E_WF_DIR = String(A.e2e_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/e2e_workflow')).replace(/\/+$/, '');
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');
const KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge')).replace(/\/+$/, '');
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
const USE_EXPERT_SKILLS = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
const EXPERT_SKILL_ROLES = new Set(['op_benchmarker']);

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
  isolated_speedup: { type: 'number' }, winner_editable: { type: 'boolean' },
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

// ---------------------------------------------------------------------------
// Prompt + agent helpers (self-contained; mirror kernel_lane.js / e2e_workflow.js).
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// Hung-agent + API-fault guard (same contract as kernel_lane.js:agentT).
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 3600000, 10);
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
// --- LLM stats: which agent ran, in which phase, in what order ---------------------------------
// Companion to the blocks of the same name in e2e_workflow.js / kernel_lane.js. Records only the
// association (phase <- agent), never timestamps: Date.now()/new Date() are unavailable in workflow
// scripts, and every duration in the final report comes from the Claude Code transcripts instead.
// Each bake-off lane hands its own timeline back on its return value, which we absorb below, so the
// whole tree is written once at the top. PURELY ADDITIVE: args.llm_stats="false" makes it a no-op.
const LLM_STATS = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
const LLM_TL = { schema: 'geak.agent_timeline/1', workflow: 'kernel_workflow', events: [], nested: [] };
function tlAgent(o, attempt, ok) {
  if (!LLM_STATS) return;
  LLM_TL.events.push({
    seq: LLM_TL.events.length,
    phase: (o && o.phase) || '',
    label: (o && o.label) || 'agent',
    attempt: attempt,
    ok: !!ok,
  });
}
function absorbNested(r) {
  if (LLM_STATS && r && r.llm_timeline) LLM_TL.nested.push(r.llm_timeline);
  return r;
}

async function agentT(p, o) {
  const label = (o && o.label) ? o.label : 'agent';
  for (let attempt = 1; attempt <= AGENT_RETRIES; attempt++) {
    try {
      if (typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) {
        const r0 = await agent(p, o);
        tlAgent(o, attempt, !!r0);
        return r0;
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
      tlAgent(o, attempt, !!r);
      return r;
    } catch (e) {
      tlAgent(o, attempt, false);
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
    `Also query ${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op and whose ` +
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
      OP_SPEC, WORKLOAD_SPEC_PATH, SKILL_DIR: WORKFLOW_DIR, KERNEL_KNOWLEDGE_DIR,
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
const DISCOVER_INTRO =
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
  'that need a running server). (5) DECIDE the author_plan as usual.';
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
    const r = absorbNested(await workflow({ scriptPath: WORKER }, {
      kernel_path: oracle.task_dir, workflow_dir: WORKFLOW_DIR,
      mode: l.mode, target_language: l.lang,
      op_spec: oracle.op_spec || OP_SPEC, workload_spec_path: oracle.workload_path || WORKLOAD_SPEC_PATH || '',
      budget: BUDGET, gpu_ids: gpu, gpu_mode: GPU_MODE, task: TASK, apply_to_original: 'false',
      exp_root: `${EVAL_DIR}/bakeoff/${l.key}`,
      use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
      perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
      llm_stats: LLM_STATS ? 'true' : 'false',
    }));
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
const ranked = cands.filter(c => c.speedup > 1.0).sort((a, b) => b.speedup - a.speedup);
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

return {
  mode: MODE,
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
  // This dispatcher's agents plus every bake-off lane's timeline, handed up so the caller can write
  // the whole run's token ledger once. See the LLM stats block above.
  llm_timeline: LLM_STATS ? LLM_TL : undefined,
};
