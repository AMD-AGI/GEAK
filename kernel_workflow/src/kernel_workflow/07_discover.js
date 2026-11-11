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
const DISCOVER_INTRO = RDNA4_ISOLATE
  ? (
  'STANDALONE kernel bake-off — there is NO live server (do not try to launch or capture from one).\n' +
  'R9700 / gfx1201 ISOLATION: do NOT consult learned KB, expert skills, AITER CDNA win cards, or CK/AITER offline tune.\n' +
  'Do NOT inject MFMA / FNUZ / Instinct guidance. Discover existing impls by measuring the frozen oracle only.\n' +
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

