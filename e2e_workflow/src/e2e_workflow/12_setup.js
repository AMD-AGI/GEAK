// PHASE: Setup + Baseline profile + Strategize  (gated; else load carried state)
// Module A's outputs, folded into the ordinary state variables at their declarations below (Module A
// runs BEFORE Profile so the Top-N is taken on the adopted config). Off-values make each fold a no-op.
let kbSeedTput = 0;
let kbSeedOverlay = '';
const kbSeedKernels = [];
// Spread into the Strategize and ConfigSweep inputs; stays {} unless Module A measured something.
let KB_REF_INPUTS = {};

let EVAL_DIR, MODEL_NAME, BASELINE_TPUT, NOISE_BAND, curFlags, curEnv, curOverlay, profile, strategy, kernelQueue, headQueue;
// Keep the original measured seed separate from the evolving timing reference.
let BASELINE_FLAGS = null, BASELINE_ENV = null, BASELINE_OVERLAY = null;
let SETUP_CONFIG = null;   // what BASELINE_TPUT was measured on, if Setup ran here

// The envelope arrives in the reply: this script cannot read EVAL_DIR (RUNTIME SANDBOX CONTRACT).
// Every malformed reply THROWS, which arms the caller's repair_delivery retry; an empty queue would
// read as "nothing to optimize".
function readAgentXPreparation(reply) {
  const prepared = reply && reply.prepared;
  if (!prepared || typeof prepared !== 'object')
    throw new Error('reply carries no prepared envelope; run agentx_kernel_preparation.mjs with ' +
      '--emit-prepared and return its stdout verbatim');
  for (const key of ['head_candidates', 'kernel_candidates'])
    if (!Array.isArray(prepared[key])) throw new Error(`prepared.${key} must be an array`);
  // Re-assert each candidate's invariants (preparedUtDefect); a bad one THROWS rather than being
  // silently dropped.
  const defects = [];
  for (const c of [...prepared.head_candidates, ...prepared.kernel_candidates]) {
    const why = !c || !c.prepared_ut ? 'carries no prepared_ut'
      : preparedUtDefect(c, c.prepared_ut.result, c.prepared_ut.phase);
    if (why) defects.push(`${(c && c.short_name) || '(unnamed)'}: ${why}`);
  }
  if (defects.length)
    throw new Error(`${defects.length} prepared candidate(s) failed re-validation — ${defects.join('; ')}`);
  if (!prepared.kernel_extraction_delivery_json && reply.delivery_path)
    prepared.kernel_extraction_delivery_json = reply.delivery_path;
  for (const key of ['failed_tasks', 'deferred_tasks'])
    if (!Array.isArray(prepared[key])) prepared[key] = [];
  prepared.config_directions = Array.isArray(prepared.config_directions) ? prepared.config_directions : [];
  return prepared;
}

async function prepareAgentXKernels(round, throughput) {
  phase('KernelExtract');
  const inputs = {
    EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, TASK, ROUND: round,
    BASELINE_LOG: `${EVAL_DIR}/baseline/server.log`, BASELINE_BENCH_DIR: `${EVAL_DIR}/baseline`,
    BASELINE_SUMMARY: `${EVAL_DIR}/baseline_summary.md`,
    EVIDENCE_DIR: `${EVAL_DIR}/kernel_extraction/round_${round}`,
    BASELINE_THROUGHPUT: BASELINE_TPUT, CURRENT_THROUGHPUT: throughput,
    CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_OVERLAY: curOverlay,
    PREVIOUS_WORKLOAD_JSON: profile ? profile.profile_workload_json || '' : '',
    PREVIOUS_RANKING_JSON: profile ? profile.kernel_ranking_json || '' : '',
    PREVIOUS_UT_DELIVERY: profile ? profile.kernel_extraction_delivery_json || '' : '',
    EXTRACTION_SELECTION_POLICY: { phases: ['PREFILL', 'DECODE'], top_k_per_phase: 5,
      overall_pct_gt: 3, combination: 'phase_top_k_and_overall_threshold', backfill: false },
    SOURCE_DELIVERY_REQUIRED: true,
    BUDGET, HEAD_BUDGET, REMAINING_MIN: remainingMin(),
    SKILL_DIR: WORKFLOW_DIR, ...TRACELENS_INPUTS, ...ANALYSIS_SKILL_INPUTS, ...KB_REF_INPUTS,
  };
  let reply = await safeAgent(roleAgent('kernel_extractor', 'prepare',
    'Complete the kernel-extract skill from the saved baseline through executable, GPU-validated UTs. ' +
    'Selection, task grouping and bootstrap replay are INTERNAL steps: save their artifacts and continue ' +
    'to source extraction, UT authoring and validation in this invocation. Prefer fixed-seed input factories ' +
    'from saved shape/layout metadata; UT execution must not start serving or capture shapes. ' +
    'Capture only missing structural state needed for faithful cases. Do not return a candidate list ' +
    'before UTs have been attempted. Reuse compatible evidence and passing UTs. Never rerun the full ' +
    'AgentX baseline for discovery. Follow TASK scope and the default per-phase selection rule. ' +
    'Prove that edits to each opt GPU leaf affect candidate correctness while the frozen baseline passes; ' +
    'restore the clean source and rerun the UT. Keep failures and budget-deferred tasks in the delivery. ' +
    'Read references/workflow-handoff.md in the skill. Pack with ' +
    '`agentx_kernel_preparation.mjs --emit-prepared` and return ONLY {delivery_path, prepared}: ' +
    'delivery_path names that file under EVAL_DIR, and prepared is that command\'s stdout envelope ' +
    'VERBATIM — the workflow cannot open the file, so an envelope you summarise or rebuild by hand ' +
    'is a corrupted handoff. Return it after all scheduled UT work is finished. Do not optimize or ' +
    'integrate.', inputs),
    { phase: 'KernelExtract', label: `kernel_extractor:prepare ${round}`, schema: AGENTX_PREPARATION_SCHEMA });
  for (let attempt = 0; ; attempt++) {
    try {
      if (!reply?.delivery_path) throw new Error('completed skill did not return delivery_path');
      const prepared = await readAgentXPreparation(reply);
      log(`Kernel extraction ready: ${prepared.head_candidates.length + prepared.kernel_candidates.length} validated UTs, ` +
        `${prepared.failed_tasks.length} failed, ${prepared.deferred_tasks.length} deferred.`);
      return prepared;
    } catch (error) {
      if (attempt >= 1) throw new Error(`AgentX UT delivery unreadable: ${error.message}`);
      reply = await safeAgent(roleAgent('kernel_extractor', 'repair_delivery',
        'Repair only the completed-stage handoff using saved files. Do not repeat selection, captures or ' +
        'passing GPU tests. Preserve every real failure and all task results. The reader failed with: ' +
        error.message + '. Re-pack with `agentx_kernel_preparation.mjs --emit-prepared` and return ' +
        '{delivery_path, prepared} for the repaired file, prepared being that stdout verbatim.',
        { ...inputs, PREVIOUS_DELIVERY_PATH: reply?.delivery_path || '' }),
        { phase: 'KernelExtract', label: `kernel_extractor:repair delivery ${round}`, schema: AGENTX_PREPARATION_SCHEMA });
    }
  }
}

if (want('setup')) {
  phase('Setup');
  const setup = await safeAgent(
    roleAgent('director', 'setup', 'Build the isolated e2e eval dir and record the baseline throughput.', {
      LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT, EVAL_DIR_OVERRIDE, MODEL_NAME_HINT, TASK,
      GPU_IDS, WORKLOAD, INIT_FLAGS, INIT_ENV, INIT_BASE_OVERLAY,
      MEASUREMENT_PURPOSE: 'parity', REPLICAS: PARITY_REPLICAS,
      SKILL_DIR: WORKFLOW_DIR, EXPECTED_GFX, EXPECTED_TARGET,
      EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
      // Only on a declared trace replay: the Director writes it to $EVAL_DIR/bench_env.sh before
      // the baseline bench, so the baseline uses the real client and metric axis.
      ...(AGENTX_ENV ? { BENCH_ENV_CONTENT: AGENTX_ENV } : {}),
    }),
    { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
  if (!setup || !setup.eval_dir) throw new Error('Setup failed: no eval_dir');
  const detectedGfx = String(setup.gfx || '').trim().toLowerCase();
  const detectedTarget = String(setup.device_target || '').trim().toLowerCase();
  const detectedPhysicalCuCount = Number(setup.physical_cu_count);
  if (!detectedGfx || !detectedTarget) {
    throw new Error('Setup failed: Director did not return structured gfx and device_target identity');
  }
  if (!Number.isFinite(detectedPhysicalCuCount) || detectedPhysicalCuCount <= 0) {
    throw new Error('Setup failed: Director did not return a positive physical_cu_count');
  }
  if (EXPECTED_GFX && detectedGfx !== EXPECTED_GFX) {
    throw new Error(`GPU architecture mismatch: expected ${EXPECTED_GFX}, detected ${detectedGfx}`);
  }
  if (EXPECTED_TARGET && EXPECTED_TARGET !== 'unknown' && detectedTarget !== EXPECTED_TARGET) {
    throw new Error(`GPU product mismatch: expected ${EXPECTED_TARGET}, detected ${detectedTarget}`);
  }
  const normalizeDeviceName = (name) =>
    String(name || '').trim().replace(/\s+/g, ' ').toLowerCase();
  if (EXPECTED_DEVICE_NAME &&
      normalizeDeviceName(setup.device_name) !== normalizeDeviceName(EXPECTED_DEVICE_NAME)) {
    log(`Setup identity telemetry differs from the deterministic probe: device_name expected ` +
        `${EXPECTED_DEVICE_NAME}, Director returned ` +
        `${String(setup.device_name || '').trim() || 'unknown'}; using expected value.`);
  }
  if (EXPECTED_PHYSICAL_CU_COUNT > 0 &&
      detectedPhysicalCuCount !== EXPECTED_PHYSICAL_CU_COUNT) {
    log(`Setup identity telemetry differs from the deterministic probe: physical_cu_count expected ` +
        `${EXPECTED_PHYSICAL_CU_COUNT}, Director returned ${detectedPhysicalCuCount}; using expected value.`);
  }
  const divergence = evalDirDivergence(EVAL_DIR_OVERRIDE, setup.eval_dir);
  if (divergence) throw new Error(`Setup failed: ${divergence}`);
  EVAL_DIR = setup.eval_dir;
  MODEL_NAME = setup.model_name || MODEL_NAME_HINT;
  BASELINE_TPUT = setup.baseline_throughput_tok_s;
  NOISE_BAND = gateNoiseBand(setup.noise_band_pct);
  // Seed flags/env win when provided (baseline was measured on them); else fall
  // back to whatever the director resolved.
  curFlags = INIT_FLAGS || (setup.server_flags && setup.server_flags.extra) || '';
  curEnv = INIT_ENV || (setup.server_env || '');
  curOverlay = INIT_BASE_OVERLAY;
  SETUP_CONFIG = { flags: curFlags, env: curEnv, overlay: curOverlay };
  // The Director may resolve launch defaults beyond INIT_*; retain its complete measured config.
  BASELINE_FLAGS = typeof (setup.server_flags && setup.server_flags.extra) === 'string'
    ? setup.server_flags.extra : (INIT_FLAGS || null);
  BASELINE_ENV = typeof setup.server_env === 'string' ? setup.server_env : (INIT_ENV || null);
  BASELINE_OVERLAY = curOverlay;
  log(`Setup done. EVAL_DIR=${EVAL_DIR}, baseline ${BASELINE_TPUT} tok/s (noise band ${NOISE_BAND}%)`);
  if (IS_AGENTX) {
    log(`Setup: AgentX graded on metric_basis=${AGENTX_METRIC_BASIS} (E2E_METRIC=${AGENTX_E2E_METRIC}, ` +
      `from ${AGENTX_METRIC_SOURCE}), concurrency ${CONC}, noise band ${NOISE_BAND}% ` +
      `(Director reported ${setup.noise_band_pct != null ? setup.noise_band_pct : 'none'}%, ` +
      `workload floor ${NOISE_BAND_DEFAULT}%).`);
  }

  // ── The trace-replay declaration must have TAKEN EFFECT, not merely been sent ──
  // BASELINE_TPUT is the denominator of every reported gain, and the declaration travels through an
  // agent, so confirm bench_env.sh exists and the baseline summary carries the requested metric
  // basis. A mismatch aborts now; an unreadable summary warns and continues.
  if (IS_AGENTX) {
    let verdict = null;
    try {
      const { execFileSync } = require('child_process');
      const out = execFileSync('python3', ['-c', `
import json, os, sys
eval_dir = sys.argv[1]
want = sys.argv[2]
env_file = os.path.join(eval_dir, 'bench_env.sh')
summary = os.path.join(eval_dir, 'baseline', 'bench_summary.json')
res = {'env_file': os.path.isfile(env_file) and os.path.getsize(env_file) > 0}
def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
try:
    with open(summary) as fh:
        _s = json.load(fh) or {}
    res['basis'] = _s.get('metric_basis') or ''
    # The request shape the baseline actually served. Reading it back is what
    # lets kernel work target the served regime without this file, or the
    # workflow, hardcoding a corpus average that would go stale.
    res['observed_isl'] = _num(_s.get('observed_isl'))
    res['observed_osl'] = _num(_s.get('observed_osl'))
except Exception as exc:
    res['basis'] = None
    res['why'] = f'{type(exc).__name__}: {exc}'
res['want'] = want
print(json.dumps(res))
`, EVAL_DIR, AGENTX_METRIC_BASIS], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 60000 });
      verdict = JSON.parse(out);
    } catch (probeErr) {
      log(`Setup: WARNING could not verify the trace-replay declaration ` +
        `(${probeErr && probeErr.message ? probeErr.message : probeErr}). ` +
        `Proceeding, but confirm ${EVAL_DIR}/bench_env.sh and the baseline metric_basis by hand.`);
    }
    if (verdict) {
      if (!verdict.env_file) {
        throw new Error(
          `Setup declared workload_kind=${WORKLOAD_KIND} but ${EVAL_DIR}/bench_env.sh was never ` +
          `written, so the baseline was measured with the synthetic client on the wrong metric axis. ` +
          `Every later delta would be relative to the wrong number. Re-run Setup.`);
      }
      if (verdict.basis === null) {
        log(`Setup: WARNING baseline/bench_summary.json unreadable (${verdict.why || 'unknown'}); ` +
          `cannot confirm metric_basis=${AGENTX_METRIC_BASIS}.`);
      } else if (verdict.basis !== AGENTX_METRIC_BASIS) {
        throw new Error(
          `Setup measured the baseline on metric_basis=${verdict.basis || '<absent>'} but this run ` +
          `declared ${AGENTX_METRIC_BASIS}. The two axes are not comparable, so the baseline is ` +
          `unusable. Check that ${EVAL_DIR}/bench_env.sh sets E2E_METRIC=${AGENTX_E2E_METRIC} and ` +
          `that the bench did not inherit another E2E_METRIC (an exported value outranks the ` +
          `file), then re-run Setup.`);
      } else {
        log(`Setup: trace-replay declaration verified (bench_env.sh present, ` +
          `baseline metric_basis=${verdict.basis}).`);
      }
      // Take the kernel-targeting shape from what the baseline served, unless the caller declared
      // one explicitly.
      if (SHAPE_IS_MEASURED) {
        log(`Setup: keeping the declared kernel-targeting shape isl=${ISL} osl=${OSL} ` +
          `(provenance ${WORKLOAD_SHAPE_PROVENANCE}); the baseline served ` +
          `isl=${verdict.observed_isl || '?'} osl=${verdict.observed_osl || '?'}.`);
      } else if (!adoptMeasuredShape(verdict.observed_isl, verdict.observed_osl,
                                     'the baseline bench')) {
        log(`Setup: the baseline summary carried no served request shape yet, so kernel ` +
          `targeting is still the synthetic ${ISL}/${OSL} placeholder — far below a trace ` +
          `replay's real lengths. This is expected when a relaunched leg has not written its ` +
          `summary yet: every later role prompt retries the read and installs the shape as ` +
          `soon as it appears (watch for "Kernel-targeting shape measured"). If that line ` +
          `never comes, kernel choices are aimed at the wrong regime (the MEASUREMENT is ` +
          `unaffected) — pass workload_spec.observed_isl/observed_osl to set it explicitly.`);
      }
    }
  }

  // =========================================================================
  // MODULE A — read the deployment KB, then VALIDATE what it says on this box.
  // After Setup (needs EVAL_DIR, bench_e2e.sh, BASELINE_TPUT, NOISE_BAND), before Profile (Top-N on
  // the adopted config), inside want('setup') (a phase-partial resume cold-starts). A stored speedup
  // counts only once this run re-measures it through the same gate.
  if (E2E_WARM_START_ON) {
    phase('WarmStart');
    KB_DIMS = {
      model: MODEL_NAME,
      gfx: String(setup.gfx || '').trim(),
      framework_version: String(setup.framework_version || A.kb_framework_version || '').trim(),
      precision: String(setup.precision || '').trim(),
      rocm_version: String(setup.rocm_version || '').trim(),
    };
    if (!KB_DIMS.gfx) {
      // Refuse to read rather than read the wrong page: an arch-less identity resolves to the
      // `gfx=unknown` rung, measured on unknown hardware.
      log('[kb] warm start skipped: read_reason=missing_arch (the Director reported no gfx; a ' +
        'cross-arch config is not a candidate, it is a guess).');
      // Still reported: "did not look, and why" differs from "looked and found nothing".
      KB_RECALL.e2e = { read_reason: 'missing_arch', tried: [], answered: '', match_tier: '',
        plane: E2E_KB_PLANE, mode: E2E_WARM_START, candidates: 0, configs: [], kernels: [] };
    } else {
      const refsDir = `${EVAL_DIR}/kb_references`;
      const cacheDir = `${EVAL_DIR}/kb_cache`;
      const KB_IDENTITY_FILE = `${EVAL_DIR}/${KB_IDENTITY_BASENAME}`;
      const resolved = await safeAgent(
        `You are the e2e warm-start resolver. Run EXACTLY this command and return its JSON stdout ` +
        `verbatim as StructuredOutput — do not add, drop, reorder, or reinterpret any field. The ` +
        `command pretty-prints its JSON over several lines; return the whole object, not the first ` +
        `line. A non-zero exit or an empty candidate list is a VALID answer (the page has never been ` +
        `written): return what it printed, do not retry with different flags, and do not invent ` +
        `candidates. It may consult the shared KB Store service first and fall back to the on-disk ` +
        `store by itself; run it as one script and do not split it into separate commands.\n` +
        '```bash\n' +
        kbResolveScript(
          `--top-n ${E2E_WARM_START_TOP_N} \\\n` +
          `  --min-speedup ${E2E_WARM_START_MIN_SPEEDUP} \\\n` +
          `  --refs-dir ${shq(refsDir)} --cache-dir ${shq(cacheDir)} \\\n` +
          // The read also leaves this run's KB address on disk, so run_e2e.py can salvage results
          // onto the right pages if this process dies before Module B. --store/--identity-plane are
          // only recorded (so a `both` run's salvage keeps its local mirror).
          `  --identity-out ${shq(KB_IDENTITY_FILE)} \\\n` +
          `  --store ${shq(E2E_KB_STORE_DIR)} --identity-plane ${E2E_KB_PLANE}`) + '\n' +
        '```',
        { phase: 'WarmStart', label: 'warm_start:resolve', schema: KB_RESOLVE_SCHEMA }) || {};
      const cands = Array.isArray(resolved.candidates) ? resolved.candidates : [];
      // Log the ladder VERBATIM: it is the only record of which address was asked, so identity drift
      // isn't mistaken for an empty store. `read_plane` is the plane that ANSWERED (`plane` if absent).
      KB_READ_PLANE = String(resolved.read_plane || resolved.plane || '');
      log(`[kb] e2e read: plane=${KB_READ_PLANE || '?'} tried=[${(resolved.tried || []).join(' | ')}] ` +
        `answered=${resolved.canonical_id || '?'} tier=${resolved.match_tier || '-'} ` +
        `sorted_by=${resolved.sorted_by || resolved.ranked_by || '-'} ` +
        `champion_metric=${resolved.champion_metric || '-'} reason=${resolved.read_reason || '?'} ` +
        `candidates=${cands.length}`);
      log(`[kb] recall_metric_basis=${E2E_KB_METRIC.basis || 'any'} ` +
        `strict=${E2E_KB_METRIC.strict} source=${E2E_KB_METRIC.source} ` +
        `measurement_metric_basis=${kbMeasurementBasis()}`);
      // The counts BEHIND a zero: nobody wrote this page, everything was retracted, everything was
      // under the floor — all `candidates=0` above. Logged separately so that line stays greppable.
      const curation = (resolved.curation && typeof resolved.curation === 'object') ? resolved.curation : {};
      if (!cands.length && Number(curation.scanned) > 0) {
        log(`[kb] the page was NOT empty: ${curation.scanned} record(s) scanned on plane ` +
          `'${curation.read_plane || KB_READ_PLANE || '?'}' at ` +
          `'${curation.canonical_id || resolved.canonical_id || '?'}' (tier ${curation.tier || '-'}), ` +
          `all curated away — retired=${curation.retired || 0} ` +
          `same_direction_collapsed=${curation.same_direction_collapsed || 0} ` +
          `below_min_speedup=${curation.below_min_speedup || 0} (floor ${curation.min_speedup}). ` +
          'This is a FLOOR result, not a cold deployment.');
      }
      // Record the ask before anything is benched, so a mid-warm-start death still reports which
      // ladder was queried.
      KB_RECALL.e2e = {
        read_reason: String(resolved.read_reason || ''),
        tried: Array.isArray(resolved.tried) ? resolved.tried : [],
        answered: String(resolved.canonical_id || ''),
        match_tier: String(resolved.match_tier || ''),
        plane: E2E_KB_PLANE, read_plane: KB_READ_PLANE, mode: E2E_WARM_START,
        recall_metric_basis: E2E_KB_METRIC.basis,
        recall_metric_strict: E2E_KB_METRIC.strict, recall_metric_source: E2E_KB_METRIC.source,
        measurement_metric_basis: kbMeasurementBasis(),
        candidates: cands.length, configs: [], kernels: [],
        // Both floors, always: what this run declined to see, and at what threshold.
        read_min_speedup: E2E_WARM_START_MIN_SPEEDUP,
        bench_min_speedup: E2E_WARM_START_BENCH_MIN_SPEEDUP,
        curation,
      };
      if (cands.length) KB_REF_DIR = refsDir;   // arms warmStartBlock() for the consumer roles
      // Armed on the same condition: a path to an empty cache reads to an agent as "the file is
      // missing" rather than "there was no record".
      if (cands.length) KB_CACHE_DIR = cacheDir;

      // How many offers are worth a server launch; a coarse rung gets its own smaller budget
      // (E2E_WARM_START_VALIDATE_N_COARSE), not zero.
      const exactTier = (resolved.match_tier || '') === 'exact';
      const benchN = E2E_WARM_START_REF_ONLY ? 0
        : exactTier ? E2E_WARM_START_VALIDATE_N
        : A.e2e_warm_start_validate_n != null ? E2E_WARM_START_VALIDATE_N
        : E2E_WARM_START_VALIDATE_N_COARSE;
      // WHICH offers: on a coarse rung the stored throughput was measured elsewhere, so re-order by
      // speedup (the rung's own metric). The top-N itself was still fetched by throughput.
      const benchOrder = exactTier ? cands : [...cands].sort(
        (x, y) => (Number(y.speedup) || 0) - (Number(x.speedup) || 0));
      // The bench floor: a record under it stays an offered reference and loses only the launch.
      // A record with no stored speedup stays benchable.
      const aboveBenchFloor = (c) =>
        c.speedup == null || Number(c.speedup) >= E2E_WARM_START_BENCH_MIN_SPEEDUP;
      const benchable = benchOrder.filter(aboveBenchFloor);
      const belowBenchFloor = benchN ? benchOrder.filter(c => !aboveBenchFloor(c)) : [];
      if (belowBenchFloor.length) {
        log(`[kb] ${belowBenchFloor.length} offer(s) stay references, not benched: stored speedup ` +
          `below the bench floor of ${E2E_WARM_START_BENCH_MIN_SPEEDUP}x — ` +
          belowBenchFloor.map(c => `${String(c.session_id || '?').slice(-12)}=${c.speedup}x`).join(' ') +
          '. A claim that size is this box\'s own drift; the config and any tuned artifact it ' +
          'carries are still offered to the roles that can use them without a launch.');
      }
      if (cands.length && !benchN) {
        log(`[kb] not benching: ${E2E_WARM_START_REF_ONLY
          ? (FAST_MODE ? 'fast mode — all optimization comes from the head track' : 'warm_start=reference')
          : `the caller set e2e_warm_start_validate_n_coarse=0 and match tier ` +
            `'${resolved.match_tier}' is not exact`}. The offers stay as references.`);
      } else if (benchable.length && !exactTier) {
        log(`[kb] benching ${Math.min(benchN, benchable.length)} of ${benchable.length} offer(s) from coarse ` +
          `tier '${resolved.match_tier}', picked by stored speedup. The stored throughput is not ` +
          `comparable to this baseline; the A/B below is measured on this box and is what counts.`);
      }

      // Seeded with the records the bench floor declined; `skipped` is never attested (not tried).
      const verdicts = belowBenchFloor.map(c => ({
        ...c, measured_tok_s: null, delta_pct: null, parity: 'n/a', outcome: 'skipped',
        why: `stored speedup ${c.speedup}x is below the bench floor of ` +
          `${E2E_WARM_START_BENCH_MIN_SPEEDUP}x — offered as a reference, not measured`,
      }));
      for (const c of benchable.slice(0, benchN)) {
        if (searchClosed(`[kb] replay of config ${c.session_id || '?'}`)) {
          verdicts.push({ ...c, measured_tok_s: null, delta_pct: null, parity: 'n/a',
            outcome: 'skipped', why: 'the final reserve began before its turn' });
          continue;
        }
        // VALIDATE THROUGH THE ORIGINAL GATE (config_tuner:sweep), so a stored config and a fresh
        // one are judged by identical evidence.
        const stored = (c.accepted_config && typeof c.accepted_config === 'object') ? c.accepted_config : {};
        const storedFlags = String(stored.flags || '');
        const storedEnv = String(stored.env || '');
        if (!storedFlags && !storedEnv) {
          verdicts.push({ ...c, measured_tok_s: null, delta_pct: null, parity: 'n/a',
            outcome: 'skipped', why: 'the record carries no config to apply' });
          continue;
        }
        // Key-by-key merge (mergeConfig): the recorded value wins every collision.
        const mf = mergeFlags(curFlags, storedFlags);
        const me = mergeEnv(curEnv, storedEnv);
        const overrides = mf.overrides.concat(me.overrides);
        if (overrides.length) {
          log(`[kb] merge ${c.session_id || '?'}: the recorded config overrides ${overrides.length} ` +
            `knob(s) this run's baseline already pins — ${describeOverrides(overrides)}. ` +
            'The recorded value is the one under test, so it wins.');
        }
        // Nothing to test: the baseline already IS the recorded configuration (a bench would file a
        // ~0% self-comparison as a loss). Left un-attested.
        if (mf.merged === renderFlagItems(parseFlagString(curFlags)) &&
            me.merged === renderEnvItems(parseEnvString(curEnv))) {
          verdicts.push({ ...c, measured_tok_s: null, delta_pct: null, parity: 'n/a',
            outcome: 'skipped',
            why: "this run's baseline already carries the whole recorded configuration" });
          continue;
        }
        const runValidation = (brief, label) => safeAgent(
          roleAgent('config_tuner', 'sweep', brief,
            {
              EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, BASELINE_THROUGHPUT: BASELINE_TPUT,
              NOISE_BAND_PCT: NOISE_BAND,
              CONFIG_DIRECTIONS: [{
                rank: 1,
                direction: 'kb_warm_start:' + (c.direction || 'unlabeled'),
                axis: 'compound (recovered configuration — do not split)',
                // ALREADY MERGED with this run's baseline. Use verbatim; do not re-combine with
                // CURRENT_FLAGS/CURRENT_ENV, or the collisions the merge just resolved come back.
                flags: mf.merged, env: me.merged,
                rationale: `Recorded under ${c.canonical_id || resolved.canonical_id} (session ` +
                  `${c.session_id || '?'}), where it measured ${c.throughput_tok_s != null ? c.throughput_tok_s : '?'}` +
                  ` tok/s (${c.speedup != null ? c.speedup + 'x' : 'speedup not recorded'}) against a ` +
                  `baseline of ${c.baseline_throughput_tok_s != null ? c.baseline_throughput_tok_s : '?'} tok/s. ` +
                  'That number is a HYPOTHESIS about this box, not a measurement of it.' +
                  (overrides.length
                    ? ` NOTE: ${overrides.length} knob(s) in it disagree with this run's baseline and ` +
                      `the recorded value was taken — ${describeOverrides(overrides)}. If the result is ` +
                      'neutral, check these first: they are where the recorded config and this box ' +
                      'were asked to be two different things.'
                    : ''),
              }],
              MERGE_OVERRIDES: overrides, MERGE_ADDED: mf.added.concat(me.added),
              CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_OVERLAY: curOverlay,
              MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
              SKILL_DIR: WORKFLOW_DIR,
            }),
          { phase: 'WarmStart', label, schema: SWEEP_SCHEMA });

        let sweep = await runValidation(
          'Validate ONE historical configuration recovered from the knowledge base. Treat it exactly ' +
          'as you would a fresh direction: same isolated-server measurement, same parity check, same ' +
          'swap-took-effect verification. THREE deviations from your role file, all deliberate:\n' +
          '(1) Do NOT decompose this direction into one-axis-at-a-time trials. A stored config is an ' +
          'ALREADY-COMPOUNDED whole that was accepted together on another box; benching its knobs ' +
          'separately measures a configuration nobody has ever run, and the parts can each be ' +
          'neutral while the whole is a win (or the reverse). Apply all of it, once, as a single ' +
          'trial.\n' +
          "(2) The direction's `flags`/`env` are ALREADY MERGED with this run's baseline, key by key. " +
          'Pass them VERBATIM to bench_e2e.sh. Do NOT append CURRENT_FLAGS/CURRENT_ENV to them — those ' +
          'are shown to you only so you can see what was overridden, and re-combining them puts the ' +
          'duplicate keys back and hands the outcome to argparse last-wins.\n' +
          '(3) Verify the swap TOOK EFFECT before you believe a null result. This config came from a ' +
          'different framework_version: a flag that was renamed or removed upstream is accepted ' +
          'silently on the command line and then ignored, which is indistinguishable from "the ' +
          'config made no difference". Grep the server log for each flag/env actually being ' +
          'honoured, and if one is not, say so in the trial notes rather than reporting a clean no-op. ' +
          'MERGE_OVERRIDES lists the knobs where the record and this box disagreed — check those in ' +
          'the log first.',
          `warm_start:validate:${c.session_id || 'cand'}`);

        // ONE repair attempt, only when the server produced no number: the tuner classifies the
        // launch failure (a stale record vs a conflict with this baseline), adapts the minimum, and
        // benches once more through the same accept gate.
        let repair = null;
        if (!measuredTrialOf(sweep, BASELINE_TPUT).measured &&
            !searchClosed(`[kb] repair of config ${c.session_id || '?'}`)) {
          const repaired = await runValidation(
            'A recovered configuration was applied to this box and the server produced NO ' +
            'measurement — it failed to launch, failed to become healthy, or died before the bench. ' +
            'This is the ONE repair attempt; there is no second.\n\n' +
            'Read your own launch log from the attempt you just made. Then, for each knob that came ' +
            'from the record (MERGE_OVERRIDES and MERGE_ADDED name them), classify it:\n' +
            '  - `upstream_gone`: this build does not have the flag/env at all — argparse rejected ' +
            'it, or it is absent from `--help`. This is the RECORD being stale.\n' +
            "  - `conflicts_with_baseline`: the build has it, but the value cannot hold together with " +
            "something this run's baseline pins (a length above the served context, a memory " +
            'fraction the rest of the config cannot fit under, a backend the baseline excludes). ' +
            'This is the PAIRING failing, and says nothing about the record.\n' +
            '  - `env_unsupported`: the build ignores or rejects the env var.\n' +
            'Then drop or adapt the MINIMUM needed to get a running server — keep as much of the ' +
            'recorded configuration as will run, and never drop a knob you did not have to — and ' +
            'bench that once, exactly as before.\n\n' +
            'Return the same JSON, plus a `repair` object: `{"attempted": true, "classification": ' +
            '"upstream_gone|conflicts_with_baseline|env_unsupported|mixed|unknown", ' +
            '"dropped_flags": ["--flag", ...], "reason_by_flag": {"--flag": "why"}, ' +
            '"launch_error": "<the line that actually killed it>"}`. If the server still will not ' +
            'come up, say so and return `best_throughput_tok_s: 0` — an honest "cannot run here" is ' +
            'worth more than a number from a configuration that is no longer the record.',
            `warm_start:repair:${c.session_id || 'cand'}`);
          if (repaired) {
            repair = (repaired.repair && typeof repaired.repair === 'object') ? repaired.repair : {};
            sweep = repaired;
          }
          log(`[kb] repair ${c.session_id || '?'}: ` + (repair
            ? `classified ${repair.classification || 'unknown'}` +
              `${(repair.dropped_flags || []).length ? `, dropped ${(repair.dropped_flags || []).join(' ')}` : ''}` +
              `${measuredTrialOf(sweep, BASELINE_TPUT).measured ? ', server came up' : ', still no measurement'}`
            : 'the repair attempt itself produced nothing'));
        }
        const { trial, measured } = measuredTrialOf(sweep, BASELINE_TPUT);
        // The role's own drop list, else what was asked for minus what ran.
        const dropped = (repair && Array.isArray(repair.dropped_flags) && repair.dropped_flags.length)
          ? repair.dropped_flags
          : (repair ? droppedByDiff(mf.merged, me.merged,
            String(trial.change || [trial.flags, trial.env].filter(Boolean).join(' | '))) : []);
        const parity = String(trial.parity || trial.output_parity || '');
        const deltaPct = BASELINE_TPUT ? ((measured - BASELINE_TPUT) / BASELINE_TPUT) * 100 : 0;
        // Both the tuner's judgement AND our arithmetic: the ratio is re-derived from the raw number
        // rather than trusting `kept: true`.
        const accept = trial.kept === true && measured > BASELINE_TPUT &&
          deltaPct > NOISE_BAND && parity !== 'fail';
        if (accept) {
          curFlags = sweep.accepted_flags || mf.merged || curFlags;
          curEnv = sweep.accepted_env || me.merged || curEnv;
          kbSeedTput = measured;
          await requireE2EValidationCheckpoint('config/e2e_validation.json', {
            phase: 'WarmStart', validation_level: 'config_sweep', gate: 'accepted',
            validation_status: 'accepted_config',
            baseline_throughput_tok_s: BASELINE_TPUT, final_throughput_tok_s: measured,
            throughput_speedup: BASELINE_TPUT ? measured / BASELINE_TPUT : 1,
            baseline_config: { flags: INIT_FLAGS, env: INIT_ENV },
            accepted_config: { flags: curFlags, env: curEnv, effective_config_digest: EFFECTIVE_CONFIG_DIGEST },
            accepted_kernels: [], accepted_heads: [], final_patch: [],
            final_overlay: { path: curOverlay },
            final_launch_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
            bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
            measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
              workload: WORKLOAD, source_artifact: `${EVAL_DIR}/config/sweep_results.json`,
              acceptance_source: 'kb_warm_start',
              acceptance: { gain_exceeds_noise: true, correctness_passed: true, noise_floor_pct: NOISE_BAND } },
            stack: { config_layers: (sweep.trials || []).filter((t) => t && t.kept === true) },
            replay: { asset_paths: [`${EVAL_DIR}/bench_e2e.sh`, `${EVAL_DIR}/config/sweep_results.json`] },
            integrity: { checkpoint_assets: [] },
          });
          log(`[kb] ADOPTED ${c.session_id || '?'} (${c.direction || 'unlabeled'}): ` +
            `${measured} tok/s, +${deltaPct.toFixed(2)}% vs baseline ${BASELINE_TPUT} (noise band ${NOISE_BAND}%)` +
            `${dropped.length ? `, with ${dropped.join(' ')} dropped to make it run here` : ''}.`);
        }
        // FOUR outcomes (see KB_ATTEST_OUTCOME), counted apart by kb/attest.py; only a loss and a
        // no-effect count against the record, as evidence toward a threshold. The role file spells
        // the per-trial text `note`.
        const notes = String(trial.notes || trial.note || (sweep && sweep.summary) || '');
        const inert = /\b(?:not (?:honou?red|recognized|recognised|applied|supported)|unrecognized|unrecognised|ignored|no such option|unknown (?:option|argument|flag)|renamed|removed upstream|did not take effect)\b/i.test(notes);
        // The repair's structured classification outranks the free-text regex.
        const conflicted = String((repair && repair.classification) || '') === 'conflicts_with_baseline';
        const outcome = accept ? 'adopted'
          : !measured ? (conflicted ? 'inapplicable' : 'not_reproduced')
            : inert ? 'not_reproduced' : 'rejected';
        if (!accept) {
          log(`[kb] ${{ not_reproduced: 'NOT REPRODUCED', inapplicable: 'INAPPLICABLE HERE' }[outcome] || 'rejected'} ` +
            `${c.session_id || '?'} (${c.direction || 'unlabeled'}): ` +
            `measured ${measured || 'n/a'} tok/s vs baseline ${BASELINE_TPUT}` +
            `${measured ? ` (${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(2)}%)` : ''}` +
            `${parity ? `, parity=${parity}` : ''}${inert ? ', the config never took effect' : ''}` +
            `${outcome === 'inapplicable' ? ", it collides with this run's baseline" : ''}` +
            ` — kept as a reference, not applied. This is this box's result, not a verdict on the ` +
            `record: it is filed as evidence toward a threshold, and retires nothing by itself.`);
        }
        verdicts.push({ ...c, measured_tok_s: measured || null, delta_pct: measured ? deltaPct : null,
          parity: parity || 'unknown', outcome, why: notes,
          // What was actually put on the box vs what the record asked for.
          overrides, applied_partial: dropped.length > 0, dropped_flags: dropped,
          repair_classification: String((repair && repair.classification) || '') });
        if (accept) break;   // adopt the first that passes; the rest stay references
      }
      // ---- Replay the stored KERNELS through the ordinary integrate gate ----
      // Same machinery as the Milestone track (runIntegrateBothLegs, INTEGRATE_SCHEMA, the isolated
      // A/B with parity probe, integAccepted). A candidate lives in its own overlay directory, active
      // only on PYTHONPATH, so rejecting one is simply not adopting it.
      const kbKernels = [];
      const seenKernel = new Set();
      for (const c of cands) {
        const bundlePath = (c.bundle && c.bundle.path) || '';
        // The record's prebuilt overlay (`overlay.tar.gz` from e2e_store._pack_overlay), found via
        // the bundle manifest (no filesystem here); without it the entry stays a reference.
        const bundleFiles = Array.isArray(c.bundle && c.bundle.files) ? c.bundle.files : [];
        const overlayTar = (bundlePath && bundleFiles.includes('overlay.tar.gz'))
          ? `${bundlePath}/files/overlay.tar.gz` : '';
        for (const k of (Array.isArray(c.accepted_kernels) ? c.accepted_kernels : [])) {
          const name = String((k && (k.name || k.short_name)) || '').trim();
          if (!name || seenKernel.has(name)) continue;   // one op recorded by several runs is one op
          seenKernel.add(name);
          kbKernels.push({ ...k, name, bundle: bundlePath, overlay_tar: overlayTar,
            kb_session_id: c.session_id || '' });
        }
      }
      let replayed = 0;
      // One overlay tarball is the whole RECORD's final overlay, so it is benched ONCE and the
      // record's other kernels are marked as covered by it.
      const overlayReplayed = new Set();
      const kernelVerdicts = [];
      for (const k of kbKernels) {
        const kind = String(k.winner_kind || k.kind || '').trim().toLowerCase();
        // `patch` lives in the record's bundle (`kernels/<name>.patch`); with no bundle there is no
        // file to hand the integrator.
        const patchPath = (kind === 'patch' && k.patch && k.bundle)
          ? `${k.bundle}/files/${k.patch}` : '';
        // An env/flag win is replayable only if the record names the env or flag that routed the op;
        // otherwise the A/B would compare two identical configurations.
        const hasRouting = !!(String(k.apply_env || '').trim() || String(k.apply_flags || '').trim());
        // A kind this lane cannot rebuild from a record (e.g. `authored`, whose workspace is gone) can
        // still be replayed from the record's built overlay; its manifest names the rebind seams, so
        // the A/B shows whether they bind here.
        const viaOverlay = !E2E_REPLAYABLE_KINDS.has(kind) && !!k.overlay_tar;
        const unreplayable =
          E2E_WARM_START_REF_ONLY ? 'read-only mode'
          : (!E2E_REPLAYABLE_KINDS.has(kind) && !k.overlay_tar)
            ? `winner_kind '${kind || 'unrecorded'}' cannot be rebuilt from a record, and this record ships no overlay.tar.gz to replay instead`
          : (viaOverlay && overlayReplayed.has(k.bundle))
            ? "already covered by this record's overlay replay — one tarball is the whole record's overlay, not one kernel's"
          : (!viaOverlay && kind === 'patch' && !patchPath) ? 'the record names a patch but its bundle holds no file'
          : (!viaOverlay && kind !== 'patch' && !hasRouting) ? `recorded as a '${kind}' win but carries no apply_env/apply_flags, so there is nothing to re-apply`
          : replayed >= E2E_WARM_START_KERNELS_N ? `replay budget of ${E2E_WARM_START_KERNELS_N} already spent`
          : searchClosed(`[kb] replay of kernel ${k.name}`) ? 'the final reserve began before its turn'
          : '';
        if (unreplayable) {
          kernelVerdicts.push({ ...k, kind, outcome: 'reference', measured_delta_pct: null,
            why: unreplayable });
          continue;
        }
        replayed++;
        if (viaOverlay) overlayReplayed.add(k.bundle);
        const isolated = Number(k.isolated_speedup) || 0;
        const kbIntegrateInputs = {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
          KERNEL_RESULT: {
            short_name: k.name, winner_kind: kind,
            // Both patch spellings: the two tracks read different ones.
            code_patch: patchPath, final_patch: patchPath,
            apply_env: String(k.apply_env || ''), apply_flags: String(k.apply_flags || ''),
            target_callable: String(k.target_callable || ''),
            source_path_in_sglang: String(k.source_path_in_sglang || ''),
            verified_isolated_speedup: isolated, pct_gpu_time: Number(k.pct_gpu_time) || 0,
            // task_dir is deliberately EMPTY and the provenance is declared foreign — see the intro.
            task_dir: '', provenance: 'knowledge_base_replay',
          },
          // Only on the prebuilt-overlay path: unpack the candidate rather than assemble it.
          ...(viaOverlay ? { KB_OVERLAY_TARBALL: k.overlay_tar } : {}),
          CURRENT_OVERLAY: kbSeedOverlay || curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
          CURRENT_THROUGHPUT: kbSeedTput || BASELINE_TPUT, SKILL_DIR: WORKFLOW_DIR,
        };
        const integ = await runIntegrateBothLegs(
          'Overlay a kernel RECOVERED FROM THE KNOWLEDGE BASE and gate it on e2e throughput. Run your ' +
          'normal isolated-server A/B — same fresh-server legs, same parity probe. Two things are different ' +
          'and you must honour both:\n' +
          (viaOverlay
            ? '(0) DO NOT BUILD THE CANDIDATE OVERLAY. This kernel was authored by another run, whose ' +
              'workspace does not exist here, so your `authored` recipe (git apply into ' +
              'authored_kernel_eval_dir/workspace, copy kernel_src, add-rebind) cannot run and must not ' +
              'be attempted. The record ships the FINISHED overlay instead. Unpack it and bench THAT ' +
              'directory as the candidate. Note the tarball is a STANDALONE overlay from another run: ' +
              'untarring it over CURRENT_OVERLAY would overwrite `_overlay_manifest.json` and drop every ' +
              'rebind already accepted here, which turns the A/B into a comparison against a candidate ' +
              'that is missing part of its own reference. Merge all manifest entries, including deferred ' +
              'installers, with the checked merge command below. Use a fresh CAND directory; a conflict ' +
              'must be reported as inapplicable, never resolved by dropping existing hooks:\n' +
              '```bash\n' +
              'CAND="$EVAL_DIR/overlay/cand_<short_name>"\n' +
              `KBO=$(mktemp -d) && tar xzf ${shq(k.overlay_tar)} -C "$KBO" --strip-components=1\n` +
              'python3 "$SKILL_DIR/scripts/overlay_setup.py" merge --overlay "$CAND" \\\n' +
              '  --from "$CURRENT_OVERLAY" --with "$KBO"\n' +
              '```\n' +
              'The manifest declares rebinds, injected modules and installers with their target sites. ' +
              'VERIFY EVERY TARGET ACTUALLY TOOK on the candidate server (installer verification API, ' +
              'load banner, and an executed seam call) before you believe ' +
              'a null result: this overlay was built against a different framework_version, and a rebind ' +
              'whose target module was renamed upstream binds nothing, silently, which is indistinguishable ' +
              'from "the kernel made no difference". If no rebind takes, report gate:"rejected" with ' +
              'reason `no_rebind_seam` rather than a clean no-op. An authored/JIT kernel on the decode ' +
              'path still has to satisfy your CUDA-graph-safety rule — the packed source is rebuilt by ' +
              "this box's own torch load(), and a build failure is a rejection, not a retry.\n"
            : '') +
          '(1) There is NO task_dir and therefore NO immutable oracle for this kernel. Your step-1 ' +
          'provenance re-check cannot run, because the workspace that produced this patch does not ' +
          'exist on this box. Do NOT claim provenance was verified. The FRESH correctness probe against ' +
          'the TRUE baseline (greedy parity, or the task-accuracy gate when ACCURACY_GATE is in your ' +
          'inputs) is the only correctness evidence available here, so run it and report its result ' +
          'plainly — if it fails or cannot be established, REJECT.\n' +
          '(2) verified_isolated_speedup is a FOREIGN number, measured on another box against another ' +
          'baseline. It is context for what to expect, never evidence. Gate on what YOU measure.\n' +
          'As always: never mutate the installed package. The overlay on PYTHONPATH is the only ' +
          'mechanism, so a rejection costs nothing to undo — you simply do not adopt the directory.',
          kbIntegrateInputs, `warm_start integrate ${k.name}`, 'WarmStart');
        const base = kbSeedTput || BASELINE_TPUT;
        if (abDone(integ) && integAccepted(integ, Number(k.pct_gpu_time) || 0, isolated) &&
            integ.e2e_throughput_tok_s > base) {
          kbSeedOverlay = integ.accepted_overlay || kbSeedOverlay;
          kbSeedTput = integ.e2e_throughput_tok_s;
          bankAccepted(kbSeedKernels, {
            short_name: k.name, backend: String(k.language || k.backend || ''), kind,
            // On the prebuilt-overlay path there is no patch: the accepted overlay (packed by
            // Module B) is what reproduces the win.
            e2e_delta_pct: integ.e2e_delta_pct, isolated, patch: patchPath,
            replayed_from_overlay: viaOverlay ? k.overlay_tar : '',
            pct_gpu_time: Number(k.pct_gpu_time) || 0,
            apply_env: String(k.apply_env || ''), apply_flags: String(k.apply_flags || ''),
            // Provenance stays on the entry: a recovered win re-verified here, not a discovery.
            from_knowledge_base: true, kb_session_id: k.kb_session_id,
          }, krOf(kbIntegrateInputs));
          kernelVerdicts.push({ ...k, kind, outcome: 'adopted',
            measured_delta_pct: integ.e2e_delta_pct, why: integ.reason || '' });
          log(`[kb] ADOPTED kernel ${k.name} (${kind}): e2e now ${kbSeedTput} tok/s (+${integ.e2e_delta_pct}%).`);
        } else {
          const reason = gateRejectReason(integ, Number(k.pct_gpu_time) || 0, isolated);
          kernelVerdicts.push({ ...k, kind, outcome: abDone(integ) ? 'rejected' : 'incomplete',
            measured_delta_pct: integ ? integ.e2e_delta_pct : null,
            why: reason || 'A/B did not complete' });
          // No corrective re-author: a stored patch that fails its fresh gate is handed to the
          // Architect as a lead, not repaired.
          log(`[kb] kernel ${k.name} (${kind}) not adopted: ${reason || 'A/B did not complete'} — kept as a reference.`);
        }
      }
      // ---- Demote what did not survive into a REFERENCE for the roles that come next ----
      // The store's `e2e_reference_<sha7>.md` stays as its claim; this box's measurements go in a
      // sibling file that overrides it, surfaced three ways: the reference dir, the prompt block
      // (WARM_START_ROLES) and the `## Inputs` entry, which always lands.
      // ---- Tell the STORE what happened ----
      // `e2e_store.py attest` counts each attempt onto the record (no score or champion moves;
      // KB_ATTEST_OUTCOME sets the buckets, kb/attest.py:should_retire the curate threshold). Only
      // config verdicts benched on this box count; a record whose bundle also claims kernels is
      // reported as config-half-only.
      const configHalfOnly = v => Array.isArray(v.accepted_kernels) && v.accepted_kernels.length > 0;
      const attestable = verdicts.filter(v => v.session_id && kbCanAttestMetric(v) &&
        ['adopted', 'rejected', 'not_reproduced', 'inapplicable'].includes(v.outcome));
      const crossMetric = verdicts.filter(v => v.session_id && !kbCanAttestMetric(v));
      if (crossMetric.length) log(`[kb] ${crossMetric.length} record(s) not attested: ` +
        'the local measurement axis does not establish their stored metric claim.');
      if (attestable.length) {
        const cmds = attestable.map(v =>
          `python3 ${shq(E2E_STORE_SCRIPT)} attest ${kbIdentityFlags()} ${kbPlaneFlags()} ` +
          `--session-id ${shq(v.session_id)} ` +
          `--outcome ${KB_ATTEST_OUTCOME[v.outcome] || 'inapplicable'} ` +
          (v.measured_tok_s ? `--measured-tok-s ${v.measured_tok_s} ` : '') +
          `--baseline-tok-s ${BASELINE_TPUT} --parity ${shq(v.parity || 'n/a')} ` +
          // The note says what actually ran (overrides, drops), not just why it ended.
          `--note ${shq([
            // The local verdict leads the note: the filed bucket (KB_ATTEST_OUTCOME) may not show it.
            `local verdict: ${v.outcome}` +
              (v.measured_tok_s ? ` (${v.measured_tok_s} tok/s vs baseline ${BASELINE_TPUT})` : ' (no measurement)'),
            configHalfOnly(v)
              ? `config half only — the record also claims ${v.accepted_kernels.length} kernel(s), ` +
                'benched in the kernel lane and NOT covered by this verdict' : '',
            (v.overrides || []).length ? `overrode ${describeOverrides(v.overrides)}` : '',
            (v.dropped_flags || []).length
              ? `dropped ${(v.dropped_flags || []).join(' ')} to make it run here` +
                `${v.repair_classification ? ` (${v.repair_classification})` : ''}` : '',
            String(v.why || ''),
          ].filter(Boolean).join('; ').slice(0, 400))} ` +
          `--measured-by ${shq('e2e_workflow:' + BACKEND)} --apply || true`);
        try {
          await safeAgent(
            `You are the e2e knowledge-base attestor. Run EXACTLY these commands in order and ` +
            `return {"ran": <how many you ran>, "note": "<anything that failed>"}. Each records ` +
            `what THIS box saw when it benched a stored record. Do NOT edit them, do NOT add or ` +
            `drop any, and do NOT retry a failure — a repeat would double-count the attempt, and ` +
            `the ledger is meant to say how often this record was tried here, not how often the ` +
            `attestor pressed the button.\n` +
            '```bash\n' + KB_ENV_PRELUDE + cmds.join('\n') + '\n```',
            { phase: 'WarmStart', label: 'kb:attest',
              schema: obj({ ran: { type: 'number' }, note: { type: 'string' } }, []) },
            1);
          log(`[kb] attested ${attestable.length} benched record(s): ` +
            attestable.map(v => `${v.session_id.slice(-12)}=${v.outcome}`).join(' '));
        } catch (e) {
          // Non-fatal, like every other KB write.
          log(`[kb] attest failed (NON-FATAL): ${String(e).slice(0, 200)}`);
        }
      }

      const allVerdicts = verdicts.concat(kernelVerdicts);
      // The measured_on_this_box.md rows in machine form, for the Report role. Set outside the
      // `if (allVerdicts.length)` guard: "offered but not benched" is itself reportable.
      if (KB_RECALL.e2e) {
        KB_RECALL.e2e.configs = verdicts.map(v => ({
          direction: String(v.direction || 'unlabeled'), session_id: String(v.session_id || ''),
          stored_tok_s: v.throughput_tok_s != null ? v.throughput_tok_s : null,
          stored_speedup: v.speedup != null ? v.speedup : null,
          // Without the stored baseline a lower `delta_pct` reads as decay when the two baselines
          // were never comparable; without the half-flag it reads as a failed kernel.
          stored_baseline_tok_s: v.baseline_throughput_tok_s != null ? v.baseline_throughput_tok_s : null,
          config_half_only: configHalfOnly(v),
          measured_tok_s: v.measured_tok_s != null ? v.measured_tok_s : null,
          delta_pct: v.delta_pct != null ? v.delta_pct : null,
          parity: String(v.parity || ''), outcome: String(v.outcome || ''),
          // What the merge and repair did, beside the number they produced.
          overrides: Array.isArray(v.overrides) ? v.overrides : [],
          applied_partial: v.applied_partial === true,
          dropped_flags: Array.isArray(v.dropped_flags) ? v.dropped_flags : [],
          why: String(v.why || '').slice(0, 300),
        }));
        KB_RECALL.e2e.kernels = kernelVerdicts.map(v => ({
          name: String(v.name || ''), kind: String(v.kind || ''),
          kb_session_id: String(v.kb_session_id || ''),
          stored_isolated: v.isolated_speedup != null ? v.isolated_speedup : null,
          measured_delta_pct: v.measured_delta_pct != null ? v.measured_delta_pct : null,
          outcome: String(v.outcome || ''), why: String(v.why || '').slice(0, 300),
        }));
      }
      if (allVerdicts.length) {
        const adoptedCfg = verdicts.filter(v => v.outcome === 'adopted');
        const adoptedKer = kernelVerdicts.filter(v => v.outcome === 'adopted');
        const rejected = allVerdicts.filter(v => v.outcome !== 'adopted');
        // Also counted inside `rejected`: to the next roles both mean "do not re-propose verbatim".
        const notReproduced = verdicts.filter(
          v => v.outcome === 'not_reproduced' || v.outcome === 'inapplicable');
        const md = [
          '# Warm start — MEASURED ON THIS BOX',
          '',
          'This file OVERRIDES the stored claims in its sibling `e2e_reference_*.md` wherever the two',
          'disagree. Those files record what another box reported; this one records what happened when',
          'this run applied the same thing here, through the same gate a fresh idea faces.',
          '',
          `- baseline: **${BASELINE_TPUT} tok/s** (noise band ${NOISE_BAND}%)`,
          `- serving: BACKEND=${BACKEND} TP=${SERVING_TP} GPU=${SERVING_GPU}, workload isl=${ISL} osl=${OSL} conc=${CONC}`,
          `- identity read: \`${resolved.canonical_id || '?'}\` (match tier \`${resolved.match_tier || '-'}\`)`,
          // `exact` covers the identity only (no flags, env or kv-cache dtype), so a record measured
          // against another baseline can reproduce smaller here for that reason alone.
          ...(verdicts.some(v => v.baseline_throughput_tok_s != null && BASELINE_TPUT &&
              Math.abs(v.baseline_throughput_tok_s - BASELINE_TPUT) / BASELINE_TPUT > NOISE_BAND / 100)
            ? ['- CAUTION: the match tier covers the identity only — not server flags, env or ' +
               'kv-cache dtype. At least one offer below was recorded against a different baseline ' +
               '(column `stored baseline`), so its stored percentage and the one measured here are ' +
               'not the same quantity.']
            : []),
          '',
          '## Configurations',
          '',
          '| stored direction | stored claim | stored baseline | measured here | delta vs baseline | parity | outcome |',
          '|---|---|---|---|---|---|---|',
          ...(verdicts.length ? verdicts.map(v =>
            `| ${v.direction || 'unlabeled'} | ${v.throughput_tok_s != null ? v.throughput_tok_s + ' tok/s' : '?'}` +
            `${v.speedup != null ? ` (${v.speedup}x)` : ''}` +
            `${configHalfOnly(v) ? ` — WHOLE BUNDLE, incl. ${v.accepted_kernels.length} kernel(s); only the config half ran in this row` : ''} | ` +
            `${v.baseline_throughput_tok_s != null ? v.baseline_throughput_tok_s + ' tok/s' : '—'} | ` +
            `${v.measured_tok_s != null ? v.measured_tok_s + ' tok/s' : 'not benched'} | ` +
            `${v.delta_pct != null ? (v.delta_pct >= 0 ? '+' : '') + v.delta_pct.toFixed(2) + '%' : '—'} | ` +
            `${v.parity || '—'} | **${v.outcome}** |`)
            : ['| _(none offered)_ | | | | | | |']),
          '',
          '## Kernels',
          '',
          '| kernel | kind | stored isolated | measured e2e delta | outcome | why |',
          '|---|---|---|---|---|---|',
          ...(kernelVerdicts.length ? kernelVerdicts.map(v =>
            `| ${v.name} | ${v.kind || '?'} | ${v.isolated_speedup ? v.isolated_speedup + 'x' : '—'} | ` +
            `${v.measured_delta_pct != null ? (v.measured_delta_pct >= 0 ? '+' : '') + v.measured_delta_pct + '%' : '—'} | ` +
            `**${v.outcome}** | ${String(v.why || '').slice(0, 160).replace(/[\\|]/g, '\\$&')} |`)
            : ['| _(none recorded)_ | | | | | |']),
          '',
          // A top-N entry that could not be reproduced is handed forward WITH its reproduction
          // material (launch script, patch paths).
          ...(notReproduced.length ? [
            '## REFERENCE ONLY — recalled but NOT reproduced here',
            '',
            'These were offered by the store and benched on this box, and either produced no number',
            'at all, never took effect (a flag renamed upstream is accepted silently and then',
            "ignored), or collided with a knob this run's baseline already pins and could not be",
            'applied here at all. They are NOT results — and the last of those is not evidence',
            'against the record either, only against the pairing. They are the closest thing this',
            'deployment has to a record of what someone else got working, and their material is',
            'below so you can read',
            'what they actually did rather than guess from a direction label.',
            '',
            ...notReproduced.flatMap(v => {
              const repro = (v.repro && typeof v.repro === 'object') ? v.repro : {};
              const root = (v.bundle && v.bundle.path) ? `${v.bundle.path}/files` : '';
              const patches = (Array.isArray(repro.kernels) ? repro.kernels : [])
                .filter(k => k && k.patch).map(k => root ? `${root}/${k.patch}` : k.patch);
              return [
                `### ${v.direction || 'unlabeled'} (session \`${v.session_id || '?'}\`)`,
                `- why it did not reproduce: ${v.outcome === 'inapplicable'
                  ? "it could not be applied to this run's baseline at all — " : ''}` +
                  `${String(v.why || 'no measurement came back').slice(0, 300)}`,
                ...((v.overrides || []).length
                  ? [`- knobs where it disagreed with this run's baseline (recorded value was ` +
                     `taken): ${describeOverrides(v.overrides)}`] : []),
                ...((v.dropped_flags || []).length
                  ? [`- dropped to get it running here: ${v.dropped_flags.join(' ')}` +
                     `${v.repair_classification ? ` (${v.repair_classification})` : ''} — so the ` +
                     'number above, if any, is for LESS than the record asked for'] : []),
                `- stored claim: ${v.throughput_tok_s != null ? v.throughput_tok_s + ' tok/s' : '?'}` +
                  `${v.speedup != null ? ` (${v.speedup}x)` : ''}, recorded elsewhere`,
                `- launch script: ${repro.launch
                  ? `\`${root ? `${root}/${repro.launch}` : repro.launch}\`` +
                    (repro.launch_origin === 'synthesized'
                      ? ' — SYNTHESIZED from the stored config, never executed as-is'
                      : ' — captured from the original run')
                  : 'none recorded (this record predates `repro`)'}`,
                `- kernel patches: ${patches.length ? patches.map(p => `\`${p}\``).join(', ') : 'none carried'}` +
                  `${repro.kernels_without_patch ? ` (${repro.kernels_without_patch} accepted kernel(s) carry no patch)` : ''}`,
                `- flags: \`${String((v.accepted_config || {}).flags || '(none)')}\``,
                `- env: \`${String((v.accepted_config || {}).env || '(none)')}\``,
                '',
              ];
            }),
          ] : []),
          '## How to use this',
          '',
          adoptedCfg.length
            ? `The adopted configuration is ALREADY in CURRENT_FLAGS / CURRENT_ENV. Do not re-propose it — ` +
              `propose only things that COMPOUND on top of it.`
            : `Nothing from the store was adopted as configuration, so CURRENT_FLAGS / CURRENT_ENV are ` +
              `unchanged from the baseline.`,
          '',
          adoptedKer.length
            ? `${adoptedKer.length} recovered kernel(s) are already in the active overlay and already ` +
              `reflected in the profile you are routing from. Their ops are DONE; look elsewhere.`
            : `No recovered kernel was adopted, so the overlay is empty and every op in the profile is ` +
              `still open.`,
          '',
          rejected.length
            ? `The ${rejected.length} rejected/unreplayed entries above are LEADS, not dead ends. A ` +
              `rejection here means the whole compounded thing did not beat this baseline on this box — ` +
              `it does NOT mean each knob inside it is worthless, and it does not mean the DIRECTION is ` +
              `wrong. Do not re-propose any of them verbatim; do feel free to propose an individual axis ` +
              `from one, or the same idea approached differently.`
            : `Nothing was rejected.`,
          '',
        ].join('\n');
        await safeAgent(
          `You are a file writer. Use the Write tool to create the file ` +
          `"${refsDir}/measured_on_this_box.md" with EXACTLY the content below, verbatim. ` +
          `Do NOT reformat, summarize, re-order rows, or change any number:\n\n` +
          '````markdown\n' + md + '\n````\n\n' +
          `Then return {"written": true, "path": "${refsDir}/measured_on_this_box.md"}.`,
          { phase: 'WarmStart', label: 'warm_start:record-measurements',
            schema: obj({ written: { type: 'boolean' }, path: { type: 'string' } }, []) },
          2);
        KB_REF_DIR = refsDir;   // the measurements exist even if the offer list was thin

        // The channel that always fires (`## Inputs` of two roles); terse on purpose.
        KB_REF_INPUTS = {
          KB_REFERENCE_DIR: refsDir,
          KB_REFERENCE_VERDICT:
            `The knowledge base offered ${allVerdicts.length} prior result(s) for this deployment ` +
            `(${resolved.canonical_id || '?'}, tier ${resolved.match_tier || '-'}). This run benched them ` +
            `and recorded what actually happened in ${refsDir}/measured_on_this_box.md — read that file, ` +
            `and treat it as overriding the stored claims in its siblings. ` +
            (adoptedCfg.length
              ? `ADOPTED config: ${adoptedCfg.map(v => v.direction || 'unlabeled').join(', ')} — it is ALREADY ` +
                `in CURRENT_FLAGS/CURRENT_ENV, so propose only things that COMPOUND on top of it, never it again. `
              : `No stored config was adopted. `) +
            (adoptedKer.length
              ? `ADOPTED kernels: ${adoptedKer.map(v => v.name).join(', ')} — already in the overlay and already ` +
                `reflected in the profile, so those ops are done. `
              : `No stored kernel was adopted. `) +
            (rejected.length
              ? `REJECTED/unreplayed: ${rejected.map(v => v.direction || v.name || '?').join(', ')}. Do not ` +
                `re-propose any of them verbatim — the compounded whole lost on this box. Their individual ` +
                `knobs may each still be a valid axis, and their declared directions are still legitimate ` +
                `ideas to reach a different way.`
              : '') +
            (notReproduced.length
              ? ` ${notReproduced.length} of those could not be reproduced AT ALL here (no number, or the ` +
                `config never took effect) — they are in the REFERENCE ONLY section of that file with their ` +
                `launch script and kernel patches attached. Read them as evidence of what worked somewhere ` +
                `else, not as something to re-run.`
              : ''),
        };
      }
    }
  }

  // A resumed overlay wins; otherwise a freshly replayed KB overlay stacks on the Hyperloom underlay.
  // Fold before Profile and ConfigSweep so every downstream measurement sees the active source stack.
  curOverlay = ST.overlay || kbSeedOverlay || curOverlay || '';

  if (searchClosed('Profile and Strategize') || searchWindowShort('Profile, kernel extraction and the kernel search')) {
    kernelQueue = [];
    headQueue = [];
  } else {
    if (KERNEL_EXTRACT_ON) {
      strategy = await prepareAgentXKernels('baseline', kbSeedTput || BASELINE_TPUT);
      profile = strategy;
    } else {
      phase('Profile');
      profile = await safeAgent(
        roleAgent('profiler', 'baseline', 'Capture a warm trace and emit the standardized Top-N.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 0,
          OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          ...TRACELENS_INPUTS, ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Profile', label: 'profiler:baseline', schema: PROFILE_SCHEMA });
      log(`Baseline profiled. ${profile ? (profile.top_kernels || []).length : 0} top kernels.`);

      phase('Strategize');
      strategy = await safeAgent(
        roleAgent('system_architect', 'strategize', 'Route the Top-N into config/kernel/host tracks by Amdahl.', {
          EVAL_DIR, PROFILE_TOPN: profile ? profile.profile_topN_json : '', BASELINE_THROUGHPUT: BASELINE_TPUT,
          WORKLOAD, BUDGET, HEAD_THRESHOLD_PCT, CONFIG_TUNE_ENABLED, SKILL_DIR: WORKFLOW_DIR,
          ...TRACELENS_INPUTS, ...ANALYSIS_SKILL_INPUTS, ...KB_REF_INPUTS,
        }),
        { phase: 'Strategize', label: 'architect:strategize', schema: STRATEGY_SCHEMA });
    }
    kernelQueue = (strategy && strategy.kernel_candidates) ? strategy.kernel_candidates.slice() : [];
    headQueue = (strategy && strategy.head_candidates) ? strategy.head_candidates.slice() : [];
    headQueue = applyOpIdentityGuard(headQueue, 'strategize');
    log(`Strategy: ${headQueue.length} head candidates, ${kernelQueue.length} kernel candidates, ${(strategy && strategy.config_directions || []).length} config directions.`);
    // strategize decided the backends -> if any candidate routed flydsl, provision it now (blocking).
    await ensureFlydslGate();
  }
} else {
  // Load carried state from a prior phase invocation (args.state).
  EVAL_DIR = ST.eval_dir || EVAL_DIR_OVERRIDE;
  if (!EVAL_DIR) throw new Error('Non-setup phase requires args.state.eval_dir (or args.eval_dir)');
  MODEL_NAME = ST.model_name || MODEL_NAME_HINT;
  BASELINE_TPUT = ST.baseline_throughput_tok_s || 0;
  NOISE_BAND = gateNoiseBand(ST.noise_band_pct);
  curFlags = ST.flags || '';
  curEnv = ST.env || '';
  curOverlay = ST.overlay || INIT_BASE_OVERLAY;
  // Older states did not retain the seed. Missing evidence must not mean an empty/pristine seed.
  BASELINE_FLAGS = typeof ST.baseline_flags === 'string' ? ST.baseline_flags : null;
  BASELINE_ENV = typeof ST.baseline_env === 'string' ? ST.baseline_env : null;
  BASELINE_OVERLAY = typeof ST.baseline_overlay === 'string' ? ST.baseline_overlay : null;
  profile = { profile_topN_json: ST.profile_topn_json || '',
    ...(KERNEL_EXTRACT_ON ? { profile_workload_json: ST.profile_workload_json || '',
      kernel_ranking_json: ST.kernel_ranking_json || '',
      kernel_extraction_delivery_json: ST.kernel_extraction_delivery_json || '',
      failed_tasks: ST.kernel_extraction_summary?.failed_tasks || [],
      deferred_tasks: ST.kernel_extraction_summary?.deferred_tasks || [],
      coverage_summary: ST.kernel_extraction_summary?.coverage_summary || {} } : {}) };
  strategy = { config_directions: ST.config_directions || [] };
  kernelQueue = ST.kernelQueue || [];
  headQueue = admitHeads(ST.headQueue || [], 'resume');
  // Dispatch replays prepared_ut.result (it cannot read disk), so a candidate without it re-runs
  // preparation, which re-packs the saved evidence rather than redoing GPU work.
  const unusable = (k) => !k || !k.prepared_ut || !k.prepared_ut.result;
  if (KERNEL_EXTRACT_ON &&
      (PHASES.includes('extract') ||
       (want('head') && (!Array.isArray(ST.headQueue) || headQueue.some(unusable))) ||
       (want('kernel') && (!Array.isArray(ST.kernelQueue) || kernelQueue.some(unusable))))) {
    // Finish the skill through GPU UT validation. Existing selection-only queues
    // are evidence to reuse, not completed tasks ready for optimization.
    strategy = await prepareAgentXKernels('resume', ST.throughput || BASELINE_TPUT);
    profile = strategy;
    headQueue = applyOpIdentityGuard(strategy.head_candidates.slice(), 'resume selection');
    kernelQueue = strategy.kernel_candidates.slice();
  }
  log(`Loaded carried state: EVAL_DIR=${EVAL_DIR}, baseline ${BASELINE_TPUT}, flags='${curFlags}', env='${curEnv}', ${headQueue.length} head + ${kernelQueue.length} kernel candidates.`);
}

