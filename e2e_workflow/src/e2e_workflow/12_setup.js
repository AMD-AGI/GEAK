// ===========================================================================
// PHASE: Setup + Baseline profile + Strategize  (gated; else load carried state)
// ===========================================================================
// Module A's outputs. They are folded into the ORDINARY state variables at those variables' real
// declaration sites further down. Module A must run BEFORE Profile so the Top-N is taken on the
// adopted config. Their off-values are 0 / '' / [], which makes each fold a no-op when the feature is off.
let kbSeedTput = 0;
let kbSeedOverlay = '';
const kbSeedKernels = [];
// Spread into the Strategize and ConfigSweep inputs. Stays `{}` unless Module A measured something:
// `{...{}}` contributes no own-properties and preserves key order, so those two prompts are
// byte-identical to the pre-feature build on an off run.
let KB_REF_INPUTS = {};

let EVAL_DIR, MODEL_NAME, BASELINE_TPUT, NOISE_BAND, curFlags, curEnv, curOverlay, profile, strategy, kernelQueue, headQueue;
// Keep the original measured seed separate from the evolving timing reference.
let BASELINE_FLAGS = null, BASELINE_ENV = null, BASELINE_OVERLAY = null;
if (want('setup')) {
  phase('Setup');
  const setup = await safeAgent(
    roleAgent('director', 'setup', 'Build the isolated e2e eval dir and record the baseline throughput.', {
      LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT, EVAL_DIR_OVERRIDE, MODEL_NAME_HINT, TASK,
      GPU_IDS, WORKLOAD, INIT_FLAGS, INIT_ENV, INIT_BASE_OVERLAY,
      MEASUREMENT_PURPOSE: 'parity', REPLICAS: PARITY_REPLICAS,
      SKILL_DIR: WORKFLOW_DIR, EXPECTED_GFX, EXPECTED_TARGET,
      EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
      // Non-empty ONLY on a declared trace replay. The Director writes it to
      // $EVAL_DIR/bench_env.sh in step 3, BEFORE the baseline bench in step 5,
      // so the baseline is measured on the real client and the real metric axis
      // rather than a synthetic sweep. Empty => the Director writes nothing.
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
  // BASELINE_TPUT is the denominator of every gain this run will report, so a
  // baseline accidentally measured with the synthetic client on the output axis
  // does not just mis-state one number -- it silently rebases the entire run,
  // and the mismatch is invisible in the result afterwards. The declaration
  // travels through an AGENT (the Director writes bench_env.sh), so confirm the
  // effect rather than trusting the instruction: the file has to exist and the
  // baseline summary has to carry the metric basis we asked for.
  // Positive mismatch => abort now, while one bench has been spent instead of
  // a full budget. Unreadable summary => say so loudly and continue, since the
  // measurement itself may still be fine and a false abort is its own failure.
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
      // Now that something has actually been served, take the kernel-targeting
      // shape from the measurement instead of from a declaration or a constant.
      // A caller who declared a shape explicitly keeps it: they may be targeting
      // a regime deliberately, and overriding that silently would be its own
      // surprise.
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
  //
  // Placed HERE, and not earlier, for three reasons that are each load-bearing:
  //   * EVAL_DIR, bench_e2e.sh, BASELINE_TPUT and NOISE_BAND do not exist until Setup returns, and a
  //     candidate cannot be judged without a baseline to judge it against.
  //   * it is before Profile, so the Top-N is captured on the ADOPTED config. This is the same
  //     discipline the flow already asserts by re-profiling after ConfigSweep: a config change moves
  //     which kernels dominate, and routing off a stale profile optimizes the wrong ops.
  //   * it is INSIDE want('setup'), so a phase-partial resume cold-starts automatically rather than
  //     re-reading and re-benching a store on every phase invocation.
  //
  // Nothing below is trusted on the store's word. A stored speedup was measured on another box, on
  // another day, against another baseline; the only thing that makes it actionable here is that this
  // run re-measured it through the same gate a fresh idea would face.
  // =========================================================================
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
      // Refuse to read rather than read the wrong page. An arch-less identity resolves to the
      // `gfx=unknown` rung, whose records were measured on hardware we cannot establish — and the
      // cost of believing one is a 20-40min server launch, not a wasted lookup.
      log('[kb] warm start skipped: read_reason=missing_arch (the Director reported no gfx; a ' +
        'cross-arch config is not a candidate, it is a guess).');
      // Still a recall FACT worth reporting: "we did not look, and here is why" is not the same
      // report as "we looked and the store was empty", and the reader cannot tell them apart from
      // an absent section.
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
          // The read leaves this run's KB ADDRESS on disk as a side effect, at the one moment the
          // dimensions are known and the process is still healthy. Module B below is the happy
          // path; when this process dies before reaching it (6 of the 9 measured runs on
          // 20260821-22), run_e2e.py salvages the numbers from disk and needs this file to know
          // which pages they belong on. Free: no extra agent, no second formatting of the dims.
          // --store/--identity-plane are recorded, not used, by the read: a `both` run reads
          // remote-first, so without them the file would tell the salvage writer "remote" and lose
          // the local mirror this run was configured to keep.
          `  --identity-out ${shq(KB_IDENTITY_FILE)} \\\n` +
          `  --store ${shq(E2E_KB_STORE_DIR)} --identity-plane ${E2E_KB_PLANE}`) + '\n' +
        '```',
        { phase: 'WarmStart', label: 'warm_start:resolve', schema: KB_RESOLVE_SCHEMA }) || {};
      const cands = Array.isArray(resolved.candidates) ? resolved.candidates : [];
      // Log the ladder VERBATIM. On a scheme with no search, "never recorded" and "recorded under an
      // address one segment different" are the same 404, and this line is the only record of which
      // question was actually asked — without it a silent identity drift looks like an empty store.
      // `read_plane` is which plane ANSWERED, `plane` only which was asked for. Falls back for a
      // resolve emitted by an older build, whose JSON has no `read_plane`.
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
      // Record the ASK before anything is benched. If this process dies mid-warm-start the report
      // still knows which ladder was queried, and on a zero-candidate read this is the entire
      // finding — see KB_RECALL's declaration for why the address matters more than the count.
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
        // Both floors, always: "what did this run decline to see, and at what threshold" is not
        // answerable after the fact from a candidate list.
        read_min_speedup: E2E_WARM_START_MIN_SPEEDUP,
        bench_min_speedup: E2E_WARM_START_BENCH_MIN_SPEEDUP,
        curation,
      };
      if (cands.length) KB_REF_DIR = refsDir;   // arms warmStartBlock() for the consumer roles
      // Armed on the same condition and never separately: the cache holds nothing until a candidate
      // is materialized, and a path to an empty directory reads to an agent as "the file is missing"
      // rather than "there was no record", which are very different things to act on.
      if (cands.length) KB_CACHE_DIR = cacheDir;

      // How many of the offers are worth a server launch. A coarse rung gets its own (smaller)
      // budget rather than zero — see E2E_WARM_START_VALIDATE_N_COARSE for why the stored number's
      // incomparability bounds how much we spend guessing, not whether the guess can be measured.
      const exactTier = (resolved.match_tier || '') === 'exact';
      const benchN = E2E_WARM_START_REF_ONLY ? 0
        : exactTier ? E2E_WARM_START_VALIDATE_N
        : A.e2e_warm_start_validate_n != null ? E2E_WARM_START_VALIDATE_N
        : E2E_WARM_START_VALIDATE_N_COARSE;
      // WHICH offers, once we know how many. The page came back ordered by absolute throughput (the
      // resolve default), and on a coarse rung that throughput was measured at some other tp or
      // workload point, so it ranks the offers by the one quantity that did not survive the move.
      // Re-order by the rung's own metric — speedup — so the launch is spent on the record that
      // claims the biggest RATIO, which is what the store promotes its coarse champions on. Note the
      // page was still FETCHED by throughput, so this reorders a throughput-biased top-N; making the
      // fetch itself follow the rung metric belongs in e2e_store.resolve, not here.
      const benchOrder = exactTier ? cands : [...cands].sort(
        (x, y) => (Number(y.speedup) || 0) - (Number(x.speedup) || 0));
      // The floor that spends money, applied HERE and not in the read. A record under it keeps its
      // place in the offer, its bundle in the cache and its paragraph in the roles' context, and
      // loses only the launch. No speedup recorded stays benchable: an unanswerable test is not a
      // failed one.
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

      // Seeded with the records the bench floor declined. `skipped` is outside the attestable set
      // on purpose: the ledger counts what was TRIED on hardware, and this was not.
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
        // VALIDATE THROUGH THE ORIGINAL GATE. This is config_tuner:sweep — the same role, the same
        // schema, the same bench_e2e.sh at the same TP/GPU, the same delta-vs-median arithmetic, the
        // same parity check and the same swap-took-effect log grep the flow already trusts for a
        // fresh idea. Reusing it rather than writing a warm-start harness is the whole reason a
        // stored config and a proposed config are judged by identical evidence.
        const stored = (c.accepted_config && typeof c.accepted_config === 'object') ? c.accepted_config : {};
        const storedFlags = String(stored.flags || '');
        const storedEnv = String(stored.env || '');
        if (!storedFlags && !storedEnv) {
          verdicts.push({ ...c, measured_tok_s: null, delta_pct: null, parity: 'n/a',
            outcome: 'skipped', why: 'the record carries no config to apply' });
          continue;
        }
        // Key-by-key, not string-concatenated — see mergeConfig. The recorded value wins every
        // collision; `overrides` is the list of knobs where this box and the record disagreed.
        const mf = mergeFlags(curFlags, storedFlags);
        const me = mergeEnv(curEnv, storedEnv);
        const overrides = mf.overrides.concat(me.overrides);
        if (overrides.length) {
          log(`[kb] merge ${c.session_id || '?'}: the recorded config overrides ${overrides.length} ` +
            `knob(s) this run's baseline already pins — ${describeOverrides(overrides)}. ` +
            'The recorded value is the one under test, so it wins.');
        }
        // Nothing left to test. Not a verdict on the record and not worth a server launch: this
        // run's baseline already IS the recorded configuration, so a bench would compare the
        // baseline with itself and file the ~0% result as a loss. Left un-attested on purpose —
        // `recalls` counts attempts on hardware, and this one never reached any.
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

        // ONE repair attempt, and only when the server produced no number at all.
        //
        // The first launch is where a box the config has never met shows up: a flag deleted upstream
        // makes argparse exit before the model loads, and a knob this run's baseline pins can be
        // incompatible with the recorded value. Both used to end as measured 0, one shot spent,
        // `not_reproduced` filed against the record — but only the first is the record's fault, and
        // a record that keeps meeting incompatible baselines was being retired for it.
        //
        // So: hand the launch failure back, make the tuner say WHICH of the two it is, let it adapt
        // the minimum needed, and bench once more. Exactly once — a repair loop against a config
        // that cannot run here learns nothing new — and through the SAME accept gate.
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
        // The role's own list when it filled one in; otherwise subtract what ran from what was
        // asked for. Either way the verdict names the knobs that did not make it onto the box.
        const dropped = (repair && Array.isArray(repair.dropped_flags) && repair.dropped_flags.length)
          ? repair.dropped_flags
          : (repair ? droppedByDiff(mf.merged, me.merged,
            String(trial.change || [trial.flags, trial.env].filter(Boolean).join(' | '))) : []);
        const parity = String(trial.parity || trial.output_parity || '');
        const deltaPct = BASELINE_TPUT ? ((measured - BASELINE_TPUT) / BASELINE_TPUT) * 100 : 0;
        // Both the tuner's own judgement AND our arithmetic. A warm-start candidate is precisely
        // where an agent is most tempted to ratify a stored claim it did not establish, so the
        // orchestrator re-derives the ratio from the raw number rather than accepting `kept: true`.
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
        // FOUR outcomes, three different things — see KB_ATTEST_OUTCOME. All counted apart
        // (kb/attest.py); only a loss and a no-effect count against the record, and then only as
        // evidence toward a threshold.
        //
        // `note` and `notes` are both read: the role file's return schema spells it `note`, and
        // reading only `notes` meant the per-trial text never reached this regex at all.
        const notes = String(trial.notes || trial.note || (sweep && sweep.summary) || '');
        const inert = /\b(?:not (?:honou?red|recognized|recognised|applied|supported)|unrecognized|unrecognised|ignored|no such option|unknown (?:option|argument|flag)|renamed|removed upstream|did not take effect)\b/i.test(notes);
        // The repair's structured verdict outranks the regex: it was reached by reading the launch
        // error, while the regex is guessing at free text written for a human.
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
          // What was actually put on the box, as opposed to what the record asked for. A reader
          // comparing this run's number with the stored one needs both, and `applied_partial`
          // is the difference between "the record lost" and "most of the record lost".
          overrides, applied_partial: dropped.length > 0, dropped_flags: dropped,
          repair_classification: String((repair && repair.classification) || '') });
        if (accept) break;   // adopt the first that passes; the rest stay references
      }
      // ---------------------------------------------------------------------
      // Replay the stored KERNELS through the ordinary integrate gate.
      //
      // Same machinery as the Milestone track, verbatim: runIntegrateBothLegs, the same
      // INTEGRATE_SCHEMA, the same isolated ref/candidate A/B with the parity probe, the same integAccepted
      // predicate. The only thing that differs is where the patch came from, and that is exactly the
      // thing the A/B is there to make irrelevant.
      //
      // Reverting is structural rather than an operation: a candidate lives in its own overlay
      // directory and is activated only by being on PYTHONPATH, so rejecting one means not adopting
      // it. Nothing is ever mutated in the install.
      // ---------------------------------------------------------------------
      const kbKernels = [];
      const seenKernel = new Set();
      for (const c of cands) {
        const bundlePath = (c.bundle && c.bundle.path) || '';
        // The record's PREBUILT overlay directory, packed by e2e_store._pack_overlay as
        // `overlay.tar.gz` (manifest + sitecustomize + `_patched/` + every overlay-root module the
        // rebinds transitively import). Named from the manifest rather than probed, because this
        // script has no filesystem: a bundle whose upload never carried the tarball simply does not
        // list it, and the entry stays a reference instead of costing two server launches to find out.
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
      // One overlay tarball is the whole RECORD's final overlay, not one kernel's. If a record banked
      // three authored kernels, restoring it restores all three at once, so it is benched ONCE and the
      // remaining kernels of that record are marked as covered by it rather than each buying their own
      // pair of server launches to measure the identical directory.
      const overlayReplayed = new Set();
      const kernelVerdicts = [];
      for (const k of kbKernels) {
        const kind = String(k.winner_kind || k.kind || '').trim().toLowerCase();
        // `patch` is stored inside the record's own bundle as `kernels/<name>.patch`, so the path only
        // resolves if the artifacts actually came down. A record whose manifest was never committed
        // downloads a knowledge document and no files, and handing the integrator a path that opens
        // nothing would burn two server launches to discover it.
        const patchPath = (kind === 'patch' && k.patch && k.bundle)
          ? `${k.bundle}/files/${k.patch}` : '';
        // An env/flag win is only replayable if the record actually says WHICH env or flag routed the
        // op. Several records in the store name the kind but carry empty apply_env/apply_flags — they
        // were written before those fields were banked. Sending one to the integrator produces an A/B
        // between two identical configurations: two full server launches to measure nothing, and a
        // 'rejected' verdict that then libels a win which may well have been real.
        const hasRouting = !!(String(k.apply_env || '').trim() || String(k.apply_flags || '').trim());
        // A kind this lane cannot BUILD from a record can still be REPLAYED when the record ships the
        // built artifact. `authored` is the case that matters: the integrator's authored path needs
        // `authored_kernel_eval_dir/workspace/` to git-apply the diff and assemble the module, and that
        // workspace is gone the moment the run ends — which is what the old blanket exclusion was
        // about. But the OUTPUT of that build, the overlay directory, is in the record: manifest,
        // sitecustomize, and the authored source (e.g. `geak_hip_extend/extend_attention_hip.hip`,
        // rebuilt by the reader's own torch load()). The rebind seam is named in the manifest
        // (`sglang...extend_attention:extend_attention_fwd` -> `geak_hip_extend:extend_attention_fwd`),
        // so it is checkable rather than unprovable, and if it does not bind here the A/B says so.
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
            // Both spellings, because the two tracks read different ones and this synthetic result
            // has to satisfy whichever the integrator reaches for.
            code_patch: patchPath, final_patch: patchPath,
            apply_env: String(k.apply_env || ''), apply_flags: String(k.apply_flags || ''),
            target_callable: String(k.target_callable || ''),
            source_path_in_sglang: String(k.source_path_in_sglang || ''),
            verified_isolated_speedup: isolated, pct_gpu_time: Number(k.pct_gpu_time) || 0,
            // task_dir is deliberately EMPTY and the provenance is declared foreign — see the intro.
            task_dir: '', provenance: 'knowledge_base_replay',
          },
          // Set only on the prebuilt-overlay path, so the integrator can tell "assemble the candidate
          // from this patch" from "the candidate already exists, unpack it and bench it".
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
            // On the prebuilt-overlay path there is no patch to point at; what reproduces this win is
            // the accepted overlay the integrator just built from the tarball, and Module B packs that
            // directory the same way the record we replayed was packed.
            e2e_delta_pct: integ.e2e_delta_pct, isolated, patch: patchPath,
            replayed_from_overlay: viaOverlay ? k.overlay_tar : '',
            pct_gpu_time: Number(k.pct_gpu_time) || 0,
            apply_env: String(k.apply_env || ''), apply_flags: String(k.apply_flags || ''),
            // Provenance stays ON the banked entry: this kernel is a recovered win re-verified here,
            // not something this run discovered, and Module B must not later claim otherwise.
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
          // No corrective re-author here. That path re-optimizes a kernel this run authored and owns;
          // a stored patch that fails its fresh gate is simply not this box's win, and the honest
          // outcome is to hand it to the Architect as a lead rather than to repair someone else's diff.
          log(`[kb] kernel ${k.name} (${kind}) not adopted: ${reason || 'A/B did not complete'} — kept as a reference.`);
        }
      }
      // ---------------------------------------------------------------------
      // Demote what did not survive into a REFERENCE for the roles that come next.
      //
      // The resolver's own `e2e_reference_<sha7>.md` is deliberately left alone. It is the pristine
      // record of what the STORE claimed, and the disagreement between that and what this box
      // measured is the interesting datum — editing the claim to match the measurement destroys the
      // only evidence that the two differ. So the measurements go in a sibling file that says, in as
      // many words, that it overrides its neighbours.
      //
      // Three channels, because the first two can each be missed. The reference DIRECTORY is only
      // read if an agent chooses to; the prompt BLOCK only reaches roles in WARM_START_ROLES; the
      // INPUTS entry lands in `## Inputs` unconditionally and is the one that always fires.
      // ---------------------------------------------------------------------
      // ---------------------------------------------------------------------
      // Tell the STORE what happened. Until now this loop's verdicts died with the run: the next
      // box to resolve this identity saw the same optimistic record, benched it, and failed the
      // same way, forever. `e2e_store.py attest` counts the attempt onto the record itself, at
      // every rung, so a later reader can see how often it has been tried here and how it went.
      // It moves no score and no champion: one box's failure is evidence, never a verdict. What it
      // adds up to is decided elsewhere — KB_ATTEST_OUTCOME for the buckets, and
      // kb/attest.py:should_retire for the threshold a separate `curate` pass acts on.
      //
      // Only candidates that were actually PUT ON THIS BOX are counted. A record listed in the
      // offer and never benched (`skipped`, or below benchN) has learned nothing about itself, and
      // counting it would put an attempt that never happened into the ledger.
      //
      // Config verdicts only. A replayed kernel that failed is evidence against the KERNEL lane's
      // record, not against the e2e run that once used it, and the two have separate ledgers —
      // `experience_store.py attest` is where that verdict belongs.
      //
      // A record's headline number covers its WHOLE bundle. When that includes kernels this lane
      // benched only the config half, so every place printing both has to say which half ran.
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
          // The note carries what was actually run, not just why it ended: a bare "no win" against
          // a partially-dropped config would read, three months later, as a verdict on the whole
          // record. Overrides first — they are what a reader has to know to interpret the number.
          `--note ${shq([
            // The local verdict leads the note. It is filed as `inapplicable` so it cannot retire
            // the record; a reader still has to be able to see what actually happened here.
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
          // Non-fatal, like every other KB write. The measurements are already on disk and already
          // in the reference file; losing the counter costs a future reader context, not this run.
          log(`[kb] attest failed (NON-FATAL): ${String(e).slice(0, 200)}`);
        }
      }

      const allVerdicts = verdicts.concat(kernelVerdicts);
      // Same rows the measured_on_this_box.md tables carry, in machine form, so the Report role
      // states the recall outcome from the orchestrator's own record rather than from whether an
      // agent happened to open a reference file. Set OUTSIDE the `if (allVerdicts.length)` guard:
      // an empty verdict list against a non-empty offer means "offered but not benched", which is
      // itself a reportable outcome.
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
          // What the merge and the repair did, in the same row as the number they produced. A
          // reader who sees a neutral `delta_pct` needs to know whether the recorded value was the
          // one that ran (`overrides`) and whether all of it ran (`applied_partial`); without them
          // the row reads as a verdict on the record when it may be a verdict on the pairing.
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
        // Split out of `rejected` for the reference section below, but deliberately still counted
        // inside it: for the roles that come next, "lost the A/B" and "never ran" are both "do not
        // re-propose this verbatim". The distinction matters to the STORE, not to the Architect.
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
          // `exact` is exact on the IDENTITY, which encodes no server flags, env or kv-cache
          // dtype — a record measured against a slower baseline reproduces smaller here for that
          // reason alone.
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
          // The plan's fourth requirement, made concrete: a top-N entry that could not be
          // reproduced is not discarded, it is handed forward WITH its reproduction material. A
          // lead the next role cannot open is a lead it will not use, so the launch script and the
          // patch paths are spelled out here rather than left to be rediscovered in the bundle.
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

        // The channel that always fires. Terse on purpose — it goes into `## Inputs` of two roles,
        // and a verdict that has to be waded through is a verdict that gets skimmed.
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

  if (searchClosed('Profile and Strategize')) {
    kernelQueue = [];
    headQueue = [];
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
  profile = { profile_topN_json: ST.profile_topn_json || '' };
  strategy = { config_directions: ST.config_directions || [] };
  kernelQueue = ST.kernelQueue || [];
  headQueue = admitHeads(ST.headQueue || [], 'resume');
  log(`Loaded carried state: EVAL_DIR=${EVAL_DIR}, baseline ${BASELINE_TPUT}, flags='${curFlags}', env='${curEnv}', ${headQueue.length} head + ${kernelQueue.length} kernel candidates.`);
}

