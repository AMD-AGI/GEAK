// ===========================================================================
// PHASE: Config sweep (Config Tuner) — FIRST, reshapes the profile
// ===========================================================================
// A carried phase-partial state still wins: it is a measurement this run already made. kbSeedTput is
// 0 when Module A did not run or adopted nothing, so `0 || BASELINE_TPUT === BASELINE_TPUT`.
let curTput = ST.throughput || kbSeedTput || BASELINE_TPUT;
if (want('config') && CONFIG_TUNE_ENABLED && strategy && (strategy.config_directions || []).length &&
    !searchClosed('ConfigSweep')) {
  phase('ConfigSweep');
  const sweep = await safeAgent(
    roleAgent('config_tuner', 'sweep', 'Sweep the ranked config axes one at a time; keep wins.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, BASELINE_THROUGHPUT: BASELINE_TPUT,
      NOISE_BAND_PCT: NOISE_BAND, CONFIG_DIRECTIONS: strategy.config_directions,
      CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_OVERLAY: curOverlay,
      MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
      SKILL_DIR: WORKFLOW_DIR, ...KB_REF_INPUTS,
    }),
    { phase: 'ConfigSweep', label: 'config_tuner:sweep', schema: SWEEP_SCHEMA });
  if (sweep && sweep.best_throughput_tok_s > curTput) {
    curFlags = sweep.accepted_flags || curFlags;
    curEnv = sweep.accepted_env || curEnv;
    curTput = sweep.best_throughput_tok_s;
    await requireE2EValidationCheckpoint('config/e2e_validation.json', {
      phase: 'ConfigSweep', validation_level: 'config_sweep', gate: 'accepted',
      validation_status: 'accepted_config',
      baseline_throughput_tok_s: BASELINE_TPUT, final_throughput_tok_s: curTput,
      throughput_speedup: BASELINE_TPUT ? curTput / BASELINE_TPUT : 1,
      baseline_config: { flags: INIT_FLAGS, env: INIT_ENV },
      accepted_config: { flags: curFlags, env: curEnv, effective_config_digest: EFFECTIVE_CONFIG_DIGEST },
      accepted_kernels: [], accepted_heads: [], final_patch: [], final_overlay: { path: curOverlay },
      final_launch_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
        workload: WORKLOAD, source_artifact: `${EVAL_DIR}/config/sweep_results.json`,
        acceptance: { gain_exceeds_noise: true, correctness_passed: true, noise_floor_pct: NOISE_BAND } },
      stack: { config_layers: (sweep.trials || []).filter((t) => t && t.kept === true) },
      replay: { asset_paths: [`${EVAL_DIR}/bench_e2e.sh`, `${EVAL_DIR}/config/sweep_results.json`] },
      integrity: { checkpoint_assets: [] },
    });
    log(`Config sweep accepted. throughput ${curTput} tok/s (${(curTput / BASELINE_TPUT).toFixed(3)}x). Re-profiling.`);
    if (!searchClosed('post-config re-profile and re-strategize')) {
      // Re-profile: config changed which kernels dominate.
      profile = await safeAgent(
        roleAgent('profiler', 'reprofile', 'Re-profile after the config sweep.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 'config',
          OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Profile', label: 'profiler:post-config', schema: PROFILE_SCHEMA });
      // Re-strategize the kernel queue against the new profile.
      const restrat = await safeAgent(
        roleAgent('system_architect', 'strategize', 'Re-route after config changed the landscape.', {
          EVAL_DIR, PROFILE_TOPN: profile ? profile.profile_topN_json : '', BASELINE_THROUGHPUT: curTput,
          WORKLOAD, BUDGET, HEAD_THRESHOLD_PCT, CONFIG_TUNE_ENABLED: false, SKILL_DIR: WORKFLOW_DIR,
          ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Strategize', label: 'architect:re-strategize', schema: STRATEGY_SCHEMA });
      if (restrat && restrat.kernel_candidates) kernelQueue = restrat.kernel_candidates.slice();
      if (restrat && restrat.head_candidates)
        headQueue = applyOpIdentityGuard(restrat.head_candidates.slice(), 're-strategize');
      // re-strategize may have (re)routed flydsl -> provision it (idempotent; no-op if already done).
      await ensureFlydslGate();
    }
  } else {
    log(`Config sweep found no win above the noise band.`);
  }
}

