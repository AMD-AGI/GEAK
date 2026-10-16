// ===========================================================================
// PHASE: Finalize + Report + Validate  (gated)
// ===========================================================================
let allAccepted = acceptedHeads.concat(acceptedKernels);
let finalize = null, report = null, validation = null;
let finalTput = curTput, finalSpeedup = BASELINE_TPUT ? curTput / BASELINE_TPUT : 1.0;
let validatedOk = false;   // did the independent Validate produce a usable (positive) number?

// --- Fix C: EVERY incomplete A/B must be finished (both legs) before final ----
// Completeness guarantee (general, not per-kernel): an A/B that measured only the
// reference leg (gate:'incomplete' / ab_complete:false) is NOT a result. Before
// finalizing we
//   (1) RECONSTRUCT from disk every overlay whose integrate_result shows the A/B
//       never completed both legs — so incompletes from ANY integrate branch, or
//       from a prior crashed/killed/resumed run (the JS has no fs and no carried
//       state across a process restart), are recovered, then
//   (2) drive EVERY pending integration to a complete ref+cand measurement
//       (runIntegrateBothLegs re-invokes the integrator to run the missing leg).
// This replaces the old "finish only the single best, and only when nothing else
// was accepted" gate, which stranded sibling incompletes (e.g. an env-config
// head) at ref-only whenever any other head/kernel had been accepted. Keys ONLY
// off the stable overlay/cand_*/integrate_result.json layout. Bounded by
// AB_FINISH_RETRIES + each agent's agentBounded guard + the outer runner budget
// + TIME_FINAL_DEADLINE_HIT (this gate must NEVER spend the final reserve; see below).
if (want('final')) {
  // Issue #429: reclaim leftover process-local capture dirs before Finalize burns more disk.
  // Run even when TIME_FINAL_DEADLINE_HIT — reclaim is cheap local FS work, not a bench.
  try {
    const { execFileSync } = require('child_process');
    const out = execFileSync('python3', [
      `${WORKFLOW_DIR}/scripts/capture_shapes.py`,
      '--reclaim-workspace', EVAL_DIR,
      '--workspace-budget', CAPTURE_WORKSPACE_BUDGET,
    ], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 300000 });
    log(`Finalize: capture workspace reclaim done (${CAPTURE_WORKSPACE_BUDGET} budget).`);
    try {
      const tel = JSON.parse(out);
      if (tel.over_budget) {
        log(`Finalize: WARNING authoritative oracles over workspace budget ` +
          `(${tel.authoritative_oracle_bytes} bytes > ${CAPTURE_WORKSPACE_BUDGET}).`);
      }
    } catch (_) { /* telemetry parse best-effort */ }
  } catch (reclaimErr) {
    log(`Finalize: capture workspace reclaim skipped (${reclaimErr && reclaimErr.message ? reclaimErr.message : reclaimErr})`);
  }
  // (0) The reserve is not ours. Each iteration below boots a server and runs two benches, so past
  // the boundary there is no time to finish even one: skip the whole gate, scan agent included.
  // Pending entries stay in pendingIntegrations and reach the caller — deferred, not discarded.
  if (TIME_FINAL_DEADLINE_HIT) {
    log(`Finalize-gate: SKIPPED entirely — the final reserve has already begun. ` +
      `${pendingIntegrations.length} pending A/B(s) deferred to the caller (surfaced in pending_integrations).`);
  } else {
  // (1) Disk-reconstruct incomplete A/Bs. The JS has no fs, so a read-only Bash
  // agent enumerates the overlay layout and reports which A/Bs never completed.
  const RECON_SCHEMA = obj({ incomplete: { type: 'array', items: obj({
    short_name: { type: 'string' }, overlay_dir: { type: 'string' },
    winner_kind: { type: 'string' }, apply_env: { type: 'string' }, apply_flags: { type: 'string' },
    op_kind: { type: 'string' }, target_callable: { type: 'string' },
    isolated: { type: 'number' }, pct_gpu_time: { type: 'number' },
    ref_present: { type: 'boolean' }, cand_present: { type: 'boolean' },
  }, ['short_name', 'overlay_dir']) } }, ['incomplete']);
  const scan = await safeAgent(
    `You are a read-only filesystem scanner. Do NOT run any benchmark, server, or A/B. ` +
    `Enumerate the directories ${EVAL_DIR}/overlay/cand_* and (if it exists) ` +
    `${EVAL_DIR}/final/overlay/cand_*. For EACH such cand_<name> dir, read its ` +
    `integrate_result.json if present. Classify the A/B as INCOMPLETE when: the file is ` +
    `MISSING, OR gate=="incomplete", OR (ab_complete is false/absent AND there is no usable ` +
    `cand/bench_summary.json in the dir). A summary is usable only when status=="complete" and ` +
    `usable_for_acceptance==true. Treat it as COMPLETE (skip it) when ab_complete:true ` +
    `OR gate is one of {accepted,stack,rejected} AND the candidate summary is usable. ` +
    `For each INCOMPLETE dir, emit one object with: short_name (the dir name minus the ` +
    `"cand_" prefix), overlay_dir (absolute path), and from integrate_result.json (use "" / ` +
    `null when absent): winner_kind, apply_env, apply_flags, op_kind, ` +
    `target_callable (or target_file), isolated (= isolated_speedup), pct_gpu_time, plus ` +
    `ref_present (true only if ref/bench_summary.json is usable) and cand_present (true only if ` +
    `cand/bench_summary.json is usable). Return ONLY compact JSON {"incomplete":[...]} (empty array if none).`,
    { phase: FINALIZE_GATE_PHASE, label: 'scan-incomplete-ab', schema: RECON_SCHEMA });
  const known = new Set(pendingIntegrations.map((p) => p.short_name));
  for (const it of ((scan && scan.incomplete) || [])) {
    if (!it || !it.short_name || known.has(it.short_name)) continue;
    known.add(it.short_name);
    pendingIntegrations.push({
      short_name: it.short_name, track: 'head', winner_kind: it.winner_kind || '',
      apply_env: it.apply_env || '', apply_flags: it.apply_flags || '',
      op_kind: it.op_kind || '', backend: '', isolated: it.isolated || 0,
      inputs: {
        EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
        KERNEL_RESULT: {
          short_name: it.short_name, op_kind: it.op_kind || '', winner_kind: it.winner_kind || '',
          target_callable: it.target_callable || '', apply_env: it.apply_env || '',
          apply_flags: it.apply_flags || '', code_patch: '', tuning_artifact: '',
          verified_isolated_speedup: it.isolated || 0, pct_gpu_time: it.pct_gpu_time,
          parity_note: 'expected_close',
        },
        // The candidate overlay is already built on disk; the integrator benches
        // it directly on resume (RESUME_AB) — it does NOT rebuild it.
        CAND_OVERLAY_DIR: it.overlay_dir, RESUME_AB: true, SKILL_DIR: WORKFLOW_DIR,
      },
    });
    log(`Finalize-gate: disk-reconstructed incomplete A/B "${it.short_name}" ` +
      `(winner_kind=${it.winner_kind || '?'}, ref=${!!it.ref_present}, cand=${!!it.cand_present}); queued to finish.`);
  }

  // (2) Drive EVERY pending integration to a complete ref+cand measurement — but never
  // past the reserve boundary. Highest isolated speedup first, so if the clock cuts the
  // queue short the most valuable A/B is the one that got finished.
  pendingIntegrations.sort((a, b) => (b.isolated || 0) - (a.isolated || 0));
  const stillIncomplete = [];
  while (pendingIntegrations.length) {
    // Break, not shift, so unfinished entries stay in pendingIntegrations for wfReturn.
    if (TIME_FINAL_DEADLINE_HIT) {
      log(`Finalize-gate: STOPPING with ${pendingIntegrations.length} A/B(s) unfinished ` +
        `(${pendingIntegrations.map((q) => q.short_name).join(', ')}) — the final reserve has begun and ` +
        `Finalize/Report/Validate must run. Deferred, not discarded: they are surfaced in pending_integrations.`);
      break;
    }
    const p = pendingIntegrations.shift();   // pop one per iteration => guaranteed termination
    log(`Finalize-gate: finishing pending A/B (${p.short_name}, ${(p.isolated || 0).toFixed(2)}x isolated); ` +
      `${pendingIntegrations.length} more queued.`);
    const integ = await runIntegrateBothLegs(
      'Finish the e2e A/B for this verified-isolated win. Run BOTH legs (reference + candidate) to ' +
      'completion, then return accepted/stack/rejected with ab_complete:true.',
      // Pin the CURRENT carried overlay/flags/env/throughput so the A/B is
      // measured against the latest accepted baseline.
      { ...p.inputs, CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_THROUGHPUT: curTput },
      `finish-integrate ${p.short_name}`, FINALIZE_GATE_PHASE);
    if (abDone(integ) && integAccepted(integ, p.pct_gpu_time, p.isolated) && integ.e2e_throughput_tok_s > curTput) {
      curOverlay = integ.accepted_overlay || curOverlay;
      if (p.track === 'head') {
        if (p.winner_kind === 'env' && p.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + p.apply_env;
        if (p.winner_kind === 'flag' && p.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + p.apply_flags;
        bankAccepted(acceptedHeads, { short_name: p.short_name, op_kind: p.op_kind, backend: p.backend, kind: p.winner_kind, ...e2eFrom(integ), isolated: p.isolated, pct_gpu_time: p.pct_gpu_time }, krOf(p.inputs));
      } else {
        bankAccepted(acceptedKernels, { short_name: p.short_name, backend: p.backend || '', kind: 'patch', ...e2eFrom(integ), isolated: p.isolated, pct_gpu_time: p.pct_gpu_time }, krOf(p.inputs));
      }
      curTput = integ.e2e_throughput_tok_s;
      finalTput = curTput; finalSpeedup = BASELINE_TPUT ? curTput / BASELINE_TPUT : 1.0;
      log(`Finalize-gate: pending win ${p.short_name} ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: p.short_name, isolated_speedup: p.isolated, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: 'finished at finalize-gate' });
    } else if (abDone(integ)) {
      log(`Finalize-gate: pending win ${p.short_name} REJECTED after a COMPLETED A/B (${integ.reason || integ.gate}).`);
      history.ledger.push({ direction: p.short_name, isolated_speedup: p.isolated, e2e_delta_pct: integ.e2e_delta_pct || 0, verdict: 'dead_end', lesson: integ.reason || 'no e2e gain' });
    } else {
      stillIncomplete.push(p.short_name);
      log(`Finalize-gate: pending win ${p.short_name} A/B STILL incomplete after ${AB_FINISH_RETRIES} finish-retries ` +
        `(${integ ? integ.reason || integ.gate : 'null/timeout'}); hard fault — surfacing it.`);
      history.ledger.push({ direction: p.short_name, isolated_speedup: p.isolated, verdict: 'incomplete', lesson: integ ? integ.reason || 'A/B could not complete both legs' : 'integrate degraded/timed out' });
    }
  }
  if (stillIncomplete.length) log(`Finalize-gate: ${stillIncomplete.length} A/B(s) could not complete both legs even after retries: ${stillIncomplete.join(', ')}.`);
  }
}
allAccepted = acceptedHeads.concat(acceptedKernels);   // refresh after Fix C may have banked a pending win
if (allAccepted.length) {
  await requireE2EValidationCheckpoint('overlay/accepted_stack/e2e_validation.json', {
    phase: 'Overlay', validation_level: 'integrator', gate: 'accepted',
    validation_status: 'accepted_intermediate',
    baseline_throughput_tok_s: BASELINE_TPUT, final_throughput_tok_s: curTput,
    throughput_speedup: BASELINE_TPUT ? curTput / BASELINE_TPUT : 1,
    baseline_config: { flags: INIT_FLAGS, env: INIT_ENV },
    accepted_config: { flags: curFlags, env: curEnv, effective_config_digest: EFFECTIVE_CONFIG_DIGEST },
    accepted_kernels: acceptedKernels, accepted_heads: acceptedHeads, final_patch: [],
    final_overlay: { path: curOverlay },
    final_launch_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
    bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
    measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
      workload: WORKLOAD, source_artifact: `${EVAL_DIR}/overlay`,
      acceptance: { gain_exceeds_noise: true, correctness_passed: true, noise_floor_pct: NOISE_BAND } },
    stack: { kernel_slots: allAccepted.map((k) => ({
      kernel_slot: k.target_callable || k.short_name || k.name || '',
      selected: true, candidate_id: k.short_name || k.name || '',
    })) },
    replay: { asset_paths: [`${EVAL_DIR}/bench_e2e.sh`, curOverlay].filter(Boolean) },
    integrity: { checkpoint_assets: [] },
  });
}
if (want('final')) {
  if (TIME_BUDGET_MS != null) log(`[time-budget] entering the final phase with ~${remainingMin()}min of the ${Math.round(TIME_BUDGET_MS / 60000)}min budget left (reserve was ${Math.round(FINAL_RESERVE_MS / 60000)}min).`);
  FINAL_PHASE_STARTED = true;   // the one place the reserve exemption is granted — by position, not label
  phase('Finalize');
  finalize = await safeAgent(
    roleAgent('e2e_integrator', 'finalize', 'Assemble the final overlay + patch + launch script bundle.', {
      EVAL_DIR, FINAL_OVERLAY: curOverlay, ACCEPTED_FLAGS: curFlags, ACCEPTED_ENV: curEnv,
      ACCEPTED_KERNELS: allAccepted, BASELINE_THROUGHPUT: BASELINE_TPUT, SKILL_DIR: WORKFLOW_DIR,
      ...TUNING_FINALIZE_INPUTS,
    }),
    { phase: 'Finalize', label: 'e2e_integrator:finalize', schema: FINALIZE_SCHEMA });
  finalTput = (finalize && finalize.final_throughput_tok_s) || curTput;
  if (finalize && finalTput > BASELINE_TPUT * (1 + NOISE_BAND / 100)) {
    await requireE2EValidationCheckpoint('final/e2e_validation.json', {
      phase: 'Finalize', validation_level: 'final_pair', gate: 'accepted',
      validation_status: 'provisional_final_pair',
      baseline_throughput_tok_s: BASELINE_TPUT, final_throughput_tok_s: finalTput,
      throughput_speedup: BASELINE_TPUT ? finalTput / BASELINE_TPUT : 1,
      baseline_config: { flags: INIT_FLAGS, env: INIT_ENV },
      accepted_config: { flags: curFlags, env: curEnv, effective_config_digest: EFFECTIVE_CONFIG_DIGEST },
      accepted_kernels: acceptedKernels, accepted_heads: acceptedHeads,
      final_patch: [finalize.final_patch].filter(Boolean),
      final_overlay: { path: finalize.final_overlay || curOverlay },
      final_launch_script: { path: finalize.final_launch_script || '' },
      bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
        workload: WORKLOAD, source_artifact: `${EVAL_DIR}/final/bench/bench_summary.json`,
        acceptance: { gain_exceeds_noise: true, correctness_passed: true, noise_floor_pct: NOISE_BAND } },
      stack: { kernel_slots: allAccepted.map((k) => ({
        kernel_slot: k.target_callable || k.short_name || k.name || '',
        selected: true, candidate_id: k.short_name || k.name || '',
      })) },
      replay: { asset_paths: [finalize.final_patch, finalize.final_overlay,
        finalize.final_launch_script, `${EVAL_DIR}/bench_e2e.sh`].filter(Boolean) },
      integrity: { checkpoint_assets: [] },
    });
  }

  phase('Report');
  report = await safeAgent(
    roleAgent('system_architect', 'report',
      'Write concise architect_report.md and final_report.md in English: measured results, ' +
      'accepted changes, limits and artifact links. Include a "## Knowledge-base recall" section ' +
      'built from KB_RECALL — see your role file. Write it even when nothing was recalled: report ' +
      'the exact canonical ids that were tried and the read_reason. On an exact-lookup store a miss ' +
      'and a never-recorded page are the same 404, so the address asked is the finding, and a reader ' +
      'who cannot see it cannot tell "no prior art" from "prior art one segment away". Zero candidates ' +
      'is NOT the same as an empty page: report KB_RECALL.e2e.curation (scanned/retired/' +
      'same_direction_collapsed/below_min_speedup, and its own read_plane) with both floors.', {
      EVAL_DIR, HISTORY: history, BASELINE_THROUGHPUT: BASELINE_TPUT, FINAL_THROUGHPUT: finalTput,
      KB_RECALL,
      ACCEPTED_CONFIG: { flags: curFlags, env: curEnv }, ACCEPTED_KERNELS: allAccepted,
      ACCEPTED_HEADS: acceptedHeads, FLAGGED_HEADS: flaggedHeads, MILESTONES: milestone, BUDGET_USED: dispatched, BUDGET, MIN_KERNEL_TASKS,
      PROFILE_TOPN: profile ? profile.profile_topN_json : '', WORKLOAD, MODEL_NAME, SKILL_DIR: WORKFLOW_DIR,
      ...ANALYSIS_SKILL_INPUTS, ...TUNING_REPORT_INPUTS,
    }),
    { phase: 'Report', label: 'architect:report', schema: REPORT_SCHEMA });

  // Validate is sized into FINAL_RESERVE_MS, so there should be room for it here; it is not time-GATED
  // (attempting it beats skipping when the clock has drifted). If the hard kill lands mid-bench the loss is
  // bounded — Report already wrote architect_report.md + final_report.md, and run_e2e.py falls back
  // (director_e2e_validation.json → best overlay's integrate_result.json), so a real win still reaches the caller.
  phase('Validate');
  validation = await safeAgent(
    roleAgent('director', 'validate', 'Independently re-measure throughput + parity; arbitrate; then reconcile the report with the validated numbers.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], BASELINE_THROUGHPUT: BASELINE_TPUT, NOISE_BAND_PCT: NOISE_BAND,
      BASELINE_FLAGS, BASELINE_ENV, BASELINE_OVERLAY,
      FINAL_OVERLAY: (finalize && finalize.final_overlay) || curOverlay,
      FINAL_FLAGS: { flags: curFlags, env: curEnv },
      CLAIMED_THROUGHPUT: finalTput, WORKLOAD, APPLY_TO_ORIGINAL,
      MEASUREMENT_MODE: VALIDATION_MEASUREMENT_MODE,
      MEASUREMENT_PURPOSE: 'validation', REPLICAS: VALIDATION_SAMPLES,
      SKILL_DIR: WORKFLOW_DIR, ...ACCURACY_INPUTS,
      // The Report phase already wrote these files with the Finalize-bundle bench (the Director had not
      // run yet). After validation the Director MUST review + rewrite their headline throughput / speedup
      // / TTFT / TPOT (and status/parity) to its authoritative same-session numbers, so report-vs-director
      // can never disagree. Paths default to the standard EVAL_DIR names if the report result is absent.
      ARCHITECT_REPORT: (report && report.report_path) || `${EVAL_DIR}/architect_report.md`,
      FINAL_REPORT: `${EVAL_DIR}/final_report.md`,
    }),
    { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });
  // A Validate that did NOT produce a usable number (e.g. its server crashed in
  // engine-core init) must NEVER erase the accepted same-session A/B win we
  // already carry in finalTput/curTput. Trust the Director's independent number
  // ONLY when it is a real positive measurement; otherwise fall back to the
  // carried best-accepted throughput so a real, parity-checked win is never
  // reported as 0 / no_gain downstream.
  validatedOk = !!(validation && validation.director_verified_throughput_tok_s > 0 && validation.throughput_speedup > 0);
  finalSpeedup = validatedOk ? validation.throughput_speedup : (BASELINE_TPUT ? finalTput / BASELINE_TPUT : finalSpeedup);
  log(`COMPLETE. ${MODEL_NAME}: ${BASELINE_TPUT} -> ${validatedOk ? validation.director_verified_throughput_tok_s : finalTput} tok/s ` +
    `(${finalSpeedup ? finalSpeedup.toFixed(3) : '?'}x, status ${validation ? validation.validation_status : '?'}` +
    `${validation && !validatedOk ? '; Validate produced no number — using carried accepted A/B win' : ''}). Results in ${EVAL_DIR}`);

  // --- FINAL experience reconciliation (post-gate) -----------------------
  // The per-milestone update_experience (in the Milestone loop above) is the ONLY
  // curate step, and it runs on that milestone's ledger. But a head/kernel whose
  // e2e A/B was still INCOMPLETE at milestone time is recorded as "Integrator
  // pending", and the Finalize-gate (Fix C) that actually CONFIRMS its e2e win runs
  // AFTER the last milestone curate — as does the Director's Validate. Without a
  // pass here, those cards stay frozen at "pending" and never receive the verified
  // gated number (see the 20260812 Qwen3.5-27B run: the +16.1% head card was left
  // at "e2e transfer NOT yet gated"). Re-curate ONCE now, with the authoritative
  // post-Validate numbers, so finalize-gate confirmations are written back.
  if (allAccepted.length && E2E_LEARNED_KB_ENABLED) {
    const verifiedTput = validatedOk ? validation.director_verified_throughput_tok_s : finalTput;
    await safeAgent(
      roleAgent('system_architect', 'update_experience',
        'FINAL RECONCILIATION (post-gate): re-curate knowledge/learned/ per learned/README.md. For every ' +
        'entry in ACCEPTED_HEADS / ACCEPTED_KERNELS whose card was written earlier THIS run but still reads ' +
        '"Integrator pending" / "e2e gate pending" (or carries no e2e number), MERGE the VERIFIED gated ' +
        'result into that card: set effect from the entry\'s e2e_delta_pct plus the run FINAL_THROUGHPUT / ' +
        'FINAL_SPEEDUP / VALIDATION_STATUS / OUTPUT_PARITY, and raise confidence per learned/README.md now ' +
        'that the e2e gate passed. Do NOT blind-append and do NOT touch rejected/dead-end cards.', {
        ROUND: 'final', FINAL_GATE: true, EVAL_DIR, MODEL_NAME, SKILL_DIR: WORKFLOW_DIR,
        ACCEPTED_HEADS: acceptedHeads, ACCEPTED_KERNELS: acceptedKernels,
        BASELINE_THROUGHPUT: BASELINE_TPUT, FINAL_THROUGHPUT: verifiedTput, FINAL_SPEEDUP: finalSpeedup,
        VALIDATION_STATUS: validatedOk ? validation.validation_status
          : (validation ? validation.validation_status || 'flagged' : 'unknown'),
        OUTPUT_PARITY: validation ? validation.output_parity : 'unknown',
        MILESTONE_RESULTS: history.ledger.filter((e) => e && e.lesson === 'finished at finalize-gate'),
        PRIOR_HISTORY: history,
        ...ANALYSIS_SKILL_INPUTS,
      }),
      { phase: 'Validate', label: 'architect:experience final', schema: EXPERIENCE_SCHEMA });
    log('Finalize: re-curated knowledge/learned/ with verified post-gate numbers (pending -> verified).');
  }
} else {
  log(`Phase(s) [${PHASES.join(',')}] done. Carried throughput ${curTput} tok/s. Pass the returned 'state' to the next phase invocation.`);
}

// State to carry into the NEXT phase invocation (args.state) when driving phase-by-phase.
const carryState = {
  backend: BACKEND,
  eval_dir: EVAL_DIR, model_name: MODEL_NAME, baseline_throughput_tok_s: BASELINE_TPUT,
  noise_band_pct: NOISE_BAND, flags: curFlags, env: curEnv, overlay: curOverlay, throughput: curTput,
  baseline_flags: BASELINE_FLAGS, baseline_env: BASELINE_ENV, baseline_overlay: BASELINE_OVERLAY,
  profile_topn_json: profile ? profile.profile_topN_json : '',
  config_directions: (strategy && strategy.config_directions) || [],
  headQueue, kernelQueue, accepted_heads: acceptedHeads, flagged_heads: flaggedHeads, accepted_kernels: acceptedKernels,
  // Full tuning-phase result, so a phase-by-phase resume does not re-run the tuning loop and the Report
  // phase still has the attribution numbers when it runs in a later invocation.
  ...(tuning ? { tuning } : {}),
  // Carry pending (verified-isolated, A/B-incomplete) wins WITH their inputs so a
  // resumed phase run can finish their A/B instead of re-discovering them.
  pending_integrations: pendingIntegrations,
  history,
};

// What this run got from the KB, and — separately — what it added on its own. The split is the
// point: a warm-started run's headline speedup is partly inherited, and reporting the whole of it as
// this run's work is how a knowledge base starts flattering itself. BASELINE_TPUT stays the true
// cold baseline throughout, so `throughput_speedup` above is still measured against the real floor.
const kbWarmStart = (E2E_WARM_START_ON && KB_DIMS) ? {
  identity: KB_DIMS, plane: E2E_KB_PLANE, read_plane: KB_READ_PLANE, mode: E2E_WARM_START,
  reference_dir: KB_REF_DIR,
  adopted_throughput_tok_s: kbSeedTput || null,
  adopted_kernels: kbSeedKernels.map(k => k.name).filter(Boolean),
  // The gain that is genuinely THIS run's: everything after the warm start's own contribution. Null
  // when nothing was adopted, because then the ordinary speedup already answers the question.
  incremental_speedup: kbSeedTput ? Number((finalTput / kbSeedTput).toFixed(6)) : null,
  // The full recall ledger, both planes. Same object the Report role was handed, so the prose in
  // final_report.md and whatever a downstream consumer reads cannot disagree about what was
  // recalled or how it fared.
  recall: KB_RECALL,
} : null;
// Attribution block for the standalone tuning phase. Because the phase measured its OWN interleaved
// pre/post legs, its contribution can be expressed both absolutely (delta%) and as a SHARE of the run's
// total gain — which is the number "how much did the tuning actually help?" is asking for. share is null
// (not 0) when the run had no net gain to apportion, so a caller never divides by a meaningless total.
function tuningReturn() {
  if (!TUNING_SKILLSET_ENABLED) return { enabled: false };
  const base = {
    enabled: true, skillset_dir: TUNING_SKILLSET_DIR, kb_enabled: TUNING_KB_ENABLED, ran: !!tuning,
  };
  if (!tuning) return { ...base, gate: want('tune') ? 'not_run' : 'phase_not_selected' };
  const finalT = validatedOk ? validation.director_verified_throughput_tok_s : finalTput;
  const totalGain = (finalT || 0) - (BASELINE_TPUT || 0);
  const tuningGain = (tuning.post_tune_throughput_tok_s || 0) - (tuning.pre_tune_throughput_tok_s || 0);
  const banked = tuning.gate === 'accepted';
  return {
    ...base,
    gate: tuning.gate,
    mode: tuning.mode || '',
    skills_used: tuning.skills_used || [],
    ops_tuned: (tuning.ops_tuned || []).map((o) => ({
      op: o.op || o.short_name || '', backend: o.backend || '',
      isolated_speedup: o.isolated_speedup || 0, engaged: o.engaged === true, artifact: o.artifact || '',
    })),
    artifacts: tuning.artifacts || [],
    // How the win reaches production. Consumers that need to reproduce the tuning outside this run read
    // deploy_bundle (its deploy.sh is idempotent); final_launch.sh already invokes it.
    deploy_bundle: tuning.deploy_bundle || '', deploy_verified: tuning.deploy_verified === true,
    cache_invalidation: tuning.cache_invalidation || [], live_tree_files: tuning.live_tree_files || [],
    apply_overlay: tuning.apply_overlay || '',
    // Finalize's independent answer to "did the tuning actually make it into the shipped bundle, and
    // does it still engage there?". null when Finalize did not run (a phase-scoped invocation).
    in_final_bundle: finalize ? (finalize.tuning_in_bundle === true) : null,
    final_bundle_engagement_recheck: (finalize && finalize.tuning_engagement_recheck) || '',
    apply_env: tuning.apply_env || '', apply_flags: tuning.apply_flags || '',
    correctness_gate: tuning.correctness_gate || 'unknown',
    engagement_verified: tuning.engagement_verified === true,
    engagement_evidence: tuning.engagement_evidence || '',
    pre_tune_throughput_tok_s: tuning.pre_tune_throughput_tok_s || 0,
    post_tune_throughput_tok_s: tuning.post_tune_throughput_tok_s || 0,
    noise_floor_pct: tuning.noise_floor_pct || 0,
    tuning_delta_pct: tuning.tuning_delta_pct || 0,
    tuning_speedup: tuning.tuning_speedup ||
      (tuning.pre_tune_throughput_tok_s ? (tuning.post_tune_throughput_tok_s / tuning.pre_tune_throughput_tok_s) : 1.0),
    ab_interleaved: tuning.ab_interleaved === true,
    ab_complete: tuning.ab_complete !== false,
    // Share of the run's TOTAL gain attributable to the tuning phase (banked wins only).
    share_of_total_gain_pct: (banked && totalGain > 0) ? +((tuningGain / totalGain) * 100).toFixed(2) : null,
    report_path: tuning.report_path || `${EVAL_DIR}/tuning/tuning_report.md`,
    summary: tuning.summary || '', reason: tuning.reason || '',
  };
}

const wfReturn = {
  // schema_version pins the CONTRACT shape run_e2e.py reads. Bump only on a
  // breaking change to the keys below; run_e2e.py keys its canonical-artifact
  // read off this so it never silently mis-parses a future shape.
  schema_version: 1,
  mode: 'e2e',
  fast_mode: FAST_MODE,   // true => ConfigSweep + Milestone skipped; HeadKernel-only within the time budget
  deep_mode: DEEP_MODE,   // true => HeadKernel runs the long cross-backend co-optimization scheduler (20h)
  backend: BACKEND,
  phases_run: PHASES,
  eval_dir: EVAL_DIR,
  model_name: MODEL_NAME,
  baseline_throughput_tok_s: BASELINE_TPUT,
  // Trust the Director's independent number ONLY when it is a real positive
  // measurement (validatedOk). A crashed/degenerate Validate falls back to the
  // carried accepted same-session A/B win — never 0 — so a real, parity-checked
  // gain is not silently demoted to no_gain by run_e2e.py.
  final_throughput_tok_s: validatedOk ? validation.director_verified_throughput_tok_s : finalTput,
  throughput_speedup: finalSpeedup,
  validation_status: validatedOk ? validation.validation_status
    : (validation ? `${validation.validation_status || 'flagged'}_no_number_used_carried_ab`
       : (want('final') ? 'unknown' : 'phase_partial')),
  output_parity: validation ? validation.output_parity : 'unknown',
  accepted_config: { flags: curFlags, env: curEnv },
  accepted_kernels: acceptedKernels,
  accepted_heads: acceptedHeads,
  // Standalone tuning-skillset phase: its own measured pre/post legs plus the share of the run's TOTAL
  // gain those legs account for. Reported separately from accepted_heads/accepted_kernels precisely
  // because the phase is standalone — "how much did tuning buy" is answerable, not folded into the total.
  tuning_skillset: tuningReturn(),
  // Verified-isolated wins whose e2e A/B never completed (timeout/hang mid-gate).
  // A REAL isolated speedup that simply ran out of A/B time — surfaced (slim) so
  // the caller can resume/finish it, never silently discarded as a rejection.
  pending_integrations: pendingIntegrations.map(p => ({
    short_name: p.short_name, track: p.track, isolated: p.isolated,
    pct_gpu_time: p.pct_gpu_time, partial: p.partial || null,
  })),
  flagged_heads: flaggedHeads,   // dominant heads surfaced but not optimized (harness/extract/no-candidate) — never silently dropped
  config_tune_enabled: CONFIG_TUNE_ENABLED,
  head_budget: HEAD_BUDGET,
  head_used: headDispatched,
  milestones: milestone,
  budget_used: dispatched,
  budget_total: BUDGET,
  final_overlay: (validation && validation.final_overlay) || (finalize && finalize.final_overlay) || curOverlay,
  // FINALIZE_SCHEMA has always defined final_patch and the Finalizer has always returned it, but it
  // was never surfaced here — so the single diff carrying the whole run's win reached neither the
  // report nor the KB (e2e_store.py's _ARTIFACT_KEYS looks up exactly result["final_patch"]).
  // final_overlay cannot stand in for it: that is a DIRECTORY, and the store's os.path.isfile filter
  // drops it, so a run with no final_patch uploads no reproducible code at all.
  ...kbResultMetadata(),
  final_patch: (finalize && finalize.final_patch) || '',
  final_launch_script: (validation && validation.final_launch_script) || (finalize && finalize.final_launch_script) || '',
  report_path: report ? report.report_path : `${EVAL_DIR}/architect_report.md`,
  // Conditional spread, never an unconditional key: with the feature off this contributes no own
  // property and both the returned object and the persisted file are byte-identical to before.
  ...(kbWarmStart ? { kb_warm_start: kbWarmStart } : {}),
  // Emitted independently of kb_warm_start. kbWarmStart is null whenever KB_DIMS was never
  // established (a preflight that produced no gfx), but the KERNEL lanes can still have recalled by
  // then — folding recall only into kbWarmStart would drop exactly the runs where knowing what was
  // recalled matters most. Absent entirely when nothing was recalled on either plane.
  ...((KB_RECALL.e2e || KB_RECALL.kernel.length) ? { kb_recall: KB_RECALL } : {}),
  state: carryState,
};

