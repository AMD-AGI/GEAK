// ===========================================================================
// PHASE: Finalize + Report + Validate  (gated)
// ===========================================================================
let allAccepted = acceptedHeads.concat(acceptedKernels);
let finalize = null, report = null, validation = null;
let finalTput = curTput, finalSpeedup = BASELINE_TPUT ? curTput / BASELINE_TPUT : 1.0;
let validatedOk = false;   // did the independent Validate produce a usable (positive) number?
let validateSkip = '', finalFromAb = false;

// --- Fix C: EVERY incomplete A/B must be finished (both legs) before final ----
// A reference-only A/B is not a result. (1) Reconstruct from disk (overlay/cand_*/
// integrate_result.json) every A/B that never completed, including from a killed or resumed run;
// (2) finish each pending one (runIntegrateBothLegs runs the missing leg). Bounded by
// AB_FINISH_RETRIES, agentBounded and TIME_FINAL_DEADLINE_HIT: never spends the final reserve.
if (want('final')) {
  // Reclaim leftover process-local capture dirs before Finalize uses more disk; runs even past
  // TIME_FINAL_DEADLINE_HIT (cheap local FS work, not a bench).
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
  // (0) Past the final-reserve boundary skip the whole gate (scan included); pending entries reach
  // the caller via pendingIntegrations: deferred, not discarded.
  if (TIME_FINAL_DEADLINE_HIT) {
    log(`Finalize-gate: SKIPPED entirely — the final reserve has already begun. ` +
      `${pendingIntegrations.length} pending A/B(s) deferred to the caller (surfaced in pending_integrations).`);
  } else {
  // (1) Disk-reconstruct incomplete A/Bs via a read-only agent (the JS has no fs).
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
        // The overlay is already built: the integrator benches it directly (RESUME_AB), no rebuild.
        CAND_OVERLAY_DIR: it.overlay_dir, RESUME_AB: true, SKILL_DIR: WORKFLOW_DIR,
      },
    });
    log(`Finalize-gate: disk-reconstructed incomplete A/B "${it.short_name}" ` +
      `(winner_kind=${it.winner_kind || '?'}, ref=${!!it.ref_present}, cand=${!!it.cand_present}); queued to finish.`);
  }

  // (2) Finish every pending A/B, never past the reserve boundary; highest isolated speedup first.
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
    if (abWindowShort(abLegs(pendingIntegrations[0].inputs), `Finalize-gate: ${pendingIntegrations[0].short_name}`)) {
      log(`Finalize-gate: STOPPING with ${pendingIntegrations.length} A/B(s) unfinished ` +
        `(${pendingIntegrations.map((q) => q.short_name).join(', ')}): the next cannot finish before the final ` +
        'reserve. Deferred to pending_integrations.');
      break;
    }
    const p = pendingIntegrations.shift();   // pop one per iteration => guaranteed termination
    log(`Finalize-gate: finishing pending A/B (${p.short_name}, ${(p.isolated || 0).toFixed(2)}x isolated); ` +
      `${pendingIntegrations.length} more queued.`);
    const integ = await runIntegrateBothLegs(
      'Finish the e2e A/B for this verified-isolated win. Run BOTH legs (reference + candidate) to ' +
      'completion, then return accepted/stack/rejected with ab_complete:true.',
      // Measured against the latest accepted baseline (current overlay/flags/env/throughput).
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
// The reserve boundary abandons a search agent's promise, not the agent, so detached search work may
// still hold GPUs. Finalize and Validate each start with run_reaper.py (selects by this run's marker,
// not by name), bounded on its own clock; a failed fence is non-fatal.
const RUN_REAPER = `${WORKFLOW_DIR}/scripts/run_reaper.py`;
const FENCE_TIMEOUT_MS = 10 * 60000;
const FENCE_SCHEMA = obj({ ok: { type: 'boolean' }, stage: { type: 'string' }, stopped: arrObj,
  survivors: arrObj, note: { type: 'string' } }, []);
async function finalPhaseFence(stage, phaseName) {
  if (!EVAL_DIR) return null;
  const cmd = `python3 ${shq(RUN_REAPER)} fence --eval-dir ${shq(EVAL_DIR)} --stage ${stage}`;
  const r = await safeAgent(
    `You are the final-phase GPU fence. Run EXACTLY this command, once, and return its JSON stdout verbatim ` +
    `as StructuredOutput; it pretty-prints over several lines, so return the whole object. It is this ` +
    `workflow's own cleanup and the sanctioned way to do what PROCESS SAFETY asks — stop only what this run ` +
    `started: it selects processes by this run's marker and eval dir, never by a name pattern, and never ` +
    `signals PID 1, the caller, or your own shell. Do not edit it, retry it, or act on what it reports; a ` +
    `non-zero exit is a valid answer, so return what it printed.\n` +
    '```bash\n' + cmd + '\n```',
    { phase: phaseName, label: `fence:${stage}`, schema: FENCE_SCHEMA }, 1, () => FENCE_TIMEOUT_MS);
  if (!r) {
    log(`[final-phase fence] ${stage}: the fence returned nothing — continuing without it (NON-FATAL).`);
    return null;
  }
  const stopped = Array.isArray(r.stopped) ? r.stopped : [];
  const survivors = Array.isArray(r.survivors) ? r.survivors : [];
  const names = [...new Set(stopped.map((p) => String((p && p.name) || (p && p.pid) || '?')))].slice(0, 8);
  log(`[final-phase fence] ${stage}: stopped ${stopped.length} process(es) still running from the search` +
    `${names.length ? ` (${names.join(', ')})` : ''}${survivors.length ? `; ${survivors.length} still up` : ''}` +
    `${r.note ? ` — ${r.note}` : ''}.`);
  return r;
}

if (want('final')) {
  if (TIME_BUDGET_MS != null) log(`[time-budget] entering the final phase with ~${remainingMin()}min of the ${Math.round(TIME_BUDGET_MS / 60000)}min budget left (reserve was ${Math.round(FINAL_RESERVE_MS / 60000)}min).`);
  FINAL_PHASE_STARTED = true;   // the one place the reserve exemption is granted — by position, not label
  phase('Finalize');
  await finalPhaseFence('finalize', 'Finalize');
  finalize = await safeAgent(
    roleAgent('e2e_integrator', 'finalize', 'Assemble the final overlay + patch + launch script bundle.' + agentxFinalizeNote(), {
      EVAL_DIR, FINAL_OVERLAY: curOverlay, ACCEPTED_FLAGS: curFlags, ACCEPTED_ENV: curEnv,
      ACCEPTED_KERNELS: allAccepted, BASELINE_THROUGHPUT: BASELINE_TPUT, SKILL_DIR: WORKFLOW_DIR,
      ...TUNING_FINALIZE_INPUTS,
    }),
    { phase: 'Finalize', label: 'e2e_integrator:finalize', schema: FINALIZE_SCHEMA });
  finalTput = (finalize && finalize.final_throughput_tok_s) || curTput;
  // The stock bundle's number is Setup's canonical one, not Finalize's search leg.
  const adoptedNothing = agentxAdoptedNothing({ setup: SETUP_CONFIG, accepted: allAccepted.length, tput: curTput,
    baseline: BASELINE_TPUT, flags: curFlags, env: curEnv, overlay: curOverlay });
  if (adoptedNothing) finalTput = BASELINE_TPUT;
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

  // Validate is sized into FINAL_RESERVE_MS and not time-gated (skips only per agentxValidateSkip);
  // if a hard kill lands mid-bench, Report's files and run_e2e.py's fallback
  // (director_e2e_validation.json, then the best overlay's integrate_result.json) still carry a win.
  phase('Validate');
  validateSkip = agentxValidateSkip(adoptedNothing, remainingMs());
  if (validateSkip === 'final_is_baseline') {
    log(`Validate SKIPPED: nothing adopted, so the final bundle is the stock one (Setup's ${BASELINE_TPUT} tok/s).`);
  } else if (validateSkip === 'no_time') {
    log(`Validate SKIPPED: ~${remainingMin()}min left, under its ${Math.round(AGENTX_VALIDATE_WINDOWS_MS / 60000)}min of timed windows.`);
  } else {
    await finalPhaseFence('validate', 'Validate');
  }
  validation = validateSkip ? null : await safeAgent(
    roleAgent('director', 'validate', 'Independently re-measure throughput + parity; arbitrate; then reconcile the report with the validated numbers.' +
      agentxClaimNote(finalTput, BASELINE_TPUT), {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], BASELINE_THROUGHPUT: BASELINE_TPUT, NOISE_BAND_PCT: NOISE_BAND,
      BASELINE_FLAGS, BASELINE_ENV, BASELINE_OVERLAY,
      FINAL_OVERLAY: (finalize && finalize.final_overlay) || curOverlay,
      FINAL_FLAGS: { flags: curFlags, env: curEnv },
      CLAIMED_THROUGHPUT: finalTput, WORKLOAD, APPLY_TO_ORIGINAL,
      MEASUREMENT_MODE: VALIDATION_MEASUREMENT_MODE,
      MEASUREMENT_PURPOSE: 'validation', REPLICAS: VALIDATION_SAMPLES,
      SKILL_DIR: WORKFLOW_DIR, ...ACCURACY_INPUTS,
      // Report wrote these before Validate ran: the Director rewrites their headline numbers (and
      // status/parity) to its own, so they never disagree. Defaults are the standard EVAL_DIR names.
      ARCHITECT_REPORT: (report && report.report_path) || `${EVAL_DIR}/architect_report.md`,
      FINAL_REPORT: `${EVAL_DIR}/final_report.md`,
    }),
    { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });
  // Trust the Director's number only when it is a real positive measurement; a Validate with no
  // usable number never erases the accepted A/B win carried in finalTput.
  validatedOk = !!(validation && validation.director_verified_throughput_tok_s > 0 && validation.throughput_speedup > 0);
  finalSpeedup = validatedOk ? validation.throughput_speedup : (BASELINE_TPUT ? finalTput / BASELINE_TPUT : finalSpeedup);
  // A search leg over a canonical baseline is no speedup: run_e2e.py reports the accepted A/B's own pair.
  finalFromAb = IS_AGENTX && !validatedOk && allAccepted.length > 0;
  log(`COMPLETE. ${MODEL_NAME}: ${BASELINE_TPUT} -> ${validatedOk ? validation.director_verified_throughput_tok_s : finalTput} tok/s ` +
    `(${finalSpeedup ? finalSpeedup.toFixed(3) : '?'}x, status ${validationStatusOf()}` +
    `${validation && !validatedOk ? '; Validate produced no number — using carried accepted A/B win' : ''}). Results in ${EVAL_DIR}`);
  if (finalFromAb) {
    log("Validate gave no number: returning no final or speedup, so run_e2e.py reports the accepted A/B's own pair.");
  }

  // --- FINAL experience reconciliation (post-gate) ----
  // Cards written as "pending" before the Finalize-gate (Fix C) and Validate confirmed them get the
  // verified post-Validate numbers here, in one re-curate.
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
  ...(KERNEL_EXTRACT_ON
    ? { workload_spec: { ...WORKLOAD_SPEC, kind: WORKLOAD_KIND },
        profile_workload_json: profile ? profile.profile_workload_json || '' : '',
        kernel_ranking_json: profile ? profile.kernel_ranking_json || '' : '',
        kernel_extraction_delivery_json: profile ? profile.kernel_extraction_delivery_json || '' : '',
        kernel_extraction_summary: {
          failed_tasks: profile?.failed_tasks || [], deferred_tasks: profile?.deferred_tasks || [],
          coverage_summary: profile?.coverage_summary || {},
        } } : {}),
  eval_dir: EVAL_DIR, model_name: MODEL_NAME, baseline_throughput_tok_s: BASELINE_TPUT,
  noise_band_pct: NOISE_BAND, flags: curFlags, env: curEnv, overlay: curOverlay, throughput: curTput,
  baseline_flags: BASELINE_FLAGS, baseline_env: BASELINE_ENV, baseline_overlay: BASELINE_OVERLAY,
  profile_topn_json: profile ? profile.profile_topN_json : '',
  config_directions: (strategy && strategy.config_directions) || [],
  // AgentX candidates carry prepared_ut.result (dispatch cannot re-read it from disk), so an
  // undispatched task survives a resume (references/workflow-handoff.md).
  headQueue, kernelQueue, accepted_heads: acceptedHeads, flagged_heads: flaggedHeads, accepted_kernels: acceptedKernels,
  // Full tuning result: a phase-by-phase resume neither re-runs tuning nor loses its attribution.
  ...(tuning ? { tuning } : {}),
  // Pending wins WITH their inputs, so a resumed run can finish their A/B.
  pending_integrations: pendingIntegrations,
  history,
};

// What this run got from the KB, and separately what it added (a warm start's headline speedup is
// partly inherited). BASELINE_TPUT stays the true cold baseline throughout.
const kbWarmStart = (E2E_WARM_START_ON && KB_DIMS) ? {
  identity: KB_DIMS, plane: E2E_KB_PLANE, read_plane: KB_READ_PLANE, mode: E2E_WARM_START,
  reference_dir: KB_REF_DIR,
  adopted_throughput_tok_s: kbSeedTput || null,
  adopted_kernels: kbSeedKernels.map(k => k.name).filter(Boolean),
  // This run's own gain past the warm start; null when nothing was adopted.
  incremental_speedup: kbSeedTput ? Number((finalTput / kbSeedTput).toFixed(6)) : null,
  // The full recall ledger (both planes), the same object the Report role was handed.
  recall: KB_RECALL,
} : null;
// Attribution for the standalone tuning phase (its own pre/post legs): delta% and its SHARE of the
// run's total gain (null, not 0, when there is no net gain to apportion).
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
    // To reproduce the tuning elsewhere: deploy_bundle (idempotent deploy.sh, run by final_launch.sh).
    deploy_bundle: tuning.deploy_bundle || '', deploy_verified: tuning.deploy_verified === true,
    cache_invalidation: tuning.cache_invalidation || [], live_tree_files: tuning.live_tree_files || [],
    apply_overlay: tuning.apply_overlay || '',
    // Finalize's check that the tuning is in the shipped bundle and engages; null if it did not run.
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

// A Validate that returned nothing is flagged, so no KB writer records the carried number.
function validationStatusOf() {
  return validatedOk ? validation.validation_status
    : (validation ? `${validation.validation_status || 'flagged'}_no_number_used_carried_ab`
      : (!want('final') ? 'phase_partial'
        : validateSkip === 'final_is_baseline' ? 'skipped_final_is_baseline'
          : (IS_AGENTX ? 'flagged_no_number_used_carried_ab' : 'unknown')));
}

const wfReturn = {
  // The contract shape run_e2e.py reads; bump only on a breaking change to the keys below.
  schema_version: 1,
  mode: 'e2e',
  fast_mode: FAST_MODE,   // true => ConfigSweep, TuningSkillset and Milestone skipped (HeadKernel-only)
  deep_mode: DEEP_MODE,   // true => HeadKernel runs the long cross-backend co-optimization scheduler
  backend: BACKEND,
  phases_run: PHASES,
  eval_dir: EVAL_DIR,
  model_name: MODEL_NAME,
  baseline_throughput_tok_s: BASELINE_TPUT,
  // The Director's number only when validatedOk; else the carried accepted A/B win, never 0 (except
  // finalFromAb's empty pair, which run_e2e.py fills from disk).
  final_throughput_tok_s: validatedOk ? validation.director_verified_throughput_tok_s : (finalFromAb ? 0 : finalTput),
  throughput_speedup: finalFromAb ? 0 : finalSpeedup,
  validation_status: validationStatusOf(),
  output_parity: validation ? validation.output_parity : 'unknown',
  accepted_config: { flags: curFlags, env: curEnv },
  accepted_kernels: acceptedKernels,
  accepted_heads: acceptedHeads,
  // Tuning phase attribution, reported apart from accepted_heads/accepted_kernels.
  tuning_skillset: tuningReturn(),
  ...(KERNEL_EXTRACT_ON ? { kernel_extraction: {
    delivery_path: profile?.kernel_extraction_delivery_json || '',
    failed_tasks: profile?.failed_tasks || [], deferred_tasks: profile?.deferred_tasks || [],
    coverage_summary: profile?.coverage_summary || {},
  } } : {}),
  // Verified-isolated wins whose e2e A/B never completed, surfaced (slim) for the caller to finish.
  pending_integrations: pendingIntegrations.map(p => ({
    short_name: p.short_name, track: p.track, isolated: p.isolated,
    pct_gpu_time: p.pct_gpu_time, partial: p.partial || null,
  })),
  flagged_heads: flaggedHeads,   // dominant heads not optimized (harness/extract/no-candidate)
  config_tune_enabled: CONFIG_TUNE_ENABLED,
  head_budget: HEAD_BUDGET,
  head_used: headDispatched,
  milestones: milestone,
  budget_used: dispatched,
  budget_total: BUDGET,
  final_overlay: (validation && validation.final_overlay) || (finalize && finalize.final_overlay) || curOverlay,
  // final_patch is the KB's reproducible code (e2e_store.py _ARTIFACT_KEYS reads
  // result["final_patch"]); final_overlay, a directory, cannot stand in for it.
  ...kbResultMetadata(),
  final_patch: (finalize && finalize.final_patch) || '',
  final_launch_script: (validation && validation.final_launch_script) || (finalize && finalize.final_launch_script) || '',
  report_path: report ? report.report_path : `${EVAL_DIR}/architect_report.md`,
  // Conditional spread: with the feature off the key is absent.
  ...(kbWarmStart ? { kb_warm_start: kbWarmStart } : {}),
  // Independent of kb_warm_start (null without KB_DIMS, while kernel lanes may still have recalled);
  // absent when nothing was recalled on either plane.
  ...((KB_RECALL.e2e || KB_RECALL.kernel.length) ? { kb_recall: KB_RECALL } : {}),
  state: carryState,
};

