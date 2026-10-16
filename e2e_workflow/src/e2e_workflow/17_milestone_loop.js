// ===========================================================================
// PHASE: Milestone loop — extract -> recursive kernel optimize -> overlay -> e2e gate
// ===========================================================================
// Floor: keep dispatching until >= MIN_KERNEL_TASKS editable-kernel tasks have run, THEN allow the
// noImprove early-stop. While below the floor the loop never stops on no-improve / empty plan.
// Budget: when time_budget_s was passed, TIME_DEADLINE_HIT stops STARTING new milestones (even below the
// floor) so the in-flight work + Finalize/Validate finish before the orchestrator's hard kill. The guard
// is inert when time_budget_s is absent (TIME_DEADLINE_HIT never set) => byte-identical default behavior.
while (want('kernel') && !TIME_DEADLINE_HIT && dispatched < BUDGET && (dispatched < MIN_KERNEL_TASKS || noImprove < 2)) {
  milestone++;
  const remaining = BUDGET - dispatched;
  const belowFloor = dispatched < MIN_KERNEL_TASKS;

  // --- (a) Plan this milestone (Architect): nominate next kernels. While BELOW the floor the Architect
  // MUST nominate (it may not stop); it draws fresh editable candidates from the re-profile + the broad
  // candidate pool, never re-using a confirmed e2e-null direction verbatim. ---
  phase('Milestone');
  const plan = (milestone === 1 && kernelQueue.length)
    ? { stop: false, kernel_candidates: kernelQueue }
    : await safeAgent(
      roleAgent('system_architect', 'plan_milestone', `Nominate next kernels — ONLY editable kernels with pct_gpu_time >= ${MILESTONE_MIN_PCT}% (below that, Amdahl says they can't move e2e; do not nominate them even to meet the floor). Each candidate MUST carry its pct_gpu_time.`, {
        EVAL_DIR, ROUND: milestone, BUDGET_REMAINING: remaining, CURRENT_THROUGHPUT: curTput,
        BASELINE_THROUGHPUT: BASELINE_TPUT, NOISE_BAND_PCT: NOISE_BAND, MILESTONE_MIN_PCT,
        MIN_KERNEL_TASKS, DISPATCHED_SO_FAR: dispatched, BELOW_MIN_FLOOR: belowFloor,
        PROFILE_TOPN: profile ? profile.profile_topN_json : '', HISTORY: history, SKILL_DIR: WORKFLOW_DIR,
        ...ANALYSIS_SKILL_INPUTS,
      }),
      { phase: 'Milestone', label: `architect:plan m${milestone}`, schema: PLAN_SCHEMA });

  const planCandsRaw = (plan && plan.kernel_candidates) ? plan.kernel_candidates : [];
  // pct_gpu_time gate: only optimize kernels above MILESTONE_MIN_PCT (a candidate missing pct is kept,
  // not silently dropped — but logged). This gate OVERRIDES the min-floor: low-pct kernels are not worth it.
  const planCands = planCandsRaw.filter(c => c.pct_gpu_time == null || c.pct_gpu_time >= MILESTONE_MIN_PCT);
  const skipped = planCandsRaw.filter(c => c.pct_gpu_time != null && c.pct_gpu_time < MILESTONE_MIN_PCT);
  if (skipped.length) log(`Milestone ${milestone}: skipped ${skipped.length} kernel(s) below ${MILESTONE_MIN_PCT}% GPU [${skipped.map(c => `${c.short_name || '?'}@${(+c.pct_gpu_time).toFixed(1)}%`).join(', ')}].`);
  if (!planCands.length) {
    if (planCandsRaw.length) log(`Milestone ${milestone}: stop — no remaining kernel clears the ${MILESTONE_MIN_PCT}% GPU bar (Amdahl: sub-threshold kernels can't move e2e). Floor is overridden by the pct gate.`);
    else if (belowFloor) log(`Milestone ${milestone}: below floor (${dispatched}/${MIN_KERNEL_TASKS}) but Architect nominated nothing — cannot fabricate candidates; stopping.`);
    else log(`Milestone ${milestone}: stop (floor ${MIN_KERNEL_TASKS} met). ${plan ? plan.reasoning || '' : ''}`);
    break;
  }

  const cands = planCands.slice(0, remaining).map((c, i) => ({
    ...c, idx: i, gpu_id: GPU_LIST[i % GPU_LIST.length],
    short_name: c.short_name || `k${milestone}_${i}`,
  }));
  dispatched += cands.length;
  log(`Milestone ${milestone}: ${cands.length} kernel candidate(s) [${cands.map(c => c.short_name).join(', ')}], dispatched ${dispatched}/${BUDGET} (floor ${MIN_KERNEL_TASKS})`);

  // --- (b) PARALLEL optimize (extract + recursive kernel layer per candidate, on distinct GPUs), then
  // SERIAL integrate. The optimize stage is independent per kernel -> run concurrently. The e2e integrate
  // stage MEASURES throughput and COMPOUNDS the overlay, so it must be serial: no two servers benched at
  // once (no timing conflict) and accepted overlays carry forward in order.
  const optimized = await parallel(cands.map((c) => async () => {
    const ext = await extractWithBaseline(
      'kernel_extractor', 'extract', 'Capture shapes + oracle; emit an immutable unittest task dir.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: c.gpu_id, WORKLOAD, KERNEL: c,
        CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
      },
      { phase: 'Milestone', label: `extract ${c.short_name}`, schema: EXTRACT_SCHEMA });
    if (!ext || ext.editable === false || ext.unittest_smoke !== 'pass' || !ext.task_dir) {
      return { c, skip: true, reason: `extraction failed/non-editable (${extractFailWhy(ext)})` };
    }
    // RECURSIVE kernel layer on the IMMUTABLE task dir (one allowed nesting level via workflow()).
    let kl;
    try {
      const r = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, laneArgs({
        kernel_path: ext.task_dir, workflow_dir: KERNEL_WF_DIR,
        use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
        ...KB_ARGS,
        budget: KERNEL_BUDGET, gpu_ids: c.gpu_id, exp_root: `${EVAL_DIR}/kernels/_exp`,
        task: 'Compare candidate backends ' + JSON.stringify(c.candidate_backends || []) +
          ' for this kernel; pick the fastest that passes the immutable unittest. ' + GRAPH_REQ + (TASK || ''),
        apply_to_original: 'false',
      }));
      noteKernelKB(r, String(c.short_name || ext.op_kind || 'milestone'));
      kl = { ran: true, kernel_eval_dir: r.eval_dir, final_patch: r.final_patch,
        final_geomean: r.final_geomean, validation_status: r.validation_status,
        note: (r.winner && r.winner.source) || '' };
    } catch (e) {
      kl = { ran: false, final_patch: '', final_geomean: 0, validation_status: 'error', note: String(e) };
    }
    return { c, ext, kl };
  }));

  // --- serial integrate (compounding overlay, isolated measurement) ---
  let milestoneImproved = false;
  for (const o of optimized) {
    if (!o) continue;
    const c = o.c;
    if (o.skip) {
      log(`  ${c.short_name}: ${o.reason}; skipping.`);
      history.ledger.push({ direction: c.short_name, verdict: 'dead_end', lesson: o.reason });
      continue;
    }
    const { ext, kl } = o;
    if (!kl || !kl.ran || !(kl.final_geomean > 1.0) || !kl.final_patch) {
      log(`  ${c.short_name}: kernel layer produced no speedup (${kl ? kl.final_geomean : '?'}x); skipping integrate.`);
      history.ledger.push({ direction: c.short_name, isolated_speedup: kl ? kl.final_geomean : 0, verdict: 'dead_end', lesson: 'no isolated speedup' });
      continue;
    }
    log(`  ${c.short_name}: kernel layer ${kl.final_geomean.toFixed(2)}x isolated. Integrating to e2e.`);
    // Build inputs ONCE so Fix C can re-issue the SAME A/B for a pending win at Finalize.
    const mileIntegrateInputs = {
      EVAL_DIR, MODEL_PATH, GPU_ID: c.gpu_id, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
      KERNEL_RESULT: { short_name: c.short_name, task_dir: ext.task_dir,
        source_path_in_sglang: ext.source_path_in_sglang, target_callable: ext.target_callable,
        final_patch: kl.final_patch, verified_isolated_speedup: kl.final_geomean, pct_gpu_time: c.pct_gpu_time },
      CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
      CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR,
    };
    const integ = await runIntegrateBothLegs(
      'Overlay the optimized kernel back; gate on e2e throughput.', mileIntegrateInputs,
      `integrate ${c.short_name}`, 'Milestone');

    // Three-state gate (see head track): incomplete A/B => PENDING, not rejected.
    // Backward-compatible: null (timeout/hang/degrade) or an explicit incomplete
    // flag => not done; a legacy return without ab_complete behaves as before.
    const abDone = !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);
    if (abDone && integAccepted(integ, c.pct_gpu_time, kl.final_geomean) && integ.e2e_throughput_tok_s > curTput) {
      curOverlay = integ.accepted_overlay || curOverlay;
      curTput = integ.e2e_throughput_tok_s;
      bankAccepted(acceptedKernels, { short_name: c.short_name, backend: kl.note || '', kind: 'patch', ...e2eFrom(integ), isolated: kl.final_geomean }, krOf(mileIntegrateInputs));
      milestoneImproved = true;
      log(`  ${c.short_name}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: c.short_name, isolated_speedup: kl.final_geomean, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
    } else {
      // Unified reject/incomplete handling (same as the head track): a FIXABLE reject OR fixable
      // crash-during-warmup (ab_complete=false) of an iso-verified editable kernel earns a corrective
      // re-author (re-optimize the EXISTING kernel; NOT charged to HEAD_BUDGET); only keep PENDING if the
      // A/B was merely incomplete for a NON-fixable reason (real transient timeout/hang). See tryCorrectiveReauthor.
      const reason = gateRejectReason(integ, c.pct_gpu_time, kl.final_geomean);
      const corr = (rejectClass(reason) !== '')
        ? await tryCorrectiveReauthor({
            short_name: c.short_name, op_kind: ext.op_kind, shapes: ext.shapes, dtype: ext.dtype, regime: c.regime,
            gpu_id: c.gpu_id, kernel_eval_dir: kl.kernel_eval_dir, task_dir: ext.task_dir, language: kl.language || '',
            isolated: kl.final_geomean, base_inputs: mileIntegrateInputs, reason, fix_class: rejectClass(reason), pct_gpu_time: c.pct_gpu_time, phase_name: 'Milestone',
            cur: { overlay: curOverlay, flags: curFlags, env: curEnv, tput: curTput },
          })
        : { banked: false };
      if (corr.banked) {
        curOverlay = corr.integ.accepted_overlay || curOverlay; curTput = corr.integ.e2e_throughput_tok_s;
        bankAccepted(acceptedKernels, { short_name: c.short_name, backend: kl.note || '', kind: 'patch', ...e2eFrom(corr.integ), isolated: corr.isolated, corrective: true, patch: corr.patch || '' }, krOf(mileIntegrateInputs));
        milestoneImproved = true;
        log(`  ${c.short_name}: ACCEPTED after corrective re-author (${reason}). e2e now ${curTput} tok/s (+${corr.integ.e2e_delta_pct}%).`);
        history.ledger.push({ direction: c.short_name, isolated_speedup: corr.isolated, e2e_delta_pct: corr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed: ${reason}` });
      } else if (!abDone) {
        pendingIntegrations.push({ track: 'milestone', short_name: c.short_name, isolated: kl.final_geomean,
          pct_gpu_time: c.pct_gpu_time, inputs: mileIntegrateInputs, backend: kl.note || '',
          partial: integ ? { gate: integ.gate, ref_med: integ.ref_med, cand_med: integ.cand_med, reason: integ.reason } : null });
        log(`  ${c.short_name}: INTEGRATE INCOMPLETE — A/B not finished (${integ ? integ.reason || integ.gate : 'null/timeout'}); kept as PENDING (not a rejection).`);
        history.ledger.push({ direction: c.short_name, isolated_speedup: kl.final_geomean, verdict: 'incomplete', lesson: integ ? integ.reason || 'A/B not finished' : 'integrate timed out/null before A/B completed' });
      } else {
        log(`  ${c.short_name}: REJECTED at e2e gate (${reason}).`);
        history.ledger.push({ direction: c.short_name, isolated_speedup: kl.final_geomean, e2e_delta_pct: integ.e2e_delta_pct || 0, verdict: 'dead_end', lesson: reason || 'no e2e gain' });
      }
    }
  }

  // --- (c) If improved: re-profile + grow the experience library ----------
  if (milestoneImproved) {
    noImprove = 0;
    profile = await safeAgent(
      roleAgent('profiler', 'reprofile', 'Re-profile the new best server.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: milestone,
        OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        ...ANALYSIS_SKILL_INPUTS,
      }),
      { phase: 'Profile', label: `profiler:reprofile m${milestone}`, schema: PROFILE_SCHEMA });
  } else {
    noImprove++;
  }

  // --- (d) Update the persistent experience library + in-run memory -------
  const exp = E2E_LEARNED_KB_ENABLED ? await safeAgent(
    roleAgent('system_architect', 'update_experience', 'Curate knowledge/learned/ (merge/insert >=2-star / archive contradicted) per learned/README.md.', {
      ROUND: milestone, EVAL_DIR, MODEL_NAME, SKILL_DIR: WORKFLOW_DIR,
      MILESTONE_RESULTS: history.ledger.slice(-cands.length),
      REPROFILE_SHIFT: profile ? profile.shift_note : '', PRIOR_HISTORY: history,
      ...ANALYSIS_SKILL_INPUTS,
    }),
    { phase: 'Milestone', label: `architect:experience m${milestone}`, schema: EXPERIENCE_SCHEMA })
    : null;
  if (exp) {
    if (exp.insights) history.insights = exp.insights;
    if (exp.bottleneck_now) history.bottleneck_now = exp.bottleneck_now;
    if (exp.suggest_next) history.suggest_next = exp.suggest_next;
  }
  history.milestones.push({ milestone, accepted: acceptedKernels.length, throughput: curTput, improved: milestoneImproved });
  log(`Milestone ${milestone} done. throughput=${curTput} tok/s (${(curTput / BASELINE_TPUT).toFixed(3)}x), noImprove=${noImprove}`);
}

