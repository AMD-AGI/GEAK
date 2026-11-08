// The Optimize phase. An optimizer starts from the state Setup/resume/WarmStart left behind and
// returns an OptimizeResult; applyOptimizeResult() hands it to the Report/Validate postlude:
//   { rounds, dispatched, cumulativeSpeedup, bestPerCase, finalWinner, roundsCommitted, history,
//     citations, profileSummary, stopReason, forcedReplans, kbCapBound }
// The greedy loop body keeps its original indentation so its prompts stay byte-identical.
async function greedyOptimize(start, evo = null, population = null) {
const evoPlanSchema = evo ? {
  ...PLAN_SCHEMA, properties: {...PLAN_SCHEMA.properties, directions: {
    ...PLAN_SCHEMA.properties.directions, items: {
      ...PLAN_SCHEMA.properties.directions.items, properties: {
        ...PLAN_SCHEMA.properties.directions.items.properties,
        parent_id: {type: 'string'}, donor_ids: {type: 'array', items: {type: 'string'}},
        approach: {type: 'string'},
      },
    },
  }},
} : PLAN_SCHEMA;
let { cumulative, bestSeen, bestPerCase, finalWinner, profileSummary } = start;
const history = start.history;   // extended in place, as before
// The planner sees the archive, its schema fields and the retained-approach policy once the archive
// holds retained source; before that its round plans from the same inputs as greedy.
const archiveHasSource = () => !!evo && evo.entries.some(row => row.patch_ref && !row.retention_error);
const evolutionInstructions = () => archiveHasSource()
  ? ` Read ${WORKFLOW_DIR}/roles/evolution.md for the retained-approach policy.` : '';
const citations = [];
const contextRef = (id, outDir) => {
  if (evo) return {path: evo.path, entry_id: id, field: 'reported_context'};
  const retained = retainedSources.find(row => row.id === id);
  return retained?.source && !retained.retention_error
    ? {path: `${retained.source.path}.json`, field: 'reported_context'}
    : {unavailable: 'no complete retained context artifact', related_report: `${outDir}/report.md`};
};
let dispatched = 0, round = 0, noImprove = 0, roundsCommitted = 0;
let plannerStopReason = '', deadlineHit = false, forcedReplans = 0, kbCapBound = 0;
await captureCanonicalSelection(history, setup.resumed || A.canonical_selection_resume === true
  ? 'resume_unknown' : 'initial', 0, finalWinner);
if (population) await population.publishCheckpoint(0, finalWinner || bench,
  await population.checkpointTree(null, 0));
const searchTiming = beginSearchWindow();
if (searchTiming) history.search_timing = searchTiming;
while (!skipLoop && dispatched < BUDGET && noImprove < MAX_NO_IMPROVE) {
  // --- (0) HARD STOP: no new round may START past the deadline ----------
  // Checked BEFORE round++ so an expired check does not inflate the round count. The in-flight round
  // is never interrupted — a killed round leaves a half-verified patch and no report.
  let left = await secondsLeft(`pre-r${round + 1}`);
  if (left <= 0) {
    deadlineHit = true;
    log(`WALL-CLOCK DEADLINE reached after ${round} round(s) (budget used ${dispatched}/${BUDGET}) — ` +
        `starting no further rounds; going straight to report + validation.`);
    break;
  }
  if (engineerCutoffReached(left)) {
    deadlineHit = true;
    log(`ENGINEER CUTOFF reached after ${round} round(s): no new round starts; delivery keeps the search cutoff.`);
    break;
  }
  if (DEADLINE_EPOCH) log(`[deadline] ${(left / 60).toFixed(0)} min remain before the hard stop.`);

  // Peers are deliberate planner references; refresh preserves this lineage's local best and budget.
  if (population && evo) await population.refreshPeers();

  round++;
  const remaining = BUDGET - dispatched;
  if (archiveHasSource()) history.evolution_archive = evo.path;
  phase('Optimize');
  // Retained delivery needs each GEAK source's actual base; EVO sources share the archive base.
  const roundBase = !EXTERNAL_OBJECTIVE && !evo ? await headCommit() : null;

  // --- (a) Plan the round (TechLead) ------------------------------------
  const planInputs = (forbidStop, facts = {}) => ({
    EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
    BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
    CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
    ...(archiveHasSource() ? {...evo.planInputs(), DEEP_COST} : {}),
    KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
    GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
    ...KB_INPUTS,
    // DRA brief (REFERENCE), from main. plan_round reads it and seeds directions[] from the ranked
    // DRA directions — see tech_lead.md plan_round. Spread conditionally, so when dra_enabled was off
    // (or the research degraded) the key is absent and the prompt is byte-identical to a build
    // without the feature. Threaded through `planInputs` rather than the call site so the forced
    // re-plan below gets it too: a replan that silently loses the research brief would plan against
    // a different input set than the round it is replacing.
    ...(researchBriefPath ? { DEEP_SEARCH_BRIEF: researchBriefPath } : {}),
    // Stated UNCONDITIONALLY, in both directions. `use_learned_kb=false` used to omit the budget and
    // nothing else, while roles/tech_lead.md still listed knowledge/learned/INDEX.md among the files
    // to read — so the off switch removed the budget that CONSTRAINS the KB and left the instruction
    // that USES it, i.e. the arm it exists to produce (a clean KB-off control) was never clean. The
    // role now gates on this value, so it has to be present whichever way it points.
    LEARNED_KB: USE_LEARNED_READ ? 'on' : 'off',
    ...(USE_LEARNED_READ ? { LEARNED_KB_BUDGET:
      `At most ${KB_DIR_CAP} of this round's directions may draw on learned cards, and you must cite ` +
      `the cards you used in that direction's \`learned_refs\`. ` +
      (KB_COLD_DIRECTION ? `At least ONE direction must be planned from the profile alone, with an ` +
        `empty \`learned_refs\` — it is this round's control, and without it a round cannot tell you ` +
        `whether the cards helped or merely crowded out what you would have tried anyway. ` : '') +
      `Read as much of the index as you like; the budget is on how much of the ROUND the KB steers, ` +
      `not on what you may look at.` } : {}),
    // Both keys absent unless a deadline was passed => byte-identical prompt when OFF.
    ...(DEADLINE_EPOCH ? { MINUTES_REMAINING: Math.round(left / 60) } : {}),
    ...(ENGINEER_RESERVE_S ? { ENGINEER_CUTOFF_EPOCH } : {}),
    ...facts,
    ...(forbidStop ? { STOP_NOT_PERMITTED:
      `You returned stop=true with ${Math.round(left / 60)} min of the wall-clock window still ` +
      `unspent. This run is a FIXED-WINDOW measurement: ending early is an invalid data point no ` +
      `matter how good the number or how well argued the rationale, and your role's own convergence ` +
      `gates (launch floor closed, gain under the promotion bar, roofline %, "only a different tech ` +
      `stack is left") do NOT authorise stopping here. Plan at least one direction. Switch lever ` +
      `rather than re-running an exhausted one — algorithm / memory / compute / host_runtime, or a ` +
      `dedicated deep_explore ground-up rewrite. If every direction really is exhausted, spend the ` +
      `round RE-MEASURING the committed best to tighten its error bars and say so. Sizing: keep the ` +
      `round finishable inside the remaining ${Math.round(left / 60)} min.` } : {}),
  });

  const recordProposedPlan = async (attempt, proposed) => {
    const path = `${EVAL_DIR}/planner_r${round}_${attempt}_${Date.now()}.json`;
    try {
      if (await writeFile(path, JSON.stringify({round, attempt, plan: proposed ?? null}, null, 2) + '\n') === null)
        log(`Round ${round}: proposed-plan artifact unavailable (${attempt}).`);
    } catch {
      log(`Round ${round}: proposed-plan artifact unavailable (${attempt}).`);
    }
  };

  let plan = await agentT(
    roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).' +
      evolutionInstructions(),
      planInputs(false, await deliveryFactsInput())),
    { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: archiveHasSource() ? evoPlanSchema : PLAN_SCHEMA, ...ENGINEER_CALL });
  await recordProposedPlan('initial', plan);
  if (PROCESS_BOUNDED) left = await secondsLeft(`planned-r${round}`);

  // NO EARLY STOP: keep refusing while the window is materially unspent. This used to refuse ONCE,
  // on the reasoning that a role repeating "stop" has run out of ideas. Measured at a 2h budget that
  // cost ~3 min (write_req_to_token_pool_triton stopped at 1.95h/2.00h) and looked harmless; the cost
  // is proportional to the budget, so at 8h the same one-shot escape hands back hours. The caller's
  // rule is "must keep optimizing until the budget is spent", so the refusal has to be as long-lived
  // as the window. Bounded by MAX_FORCED_REPLANS so a role that cannot produce a direction at all
  // degrades to stopping instead of spinning the window away on planning.
  // With an engineer reserve, refuse a stop only while engineering time above the 7 s floor remains.
  while (DEADLINE_EPOCH && (ENGINEER_RESERVE_S ? left - ENGINEER_RESERVE_S > Math.max(NO_STOP_S, 7) : left > NO_STOP_S)
         && forcedReplans < MAX_FORCED_REPLANS &&
         (!plan || plan.stop || !plan.directions || plan.directions.length === 0)) {
    forcedReplans++;
    log(`Round ${round}: stop=true with ${(left / 60).toFixed(0)} min left — refusing (#${forcedReplans}), re-planning.`);
    plan = await agentT(
      roleAgent('tech_lead', 'plan_round', 'Your stop was refused: plan at least one direction.' + evolutionInstructions(),
        planInputs(true, await deliveryFactsInput())),
      { phase: 'Optimize', label: `tech_lead:replan r${round}#${forcedReplans}`, schema: archiveHasSource() ? evoPlanSchema : PLAN_SCHEMA, ...ENGINEER_CALL });
    await recordProposedPlan(`replan${forcedReplans}`, plan);
    left = await secondsLeft(`replan-r${round}`);
  }
  if (engineerCutoffReached(left)) {
    deadlineHit = true;   // Planning reached the engineer cutoff: dispatch nothing.
    log(`Round ${round}: the engineer cutoff was reached while planning; no engineer starts.`);
    break;
  }

  if (!plan || plan.stop || !plan.directions || plan.directions.length === 0) {
    if (PROCESS_BOUNDED && left <= 7) {
      deadlineHit = true; // No agent may start inside its seven-second cleanup reserve.
      log(`Round ${round}: the search process cutoff was reached while planning.`);
    } else {
      log(`Round ${round}: TechLead chose to stop. ${plan ? plan.reasoning || '' : ''}`);
    }
    break;
  }

  let directions = plan.directions.slice(0, remaining).map((d, i) => ({
    ...d,
    idx: i,
    id: d.id || `r${round}_d${i}`,
    gpu_id: GPU_MODE === 'pin' ? GPU_LIST[i % GPU_LIST.length] : GPU_POOL,
    out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
  }));
  // deep_explore is a DEDICATED-ROUND, heavyweight mandate: if the plan includes one, run ONLY it this
  // round (its broad ground-up rewrite touches many files and can't be merged with specialist patches),
  // and charge DEEP_COST against the budget. Otherwise each specialist direction costs 1.
  const deepDir = directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) directions = [deepDir];
  if (evo) {
    let allocated = 0;
    directions = directions.filter(d => {
      const cost = d.specialty === 'deep_explore' ? DEEP_COST : 1;
      if (allocated + cost > remaining) return false;
      allocated += cost;
      return true;
    });
    if (!directions.length) { plannerStopReason = 'budget'; break; }
    directions = await evo.prepare(directions, round);
  }
  const roundCost = directions.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  dispatched += roundCost;
  log(`Round ${round}: ${directions.length} direction(s) [${directions.map(d => d.specialty).join(', ')}], cost ${roundCost}, budget ${dispatched}/${BUDGET}`);
  // Preserve issued direction labels even when an engineer returns no result or the round hits its cutoff.
  // learned_refs here records issued references; later citation-cap accounting uses the working directions and may strip their attribution.
  const directionHistory = directions.map(d => ({id: d.id, title: d.title, specialty: d.specialty,
    focus_files: Array.isArray(d.focus_files) ? d.focus_files : [], parent_id: d.parent_id || null, learned_refs: d.learned_refs || []}));

  // Individual and newly integrated native sources use the same verifier and custody path.
  const verifyCandidate = async (d, eng, patch, recovered = false,
    engineerWorkspace = `${d.out_dir}/workspace`) => {
    // The verifier gets a disposable copy. Custody files remain inputs to host promotion only.
    const verification = !EXTERNAL_OBJECTIVE
      ? await verificationCopy(patch, `${d.out_dir}/verification_input.diff`) : null;
    if (!EXTERNAL_OBJECTIVE && !verification) {
      return {d, eng, ver: null, patch, verification_input_error: 'source copy unavailable'};
    }
    const measured = verification ? verification.path : patch;
    return observeRole('verify', d.id, () => agentT(
      roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
        CANONICAL: evo ? evo.base : CANONICAL, PATCH: measured, VERIFY_DIR: `${d.out_dir}/verify`,
        ...(engineerWorkspace ? { ENGINEER_WORKSPACE: engineerWorkspace } : {}),
        EVAL_DIR, WORKFLOW_DIR, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
        ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
      }),
      { phase: 'Verify', label: `verify ${d.id}${recovered ? ' (recovered)' : ''}`, schema: VERIFY_SCHEMA }
    )).then(async (ver) => {
      ver = await checkedExternalMeasurement(ver, evo ? evo.base : CANONICAL, measured);
      const verificationStatus = verification ? await verificationIntact(verification, patch) : true;
      if (verificationStatus === false) {
        ver = {...ver, status: 'source_changed', correctness: 'unknown', measurement_valid: false};
      }
      const retained = evo ? await evo.record(d, eng, verificationStatus === null ? null : ver, patch) : null;
      // Register every settled result before another direction or the round can hit its cutoff.
      // Keep failed custody of an authoritative win visible; it must fail delivery explicitly.
      const pendingVerification = verificationStatus === null ? verification : null;
      await retainResult(d, eng, ver, patch, evo, roundBase, retained, pendingVerification);
      return {d, eng, ver, patch, retained, verification: pendingVerification, verification_status: verificationStatus};
    });
  };

  // --- (b,c) Optimize -> Verify, pipelined per direction ----------------
  const results = await pipeline(
    directions,
    async (d) => {
      if (evo) {
        try { await evo.prepareWorkspace(d); }
        catch (error) {
          log(`[evolution] ${d.id}: parent initialization failed: ${String(error)}`);
          await evo.record(d, {status: 'failed', notes: 'parent initialization failed'}, null, null);
          return {d, eng: null, initialization_failed: true};
        }
      }
      const isDeep = d.specialty === 'deep_explore';
      // Before any retained source exists, EVO engineers get the greedy development rules; the
      // evolution policy applies once there is an archive source to develop from.
      const evoEngineer = archiveHasSource();
      // deep_explore reads its own role (broad authority + own iteration loop); specialists read engineer.md.
      const readLine = isDeep
        ? `Then Read ${WORKFLOW_DIR}/roles/deep_engineer.md and the knowledge files under ${WORKFLOW_DIR}/knowledge/ ` +
          (USE_LEARNED_READ ? '' : `— EXCLUDING knowledge/learned/, which is off for this run: do not open ` +
            `INDEX.md or any card there. `) +
          `(you have broad authority — combine algorithm + memory + compute + host_runtime levers in one ` +
          `coherent rewrite), and follow them. You MAY edit ANY modifiable source (kernel + Python wrapper ` +
          `+ C++ binding), not just focus_files. Run your OWN multi-iteration measure→(self-)profile→rewrite ` +
          `loop and push to the TARGET; keep the best correct version.`
        : `Then Read ${WORKFLOW_DIR}/roles/engineer.md and ${WORKFLOW_DIR}/knowledge/self_monitoring.md and the ` +
          `knowledge files for your specialty, and follow them.`;
      const workspaceInstruction = evo
        ? `Your private workspace at ${d.out_dir}/workspace has already been initialized from the selected parent by the workflow. Begin engineering there. Do not recreate it or reset its Git baseline.\n`
        : `First create YOUR private workspace, then optimize.
\`\`\`bash
set -e
# Use the materializer to exclude build artifacts and preserve symlinks.
# It shares immutable dependencies; candidate builds stay private to the workspace.
mkdir -p ${d.out_dir}
bash ${WORKFLOW_DIR}/scripts/materialize_workspace.sh \\
  --src ${CANONICAL} --dst ${d.out_dir}/workspace \\
  --shared-root ${EVAL_DIR}/_shared --link-aiter
git -C ${d.out_dir}/workspace init -q
git -C ${d.out_dir}/workspace add -A
git -C ${d.out_dir}/workspace -c user.name=GEAK -c user.email=geak@localhost commit -qm 'Candidate baseline'
\`\`\`
`;
      const facts = await deliveryFactsInput();   // after workspace preparation, at dispatch
      return agentT(
      `You are Engineer ${d.id} (specialty=${d.specialty}) for round ${round}.
${workspaceInstruction}${readLine} If KK_OPERATOR is non-empty, also consult the operator/language SOTA cards under
KERNEL_KNOWLEDGE_DIR per your role's "operator/language SOTA knowledge (REFERENCE ONLY)" section
(facts/how-to only; measure everything; ${evoEngineer ? 'retain useful slower approaches; only verified improvements replace the best' : 'never go below baseline'}).
Save best_patch.diff via \`bash ${WORKFLOW_DIR}/scripts/save_candidate_patch.sh <KERNEL_PATH> ${d.out_dir}/best_patch.diff ${evo ? d.evolution_context.workspace_base_commit : "HEAD"} <editable-source-paths>\` ${evoEngineer ? 'after each useful change, including a correct but slower implementation; retain an unfinished attempt separately with its limitations. The patch must include the inherited parent changes against the initial baseline commit. Read ' + WORKFLOW_DIR + '/roles/evolution.md' : "when COMMANDMENT's primary speedup exceeds " + CANDIDATE_FLOOR_TXT}. Use its audited oracle aggregation. For an extracted task the tool captures only kernel_src, including new source files. Otherwise pass the task's declared editable files/directories, including any new backend/build source dependencies; exclude logs, measurements and tool artifacts. Do not replace this source capture with bare git diff.

## Inputs
${cfg({
        ...(TASK ? { TASK } : {}),
        SPECIALTY: d.specialty,
        ...(evoEngineer ? {EVOLUTION: evo.engineerContext(d), EVOLUTION_ARCHIVE: evo.path} : {}),
        DIRECTION: { id: d.id, title: d.title, focus_files: d.focus_files || [], expected_speedup: d.expected_speedup, prompt: d.prompt },
        ...(isDeep ? { TARGET: (d.expected_speedup ? `Investigate the proposed ${d.expected_speedup}x speedup. ` : '') +
          'Improve the measured primary metric within the assigned budget. Work toward any supplied engineering target using the fixed-reference method in knowledge/kernel_hardware_evidence.md. ' +
          'For a declared fraction q and supported complete-region reference T_ref, use T_goal = T_ref / q with the existing evaluator case aggregation; keep the reference fixed across comparable revisions. ' +
          'An unsupported reference remains unknown. Independent unprofiled full-call measurement decides fitness; reference attainment does not change correctness, ranking or the assigned budget.' } : {}),
        KERNEL_PATH: `${d.out_dir}/workspace`,
        OUTPUT_DIR: d.out_dir,
        CANONICAL: evo ? evo.base : CANONICAL, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
        codebase_context: `${EVAL_DIR}/codebase_context.md`,
        profiling_summary: profileSummary ? profileSummary.summary_path : '',
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
        ...KB_INPUTS,
        ...(ENGINEER_RESERVE_S ? { ENGINEER_CUTOFF_EPOCH } : {}),
        ...facts,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.` +
      // Built inline, not via roleAgent(), so the injection has to be appended here too.
      expertSkillsBlock(isDeep ? 'deep_engineer' : 'engineer') + EXTERNAL_METRIC_INSTRUCTIONS,
      { phase: 'Optimize', label: `${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`, schema: ENG_SCHEMA,
        ...ENGINEER_CALL }   // stopped at the engineer cutoff; its saved patch then goes to recovered verify
    ).then(async (eng) => {
      eng = await checkedExternalMeasurement(eng, `${d.out_dir}/workspace`);
      const patch = `${d.out_dir}/best_patch.diff`;
      const retained = evo ? await evo.record(d, eng, null, patch) : null;
      const measured = !EXTERNAL_OBJECTIVE
        ? (evo ? retained?.patch_ref?.path : await pinSource(patch, d.out_dir)) || patch : patch;
      await retainResult(d, eng, null, measured, evo, roundBase, retained);
      return {d, eng, patch: measured};
    });
    },

    async (prev) => {
      const { d, eng } = prev;
      if (prev.initialization_failed) return {...prev, ver: null, patch: null};
      const patch = prev.patch || `${d.out_dir}/best_patch.diff`;
      // A failed or missing response can still leave a valid patch. Skip verification
      // only after a valid measurement at or below the floor. Verify checks whether
      // the patch exists, applies cleanly, and passes the oracle. In EVO the skipped source stays
      // retained as unqualified evidence; it is verified before any promotion or delivery.
      const trustworthyBelowBaseline = eng && eng.status !== 'failed' && !(primSpeedup(eng) > CANDIDATE_FLOOR) &&
        eng.status !== 'invalid_measurement' && eng.measurement_valid === true &&
        (!HAS_WORKLOAD || Number.isFinite(primSpeedup(eng))) &&
        (!EXTERNAL_OBJECTIVE || (eng.external_measurement_valid === true && Number.isFinite(primSpeedup(eng))));
      if (trustworthyBelowBaseline) {
        return { d, eng, ver: null };
      }
      const recovered = !eng || eng.status === 'failed';
      if (recovered) log(`Round ${round} dir ${d.id}: engineer return ${eng ? 'failed' : 'missing'} — ` +
        `sending best_patch.diff to verify anyway (oracle decides).`);
      return verifyCandidate(d, eng, patch, recovered);
    }
  );

  const clean = results.filter(Boolean);
  if (evo && PROCESS_BOUNDED && await secondsLeft(`archive-r${round}`) <= 7) {
    // The pipeline has settled. Use the existing finalization allowance only to
    // retain completed work; no new engineer or GPU evaluation starts here.
    setProcessDeadline(DEADLINE_EPOCH - FINAL_VALIDATION_RESERVE_S);
    for (const r of clean) {
      const patch = r.patch || `${r.d.out_dir}/best_patch.diff`;
      const binding = r.verification ? await verificationIntact(r.verification, patch) : true;
      const retained = await evo.record(r.d, r.eng, binding === true ? r.ver : null, patch);
      await retainResult(r.d, r.eng, r.ver, patch, evo, roundBase, retained, r.verification);
    }
    history.rounds.push({round, status: 'deadline', improved: false, cumulative,
      directions: directionHistory});
    deadlineHit = true;
    break;
  }
  const isQualified = r => r.ver && (!r.verification || r.verification_status === true) &&
    says(r.ver.status, 'verified') && says(r.ver.correctness, 'pass') && primSpeedup(r.ver) > CANDIDATE_FLOOR;
  const verified = clean.filter(isQualified);

  // --- (d) Build candidate list; integrate if >=2 verified --------------
  // `geomean` here is the PRIMARY metric used for sorting/gating/cumulative: the time-weighted
  // ratio-of-sums when a workload spec is active, else the unweighted geomean (unchanged behavior).
  // The raw unweighted geomean is retained separately for the report.
  const candidate = r => ({
    source: `engineer ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
    geomean: primSpeedup(r.ver), geomean_unweighted: r.ver.verified_geomean,
    weighted: r.ver.verified_weighted != null ? r.ver.verified_weighted : null,
    arithmetic: r.ver.verified_arithmetic || r.ver.verified_geomean,
    per_case: r.ver.per_case || [], patch: r.patch,
    ...(EXTERNAL_OBJECTIVE ? {reported_per_case: r.ver.reported_per_case ?? null} : {}),
    reported_context: reportedEvidence(r.d, r.eng, r.ver, r.patch),
    ...externalFields(r.ver),
  });
  let candidates = verified.map(candidate);

  let integrate = null;
  if (verified.length >= 2) {
    phase('Merge');
    integrate = await observeRole('integrate', `r${round}`, () => agentT(
      roleAgent('integrator', 'integrate', 'Combine this round\'s verified patches into one best implementation.', {
        CANONICAL: evo ? evo.base : CANONICAL, INTEGRATE_DIR: `${EVAL_DIR}/round_${round}/integrate`,
        EVAL_DIR, WORKFLOW_DIR, GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        BEST_INDIVIDUAL: Math.max(...candidates.map(c => c.geomean)),
        PATCHES: verified.map(r => ({ id: r.d.id, specialty: r.d.specialty, title: r.d.title,
          strategy: r.eng ? r.eng.strategy : '', verified_geomean: r.ver.verified_geomean,
          ...externalFields(r.ver),
          files: r.d.focus_files || [], patch: r.patch,
          reported_context: boundedContext(candidates.find(c => c.id === r.d.id).reported_context,
            contextRef(r.d.id, r.d.out_dir), 8192) })),
        ...(evo && A.evolution_case_evidence_view === true
          ? {CANDIDATE_CASE_EVIDENCE: evo.caseInputs(verified.map(r => r.d.id))} : {}),
        INSIGHTS: history.insights,
      }),
      { phase: 'Merge', label: `integrate r${round}`, schema: INTEGRATE_SCHEMA }));
    if (integrate && integrate.best) integrate.best = await checkedExternalMeasurement(
      integrate.best, evo ? evo.base : CANONICAL, integrate.best.patch_file);
    const integPrim = integrate && integrate.best ? primSpeedup(EXTERNAL_OBJECTIVE ? integrate.best : {
      verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean,
    }) : 0;
    if (!EXTERNAL_OBJECTIVE && integrate && integrate.best) {
      const d = {id: `r${round}_integrated`, parent_id: 'ORIGINAL', out_dir: `${EVAL_DIR}/round_${round}/integrate`,
        donor_ids: verified.map(r => r.d.id), title: 'integrated', specialty: 'integrate', gpu_id: GPU_POOL};
      // Preserve the complete integrator report as provisional evidence, before any verifier runs.
      d.reported_context = {...reportedEvidence(d, null, null, integrate.best.patch_file,
        candidates.map(c => c.reported_context)),
        integrator: {...integrate.best, reported_by: 'integrator', notes: integrate.notes}};
      integrate.best.reported_context = d.reported_context;
      const retained = evo ? await evo.record(d, null, null, integrate.best.patch_file) : null;
      const patch = (evo ? retained?.patch_ref?.path : await pinSource(integrate.best.patch_file, d.out_dir))
        || integrate.best.patch_file;
      await retainResult(d, null, null, patch, evo, roundBase, retained);
      if (integrate.conclusion === 'improved' && integPrim > Math.max(...candidates.map(c => c.geomean))) {
        // No new allowance: this independent measurement shares the ordinary search cutoff.
        const r = await verifyCandidate(d, null, patch, false, null);
        integrate.verification = {result: r.ver, patch: r.patch,
          custody: r.verification_status ?? null, input_error: r.verification_input_error || null};
        if (isQualified(r) && !(r.retained?.retention_error || r.retained?.last_record_error))
          candidates.push({...candidate(r), source: 'integrated'});
      }
    }
    // External integration keeps the existing host-checked measurement and identity route.
    if (EXTERNAL_OBJECTIVE && integrate && integrate.conclusion === 'improved' && integrate.best &&
      isQualified({ver: integrate.best}) &&
      integPrim > Math.max(...candidates.map(c => c.geomean))) {
      integrate.best.reported_context = {...reportedEvidence({id: `r${round}_integrated`},
        null, null, integrate.best.patch_file, candidates.map(c => c.reported_context)),
        integrator: {...reportedDetails(integrate.best), notes: integrate.notes,
          reported_included_ids: integrate.best.patches || null}};
      candidates.push({
        reported_context: integrate.best.reported_context,
        source: 'integrated', id: `r${round}_integrated`, title: 'integrated', specialty: 'integrate',
        geomean: integPrim, geomean_unweighted: integrate.best.geomean,
        weighted: integrate.best.weighted != null ? integrate.best.weighted : null,
        arithmetic: integrate.best.arithmetic || integrate.best.geomean,
        per_case: integrate.best.per_case || [], patch: integrate.best.patch_file,
        reported_per_case: integrate.best.reported_per_case ?? null,
        ...externalFields(integrate.best),
      });
    }
  }

  if (EXTERNAL_OBJECTIVE && integrate && integrate.best && candidates.some(c => c.id === `r${round}_integrated`)) {
    const d = {id: `r${round}_integrated`, parent_id: 'ORIGINAL', out_dir: `${EVAL_DIR}/round_${round}/integrate`,
      donor_ids: verified.map(r => r.d.id), title: 'integrated', specialty: 'integrate',
      reported_context: integrate.best.reported_context};
    const ver = {...integrate.best, reported_by: 'integrator',
      verified_geomean: integrate.best.geomean, verified_weighted: integrate.best.weighted};
    const retained = evo ? await evo.record(d, null, ver, integrate.best.patch_file) : null;
    await retainResult(d, null, ver, integrate.best.patch_file, evo, roundBase, retained);
    if (retained && (retained.retention_error || retained.last_record_error)) {
      candidates = candidates.filter(c => c.id !== `r${round}_integrated`);
    }
  }
  candidates.sort((a, b) => b.geomean - a.geomean);
  let winner = candidates[0] || null;
  let improved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));
  let failedCommitRecovery = null;
  // Separate question from `improved`: is the SEARCH advancing, not did it beat the incumbent. The
  // `bestSeen > 0` guard keeps round 1 deciding on `improved` alone; from then on bestSeen >= cumulative
  // at the default floor, so at the default PROGRESS_DELTA this implies `improved` and changes nothing.
  // A round with NO candidate is never progress, so a dead round still counts against MAX_NO_IMPROVE.
  const madeProgress = !!(winner && bestSeen > 0 && winner.geomean > bestSeen * (1 + PROGRESS_DELTA));

  // --- (e) Commit the winner into the canonical workspace ---------------
  if (improved && evo) {
    let promotion = null;
    try { promotion = await evolutionPromotionPatch(evo, winner, round); }
    catch (error) { log(`Round ${round}: could not prepare promotion: ${String(error)}`); }
    if (!promotion) {
      log(`Round ${round}: promotion patch failed; retaining the previous validated source.`);
      improved = false;
    } else {
      winner = {...winner, archive_patch: winner.patch, patch: promotion.patch, source_tree: promotion.tree};
    }
  }
  if (improved && !evo && !EXTERNAL_OBJECTIVE) {
    let promotion = null;
    try {
      promotion = await promotionPatch(roundBase, pinned(winner.patch),
        `${EVAL_DIR}/delivery/round_${round}.index`, `${EVAL_DIR}/delivery/round_${round}.diff`);
    } catch (error) { log(`Round ${round}: could not bind the GEAK winner tree: ${String(error)}`); }
    if (!promotion) improved = false;
    else winner = {...winner, source_tree: promotion.tree};
  }
  if (improved) {
    const checkpointTree = population ? await population.checkpointTree(winner, round) : null;
    await captureCanonicalSelection(history, 'before_commit', round, finalWinner);
    let commitBase = null, priorUntracked = [], commitResult = null, commitDeadline = false;
    try {
      if (!EXTERNAL_OBJECTIVE || PROCESS_BOUNDED || evo) {
        const top = ((await git(CANONICAL, 'rev-parse', '--show-toplevel')) || '').trim();
        if (top !== CANONICAL.replace(/\/+$/, '')) {
          throw new Error('bounded commit requires an isolated canonical repository');
        }
        const untracked = await git(CANONICAL, 'ls-files', '--others', '--exclude-standard', '-z');
        commitBase = ((await git(CANONICAL, 'rev-parse', '--verify', 'HEAD^{commit}')) || '').trim();
        if (untracked === null || !/^[0-9a-f]{40,64}$/.test(commitBase)) {
          throw new Error('could not pin the previous committed canonical source');
        }
        priorUntracked = untracked.split('\0').filter(Boolean);
        const status = await git(CANONICAL, 'status', '--porcelain', '--untracked-files=no');
        if (status === null) throw new Error('could not inspect the canonical source before commit');
        if (status !== '') {
          const dirty = await git(CANONICAL, 'diff', '--binary', commitBase, '--', '.');
          const dirtyPath = `${EVAL_DIR}/precommit_round_${round}.diff`;
          if (dirty === null
              || await hostRun('sh', ['-c', 'set -C; cat > "$1"', 'sh', dirtyPath], {input: dirty}) === null
              || await git(CANONICAL, 'restore', '--source', commitBase, '--staged', '--worktree', '--', '.') === null
              || ((await git(CANONICAL, 'rev-parse', 'HEAD')) || '').trim() !== commitBase
              || await git(CANONICAL, 'status', '--porcelain', '--untracked-files=no') !== '') {
            throw new Error('could not retain and restore tracked changes before commit');
          }
          log(`Round ${round}: retained tracked changes in ${dirtyPath} before restoring the committed source.`);
        }
      }
      await preservePatchObstructions(winner.patch, `round_${round}`);
      commitResult = await observeRole('commit', `r${round}`, () => agentT(
      `You are the TechLead committing round ${round}'s winning patch into the canonical workspace.
\`\`\`bash
set -e
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
# Stage only the selected patch, preserving unrelated untracked harnesses and caches.
# This is the same exact-index apply used by host delivery; do not hand-merge measured bytes.
git apply --index -- ${winner.patch}
git -c user.email=team@workflow -c user.name=team commit -q -m "round ${round} winner: ${winner.source} (${winner.geomean.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
Run the block with fail-fast shell behavior. If the exact apply or commit fails, return committed=false
and explain the error; do not reconstruct, remeasure or add unrelated files. Correctness belongs to the
already verified source, and the host checks the resulting tree. Return JSON {committed, current_best_diff, note}.`,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA }));
    } catch (error) {
      if (EXTERNAL_OBJECTIVE && !evo && (!PROCESS_BOUNDED || error.code !== 'GEAK_PROCESS_DEADLINE')) throw error;
      commitDeadline = error.code === 'GEAK_PROCESS_DEADLINE';
      if (evo) log(`Round ${round}: commit attempt failed; retaining the previous validated source.`);
    }
    if ((!EXTERNAL_OBJECTIVE || evo) && commitResult && commitResult.committed === true
        && await git(CANONICAL, 'diff', '--exit-code', winner.source_tree, 'HEAD', '--', '.') !== '') {
      commitResult = null;
    }
    // GEAK delivery: a commit stands only while its snapshot holds the bytes pinned before verification. Checked
    // after the commit, so a change by the verifier, integrator or commit agent restores the incumbent.
    if (!EXTERNAL_OBJECTIVE && !evo && commitResult && commitResult.committed === true
        && !await pinIntact(winner.patch)) commitResult = null;
    if ((!EXTERNAL_OBJECTIVE || PROCESS_BOUNDED || evo) && (!commitResult || commitResult.committed !== true)) {
      // A qualified winner (individual or integrated) whose commit did not finish stays deliverable,
      // by its pinned bytes (EVO archive row or GEAK snapshot), never by the current promotion file.
      if (!EXTERNAL_OBJECTIVE) keepForDelivery({key: `round_${round}_winner`, id: winner.id, d: null, eng: null,
        reported_context: winner.reported_context,
        ver: {status: 'verified', correctness: 'pass', verified_geomean: winner.geomean_unweighted,
          verified_weighted: winner.weighted, per_case: winner.per_case},
        evidence: winner.geomean, archived: !!evo, source: evo ? null : pinned(winner.patch),
        base: evo ? evo.baseCommit : commitBase});
      let atDeadline = commitDeadline || await secondsLeft(`uncommitted-r${round}`) <= 7;
      // Search has settled. Only host recovery may use the existing Report allowance.
      if (PROCESS_BOUNDED) setProcessDeadline(DEADLINE_EPOCH - FINAL_VALIDATION_RESERVE_S);
      const recovery = commitBase ? await restoreUncommittedRound(commitBase, priorUntracked, round) : null;
      await captureCanonicalSelection(history, 'after_restore', round, finalWinner);
      if (!atDeadline && PROCESS_BOUNDED) {
        setProcessDeadline(DEADLINE_EPOCH - SEARCH_RESERVE_S);
        atDeadline = await secondsLeft(`restored-r${round}`) <= 7;
      }
      const status = atDeadline ? 'deadline_uncommitted' : 'commit_failed';
      if (atDeadline) {
        history.rounds.push({round, status, cumulative, improved: false,
          directions: directionHistory,
          winner: {source: winner.source, geomean: winner.geomean, patch: winner.patch}, recovery});
        plannerStopReason = status;
        log(`Round ${round}: commit did not finish at the cutoff; retained the previous winner for Report/Validate.`);
        break;
      }
      log(`Round ${round}: commit failed; retained the previous winner and continuing search.`);
      improved = false;
      failedCommitRecovery = recovery;
    }
    if (improved) {
      await captureCanonicalSelection(history, 'after_commit', round, winner);
      cumulative = winner.geomean;
      bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
      finalWinner = winner;
      roundsCommitted += 1;
      if (evo) {
        const selected = await evo.selectBest(winner.id);
        if (!selected && !EXTERNAL_OBJECTIVE) deliveryIncomplete.push({id: winner.id,
          key: `round_${round}_winner`, status: 'archive_selection_failed', evidence: cumulative});
      }

      if (population) await population.publishCheckpoint(round, winner, checkpointTree);

      // --- (f) Re-profile the new best ------------------------------------
      profileSummary = await agentT(
        roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
          WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: round,
          COMMANDMENT, PREVIOUS_METRICS: profileSummary,
        }),
        { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
    }
  }

  if (winner && winner.geomean > bestSeen) bestSeen = winner.geomean;
  if (madeProgress || improved) { noImprove = 0; } else { noImprove++; }

  // --- update cross-round memory (insight blackboard + hypothesis ledger)
  const mem = await agentT(
    roleAgent('tech_lead', 'update_memory', 'Distill durable insights + update the hypothesis ledger.' + evolutionInstructions(), {
      EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR,
      ...(archiveHasSource() ? evo.memoryInputs(clean.map(r => r.d)) : {}),
      ROUND_RESULTS: clean.map(r => ({ id: r.d.id, title: r.d.title, specialty: r.d.specialty,
        expected: r.d.expected_speedup, claimed: r.eng ? r.eng.speedup_geomean : 0,
        verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
        notes: r.eng ? r.eng.notes : '', reported_context: boundedContext(
          reportedEvidence(r.d, r.eng, r.ver, r.patch), contextRef(r.d.id, r.d.out_dir), 2048) })),
      ...(EXTERNAL_OBJECTIVE ? { EXTERNAL_RESULTS: clean.map(r => ({id: r.d.id,
        engineer: externalFields(r.eng), verified: externalFields(r.ver)})) } : {}),
      INTEGRATE: integrate ? {...integrate, ...(integrate.best ? {best: {...integrate.best,
        reported_context: boundedContext(integrate.best.reported_context,
          contextRef(`r${round}_integrated`, `${EVAL_DIR}/round_${round}/integrate`), 8192)}} : {})} : null,
      WINNER: winner ? {source: winner.source, geomean: winner.geomean,
        reported_context: boundedContext(winner.reported_context, contextRef(winner.id, `${EVAL_DIR}/round_${round}`), 2048)} : null,
      IMPROVED: improved, REPROFILE_SHIFT: profileSummary ? profileSummary.shift_note : '',
      PRIOR_HISTORY: history,
      ...(STATE_DIR ? { STATE_DIR, CANONICAL, CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase } : {}),
      ...(SHARED_KB ? { SHARED_KB, TARGET_LANGUAGE } : {}),
    }),
    { phase: 'Optimize', label: `tech_lead:memory r${round}`, schema: MEMORY_SCHEMA });
  if (mem) {
    if (mem.insights) history.insights = mem.insights;
    if (mem.ledger) history.ledger = history.ledger.concat(mem.ledger);
    if (mem.bottleneck_now) history.bottleneck_now = mem.bottleneck_now;
    if (mem.suggest_next) history.suggest_next = mem.suggest_next;
  }
  // Enforce the round's KB budget. Honest about what this can and cannot do: stripping a citation
  // does NOT un-influence a direction that was already planned with a card in mind — only the prompt
  // above can do that. What the strip buys is OBSERVABILITY: `kbCapBound` counts the rounds where the
  // planner overran its budget, so "the KB quietly took over planning" becomes a number in the run's
  // return value instead of something you infer from a speedup that got worse.
  if (USE_LEARNED_READ) {
    let seeded = 0; const stripped = [];
    for (const d of directions) {
      if (!d.learned_refs || !d.learned_refs.length) continue;
      if (seeded < KB_DIR_CAP) { seeded++; continue; }
      stripped.push(d.id); d.learned_refs = [];
    }
    // A round where EVERY direction is KB-seeded has no control in it at all.
    if (KB_COLD_DIRECTION && directions.length && directions.every(d => (d.learned_refs || []).length)) {
      const last = directions[directions.length - 1];
      stripped.push(`${last.id}(forced cold)`); last.learned_refs = [];
    }
    if (stripped.length) {
      kbCapBound++;
      log(`Round ${round}: KB budget ${KB_DIR_CAP} exceeded — un-cited ${stripped.join(', ')} ` +
          `(the directions still run). Rounds where the cap bound so far: ${kbCapBound}.`);
    }
  }

  // Join the planner's declared citations against what the verifier independently measured. Derived
  // here rather than self-reported: the planner said which card seeded which direction, the verifier
  // re-measured that direction without knowing, and their join is a fact neither one could fake.
  // `became_winner` and not `verified > 1.0` is the standing test — verified_geomean is measured
  // against the FROZEN baseline, so once a kernel sits at 2.5x cumulative every non-regressing
  // direction clears 1.0 and a card would accrue credit for advancing nothing.
  for (const r of clean) {
    for (const cardRef of (r.d.learned_refs || [])) {
      citations.push({
        card: cardRef, round, direction: r.d.id, specialty: r.d.specialty,
        // null, NOT 0, when the verifier produced nothing. `kb.py` scores <=1.0 as a real loss, so
        // encoding "no evidence" as 0 charged a card for a verifier that crashed or timed out.
        cited_then_verified: EXTERNAL_OBJECTIVE ? primSpeedup(r.ver)
          : (r.ver && Number.isFinite(r.ver.verified_geomean) ? r.ver.verified_geomean : null),
        // `winner` is only the best candidate OF THIS ROUND; `improved` is whether it beat the
        // incumbent by MIN_IMPROVE. Crediting on `winner` alone let a 1.5x round winner confirm a
        // card while the committed best already sat at 2x — a confirmation for advancing nothing,
        // which is precisely what the three-state scoring was written to avoid. Require both.
        became_winner: !!(winner && winner.id === r.d.id && improved),
      });
    }
  }
  history.rounds.push({
    round,
    ...(failedCommitRecovery ? {status: 'commit_failed', recovery: failedCommitRecovery} : {}),
    directions: directionHistory,
    results: clean.map(r => ({ id: r.d.id, claimed: r.eng ? r.eng.speedup_geomean : 0,
      verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
      ...(EXTERNAL_OBJECTIVE ? {engineer_external: externalFields(r.eng), verified_external: externalFields(r.ver)} : {}) })),
    integrate: integrate ? { conclusion: integrate.conclusion, geomean: integrate.best ? integrate.best.geomean : 0,
      ...(!EXTERNAL_OBJECTIVE ? {verification: integrate.verification || null} : {}) } : null,
    winner: winner ? { source: winner.source, geomean: winner.geomean } : null,
    improved, cumulative,
  });
  log(`Round ${round} done. winner=${winner ? winner.source + ' ' + winner.geomean.toFixed(2) + 'x' : 'none'}, cumulative=${cumulative.toFixed(2)}x, noImprove=${noImprove}`);

  // Issue #429: reclaim superseded heavy copies after each round, then CONTINUE optimizing.
  // Disk pressure must never abort the loop — reclaim (and under pressure, --force-heavy) only.
  // Agents (not this JS runtime) execute the bash; the prompt below is also mirrored into the
  // next round's tech_lead context via STORAGE_NOTE so it is not dropped on a dry orchestrator.
  const reclaimCmd =
    `bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${EVAL_DIR} --keep-round ${round}` +
    `; bytes=$(du -sb ${EVAL_DIR} 2>/dev/null | awk '{print $1}');` +
    ` soft=\${GEAK_EVAL_SOFT_BYTES:-34359738368}; hard=\${GEAK_EVAL_HARD_BYTES:-68719476736};` +
    ` if [ -n "$bytes" ] && [ "$bytes" -gt "$hard" ]; then` +
    `   bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${EVAL_DIR} --keep-round ${round} --force-heavy;` +
    ` elif [ -n "$bytes" ] && [ "$bytes" -gt "$soft" ]; then` +
    `   bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${EVAL_DIR} --keep-round ${round};` +
    ` fi` +
    `; echo STORAGE_RECLAIM_DONE round=${round}`;
  await agentT(
    `Storage reclaim after round ${round} (issue #429). Run EXACTLY this bash, then return ` +
    `{ok:true, note:"reclaimed"}. Do NOT stop optimizing — reclaim frees disk so later rounds can run.\n` +
    `\`\`\`bash\n${reclaimCmd}\n\`\`\``,
    { phase: 'Optimize', label: `storage:reclaim r${round}`,
      schema: { type: 'object', additionalProperties: true,
        properties: { ok: { type: 'boolean' }, note: { type: 'string' } },
        required: ['ok'] } });
}
// End-of-run delivery commits the best verified retained source above the incumbent; search promotion keeps MIN_IMPROVE.
if (!EXTERNAL_OBJECTIVE && retainedSources.length) {
  // Qualified custody and host promotion use the existing Report allowance, even with no optional
  // verification allocation. This does not extend the run or the final-validation cutoff.
  if (PROCESS_BOUNDED) setProcessDeadline(DEADLINE_EPOCH - FINAL_VALIDATION_RESERVE_S);
  const dir = `${EVAL_DIR}/delivery`, rows = [];
  const digest = async (file) => ((await hostRun('sha256sum', ['--', file])) || '').slice(0, 64);
  // incumbentEvidence: verifier-bound re-measurement of the already-committed program; it raises the bar.
  let best = null, incumbentEvidence = 0;
  for (const c of [...retainedSources].sort((a, b) => b.evidence - a.evidence)) {
    if (!(c.evidence > Math.max(cumulative, incumbentEvidence, best ? best.primary : 0))) continue;
    const row = {key: c.key, id: c.id, basis: c.ver ? 'verifier' : 'engineer_local', evidence: c.evidence,
      status: 'unmeasured', source: c.source, base_commit: c.base, original_patch: c.original_patch,
      archive: c.archived ? evo.path : null};
    rows.push(row);
    if (!c.ver && !DELIVERY_RESERVE_S) continue;
    if (PROCESS_BOUNDED && setProcessDeadline.secondsLeft() <= 7) {
      row.status = 'incomplete'; continue;
    }
    // A successful cutoff checkpoint supersedes an earlier failed registration.
    const latest = c.archived ? evo.entries.find(r => r.id === c.id && !r.retention_error
      && !r.last_record_error) : null;
    const src = latest?.patch_ref || c.source || null;
    let promo = null, frozen = null;
    try {
      if (!src || !/^[0-9a-f]{40,64}$/.test(c.base || '')) {
        row.status = 'unpinned'; row.error = c.retention_error || 'missing source or base'; continue;
      }
      row.source = src;
      if (await digest(src.path) !== src.sha256) { row.status = 'source_changed'; continue; }
      if (c.ver && c.verification) {
        const binding = await verificationIntact(c.verification, src.path);
        if (binding !== true) {
          row.status = binding === false ? 'source_changed' : 'incomplete'; continue;
        }
      }
      promo = await hostRun('mkdir', ['-p', dir]) === null ? null
        : await promotionPatch(c.base, src, `${dir}/${c.key}.index`, `${dir}/${c.key}.diff`);
      frozen = promo && evo ? `${dir}/${c.key}.source.diff` : promo && promo.patch;
      if (promo && evo && (await hostRun('cp', ['--', src.path, frozen]) === null
          || await digest(frozen) !== src.sha256)) promo = null;
    } catch (error) { log(`[delivery] ${c.key}: ${String(error)}`); promo = null; }
    if (!promo) { row.status = 'assembly_failed'; continue; }
    if (promo.tree === ((await git(CANONICAL, 'rev-parse', 'HEAD^{tree}')) || '').trim()) {
      if (c.ver && c.verification) incumbentEvidence = Math.max(incumbentEvidence, c.evidence);
      // A selected source that does not beat the incumbent's verified evidence is not delivered.
      if (best && best.primary <= incumbentEvidence) { best.row.status = 'below_incumbent'; best = null; }
      row.status = 'already_committed'; continue;
    }
    let ver = c.ver;
    if (!ver) {
      // Only additional verification spends the optional allocation. An exhausted allocation cannot
      // prevent promotion of a result whose authoritative verification has already settled.
      setProcessDeadline(DEADLINE_EPOCH - SEARCH_RESERVE_S + DELIVERY_RESERVE_S);
      if (setProcessDeadline.secondsLeft() <= 7) {
        setProcessDeadline(DEADLINE_EPOCH - FINAL_VALIDATION_RESERVE_S); continue;
      }
      const frozenSha = await digest(frozen);
      try { ver = await agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL: evo ? evo.base : CANONICAL, PATCH: frozen, VERIFY_DIR: `${dir}/${c.key}/verify`,
          ENGINEER_WORKSPACE: `${c.d.out_dir}/workspace`,
          EVAL_DIR, WORKFLOW_DIR, GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
        }),
        { phase: 'Verify', label: `verify ${c.id} (delivery)`, schema: VERIFY_SCHEMA }); }
      finally { setProcessDeadline(DEADLINE_EPOCH - FINAL_VALIDATION_RESERVE_S); }
      // The result binds only to the bytes it measured.
      if (ver && (await digest(frozen) !== frozenSha || await digest(src.path) !== src.sha256)) {
        row.status = 'source_changed'; continue;
      }
    }
    const ok = ver && says(ver.status, 'verified') && says(ver.correctness, 'pass') && primSpeedup(ver) > 0;
    if (ok && evo && c.d) {
      const kept = await evo.record(c.d, c.eng, ver, src.path);
      if (kept && (kept.retention_error || kept.last_record_error)) row.archive_record_status = 'failed';
    }
    Object.assign(row, {status: !ver ? 'unmeasured' : ok ? 'qualified' : String(ver.status || 'unknown'),
      primary: ok ? primSpeedup(ver) : null, source_sha256: src.sha256, tree: promo.tree});
    if (ok && row.primary > Math.max(cumulative, incumbentEvidence) && (!best || row.primary > best.primary)) {
      best = {c, ver, promo, primary: row.primary, row};
    }
  }
  // Qualification has settled; only the host commit uses the existing Report allowance.
  if (PROCESS_BOUNDED) setProcessDeadline(DEADLINE_EPOCH - FINAL_VALIDATION_RESERVE_S);
  if (best && best.primary > Math.max(cumulative, incumbentEvidence)
      && await commitExactTree(best.promo, `delivery ${best.c.key}`)) {
    best.row.status = 'delivered';
    cumulative = best.primary;
    bestPerCase = best.ver.per_case && best.ver.per_case.length ? best.ver.per_case : bestPerCase;
    finalWinner = {source: `delivery ${best.c.key}`, id: best.c.id, title: best.c.d ? best.c.d.title : 'delivery',
      geomean: best.primary, geomean_unweighted: best.ver.verified_geomean,
      weighted: best.ver.verified_weighted != null ? best.ver.verified_weighted : null,
      per_case: best.ver.per_case || [], patch: best.promo.patch, source_tree: best.promo.tree,
      reported_context: reportedEvidence({...best.c.d, id: best.c.id, reported_context: best.c.reported_context},
        best.c.eng, best.ver, best.promo.patch)};
    roundsCommitted += 1;
    if (evo && !await evo.selectBest(best.c.id)) best.row.archive_selection_status = 'failed';
  } else if (best) {
    best.row.status = best.primary > Math.max(cumulative, incumbentEvidence) ? 'commit_failed' : 'below_incumbent';
  }
  history.delivery = {reserve_s: DELIVERY_RESERVE_S, candidates: rows, incumbent_evidence: incumbentEvidence,
    delivered: rows.find(row => row.status === 'delivered')?.key || null};
  const unresolved = [...deliveryIncomplete, ...rows.filter(row => row.basis === 'verifier'
    && row.evidence > Math.max(cumulative, incumbentEvidence) && !['delivered', 'already_committed'].includes(row.status))];
  if (best && best.row.archive_selection_status === 'failed') unresolved.push(best.row);
  history.delivery.incomplete = unresolved;
  const receipt = `${dir}/outcome.json`;
  try {
    if (await hostRun('mkdir', ['-p', dir]) === null
        || await writeFile(`${receipt}.tmp`, JSON.stringify(history.delivery)) === null
        || await hostRun('mv', ['--', `${receipt}.tmp`, receipt]) === null) {
      throw new Error('receipt write unavailable');
    }
  } catch (error) {
    unresolved.push({status: 'receipt_write_failed', path: receipt, error: String(error)});
  }
  if (unresolved.length) log(`[delivery] ${unresolved.length} incomplete source(s); retained evidence ${receipt}.`);
  log(`[delivery] ${rows.length} retained source(s); delivered ${history.delivery.delivered || 'none (incumbent kept)'}.`);
}
return {
  rounds: round, dispatched, cumulativeSpeedup: cumulative, bestPerCase, finalWinner, roundsCommitted,
  history, citations, profileSummary, forcedReplans, kbCapBound,
  // Which condition actually ended the loop (see `stopped_by` in the return value).
  stopReason: plannerStopReason === 'deadline_uncommitted' ? plannerStopReason : deadlineHit ? 'deadline'
    : (dispatched >= BUDGET) ? 'budget'
    : (noImprove >= MAX_NO_IMPROVE) ? 'no_improve'
    : (plannerStopReason || 'tech_lead_stop'),
};
}

// Every retained evolutionary patch is relative to one run base. Build the exact
// proposed tree in a private Git index, then diff it against the current winner.
// This avoids applying an old parent's patch to a different canonical source.
async function evolutionPromotionPatch(evo, winner, round) {
  return promotionPatch(evo.baseCommit, evo.entries.find(row => row.id === winner.id)?.patch_ref,
    `${EVAL_DIR}/evolution/promotion-${round}.index`, `${EVAL_DIR}/evolution/promotion-${round}.diff`);
}
// The same construction for any pinned source {path, sha256} on its recorded base commit.
async function promotionPatch(base, source, index, patch) {
  const indexed = (...args) => hostRun('env', [`GIT_INDEX_FILE=${index}`, 'git', '-C', CANONICAL, ...args]);
  const sourceMatches = async () => source &&
    ((await hostRun('sha256sum', ['--', source.path])) || '').split(/\s/)[0] === source.sha256;
  if (!await sourceMatches()) return null;
  if (await indexed('read-tree', base) === null
      || await indexed('apply', '--cached', '--', source.path) === null || !await sourceMatches()) return null;
  const tree = ((await indexed('write-tree')) || '').trim();
  if (!/^[0-9a-f]{40,64}$/.test(tree)) return null;
  const diff = await indexed('diff', '--cached', '--binary', 'HEAD', '--', '.');
  if (diff === null || await writeFile(patch, diff) === null) return null;
  return {tree, patch};
}

async function restoreUncommittedRound(previous, priorUntracked, round) {
  const unfinished = ((await git(CANONICAL, 'rev-parse', 'HEAD')) || '').trim();
  const retainedRef = `refs/geak/incomplete-round-${round}`;
  const diffPath = `${EVAL_DIR}/uncommitted_round_${round}.diff`;
  const pending = await git(CANONICAL, 'ls-files', '--others', '--exclude-standard', '-z');
  if (pending === null) throw new Error('could not inspect unfinished canonical files');
  const newFiles = pending.split('\0').filter(name => name && !priorUntracked.includes(name));
  // Include newly created source bytes in the retained diff before removing them.
  if (newFiles.length && await git(CANONICAL, '--literal-pathspecs', 'add', '--intent-to-add', '--', ...newFiles) === null) {
    throw new Error('could not retain new unfinished canonical files');
  }
  const partial = await git(CANONICAL, 'diff', '--binary', previous, '--', '.');
  if (!/^[0-9a-f]{40,64}$/.test(unfinished) || partial === null
      || await hostRun('sh', ['-c', 'set -C; cat > "$1"', 'sh', diffPath], {input: partial}) === null
      || await git(CANONICAL, 'update-ref', retainedRef, unfinished, '') === null
      // Mixed reset preserves untracked files, including files staged by an unfinished add -A.
      || await git(CANONICAL, 'reset', '--mixed', previous) === null
      || await git(CANONICAL, 'restore', '--source', previous, '--worktree', '--', '.') === null) {
    throw new Error('could not retain and restore the unfinished canonical commit');
  }
  const untracked = await git(CANONICAL, 'ls-files', '--others', '--exclude-standard', '-z');
  if (untracked === null) throw new Error('could not inspect unfinished canonical files');
  const added = untracked.split('\0').filter(name => name && !priorUntracked.includes(name));
  if (added.length && await git(CANONICAL, '--literal-pathspecs', 'clean', '-f', '--', ...added) === null) {
    throw new Error('could not remove unfinished canonical files');
  }
  const baseline = ((await git(CANONICAL, 'rev-list', '--max-parents=0', previous)) || '').trim();
  const patch = /^[0-9a-f]{40,64}$/.test(baseline)
    ? await git(CANONICAL, 'diff', '--binary', baseline, previous, '--', '.') : null;
  if (patch === null || await writeFile(`${EVAL_DIR}/current_best.diff`, patch) === null
      || ((await git(CANONICAL, 'rev-parse', 'HEAD')) || '').trim() !== previous
      || await git(CANONICAL, 'status', '--porcelain', '--untracked-files=no') !== '') {
    throw new Error('restored canonical source differs from the previous committed winner');
  }
  return {previous_commit: previous, unfinished_commit: unfinished, retained_ref: retainedRef,
    partial_patch: diffPath};
}

// Retained delivery state. Each entry references existing outputs: an EVO archive row (pinned copy) or a
// GEAK snapshot taken before its verification, plus the base commit it was written against.
async function nativeDeliveryBoundary(call) {
  // Only an explicit host capture/export failure has the delivery error class.
  return await call();
}
function nativeDeliveryError(reason) {
  return Object.assign(new Error(`GEAK_NATIVE_DELIVERY_INCOMPLETE: ${reason}; retained sources ${EVAL_DIR}/delivery; archive ${EVAL_DIR}/evolution/archive.json`),
    {code: 'GEAK_NATIVE_DELIVERY_INCOMPLETE'});
}
const retainedSources = [], deliveryIncomplete = [], PINS = new Map();
function keepForDelivery(entry) {
  const index = retainedSources.findIndex(c => c.key === entry.key);
  if (index >= 0) retainedSources[index] = entry;
  else retainedSources.push(entry);
}
async function retainResult(d, eng, ver, patch, evo, base, retained, verification = null) {
  if (EXTERNAL_OBJECTIVE) return; // External measurements retain their own authority and contract.
  const qualified = ver && says(ver.status, 'verified') && says(ver.correctness, 'pass');
  const local = !ver && eng && eng.measurement_valid === true
    && !says(eng.status, 'failed') && !says(eng.status, 'invalid');
  const entry = {key: d.out_dir.slice(EVAL_DIR.length + 1).replace(/\//g, '_'), id: d.id, d, eng,
    ver: qualified ? ver : null, evidence: qualified ? primSpeedup(ver) : local ? primSpeedup(eng) : null,
    archived: !!evo, source: evo ? retained?.patch_ref || null : pinned(patch),
    original_patch: patch, base: evo ? evo.baseCommit : base, verification,
    reported_context: retained?.reported_context || reportedEvidence(d, eng, ver, patch),
    retention_error: retained && (retained.retention_error || retained.last_record_error) || null};
  keepForDelivery(entry);
  // EVO record() already checkpoints source and evidence atomically. GEAK keeps the same evidence
  // beside its immutable source snapshot so a partial round leaves reviewable custody.
  if (!evo && entry.source) {
    try {
      const receipt = `${entry.source.path}.json`;
      if (await writeFile(`${receipt}.tmp`, JSON.stringify(entry)) === null
          || await hostRun('mv', ['--', `${receipt}.tmp`, receipt]) === null)
        entry.retention_error = 'could not retain source evidence';
    } catch (error) { entry.retention_error = String(error); }
  }
}
const pinned = (path) => PINS.has(path) ? {path, sha256: PINS.get(path)} : null;
async function verificationCopy(source, target) {
  try {
    const sha = ((await hostRun('sha256sum', ['--', source])) || '').slice(0, 64);
    if (!/^[0-9a-f]{64}$/.test(sha)
        || await hostRun('cp', ['--no-clobber', '--', source, target]) === null
        || await hostRun('chmod', ['u+w', '--', target]) === null) return null;
    const copy = {path: target, sha256: sha};
    return await verificationIntact(copy, source) ? copy : null;
  } catch (error) { return null; }
}
async function verificationIntact(copy, source) {
  try {
    const measured = await hostRun('sha256sum', ['--', copy.path]);
    const retained = await hostRun('sha256sum', ['--', source]);
    if (measured === null || retained === null) return null;
    return measured.slice(0, 64) === copy.sha256 && retained.slice(0, 64) === copy.sha256;
  } catch (error) { return null; }
}
// True for an unpinned path; false once a pinned snapshot's bytes differ from their pin (or cannot be read).
async function pinIntact(path) {
  try { return !PINS.has(path) || ((await hostRun('sha256sum', ['--', path])) || '').slice(0, 64) === PINS.get(path); }
  catch (error) { return false; }
}
// Copy a GEAK source once into an immutable snapshot; null when it cannot be pinned.
async function pinSource(path, owner) {
  const pin = `${EVAL_DIR}/delivery/sources/${owner.slice(EVAL_DIR.length + 1).replace(/\//g, '_')}.diff`;
  try {
    if (await hostRun('mkdir', ['-p', `${EVAL_DIR}/delivery/sources`]) === null
        || await hostRun('cp', ['--no-clobber', '--', path, pin]) === null
        || await hostRun('chmod', ['a-w', '--', pin]) === null) return null;
    const sha = ((await hostRun('sha256sum', ['--', pin])) || '').slice(0, 64);
    if (!/^[0-9a-f]{64}$/.test(sha) || ((await hostRun('sha256sum', ['--', path])) || '').slice(0, 64) !== sha) return null;
    PINS.set(pin, sha);
    return pin;
  } catch (error) { return null; }
}
async function headCommit() {
  try {
    const head = ((await git(CANONICAL, 'rev-parse', '--verify', 'HEAD^{commit}')) || '').trim();
    return /^[0-9a-f]{40,64}$/.test(head) ? head : null;
  } catch (error) { return null; }
}
// An added source path may already exist untracked in CANONICAL. Preserve only
// paths named by this patch before exact apply; never clean unrelated workspace files.
async function preservePatchObstructions(patch, tag) {
  const pending = await git(CANONICAL, 'ls-files', '--others', '-z'); // includes ignored obstructions
  const changes = await git(CANONICAL, 'apply', '--numstat', '-z', '--', patch);
  if (pending === null || changes === null) throw new Error('cannot inspect patch obstructions');
  const touched = new Set(changes.split('\0').filter(Boolean).map(row => row.split('\t').slice(2).join('\t')));
  const paths = pending.split('\0').filter(name => name && touched.has(name));
  if (!paths.length) return;
  const dir = `${EVAL_DIR}/delivery/obstructions/${tag}`;
  if (await hostRun('mkdir', ['-p', `${EVAL_DIR}/delivery/obstructions`]) === null
      || await hostRun('mkdir', ['--', dir]) === null) throw new Error('cannot create obstruction retention directory');
  for (const name of paths) {
    if (name.startsWith('/') || name.split('/').includes('..')) throw new Error('invalid obstruction path');
    const parent = name.includes('/') ? name.slice(0, name.lastIndexOf('/')) : '';
    if ((parent && await hostRun('mkdir', ['-p', '--', `${dir}/${parent}`]) === null)
        || await hostRun('mv', ['--', `${CANONICAL}/${name}`, `${dir}/${name}`]) === null) {
      throw new Error(`could not preserve obstruction ${name}; retained files ${dir}`);
    }
    log(`[delivery] preserved pre-existing untracked ${name} at ${dir}/${name} before exact apply.`);
  }
}

// Commit exactly the qualified tree; on any mismatch restore the previous commit (incumbent bytes).
async function commitExactTree(promo, message) {
  const previous = await headCommit();
  const untracked = previous ? await git(CANONICAL, 'ls-files', '--others', '--exclude-standard', '-z') : null;
  if (!previous || untracked === null) return false;
  let ok = false;
  try {
    const root = ((await git(CANONICAL, 'rev-list', '--max-parents=0', 'HEAD')) || '').trim();
    await preservePatchObstructions(promo.patch, 'final');
    ok = /^[0-9a-f]{40,64}$/.test(root) && await git(CANONICAL, 'apply', '--index', '--', promo.patch) !== null
      && await git(CANONICAL, '-c', 'user.email=team@workflow', '-c', 'user.name=team', 'commit', '-q', '-m', message) !== null
      && await git(CANONICAL, 'diff', '--exit-code', promo.tree, 'HEAD', '--', '.') === '';
    const best = ok ? await git(CANONICAL, 'diff', '--binary', root, 'HEAD', '--', '.') : null;
    ok = ok && best !== null && await writeFile(`${EVAL_DIR}/current_best.diff`, best) !== null;
  } catch (error) { log(`[delivery] commit failed: ${String(error)}`); ok = false; }
  if (!ok) await restoreUncommittedRound(previous, untracked.split('\0').filter(Boolean), 'delivery');
  return ok;
}

// Set the variables the Report/Validate postlude reads from one OptimizeResult.
let stopReason = '';
function applyOptimizeResult(r) {
  round = r.rounds;
  dispatched = r.dispatched;
  cumulative = r.cumulativeSpeedup;
  bestPerCase = r.bestPerCase;
  finalWinner = r.finalWinner;
  roundsCommitted = r.roundsCommitted;
  Object.assign(history, r.history);
  citations.push(...r.citations);
  profileSummary = r.profileSummary;
  stopReason = r.stopReason;
  deadlineHit = r.stopReason === 'deadline' || r.stopReason === 'deadline_uncommitted';
  forcedReplans = r.forcedReplans;
  kbCapBound = r.kbCapBound;
}
