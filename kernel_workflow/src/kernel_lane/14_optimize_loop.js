// The Optimize phase. An optimizer starts from the state Setup/resume/WarmStart left behind and
// returns an OptimizeResult; applyOptimizeResult() hands it to the Report/Validate postlude:
//   { rounds, dispatched, cumulativeSpeedup, bestPerCase, finalWinner, roundsCommitted, history,
//     citations, profileSummary, stopReason, forcedReplans, kbCapBound }
// The greedy loop body keeps its original indentation so its prompts stay byte-identical.
async function greedyOptimize(start) {
let { cumulative, bestSeen, bestPerCase, finalWinner, profileSummary } = start;
const history = start.history;   // extended in place, as before
const citations = [];
let dispatched = 0, round = 0, noImprove = 0, roundsCommitted = 0;
let plannerStopReason = '', deadlineHit = false, forcedReplans = 0, kbCapBound = 0;
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
  if (DEADLINE_EPOCH) log(`[deadline] ${(left / 60).toFixed(0)} min remain before the hard stop.`);

  round++;
  const remaining = BUDGET - dispatched;
  phase('Optimize');

  // --- (a) Plan the round (TechLead) ------------------------------------
  const planInputs = (forbidStop) => ({
    EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
    BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
    CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
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

  let plan = await agentT(
    roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).',
      planInputs(false)),
    { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });

  // NO EARLY STOP: keep refusing while the window is materially unspent. This used to refuse ONCE,
  // on the reasoning that a role repeating "stop" has run out of ideas. Measured at a 2h budget that
  // cost ~3 min (write_req_to_token_pool_triton stopped at 1.95h/2.00h) and looked harmless; the cost
  // is proportional to the budget, so at 8h the same one-shot escape hands back hours. The caller's
  // rule is "must keep optimizing until the budget is spent", so the refusal has to be as long-lived
  // as the window. Bounded by MAX_FORCED_REPLANS so a role that cannot produce a direction at all
  // degrades to stopping instead of spinning the window away on planning.
  while (DEADLINE_EPOCH && left > NO_STOP_S && forcedReplans < MAX_FORCED_REPLANS &&
         (!plan || plan.stop || !plan.directions || plan.directions.length === 0)) {
    forcedReplans++;
    log(`Round ${round}: stop=true with ${(left / 60).toFixed(0)} min left — refusing (#${forcedReplans}), re-planning.`);
    plan = await agentT(
      roleAgent('tech_lead', 'plan_round', 'Your stop was refused: plan at least one direction.',
        planInputs(true)),
      { phase: 'Optimize', label: `tech_lead:replan r${round}#${forcedReplans}`, schema: PLAN_SCHEMA });
    left = await secondsLeft(`replan-r${round}`);
  }

  if (plan && Array.isArray(plan.directions) && ROOFLINE_STATUS.startsWith('unknown')) {
    const before = plan.directions.length;
    plan.directions = plan.directions.filter((d) =>
      !/roofline|peak\s*(?:flops|bandwidth)|%\s*of\s*peak/i.test(
        `${d && d.title || ''} ${d && d.prompt || ''} ${d && d.why || ''}`));
    if (plan.directions.length !== before) {
      log(`Round ${round}: removed ${before - plan.directions.length} direction(s) that relied on unknown peaks.`);
    }
    if (before > 0 && plan.directions.length === 0) {
      plannerStopReason = 'all_directions_filtered_unknown_peaks';
    }
  }

  if (!plan || plan.stop || !plan.directions || plan.directions.length === 0) {
    if (plannerStopReason === 'all_directions_filtered_unknown_peaks') {
      log(`Round ${round}: harness stopped after all directions were filtered because calibrated peaks are unknown.`);
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
  const roundCost = directions.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  dispatched += roundCost;
  log(`Round ${round}: ${directions.length} direction(s) [${directions.map(d => d.specialty).join(', ')}], cost ${roundCost}, budget ${dispatched}/${BUDGET}`);

  // --- (b,c) Optimize -> Verify, pipelined per direction ----------------
  const results = await pipeline(
    directions,
    (d) => {
      const isDeep = d.specialty === 'deep_explore';
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
      return agentT(
      `You are Engineer ${d.id} (specialty=${d.specialty}) for round ${round}.
First create YOUR private workspace, then optimize.
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
${readLine} If KK_OPERATOR is non-empty, also consult the operator/language SOTA cards under
KERNEL_KNOWLEDGE_DIR per your role's "operator/language SOTA knowledge (REFERENCE ONLY)" section
(facts/how-to only; measure everything; never go below baseline).
Save best_patch.diff via \`cd <KERNEL_PATH> && git diff > ${d.out_dir}/best_patch.diff\` when COMMANDMENT's primary speedup exceeds ${CANDIDATE_FLOOR_TXT}. Use its audited oracle aggregation. Before saving, include any new editable source files with \`git add -N -- <new-source-files>\`; exclude logs, measurements and build artifacts.

## Inputs
${cfg({
        SPECIALTY: d.specialty,
        DIRECTION: { id: d.id, title: d.title, focus_files: d.focus_files || [], expected_speedup: d.expected_speedup, prompt: d.prompt },
        ...(isDeep ? { TARGET: (d.expected_speedup ? `Investigate the proposed ${d.expected_speedup}x speedup. ` : '') +
          'Improve the measured primary metric within the assigned budget. Use roofline estimates only with supported work counts and product-specific limits; they are not required attainment targets.' } : {}),
        KERNEL_PATH: `${d.out_dir}/workspace`,
        OUTPUT_DIR: d.out_dir,
        CANONICAL, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
        codebase_context: `${EVAL_DIR}/codebase_context.md`,
        profiling_summary: profileSummary ? profileSummary.summary_path : '',
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
        ...KB_INPUTS,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.` +
      // Built inline, not via roleAgent(), so the injection has to be appended here too.
      expertSkillsBlock(isDeep ? 'deep_engineer' : 'engineer') + EXTERNAL_METRIC_INSTRUCTIONS,
      { phase: 'Optimize', label: `${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`, schema: ENG_SCHEMA }
    ).then(async (eng) => ({ d, eng: await checkedExternalMeasurement(eng, `${d.out_dir}/workspace`) }));
    },

    (prev) => {
      const { d, eng } = prev;
      const patch = `${d.out_dir}/best_patch.diff`;
      // A failed or missing response can still leave a valid patch. Skip verification
      // only after a valid measurement at or below the floor. Verify checks whether
      // the patch exists, applies cleanly, and passes the oracle.
      const trustworthyBelowBaseline = eng && eng.status !== 'failed' && !(primSpeedup(eng) > CANDIDATE_FLOOR) &&
        eng.status !== 'invalid_measurement' && eng.measurement_valid === true &&
        (!EXTERNAL_OBJECTIVE || (eng.external_measurement_valid === true && Number.isFinite(primSpeedup(eng))));
      if (trustworthyBelowBaseline) {
        return { d, eng, ver: null };
      }
      const recovered = !eng || eng.status === 'failed';
      if (recovered) log(`Round ${round} dir ${d.id}: engineer return ${eng ? 'failed' : 'missing'} — ` +
        `sending best_patch.diff to verify anyway (oracle decides).`);
      return agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          ENGINEER_WORKSPACE: `${d.out_dir}/workspace`,
          EVAL_DIR, WORKFLOW_DIR, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
        }),
        { phase: 'Verify', label: `verify ${d.id}${recovered ? ' (recovered)' : ''}`, schema: VERIFY_SCHEMA }
      ).then(async (ver) => ({ d, eng, ver: await checkedExternalMeasurement(ver, CANONICAL, patch), patch }));
    }
  );

  const clean = results.filter(Boolean);
  const verified = clean.filter(r => r.ver && says(r.ver.status, 'verified') &&
    says(r.ver.correctness, 'pass') && primSpeedup(r.ver) > CANDIDATE_FLOOR);

  // --- (d) Build candidate list; integrate if >=2 verified --------------
  // `geomean` here is the PRIMARY metric used for sorting/gating/cumulative: the time-weighted
  // ratio-of-sums when a workload spec is active, else the unweighted geomean (unchanged behavior).
  // The raw unweighted geomean is retained separately for the report.
  let candidates = verified.map(r => ({
    source: `engineer ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
    geomean: primSpeedup(r.ver), geomean_unweighted: r.ver.verified_geomean,
    weighted: r.ver.verified_weighted != null ? r.ver.verified_weighted : null,
    arithmetic: r.ver.verified_arithmetic || r.ver.verified_geomean,
    per_case: r.ver.per_case || [], patch: r.patch,
    ...externalFields(r.ver),
  }));

  let integrate = null;
  if (verified.length >= 2) {
    phase('Merge');
    integrate = await agentT(
      roleAgent('integrator', 'integrate', 'Combine this round\'s verified patches into one best implementation.', {
        CANONICAL, INTEGRATE_DIR: `${EVAL_DIR}/round_${round}/integrate`,
        EVAL_DIR, WORKFLOW_DIR, GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        BEST_INDIVIDUAL: Math.max(...candidates.map(c => c.geomean)),
        PATCHES: verified.map(r => ({ id: r.d.id, specialty: r.d.specialty, title: r.d.title,
          strategy: r.eng ? r.eng.strategy : '', verified_geomean: r.ver.verified_geomean,
          ...externalFields(r.ver),
          files: r.d.focus_files || [], patch: r.patch })),
        INSIGHTS: history.insights,
      }),
      { phase: 'Merge', label: `integrate r${round}`, schema: INTEGRATE_SCHEMA });
    if (integrate && integrate.best) integrate.best = await checkedExternalMeasurement(
      integrate.best, CANONICAL, integrate.best.patch_file);
    const integPrim = integrate && integrate.best ? primSpeedup(EXTERNAL_OBJECTIVE ? integrate.best : {
      verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean,
    }) : 0;
    if (integrate && integrate.conclusion === 'improved' && integrate.best &&
      integPrim > Math.max(...candidates.map(c => c.geomean))) {
      candidates.push({
        source: 'integrated', id: `r${round}_integrated`, title: 'integrated', specialty: 'integrate',
        geomean: integPrim, geomean_unweighted: integrate.best.geomean,
        weighted: integrate.best.weighted != null ? integrate.best.weighted : null,
        arithmetic: integrate.best.arithmetic || integrate.best.geomean,
        per_case: integrate.best.per_case || [], patch: integrate.best.patch_file,
        ...externalFields(integrate.best),
      });
    }
  }

  candidates.sort((a, b) => b.geomean - a.geomean);
  const winner = candidates[0] || null;
  const improved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));
  // Separate question from `improved`: is the SEARCH advancing, not did it beat the incumbent. The
  // `bestSeen > 0` guard keeps round 1 deciding on `improved` alone; from then on bestSeen >= cumulative
  // at the default floor, so at the default PROGRESS_DELTA this implies `improved` and changes nothing.
  // A round with NO candidate is never progress, so a dead round still counts against MAX_NO_IMPROVE.
  const madeProgress = !!(winner && bestSeen > 0 && winner.geomean > bestSeen * (1 + PROGRESS_DELTA));

  // --- (e) Commit the winner into the canonical workspace ---------------
  if (improved) {
    await agentT(
      `You are the TechLead committing round ${round}'s winning patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
# Try a plain apply first, then a 3-way apply (auto-reconciles context-line drift against the blobs)
# before falling back to a manual reconstruction. --3way resolves most "patch does not apply" cases
# that are just context offsets, so the manual path is only hit on a genuine semantic conflict.
git apply ${winner.patch} || git apply --3way ${winner.patch}
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "round ${round} winner: ${winner.source} (${winner.geomean.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH \`git apply\` and \`git apply --3way\` fail, inspect the patch and apply it manually (edit the
files to match the patch's intent), then \`add -A\` + commit. The applied source is NOT guaranteed to
match the patch verbatim after a hand-merge, so after committing, RE-RUN the COMMANDMENT correctness
check (cd ${CANONICAL} && the COMMANDMENT CORRECTNESS cmd via gpu_lock); only report committed=true if
it still passes. (When a clean \`git apply\`/\`--3way\` succeeds, correctness was already verified and a
re-check is not required.) Return JSON {committed, current_best_diff, note}.`,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA });
    cumulative = winner.geomean;
    bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
    finalWinner = winner;
    roundsCommitted += 1;

    // --- (f) Re-profile the new best ------------------------------------
    profileSummary = await agentT(
      roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: round,
        COMMANDMENT, PREVIOUS_METRICS: profileSummary,
      }),
      { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
  }

  if (winner && winner.geomean > bestSeen) bestSeen = winner.geomean;
  if (madeProgress || improved) { noImprove = 0; } else { noImprove++; }

  // --- update cross-round memory (insight blackboard + hypothesis ledger)
  const mem = await agentT(
    roleAgent('tech_lead', 'update_memory', 'Distill durable insights + update the hypothesis ledger.', {
      EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR,
      ROUND_RESULTS: clean.map(r => ({ id: r.d.id, title: r.d.title, specialty: r.d.specialty,
        expected: r.d.expected_speedup, claimed: r.eng ? r.eng.speedup_geomean : 0,
        verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
        notes: r.eng ? r.eng.notes : '' })),
      ...(EXTERNAL_OBJECTIVE ? { EXTERNAL_RESULTS: clean.map(r => ({id: r.d.id,
        engineer: externalFields(r.eng), verified: externalFields(r.ver)})) } : {}),
      INTEGRATE: integrate, WINNER: winner ? { source: winner.source, geomean: winner.geomean } : null,
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
    directions: directions.map(d => ({ id: d.id, title: d.title, specialty: d.specialty,
      learned_refs: d.learned_refs || [] })),
    results: clean.map(r => ({ id: r.d.id, claimed: r.eng ? r.eng.speedup_geomean : 0,
      verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
      ...(EXTERNAL_OBJECTIVE ? {engineer_external: externalFields(r.eng), verified_external: externalFields(r.ver)} : {}) })),
    integrate: integrate ? { conclusion: integrate.conclusion, geomean: integrate.best ? integrate.best.geomean : 0 } : null,
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
return {
  rounds: round, dispatched, cumulativeSpeedup: cumulative, bestPerCase, finalWinner, roundsCommitted,
  history, citations, profileSummary, forcedReplans, kbCapBound,
  // Which condition actually ended the loop (see `stopped_by` in the return value).
  stopReason: deadlineHit ? 'deadline'
    : (dispatched >= BUDGET) ? 'budget'
    : (noImprove >= MAX_NO_IMPROVE) ? 'no_improve'
    : (plannerStopReason || 'tech_lead_stop'),
};
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
  deadlineHit = r.stopReason === 'deadline';
  forcedReplans = r.forcedReplans;
  kbCapBound = r.kbCapBound;
}
