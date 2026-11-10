// Native planner construction shared by ordinary Optimize and a source-bound decision-only call.
function nativeEvolutionPlannerSchema(hasArchive) {
  return hasArchive ? {
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
}

function nativePlannerCall(context) {
  const {EVAL_DIR, round, remaining, cumulative, handoffMetric, BASELINE_GEOMEAN_MS, profileSummary, bestPerCase, finalWinner, measurementRef, history, archiveHasSource, evo, GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS, researchBriefPath, BASELINE_PER_CASE, getLeft, KK_OPERATOR, KK_LANGUAGE, KK_REFS, ENGINEER_CALL, DEADLINE_EPOCH, ENGINEER_RESERVE_S, ENGINEER_CUTOFF_EPOCH, evoPlanSchema, allowInitialBenchmark = true, factsInput = deliveryFactsInput} = context;
  let caseMeasurementObservation = null, constructedCall = null;
  const planInputs = (forbidStop, facts = {}) => {
    const left = getLeft();
    const inputs = {
    EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative, PRIMARY_METRIC: handoffMetric,
    BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
    ...currentBestCaseInputs(bestPerCase, finalWinner, measurementRef(finalWinner),
      {baselinePerCase: BASELINE_PER_CASE, evalDir: EVAL_DIR, allowInitialBenchmark}), HISTORY: history,
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
    };
    if (CASE_MEASUREMENT_VIEW !== undefined) {
      try {
        const visibleEntries = inputs.EVOLUTION_ENTRIES;
        const initialOriginal = allowInitialBenchmark && !finalWinner && bestPerCase === BASELINE_PER_CASE;
        const rawCases = value => value ? {native_primary: primSpeedup(value),
          per_case: value.per_case ?? null, evidence_context: compactCaseMeasurementContext(value)} : null;
        const decisionView = {
          CURRENT_BEST_PER_CASE: inputs.CURRENT_BEST_PER_CASE,
          CURRENT_BEST_CASE_MEASUREMENT: inputs.CURRENT_BEST_CASE_MEASUREMENT ?? null,
          archive_present: Array.isArray(visibleEntries),
          archive_entries: Array.isArray(visibleEntries) ? visibleEntries.map(row => ({id: row.id,
            case_evidence_present: Object.hasOwn(row, 'case_evidence'), case_evidence: row.case_evidence ?? null})) : null,
        };
        // Copy now, before the planner runs or later archive/verifier updates.
        // This is an observation only, never another prompt or acceptance input.
        caseMeasurementObservation = JSON.parse(JSON.stringify({mode: CASE_MEASUREMENT_VIEW,
          authority: 'constructed planner case inputs and retained reported measurements; not proof of model reads',
          decision_view: decisionView, decision_view_bytes: contextBytes(decisionView),
          raw_current_best: {initial_original: initialOriginal,
            source_ref: initialOriginal ? {path: `${EVAL_DIR}/baseline_timing.json`} : measurementRef(finalWinner),
            native_primary: initialOriginal ? null : cumulative,
            per_case: bestPerCase, evidence_context: compactCaseMeasurementContext(finalWinner)
              || compactCaseMeasurementContext(finalWinner?.reported_context?.verifier)},
          raw_archive_entries: Array.isArray(visibleEntries) ? visibleEntries.map(visible => {
            const row = evo?.entries.find(entry => entry.id === visible.id);
            return {id: visible.id, present: !!row, patch_ref: row?.patch_ref ?? null,
              engineer: rawCases(row?.engineer), verifier: rawCases(row?.verifier)};
          }) : null}));
      } catch (_) {
        caseMeasurementObservation = {mode: CASE_MEASUREMENT_VIEW, unavailable: 'case_snapshot_serialization_failed'};
      }
    }
    return inputs;
  };

  const callPlanner = async (attempt, intro, forbidStop, label) => {
    const inputs = planInputs(forbidStop, await factsInput());
    const prompt = roleAgent('tech_lead', 'plan_round', intro, inputs);
    const schema = archiveHasSource() ? evoPlanSchema : PLAN_SCHEMA;
    let plannerObservation;
    if (A.record_planner_inputs === true || context.recordPlannerInputs === true) {
      const components = {};
      const component = (name, value) => {
        try {
          const serialized = JSON.stringify(value);
          const representedValue = serialized === undefined ? undefined : JSON.parse(serialized);
          const state = representedValue === undefined ? 'absent' : representedValue === null ? 'null'
            : representedValue === '' || (Array.isArray(representedValue) ? representedValue.length === 0
              : typeof representedValue === 'object' && Object.keys(representedValue).length === 0) ? 'empty' : 'present';
          components[name] = {state, ...(serialized === undefined ? {} : {serialized})};
        } catch (_) { components[name] = {state: 'error'}; }
      };
      component('inputs', inputs);
      component('role_prompt', prompt);
      component('response_schema', schema);
      component('prepared_original_ref', A.prepared_original_ref);
      component('TASK', Object.hasOwn(inputs, 'TASK') ? inputs.TASK
        : typeof TASK === 'string' && TASK ? TASK : undefined);
      for (const key of new Set(['PROFILE_SUMMARY', 'HISTORY', 'EVOLUTION_ENTRIES', 'DEEP_COST',
        'DEEP_SEARCH_BRIEF', 'LEARNED_KB_BUDGET', 'MINUTES_REMAINING', 'ENGINEER_CUTOFF_EPOCH',
        'STOP_NOT_PERMITTED', ...Object.keys(inputs)])) component(`inputs.${key}`, inputs[key]);
      plannerObservation = {eval_dir: EVAL_DIR, round, logical_attempt: attempt, components};
    }
    const options = { phase: 'Optimize', label, schema, ...ENGINEER_CALL,
      ...(plannerObservation ? {plannerObservation} : {}) };
    if (context.recordPlannerInputs === true) constructedCall = JSON.parse(JSON.stringify({
      boundary: 'before_agentT', prompt, options, inputs}));
    return agentT(prompt, options);
  };

  return {callPlanner, planInputs, caseMeasurementObservation: () => caseMeasurementObservation,
    constructedCall: () => constructedCall};
}

// Native round sizing; source-only decisions with unknown budget do not invoke this allocator.
function nativeDirectionSelection(plan, remaining, round, evolution) {
  let directions = plan.directions.slice(0, remaining).map((d, i) => ({
    ...d, idx: i, id: d.id || `r${round}_d${i}`}));
  const deepDir = directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) directions = [deepDir];
  if (evolution) {
    let allocated = 0;
    directions = directions.filter(d => {
      const cost = d.specialty === 'deep_explore' ? DEEP_COST : 1;
      if (allocated + cost > remaining) return false;
      allocated += cost;
      return true;
    });
  }
  return directions;
}
