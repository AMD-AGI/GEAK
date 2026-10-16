// ===========================================================================
// Write this run's outcome back — the producer half of the loop. One call files all three: the
// kb_artifacts/ tree (which the remote payload is derived FROM, and which a curation pass edits),
// the on-disk store the read addresses by key, and the service when credentialed.
// The script applies its own gate and never fails the run; the `finalPrimary > 1.0`
// pre-check just avoids spending an agent on a run that cannot pass the gate anyway.
// ===========================================================================
let kb_written = null;
if (KB_WRITE_OK && GFX && Number.isFinite(finalPrimary) && finalPrimary > 1.0) {
  const kernelClass = (analysis && analysis.kernel_type) || 'unknown';
  const finalPatch = report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`;
  const reportPath = report && report.report_path ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`;
  // What the store cannot infer from a diff but needs to curate this entry later. `direction` is the
  // IDEA, and only a warm start supplies a real one: a round title is free-form per run, so kebabbing
  // it would mint a unique label every time and group nothing. Unlabeled is honest — resolve treats
  // such an entry as its own direction, and a curation pass assigns the shared label later.
  const winnerDirection = kebab(warm_start.direction || '');
  const metricKind = HAS_WORKLOAD ? 'time_weighted' : 'geomean';
  // Commas separate the list, so a case name may not contain one.
  const caseNames = (bestPerCase || []).map(c => c && String(c.name || '').replace(/,/g, ';'))
    .filter(Boolean).join(',');
  // write-remote runs the directory write first and files the same entry in the store, so the two
  // planes cannot drift; its extra `remote` field rides along under the schema's open object.
  // `--plane both` extends that ordering one hop further: directory tree, then local store, then the
  // service. Unlike the read there is no branch on credentials here, because `_open_plane` already
  // makes the right call — a `both` without them degrades to the local planes and reports
  // `remote_unavailable` rather than failing, which is the behaviour we would have written by hand.
  const remoteWriteOn = KB_REMOTE !== 'off' && !!KB_STORE_DIR;
  const writeCmd = remoteWriteOn
    ? `write-remote --plane both --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
    : KB_MODE === 'store'
      ? `write-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
      : 'write';
  kb_written = await agentT(
    `You are the experience writer. Run EXACTLY this command (it applies its own gates and prints a ` +
    `single-line JSON) and return that JSON verbatim as StructuredOutput. If the command errors, return ` +
    `{"written": false, "reason": "io_error"}` +
    (remoteWriteOn
      ? ` — do NOT retry it and do NOT adjust its arguments. It may reach the shared KB Store service, ` +
        `and every write that service accepts is PERMANENT (it exposes no delete), so a second attempt ` +
        `creates a second permanent record instead of fixing the first.`
      : `.`) + `
\`\`\`bash
${remoteWriteOn ? KB_ENV_PRELUDE + '\n' : ''}python3 ${JSON.stringify(EXPERIENCE_STORE)} ${writeCmd} --root ${JSON.stringify(KB_ARTIFACTS_DIR)} \\
  --kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\
  --gfx ${GFX} --kernel-class ${JSON.stringify(kernelClass)} \\
  --speedup ${finalPrimary} --baseline-wall-ms ${BASELINE_GEOMEAN_MS} \\
  --patch ${JSON.stringify(finalPatch)} --eval-dir ${JSON.stringify(EVAL_DIR)} \\
  --report ${JSON.stringify(reportPath)} --metric-kind ${metricKind} \\
  --direction ${JSON.stringify(winnerDirection)} --case-names ${JSON.stringify(caseNames)}\
${OP_SPEC.dtype ? ` \\\n  --precision ${JSON.stringify(String(OP_SPEC.dtype))}` : ''}\
${warm_start.exp_dir ? ` \\\n  --parent ${JSON.stringify(warm_start.exp_dir)}` : ''}
\`\`\``,
    { phase: 'Validate', label: 'kb:write', schema: WARMSTART_WRITE_SCHEMA });
  const remoteWrite = (kb_written && kb_written.remote) || null;
  if (remoteWrite) {
    log(remoteWrite.written
      // `replaced` distinguishes the two write outcomes the store has: a NEW patch appends a session
      // under the same key, the SAME patch remeasured lands back on its own content-addressed one.
      ? `[kb] plane=store ${remoteWrite.replaced ? 'updated' : 'appended'} ` +
        `${remoteWrite.canonical_id}/${remoteWrite.session_id}${remoteWrite.champion ? ' (champion)' : ''}`
      : `[kb] plane=store not written: ${remoteWrite.reason || 'unknown'}`);
  }
  log(kb_written && kb_written.written
    ? `[kb] experience written: ${kb_written.slug} (speedup ${finalPrimary.toFixed(2)}, direction ` +
      `${winnerDirection || 'unlabeled'})`
    : kb_written && kb_written.reason === 'duplicate_impl'
      // Not a failure — the expected outcome of adopting a warm start and not beating it.
      ? `[kb] experience already known — counted as reproduction #${kb_written.reproductions || '?'} of ` +
        `${kb_written.reproduced || kb_written.dir}`
      : `[kb] experience not written: ${kb_written ? kb_written.reason : 'writer returned nothing'}`);
}

// finalPrimary is the total vs the pristine baseline; when a warm-start patch was adopted, split out
// the delta ABOVE it so a KB-derived gain is never reported as this run's own work.
//
// With no round committed after the adopt, the workspace still holds exactly the adopted code, and
// the two numbers are two measurements of the SAME thing taken at different times — the ratio is
// bench noise (~2% on this box; `_gemm_a16_w16_kernel` reported 0.98 and read as a regression).
// Report 1.0 and say why, rather than dressing the noise up as a delta.
const noRoundsAfterAdopt = !!(warm_start.adopted && roundsCommitted === 0);
const incrementalSpeedup = warm_start.adopted && warm_start.adopted_speedup
  ? (noRoundsAfterAdopt ? 1.0
     : Number.isFinite(finalPrimary) ? finalPrimary / warm_start.adopted_speedup : null)
  : finalPrimary;

return {
  mode: MODE,
  // `instance` = this lane's eval dir: a stable identity so the parser dedupes a timeline reached
  // twice (parent-merge + glob) without collapsing two distinct lanes that share a shape.
  llm_timeline: LLM_STATS ? { ...LLM_TL, instance: EVAL_DIR } : undefined,
  target_language: MODE === 'author' ? TARGET_LANGUAGE : undefined,
  authored: MODE === 'author' ? true : undefined,
  eval_dir: EVAL_DIR,
  kernel_name: KERNEL_NAME,
  workload_aligned: HAS_WORKLOAD,
  final_speedup: finalPrimary,                 // PRIMARY metric (weighted when workload-aligned)
  final_weighted: finalWeighted,
  final_geomean: finalGeomean,
  final_arithmetic: validation ? validation.director_verified_speedup_arithmetic : null,
  tech_lead_reported_geomean: report ? report.final_speedup_geomean : cumulative,
  validation_status: validation ? validation.validation_status : 'unknown',
  // What KIND of number `final_speedup` is, so bake-off ranking and campaign summaries don't read a
  // host_bound/unprimed float as a clean device-time win. 'not_applicable' = this lane's own baseline
  // ratio, no frozen-oracle receipt behind it.
  timing_basis: validation ? validation.timing_basis || 'unknown' : 'unknown',
  rounds: report ? report.rounds : round,
  budget_used: dispatched,
  budget_total: BUDGET,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
  // null when the step was off, or the run earned nothing worth a card.
  // Which condition actually ended the loop. RECORDED, not inferred: on the previous campaign this
  // existed only in the driver's return value, which the caller sees after ALL its kernels finish —
  // so a parent-process death erased it for every kernel already done, and the realised window had
  // to be reconstructed from queue timestamps. It is the whole point of an enforced window that you
  // can tell a run that used its budget from one that gave up.
  stopped_by: stopReason,
  deadline_hit: deadlineHit,
  forced_replans: forcedReplans,
  // What the plan cited and whether it carried its round. Returned ALWAYS, including when nothing was
  // cited (an empty array is the finding: the KB was read and nothing in it was worth acting on).
  learned_citations: citations,
  // Rounds where the planner overran the KB budget. A KB that binds every round is one that has taken
  // over planning, which is the regression this budget exists to catch.
  kb_cap_bound_rounds: kbCapBound,
  // The monoculture canary, and it costs no GPU time: how many DISTINCT (specialty, focus_files)
  // directions this run explored vs how many it issued. A KB that helps raises the verified speedup;
  // a KB that cages lowers this without raising that. Reported on KB-less runs too — those are the
  // baseline it has to be read against.
  direction_entropy: (() => {
    const seen = new Set(); let n = 0;
    for (const r of history.rounds) for (const d of (r.directions || [])) {
      n++; seen.add(`${d.specialty}|${(d.focus_files || []).slice().sort().join(',')}`);
    }
    return n ? { distinct: seen.size, issued: n, ratio: +(seen.size / n).toFixed(3) } : null;
  })(),
  learned_card: learned_card && learned_card.card_path
    ? { action: learned_card.action || '', card_path: learned_card.card_path, key: learned_card.key || '' }
    : null,
  // Warm-start (local experience KB) outcome. adopted=false + read_reason on a cold start.
  warm_start: {
    adopted: warm_start.adopted,
    read_reason: warm_start.read_reason,
    slug: warm_start.slug || '',
    // How the store was reached and what it filtered out: tells a genuine cold start from a
    // name/language mismatch that only looks like an empty KB.
    match_tier: warm_start.match_tier || '',
    filtered: warm_start.filtered || null,
    adopted_speedup: warm_start.adopted ? warm_start.adopted_speedup : null,
    total_speedup: Number.isFinite(finalPrimary) ? finalPrimary : null,
    incremental_speedup: incrementalSpeedup,
    // true = the run adopted a stored patch and never improved on it; incremental is 1.0 by
    // definition, not by measurement.
    no_rounds_after_adopt: noRoundsAfterAdopt,
    rounds_committed: roundsCommitted,
    incremental_improved: !!(warm_start.adopted && Number.isFinite(incrementalSpeedup) && incrementalSpeedup > 1 + MIN_IMPROVE),
    returned_after_read_kb: !!warm_start.returned_after_read_kb,
    candidates: warm_start.candidates,
  },
  kb_written: kb_written && kb_written.written ? { slug: kb_written.slug, dir: kb_written.dir } : null,
};
