// Optional planner delivery of existing profile context. Role/tool prompts and engineer
// profileSummary remain unchanged. A source-at-read hash never qualifies a profile's producer.
const profileEvidenceState = {summary: null, summary_origin: null, latest_attempt: null, paths: []};

async function profileAgent(prompt, options, profileRound) {
  if (!PLANNER_PROFILE_EVIDENCE) return agentT(prompt, options);
  const origin = {role: 'profile_engineer', round: profileRound, phase: options.phase};
  profileEvidenceState.latest_attempt = {...origin, attempt: null, outcome: 'no_end', error_code: null};
  const report = await agentT(prompt, options, outcome => {
    profileEvidenceState.latest_attempt = {...origin, ...outcome};
  });
  if (report) {
    // Retain the returned value, not a promise that its mutable artifact paths preserve old bytes.
    profileEvidenceState.summary = JSON.parse(JSON.stringify(report));
    profileEvidenceState.summary_origin = {...origin,
      attempt: profileEvidenceState.latest_attempt.attempt, currency: 'CURRENT'};
  } else if (profileEvidenceState.summary_origin) {
    profileEvidenceState.summary_origin = {...profileEvidenceState.summary_origin, currency: 'HISTORICAL'};
  }
  const paths = profileRound === 0
    ? [`${EVAL_DIR}/profile_output`, `${EVAL_DIR}/baseline_metrics.json`, `${EVAL_DIR}/profiling_summary.md`]
    : [`${EVAL_DIR}/profile_output_r${profileRound}`, `${EVAL_DIR}/round_${profileRound}_metrics.json`,
      `${EVAL_DIR}/round_${profileRound}_shift_analysis.md`];
  if (typeof profileEvidenceState.summary?.summary_path === 'string')
    paths.push(profileEvidenceState.summary.summary_path);
  // Keep only explicit known paths. Availability is checked afresh before each planner call.
  profileEvidenceState.paths = [...profileEvidenceState.paths.filter(path => !paths.includes(path)),
    ...new Set(paths)].slice(-24);
  return report;
}

async function plannerProfileEvidenceInputs() {
  if (!PLANNER_PROFILE_EVIDENCE) return {};
  const imports = Array.isArray(A.population_lineage?.imports) ? A.population_lineage.imports : [];
  const populationRoot = imports.length && typeof A.population_lineage?.root === 'string'
    ? A.population_lineage.root : null;
  const index = USE_LEARNED_READ ? `${LEARNED_DIR}/INDEX.md` : null;
  const known = [...new Set([...profileEvidenceState.paths, COMMANDMENT,
    ...(populationRoot ? [populationRoot] : []), ...(index ? [index] : [])])];
  let metadata = null;
  try {
    const result = await runProcess({cmd: 'python3', args: ['-I', '-B',
      `${WORKFLOW_DIR}/scripts/read_profile_evidence.py`, CANONICAL,
      JSON.stringify(A.editable_source_paths === undefined ? ['kernel_src'] : A.editable_source_paths),
      JSON.stringify(known)], timeoutMs: 30000, graceMs: 2000, settleMs: 2000});
    if (result.status === 'completed' && result.code === 0 && result.group_cleared
        && !result.strays_killed && !result.stdio_held) metadata = JSON.parse(result.stdout);
  } catch (_) { /* Missing host metadata leaves provenance explicitly unknown. */ }
  const available = path => metadata?.paths?.find(row => row.path === path && row.available);
  const items = (metadata?.paths || []).filter(row => profileEvidenceState.paths.includes(row.path))
    .map(row => ({path: row.path, kind: row.kind, available: row.available,
      artifact_sha256: row.sha256, bytes: row.bytes, produced_by: null,
      evidence_source_sha256: null, applicability: 'UNKNOWN', currency: 'UNKNOWN'}));
  const commands = [];
  if (populationRoot && available(populationRoot)) commands.push({
    argv: ['python3', `${WORKFLOW_DIR}/scripts/query_population.py`, '--root', populationRoot,
      '--checkpoints-only', ...imports.flatMap(id => ['--lineage', id])],
    purpose: 'Read checkpoints from declared imports only. Check source compatibility before use; no new evaluation.'});
  if (index && available(index)) commands.push({
    argv: ['python3', `${WORKFLOW_DIR}/scripts/kb.py`, '--kb-dir', LEARNED_DIR, 'match',
      '--operator', KK_OPERATOR, '--device', GFX, '--kernel-name', KERNEL_NAME, '--max', '3'],
    purpose: 'Read existing learned-card matches under the unchanged learned-KB policy.'});
  return {
    PROFILE_SUMMARY: profileEvidenceState.summary,
    EVIDENCE_STATUS: {
      current_source_sha256: metadata?.current_source_sha256 || null,
      current_source_paths: metadata?.source_paths || [],
      current_source_status: metadata?.source_status || 'unknown',
      current_source_reason: metadata?.source_reason || (metadata ? null : 'metadata_unavailable'),
      latest_attempt: profileEvidenceState.latest_attempt,
      retained_profile_return: profileEvidenceState.summary_origin,
      items,
      authority: 'Current source hash covers only the declared editable closure at this read. ' +
        'No host-verified profile-producing-source receipt exists on this route, so applicability is UNKNOWN. ' +
        'CURRENT/HISTORICAL on the retained return describe its attempt origin only. Existing paths are mutable; ' +
        'their current bytes are not retained historical artifacts or proof of executed source.'
    },
    EVIDENCE_HANDOFF: {
      paths: (metadata?.paths || []).filter(row => row.available).map(row => ({path: row.path, kind: row.kind})),
      commandment: available(COMMANDMENT) ? {path: COMMANDMENT,
        instruction: 'Read the existing PROFILE command here. Run it only in roles holding GPU_ID.',
        last_recorded_duration_s: null} : null,
      commands,
      scope: 'Existing paths and commands only. No GPU grant, new profile, replacement bundle or query subsystem.'
    }
  };
}
