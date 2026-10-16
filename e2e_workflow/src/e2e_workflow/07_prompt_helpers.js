// ---------------------------------------------------------------------------
// Prompt helpers (mirror the single-kernel workflow).
// ---------------------------------------------------------------------------
// Workflow scripts have no filesystem API. The writer agent is therefore the
// one component that can atomically snapshot mutable replay assets while the
// measured stack still exists. The intent remains structured so every phase
// feeds the same schema-v2 contract rather than inventing a private summary.
let lastCommittedCheckpoint = null;
async function persistE2EValidationCheckpoint(relativePath, intent) {
  if (!EVAL_DIR) return null;
  const target = `${EVAL_DIR}/${relativePath}`;
  const payload = {
    schema_version: 2, checkpoint_type: 'e2e_validation', committed: true,
    eval_dir: EVAL_DIR,
    ...(lastCommittedCheckpoint ? { parent_checkpoint: lastCommittedCheckpoint } : {}),
    ...intent,
  };
  const written = await safeAgent(
    `You are an E2E checkpoint writer. Materialize the schema-v2 checkpoint intent below at ` +
    `"${target}". Before writing, inspect the referenced measured artifacts and populate ` +
    `measurement.legs, replay commands, stack digests, and integrity.checkpoint_assets. Copy every ` +
    `script/patch/overlay/table/manifest used to reproduce this measurement beneath the checkpoint's ` +
    `checkpoint_assets/ directory; record each relative snapshot path and SHA-256. Canonicalize JSON ` +
    `(sorted keys, compact separators), calculate checkpoint_sha256 excluding that field, then write via ` +
    `a temporary file, fsync it, os.replace(), and fsync the parent directory. Never infer a missing ` +
    `A/B leg, engagement, correctness, asset, or selected kernel slot. Return the written path and digest.\n\n` +
    '```json\n' + JSON.stringify(payload, null, 2) + '\n```',
    { phase: intent.phase || 'Finalize', label: `persist-e2e-checkpoint:${relativePath}`,
      schema: CHECKPOINT_WRITE_SCHEMA },
    2
  );
  // Candidate-local A/B snapshots are diagnostic evidence, never a replay
  // parent; only a complete accepted stack may advance the digest chain.
  if (
    payload.committed === true && written && written.written === true
    && !relativePath.includes("/candidate_")
  ) {
    lastCommittedCheckpoint = {
      path: relativePath, checkpoint_sha256: written.checkpoint_sha256,
    };
  }
  return written;
}

// Candidate-local incomplete A/B records stay best-effort diagnostics, but an
// accepted config/tuning/overlay/final result must never silently continue
// without the checkpoint that makes it replayable after an interrupted run.
async function requireE2EValidationCheckpoint(relativePath, intent) {
  const written = await persistE2EValidationCheckpoint(relativePath, intent);
  if (!written || written.written !== true || !written.checkpoint_sha256) {
    throw new Error(
      `checkpoint_write_failed: ${relativePath}; accepted result cannot be committed without a replayable checkpoint`
    );
  }
  return written;
}

// Expert-skills prompt injection. PURELY ADDITIVE: returns '' whenever the feature is OFF or the role
// is not a skills consumer, so roleAgent's output is byte-identical to the pre-feature build in those
// cases. When ON, it appends a short advisory pointer that tells the agent to Read the fragment file
// and query the skills index. (Workflow scripts have no fs access; the agent does the reading.)
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role)) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits the current bottleneck/op and whose ` +
    `validation_status is \`validated\`, and treat each as a HIGH-PRIOR candidate to reproduce — advisory ` +
    `only, never overriding your on-box A/B, never reducing a result below the measured baseline.`;
}

