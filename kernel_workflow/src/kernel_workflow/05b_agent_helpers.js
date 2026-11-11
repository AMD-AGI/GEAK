function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also query ${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op, whose \`scope\` ` +
    `is \`kernel\` (\`tuning\` entries belong to the e2e tuning phase and match every operator), and whose ` +
    `validation_status is \`validated\`, and treat each as a HIGH-PRIOR candidate — advisory only, never ` +
    `overriding your isolated A/B vs the oracle, never reducing a result below the measured baseline.`;
}

// GPU semaphore: 1 GPU per lane, EXCLUSIVE. size==1 serializes lanes. (Lifted from e2e_workflow.js.)
