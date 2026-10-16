
// Role prompt: the agent reads its role file and knowledge itself; the script passes only inputs.
// roleAgentFrom reads the role from any roles/ dir (kernel_workflow uses it for e2e's op_benchmarker,
// referenced in place). expertSkillsBlock is declared by each workflow (its text differs per workflow).
function roleAgentFrom(dir, role, phase, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${dir}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${dir}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role);
}
// This workflow's own roles live under WORKFLOW_DIR/roles/.
function roleAgent(role, phase, intro, inputs) {
  return roleAgentFrom(WORKFLOW_DIR, role, phase, intro, inputs);
}
