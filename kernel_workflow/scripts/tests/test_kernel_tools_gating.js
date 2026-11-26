// The kernel-tools prompt block is opt-in: it is empty unless use_expert_skills is on, empty for
// roles outside KERNEL_TOOL_ROLES, and empty when use_kernel_tools=false. Runs the shipped source
// region of kernel_lane.js (from `const KERNEL_TOOLS_REQUESTED` to the end of kernelToolsBlock).
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const src = fs.readFileSync(path.join(__dirname, '..', '..', 'kernel_lane.js'), 'utf8');
const start = src.indexOf('const KERNEL_TOOLS_REQUESTED');
const fn = src.indexOf('function kernelToolsBlock(role)');
const end = src.indexOf('\n}\n', fn) + 3;
assert(start > 0 && fn > start && end > fn, 'kernelToolsBlock region not found');
const region = src.slice(start, end);
function block(useExpert, args, role) {
  const f = new Function('A', 'USE_EXPERT_SKILLS', 'WORKFLOW_DIR', 'EXPERT_SKILLS_DIR', 'role',
                         region + '\nreturn kernelToolsBlock(role);');
  return f(args, useExpert, '/wf', '/x/expert_skills', role);
}
assert.strictEqual(block(false, {}, 'engineer'), '', 'default run must not inject kernel tools');
assert.strictEqual(block(false, { use_kernel_tools: 'true' }, 'engineer'), '');
assert.strictEqual(block(true, { use_kernel_tools: 'false' }, 'engineer'), '');
assert.strictEqual(block(true, {}, 'verify_engineer'), '');
const on = block(true, {}, 'engineer');
assert(on.includes('## Kernel tools') &&
       on.includes('/x/expert_skills/skills/gluon_authoring/scripts/q.py --kind tool'));
assert(!on.includes('CARD.md'), 'the kernel-tools block still points at the removed CARD.md');
// The tool cards live in the Gluon fact store, outside every knowledge/ directory: default roles are
// told to read ${WORKFLOW_DIR}/knowledge/ (e.g. deep_explore), so a card there would reach the default path.
const REPO = path.join(__dirname, '..', '..', '..');
const PACK = path.join(REPO, 'perf_knowledge', 'expert_skills', 'skills', 'gluon_authoring');
assert(fs.existsSync(path.join(PACK, 'facts.jsonl')), 'fact store (kind:tool cards) missing');
assert(fs.existsSync(path.join(PACK, 'scripts', 'q.py')), 'q.py missing');
assert(!fs.existsSync(path.join(REPO, 'kernel_workflow', 'scripts', 'kernel_tools', 'CARD.md')),
       'kernel_tools/CARD.md is superseded by the kind:tool cards');
assert(!PACK.split(path.sep).includes('knowledge'), 'the fact store must not sit under a knowledge/ dir');
function walk(dir, out) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (e.name === '.git' || e.name === 'node_modules') continue;
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out); else out.push(p);
  }
  return out;
}
const inKnowledge = walk(REPO, []).filter((p) => p.split(path.sep).includes('knowledge'));
assert(inKnowledge.length > 0, 'no knowledge/ directory found');
for (const p of inKnowledge) {
  assert(!/kernel_tools\.md$|CARD\.md$|facts\.jsonl$/.test(p), `kernel-tools card under a default knowledge dir: ${p}`);
  if (/\.(md|txt)$/.test(p)) {
    assert(!fs.readFileSync(p, 'utf8').includes('q.py --kind tool'), `kernel-tools card pointer under a knowledge dir: ${p}`);
  }
}
for (const absolute of ['Run every profile_kernel.sh', 'do not recall', 'within ~3%']) {
  assert(!on.includes(absolute), `imperative wording left: ${absolute}`);
}
// The default role prompt is base + '' + '' + multiGpuBlock: neither the kernel-tools block nor any
// GEAK_KERNEL_TOOLS / KERNEL_TOOLS_PROFILE instruction reaches it.
assert(/return base \+ expertSkillsBlock\(role\) \+ kernelToolsBlock\(role\) \+ multiGpuBlock\(role\);/.test(src));
for (const v of ['GEAK_KERNEL_TOOLS', 'KERNEL_TOOLS_PROFILE']) {
  const outside = src.slice(0, start) + src.slice(end);
  assert(!outside.includes(v), `${v} appears in kernel_lane.js outside the gated kernel-tools block`);
}

// gpu_lock.sh: without GEAK_KERNEL_TOOLS / KERNEL_TOOLS_PROFILE it ignores GEAK_GPU_LOCK_DIR and exports
// no GEAK_GPU_LOCK_HELD (the default env); with the flag it does both. No GPU work: the command is echo.
const { spawnSync } = require('child_process');
const os = require('os');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kt_gate_'));
const LOCK = path.join(__dirname, '..', 'gpu_lock.sh');
function lockRun(extra) {
  const env = { ...process.env, GEAK_GPU_REQUIRE_IDLE: '0', KERNEL_ENV_KEEP_ARCH: '1', KERNEL_ENV_SKIP_ENUM_REAP: '1',
    KERNEL_EXTRACT_LOCK_DIR: path.join(tmp, 'default'), GEAK_GPU_LOCK_DIR: path.join(tmp, 'override'), ...extra };
  for (const k of ['GEAK_GPU_BROKER', 'GEAK_GPU_ALLOWED', 'GEAK_GPU_LOCK_HELD', 'GEAK_GPU_LOCKED_SET',
    'GEAK_GPU_LEASE_ACTIVE', 'GEAK_GPU_LEASE_IDS', 'GEAK_KERNEL_TOOLS', 'KERNEL_TOOLS_PROFILE']) {
    if (!(k in extra)) delete env[k];
  }
  const r = spawnSync('bash', [LOCK, '7', 'bash', '-c', 'echo "held=${GEAK_GPU_LOCK_HELD-unset}"'],
    { cwd: tmp, env, encoding: 'utf8', timeout: 60000 });
  assert.strictEqual(r.status, 0, r.stderr);
  return r.stdout.trim();
}
const dflt = lockRun({});
assert.strictEqual(dflt, 'held=unset', `default run exported GEAK_GPU_LOCK_HELD: ${dflt}`);
assert(fs.existsSync(path.join(tmp, 'default', 'gpu_7.lock')) && !fs.existsSync(path.join(tmp, 'override', 'gpu_7.lock')),
  'default run must lock in the default namespace, not GEAK_GPU_LOCK_DIR');
assert.strictEqual(lockRun({ GEAK_KERNEL_TOOLS: '1' }), 'held=7');
assert(fs.existsSync(path.join(tmp, 'override', 'gpu_7.lock')), 'kernel-tools run honours GEAK_GPU_LOCK_DIR');
assert.strictEqual(lockRun({ KERNEL_TOOLS_PROFILE: '1' }), 'held=7');
console.log('PASS: kernel-tools block is gated on use_expert_skills and signal-gated in wording; ' +
  'default prompt and gpu_lock env unchanged');
