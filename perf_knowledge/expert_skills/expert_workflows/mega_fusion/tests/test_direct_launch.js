#!/usr/bin/env node
// workflow.js is launched directly (no dispatcher fills in its args), so its defaults must resolve:
// mode defaults to mega, MEGA_DIR defaults to this package from workflow_dir alone, and the first
// agent prompt points at files that exist.
'use strict';

const fs = require('fs');
const path = require('path');

const PKG = path.resolve(__dirname, '..');
const ROOT = path.resolve(PKG, '..', '..', '..', '..');
const WF_DIR = path.join(ROOT, 'kernel_workflow');
let failures = 0;
const ok = (cond, msg, got) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg, got === undefined ? '' : `(got ${got})`); failures++; }
};

const src = fs.readFileSync(path.join(PKG, 'workflow.js'), 'utf8')
  .replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

// Runs workflow.js with every agent answering null; returns the prompts it sent and its error.
async function launch(args) {
  const prompts = [];
  const agent = async (prompt) => { prompts.push(prompt); return null; };
  const parallel = (ts) => Promise.all(ts.map((t) => Promise.resolve().then(t).catch(() => null)));
  const pipeline = async () => [];
  const fn = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', src);
  let error = null;
  try {
    await fn(agent, parallel, pipeline, () => {}, () => {}, async () => null,
      { agent_timeout_ms: 0, ...args }, {});
  } catch (e) { error = String(e && e.message || e); }
  return { prompts, error };
}

(async () => {
  const base = { kernel_path: '/tmp/op_task', workflow_dir: WF_DIR, gpu_ids: '0,1', gpus_per_job: 2 };

  console.log('\n# 1. no mode / expert_workflow_dir: runs mode=mega from this package');
  {
    const { prompts, error } = await launch(base);
    ok(prompts.length > 0, 'reaches the first agent (Setup)', error);
    const first = prompts[0] || '';
    ok(first.includes(`MEGA_DIR=${PKG} `), 'MEGA_DIR defaults to this package', first.slice(0, 200));
    ok(first.includes(`SKILL_DIR=${WF_DIR} `), 'SKILL_DIR is the kernel_workflow dir');
    const role = (first.match(/First Read (\S+\.md)/) || [])[1];
    ok(role === path.join(PKG, 'roles', 'director.md') && fs.existsSync(role), 'director role file exists', role);
  }

  console.log('\n# 2. other modes are refused before any agent runs');
  {
    const { prompts, error } = await launch({ ...base, mode: 'optimize' });
    ok(prompts.length === 0 && /runs mode=mega only/.test(error || ''), 'mode=optimize throws', error);
  }

  console.log('\n# 3. workflow_dir is required');
  {
    const { prompts, error } = await launch({ ...base, workflow_dir: '' });
    ok(prompts.length === 0 && /workflow_dir is required/.test(error || ''), 'missing workflow_dir throws', error);
  }

  if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
  console.log('\nall direct-launch checks passed');
})();
