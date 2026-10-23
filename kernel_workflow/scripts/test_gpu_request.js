#!/usr/bin/env node
// Multi-GPU job resolution in kernel_lane.js: gpus_per_job / job_gpu_ids -> one gpu_lock.sh spec
// shared by every GPU consumer, with the single-GPU default unchanged.
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const lane = fs.readFileSync(path.join(ROOT, 'kernel_lane.js'), 'utf8');
const dispatcher = fs.readFileSync(path.join(ROOT, 'kernel_workflow.js'), 'utf8');
const e2e = fs.readFileSync(path.join(ROOT, '..', 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
function ok(cond, msg) {
  if (cond) console.log(`  ok: ${msg}`);
  else { console.error(`  FAIL: ${msg}`); failures++; }
}
function equal(actual, expected, msg) {
  ok(JSON.stringify(actual) === JSON.stringify(expected),
    `${msg} (expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)})`);
}

const match = lane.match(/function resolveGpuRequest\(A, gpuList, detected\) \{[\s\S]*?\n\}\n/);
if (!match) {
  console.error('FAIL: resolveGpuRequest not found in kernel_lane.js');
  process.exit(1);
}
const resolveGpuRequest = new Function(`${match[0]}; return resolveGpuRequest;`)();
const resolve = (A, detected) => resolveGpuRequest(A, String(A.gpu_ids != null ? A.gpu_ids : '0')
  .split(',').map(s => s.trim()).filter(Boolean), detected);

console.log('\n# resolution');
equal(resolve({}), { gpusPerJob: 1, fixedIds: [], jobSpec: '', explicit: false }, 'default: one GPU, no group spec');
equal(resolve({ gpu_ids: '0,1' }).jobSpec, '', 'a plain pool is not a multi-GPU job');
equal(resolve({ gpu_ids: '0,1,2,3', gpus_per_job: 2 }).jobSpec, 'pool:2:0,1,2,3', 'any N of the pool');
const fixed = resolve({ gpu_ids: '0,1,2,3', job_gpu_ids: '3,1' });
equal([fixed.gpusPerJob, fixed.jobSpec], [2, 'group:3,1'], 'fixed group infers its size');
equal(resolve({ gpu_ids: '0,1,2,3', op_spec: { resource: { gpus_per_job: 2, job_gpu_ids: '2,3' } } }).jobSpec,
  'group:2,3', 'op_spec.resource is honored');
equal(resolve({ gpu_ids: '0,1,2,3', gpus_per_job: 4, op_spec: { resource: { gpus_per_job: 2 } } }).jobSpec,
  'pool:4:0,1,2,3', 'explicit args win over op_spec.resource');

console.log('\n# Setup-detected requirement (no explicit request)');
const POOL8 = { gpu_ids: '0,1,2,3,4,5,6,7' };
const ev = 'unittest.py:12: torchrun --nproc_per_node=8';
equal(resolve(POOL8, { kind: 'single', gpus_per_job: 1 }).jobSpec, '', 'single stays single-GPU');
equal(resolve(POOL8, { kind: 'fixed', gpus_per_job: 8, evidence: ev }).jobSpec,
  'pool:8:0,1,2,3,4,5,6,7', 'fixed 8 leases 8 of the pool');
equal(resolve(POOL8, { kind: 'fixed', gpus_per_job: 2 }).jobSpec, 'pool:2:0,1,2,3,4,5,6,7', 'fixed 2');
equal(resolve(POOL8, { kind: 'scalable' }).jobSpec, 'pool:8:0,1,2,3,4,5,6,7', 'scalable uses the whole pool');
equal(resolve(POOL8, undefined).jobSpec, '', 'no finding stays single-GPU');
equal(resolve({ ...POOL8, gpus_per_job: 4 }, { kind: 'fixed', gpus_per_job: 8 }).jobSpec,
  'pool:4:0,1,2,3,4,5,6,7', 'an explicit gpus_per_job wins over detection');
equal(resolve({ ...POOL8, op_spec: { resource: { gpus_per_job: 2 } } }, { kind: 'unknown' }).jobSpec,
  'pool:2:0,1,2,3,4,5,6,7', 'op_spec.resource wins and skips an unknown finding');
for (const [label, A, d, re] of [
  ['unknown stops with a request for gpus_per_job', POOL8, { kind: 'unknown', evidence: ev }, /could not tell how many GPUs/],
  ['more than the pool stops and cites the evidence', { gpu_ids: '0,1' }, { kind: 'fixed', gpus_per_job: 8, evidence: ev }, /detected: unittest\.py:12.*exceeds gpu_ids pool size 2/],
]) {
  let msg = '';
  try { resolve(A, d); } catch (e) { msg = e.message; }
  ok(re.test(msg), `${label} (${msg})`);
}

console.log('\n# invalid requests throw');
for (const bad of [
  { gpu_ids: '' },
  { gpu_ids: '0,0' },
  { gpu_ids: '0,00' },
  { gpu_ids: '0,1', gpus_per_job: 3 },
  { gpu_ids: '0,1', gpus_per_job: '2x' },
  { gpu_ids: '0,1', gpus_per_job: 1.5 },
  { gpu_ids: '0,1', gpus_per_job: 0 },
  { gpu_ids: '0,1', job_gpu_ids: '0,2' },
  { gpu_ids: '0,1', job_gpu_ids: '0,1', gpus_per_job: 1 },
]) {
  let threw = false;
  try { resolve(bad); } catch (_e) { threw = true; }
  ok(threw, `rejects ${JSON.stringify(bad)}`);
}

console.log('\n# propagation');
ok(!/GPU_ID: GPU_LIST\[0\]/.test(lane), 'no phase takes GPU_LIST[0] directly');
ok(/GPU_POOL = GPU_JOB \|\|/.test(lane), 'GPU_POOL is the group spec for multi-GPU jobs');
ok(/if \(!GPU_RESOURCE\.explicit && setup\.gpu_requirement\) \{\n  applyGpuResource\(resolveGpuRequest\(A, GPU_LIST, setup\.gpu_requirement\)\)/.test(lane),
  'an implicit request is re-sized from Setup before any GPU phase');
ok(lane.indexOf('applyGpuResource(resolveGpuRequest(A, GPU_LIST, setup.gpu_requirement))') <
   lane.indexOf("phase('Author')"), 'resizing happens before the first GPU phase');
ok(/GPU_MODE === 'pin' && !GPU_JOB/.test(lane), 'pinned placement does not split a group');
ok(/return base \+ expertSkillsBlock\(role\) \+ multiGpuBlock\(role\);/.test(lane), 'role prompts get the multi-GPU block');
ok(/function multiGpuBlock\(role\) \{\n  if \(!GPU_JOB\) return '';/.test(lane), 'single-GPU prompts are unchanged');
ok(/mode=bakeoff does not support gpus_per_job/.test(dispatcher), 'bakeoff rejects multi-GPU jobs');
ok(/gpus_per_job: A\.gpus_per_job, job_gpu_ids: A\.job_gpu_ids/.test(e2e),
  'e2e single-kernel pass-through forwards the multi-GPU args');

if (failures) {
  console.error(`\nFAILED: ${failures}`);
  process.exit(1);
}
console.log('\nPASS: multi-GPU job resolution');
