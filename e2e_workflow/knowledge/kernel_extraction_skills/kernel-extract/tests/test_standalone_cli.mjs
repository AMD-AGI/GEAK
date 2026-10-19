// CPU fixtures test evidence transport; they do not claim real GPU execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { sourceSnapshot, fileRef } from '../scripts/kernel_source_delivery.mjs';

const cli = fileURLToPath(new URL('../scripts/kernel_extraction_delivery.mjs', import.meta.url));

async function fixture(t) {
  const task = await fs.mkdtemp(path.join(os.tmpdir(), 'standalone source fixture '));
  t.after(() => fs.rm(task, {recursive: true, force: true}));
  await fs.mkdir(path.join(task, 'kernel_src'));
  await fs.mkdir(path.join(task, 'run'));
  const write = async (name, value) => {
    await fs.writeFile(path.join(task, name), JSON.stringify(value));
    return fileRef(task, name);
  };
  await write('meta.json', {target_callable: 'pkg:leaf'});
  for (const name of ['cases.py', 'unittest.py', 'kernel_src/leaf.cu'])
    await fs.writeFile(path.join(task, name), '// CPU-only source fixture\n');
  const snapshot = await sourceSnapshot(task);
  const kernel = {entity_kind: 'gpu_kernel', device_kernel: 'leaf_kernel',
    selected_device_kernels: ['leaf_kernel'], source_delivery_required: true,
    seam_candidates: [{target_callable: 'pkg:leaf', role: 'inner_launcher', depth: 1,
      device_kernels: ['leaf_kernel']}]};
  const selection = await write('run/selection.json', {
    contract: 'kernel_selection', ok: true, target_callable: 'pkg:leaf',
    device_kernel: 'leaf_kernel', total_calls_observed: 1, target_marker_calls: 1,
    matched_kernel_calls: 1, deepest_verified: true, candidate_targets_tested: ['pkg:leaf'],
    live_candidate_targets: ['pkg:leaf'], deeper_live_candidates: [], failed: [],
  });
  const ut = await write('run/ut.json', {ok: true, exit_code: 0,
    target_callable: 'pkg:leaf', snapshot_sha256: snapshot.snapshot_sha256});
  const leaf = {device_kernel: 'leaf_kernel', source_files: ['kernel_src/leaf.cu']};
  for (const [field,kind] of [['execution_ref','candidate_execution'],
                             ['negative_control_ref','numerical_rejection']]) {
    await fs.writeFile(path.join(task, `run/${kind}.log`), 'CPU TEST FIXTURE, not GPU evidence\n');
    leaf[field] = await write(`run/${kind}.json`, {kind, ok: true,
      device_kernel: 'leaf_kernel', target_callable: 'pkg:leaf',
      snapshot_sha256: snapshot.snapshot_sha256,
      command: 'fixture-only', gpu: 'synthetic', invocations: 1,
      launch_succeeded: true, numerical_mismatches: 1,
      log_ref: await fileRef(task, `run/${kind}.log`)});
  }
  const source = await write('run/source.json', {contract:'kernel_source_delivery/v1',
    target_callable:'pkg:leaf', ...snapshot, ut_result_ref:ut, kernels:[leaf]});
  const result = {editable:true, task_dir:task, target_callable:'pkg:leaf',
    unittest_smoke:'pass', baseline_frozen:true,
    selection_validation_ref:selection, source_delivery_ref:source};
  await write('run/request.json',kernel);
  await write('run/draft.json',result);
  const args=[cli,'--kernel',path.join(task,'run/request.json'),'--result',path.join(task,'run/draft.json'),
    '--phase','extract','--out',path.join(task,'run/RETURN.json')];
  const run=()=>spawnSync(process.execPath,args,{cwd:os.tmpdir(),encoding:'utf8'});
  return {task,kernel,result,write,run};
}

test('standalone CLI accepts consistent local source/selection records without a runner',async t=>{
  const f=await fixture(t), result=f.run();
  assert.equal(result.status,0,result.stdout+result.stderr);
  assert.equal(JSON.parse(result.stdout).ok,true);
  assert.deepEqual(JSON.parse(await fs.readFile(path.join(f.task,'run/RETURN.json'),'utf8')),f.result);
});
test('CLI source declaration and records cannot be omitted',async t=>{
  const f=await fixture(t);
  delete f.kernel.source_delivery_required;
  await f.write('run/request.json',f.kernel);
  const result=f.run();
  assert.notEqual(result.status,0);
  assert.match(result.stderr,/source_delivery_required=true/);
  await f.write('run/request.json',{...f.kernel,source_delivery_required:true});
  delete f.result.source_delivery_ref;
  await f.write('run/draft.json',f.result);
  assert.notEqual(f.run().status,0);
});
test('changed source rejects stale evidence without overwriting accepted return',async t=>{
  const f=await fixture(t);
  assert.equal(f.run().status,0);
  const accepted=await fs.readFile(path.join(f.task,'run/RETURN.json'),'utf8');
  await fs.appendFile(path.join(f.task,'kernel_src/leaf.cu'),'// changed\n');
  const result=f.run();
  assert.equal(result.status,1,result.stdout+result.stderr);
  assert.equal(JSON.parse(result.stdout).checks.source_delivery.ok,false);
  assert.equal(await fs.readFile(path.join(f.task,'run/RETURN.json'),'utf8'),accepted);
});
