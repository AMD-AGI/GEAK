#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Exercise the actual workflow bodies with deterministic agent responses, without a GPU or model.
'use strict';
const assert = require('assert');
const path = require('path');
const {execFileSync} = require('child_process');
const lane = require('./hardware_context_test_harness');
const bakeoff = require('./test_mode_dispatch');
const e2e = require('../../e2e_workflow/scripts/test_rdna4_e2e_policy');
const root = path.resolve(__dirname, '../..');
function context(name, cu) {
  return JSON.parse(execFileSync('python3', ['-c',
    'import sys,json; sys.path.insert(0,"scripts"); import hardware_context as h; print(json.dumps(h.build(json.loads(sys.argv[1]))))',
    JSON.stringify({gfx:'gfx942', marketing_name:`AMD Instinct ${name}`, physical_cu_count:cu})], {cwd:root}));
}
function inputs(c) {
  return {expected_gfx:c.observed.gfx, expected_target:'unknown',
    expected_device_name:c.observed.marketing_name, expected_physical_cu_count:c.observed.physical_cu_count,
    hardware_context:c, hardware_context_path:'/tmp/eval/hardware_context.json', expected_sku:c.sku};
}
function assertContext(prompt, c) {
  assert(prompt.includes('"DEVICE_SKU":"' + c.sku + '"'));
  assert(prompt.includes('"physical_cu_count":' + c.observed.physical_cu_count));
  assert(prompt.includes(c.catalog_sha256));
  assert(prompt.includes('TraceLens retains native models/calibration'));
}

(async () => {
  const c = context('MI308X', 80), other = context('MI300X', 304), unknown = context('MI300X', 80);
  for (const hw of [c, unknown]) {
    for (const language of ['triton', 'gluon', 'hip']) {
      const run = await lane.run('gfx942', '', {hardwareContext:hw, deviceName:hw.observed.marketing_name,
        physicalCuCount:80, language, budget:1, planDirections:[{id:'scope', specialty:'deep_explore',
          mode:'deep_explore', hypothesis:'resource-aware grid', out_dir:'/tmp/eval/deep', gpu_id:'0'}]});
      for (const label of ['tech_lead:analyze', 'profile_engineer:baseline']) {
        assertContext(run.prompts.find(p => p.label === label).prompt, hw);
      }
      const workers = run.prompts.filter(p => /^(deep|eng) /.test(p.label));
      assert(workers.length > 0);
      workers.forEach(p => assertContext(p.prompt, hw));
      assert.deepStrictEqual(run.result.hardware_context, hw);
    }
  }
  const freeze = {device_gfx:'gfx942', device_target:'unknown', device_name:c.observed.marketing_name,
    physical_cu_count:80, hardware_context:c, hardware_context_path:'/tmp/eval/hardware_context.json'};
  const b = bakeoff.build({kernel_path:'/tmp/k', workflow_dir:path.join(root,'kernel_workflow'), mode:'bakeoff',
    use_expert_skills:'false', backends:['hip','gluon']}, {
    agent:bakeoff.healthy('triton', [], freeze), lane:{hip:1.2,gluon:1.1}});
  await b.run();
  assert.strictEqual(b.trace.workflowCalls.length, 3);
  for (const call of b.trace.workflowCalls) {
    assert.deepStrictEqual(call.args.hardware_context, c);
    assert.strictEqual(call.args.expected_sku, 'mi308x');
  }
  const pass = e2e.build({kernel_path:'/tmp/k', ...inputs(c)});
  await pass.run();
  assert.deepStrictEqual(pass.trace.workflowCalls[0].args.hardware_context, c);
  const modelArgs = {model_path:'/models/m', ...inputs(c), phases:'setup', warm_start:'off', llm_stats:'false'};
  const setup = e2e.build(modelArgs, {setupIdentity:{gfx:'gfx942',target:'unknown',name:c.observed.marketing_name,
    cu:80,hardware_context:c}});
  const result = await setup.run();
  assert.deepStrictEqual(result.state.hardware_context, c);
  assert.strictEqual(result.state.hardware_context_path, '/tmp/eval/hardware_context.json');
  const resumed = e2e.build({...modelArgs, phases:'final', state:result.state});
  await resumed.run();
  const profiling = setup.trace.agentPrompts.filter(p => /profile/.test(p.label));
  assert(profiling.length > 0, 'the baseline profile role must run');
  profiling.forEach(p => assertContext(p.prompt, c));
  assert(resumed.trace.agentPrompts.length > 0);
  resumed.trace.agentPrompts.forEach(p => assertContext(p.prompt, c));
  await assert.rejects(e2e.build({...modelArgs, state:{hardware_context:other}}).run, /hardware_context/);
  await assert.rejects(lane.run('gfx942','',{hardwareContext:c, deviceName:c.observed.marketing_name,
    physicalCuCount:80, setupContext:other}), /hardware_context/);
  await assert.rejects(lane.run('gfx942','',{hardwareContext:c, deviceName:c.observed.marketing_name,
    physicalCuCount:80, expectedSku:'mi300x'}), /expected_sku/);
  await assert.rejects(lane.run('gfx942','',{hardwareContext:c, deviceName:c.observed.marketing_name,
    physicalCuCount:80, setupSku:'mi300x'}), /device_sku/);
  // Old callers still execute, but no SKU is invented from their ISA.
  const old = await lane.run('gfx942');
  assert.strictEqual(old.result.hardware_context, null);
  console.log('PASS: shared hardware context reaches real kernel/bakeoff/E2E/resume routes and rejects mismatches.');
})().catch(e => {console.error(e); process.exit(1);});
