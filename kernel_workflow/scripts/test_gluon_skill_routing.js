#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';

const fs = require('fs');
const path = require('path');

const wfDir = path.resolve(__dirname, '..');
const geakRoot = path.resolve(wfDir, '..');
const body = fs.readFileSync(path.join(wfDir, 'kernel_lane.js'), 'utf8')
  .replace(/^export const meta/m, 'const meta');

async function run(deviceGfx, expectedGfx = '', extras = {}) {
  const prompts = [];
  const logs = [];
  const {
    deviceName = '',
    deviceTarget,
    physicalCuCount,
    omitGfx = false,
    omitTarget = false,
    expectedTarget = '',
    expectedDeviceName = '',
    expectedPhysicalCuCount = 0,
    budget = 0,
    planDirections = null,
  } = extras;
  const globals = {
    args: {
      kernel_path: '/tmp/kernel',
      workflow_dir: wfDir,
      budget,
      expected_gfx: expectedGfx,
      expected_target: expectedTarget,
      expected_device_name: expectedDeviceName,
      expected_physical_cu_count: expectedPhysicalCuCount,
      use_expert_skills: 'false',
      ...(extras.language ? {target_language: extras.language} : {}),
      ...(extras.useGluon === undefined ? {} : {use_gluon_skill: extras.useGluon}),
      use_learned_kb: 'true',
    },
    phase: () => {},
    log: (message) => logs.push(String(message)),
    workflow: async () => null,
    parallel: async (items) => Promise.all(items.map((item) => item())),
    pipeline: async (items, ...stages) => Promise.all(items.map(async (item) => {
      let result = item;
      for (const stage of stages) result = await stage(result);
      return result;
    })),
    budget: { total: null, spent: () => 0, remaining: () => budget },
    agent: async (prompt, options) => {
      prompts.push({ label: options.label, prompt });
      if (options.label === 'director:setup') {
        const setup = {
          eval_dir: '/tmp/eval',
          workspace: '/tmp/eval/workspace',
          baseline_dir: '/tmp/eval/baseline',
          kernel_name: 'kernel',
          device_name: deviceName || 'fixture',
          baseline_frozen: true,
          source_files: ['kernel.py'],
        };
        if (!omitGfx) setup.device_gfx = deviceGfx;
        if (!omitTarget) {
          setup.device_target = deviceTarget != null
            ? deviceTarget
            : 'unknown';
        }
        setup.physical_cu_count = physicalCuCount != null
          ? physicalCuCount
          : (deviceGfx === 'gfx1201' ? 64 : 256);
        return setup;
      }
      if (options.label === 'tech_lead:analyze') {
        return {
          kernel_type: 'triton',
          modifiable_files: ['kernel.py'],
          kk_operator: 'dense_gemm',
          kk_language: extras.sourceLanguage || 'triton',
          kk_refs: ['optimization/mfma_scheduling.md'],
        };
      }
      if (options.label === 'benchmark_engineer') {
        return {
          commandment_path: '/tmp/eval/COMMANDMENT.md',
          baseline_per_case: [{ name: 'x', latency_ms: 1 }],
          baseline_geomean_ms: 1,
        };
      }
      if (options.label === 'profile_engineer:baseline') {
        return { bottleneck: 'unknown', device: deviceGfx, summary_path: '', profile_roofline_json: '/tmp/native-evidence.json' };
      }
      if (options.label.startsWith('tech_lead:plan')) {
        return { stop: false, directions: planDirections || [] };
      }
      if (options.label === 'update_experience') {
        return { action: 'skipped', card_path: '', key: '', note: 'test' };
      }
      return null;
    },
  };
  const fn = new Function(
    ...Object.keys(globals),
    `return (async () => { ${body} })();`,
  );
  return { result: await fn(...Object.values(globals)), prompts, logs };
}


const assert = require('assert');
(async () => {
  const marker = '## Triton/Gluon method (automatic language reference)';
  for (const gfx of ['gfx942', 'gfx950', 'gfx1201', 'gfx1151']) {
    for (const language of ['triton', 'gluon', 'hip', 'flydsl']) {
      for (const useGluon of [undefined, 'false']) {
        const {prompts} = await run(gfx, '', {language, useGluon});
        const expected = ['gfx942','gfx950'].includes(gfx) && ['triton','gluon'].includes(language) && useGluon !== 'false';
        for (const label of ['tech_lead:analyze', 'profile_engineer:baseline']) {
          const prompt = prompts.find(x => x.label === label).prompt;
          assert.strictEqual(prompt.includes(marker), expected, `${gfx}/${language}/${useGluon}/${label}`);
          if (expected) {
            assert(prompt.includes('draft means unvalidated'));
            assert(prompt.includes('kernel-roofline'));
            assert(prompt.includes(language === 'triton' ? 'Keep the requested Triton backend' : 'incumbent entry'));
          }
        }
        assert(!prompts.find(x => x.label === 'benchmark_engineer').prompt.includes(marker));
      }
    }
  }
  const incumbent = await run('gfx950', '', {sourceLanguage:'gluon'});
  assert(incumbent.prompts.find(x => x.label === 'profile_engineer:baseline').prompt.includes('incumbent entry'));
  const deep = await run('gfx950', '', {language:'gluon', budget:1, planDirections:[{
    id:'gluon', specialty:'deep_explore', mode:'deep_explore', hypothesis:'explicit pipeline',
    out_dir:'/tmp/eval/deep', gpu_id:'0'
  }]});
  // The exact direction schema is owned by the lane; ensure any dispatched worker carries the method.
  const workers = deep.prompts.filter(x => /^(deep|eng) /.test(x.label));
  assert(workers.length > 0, 'a real Optimize worker must be dispatched');
  assert(workers.every(x => x.prompt.includes(marker)));
  assert(workers.every(x => x.prompt.includes('/tmp/native-evidence.json')), 'native evidence reaches the optimizing worker');
  // Invoke the actual e2e central lane-argument builder with controlled dependencies.
  const e2e = fs.readFileSync(path.join(geakRoot, 'e2e_workflow/e2e_workflow.js'), 'utf8');
  const laneArgsSource = e2e.match(/const laneArgs = \(wfArgs\) => \{[\s\S]*?^\};/m)[0];
  for (const setting of ['auto', 'false']) {
    const deps = {LANE_USE_LEARNED_KB:'false', USE_GLUON_SKILL:setting, A:{}, RDNA4_ISOLATE:false,
      EXPECTED_GFX:'', EXPECTED_TARGET:'', EXPECTED_DEVICE_NAME:'', EXPECTED_PHYSICAL_CU_COUNT:0,
      ablLaneArgs:()=>({})};
    const laneArgs = new Function(...Object.keys(deps), laneArgsSource+'; return laneArgs;')(...Object.values(deps));
    assert.strictEqual(laneArgs({target_language:'gluon'}).use_gluon_skill, setting);
  }
  console.log('PASS: actual lane prompts route Triton/Gluon guidance by target/GPU, preserve draft and opt-out, including Optimize.');
})().catch(e => { console.error(e); process.exit(1); });
