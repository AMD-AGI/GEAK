#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';

// Policy test for the RDNA 3.5 APUs (gfx115x; Strix Halo is gfx1151). Mirrors the RDNA4
// harness: kernel_lane.js and kernel_workflow.js are read as TEXT, their ESM export is
// stripped, and the body runs under stubbed Workflow runtime globals. No GPU, no npm install.
//
// Every isolation assertion has a control that must NOT be isolated (gfx950, gfx1250, discrete
// RDNA3 gfx1100, the gfx1103 iGPU), so a build that isolated unconditionally fails here too.

const fs = require('fs');
const path = require('path');

const wfDir = path.resolve(__dirname, '..');
const laneBody = fs.readFileSync(path.join(wfDir, 'kernel_lane.js'), 'utf8')
  .replace(/^export const meta/m, 'const meta');
const dispatcherBody = fs.readFileSync(path.join(wfDir, 'kernel_workflow.js'), 'utf8')
  .replace(/^export const meta/m, 'const meta');

async function run(deviceGfx, expectedGfx = '', extras = {}) {
  const prompts = [];
  const logs = [];
  const {
    deviceName = 'fixture',
    deviceTarget = 'unknown',
    physicalCuCount = 40,
    budget = 0,
    planDirections = null,
  } = extras;
  const globals = {
    args: {
      kernel_path: '/tmp/kernel',
      workflow_dir: wfDir,
      budget,
      expected_gfx: expectedGfx,
      expected_target: '',
      expected_device_name: '',
      expected_physical_cu_count: 0,
      use_expert_skills: 'true',
      use_learned_kb: 'true',
    },
    phase: () => {},
    log: (message) => logs.push(String(message)),
    workflow: async () => null,
    parallel: async (items) => Promise.all(items.map((item) => item())),
    pipeline: async () => [],
    budget: { total: null, spent: () => 0, remaining: () => 0 },
    agent: async (prompt, options) => {
      prompts.push({ label: options.label, prompt });
      if (options.label === 'director:setup') {
        return {
          eval_dir: '/tmp/eval',
          workspace: '/tmp/eval/workspace',
          baseline_dir: '/tmp/eval/baseline',
          kernel_name: 'kernel',
          device_name: deviceName,
          device_gfx: deviceGfx,
          device_target: deviceTarget,
          physical_cu_count: physicalCuCount,
          baseline_frozen: true,
          source_files: ['kernel.py'],
        };
      }
      if (options.label === 'tech_lead:analyze') {
        return {
          kernel_type: 'triton',
          modifiable_files: ['kernel.py'],
          kk_operator: 'dense_gemm',
          kk_language: 'triton',
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
        return { bottleneck: 'unknown', device: deviceGfx, summary_path: '' };
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
    `return (async () => { ${laneBody} })();`,
  );
  return { result: await fn(...Object.values(globals)), prompts, logs };
}

// The bake-off dispatcher learns the ISA from the oracle_freezer, then launches one lane per
// backend. Its Discover prompt and the lane args are the observable policy.
async function runDispatcher(detectedGfx, expectedGfx = '', physicalCuCount = 40) {
  const prompts = [];
  const lanes = [];
  const globals = {
    args: {
      kernel_path: '/tmp/kernel',
      workflow_dir: wfDir,
      mode: 'bakeoff',
      backends: 'triton',
      budget: 1,
      expected_gfx: expectedGfx,
      use_expert_skills: 'true',
    },
    phase: () => {},
    log: () => {},
    workflow: async (ref, laneArgs) => { lanes.push(laneArgs); return null; },
    parallel: async (items) => Promise.all(items.map((item) => item())),
    pipeline: async () => [],
    budget: { total: null, spent: () => 0, remaining: () => 0 },
    agent: async (prompt, options) => {
      prompts.push({ label: options.label, prompt });
      if (options.label === 'oracle_freezer') {
        return {
          smoke: 'pass', task_dir: '/tmp/task', baseline_frozen: true, eval_dir: '/tmp/eval',
          op_kind: 'gemm', device_gfx: detectedGfx, device_target: 'unknown', device_name: '',
          physical_cu_count: physicalCuCount,
        };
      }
      return null;
    },
  };
  const fn = new Function(
    ...Object.keys(globals),
    `return (async () => { ${dispatcherBody} })();`,
  );
  await fn(...Object.values(globals));
  const discover = prompts.find((entry) => entry.label === 'op_benchmarker:discover');
  if (!discover || lanes.length === 0) throw new Error(`${detectedGfx}: bake-off did not reach a lane`);
  return { discover: discover.prompt, lane: lanes[0] };
}

function analyzeOf(prompts) {
  return prompts.find((entry) => entry.label === 'tech_lead:analyze').prompt;
}

function mustInclude(hay, needle, msg) {
  if (!hay.includes(needle)) throw new Error(msg || `missing ${needle}`);
}

function mustExclude(hay, needle, msg) {
  if (hay.includes(needle)) throw new Error(msg || `unexpected ${needle}`);
}

const SKILLS = 'Expert skills (ADVISORY';

(async () => {
  // --- A. Lane: gfx115x is isolated whether declared or only detected. ------
  for (const [label, detected, expected] of [
    ['declared gfx1151', 'gfx1151', 'gfx1151'],
    ['detected gfx1151', 'gfx1151', ''],
    ['detected gfx1150', 'gfx1150', ''],
  ]) {
    const lane = await run(detected, expected);
    const analyze = analyzeOf(lane.prompts);
    mustInclude(analyze, `- GFX: ${detected}`, `${label}: GFX did not reach the analysis prompt`);
    mustExclude(analyze, SKILLS, `${label}: received CDNA-biased external expert skills`);
    if (lane.prompts.some((entry) => entry.label.includes('warm_start'))) {
      throw new Error(`${label}: attempted warm-start calls`);
    }
  }

  // --- B. Lane controls: the predicate is gfx115x, not "any gfx11" or "everything". -----
  for (const [detected, cu] of [['gfx950', 256], ['gfx1250', 256], ['gfx1100', 96], ['gfx1103', 12]]) {
    const lane = await run(detected, '', { physicalCuCount: cu });
    mustInclude(analyzeOf(lane.prompts), SKILLS, `${detected} lost the expert-skill path`);
  }

  // --- C. Identity gates still apply on this arch. --------------------------
  await run('gfx1151', 'gfx1201').then(
    () => { throw new Error('expected/detected architecture mismatch was not rejected'); },
    (error) => { if (!/architecture mismatch/.test(String(error.message))) throw error; },
  );
  await run('gfx1151', 'gfx1151', { physicalCuCount: 0 }).then(
    () => { throw new Error('non-positive physical CU count was not rejected'); },
    (error) => { if (!/positive physical_cu_count/.test(String(error.message))) throw error; },
  );

  // --- D. Bake-off dispatcher: detection alone isolates gfx115x. ------------
  for (const detected of ['gfx1151', 'gfx1150']) {
    const { discover, lane } = await runDispatcher(detected);
    mustInclude(discover, `RDNA 3.5 / ${detected} ISOLATION`, `${detected}: Discover is not isolated`);
    if (lane.perf_knowledge_dir !== '' || lane.warm_start !== 'off' || lane.use_expert_skills !== 'false') {
      throw new Error(`${detected}: bake-off lane kept CDNA inputs: ${JSON.stringify(lane)}`);
    }
  }
  {
    const { discover, lane } = await runDispatcher('gfx1201');
    mustInclude(discover, 'R9700 / gfx1201 ISOLATION', 'the gfx1201 isolation label changed');
    if (lane.perf_knowledge_dir !== '') throw new Error('gfx1201 bake-off lane lost its isolation');
  }
  for (const [detected, cu] of [['gfx950', 256], ['gfx1100', 96]]) {
    const { discover, lane } = await runDispatcher(detected, '', cu);
    mustExclude(discover, 'ISOLATION', `${detected}: Discover was isolated`);
    if (!lane.perf_knowledge_dir || lane.warm_start === 'off' || lane.use_expert_skills !== 'true') {
      throw new Error(`${detected}: bake-off lane lost CDNA inputs: ${JSON.stringify(lane)}`);
    }
  }

  // --- E. Roofline status: gfx1151 is tabulated only at 40 CU (the 8060S row). ----
  // roofline_tools.resolve_peaks(gfx, cu=...) draws the same line; see test_roofline_skill.py.
  for (const [gfx, cu, status] of [
    ['gfx1151', 40, 'tabulated-gfx1151'],
    ['gfx1151', 32, 'unknown-gfx115x-not-tabulated'],
    ['gfx1150', 16, 'unknown-gfx115x-not-tabulated'],
    ['gfx1152', 40, 'unknown-gfx115x-not-tabulated'],
    ['gfx1100', 96, 'non-rdna'],
    ['gfx950', 256, 'non-rdna'],
  ]) {
    const lane = await run(gfx, '', { physicalCuCount: cu });
    mustInclude(analyzeOf(lane.prompts), `- ROOFLINE_STATUS: ${status}`, `${gfx} at ${cu} CU is not ${status}`);
  }
  // The planner drops roofline directions on any unknown-* status, so the status is load-bearing:
  // the same plan is filtered on the 32-CU part and kept on the 40-CU part.
  const roofPlan = [{
    id: 'roof', title: 'Use roofline peak bandwidth', prompt: 'Reach 90% of peak',
    specialty: 'memory', focus_files: ['kernel.py'],
  }];
  const filtered = await run('gfx1151', '', { physicalCuCount: 32, budget: 1, planDirections: roofPlan });
  if (filtered.result.stopped_by !== 'all_directions_filtered_unknown_peaks') {
    throw new Error(`the 32-CU part kept a roofline direction: ${filtered.result.stopped_by}`);
  }
  const kept = await run('gfx1151', '', { physicalCuCount: 40, budget: 1, planDirections: roofPlan });
  if (kept.result.stopped_by === 'all_directions_filtered_unknown_peaks') {
    throw new Error('the tabulated 40-CU part lost its roofline direction');
  }
  // Every role that reads ROOFLINE_STATUS names both gfx115x statuses.
  for (const role of ['profile_engineer.md', 'tech_lead.md']) {
    const text = fs.readFileSync(path.join(wfDir, 'roles', role), 'utf8');
    mustInclude(text, 'tabulated-gfx1151', `${role} does not explain tabulated-gfx1151`);
    mustInclude(text, 'unknown-gfx115x-not-tabulated', `${role} does not explain the gfx115x unknown`);
  }
  mustInclude(fs.readFileSync(path.join(wfDir, 'roles', 'deep_engineer.md'), 'utf8'), 'tabulated-gfx1151',
    'deep_engineer.md does not say when the gfx1151 row applies');

  console.log('PASS: gfx115x is isolated from CDNA inputs in the lane and the bake-off; controls are not;');
  console.log('      gfx1151 roofline peaks apply only at 40 CU.');
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
