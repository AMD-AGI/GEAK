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
    `return (async () => { ${body} })();`,
  );
  return { result: await fn(...Object.values(globals)), prompts, logs };
}

function mustInclude(hay, needle, msg) {
  if (!hay.includes(needle)) throw new Error(msg || `missing ${needle}`);
}

(async () => {
  const { prompts } = await run('gfx1201', 'gfx1201', {
    deviceTarget: 'r9700', deviceName: 'AMD Radeon AI PRO R9700',
    physicalCuCount: 64, expectedTarget: 'r9700',
    expectedDeviceName: 'AMD Radeon AI PRO R9700', expectedPhysicalCuCount: 64,
  });
  const analyze = prompts.find((entry) => entry.label === 'tech_lead:analyze').prompt;
  if (!analyze.includes('- GFX: gfx1201') || !analyze.includes('- KERNEL_KNOWLEDGE_DIR: ')) {
    throw new Error('gfx1201 analysis did not receive the fail-closed architecture policy');
  }
  mustInclude(analyze, '- DEVICE_TARGET: r9700');
  mustInclude(analyze, '- PHYSICAL_CU_COUNT: 64');
  if (analyze.includes('Expert skills (ADVISORY')) {
    throw new Error('gfx1201 analysis received external expert-skill guidance');
  }
  if (prompts.some((entry) => entry.label.includes('warm_start'))) {
    throw new Error(`gfx1201 attempted warm-start calls: ${prompts.map((entry) => entry.label).join(', ')}`);
  }

  const unknown = await run('gfx1201', 'gfx1201', {
    deviceTarget: 'unknown', deviceName: 'Some other gfx1201 board',
  });
  const unknownAnalyze = unknown.prompts.find((entry) => entry.label === 'tech_lead:analyze').prompt;
  if (!unknownAnalyze.includes('unknown-device-not-r9700')) {
    throw new Error('uncalibrated gfx1201 device did not receive the hard unknown-peak gate');
  }
  if (unknownAnalyze.includes('calibrated-r9700')) {
    throw new Error('gfx1201 architecture alone was promoted to calibrated R9700');
  }

  await run('gfx1201', 'gfx1201', { omitGfx: true }).then(
    () => { throw new Error('missing device_gfx was not rejected'); },
    (error) => {
      if (!/device_gfx/.test(String(error.message))) throw error;
    },
  );
  await run('gfx1201', 'gfx1201', { omitTarget: true }).then(
    () => { throw new Error('missing device_target was not rejected'); },
    (error) => {
      if (!/device_target/.test(String(error.message))) throw error;
    },
  );

  await run('gfx1200').then(
    () => { throw new Error('gfx1200 was not rejected'); },
    (error) => {
      if (!/gfx1200/.test(String(error.message))) throw error;
    },
  );
  await run('gfx950', 'gfx1201').then(
    () => { throw new Error('expected/detected architecture mismatch was not rejected'); },
    (error) => {
      if (!/architecture mismatch/.test(String(error.message))) throw error;
    },
  );
  await run('gfx1201', 'gfx1201', { deviceTarget: 'unknown', expectedTarget: 'r9700' }).then(
    () => { throw new Error('expected r9700 vs unknown product was not rejected'); },
    (error) => {
      if (!/product mismatch/.test(String(error.message))) throw error;
    },
  );
  const identityDrift = await run('gfx1201', 'gfx1201', {
    deviceTarget: 'r9700', physicalCuCount: 32,
    deviceName: '  Radeon AI PRO R9700 (paraphrased)  ',
    expectedTarget: 'r9700',
    expectedDeviceName: 'AMD Radeon AI PRO R9700',
    expectedPhysicalCuCount: 64,
  });
  const driftAnalyze = identityDrift.prompts.find(
    (entry) => entry.label === 'tech_lead:analyze').prompt;
  mustInclude(driftAnalyze, '- PHYSICAL_CU_COUNT: 64',
    'deterministic expected CU count was not used for policy');
  if (identityDrift.logs.filter((line) => /identity telemetry differs/.test(line)).length !== 2) {
    throw new Error(`Director name/CU drift was not logged as telemetry: ${identityDrift.logs.join(' | ')}`);
  }
  await run('gfx1201', 'gfx1201', {
    deviceTarget: 'r9700', physicalCuCount: 0, expectedTarget: 'r9700',
  }).then(
    () => { throw new Error('non-positive physical CU count was not rejected'); },
    (error) => {
      if (!/positive physical_cu_count/.test(String(error.message))) throw error;
    },
  );

  const allFiltered = await run('gfx1201', 'gfx1201', {
    deviceTarget: 'unknown',
    budget: 1,
    planDirections: [{
      id: 'roof', title: 'Use roofline peak bandwidth', prompt: 'Reach 90% of peak',
      specialty: 'memory', focus_files: ['kernel.py'],
    }],
  });
  if (allFiltered.result.stopped_by !== 'all_directions_filtered_unknown_peaks') {
    throw new Error(`all-filtered plan reported ${allFiltered.result.stopped_by}`);
  }
  if (allFiltered.logs.some((line) => /TechLead chose to stop/.test(line))) {
    throw new Error('harness filtering was still attributed to the TechLead');
  }

  const partlyFiltered = await run('gfx1201', 'gfx1201', {
    deviceTarget: 'unknown',
    budget: 1,
    planDirections: [
      {
        id: 'roof', title: 'Use roofline peak bandwidth', prompt: 'Reach peak',
        specialty: 'memory', focus_files: ['kernel.py'],
      },
      {
        id: 'fusion', title: 'Fuse dispatches', prompt: 'Reduce launches',
        specialty: 'host_runtime', focus_files: ['kernel.py'],
      },
    ],
  });
  if (partlyFiltered.result.stopped_by !== 'budget') {
    throw new Error(`partially filtered plan did not keep its valid direction: ${partlyFiltered.result.stopped_by}`);
  }

  const cdna5 = await run('gfx1250');
  const cdna5Analyze = cdna5.prompts.find((entry) => entry.label === 'tech_lead:analyze').prompt;
  if (cdna5Analyze.includes('unknown-device-not-r9700')) {
    throw new Error('gfx1250 CDNA5 was treated as uncalibrated client RDNA');
  }
  if (!cdna5Analyze.includes('Expert skills (ADVISORY')) {
    throw new Error('gfx1250 did not keep the CDNA expert-skill path');
  }

  const director = fs.readFileSync(path.join(wfDir, 'roles/director.md'), 'utf8');
  mustInclude(director, 'device_target');
  mustInclude(director, 'physical_cu_count');
  mustInclude(director, 'AMD Radeon AI PRO R9700');

  const triton = fs.readFileSync(path.join(wfDir, 'knowledge/triton_optimization.md'), 'utf8');
  if (/WMMA \/ RDNA4[\s\S]*attention_prefill_fmha\/backends\/triton\.md/.test(triton)) {
    throw new Error('RDNA4 triton_optimization still routes to the CDNA FMHA Triton card');
  }
  if (/attention_prefill_fmha/.test(triton)) {
    throw new Error('triton_optimization.md still names the CDNA FMHA card on the RDNA4 path');
  }

  const dispatcher = fs.readFileSync(path.join(wfDir, 'kernel_workflow.js'), 'utf8');
  mustInclude(dispatcher, 'expected_gfx: LANE_GFX');
  mustInclude(dispatcher, 'expected_target: LANE_TARGET');
  mustInclude(dispatcher, 'RDNA4_ISOLATE');
  mustInclude(dispatcher, 'R9700 / gfx1201 ISOLATION');

  const e2e = fs.readFileSync(path.join(geakRoot, 'e2e_workflow/e2e_workflow.js'), 'utf8');
  mustInclude(e2e, 'R9700 E2E is vLLM-only');
  mustInclude(e2e, 'R9700_E2E');

  const rdna = fs.readFileSync(path.join(wfDir, 'knowledge/amd_rdna4.md'), 'utf8');
  mustInclude(rdna, 'rocprofv3-avail list --pmc');
  mustInclude(rdna, 'rocprofv3 -L');
  mustInclude(rdna, 'accepts `s_alloc_vgpr`');
  mustInclude(rdna, 'Use `gfx` for ISA facts');
  mustInclude(rdna, 'unvalidated seed');
  if (rdna.includes('rocprofv3 --list-counters')) {
    throw new Error('amd_rdna4.md still documents rocprofv3 --list-counters');
  }

  const profileGuide = fs.readFileSync(path.join(wfDir, 'knowledge/profiling_guide.md'), 'utf8');
  mustInclude(profileGuide, 'Active Threads < 32');
  mustInclude(profileGuide, 'do **not** use the 512 combined-VGPR formula');
  mustInclude(profileGuide, 'rocprofv3-avail list --pmc');
  if (profileGuide.includes('rocprofv3 --list-counters')) {
    throw new Error('profiling_guide.md still documents rocprofv3 --list-counters as the RDNA4 PMC listing command');
  }

  const pitfalls = fs.readFileSync(
    path.join(geakRoot, 'perf_knowledge/hardware/rdna4_gfx1201/pitfalls.md'), 'utf8');
  mustInclude(pitfalls, 'rocprofv3 -L');
  mustInclude(pitfalls, 'CLI rename');
  const platformIssues = fs.readFileSync(
    path.join(geakRoot, 'perf_knowledge/expert_skills/skills/gluon_authoring/references/pitfalls/platform-known-issues.md'),
    'utf8');
  mustInclude(platformIssues, 'rocprofv3 -L');
  mustInclude(platformIssues, 'R9700-only');

  const occupancy = fs.readFileSync(
    path.join(geakRoot, 'perf_knowledge/hardware/rdna4_gfx1201/occupancy.md'), 'utf8');
  mustInclude(occupancy, 'pointer, not a second prose copy');
  if (occupancy.includes('| VGPRs/wave |')) {
    throw new Error('orphan occupancy card still duplicates the workflow table');
  }

  const setupComment = e2e.slice(e2e.indexOf('const SETUP_SCHEMA'), e2e.indexOf('const PROFILE_SCHEMA'));
  if (/OPTIONAL here|required` is unchanged/.test(setupComment)) {
    throw new Error('E2E Setup schema comment still says required identity is optional');
  }
  const e2eDirector = fs.readFileSync(
    path.join(geakRoot, 'e2e_workflow/roles/director.md'), 'utf8');
  if (/gfx target[^\\n]*or \\\\\"\\\\"/.test(e2eDirector)) {
    throw new Error('E2E Director still permits an empty required gfx');
  }
  const apiReference = fs.readFileSync(
    path.join(geakRoot, 'docs/reference/api-reference.md'), 'utf8');
  mustInclude(apiReference, 'expected_gfx: \"gfx950\"');
  mustInclude(apiReference, 'expected_target: \"unknown\"');
  const runE2eDocs = fs.readFileSync(
    path.join(geakRoot, 'interface/run_e2e.md'), 'utf8');
  mustInclude(runE2eDocs, 'GEAK_GPU_IDENTITY_JSON');
  mustInclude(runE2eDocs, 'unavailable_dry_run');

  const curator = fs.readFileSync(path.join(wfDir, 'roles/update_experience.md'), 'utf8');
  mustInclude(curator, 'CURATION_ISOLATE');

  console.log('PASS: R9700 identity is structured; gfx1201 knowledge is fail-closed; bakeoff/E2E/docs contracts hold.');
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
