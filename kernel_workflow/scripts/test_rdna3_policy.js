#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';

// Policy test for RDNA 3.5 APUs (gfx1151 / Strix Halo). Mirrors the RDNA4
// harness: kernel_lane.js is read as TEXT, its ESM export is stripped, and the
// body runs under stubbed Workflow runtime globals. No GPU, no npm install.

const fs = require('fs');
const path = require('path');

const wfDir = path.resolve(__dirname, '..');
const body = fs.readFileSync(path.join(wfDir, 'kernel_lane.js'), 'utf8')
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
    `return (async () => { ${body} })();`,
  );
  return { result: await fn(...Object.values(globals)), prompts, logs };
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

(async () => {
  // --- A. gfx1151 reaches policy as itself, isolated from CDNA inputs. -------
  const halo = await run('gfx1151', 'gfx1151', { physicalCuCount: 40 });
  const haloAnalyze = analyzeOf(halo.prompts);
  mustInclude(haloAnalyze, '- GFX: gfx1151', 'gfx1151 did not reach the analysis prompt');
  mustInclude(haloAnalyze, '- PHYSICAL_CU_COUNT: 40');
  mustExclude(haloAnalyze, 'Expert skills (ADVISORY',
    'gfx1151 received CDNA-biased external expert skills');
  if (halo.prompts.some((entry) => entry.label.includes('warm_start'))) {
    throw new Error('gfx1151 attempted warm-start calls');
  }

  // --- B. The tabulated-peaks status must NOT look "unknown". ---------------
  // perf_knowledge .../rdna35_gfx1151/peaks.md tabulates this ISA, and L1456
  // deletes every roofline direction when the status starts with "unknown".
  // Getting this wrong silently empties the plan on a fully-tabulated part.
  mustInclude(haloAnalyze, 'tabulated-isa-gfx1151',
    'the 40-CU part did not get the tabulated-peaks status');
  mustExclude(haloAnalyze, 'unknown-gfx1151',
    'the tabulated part was marked unknown and will lose roofline directions');

  // --- C. A different SKU on the same ISA must fall back to hard unknown. ---
  // gfx1151 is shared by the 8060S (40 CU), 8050S and 8040S, so CU count is the
  // only on-box discriminator and the tabulated peaks are not transferable.
  const smallSku = await run('gfx1151', 'gfx1151', { physicalCuCount: 32 });
  const smallAnalyze = analyzeOf(smallSku.prompts);
  mustInclude(smallAnalyze, 'unknown-gfx1151-cu-mismatch',
    'a non-tabulated gfx1151 SKU was handed the 8060S peaks');
  mustExclude(smallAnalyze, 'tabulated-isa-gfx1151');

  const filtered = await run('gfx1151', 'gfx1151', {
    physicalCuCount: 32,
    budget: 1,
    planDirections: [{
      id: 'roof', title: 'Use roofline peak bandwidth', prompt: 'Reach 90% of peak',
      specialty: 'memory', focus_files: ['kernel.py'],
    }],
  });
  if (filtered.result.stopped_by !== 'all_directions_filtered_unknown_peaks') {
    throw new Error(`unknown-peak SKU did not filter roofline directions: ${filtered.result.stopped_by}`);
  }
  // ... and the same plan on the tabulated SKU must survive, which is what
  // proves the status string is load-bearing in both directions.
  const kept = await run('gfx1151', 'gfx1151', {
    physicalCuCount: 40,
    budget: 1,
    planDirections: [{
      id: 'roof', title: 'Use roofline peak bandwidth', prompt: 'Reach 90% of peak',
      specialty: 'memory', focus_files: ['kernel.py'],
    }],
  });
  if (kept.result.stopped_by === 'all_directions_filtered_unknown_peaks') {
    throw new Error('the tabulated 40-CU part lost its roofline direction');
  }

  // --- D. Reverse controls: the predicate is gfx115x, not "any gfx11". ------
  // gfx1100 is discrete RDNA3 (Navi 31), not an APU and not tabulated. If the
  // regex is ever widened to gfx11\d\d this assertion is what catches it.
  const navi31 = await run('gfx1100', 'gfx1100', { physicalCuCount: 96 });
  const navi31Analyze = analyzeOf(navi31.prompts);
  // Note: gfx11* routes to knowledge/amd_ryzen.md, which NAMES gfx1151, so the
  // assertion is on the policy status, not on the substring.
  mustExclude(navi31Analyze, 'tabulated-isa-gfx1151',
    'discrete RDNA3 was handed the Strix Halo peaks');
  mustExclude(navi31Analyze, 'unknown-gfx1151-cu-mismatch');
  mustInclude(navi31Analyze, '- GFX: gfx1100');

  const cdna5 = await run('gfx1250', '', { physicalCuCount: 256 });
  const cdna5Analyze = analyzeOf(cdna5.prompts);
  mustExclude(cdna5Analyze, 'tabulated-isa-gfx1151');
  mustInclude(cdna5Analyze, 'Expert skills (ADVISORY',
    'gfx1250 lost the CDNA expert-skill path');

  // --- E. Identity gates still apply on this arch. --------------------------
  await run('gfx1151', 'gfx1201').then(
    () => { throw new Error('expected/detected architecture mismatch was not rejected'); },
    (error) => { if (!/architecture mismatch/.test(String(error.message))) throw error; },
  );
  await run('gfx1151', 'gfx1151', { physicalCuCount: 0 }).then(
    () => { throw new Error('non-positive physical CU count was not rejected'); },
    (error) => { if (!/positive physical_cu_count/.test(String(error.message))) throw error; },
  );

  // --- F. Source invariants. ------------------------------------------------
  // RDNA4_ISOLATE keeps its exact meaning; the union is the policy input. Two
  // RDNA4 tests match RDNA4_ISOLATE as source text, so this must stay a union
  // rather than a rename.
  const lane = fs.readFileSync(path.join(wfDir, 'kernel_lane.js'), 'utf8');
  if (!/const CLIENT_RDNA_ISOLATE = RDNA4_ISOLATE \|\| RDNA35_ISOLATE/.test(lane)) {
    throw new Error('kernel_lane.js lost the client-RDNA union');
  }
  mustInclude(lane, "const RDNA4_ISOLATE = GFX === 'gfx1201';",
    'RDNA4_ISOLATE no longer means exactly gfx1201');
  mustInclude(lane, 'REQUIRED_PLATFORMS: CLIENT_RDNA_ISOLATE ? [GFX] : []',
    'curation still pins a hard-coded platform');

  const dispatcher = fs.readFileSync(path.join(wfDir, 'kernel_workflow.js'), 'utf8');
  mustInclude(dispatcher, 'CLIENT_RDNA_ISOLATE');
  mustInclude(dispatcher, 'Strix Halo / gfx1151 ISOLATION');
  mustInclude(dispatcher, 'R9700 / gfx1201 ISOLATION',
    'the RDNA4 isolation label was replaced instead of joined');
  // Interpolating the label turned a quoted fragment into a template literal. Mismatching the
  // closing delimiter does not throw here — it silently swallows the following concatenated
  // fragments into one string, so the prompt loses instructions with no error anywhere.
  try {
    // eslint-disable-next-line no-new-func
    new Function(`return (async () => { ${dispatcher.replace(/^export const meta/m, 'const meta')} })`);
  } catch (error) {
    throw new Error(`kernel_workflow.js no longer parses: ${error.message}`);
  }

  console.log('PASS: gfx1151 is isolated, SKU-gated on peaks, and the gfx115x predicate is pinned.');
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
