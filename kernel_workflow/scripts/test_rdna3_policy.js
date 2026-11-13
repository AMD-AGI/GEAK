#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// W7900 / gfx1100 policy at the kernel layer, exercised against the REAL kernel_lane.js and
// kernel_workflow.js sources (no GPU, no agent — the Workflow runtime globals are stubbed).
//
//   A) Lane: an exact W7900 identity gets RDNA3 isolation (no CDNA knowledge, expert skills or warm
//      start), the calibrated-w7900 roofline status, and COUNTER_GUIDANCE=gfx1100-safe-set only
//      when profiling is on (args.profile=on) and gpu_perf_level declares profile_standard (else
//      unavailable). R9700 and CDNA prompts carry no COUNTER_GUIDANCE (negative controls).
//   B) Lane refusals: gfx1100 without the W7900 product, and a W7900 product on another ISA, fail
//      closed before or right after Setup.
//   C) Bakeoff: the W7900 identity is forwarded to every lane, Discover gets the RDNA3 isolation
//      intro, and an unknown gfx1100 board is refused before Discover.
//   D) Knowledge contracts: gfx1100 routes to amd_rdna3.md, not the Ryzen APU guide.
//
// Run:  node kernel_workflow/scripts/test_rdna3_policy.js
'use strict';

const fs = require('fs');
const path = require('path');

const wfDir = path.resolve(__dirname, '..');
const geakRoot = path.resolve(wfDir, '..');
const laneBody = fs.readFileSync(path.join(wfDir, 'kernel_lane.js'), 'utf8')
  .replace(/^export const meta/m, 'const meta');
const dispatcherBody = fs.readFileSync(path.join(wfDir, 'kernel_workflow.js'), 'utf8')
  .replace(/^export const meta/m, 'const meta');

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail ? '->  ' + detail : ''); failures++; }
  else console.log('  ok:', msg);
};

const W7900_NAME = 'AMD Radeon Pro W7900 Dual Slot';
const W7900 = {
  deviceGfx: 'gfx1100', deviceTarget: 'w7900', deviceName: W7900_NAME, physicalCuCount: 96,
  expectedGfx: 'gfx1100', expectedTarget: 'w7900',
  expectedDeviceName: W7900_NAME, expectedPhysicalCuCount: 96,
};

async function runLane(opts) {
  const o = { budget: 0, planDirections: null, ...opts };
  const prompts = [];
  const logs = [];
  const globals = {
    args: {
      kernel_path: '/tmp/kernel',
      workflow_dir: wfDir,
      budget: o.budget,
      expected_gfx: o.expectedGfx || '',
      expected_target: o.expectedTarget || '',
      expected_device_name: o.expectedDeviceName || '',
      expected_physical_cu_count: o.expectedPhysicalCuCount || 0,
      ...(o.gpuPerfLevel != null ? { gpu_perf_level: o.gpuPerfLevel } : {}),
      ...(o.profile != null ? { profile: o.profile } : {}),
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
          device_name: o.deviceName || 'fixture',
          baseline_frozen: true,
          source_files: ['kernel.py'],
          device_gfx: o.deviceGfx,
          device_target: o.deviceTarget,
          physical_cu_count: o.physicalCuCount || 64,
        };
      }
      if (options.label === 'tech_lead:analyze') {
        return {
          kernel_type: 'triton', modifiable_files: ['kernel.py'],
          kk_operator: '', kk_language: 'triton', kk_refs: [],
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
        return { bottleneck: 'unknown', device: o.deviceGfx, summary_path: '' };
      }
      if (options.label.startsWith('tech_lead:plan')) {
        return { stop: false, directions: o.planDirections || [] };
      }
      return null;
    },
  };
  const fn = new Function(...Object.keys(globals), `return (async () => { ${laneBody} })();`);
  return { result: await fn(...Object.values(globals)), prompts, logs };
}

const promptOf = (run, label) => ((run.prompts.find((p) => p.label.startsWith(label)) || {}).prompt || '');

async function laneError(opts) {
  const prompts = [];
  try {
    const r = await runLane(opts);
    prompts.push(...r.prompts);
    return { msg: '', prompts };
  } catch (error) {
    return { msg: String(error && error.message), prompts };
  }
}

function buildBakeoff(argsObj, identity) {
  const trace = { workflowCalls: [], agentLabels: [], prompts: [] };
  const g = {
    args: { kernel_path: '/tmp/k', workflow_dir: wfDir, mode: 'bakeoff', backends: ['hip'],
      gpu_ids: '0', use_expert_skills: 'true', ...argsObj },
    phase: () => {},
    log: () => {},
    workflow: async (ref, a) => {
      trace.workflowCalls.push({ scriptPath: ref.scriptPath, args: a });
      return { validation_status: 'accepted', final_speedup: 1.0 };
    },
    agent: async (p, o) => {
      const label = (o && o.label) || '';
      trace.agentLabels.push(label);
      trace.prompts.push({ label, prompt: p });
      if (label.startsWith('oracle_freezer')) {
        return {
          op_kind: 'gemm', task_dir: '/tmp/oracle', smoke: 'pass', baseline_frozen: true,
          eval_dir: '/tmp/eval', live_backend: 'hip', candidate_backends: [], ...identity,
        };
      }
      if (label.startsWith('op_benchmarker')) {
        return {
          gate: 'pass', isolated_speedup: 1.0, best_known_ms: 1.0, baseline_ms: 1.0,
          tuned_speedup: 0, author_plan: [],
        };
      }
      if (label.startsWith('bakeoff:report')) return { report_path: '/tmp/eval/bakeoff_report.md' };
      return null;
    },
    parallel: async (thunks) => Promise.all(thunks.map((t) => t().catch(() => null))),
    pipeline: async (items, ...stages) => Promise.all(items.map(async (it, i) => {
      let v = it;
      for (const st of stages) v = await st(v, it, i);
      return v;
    })),
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
  };
  const fn = new Function(...Object.keys(g), `return (async () => { ${dispatcherBody} })();`);
  return {
    run: async () => {
      try { await fn(...Object.values(g)); return ''; } catch (e) { return String(e && e.message); }
    },
    trace,
  };
}

const W7900_FREEZE = {
  device_gfx: 'gfx1100', device_target: 'w7900', device_name: W7900_NAME, physical_cu_count: 96,
};

(async () => {
  console.log('\n# A. lane: exact W7900 identity gets RDNA3 isolation and its own roofline policy');
  {
    const run = await runLane(W7900);
    const analyze = promptOf(run, 'tech_lead:analyze');
    ok(analyze.includes('- GFX: gfx1100'), 'analyze receives GFX gfx1100');
    ok(analyze.includes('- DEVICE_TARGET: w7900'), 'analyze receives DEVICE_TARGET w7900');
    ok(analyze.includes('- PHYSICAL_CU_COUNT: 96'), 'analyze receives 96 CUs');
    ok(analyze.includes('- ROOFLINE_STATUS: calibrated-w7900'),
      'analyze receives the W7900 roofline status');
    ok(!analyze.includes('calibrated-r9700'), 'W7900 never inherits R9700 peaks');
    ok(analyze.includes('- COUNTER_GUIDANCE: unavailable'), 'analyze is told counters are unavailable');
    ok(/- KERNEL_KNOWLEDGE_DIR: *\n/.test(analyze), 'perf_knowledge dir is blanked');
    ok(!analyze.includes('Expert skills (ADVISORY'), 'no CDNA expert-skill block');
    ok(!run.prompts.some((p) => p.label.includes('warm_start')), 'no warm-start calls',
      run.prompts.map((p) => p.label).join(', '));
    ok(!run.prompts.some((p) => p.label.startsWith('profile_engineer')),
      'profiling stays off by default on the W7900');
    const on = await runLane({ ...W7900, profile: 'on' });
    const profile = promptOf(on, 'profile_engineer:baseline');
    ok(profile.includes('- COUNTER_GUIDANCE: unavailable'), 'profile engineer is told counters are unavailable');
    ok(profile.includes('- ROOFLINE_STATUS: calibrated-w7900'), 'profile engineer gets the roofline status');
  }
  {
    const run = await runLane({ ...W7900, budget: 1, planDirections: [{
      id: 'roof', title: 'Use roofline peak bandwidth', prompt: 'Reach 90% of peak',
      specialty: 'memory', focus_files: ['kernel.py'],
    }] });
    ok(run.result && run.result.stopped_by === 'budget',
      'peak-based directions are kept on the W7900', run.result && run.result.stopped_by);
    ok(!run.logs.some((line) => /relied on unknown peaks/.test(line)),
      'no W7900 direction is removed as relying on unknown peaks');
    const plan = promptOf(run, 'tech_lead:plan');
    ok(plan.includes('- COUNTER_GUIDANCE: unavailable'), 'plan_round is told counters are unavailable');
    ok(plan.includes('- LEARNED_KB: off'), 'plan_round has the learned KB off');
  }
  for (const [level, expected] of [
    ['stable_std', 'gfx1100-safe-set'], ['profile_standard', 'gfx1100-safe-set'],
    [' STABLE_STD ', 'gfx1100-safe-set'], ['AMDSMI_DEV_PERF_LEVEL_STABLE_STD', 'gfx1100-safe-set'],
    ['auto', 'unavailable'], ['manual', 'unavailable'], ['AMDSMI_DEV_PERF_LEVEL_AUTO', 'unavailable'],
    ['', 'unavailable'],
  ]) {
    const run = await runLane({ ...W7900, gpuPerfLevel: level, profile: 'on' });
    const counterLines = run.prompts.flatMap((p) => p.prompt.match(/- COUNTER_GUIDANCE: .*/g) || []);
    const line = `- COUNTER_GUIDANCE: ${expected}`;
    ok(promptOf(run, 'tech_lead:analyze').includes(line) && promptOf(run, 'profile_engineer:baseline').includes(line)
      && counterLines.every((l) => l === line),
      `W7900 profile=on gpu_perf_level=${JSON.stringify(level)} gives COUNTER_GUIDANCE ${expected}`,
      [...new Set(counterLines)].join(' | '));
    const off = await runLane({ ...W7900, gpuPerfLevel: level });
    const offLines = off.prompts.flatMap((p) => p.prompt.match(/- COUNTER_GUIDANCE: .*/g) || []);
    ok(offLines.length > 0 && offLines.every((l) => l === '- COUNTER_GUIDANCE: unavailable'),
      `W7900 profile off gpu_perf_level=${JSON.stringify(level)} gives COUNTER_GUIDANCE unavailable`,
      [...new Set(offLines)].join(' | '));
  }
  {
    const r9700 = await runLane({
      deviceGfx: 'gfx1201', deviceTarget: 'r9700', deviceName: 'AMD Radeon AI PRO R9700',
      physicalCuCount: 64, expectedGfx: 'gfx1201', expectedTarget: 'r9700', gpuPerfLevel: 'stable_std',
    });
    ok(!r9700.prompts.some((p) => p.prompt.includes('COUNTER_GUIDANCE')),
      'R9700 prompts carry no COUNTER_GUIDANCE, even with gpu_perf_level');
    ok(promptOf(r9700, 'tech_lead:analyze').includes('- ROOFLINE_STATUS: calibrated-r9700'),
      'R9700 keeps its calibrated status');
    const cdna = await runLane({
      deviceGfx: 'gfx950', deviceTarget: 'unknown', physicalCuCount: 256, gpuPerfLevel: 'profile_standard',
    });
    ok(!cdna.prompts.some((p) => p.prompt.includes('COUNTER_GUIDANCE')),
      'CDNA prompts carry no COUNTER_GUIDANCE, even with gpu_perf_level');
    ok(!cdna.prompts.some((p) => /gpu_perf_level|GPU_PERF_LEVEL/.test(p.prompt)),
      'the perf level itself never reaches a prompt');
    ok(promptOf(cdna, 'tech_lead:analyze').includes('Expert skills (ADVISORY'),
      'CDNA keeps the expert-skill path (isolation is not unconditional)');
  }

  console.log('\n# B. lane refusals');
  for (const [desc, opts, re] of [
    ['expected gfx1100 + unknown', { ...W7900, expectedTarget: 'unknown' },
      /expected_gfx=gfx1100 requires expected_target=w7900/],
    ['expected gfx1100 alone', { ...W7900, expectedTarget: '' },
      /expected_gfx=gfx1100 requires expected_target=w7900/],
    ['expected w7900 on gfx1201', { ...W7900, expectedGfx: 'gfx1201' },
      /expected_target=w7900 requires expected_gfx=gfx1100/],
    ['expected r9700 on gfx1100', { ...W7900, expectedTarget: 'r9700' },
      /expected_target=r9700 requires expected_gfx=gfx1201/],
  ]) {
    const { msg, prompts } = await laneError(opts);
    ok(re.test(msg), `${desc} is refused before Setup`, msg || 'no throw');
    ok(prompts.length === 0, `${desc} spawns no agent`);
  }
  for (const [desc, opts, re] of [
    ['detected gfx1100 + unknown (no pins)',
      { deviceGfx: 'gfx1100', deviceTarget: 'unknown', deviceName: 'AMD Radeon RX 7900 XTX', physicalCuCount: 96 },
      /only the exact Radeon Pro W7900/],
    ['detected gfx1100 + unknown (W7900 pinned)',
      { ...W7900, deviceTarget: 'unknown' }, /only the exact Radeon Pro W7900|product mismatch/],
    ['detected w7900 on gfx1201',
      { deviceGfx: 'gfx1201', deviceTarget: 'w7900', physicalCuCount: 96 }, /w7900 requires gfx1100/],
  ]) {
    const { msg } = await laneError(opts);
    ok(re.test(msg), `${desc} is refused after Setup`, msg || 'no throw');
  }
  ok(laneBody.includes("{ name: 'W7900', isa: /^gfx1100$/, isolate: true, matrixCore: 'WMMA' }"),
    'gfx1100 is an isolated WMMA family in the lane');
  ok(laneBody.includes('REQUIRED_PLATFORMS: GPU_FAMILY.isolate ? [GFX] : []'),
    'lane curation is scoped to the detected ISA (gfx1100 on the W7900)');
  ok(laneBody.includes("CURATION_ISOLATE: GPU_FAMILY.isolate ? 'true' : 'false'"),
    'lane curation isolation covers every isolated family');

  console.log('\n# C. bakeoff: W7900 identity is forwarded and Discover is isolated');
  {
    const { run, trace } = buildBakeoff({ expected_gfx: 'gfx1100', expected_target: 'w7900' }, W7900_FREEZE);
    const msg = await run();
    ok(msg === '', 'W7900 bakeoff runs', msg);
    const lane = (trace.workflowCalls[0] || {}).args || {};
    ok(lane.expected_gfx === 'gfx1100', 'lane expected_gfx gfx1100', JSON.stringify(lane.expected_gfx));
    ok(lane.expected_target === 'w7900', 'lane expected_target w7900', JSON.stringify(lane.expected_target));
    ok(lane.expected_physical_cu_count === 96, 'lane physical CU count 96');
    ok(lane.warm_start === 'off', 'W7900 bakeoff warm_start=off', lane.warm_start);
    ok(lane.perf_knowledge_dir === '', 'W7900 bakeoff empty perf_knowledge_dir', lane.perf_knowledge_dir);
    ok(lane.use_expert_skills === 'false', 'W7900 bakeoff expert skills off', lane.use_expert_skills);
    ok(!('gpu_perf_level' in lane), 'an undeclared perf level is not forwarded', lane.gpu_perf_level);
    const discover = (trace.prompts.find((p) => p.label.startsWith('op_benchmarker')) || {}).prompt || '';
    ok(discover.includes('W7900 / gfx1100 ISOLATION'), 'Discover gets the W7900 isolation intro');
    ok(discover.includes('On gfx1100, plan lanes on wave32 WMMA with FP16, BF16, INT8 or INT4 inputs.'),
      'Discover is told which WMMA inputs gfx1100 lanes use');
    ok(!discover.includes('R9700 / gfx1201 ISOLATION'), 'Discover does not call the W7900 an R9700');
  }
  {
    const { run, trace } = buildBakeoff(
      { expected_gfx: 'gfx1100', expected_target: 'w7900', gpu_perf_level: 'stable_std' }, W7900_FREEZE);
    const msg = await run();
    ok(msg === '', 'W7900 bakeoff with a declared perf level runs', msg);
    const lanes = trace.workflowCalls.map((c) => c.args || {});
    ok(lanes.length > 0 && lanes.every((a) => a.gpu_perf_level === 'stable_std'),
      'bakeoff forwards gpu_perf_level to every lane', JSON.stringify(lanes.map((a) => a.gpu_perf_level)));
  }
  {
    const { run, trace } = buildBakeoff({}, W7900_FREEZE);
    const msg = await run();
    ok(msg === '', 'unpinned bakeoff on a detected W7900 runs', msg);
    const lane = (trace.workflowCalls[0] || {}).args || {};
    ok(lane.expected_target === 'w7900' && lane.expected_gfx === 'gfx1100',
      'detected W7900 identity is forwarded', JSON.stringify(lane));
    ok(lane.warm_start === 'off' && lane.perf_knowledge_dir === '',
      'detected W7900 is isolated after Freeze');
  }
  {
    const { run, trace } = buildBakeoff({ expected_gfx: 'gfx1201', expected_target: 'r9700' }, {
      device_gfx: 'gfx1201', device_target: 'r9700',
      device_name: 'AMD Radeon AI PRO R9700', physical_cu_count: 64,
    });
    await run();
    const discover = (trace.prompts.find((p) => p.label.startsWith('op_benchmarker')) || {}).prompt || '';
    ok(discover.includes('R9700 / gfx1201 ISOLATION') && !discover.includes('FP8 or MX'),
      'R9700 Discover intro is unchanged');
  }
  for (const [desc, argsObj, identity, re, beforeFreeze] of [
    ['unpinned gfx1100 + unknown', {}, { ...W7900_FREEZE, device_target: 'unknown',
      device_name: 'AMD Radeon Pro W7800' }, /only the exact Radeon Pro W7900/, false],
    ['W7900 pinned, Freeze reports unknown', { expected_gfx: 'gfx1100', expected_target: 'w7900' },
      { ...W7900_FREEZE, device_target: 'unknown' }, /product mismatch: expected w7900/, false],
    ['Freeze reports w7900 on gfx1201', {}, { ...W7900_FREEZE, device_gfx: 'gfx1201' },
      /w7900 requires gfx1100/, false],
    ['expected gfx1100 + unknown', { expected_gfx: 'gfx1100', expected_target: 'unknown' },
      W7900_FREEZE, /expected_gfx=gfx1100 requires expected_target=w7900/, true],
    ['expected w7900 on gfx1201', { expected_gfx: 'gfx1201', expected_target: 'w7900' },
      W7900_FREEZE, /expected_target=w7900 requires expected_gfx=gfx1100/, true],
    ['expected w7800', { expected_gfx: 'gfx1100', expected_target: 'w7800' },
      W7900_FREEZE, /invalid bakeoff identity/, true],
  ]) {
    const { run, trace } = buildBakeoff(argsObj, identity);
    const msg = await run();
    ok(re.test(msg), `${desc} is refused`, msg || 'no throw');
    ok(!trace.agentLabels.some((l) => l.startsWith('op_benchmarker')), `${desc} never reaches Discover`);
    ok(trace.workflowCalls.length === 0, `${desc} launches no lane`);
    if (beforeFreeze) ok(trace.agentLabels.length === 0, `${desc} spawns no agent`);
  }
  ok(dispatcherBody.includes('REQUIRED_PLATFORMS: GPU_FAMILY.isolate ? [LANE_GFX] : []'),
    'bakeoff curation is scoped to the lane ISA (gfx1100 on the W7900)');

  console.log('\n# D. knowledge contracts');
  {
    const rdna3Path = path.join(wfDir, 'knowledge/amd_rdna3.md');
    const rdna3 = fs.existsSync(rdna3Path) ? fs.readFileSync(rdna3Path, 'utf8') : '';
    ok(!!rdna3, 'knowledge/amd_rdna3.md exists');
    for (const needle of ['gfx1100', 'w7900', 'AMD Radeon Pro W7900', 'wave32', '_w32',
                          'calibrated-w7900', 'Matrix BF16 | not published', 'COUNTER_GUIDANCE',
                          'gfx1100-safe-set', 'profile_standard', 'memory.hbm_read_bandwidth',
                          'memory.lds_bank_conflict_percent', 'memory.l2_hit_rate', 'metrix list metrics',
                          '2f61453a', 'FP16, BF16, INT8 and INT4', '64 KiB', 'rocprofv3 --kernel-trace']) {
      ok(rdna3.includes(needle), `amd_rdna3.md mentions ${needle}`);
    }
    for (const other of ['R9700', 'RDNA4', 'CDNA', 'Instinct', 'gfx12', 'MFMA', '160 KiB',
                         'Unsupported on gfx1100']) {
      ok(!rdna3.includes(other), `amd_rdna3.md describes the W7900 without ${other}`);
    }
    ok(!/MFMA is available|use MFMA/i.test(rdna3), 'amd_rdna3.md does not recommend MFMA');
    const ryzen = fs.readFileSync(path.join(wfDir, 'knowledge/amd_ryzen.md'), 'utf8');
    ok(/gfx115x/.test(ryzen.split('\n').slice(0, 6).join('\n')), 'amd_ryzen.md header is narrowed to gfx115x');
    for (const role of ['tech_lead.md', 'profile_engineer.md', 'engineer.md', 'deep_engineer.md',
                        'author_engineer.md', 'director.md', 'oracle_freezer.md']) {
      const text = fs.readFileSync(path.join(wfDir, 'roles', role), 'utf8');
      ok(/gfx1100[^\n]*amd_rdna3\.md|amd_rdna3\.md[^\n]*gfx1100/.test(text), `${role} routes gfx1100 to amd_rdna3.md`);
      ok(!text.split('\n').some((line) => line.includes('gfx11*') && line.includes('amd_ryzen.md')),
        `${role} no longer routes all of gfx11 to amd_ryzen.md`);
    }
    const director = fs.readFileSync(path.join(wfDir, 'roles/director.md'), 'utf8');
    ok(director.includes(W7900_NAME.replace(' Dual Slot', '')), 'kernel director names the W7900 product');
    const e2eDirector = fs.readFileSync(path.join(geakRoot, 'e2e_workflow/roles/director.md'), 'utf8');
    ok(/w7900/.test(e2eDirector), 'e2e director documents the w7900 target');
    const curator = fs.readFileSync(path.join(wfDir, 'roles/update_experience.md'), 'utf8');
    ok(/gfx1100/.test(curator), 'update_experience documents the gfx1100 platform scope');
  }

  console.log(failures === 0
    ? '\nPASS: W7900 / gfx1100 is isolated, product-scoped and fail-closed in the lane and bakeoff.'
    : `\nFAILED: ${failures} assertion(s).`);
  process.exit(failures === 0 ? 0 : 1);
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
