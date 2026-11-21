#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// One W7900 handoff followed end to end against the REAL sources (no GPU, no agent, no server):
//
//   handoff.json --run_e2e.py --dry-run--> mapped_args --e2e_workflow.js laneArgs()--> lane args
//     --kernel_lane.js--> lane prompts;   kernel_workflow.js (optimize, {...A}) --> lane args
//
//   A) gpu_perf_level and the W7900 identity survive every hop; an E2E run never turns lane
//      profiling on.
//   B) Roofline contract: the lane's ROOFLINE_STATUS=calibrated-w7900 is backed by the
//      `product: w7900` section of peaks.md, which roofline_tools resolves only for that product,
//      with no BF16, FP8 or INT4 ceiling.
//   C) COUNTER_GUIDANCE is gfx1100-safe-set only with profile=on and a declared profile_standard
//      level, and unavailable otherwise. R9700 and CDNA handoffs carry none.
//
// Needs python3 with PyYAML (run_e2e.py imports it).
// Run:  node kernel_workflow/scripts/test_w7900_handoff_contract.js
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const wfDir = path.resolve(__dirname, '..');
const geakRoot = path.resolve(wfDir, '..');
const e2eDir = path.join(geakRoot, 'e2e_workflow');
const strip = (file) => fs.readFileSync(file, 'utf8').replace(/^export const meta/m, 'const meta');
const LANE = strip(path.join(wfDir, 'kernel_lane.js'));
const DISPATCHER = strip(path.join(wfDir, 'kernel_workflow.js'));
const E2E = strip(path.join(e2eDir, 'e2e_workflow.js'));
const ROOFLINE_DIR = path.join(e2eDir, 'knowledge/analysis_skills/roofline');

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail ? '->  ' + detail : ''); failures++; }
  else console.log('  ok:', msg);
};

const BOARDS = {
  w7900: { gfx: 'gfx1100', target: 'w7900', marketing_name: 'AMD Radeon Pro W7900 Dual Slot', physical_cu_count: 96 },
  r9700: { gfx: 'gfx1201', target: 'r9700', marketing_name: 'AMD Radeon AI PRO R9700', physical_cu_count: 64 },
  mi300x: { gfx: 'gfx942', target: 'unknown', marketing_name: 'AMD Instinct MI300X', physical_cu_count: 304 },
};

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'w7900-handoff-'));

// Hop 1: the operator's handoff through run_e2e.py's own CLI, as Hyperloom invokes it.
function mappedArgs(board, extra) {
  const b = BOARDS[board];
  const dir = fs.mkdtempSync(path.join(tmp, `${board}-`));
  const handoff = {
    schema_version: 2, model_path: '/models/fixture', framework: 'vllm', tp: 1, gpu_ids: '0',
    expected_gfx: b.gfx, expected_target: b.target,
    expected_device_name: b.marketing_name, expected_physical_cu_count: b.physical_cu_count,
    workload: { isl: 1024, osl: 128, conc: 8 }, exp_root: path.join(dir, 'geak'), ...extra,
  };
  fs.writeFileSync(path.join(dir, 'handoff.json'), JSON.stringify(handoff));
  const out = execFileSync('python3', [path.join(geakRoot, 'interface/run_e2e.py'),
    path.join(dir, 'handoff.json'), path.join(dir, 'result.json'), '--dry-run'], {
    encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, GEAK_GPU_IDENTITY_JSON: JSON.stringify({ ...b, visible_gpu_agents: 1 }) },
  });
  return JSON.parse(out).mapped_args;
}

// Hop 2: e2e_workflow.js with those args. The single-kernel pass-through is the one lane call that
// needs no server; it builds its lane args with the same laneArgs() as every milestone lane (D).
async function e2eLaneArgs(mapped) {
  const { model_path: _ignored, ...rest } = mapped;
  const calls = [];
  const g = {
    args: { ...rest, kernel_path: '/tmp/kernel', workflow_dir: e2eDir },
    phase: () => {},
    log: () => {},
    workflow: async (ref, a) => {
      calls.push({ scriptPath: ref.scriptPath, args: a });
      return { eval_dir: '/tmp/eval', final_geomean: 1.0, final_patch: '', validation_status: 'accepted' };
    },
    agent: async () => null,
    parallel: async (thunks) => Promise.all(thunks.map((t) => t())),
    pipeline: async (items) => items,
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
  };
  await new Function(...Object.keys(g), `return (async () => { ${E2E} })();`)(...Object.values(g));
  return calls;
}

// Hop 2b: a direct kernel-workflow call, whose optimize path spreads {...A} into the lane.
async function dispatcherLaneArgs(args) {
  const calls = [];
  const g = {
    args: { kernel_path: '/tmp/kernel', workflow_dir: wfDir, ...args },
    phase: () => {},
    log: () => {},
    workflow: async (ref, a) => { calls.push({ scriptPath: ref.scriptPath, args: a }); return { validation_status: 'accepted' }; },
    agent: async () => null,
    parallel: async (thunks) => Promise.all(thunks.map((t) => t())),
    pipeline: async (items) => items,
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
  };
  await new Function(...Object.keys(g), `return (async () => { ${DISPATCHER} })();`)(...Object.values(g));
  return calls;
}

// Hop 3: kernel_lane.js with the lane args; Setup reports the same board.
async function lanePrompts(laneArgs, board) {
  const b = BOARDS[board];
  const prompts = [];
  const g = {
    args: { ...laneArgs, workflow_dir: wfDir, budget: 0 },
    phase: () => {},
    log: () => {},
    workflow: async () => null,
    parallel: async (items) => Promise.all(items.map((item) => item())),
    pipeline: async () => [],
    budget: { total: null, spent: () => 0, remaining: () => 0 },
    agent: async (prompt, options) => {
      prompts.push({ label: options.label, prompt });
      if (options.label === 'director:setup') {
        return {
          eval_dir: '/tmp/eval', workspace: '/tmp/eval/workspace', baseline_dir: '/tmp/eval/baseline',
          kernel_name: 'kernel', device_name: b.marketing_name, baseline_frozen: true,
          source_files: ['kernel.py'], device_gfx: b.gfx, device_target: b.target,
          physical_cu_count: b.physical_cu_count,
        };
      }
      if (options.label === 'tech_lead:analyze') {
        return { kernel_type: 'triton', modifiable_files: ['kernel.py'], kk_operator: '', kk_language: 'triton', kk_refs: [] };
      }
      if (options.label === 'benchmark_engineer') {
        return { commandment_path: '/tmp/eval/COMMANDMENT.md', baseline_per_case: [{ name: 'x', latency_ms: 1 }], baseline_geomean_ms: 1 };
      }
      if (options.label === 'profile_engineer:baseline') return { bottleneck: 'unknown', summary_path: '' };
      return null;
    },
  };
  await new Function(...Object.keys(g), `return (async () => { ${LANE} })();`)(...Object.values(g));
  return prompts;
}

const counterLines = (prompts) => [...new Set(prompts.flatMap((p) => p.prompt.match(/- COUNTER_GUIDANCE: .*/g) || []))];
const promptOf = (prompts, label) => ((prompts.find((p) => p.label.startsWith(label)) || {}).prompt || '');

const RESOLVE_PY = `
import json, os, sys
roofline_dir, gfx, product = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
sys.path.insert(0, roofline_dir)
import roofline_tools as rt
p = rt.resolve_peaks(os.path.join(roofline_dir, "peaks.md"), gfx, product=product)
dtypes = ("fp32", "fp16", "int8", "bf16", "fp8", "int4", "int4_w4a16")
print(json.dumps(None if p is None else {"peaks": p, "ceilings": {d: rt.peak_flops_for(p, d) for d in dtypes}}))
`;
const resolvePeaks = (gfx, product) => JSON.parse(execFileSync('python3',
  ['-c', RESOLVE_PY, ROOFLINE_DIR, gfx, JSON.stringify(product)], { encoding: 'utf8' }));

(async () => {
  console.log('\n# A. the W7900 handoff reaches the lane intact');
  const mapped = mappedArgs('w7900', { gpu_perf_level: 'AMDSMI_DEV_PERF_LEVEL_STABLE_STD' });
  ok(mapped.gpu_perf_level === 'amdsmi_dev_perf_level_stable_std', 'run_e2e maps gpu_perf_level, lower-cased',
    JSON.stringify(mapped.gpu_perf_level));
  ok(mapped.expected_gfx === 'gfx1100' && mapped.expected_target === 'w7900' && mapped.expected_physical_cu_count === 96,
    'run_e2e maps the W7900 identity', JSON.stringify([mapped.expected_gfx, mapped.expected_target, mapped.expected_physical_cu_count]));
  ok(mapped.backend === 'vllm', 'the W7900 handoff selects vLLM', mapped.backend);
  ok(!('profile' in mapped), 'run_e2e sends no lane profile switch');
  const e2eCalls = await e2eLaneArgs(mapped);
  const lane = (e2eCalls[0] || {}).args || {};
  ok(e2eCalls.length === 1 && String(e2eCalls[0].scriptPath).endsWith('kernel_lane.js'),
    'e2e_workflow delegates to kernel_lane.js', JSON.stringify(e2eCalls.map((c) => c.scriptPath)));
  ok(lane.gpu_perf_level === 'amdsmi_dev_perf_level_stable_std', 'laneArgs forwards gpu_perf_level', JSON.stringify(lane.gpu_perf_level));
  ok(lane.expected_gfx === 'gfx1100' && lane.expected_target === 'w7900', 'laneArgs forwards the W7900 identity');
  ok(!('profile' in lane), 'an E2E lane is not asked to profile', JSON.stringify(lane.profile));
  const sites = E2E.split('\n').filter((l) => /(^|[^.\w])workflow\(/.test(l) && !/^\s*\/\//.test(l));
  ok(sites.length === 4 && sites.every((l) => l.includes('laneArgs(')),
    'every kernel-lane workflow() call in e2e_workflow.js goes through laneArgs()', sites.join(' | '));

  console.log('\n# B. roofline contract: calibrated-w7900 is the peaks.md w7900 section');
  const e2ePrompts = await lanePrompts(lane, 'w7900');
  ok(promptOf(e2ePrompts, 'tech_lead:analyze').includes('- ROOFLINE_STATUS: calibrated-w7900'),
    'the E2E lane plans against calibrated-w7900');
  const w7900 = resolvePeaks('gfx1100', 'w7900');
  ok(w7900 && w7900.peaks.product === 'w7900' && w7900.peaks.source === 'table' && w7900.peaks.confidence === 'high',
    'roofline_tools resolves the w7900 table section', JSON.stringify(w7900 && w7900.peaks));
  const c = (w7900 && w7900.ceilings) || {};
  ok(w7900 && w7900.peaks.hbm_bw_bytes_s === 8.64e11 && w7900.peaks.cu === 96,
    'w7900 memory roof 864 GB/s on 96 CUs');
  ok(c.fp16 === 1.23e14 && c.int8 === 1.23e14 && c.fp32 === 6.13e13, 'w7900 FP16, INT8 and FP32 ceilings', JSON.stringify(c));
  ok(c.bf16 === null && c.fp8 === null && c.int4 === null && c.int4_w4a16 === null,
    'no BF16, FP8 or INT4 ceiling on the w7900', JSON.stringify(c));
  ok(resolvePeaks('gfx1100', 'unknown') === null && resolvePeaks('gfx1100', null) === null,
    'gfx1100 without the w7900 product has no roofline');

  console.log('\n# C. COUNTER_GUIDANCE: profile x declared level');
  ok(counterLines(e2ePrompts).join() === '- COUNTER_GUIDANCE: unavailable',
    'an E2E lane gets COUNTER_GUIDANCE unavailable, even at stable_std', counterLines(e2ePrompts).join(' | '));
  ok(!e2ePrompts.some((p) => p.label.startsWith('profile_engineer')), 'an E2E lane runs no profile engineer');
  const autoLane = ((await e2eLaneArgs(mappedArgs('w7900', { gpu_perf_level: 'auto' })))[0] || {}).args || {};
  for (const [desc, args, expected] of [
    ['E2E lane args + profile=on at stable_std', { ...lane, profile: 'on' }, 'gfx1100-safe-set'],
    ['E2E lane args + profile=on at auto', { ...autoLane, profile: 'on' }, 'unavailable'],
    ['E2E lane args + profile=off at stable_std', { ...lane, profile: 'off' }, 'unavailable'],
  ]) {
    const prompts = await lanePrompts(args, 'w7900');
    ok(counterLines(prompts).join() === `- COUNTER_GUIDANCE: ${expected}`, `${desc} gives ${expected}`, counterLines(prompts).join(' | '));
    if (args.profile === 'on') {
      ok(promptOf(prompts, 'profile_engineer:baseline').includes(`- COUNTER_GUIDANCE: ${expected}`),
        `${desc}: the profile engineer gets it too`);
    }
  }
  const direct = await dispatcherLaneArgs({
    expected_gfx: 'gfx1100', expected_target: 'w7900', expected_physical_cu_count: 96,
    gpu_perf_level: 'stable_std', profile: 'on',
  });
  const directLane = (direct[0] || {}).args || {};
  ok(directLane.gpu_perf_level === 'stable_std' && directLane.profile === 'on',
    'kernel_workflow optimize forwards gpu_perf_level and profile', JSON.stringify([directLane.gpu_perf_level, directLane.profile]));
  const directPrompts = await lanePrompts(directLane, 'w7900');
  ok(counterLines(directPrompts).join() === '- COUNTER_GUIDANCE: gfx1100-safe-set',
    'a direct profile=on kernel run at stable_std gets the safe set', counterLines(directPrompts).join(' | '));
  for (const board of ['r9700', 'mi300x']) {
    const m = mappedArgs(board, { gpu_perf_level: 'stable_std' });
    const l = ((await e2eLaneArgs(m))[0] || {}).args || {};
    ok(l.gpu_perf_level === 'stable_std', `${board} lane args carry the declared level`);
    const prompts = await lanePrompts({ ...l, profile: 'on' }, board);
    ok(prompts.length > 0 && !prompts.some((p) => p.prompt.includes('COUNTER_GUIDANCE')),
      `${board} prompts carry no COUNTER_GUIDANCE`);
    ok(!prompts.some((p) => /gpu_perf_level|GPU_PERF_LEVEL/.test(p.prompt)), `${board}: the level itself never reaches a prompt`);
  }

  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(failures === 0
    ? '\nPASS: a W7900 handoff keeps its identity, perf level and roofline from run_e2e.py to the lane prompts.'
    : `\nFAILED: ${failures} assertion(s).`);
  process.exit(failures === 0 ? 0 : 1);
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
