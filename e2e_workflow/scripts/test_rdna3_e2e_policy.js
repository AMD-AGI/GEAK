#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// W7900 / gfx1100 policy at the E2E layer, exercised against the REAL e2e_workflow.js source
// (no GPU, no server, no model — the Workflow runtime globals are stubbed). Modelled on
// test_rdna4_e2e_policy.js:
//
//   A) laneArgs() ISOLATION. A declared W7900 run gives every lane use_learned_kb=false,
//      use_expert_skills=false, perf_knowledge_dir='', warm_start=off and the forwarded identity.
//   B) FAIL-CLOSED identity. gfx1100 without the exact w7900 product, w7900 on another ISA, and a
//      Setup result that disagrees are all refused before any serving or kernel work.
//   C) SERVING BACKEND. There is no validated W7900 SGLang/ATOM image: vLLM is the default and the
//      only accepted backend.
//   D) Effective role inputs: the learned E2E KB is off and the CDNA tuning phase is disabled.
//   E) No FP8 matrix lever: the fused-fp8 prompts name an in-register upcast on the W7900, not an
//      fp8 WMMA GEMM, and the CDNA and R9700 text is unchanged.
//
// Run:  node e2e_workflow/scripts/test_rdna3_e2e_policy.js
'use strict';

const fs = require('fs');
const path = require('path');

const E2E_DIR = path.resolve(__dirname, '..');
const BODY = fs.readFileSync(path.join(E2E_DIR, 'e2e_workflow.js'), 'utf8')
  .replace(/^export const meta/m, 'const meta');
const PROBE_STOP = 'E2E_PROBE_STOP';

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail ? '->  ' + detail : ''); failures++; }
  else console.log('  ok:', msg);
};

function build(argsObj, opts) {
  const o = opts || {};
  const trace = { workflowCalls: [], phases: [], logs: [], agentPrompts: [] };
  const g = {
    args: { workflow_dir: E2E_DIR, ...argsObj },
    phase: (t) => {
      trace.phases.push(t);
      if (o.stopAtFirstPhase) throw new Error(PROBE_STOP);
      if (o.stopAtPhase && trace.phases.length >= o.stopAtPhase) throw new Error(PROBE_STOP);
    },
    log: (m) => trace.logs.push(m),
    workflow: async (ref, a) => {
      trace.workflowCalls.push({ scriptPath: ref.scriptPath, args: a });
      return {
        eval_dir: '/tmp/eval', final_geomean: 1.2, final_patch: '/tmp/p.diff',
        validation_status: 'accepted',
      };
    },
    agent: async (prompt, options) => {
      trace.agentPrompts.push({ label: options && options.label, prompt });
      if (options && options.label === 'director:setup' && o.setupIdentity) {
        return {
          eval_dir: '/tmp/eval', model_name: 'fixture',
          baseline_throughput_tok_s: 100, noise_band_pct: 0.5,
          gfx: o.setupIdentity.gfx, device_target: o.setupIdentity.target,
          device_name: o.setupIdentity.name, physical_cu_count: o.setupIdentity.cu,
        };
      }
      return null;
    },
    parallel: async (thunks) => Promise.all(thunks.map((t) => t())),
    pipeline: async (items) => items,
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
  };
  // opts.body runs a probe copy of the source and resolves to what it returns.
  const fn = new Function(...Object.keys(g), `return (async () => { ${o.body || BODY} })();`);
  return {
    run: async () => {
      try { const r = await fn(...Object.values(g)); return o.body ? r : ''; } catch (e) { return String(e && e.message); }
    },
    trace,
  };
}

const W7900 = {
  expected_gfx: 'gfx1100',
  expected_target: 'w7900',
  expected_device_name: 'AMD Radeon Pro W7900 Dual Slot',
  expected_physical_cu_count: 96,
};
const W7900_SETUP = { gfx: 'gfx1100', target: 'w7900', name: 'AMD Radeon Pro W7900 Dual Slot', cu: 96 };

(async () => {
  console.log('\n# A. laneArgs() isolation on a declared W7900 run');
  {
    const { run, trace } = build({ kernel_path: '/tmp/k', ...W7900 });
    const msg = await run();
    ok(msg === '', 'the pass-through completes', msg);
    const call = trace.workflowCalls[0];
    ok(!!call && String(call.scriptPath).endsWith('kernel_lane.js'), 'delegates to kernel_lane.js',
      call && call.scriptPath);
    const a = (call && call.args) || {};
    ok(a.use_learned_kb === 'false', 'use_learned_kb=false', JSON.stringify(a.use_learned_kb));
    ok(a.use_expert_skills === 'false', 'use_expert_skills=false', JSON.stringify(a.use_expert_skills));
    ok(a.perf_knowledge_dir === '', "perf_knowledge_dir=''", JSON.stringify(a.perf_knowledge_dir));
    ok(a.warm_start === 'off', 'warm_start=off', JSON.stringify(a.warm_start));
    ok(a.expected_gfx === 'gfx1100', 'expected_gfx forwarded', JSON.stringify(a.expected_gfx));
    ok(a.expected_target === 'w7900', 'expected_target forwarded', JSON.stringify(a.expected_target));
    ok(a.expected_device_name === 'AMD Radeon Pro W7900 Dual Slot', 'expected_device_name forwarded');
    ok(a.expected_physical_cu_count === 96, 'expected_physical_cu_count forwarded');
    ok(!('gpu_perf_level' in a), 'an undeclared perf level is not forwarded', JSON.stringify(a.gpu_perf_level));
  }
  {
    const { run, trace } = build({ kernel_path: '/tmp/k', ...W7900, gpu_perf_level: ' STABLE_STD ' });
    const msg = await run();
    ok(msg === '', 'the pass-through with a declared perf level completes', msg);
    const a = (trace.workflowCalls[0] && trace.workflowCalls[0].args) || {};
    ok(a.gpu_perf_level === 'stable_std', 'laneArgs forwards gpu_perf_level, lower-cased',
      JSON.stringify(a.gpu_perf_level));
  }

  console.log('\n# B. gfx1100 is fail-closed without the exact W7900 product');
  for (const [desc, argsObj, re] of [
    ['pass-through gfx1100 alone', { kernel_path: '/tmp/k', expected_gfx: 'gfx1100' },
      /expected_gfx=gfx1100 requires expected_target=w7900/],
    ['pass-through gfx1100 + unknown', { kernel_path: '/tmp/k', expected_gfx: 'gfx1100', expected_target: 'unknown' },
      /expected_gfx=gfx1100 requires expected_target=w7900/],
    ['serving gfx1100 + unknown', { model_path: '/models/m', expected_gfx: 'gfx1100', expected_target: 'unknown' },
      /expected_gfx=gfx1100 requires expected_target=w7900/],
    ['serving w7900 on gfx1201', { model_path: '/models/m', ...W7900, expected_gfx: 'gfx1201' },
      /expected_target=w7900 requires expected_gfx=gfx1100/],
    ['pass-through w7900 without an ISA', { kernel_path: '/tmp/k', expected_target: 'w7900' },
      /expected_target=w7900 requires expected_gfx=gfx1100/],
    ['serving r9700 on gfx1100', { model_path: '/models/m', ...W7900, expected_target: 'r9700' },
      /expected_target=r9700 requires expected_gfx=gfx1201/],
  ]) {
    const { run, trace } = build(argsObj, { stopAtFirstPhase: true });
    const msg = await run();
    ok(re.test(msg), `${desc} is refused`, msg || 'no throw');
    ok(trace.phases.length === 0 && trace.workflowCalls.length === 0 && trace.agentPrompts.length === 0,
      `${desc} enters no phase, lane or agent`);
  }
  {
    const { run } = build({ model_path: '/models/m', ...W7900 }, {
      stopAtPhase: 2, setupIdentity: { ...W7900_SETUP, target: 'unknown' },
    });
    const msg = await run();
    ok(/GPU product mismatch: expected w7900, detected unknown/.test(msg),
      'Setup reporting unknown on a pinned W7900 is refused', msg || 'no throw');
  }
  {
    // Without pins, Setup alone must not be able to put an unknown gfx1100 into serving policy.
    // Only reachable through the Setup check: the early identity guard requires a pin for model mode.
    ok(/if \(detectedGfx === 'gfx1100' && detectedTarget !== 'w7900'\)/.test(BODY),
      'Setup has its own gfx1100 product check');
  }

  console.log('\n# C. serving backend policy (no validated W7900 SGLang/ATOM image)');
  for (const backend of ['sglang', 'atom']) {
    const { run } = build({ model_path: '/models/m', backend, ...W7900 }, { stopAtFirstPhase: true });
    const msg = await run();
    ok(new RegExp(`W7900 E2E is vLLM-only \\(got backend=${backend}\\)`).test(msg),
      `explicit backend=${backend} fails closed`, msg || 'no throw');
  }
  {
    const { run } = build({ model_path: '/models/m', ...W7900 }, { stopAtFirstPhase: true });
    const msg = await run();
    ok(msg === PROBE_STOP, 'W7900 default backend is vllm, not sglang', msg || 'no throw');
  }
  {
    const { run } = build({ model_path: '/models/m', backend: 'vllm', ...W7900 }, { stopAtFirstPhase: true });
    const msg = await run();
    ok(msg === PROBE_STOP, 'explicit backend=vllm is accepted', msg || 'no throw');
  }

  console.log('\n# D. effective role inputs on a W7900 run');
  {
    const { run, trace } = build({ model_path: '/models/m', ...W7900 },
      { stopAtPhase: 2, setupIdentity: W7900_SETUP });
    const msg = await run();
    ok(msg === PROBE_STOP, 'W7900 setup reached the next phase', msg || 'no throw');
    const directorPrompt = (trace.agentPrompts.find((p) => p.label === 'director:setup') || {}).prompt || '';
    ok(/E2E_LEARNED_KB: off/.test(directorPrompt), 'the E2E learned KB is off', directorPrompt.slice(-400));
    ok(/EXPECTED_TARGET: w7900/.test(directorPrompt), 'the Director is told the expected product');
    ok(/EXPECTED_GFX: gfx1100/.test(directorPrompt), 'the Director is told the expected ISA');
  }
  {
    ok(BODY.includes("{ name: 'W7900', isa: /^gfx1100$/, isolate: true, matrixCore: 'WMMA' }"),
      'gfx1100 is an isolated WMMA family (GPU_FAMILIES)');
    ok(BODY.includes('W7900: { vllmOnly: true, noFp8Matrix: true }'),
      'the W7900 E2E family is vLLM-only and has no fp8 matrix op');
    ok(/const TUNING_SKILLSET_ENABLED = !GPU_FAMILY\.isolate/.test(BODY), 'the CDNA tuning skillset is off on isolated families');
    ok(/const TUNING_KB_ENABLED = !GPU_FAMILY\.isolate/.test(BODY), 'the CDNA tuning KB is off on isolated families');
    ok(/const USE_EXPERT_SKILLS = !GPU_FAMILY\.isolate/.test(BODY), 'expert skills are off on isolated families');
    ok(/if \(GPU_FAMILY\.isolate\) KERNEL_KNOWLEDGE_DIR = '';/.test(BODY), 'kernel knowledge dir is blanked on isolated families');
    ok(/MATRIX_CORE_NAME === 'WMMA'\s*\n\s*\? ' DIRECTION=wmma-layout/.test(BODY), 'deep steers use the WMMA layout direction');
    const meta = BODY.slice(0, BODY.indexOf('};'));
    ok(/Radeon Pro W7900 \(gfx1100, vLLM only/.test(meta), 'meta names the W7900 and its vLLM-only constraint');
    ok(/gfx1100 without the exact W7900 product is refused/.test(meta), 'meta states the gfx1100 refusal');
  }

  console.log('\n# E. no FP8 matrix lever on the W7900');
  {
    // gfx1100 WMMA has no fp8 input type. GRAPH_REQ is built before Setup, so a probe copy of the
    // source returns it with FP8_MMA, the words every fused-fp8 prompt uses for the matrix op.
    const probe = BODY.replace(/(\nconst GRAPH_REQ = [\s\S]*?\n\) : '';\n)/, '$1return { GRAPH_REQ, FP8_MMA };\n');
    ok(probe !== BODY, 'GRAPH_REQ is found');
    for (const [identity, words] of [[W7900, 'fp8-weight (upcast in registers)'],
      [{ expected_gfx: 'gfx1201', expected_target: 'r9700' }, 'fp8 WMMA'],
      [{ expected_gfx: 'gfx950', expected_target: 'unknown' }, 'fp8 MFMA']]) {
      const r = await build({ model_path: '/models/m', backend: 'vllm', ...identity }, { body: probe }).run();
      ok(r && r.FP8_MMA === words && r.GRAPH_REQ.includes(`fused quant (one ${words}, kill the dequant)`),
        `${identity.expected_target} fused-fp8 prompts say "${words}"`, (r && r.FP8_MMA) || r);
      if (identity === W7900) ok(r && !/fp8 WMMA/.test(r.GRAPH_REQ), 'W7900 GRAPH_REQ offers no fp8 WMMA GEMM');
    }
    ok(/one \$\{FP8_MMA\} ` \+/.test(BODY) && /run ONE \$\{FP8_MMA\} GEMM/.test(BODY),
      'both fused-fp8 author tasks name the matrix op through FP8_MMA');
  }

  console.log(failures === 0
    ? '\nPASS: E2E isolates W7900 lanes, refuses other gfx1100 boards, and is vLLM-only on the W7900.'
    : `\nFAILED: ${failures} assertion(s).`);
  process.exit(failures === 0 ? 0 : 1);
})();
