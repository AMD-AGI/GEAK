#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// R9700 / gfx1201 policy at the E2E layer, exercised against the REAL e2e_workflow.js source
// (no GPU, no server, no model — the Workflow runtime globals are stubbed).
//
// Why this file exists: the kernel layer's isolation is worthless if the layer ABOVE it hands the
// lane a CDNA knowledge base anyway. e2e_workflow.js funnels every lane invocation through one
// `laneArgs()` helper, so that helper is the whole contract — and it had never been executed by a
// test, only string-matched. Two invariants:
//
//   A) laneArgs() ISOLATION. On a declared R9700 run every lane gets use_learned_kb=false,
//      use_expert_skills=false, perf_knowledge_dir='', warm_start=off and the expected identity
//      forwarded. A CDNA run of the same shape keeps all of it (the negative control — without it
//      these assertions would also pass on a build that isolates unconditionally).
//   B) SERVING BACKEND policy. There is no validated R9700 SGLang image, so an explicit
//      backend=sglang or backend=atom must fail closed and the default must be vllm.
//
// The single-kernel pass-through is the entry point used for (A): it reaches `laneArgs()` after only
// pure config, with no agent or server call in between.
//
// Run:  node e2e_workflow/scripts/test_rdna4_e2e_policy.js
'use strict';

const fs = require('fs');
const path = require('path');

const E2E_DIR = path.resolve(__dirname, '..');
const GEAK_ROOT = path.resolve(E2E_DIR, '..');
const BODY = fs.readFileSync(path.join(E2E_DIR, 'e2e_workflow.js'), 'utf8')
  .replace(/^export const meta/m, 'const meta');

// Thrown from the `phase` stub to stop a model_path run the moment config has been resolved.
// Reaching it means every top-level guard above it accepted the args.
const PROBE_STOP = 'E2E_PROBE_STOP';

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail ? '->  ' + detail : ''); failures++; }
  else console.log('  ok:', msg);
};

function build(argsObj, opts) {
  const o = opts || {};
  const trace = { workflowCalls: [], phases: [], logs: [] };
  const g = {
    args: { workflow_dir: E2E_DIR, ...argsObj },
    phase: (t) => {
      trace.phases.push(t);
      if (o.stopAtFirstPhase) throw new Error(PROBE_STOP);
    },
    log: (m) => trace.logs.push(m),
    workflow: async (ref, a) => {
      trace.workflowCalls.push({ scriptPath: ref.scriptPath, args: a });
      return {
        eval_dir: '/tmp/eval', final_geomean: 1.2, final_patch: '/tmp/p.diff',
        validation_status: 'accepted',
      };
    },
    agent: async () => null,
    parallel: async (thunks) => Promise.all(thunks.map((t) => t())),
    pipeline: async (items) => items,
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
  };
  const fn = new Function(...Object.keys(g), `return (async () => { ${BODY} })();`);
  return { run: () => fn(...Object.values(g)), trace };
}

const R9700 = {
  expected_gfx: 'gfx1201',
  expected_target: 'r9700',
  expected_device_name: 'AMD Radeon AI PRO R9700',
  expected_physical_cu_count: 64,
};

(async () => {
  console.log('\n# A. laneArgs() isolation on a declared R9700 run');
  {
    const { run, trace } = build({ kernel_path: '/tmp/k', ...R9700 });
    await run();
    const call = trace.workflowCalls[0];
    ok(!!call, 'the pass-through reached the kernel layer', JSON.stringify(trace.phases));
    const a = (call && call.args) || {};
    ok(String(call && call.scriptPath).endsWith('kernel_lane.js'), 'delegates to kernel_lane.js',
      call && call.scriptPath);
    ok(a.use_learned_kb === 'false', 'use_learned_kb=false', JSON.stringify(a.use_learned_kb));
    ok(a.use_expert_skills === 'false', 'use_expert_skills=false', JSON.stringify(a.use_expert_skills));
    ok(a.perf_knowledge_dir === '', "perf_knowledge_dir=''", JSON.stringify(a.perf_knowledge_dir));
    ok(a.warm_start === 'off', 'warm_start=off', JSON.stringify(a.warm_start));
    ok(a.expected_gfx === 'gfx1201', 'expected_gfx forwarded', JSON.stringify(a.expected_gfx));
    ok(a.expected_target === 'r9700', 'expected_target forwarded', JSON.stringify(a.expected_target));
    ok(a.expected_device_name === 'AMD Radeon AI PRO R9700', 'expected_device_name forwarded');
    ok(a.expected_physical_cu_count === 64, 'expected_physical_cu_count forwarded');
  }

  console.log('\n# B. the same run on CDNA keeps every knowledge input (negative control)');
  {
    const { run, trace } = build({ kernel_path: '/tmp/k' });
    await run();
    const a = (trace.workflowCalls[0] || {}).args || {};
    ok(a.warm_start !== 'off', 'warm_start NOT forced off', JSON.stringify(a.warm_start));
    ok(a.perf_knowledge_dir !== '', 'perf_knowledge_dir NOT blanked', JSON.stringify(a.perf_knowledge_dir));
    ok(a.expected_target === undefined, 'no R9700 identity invented', JSON.stringify(a.expected_target));
  }

  console.log('\n# C. gfx1201 alone isolates ISA knowledge but never manufactures R9700');
  {
    // The lane guard keys on either half; a caller that passes only the ISA must not silently get
    // the CDNA corpus.
    const { run, trace } = build({ kernel_path: '/tmp/k', expected_gfx: 'gfx1201' });
    await run();
    const a = (trace.workflowCalls[0] || {}).args || {};
    ok(a.use_learned_kb === 'false', 'use_learned_kb=false', JSON.stringify(a.use_learned_kb));
    ok(a.perf_knowledge_dir === '', "perf_knowledge_dir=''", JSON.stringify(a.perf_knowledge_dir));
    ok(a.expected_target === undefined, 'expected_target is not synthesized', JSON.stringify(a.expected_target));
  }

  console.log('\n# D. serving backend policy (there is no validated R9700 SGLang image)');
  {
    const { run } = build({ model_path: '/models/m', backend: 'sglang', ...R9700 },
      { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(/R9700 E2E is vLLM-only/.test(msg), 'explicit backend=sglang fails closed', msg || 'no throw');
  }
  {
    // ATOM is an Instinct-only backend; the guard must reject every non-vllm backend, not just sglang.
    const { run } = build({ model_path: '/models/m', backend: 'atom', ...R9700 },
      { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(/R9700 E2E is vLLM-only \(got backend=atom\)/.test(msg), 'explicit backend=atom fails closed',
      msg || 'no throw');
  }
  {
    // No backend passed: the default must be vllm, NOT the global sglang default. If it defaulted to
    // sglang the guard above would fire; reaching the probe stop proves it did not.
    const { run } = build({ model_path: '/models/m', ...R9700 }, { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(msg === PROBE_STOP, 'R9700 default backend is vllm, not sglang', msg || 'no throw');
  }
  {
    // CDNA is unaffected: sglang remains the default and is allowed.
    const { run } = build({
      model_path: '/models/m', expected_gfx: 'gfx950', expected_target: 'unknown',
    }, { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(msg === PROBE_STOP, 'CDNA sglang default still accepted', msg || 'no throw');
  }
  {
    const { run } = build({
      model_path: '/models/m', expected_gfx: 'gfx1201', expected_target: 'unknown',
    }, { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(/product is not confirmed as r9700/.test(msg),
      'unknown gfx1201 product fails before serving policy', msg || 'no throw');
  }
  {
    const { run } = build({ model_path: '/models/m' }, { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(/requires structured GPU identity/.test(msg),
      'model-mode direct entry fails without early identity', msg || 'no throw');
  }

  console.log('\n# E. the advertised target matches what the code enforces');
  {
    const meta = BODY.slice(0, BODY.indexOf('};'));
    ok(/Radeon AI PRO R9700/.test(meta), 'meta names the validated product, not bare gfx1201');
    ok(/vLLM only|vllm only|vLLM-only/i.test(meta), 'meta states the vLLM-only constraint');
    const kernelLane = fs.readFileSync(path.join(GEAK_ROOT, 'kernel_workflow/kernel_lane.js'), 'utf8');
    ok(/expected_gfx/.test(kernelLane), 'the lane worker accepts the identity e2e forwards');
  }

  console.log(failures === 0
    ? '\nPASS: E2E isolates R9700 lanes, leaves CDNA untouched, and fails closed on non-vLLM backends.'
    : `\nFAILED: ${failures} assertion(s).`);
  process.exit(failures === 0 ? 0 : 1);
})();
