#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// RDNA 3.5 APU (gfx1151 / Strix Halo) policy at the E2E layer, exercised against the REAL
// e2e_workflow.js source (no GPU, no server, no model — Workflow runtime globals are stubbed).
//
// Three invariants, each of which has a silent failure mode:
//
//   A) ISOLATION. gfx1151 must reach the kernel lane with the CDNA knowledge inputs off, exactly
//      as gfx1201 does. Without it a Strix Halo run is steered by MI-series priors.
//   B) BACKEND. There is no validated gfx1151 SGLang image (ci/docker_setup/docker_default.json
//      carries no sglang.gfx1151 key), so the default must be vllm and anything else must fail
//      closed. The pre-existing default was the global 'sglang', i.e. a guaranteed late failure.
//   C) fp8 TRUTH. gfx1201 has fp8 WMMA; gfx1151 has no fp8 matrix instruction at all. Renaming
//      MFMA->WMMA is not enough — prompts that say "fold the scale into one fp8 WMMA" would be
//      prescribing hardware that does not exist on this part.
//
// R9700 and CDNA appear throughout as negative controls: every assertion here would also pass on
// a build that isolated unconditionally or that had simply widened the gfx1201 branch.
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
          eval_dir: '/tmp/eval',
          model_name: 'fixture',
          baseline_throughput_tok_s: 100,
          noise_band_pct: 0.5,
          gfx: o.setupIdentity.gfx,
          device_target: o.setupIdentity.target,
          device_name: o.setupIdentity.name,
          physical_cu_count: o.setupIdentity.cu,
        };
      }
      return null;
    },
    parallel: async (thunks) => Promise.all(thunks.map((t) => t())),
    pipeline: async (items) => items,
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
  };
  const fn = new Function(...Object.keys(g), `return (async () => { ${BODY} })();`);
  return { run: () => fn(...Object.values(g)), trace };
}

// gfx1151 has no product key: the ISA ships as the 8060S / 8050S / 8040S behind one generic
// nameplate, so identity carries target 'unknown' and the CU count is the only discriminator.
const HALO = {
  expected_gfx: 'gfx1151',
  expected_target: 'unknown',
  expected_device_name: 'AMD Radeon Graphics',
  expected_physical_cu_count: 40,
};

(async () => {
  console.log('\n# A. laneArgs() isolation on a Strix Halo run');
  {
    const { run, trace } = build({ kernel_path: '/tmp/k', ...HALO });
    await run();
    const call = trace.workflowCalls[0];
    ok(!!call, 'the pass-through reached the kernel layer', JSON.stringify(trace.phases));
    const a = (call && call.args) || {};
    ok(a.use_learned_kb === 'false', 'use_learned_kb=false', JSON.stringify(a.use_learned_kb));
    ok(a.use_expert_skills === 'false', 'use_expert_skills=false', JSON.stringify(a.use_expert_skills));
    ok(a.perf_knowledge_dir === '', "perf_knowledge_dir=''", JSON.stringify(a.perf_knowledge_dir));
    ok(a.warm_start === 'off', 'warm_start=off', JSON.stringify(a.warm_start));
    ok(a.expected_gfx === 'gfx1151', 'expected_gfx forwarded', JSON.stringify(a.expected_gfx));
    ok(a.expected_physical_cu_count === 40, 'expected_physical_cu_count forwarded',
      JSON.stringify(a.expected_physical_cu_count));
  }

  console.log('\n# B. negative controls: CDNA keeps everything, R9700 is unchanged');
  {
    const { run, trace } = build({ kernel_path: '/tmp/k' });
    await run();
    const a = (trace.workflowCalls[0] || {}).args || {};
    ok(a.warm_start !== 'off', 'CDNA warm_start NOT forced off', JSON.stringify(a.warm_start));
    ok(a.perf_knowledge_dir !== '', 'CDNA perf_knowledge_dir NOT blanked',
      JSON.stringify(a.perf_knowledge_dir));
  }
  {
    // Widening isolation must not narrow it: R9700 keeps its own isolation and its product key.
    const { run, trace } = build({
      kernel_path: '/tmp/k', expected_gfx: 'gfx1201', expected_target: 'r9700',
      expected_device_name: 'AMD Radeon AI PRO R9700', expected_physical_cu_count: 64,
    });
    await run();
    const a = (trace.workflowCalls[0] || {}).args || {};
    ok(a.use_learned_kb === 'false', 'R9700 isolation preserved', JSON.stringify(a.use_learned_kb));
    ok(a.expected_target === 'r9700', 'R9700 product key preserved', JSON.stringify(a.expected_target));
  }

  console.log('\n# C. no product identity is synthesized for an APU ISA');
  {
    const { run, trace } = build({ kernel_path: '/tmp/k', expected_gfx: 'gfx1151' });
    await run();
    const a = (trace.workflowCalls[0] || {}).args || {};
    ok(a.use_learned_kb === 'false', 'bare gfx1151 still isolates', JSON.stringify(a.use_learned_kb));
    ok(a.expected_target === undefined, 'no device_target invented',
      JSON.stringify(a.expected_target));
  }

  console.log('\n# D. serving backend policy (there is no validated gfx1151 SGLang image)');
  {
    const { run } = build({ model_path: '/models/m', backend: 'sglang', ...HALO },
      { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(/gfx1151 E2E is vLLM-only \(got backend=sglang\)/.test(msg),
      'explicit backend=sglang fails closed', msg || 'no throw');
  }
  {
    const { run } = build({ model_path: '/models/m', backend: 'atom', ...HALO },
      { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(/gfx1151 E2E is vLLM-only \(got backend=atom\)/.test(msg),
      'explicit backend=atom fails closed', msg || 'no throw');
  }
  {
    // No backend passed. Before this change the global default 'sglang' applied, which is the whole
    // bug: reaching the probe stop proves the default is now vllm.
    const { run } = build({ model_path: '/models/m', ...HALO }, { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(msg === PROBE_STOP, 'gfx1151 default backend is vllm, not sglang', msg || 'no throw');
  }
  {
    // The R9700 message is matched verbatim by test_rdna4_e2e_policy.js; the new branch must not
    // have swallowed it.
    const { run } = build({
      model_path: '/models/m', backend: 'sglang',
      expected_gfx: 'gfx1201', expected_target: 'r9700',
    }, { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(/R9700 E2E is vLLM-only/.test(msg), 'the R9700 message is unchanged', msg || 'no throw');
  }
  {
    const { run } = build({
      model_path: '/models/m', expected_gfx: 'gfx950', expected_target: 'unknown',
    }, { stopAtFirstPhase: true });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(msg === PROBE_STOP, 'CDNA sglang default still accepted', msg || 'no throw');
  }

  console.log('\n# E. effective role inputs');
  {
    const { run, trace } = build(
      { model_path: '/models/m', ...HALO },
      {
        stopAtPhase: 2,
        setupIdentity: {
          gfx: 'gfx1151', target: 'unknown', name: 'AMD Radeon Graphics', cu: 40,
        },
      },
    );
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(msg === PROBE_STOP, 'gfx1151 setup reached the next phase', msg || 'no throw');
    const directorPrompt = (trace.agentPrompts.find((p) => p.label === 'director:setup') || {}).prompt || '';
    ok(/E2E_LEARNED_KB: off/.test(directorPrompt),
      'effective gfx1151 role inputs disable the E2E learned KB', directorPrompt.slice(-500));
  }

  console.log('\n# F. fp8 truth: the matrix-core rename is not sufficient');
  {
    ok(/const HAS_FP8_MATRIX = !RDNA35_ISA/.test(BODY),
      'the fp8 matrix path is a separate fact from the matrix-core name');
    // The core name may name an fp8 op in exactly one place: the HAS_FP8_MATRIX branch that
    // decides whether this ISA has such an op at all. Any other occurrence is a prompt that
    // reached the model without passing that question.
    const fp8ByCoreName = BODY.match(/one fp8 \$\{MATRIX_CORE_NAME\}/g) || [];
    ok(fp8ByCoreName.length === 1, 'the core name names an fp8 op only inside the fp8 branch',
      `n=${fp8ByCoreName.length}`);
    ok(/HAS_FP8_MATRIX\n\s+\? `one fp8 \$\{MATRIX_CORE_NAME\} GEMM`/.test(BODY),
      'that one occurrence is the HAS_FP8_MATRIX branch');
    ok(/const CLIENT_RDNA_ISOLATE = RDNA4_ISOLATE \|\| RDNA35_ISA/.test(BODY),
      'client-RDNA isolation is a union, not a rename of RDNA4_ISOLATE');
    ok(/const VLLM_ONLY = R9700_E2E \|\| RDNA35_ISA/.test(BODY),
      'the vLLM-only set is one named predicate');
  }
  {
    // Behavioural half of the same contract: the deep-dive steers must offer the WMMA layout
    // direction (not matrix_instr_nonkdim, which is an MFMA-only knob) on this ISA.
    const halo = build({ kernel_path: '/tmp/k', ...HALO });
    await halo.run();
    const mfmaOnlyKnob = BODY.indexOf('matrix_instr_nonkdim');
    ok(mfmaOnlyKnob === -1 || /CLIENT_RDNA_ISOLATE\n\s+\? ' DIRECTION=wmma-layout:/.test(BODY),
      'the MFMA-only tuning knob is gated behind the client-RDNA branch');
  }

  console.log(failures === 0
    ? '\nPASS: gfx1151 is isolated, vLLM-only, and never promised an fp8 matrix instruction.'
    : `\nFAIL: ${failures} assertion(s)`);
  process.exit(failures === 0 ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(1); });
