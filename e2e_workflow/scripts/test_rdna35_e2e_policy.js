#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// RDNA 3.5 APU (gfx115x; Strix Halo is gfx1151) policy at the E2E layer, exercised against the
// REAL e2e_workflow.js source (no GPU, no server, no model — Workflow runtime globals are stubbed).
//
// ISOLATION: gfx1151 must reach the kernel lane with the CDNA knowledge inputs off, exactly as
// gfx1201 does. Without it a Strix Halo run is steered by MI-series priors, silently.
//
// Controls: gfx950 must keep every input, and R9700 must keep its own isolation. The assertions
// use warm_start / perf_knowledge_dir, which differ between the two policies; use_learned_kb
// does not (e2e lanes default it off on every architecture), so it cannot serve as a control.
//
// Run:  node e2e_workflow/scripts/test_rdna35_e2e_policy.js
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
  const body = o.body || BODY;
  const trace = { workflowCalls: [], phases: [], logs: [], agentPrompts: [] };
  const g = {
    args: { workflow_dir: E2E_DIR, ...argsObj },
    phase: (t) => {
      trace.phases.push(t);
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
  const fn = new Function(...Object.keys(g), `return (async () => { ${body} })();`);
  return { run: () => fn(...Object.values(g)), trace };
}

async function laneArgsFor(identity) {
  const { run, trace } = build({ kernel_path: '/tmp/k', ...identity });
  await run();
  return (trace.workflowCalls[0] || {}).args || {};
}

// gfx1151 has no product key (the ISA ships as the 8060S, 8050S and 8040S), so identity carries
// target 'unknown'. On ROCm 7.2.3 rocminfo reports an empty marketing name for it.
const HALO = {
  expected_gfx: 'gfx1151',
  expected_target: 'unknown',
  expected_device_name: '',
  expected_physical_cu_count: 40,
};
const MI355X = {
  expected_gfx: 'gfx950',
  expected_target: 'unknown',
  expected_device_name: 'AMD Instinct MI355X',
  expected_physical_cu_count: 256,
};
const R9700 = {
  expected_gfx: 'gfx1201',
  expected_target: 'r9700',
  expected_device_name: 'AMD Radeon AI PRO R9700',
  expected_physical_cu_count: 64,
};

(async () => {
  console.log('\n# laneArgs() isolation on a Strix Halo run');
  {
    const a = await laneArgsFor(HALO);
    ok(a.use_expert_skills === 'false', 'use_expert_skills=false', JSON.stringify(a.use_expert_skills));
    ok(a.perf_knowledge_dir === '', "perf_knowledge_dir=''", JSON.stringify(a.perf_knowledge_dir));
    ok(a.warm_start === 'off', 'warm_start=off', JSON.stringify(a.warm_start));
    ok(a.expected_gfx === 'gfx1151', 'expected_gfx forwarded', JSON.stringify(a.expected_gfx));
    ok(a.expected_physical_cu_count === 40, 'expected_physical_cu_count forwarded',
      JSON.stringify(a.expected_physical_cu_count));
  }
  {
    // No target and no CU count: the ISA alone isolates, and no product identity is invented.
    const a = await laneArgsFor({ expected_gfx: 'gfx1150' });
    ok(a.warm_start === 'off' && a.perf_knowledge_dir === '', 'bare gfx1150 isolates',
      JSON.stringify([a.warm_start, a.perf_knowledge_dir]));
    ok(a.expected_target === undefined, 'no device_target invented', JSON.stringify(a.expected_target));
  }

  console.log('\n# controls: gfx950 keeps every input, R9700 keeps its own isolation');
  {
    const a = await laneArgsFor(MI355X);
    ok(a.warm_start === 'on', 'gfx950 warm_start stays on', JSON.stringify(a.warm_start));
    ok(a.perf_knowledge_dir !== '', 'gfx950 perf_knowledge_dir kept', JSON.stringify(a.perf_knowledge_dir));
  }
  for (const gfx of ['gfx1100', 'gfx1103', 'gfx1250']) {
    const a = await laneArgsFor({ expected_gfx: gfx });
    ok(a.warm_start === 'on' && a.perf_knowledge_dir !== '', `${gfx} is not isolated`,
      JSON.stringify([a.warm_start, a.perf_knowledge_dir]));
  }
  {
    const a = await laneArgsFor(R9700);
    ok(a.warm_start === 'off' && a.perf_knowledge_dir === '', 'R9700 isolation preserved',
      JSON.stringify([a.warm_start, a.perf_knowledge_dir]));
    ok(a.expected_target === 'r9700', 'R9700 product key preserved', JSON.stringify(a.expected_target));
  }

  console.log('\n# serving backend policy (no other backend is validated on gfx1151)');
  const firstPhaseError = async (args) => {
    const { run } = build(args, { stopAtPhase: 1 });
    try { await run(); } catch (e) { return e.message; }
    return '';
  };
  const directorBackend = async (identity) => {
    const { run, trace } = build({ model_path: '/models/m', ...identity }, {
      stopAtPhase: 2,
      setupIdentity: {
        gfx: identity.expected_gfx, target: identity.expected_target,
        name: identity.expected_device_name, cu: identity.expected_physical_cu_count,
      },
    });
    try { await run(); } catch (e) { /* probe stop */ }
    const prompt = (trace.agentPrompts.find((p) => p.label === 'director:setup') || {}).prompt || '';
    return (prompt.match(/BACKEND=(\w+)/) || [])[1];
  };
  for (const backend of ['sglang', 'atom']) {
    const msg = await firstPhaseError({ model_path: '/models/m', backend, ...HALO });
    ok(new RegExp(`^gfx1151 E2E is vLLM-only \\(got backend=${backend}\\)`).test(msg),
      `explicit backend=${backend} fails closed`, msg || 'no throw');
  }
  {
    const msg = await firstPhaseError({
      model_path: '/models/m', backend: 'sglang', expected_gfx: 'gfx1150', expected_target: 'unknown',
    });
    ok(/^gfx1150 E2E is vLLM-only/.test(msg) && /RDNA 3\.5 APUs/.test(msg),
      'the refusal names the ISA it saw', msg || 'no throw');
  }
  {
    // The default case does not throw either way, so read the backend the Director is told to use.
    const halo = await directorBackend(HALO);
    ok(halo === 'vllm', 'gfx1151 default backend is vllm', String(halo));
    const mi = await directorBackend(MI355X);
    ok(mi === 'sglang', 'gfx950 default backend is still sglang', String(mi));
  }
  {
    // atom rather than sglang: which backends R9700 accepts is the R9700 policy's business.
    const msg = await firstPhaseError({ model_path: '/models/m', backend: 'atom', ...R9700 });
    ok(/^R9700 E2E is vLLM/.test(msg) && /got backend=atom/.test(msg),
      'the R9700 guard fires with its own message', msg || 'no throw');
  }
  {
    // A single-kernel pass-through never serves, so an explicit backend must not stop it.
    const { run, trace } = build({ kernel_path: '/tmp/k', backend: 'sglang', ...HALO });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(!msg && trace.workflowCalls.length === 1, 'kernel pass-through ignores the serving backend',
      msg || `${trace.workflowCalls.length} lane call(s)`);
  }

  console.log('\n# effective role inputs: the E2E learned KB');
  for (const [name, identity, want] of [['gfx1151', HALO, 'off'], ['gfx950', MI355X, 'on']]) {
    const { run, trace } = build(
      { model_path: '/models/m', ...identity },
      {
        stopAtPhase: 2,
        setupIdentity: {
          gfx: identity.expected_gfx, target: identity.expected_target,
          name: identity.expected_device_name, cu: identity.expected_physical_cu_count,
        },
      },
    );
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(msg === PROBE_STOP, `${name} setup reached the next phase`, msg || 'no throw');
    const directorPrompt = (trace.agentPrompts.find((p) => p.label === 'director:setup') || {}).prompt || '';
    ok(new RegExp(`E2E_LEARNED_KB: ${want}`).test(directorPrompt),
      `${name} role inputs carry E2E_LEARNED_KB: ${want}`, directorPrompt.slice(-500));
  }

  console.log('\n# source invariants');
  {
    ok(/const CLIENT_RDNA_ISOLATE = RDNA4_ISOLATE \|\| RDNA35_ISA/.test(BODY),
      'client-RDNA isolation is a union, not a rename of RDNA4_ISOLATE');
    ok(/const VLLM_DEFAULT = RDNA4_PRODUCT_E2E \|\| RDNA35_ISA/.test(BODY),
      'the vllm-default set is one named predicate');
    // The head-kernel deep-dive offers the WMMA layout direction instead of the MFMA-only
    // matrix_instr_nonkdim / kpack knobs on every isolated ISA.
    ok(/CLIENT_RDNA_ISOLATE\n\s+\? ' DIRECTION=wmma-layout:/.test(BODY),
      'the MFMA-only tuning knobs are gated behind the client-RDNA branch');
  }

  console.log('\n# fp8 prompts: no fp8 matrix op is prescribed where the ISA has none');
  {
    // Render the prompt fragments for real: return them right after GRAPH_REQ is built.
    const PROBE_BODY = BODY.replace(/(\nconst GRAPH_REQ = [\s\S]*?\n\) : '';\n)/,
      '$1return { GRAPH_REQ, FP8_FUSED_QUANT, FP8_FUSED_GEMM, FP8_FUSED_RUN };\n');
    ok(PROBE_BODY !== BODY, 'the GRAPH_REQ probe point exists');
    const render = async (identity) => build(
      { model_path: '/models/m', backend: 'vllm', ...identity }, { body: PROBE_BODY }).run();
    const halo = await render(HALO);
    const all = (r) => [r.GRAPH_REQ, r.FP8_FUSED_QUANT, r.FP8_FUSED_GEMM, r.FP8_FUSED_RUN].join(' | ');
    ok(!/fp8 (WMMA|MFMA)/.test(all(halo)) && /upcasts them in registers/.test(halo.GRAPH_REQ),
      'gfx1151 is told to upcast in registers, not to use an fp8 matrix op', all(halo).slice(0, 300));
    // Every other ISA renders exactly what it did before this change.
    for (const [name, identity, core] of [['R9700', R9700, 'WMMA'], ['gfx950', MI355X, 'MFMA']]) {
      const r = await render(identity);
      ok(r.GRAPH_REQ.includes(`fused quant (one fp8 ${core}, kill the dequant)`)
        && r.FP8_FUSED_GEMM === `one fp8 ${core} GEMM` && r.FP8_FUSED_RUN === `run ONE fp8 ${core} GEMM`,
        `${name} keeps the fp8 ${core} wording`, all(r).slice(0, 300));
    }
    // The core name may name an fp8 op only in the three per-site constants, so a new prompt
    // cannot reintroduce "fp8 WMMA" on gfx115x without going through FP8_MATRIX_ABSENT.
    const byCore = BODY.split('\n').filter((l) => /fp8 \$\{MATRIX_CORE_NAME\}/.test(l));
    ok(byCore.length === 3 && byCore.every((l) => /^const FP8_FUSED_(QUANT|GEMM|RUN) = FP8_MATRIX_ABSENT/.test(l)),
      'fp8 is joined to the core name only inside the FP8_FUSED_* constants', byCore.join(' || '));
    ok(!/fp8 (WMMA|MFMA)/.test(BODY), 'no prompt hard-codes an fp8 matrix op');
  }

  console.log(failures === 0
    ? '\nPASS: gfx115x E2E policy holds.'
    : `\nFAIL: ${failures} assertion(s)`);
  process.exit(failures === 0 ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(1); });
