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
  const fn = new Function(...Object.keys(g), `return (async () => { ${BODY} })();`);
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
    // The head-kernel deep-dive offers the WMMA layout direction instead of the MFMA-only
    // matrix_instr_nonkdim / kpack knobs on every isolated ISA.
    ok(/CLIENT_RDNA_ISOLATE\n\s+\? ' DIRECTION=wmma-layout:/.test(BODY),
      'the MFMA-only tuning knobs are gated behind the client-RDNA branch');
  }

  console.log(failures === 0
    ? '\nPASS: gfx115x E2E policy holds.'
    : `\nFAIL: ${failures} assertion(s)`);
  process.exit(failures === 0 ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(1); });
