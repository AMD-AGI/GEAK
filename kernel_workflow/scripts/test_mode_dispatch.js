#!/usr/bin/env node
// Regression guard for the kernel_workflow.js DISPATCHER (no GPU, no agent, no network).
//
// Two invariants under test, both exercised against the REAL kernel_workflow.js source:
//   A-D) mode dispatch — args.mode routes to the single-language worker (optimize|author) or to the
//        multi-language bake-off (bakeoff); it normalizes case/whitespace, defaults to optimize, throws
//        on anything else (never silently downgrades), and forwards every arg to the worker untouched.
//   E-J) bake-off LANE ROUTING — the live language optimizes its existing impl, other languages follow
//        Discover's author_plan.route (rewrite -> optimize, author -> author), explicit args.backends
//        overrides auto-discovery, and the winner is picked by speedup on the one frozen baseline.
//
// The Workflow runtime globals (args/phase/log/workflow/agent/parallel/pipeline/budget) are stubbed, so
// this runs in milliseconds and spawns nothing. The runtime wraps a workflow script body in an async
// function — that is why top-level `return` is legal in the script and why we rebuild it the same way.
//
// Run:  node kernel_workflow/scripts/test_mode_dispatch.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const WF_DIR = path.join(ROOT, 'kernel_workflow');
const DISPATCHER = path.join(WF_DIR, 'kernel_workflow.js');
const BODY = fs.readFileSync(DISPATCHER, 'utf8').replace(/^export const meta/m, 'const meta');

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail ? '->  ' + detail : ''); failures++; }
  else console.log('  ok:', msg);
};

// Rebuild the script body with stubbed runtime globals. `stubs.agent` decides what each phase's agent
// returns; `stubs.lane` maps a lane's target_language to its final_speedup.
function build(argsObj, stubs) {
  const s = stubs || {};
  const trace = { phases: [], logs: [], lanes: [], workflowCalls: [], agentLabels: [] };
  const g = {
    args: argsObj,
    phase: (t) => trace.phases.push(t),
    log: (m) => trace.logs.push(m),
    workflow: async (ref, a) => {
      trace.workflowCalls.push({ scriptPath: ref.scriptPath, args: a });
      trace.lanes.push({ lang: a.target_language, mode: a.mode, gpus: a.gpu_ids });
      const sp = s.lane && s.lane[a.target_language] !== undefined ? s.lane[a.target_language] : 1.0;
      return { validation_status: 'validated', final_speedup: sp };
    },
    agent: async (p, o) => {
      const label = (o && o.label) || '';
      trace.agentLabels.push(label);
      return s.agent ? s.agent(label) : null;
    },
    parallel: async (thunks) => Promise.all(thunks.map((t) => t().catch(() => null))),
    pipeline: async (items, ...stages) => Promise.all(items.map(async (it, i) => {
      let v = it;
      for (const st of stages) v = await st(v, it, i);
      return v;
    })),
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
  };
  const fn = new Function(...Object.keys(g), `return (async () => { ${BODY} })();`);
  return { run: () => fn(...Object.values(g)), trace };
}

const BASE = { kernel_path: '/tmp/k', workflow_dir: WF_DIR };

// Freeze that reports a failed smoke: enough to prove bake-off was ENTERED (phase 'Freeze' ran) without
// paying for Discover/Bakeoff/Report. Used by the routing table below.
const freezeFails = (label) =>
  label.startsWith('oracle_freezer') ? { op_kind: 'gemm', task_dir: '/tmp/o', smoke: 'fail' } : null;

// A healthy freeze + a Discover result, parameterized by live backend and author_plan.
const healthy = (liveBackend, authorPlan) => (label) => {
  if (label.startsWith('oracle_freezer')) return {
    op_kind: 'gemm', task_dir: '/tmp/oracle', smoke: 'pass', baseline_frozen: true,
    eval_dir: '/tmp/eval', live_backend: liveBackend, candidate_backends: [],
  };
  if (label.startsWith('op_benchmarker')) return {
    gate: 'pass', isolated_speedup: 1.0, best_known_ms: 1.0, baseline_ms: 1.0,
    tuned_speedup: 0, author_plan: authorPlan,
  };
  if (label.startsWith('bakeoff:report')) return { report_path: '/tmp/eval/bakeoff_report.md' };
  return null;
};

const lanesOf = (trace) => trace.lanes.map((l) => `${l.lang}:${l.mode}`).sort().join(', ');

(async () => {
  // -------------------------------------------------------------------------
  console.log('\n# A. mode -> route (worker pass-through vs bake-off)');
  const ROUTES = [
    ['mode absent (default)', undefined, 'lane'],
    ['mode=optimize', 'optimize', 'lane'],
    ['mode=author', 'author', 'lane'],
    ['mode=bakeoff', 'bakeoff', 'bakeoff'],
    ['mode="  BaKeOff  " (trim + lowercase)', '  BaKeOff  ', 'bakeoff'],
    ['mode="OPTIMIZE" (uppercase)', 'OPTIMIZE', 'lane'],
    ['mode="" (empty string)', '', 'lane'],
    ['mode=null', null, 'lane'],
  ];
  for (const [label, mode, expect] of ROUTES) {
    const a = { ...BASE };
    if (mode !== undefined) a.mode = mode;
    const { run, trace } = build(a, { agent: freezeFails });
    let err = null;
    try { await run(); } catch (e) { err = e; }
    const toLane = trace.workflowCalls.some((c) => String(c.scriptPath).endsWith('kernel_lane.js'));
    const got = err ? `throw(${err.message})` : (toLane ? 'lane' : (trace.phases.includes('Freeze') ? 'bakeoff' : 'none'));
    ok(got === expect, `${label} -> ${expect}`, got);
  }

  // -------------------------------------------------------------------------
  console.log('\n# B. an unknown mode must THROW (never silently downgrade)');
  // 'auto' is listed deliberately: the multi-language fan-out is spelled `bakeoff` and ONLY `bakeoff`.
  // There is no `auto` alias and no auto-detection of intent — this assertion keeps it that way.
  for (const bad of ['auto', 'bake-off', 'bakoff', 'tune', 'BAKEOFF2']) {
    const { run } = build({ ...BASE, mode: bad }, { agent: freezeFails });
    let msg = null;
    try { await run(); } catch (e) { msg = e.message; }
    ok(!!msg && /unknown mode/.test(msg), `mode='${bad}' throws`, msg || 'no throw');
  }

  // -------------------------------------------------------------------------
  console.log('\n# C. pass-through forwards every arg to the worker unchanged');
  {
    const a = { ...BASE, mode: 'author', target_language: 'triton', budget: 3,
                gpu_ids: '2,3', task: 'make it fast', backends: ['hip', 'ck'] };
    const { run, trace } = build(a, {});
    await run();
    const c = trace.workflowCalls[0];
    ok(trace.workflowCalls.length === 1, 'exactly one workflow() call', `n=${trace.workflowCalls.length}`);
    ok(String(c.scriptPath).endsWith(path.sep + 'kernel_lane.js'), 'scriptPath = kernel_lane.js', c.scriptPath);
    for (const k of ['mode', 'target_language', 'budget', 'gpu_ids', 'task', 'kernel_path']) {
      ok(JSON.stringify(c.args[k]) === JSON.stringify(a[k]), `forwards ${k}`, JSON.stringify(c.args[k]));
    }
    ok(trace.agentLabels.length === 0, 'no agent spawned on pass-through', `n=${trace.agentLabels.length}`);
  }

  // -------------------------------------------------------------------------
  console.log('\n# D. required args');
  {
    let m1 = null, m2 = null;
    try { await build({ workflow_dir: WF_DIR }, {}).run(); } catch (e) { m1 = e.message; }
    try { await build({ kernel_path: '/tmp/k' }, {}).run(); } catch (e) { m2 = e.message; }
    ok(/kernel_path is required/.test(m1 || ''), 'missing kernel_path throws', m1);
    ok(/workflow_dir is required/.test(m2 || ''), 'missing workflow_dir throws', m2);
  }

  // -------------------------------------------------------------------------
  console.log('\n# E. auto-discovery: live language optimizes, others follow author_plan.route');
  {
    const { run, trace } = build({ ...BASE, mode: 'bakeoff', gpu_ids: '0,1,2,3' }, {
      agent: healthy('triton', [{ language: 'hip', route: 'author' },
                                { language: 'ck', route: 'rewrite' },
                                { language: 'flydsl', route: 'author' }]),
    });
    await run();
    ok(trace.lanes.length === 4, 'opens 4 lanes', lanesOf(trace));
    ok(trace.lanes.some((l) => l.lang === 'triton' && l.mode === 'optimize'), 'live triton -> optimize', lanesOf(trace));
    ok(trace.lanes.some((l) => l.lang === 'hip' && l.mode === 'author'), 'route=author -> hip:author', lanesOf(trace));
    ok(trace.lanes.some((l) => l.lang === 'ck' && l.mode === 'optimize'), 'route=rewrite -> ck:optimize', lanesOf(trace));
    ok(trace.lanes.some((l) => l.lang === 'flydsl' && l.mode === 'author'), 'flydsl:author', lanesOf(trace));
  }

  // -------------------------------------------------------------------------
  console.log('\n# F. explicit args.backends ADD rewrites but can NEVER drop the incumbent language');
  {
    // input is triton; user asks ONLY for hip+ck rewrites. The incumbent triton:optimize lane MUST still
    // run — a bake-off must always be able to win by simply optimizing what we already have, not only by
    // rewriting into another language. args.backends is additive over the incumbent, not a replacement.
    const { run, trace } = build({ ...BASE, mode: 'bakeoff', backends: ['hip', 'ck'] }, {
      agent: healthy('triton', [{ language: 'ck', route: 'author' }]),
    });
    await run();
    ok(trace.lanes.some((l) => l.lang === 'triton' && l.mode === 'optimize'),
      'incumbent triton:optimize force-included even though backends omitted it', lanesOf(trace));
    ok(lanesOf(trace) === 'ck:author, hip:author, triton:optimize',
      'lanes = incumbent optimize + the 2 requested rewrites', lanesOf(trace));
  }
  {
    // when the incumbent IS listed in backends it collapses to a single optimize lane (no duplicate).
    const { run, trace } = build({ ...BASE, mode: 'bakeoff', backends: ['hip', 'triton'] }, {
      agent: healthy('triton', [{ language: 'ck', route: 'author' }]), // ck planned but not requested
    });
    await run();
    ok(trace.lanes.length === 2, 'listed incumbent does not duplicate (hip + triton only)', lanesOf(trace));
    ok(trace.lanes.some((l) => l.lang === 'triton' && l.mode === 'optimize'), 'triton stays optimize (= live)', lanesOf(trace));
  }

  // -------------------------------------------------------------------------
  console.log('\n# G. explicit backend NOT in author_plan falls back to author (documented behavior)');
  {
    const { run, trace } = build({ ...BASE, mode: 'bakeoff', backends: ['hip'] }, {
      agent: healthy('triton', [{ language: 'hip', route: 'rewrite' }]),
    });
    await run();
    ok(trace.lanes.some((l) => l.lang === 'hip' && l.mode === 'optimize'),
      'hip honors route=rewrite -> optimize', lanesOf(trace));
  }
  {
    const { run, trace } = build({ ...BASE, mode: 'bakeoff', backends: ['hip'] }, {
      agent: healthy('triton', []), // Discover never mentioned hip
    });
    await run();
    // kernel_workflow.js: `planByLang[lang] || 'author'` — an explicitly requested backend that Discover
    // did not plan gets AUTHORED even if an editable impl exists on the box. Leave backends empty to let
    // Discover decide the route.
    ok(trace.lanes.some((l) => l.lang === 'hip' && l.mode === 'author'),
      'hip absent from plan -> author fallback', lanesOf(trace));
  }

  // -------------------------------------------------------------------------
  console.log('\n# H. backends accepts a comma string and normalizes case/whitespace');
  {
    const { run, trace } = build({ ...BASE, mode: 'bakeoff', backends: ' HIP , Triton ,ck ' }, {
      agent: healthy('triton', []),
    });
    await run();
    ok(lanesOf(trace) === 'ck:author, hip:author, triton:optimize', 'parsed to 3 lowercase lanes', lanesOf(trace));
  }

  // -------------------------------------------------------------------------
  console.log('\n# I. no viable lane -> clean abort, nothing spawned');
  {
    const { run, trace } = build({ ...BASE, mode: 'bakeoff' }, { agent: healthy('', []) });
    const r = await run();
    ok(r && r.validation_status === 'no_lanes', 'validation_status=no_lanes', JSON.stringify(r));
    ok(trace.lanes.length === 0, 'no lane opened', `n=${trace.lanes.length}`);
  }

  // -------------------------------------------------------------------------
  console.log('\n# J. winner is the fastest lane on the one frozen baseline');
  {
    const { run } = build({ ...BASE, mode: 'bakeoff', gpu_ids: '0,1,2' }, {
      agent: healthy('triton', [{ language: 'hip', route: 'author' }, { language: 'ck', route: 'author' }]),
      lane: { triton: 1.10, hip: 1.85, ck: 1.42 },
    });
    const r = await run();
    const w = r && r.winner;
    ok(w && w.lang === 'hip' && Math.abs(w.speedup - 1.85) < 1e-9, 'winner = hip @1.85x', JSON.stringify(w));
    ok(w && w.mode === 'author', 'winner.mode = author', JSON.stringify(w));
  }

  // -------------------------------------------------------------------------
  console.log('\n# K. no candidate beats the frozen baseline (all <= 1.0x) -> winner:null / no_winner');
  {
    // Every lane is slower-than-or-equal-to the frozen baseline. A lane that merely "did not fail" must
    // NOT win by default — the winner filter is `speedup > 1.0`, mirroring the env-tune guard. This is the
    // wvSplitK case: the only non-failed lane came in at ~0.17x and the original HIP kernel is kept.
    const { run } = build({ ...BASE, mode: 'bakeoff', gpu_ids: '0,1,2' }, {
      agent: healthy('triton', [{ language: 'hip', route: 'author' }, { language: 'ck', route: 'author' }]),
      lane: { triton: 0.17, hip: 1.0, ck: 0.93 },
    });
    const r = await run();
    ok(r && r.winner == null, 'winner is null when no candidate > 1.0x', JSON.stringify(r && r.winner));
    ok(r && r.validation_status === 'no_winner', 'validation_status=no_winner', r && r.validation_status);
    ok(Array.isArray(r && r.candidates) && r.candidates.length === 3,
      'all 3 candidates still reported for transparency', JSON.stringify(r && (r.candidates || []).length));
  }

  console.log(failures === 0
    ? '\nPASS: mode dispatch + bake-off lane routing behave as specified.'
    : `\nFAILED: ${failures} assertion(s).`);
  process.exit(failures === 0 ? 0 : 1);
})();
