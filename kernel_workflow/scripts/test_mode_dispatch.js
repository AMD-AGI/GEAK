#!/usr/bin/env node
// Regression guard for the kernel_workflow.js DISPATCHER (no GPU, no agent, no network).
//
// Invariants under test, all exercised against the REAL kernel_workflow.js source:
//   A-D) mode dispatch — args.mode routes to the single-language worker (optimize|author) or to the
//        multi-language bake-off (bakeoff); it normalizes case/whitespace, defaults to optimize, throws
//        on anything else (never silently downgrades), and forwards every arg to the worker untouched.
//   E-J) bake-off LANE ROUTING — the live language optimizes its existing impl, other languages follow
//        Discover's author_plan.route (rewrite -> optimize, author -> author), explicit args.backends
//        overrides auto-discovery, and the winner is picked by speedup on the one frozen baseline.
//   L-M) bake-off IDENTITY — expected/detected identity reaches every lane, gfx1201 isolates CDNA
//        knowledge, and an unvalidated ISA (gfx1200) is refused before Discover.
//   N-S) optional POPULATION — independent workflows share command-level GPUs, keep private paths,
//        preserve native argv identities, select through the source/evidence query, and can receive
//        a revisable exploration focus through the inherited task without changing the objective.
//
// The Workflow runtime globals (args/phase/log/workflow/agent/parallel/pipeline/budget) are stubbed, so
// model calls are stubbed; population path checks use local CPU Python processes in a temporary tree.
// The runtime wraps a workflow script body in an async
// function — that is why top-level `return` is legal in the script and why we rebuild it the same way.
//
// Run:  node kernel_workflow/scripts/test_mode_dispatch.js
'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawnSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const WF_DIR = path.join(ROOT, 'kernel_workflow');
const dispatcherIndex = process.argv.indexOf('--dispatcher');
const DISPATCHER = dispatcherIndex < 0 ? path.join(WF_DIR, 'kernel_workflow.js') : process.argv[dispatcherIndex + 1];
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
  const trace = { phases: [], logs: [], lanes: [], workflowCalls: [], agentLabels: [], processCalls: [] };
  const g = {
    args: argsObj,
    phase: (t) => trace.phases.push(t),
    log: (m) => trace.logs.push(m),
    workflow: async (ref, a) => {
      trace.workflowCalls.push({ scriptPath: ref.scriptPath, args: a });
      trace.lanes.push({ lang: a.target_language, mode: a.mode, gpus: a.gpu_ids });
      if (s.workflow) return s.workflow(a, ref);
      const sp = s.lane && s.lane[a.target_language] !== undefined ? s.lane[a.target_language] : 1.0;
      return { validation_status: 'accepted', final_speedup: sp };
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
    runProcess: s.runProcess ? async (options) => {
      trace.processCalls.push(options);
      return s.runProcess(options);
    } : undefined,
  };
  const fn = new Function(...Object.keys(g), `return (async () => { ${BODY} })();`);
  return { run: () => fn(...Object.values(g)), trace };
}

const BASE = {
  kernel_path: '/tmp/k', workflow_dir: WF_DIR,
  expected_gfx: 'gfx950', expected_target: 'unknown',
};

// Freeze that reports a failed smoke: enough to prove bake-off was ENTERED (phase 'Freeze' ran) without
// paying for Discover/Bakeoff/Report. Used by the routing table below.
const freezeFails = (label) =>
  label.startsWith('oracle_freezer') ? { op_kind: 'gemm', task_dir: '/tmp/o', smoke: 'fail' } : null;

// A healthy freeze + a Discover result, parameterized by live backend and author_plan.
const healthy = (liveBackend, authorPlan, identity = {
  device_gfx: 'gfx950', device_target: 'unknown',
  device_name: 'AMD Instinct MI355X', physical_cu_count: 256,
}) => (label) => {
  if (label.startsWith('oracle_freezer')) return {
    op_kind: 'gemm', task_dir: '/tmp/oracle', smoke: 'pass', baseline_frozen: true,
    eval_dir: '/tmp/eval', live_backend: liveBackend, candidate_backends: [],
    ...identity,
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
  for (const mode of ['optimize', 'author']) for (const handoff of [undefined, false, true, 'true']) {
    const a = { ...BASE, mode, target_language: 'triton', budget: 3,
                gpu_ids: '2,3', task: 'make it fast', backends: ['hip', 'ck'],
                ...(handoff !== undefined ? {evolution_evidence_handoff: handoff} : {}) };
    const { run, trace } = build(a, {});
    await run();
    const c = trace.workflowCalls[0];
    ok(trace.workflowCalls.length === 1, 'exactly one workflow() call', `n=${trace.workflowCalls.length}`);
    ok(String(c.scriptPath).endsWith(path.sep + 'kernel_lane.js'), 'scriptPath = kernel_lane.js', c.scriptPath);
    for (const k of ['mode', 'target_language', 'budget', 'gpu_ids', 'task', 'kernel_path',
      'evolution_evidence_handoff']) {
      ok(JSON.stringify(c.args[k]) === JSON.stringify(a[k]), `forwards ${k}`, JSON.stringify(c.args[k]));
    }
    ok(trace.agentLabels.length === 0, 'no agent spawned on pass-through', `n=${trace.agentLabels.length}`);
  }

  // -------------------------------------------------------------------------
  console.log('\n# D. required args');
  {
    let m1 = null, m2 = null, m3 = null;
    try { await build({ workflow_dir: WF_DIR }, {}).run(); } catch (e) { m1 = e.message; }
    try { await build({ kernel_path: '/tmp/k' }, {}).run(); } catch (e) { m2 = e.message; }
    try {
      await build(
        { kernel_path: '/tmp/k', workflow_dir: WF_DIR, mode: 'bakeoff' },
        { agent: healthy('hip', [], {}) },
      ).run();
    } catch (e) { m3 = e.message; }
    ok(/kernel_path is required/.test(m1 || ''), 'missing kernel_path throws', m1);
    ok(/workflow_dir is required/.test(m2 || ''), 'missing workflow_dir throws', m2);
    ok(/did not return structured device_gfx and device_target/.test(m3 || ''),
      'bakeoff refuses to Discover when Freeze cannot establish identity', m3);
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

  console.log('\n# L. bakeoff forwards expected identity and fail-closes R9700 knowledge before Discover');
  for (const target of ['mi350x', 'mi355x']) {
    const { run, trace } = build({
      ...BASE, mode: 'bakeoff', expected_gfx: 'gfx950', expected_target: target,
      backends: ['hip'], gpu_ids: '0',
    }, { agent: healthy('hip', [], {
      device_gfx: 'gfx950', device_target: target,
      device_name: `AMD Instinct ${target.toUpperCase()}`, physical_cu_count: 256,
    }) });
    await run();
    ok(trace.workflowCalls[0]?.args.expected_target === target,
      `${target} survives bakeoff dispatch`);
  }
  {
    const { run, trace } = build({
      ...BASE, mode: 'bakeoff', expected_gfx: 'gfx1201', expected_target: 'r9700',
      backends: ['hip'], gpu_ids: '0', use_expert_skills: 'true',
    }, { agent: healthy('hip', [], {
      device_gfx: 'gfx1201', device_target: 'r9700',
      device_name: 'AMD Radeon AI PRO R9700', physical_cu_count: 64,
    }) });
    await run();
    const lane = trace.workflowCalls[0];
    ok(lane && lane.args.expected_gfx === 'gfx1201', 'lane expected_gfx', JSON.stringify(lane && lane.args));
    ok(lane && lane.args.expected_target === 'r9700', 'lane expected_target');
    ok(lane && lane.args.expected_physical_cu_count === 64, 'lane physical CU count');
    ok(lane && lane.args.warm_start === 'off', 'R9700 bakeoff warm_start=off', lane && lane.args.warm_start);
    ok(lane && lane.args.perf_knowledge_dir === '', 'R9700 bakeoff empty perf_knowledge_dir',
      lane && lane.args.perf_knowledge_dir);
    ok(lane && lane.args.use_expert_skills === 'false', 'R9700 bakeoff expert skills off',
      lane && lane.args.use_expert_skills);
  }
  {
    const { run, trace } = build({
      ...BASE, mode: 'bakeoff', expected_gfx: 'gfx1201', expected_target: 'unknown',
      backends: ['hip'], gpu_ids: '0', use_expert_skills: 'true',
    }, { agent: healthy('hip', [], {
      device_gfx: 'gfx1201', device_target: 'unknown',
      device_name: 'Another gfx1201 Product', physical_cu_count: 64,
    }) });
    await run();
    const lane = trace.workflowCalls[0];
    ok(lane && lane.args.expected_target === 'unknown',
      'unknown gfx1201 product is forwarded without R9700 synthesis');
    ok(lane && lane.args.warm_start === 'off',
      'unknown gfx1201 still isolates from CDNA warm-start');
    ok(lane && lane.args.perf_knowledge_dir === '',
      'unknown gfx1201 still isolates from CDNA performance knowledge');
  }
  {
    const { run, trace } = build({
      kernel_path: '/tmp/k', workflow_dir: WF_DIR, mode: 'bakeoff',
      backends: ['hip'], gpu_ids: '0', use_expert_skills: 'true',
    }, { agent: healthy('hip', [], {
      device_gfx: 'gfx1201', device_target: 'r9700',
      device_name: 'AMD Radeon AI PRO R9700', physical_cu_count: 64,
    }) });
    await run();
    const lane = trace.workflowCalls[0];
    ok(lane && lane.args.expected_gfx === 'gfx1201',
      'auto-detected bakeoff forwards the detected ISA');
    ok(lane && lane.args.expected_target === 'r9700',
      'auto-detected bakeoff forwards exact structured product identity');
    ok(lane && lane.args.use_expert_skills === 'false',
      'auto-detected gfx1201 isolates before Discover and lane dispatch');
  }

  console.log('\n# M. an unvalidated ISA (gfx1200) is rejected before Discover');
  {
    const { run, trace } = build({
      kernel_path: '/tmp/k', workflow_dir: WF_DIR, mode: 'bakeoff',
      backends: ['hip'], gpu_ids: '0',
    }, { agent: healthy('hip', [], {
      device_gfx: 'gfx1200', device_target: 'unknown',
      device_name: 'AMD Radeon Graphics', physical_cu_count: 56,
    }) });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(/gfx1200/.test(msg), 'Freeze-detected gfx1200 throws', msg || 'no throw');
    ok(!trace.phases.includes('Discover'), 'detected gfx1200 never reaches Discover',
      JSON.stringify(trace.phases));
    ok(!trace.agentLabels.some((l) => l.startsWith('op_benchmarker')),
      'no op_benchmarker agent spawned', JSON.stringify(trace.agentLabels));
    ok(trace.lanes.length === 0, 'no lane opened', lanesOf(trace));
  }
  {
    const { run, trace } = build({
      ...BASE, mode: 'bakeoff', expected_gfx: 'gfx1200', expected_target: 'unknown',
    }, { agent: healthy('hip', []) });
    let msg = '';
    try { await run(); } catch (e) { msg = e.message; }
    ok(/gfx1200 is not supported/.test(msg), 'expected gfx1200 throws', msg || 'no throw');
    ok(trace.agentLabels.length === 0, 'expected gfx1200 spawns no agent, not even Freeze',
      JSON.stringify(trace.agentLabels));
  }

  console.log('\n# N. declared population keeps private lineages and a shared command-level GPU pool');
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'geak-population-dispatch-'));
  try {
    const task = path.join(temp, 'task');
    const seed = path.join(temp, 'seed');
    fs.mkdirSync(task); fs.mkdirSync(seed);
    fs.writeFileSync(path.join(task, 'kernel.py'), 'original source\n');
    for (const key of ['learned', 'artifacts', 'store']) {
      fs.mkdirSync(path.join(seed, key));
      fs.writeFileSync(path.join(seed, key, 'context.txt'), 'shared seed\n');
    }
    const populationArgs = (name) => ({ ...BASE, kernel_path: task, mode: 'optimize', optimizer: 'evo',
      eval_dir: path.join(temp, name), gpu_ids: '0, 2', gpu_mode: 'pin', budget: 9,
      learned_dir: path.join(seed, 'learned'), kb_artifacts_dir: path.join(seed, 'artifacts'),
      kb_store_dir: path.join(seed, 'store'), kb_remote: 'auto', llm_stats: 'false',
      lineages: [{ id: 'ordinary', optimizer: 'greedy', imports: [] },
        { id: 'explorer', optimizer: 'evo', imports: ['ordinary'] }],
    });
    const clean = stdout => ({ status: 'completed', code: 0, group_cleared: true,
      strays_killed: false, stdio_held: false, stdout, stderr: '' });
    function hostWithQuery(query) {
      return async options => {
        if (query && options.args[0] === `${WF_DIR}/scripts/query_population.py`) return clean(JSON.stringify(query()));
        const result = spawnSync(options.cmd, options.args, { encoding: 'utf8', timeout: options.timeoutMs, input: options.input });
        return { ...clean(result.stdout), code: result.status, stderr: result.stderr || String(result.error || '') };
      };
    }
    const returned = {};
    let active = 0, peak = 0;
    const args = populationArgs('run');
    const { run, trace } = build(args, {
      workflow: async child => {
        active++; peak = Math.max(peak, active);
        await new Promise(resolve => setTimeout(resolve, child.optimizer === 'greedy' ? 10 : 1));
        active--;
        if (child.optimizer === 'evo') throw new Error('peer failed after search');
        const result = { mode: 'optimize', eval_dir: child.eval_dir, final_speedup: 1.4,
          final_patch: `${child.eval_dir}/winner.diff`, validation_status: 'accepted' };
        returned[child.population_lineage.id] = result;
        return result;
      },
      runProcess: hostWithQuery(() => ({ schema: 'geak.population.query/1', candidates: [],
        selected: { lineage_id: 'ordinary', kind: 'final', qualification: 'legacy_verifier',
          speedup: 1.4, result: returned.ordinary } })),
    });
    const result = await run();
    ok(peak === 2, 'all workflows can overlap without a whole-lineage GPU reservation');
    ok(trace.workflowCalls.length === 2, 'declared lineages each launch one existing worker');
    ok(result.population.selected.id === 'ordinary' && result.final_speedup === 1.4,
      'an eligible ordinary member survives a failed peer');
    ok(result.population.lineages.find(row => row.id === 'explorer').status === 'failed',
      'failed member disposition is retained');
    for (const call of trace.workflowCalls) {
      const a = call.args, id = a.population_lineage.id;
      ok(!Object.prototype.hasOwnProperty.call(a, 'task'), `${id} without focus preserves an absent task`);
      ok(a.gpu_ids === '0,2' && a.gpu_mode === 'pool', `${id} receives the same normalized GPU pool`);
      ok(a.budget === 9 && a.apply_to_original === 'false' && !('lineages' in a),
        `${id} keeps its own budget and isolated worker invocation`);
      ok(a.kb_remote === 'auto' && a.kb_remote_write === 'off', `${id} retains remote read policy with private writes`);
      ok(a.population_lineage.root === `${args.eval_dir}/population`, `${id} has the shared publication root`);
      ok(a.continuing_lineage === (a.optimizer === 'evo' ? true : undefined), `${id} receives the appropriate continuation policy`);
      for (const key of ['eval_dir', 'exp_root', 'state_dir', 'learned_dir', 'kb_artifacts_dir', 'kb_store_dir']) {
        ok(a[key].startsWith(`${args.eval_dir}/lineages/${id}/`), `${id} owns ${key}`);
      }
      fs.writeFileSync(path.join(a.learned_dir, 'context.txt'), id);
    }
    ok(fs.readFileSync(path.join(seed, 'learned', 'context.txt'), 'utf8') === 'shared seed\n',
      'private learned writes preserve original read context');
    ok(fs.readFileSync(path.join(task, 'kernel.py'), 'utf8') === 'original source\n', 'population leaves the original source untouched');
    const manifest = JSON.parse(fs.readFileSync(`${args.eval_dir}/population/manifest.json`, 'utf8'));
    ok(manifest.schema === 'geak.population/1' && manifest.objective.kind === 'geomean'
      && manifest.lineages.length === 2, 'fresh trusted manifest records all lineage identities and objective');
    ok(fs.existsSync(`${args.eval_dir}/lineages/ordinary/eval/population_final_result.json`),
      'host preserves returned final results for read-only selection');
    ok(fs.existsSync(`${args.eval_dir}/population/result.json`), 'population result is persisted');

    console.log('\n# O. native argv rewrite preserves option identities and distinct evaluation state');
    const nativeArgs = populationArgs('native');
    nativeArgs.objective = { kind: 'external_primary', primary_regime: 'decode' };
    const nativeCommand = (state, action) => ['python3', '/native/runner.py', '--task-root', task,
      '--state-root', state, '--gpu-lock-file', '/tmp/gpu.lock', '--epoch', 'test', action];
    nativeArgs.external_measurement_command = nativeCommand('/tmp/development-state', 'measure');
    nativeArgs.external_result_command = nativeCommand('/tmp/development-state', 'read');
    nativeArgs.external_final_measurement_command = nativeCommand('/tmp/final-state', 'measure');
    nativeArgs.external_final_result_command = nativeCommand('/tmp/final-state', 'read');
    nativeArgs.lineages[1].focus = 'Explore parallel reductions while retaining the full native task.';
    Object.assign(nativeArgs, { deadline_epoch: 2000000000, search_reserve_s: 300,
      final_validation_reserve_s: 60, final_cleanup_reserve_s: 10, population_final_reserve_s: 100 });
    const nativeBuild = build(nativeArgs, {
      workflow: async a => ({ eval_dir: a.eval_dir, validation_status: 'invalid_measurement', final_speedup: 99 }),
      runProcess: hostWithQuery(() => ({ schema: 'geak.population.query/1', selected: null, candidates: [], finalists: [] })),
    });
    const nativeResult = await nativeBuild.run();
    const optionValue = (argv, option) => argv[argv.indexOf(option) + 1];
    for (const call of nativeBuild.trace.workflowCalls) {
      const a = call.args;
      ok(a.objective === nativeArgs.objective,
        `${a.population_lineage.id}: focus leaves the full native primary objective unchanged`);
      ok(a.deadline_epoch === nativeArgs.deadline_epoch - 100
        && a.deadline_epoch - a.search_reserve_s === nativeArgs.deadline_epoch - nativeArgs.search_reserve_s,
        `${a.population_lineage.id}: common final reserve preserves the declared search cutoff`);
      ok(optionValue(a.external_measurement_command, '--state-root') === optionValue(a.external_result_command, '--state-root'),
        `${a.population_lineage.id}: measurement and reader share private development state`);
      ok(optionValue(a.external_result_command, '--state-root') !== optionValue(a.external_final_result_command, '--state-root'),
        `${a.population_lineage.id}: separate final state stays separate`);
      ok(optionValue(a.external_result_command, '--gpu-pool') === '0,2'
        && !a.external_result_command.includes('--gpu-lock-file') && a.external_result_command.at(-1) === 'read',
        `${a.population_lineage.id}: exact options change without positional guessing`);
    }
    ok(nativeArgs.external_result_command.includes('--gpu-lock-file')
      && optionValue(nativeArgs.external_result_command, '--state-root') === '/tmp/development-state',
      'native caller argv is unchanged');
    ok(nativeResult.final_speedup === null && nativeResult.population.selected === null,
      'unqualified self-reported native scalars never become a population winner');
    ok(nativeResult.population.common_final.confirmations.length === 2
      && nativeResult.population.common_final.confirmations.every(row => row.status === 'no_retained_candidate'),
      'every configured member without a deliverable candidate has an explicit disposition');
    for (const reserve of [undefined, 0, 250]) {
      const invalidReserve = build({ ...nativeArgs, population_final_reserve_s: reserve }, {
        runProcess: hostWithQuery(null),
      });
      let error = null;
      try { await invalidReserve.run(); } catch (e) { error = e; }
      ok(!!error && invalidReserve.trace.workflowCalls.length === 0,
        `missing or infeasible common final reserve (${reserve}) fails before search`);
    }

    console.log('\n# P. invalid population configuration fails before any worker launches');
    const invalid = [
      ['wrong mode', { mode: 'author' }], ['wrong optimizer', { optimizer: 'greedy' }],
      ['original writeback', { apply_to_original: true }],
      ['relative output', { eval_dir: 'relative' }], ['empty members', { lineages: [] }],
      ['duplicate IDs', { lineages: [{ id: 'a', optimizer: 'greedy' }, { id: 'a', optimizer: 'evo' }] }],
      ['foreign import', { lineages: [{ id: 'a', optimizer: 'evo', imports: ['missing'] }] }],
      ['greedy import', { lineages: [{ id: 'a', optimizer: 'greedy', imports: ['b'] }, { id: 'b', optimizer: 'evo' }] }],
      ['unknown member key', { lineages: [{ id: 'a', optimizer: 'greedy', focus: 'Explore staging.', niche: 'cell' }] }],
      ['invalid GPU pool', { gpu_ids: '0,0' }],
      ['combined state flag', { external_result_command: ['reader', '--state-root=/tmp/state', '--gpu-pool', '0', 'read'] }],
      ['duplicate state flag', { external_result_command: ['reader', '--state-root', '/tmp/a', '--state-root', '/tmp/b', '--gpu-pool', '0', 'read'] }],
    ];
    for (const focus of [undefined, null, false, 0, [], {}, '', ' \n\t\u2003']) {
      invalid.push([`invalid focus (${String(focus)})`, {
        lineages: [{ id: 'a', optimizer: 'greedy', focus }],
      }]);
    }
    for (const [name, override] of invalid) {
      const trial = build({ ...populationArgs('invalid'), ...override }, { runProcess: hostWithQuery(() => null) });
      let error = null;
      try { await trial.run(); } catch (e) { error = e; }
      ok(!!error && trial.trace.workflowCalls.length === 0, `${name} fails before launching a lineage`);
      if (name.startsWith('invalid focus')) ok(error && /focus must be a nonempty string/.test(error.message)
        && trial.trace.processCalls.length === 0, `${name} is rejected before any host mutation`);
    }
    const stale = build(args, { runProcess: hostWithQuery(() => null) });
    let reused = null;
    try { await stale.run(); } catch (error) { reused = error; }
    ok(!!reused && stale.trace.workflowCalls.length === 0, 'reused output roots cannot import stale population evidence');

    console.log('\n# Q. real checkpoint query ranks preserved lineage finals');
    const realQuery = build(populationArgs('real_query'), {
      workflow: async a => {
        fs.mkdirSync(a.eval_dir, { recursive: true });
        const patch = `${a.eval_dir}/final.diff`;
        fs.writeFileSync(patch, 'diff --git a/kernel.py b/kernel.py\n--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-original source\n+better source\n');
        return { mode: 'optimize', eval_dir: a.eval_dir, final_patch: patch,
          final_speedup: a.optimizer === 'greedy' ? 1.2 : 9,
          validation_status: a.optimizer === 'greedy' ? 'accepted' : 'flagged' };
      },
      runProcess: hostWithQuery(null),
    });
    const queried = await realQuery.run();
    ok(queried.population.selected.id === 'ordinary' && queried.final_speedup === 1.2,
      'real query preserves the accepted ordinary member over an unqualified faster claim');
    ok(queried.population.query.candidates.find(row => row.lineage_id === 'explorer').status === 'disqualified',
      'real query retains the rejected final disposition');
    ok(queried.eval_dir === path.join(temp, 'real_query'), 'public result retains the caller eval_dir');

    console.log('\n# R. all members receive a common native final before generic selection');
    const nativeTask = path.join(temp, 'native_task');
    const fixture = spawnSync('python3', ['-c',
      'import json,sys; from pathlib import Path; from tests.native.fixtures import fixture; p=Path(sys.argv[1]); c=fixture(p); (p/"task.json").write_text(json.dumps(c))',
      nativeTask], { cwd: ROOT, encoding: 'utf8' });
    if (fixture.status !== 0) throw new Error(fixture.stderr);
    const synthetic = path.join(temp, 'synthetic_native.py'), audit = path.join(temp, 'native_calls.jsonl');
    // Native capture and readers are real. Only the UT execution is replaced by
    // deterministic fixture evidence, so this exercises the handoff without a GPU.
    fs.writeFileSync(synthetic, `import json,sys\nfrom pathlib import Path\nsys.path.insert(0,${JSON.stringify(ROOT)})\n` + String.raw`from scripts.native_runtime import direct_external_command as native
from tests.native import fixtures
argv=sys.argv[1:]; i=argv.index('--audit-file'); audit=Path(argv[i+1]); del argv[i:i+2]
with audit.open('a') as stream: stream.write(json.dumps(argv)+'\n')
state=argv[argv.index('--state-root')+1]
def evidence(request,workspace,out,**options):
 common='caller_final_' in state
 explorer=any(b'explorer' in Path(p).read_bytes() for p in request['source_objects'].values())
 if 'caller_final_failure' in state: return 3
 if 'caller_final_peer_failure' in state and explorer: return 3
 original=(not explorer) if not common else ('caller_final_no_gain' in state or ('caller_final_gain' in state and explorer))
 for name,raw in fixtures.raw_reports(request,original=original).items():
  target=Path(out)/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
 return 0
native.run_native=evidence
i=argv.index('--gpu-pool'); argv[i:i+2]=['--gpu-lock-file',str(audit.parent/'synthetic-gpu.lock')]
raise SystemExit(native.main(argv))
`);
    const cpu = (cmd, commandArgs, cwd) => {
      const process = spawnSync(cmd, commandArgs, { cwd, encoding: 'utf8', timeout: 30000 });
      if (process.status !== 0) throw new Error(process.stderr || `command failed: ${cmd}`);
      return process.stdout;
    };
    const invoke = (command, more) => cpu(command[0], [...command.slice(1), ...more], ROOT);
    for (const finalOutcome of ['gain', 'no_gain', 'failure', 'peer_failure', 'deadline']) {
      const a = { ...populationArgs(`handoff_${finalOutcome}`), kernel_path: nativeTask,
        objective: { kind: 'external_primary', primary_regime: 'extend' } };
      const command = (stage, action) => ['python3', synthetic, '--audit-file', audit, '--task-root', nativeTask,
        '--state-root', path.join(temp, stage === 'final' ? `caller_final_${finalOutcome}` : `caller_development_${finalOutcome}`),
        '--gpu-pool', '0,2', '--epoch', stage, action];
      a.external_measurement_command = command('development', 'measure');
      a.external_result_command = command('development', 'read');
      a.external_final_measurement_command = command('final', 'measure');
      a.external_final_result_command = command('final', 'read');
      if (finalOutcome === 'peer_failure') a.lineages.push({id: 'missing', optimizer: 'evo', imports: []});
      const callerState = optionValue(a.external_final_result_command, '--state-root');
      const host = hostWithQuery(null);
      let commonAttempts = 0, blocked = false, publicPatchDuringConfirmation = false;
      const handoff = build(a, {
        workflow: async child => {
          if (child.population_lineage.id === 'missing') throw new Error('member exited without a retained candidate');
          const workspace = `${child.eval_dir}/workspace`;
          fs.mkdirSync(child.eval_dir, { recursive: true });
          fs.cpSync(nativeTask, workspace, { recursive: true });
          cpu('git', ['init', '-q'], workspace); cpu('git', ['add', '-A'], workspace);
          fs.writeFileSync(`${workspace}/kernel_src/module.py`, `# ${child.population_lineage.id} candidate\n`);
          const patch = `${child.eval_dir}/host_final_patch.diff`;
          fs.writeFileSync(patch, cpu('git', ['diff', '--binary'], workspace));
          const capture = JSON.parse(invoke(child.external_final_measurement_command, ['--workspace', workspace, '--capture-only']));
          const pair = ['--candidate-id', capture.candidate_id, '--evaluation-id', capture.evaluation_id];
          invoke(child.external_final_measurement_command, ['--captured', ...pair]);
          const observed = JSON.parse(invoke(child.external_final_result_command, ['--workspace', nativeTask, '--patch', patch, ...pair]));
          return { ...observed, mode: 'optimize', eval_dir: child.eval_dir, final_speedup: observed.primary_speedup,
            final_selection: capture, final_patch: observed.native_reproducible_gain ? patch : '',
            final_patch_sha256: observed.native_reproducible_gain ? observed.patch_sha256 : null };
        },
        runProcess: async options => {
          const commonMeasurement = optionValue(options.args, '--state-root') === callerState
            && options.args.includes('--captured') && options.args.includes('measure');
          if (commonMeasurement) {
            commonAttempts++;
            publicPatchDuringConfirmation ||= fs.existsSync(`${a.eval_dir}/host_final_patch.diff`);
            if (finalOutcome === 'deadline' && commonAttempts === 2) blocked = true;
          }
          if (blocked) throw Object.assign(new Error('common final process deadline'), {code: 'GEAK_PROCESS_DEADLINE'});
          return host(options);
        },
      });
      const r = await handoff.run();
      const calls = fs.readFileSync(audit, 'utf8').trim().split('\n').map(line => JSON.parse(line));
      const measured = calls.filter(argv => optionValue(argv, '--state-root') === callerState
        && argv.includes('measure') && !argv.includes('--capture-only'));
      ok(commonAttempts === 2 && measured.length === (finalOutcome === 'deadline' ? 1 : 2)
        && measured.every(argv => argv.includes('--captured') && optionValue(argv, '--gpu-pool') === '0'),
        `${finalOutcome}: all retained member sources are attempted on the same single GPU`);
      ok(!publicPatchDuringConfirmation, `${finalOutcome}: public final patch is created only after common selection`);
      ok(r.eval_dir === a.eval_dir && r.population.search_selected.lineage_id === 'explorer',
        `${finalOutcome}: caller output root and earlier search winner remain recorded`);
      const confirmations = r.population.common_final.confirmations;
      ok(confirmations.length === a.lineages.length && confirmations.filter(row => row.source).every(row => row.patch_ref
        && row.process_costs.some(cost => cost.operation === 'measure') && row.duration_ms === null),
        `${finalOutcome}: every source, confirmation status and available process cost is retained`);
      if (finalOutcome === 'failure') {
        ok(r.validation_status === 'invalid_measurement' && r.final_speedup === null && r.final_patch === '',
          'failed common measurement cannot borrow an earlier scalar or export claim');
        ok(!r.population.common_final.complete && !fs.existsSync(`${a.eval_dir}/host_final_patch.diff`),
          'failed comparison has no public winner alias');
      } else {
        const winningPatch = r.final_patch || r.population.common_final.patch_ref.path;
        const read = JSON.parse(invoke(a.external_final_result_command, ['--workspace', nativeTask,
          '--patch', winningPatch, '--candidate-id', r.final_selection.candidate_id,
          '--evaluation-id', r.final_selection.evaluation_id]));
        ok(read.source_id === r.final_selection.source_id && read.primary_speedup === r.final_speedup,
          `${finalOutcome}: caller original reader resolves exact returned source and IDs`);
        ok(finalOutcome === 'no_gain' ? r.final_patch === '' && r.outcome === 'no_improvement'
          : !!r.final_patch && r.outcome === 'improved', `${finalOutcome}: public export follows common native gain qualification`);
      }
      if (finalOutcome === 'gain') {
        ok(r.population.selected.id === 'ordinary' && r.final_speedup > 2 && r.population.common_final.complete,
          'common rank reversal preserves the included ordinary member and selects its confirmed score');
      } else if (finalOutcome === 'peer_failure') {
        ok(r.population.selected.id === 'ordinary' && !r.population.common_final.complete
          && confirmations.find(row => row.id === 'explorer').status === 'failed'
          && confirmations.find(row => row.id === 'missing').status === 'no_retained_candidate',
          'a failed final peer and missing member cannot disqualify the ordinary confirmed winner');
      } else if (finalOutcome === 'deadline') {
        ok(r.population.selected.id === 'explorer' && confirmations.find(row => row.id === 'ordinary').status === 'deadline'
          && !r.population.common_final.complete && !!r.population.common_final.delivery_alias_error
          && !!r.population.persistence_error && !fs.existsSync(`${a.eval_dir}/host_final_patch.diff`),
          'late cutoff returns the earlier confirmed winner with explicit alias and persistence failures');
        ok(fs.existsSync(`${a.eval_dir}/population/explorer/common_final_result.json`)
          && r.final_patch === `${a.eval_dir}/population/explorer/common_final_patch.diff`,
          'completed confirmation is persisted before the later cutoff and remains a valid public patch');
      }
    }

    console.log('\n# S. optional focus reaches either optimizer and preserves unfocused task arguments');
    const shortFocus = '  Explore producer/consumer staging.\nKeep intermediate reuse local: \u03bb, `code`, $values, $(literal).  ';
    const taskCases = [
      ['absent', {}], ['undefined', { task: undefined }], ['empty', { task: '' }],
      ['verbatim', { task: '  Optimize the full task.\r\nPreserve literal `code`, $values and trailing whitespace.\n\t' }],
      ['large_focus', { task: 'Optimize the full task.' }],
    ];
    for (const [name, taskFields] of taskCases) {
      const focus = name === 'large_focus' ? shortFocus.repeat(1000) : shortFocus;
      const a = { ...populationArgs(`focus_${name}`), ...taskFields,
        lineages: [{ id: 'a', optimizer: 'greedy', imports: [] },
          { id: 'b', optimizer: 'greedy', imports: [], focus },
          { id: 'c', optimizer: 'evo', imports: ['a', 'b'], focus }],
      };
      const before = JSON.stringify(a);
      const focused = build(a, {
        workflow: async child => {
          if (child.population_lineage.id === 'c') throw new Error('focused member failed');
          return { eval_dir: child.eval_dir, validation_status: 'invalid_measurement' };
        },
        runProcess: hostWithQuery(() => ({ schema: 'geak.population.query/1', selected: null, candidates: [] })),
      });
      const r = await focused.run();
      ok(focused.trace.workflowCalls.length === 3 && focused.trace.agentLabels.length === 0,
        `${name}: focus launches one existing worker per member without new planner calls`);
      const children = Object.fromEntries(focused.trace.workflowCalls.map(call => [call.args.population_lineage.id, call.args]));
      ok(children.a.task === a.task
        && Object.prototype.hasOwnProperty.call(children.a, 'task') === Object.prototype.hasOwnProperty.call(a, 'task'),
        `${name}: unfocused ordinary member preserves the original task value and property presence`);
      for (const id of ['b', 'c']) {
        const child = children[id];
        ok(typeof child.task === 'string' && child.task.startsWith(a.task || '') && child.task.includes(focus),
          `${name}: ${child.optimizer} receives the verbatim task prefix and focus through task`);
      }
      ok(children.b.task === children.c.task, `${name}: a matching focus produces matching task steering across optimizers`);
      ok(children.b.continuing_lineage === undefined && children.c.continuing_lineage === true,
        `${name}: focus preserves each optimizer's continuation policy`);
      const m = JSON.parse(fs.readFileSync(`${a.eval_dir}/population/manifest.json`, 'utf8'));
      const persisted = JSON.parse(fs.readFileSync(`${a.eval_dir}/population/result.json`, 'utf8'));
      for (const [label, rows] of [['manifest', m.lineages], ['returned result', r.population.lineages],
        ['persisted result', persisted.population.lineages]]) {
        ok(!Object.prototype.hasOwnProperty.call(rows.find(row => row.id === 'a'), 'focus')
          && rows.filter(row => row.id !== 'a').every(row => row.focus === focus),
          `${name}: ${label} records only explicitly supplied focus strings without trimming`);
      }
      ok(r.population.lineages.find(row => row.id === 'c').status === 'failed',
        `${name}: failed focused member retains its metadata and disposition`);
      ok(JSON.stringify(a) === before, `${name}: population leaves caller arguments unchanged`);
    }
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }

  console.log(failures === 0
    ? '\nPASS: mode dispatch + bake-off lane routing behave as specified.'
    : `\nFAILED: ${failures} assertion(s).`);
  process.exit(failures === 0 ? 0 : 1);
})();
