#!/usr/bin/env node
// Regression guard for the config_aware (operator family) path of kernel_workflow.js. No GPU, no agent,
// no network: the Workflow runtime globals are stubbed and the REAL dispatcher body is executed, the
// same way test_mode_dispatch.js does it.
//
//   A) the flag: only "true"/"on" take the family path; everything else keeps today's dispatcher,
//      including the kernel_path requirement
//   B) family_spec validation happens before any agent or lane is started
//   C) planner output is validated; a bad plan falls back to one group per config
//   D) lane arguments: overrides -> defaults -> lane default, one GPU id, own eval_dir,
//      kernel_lane.js only, apply_to_original forced "false"
//   E) lanes launch together; one GPU serializes them; execution_order only sorts the report
//   F) lane failures are isolated and every status is mapped
//   G) catalog: only optimized configs, exact coverage, every other config in fallback exactly once
//   H) aggregate only across one timing basis
//   I) persisted artifacts are the returned objects, verbatim
//   J) the family eval dir: pinned, created by the setup agent, or refused
//
// Run:  node kernel_workflow/scripts/test_config_aware.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const WF_DIR = path.join(ROOT, 'kernel_workflow');
const BODY = fs.readFileSync(path.join(WF_DIR, 'kernel_workflow.js'), 'utf8')
  .replace(/^export const meta/m, 'const meta');

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail ? '->  ' + detail : ''); failures++; }
  else console.log('  ok:', msg);
};

// stubs.agent(label, prompt) answers agent calls; stubs.lane(args) answers (or throws for) lane calls.
function build(argsObj, stubs) {
  const s = stubs || {};
  const trace = { phases: [], logs: [], workflowCalls: [], agentCalls: [], inFlight: 0, maxInFlight: 0 };
  const g = {
    args: argsObj,
    phase: (t) => trace.phases.push(t),
    log: (m) => trace.logs.push(m),
    workflow: async (ref, a) => {
      trace.workflowCalls.push({ scriptPath: ref.scriptPath, args: a });
      trace.inFlight++;
      trace.maxInFlight = Math.max(trace.maxInFlight, trace.inFlight);
      try {
        return s.lane ? await s.lane(a) : { validation_status: 'accepted', final_speedup: 1.0 };
      } finally { trace.inFlight--; }
    },
    agent: async (p, o) => {
      const label = (o && o.label) || '';
      trace.agentCalls.push({ label, prompt: p });
      return s.agent ? s.agent(label, p) : null;
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

const FAM_DIR = '/exp/family_gemm_test';
const cfg = (id, extra) => ({
  id, kernel_path: `/tasks/${id}`, op_spec: { op_kind: 'gemm', dtype: 'bf16' }, ...(extra || {}),
});
const spec = (extra) => ({
  schema_version: 1, family_id: 'gemm_n2048_k1024', operator: 'gemm',
  deployment: { signature_axes: ['M', 'N', 'K'] },
  defaults: { mode: 'optimize', target_language: 'triton', budget: 4 },
  configs: [
    cfg('m1', { signature: [1, 2048, 1024] }),
    cfg('m2', { signature: [2, 2048, 1024] }),
    cfg('prefill', { signature: [4096, 2048, 1024], overrides: { mode: 'author', target_language: 'flydsl', budget: 2 } }),
  ],
  ...(extra || {}),
});
const famArgs = (extra) => ({
  config_aware: 'true', workflow_dir: WF_DIR, eval_dir: FAM_DIR, gpu_ids: '0,1,2', family_spec: spec(), ...(extra || {}),
});
const writerOk = (label, prompt) => {
  if (!label.startsWith('file_writer')) return null;
  return { written: [...prompt.matchAll(/^### (\S+)$/gm)].map((m) => m[1]) };
};
const withPlanner = (plan) => (label, prompt) =>
  label === 'operator_family_planner:plan' ? plan : writerOk(label, prompt);
const laneFor = (trace, id) => trace.workflowCalls.find((c) => c.args.eval_dir === `${FAM_DIR}/configs/${id}`);
const persisted = (trace) => {
  const files = {};
  for (const c of trace.agentCalls.filter((x) => x.label.startsWith('file_writer:family'))) {
    for (const m of c.prompt.matchAll(/^### (\S+)\n````\n([\s\S]*?)\n````/gm)) files[m[1]] = m[2];
  }
  return files;
};
const throwsOf = async (args, stubs) => {
  const b = build(args, stubs || {});
  try { await b.run(); return { msg: null, trace: b.trace }; } catch (e) { return { msg: e.message, trace: b.trace }; }
};

(async () => {
  console.log('\n# A. the config_aware flag');
  for (const v of [undefined, 'false', 'off', 'yes', '1', '', 'FALSE']) {
    const a = { kernel_path: '/tmp/k', workflow_dir: WF_DIR, family_spec: spec() };
    if (v !== undefined) a.config_aware = v;
    const { run, trace } = build(a, {});
    await run();
    const c = trace.workflowCalls[0];
    ok(trace.workflowCalls.length === 1 && c.args.kernel_path === '/tmp/k' && !c.args.eval_dir,
      `config_aware=${JSON.stringify(v)} keeps the single-lane pass-through`, JSON.stringify(c && c.args.eval_dir));
  }
  for (const v of [undefined, 'false', 'off']) {
    const a = { workflow_dir: WF_DIR, family_spec: spec() };
    if (v !== undefined) a.config_aware = v;
    const { msg } = await throwsOf(a);
    ok(/kernel_path is required/.test(msg || ''), `config_aware=${JSON.stringify(v)} still requires kernel_path`, msg);
  }
  for (const v of ['true', 'on', 'TRUE', ' On ']) {
    const { run, trace } = build(famArgs({ config_aware: v }), { agent: writerOk });
    const r = await run();
    ok(r && r.mode === 'config_aware' && trace.workflowCalls.length === 3,
      `config_aware=${JSON.stringify(v)} runs one lane per config without a parent kernel_path`,
      `n=${trace.workflowCalls.length}`);
  }
  {
    const { run, trace } = build({ ...famArgs(), mode: 'author', kernel_path: '/ignored' }, { agent: writerOk });
    await run();
    ok(trace.workflowCalls.every((c) => c.args.kernel_path !== '/ignored'), 'parent kernel_path is not used by any lane');
    ok(laneFor(trace, 'm1').args.mode === 'optimize', 'parent mode=author is ignored; the config resolves its own');
  }

  console.log('\n# B. validation happens before any agent or lane');
  const BAD = [
    ['family_spec missing', famArgs({ family_spec: undefined }), /already-parsed JSON object/],
    ['family_spec as a string', famArgs({ family_spec: JSON.stringify(spec()) }), /already-parsed JSON object/],
    ['schema_version 2', famArgs({ family_spec: spec({ schema_version: 2 }) }), /schema_version must be 1/],
    ['empty family_id', famArgs({ family_spec: spec({ family_id: ' ' }) }), /family_id/],
    ['missing operator', famArgs({ family_spec: spec({ operator: '' }) }), /operator must be/],
    ['empty configs', famArgs({ family_spec: spec({ configs: [] }) }), /configs must be a non-empty array/],
    ['duplicate config id', famArgs({ family_spec: spec({ configs: [cfg('a'), cfg('a')] }) }), /duplicate config id 'a'/],
    ['unsafe config id', famArgs({ family_spec: spec({ configs: [cfg('../x')] }) }), /configs\[0\]\.id/],
    ['relative kernel_path', famArgs({ family_spec: spec({ configs: [cfg('a', { kernel_path: 'tasks/a' })] }) }), /absolute path/],
    ['missing op_spec', famArgs({ family_spec: spec({ configs: [cfg('a', { op_spec: undefined })] }) }), /op_spec must be an object/],
    ['mode=bakeoff in a config', famArgs({ family_spec: spec({ configs: [cfg('a', { overrides: { mode: 'bakeoff' } })] }) }), /mode must be optimize or author/],
    ['no target_language anywhere', famArgs({ family_spec: spec({ defaults: {}, configs: [cfg('a')] }) }), /target_language must resolve/],
    ['budget 0', famArgs({ family_spec: spec({ defaults: { target_language: 'hip', budget: 0 } }) }), /budget must be a positive integer/],
    ['budget 1.5', famArgs({ family_spec: spec({ defaults: { target_language: 'hip', budget: 1.5 } }) }), /budget must be a positive integer/],
    ['bad gpu_mode', famArgs({ family_spec: spec({ defaults: { target_language: 'hip', gpu_mode: 'shared' } }) }), /gpu_mode must be one of/],
    ['apply_to_original=true in overrides', famArgs({ family_spec: spec({ configs: [cfg('a', { overrides: { apply_to_original: 'true' } })] }) }), /apply_to_original must be false/],
    ['unknown override field', famArgs({ family_spec: spec({ configs: [cfg('a', { overrides: { kernel_path: '/x' } })] }) }), /overrides\.kernel_path is not a supported field/],
    ['signature rank vs axes', famArgs({ family_spec: spec({ configs: [cfg('a', { signature: [1, 2] })] }) }), /signature rank 2 does not match deployment\.signature_axes/],
    ['signature rank across configs', famArgs({ family_spec: spec({ deployment: {}, configs: [cfg('a', { signature: [1, 2] }), cfg('b', { signature: [1, 2, 3] })] }) }), /does not match the other configs/],
    ['non-integer signature', famArgs({ family_spec: spec({ configs: [cfg('a', { signature: [1, 2.5, 3] })] }) }), /signature must be/],
    ['signature outside its range', famArgs({ family_spec: spec({ configs: [cfg('a', { signature: [1, 2, 3], signature_ranges: { 0: [2, 8] } })] }) }), /outside signature_ranges/],
    ['same signature and contract twice', famArgs({ family_spec: spec({ configs: [cfg('a', { signature: [1, 2, 3] }), cfg('b', { signature: [1, 2, 3] })] }) }), /same signature and contract as config 'a'/],
    ['negative workload_weight', famArgs({ family_spec: spec({ configs: [cfg('a', { workload_weight: -1 })] }) }), /workload_weight/],
    ['non-object contract.graph', famArgs({ family_spec: spec({ configs: [cfg('a', { contract: { graph: 'x' } })] }) }), /contract\.graph must be an object/],
    ['parent mode=bakeoff', famArgs({ mode: 'bakeoff' }), /does not support mode=bakeoff/],
    ['parent unknown mode', famArgs({ mode: 'auto' }), /unknown mode='auto' with config_aware/],
    ['parent apply_to_original=true', famArgs({ apply_to_original: 'true' }), /apply_to_original must be "false"/],
    ['duplicate gpu id', famArgs({ gpu_ids: '0,0' }), /lists a GPU twice/],
    ['relative eval_dir', famArgs({ eval_dir: 'exp/x' }), /eval_dir must be an absolute path/],
    ['gpus_per_job=2', famArgs({ gpus_per_job: 2 }), /gpus_per_job > 1/],
    ['lane script is the dispatcher', famArgs({ kernel_lane_script: `${WF_DIR}/kernel_workflow.js`.replace(/\\/g, '/') }), /must run kernel_lane\.js/],
    ['target pin conflicts with parent', famArgs({ expected_gfx: 'gfx942', expected_target: 'unknown', family_spec: spec({ target: { expected_gfx: 'gfx950', expected_target: 'unknown' } }) }), /conflict with family_spec\.target/],
  ];
  for (const [label, a, re] of BAD) {
    const { msg, trace } = await throwsOf(a, { agent: writerOk });
    ok(re.test(msg || '') && trace.agentCalls.length === 0 && trace.workflowCalls.length === 0,
      `${label}: rejected before any agent or lane`, msg || 'no throw');
  }
  {
    const s = spec({ configs: [cfg('a', { kernel_path: 'x' }), cfg('a', { op_spec: null })] });
    const { msg } = await throwsOf(famArgs({ family_spec: s }));
    ok(/\(3 problem\(s\)\)/.test(msg || ''), 'every problem is reported in one error', msg);
  }
  {
    const s = spec({ configs: [
      cfg('g8k', { signature: [1, 2048, 1024], contract: { graph: { maximum_context: 8192 } } }),
      cfg('g16k', { signature: [1, 2048, 1024], contract: { graph: { maximum_context: 16384 } } }),
    ] });
    const { msg } = await throwsOf(famArgs({ family_spec: s }), { agent: writerOk });
    ok(msg === null, 'same signature with different graph contracts are two valid configs', msg);
  }

  console.log('\n# C. planner validation and fallback');
  const goodPlan = {
    groups: [
      { group_id: 'decode', config_ids: ['m1', 'm2'], strategy: 'packed GEMV', reasoning: 'tiny M',
        implementation_form: 'per_signature', confidence: 0.7, config_hints: { m1: 'one wave per column', zz: 'drop me' } },
      { group_id: 'prefill', config_ids: ['prefill'], strategy: 'MFMA tiles', implementation_form: 'bogus', confidence: 3 },
    ],
    execution_order: ['prefill', 'm2', 'm1'],
  };
  {
    const { run, trace } = build(famArgs(), { agent: withPlanner(goodPlan) });
    const r = await run();
    ok(r.plan.source === 'planner' && r.plan.groups.length === 2, 'a valid plan is used', JSON.stringify(r.plan.groups.map((g) => g.group_id)));
    ok(r.plan.groups[0].config_hints.m1 && !('zz' in r.plan.groups[0].config_hints), 'hints for unknown ids are dropped');
    ok(r.plan.groups[1].implementation_form === null && r.plan.groups[1].confidence === null,
      'an unknown implementation_form and an out-of-range confidence are nulled');
    ok(trace.workflowCalls.length === 3, 'a group of two still runs one lane per config', `n=${trace.workflowCalls.length}`);
    const t = laneFor(trace, 'm1').args.task;
    ok(/group=decode/.test(t) && /packed GEMV/.test(t) && /one wave per column/.test(t) && /may drop this hint/.test(t),
      'the lane task carries group, strategy, hint and the measurements-override reminder', t);
    ok(/siblings[^\n]*m2\[2, 2048, 1024\][^\n]*prefill/.test(t), 'the lane task lists sibling configs', t);
    ok(r.configs.map((c) => c.config_id).join(',') === 'prefill,m2,m1', 'results follow the planner execution_order');
  }
  const BAD_PLANS = [
    ['planner returned nothing', null, /no object/],
    ['no groups', { groups: [] }, /no groups/],
    ['omitted config', { groups: [{ group_id: 'g', config_ids: ['m1', 'm2'] }] }, /omitted config\(s\): prefill/],
    ['duplicate assignment', { groups: [{ group_id: 'g', config_ids: ['m1', 'm2'] }, { group_id: 'h', config_ids: ['m2', 'prefill'] }] }, /assigned to both/],
    ['unknown config', { groups: [{ group_id: 'g', config_ids: ['m1', 'm2', 'prefill', 'm9'] }] }, /unknown config 'm9'/],
    ['empty group id', { groups: [{ group_id: '', config_ids: ['m1', 'm2', 'prefill'] }] }, /group_id is empty/],
    ['duplicate group id', { groups: [{ group_id: 'g', config_ids: ['m1'] }, { group_id: 'g', config_ids: ['m2', 'prefill'] }] }, /duplicate group_id/],
  ];
  for (const [label, plan, re] of BAD_PLANS) {
    const { run, trace } = build(famArgs(), { agent: withPlanner(plan) });
    const r = await run();
    ok(r.plan.source === 'fallback' && re.test(r.plan.fallback_reason || '') &&
       r.plan.groups.length === 3 && r.plan.groups.every((g) => g.config_ids.length === 1 && g.config_ids[0] === g.group_id) &&
       trace.workflowCalls.length === 3,
      `${label} -> singleton groups, every config still runs`, r.plan.fallback_reason);
  }
  {
    const s = spec({ configs: [
      cfg('a', { contract: { semantics: { precision: 'bf16' } } }),
      cfg('b', { contract: { semantics: { precision: 'fp8' } } }),
    ] });
    const { run } = build(famArgs({ family_spec: s }), { agent: withPlanner({ groups: [{ group_id: 'g', config_ids: ['a', 'b'] }] }) });
    const r = await run();
    ok(r.plan.source === 'fallback' && /mixes configs with different contract\.semantics/.test(r.plan.fallback_reason),
      'a group mixing semantic contracts is rejected', r.plan.fallback_reason);
  }
  {
    const s = spec({ configs: [
      cfg('a', { contract: { graph: { maximum_context: 8192, capture_safe: true } } }),
      cfg('b', { contract: { graph: { capture_safe: true, maximum_context: 8192 } } }),
    ] });
    const { run } = build(famArgs({ family_spec: s }), { agent: withPlanner({ groups: [{ group_id: 'g', config_ids: ['a', 'b'] }] }) });
    const r = await run();
    ok(r.plan.source === 'planner', 'equal contracts written in a different key order may share a group', r.plan.fallback_reason);
  }
  {
    const plan = { ...goodPlan, execution_order: ['m1', 'm1', 'prefill'] };
    const { run } = build(famArgs(), { agent: withPlanner(plan) });
    const r = await run();
    ok(r.plan.source === 'planner' && r.plan.order_source === 'default', 'a bad execution_order keeps the groups and uses the default order');
  }
  {
    const s = spec({ configs: [cfg('a', { priority: 3 }), cfg('b'), cfg('c', { priority: 1 }), cfg('d', { priority: 3 })] });
    const { run } = build(famArgs({ family_spec: s }), { agent: writerOk });
    const r = await run();
    ok(r.plan.execution_order.join(',') === 'c,a,d,b', 'default order: priority ascending, then input order, unprioritized last',
      r.plan.execution_order.join(','));
  }

  console.log('\n# D. lane arguments');
  {
    const s = spec();
    s.defaults = { mode: 'optimize', target_language: 'Triton', budget: '3', use_learned_kb: 'false', apply_to_original: 'false', gpu_ids: '7' };
    s.configs[1].overrides = { budget: 9, warm_start: 'OFF', task: 'focus on the epilogue' };
    s.configs.push(cfg('nosig', { overrides: { target_language: 'hip' } }));
    const { run, trace } = build(famArgs({
      family_spec: s, gpu_ids: '0,1,2,3', kb_mode: 'local', use_expert_skills: 'true',
      target_language: 'ck', budget: 99, routing: 'on',
    }), { agent: writerOk });
    await run();
    const [m1, m2, pf, ns] = ['m1', 'm2', 'prefill', 'nosig'].map((id) => laneFor(trace, id).args);
    ok(m1.target_language === 'triton' && m1.budget === 3 && m1.mode === 'optimize', 'm1 inherits defaults (language lowercased, budget parsed)',
      JSON.stringify([m1.target_language, m1.budget, m1.mode]));
    ok(m2.budget === 9 && m2.warm_start === 'off' && /caller task: focus on the epilogue/.test(m2.task), 'm2 overrides beat defaults');
    ok(pf.mode === 'author' && pf.target_language === 'flydsl' && pf.budget === 2, 'prefill override mode/language/budget');
    ok(ns.target_language === 'hip' && ns.budget === 3, 'a config without signature still runs with its own language');
    ok([m1, m2, pf, ns].every((a) => a.use_learned_kb === 'false'), 'use_learned_kb comes from defaults');
    ok(m1.warm_start === undefined && m1.update_experience === undefined, 'unset KB switches stay absent (lane default)');
    ok([m1, m2, pf, ns].every((a) => a.kb_mode === 'local' && a.use_expert_skills === 'true'), 'recognized parent knobs are forwarded');
    ok([m1, m2, pf, ns].every((a) => a.routing === undefined && a.family_spec === undefined && a.config_aware === undefined),
      'unrecognized parent args are not forwarded');
    ok(trace.logs.some((l) => /parent .*target_language.*budget.* ignored/.test(l)), 'ignored parent args are logged');
    ok([m1, m2, pf, ns].every((a) => a.apply_to_original === 'false'), 'apply_to_original is "false" on every lane');
    ok([m1, m2, pf, ns].every((a) => /^\d$/.test(a.gpu_ids)), 'every lane gets exactly one GPU id',
      JSON.stringify([m1, m2, pf, ns].map((a) => a.gpu_ids)));
    ok(new Set([m1, m2, pf, ns].map((a) => a.eval_dir)).size === 4 &&
       [m1, m2, pf, ns].every((a, i) => a.eval_dir === `${FAM_DIR}/configs/${['m1', 'm2', 'prefill', 'nosig'][i]}`),
      'every lane gets ${family_eval_dir}/configs/<id>');
    ok(trace.workflowCalls.every((c) => /kernel_lane\.js$/.test(c.scriptPath)), 'the child script is kernel_lane.js');
    ok([m1, m2, pf, ns].every((a) => a.kernel_path === `/tasks/${['m1', 'm2', 'prefill', 'nosig'][[m1, m2, pf, ns].indexOf(a)]}`),
      'each lane gets its own config kernel_path');
    ok(m1.workflow_dir === WF_DIR, 'workflow_dir is forwarded');
  }
  {
    const s = spec();
    s.defaults.gpu_ids = '5';
    const { run, trace } = build({ config_aware: 'on', workflow_dir: WF_DIR, eval_dir: FAM_DIR, family_spec: s }, { agent: writerOk });
    await run();
    ok(trace.workflowCalls.every((c) => c.args.gpu_ids === '5'), 'without parent gpu_ids the pool comes from defaults.gpu_ids');
  }
  {
    const { run, trace } = build(famArgs({ family_spec: spec({ target: { expected_gfx: 'GFX950', expected_target: 'unknown' } }) }), { agent: writerOk });
    await run();
    ok(trace.workflowCalls.every((c) => c.args.expected_gfx === 'gfx950' && c.args.expected_target === 'unknown'),
      'family_spec.target is forwarded to every lane as the expected identity');
  }

  console.log('\n# E. concurrency');
  {
    const gates = [];
    const { run, trace } = build(famArgs({ gpu_ids: '0,1,2' }), {
      agent: writerOk,
      lane: () => new Promise((resolve) => gates.push(() => resolve({ validation_status: 'accepted', final_speedup: 1.2, final_patch: '/p' }))),
    });
    const p = run();
    for (let i = 0; i < 20 && gates.length < 3; i++) await new Promise((r) => setImmediate(r));
    ok(gates.length === 3, 'with 3 GPUs all 3 lanes are in flight before any finishes', `started=${gates.length}`);
    ok(new Set(trace.workflowCalls.map((c) => c.args.gpu_ids)).size === 3, 'the 3 lanes hold 3 different GPU ids');
    gates.forEach((f) => f());
    await p;
  }
  {
    const { run, trace } = build(famArgs({ gpu_ids: '0' }), { agent: writerOk,
      lane: async () => { await new Promise((r) => setTimeout(r, 2)); return { validation_status: 'accepted', final_speedup: 1.1, final_patch: '/p' }; } });
    await run();
    ok(trace.maxInFlight === 1 && trace.workflowCalls.length === 3, 'one GPU serializes the lanes', `max=${trace.maxInFlight}`);
  }
  {
    const { run, trace } = build(famArgs({ gpu_ids: '0,1' }), { agent: writerOk,
      lane: async () => { await new Promise((r) => setTimeout(r, 2)); return { validation_status: 'accepted', final_speedup: 1.1, final_patch: '/p' }; } });
    const r = await run();
    ok(trace.maxInFlight === 2, 'concurrency = min(configs, gpu_ids)', `max=${trace.maxInFlight}`);
    ok(r.plan.concurrency === 2, 'the plan records the concurrency');
  }

  console.log('\n# F. failure isolation and status mapping');
  const LANES = {
    m1: () => ({ validation_status: 'accepted', final_speedup: 1.5, final_patch: '/e/m1/final_patch.diff',
      timing_basis: 'clean', workload_aligned: false, eval_dir: '/e/m1', report_path: '/e/m1/tech_lead_report.md',
      warm_start: { adopted: false, read_reason: 'cold' }, kb_written: { slug: 'gemm__triton__gfx950', dir: '/kb' } }),
    m2: () => { throw new Error('lane exploded'); },
    prefill: () => null,
    a1: () => ({ validation_status: 'accepted', final_speedup: 0.9, final_patch: '/p', timing_basis: 'host_bound', workload_aligned: false }),
    a2: () => ({ validation_status: 'flagged', final_speedup: 2.0, final_patch: '/p', reason: 'correctness drift' }),
    a3: () => ({ validation_status: 'author_failed', final_speedup: 0 }),
    a4: () => ({ validation_status: 'no_baseline' }),
    a5: () => ({ validation_status: 'unknown', final_speedup: 1.3 }),
    a6: () => ({ final_speedup: 1.3 }),
    a7: () => ({ validation_status: 'accepted', final_speedup: 'fast' }),
    a8: () => ({ validation_status: 'accepted', final_speedup: 1.4, final_patch: '' }),
  };
  const ids = Object.keys(LANES);
  const mixed = spec({ configs: ids.map((id, i) => cfg(id, { signature: [i + 1, 2048, 1024],
    ...(id === 'a1' ? { incumbent: { source_file: 'gemm_m5.py' } } : {}) })) });
  const idOf = (a) => a.eval_dir.split('/').pop();
  const mixedRun = build(famArgs({ family_spec: mixed, gpu_ids: '0,1,2,3' }), { agent: writerOk, lane: (a) => LANES[idOf(a)]() });
  const R = await mixedRun.run();
  const st = Object.fromEntries(R.configs.map((c) => [c.config_id, c.status]));
  const EXPECT = { m1: 'optimized', m2: 'lane_failed', prefill: 'invalid_result', a1: 'accepted_no_improvement',
    a2: 'flagged', a3: 'author_failed', a4: 'no_baseline', a5: 'not_accepted', a6: 'invalid_result',
    a7: 'invalid_result', a8: 'invalid_result' };
  for (const [id, want] of Object.entries(EXPECT)) ok(st[id] === want, `${id} -> ${want}`, st[id]);
  ok(R.configs.length === ids.length, 'a thrown lane does not drop any sibling result');
  const m1r = R.configs.find((c) => c.config_id === 'm1');
  ok(m1r.speedup_vs_stock === 1.5 && m1r.speedup_vs_incumbent === null && m1r.timing_basis === 'clean' &&
     m1r.metric_kind === 'geomean' && m1r.kb_written.slug === 'gemm__triton__gfx950' && m1r.warm_start.read_reason === 'cold',
    'structured lane fields are preserved');
  ok(R.configs.find((c) => c.config_id === 'a2').lane_validation_status === 'flagged', 'the raw lane status is kept separately');
  ok(/lane exploded/.test(R.configs.find((c) => c.config_id === 'm2').error), 'the thrown error is recorded');
  ok(R.summary.total === 11 && R.summary.optimized === 1 && R.summary.accepted_no_improvement === 1 &&
     R.summary.flagged === 1 && R.summary.failed === 8, 'summary counts', JSON.stringify(R.summary));
  ok(R.validation_status === 'ok', 'validation_status=ok when at least one config is optimized');

  console.log('\n# G. implementation catalog');
  const cat = R.catalog;
  ok(cat.implementations.length === 1 && cat.implementations[0].config_ids[0] === 'm1', 'only optimized configs enter implementations');
  const impl = cat.implementations[0];
  ok(JSON.stringify(impl.coverage) === JSON.stringify({ exact: [[1, 2048, 1024]], ranges: [] }), 'coverage is the exact own signature, no ranges');
  ok(impl.artifact.kind === 'lane_patch' && impl.artifact.path === '/e/m1/final_patch.diff', 'the artifact is the lane patch');
  ok(impl.deployment_ready === false && cat.deployment_ready === false && impl.speedup_vs_incumbent === null &&
     impl.candidate_class === 'optimization_candidate', 'not deployment-ready; no incumbent speedup claimed');
  ok(cat.cross_config_evaluation.status === 'unsupported', 'cross-config evaluation is reported as unsupported');
  const placed = [...cat.implementations.flatMap((i) => i.config_ids), ...cat.fallback.map((f) => f.config_id)];
  ok(placed.length === ids.length && new Set(placed).size === ids.length, 'every config appears exactly once in the catalog');
  ok(cat.fallback.find((f) => f.config_id === 'a1').selected === 'incumbent' &&
     cat.fallback.find((f) => f.config_id === 'a2').selected === 'stock', 'fallback selects incumbent when supplied, else stock');
  ok(/lane threw: lane exploded/.test(cat.fallback.find((f) => f.config_id === 'm2').reason), 'fallback reason names the failure');
  {
    const s = spec({ configs: [cfg('a'), cfg('b')] });
    const { run } = build(famArgs({ family_spec: s }), { agent: writerOk,
      lane: () => ({ validation_status: 'accepted', final_speedup: 1.2, final_patch: '/p', timing_basis: 'clean', workload_aligned: false }) });
    const r = await run();
    ok(r.catalog.implementations.every((i) => i.coverage.exact.length === 0 && i.orchestration_key === i.config_ids[0]),
      'without a signature the config id is the orchestration key and coverage is empty');
  }

  console.log('\n# H. family aggregate');
  ok(R.aggregate.value === null && /timing_basis not comparable/.test(R.aggregate.reason),
    'accepted configs with mixed timing bases (clean vs host_bound) are not aggregated', R.aggregate.reason);
  {
    const s = spec({ configs: [cfg('a'), cfg('b'), cfg('c')] });
    const sp = { a: 2.0, b: 0.5, c: 4.0 };
    const { run } = build(famArgs({ family_spec: s }), { agent: writerOk,
      lane: (a) => idOf(a) === 'c' ? { validation_status: 'flagged', final_speedup: 4.0 }
        : { validation_status: 'accepted', final_speedup: sp[idOf(a)], final_patch: '/p', timing_basis: 'clean', workload_aligned: false } });
    const r = await run();
    ok(Math.abs(r.aggregate.value - 1.0) < 1e-12 && r.aggregate.timing_basis === 'clean' && r.aggregate.metric === 'geomean',
      'geomean over accepted configs (2.0, 0.5) = 1.0', JSON.stringify(r.aggregate));
    ok(r.aggregate.complete === false && r.aggregate.excluded.length === 1 && r.aggregate.excluded[0].config_id === 'c' &&
       /partial family/.test(r.aggregate.reason), 'the excluded config and the partial status are reported');
  }
  {
    const s = spec({ configs: [cfg('a'), cfg('b')] });
    const { run } = build(famArgs({ family_spec: s }), { agent: writerOk,
      lane: (a) => ({ validation_status: 'accepted', final_speedup: 1.5, final_patch: '/p', timing_basis: 'clean', workload_aligned: idOf(a) === 'a' }) });
    const r = await run();
    ok(r.aggregate.value === null && /metric kind not comparable/.test(r.aggregate.reason), 'weighted and geomean results are not mixed');
  }
  {
    const s = spec({ configs: [cfg('a'), cfg('b')] });
    const { run } = build(famArgs({ family_spec: s }), { agent: writerOk,
      lane: () => ({ validation_status: 'accepted', final_speedup: 1.5, final_patch: '/p', timing_basis: 'unknown', workload_aligned: false }) });
    const r = await run();
    ok(r.aggregate.value === null, 'an unknown timing basis is never aggregated');
  }

  console.log('\n# I. persisted artifacts are the returned objects');
  {
    const files = persisted(mixedRun.trace);
    const P = (n) => `${FAM_DIR}/${n}`;
    ok(R.plan_path === P('family_plan.json') && R.results_path === P('family_results.json') &&
       R.report_path === P('family_report.md') && R.catalog_path === P('implementation_catalog.json'), 'artifact paths');
    ok(files[P('family_plan.json')] && JSON.stringify(JSON.parse(files[P('family_plan.json')])) === JSON.stringify(R.plan), 'family_plan.json == returned plan');
    const res = files[P('family_results.json')] && JSON.parse(files[P('family_results.json')]);
    ok(res && JSON.stringify(res.configs) === JSON.stringify(R.configs) && JSON.stringify(res.summary) === JSON.stringify(R.summary) &&
       JSON.stringify(res.aggregate) === JSON.stringify(R.aggregate), 'family_results.json == returned configs/summary/aggregate');
    ok(files[P('implementation_catalog.json')] && JSON.stringify(JSON.parse(files[P('implementation_catalog.json')])) === JSON.stringify(R.catalog),
      'implementation_catalog.json == returned catalog');
    const md = files[P('family_report.md')] || '';
    ok(/# Operator family report: gemm_n2048_k1024/.test(md) && /\| m2 \| lane_failed \|/.test(md) && /## Failed configs/.test(md),
      'family_report.md lists every config and its status');
    ok(R.artifacts_persisted === true, 'artifacts_persisted=true when the writer confirms every path');
  }
  {
    const { run } = build(famArgs(), { agent: (label) => label === 'file_writer:family_artifacts' ? { written: [] } : null });
    const r = await run();
    ok(r.artifacts_persisted === false && r.configs.length === 3, 'a failed write is reported, results are still returned');
  }

  console.log('\n# J. family eval dir');
  {
    const made = '/root/exp/family_gemm_n2048_k1024_20261123T000000Z_42';
    const { run, trace } = build(famArgs({ eval_dir: undefined, exp_root: '/root/exp' }), {
      agent: (label, prompt) => label === 'file_writer:family_setup' ? { eval_dir: made + '/' } : writerOk(label, prompt) });
    const r = await run();
    ok(trace.agentCalls[0].label === 'file_writer:family_setup', 'without eval_dir the setup agent runs first');
    ok(r.eval_dir === made && trace.workflowCalls.every((c) => c.args.eval_dir.startsWith(`${made}/configs/`)),
      'lanes use the created family dir', r.eval_dir);
  }
  for (const [label, answer] of [['outside exp_root', { eval_dir: '/tmp/elsewhere' }], ['no answer', null],
    ['bare prefix', { eval_dir: '/root/exp/family_gemm_n2048_k1024_' }]]) {
    const { run, trace } = build(famArgs({ eval_dir: undefined, exp_root: '/root/exp' }), {
      agent: (l) => l === 'file_writer:family_setup' ? answer : null });
    const r = await run();
    ok(r.validation_status === 'setup_failed' && trace.workflowCalls.length === 0, `setup ${label} -> setup_failed, no lane`);
  }
  {
    const { trace } = await (async () => { const b = build(famArgs(), { agent: writerOk }); await b.run(); return b; })();
    ok(!trace.agentCalls.some((c) => c.label === 'file_writer:family_setup'), 'a pinned eval_dir is honored without a setup agent');
  }

  console.log(failures === 0
    ? '\nPASS: config_aware operator-family path behaves as specified.'
    : `\nFAILED: ${failures} assertion(s).`);
  process.exit(failures === 0 ? 0 : 1);
})();
