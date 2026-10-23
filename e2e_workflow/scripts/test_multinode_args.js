#!/usr/bin/env node
// args.multinode -> topology, the caller's mn.py check verdict and the SERVING env from the parsed spec,
// EXTRACTED from the shipped e2e_workflow.js.
//
// Run:  node e2e_workflow/scripts/test_multinode_args.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const eq = (got, want, msg) => {
  const a = JSON.stringify(got), b = JSON.stringify(want);
  if (a !== b) { console.error('  FAIL:', msg, '\n    got: ', a, '\n    want:', b); failures++; }
  else console.log('  ok:', msg);
};

const mStart = src.indexOf('const SERVING_TP =');
const mEnd = src.indexOf('// ---- WALL-CLOCK BUDGET', mStart);
if (mStart < 0 || mEnd < 0) { console.error('FAIL: serving-config block not found in e2e_workflow.js'); process.exit(1); }
// The Workflow sandbox has no `require`; the block must load without one.
const load = (A, gpuIds = '0') => new Function('A', 'GPU_IDS', 'GPU_LIST', 'WORKFLOW_DIR',
  `${src.slice(mStart, mEnd)}\nreturn { SERVING_TOPOLOGY, SERVING_GPU, MN, MN_PY, useMnSpec,
    vars: () => ({ MN_SPEC, MN_STATE, MN_SERVING_ENV, MN_GPU_EXEC }) };`)(
  A, gpuIds, gpuIds.split(','), '/w/e2e_workflow');

const IPS = ['10.10.0.166', '10.10.0.32'];
const mn = (over = {}) => ({ nodes: IPS, shared_fs: '/shared/geak/', ...over });
const SPEC = { topology: 'pd_disagg', nodes: [{ host: 'p1' }, { host: 'd1' }], shared_fs: '/shared/geak',
  state: '/shared/geak/r1/mn_state.json', tp_by_role: { prefill: 8, decode: 4 } };

console.log('single host');
{
  const r = load({ tp: 8 }, '0,1,2,3,4,5,6,7');
  eq([r.MN, r.SERVING_TOPOLOGY, r.vars().MN_SERVING_ENV, r.vars().MN_GPU_EXEC], [null, '', '', ''],
    'no multinode => nothing set');
  eq(r.SERVING_GPU, '0,1,2,3,4,5,6,7', 'GPU set is the first TP ids, as before');
  eq(load({ tp: 8, multinode: mn({ nodes: IPS.slice(0, 1) }) }).SERVING_TOPOLOGY, 'tp_span',
    'one named node is a single-machine server reached through the pin');
}

console.log('the block before the check');
{
  const r = load({ tp: 16, multinode: mn() }, '0,1,2,3,4,5,6,7');
  eq(r.SERVING_TOPOLOGY, 'tp_span', 'topology defaults to tp_span');
  eq(r.SERVING_GPU, '0,1,2,3,4,5,6,7', 'GPU is rank0\'s gpu_ids verbatim; the pin holds each rank\'s cards');
  eq([r.vars().MN_SPEC, r.vars().MN_SERVING_ENV], [null, ''], 'nothing is pinned until mn.py check passes');
  eq(r.MN_PY, '/w/e2e_workflow/scripts/multi_node/mn.py', 'mn.py from this workflow tree');
  eq(load({ tp: 16, serving_gpu: '0,1', multinode: mn() }).SERVING_GPU, '0,1', 'serving_gpu always wins');
  eq(load({ tp: 8, multinode: mn({ topology: 'pd_disagg' }) }).SERVING_TOPOLOGY, 'pd_disagg', 'pd_disagg is read');
}

console.log('useMnSpec');
{
  const r = load({ tp: 8, multinode: mn({ topology: 'pd_disagg' }) });
  r.useMnSpec(SPEC);
  const v = r.vars();
  eq([v.MN_STATE, v.MN_SPEC], ['/shared/geak/r1/mn_state.json', SPEC],
    'the pin and every other multinode field come from the parsed spec');
  eq(v.MN_SERVING_ENV, 'BENCH_LAUNCHER=multinode GEAK_MN_STATE=/shared/geak/r1/mn_state.json',
    'SERVING env carries the launcher and the pin only');
  eq(v.MN_GPU_EXEC, 'GEAK_MN_STATE=/shared/geak/r1/mn_state.json python3 /w/e2e_workflow/scripts/multi_node/mn.py exec',
    'an explicit node is reached through mn.py exec against the pin');
}

console.log('phase-by-phase resume');
{
  const r = load({ tp: 8, phases: 'head', state: { multinode: SPEC } });
  eq([r.SERVING_TOPOLOGY, r.MN], ['pd_disagg', SPEC], 'a resume without args.multinode re-checks the carried spec');
  eq(load({ tp: 16, multinode: mn(), state: { multinode: SPEC } }).SERVING_TOPOLOGY, 'tp_span',
    'args.multinode wins over the carried block');
  eq(/const carryState = \{\s*backend: BACKEND,\s*\.\.\.\(MN_SPEC \? \{ multinode: MN_SPEC \} : \{\}\),/.test(src), true,
    'the returned state carries the parsed spec');
}

console.log('check + Setup wiring');
const checkAt = src.indexOf('let verdict = A.multinode_check || await agentMnCheck();');
const setupAt = src.indexOf("if (want('setup')) {");
eq(checkAt > 0 && checkAt < setupAt, true, 'every invocation has a check verdict before Setup');
eq(src.slice(checkAt, setupAt).includes('useMnSpec(verdict.spec)'), true, 'a passed check hands its spec to useMnSpec');
eq(/multinode_check: \{ verdict: 'refused', problems/.test(src.slice(checkAt, setupAt)), true,
  'a refusal returns before Setup, with the problems');

const verdictTests = (async () => {
  console.log('the verdict: the caller\'s, else one agent runs mn.py check');
  const cStart = src.indexOf('async function agentMnCheck()');
  const cEnd = src.indexOf('// The envelope arrives in the reply', cStart);
  const agentRun = src.slice(src.indexOf('async function agentRun('), src.indexOf('async function withDiskSelection('));
  const run = async (A, reply, over = {}) => {
    const calls = [], used = [];
    const safeAgent = async (prompt, opts) => { calls.push({ prompt, opts }); return { stdout: reply }; };
    const ret = await new Function('A', 'MN', 'SERVING_TP', 'ST', 'EVAL_DIR_OVERRIDE', 'EXP_ROOT', 'MN_PY', 'shq',
      'safeAgent', 'obj', 'useMnSpec', 'log', 'MN_SPEC', 'MN_STATE', 'MN_SERVING_ENV',
      `${agentRun}\nreturn (async () => { ${src.slice(cStart, cEnd)}\nreturn 'passed'; })();`)(
      A, A.multinode, 8, over.ST || {}, over.EVAL_DIR_OVERRIDE || '', '/shared/geak/exp', '/w/mn.py',
      (s) => `'${s}'`, safeAgent, () => ({}), (s) => used.push(s), () => {}, SPEC, '', '');
    return { ret, calls, used };
  };
  const A = { tp: 8, multinode: mn({ topology: 'pd_disagg' }) };
  const ok = (hosts) => 'MN_CHECK=' + JSON.stringify({ verdict: 'ok', spec: { ...SPEC, nodes: hosts.map((host) => ({ host })) } });

  let r = await run({ ...A, multinode_check: { verdict: 'ok', spec: SPEC } }, '');
  eq([r.ret, r.calls.length, r.used.length], ['passed', 0, 1], 'the caller\'s verdict is used as is, no agent');

  r = await run(A, ok(IPS.slice().reverse()), { EVAL_DIR_OVERRIDE: '/shared/geak/exp/e1' });
  eq([r.ret, r.calls.length, r.used.length], ['passed', 1, 1], 'no verdict passed: one agent runs the check');
  eq(r.calls[0].prompt.includes("python3 /w/mn.py check --owner claude | grep '^MN_CHECK=' | tail -n 1"), true,
    'the agent runs mn.py check with the Claude Code process as the pin owner');
  eq(r.calls[0].prompt.includes('"work_dir":"/shared/geak/exp/e1"') && r.calls[0].prompt.includes('"tp":8'), true,
    'the spec carries the run directory and the engine TP, as run_e2e.py\'s does');
  eq((await run(A, ok(IPS))).calls[0].prompt.includes('"work_dir":"/shared/geak/exp"'), true,
    'before Setup names an eval dir, the tools go under exp_root');

  r = await run(A, ok(['10.9.9.9', IPS[1]]));
  eq([r.ret.multinode_check.verdict, r.used.length], ['refused', 0], 'a relayed verdict about other machines is refused');
  r = await run(A, 'MN_CHECK={"verdict":"ok"');
  eq(/printed no verdict line/.test(r.ret.error), true, 'a garbled line is refused');
  r = await run(A, 'MN_CHECK=' + JSON.stringify({ verdict: 'refused', problems: ['shared_fs /x is not a directory here'] }));
  eq(r.ret.multinode_check.problems, ['shared_fs /x is not a directory here'], 'the check\'s own refusal is passed on');
})();
eq(/runMn|\[MN_PY, \.\.\.args\]/.test(src), false, 'the workflow spawns no mn.py itself');
eq(/BACKEND=\$\{BACKEND\} TP=\$\{SERVING_TP\} GPU=\$\{SERVING_GPU\}\$\{MN \? ` \$\{MN_SERVING_ENV\}` : ''\}/.test(src), true,
  'every role\'s SERVING env line appends the multinode env');
eq(src.includes('CAPTURE_TP: ((MN_SPEC && MN_SPEC.tp_by_role) || {})[role] || SERVING_TP'), true,
  'a PD capture shard uses its own engine\'s TP');
eq(src.includes('${MN_GPU_EXEC} --target <master|rank<N>|prefill|decode|host> [--gpu <id>] -- <command>'), true,
  'every role is told how to reach a node explicitly');
eq(/MN_CHECK_CMD|MN_CHECK_SCHEMA|GEAK_REMOTE_GPU|GEAK_MN_EXT_SHARED_FS/.test(src), false, 'no trace of the old check agent');

verdictTests.then(() => {
  if (failures) { console.error(`\n${failures} FAILED`); process.exit(1); }
  console.log('\nall passed');
}, (e) => { console.error('FAIL:', e); process.exit(1); });
