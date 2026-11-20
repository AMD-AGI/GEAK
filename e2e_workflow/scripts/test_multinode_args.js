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
const checkAt = src.indexOf('const verdict = A.multinode_check;');
const setupAt = src.indexOf("if (want('setup')) {");
eq(checkAt > 0 && checkAt < setupAt, true, 'every invocation has a check verdict before Setup');
eq(src.slice(checkAt, setupAt).includes('useMnSpec(verdict.spec)'), true, 'a passed check hands its spec to useMnSpec');
eq(/multinode_check: \{ verdict: 'refused', problems/.test(src.slice(checkAt, setupAt)), true,
  'a refusal returns before Setup, with the problems');

const verdictTests = (async () => {
  console.log('the verdict: run_e2e.py\'s mn.py check, relayed');
  const cStart = src.indexOf('const mnHosts =');
  const cEnd = src.indexOf('// The envelope arrives in the reply', cStart);
  const run = (A) => {
    const used = [], logs = [];
    const ret = new Function('A', 'MN', 'useMnSpec', 'log', 'MN_SPEC', 'MN_STATE', 'IS_PD', 'PD_ROLES',
      `return (() => { ${src.slice(cStart, cEnd)}\nreturn 'passed'; })();`)(
      A, A.multinode, (s) => used.push(s), (m) => logs.push(m), SPEC, '',
      A.multinode.topology === 'pd_disagg', ['prefill', 'decode']);
    return { ret, used, logs };
  };
  const A = { tp: 8, multinode: mn({ topology: 'pd_disagg' }) };
  const at = (hosts) => ({ ...SPEC, nodes: hosts.map((host) => ({ host })) });

  let r = run({ ...A, multinode_check: { verdict: 'ok', spec: at(IPS) } });
  eq([r.ret, r.used.length], ['passed', 1], 'a complete relayed verdict is used as is');
  const pinned = { tp: 8, multinode: mn({ topology: 'pd_disagg', commands_in_pin: true }) };
  r = run({ ...pinned, multinode_check: { verdict: 'ok', spec: { ...at(IPS), commands_in_pin: true } } });
  eq([r.ret, r.used.length], ['passed', 1], "run_e2e.py's command-less verdict is used as is");

  const drop = (k) => { const s = at(IPS); delete s[k]; return s; };
  for (const [why, bad, shown] of [
    ['no verdict', undefined, 'no verdict'],
    ['no spec', { verdict: 'ok' }, 'no spec'],
    ['no spec.state', { verdict: 'ok', spec: drop('state') }, 'no spec.state'],
    ['PD without tp_by_role', { verdict: 'ok', spec: drop('tp_by_role') }, 'spec.tp_by_role'],
    ['other machines', { verdict: 'ok', spec: at(['10.9.9.9', IPS[1]]) }, 'are not args.multinode'],
  ]) {
    r = run({ ...A, multinode_check: bad });
    eq([r.ret.multinode_check.verdict, r.used.length, r.ret.error.includes(shown), /through run_e2e\.py/.test(r.ret.error)],
      ['refused', 0, true, true], `${why}: refused before Setup, nothing re-runs the check`);
  }
  r = run({ ...A, multinode_check: { verdict: 'refused', problems: ['rank1: GPUs busy'] } });
  eq([r.ret.multinode_check.verdict, r.ret.multinode_check.problems, r.used.length],
    ['refused', ['rank1: GPUs busy'], 0], 'a relayed refusal stands with its own problems');
  r = run({ ...A, multinode_check: { verdict: 'refused' } });
  eq(r.ret.multinode_check.problems, ['mn.py check refused'], 'a refusal without problems still says so');
  eq(run({ tp: 16, multinode: mn(), multinode_check: { verdict: 'ok', spec: drop('tp_by_role') } }).ret, 'passed',
    'tp_span needs no per-engine TP');
})();
eq(/runMn|\[MN_PY, \.\.\.args\]/.test(src), false, 'the workflow spawns no mn.py itself');
eq(/BACKEND=\$\{BACKEND\} TP=\$\{SERVING_TP\} GPU=\$\{SERVING_GPU\}\$\{MN \? ` \$\{MN_SERVING_ENV\}` : ''\}/.test(src), true,
  'every role\'s SERVING env line appends the multinode env');
eq(src.includes('CAPTURE_TP: SERVING_TP, CAPTURE_GPU: SERVING_GPU'), true,
  'a capture leg is a serving leg: each PD engine keeps the TP its pin records');
eq(src.includes("${MN ? `MULTINODE: also Read ${WORKFLOW_DIR}/roles/_fragments/multinode_driver.md and follow it.\\n` : ''}"), true,
  'every role reads the multinode driver rules');
eq(/MN_SPEC \? \{ MN_NODES: MN_SPEC\.nodes\.length, MN_STATE, MN_SHARED_FS: MN_SPEC\.shared_fs, MN_GPU_EXEC,/.test(src), true,
  'every role gets the pin, the shared fs and the exec command as Inputs');
const driverMd = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'roles', '_fragments', 'multinode_driver.md'), 'utf8');
eq(['<MN_GPU_EXEC> --target <master|rank<N>|prefill|decode|host> [--gpu <id>] -- <command>', 'sees NO GPU',
  'never copy a `GEAK_MN_PD_*` value into them'].every((s) => driverMd.includes(s)), true,
  'the fragment tells every role how to reach a node and keeps the PD env rule');
eq(/MN_CHECK_CMD|MN_CHECK_SCHEMA|GEAK_REMOTE_GPU|GEAK_MN_EXT_SHARED_FS|agentMnCheck|agentRun|--env-pin/.test(src), false,
  'no check agent: only run_e2e.py runs mn.py check');

verdictTests.then(() => {
  if (failures) { console.error(`\n${failures} FAILED`); process.exit(1); }
  console.log('\nall passed');
}, (e) => { console.error('FAIL:', e); process.exit(1); });
