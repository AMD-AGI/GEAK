#!/usr/bin/env node
// The PD helpers of e2e_workflow.js (directions guard, engine capture, per-engine variables),
// EXTRACTED from the shipped source.
//
// Run:  node e2e_workflow/scripts/test_pd_guards.js
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

const PD_CONSTS = "const PD_ROLES = ['prefill', 'decode'], PD_KINDS = ['flags', 'env'];";
const mStart = src.indexOf('// The one PD_KINDS x PD_ROLES walk');
const pdEnd = src.indexOf('function roleAgent(', mStart);
if (!src.includes(PD_CONSTS) || mStart < 0 || pdEnd < 0) {
  console.error('FAIL: PD helper block not found in e2e_workflow.js'); process.exit(1);
}
const PD_BLOCK = `${PD_CONSTS}\n${src.slice(mStart, pdEnd)}`;
const load = (topology, prof) => new Function('SERVING_TOPOLOGY', 'IS_PD', 'profile', 'log',
  `${PD_BLOCK}\nreturn { applyPdDirectionGuard, pdOut, pdVar, pdRole, pdNew, engineGpu };`)(
  topology, topology === 'pd_disagg', prof, () => {});

const BY_ROLE = { profile_topN_by_role: { prefill: '/p.json', decode: '/d.json' } };

// ------------------------------------------------------------------ A. directions
console.log('A. directions');
{
  const { applyPdDirectionGuard } = load('pd_disagg', BY_ROLE);
  const strat = { config_directions: [
    { axis: 'chunked-prefill-size', engine: 'prefill' },
    { axis: 'attention-backend', engine: 'decode' },
    { axis: 'kv-cache-dtype', engine: '' },
  ] };
  eq(applyPdDirectionGuard(strat, 't'), [], 'both engines covered -> no gap');
  const legacy = { config_directions: [{ axis: 'chunk', engine: 'prefill' }, { axis: 'x', role: 'decode' }] };
  eq(applyPdDirectionGuard(legacy, 't'), ['decode'], 'a row\'s `role` is not an engine: only `engine` is read');
}
{
  const { applyPdDirectionGuard } = load('pd_disagg', BY_ROLE);
  eq(applyPdDirectionGuard({ config_directions: [{ axis: 'chunk', engine: 'prefill' }] }, 't'),
    ['decode'], 'a profiled engine with no direction is named');
}
{
  const { applyPdDirectionGuard } = load('pd_disagg', { profile_topN_by_role: { prefill: '/p.json' } });
  eq(applyPdDirectionGuard({ config_directions: [{ axis: 'chunk', engine: 'prefill' }] }, 't'),
    [], 'an engine that was never profiled is not a gap (the profiler already said so)');
}
{
  const { applyPdDirectionGuard } = load('tp_span', BY_ROLE);
  const strat = { config_directions: [{ axis: 'x', engine: 'Decode' }] };
  eq(applyPdDirectionGuard(strat, 't'), [], 'off PD the guard reports nothing');
  eq(strat.config_directions[0].engine, 'Decode', '...and changes nothing');
}

// ------------------------------------------------------------------ C. engine capture
// One patch runs on both PD engines: the capture hooks both, and the cases span both regimes.
console.log('C. engine capture');
const cStart = src.indexOf('function engineWorkload(');
const cEnd = src.indexOf('// PHASE: TuningSkillset', cStart);
const WORKLOAD = { profile_workload_json: '/w.json',
  profile_workload_by_role: { prefill: '/w.prefill.json', decode: '/w.decode.json' } };
if (cStart < 0 || cEnd < 0) { console.error('FAIL: capture block not found in e2e_workflow.js'); process.exit(1); }
const loadC = (topology) => new Function(
  'SERVING_TOPOLOGY', 'IS_PD', 'SERVING_TP', 'SERVING_GPU', 'curOverlay', 'CONC', 'profile',
  `${src.slice(cStart, cEnd)}\nreturn { engineWorkload, captureShard, engineRegimes };`)(
  topology, topology === 'pd_disagg', 8, '0,1,2,3,4,5,6,7', '/ovl/shared', 16, WORKLOAD);
{
  eq(loadC('pd_disagg').captureShard(), {
    CAPTURE_ON_ENGINE: 'pd_disagg', CAPTURE_ENGINE_BASE_OVERLAY: '/ovl/shared',
    CAPTURE_ENGINE_LOG: 'every engine\'s "log" in mn_launch.json pd.engines', CAPTURE_TP: 8,
    CAPTURE_GPU: '0,1,2,3,4,5,6,7',
  }, 'PD captures on both engines, seeded from the one shared overlay');
  eq(loadC('tp_span').captureShard(), {
    CAPTURE_ON_ENGINE: 'tp_span', CAPTURE_ENGINE_BASE_OVERLAY: '/ovl/shared',
    CAPTURE_ENGINE_LOG: 'server.log', CAPTURE_TP: 8, CAPTURE_GPU: '0,1,2,3,4,5,6,7',
  }, 'tp_span captures on the spanning server, seeded from the shared overlay');
  eq(loadC('').captureShard(), {}, 'a single-host run keeps the isolated capture');
}
for (const topology of ['', 'pd_disagg']) {
  const { engineRegimes, engineWorkload } = loadC(topology);
  eq([engineRegimes().REQUIRE_DECODE_BUCKET, /chunk size, ~thousands/.test(engineRegimes().PREFILL_M_NOTE)],
    [true, true], `${topology || 'single host'}: the cases span both regimes`);
  eq(engineWorkload().PROFILE_WORKLOAD_JSON, '/w.json', `${topology || 'single host'}: the merged workload`);
  eq('SERVING_ENGINE' in engineWorkload(), false, `${topology || 'single host'}: no engine narrows the extraction`);
}

// ------------------------------------------------------------------ D. per-engine variables + prompt
console.log('D. per-engine variables + prompt');
{
  const { pdOut, pdRole, pdNew } = load('pd_disagg', BY_ROLE);
  eq(pdOut(true, pdNew({ flags: { prefill: '--a 1' }, env: { decode: 'X=1' }, overlay: { decode: '/o' } })),
    { GEAK_MN_PD_SWEEP_ARGS_PREFILL: '--a 1', GEAK_MN_PD_SWEEP_ENV_DECODE: 'X=1' },
    'each engine config layer maps to its own variable; an engine has no overlay of its own');
  eq([pdRole({ engine: 'decode' }), pdRole({ role: 'prefill' }), pdRole({ engine: 'both' }), pdRole(null)],
    ['decode', '', '', ''], 'pdRole: the `engine` field only; anything else is shared');
  eq(load('tp_span', BY_ROLE).pdRole({ engine: 'decode' }), '', 'off PD every head is shared');
  const { engineGpu } = load('pd_disagg', BY_ROLE);
  eq([engineGpu({ engine: 'decode' }, '3'), engineGpu({ engine: 'prefill' }, '0,1'), engineGpu({}, '2')],
    ['decode:3', 'prefill:0,1', '2'], 'a kernel lane card names its engine; a shared head stays bare');
  eq(load('tp_span', BY_ROLE).engineGpu({ engine: 'decode' }, '3'), '3', 'off PD a card is never prefixed');
}
{
  const at = src.indexOf('function roleAgent(');
  const body = src.slice(at, src.indexOf('\n}\n', at));
  const driver = fs.readFileSync(path.join(__dirname, '..', 'roles', '_fragments', 'multinode_driver.md'), 'utf8');
  const pdRule = driver.slice(driver.indexOf('## PD (`SERVING_TOPOLOGY=pd_disagg`)'));
  eq([body.includes("${MN ? `MULTINODE: also Read ${WORKFLOW_DIR}/roles/_fragments/multinode_driver.md"),
    pdRule.includes('Pass each `GEAK_MN_PD_*` input') && pdRule.includes('never copy a `GEAK_MN_PD_*` value into them'),
    /forced\)\.`;/.test(body)], [true, true, true],
    'the PD rule is in the multinode fragment\'s pd_disagg section; the prompt tail is unchanged off PD');
}

// ------------------------------------------------------------------ E. per-engine state
console.log('E. per-engine state');
const loadS = (topology) => new Function('SERVING_TOPOLOGY', 'profile', 'log',
  `let curOverlay = '/shared', curFlags = '', curEnv = '';
   const IS_PD = SERVING_TOPOLOGY === 'pd_disagg';
   ${PD_BLOCK}
   let curPd = pdNew({ flags: { decode: '--d 1' } });
   return { foldAccept, setPd: (kind, byRole) => pdSet(curPd, { [kind]: byRole }), pdOut, engineIntegrateInputs,
     state: () => ({ curPd, curOverlay, curFlags }) };`)(topology, BY_ROLE, () => {});
{
  const LOG_OF = (role) => `the ${role} engine's "log" in mn_launch.json pd.engines`;
  const s = loadS('pd_disagg');
  eq(s.foldAccept({ engine: 'prefill' }, { overlay: '/p', flags: '--x 1' }), 'prefill', 'a prefill flag folds onto prefill');
  eq([s.state().curPd, s.state().curOverlay],
    [{ prefill: { flags: '--x 1', env: '' }, decode: { flags: '--d 1', env: '' } }, '/p'],
    '...the decode layer is untouched, and the code lands on both engines');
  eq([s.foldAccept({ engine: 'decode' }, { overlay: '/d' }), s.state().curOverlay], ['', '/d'],
    'a tagged head whose win is only code banks as shared');
  eq([s.foldAccept({}, { overlay: '/s2' }), s.state().curOverlay], ['', '/s2'], 'an untagged head stays shared');
  const integ = s.engineIntegrateInputs({ engine: 'prefill' });
  eq(['CURRENT_OVERLAY' in integ, 'SHARED_OVERLAY' in integ, 'ENGINE_OVERLAY_VAR' in integ], [false, false, false],
    'the integrator builds on the overlay its caller passes: the shared one');
  eq([integ.ENGINE_LOG, s.engineIntegrateInputs({ engine: 'decode' }).ENGINE_LOG],
    [LOG_OF('prefill'), LOG_OF('decode')], 'the integrator reads the log launch recorded for its engine');
  s.setPd('flags', { prefill: '--y 2' });
  eq([s.state().curPd.prefill, s.state().curPd.decode], [{ flags: '--y 2', env: '' }, { flags: '--d 1', env: '' }],
    'a reply replaces the engine it names; the one it leaves out keeps its layer');
  s.setPd('env', undefined);
  eq(s.state().curPd.decode, { flags: '--d 1', env: '' }, 'a sweep without that record leaves the kind alone');
  s.setPd('flags', { decode: '' });
  eq(s.state().curPd.decode, { flags: '', env: '' }, '"" clears an engine');
  eq(s.pdOut(false), { flags_by_role: { prefill: '--y 2', decode: '' } },
    'records carry only the non-empty kinds, and never an overlay');
}
{
  const s = loadS('pd_disagg');
  s.setPd('flags', { both: '--z' });
  eq(s.state().curPd.prefill.flags, '', 'a key that is no engine changes nothing (the schema refuses it)');
  s.setPd('flags', '--not-a-map');
  eq(s.state().curPd.decode.flags, '--d 1', 'a non-map record changes nothing');
}

// ------------------------------------------------------------------ G. engine names live in the schemas
console.log('G. engine names live in the schemas');
{
  const frag = (a, b) => { const i = src.indexOf(a); return src.slice(i, src.indexOf(b, i)); };
  const schemas = (topology) => new Function('IS_PD', 'PD_ROLES',
    `${frag('const obj = (props', 'const SETUP_SCHEMA')}
     ${frag('const STRATEGY_SCHEMA', '// Returned after source extraction')}
     ${frag('const PLAN_SCHEMA', '// Extraction smoke gate')}
     ${frag('const SWEEP_SCHEMA', '// `e2e_store.py resolve` output')}
     return { STRATEGY_SCHEMA, PLAN_SCHEMA, SWEEP_SCHEMA, arrObj };`)(topology === 'pd_disagg', ['prefill', 'decode']);
  const pd = schemas('pd_disagg');
  const ENUM = ['', 'prefill', 'decode'];
  eq([pd.STRATEGY_SCHEMA.properties.config_directions.items.properties.engine.enum,
    pd.STRATEGY_SCHEMA.properties.head_candidates.items.properties.engine.enum,
    pd.STRATEGY_SCHEMA.properties.kernel_candidates.items.properties.engine.enum,
    pd.PLAN_SCHEMA.properties.head_candidates.items.properties.engine.enum], [ENUM, ENUM, ENUM, ENUM],
    'PD: every queue row names prefill, decode or "" (both)');
  const byRole = pd.SWEEP_SCHEMA.properties.accepted_flags_by_role;
  eq([Object.keys(byRole.properties), byRole.additionalProperties], [['prefill', 'decode'], false],
    'PD: a per-engine map has exactly the two engines');
  for (const topology of ['', 'tp_span']) {
    const s = schemas(topology);
    eq([s.STRATEGY_SCHEMA.properties.config_directions, s.STRATEGY_SCHEMA.properties.kernel_candidates,
      Object.keys(s.STRATEGY_SCHEMA.properties.head_candidates.items.properties), s.PLAN_SCHEMA.properties.head_candidates,
      'accepted_flags_by_role' in s.SWEEP_SCHEMA.properties],
      [s.arrObj, s.arrObj, ['short_name', 'entity_kind', 'device_kernel'], s.arrObj, false],
      `${topology || 'single host'}: the queue and sweep schemas are main's`);
  }
  const { pdRole } = load('pd_disagg', BY_ROLE);
  eq([pdRole({ engine: 'decode' }), pdRole({ engine: 'Decode' }), pdRole({ engine: '' })], ['decode', '', ''],
    'pdRole reads the schema-checked name as is');
}
{
  const s = loadS('tp_span');
  eq([s.foldAccept({ engine: 'prefill' }, { flags: '--z' }), s.state().curFlags], ['', '--z'], 'off PD everything is shared');
}
eq((src.match(/pdSet\(curPd, \{ flags: sweep\.accepted_flags_by_role, env: sweep\.accepted_env_by_role \}\);/g) || []).length, 2,
  'both sweep sites fold the per-engine stack through pdSet');
eq(/\b(engineOf|engineLog|takeSweepPd|normalizePd|pdLayers|setPd|pdVars|pdConfigFields)\b/.test(src), false, 'the old helpers are gone');

// ------------------------------------------------------------------ F. role files stay single-node
console.log('F. role files stay single-node');
{
  const roles = path.join(__dirname, '..', 'roles');
  const read = (f) => fs.readFileSync(path.join(roles, f), 'utf8');
  const MN_ONLY = /_by_role|GEAK_MN_|TRACELENS_TRACE_FILES|mn\.py|pd_disagg|mn_trace|multi-?node|CAPTURE_ON_ENGINE|SERVING_ENGINE|MN_GPU_EXEC|disaggregat/i;
  const MN_ROLES = ['director', 'profiler', 'system_architect', 'kernel_extractor', 'op_benchmarker', 'e2e_integrator'];
  for (const f of [...MN_ROLES, 'config_tuner'].map((r) => `${r}.md`).concat('../../kernel_workflow/roles/benchmark_engineer.md'))
    eq(MN_ONLY.test(read(f)), false, `${f} carries no multinode/PD text`);
  eq(read('config_tuner.md').includes(
    'EXTRA_SERVER_ARGS="<current flags, merged with this flag>" EXTRA_ENV="<current env, merged with this env>"'),
    true, 'config_tuner.md keeps the single-node sweep command');
  const pd = read('_fragments/tuning_pd.md');
  eq([/^## config_tuner$/m.test(pd), /^## tuning_specialist$/m.test(pd),
    pd.includes('accepted_flags_by_role'), pd.includes('apply_env_by_role')], [true, true, true, true],
    'tuning_pd.md has a section per tuning role, each with its own reply fields');
  eq(read('_fragments/profiler_mn.md').includes('profile_topN_by_role'), true,
    'profiler_mn.md carries the per-engine reply fields');

  const fn = (name) => src.match(new RegExp(`function ${name}\\(role\\) \\{[\\s\\S]*?\\n\\}`))[0];
  const fragmentList = src.match(/const MN_ROLE_FRAGMENTS = .*\n/)[0];
  const block = (name, topology, extractOn = false) => new Function('SERVING_TOPOLOGY', 'WORKFLOW_DIR', 'KERNEL_EXTRACT_ON',
    `${fragmentList}${fn(name)}\nreturn ${name};`)(topology, '/wf', extractOn);
  const ALL = ['config_tuner', 'tuning_specialist', 'author_engineer', ...MN_ROLES];
  const fires = (name, topology, extractOn) => ALL.filter((r) => block(name, topology, extractOn)(r));
  eq([fires('pdFragmentBlock', ''), fires('pdFragmentBlock', 'tp_span'), fires('pdFragmentBlock', 'pd_disagg')],
    [[], [], ['config_tuner', 'tuning_specialist']], 'tuning_pd.md is injected for the tuning roles on PD only');
  eq(block('pdFragmentBlock', 'pd_disagg')('config_tuner').includes('`config_tuner` section'), true,
    'the injection names the role\'s section');
  eq([fires('mnRoleBlock', ''), fires('mnRoleBlock', 'tp_span'), fires('mnRoleBlock', 'pd_disagg')],
    [[], MN_ROLES, MN_ROLES], '<role>_mn.md is injected for its role on multinode only');
  eq(fires('mnRoleBlock', 'tp_span', true), MN_ROLES.filter((r) => r !== 'kernel_extractor'),
    'a standalone extraction reads the skill, not kernel_extractor_mn.md');
  for (const r of MN_ROLES) {
    const frag = block('mnRoleBlock', 'tp_span')(r).match(/\/wf\/roles\/(_fragments\/\S+\.md)/)[1];
    eq([frag, read(frag).includes('`SERVING_TOPOLOGY`') || read(frag).includes('`MN_STATE`')],
      [`_fragments/${r}_mn.md`, true], `${frag} exists and says when it applies`);
  }
  eq(/\+ pdFragmentBlock\(role\) \+ mnRoleBlock\(role\) \+/.test(src), true, 'roleAgent appends both injectors');
  // The multinode text the role files dropped is in their fragments.
  for (const [frag, phrase] of [['director_mn.md', 'KVTransferError'], ['director_mn.md', 'gpu_identity.py'],
    ['e2e_integrator_mn.md', 'ENGINE_ENV_VAR'], ['system_architect_mn.md', 'bottleneck_engine'],
    ['op_benchmarker_mn.md', 'harness_lib.py'], ['kernel_extractor_mn.md', 'CAPTURE_ENGINE_BASE_OVERLAY'],
    ['kernel_extractor_mn.md', 'prefill_m_buckets'], ['multinode_driver.md', 'per-machine copy']])
    eq(read(`_fragments/${frag}`).includes(phrase), true, `${frag} carries ${phrase}`);
}

if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
console.log('\nall PD guard checks passed');
