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

const mStart = src.indexOf("const PD_ROLES = ['prefill', 'decode'];");
const mEnd = src.indexOf('function pdConfigFields(', mStart);
if (mStart < 0 || mEnd < 0) { console.error('FAIL: PD helper block not found in e2e_workflow.js'); process.exit(1); }
const load = (topology, prof) => new Function('SERVING_TOPOLOGY', 'profile', 'log',
  `${src.slice(mStart, mEnd)}\nreturn { applyPdDirectionGuard, pdVars, pdVar, pdRole, pdLayers, engineGpu };`)(
  topology, prof, () => {});

const BY_ROLE = { profile_topN_by_role: { prefill: '/p.json', decode: '/d.json' } };

// ------------------------------------------------------------------ A. directions
console.log('A. directions');
{
  const { applyPdDirectionGuard } = load('pd_disagg', BY_ROLE);
  const strat = { config_directions: [
    { axis: 'chunked-prefill-size', engine: 'Prefill' },
    { axis: 'attention-backend', engine: 'decode' },
    { axis: 'kv-cache-dtype', engine: 'both' },
  ] };
  eq(applyPdDirectionGuard(strat, 't'), [], 'both engines covered -> no gap');
  eq(strat.config_directions.map((d) => d.engine), ['prefill', 'decode', ''],
    'engine is normalized: case, and an unknown value becomes shared');
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
// A PD head's M set is captured on its own engine in the measured lifecycle.
console.log('C. engine capture');
const cStart = src.indexOf('function engineWorkload(');
const cEnd = src.indexOf('// PHASE: TuningSkillset', cStart);
const WORKLOAD = { profile_workload_json: '/w.json',
  profile_workload_by_role: { prefill: '/w.prefill.json', decode: '/w.decode.json' } };
if (cStart < 0 || cEnd < 0) { console.error('FAIL: capture block not found in e2e_workflow.js'); process.exit(1); }
const loadC = (topology, A = {}, tpByRole = topology === 'pd_disagg' ? { prefill: 8, decode: 8 } : null) => new Function(
  'SERVING_TOPOLOGY', 'SERVING_TP', 'SERVING_GPU', 'MN_SPEC', 'curPd', 'curOverlay', 'CONC', 'A', 'pdRole', 'profile',
  'pdVar',
  `${src.slice(cStart, cEnd)}\nreturn { engineWorkload, captureShard, engineRegimes };`)(
  topology, 8, '0,1,2,3,4,5,6,7', tpByRole && { tp_by_role: tpByRole }, { prefill: { overlay: '/ovl/pre' }, decode: { overlay: '' } },
  '/ovl/shared', 16, A, load(topology, BY_ROLE).pdRole, WORKLOAD, load(topology, BY_ROLE).pdVar);
const LOG_OF = (role) => `the ${role} engine's "log" in mn_launch.json pd.engines`;
{
  const { captureShard } = loadC('pd_disagg');
  eq(captureShard('prefill'), {
    CAPTURE_ON_ENGINE: 'prefill', CAPTURE_ENGINE_OVERLAY_VAR: 'GEAK_MN_PD_OVERLAY_PREFILL',
    CAPTURE_ENGINE_BASE_OVERLAY: '/ovl/pre', CAPTURE_SHARED_OVERLAY: '/ovl/shared',
    CAPTURE_ENGINE_LOG: LOG_OF('prefill'), CAPTURE_TP: 8, CAPTURE_GPU: '0,1,2,3,4,5,6,7',
  }, 'a prefill head captures on the prefill engine, seeded from its own overlay');
  eq(captureShard('decode').CAPTURE_ENGINE_LOG, LOG_OF('decode'),
    'a decode head reads the log launch recorded for the decode engine');
  eq(loadC('pd_disagg', {}, { prefill: 8, decode: 4 }).captureShard('decode').CAPTURE_TP, 4,
    'a decode engine with its own TP captures at that TP');
  eq(captureShard(''), {}, 'an untagged head keeps the isolated capture');
  eq(loadC('').captureShard('prefill'), {}, 'a single-host run keeps the isolated capture');
}
{
  const span = loadC('tp_span').captureShard('');
  eq(span, {
    CAPTURE_ON_ENGINE: 'tp_span', CAPTURE_ENGINE_OVERLAY_VAR: 'OVERLAY_PYTHONPATH',
    CAPTURE_ENGINE_BASE_OVERLAY: '/ovl/shared', CAPTURE_SHARED_OVERLAY: '',
    CAPTURE_ENGINE_LOG: 'server.log', CAPTURE_TP: 8, CAPTURE_GPU: '0,1,2,3,4,5,6,7',
  }, 'tp_span captures on the spanning server, seeded from the shared overlay');
  eq(loadC('tp_span').captureShard('decode'), span, '...whatever the head\'s role says');
}
{
  const { engineRegimes } = loadC('pd_disagg');
  const note = engineRegimes({ engine: 'prefill' }).PREFILL_M_NOTE;
  eq(/meta\.shape_counts/.test(note) && !/chunk size, ~thousands/.test(note), true,
    'the prefill M set is the capture, not the chunk size');
}
{
  // A head candidate's own `role` is its seam kind, never an engine.
  const seam = { role: 'op_seam' };
  eq(loadC('').engineWorkload(seam), { PROFILE_WORKLOAD_JSON: '/w.json' },
    'single host: a seam role adds no SERVING_ENGINE');
  eq(loadC('pd_disagg').engineWorkload(seam), { PROFILE_WORKLOAD_JSON: '/w.json' },
    'PD: an untagged head uses the merged workload');
  eq(loadC('pd_disagg').engineWorkload({ engine: 'decode', role: 'op_seam' }).PROFILE_WORKLOAD_JSON, '/w.decode.json',
    'PD: a decode head uses the decode workload');
  eq(loadC('').engineRegimes({ role: 'prefill' }).REQUIRE_DECODE_BUCKET, true,
    'single host keeps the decode bucket whatever the role says');
}

// ------------------------------------------------------------------ D. per-engine variables + prompt
console.log('D. per-engine variables + prompt');
{
  const { pdVars, pdRole, pdLayers } = load('pd_disagg', BY_ROLE);
  eq(pdVars(pdLayers({ flags: { prefill: '--a 1' }, env: { decode: 'X=1' }, overlay: { decode: '/o' } })),
    { GEAK_MN_PD_SWEEP_ARGS_PREFILL: '--a 1', GEAK_MN_PD_SWEEP_ENV_DECODE: 'X=1', GEAK_MN_PD_OVERLAY_DECODE: '/o' },
    'each engine layer maps to its own variable; empty ones are left out');
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
  eq([/\$\{SERVING_TOPOLOGY === 'pd_disagg' \? `PD: pass each GEAK_MN_PD_\* Input/.test(body),
    /forced\)\.`;/.test(body)], [true, true],
    'the PD rule is only in a pd_disagg prompt; the prompt tail is unchanged off PD');
}

// ------------------------------------------------------------------ E. per-engine state
console.log('E. per-engine state');
const pdEnd = src.indexOf('\n}\n', src.indexOf('function pdConfigFields(')) + 3;
const loadS = (topology) => new Function('SERVING_TOPOLOGY', 'profile', 'log',
  `let curOverlay = '/shared', curFlags = '', curEnv = '';
   ${src.slice(mStart, pdEnd)}
   let curPd = pdLayers({ flags: { decode: '--d 1' } });
   return { foldAccept, setPd, pdConfigFields, engineIntegrateInputs,
     state: () => ({ curPd, curOverlay, curFlags }) };`)(topology, BY_ROLE, () => {});
{
  const s = loadS('pd_disagg');
  eq(s.foldAccept({ engine: 'prefill' }, { overlay: '/p', flags: '--x 1' }), 'prefill', 'a prefill head folds onto prefill');
  eq(s.state().curPd, { prefill: { flags: '--x 1', env: '', overlay: '/p' }, decode: { flags: '--d 1', env: '', overlay: '' } },
    '...and the decode layer is untouched');
  eq([s.foldAccept({}, { overlay: '/s2' }), s.state().curOverlay], ['', '/s2'], 'an untagged head stays shared');
  eq(s.engineIntegrateInputs({ engine: 'prefill' }).CURRENT_OVERLAY, '/p', 'the integrator builds on the engine layer');
  eq([s.engineIntegrateInputs({ engine: 'prefill' }).ENGINE_LOG, s.engineIntegrateInputs({ engine: 'decode' }).ENGINE_LOG],
    [LOG_OF('prefill'), LOG_OF('decode')], 'the integrator reads the log launch recorded for its engine');
  s.setPd('flags', { prefill: '--y 2' });
  eq(s.state().curPd.prefill, { flags: '--y 2', env: '', overlay: '/p' },
    'a sweep replaces the flags of every engine and leaves the overlay alone');
  s.setPd('env', undefined);
  eq(s.state().curPd.decode, { flags: '', env: '', overlay: '' }, 'a sweep without that record leaves the kind alone');
  eq(s.pdConfigFields(), { flags_by_role: { prefill: '--y 2', decode: '' }, overlay_by_role: { prefill: '/p', decode: '' } },
    'records carry only the non-empty kinds');
}
{
  const s = loadS('tp_span');
  eq([s.foldAccept({ engine: 'prefill' }, { flags: '--z' }), s.state().curFlags], ['', '--z'], 'off PD everything is shared');
}
eq(/Object\.assign\(fixInputs, engineIntegrateInputs\(\{ engine: base\.SERVING_ENGINE \}\)\);/.test(src), true,
  'a corrective re-gate re-applies the engine layer over the shared overlay');
eq((src.match(/setPd\('flags', sweep\.accepted_flags_by_role\); setPd\('env', sweep\.accepted_env_by_role\);/g) || []).length, 2,
  'both sweep sites fold the per-engine stack through setPd');
eq(/\b(engineOf|engineLog|takeSweepPd|normalizePd)\b/.test(src), false, 'the old helpers are gone');

if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
console.log('\nall PD guard checks passed');
