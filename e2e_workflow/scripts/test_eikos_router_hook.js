// Offline test of the Eikos routing hook AS SHIPPED in e2e_workflow.js: the region between
// <<EIKOS-ROUTER-START>> and <<EIKOS-ROUTER-END>> plus the B5 static router it falls back to,
// evaluated with the helper process stubbed. It establishes hook behaviour at the wrapper
// boundary only -- not the helper's decisions, and not workflow replay on resume.
//
//   node e2e_workflow/scripts/test_eikos_router_hook.js
'use strict';
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(path.join(__dirname, '..', 'e2e_workflow.js'), 'utf8');
function slice(from, to) {
  const i = SRC.indexOf(from), j = SRC.indexOf(to, i);
  if (i < 0 || j < 0) throw new Error('marker not found: ' + from);
  return SRC.slice(i, j + to.length);
}
const B5 = slice('const ABL_CHEAP_LABELS', 'return ABL_CHEAP_LABELS.test(label) ? \'low\' : null;\n}');
const HOOK = slice('// <<EIKOS-ROUTER-START>>', '// <<EIKOS-ROUTER-END>>');

// helper: what the stubbed eikos_router.py prints, a function of the request, or an Error to throw.
function load({ on = true, b5 = true, helper = null, noRequire = false } = {}) {
  const calls = [], events = [], logs = [];
  const req = noRequire
    ? () => { throw new Error('require is not defined'); }
    : (mod) => {
      if (mod !== 'child_process') throw new Error('unexpected require ' + mod);
      return { execFileSync: (cmd, args, o) => {
        const r = JSON.parse(o.input); calls.push({ cmd, args, req: r });
        const out = typeof helper === 'function' ? helper(r) : helper;
        if (out instanceof Error) throw out;
        return typeof out === 'string' ? out : JSON.stringify(out);
      } };
    };
  const f = new Function('ABL', 'ablEvent', 'log', 'require', 'WORKFLOW_DIR', 'A', 'EVAL_DIR',
    'EIKOS_ROUTER_ON', 'EIKOS_ROUTER_TIMEOUT_MS',
    B5 + '\n' + HOOK + '\nreturn { routeOptsFor, eikosRouteFor };');
  const api = f((k) => k === 'B5' && b5, (e) => events.push(e), (m) => logs.push(m), req,
    '/wf', { exp_root: '/exp' }, '/eval', on, 15000);
  return { ...api, calls, events, logs };
}

let n = 0, fails = 0;
function ok(name, cond) { n++; if (cond) console.log('PASS  ' + name); else { fails++; console.log('FAIL  ' + name); } }

const CHEAP = { label: 'storage:reclaim r1', phase: 'Optimize', schema: { s: 1 } };
const HOST = { model: null, effort: null, tier: null, source: 'host', reason: 'eikos unavailable or invalid' };
const EIKOS_CHEAP = { model: 'claude-haiku-4-5-20251001', effort: 'low', tier: 'cheap', source: 'eikos', confidence: 0.9 };
const EIKOS_THINKER = { model: 'claude-opus-5-5', effort: null, tier: 'thinker', source: 'eikos', escalated: true };

// ---- router OFF: the host path, byte-identical to a build without the router
{
  const h = load({ on: false, b5: true, helper: () => { throw new Error('must not run'); } });
  const o = h.routeOptsFor(CHEAP, 0, 'task');
  ok('OFF + B5 on: static route sets effort low, helper never runs', o.effort === 'low' && h.calls.length === 0);
}
{
  const h = load({ on: false, b5: false });
  const inOpts = { ...CHEAP, effort: 'max' };
  ok('OFF + B5 off: caller options returned untouched (same object)', h.routeOptsFor(inOpts, 0, 't') === inOpts);
}

// ---- router ON but no Eikos decision: the host must decide exactly as without the router
for (const [name, helper] of [
  ['helper answers "host"', HOST],
  ['helper process throws (timeout / non-zero exit)', new Error('ETIMEDOUT')],
  ['helper prints garbage', 'not json'],
  ['helper answers from an old Jev source', { ...EIKOS_CHEAP, source: 'jev' }],
]) {
  {
    const h = load({ on: true, b5: false, helper });
    const inOpts = { ...CHEAP, effort: 'max', model: 'claude-opus-5-5' };
    const o = h.routeOptsFor(inOpts, 0, 't');
    ok(`ON + B5 off + ${name}: incoming effort/model untouched (Astra's reproduction)`,
      o === inOpts && o.effort === 'max' && h.events.length === 0);
  }
  {
    const h = load({ on: true, b5: true, helper });
    const o = h.routeOptsFor(CHEAP, 0, 't');
    ok(`ON + B5 on + ${name}: B5 static route applies`, o.effort === 'low' && !o.model &&
      h.events.length === 1 && h.events[0].tier === 'cheap' && !h.events[0].source);
  }
}
{
  const h = load({ on: true, b5: false, noRequire: true });
  const inOpts = { ...CHEAP };
  ok('native Workflow (no require): host decides, failure logged', h.routeOptsFor(inOpts, 0, 't') === inOpts &&
    h.logs.some((m) => m.includes('host decides')));
}

// ---- a real Eikos decision is applied, and only its fields change
{
  const h = load({ on: true, b5: false, helper: EIKOS_CHEAP });
  const inOpts = { ...CHEAP, effort: 'max' };
  const o = h.routeOptsFor(inOpts, 0, 'full task text');
  ok('ON + Eikos cheap: model and effort set, other options preserved',
    o.model === 'claude-haiku-4-5-20251001' && o.effort === 'low' && o.schema === CHEAP.schema &&
    o.phase === 'Optimize' && o !== inOpts && inOpts.effort === 'max');
  ok('ON + Eikos cheap: event records source eikos', h.events[0] && h.events[0].source === 'eikos');
}
{
  const h = load({ on: true, b5: true, helper: EIKOS_THINKER });
  const o = h.routeOptsFor({ ...CHEAP, effort: 'high' }, 0, 't');
  ok('ON + Eikos thinker: model set, incoming effort kept (thinker sets no effort)',
    o.model === 'claude-opus-5-5' && o.effort === 'high');
}

// ---- who is asked at all
{
  const h = load({ on: true, b5: true, helper: EIKOS_CHEAP });
  const o = h.routeOptsFor(CHEAP, 1, 't');
  ok('retry: helper never asked, and B5 does not route a retry either', h.calls.length === 0 && o.effort === undefined);
}
{
  const h = load({ on: true, b5: true, helper: EIKOS_CHEAP });
  const inOpts = { label: 'engineer d1:algorithm', phase: 'Optimize' };
  ok('label not cheap-eligible: helper never asked, options untouched',
    h.routeOptsFor(inOpts, 0, 't') === inOpts && h.calls.length === 0);
}
{
  const big = 'q'.repeat(20000) + 'CRITICAL TAIL';
  const h = load({ on: true, b5: false, helper: HOST });
  h.routeOptsFor(CHEAP, 0, big);
  ok('the helper receives the FULL prompt, not a prefix', h.calls[0].req.task === big);
  ok('the helper is eikos_router.py with the run cache dir',
    h.calls[0].args.includes('/wf/scripts/eikos_router.py') && h.calls[0].args.includes('/exp'));
}

console.log(`\n${fails === 0 ? 'ALL PASS' : 'FAIL'} — ${n - fails}/${n} checks`);
process.exit(fails === 0 ? 0 : 1);
