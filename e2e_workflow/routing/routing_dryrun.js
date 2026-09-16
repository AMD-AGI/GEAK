// Expt-3 routing DRY-RUN (offline, no GPU, no API call). Executes the SHIPPED inline routing region
// (between the // <<ROUTING-INLINE-START>> / // <<ROUTING-INLINE-END>> sentinels) pulled straight out
// of e2e_workflow.js and kernel_workflow/kernel_lane.js, so any drift between the tested module and
// what actually ships is caught here.
//
// The two entry points ship DIFFERENT routing seams and are exercised through DIFFERENT surfaces:
//   * e2e_workflow.js — the FULL cascade (__routeDecide -> __routeEscalate + verbatim gate). Proves:
//       (1) OFF -> __routeDecide null on EVERY scope (byte-identical run).
//       (2) inline ROUTE_TIER_MAP has not drifted from the canonical module TIER_MAP.
//       (3) ON + verifier present (Path B) -> mapped -> Sonnet, gated.
//       (4) ON + NO verifier (Path A) -> gated=false -> the seam SUPPRESSES (stays strong).
//       (5) __routeEscalate: cheap-first, one strong fallback on a failed gate, both attempts recorded.
//   * kernel_lane.js — the MINIMAL model-map seam (__routeModel only; NO cascade, by design). Proves:
//       (1) OFF -> __routeModel undefined on EVERY scope (byte-identical run).
//       (2) inline ROUTE_TIER_MAP is exactly the 3 expected cheap helper scopes (no drift).
//       (3) ON -> mapped helper -> Sonnet; unmapped -> undefined (pinned strong).
//   Both: (6) PILOT REACHABILITY — at least one mapped scope is an ACTUAL call site in the entry point.
//   The kernel seam is deliberately NOT the cascade API; asserting the cascade symbols on kernel_lane.js
//   would be testing an API it does not (and should not) ship.
//
//   NODE=$(ls -t /home/aditysin/.cursor-server/bin/linux-x64/*/node | head -1)
//   "$NODE" e2e_workflow/routing/routing_dryrun.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');   // /home/aditysin/PROJECTS/GEAK
const CANON = require('./tier_map.js');
const NUL = String.fromCharCode(0);
const CHEAP = 'claude-sonnet-5';

// Per-entry-point config. `kind` selects which routing SURFACE to exercise. `mapped` lists the scopes
// that SHOULD route, each with a `reach` token: the distinctive source substring that proves the scope
// is an actual call site (grepped OUTSIDE the inline region). `unmapped` scopes must NOT route.
const FILES = [
  {
    name: 'e2e_workflow.js', file: path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'),
    kind: 'cascade', expectReachable: true,
    mapped: [
      { phase: 'Validate', label: 'persist-workflow-return', reach: "label: 'persist-workflow-return'" },
      { phase: 'WarmStart', label: 'warm_start:record-measurements', reach: "label: 'warm_start:record-measurements'" },
    ],
    unmapped: [
      { phase: 'WarmStart', label: 'warm_start:record-measurements r2' }, // dynamic suffix stays mapped -> see below
      { phase: 'Optimize', label: 'system_architect:strategize' },
      { phase: 'Optimize', label: 'director:plan' },
      { phase: 'Finalize', label: 'e2e_integrator:overlay' },
      { phase: 'Optimize', label: 'persist-workflow-return' },           // right label, WRONG phase
      { phase: 'KB', label: 'kb:write' },
      { phase: '', label: 'agent' },
    ],
  },
  {
    name: 'kernel_lane.js', file: path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'),
    kind: 'model_map', expectReachable: true,
    mapped: [
      { phase: 'Optimize', label: 'clock', reach: 'clock ${tag}' },
      { phase: 'Optimize', label: 'storage:reclaim', reach: 'storage:reclaim r${round}' },
      { phase: 'WarmStart', label: 'warm_start:resolve', reach: "label: 'warm_start:resolve'" },
    ],
    unmapped: [
      { phase: 'Optimize', label: 'tech_lead:plan_round' },
      { phase: 'WarmStart', label: 'clock' },        // right label, WRONG phase
      { phase: '', label: 'agent' },
    ],
    // The canonical inline kernel map (drift guard): exactly these 3 cheap scopes, nothing else.
    expectMap: (() => {
      const m = {};
      m['Optimize' + NUL + 'clock'] = CHEAP;
      m['Optimize' + NUL + 'storage:reclaim'] = CHEAP;
      m['WarmStart' + NUL + 'warm_start:resolve'] = CHEAP;
      return m;
    })(),
  },
];

// A dynamic-suffix variant that MUST still route (same static prefix). Kept out of `unmapped` above.
const E2E_DYNAMIC_MAPPED = { phase: 'WarmStart', label: 'warm_start:record-measurements r2' };

// Extract the sentinel region and eval it, injecting the globals it closes over. Symbol-TOLERANT: the
// two entry points export different routing symbols (cascade vs model-map), so the return probes each
// with typeof and yields undefined for any a given file does not define — no ReferenceError either way.
function loadInline(src, argsObj, envObj, logs, requireImpl, agentImpl) {
  const A_START = '// <<ROUTING-INLINE-START>>';
  const A_END = '// <<ROUTING-INLINE-END>>';
  const i = src.indexOf(A_START);
  const j = src.indexOf(A_END);
  if (i < 0 || j < 0) throw new Error('inline sentinels not found in ' + (src.length) + ' bytes');
  const block = src.slice(i, j + A_END.length);
  const want = ['__routeDecide', '__routeEscalate', '__routeCheckVerbatim', '__routeValidate',
    '__routeExpected', '__routeModel', '__routeLabelPrefix', 'ROUTING_ON', 'ROUTE_TIER_MAP',
    '__routeAttempts'];
  const ret = 'return {' + want.map((s) => s + ': (typeof ' + s + " !== 'undefined' ? " + s + ' : undefined)').join(', ') + '};';
  const factory = new Function(
    'A', 'log', 'process', 'require', 'setTimeout', 'agent', '__routeAgentTimeoutMs',
    block + '\n' + ret);
  const fakeProcess = { env: envObj || {} };
  const req = requireImpl || function () { throw new Error('no require (Path A)'); };
  const timeout = function () { return 0; };
  return factory(argsObj || {}, (m) => logs.push(String(m)), fakeProcess, req, setTimeout, agentImpl || (async () => null), timeout);
}

let fails = 0, n = 0;
function ok_(name, cond) { n++; if (cond) { console.log(`PASS  ${name}`); } else { console.log(`FAIL  ${name}`); fails++; } }

// A require that supplies an fs backed by an in-memory disk (simulates Path B's real verifier).
function fsRequire(disk) {
  return function (mod) {
    if (mod !== 'fs') throw new Error('only fs stubbed');
    return { readFileSync: function (p) { if (!(p in disk)) { const e = new Error('ENOENT'); throw e; } return disk[p]; } };
  };
}

function reachabilityCheck(F, src) {
  // Does the entry point actually CALL a mapped scope? Scan the file text OUTSIDE the inline routing
  // region for each mapped scope's distinctive `reach` token.
  const regionA = src.indexOf('// <<ROUTING-INLINE-START>>');
  const regionB = src.indexOf('// <<ROUTING-INLINE-END>>') + '// <<ROUTING-INLINE-END>>'.length;
  const outside = src.slice(0, regionA) + src.slice(regionB);
  const reached = F.mapped.filter((m) => outside.indexOf(m.reach) >= 0).map((m) => m.label);
  const isReachable = reached.length >= 1;
  console.log(`  reachable mapped labels in ${F.name}: ${reached.length ? reached.join(', ') : '(none)'}`);
  if (F.expectReachable) {
    ok_(`${F.name}: >=1 mapped scope is an ACTUAL call site (pilot can demonstrate routing)`, isReachable);
  } else {
    ok_(`${F.name}: ZERO mapped call sites here — documented`, !isReachable);
  }
}

async function runCascade(F, src) {
  // (1) OFF (no args, no env): NOTHING routes, even with a real fs require present.
  const off = loadInline(src, {}, {}, [], fsRequire({}));
  ok_(`${F.name}: routing OFF by default`, off.ROUTING_ON === false);
  let offRoutes = 0;
  for (const s of F.mapped.concat(F.unmapped)) if (off.__routeDecide({ phase: s.phase, label: s.label }) !== null) offRoutes++;
  ok_(`${F.name}: OFF -> __routeDecide null on ALL scopes (byte-identical)`, offRoutes === 0);

  // (2) drift guard: inline ROUTE_TIER_MAP === canonical module TIER_MAP (keys carry U+0000 in both).
  ok_(`${F.name}: inline map matches canonical module (no drift)`,
      JSON.stringify(off.ROUTE_TIER_MAP) === JSON.stringify(CANON.TIER_MAP));
  const someKey = Object.keys(off.ROUTE_TIER_MAP)[0];
  ok_(`${F.name}: scope keys use U+0000 separator (not a space)`, someKey.indexOf(NUL) >= 0 && someKey.indexOf(' ') < 0);

  // (3) ON + verifier present (Path B sim): mapped -> Sonnet + gated; unmapped -> null.
  const onB = loadInline(src, { routing: 'true' }, {}, [], fsRequire({}));
  ok_(`${F.name}: routing ON when A.routing='true'`, onB.ROUTING_ON === true);
  let good = true, routed = 0;
  for (const s of F.mapped.concat([E2E_DYNAMIC_MAPPED])) {
    const d = onB.__routeDecide({ phase: s.phase, label: s.label });
    if (!d || d.model !== CHEAP || d.gated !== true || d.kind !== 'verbatim_write') good = false; else routed++;
  }
  for (const s of F.unmapped.filter((s) => !(s.phase === E2E_DYNAMIC_MAPPED.phase && s.label === E2E_DYNAMIC_MAPPED.label))) {
    if (onB.__routeDecide({ phase: s.phase, label: s.label }) !== null) good = false;
  }
  ok_(`${F.name}: ON+verifier -> mapped route to Sonnet (gated), all others pinned`, good);
  ok_(`${F.name}: ON routed count == mapped(+dynamic) count`, routed === F.mapped.length + 1);

  // (4) ON + NO verifier (Path A sim: require throws): mapped -> decision.gated=false -> seam SUPPRESSES.
  const onA = loadInline(src, { routing: 'true' }, {}, [], /* requireImpl */ null);
  const dA = onA.__routeDecide({ phase: 'Validate', label: 'persist-workflow-return' });
  ok_(`${F.name}: ON+no-verifier (Path A) -> mapped decision has gated=false (seam will suppress)`,
      dA && dA.model === CHEAP && dA.gated === false);
  const weak = onA.__routeCheckVerbatim('create the file "/x" with\n```\nq\n```', { path: '/x' }, null, null);
  ok_(`${F.name}: no-verifier validator returns ok=false (weak != pass)`, weak.ok === false && weak.weak === true);

  // (5) drive the SHIPPED __routeEscalate: cheap writes WRONG bytes -> strong fallback fixes it.
  const EXPECT_PATH = '/tmp/dryrun/workflow_return.json';
  const EXPECT_CONTENT = '{\n  "a": 1\n}';
  const PROMPT = 'Use the Write tool to create the file "' + EXPECT_PATH +
    '" with EXACTLY the content below, verbatim:\n\n```json\n' + EXPECT_CONTENT + '\n```\n\nThen return a receipt.';
  const disk = {};
  const calls = [];
  const agentImpl = async (p, o) => {
    calls.push(o.model);
    disk[EXPECT_PATH] = (o.model === CANON.MODEL_STRONG) ? EXPECT_CONTENT : '{\n  "a": 999\n}';
    return { written: true, path: EXPECT_PATH };
  };
  const esc = loadInline(src, { routing: 'true' }, {}, [], fsRequire(disk), agentImpl);
  const dEsc = esc.__routeDecide({ phase: 'Validate', label: 'persist-workflow-return' });
  const rEsc = await esc.__routeEscalate(PROMPT, { phase: 'Validate', label: 'persist-workflow-return' }, dEsc);
  ok_(`${F.name}: escalate made 2 calls, cheap(Sonnet) then strong(Opus)`,
      calls.length === 2 && calls[0] === CHEAP && calls[1] === CANON.MODEL_STRONG);
  ok_(`${F.name}: escalate recorded BOTH attempts (fail then pass)`,
      esc.__routeAttempts.length === 2 && esc.__routeAttempts[0].ok === false &&
      esc.__routeAttempts[1].ok === true && esc.__routeAttempts[1].escalated === true);
  ok_(`${F.name}: escalate returned the strong result (final artifact matches)`,
      rEsc && rEsc.path === EXPECT_PATH && disk[EXPECT_PATH] === EXPECT_CONTENT);
}

async function runModelMap(F, src) {
  // (1) OFF: __routeModel returns undefined for EVERY scope -> byte-identical.
  const off = loadInline(src, {}, {}, [], fsRequire({}));
  ok_(`${F.name}: routing OFF by default`, off.ROUTING_ON === false);
  ok_(`${F.name}: model-map seam present (__routeModel), cascade API absent (by design)`,
      typeof off.__routeModel === 'function' && off.__routeDecide === undefined && off.__routeEscalate === undefined);
  let offRoutes = 0;
  for (const s of F.mapped.concat(F.unmapped)) if (off.__routeModel({ phase: s.phase, label: s.label }) !== undefined) offRoutes++;
  ok_(`${F.name}: OFF -> __routeModel undefined on ALL scopes (byte-identical)`, offRoutes === 0);

  // (2) drift guard: the inline kernel map is exactly the 3 expected cheap helper scopes.
  ok_(`${F.name}: inline map == expected 3 cheap helper scopes (no drift)`,
      JSON.stringify(off.ROUTE_TIER_MAP) === JSON.stringify(F.expectMap));
  const someKey = Object.keys(off.ROUTE_TIER_MAP)[0];
  ok_(`${F.name}: scope keys use U+0000 separator (not a space)`, someKey.indexOf(NUL) >= 0 && someKey.indexOf(' ') < 0);

  // (3) ON: mapped helper -> Sonnet (incl. dynamic suffix on the same static prefix); unmapped -> undefined.
  const on = loadInline(src, { routing: 'true' }, {}, []);
  ok_(`${F.name}: routing ON when A.routing='true'`, on.ROUTING_ON === true);
  let good = true, routed = 0;
  const dyn = [{ phase: 'Optimize', label: 'clock r1' }, { phase: 'Optimize', label: 'storage:reclaim r3' }];
  for (const s of F.mapped.concat(dyn)) {
    if (on.__routeModel({ phase: s.phase, label: s.label }) !== CHEAP) good = false; else routed++;
  }
  for (const s of F.unmapped) if (on.__routeModel({ phase: s.phase, label: s.label }) !== undefined) good = false;
  ok_(`${F.name}: ON -> mapped helper routes to Sonnet, all others pinned`, good);
  ok_(`${F.name}: ON routed count == mapped(+dynamic) count`, routed === F.mapped.length + dyn.length);
}

(async () => {
  for (const F of FILES) {
    const src = fs.readFileSync(F.file, 'utf8');
    console.log(`\n===== ${F.name} (${F.kind}) =====`);
    if (F.kind === 'cascade') await runCascade(F, src);
    else await runModelMap(F, src);
    reachabilityCheck(F, src);
  }

  // (7) Opt-in propagation: each dispatcher forwards A.routing to its child lanes the same guarded way
  // it forwards llm_stats, so a routed parent reaches every lane (and an unset parent stays OFF).
  console.log('\n===== routing switch forwarded to child lanes =====');
  for (const f of [path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'),
                   path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js')]) {
    const s = fs.readFileSync(f, 'utf8');
    ok_(`${path.basename(f)}: forwards A.routing to child lanes (guarded)`,
        /A\.routing\s*!=\s*null\s*\?\s*\{\s*routing:\s*String\(A\.routing\)/.test(s));
  }

  console.log(`\n${fails === 0 ? 'DRY-RUN PASS' : 'DRY-RUN FAIL'} — ${n - fails}/${n} checks`);
  process.exit(fails === 0 ? 0 : 1);
})();
