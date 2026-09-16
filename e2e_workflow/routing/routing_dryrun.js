// Expt-3 routing DRY-RUN (offline, no GPU, no API call). Executes the SHIPPED inline routing region
// (between the // <<ROUTING-INLINE-START>> / // <<ROUTING-INLINE-END>> sentinels) pulled straight out
// of e2e_workflow.js and kernel_workflow/kernel_lane.js, so any drift between the tested module and
// what actually ships is caught here. It proves:
//   (1) OFF by default -> __routeDecide() returns null for EVERY scope (byte-identical run).
//   (2) the inline ROUTE_TIER_MAP has not drifted from the canonical module TIER_MAP.
//   (3) ON + mapped + a verifier present in the runtime (Path B) -> decision routes to Sonnet, gated.
//   (4) ON + mapped + NO verifier in the runtime (Path A) -> decision.gated=false -> the seam SUPPRESSES
//       (stays strong); a cheap attempt that cannot be gated is never accepted.
//   (5) driving the shipped __routeEscalate: cheap-first, and on a failed artifact gate exactly one
//       strong fallback (bypassing the cheap map), with BOTH attempts recorded and a hard cap of 2.
//   (6) PILOT REACHABILITY (Astra CRITICAL #1): at least one mapped scope is an ACTUAL call site in the
//       entry point under test. e2e_workflow.js reaches >=1; kernel_lane.js currently reaches ZERO, so
//       a standalone knn/kernel-lane A/B would route nothing — flagged loudly, not silently passed.
//
//   NODE=$(ls -t /home/aditysin/.cursor-server/bin/linux-x64/*/node | head -1)
//   "$NODE" e2e_workflow/routing/routing_dryrun.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');   // /home/aditysin/PROJECTS/GEAK
const CANON = require('./tier_map.js');
const NUL = String.fromCharCode(0);

const FILES = [
  { name: 'e2e_workflow.js', file: path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), expectReachable: true },
  { name: 'kernel_lane.js', file: path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'), expectReachable: false },
];

// The two mapped scopes (phase + static label prefix) and their raw label literals (for reachability).
const MAPPED = [
  { phase: 'Validate', label: 'file_writer:persist:workflow-return' },
  { phase: 'WarmStart', label: 'warm_start:record-measurements' },
];

const SCOPES = [
  { phase: 'Validate', label: 'file_writer:persist:workflow-return', mapped: true },
  { phase: 'WarmStart', label: 'warm_start:record-measurements', mapped: true },
  { phase: 'WarmStart', label: 'warm_start:record-measurements r2', mapped: true }, // dynamic suffix, same prefix
  { phase: 'Optimize', label: 'system_architect:strategize', mapped: false },
  { phase: 'Optimize', label: 'director:plan', mapped: false },
  { phase: 'Finalize', label: 'e2e_integrator:overlay', mapped: false },
  { phase: 'Optimize', label: 'file_writer:persist:workflow-return', mapped: false }, // right label, WRONG phase
  { phase: 'Optimize', label: 'clock r1', mapped: false },
  { phase: 'Optimize', label: 'storage:reclaim r3', mapped: false },
  { phase: 'KB', label: 'kb:write', mapped: false },
  { phase: '', label: 'agent', mapped: false },
];

// Extract the sentinel region and eval it, injecting the globals it closes over. `requireImpl` lets us
// simulate a runtime WITH a real fs verifier (Path B) or WITHOUT one (Path A: pass a throwing require).
function loadInline(src, argsObj, envObj, logs, requireImpl, agentImpl) {
  const A_START = '// <<ROUTING-INLINE-START>>';
  const A_END = '// <<ROUTING-INLINE-END>>';
  const i = src.indexOf(A_START);
  const j = src.indexOf(A_END);
  if (i < 0 || j < 0) throw new Error('inline sentinels not found');
  const block = src.slice(i, j + A_END.length);
  const factory = new Function(
    'A', 'log', 'process', 'require', 'setTimeout', 'agent', '__routeAgentTimeoutMs',
    block + '\nreturn { __routeDecide, __routeEscalate, __routeCheckVerbatim, __routeValidate, __routeExpected, ROUTING_ON, ROUTE_TIER_MAP, __routeAttempts };');
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

(async () => {
  for (const F of FILES) {
    const src = fs.readFileSync(F.file, 'utf8');
    console.log(`\n===== ${F.name} =====`);

    // (1) OFF (no args, no env): NOTHING routes. With a real fs require present, still nothing.
    let logsOff = [];
    const off = loadInline(src, {}, {}, logsOff, fsRequire({}));
    ok_(`${F.name}: routing OFF by default`, off.ROUTING_ON === false);
    let offRoutes = 0;
    for (const s of SCOPES) if (off.__routeDecide({ phase: s.phase, label: s.label }) !== null) offRoutes++;
    ok_(`${F.name}: OFF -> __routeDecide null on ALL scopes (byte-identical)`, offRoutes === 0);

    // (2) drift guard: inline ROUTE_TIER_MAP === canonical module TIER_MAP (keys carry U+0000 in both).
    ok_(`${F.name}: inline map matches canonical module (no drift)`,
        JSON.stringify(off.ROUTE_TIER_MAP) === JSON.stringify(CANON.TIER_MAP));
    // and the keys really use the NUL separator, not a space
    const someKey = Object.keys(off.ROUTE_TIER_MAP)[0];
    ok_(`${F.name}: scope keys use U+0000 separator (not a space)`, someKey.indexOf(NUL) >= 0 && someKey.indexOf(' ') < 0);

    // (3) ON + verifier present (Path B sim): mapped -> Sonnet + gated; unmapped -> null.
    let logsOnB = [];
    const onB = loadInline(src, { routing: 'true' }, {}, logsOnB, fsRequire({}));
    ok_(`${F.name}: routing ON when A.routing='true'`, onB.ROUTING_ON === true);
    let good = true, routed = 0;
    for (const s of SCOPES) {
      const d = onB.__routeDecide({ phase: s.phase, label: s.label });
      if (s.mapped) {
        if (!d || d.model !== 'claude-sonnet-5' || d.gated !== true || d.kind !== 'verbatim_write') good = false;
        if (d) routed++;
      } else if (d !== null) good = false;
    }
    ok_(`${F.name}: ON+verifier -> mapped route to Sonnet (gated), all others pinned`, good);
    ok_(`${F.name}: ON routed count == mapped count (${SCOPES.filter(s => s.mapped).length})`,
        routed === SCOPES.filter(s => s.mapped).length);

    // (4) ON + NO verifier (Path A sim: require throws): mapped -> decision.gated=false -> seam SUPPRESSES.
    const onA = loadInline(src, { routing: 'true' }, {}, [], /* requireImpl */ null);
    const dA = onA.__routeDecide({ phase: 'Validate', label: 'file_writer:persist:workflow-return' });
    ok_(`${F.name}: ON+no-verifier (Path A) -> mapped decision has gated=false (seam will suppress)`,
        dA && dA.model === 'claude-sonnet-5' && dA.gated === false);
    // the un-gateable cheap attempt must never be ACCEPTED by the validator
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
    let logsEsc = [];
    const esc = loadInline(src, { routing: 'true' }, {}, logsEsc, fsRequire(disk), agentImpl);
    const dEsc = esc.__routeDecide({ phase: 'Validate', label: 'file_writer:persist:workflow-return' });
    const rEsc = await esc.__routeEscalate(PROMPT, { phase: 'Validate', label: 'file_writer:persist:workflow-return' }, dEsc);
    ok_(`${F.name}: escalate made 2 calls, cheap(Sonnet) then strong(Opus)`,
        calls.length === 2 && calls[0] === 'claude-sonnet-5' && calls[1] === CANON.MODEL_STRONG);
    ok_(`${F.name}: escalate recorded BOTH attempts (fail then pass)`,
        esc.__routeAttempts.length === 2 && esc.__routeAttempts[0].ok === false &&
        esc.__routeAttempts[1].ok === true && esc.__routeAttempts[1].escalated === true);
    ok_(`${F.name}: escalate returned the strong result (final artifact matches)`,
        rEsc && rEsc.path === EXPECT_PATH && disk[EXPECT_PATH] === EXPECT_CONTENT);

    // (6) PILOT REACHABILITY: does the entry point actually CALL a mapped scope? Scan the file text
    // OUTSIDE the inline routing region for the mapped label literals.
    const regionA = src.indexOf('// <<ROUTING-INLINE-START>>');
    const regionB = src.indexOf('// <<ROUTING-INLINE-END>>') + '// <<ROUTING-INLINE-END>>'.length;
    const outside = src.slice(0, regionA) + src.slice(regionB);
    const reached = MAPPED.filter(m => outside.indexOf(m.label) >= 0).map(m => m.label);
    const isReachable = reached.length >= 1;
    console.log(`  reachable mapped labels in ${F.name}: ${reached.length ? reached.join(', ') : '(none)'}`);
    if (F.expectReachable) {
      ok_(`${F.name}: >=1 mapped scope is an ACTUAL call site (pilot can demonstrate routing)`, isReachable);
    } else {
      ok_(`${F.name}: ZERO mapped call sites here — documented; a standalone ${F.name} A/B routes nothing`, !isReachable);
      if (!isReachable) console.log(`  NOTE: ${F.name} has no eligible scope yet. A knn/kernel-lane GPU A/B would route nothing — route the pilot through e2e OR qualify a kernel_lane helper first (Astra CRITICAL #1).`);
    }
  }

  console.log(`\n${fails === 0 ? 'DRY-RUN PASS' : 'DRY-RUN FAIL'} — ${n - fails}/${n} checks`);
  process.exit(fails === 0 ? 0 : 1);
})();
