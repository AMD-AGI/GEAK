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
//   * kernel_lane.js — the four-lane COST LADDER (policy: routing/SKILL.md; no verbatim cascade). Proves:
//       (1) OFF -> __routeModel undefined on EVERY scope, and agentT() short-circuits (byte-identical run).
//       (2) inline ROUTE_TIER_MAP is exactly the 3 Haiku helper scopes; ROUTE_LANES == canonical LANES.
//       (3) ON static: helpers -> Haiku 5.5, brain -> Opus 5.5, decider -> Sonnet 5.5, workers -> classified.
//       (4) the ladder: first-sight gate, retry vs escalate after a failure, top-lane cap, floors,
//           kill switch — each driven through the SHIPPED __routePick / __routeOutcome.
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
const CHEAP = 'claude-sonnet-5-5';             // the e2e cascade's routed model
const HAIKU = 'claude-haiku-5-5';              // the kernel ladder's SMALL lane / helper model

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
    // Workers: no static model (they are classified). Includes a helper label in the WRONG phase.
    unmapped: [
      { phase: 'Optimize', label: 'eng d1:memory' },
      { phase: 'WarmStart', label: 'clock' },        // right label, WRONG phase -> a worker, not a helper
      { phase: '', label: 'agent' },
    ],
    // The canonical inline kernel map (drift guard): exactly these 3 Haiku helper scopes, nothing else.
    expectMap: (() => {
      const m = {};
      m['Optimize' + NUL + 'clock'] = HAIKU;
      m['Optimize' + NUL + 'storage:reclaim'] = HAIKU;
      m['WarmStart' + NUL + 'warm_start:resolve'] = HAIKU;
      return m;
    })(),
  },
];

// A dynamic-suffix variant that MUST still route (same static prefix). Kept out of `unmapped` above.
const E2E_DYNAMIC_MAPPED = { phase: 'WarmStart', label: 'warm_start:record-measurements r2' };

// Extract the sentinel region and eval it, injecting the globals it closes over. Symbol-TOLERANT: the
// two entry points export different routing symbols (cascade vs model-map), so the return probes each
// with typeof and yields undefined for any a given file does not define — no ReferenceError either way.
function loadInline(src, argsObj, envObj, logs, requireImpl, agentImpl, budgetObj) {
  const A_START = '// <<ROUTING-INLINE-START>>';
  const A_END = '// <<ROUTING-INLINE-END>>';
  const i = src.indexOf(A_START);
  const j = src.indexOf(A_END);
  if (i < 0 || j < 0) throw new Error('inline sentinels not found in ' + (src.length) + ' bytes');
  const block = src.slice(i, j + A_END.length);
  const want = ['__routeDecide', '__routeEscalate', '__routeCheckVerbatim', '__routeValidate',
    '__routeExpected', '__routeModel', '__routeLabelPrefix', 'ROUTING_ON', 'ROUTE_TIER_MAP',
    '__routeAttempts', '__routePick', '__routeOutcome', '__routeStep', '__routeScope', '__routeLadder',
    '__routeAudit', '__routeReport', 'ROUTE_LANES', 'ROUTE_DECISION_SCHEMA'];
  const ret = 'return {' + want.map((s) => s + ': (typeof ' + s + " !== 'undefined' ? " + s + ' : undefined)').join(', ') + '};';
  const factory = new Function(
    'A', 'log', 'process', 'require', 'setTimeout', 'agent', '__routeAgentTimeoutMs', 'budget',
    block + '\n' + ret);
  const fakeProcess = { env: envObj || {} };
  const req = requireImpl || function () { throw new Error('no require (Path A)'); };
  const timeout = function () { return 0; };
  return factory(argsObj || {}, (m) => logs.push(String(m)), fakeProcess, req, setTimeout, agentImpl || (async () => null), timeout, budgetObj);
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
  const OPUS55 = 'claude-opus-5-5', OPUS46 = 'claude-opus-4-6';
  // (1) OFF: __routeModel undefined for EVERY scope, and agentT() hands straight to the unchanged core.
  const off = loadInline(src, {}, {}, [], fsRequire({}));
  ok_(`${F.name}: routing OFF by default`, off.ROUTING_ON === false);
  ok_(`${F.name}: ladder seam present (__routeModel/__routePick), cascade API absent (by design)`,
      typeof off.__routeModel === 'function' && typeof off.__routePick === 'function' &&
      off.__routeDecide === undefined && off.__routeEscalate === undefined);
  let offRoutes = 0;
  for (const s of F.mapped.concat(F.unmapped).concat([{ phase: 'Optimize', label: 'tech_lead:plan r1' }]))
    if (off.__routeModel({ phase: s.phase, label: s.label }) !== undefined) offRoutes++;
  ok_(`${F.name}: OFF -> __routeModel undefined on ALL scopes`, offRoutes === 0);
  ok_(`${F.name}: OFF -> agentT() short-circuits to agentTCore before any routing (byte-identical)`,
      /async function agentT\(p, o\) \{\n  if \(!ROUTING_ON\) return agentTCore\(p, o\);/.test(src));

  // (2) drift guards.
  ok_(`${F.name}: inline helper map == 3 Haiku helper scopes (no drift)`,
      JSON.stringify(off.ROUTE_TIER_MAP) === JSON.stringify(F.expectMap));
  ok_(`${F.name}: inline ROUTE_LANES == canonical tier_map LANES`,
      JSON.stringify(off.ROUTE_LANES) === JSON.stringify(CANON.LANES));
  ok_(`${F.name}: top lane is GEAK's default model (tier_map MODEL_STRONG)`, off.ROUTE_LANES[3] === CANON.MODEL_STRONG);
  const someKey = Object.keys(off.ROUTE_TIER_MAP)[0];
  ok_(`${F.name}: scope keys use U+0000 separator (not a space)`, someKey.indexOf(NUL) >= 0 && someKey.indexOf(' ') < 0);

  // (3) ON, static part.
  const on = loadInline(src, { routing: 'true' }, {}, []);
  ok_(`${F.name}: routing ON when A.routing='true'`, on.ROUTING_ON === true);
  let good = true;
  const dyn = [{ phase: 'Optimize', label: 'clock r1' }, { phase: 'Optimize', label: 'storage:reclaim r3' }];
  for (const s of F.mapped.concat(dyn)) if (on.__routeModel(s) !== HAIKU) good = false;
  ok_(`${F.name}: ON -> every helper (incl. dynamic suffix) -> Haiku`, good);
  ok_(`${F.name}: ON -> brain (tech_lead:*, director:*) -> Opus 5.5`,
      on.__routeModel({ phase: 'Optimize', label: 'tech_lead:plan r2' }) === OPUS55 &&
      on.__routeModel({ phase: 'Validate', label: 'director:validate' }) === OPUS55);
  ok_(`${F.name}: ON -> decider (route:*) -> Sonnet 5.5, never classified itself`,
      on.__routeModel({ phase: 'Optimize', label: 'route:classify eng:memory' }) === CHEAP);
  ok_(`${F.name}: ON -> workers have no static model (they get classified)`,
      F.unmapped.every((s) => on.__routeModel(s) === undefined));
  ok_(`${F.name}: ladder scope identity`,
      on.__routeScope('eng d3:memory') === 'eng:memory' && on.__routeScope('eng d7:memory') === 'eng:memory' &&
      on.__routeScope('deep d2:deep_explore') === 'deep:deep_explore' && on.__routeScope('verify d3 (recovered)') === 'verify' &&
      on.__routeScope('researcher:q q1') === 'researcher:q' && on.__routeScope('integrate r2') === 'integrate');

  // A scripted decider: each classify call pops the next canned answer and records how it was asked.
  const dec = (c, conf, extra) => Object.assign({ complexity: c, action: 'continue', scope_drift: false,
                                                  confidence: conf, risk_flag: 'none' }, extra || {});
  function harness(args, answers, budgetObj) {
    const asked = [];
    const agentImpl = async (p, o) => { asked.push({ p, o }); return answers.length ? answers.shift() : null; };
    const L = loadInline(src, Object.assign({ routing: 'true' }, args || {}), {}, [], null, agentImpl, budgetObj);
    return { L, asked };
  }
  const W = (label) => ({ phase: 'Optimize', label });

  // (4a) first sight: the task must prove it needs more than SMALL.
  {
    const { L, asked } = harness({}, [dec('high', 0.40)]);
    const r1 = await L.__routePick('optimize the kernel', W('eng d1:memory'));
    const r2 = await L.__routePick('optimize again', W('eng d4:memory'));
    ok_(`${F.name}: unsure small lane (conf 0.40 < 0.70) -> classified lane (high -> Opus 4.6)`, r1.model === OPUS46);
    ok_(`${F.name}: decider asked ONCE per scope (same scope re-used, no re-classify on success)`, asked.length === 1 && r2.model === OPUS46);
    ok_(`${F.name}: decider call: route:classify label, low effort, typed schema (agentT pins route:* to Sonnet 5.5)`,
        asked[0].o.model === undefined && asked[0].o.label === 'route:classify eng:memory' &&
        asked[0].o.effort === 'low' && asked[0].o.schema === L.ROUTE_DECISION_SCHEMA);
    ok_(`${F.name}: decider prompt carries the ledger role header (its cost lands in the router bucket)`,
        /^You are the ([A-Za-z0-9_.\-]+)\.\s*PHASE=([A-Za-z0-9_.\-]+)\./.test(asked[0].p) &&
        asked[0].p.indexOf('You are the route_classifier. PHASE=Optimize.') === 0);
  }
  {
    const { L } = harness({}, [dec('high', 0.90)]);
    ok_(`${F.name}: confident small lane (conf 0.90) -> SMALL even when complexity says high`,
        (await L.__routePick('x', W('eng d1:compute'))).model === HAIKU);
  }
  {
    const { L } = harness({}, [null]);
    ok_(`${F.name}: no valid decider answer -> default SMALL`, (await L.__routePick('x', W('verify d1'))).model === HAIKU);
  }
  {
    const { L } = harness({}, [{ complexity: 'huge', confidence: 2 }]);
    ok_(`${F.name}: malformed decider answer -> default SMALL`, (await L.__routePick('x', W('verify d1'))).model === HAIKU);
  }

  // (4b) after a failure: retry on the lane unless a gate trips, then ONE step up.
  {
    const { L, asked } = harness({}, [dec('small', 0.95), dec('small', 0.95), dec('small', 0.95), dec('small', 0.95)]);
    await L.__routePick('x', W('eng d1:memory'));
    L.__routeOutcome('eng:memory', false, 'verify correctness=fail');
    const a = await L.__routePick('x', W('eng d2:memory'));
    ok_(`${F.name}: failure + confident decider -> retry on the same lane`, a.model === HAIKU && asked.length === 2);
    ok_(`${F.name}: re-classify prompt carries the compact failure evidence`, asked[1].p.indexOf('verify correctness=fail') >= 0);
    L.__routeOutcome('eng:memory', false, 'f2'); await L.__routePick('x', W('eng d3:memory'));
    L.__routeOutcome('eng:memory', false, 'f3');
    ok_(`${F.name}: 3rd failure on a lane -> escalate one lane regardless of confidence`,
        (await L.__routePick('x', W('eng d4:memory'))).model === CHEAP);
    ok_(`${F.name}: escalation reason recorded`, /3 failures on this lane/.test(L.__routeLadder['eng:memory'].why));
  }
  {
    const { L } = harness({}, [dec('small', 0.95), dec('small', 0.30)]);
    await L.__routePick('x', W('verify d1'));
    L.__routeOutcome('verify', false, 'no result');
    ok_(`${F.name}: failure + unsure decider (conf 0.30) -> escalate one lane, not a jump`,
        (await L.__routePick('x', W('verify d2'))).model === CHEAP);
  }
  {
    const { L } = harness({}, [dec('small', 0.95), dec('small', 0.95, { risk_flag: 'security' })]);
    await L.__routePick('x', W('integrate r1'));
    L.__routeOutcome('integrate', false, 'no result');
    ok_(`${F.name}: failure + risk_flag=security -> escalate`, (await L.__routePick('x', W('integrate r2'))).model === CHEAP);
  }
  {
    const { L } = harness({}, [dec('small', 0.95)]);
    await L.__routePick('x', W('commit r1'));
    L.__routeOutcome('commit', true, '');
    await L.__routePick('x', W('commit r2'));
    ok_(`${F.name}: success -> no re-classification, lane kept`, L.__routeAudit.filter((e) => e.event === 'decide').length === 1);
  }

  // (4c) caps and knobs.
  {
    const { L } = harness({}, [dec('escalate', 0.10), dec('escalate', 0.10)]);
    const a = await L.__routePick('x', W('eng d1:algo'));
    const b = await L.__routePick('x', W('eng d2:layout'));
    ok_(`${F.name}: first worker Opus 5.5 dispatch allowed, second capped to Opus 4.6 (max 1 per run)`,
        a.model === OPUS55 && b.model === OPUS46 && L.__routeAudit.some((e) => e.event === 'dispatch' && e.capped));
  }
  {
    const { L } = harness({ route_floors: JSON.stringify({ verify: 1 }) }, [dec('small', 0.99)]);
    ok_(`${F.name}: route_floors raises a scope's starting lane (operator-approved floor)`,
        (await L.__routePick('x', W('verify d1'))).model === CHEAP);
  }
  {
    const { L } = harness({ route_conf_escalate: '0.95' }, [dec('medium', 0.90)]);
    ok_(`${F.name}: thresholds are args, not constants (conf gate 0.95 -> 0.90 counts as unsure)`,
        (await L.__routePick('x', W('reprofile r1'))).model === CHEAP);
  }
  {
    // The session had already spent far more than the cap before this lane started (as observed live:
    // 4,727,204). The cap is on growth since the lane started, so routing must proceed normally.
    let spent = 4727204;
    const { L } = harness({ route_max_output_tokens: '1000000' }, [dec('small', 0.9)], { spent: () => spent });
    ok_(`${F.name}: cap counts growth since the lane started, not the session total`,
        (await L.__routePick('x', W('benchmark_engineer'))).model === HAIKU && L.__routeReport().killed === false &&
        L.__routeReport().session_output_tokens_at_start === 4727204);
  }
  {
    let spent = 10;
    const { L, asked } = harness({ route_max_output_tokens: '100' }, [dec('small', 0.9)], { spent: () => spent });
    const a = await L.__routePick('x', W('verify d1'));
    spent = 110;
    const b = await L.__routePick('x', W('verify d2'));
    const c = await L.__routePick('x', { phase: 'Optimize', label: 'tech_lead:report' });
    ok_(`${F.name}: under the output cap -> dispatch; at the cap -> worker skipped (kill switch)`,
        a.model === HAIKU && b.skip === true && asked.length === 1);
    ok_(`${F.name}: kill switch leaves brain dispatches running (report + validation are not lost)`, c.model === OPUS55);
    const rep = L.__routeReport();
    ok_(`${F.name}: report records the kill, thresholds and full audit`,
        rep.killed === true && rep.thresholds.max_output_tokens === 100 && rep.spent_output_tokens === 100 &&
        rep.audit.some((e) => e.event === 'kill_switch'));
  }
  {
    const { L } = harness({}, [dec('small', 0.9)]);   // no budget global at all (older runtime)
    ok_(`${F.name}: no budget API -> kill switch stays off, routing still works`,
        (await L.__routePick('x', W('verify d1'))).model === HAIKU && L.__routeReport().killed === false);
  }
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
  {
    const s = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js'), 'utf8');
    ok_('kernel_workflow.js: forwards every route_* knob the ladder reads',
        ['route_conf_escalate', 'route_max_retries_per_lane', 'route_max_top_escalations',
         'route_max_output_tokens', 'route_floors', 'route_decider'].every((k) => s.indexOf(`'${k}'`) >= 0));
  }

  console.log(`\n${fails === 0 ? 'DRY-RUN PASS' : 'DRY-RUN FAIL'} — ${n - fails}/${n} checks`);
  process.exit(fails === 0 ? 0 : 1);
})();
