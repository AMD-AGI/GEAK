// Offline tests for the cost ladder's Eikos decider (route_decider=eikos). No model, no GPU, no network
// beyond loopback: a fake Eikos server answers on 127.0.0.1, and a fake carrier runs the exact command
// the lane builds -- the real eikos_decide.py -- and relays its JSON, optionally tampered with.
//
//   node kernel_workflow/scripts/test_eikos_route.js
//
// Contract under test: research/eikos_ladder_routing_proposal_20261009.md rev 2, section 7.
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const { execFile } = require('child_process');

const KDIR = path.resolve(__dirname, '..');
const ROOT = path.resolve(KDIR, '..');
const SRC = fs.readFileSync(path.join(KDIR, 'kernel_lane.js'), 'utf8');
const SPEC = JSON.parse(fs.readFileSync(path.join(ROOT, 'e2e_workflow/scripts/eikos_questions/lane_sufficiency.json'), 'utf8'));
const H5 = 'claude-haiku-5-5', S5 = 'claude-sonnet-5-5', O46 = 'claude-opus-4-6', O55 = 'claude-opus-5-5';

let n = 0, fails = 0;
function ok(name, cond) { n++; if (cond) console.log(`PASS  ${name}`); else { fails++; console.log(`FAIL  ${name}`); } }

function region(a, b) {
  const i = SRC.indexOf(a), j = SRC.indexOf(b);
  if (i < 0 || j < 0) throw new Error(`sentinels ${a} / ${b} not found`);
  return SRC.slice(i, j + b.length);
}
const ROUTING = region('// <<ROUTING-INLINE-START>>', '// <<ROUTING-INLINE-END>>');
const SHADOW = region('// <<EIKOS-SHADOW-START>>', '// <<EIKOS-SHADOW-END>>');
const AGENT_T = SRC.slice(SRC.indexOf('async function agentT(p, o) {'), SRC.indexOf('async function agentTCore(p, o) {'));

// ---- fake Eikos server -------------------------------------------------------------------------
const QID_LANE = {};
Object.keys(SPEC.estimate_questions).forEach((lane) => { QID_LANE[SPEC.estimate_questions[lane]] = lane; });
const server = { probs: {}, seen: [], raw: null };
const TMP = [];
function answer(p) { return { type: 'noul', probability: p, noul: p, value: p >= 0.5, confidence: Math.max(p, 1 - p) }; }
const srv = http.createServer((req, res) => {
  let body = '';
  req.on('data', (c) => { body += c; });
  req.on('end', () => {
    const sent = JSON.parse(body);
    server.seen.push(sent);
    const answers = {};
    Object.keys(sent.questions).forEach((q) => { answers[q] = answer(server.probs[QID_LANE[q]]); });
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(server.raw != null ? server.raw : JSON.stringify({ answers }));
  });
});

// ---- the lane's routing region, the shadow helpers and agentT, with stubs -------------------------
function lane(args, opts) {
  opts = opts || {};
  const logs = [], carried = [], workers = [];
  const evalDir = opts.evalDir || fs.mkdtempSync(path.join(os.tmpdir(), 'eikos-route-'));
  TMP.push(evalDir);
  const url = `http://127.0.0.1:${srv.address().port}`;
  async function agent(prompt, o) {
    if (String(o.label).indexOf('eikos:route') === 0) {
      carried.push({ prompt, o });
      if (opts.carrier) return opts.carrier(prompt, o, run);
      return run(prompt);
    }
    workers.push({ prompt, o });
    return opts.worker ? opts.worker(prompt, o) : { ok: true };
  }
  function run(prompt) {
    const cmd = /```bash\n([\s\S]*?)\n```/.exec(prompt)[1];
    return new Promise((resolve, reject) => {
      execFile('bash', ['-c', cmd], { env: Object.assign({}, process.env, { GEAK_EIKOS_URL: url }), timeout: 60000 },
        (err, stdout) => (err ? reject(err) : resolve(JSON.parse(stdout))));
    });
  }
  const names = ['__routePick', '__routeOutcome', '__routeReduce', '__routeReport', '__routeCard', '__routeScopeClass',
    '__routeParseEstimates', '__routeLadder', '__routeAudit', '__routeTasks', '__routeSkipped', 'ROUTE_EIKOS_ON',
    'ROUTE_EIKOS_TAU', 'ROUTE_PRICE_ORDER', 'ROUTE_LANES', 'ROUTE_CLASS_FLOOR', 'agentT', 'eikosSha256',
    'ROUTE_DECIDER'];
  const ret = 'return {' + names.map((s) => `${s}: (typeof ${s} !== 'undefined' ? ${s} : undefined)`).join(', ') +
    ', setClock: (v) => { __routeClock = v; }, topUsed: () => __routeTopUsed, consults: () => __routeConsults };';
  const f = new Function('A', 'log', 'process', 'require', 'setTimeout', 'agent', 'budget', 'tlAgent', 'WORKFLOW_DIR',
    'EVAL_DIR', 'DEADLINE_EPOCH', 'dispatched', 'BUDGET', 'agentTCore', 'cumulative', 'bestSeen', 'noImprove',
    'MAX_NO_IMPROVE', 'NO_STOP_S', 'forcedReplans', 'MAX_FORCED_REPLANS', 'MIN_IMPROVE', 'PROGRESS_DELTA',
    SHADOW + '\n' + ROUTING + '\n' + AGENT_T + '\n' + ret);
  const coreCalls = [];
  const api = f(Object.assign({ routing: 'true', route_decider: 'eikos' }, args || {}), (m) => logs.push(String(m)),
    { env: {} }, () => { throw new Error('no require'); }, opts.setTimeout || setTimeout, agent, undefined, null, KDIR,
    evalDir, opts.deadline || 0, 3, 6, async (p, o) => { coreCalls.push({ p, o }); return agent(p, o); },
    1, 1, 0, 2, 600, 0, 3, 0.01, 0.01);
  return Object.assign(api, { logs, carried, workers, evalDir, coreCalls });
}
const W = (label, phase) => ({ phase: phase || 'Optimize', label });
const ENG = (id, spec) => `eng ${id}:${spec || 'memory'}`;
function direction(L, id, spec, extra) {
  L.__routeTasks[ENG(id, spec)] = Object.assign({ id, title: 'tile the K loop', specialty: spec || 'memory',
    expected_speedup: 1.3, focus_files: ['kernel.py'], prompt: 'Tile K by 64.' }, extra || {});
}
const ROLE_PROMPT = 'You are the benchmark_engineer. PHASE=setup.\nFirst Read x and follow it.\n' +
  'Do all filesystem/shell work yourself (Bash/Read/Write). Build the COMMANDMENT and record a reliable baseline.\n\n## Inputs\n{}';

async function main() {
  await new Promise((r) => srv.listen(0, '127.0.0.1', r));
  const last = (L, ev) => L.__routeAudit.filter((e) => e.event === ev).slice(-1)[0];

  // ---- the pure reducer ---------------------------------------------------------------------------
  {
    const L = lane();
    const R = (x) => L.__routeReduce(Object.assign({ cls: 'code_change', tau: 0.9, current: null, floorModel: S5,
      opFloorModel: null, failed: null, failsOnFailed: 0, maxFails: 1, backend: 'ok', estimates: null, quotaLeft: 1,
      available: [H5, S5, O46, O55] }, x));
    ok('price order is declared, not the lane order: Opus 4.6 is dearer than Opus 5.5',
      JSON.stringify(L.ROUTE_PRICE_ORDER) === JSON.stringify([H5, S5, O55, O46]) &&
      L.ROUTE_PRICE_ORDER.slice().sort().join() === L.ROUTE_LANES.slice().sort().join());
    let r = R({ cls: 'paperwork', tau: 0.7, floorModel: H5, estimates: { [H5]: 0.95, [S5]: 0.5, [O55]: 0.92, [O46]: 0.99 } });
    ok('non-monotone estimates are taken as given: the cheapest qualifying lane wins', r.model === H5 && !r.override);
    r = R({ estimates: { [H5]: 0.99, [S5]: 0.91, [O55]: 0.97, [O46]: 0.99 } });
    ok('a lane below the class floor is never a candidate, however sure Eikos is', r.model === S5 && r.candidates.indexOf(H5) < 0);
    r = R({ estimates: { [H5]: 0.25, [S5]: 0.25, [O55]: 0.25, [O46]: 0.25 } });
    ok('uniform uncertainty qualifies no lane: a recorded host override, not a "sufficient" top lane',
      r.qualifying.length === 0 && r.override && r.override.kind === 'no_lane_qualified' && r.override.tau === 0.9);
    r = R({ estimates: { [H5]: 0.2, [S5]: 0.4, [O55]: 0.85, [O46]: 0.6 } });
    ok('with no lane qualifying, the override takes the highest estimate, not the dearest lane', r.model === O55);
    r = R({ current: O55, backend: 'timeout' });
    ok('a missing answer keeps an escalated scope where it is (never back to the floor)', r.model === O55 && /no usable estimate \(timeout\)/.test(r.reason));
    r = R({ backend: 'ineligible' });
    ok('a missing answer on first sight takes the class floor', r.model === S5);
    r = R({ failed: S5, failsOnFailed: 1, current: S5, estimates: { [H5]: 1, [S5]: 1, [O55]: 0.95, [O46]: 0.99 } });
    ok('a code change climbs at once after a work-product failure: the failed lane is out', r.model === O55 && r.excluded[0] === S5);
    r = R({ cls: 'evidence', tau: 0.8, failed: S5, failsOnFailed: 1, maxFails: 2, current: S5, estimates: { [H5]: 0, [S5]: 0.9, [O55]: 0.9, [O46]: 0.9 } });
    ok('evidence work may retry once on the failed lane', r.model === S5 && r.excluded.length === 0);
    r = R({ quotaLeft: 0, estimates: { [H5]: 0, [S5]: 0.1, [O55]: 0.99, [O46]: 0.95 } });
    ok('a used-up top-lane quota is filtered before selection (no substitution afterwards)',
      r.model === O46 && r.filtered.some((x) => x.model === O55 && /quota/.test(x.why)));
    r = R({ current: O46, failed: O46, failsOnFailed: 1, quotaLeft: 0, estimates: { [H5]: 1, [S5]: 1, [O55]: 1, [O46]: 1 } });
    ok('quota against a hard floor: no feasible lane is a deferral, not a cheaper lane', r.model === null && /no feasible lane/.test(r.reason));
    r = R({ available: [H5], estimates: { [H5]: 1, [S5]: 1, [O55]: 1, [O46]: 1 } });
    ok('no available lane at or above the floor is a deferral', r.model === null);
    r = R({ opFloorModel: O55, estimates: { [H5]: 1, [S5]: 1, [O55]: 0.95, [O46]: 1 } });
    ok('an operator floor raises the lower bound', r.model === O55);
  }

  // ---- card, transport and the whole decision ------------------------------------------------------
  {
    server.probs = { [H5]: 0.31, [S5]: 0.93, [O46]: 0.95, [O55]: 0.97 };
    server.seen = [];
    // A quota of 5 top-lane dispatches keeps this sequence about failure kinds; the quota has its own test.
    const L = lane({ route_max_top_escalations: '5' }, { deadline: 1 });
    L.setClock({ left_s: 5400 });
    direction(L, 'r2_d1');
    const pick = await L.__routePick('optimize prompt', W(ENG('r2_d1'), 'Optimize'));
    const dec = last(L, 'decide');
    ok('the default decider is still Sonnet; eikos only when chosen', lane({ route_decider: undefined }).ROUTE_EIKOS_ON === false && L.ROUTE_EIKOS_ON === true);
    ok('first sight: one carrier call, verified relay, cheapest lane at or above tau', pick.model === S5 && L.carried.length === 1 &&
      dec.backend === 'ok' && dec.relay_ok && dec.decision_ok && dec.receipt_ok && dec.trigger === 'first_sight');
    ok('the carrier is Haiku 5.5, unrouted, labelled as router cost', L.carried[0].o.model === H5 &&
      /^You are the route_carrier\. PHASE=Optimize\./.test(L.carried[0].prompt) && L.workers.length === 0);
    const card = JSON.parse(server.seen[0].state);
    ok('the card carries the direction, acceptance, budget and lane configuration', card.task.direction.id === 'r2_d1' &&
      card.task.goal_source === 'direction' && /verify/.test(card.task.acceptance) && card.budget.minutes_left === 90 &&
      card.budget.minutes_left_source === 'round_clock' && card.round === 2 && card.scope_class === 'code_change' &&
      JSON.stringify(card.config.lanes) === JSON.stringify(L.ROUTE_LANES));
    ok('the card holds a digest of the prompt, never the prompt and never a routing answer',
      card.task.prompt_sha256 === L.eikosSha256('optimize prompt') && card.task.prompt_chars === 15 &&
      JSON.stringify(card).indexOf('optimize prompt') < 0 && !('answer' in card) && !('estimates' in card));
    ok('Eikos was asked the four lane questions from the spec', JSON.stringify(server.seen[0].questions) === JSON.stringify(SPEC.questions));
    const again = await L.__routePick('optimize prompt', W(ENG('r2_d1')));
    ok('the same task does not ask again', again.model === S5 && L.carried.length === 1);

    // replay: a fresh lane on the same EVAL_DIR asks with the same card and gets the persisted estimates
    server.probs = { [H5]: 0.99, [S5]: 0.99, [O46]: 0.99, [O55]: 0.99 };
    const L2 = lane({ route_max_top_escalations: '5' }, { deadline: 1, evalDir: L.evalDir });   // same EVAL_DIR, as on resume
    L2.setClock({ left_s: 5400 });
    direction(L2, 'r2_d1');
    const p2 = await L2.__routePick('optimize prompt', W(ENG('r2_d1')));
    const d2 = last(L2, 'decide');
    ok('a resumed lane re-asking the same card gets the persisted decision (same key, same estimates)',
      p2.model === S5 && d2.logical_key === dec.logical_key && d2.estimates[H5] === 0.31);

    // a changed task in the same ladder is a new occurrence
    server.probs = { [H5]: 0.31, [S5]: 0.5, [O46]: 0.92, [O55]: 0.91 };
    direction(L, 'r3_d0', 'memory', { title: 'vectorize the epilogue' });
    const p3 = await L.__routePick('another prompt', W(ENG('r3_d0')));
    const d3 = last(L, 'decide');
    ok('a new direction in the same ladder is reassessed: a new key, changed_task, never below the current lane',
      L.carried.length === 2 && d3.trigger === 'changed_task' && d3.logical_key !== dec.logical_key && d3.seq === 2 && p3.model === O55);

    // failure kinds
    L.__routeOutcome('eng:memory', false, 'verify status=correctness_failed', 'invalid_measurement', 'verify r3_d0');
    await L.__routePick('another prompt', W(ENG('r3_d0')));
    ok('an invalid measurement is recorded but changes nothing and asks nothing', L.carried.length === 2 &&
      last(L, 'fail').kind === 'invalid_measurement' && last(L, 'fail').reconsult === false);
    L.__routeOutcome('eng:memory', false, 'engineer returned nothing', 'no_result', 'engineer r3_d0');
    await L.__routePick('another prompt', W(ENG('r3_d0')));
    ok('a missing result is not evidence about the lane either', L.carried.length === 2 && L.__routeLadder['eng:memory'].model === O55);
    server.probs = { [H5]: 0.1, [S5]: 0.2, [O46]: 0.95, [O55]: 0.97 };
    L.__routeOutcome('eng:memory', false, 'verify status=apply_failed', 'work_product', 'verify r3_d0');
    const p4 = await L.__routePick('another prompt', W(ENG('r3_d0')));
    const c4 = JSON.parse(server.seen[server.seen.length - 1].state);
    ok('a work-product failure re-asks with the evidence and climbs past the failed lane', p4.model === O46 &&
      c4.occurrence.trigger === 'work_product_failure' && c4.occurrence.prior.slice(-1)[0].outcome === 'work_product' &&
      c4.occurrence.prior.slice(-1)[0].source === 'verify r3_d0' && last(L, 'decide').excluded[0] === O55);
    L.__routeOutcome('eng:memory', false, 'verify status=apply_failed', 'work_product', 'verify r3_d0');
    const p5 = await L.__routePick('another prompt', W(ENG('r3_d0')));
    ok('retries after escalation stop at the last lane: a deferral, recorded, not a silent cheaper lane',
      p5.skip === true && p5.deferred === true && /deferred/.test(last(L, 'skip').why));
    const t = await L.agentT('another prompt', W(ENG('r3_d0')));
    ok('agentT dispatches nothing for a deferred worker and marks it skipped', t === null && L.__routeSkipped[ENG('r3_d0')] === true &&
      L.logs.some((m) => /no feasible lane \(deferred\)/.test(m)));
    const rep = L.__routeReport();
    ok('the report names the Eikos decider and its thresholds as unvalidated experimental settings', rep.decider === 'eikos' &&
      rep.eikos.tau_status === 'unvalidated_experimental' && rep.eikos.tau.code_change === 0.9 && rep.eikos.consults === 4);
  }

  {
    // non-improvement asks again without excluding the lane
    server.probs = { [H5]: 0.2, [S5]: 0.92, [O46]: 0.95, [O55]: 0.97 };
    const L = lane();
    direction(L, 'r1_d0', 'algorithm');
    await L.__routePick('p', W(ENG('r1_d0', 'algorithm')));
    L.__routeOutcome('eng:algorithm', false, 'engineer reported 0.98x <= floor 1', 'non_improvement', 'engineer r1_d0');
    const p = await L.__routePick('p', W(ENG('r1_d0', 'algorithm')));
    ok('a non-improvement asks again but keeps the lane available', L.carried.length === 2 && p.model === S5 &&
      last(L, 'decide').trigger === 'non_improvement' && last(L, 'decide').excluded.length === 0);
  }

  // ---- what reaches the host when the relay is bad --------------------------------------------------
  async function relayCase(name, carrier, expect, extra) {
    server.probs = { [H5]: 0.99, [S5]: 0.99, [O46]: 0.99, [O55]: 0.99 };
    const L = lane(extra && extra.args, Object.assign({ carrier }, extra && extra.opts));
    direction(L, 'r1_d0');
    const pick = await L.__routePick('p', W(ENG('r1_d0')));
    const d = last(L, 'decide');
    ok(name, d.backend === expect && pick.model === S5 && d.estimates === null);
    return L;
  }
  await relayCase('altered estimates in the relay are caught (decision digest), host falls back to the floor',
    async (p, o, run) => Object.assign(await run(p), { estimates: 'claude-haiku-5-5=1.0;claude-sonnet-5-5=1.0;claude-opus-4-6=1.0;claude-opus-5-5=1.0' }),
    'decision_altered');
  await relayCase('a different state reached the script: relay mismatch, host falls back',
    async (p, o, run) => Object.assign(await run(p), { state_sha256: '0'.repeat(64) }), 'relay_mismatch');
  await relayCase('a carrier that throws is a backend error, not a worker failure',
    async () => { throw new Error('API 529'); }, 'error');
  await relayCase('a carrier that never returns times out on its own bound',
    () => new Promise(() => {}), 'timeout', { opts: { setTimeout: (fn) => setTimeout(fn, 5) } });
  await relayCase('a carrier that returns nothing is no result', async () => null, 'no_result');
  {
    server.raw = JSON.stringify({ answers: { passes_on_haiku_5_5: answer(1.7) } });
    const L = lane();
    direction(L, 'r1_d0');
    const pick = await L.__routePick('p', W(ENG('r1_d0')));
    server.raw = null;
    ok('an unusable Eikos answer is malformed (status relayed intact), host falls back', last(L, 'decide').backend === 'malformed' &&
      last(L, 'decide').decision_ok === true && pick.model === S5);
  }
  {
    const L = lane();
    direction(L, 'r1_d0', 'memory', { prompt: 'x'.repeat(4001) });
    const pick = await L.__routePick('p', W(ENG('r1_d0')));
    ok('a direction too long for the card is ineligible: Eikos is not asked, nothing is truncated',
      L.carried.length === 0 && last(L, 'decide').backend === 'ineligible' && /direction/.test(last(L, 'decide').ineligible) && pick.model === S5);
  }
  {
    server.probs = { [H5]: 0.99, [S5]: 0.99, [O46]: 0.99, [O55]: 0.99 };
    const L = lane({ route_eikos_max_consults: '1' });
    await L.__routePick(ROLE_PROMPT, W('benchmark_engineer', 'Benchmark'));
    await L.__routePick('p', W('commit r1', 'Merge'));
    ok('the consultation budget bounds carrier calls; the rest fall back', L.carried.length === 1 &&
      last(L, 'decide').backend === 'consult_budget_exhausted' && L.__routeLadder.commit.model === H5);
    const c = JSON.parse(server.seen[server.seen.length - 1].state);
    ok('a role prompt\'s own goal sentence is the card\'s goal', c.task.goal_source === 'role_intro' &&
      c.task.goal === 'Build the COMMANDMENT and record a reliable baseline.' && c.scope_class === 'evidence' && c.round === 0);
  }
  {
    // quota used by another scope between decision and dispatch
    server.probs = { [H5]: 0.1, [S5]: 0.5, [O46]: 0.95, [O55]: 0.97 };
    const L = lane();
    direction(L, 'r1_d0', 'memory'); direction(L, 'r1_d1', 'compute');
    const a = await L.__routePick('p', W(ENG('r1_d0', 'memory')));
    const b = await L.__routePick('p', W(ENG('r1_d1', 'compute')));
    ok('the top-lane quota is honoured for the second scope by filtering, before its choice', a.model === O55 && b.model === O46 &&
      last(L, 'decide').filtered.some((x) => x.model === O55));
    L.__routeLadder['eng:memory'].task_key = 'direction:r1_d0';
    const again = await L.__routePick('p', W(ENG('r1_d0', 'memory')));
    ok('a decided top lane whose quota ran out is re-reduced and recorded as an override', again.model === O46 &&
      last(L, 'override').kind === 'quota_used_before_dispatch' && last(L, 'override').from === O55);
  }
  {
    // fixed scopes, model and effort overrides through agentT
    server.probs = { [H5]: 0.95, [S5]: 0.95, [O46]: 0.95, [O55]: 0.95 };
    const L = lane();
    await L.agentT('p', W('tech_lead:plan r1'));
    await L.agentT('p', W('clock r1'));
    ok('brain and fixed helpers are never sent to Eikos', L.carried.length === 0 &&
      L.coreCalls[0].o.model === O55 && L.coreCalls[1].o.model === H5);
    await L.agentT('p', Object.assign(W('kb:cite', 'Validate'), { effort: 'low' }));
    const c = JSON.parse(server.seen[server.seen.length - 1].state);
    ok('agentT applies the decided model without mutating the caller\'s options, and the card records effort',
      L.coreCalls[2].o.model === H5 && L.coreCalls[2].o.effort === 'low' && c.config.effort === 'low' && c.scope_class === 'paperwork');
  }
  {
    const L = lane({ route_eikos_tau: '{"code_change": 0.5, "evidence": 2, "paperwork": "x"}' });
    ok('thresholds are per-run settings; invalid values keep the built-in experimental ones',
      L.ROUTE_EIKOS_TAU.code_change === 0.5 && L.ROUTE_EIKOS_TAU.evidence === 0.8 && L.ROUTE_EIKOS_TAU.paperwork === 0.7);
    ok('scope classes: code changes, evidence, paperwork, and the strictest class for anything unknown',
      L.__routeScopeClass('eng:memory') === 'code_change' && L.__routeScopeClass('integrate') === 'code_change' &&
      L.__routeScopeClass('verify') === 'evidence' && L.__routeScopeClass('kb:write') === 'evidence' &&
      L.__routeScopeClass('kb:cite') === 'paperwork' && L.__routeScopeClass('commit') === 'paperwork' &&
      L.__routeScopeClass('mystery') === 'code_change');
  }

  // ---- the standalone and nested paths forward every setting ---------------------------------------
  const KEYS = ['route_decider', 'route_eikos_tau', 'route_eikos_max_consults', 'route_eikos_carrier_model',
    'route_eikos_carrier_timeout_ms'];
  const kw = fs.readFileSync(path.join(KDIR, 'kernel_workflow.js'), 'utf8');
  const e2e = fs.readFileSync(path.join(ROOT, 'e2e_workflow/e2e_workflow.js'), 'utf8');
  const e2eLane = e2e.slice(e2e.indexOf('const laneArgs = (wfArgs) => {'), e2e.indexOf('return out;', e2e.indexOf('const laneArgs = (wfArgs) => {')));
  ok('kernel_workflow.js forwards the Eikos routing settings to each lane', KEYS.every((k) => kw.indexOf(`'${k}'`) >= 0));
  ok('e2e_workflow.js forwards them to its nested kernel lanes', KEYS.every((k) => e2eLane.indexOf(`'${k}'`) >= 0));
  ok('the lane reads every forwarded setting', KEYS.every((k) => ROUTING.indexOf(`A.${k}`) >= 0));

  srv.close();
  TMP.forEach((d) => fs.rmSync(d, { recursive: true, force: true }));
  console.log(fails ? `FAILED — ${fails}/${n} checks failed` : `ALL PASS — ${n}/${n} checks`);
  process.exit(fails ? 1 : 0);
}
main().catch((e) => { console.error(e); process.exit(2); });
