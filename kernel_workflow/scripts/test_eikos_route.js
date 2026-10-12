// Offline tests for the cost ladder's Eikos decider (route_decider=eikos). No model, no GPU, no network
// beyond loopback: a fake Eikos server answers on 127.0.0.1, and a fake carrier runs the exact command
// the lane builds -- the real eikos_decide.py -- and relays its JSON, optionally tampered with.
//
//   node kernel_workflow/scripts/test_eikos_route.js
//
// Contract under test: research/eikos_ladder_routing_proposal_20261009.md rev 2, section 7, and the
// corrections of Astra's S0 plan review (2026-10-12); her counterexamples are regression cases here.
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
const P = (h, s, o46, o55) => ({ [H5]: h, [S5]: s, [O46]: o46, [O55]: o55 });

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
    '__routeParseEstimates', '__routeHostInput', '__routeLadder', '__routeAudit', '__routeTasks', '__routeSkipped',
    '__routeDispatches', '__routeConfigErrors', 'ROUTE_EIKOS_ON', 'ROUTE_EIKOS_TAU', 'ROUTE_EIKOS_MAX_CONSULTS',
    'ROUTE_PRICE_ORDER', 'ROUTE_LANES', 'ROUTE_CLASS_FLOOR', 'agentT', 'eikosSha256', 'eikosCanonical',
    'EIKOS_DECISION_FIELDS', 'ROUTE_DECIDER'];
  const ret = 'return {' + names.map((s) => `${s}: (typeof ${s} !== 'undefined' ? ${s} : undefined)`).join(', ') +
    ', setClock: (v) => { __routeClock = v; }, topUsed: () => __routeTopUsed, consults: () => __routeConsults };';
  const f = new Function('A', 'log', 'process', 'require', 'setTimeout', 'agent', 'budget', 'tlAgent', 'WORKFLOW_DIR',
    'EVAL_DIR', 'DEADLINE_EPOCH', 'dispatched', 'BUDGET', 'agentTCore', 'cumulative', 'bestSeen', 'noImprove',
    'MAX_NO_IMPROVE', 'NO_STOP_S', 'forcedReplans', 'MAX_FORCED_REPLANS', 'MIN_IMPROVE', 'PROGRESS_DELTA',
    'AGENT_TIMEOUT_MS', 'AGENT_RETRIES', 'CANDIDATE_FLOOR',
    SHADOW + '\n' + ROUTING + '\n' + AGENT_T + '\n' + ret);
  const coreCalls = [];
  const api = f(Object.assign({ routing: 'true', route_decider: 'eikos' }, args || {}), (m) => logs.push(String(m)),
    { env: {} }, () => { throw new Error('no require'); }, opts.setTimeout || setTimeout, agent, opts.budget, null, KDIR,
    evalDir, opts.deadline || 0, 3, 6, async (p, o) => { coreCalls.push({ p, o }); return agent(p, o); },
    1, 1, 0, 2, 600, 0, 3, 0.01, 0.01, 1800000, 2, 1.05);
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
// Re-sign a relayed envelope the way a faithful script would have, so a check other than the digest is hit.
function resign(L, env) {
  const dec = {};
  L.EIKOS_DECISION_FIELDS.concat(['estimates', 'questions_version', 'policy_version', 'decision_host_sha256'])
    .forEach((k) => { dec[k] = env[k] === undefined ? null : env[k]; });
  return Object.assign(env, { decision_sha256: L.eikosSha256(L.eikosCanonical(dec)) });
}

async function main() {
  await new Promise((r) => srv.listen(0, '127.0.0.1', r));
  const last = (L, ev) => L.__routeAudit.filter((e) => e.event === ev).slice(-1)[0];
  const all = (L, ev) => L.__routeAudit.filter((e) => e.event === ev);

  // ---- the pure reducer ---------------------------------------------------------------------------
  {
    const L = lane();
    const BASE = L.__routeHostInput('eng:memory', 'code_change');
    const R = (x) => L.__routeReduce(Object.assign({}, BASE, { backend: 'ok', estimates: null }, x));
    ok('the host input of a fresh code-change scope: policy rank, list-price order, floors, quota, availability',
      JSON.stringify(BASE) === JSON.stringify({ reducer_version: 'lane_reduce.v1', scope_class: 'code_change', tau: 0.9,
        policy_rank: [H5, S5, O46, O55], price_order: [H5, S5, O55, O46], class_floor_rank: 1, operator_floor_rank: 0,
        scope_floor_rank: 1, effective_rank: -1, excluded: [], top_quota_left: 1, available: [H5, S5, O46, O55] }));
    let r = R({ scope_class: 'paperwork', tau: 0.7, class_floor_rank: 0, scope_floor_rank: 0, estimates: P(0.95, 0.5, 0.99, 0.92) });
    ok('non-monotone estimates are taken as given: the cheapest qualifying lane wins', r.model === H5 && r.threshold_met === true);
    r = R({ estimates: P(0.99, 0.91, 0.99, 0.97) });
    ok('a lane below the class floor is never allowed, however sure Eikos is', r.model === S5 && r.allowed.indexOf(H5) < 0);
    r = R({ estimates: P(0.1, 0.93, 0.99, 0.97) });
    ok('the chosen lane\'s own estimate and the maximum estimate are recorded apart',
      r.chosen_estimate === 0.93 && r.max_estimate === 0.99 && r.override === null);
    // Astra's crossovers: the policy rank decides what is allowed, the price order only what is cheaper.
    r = R({ scope_floor_rank: 2, excluded: [H5, S5], estimates: P(1, 1, 0.95, 0.95) });
    ok('crossover 1: after an Opus 4.6-rank floor, the cheaper Opus 5.5 is still allowed and chosen by price',
      r.model === O55 && JSON.stringify(r.allowed) === JSON.stringify([O46, O55]));
    r = R({ scope_floor_rank: 3, excluded: [H5, S5, O46], estimates: P(1, 1, 1, 0.95) });
    ok('crossover 2: after an Opus 4.6 code failure, Opus 5.5 remains feasible (not "no lane")', r.model === O55);
    r = R({ operator_floor_rank: 3, top_quota_left: 0, estimates: P(1, 1, 0.99, 1) });
    ok('crossover 3: an operator floor at lane 3 never admits the dearer Opus 4.6 when the quota is used: deferral',
      r.model === null && r.allowed.length === 1 && /no feasible lane/.test(r.reason));
    r = R({ scope_floor_rank: 4, excluded: [H5, S5, O46, O55], estimates: P(1, 1, 1, 1) });
    ok('an Opus 5.5 code failure leaves nothing allowed: a deferral, not a fall back to Opus 4.6', r.model === null);
    r = R({ estimates: P(0.25, 0.25, 0.25, 0.25) });
    ok('uniform uncertainty qualifies no lane: a recorded policy override with tau, chosen and max estimate',
      r.threshold_met === false && r.override && r.override.kind === 'no_lane_qualified' &&
      r.override.policy === 'highest_estimate' && r.override.tau === 0.9 && r.chosen_estimate === 0.25 && r.max_estimate === 0.25);
    r = R({ estimates: P(0.2, 0.4, 0.6, 0.85) });
    ok('the override takes the highest estimate, not the dearest lane', r.model === O55 && r.override.chosen_estimate === 0.85);
    r = R({ estimates: P(0.2, 0.5, 0.5, 0.5) });
    ok('override ties keep the cheaper lane', r.model === S5);
    r = R({ backend: 'timeout' });
    ok('a missing answer is unknown, not a miss: the cheapest feasible allowed lane, threshold_met null',
      r.model === S5 && r.threshold_met === null && r.override === null && /no usable estimate \(timeout\)/.test(r.reason));
    r = R({ backend: 'timeout', scope_floor_rank: 2, excluded: [H5, S5] });
    ok('a missing answer after a Sonnet code failure still excludes Sonnet', r.model === O55);
    r = R({ backend: 'timeout', scope_floor_rank: 3, top_quota_left: 0 });
    ok('a missing answer still obeys the quota: with nothing feasible it defers', r.model === null);
    ok('every unusable backend status falls back the same way, under the same constraints',
      ['ineligible', 'malformed', 'unavailable', 'consult_budget_exhausted', 'write_failed', 'not_persisted',
       'relay_mismatch', 'decision_altered', 'question_mismatch', 'error', 'no_result']
        .every((b) => R({ backend: b, estimates: P(1, 1, 1, 1) }).model === S5));
    r = R({ top_quota_left: 0, estimates: P(0, 0.1, 0.95, 0.99) });
    ok('a used-up top-lane quota is filtered before selection', r.model === O46 &&
      r.filtered.some((x) => x.model === O55 && /quota/.test(x.why)));
    r = R({ available: [H5], estimates: P(1, 1, 1, 1) });
    ok('no available lane at or above the floor is a deferral', r.model === null);
    r = R({ operator_floor_rank: 3, estimates: P(1, 1, 1, 0.95) });
    ok('an operator floor raises the lower bound', r.model === O55);
    r = R({ effective_rank: 2, estimates: P(1, 0.99, 0.95, 0.95) });
    ok('the scope\'s effective lane is a floor by policy rank: a confident Sonnet is not taken, price picks among the rest',
      r.model === O55 && r.allowed.join() === [O46, O55].join());
    r = R({ effective_rank: 3, top_quota_left: 0, estimates: P(1, 1, 1, 1) });
    ok('an effective Opus 5.5 lane with the quota used defers rather than dropping', r.model === null);
    const x = Object.assign({}, BASE, { backend: 'ok', estimates: P(0.2, 0.93, 0.9, 0.5) });
    const before = JSON.stringify(x);
    ok('the reducer is pure: same input, same output, input untouched',
      JSON.stringify(L.__routeReduce(x)) === JSON.stringify(L.__routeReduce(x)) && JSON.stringify(x) === before);
    ok('price order is the declared list-price proxy, over exactly the lanes',
      JSON.stringify(L.ROUTE_PRICE_ORDER) === JSON.stringify([H5, S5, O55, O46]) &&
      L.ROUTE_PRICE_ORDER.slice().sort().join() === L.ROUTE_LANES.slice().sort().join());
  }

  // ---- strict wire parsing ------------------------------------------------------------------------
  {
    const L = lane();
    const g = L.__routeParseEstimates(`${H5}=0.31;${S5}=1.0;${O46}=5e-05;${O55}=0.97`);
    ok('the estimates string parses exactly as Python wrote it', g && g[H5] === 0.31 && g[S5] === 1 && g[O46] === 5e-05);
    const bad = [`${H5}=0.9garbage;${S5}=1.0;${O46}=0.5;${O55}=0.5`, `${H5}=+0.3;${S5}=1.0;${O46}=0.5;${O55}=0.5`,
      `${H5}=1.5;${S5}=1.0;${O46}=0.5;${O55}=0.5`, `${H5}=NaN;${S5}=1.0;${O46}=0.5;${O55}=0.5`,
      `${H5}= 0.3;${S5}=1.0;${O46}=0.5;${O55}=0.5`, `${S5}=0.3;${H5}=1.0;${O46}=0.5;${O55}=0.5`,
      `${H5}=0.3;${S5}=1.0;${O46}=0.5`, `${H5}=0.3;${S5}=1.0;${O46}=0.5;${O55}=0.5;${O55}=0.5`,
      `${H5}=0.3;${H5}=1.0;${O46}=0.5;${O55}=0.5`, `${H5}=0x1;${S5}=1.0;${O46}=0.5;${O55}=0.5`,
      `${H5}=0.3=1;${S5}=1.0;${O46}=0.5;${O55}=0.5`, `${H5}=-0.0;${S5}=1.0;${O46}=0.5;${O55}=0.5`, '', null];
    ok('prefixes, signs, out-of-range, wrong order, missing, extra and duplicate lanes are all rejected',
      bad.every((b) => L.__routeParseEstimates(b) === null));
  }

  // ---- card, transport and the whole decision ------------------------------------------------------
  {
    server.probs = P(0.31, 0.93, 0.95, 0.97);
    server.seen = [];
    const L = lane({}, { deadline: 1 });
    L.setClock({ left_s: 5400 });
    direction(L, 'r2_d1');
    const pick = await L.__routePick('optimize prompt', W(ENG('r2_d1'), 'Optimize'));
    const dec = last(L, 'decide');
    ok('the default decider is still Sonnet; eikos only when chosen', lane({ route_decider: undefined }).ROUTE_EIKOS_ON === false && L.ROUTE_EIKOS_ON === true);
    ok('first sight: one carrier call, verified and persisted, cheapest lane at or above tau', pick.model === S5 &&
      L.carried.length === 1 && dec.backend === 'ok' && dec.relay_ok && dec.decision_ok && dec.receipt_ok &&
      dec.persisted === true && dec.trigger === 'first_sight' && dec.threshold_met === true && dec.chosen_estimate === 0.93);
    ok('the carrier is Haiku 5.5, unrouted, labelled as router cost, and sends the host input',
      L.carried[0].o.model === H5 && /^You are the route_carrier\. PHASE=Optimize\./.test(L.carried[0].prompt) &&
      /--host-json /.test(L.carried[0].prompt) && L.workers.length === 0);
    const card = JSON.parse(server.seen[0].state);
    ok('the card carries the direction verbatim, the acceptance event, budget and attempt configuration',
      L.eikosCanonical(card.task.direction) === L.eikosCanonical(L.__routeTasks[ENG('r2_d1')]) &&
      card.task.goal_source === 'direction' && /above the candidate floor \(1\.05\)/.test(card.task.acceptance) &&
      /does not meet/.test(card.task.acceptance) && card.budget.minutes_left === 90 &&
      card.budget.minutes_left_source === 'round_clock' && card.round === 2 && card.scope_class === 'code_change' &&
      JSON.stringify(card.config.lanes) === JSON.stringify(L.ROUTE_LANES) && card.config.attempt_timeout_s === 1800 &&
      card.config.transport_retries === 2);
    ok('the whole dispatch prompt goes in as digest and length only; no routing answer rides along',
      card.task.dispatch_prompt_sha256 === L.eikosSha256('optimize prompt') && card.task.dispatch_prompt_chars === 15 &&
      JSON.stringify(card).indexOf('optimize prompt') < 0 && !('answer' in card) && !('estimates' in card));
    ok('Eikos is asked the four lane questions and never sees the host\'s thresholds or quota',
      JSON.stringify(server.seen[0].questions) === JSON.stringify(SPEC.questions) &&
      JSON.stringify(server.seen[0]).indexOf('reducer_version') < 0 && JSON.stringify(server.seen[0]).indexOf('top_quota') < 0);
    const again = await L.__routePick('optimize prompt', W(ENG('r2_d1')));
    ok('the same task dispatched again reuses its occurrence: no new question, a new dispatch record',
      again.model === S5 && L.carried.length === 1 && again.dispatch_id !== pick.dispatch_id &&
      L.__routeDispatches[ENG('r2_d1')].length === 2 && L.__routeDispatches[ENG('r2_d1')].every((d) => d.seq === 1));

    // failure kinds, each bound to its own dispatch (the label's first open dispatch)
    L.__routeOutcome('eng:memory', false, 'verify status=invalid_measurement', 'invalid_measurement', 'verify r2_d1', ENG('r2_d1'));
    L.__routeOutcome('eng:memory', false, 'engineer returned nothing', 'no_result', 'engineer r2_d1', ENG('r2_d1'));
    ok('an invalid measurement and a missing result each close one dispatch and change nothing',
      L.__routeDispatches[ENG('r2_d1')].map((d) => d.outcome).join() === 'invalid_measurement,no_result' &&
      JSON.stringify(L.__routeLadder['eng:memory'].fails_by_lane) === '{}' && all(L, 'fail').every((e) => !e.reconsult_next_attempt));
    await L.__routePick('optimize prompt', W(ENG('r2_d1')));
    ok('...and ask nothing', L.carried.length === 1);
    L.__routeOutcome('eng:memory', false, 'verify status=apply_failed', 'work_product', 'verify r2_d1', ENG('r2_d1'));
    L.__routeOutcome('eng:memory', false, 'verify status=apply_failed', 'work_product', 'verify r2_d1', ENG('r2_d1'));
    ok('a work-product failure counts once against its own lane; a duplicate outcome is recorded and ignored',
      L.__routeLadder['eng:memory'].fails_by_lane[S5] === 1 && L.__routeLadder['eng:memory'].floor_rank === 2 &&
      last(L, 'outcome_unmatched') && last(L, 'outcome_unmatched').kind === 'work_product');
    server.probs = P(0.1, 0.99, 0.95, 0.97);
    const p4 = await L.__routePick('optimize prompt', W(ENG('r2_d1')));
    const c4 = JSON.parse(server.seen[server.seen.length - 1].state);
    const d4 = last(L, 'decide');
    ok('the next attempt is a new occurrence: asked again with the evidence, Sonnet excluded however sure Eikos is',
      p4.model === O55 && L.carried.length === 2 && d4.seq === 2 && c4.occurrence.trigger === 'work_product_failure' &&
      c4.occurrence.prior.slice(-1)[0].outcome === 'work_product' && c4.occurrence.prior.slice(-1)[0].lane === S5 &&
      c4.occurrence.prior.slice(-1)[0].source === 'verify r2_d1' && d4.reduce.allowed.indexOf(S5) < 0 && L.topUsed() === 1);
    L.__routeOutcome('eng:memory', false, 'verify status=correctness_failed', 'work_product', 'verify r2_d1', ENG('r2_d1'));
    const p5 = await L.__routePick('optimize prompt', W(ENG('r2_d1')));
    ok('after the top lane fails a code change nothing is allowed: deferred without asking Eikos',
      p5.skip === true && p5.deferred === true && L.carried.length === 2 && last(L, 'decide').backend === 'not_asked_no_feasible_lane');
    const t = await L.agentT('optimize prompt', W(ENG('r2_d1')));
    ok('agentT dispatches nothing for a deferred worker, marks it skipped and logs the reason', t === null &&
      L.__routeSkipped[ENG('r2_d1')] === true && L.logs.some((m) => /no feasible lane \(deferred\)/.test(m)) &&
      L.coreCalls.length === 0 && L.carried.length === 2);
    const rep = L.__routeReport();
    ok('the report names the decider, labels its settings experimental and counts qualified choices apart from the rest',
      rep.decider === 'eikos' && rep.eikos.tau_status === 'unvalidated_experimental' && rep.eikos.tau.code_change === 0.9 &&
      rep.eikos.consults === 2 && rep.eikos.choices.eikos_qualified === 2 && rep.eikos.choices.deferred === 1 &&
      rep.eikos.choices.override_no_lane_qualified === 0 && /list-price proxy/.test(rep.eikos.price_order_is) &&
      rep.config_errors.length === 0);
  }

  // ---- resume: the same run replayed ---------------------------------------------------------------
  {
    // A Workflow resume replays the script and returns cached results for unchanged agent() calls; the
    // `journal` stands in for that cache. The third run has no cache and relies on eikos_decide.py's own
    // persistence. Every run must reach the same decisions, counters and attributions.
    async function script(L) {
      L.setClock({ left_s: 3000 });
      direction(L, 'r1_d0');
      const out = [];
      out.push((await L.__routePick('p0', W(ENG('r1_d0')))).model);
      L.__routeOutcome('eng:memory', false, 'verify status=apply_failed', 'work_product', 'verify r1_d0', ENG('r1_d0'));
      L.__routeOutcome('eng:memory', false, 'verify status=apply_failed', 'work_product', 'verify r1_d0', ENG('r1_d0'));
      out.push((await L.__routePick('p0', W(ENG('r1_d0')))).model);          // escalated; the top lane is reserved
      direction(L, 'r1_d1', 'compute');
      out.push((await L.__routePick('p1', W(ENG('r1_d1', 'compute')))).model); // another scope starts at its own floor
      return { out, top: L.topUsed(), fails: L.__routeLadder['eng:memory'].fails_by_lane,
               floor: L.__routeLadder['eng:memory'].floor_rank,
               decisions: all(L, 'decide').map((e) => [e.task_key, e.seq, e.decided, e.logical_key, e.estimates]),
               fail_events: all(L, 'fail').length, unmatched: all(L, 'outcome_unmatched').length };
    }
    const journal = new Map();
    let misses = 0;
    server.probs = P(0.1, 0.95, 0.96, 0.97);
    const L1 = lane({}, { deadline: 1, carrier: async (p, o, run) => { const env = await run(p); journal.set(p, env); return env; } });
    const recorded = await script(L1);
    server.probs = P(0.99, 0.99, 0.99, 0.99);                                 // any redraw would now differ
    const seenBefore = server.seen.length;
    const L2 = lane({}, { deadline: 1, evalDir: L1.evalDir,
      carrier: async (p, o, run) => { if (journal.has(p)) return journal.get(p); misses++; return run(p); } });
    const replayed = await script(L2);
    const L3 = lane({}, { deadline: 1, evalDir: L1.evalDir });
    const relived = await script(L3);
    ok('the run escalates past the failed Sonnet, counts the duplicate outcome once, and another scope starts at its own floor',
      JSON.stringify(recorded.out) === JSON.stringify([S5, O55, S5]) && recorded.top === 1 && recorded.fails[S5] === 1 &&
      recorded.floor === 2 && recorded.fail_events === 1 && recorded.unmatched === 1);
    ok('a resumed run with the Workflow cache asks nothing new: the same frozen cards and host inputs',
      misses === 0 && journal.size === 3 && server.seen.length === seenBefore);
    ok('...and reaches the same decisions, quota use, failure counts, floor and attributions',
      JSON.stringify(replayed) === JSON.stringify(recorded));
    ok('a resumed run without the cache gets the persisted estimates from eikos_decide.py, never a redraw',
      JSON.stringify(relived) === JSON.stringify(recorded) && server.seen.length === seenBefore &&
      all(L3, 'decide').every((e) => e.backend === 'ok' && e.persisted === true && e.interventions.length === 0));
    // The same EVAL_DIR resumed under a stricter operator floor: the stored estimates stand, the host
    // reduces on the current constraints and records the intervention.
    const L4 = lane({ route_floors: '{"eng:memory": 3}' }, { deadline: 1, evalDir: L1.evalDir });
    L4.setClock({ left_s: 3000 });
    direction(L4, 'r1_d0');
    const p4 = await L4.__routePick('p0', W(ENG('r1_d0')));
    const d4 = last(L4, 'decide');
    ok('resumed under a changed host input: same key, persisted estimates, current constraints, recorded intervention',
      p4.model === O55 && d4.logical_key === recorded.decisions[0][3] && d4.estimates[H5] === 0.1 &&
      d4.interventions.some((s) => /different host input/.test(s)) && server.seen.length === seenBefore);
  }

  // ---- policy rank versus price, end to end --------------------------------------------------------
  {
    server.probs = P(0.1, 0.95, 0.95, 0.5);
    const L = lane({ route_max_top_escalations: '5' });
    direction(L, 'r1_d0', 'compute');
    const lbl = ENG('r1_d0', 'compute');
    const models = [(await L.__routePick('p', W(lbl))).model];
    L.__routeOutcome('eng:compute', false, 'apply_failed', 'work_product', 'verify r1_d0', lbl);
    models.push((await L.__routePick('p', W(lbl))).model);
    L.__routeOutcome('eng:compute', false, 'correctness_failed', 'work_product', 'verify r1_d0', lbl);
    models.push((await L.__routePick('p', W(lbl))).model);
    const d = last(L, 'decide');
    ok('Sonnet fails, then Opus 4.6 (the only qualifying lane) fails, then Opus 5.5 is still allowed',
      JSON.stringify(models) === JSON.stringify([S5, O46, O55]));
    ok('...taken as a recorded override because its estimate is under tau', d.threshold_met === false &&
      d.override.kind === 'no_lane_qualified' && d.chosen_estimate === 0.5 && d.max_estimate === 0.5);
  }
  {
    // a carrier timeout after a code failure still excludes the failed lane
    let hang = false;
    server.probs = P(0.1, 0.99, 0.95, 0.97);
    const L = lane({}, { carrier: (p, o, run) => (hang ? new Promise(() => {}) : run(p)),
                         setTimeout: (fn, ms) => setTimeout(fn, hang ? 5 : ms) });
    direction(L, 'r1_d0');
    const a = await L.__routePick('p', W(ENG('r1_d0')));
    L.__routeOutcome('eng:memory', false, 'apply_failed', 'work_product', 'verify r1_d0', ENG('r1_d0'));
    hang = true;
    const b = await L.__routePick('p', W(ENG('r1_d0')));
    ok('Sonnet fails a code change, the next consultation times out: Opus 5.5, never Sonnet again',
      a.model === S5 && b.model === O55 && last(L, 'decide').backend === 'timeout' &&
      /no usable estimate \(timeout\)/.test(last(L, 'decide').why));
  }

  // ---- parallel same-scope tasks and late results ---------------------------------------------------
  {
    let inflight = 0, maxInflight = 0;
    server.probs = P(0.31, 0.93, 0.95, 0.97);
    const L = lane({ route_max_top_escalations: '5' }, { carrier: async (p, o, run) => {
      inflight++; maxInflight = Math.max(maxInflight, inflight);
      try { return await run(p); } finally { inflight--; } } });
    direction(L, 'r1_d0'); direction(L, 'r1_d1');
    const [a, b] = await Promise.all([L.__routePick('p0', W(ENG('r1_d0'))), L.__routePick('p1', W(ENG('r1_d1')))]);
    ok('two in-flight directions in one scope are decided one at a time, each its own occurrence',
      maxInflight === 1 && a.model === S5 && b.model === S5 &&
      JSON.stringify(all(L, 'decide').map((e) => e.seq)) === '[1,2]' && a.dispatch_id !== b.dispatch_id);
    server.probs = P(0.1, 0.5, 0.5, 0.97);
    direction(L, 'r1_d2');
    const c = await L.__routePick('p2', W(ENG('r1_d2')));
    L.__routeOutcome('eng:memory', false, 'first task patch did not apply', 'work_product', 'verify r1_d0', ENG('r1_d0'));
    const f0 = last(L, 'fail');
    ok('a delayed failure is charged to its own dispatch\'s lane, not the scope\'s latest decision',
      c.model === O55 && f0.model === S5 && f0.dispatch_id === a.dispatch_id && f0.seq === 1 &&
      L.__routeLadder['eng:memory'].fails_by_lane[S5] === 1 && !(O55 in L.__routeLadder['eng:memory'].fails_by_lane));
    const b2 = await L.__routePick('p1', W(ENG('r1_d1')));
    const ov = last(L, 'override');
    ok('a repeated dispatch whose lane a failure has since excluded is re-reduced on its own estimates, recorded',
      b2.model === O55 && ov.kind === 'decided_lane_no_longer_feasible' && ov.from === S5 && ov.to === O55 &&
      L.carried.length === 3);
    L.__routeOutcome('eng:memory', false, 'engineer reported 0.98x <= floor 1.05', 'non_improvement', 'engineer r1_d1', ENG('r1_d1'));
    const b3 = await L.__routePick('p1', W(ENG('r1_d1')));
    ok('the first open dispatch takes the outcome; a non-improvement asks again (a new occurrence) without exclusion',
      L.__routeDispatches[ENG('r1_d1')][0].outcome === 'non_improvement' && L.carried.length === 4 &&
      last(L, 'decide').trigger === 'non_improvement' && last(L, 'decide').seq === 4 && b3.model === O55 &&
      last(L, 'decide').reduce.excluded.join() === S5);
    L.__routeOutcome('eng:memory', false, 'late: patch did not apply', 'work_product', 'verify r1_d1', ENG('r1_d1'));
    const late = last(L, 'fail');
    ok('a late failure from a superseded occurrence counts against its own lane and schedules nothing',
      late.late === true && late.seq === 2 && late.model === O55 && late.reconsult_next_attempt === false &&
      L.__routeLadder['eng:memory'].fails_by_lane[O55] === 1);
    const b4 = await L.__routePick('p1', W(ENG('r1_d1')));
    ok('...but the next dispatch still obeys it: no new question, Opus 5.5 now excluded, deferred',
      L.carried.length === 4 && b4.skip === true && b4.deferred === true && last(L, 'override').to === null);
  }

  // ---- quota and the kill switch ---------------------------------------------------------------------
  {
    server.probs = P(0.1, 0.5, 0.5, 0.97);
    const L = lane();
    direction(L, 'r1_d0');
    const a = await L.__routePick('p', W(ENG('r1_d0')));
    const b = await L.__routePick('p', W(ENG('r1_d0')));
    ok('a second dispatch of a top-lane occurrence with the quota used never drops below its decided lane: deferred',
      a.model === O55 && b.skip === true && b.deferred === true &&
      last(L, 'override').kind === 'decided_lane_no_longer_feasible' && last(L, 'override').to === null && L.topUsed() === 1);
  }
  {
    // The scope's effective lane survives new occurrences (Astra, 2026-10-12): no silent drop to Sonnet.
    server.probs = P(0.1, 0.5, 0.5, 0.97);
    const L = lane();
    direction(L, 'r1_d0');
    const a = await L.__routePick('p', W(ENG('r1_d0')));
    server.probs = P(0.1, 0.99, 0.5, 0.5);                                    // Sonnet would now qualify
    direction(L, 'r1_d1');
    const b = await L.__routePick('p1', W(ENG('r1_d1')));
    ok('after Opus 5.5 is dispatched, a new task in the scope with the quota used is deferred, not sent to Sonnet',
      a.model === O55 && L.__routeLadder['eng:memory'].effective_rank === 3 && b.skip === true && b.deferred === true &&
      last(L, 'decide').backend === 'not_asked_no_feasible_lane' && L.carried.length === 1);
    L.__routeOutcome('eng:memory', false, 'engineer reported 0.98x <= floor 1.05', 'non_improvement', 'engineer r1_d0', ENG('r1_d0'));
    const c = await L.__routePick('p', W(ENG('r1_d0')));
    ok('...and so is the next attempt after its non-improvement', c.skip === true && c.deferred === true &&
      last(L, 'decide').trigger === 'non_improvement' && L.carried.length === 1);
    const M = lane({ route_max_top_escalations: '5' });
    server.probs = P(0.1, 0.5, 0.5, 0.97);
    direction(M, 'r1_d0'); direction(M, 'r1_d1');
    await M.__routePick('p', W(ENG('r1_d0')));
    server.probs = P(0.1, 0.99, 0.5, 0.5);
    const d = await M.__routePick('p1', W(ENG('r1_d1')));
    ok('with quota left, the new task stays on the effective lane as a recorded override, not a drop to Sonnet',
      d.model === O55 && last(M, 'decide').threshold_met === false && last(M, 'decide').reduce.allowed.join() === O55);
  }
  {
    const spent = { v: 0 };
    server.probs = P(0.1, 0.5, 0.5, 0.97);
    const L = lane({ route_max_output_tokens: '1000' }, { budget: { spent: () => spent.v },
      carrier: async (p, o, run) => { spent.v = 5000; return run(p); } });
    direction(L, 'r1_d0'); direction(L, 'r1_d1');
    const r = await L.agentT('p', W(ENG('r1_d0')));
    ok('a consultation that crosses the output cap stops the worker: no dispatch, quota released, killed reported',
      r === null && L.carried.length === 1 && L.coreCalls.length === 0 && L.topUsed() === 0 &&
      last(L, 'decide').decided === O55 && /kill switch/.test(last(L, 'skip').why) && L.__routeReport().killed === true &&
      L.logs.some((m) => /output-token cap reached/.test(m)) && L.__routeSkipped[ENG('r1_d0')] === true &&
      L.__routeLadder['eng:memory'].effective_rank === -1);
    await L.agentT('p', W(ENG('r1_d1')));
    ok('once tripped, the switch is checked before the carrier: nothing more is asked', L.carried.length === 1 && L.coreCalls.length === 0);
  }

  // ---- what reaches the host when the relay or the record is bad -------------------------------------
  async function relayCase(name, carrier, expect, extra) {
    server.probs = P(0.99, 0.99, 0.99, 0.99);
    let L = null;
    L = lane(extra && extra.args, Object.assign({ carrier: carrier && ((p, o, run) => carrier(p, o, run, L)) }, extra && extra.opts));
    if (extra && extra.prepare) extra.prepare(L);
    direction(L, 'r1_d0');
    const pick = await L.__routePick('p', W(ENG('r1_d0')));
    const d = last(L, 'decide');
    ok(name, d.backend === expect && pick.model === S5 && d.estimates === null && d.threshold_met === null);
  }
  await relayCase('altered estimates in the relay are caught (decision digest), host falls back',
    async (p, o, run) => Object.assign(await run(p), { estimates: `${H5}=1.0;${S5}=1.0;${O46}=1.0;${O55}=1.0` }), 'decision_altered');
  await relayCase('a different card reached the script: relay mismatch',
    async (p, o, run) => Object.assign(await run(p), { state_sha256: '0'.repeat(64) }), 'relay_mismatch');
  await relayCase('a different host input reached the script: relay mismatch',
    async (p, o, run) => Object.assign(await run(p), { host_sha256: '0'.repeat(64) }), 'relay_mismatch');
  await relayCase('an answer to another question version is refused even when consistently signed',
    async (p, o, run, L) => resign(L, Object.assign(await run(p), { questions_version: 'lane_sufficiency.v0' })), 'question_mismatch');
  await relayCase('a non-strict estimates token is malformed even when consistently signed',
    async (p, o, run, L) => resign(L, Object.assign(await run(p), { estimates: `${H5}=0.99garbage;${S5}=0.99;${O46}=0.99;${O55}=0.99` })),
    'malformed');
  await relayCase('a carrier that throws is a backend error, not a worker failure', async () => { throw new Error('API 529'); }, 'error');
  await relayCase('a carrier that never returns times out on its own bound', () => new Promise(() => {}), 'timeout',
    { opts: { setTimeout: (fn) => setTimeout(fn, 5) } });
  await relayCase('a carrier that returns nothing is no result', async () => null, 'no_result');
  await relayCase('a real receipt append failure: persisted but not receipted, so not used', null, 'not_persisted',
    { prepare: (L) => fs.mkdirSync(path.join(L.evalDir, 'eikos', 'lane_sufficiency.attempts.jsonl'), { recursive: true }) });
  await relayCase('a real index write failure is write_failed, so not used', null, 'write_failed',
    { prepare: (L) => fs.mkdirSync(path.join(L.evalDir, 'eikos', 'lane_sufficiency.index.json'), { recursive: true }) });
  {
    server.raw = JSON.stringify({ answers: { passes_on_haiku_5_5: answer(1.7) } });
    const L = lane();
    direction(L, 'r1_d0');
    const pick = await L.__routePick('p', W(ENG('r1_d0')));
    server.raw = null;
    ok('an unusable Eikos answer is malformed (status relayed intact), host falls back', last(L, 'decide').backend === 'malformed' &&
      last(L, 'decide').decision_ok === true && pick.model === S5);
  }

  // ---- task identity and eligibility -------------------------------------------------------------------
  {
    server.probs = P(0.99, 0.99, 0.99, 0.99);
    const L = lane();
    direction(L, 'r1_d0');
    await L.__routePick('p', W(ENG('r1_d0')));
    direction(L, 'r1_d0', 'memory', { prompt: 'Tile K by 128 instead.' });
    await L.__routePick('p', W(ENG('r1_d0')));
    await L.__routePick('p, with a new workspace path', W(ENG('r1_d0')));
    await L.__routePick('p, with a new workspace path', Object.assign(W(ENG('r1_d0')), { effort: 'high' }));
    ok('a changed direction text, dispatch prompt or effort under the same direction ID is a new occurrence, asked again',
      L.carried.length === 4 && JSON.stringify(all(L, 'decide').map((e) => e.trigger)) ===
        JSON.stringify(['first_sight', 'changed_task', 'changed_task', 'changed_task']) &&
      JSON.parse(server.seen[server.seen.length - 1].state).config.effort === 'high');
  }
  {
    const L = lane();
    const a = await L.__routePick(ROLE_PROMPT, W('benchmark_engineer', 'Benchmark'));
    const v0 = await L.__routePick('verify d0', W('verify r1_d0', 'Verify'));
    const v1 = await L.__routePick('verify d1', W('verify r1_d1', 'Verify'));
    const d = all(L, 'decide');
    ok('S0 has no task projection for non-engineer scopes: never asked, routed by host policy at the class floor',
      L.carried.length === 0 && d.every((e) => e.backend === 'ineligible' && /no task projection/.test(e.ineligible)) &&
      a.model === S5 && v0.model === S5 && v1.model === S5);
    ok('two different verification tasks are two occurrences, not one scope-wide decision',
      d.length === 3 && d[1].task_key !== d[2].task_key && d[2].seq === 2);
    const L2 = lane();
    direction(L2, 'r1_d0', 'memory', { prompt: 'x'.repeat(4001) });
    const p = await L2.__routePick('p', W(ENG('r1_d0')));
    ok('a direction too long for the card is ineligible: Eikos is not asked, nothing is truncated',
      L2.carried.length === 0 && /does not fit/.test(last(L2, 'decide').ineligible) && p.model === S5);
  }
  {
    server.probs = P(0.99, 0.99, 0.99, 0.99);
    const L = lane({ route_eikos_max_consults: '1' });
    direction(L, 'r1_d0'); direction(L, 'r1_d1');
    await L.__routePick('p', W(ENG('r1_d0')));
    const b = await L.__routePick('p', W(ENG('r1_d1')));
    ok('the consultation budget bounds carrier calls; the rest fall back under the same constraints',
      L.carried.length === 1 && last(L, 'decide').backend === 'consult_budget_exhausted' && b.model === S5);
  }
  {
    server.probs = P(0.95, 0.95, 0.95, 0.95);
    const L = lane();
    await L.agentT('p', W('tech_lead:plan r1'));
    await L.agentT('p', W('clock r1'));
    ok('brain and fixed helpers are never sent to Eikos', L.carried.length === 0 &&
      L.coreCalls[0].o.model === O55 && L.coreCalls[1].o.model === H5);
    direction(L, 'r1_d0');
    const o = Object.assign(W(ENG('r1_d0')), { effort: 'high' });
    await L.agentT('p', o);
    ok('agentT applies the decided model without mutating the caller\'s options; the card records effort',
      L.coreCalls[2].o.model === S5 && L.coreCalls[2].o.effort === 'high' && o.model === undefined &&
      JSON.parse(server.seen[server.seen.length - 1].state).config.effort === 'high');
  }

  // ---- settings -------------------------------------------------------------------------------------
  {
    const L = lane({ route_eikos_tau: '{"code_change": 0.5, "evidence": 2, "paperwork": "x", "mystery": 0.1}' });
    ok('thresholds are per-run settings; each invalid value is recorded and keeps the built-in experimental one',
      L.ROUTE_EIKOS_TAU.code_change === 0.5 && L.ROUTE_EIKOS_TAU.evidence === 0.8 && L.ROUTE_EIKOS_TAU.paperwork === 0.7 &&
      L.__routeConfigErrors.length === 3 && L.__routeReport().config_errors.length === 3);
    const U = lane({ route_decider: 'eikoss' });
    ok('an unknown decider is recorded and never activates Eikos', U.ROUTE_EIKOS_ON === false &&
      U.__routeConfigErrors.some((e) => /route_decider/.test(e)));
    const C = lane({ route_eikos_max_consults: '2.5', route_eikos_carrier_timeout_ms: '-1', route_max_top_escalations: 'x',
                     route_floors: '[1]', route_conf_escalate: '1.5' });
    ok('non-integer, negative and malformed settings are recorded, with the value used instead',
      C.ROUTE_EIKOS_MAX_CONSULTS === 40 && ['route_eikos_max_consults', 'route_eikos_carrier_timeout_ms',
        'route_max_top_escalations', 'route_floors', 'route_conf_escalate'].every((k) => C.__routeConfigErrors.some((e) => e.indexOf(k) === 0)));
    const F = lane({ route_floors: '{"eng:memory": 7}' });
    ok('an out-of-range operator floor is recorded', F.__routeConfigErrors.some((e) => /route_floors\.eng:memory/.test(e)));
    ok('a clean configuration records nothing', lane({ route_floors: '{"eng:memory": 2}', route_eikos_tau: '{"evidence": 0.75}',
      route_eikos_max_consults: '3' }).__routeConfigErrors.length === 0);
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
  ok('the lane reads every forwarded setting', KEYS.every((k) => ROUTING.indexOf(`A.${k}`) >= 0 || ROUTING.indexOf(`'${k}'`) >= 0));

  srv.close();
  TMP.forEach((d) => fs.rmSync(d, { recursive: true, force: true }));
  console.log(fails ? `FAILED — ${fails}/${n} checks failed` : `ALL PASS — ${n}/${n} checks`);
  process.exit(fails ? 1 : 0);
}
main().catch((e) => { console.error(e); process.exit(2); });
