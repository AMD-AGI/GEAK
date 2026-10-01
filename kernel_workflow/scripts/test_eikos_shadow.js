// Offline test of the Eikos shadow pilot AS SHIPPED in kernel_lane.js: the region between
// <<EIKOS-SHADOW-START>> and <<EIKOS-SHADOW-END>>, evaluated with the lane's globals stubbed. The
// carrier stub runs the generated command through REAL bash into the real eikos_decide.py, against
// a fake local Eikos server — so shell quoting, the relay echo and receipts are exercised end to end.
// Synthetic mechanics evidence only: no model, no GPU, no decision-quality claim.
//
//   node kernel_workflow/scripts/test_eikos_shadow.js
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const { execFile } = require('child_process');

const WF = path.resolve(__dirname, '..');
const SRC = fs.readFileSync(path.join(WF, 'kernel_lane.js'), 'utf8');
const i = SRC.indexOf('// <<EIKOS-SHADOW-START>>'), j = SRC.indexOf('// <<EIKOS-SHADOW-END>>');
if (i < 0 || j < 0) throw new Error('shadow markers not found');
const REGION = SRC.slice(i, j);

let n = 0, fails = 0;
function ok(name, cond) { n++; if (cond) console.log('PASS  ' + name); else { fails++; console.log('FAIL  ' + name); } }

function load(cfg) {
  const c = Object.assign({
    A: { eikos_shadow: 'round_continue' }, EVAL_DIR: '/tmp/x', DEADLINE_EPOCH: 0, NO_STOP_S: 900,
    MAX_FORCED_REPLANS: 6, forcedReplans: 0, dispatched: 2, BUDGET: 6, cumulative: 1.21, bestSeen: 1.234,
    noImprove: 1, MAX_NO_IMPROVE: 2, MIN_IMPROVE: 0.01, PROGRESS_DELTA: 0.01,
    agent: async () => { throw new Error('agent must not be called'); }, setTimeout, clearTimeout,
  }, cfg || {});
  const calls = [], logs = [];
  const agentSpy = async (p, o) => { calls.push({ p, o }); return c.agent(p, o); };
  const body = `let forcedReplans = c.forcedReplans, dispatched = c.dispatched, cumulative = c.cumulative,
    bestSeen = c.bestSeen, noImprove = c.noImprove;
  const A = c.A, EVAL_DIR = c.EVAL_DIR, WORKFLOW_DIR = c.WF, DEADLINE_EPOCH = c.DEADLINE_EPOCH,
    NO_STOP_S = c.NO_STOP_S, MAX_FORCED_REPLANS = c.MAX_FORCED_REPLANS, BUDGET = c.BUDGET,
    MAX_NO_IMPROVE = c.MAX_NO_IMPROVE, MIN_IMPROVE = c.MIN_IMPROVE, PROGRESS_DELTA = c.PROGRESS_DELTA;
  ${REGION}
  return { EIKOS_SHADOW_ON, EIKOS_SHADOW_LOG, eikosSnapshot, eikosStopPermitted, eikosRawChoice,
           eikosCommand, eikosShellQuote, eikosSha256, eikosCanonical, eikosShadowRound, setLast: (o) => { eikosLastOutcome = o; } };`;
  const f = new Function('c', 'agent', 'log', 'setTimeout', 'clearTimeout', 'unescape', 'encodeURIComponent', body);
  const api = f(Object.assign({ WF }, c), agentSpy, (m) => logs.push(m), c.setTimeout, c.clearTimeout,
                unescape, encodeURIComponent);
  return Object.assign(api, { calls, logs });
}

const CONT = { stop: false, directions: [{ specialty: 'memory' }] };
const STOP = { stop: true, directions: [] };

// A carrier that does exactly what a faithful agent should: run the fenced command, return stdout.
function bashCarrier(env) {
  return (p) => new Promise((resolve, reject) => {
    const m = /```bash\n([\s\S]*?)\n```/.exec(p);
    if (!m) return reject(new Error('no command in prompt'));
    execFile('bash', ['-c', m[1]], { env: Object.assign({}, process.env, env), timeout: 60000 },
      (err, stdout) => (err ? reject(err) : resolve(JSON.parse(stdout))));
  });
}

(async () => {
  // Fake Eikos: answers continue at 0.91.
  let hits = 0;
  const srv = http.createServer((req, res) => {
    let b = ''; req.on('data', (d) => { b += d; }); req.on('end', () => {
      hits++;
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ answers: {
        next_step: { type: 'choice', choice: 'continue', probabilities: { continue: 0.91, stop: 0.09 }, confidence: 0.91 },
        stalled: { type: 'noul', probability: 0.4, noul: 0.4, value: false, confidence: 0.6 } } }));
    });
  });
  await new Promise((r) => srv.listen(0, '127.0.0.1', r));
  const URL = `http://127.0.0.1:${srv.address().port}`;

  // ---- off by default
  ok('OFF unless eikos_shadow names round_continue', load({ A: {} }).EIKOS_SHADOW_ON === false &&
     load({ A: { eikos_shadow: 'other' } }).EIKOS_SHADOW_ON === false && load().EIKOS_SHADOW_ON === true);
  ok('every lane hook site is gated on EIKOS_SHADOW_ON',
     /let eikosSnap = EIKOS_SHADOW_ON \?/.test(SRC) && /const eikosFirstChoice = EIKOS_SHADOW_ON \?/.test(SRC) &&
     /if \(EIKOS_SHADOW_ON\) eikosSnap = eikosSnapshot/.test(SRC) && /const eikosRec = EIKOS_SHADOW_ON\s*\?/.test(SRC) &&
     /if \(EIKOS_SHADOW_ON\) directions\.forEach/.test(SRC) && /eikos_shadow: EIKOS_SHADOW_ON \?/.test(SRC));
  ok('shadow call sits BEFORE the host stop check (TechLead decision already final)',
     SRC.indexOf('await eikosShadowRound(') < SRC.indexOf("log(`Round ${round}: TechLead chose to stop."));

  // ---- frozen state
  {
    const L = load({ DEADLINE_EPOCH: 0 });
    const s = L.eikosSnapshot(3, Infinity, 'pre_plan_clock');
    ok('state: honest names and unknowns (tracked incumbent, noise unknown, no-deadline time unknown)',
       s.tracked_incumbent_speedup === 1.21 && !('best_committed_speedup' in s) && s.noise_band === 'unknown' &&
       s.minutes_left === 'unknown' && s.minutes_left_source === 'no_deadline' && s.round === 3 &&
       s.directions_used === 2 && s.last_round_outcome === 'none');
    const D = load({ DEADLINE_EPOCH: 1 });
    const t = D.eikosSnapshot(2, 1830, 'pre_replan_clock');
    ok('state: minutes from the clock read before the attempt, with its source', t.minutes_left === 31 && t.minutes_left_source === 'pre_replan_clock');
    const fields = JSON.parse(fs.readFileSync(path.join(WF, '..', 'e2e_workflow', 'scripts', 'eikos_questions', 'round_continue.json'), 'utf8')).required_state;
    ok('state carries exactly the question file\'s required fields', JSON.stringify(Object.keys(s).sort()) === JSON.stringify(fields.slice().sort()));
  }

  // ---- eligibility = the host's own refusal predicate
  {
    const fixed = load({ DEADLINE_EPOCH: 1, forcedReplans: 0 });
    ok('fixed window with time left -> not permitted', fixed.eikosStopPermitted(3600).reason === 'fixed_window');
    ok('fixed window inside the last NO_STOP_S -> permitted', fixed.eikosStopPermitted(600).permitted === true);
    ok('forced re-plan cap exhausted -> permitted (host allows degraded stop)',
       load({ DEADLINE_EPOCH: 1, forcedReplans: 6 }).eikosStopPermitted(3600).permitted === true);
    ok('unreadable clock -> skipped as clock_unavailable', fixed.eikosStopPermitted(Infinity).reason === 'clock_unavailable');
    ok('no deadline -> permitted', load({ DEADLINE_EPOCH: 0 }).eikosStopPermitted(Infinity).permitted === true);
  }

  // ---- ineligible rounds make ZERO Eikos calls (the historical fixed-window case)
  {
    const L = load({ DEADLINE_EPOCH: 1 });
    const r = await L.eikosShadowRound(1, L.eikosSnapshot(1, 9000, 'pre_plan_clock'), 'continue', CONT, 0, 9000);
    ok('fixed-window round: skipped, zero agent calls, zero Eikos calls',
       r.carrier.status === 'skipped' && r.carrier.eikos_calls === 0 && L.calls.length === 0 && hits === 0 &&
       r.effective_baseline_action === 'continue' && r.host_stop_reason === 'none');
  }

  // ---- synthetic eligible fixture: real bash -> real eikos_decide.py -> fake Eikos
  const evalDir = fs.mkdtempSync(path.join(os.tmpdir(), "eikos shadow it's ")) ;
  const extraDirs = [];
  const evalDir2 = () => { const d = fs.mkdtempSync(path.join(os.tmpdir(), 'eikos shadow 2 ')); extraDirs.push(d); return d; };
  {
    // Hostile text first, in its own dir. The script refuses this outcome object (not the lane's
    // shape), but bash parses the command BEFORE the script validates anything, so the quoting is
    // still what stands between this text and the shell.
    const H = load({ DEADLINE_EPOCH: 0, EVAL_DIR: evalDir2(), agent: bashCarrier({ GEAK_EIKOS_URL: URL }) });
    const marker = path.join(os.tmpdir(), `eikos_injection_marker_${process.pid}`);
    try { fs.unlinkSync(marker); } catch (e) {}
    H.setLast({ note: `it's a "quoted"\nmulti-line outcome; $(touch ${marker}) \`touch ${marker}\``, verified_candidates: 1 });
    const h = await H.eikosShadowRound(2, H.eikosSnapshot(2, Infinity, 'pre_plan_clock'), 'continue', CONT, 0, Infinity);
    ok('state with quotes, newline, $(...) and backticks reached the script intact (digest relay_ok)', h.relay_ok === true);
    ok('hostile outcome refused by the state check before any Eikos request',
       h.envelope && h.envelope.status === 'state_invalid' && hits === 0);

    const L = load({ DEADLINE_EPOCH: 0, EVAL_DIR: evalDir, agent: bashCarrier({ GEAK_EIKOS_URL: URL }) });
    L.setLast({ verified_candidates: 1, winner_speedup: 1.31, improved: true, made_progress: true,
                commit_reported: 'not_captured', tracked_incumbent_after: 1.31 });   // the lane's real shape
    const snap = L.eikosSnapshot(2, Infinity, 'pre_plan_clock');
    const r = await L.eikosShadowRound(2, snap, 'continue', CONT, 0, Infinity);
    ok('eligible: carrier ran once on the fixed Sonnet 5.5 model, with schema', L.calls.length === 1 &&
       L.calls[0].o.model === 'claude-sonnet-5-5' && L.calls[0].o.label === 'eikos:round_continue r2' && !!L.calls[0].o.schema);
    ok('envelope ok: choice continue, would_be_action continue at 0.91 >= 0.8',
       r.envelope && r.envelope.status === 'ok' && r.envelope.choice === 'continue' && r.envelope.would_be_action === 'continue');
    ok('envelope carries the state digest, not the state', r.envelope && !('state_raw' in r.envelope) && /^[0-9a-f]{64}$/.test(r.envelope.state_sha256));
    ok('decision digest verified (decision_ok) on the real script output', r.decision_ok === true);
    ok('receipt written and reported (receipt_ok); confidence travels as a string', r.receipt_ok === true && typeof r.envelope.confidence === 'string');
    ok('receipts written under the lane dir (path with space and apostrophe)',
       fs.existsSync(path.join(evalDir, 'eikos', 'round_continue.index.json')));
    ok('nothing in the state executed as shell ($(touch ...) and backticks left no marker)', !fs.existsSync(marker) && hits === 1);
    // a repeated execution for the same round/state returns the first decision without asking again
    const r2 = await L.eikosShadowRound(2, snap, 'continue', CONT, 0, Infinity);
    ok('repeat for the same key: reused first decision, no second Eikos request',
       r2.envelope && r2.envelope.reused === true && r2.envelope.decision_attempt_id === r.envelope.attempt_id && hits === 1);
  }

  // ---- stop: outcome not observed; host reason kept separate from the model's raw choice
  {
    const L = load({ DEADLINE_EPOCH: 0, EVAL_DIR: evalDir, agent: bashCarrier({ GEAK_EIKOS_URL: URL }) });
    const r = await L.eikosShadowRound(3, L.eikosSnapshot(3, Infinity, 'pre_plan_clock'), 'stop', STOP, 0, Infinity);
    ok('baseline stop: effective stop, tech_lead_stop, next round not_observed',
       r.effective_baseline_action === 'stop' && r.host_stop_reason === 'tech_lead_stop' &&
       r.raw_model_choice.final === 'stop' && r.outcome && r.outcome.next_round === 'not_observed');
    const E = load({ DEADLINE_EPOCH: 1, forcedReplans: 6, EVAL_DIR: evalDir, agent: bashCarrier({ GEAK_EIKOS_URL: URL }) });
    const e = await E.eikosShadowRound(4, E.eikosSnapshot(4, 3600, 'pre_replan_clock'), 'stop', { stop: false, directions: [] }, 6, 3600);
    ok('cap exhausted + empty plan: forced_window_exhausted, raw first/final kept apart',
       e.host_stop_reason === 'forced_window_exhausted' && e.raw_model_choice.first === 'stop' &&
       e.raw_model_choice.final === 'empty_directions' && e.forced_replans_this_round === 6);
  }

  // ---- carrier failure modes
  {
    const fire = (fn) => { Promise.resolve().then(fn); return 1; };
    const T = load({ DEADLINE_EPOCH: 0, agent: () => new Promise(() => {}), setTimeout: fire, clearTimeout: () => {} });
    const t = await T.eikosShadowRound(2, T.eikosSnapshot(2, Infinity, 'pre_plan_clock'), 'continue', CONT, 0, Infinity);
    ok('carrier wait bound: timeout recorded honestly (not cancelled; late finish cannot change the decision)',
       t.carrier.status === 'timeout' && /not cancelled/.test(t.carrier.note) && T.calls.length === 1);
    const X = load({ DEADLINE_EPOCH: 0, agent: async () => { throw new Error('boom'); } });
    const x = await X.eikosShadowRound(2, X.eikosSnapshot(2, Infinity, 'pre_plan_clock'), 'continue', CONT, 0, Infinity);
    ok('carrier error: recorded, one attempt only (no retries)', x.carrier.status === 'error' && X.calls.length === 1);
    const M = load({ DEADLINE_EPOCH: 0, agent: async () => ({ status: 'ok', choice: 'stop', state_sha256: '0'.repeat(64) }) });
    const m = await M.eikosShadowRound(2, M.eikosSnapshot(2, Infinity, 'pre_plan_clock'), 'continue', CONT, 0, Infinity);
    ok('altered relay detected (relay_ok false)', m.relay_ok === false);
  }
  {
    // A carrier that flips the choice but keeps the script's digest is caught.
    const real = load({ DEADLINE_EPOCH: 0, EVAL_DIR: evalDir2(), agent: bashCarrier({ GEAK_EIKOS_URL: URL }) });
    const snap = real.eikosSnapshot(5, Infinity, 'pre_plan_clock');
    const honest = await real.eikosShadowRound(5, snap, 'continue', CONT, 0, Infinity);
    const flip = load({ DEADLINE_EPOCH: 0, agent: async () => Object.assign({}, honest.envelope, { choice: 'stop', would_be_action: 'stop' }) });
    const f = await flip.eikosShadowRound(5, snap, 'continue', CONT, 0, Infinity);
    ok('carrier that alters the decision is detected (decision_ok false), state still ok', f.decision_ok === false && f.relay_ok === true);
    const extra = load({ DEADLINE_EPOCH: 0, agent: async () => Object.assign({}, honest.envelope, { status_: 'ok' }) });
    const x2 = await extra.eikosShadowRound(5, snap, 'continue', CONT, 0, Infinity);
    ok('an added non-decision field (as observed live) does not break decision_ok', x2.decision_ok === true);
    const Z = load({ DEADLINE_EPOCH: 0, agent: async () => null });
    const z = await Z.eikosShadowRound(2, Z.eikosSnapshot(2, Infinity, 'pre_plan_clock'), 'continue', CONT, 0, Infinity);
    ok('no result: recorded as no_result', z.carrier.status === 'no_result');
  }

  ok('shell quoting round-trips an apostrophe', load().eikosShellQuote("a'b") === "'a'\\''b'");
  {
    // Astra's digest counterexample: Python confidence 1.0 must not read as an altered decision.
    const { execFileSync } = require('child_process');
    const SCRIPTS = path.join(WF, '..', 'e2e_workflow', 'scripts');
    const pyFields = JSON.parse(execFileSync('python3', ['-B', '-c', 'import sys,json;sys.path.insert(0,sys.argv[1]);import eikos_decide as e;print(json.dumps(list(e.DECISION_FIELDS)))', SCRIPTS]).toString());
    const jsFields = SRC.slice(SRC.indexOf('const EIKOS_DECISION_FIELDS'), SRC.indexOf('];', SRC.indexOf('const EIKOS_DECISION_FIELDS')));
    ok('JS digest field list equals Python DECISION_FIELDS', pyFields.every((f) => jsFields.includes(`'${f}'`)) && (jsFields.match(/'/g) || []).length === pyFields.length * 2);
    let all = true;
    for (const conf of ['1.0', '0.92', '0.0078125', '1e-07', '5e-324', '0.0']) {
      const L = load({ DEADLINE_EPOCH: 0 });
      const snap = L.eikosSnapshot(2, Infinity, 'pre_plan_clock');
      const st = JSON.stringify(snap);
      const code = 'import sys,json;sys.path.insert(0,sys.argv[1]);import eikos_decide as e;print(json.dumps(e.compact({"attempt_id":"a","decision_attempt_id":"a","logical_key":"k","status":"ok","choice":"continue","confidence":float(sys.argv[2]),"would_be_action":"continue","reused":False,"persisted":True,"receipt":"written","state_sha256":e.sha256(sys.argv[3])})))';
      const env = JSON.parse(execFileSync('python3', ['-B', '-c', code, SCRIPTS, conf, st]).toString());
      const L2 = load({ DEADLINE_EPOCH: 0, agent: async () => env });
      const res = await L2.eikosShadowRound(2, snap, 'continue', CONT, 0, Infinity);
      if (!(res.decision_ok === true && res.relay_ok === true)) { all = false; console.log('   mismatch for confidence', conf); }
    }
    ok('decision_ok holds for confidence 1.0, 0.92, 0.0078125, 1e-07, 5e-324, 0.0 from real Python output', all);
    const R = load({ DEADLINE_EPOCH: 0, agent: async () => ({ status: 'ok', receipt: 'failed', state_sha256: 'x' }) });
    const rr = await R.eikosShadowRound(2, R.eikosSnapshot(2, Infinity, 'pre_plan_clock'), 'continue', CONT, 0, Infinity);
    ok('receipt failure is a recorded collection failure (receipt_ok false)', rr.receipt_ok === false);
  }
  {
    const crypto = require('crypto');
    const L = load();
    const samples = ['', 'abc', "it's \"q\"\n$(x) `y`", 'é ✓ 漢字 🚀', 'x'.repeat(1000), JSON.stringify({ a: [1, 2], b: 'ü' })];
    const { execFileSync } = require('child_process');
    const obj = { b: [1, 2.5, 'é'], a: { z: null, y: true }, c: 0.8438951373100281, d: "it's" };
    const py = execFileSync('python3', ['-c', 'import json,sys;print(json.dumps(json.loads(sys.argv[1]),sort_keys=True,separators=(",",":"),ensure_ascii=False),end="")', JSON.stringify(obj)]).toString('utf8');
    ok('JS canonical JSON matches Python json.dumps(sort_keys, compact, ensure_ascii=False)', L.eikosCanonical(obj) === py);
    ok('pure-JS SHA-256 matches node crypto (ASCII, UTF-8, multi-block)',
       samples.every((t) => L.eikosSha256(t) === crypto.createHash('sha256').update(t, 'utf8').digest('hex')));
  }
  srv.close();
  fs.rmSync(evalDir, { recursive: true, force: true });
  extraDirs.forEach((d) => fs.rmSync(d, { recursive: true, force: true }));
  console.log(`\n${fails === 0 ? 'ALL PASS' : 'FAIL'} — ${n - fails}/${n} checks`);
  process.exit(fails === 0 ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(1); });
