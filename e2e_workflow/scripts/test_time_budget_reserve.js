#!/usr/bin/env node
// Regression guard for the wall-clock budget -> final reserve contract (no GPU, no model needed).
//
// Hyperloom #1202 round 2: three 20260823 sessions burned their whole budget and shipped NO final
// report despite a configured reserve. Two defects, both pinned here:
//
//   1. ELAPSED_MS under-counted -- a self-rearming setTimeout chain compounds scheduler lateness, so
//      remainingMs() reported time the run did not have.
//   2. The Finalize-gate had no wall-clock bound: its drain loop boots a server and benches two legs
//      per iteration, unbounded. That loop is where all three runs were SIGKILLed.
//
// The clock is EXTRACTED from the real source and run against a virtual scheduler that injects
// lateness; a reimplementation here would pass while the shipped code drifted. The Finalize-gate is
// checked structurally (200 lines of async orchestration, not executable standalone).
//
// Run:  node e2e_workflow/scripts/test_time_budget_reserve.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const WORKFLOW = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const src = fs.readFileSync(WORKFLOW, 'utf8');

// ---------------------------------------------------------------- virtual scheduler
// Fires timers in due order, each LATE by `lateness` ms. Timers armed inside a callback are relative
// to the (already late) virtual now -- which is precisely how a self-rearming chain accumulates drift.
function runVirtualClock(arm, { horizonMs, lateness }) {
  let now = 0, seq = 0;
  const pending = [];
  const setTimeout_ = (fn, delay) => {
    pending.push({ due: now + Math.max(0, delay | 0), seq: seq++, fn });
    return { unref() {} };
  };
  const elapsed = arm(setTimeout_);
  const samples = [];
  for (;;) {
    let best = -1;
    for (let i = 0; i < pending.length; i++) {
      if (best === -1 || pending[i].due < pending[best].due
        || (pending[i].due === pending[best].due && pending[i].seq < pending[best].seq)) best = i;
    }
    if (best === -1) break;
    const t = pending.splice(best, 1)[0];
    now = t.due + lateness;          // the callback runs LATE
    if (now > horizonMs) break;
    t.fn();
    samples.push({ wall: now, elapsed: elapsed() });
  }
  return { samples, finalWall: now, finalElapsed: elapsed() };
}

// The shipped ladder, lifted verbatim out of e2e_workflow.js.
const cStart = src.indexOf('let ELAPSED_MS = 0;');
const cEnd = src.indexOf('const remainingMs =');
ok(cStart !== -1 && cEnd !== -1 && cStart < cEnd, 'elapsed-clock block located in e2e_workflow.js');
if (failures) process.exit(1);
const clockSrc = src.slice(cStart, cEnd);
const armShipped = (TIME_BUDGET_MS) => (setTimeout) => new Function(
  'TIME_BUDGET_MS', 'setTimeout',
  `${clockSrc}\nreturn () => ELAPSED_MS;`)(TIME_BUDGET_MS, setTimeout);

// The PRE-FIX clock, kept only as the counter-example this test discriminates against.
const armChain = (TIME_BUDGET_MS) => (setTimeout) => {
  let ELAPSED_MS = 0;
  const CLOCK_TICK_MS = 60000;
  const arm = () => { const t = setTimeout(tick, CLOCK_TICK_MS); if (t && t.unref) t.unref(); };
  const tick = () => { ELAPSED_MS += CLOCK_TICK_MS; arm(); };
  if (TIME_BUDGET_MS != null) arm();
  return () => ELAPSED_MS;
};

const H = 12 * 3600 * 1000;          // 12h budget: the real Hyperloom KERNEL-phase shape
const LATE = 5000;                   // 5s of scheduler lag per rung on a saturated event loop

console.log('\n# the clock tracks real time under scheduler lateness');
const shipped = runVirtualClock(armShipped(H), { horizonMs: H, lateness: LATE });
const chain = runVirtualClock(armChain(H), { horizonMs: H, lateness: LATE });

const worst = (r) => r.samples.reduce((m, s) => Math.max(m, s.wall - s.elapsed), 0);
const shippedDrift = worst(shipped), chainDrift = worst(chain);
ok(shippedDrift <= 60000 + LATE + 1,
  `absolute ladder stays within one step+lag of real time (worst under-count ${Math.round(shippedDrift / 1000)}s)`);
ok(chainDrift > 30 * 60000,
  `the pre-fix chain compounds to >30min of under-count (${Math.round(chainDrift / 60000)}min) -- ` +
  'this is what spent the reserve');
ok(shippedDrift * 20 < chainDrift, 'ladder drift is more than an order of magnitude below the chain');

console.log('\n# the clock reaches the end of the budget');
// remainingMs() must actually be able to hit 0, or the deadline guards never trip.
const toEnd = runVirtualClock(armShipped(H), { horizonMs: H + 10 * 60000, lateness: LATE });
ok(toEnd.finalElapsed >= H, `ELAPSED_MS reaches the full budget (${Math.round(toEnd.finalElapsed / 60000)}min >= ${H / 60000}min)`);

console.log('\n# an absurd budget cannot flood the event loop with timers');
let armed = 0;
new Function('TIME_BUDGET_MS', 'setTimeout', `${clockSrc}\nreturn () => ELAPSED_MS;`)(
  400 * 24 * 3600 * 1000, () => { armed++; return { unref() {} }; });
ok(armed <= 2049, `a 400-day budget arms ${armed} rungs, not ${Math.round(400 * 24 * 60)}`);

console.log('\n# the default reserve is one hour');
// Pinned because it has already been moved twice (3600 -> 3000 -> 3600). Large models spend the final
// phase mostly waiting on server boots, and a p75-sized reserve is what left three of them with no report.
{
  const rStart = src.indexOf('const FINAL_RESERVE_CAP_FRAC');
  const rSrc = src.slice(rStart, src.indexOf('const TIME_BUDGET_EFFECTIVE_MS', rStart));
  const reserveFor = (TIME_BUDGET_MS, A) => new Function('TIME_BUDGET_MS', 'A',
    `${rSrc}\nreturn FINAL_RESERVE_MS;`)(TIME_BUDGET_MS, A);
  ok(reserveFor(12 * 3600 * 1000, {}) === 60 * 60000,
    'a 12h budget reserves the full 60min default (the 20% cap does not bind)');
  ok(reserveFor(3 * 3600 * 1000, {}) === Math.floor(0.2 * 3 * 3600 * 1000),
    'below a 5h budget the 20% cap binds instead of the default (3h -> 36min)');
  ok(reserveFor(12 * 3600 * 1000, { final_reserve_s: 5400 }) === 90 * 60000,
    'GEAK_FINAL_RESERVE_S still widens it, up to the same 20% cap');
  ok(reserveFor(12 * 3600 * 1000, { final_reserve_s: 19800 }) === Math.floor(0.2 * 12 * 3600 * 1000),
    'a reserve wider than 20% of the budget is still capped by default');
  ok(reserveFor(16 * 3600 * 1000, { final_reserve_s: 19800, final_reserve_cap_frac: 0.4 }) === 19800 * 1000,
    'GEAK_FINAL_RESERVE_CAP_FRAC raises the cap for a final phase that needs longer legs');
  ok(reserveFor(12 * 3600 * 1000, { final_reserve_s: 40000, final_reserve_cap_frac: 0.9 }) === 0.5 * 12 * 3600 * 1000,
    'the cap itself is clamped to half the budget, so the search keeps the other half');
  ok(reserveFor(3 * 3600 * 1000, { final_reserve_cap_frac: 'junk' }) === Math.floor(0.2 * 3 * 3600 * 1000),
    'a junk cap falls back to the 20% default');
}

console.log('\n# a trace replay reserves what its own final phase costs');
// Finalize benches one 900s search leg and Validate two canonical 3600s legs, each behind a boot and
// the replay's own warmup and drain: about four hours, against a one-hour default. The GLM-5.2 HL+GEAK
// run only had room for it because its operator set an 8h reserve by hand.
{
  const rStart = src.indexOf('const FINAL_RESERVE_CAP_FRAC');
  const rSrc = src.slice(rStart, src.indexOf('const TIME_BUDGET_EFFECTIVE_MS', rStart));
  const reserve = (TIME_BUDGET_MS, A) => new Function('TIME_BUDGET_MS', 'A',
    `${rSrc}\nreturn { FINAL_RESERVE_MS, AGENTX_FINAL_PHASE_MS, AGENTX_PRE_FINAL_MIN_MS, AGENTX_MIN_BUDGET_MS,
      RESERVE_IS_AGENTX };`)(TIME_BUDGET_MS, A);
  const HR = 3600 * 1000, MN = 60000;
  const AX = { workload_kind: 'agentx_trace_replay' };
  const ax12 = reserve(12 * HR, AX);
  ok(ax12.AGENTX_FINAL_PHASE_MS === 235 * MN && ax12.FINAL_RESERVE_MS === 235 * MN,
    'a 12h AgentX budget reserves 235min (one 900s and two 3600s legs at +20min each, plus 40min), not 60min');
  ok(reserve(70920 * 1000, { ...AX, final_reserve_s: 28800, final_reserve_cap_frac: 0.45 }).FINAL_RESERVE_MS === 28800 * 1000,
    'a wider reserve the operator configured is kept (GEAK_FINAL_RESERVE_S=28800 on a 19h42m budget)');
  // The run's minimum: Setup's baseline leg (+10min), one A/B on the search window (the KB warm start's)
  // and the final phase. Any budget that holds it gives the final phase its whole cost.
  ok(ax12.AGENTX_PRE_FINAL_MIN_MS === (600 + 4800 + 2 * 2100 + 600) * 1000 && ax12.AGENTX_MIN_BUDGET_MS === 405 * MN,
    "the minimum is Setup's leg (90min), one A/B (80min) and the final phase (235min): 405min, 6h45m");
  ok(reserve(405 * MN, AX).FINAL_RESERVE_MS === 235 * MN && reserve(7 * HR, AX).FINAL_RESERVE_MS === 235 * MN,
    'a budget at the minimum, or 7h, still reserves the whole final phase');
  ok(reserve(6 * HR, AX).FINAL_RESERVE_MS === 190 * MN,
    'under it, Setup and that A/B keep their 170min and the final phase gets the rest (6h -> 190min)');
  ok(reserve(3 * HR, AX).FINAL_RESERVE_MS === 36 * MN,
    'a budget that cannot even hold Setup and one A/B keeps the configured reserve (20% of 3h)');
  ok(/\(TIME_BUDGET_MS < AGENTX_MIN_BUDGET_MS\s*\? ' WARNING: the budget is under it, so this run may not return a validated gain/.test(src),
    'the startup log warns on a budget under the minimum');
  ok(reserve(12 * HR, { workload_spec: { kind: 'agentx_trace_replay', duration_s: 1800, geak_loop_duration_s: 600 } })
    .AGENTX_FINAL_PHASE_MS === (2400 + 1800 + 2 * 3000) * 1000, 'it follows the windows the run declares');
  ok(reserve(24 * HR, { ...AX, validation_measurement_mode: 'isolated_server', validation_replicas: 3 })
    .FINAL_RESERVE_MS === (2400 + 2100 + 6 * 4800) * 1000, 'isolated validation counts every fresh server: six legs');
  ok(reserve(12 * HR, { ...AX, agentx_leg_overhead_s: 600 }).AGENTX_FINAL_PHASE_MS === (2400 + 1500 + 2 * 4200) * 1000,
    'agentx_leg_overhead_s sizes the boot, warmup and drain of every leg');
  ok(reserve(null, AX).FINAL_RESERVE_MS === null, 'no time_budget_s => no reserve, as before');
  const SYNTHETIC = [{}, { workload_kind: 'synthetic' }, { workload_spec: { kind: 'synthetic' } },
    { workload_kind: 'synthetic', state: { workload_spec: { kind: 'agentx_trace_replay' } } }];
  ok(SYNTHETIC.every((A) => [3, 5, 12, 24].every((h) => {
    const r = reserve(h * HR, A);
    return !r.RESERVE_IS_AGENTX && r.AGENTX_FINAL_PHASE_MS === 0
      && r.FINAL_RESERVE_MS === Math.min(3600 * 1000, Math.floor(h * HR * 0.2));
  })), 'every other workload keeps exactly the reserve it had');

  // The declaration is evaluated after the reserve, so the reserve resolves AgentX from the raw args;
  // run the real declaration beside it so the two can never disagree about what this run is.
  const dStart = src.indexOf('const WORKLOAD_SPEC =');
  const dEnd = src.indexOf('\n', src.indexOf('const AGENTX = IS_AGENTX'));
  const declared = (A) => new Function('A', `${src.slice(dStart, dEnd)}\nreturn { IS_AGENTX, AGENTX };`)(A);
  const SHAPES = [...SYNTHETIC, AX, { workload_spec: { kind: 'agentx_trace_replay', duration_s: 1800 } },
    { state: { workload_spec: { kind: 'agentx_trace_replay', geak_loop_duration_s: 600 } } },
    { workload_kind: 'agentx_trace_replay', state: { workload_spec: { kind: 'agentx_trace_replay', duration_s: 1800 } } },
    { workload_kind: 'agentx_trace_replay', workload_spec: 'junk' }];
  const disagree = SHAPES.filter((A) => {
    const d = declared(A), r = reserve(12 * HR, A);
    if (d.IS_AGENTX !== r.RESERVE_IS_AGENTX) return true;
    return d.IS_AGENTX && (r.AGENTX_FINAL_PHASE_MS !==
      1000 * (2400 + (d.AGENTX.geak_loop_duration_s + 1200) + 2 * (d.AGENTX.duration_s + 1200))
      || r.AGENTX_PRE_FINAL_MIN_MS !==
      1000 * (600 + (d.AGENTX.duration_s + 1200) + 2 * (d.AGENTX.geak_loop_duration_s + 1200) + 600));
  });
  ok(disagree.length === 0,
    `the reserve resolves AgentX and its windows exactly as the declaration does (${JSON.stringify(disagree)})`);
}

console.log('\n# the reserve boundary is armed and gates the Finalize-gate');
ok(/let TIME_FINAL_DEADLINE_HIT = false;/.test(src), 'TIME_FINAL_DEADLINE_HIT exists');
ok(/TIME_FINAL_DEADLINE_HIT = true;/.test(src) && /}, TIME_BUDGET_EFFECTIVE_MS\);/.test(src),
  'it is armed by an absolute timer at TIME_BUDGET_EFFECTIVE_MS (budget minus reserve)');

// The drain loop is the exact code that ran for 3-4h past the deadline in all three sessions. Its
// deadline check must come BEFORE the shift()/integrate work, every iteration -- not once on entry.
const gate = src.indexOf("if (want('final'))");
ok(gate !== -1, "Finalize-gate located");
const loop = src.indexOf('while (pendingIntegrations.length)', gate);
ok(loop !== -1 && loop > gate, 'pendingIntegrations drain loop located inside the gate');
const body = src.slice(loop, src.indexOf('runIntegrateBothLegs', loop));
ok(/TIME_FINAL_DEADLINE_HIT/.test(body) && /\bbreak;/.test(body),
  'the drain loop breaks on TIME_FINAL_DEADLINE_HIT before reaching runIntegrateBothLegs');
ok(src.slice(gate, loop).includes('TIME_FINAL_DEADLINE_HIT'),
  'the gate is also skipped wholesale when the reserve has already begun');
// Deferred, not discarded: the caller must still see the unfinished A/Bs.
ok(/pending_integrations/.test(src.slice(gate, loop + 4000)),
  'unfinished A/Bs are surfaced to the caller rather than dropped');

console.log('\n# the per-agent reserve cap is waived by POSITION, not by a self-reported label');
// The round-2 hole: agentTimeoutFor() exempted opts.phase in ['Finalize','Report','Validate'], and the
// Finalize-GATE tags its agents 'Finalize' too -- so optimization work inherited the exemption and ran
// unbounded. A self-reported label fails OPEN. Executed, not grepped: lifted from the real source.
const aStart = src.indexOf('let FINAL_PHASE_STARTED = false;');
const aEnd = src.indexOf('\n}', src.indexOf('function agentTimeoutFor')) + 2;
ok(aStart !== -1 && aEnd > aStart, 'agentTimeoutFor located');
const mk = (TIME_BUDGET_MS, ELAPSED_MS) => new Function(
  'TIME_BUDGET_MS', 'AGENT_TIMEOUT_MS', 'FINAL_RESERVE_MS', 'remainingMs',
  `${src.slice(aStart, aEnd)}
   return { agentTimeoutFor, enterFinalPhase: () => { FINAL_PHASE_STARTED = true; } };`)(
  TIME_BUDGET_MS, 2 * 3600 * 1000, 60 * 60000, () => Math.max(0, TIME_BUDGET_MS - ELAPSED_MS));

const RESERVE = 60 * 60000, AGENT_MAX = 2 * 3600 * 1000;
// 40min left of a 12h budget: less than the 60min reserve, so a capped agent floors at 2min.
const near = mk(H, H - 40 * 60000);
ok(near.agentTimeoutFor() === 120000,
  'before the final phase, an agent floors at 2min near the deadline — no caller can opt out');
near.enterFinalPhase();
ok(near.agentTimeoutFor() === AGENT_MAX,
  'once phase(\'Finalize\') has actually run, the cap lifts — the final phase owns the reserve');

// Mid-run the cap is the real remaining-minus-reserve, not the flat AGENT_TIMEOUT_MS.
const mid = mk(H, 10 * 3600 * 1000);
ok(mid.agentTimeoutFor() === AGENT_MAX - RESERVE,
  'with 2h left an agent gets 2h minus the 60min reserve, not a flat 2h');

// No budget => the feature is inert and every agent keeps the plain timeout.
const noBudget = new Function('TIME_BUDGET_MS', 'AGENT_TIMEOUT_MS', 'FINAL_RESERVE_MS', 'remainingMs',
  `${src.slice(aStart, aEnd)}\nreturn agentTimeoutFor;`)(null, AGENT_MAX, null, () => Infinity);
ok(noBudget() === AGENT_MAX, 'no time_budget_s => byte-identical to a build without the feature');

// The flag must be granted in exactly ONE place, and that place must be the final phase entry.
ok((src.match(/FINAL_PHASE_STARTED = true/g) || []).length === 1,
  'the exemption is granted in exactly one place');
const grant = src.indexOf('FINAL_PHASE_STARTED = true');
ok(src.slice(grant, grant + 200).includes("phase('Finalize')"),
  "it is granted immediately before phase('Finalize'), so position is what decides");
// Nothing may pass a phase/label to buy the exemption any more.
ok(/function agentTimeoutFor\(\)/.test(src) && !/FINAL_PHASE_LABELS/.test(src),
  'agentTimeoutFor takes no arguments — there is no label left to get wrong');
ok(src.slice(gate, grant).indexOf('FINAL_PHASE_STARTED = true') === -1,
  'the Finalize-gate runs entirely before the exemption is granted');

console.log('\n# nothing measured starts once a new agent would only get the 2min floor');
// The 20261011 GLM run: WarmStart's KB integrator was cut off at the boundary, three FINISH retries
// each got 2min and were abandoned too, then Profile dispatched — and that abandoned profiler held the
// GPUs for an hour while Finalize waited. Every one of those agents ran on after its promise was dropped.
const fStart = aStart;
const fEnd = src.indexOf('\n}', src.indexOf('function atReserveFloor')) + 2;
ok(fEnd > src.indexOf('function atReserveFloor') && src.indexOf('function atReserveFloor') > aEnd,
  'atReserveFloor located right after agentTimeoutFor');
const floorAt = (TIME_BUDGET_MS, leftMs) => new Function(
  'TIME_BUDGET_MS', 'AGENT_TIMEOUT_MS', 'FINAL_RESERVE_MS', 'remainingMs',
  `${src.slice(fStart, fEnd)}
   return { atReserveFloor, agentTimeoutFor, enterFinalPhase: () => { FINAL_PHASE_STARTED = true; } };`)(
  TIME_BUDGET_MS, AGENT_MAX, RESERVE, () => leftMs);
ok(!floorAt(H, RESERVE + 3 * 60000).atReserveFloor(), '3min before the boundary an agent still gets real time');
const edge = floorAt(H, RESERVE + 60000);
ok(edge.atReserveFloor() && edge.agentTimeoutFor() === 120000,
  'one rung of clock lag before the boundary already counts — that agent would get the floor');
const inside = floorAt(H, 40 * 60000);
ok(inside.atReserveFloor(), 'inside the reserve: true');
inside.enterFinalPhase();
ok(!inside.atReserveFloor(), 'never once the final phase has started — it owns the reserve');
ok(!new Function('TIME_BUDGET_MS', 'AGENT_TIMEOUT_MS', 'FINAL_RESERVE_MS', 'remainingMs',
  `${src.slice(fStart, fEnd)}\nreturn atReserveFloor;`)(null, AGENT_MAX, null, () => Infinity)(),
'no time_budget_s => never');

const scSrc = (src.match(/function searchClosed\(what\) \{[\s\S]*?\n\}/) || [])[0];
ok(!!scSrc, 'searchClosed(what) located');
const closedFor = (flags) => {
  const logs = [];
  const fn = new Function('FINAL_PHASE_STARTED', 'TIME_FINAL_DEADLINE_HIT', 'atReserveFloor', 'log',
    `${scSrc}\nreturn searchClosed;`)(flags.final, flags.hit, () => flags.floor, (m) => logs.push(m));
  return { closed: fn('Profile'), logs };
};
ok(!closedFor({ final: false, hit: false, floor: false }).closed, 'mid-search: open, and silent');
const byTimer = closedFor({ final: false, hit: true, floor: false });
ok(byTimer.closed && /Profile SKIPPED/.test(byTimer.logs.join('\n')),
  'closed by the drift-free boundary timer, and the skip is logged by name');
ok(closedFor({ final: false, hit: false, floor: true }).closed, 'closed by the floor rule just before it');
ok(!closedFor({ final: true, hit: true, floor: false }).closed, 'never closes anything in the final phase');

// Every dispatch ahead of the Finalize-gate that could straddle the boundary checks first.
for (const [gateArg, dispatch] of [
  ["'Profile and Strategize'", "phase('Profile');"],
  ["'ConfigSweep'", "phase('ConfigSweep');"],
  ["'post-config re-profile and re-strategize'", "label: 'profiler:post-config'"],
  ["'TuningSkillset'", "phase('TuningSkillset');"],
  ["'post-tuning re-profile and re-strategize'", "label: 'profiler:post-tuning'"],
  ["'post-head re-profile'", "label: 'profiler:post-head'"],
  ['`[kb] replay of config', 'const runValidation'],
  ['`[kb] repair of config', 'warm_start:repair:'],
  ['`[kb] replay of kernel', 'runIntegrateBothLegs('],
]) {
  const g = src.indexOf(`searchClosed(${gateArg}`);
  const d = g === -1 ? -1 : src.indexOf(dispatch, g);
  // Only the gated call itself may sit between the gate and its marker.
  const between = d > g ? src.slice(g, d) : '';
  const dispatches = (between.match(/\b(?:safeAgent|runIntegrateBothLegs|runValidation)\(/g) || []).length;
  ok(g !== -1 && d > g && dispatches <= 1 && d - g < 4000, `${dispatch} is gated by searchClosed(${gateArg}...)`);
}
ok(/if \(searchClosed\('Profile and Strategize'\) \|\| searchWindowShort\('[^']+'\)\) \{\s*kernelQueue = \[\];\s*headQueue = \[\];/.test(src),
  'skipping Strategize still leaves both queues arrays (they are declared without a value)');

console.log('\n# a trace replay starts no search work that would run into the reserve');
// The GLM-5.2 run again: a head's second author lane started 36min before the boundary and ran 4h08m
// past it, and an A/B was dispatched inside the reserve. The lane was already running when the boundary
// closed the search, so the checks above could not stop it; these decide before work starts.
const gStart = src.indexOf('// --- TRACE-REPLAY SEARCH vs THE RESERVE');
const gEnd = src.indexOf('// GPU semaphore (FAST MODE only', gStart);
ok(gStart !== -1 && gEnd > gStart, 'the trace-replay search guards located');
const lineOf = (needle) => src.slice(src.indexOf(needle), src.indexOf('\n', src.indexOf(needle)));
const guardHarnessSrc = [
  'let ELAPSED_MS = 0;', lineOf('const remainingMs ='), lineOf('const obj = '),
  lineOf('const AGENT_TIMEOUT_FLOOR_MS ='),
  src.slice(src.indexOf('const FINAL_RESERVE_CAP_FRAC'), src.indexOf('const TIME_BUDGET_EFFECTIVE_MS')),
  src.slice(src.indexOf('const WORKLOAD_SPEC ='), src.indexOf('\n', src.indexOf('const AGENTX = IS_AGENTX'))),
  src.slice(gStart, gEnd),
  'ELAPSED_MS = TIME_BUDGET_MS == null ? 0 : TIME_BUDGET_MS - FINAL_RESERVE_MS - WINDOW_MS;',
  'return { abWindowShort, laneWindowShort, searchWindowShort, laneDeadlineArgs, laneReserveCapMs, agentxAbMs,',
  '  agentxLaneTailMs, abLegs, advance: (ms) => { ELAPSED_MS += ms; } };',
].join('\n');
const EPOCH = 1800000000;
const guardsFor = ({ A = { workload_kind: 'agentx_trace_replay' }, budgetMs = 12 * 3600 * 1000, windowMin = 0,
  finalStarted = false, clock = () => ({ epoch: EPOCH }) } = {}) => {
  const logs = [], reads = [];
  const api = new Function('A', 'TIME_BUDGET_MS', 'FINAL_PHASE_STARTED', 'WINDOW_MS', 'safeAgent', 'log',
    guardHarnessSrc)(A, budgetMs, finalStarted, windowMin * 60000,
    async (prompt, opts, tries, timeoutFor) => {
      reads.push({ prompt, opts, tries, timeoutMs: timeoutFor() });
      return clock(reads.length);
    }, (m) => logs.push(String(m)));
  return Object.assign(api, { logs, reads });
};
{
  const g = guardsFor({ windowMin: 300 });
  ok(g.agentxAbMs(2) === 80 * 60000 && g.agentxAbMs(1) === 45 * 60000 && g.agentxLaneTailMs() === 140 * 60000,
    'an A/B is two 900s legs at +20min each plus 10min (80min), 45min with its reference reused; a lane tail is that plus a round');
  ok(g.abLegs({}) === 2 && g.abLegs(undefined) === 2 && g.abLegs({ REUSE_REF: true, SHARED_REF_MED: 1 }) === 1
    && g.abLegs({ RESUME_AB: true }) === 1, 'a shared or on-disk reference leg is not counted twice');
  const short = guardsFor({ windowMin: 79 });
  ok(short.abWindowShort(2, 'head k') && /head k DEFERRED — ~79min left/.test(short.logs.join('\n')),
    'an A/B with 79min left before the boundary is deferred, and says so by name');
  ok(!guardsFor({ windowMin: 81 }).abWindowShort(2, 'x'), 'with 81min left it runs');
  ok(!guardsFor({ windowMin: 50 }).abWindowShort(1, 'x') && guardsFor({ windowMin: 44 }).abWindowShort(1, 'x'),
    'a one-leg A/B needs 45min');
  ok(g.agentxAbMs(2, 2) === 150 * 60000,
    "a speedup review's confirmation (two fresh legs of two timed passes) is costed as four legs: 150min");
  const review = guardsFor({ windowMin: 120 });
  ok(review.abWindowShort(2, 'h1: speedup review', 2)
    && /h1: speedup review DEFERRED — ~120min left .* ~150min a 2-leg trace-replay A\/B with 2 timed passes a leg takes/
      .test(review.logs.join('\n')) && !guardsFor({ windowMin: 120 }).abWindowShort(2, 'x'),
  'with 120min left an ordinary A/B runs but a confirmation is deferred, and says so by name');
  const late = guardsFor({ windowMin: 199 });
  ok(late.laneWindowShort('author lane k:triton') && /author lane k:triton NOT STARTED/.test(late.logs.join('\n')),
    'a lane with 199min left is not started: one round plus its A/B tail would not fit');
  ok(!guardsFor({ windowMin: 201 }).laneWindowShort('x'), 'with 201min left it starts');
  // After the KB warm start: kernel extraction (70min) and one lane (200min) must both still fit, or
  // the run goes straight to Finalize and Validate.
  const kbOnly = guardsFor({ windowMin: 269 });
  ok(kbOnly.searchWindowShort('Profile, kernel extraction and the kernel search')
    && /kernel search SKIPPED — ~269min left .* ~270min kernel extraction plus one kernel lane/.test(kbOnly.logs.join('\n'))
    && /Going straight to Finalize and Validate/.test(kbOnly.logs.join('\n')),
  'with 269min left after the warm start the search is skipped, and the run says it goes straight to Validate');
  ok(!guardsFor({ windowMin: 271 }).searchWindowShort('x'), 'with 271min left the search runs');
  ok(guardsFor({ windowMin: 300 }).laneReserveCapMs() === 300 * 60000
    && guardsFor({ windowMin: 1 }).laneReserveCapMs() === 120000,
  'a running lane is never waited for past the boundary (floored at the 2min agent floor)');
  const off = [guardsFor({ A: {}, windowMin: 1 }), guardsFor({ budgetMs: null }), guardsFor({ windowMin: 1, finalStarted: true })];
  ok(off.every((o) => !o.abWindowShort(2, 'x') && !o.laneWindowShort('x') && !o.searchWindowShort('x')
    && o.laneReserveCapMs() === null && o.logs.length === 0),
  'inert, and silent, for a synthetic run, without time_budget_s, and in the final phase');
}

// Where they are consulted. The integrate entry point covers every A/B site (head, milestone,
// Finalize-gate, disk-recovered) and the speedup review the confirmation it adds; the lane wrappers
// cover every lane the search starts.
{
  const ribSrc = src.slice(src.indexOf('async function runIntegrateBothLegs('),
    src.indexOf('if (integ && integ.short_name)', src.indexOf('async function runIntegrateBothLegs(')));
  ok(/^async function runIntegrateBothLegs\([^)]*\) \{\s*\/\/[^\n]*\n\s*if \(abWindowShort\(abLegs\(inputs\), label\)\) \{\s*return \{ gate: 'incomplete', ab_complete: false,/.test(ribSrc),
    'runIntegrateBothLegs defers an A/B that cannot finish before anything is dispatched, as incomplete (kept pending)');
  const rvSrc = src.slice(src.indexOf('async function reviewSpeedup('), src.indexOf('async function runIntegrateBothLegs('));
  const rvGate = rvSrc.indexOf('if (abWindowShort(abLegs(reviewInputs), `${label}: speedup review`, reviewInputs.REPLICAS)) {');
  ok(rvGate > rvSrc.indexOf('REPLICAS: Math.max(2,') && rvGate < rvSrc.indexOf('await safeAgent(')
    && /^\s*pending\.reason = 'speedup_review_inconclusive: deferred[^']*';\s*return pending;/.test(rvSrc.slice(rvSrc.indexOf('\n', rvGate))),
  'a speedup review checks its confirmation, counted with its timed passes, before dispatching it');
  const rx = (name) => new Function(`return ${src.match(new RegExp(`const ${name} = (/.*/[a-z]*);`))[1]};`)();
  const reason = (ribSrc.match(/reason: '(deferred: [^']+)'/) || [])[1] || '';
  ok(reason && !rx('FIXABLE_REJECT_RX').test(reason) && !rx('CORRECTNESS_REJECT_RX').test(reason),
    'its reason reads as neither a fixable nor a correctness reject, so no repair loop starts on a deferral');
  ok(/if \(searchClosed\(`\$\{label\}: FINISH retry`\)\) break;\s*if \(abWindowShort\(1, `\$\{label\}: FINISH retry`\)\) break;/.test(ribSrc),
    'nor does a one-leg FINISH retry that cannot finish');
  const fg = src.indexOf('while (pendingIntegrations.length)', src.indexOf("if (want('final'))"));
  const fgBody = src.slice(fg, src.indexOf('runIntegrateBothLegs', fg));
  ok(fgBody.indexOf('abWindowShort(abLegs(pendingIntegrations[0].inputs)') !== -1
    && fgBody.indexOf('abWindowShort(abLegs(pendingIntegrations[0].inputs)') < fgBody.indexOf('pendingIntegrations.shift()')
    && /abWindowShort\(abLegs\(pendingIntegrations\[0\]\.inputs\)[\s\S]*?\bbreak;/.test(fgBody),
  'the Finalize-gate stops, leaving the rest in pending_integrations, at the first A/B that cannot finish');
  const authorGate = src.indexOf('if (laneWindowShort(`author lane ${h.short_name}:${lang}`)) {');
  ok(authorGate > src.indexOf("const lang = ap.language || 'triton';") && authorGate < src.indexOf('const AUTHOR_TRIES', authorGate)
    && /verdict: 'paused'[\s\S]{0,200}break;/.test(src.slice(authorGate, authorGate + 400)),
  "a head's next author language is not started without room for a lane, and is ledgered as paused");
  const ms = src.indexOf("phase('Milestone');");
  ok(src.indexOf('await reserveBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }', ms) > ms
    && src.indexOf('workflow({ scriptPath: KERNEL_WF_SCRIPT }', ms) === -1,
  'every Milestone lane goes through reserveBoundedWorkflow');
  for (const w of ['fastBoundedWorkflow', 'reserveBoundedWorkflow']) {
    const body = src.slice(src.indexOf(`async function ${w}(`), src.indexOf('\n}', src.indexOf(`async function ${w}(`)));
    ok(body.includes('if (laneWindowShort(') && body.includes('await laneDeadlineArgs(label, ')
      && body.includes('laneReserveCapMs()'), `${w} checks the start window, sets the round deadline and the backstop`);
  }
}

console.log('\n# a required checkpoint keeps its own bound at the reserve boundary');
// The 20261014 GLM run: the accepted-stack checkpoint was dispatched as the reserve began, got the 2min
// floor and was cut off mid-write. The throw that followed discarded a +36% stack with 6h of reserve left.
const kStart = src.indexOf('function checkpointTimeoutFor');
const kEnd = src.indexOf('\n}', kStart) + 2;
ok(kStart > fEnd, 'checkpointTimeoutFor located after atReserveFloor');
if (kStart === -1) process.exit(1);
const COMMIT = 30 * 60000;
const commitAt = (TIME_BUDGET_MS, leftMs, agentMax = AGENT_MAX) => new Function(
  'TIME_BUDGET_MS', 'AGENT_TIMEOUT_MS', 'FINAL_RESERVE_MS', 'remainingMs',
  `${src.slice(aStart, kEnd)}
   return { agentTimeoutFor, checkpointTimeoutFor, enterFinalPhase: () => { FINAL_PHASE_STARTED = true; } };`)(
  TIME_BUDGET_MS, agentMax, RESERVE, () => leftMs);
const kBoundary = commitAt(H, RESERVE);
ok(kBoundary.agentTimeoutFor() === 120000 && kBoundary.checkpointTimeoutFor() === COMMIT,
  'at the boundary an optimization agent still floors at 2min, and a required checkpoint gets 30min');
ok(commitAt(H, 10 * 60000).checkpointTimeoutFor() === COMMIT, 'deep inside the reserve it still gets 30min');
const kMid = commitAt(H, 2 * 3600 * 1000);
ok(kMid.checkpointTimeoutFor() === kMid.agentTimeoutFor() && kMid.agentTimeoutFor() === AGENT_MAX - RESERVE,
  'mid-run it keeps the ordinary bound, which is the longer one there');
const kFinal = commitAt(H, RESERVE);
kFinal.enterFinalPhase();
ok(kFinal.checkpointTimeoutFor() === AGENT_MAX, 'in the final phase it is the plain AGENT_TIMEOUT_MS, as before');
ok(commitAt(H, RESERVE, 10 * 60000).checkpointTimeoutFor() === 10 * 60000,
  'an agent_timeout_ms below 30min still caps it');
ok(new Function('TIME_BUDGET_MS', 'AGENT_TIMEOUT_MS', 'FINAL_RESERVE_MS', 'remainingMs',
  `${src.slice(aStart, kEnd)}\nreturn checkpointTimeoutFor;`)(null, AGENT_MAX, null, () => Infinity)() === AGENT_MAX,
'no time_budget_s => byte-identical');
// Granted where the commit happens, not by a label a caller sets, so the round-2 hole stays closed.
ok((src.match(/\bcheckpointTimeoutFor\b/g) || []).length === 2,
  'checkpointTimeoutFor is defined once and passed at exactly one call site');
ok(/async function requireE2EValidationCheckpoint\(relativePath, intent\) \{\s*const written = await persistE2EValidationCheckpoint\(relativePath, intent, checkpointTimeoutFor\);/.test(src),
  'that call site is requireE2EValidationCheckpoint, so candidate-local diagnostics keep the ordinary cap');

// The whole commit path, executed: requireE2EValidationCheckpoint -> persistE2EValidationCheckpoint ->
// safeAgent -> agentBounded. Time is scaled (1min -> 4ms) so the writer and the guard really race.
const persistSrc = src.slice(src.indexOf('let lastCommittedCheckpoint = null;'),
  src.indexOf('// Expert-skills prompt injection.'));
const dispatchSrc = src.slice(src.indexOf('function agentBounded('), src.indexOf('// --- FlyDSL provisioning gate'));
ok(persistSrc.includes('async function requireE2EValidationCheckpoint(') &&
  dispatchSrc.includes('async function safeAgent('), 'checkpoint writer and dispatch loop located');
const SCALE = 15000;
const commitHarness = ({ leftMs = RESERVE, retry = false, writer }) => {
  const calls = [], logs = [];
  const api = new Function(
    'TIME_BUDGET_MS', 'AGENT_TIMEOUT_MS', 'FINAL_RESERVE_MS', 'remainingMs', 'AGENT_TIMEOUT_RETRY',
    'withProcessSafety', 'ablEffortFor', 'ablEvent', 'ABL', 'tlAgent', 'LLM_TL', 'log', 'agent',
    'EVAL_DIR', 'CHECKPOINT_WRITE_SCHEMA', 'setTimeout',
    `${src.slice(aStart, kEnd)}\n${dispatchSrc}\n${persistSrc}
     return { persistE2EValidationCheckpoint, requireE2EValidationCheckpoint, ABANDONED_RUNS };`)(
    H, AGENT_MAX, RESERVE, () => leftMs, retry,
    (p) => p, () => null, () => {}, () => false, () => null, { nested: [] }, (m) => logs.push(m),
    (prompt, opts) => { calls.push(opts.label); return writer(calls.length); },
    '/eval', {}, (fn, ms) => setTimeout(fn, ms / SCALE));
  return { ...api, calls, logs };
};
const STACK = 'overlay/accepted_stack/e2e_validation.json';
const WRITTEN = { written: true, path: `/eval/${STACK}`, checkpoint_sha256: 'ab12' };
const eightMinWrite = () => new Promise((resolve) => setTimeout(() => resolve(WRITTEN), 8 * 60000 / SCALE));

console.log("\n# an integrator abandoned at the reserve boundary keeps its verdict");
const rib = src.slice(src.indexOf('async function runIntegrateBothLegs('),
  src.indexOf('if (integ && integ.short_name)', src.indexOf('async function runIntegrateBothLegs(')));
ok(/let integ = await adoptAbandonedVerdict\(await safeAgent\(/.test(rib),
  'the first integrate attempt goes through adoptAbandonedVerdict');
ok(/searchClosed\(`\$\{label\}: FINISH retry`\)\) break;/.test(rib) &&
  rib.indexOf('searchClosed(') < rib.indexOf('tries++'),
  'no FINISH retry starts inside the reserve (checked before each one)');
ok(/integ = await adoptAbandonedVerdict\(await safeAgent\([\s\S]*?label: finishLabel, [^}]*\}\), finishLabel\);/.test(rib),
  'a FINISH attempt abandoned at the boundary is adoptable too');

const adSrc = src.slice(src.indexOf('const LATE_VERDICT_RESERVE_FRAC'),
  src.indexOf('// Run ONE integrate A/B and GUARANTEE'));
ok(adSrc.includes('async function adoptAbandonedVerdict('), 'adoptAbandonedVerdict located');
const adoptHarness = ({ floor = true, leftMs = RESERVE, run, integ = null, reserve = RESERVE, finalPhase = 0 }) => {
  const waits = [];
  const runs = new Map(run ? [['warm_start integrate k', run]] : []);
  const fn = new Function('ABANDONED_RUNS', 'atReserveFloor', 'remainingMs', 'FINAL_RESERVE_MS',
    'AGENTX_FINAL_PHASE_MS', 'log', 'LLM_TL', 'setTimeout', `${adSrc}\nreturn adoptAbandonedVerdict;`)(
    runs, () => floor, () => leftMs, reserve, finalPhase, () => {}, { nested: [] },
    (cb, ms) => { waits.push(ms); if (run === NEVER) cb(); return 0; });
  return { adopt: () => fn(integ, 'warm_start integrate k'), waits, runs };
};
const NEVER = new Promise(() => {});
const verdict = { gate: 'accepted', ab_complete: true, e2e_delta_pct: 23.99 };

(async () => {
  const late = adoptHarness({ run: Promise.resolve(verdict) });
  ok(await late.adopt() === verdict, 'the late return of the abandoned attempt is the adopted verdict');
  ok(late.waits.length === 1 && late.waits[0] === RESERVE / 4,
    `the wait is bounded at the first quarter of the reserve (${late.waits[0] / 60000}min of ${RESERVE / 60000}min)`);
  ok(late.runs.size === 0, 'the entry is consumed, so a later A/B with the same label starts clean');

  const silent = adoptHarness({ run: NEVER });
  ok(await silent.adopt() === null, 'an attempt that never returns leaves the A/B incomplete, not stuck');

  const failed = adoptHarness({ run: Promise.reject(new Error('agent crashed')) });
  ok(await failed.adopt() === null, 'an attempt that throws is incomplete too');

  const early = adoptHarness({ floor: false, run: Promise.resolve(verdict) });
  ok(await early.adopt() === null && early.waits.length === 0,
    'mid-search nothing is awaited: FINISH retries proceed exactly as before');

  const spent = adoptHarness({ leftMs: RESERVE * 0.7, run: Promise.resolve(verdict) });
  ok(await spent.adopt() === null && spent.waits.length === 0,
    'past the first quarter of the reserve nothing is awaited — the rest belongs to Finalize/Validate');

  const got = { gate: 'rejected', ab_complete: true };
  const returned = adoptHarness({ integ: got, run: Promise.resolve(verdict) });
  ok(await returned.adopt() === got && returned.waits.length === 0, 'a verdict already in hand is never replaced');

  // A trace replay's wait stops where its final phase's own cost begins: past that, every minute
  // waited is a minute Validate's second canonical leg does not get.
  const AX_PHASE = 235 * 60000;
  const axTight = adoptHarness({ reserve: AX_PHASE, finalPhase: AX_PHASE, leftMs: AX_PHASE, run: Promise.resolve(verdict) });
  ok(await axTight.adopt() === null && axTight.waits.length === 0,
    'a trace replay whose reserve is exactly its final phase awaits nothing: the wait could only cut Validate');
  const axMid = adoptHarness({ reserve: 300 * 60000, finalPhase: AX_PHASE, leftMs: 300 * 60000, run: Promise.resolve(verdict) });
  ok(await axMid.adopt() === verdict && axMid.waits.length === 1 && axMid.waits[0] === 65 * 60000,
    `with a 300min reserve it waits ${axMid.waits[0] / 60000}min, not the quarter (75min): the rest is the final phase's`);
  const axWide = adoptHarness({ reserve: 480 * 60000, finalPhase: AX_PHASE, leftMs: 480 * 60000, run: Promise.resolve(verdict) });
  ok(await axWide.adopt() === verdict && axWide.waits[0] === 120 * 60000,
    'a reserve far wider than the final phase still waits its first quarter (120min of 480min)');

  console.log('\n# the accepted-stack commit at the boundary, end to end (agent_timeout_retry=false, as in the GLM run)');
  const cut = commitHarness({ writer: eightMinWrite });
  ok(await cut.persistE2EValidationCheckpoint(STACK, { phase: 'Overlay' }) === null &&
    cut.logs.some((m) => /exceeded 2min/.test(m)),
  'through the ordinary cap an 8min write at the boundary is cut off at 2min');
  const commit = commitHarness({ writer: eightMinWrite });
  const landed = await commit.requireE2EValidationCheckpoint(STACK, { phase: 'Overlay' }).catch((e) => e);
  ok(landed === WRITTEN && commit.calls.length === 1 && !commit.logs.some((m) => /hung-agent guard/.test(m)),
    'through requireE2EValidationCheckpoint the same write lands, in one dispatch');

  const stuck = commitHarness({ retry: true, writer: () => new Promise(() => {}) });
  let threw = '';
  try { await stuck.requireE2EValidationCheckpoint(STACK, { phase: 'Overlay' }); } catch (e) { threw = String(e); }
  ok(/checkpoint_write_failed/.test(threw) && stuck.logs.some((m) => /exceeded 30min/.test(m)),
    'a writer that never returns still fails the commit, after 30min');
  ok(stuck.calls.length === 1 && stuck.ABANDONED_RUNS.has(`persist-e2e-checkpoint:${STACK}`),
    'even with agent_timeout_retry on, it is not retried beside itself: one dispatch, the attempt kept');

  const flaky = commitHarness({ writer: (n) => Promise.resolve(n === 1 ? null : WRITTEN) });
  ok(await flaky.requireE2EValidationCheckpoint(STACK, { phase: 'Overlay' }).catch((e) => e) === WRITTEN &&
    flaky.calls.length === 2, 'a writer that returns empty inside the reserve is retried once, and the retry commits');

  console.log("\n# a trace-replay lane's round deadline, and the bound on waiting for it");
  const dl = guardsFor({ windowMin: 300 });
  const first = await dl.laneDeadlineArgs('lane k', 'HeadKernel');
  ok(first.deadline_epoch === EPOCH + 160 * 60 && first.no_stop_s === 160 * 60 + 86400,
    'with 300min left the lane starts no round after 160min: its last round and its A/B fit before the boundary');
  ok(dl.reads.length === 1 && /date \+%s/.test(dl.reads[0].prompt) && dl.reads[0].tries === 1
    && dl.reads[0].timeoutMs === 5 * 60000 && dl.reads[0].opts.phase === 'HeadKernel' && dl.reads[0].opts.label === 'clock',
  'the wall clock is one `date +%s` read, one try, bounded at 5min');
  dl.advance(30 * 60000);
  const second = await dl.laneDeadlineArgs('lane j', 'Milestone');
  ok(second.deadline_epoch === first.deadline_epoch && dl.reads.length === 1,
    'a lane started 30min later gets the same absolute stop, from the same read: the boundary does not move');
  const lateLane = guardsFor({ windowMin: 100 });
  ok(Object.keys(await lateLane.laneDeadlineArgs('x', 'HeadKernel')).length === 0 && lateLane.reads.length === 0,
    'no deadline (and no read) once no round fits: the start check refuses such a lane before it gets here');
  const blind = guardsFor({ windowMin: 300, clock: () => null });
  ok(Object.keys(await blind.laneDeadlineArgs('x', 'HeadKernel')).length === 0
    && /wall clock unreadable/.test(blind.logs.join('\n')),
  'an unreadable clock leaves the lane without a deadline (the backstop still bounds it), and says so');
  await blind.laneDeadlineArgs('y', 'HeadKernel');
  ok(blind.reads.length === 2, 'and a failed read is not cached: the next lane tries again');
  const synth = guardsFor({ A: {}, windowMin: 300 });
  ok(Object.keys(await synth.laneDeadlineArgs('x', 'HeadKernel')).length === 0 && synth.reads.length === 0,
    'a synthetic run never reads the clock and passes the lane nothing new');

  const rbwStart = src.indexOf('async function reserveBoundedWorkflow(');
  const rbwEnd = src.indexOf('\n}', src.indexOf('function logLaneAtReserve(')) + 2;
  ok(rbwStart !== -1 && rbwEnd > rbwStart, 'reserveBoundedWorkflow and its race located');
  const laneHarness = ({ short = false, deadline = {}, capMs = null, result }) => {
    const calls = [], logs = [];
    const fn = new Function('laneWindowShort', 'laneDeadlineArgs', 'workflow', 'laneArgs', 'noteKernelKB',
      'laneReserveCapMs', 'setTimeout', 'clearTimeout', 'log',
      `${src.slice(rbwStart, rbwEnd)}\nreturn reserveBoundedWorkflow;`)(
      () => short, async () => deadline, (ref, args) => { calls.push(args); return result; },
      (a) => ({ ...a, lane_defaults: true }), (r) => r, () => capMs,
      (cb) => { if (result === NEVER) cb(); return 1; }, () => {}, (m) => logs.push(String(m)));
    return { run: () => fn({ scriptPath: 'kernel_lane.js' }, { kernel_path: '/task' }, 'k'), calls, logs };
  };
  const LANE = { final_patch: 'p', final_geomean: 1.2 };
  const plain = laneHarness({ result: Promise.resolve(LANE) });
  ok(await plain.run() === LANE && plain.calls[0].lane_defaults && !('deadline_epoch' in plain.calls[0]),
    'guards off: the plain lane call, lane defaults injected and nothing added');
  const timed = laneHarness({ deadline: { deadline_epoch: 7, no_stop_s: 9 }, capMs: 60000, result: Promise.resolve(LANE) });
  ok(await timed.run() === LANE && timed.calls[0].deadline_epoch === 7 && timed.calls[0].no_stop_s === 9
    && timed.calls[0].lane_defaults, 'a trace replay hands the lane its round deadline beside the lane defaults');
  const hungLane = laneHarness({ capMs: 60000, result: NEVER });
  ok(await hungLane.run() === null && /still running at the final reserve/.test(hungLane.logs.join('\n')),
    'a lane still running at the boundary is abandoned (null) so Finalize starts on time');
  const refused = laneHarness({ short: true, result: Promise.resolve(LANE) });
  ok(await refused.run() === null && refused.calls.length === 0, 'a lane refused by the start check never runs');

  console.log(failures ? `\nFAILED (${failures})` : '\nPASS');
  process.exit(failures ? 1 : 0);
})();
