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
ok(/if \(searchClosed\('Profile and Strategize'\)\) \{\s*kernelQueue = \[\];\s*headQueue = \[\];/.test(src),
  'skipping Strategize still leaves both queues arrays (they are declared without a value)');

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
  src.indexOf('async function runIntegrateBothLegs('));
ok(adSrc.includes('async function adoptAbandonedVerdict('), 'adoptAbandonedVerdict located');
const adoptHarness = ({ floor = true, leftMs = RESERVE, run, integ = null }) => {
  const waits = [];
  const runs = new Map(run ? [['warm_start integrate k', run]] : []);
  const fn = new Function('ABANDONED_RUNS', 'atReserveFloor', 'remainingMs', 'FINAL_RESERVE_MS', 'log',
    'LLM_TL', 'setTimeout', `${adSrc}\nreturn adoptAbandonedVerdict;`)(
    runs, () => floor, () => leftMs, RESERVE, () => {}, { nested: [] },
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

  console.log(failures ? `\nFAILED (${failures})` : '\nPASS');
  process.exit(failures ? 1 : 0);
})();
