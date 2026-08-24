#!/usr/bin/env node
// Regression guard for the wall-clock budget -> final reserve contract (no GPU, no model needed).
//
// Hyperloom #1202 round 2: three 20260823 sessions (Qwen3.5-27B-FP8, gpt-oss-120b, DeepSeek-V4-Pro)
// each burned their whole budget and shipped NO final report, despite a 50min reserve being configured
// and never overridden. Two independent defects put them there, and this file pins both:
//
//   1. ELAPSED_MS under-counted. The clock was a self-rearming setTimeout chain, so every rung's
//      scheduler lateness was added to the next rung's start and the error compounded without bound.
//      Under-counting is the dangerous direction: remainingMs() reports time the run does not have.
//   2. The Finalize-gate had no wall-clock bound at all. TIME_DEADLINE_HIT only stops STARTING new
//      head/milestone work; the pendingIntegrations drain loop that runs after it boots a server and
//      benches two legs per iteration, unbounded. That loop is where all three runs were SIGKILLed.
//
// The clock is EXTRACTED from the real workflow source and executed against a virtual scheduler that
// injects lateness -- a reimplementation here would pass while the shipped code drifted, which is the
// failure this test exists to catch. The Finalize-gate is checked structurally (it is 200 lines of
// async agent orchestration and cannot be executed standalone), asserting the deadline check precedes
// the work in the shipped bytes.
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
// The round-2 hole: agentTimeoutFor() used to exempt opts.phase in ['Finalize','Report','Validate'],
// and the Finalize-GATE tags its integrator agents 'Finalize' too -- so measured optimization work
// (server boot + two benches, x AB_FINISH_RETRIES) inherited an exemption written for the final phase
// and ran up to 4 x AGENT_TIMEOUT_MS with no budget bound at all. A self-reported label fails OPEN.
// The exemption is now the FINAL_PHASE_STARTED flag, set once at the phase('Finalize') call site.
// Executed, not grepped: the real function is lifted from source and asked for a verdict.
const aStart = src.indexOf('let FINAL_PHASE_STARTED = false;');
const aEnd = src.indexOf('\n}', src.indexOf('function agentTimeoutFor')) + 2;
ok(aStart !== -1 && aEnd > aStart, 'agentTimeoutFor located');
const mk = (TIME_BUDGET_MS, ELAPSED_MS) => new Function(
  'TIME_BUDGET_MS', 'AGENT_TIMEOUT_MS', 'FINAL_RESERVE_MS', 'remainingMs',
  `${src.slice(aStart, aEnd)}
   return { agentTimeoutFor, enterFinalPhase: () => { FINAL_PHASE_STARTED = true; } };`)(
  TIME_BUDGET_MS, 2 * 3600 * 1000, 50 * 60000, () => Math.max(0, TIME_BUDGET_MS - ELAPSED_MS));

const RESERVE = 50 * 60000, AGENT_MAX = 2 * 3600 * 1000;
// 40min left of a 12h budget: less than the 50min reserve, so a capped agent floors at 2min.
const near = mk(H, H - 40 * 60000);
ok(near.agentTimeoutFor() === 120000,
  'before the final phase, an agent floors at 2min near the deadline — no caller can opt out');
near.enterFinalPhase();
ok(near.agentTimeoutFor() === AGENT_MAX,
  'once phase(\'Finalize\') has actually run, the cap lifts — the final phase owns the reserve');

// Mid-run the cap is the real remaining-minus-reserve, not the flat AGENT_TIMEOUT_MS.
const mid = mk(H, 10 * 3600 * 1000);
ok(mid.agentTimeoutFor() === AGENT_MAX - RESERVE,
  'with 2h left an agent gets 2h minus the 50min reserve, not a flat 2h');

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

console.log(failures ? `\nFAILED (${failures})` : '\nPASS');
process.exit(failures ? 1 : 0);
