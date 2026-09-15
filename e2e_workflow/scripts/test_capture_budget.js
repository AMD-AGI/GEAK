// A retry COUNT never bounded extraction cost. BASELINE_EXTRACT_RETRIES (3 -> 4 invocations) x
// safeAgent's 3 internal tries x AGENT_TIMEOUT_MS (2h with no global budget) = a 24h envelope on one
// head, and the global ELAPSED clock does not even exist unless time_budget_s was passed. The
// 20260907 session spent 13h34m on one gemm head and 2h45m on one moe head, neither of which ever
// reached the optimization lane. A per-head wall-clock budget bounds the thing that actually costs.
//
// e2e_workflow.js is a workflow script (top-level `export const meta`), so it cannot be require()d.
// These tests extract the real definitions from the source and EXECUTE them, so editing the test
// alone cannot satisfy them.
const fs = require('fs');
const path = require('path');
const assert = require('assert');

const SRC = fs.readFileSync(path.join(__dirname, '..', 'e2e_workflow.js'), 'utf8');
let failures = 0;
function test(name, fn) {
  try { fn(); console.log(`  ok: ${name}`); }
  catch (err) { failures++; console.log(`  FAIL: ${name}\n    ${err && err.message}`); }
}
function extract(startsWith, endsWith) {
  const start = SRC.indexOf(startsWith);
  assert.ok(start >= 0, `source no longer contains \`${startsWith}\``);
  const end = SRC.indexOf(endsWith, start);
  assert.ok(end >= 0, `no \`${endsWith}\` after \`${startsWith}\``);
  return SRC.slice(start, end + endsWith.length);
}
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// eslint-disable-next-line no-new-func
const stopwatch = new Function(`${extract('function stopwatch(budgetMs) {', '\n}')}\nreturn stopwatch;`)();

console.log('the stopwatch reads elapsed time without Date.now()');

test('a fresh stopwatch has its whole budget left', () => {
  const sw = stopwatch(60000);
  assert.strictEqual(sw.remainingMs(), 60000);
  assert.strictEqual(sw.spentMin(), 0);
  sw.stop();
});

test('it does not use Date.now / new Date / Math.random', () => {
  // All three throw in Workflow scripts because they would break resume. A stopwatch that reached for
  // one would not fail here -- it would fail at runtime, mid-session, on the first head.
  const body = extract('function stopwatch(budgetMs) {', '\n}');
  assert.ok(!/Date\.now|new Date|Math\.random/.test(body), 'stopwatch reads a forbidden clock');
});

test('it does not advance before a rung is due', async () => {
  const sw = stopwatch(120000);          // step = max(30s, 120s/240) = 30s
  await wait(70);
  assert.strictEqual(sw.remainingMs(), 120000, 'advanced before any rung could fire');
  sw.stop();
});

// The rungs floor at 30s, so real time cannot drive them in a unit test. Build the stopwatch over an
// INJECTED setTimeout instead and fire the rungs by hand. Without this, a stopwatch whose `elapsed`
// never updates passes every other test here -- and silently restores the unbounded behaviour.
function fakeClock() {
  const rungs = [];
  const fake = (fn, ms) => { rungs.push({ fn, ms, cleared: false }); return { unref() { return this; } }; };
  // eslint-disable-next-line no-new-func
  const build = new Function('setTimeout', 'clearTimeout',
    `${extract('function stopwatch(budgetMs) {', '\n}')}\nreturn stopwatch;`);
  const make = build(fake, (h) => { const r = rungs.find((x) => x.h === h); if (r) r.cleared = true; });
  return { rungs, make };
}

test('elapsed advances as rungs fire, and the budget runs out', () => {
  const { rungs, make } = fakeClock();
  const sw = make(120000);                                  // step 30s -> rungs at 30/60/90/120/150s
  assert.deepStrictEqual(rungs.map((r) => r.ms), [30000, 60000, 90000, 120000, 150000]);
  rungs[0].fn();
  assert.strictEqual(sw.remainingMs(), 90000, 'a fired rung did not advance the clock');
  assert.strictEqual(sw.spentMin(), 1);
  rungs[3].fn();
  assert.strictEqual(sw.remainingMs(), 0, 'the budget never runs out');
  rungs[4].fn();
  assert.strictEqual(sw.remainingMs(), 0, 'remaining went negative past the budget');
});

test('a late rung cannot rewind the clock', () => {
  // Rungs are armed at ABSOLUTE offsets and taken as a max precisely so scheduler lateness reorders
  // them harmlessly. Re-running an earlier rung after a later one must not hand back time already spent.
  const { rungs, make } = fakeClock();
  const sw = make(120000);
  rungs[2].fn();                                            // 90s
  assert.strictEqual(sw.remainingMs(), 30000);
  rungs[0].fn();                                            // 30s, arriving late
  assert.strictEqual(sw.remainingMs(), 30000, 'a late rung rewound the clock and bought back budget');
});

test('one rung is armed PAST the budget so it can reach zero', () => {
  const { rungs, make } = fakeClock();
  make(120000);
  assert.ok(rungs[rungs.length - 1].ms > 120000,
    'no rung past the budget: the last tick lands short and remaining never reaches 0');
});

test('a budget of zero or less is unenforceable and reports Infinity', () => {
  // Never abort work you cannot time: an unarmed stopwatch must leave callers exactly as they were.
  for (const bad of [0, -1, NaN]) {
    const sw = stopwatch(bad);
    assert.strictEqual(sw.remainingMs(), Infinity, `budget ${bad} armed a guard anyway`);
    sw.stop();
  }
});

test('stop() is idempotent and safe after expiry', () => {
  const sw = stopwatch(60000);
  sw.stop(); sw.stop();
  assert.doesNotThrow(() => sw.remainingMs());
});

test('rung count stays bounded for an absurd budget', () => {
  // A 24h budget at a fixed 30s tick would arm 2880 timers per head. The step widens instead.
  const body = extract('const step = Math.max(', ';');
  assert.ok(/Math\.max\(30000/.test(body) && /\/ 240\)/.test(body),
    'step no longer both floors granularity and caps the rung count');
});

console.log('the per-head budget is wired into the funnels');

test('agentBounded caps an attempt at what the caller has left', () => {
  const block = extract('function agentBounded(rawPrompt, opts) {', 'const timeoutMs =');
  assert.ok(/opts\.timeoutCapMs/.test(block), 'no per-call timeout cap');
  assert.ok(/Math\.min\(agentTimeoutFor\(\)/.test(SRC.slice(SRC.indexOf(block))),
    'the cap does not tighten the global hung-guard');
  // Absent the opt, every existing call site must be byte-identical: min(global, Infinity) = global.
  assert.ok(/: Infinity;/.test(block), 'a call site without the opt no longer defaults to Infinity');
});

test('safeAgent stops retrying when the caller says the budget is spent', () => {
  const block = extract('async function safeAgent(prompt, opts, tries = 3) {', 'try {');
  assert.ok(/opts\.abortIf/.test(block), 'safeAgent internal retries are still invisible to callers');
  assert.ok(/return null;/.test(block), 'abort does not short-circuit the retry loop');
  assert.ok(/typeof opts\.abortIf === 'function'/.test(block),
    'a caller that passes no abortIf could be broken by a truthy non-function');
});

test('extractWithBaseline checks the budget BEFORE paying for an attempt', () => {
  const loop = extract('while (smokeOk(ext) && !complete(ext) && tries < BASELINE_EXTRACT_RETRIES) {',
    'tries++;');
  assert.ok(/if \(spent\(\)\)/.test(loop), 'the budget is not checked at the top of the retry loop');
  assert.ok(/break;/.test(loop), 'a spent budget does not leave the loop');
  assert.ok(loop.indexOf('if (spent())') < loop.indexOf('tries++'),
    'the budget is checked after the attempt was already counted');
});

test('both agent call sites in the extraction are budget-bounded', () => {
  const fn = extract('async function extractWithBaseline(role, phase, intro, inputs, opts) {',
    '\n}\n');
  const bounded = (fn.match(/bounded\(\)\)/g) || []).length;
  assert.strictEqual(bounded, 2,
    `expected the first attempt AND the retry to be bounded, found ${bounded}`);
  assert.ok(!/\n      opts\);/.test(fn), 'a call site still passes raw opts and escapes the budget');
});

test('the stopwatch is always stopped', () => {
  const fn = extract('async function extractWithBaseline(role, phase, intro, inputs, opts) {',
    '\n}\n');
  assert.ok(/} finally {/.test(fn) && /budget\.stop\(\);/.test(fn),
    'an early return leaks 240 timers per head');
});

test('a budget cut is reported as a budget cut, not as exhausted retries', () => {
  // The two are not the same failure: one is raised with a knob, the other is not. A head cut at the
  // budget that reads as "unextractable" is a head nobody retries with more time.
  const fn = extract('async function extractWithBaseline(role, phase, intro, inputs, opts) {',
    '\n}\n');
  assert.ok(/capture_budget_spent/.test(fn), 'the verdict does not distinguish a budget cut');
  assert.ok(/args\.capture_budget_s/.test(fn), 'the message does not name the knob that raises it');
  // Including the case where nothing ever smoke-passed, which used to return unmarked.
  assert.ok(/budgetSpent && !smokeOk\(ext\)/.test(fn),
    'a never-smoke-passing head cut by the budget is reported as an ordinary failure');
});

console.log('the default is calibrated to measured runs, not to a round number');

test('the default budget keeps the only observed success and cuts both failures', () => {
  const line = extract('const CAPTURE_BUDGET_MS =', ';');
  const seconds = Number((line.match(/:\s*(\d+),\s*10\)/) || [])[1]);
  assert.ok(Number.isFinite(seconds), `could not read the default from: ${line}`);
  const minutes = seconds / 60;
  // 20260907: moe1+moe2 succeeded in 63min; moe1_silu failed at 165min; gemm failed at 816min.
  assert.ok(minutes > 63, `${minutes}min would have killed the one extraction that ever succeeded (63min)`);
  assert.ok(minutes < 165, `${minutes}min does not cut moe1_silu (165min, never reached the lane)`);
});

test('the minimum attempt is at least one server boot', () => {
  const line = extract('const CAPTURE_MIN_ATTEMPT_MS =', ';');
  const seconds = Number((line.match(/:\s*(\d+),\s*10\)/) || [])[1]);
  assert.ok(Number.isFinite(seconds), `could not read the default from: ${line}`);
  assert.ok(seconds >= 900, `${seconds / 60}min is below a measured server boot (15-30min)`);
});

test('both budgets are overridable', () => {
  assert.ok(/A\.capture_budget_s/.test(SRC), 'capture_budget_s is not a knob');
  assert.ok(/A\.capture_min_attempt_s/.test(SRC), 'capture_min_attempt_s is not a knob');
});

(async () => {
  await wait(120);
  console.log(failures ? `\nFAIL (${failures})` : '\nPASS');
  process.exit(failures ? 1 : 0);
})();
