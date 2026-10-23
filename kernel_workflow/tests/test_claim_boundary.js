#!/usr/bin/env node
// Behavioral test for the claim boundary: the three decisions that determine whether a measurement
// that happened on hardware reaches the scoring harness.
//
// WHY THIS EXISTS. Once a real, reproducible, bit-identical +20.6% at one route guard was
// measured three separate times and scored zero times — once because the engineer's declared patch
// did not exist (the effect lived only in bench CLI flags), once because the engineer wrote a
// correct claim to disk and kept measuring past the round's deadline, so the round closed with no
// StructuredOutput. The instrument worked every time. The loss was entirely at this boundary. The
// repairs went in without a test, which means nothing would have gone red if either had regressed
// to silence — and silence is exactly the failure mode. Hence this file.
//
// WHY IT IS NOT A SOURCE-GREP TEST. Same house rule as test_task_graph_gate.js: it LIFTS the shipped
// predicates out of kernel_mega.js between the `<<REPLAY:claim_boundary>>` markers and runs
// them. If the shipped code changes, this test runs the changed code.
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_mega.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

// --- lift ------------------------------------------------------------------------------------
const m = src.match(
  /\/\/ <<REPLAY:claim_boundary>>[\s\S]*?\n([\s\S]*?)\n\s*\/\/ <<\/REPLAY:claim_boundary>>/);
if (!m) {
  console.error('FAILED: the <<REPLAY:claim_boundary>> markers are missing from kernel_mega.js.');
  console.error('Without them this test cannot reach the shipped predicates and would silently start');
  console.error('testing nothing. Restore the markers around claimBoundary().');
  process.exit(1);
}
// eslint-disable-next-line no-new-func
const claimBoundary = new Function(`${m[1]}\nreturn claimBoundary;`)();
ok(typeof claimBoundary === 'function', 'claimBoundary lifted from kernel_mega.js and is callable');

// Purity: a region that closed over run state would throw here rather than in production.
ok((() => { try { claimBoundary(() => 0); return true; } catch { return false; } })(),
   'the lifted region is pure (no closure over run state)');

// The real primSpeedup consults HAS_WORKLOAD, which is run state, so the shipped code injects it.
// Here we inject the simplest thing that reproduces its contract: a missing claim scores 0.
const speedupOf = (o) => (o && Number.isFinite(o.speedup_geomean) ? o.speedup_geomean : 0);
const CB = claimBoundary(speedupOf);
const MCB = claimBoundary(speedupOf, true);

// --- 1. recovery fires on every way an engineer can fail to hand a claim back -------------------
console.log('\n# when to go looking on disk');
for (const [what, eng] of [
  ['null (the agent died or timed out)', null],
  ['undefined', undefined],
  ['a claim object with no per_case at all', { speedup_geomean: 1.0594, status: 'ok' }],
  ['per_case present but empty', { per_case: [], speedup_geomean: 1.0594 }],
  ['per_case is not an array', { per_case: 'four guards, all resolved' }],
]) {
  ok(CB.needsRecovery(eng) === true, `${what} -> recover`);
}
// The third case is the one that actually happened and the one an eyeball test misses: a WELL-FORMED
// claim with a headline speedup and no per-case evidence. It reads like a result. It is not one.

console.log('\n# when not to');
ok(CB.needsRecovery({ per_case: [{ guard: '512_skew', pct: 20.6 }] }) === false,
   'a claim with at least one case is left alone -- recovery must never re-enter a good round');
ok(MCB.needsRecovery({
  per_case: [{ guard: '8192_uniform', pct: 3.2 }], claim_complete: false,
}) === true,
'Mega recovers a partial claim even when it already contains per-case rows');
ok(MCB.needsRecovery({
  per_case: [{ guard: '8192_uniform', pct: 3.2 }], claim_complete: true,
}) === false,
'Mega accepts the same claim only after its evidence manifest is complete');

// --- 2. recovery cannot manufacture a claim ------------------------------------------------------
// The recovery agent is told to return `per_case: []` when there is nothing on disk. That answer has
// to be REJECTED, or "nothing was there" becomes indistinguishable from "something was recovered"
// and the phase reports a recovery that recovered nothing.
console.log('\n# what recovery is allowed to accept');
ok(CB.recovered({ per_case: [{ guard: '512_skew', pct: 20.6 }] }) === true,
   'a claim read off disk with cases is accepted');
for (const [what, d] of [
  ['the agent returned nothing', null],
  ['the agent reported an empty disk (per_case: [])', { per_case: [], notes: 'no claim on disk' }],
  ['a headline with no cases', { speedup_geomean: 1.21 }],
]) {
  ok(CB.recovered(d) === false, `${what} -> not a recovery`);
}
// Symmetry is the property that matters: recovery must not accept anything the caller would have
// rejected, or the loop oscillates -- accept, re-check, recover again.
for (const c of [null, undefined, {}, { per_case: [] }, { per_case: [{ g: 1 }] }]) {
  ok(CB.needsRecovery(c) === !CB.recovered(c),
     `usability is one definition, not two (${JSON.stringify(c)})`);
}

// --- 3. an unbacked claim is a reporting failure, and only then --------------------------------
// This is the discrimination the label is worth having for. `apply_failed` under a WINNING claim is
// a measurement that has been lost. `apply_failed` under a null claim is an ordinary dead direction.
// Calling both a reporting failure would train readers to ignore the word.
console.log('\n# unbacked claims');
const R = (over) => Object.assign({
  eng: { per_case: [{ guard: '512_skew', pct: 20.6 }], speedup_geomean: 1.206 },
  ver: { status: 'apply_failed' },
}, over || {});

ok(CB.unbacked(R()) === true,
   'a winning claim whose patch does not apply IS unbacked -- the wave-8 case');
ok(CB.unbacked(R({ eng: { per_case: [{}], speedup_geomean: 0.98 } })) === false,
   'a LOSING claim whose patch does not apply is just a dead direction, not a reporting failure');
ok(CB.unbacked(R({ eng: { per_case: [{}], speedup_geomean: 1.0 } })) === false,
   'exactly 1.000x is not a win -- the threshold is strict, as it is everywhere else in the loop');
ok(CB.unbacked(R({ ver: { status: 'ok' } })) === false,
   'a patch that applied is not unbacked however good the claim');
ok(CB.unbacked(R({ ver: { status: 'regression' } })) === false,
   'a patch that applied and lost is not unbacked either');
ok(CB.unbacked(R({ ver: null })) === false,
   'no verify verdict at all is not an unbacked claim -- it is a different, earlier failure');
ok(CB.unbacked(R({ eng: null })) === false,
   'no engineer claim means nothing was claimed, so nothing is unbacked');
ok(CB.unbacked(null) === false, 'a missing result does not crash the sweep');

// --- 4. wiring ----------------------------------------------------------------------------------
// The predicates can be perfect and never called. These are the connections to the phase.
console.log('\n# wiring');
ok(/const CLAIM = claimBoundary\(primSpeedup, MODE === 'mega'\)/.test(src),
   'the predicates are instantiated with the real primSpeedup, not a local copy');
ok(/if \(CLAIM\.needsRecovery\(eng\)\)/.test(src), 'Optimize consults needsRecovery');
ok(/if \(CLAIM\.recovered\(onDisk\)\)/.test(src), 'Optimize gates the recovered claim through recovered()');
ok(/if \(!CLAIM\.unbacked\(r\)\) continue/.test(src), 'the post-verify sweep consults unbacked');

// Recovery must not be allowed to become a second measurement. That instruction lives in the prompt,
// and the prompt is the only thing standing between "read the bytes" and "take a lease and re-run",
// so its load-bearing clauses are asserted here.
const recoverPrompt = (src.match(/did not return a usable claim\.[\s\S]{0,1400}?\}\),/) || [''])[0];
ok(/RECOVER ONLY/.test(recoverPrompt), 'the recovery agent is told RECOVER ONLY');
ok(/do NOT take a lease/i.test(recoverPrompt), 'the recovery agent is forbidden a GPU lease');
ok(/do NOT re-measure/i.test(recoverPrompt), 'the recovery agent is forbidden a fresh measurement');
ok(/UNRESOLVED/.test(recoverPrompt),
   'recovery must carry UNRESOLVED guards through -- laundering them into wins is the obvious abuse');
ok(/per_case: \[\]/.test(recoverPrompt),
   'the agent is given a way to say "nothing on disk" that is not a fabricated claim');
ok(/mtimes\/claim_complete/.test(recoverPrompt) && /declaration is stale/.test(recoverPrompt),
   'a newer completed aggregate supersedes a stale worker_result instead of being ignored');

// And the unbacked log line must say what to DO, since its whole purpose is to stop the direction
// being filed as tried-and-failed.
const unbackedLog = (src.match(/CLAIM NOT BACKED BY A PATCH[\s\S]{0,700}?\);/) || [''])[0];
// (the phrase is split across a string concatenation in the source, so match the halves)
ok(/REPORTING/.test(unbackedLog) && /not a null result/.test(unbackedLog),
   'the log names it a reporting failure, explicitly not a null result');
ok(/re-dispatch it with the patch written first/.test(unbackedLog),
   'the log says what the next round must do differently');

console.log(failures
  ? `\nFAILED: ${failures} assertion(s).`
  : '\nPASS: the claim boundary discriminates and is wired end-to-end.');
process.exit(failures ? 1 : 0);
