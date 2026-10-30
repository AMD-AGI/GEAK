#!/usr/bin/env node
// Guards for the three checks added after a run reported 1.000x on a tree that was missing an
// already-measured +4.71% win, with nothing in the workflow able to tell "found nothing" apart from
// "cannot see anything":
//
//   1. POSITIVE CONTROL  — a known-effect change run during Benchmark, before any budget is spent.
//                          Calibrates the DETECTION floor (the null arm only calibrates noise).
//   2. PRIOR ART         — directions already implemented in/beside the tree, surfaced at Analyze.
//   3. PROVENANCE        — reps + null_arm_pct on every claimed win, marking thin ones PROVISIONAL.
//
// These are prose-plus-schema changes across four files, which is exactly the kind of thing that
// gets reflowed away. The assertions below are the ratchet.

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

let failures = 0;
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}

const wf = read('kernel_mega.js');
const bench = read('roles/mega/benchmark_engineer.md');
const lead = read('roles/mega/tech_lead.md');
const verify = read('roles/mega/verify_engineer.md');

// --- 1. positive control ---------------------------------------------------
console.log('\n# positive control');
ok(/const POSITIVE_CONTROL = \(A\.positive_control/.test(wf),
   'kernel_mega.js reads args.positive_control');
// Was `positive_control: { type: 'object', additionalProperties: true }` — an opaque bag. Every
// field the gate reads off it defaults to a PASSING reading when absent (`pc.ran !== false` is true
// for a missing `ran`; a missing `switch_present` makes the pre-flight inert), so a renamed or
// dropped field validated and was then interpreted as a control that ran and passed.
ok(/positive_control: obj\(\{/.test(wf),
   'BENCH_SCHEMA declares positive_control\'s shape rather than accepting any object');
{
  const seg = wf.slice(wf.indexOf('positive_control: obj({'), wf.indexOf('positive_control: obj({') + 1400);
  for (const f of ['ran', 'switch_present', 'measured_pct', 'null_arm_pct',
                   'control_pairs_pct', 'null_pairs_pct'])
    ok(new RegExp(`\\b${f}:`).test(seg), `  and declares ${f}, which the gate reads`);
  ok(/\}, \['ran', 'claim_complete', 'switch_present', 'measured_pct'\]\)/.test(seg),
     '  and requires completion plus the three whose absence would otherwise read as a passing control');
}
// The gate must sit between Benchmark and the optimize loop. If it ran later, the budget it exists
// to protect would already be spent.
const iBench = wf.indexOf("phase('Benchmark')");
const iGate = wf.indexOf('Positive control FAILED');
const iLoop = wf.indexOf('while (dispatched < BUDGET');
ok(iBench > 0 && iGate > iBench && iLoop > iGate,
   'the gate fires after Benchmark and before the optimization loop');
ok(/if \(PC_ABORT\) throw new Error\(why\)/.test(wf),
   'a failed control aborts by default rather than warning');
ok(/abort_on_fail !== false/.test(wf),
   'abort is opt-out, not opt-in');
ok(/unfalsifiable/.test(wf),
   'the code states WHY a missing control makes 1.000x unfalsifiable');

// The benchmark engineer must be told to interleave the control like a real candidate, with a null
// arm — a control measured more loosely than the thing it certifies proves nothing.
ok(/POSITIVE CONTROL/.test(bench), 'benchmark_engineer has a positive-control step');
ok(/≥5 pairs/.test(bench), 'the control requires >=5 interleaved pairs');
ok(/null arm/i.test(bench), 'the control requires a null arm alongside it');
ok(/do not quietly retry until it passes/i.test(bench),
   'the engineer is barred from retrying the control until it passes');
ok(/positive = faster/.test(bench), 'the sign convention for measured_pct is pinned down');

// The worst null pair is a sample MAXIMUM: it can only grow with pair count, so the same real effect
// scores worse at n=10 than at n=5. An engineer who spots that has a standing incentive to sample
// shallowly and call the flattering number a floor -- and one run reached exactly that conclusion
// ("operate at 6-8 pairs, never 40"), which on a bimodal guard hides the tail that decides the case.
// The role has to name the incentive and route around it, or the rule teaches undersampling.
ok(/sample maximum/.test(bench),
   'the role states that the worst null pair grows with pair count');
ok(/that is not a reason to stop\s*\n?deepening/.test(bench),
   'and refuses the undersampling that follows from noticing it');
ok(/Deep sample, judge by separation/.test(bench),
   'the way out is named: a statistic that deepening does not penalise');
ok(/never 40/.test(bench),
   'the tempting wrong conclusion is quoted, not just contradicted in the abstract');
// A residual that is merely large is not evidence of a host cost when the timers are independently
// rank-reduced -- max-of-sums minus sum-of-maxes is signed, and was measured NEGATIVE on this box.
ok(/max-of-sums minus sum-of-maxes|max-of-sums\*? minus \*?sum-of-maxes/.test(bench),
   'residual attribution carries the arithmetic that makes a large residual uninformative');
ok(/moves \*between paired arms\* while both kernel timers hold still/.test(bench),
   'and states what a residual has to do before it counts as evidence');

// --- 2. prior art ----------------------------------------------------------
console.log('\n# prior art');
ok(/prior_art: \{ type: 'array'/.test(wf), 'ANALYZE_SCHEMA carries prior_art');
ok(/PRIOR ART NOT IN BASELINE/.test(wf),
   'absent-from-baseline prior art gets its own distinct, louder log line');
ok(/pointed at the wrong target/.test(wf),
   'the log names the real failure mode: optimizing the wrong tree');
ok(/Measure it, do not re-derive it/.test(wf),
   'present-but-off prior art is routed to measurement, not engineering');
ok(/Search for PRIOR ART before you plan anything/i.test(lead),
   'tech_lead is told to search for prior art at analyze time');
// NB: this file wraps at ~100 cols; match across a hard wrap. A plain-space regex here has broken
// silently before when a paragraph was re-flowed.
ok(/~1000 lines of already-written, already-measured/.test(lead),
   'the tech_lead rule carries the incident that motivated it');
// Wording tightened after the key was omitted outright and the distinction had to be enforced in
// code as well as prose — see tests/test_prior_art_provenance.js.
ok(/`\[\]` is a real answer that is not the same as omitting it/.test(lead),
   'the [] vs omitted distinction is spelled out so "no prior art" is a claim, not a silence');

// --- 3. result provenance --------------------------------------------------
console.log('\n# result provenance');
ok(/reps: \{ type: 'number' \}, null_arm_pct: \{ type: 'number' \}/.test(wf),
   'VERIFY_SCHEMA carries reps and null_arm_pct');
ok(/r\.provisional = reasons\.join/.test(wf), 'thin results are marked provisional');
ok(/PROVISIONAL \$\{r\.d\.id\}/.test(wf), 'provisional results are logged per direction');
// The point of PROVISIONAL is that it does NOT delete the result. Dropping under-repped wins would
// re-create the exact failure this was built to catch (a real win discarded on weak evidence).
ok(/does NOT reject the result/.test(wf),
   'the code states that provisional marking is not rejection');
ok(/claim \$\{claimPct\.toFixed\(2\)\}% <= null arm/.test(wf),
   'a claim at or below the null arm is flagged');
ok(/Under-repped wins are \*\*not discarded\*\*/.test(verify),
   'verify_engineer is told thin results survive as provisional');
ok(/reps` is the count of interleaved\s+A,B pairs behind the median/.test(verify),
   'reps is defined as pairs, not process launches');

// --- 4. it has to reach the report ----------------------------------------
console.log('\n# the report must carry it');
ok(/Measurement confidence/.test(lead), 'the report has a measurement-confidence section');
ok(/1\.000x with no positive control is not a\s+finding/.test(lead),
   'the report is required to disown an uncalibrated zero');
ok(/required even \(especially\) when the final speedup is\s+1\.000x/.test(lead),
   'the section is mandatory precisely in the case that produced this fix');

// --- 5. the gate's ARITHMETIC, exercised on the real source ----------------
// Everything above only proves the text is present. This lifts the actual in-band expression out of
// kernel_mega.js and runs it, so a subtly wrong comparison (>= vs >, NaN slipping through as
// "pass") fails here instead of in production. Extracting rather than re-typing is deliberate: a
// hand-copied duplicate of the logic drifts and then tests itself.
console.log('\n# gate arithmetic (extracted from source, not re-implemented)');
const gateSrc = wf.match(
  /(const mLo = Math\.min[\s\S]*?)\n  const ok = ran && !tooSmall && !absurd && \(!overshoot \|\| nullQuiet\);/,
);
ok(!!gateSrc, 'the decision expression can be located in kernel_mega.js');
let decide = null;
if (gateSrc) {
  decide = new Function('pc', 'POSITIVE_CONTROL', `
    const lo = Number(POSITIVE_CONTROL.expected_pct_lo);
    const hi = Number(POSITIVE_CONTROL.expected_pct_hi);
    const got = Number(pc.measured_pct);
    ${gateSrc[1]}
    const ok = ran && !tooSmall && !absurd && (!overshoot || nullQuiet);
    return { ok, overshoot, tooSmall, absurd };
  `);
  const band = { expected_pct_lo: 3.55, expected_pct_hi: 4.93 };  // the real +4.71% guard
  const quiet = 0.38;   // the null arm actually observed alongside the +5.13% reading
  const cases = [
    [{ ran: true, measured_pct: 4.71, null_arm_pct: quiet }, true, 'the known +4.71% passes'],
    [{ ran: true, measured_pct: 3.55, null_arm_pct: quiet }, true, 'the lower bound is inclusive'],
    [{ ran: true, measured_pct: 4.93, null_arm_pct: quiet }, true, 'the upper bound is inclusive'],
    // The incident: a real reproduction 0.20pp over the ceiling with a quiet null arm. The whole
    // point of the asymmetry is that this must NOT kill the run.
    [{ ran: true, measured_pct: 5.13, null_arm_pct: quiet }, true,
     'a 0.20pp overshoot with a quiet null arm PASSES (the incident that motivated the split)'],
    [{ ran: true, measured_pct: 5.13, null_arm_pct: 2.9 }, false,
     'the same overshoot with a LOUD null arm fails — that excess is drift, not sensitivity'],
    [{ ran: true, measured_pct: 5.13 }, false,
     'an overshoot with an UNREPORTED null arm fails: nothing rules out drift'],
    [{ ran: true, measured_pct: 0.4, null_arm_pct: quiet }, false,
     'a harness that sees almost nothing still fails — the lower bound is untouched'],
    [{ ran: true, measured_pct: -4.71, null_arm_pct: quiet }, false,
     'the right magnitude with the WRONG SIGN fails'],
    [{ ran: true, measured_pct: 40.0, null_arm_pct: quiet }, false,
     'an absurd delta still fails: past 2x the ceiling it is not this change being measured'],
    [{ ran: true, measured_pct: 9.86, null_arm_pct: quiet }, true,
     'exactly at the plausibility ceiling is still a pass (absurd is strictly above)'],
    [{ ran: false }, false, 'a control that did not run fails'],
    [{ ran: true }, false, 'a missing measurement is not silently a pass'],
    [{ ran: true, measured_pct: 'n/a' }, false, 'an unparseable measurement is not a pass'],
    [{ ran: true, measured_pct: NaN }, false, 'NaN is not a pass'],
    [{}, false, 'an empty result object is not a pass'],
  ];
  for (const [pc, want, msg] of cases) ok(decide(pc, band).ok === want, msg);
  // An explicit ceiling overrides the 2*hi default.
  ok(decide({ ran: true, measured_pct: 6.0, null_arm_pct: quiet },
            { ...band, implausible_pct: 5.5 }).ok === false,
     'an explicit implausible_pct overrides the 2x default');
  // A pass that overshot must be DISTINGUISHABLE from a clean pass, or the caveat cannot travel.
  ok(decide({ ran: true, measured_pct: 5.13, null_arm_pct: quiet }, band).overshoot === true,
     'an overshooting pass is flagged as such, so the report can carry the scale caveat');
  ok(decide({ ran: true, measured_pct: 4.71, null_arm_pct: quiet }, band).overshoot === false,
     'a clean pass is not flagged as an overshoot');
}

// A per-case noise floor that only lives in an insight is decoration. The aggregate is where an
// unreadable number turns into a claim, so the rule has to bind at aggregation time.
console.log('\n# unresolved cases must not be aggregated at face value');
ok(/CONTRIBUTES 1\.000 TO THE HEADLINE/.test(lead),
   'a case inside its own same-arm spread folds in as 1.000, not as its point estimate');
ok(/UNRESOLVED \(delta, n, wins\/n, spread\)/.test(lead),
   'the table prints the evidence that made the case unresolved, not just the verdict');
ok(/same-arm spread/.test(lead) && /sign test/.test(lead),
   'both readability tests are named: magnitude vs spread AND the paired sign test');
// Without the incident the rule reads as pedantry and gets skipped under budget pressure.
ok(/1\.01769x/.test(lead) && /-4\.14%/.test(lead) && /1\.0021x/.test(lead),
   'the 1.56pp inflation incident keeps its three numbers in the role file');

// A measurement that exists on disk but not in the return value is still a measurement. The workflow
// cannot read files, so it has to ASK — and the recovery must be recovery, not a silent re-run.
console.log('\n# an unreturned baseline is recovered from disk, not thrown away');
ok(/label: 'benchmark_recover'/.test(wf), 'a recovery agent runs before the phase aborts');
ok(/no baseline recorded, and none recoverable from EVAL_DIR/.test(wf),
   'the abort still exists — recovery narrows the failure, it does not remove it');
ok(/Do NOT run any GPU command and do/.test(wf) && /a fresh measurement here is a/.test(wf),
   'the recovery agent is forbidden from re-measuring, which would hide the cost of the failure');
ok(/claim_complete:true IS the positive/.test(wf),
   'a completed positive control on disk is reused rather than re-run — it is the phase\'s costliest step');
// The role side: persist first, then do the expensive interruptible things.
ok(/PERSIST THE BASELINE THE MOMENT IT EXISTS/.test(bench),
   'the engineer writes the baseline before the control and correctness, not after');
ok(/READ THEM AND REUSE THEM/.test(bench),
   'an interrupted attempt\'s artifacts are inputs to the retry, not litter');


// Same failure class as the unreturned baseline, one level down: a direction that measured and then
// died before writing its result is recorded as a zero. Both happened in the same wave.
console.log('\n# a killed engineer must leave a partial result, not a silence');
const eng = read('roles/mega/engineer.md');
ok(/REWRITE IT AFTER EVERY UNIT OF EVIDENCE/.test(eng),
   'the result file is refreshed per rung/arm, not written once at the end');
ok(/before your first long-running\s*\n?command/.test(eng),
   'the first write happens before the first thing that can kill the agent');
ok(/15-rung hardware bisection/.test(eng) && /scored \*\*zero\*\*/.test(eng),
   'the incident keeps its cost: a completed bisection recorded as no result');
ok(/is a \*\*zero\*\*, and the difference is entirely in when you wrote the file/.test(eng),
   'the rule states the consequence in the ledger, not just the practice');

// The caveat has to reach the report, or "passed with a known upward bias" degrades to "passed".
console.log('\n# an overshooting pass carries its caveat forward');
ok(/let PC_OVERSHOOT = ''/.test(wf), 'the overshoot caveat is captured in a variable');
ok(/POSITIVE CONTROL OVERSHOOT/.test(wf), 'it is logged distinctly from a clean pass');
ok(/POSITIVE_CONTROL_OVERSHOOT: PC_OVERSHOOT/.test(wf),
   'it is threaded into the report phase');
ok(/reads \\nHIGH by roughly|reads `? ?HIGH by roughly|HIGH by roughly/.test(wf),
   'the caveat quantifies the bias rather than just noting it');


// A marker proves the host path ran. Under a JIT with a disk cache, two arms can print two markers
// and execute one binary — which reads 1.000 with every existing gate satisfied. This closed an
// entire optimization axis on a void experiment before it was caught by a retroactive audit.
console.log('\n# arms must be proved to be DIFFERENT CODE, not just different markers');
ok(/IT DOES NOT PROVE THE ARMS COMPILED TO DIFFERENT CODE/.test(verify),
   'the marker/binary distinction is stated where activation is judged');
ok(/artifact-distinctness proof/.test(verify),
   'a positive obligation is named, not just a warning');
ok(/cache keys, IR\/ISA hashes, or binary paths/.test(verify),
   'the acceptable evidence forms are concrete enough to run');
ok(/name-normalised/.test(verify),
   'a symbol rename cannot fake a distinct artifact');
ok(/Same hash across arms ⇒ `status:"inactive"`/.test(verify),
   'identical artifacts are a VOID experiment, never a null result');
ok(/machine-global/.test(verify),
   'the separate-checkout escape hatch is closed: a global cache root defeats it');
ok(/Once a retroactive audit/.test(verify) && /had to be reopened/.test(verify),
   'the incident keeps its consequence: a closed axis was reopened');

// The rule above was prose in a role file for one wave. Prose is advisory — the pool-availability
// rule sat in exactly this state and a whole wave walked past it. The script now compares the hashes
// itself, so a verifier that buries them in `notes` cannot close an axis on one binary.
console.log('\n# the script compares the artifacts itself, so the rule is not merely advisory');
ok(/artifact_distinct: \{ type: 'string' \}/.test(wf),
   'VERIFY_SCHEMA carries a dedicated artifact_distinct field');
ok(/artifact_hash_base: \{ type: 'string' \}/.test(wf) &&
   /artifact_hash_candidate: \{ type: 'string' \}/.test(wf),
   'the two hashes are structured fields the script can compare, not prose in notes');
ok(/ARTIFACT DISTINCTNESS/.test(wf) && /SAME BINARY/.test(wf),
   'an unproven comparison and a proven-identical one get distinct log lines');
// The gate must run after activation, so a patch that never executed is reported as inactive rather
// than mislabelled as a same-binary void — the two have different remedies.
const iAct = wf.indexOf('--- ACTIVATION: did the candidate');
const iArt = wf.indexOf('--- ARTIFACT DISTINCTNESS');
const iVerifiedFilter = wf.indexOf("const verified = clean.filter(");
ok(iAct > 0 && iArt > iAct && iVerifiedFilter > iArt,
   'the artifact gate runs after activation and before the winner filter');
ok(/is VOID, not a null/.test(wf),
   'an identical-artifact result is voided rather than filed as a negative finding');
ok(/`n\/a` \(not a JIT candidate\) and `unknown`/.test(wf),
   'n/a and unknown stay non-fatal, so the verifier is not pushed into a false "yes"');
ok(/Report this in the three dedicated fields, not only in prose/.test(verify),
   'the verifier is told the script reads the fields, not the notes');

// Exercise the actual comparison rather than trusting the text around it. Lifted from source so a
// flipped condition fails here instead of in a wave.
console.log('\n# artifact comparison arithmetic (extracted from source)');
const artSrc = wf.match(
  /(const ad = String\(r\.ver\.artifact_distinct[\s\S]*?const sameHash = [^\n]*\n)\s*if \(ad !== 'no' && !sameHash\) \{/,
);
ok(!!artSrc, 'the artifact decision expression can be located in kernel_mega.js');
if (artSrc) {
  const decideArt = new Function('ver', `
    const r = { ver };
    ${artSrc[1]}
    return { fatal: !(ad !== 'no' && !sameHash), sameHash };
  `);
  const cases = [
    [{ artifact_distinct: 'yes', artifact_hash_base: 'aaa', artifact_hash_candidate: 'bbb' }, false,
     'differing hashes with a yes verdict survive'],
    [{ artifact_distinct: 'yes', artifact_hash_base: 'aaa', artifact_hash_candidate: 'aaa' }, true,
     'IDENTICAL hashes void the result even when the verifier claimed yes — the hashes outrank the verdict'],
    [{ artifact_distinct: 'no', artifact_hash_base: 'aaa', artifact_hash_candidate: 'bbb' }, true,
     'an explicit no voids it even without matching hashes'],
    [{ artifact_distinct: 'n/a' }, false, 'a non-JIT candidate is not penalised'],
    [{ artifact_distinct: 'unknown' }, false, 'an unrunnable proof warns but does not void'],
    [{}, false, 'a silent verifier does not void the result — activation already covers silence'],
    [{ artifact_distinct: 'NO' }, true, 'the verdict comparison is case-insensitive'],
    [{ artifact_hash_base: '  aaa ', artifact_hash_candidate: 'aaa' }, true,
     'hashes are trimmed before comparison, so whitespace cannot fake a difference'],
    [{ artifact_hash_base: '', artifact_hash_candidate: '' }, false,
     'two EMPTY hashes are not "identical artifacts" — that is a verifier that reported nothing'],
  ];
  for (const [ver, wantFatal, msg] of cases) ok(decideArt(ver).fatal === wantFatal, msg);
}

// A positive control that requires a FINISHED OPTIMIZATION does not generalise: most runs have no
// known win lying around, and in a capability evaluation the one that does is the answer itself,
// applied — the control workspace leaks harder than a reference checkout. The gate only ever asked
// whether the loop can resolve an effect of a given size, and a deliberate SLOWDOWN answers that
// just as well. So the arithmetic works on magnitude plus an expected sign. These cases pin that a
// negative expectation is a first-class control and that the old positive behaviour is unchanged.
console.log('\n# the positive control accepts a synthetic slowdown, not only a known win');
if (decide) {
  const decidePC = (lo, hi, got, nullPct, implausible) => decide(
    { ran: true, measured_pct: got, null_arm_pct: nullPct },
    { expected_pct_lo: lo, expected_pct_hi: hi, implausible_pct: implausible },
  );
  const cases = [
    // lo,   hi,  got,  null, implausible, expect
    [3.55, 4.93, 4.56, 0.07, undefined, { ok: true },
     'the classic known-win control still passes exactly as before'],
    [3.55, 4.93, 1.00, 0.07, undefined, { tooSmall: true },
     'a known win read far under its floor is still an insensitivity abort'],
    [3.55, 4.93, -4.71, 0.07, undefined, { tooSmall: true },
     'wrong sign is insensitivity, not overshoot — the loop did not track the change it was handed'],
    [3.55, 4.93, 5.13, 0.07, undefined, { ok: true, overshoot: true },
     'an overshoot beside a quiet null arm passes and carries the caveat'],
    [3.55, 4.93, 5.13, 3.00, undefined, { ok: false, overshoot: true },
     'the same overshoot beside a LOUD null arm fails — the excess is drift, not effect'],
    // The whole point: a deliberate slowdown, which the old positive-only arithmetic called absurd.
    [-4.0, -2.0, -3.0, 0.07, undefined, { ok: true },
     'a synthetic slowdown control inside its band PASSES (this is the case that used to abort)'],
    [-4.0, -2.0, -0.5, 0.07, undefined, { tooSmall: true },
     'a slowdown the loop barely registers is an insensitivity abort'],
    [-4.0, -2.0, -12.0, 0.07, undefined, { absurd: true },
     'a slowdown far beyond the plausibility ceiling is measuring something else'],
    [-4.0, -2.0, 3.0, 0.07, undefined, { tooSmall: true },
     'a slowdown control that came back FASTER is a wiring error, caught as wrong sign'],
    [-4.0, -2.0, -9.0, 0.07, 10, { ok: true, overshoot: true },
     'implausible_pct is read as a magnitude, so it bounds a negative control too'],
  ];
  for (const [lo, hi, got, nul, imp, want, msg] of cases) {
    const r = decidePC(lo, hi, got, nul, imp);
    ok(Object.entries(want).every(([k, v]) => r[k] === v), msg);
  }
}

console.log(
  failures === 0
    ? '\nPASS: detection floor, prior art, and result provenance are all gated and reach the report.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
