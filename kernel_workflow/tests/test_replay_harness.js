#!/usr/bin/env node
// The replay harness is the first thing in this suite that tests BEHAVIOUR rather than prose, so it
// had better be tested that way itself.
//
// Every other suite here asserts that a sentence is present in a role file or that a regex matches
// the workflow source. That catches a deletion and nothing else: a change to the gate arithmetic
// that silently re-decides finished runs passes all of them. `scripts/replay_runs.js` exists to
// close that hole by re-deciding recorded runs on no GPU. This file proves it can:
//
//   1. it computes the paired effect the way the discipline requires (within-rep, rank-max, VOID
//      arms dropped whole);
//   2. it pairs the two arms of a control by WORKSPACE, not by whichever arm is declared base —
//      the mistake that made a real +4.5% control read +2.5% and abort a run;
//   3. a change to the gate in kernel_mega.js CHANGES ITS OUTPUT. Without this assertion the
//      harness could be a no-op that always prints the same thing, which would be worse than having
//      no harness, because the clean --check would be read as evidence.
//
// The corpus is synthesised here rather than shipped, so the test runs in any checkout and carries
// no measurement from any particular machine.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const REPLAY = path.join(ROOT, 'scripts', 'replay_runs.js');
const WF = path.join(ROOT, 'kernel_mega.js');

let failures = 0;
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}
function run(args) {
  try { return { code: 0, out: execFileSync('node', [REPLAY, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) }; }
  catch (e) { return { code: e.status == null ? -1 : e.status, out: `${e.stdout || ''}${e.stderr || ''}` }; }
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'gk_replay_test_'));
process.on('exit', () => { try { fs.rmSync(tmp, { recursive: true, force: true }); } catch { /* best effort */ } });

// ---------------------------------------------------------------- synthetic corpus
// Four arms in the shape a real control takes: the run's own baseline, a byte-identical null copy of
// it, and a matched off/on pair living together in a separate frozen workspace. The frozen pair is
// 4% apart; the frozen tree as a whole is 2% off the run's baseline, which is exactly the confounder
// that makes cross-workspace pairing wrong.
const corpus = path.join(tmp, 'runs', 'team_x', 'kern');
fs.mkdirSync(corpus, { recursive: true });
const REPS = 6;
const runs = [];
const mk = (arm, guard, rep, max) => ({ arm, guard, rep, rc: 0, ok: true, marker_ok: true, mean: max * 0.99, max });
for (let rep = 1; rep <= REPS; rep++) {
  const drift = 1 + 0.004 * Math.sin(rep);       // shared, so pairing cancels it
  const base = 4.60 * drift;
  runs.push(mk('base', 'g1', rep, base));
  runs.push(mk('null', 'g1', rep, base * 1.0005));  // byte-identical: quiet
  runs.push(mk('off', 'g1', rep, base * 1.02));     // frozen tree, switch off — 2% SLOWER than base
  runs.push(mk('on', 'g1', rep, base * 1.02 * 0.96)); // switch on — 4% faster than its own off arm
}
// One dead rep on the candidate: a crashed arm must remove the whole pair, not half of it. And one
// rep that completed and reported a latency but never printed the path marker — the more dangerous
// case, because it carries a plausible number that silently belongs to the fallback path.
runs.push({ arm: 'on', guard: 'g1', rep: 99, rc: 1, ok: false, max: null, marker_ok: false });
runs.push(mk('off', 'g1', 99, 4.60));
runs.push(Object.assign(mk('on', 'g1', 98, 4.10), { marker_ok: false }));
runs.push(mk('off', 'g1', 98, 4.60));
fs.writeFileSync(path.join(corpus, 'setup_ab_control.json'), JSON.stringify({
  arms: [
    { name: 'base', ws: '/w/run', env: {} },
    { name: 'null', ws: '/w/run_copy', env: {} },
    { name: 'off', ws: '/w/frozen', env: {} },
    { name: 'on', ws: '/w/frozen', env: { OPT_SWITCH: '1' } },
  ],
  guards: ['g1'], base: 'base', runs,
}, null, 1));

const RUNS_DIR = path.join(tmp, 'runs');

console.log('\n# the harness reads a corpus and reports what it dropped');
let r = run(['--runs', RUNS_DIR]);
ok(r.code === 0, `replay exits clean (${r.code})`);
ok(/setup_ab_control\.json/.test(r.out), 'the A/B artifact is discovered');
ok(/void=2/.test(r.out), 'the two unusable runs are counted VOID; their healthy partners are not');
ok(/rc=1/.test(r.out) && /path marker absent/.test(r.out),
   'each VOID run says WHY — a nonzero exit and a missing path marker are different failures');
ok(/dropped:[^\n]*rep98[^\n]*on\(path marker absent\)/.test(r.out) && /rep99/.test(r.out),
   'a rep with a VOID arm is dropped WHOLE and named — half a pair is not a pair');

console.log('\n# the two arms of a control are paired by workspace, not by declared base');
ok(/on vs off \(same-workspace\)/.test(r.out),
   'the frozen pair is matched to each other');
ok(!/on vs base/.test(r.out),
   'the candidate is NOT compared against the run\'s own baseline across workspaces — the pairing error that aborted a passing control');
const m = r.out.match(/measured=([-\d.]+)%\s+\((\d+) pairs/);
ok(!!m, 'a paired effect is reported with its pair count');
if (m) {
  const pct = Number(m[1]);
  ok(Math.abs(pct - 4.0) < 0.15, `the paired effect recovers the injected 4% (got ${pct}%)`);
  ok(Number(m[2]) === REPS, `the crashed rep is excluded from the pair count (${m[2]} of ${REPS})`);
}
const nul = r.out.match(/null=([-\d.]+)%/);
ok(nul && Math.abs(Number(nul[1])) < 0.2,
   'the null arm is paired against the baseline it copies, so it reads quiet rather than picking up the cross-workspace offset');

console.log('\n# both signs are decided on the same data');
ok(/band=speedup_3\.55\.\.4\.93[\s\S]*?-> PASS/.test(r.out),
   'a +4% effect passes a speedup band');
ok(/band=slowdown_-5\.0\.\.-2\.5[^\n]*\n?[^\n]*-> FAIL_WRONG_SIGN/.test(r.out),
   'the same +4% effect is WRONG SIGN against a slowdown band — the gate is sign-aware on real records, not just on hand-written cases');

console.log('\n# snapshot / --check is a working regression gate');
const snap = path.join(tmp, 'snap.txt');
ok(run(['--runs', RUNS_DIR, '--snapshot', snap]).code === 0 && fs.existsSync(snap), 'a snapshot can be recorded');
ok(run(['--runs', RUNS_DIR, '--check', snap]).code === 0, 'an unchanged workflow re-checks clean');

// ---------------------------------------------------------------- falsification
// The assertion that gives every other one its meaning: perturb the live gate and the snapshot must
// break. A harness whose --check cannot fail is a harness that certifies nothing.
console.log('\n# a change to the live gate is detected');
const orig = fs.readFileSync(WF, 'utf8');
let restored = false;
const restore = () => { if (!restored) { fs.writeFileSync(WF, orig); restored = true; } };
process.on('exit', restore);
try {
  // Swap the floor of the sensitivity band for its ceiling, inside the replayed region: an in-band
  // effect then reads as insensitivity. Chosen because it is the kind of change that is invisible in
  // a source diff review and catastrophic in a run.
  const patched = orig.replace(
    'const mLo = Math.min(Math.abs(lo), Math.abs(hi));',
    'const mLo = Math.max(Math.abs(lo), Math.abs(hi));');
  ok(patched !== orig, 'the perturbation target exists in the replayed region');
  fs.writeFileSync(WF, patched);
  const after = run(['--runs', RUNS_DIR, '--check', snap]);
  ok(after.code === 4, `--check reports a difference (exit ${after.code}, expected 4)`);
  ok(/re-decided a finished run/.test(after.out), 'the failure says what it means: a finished run would now be decided differently');
  ok(/^\s*[-+] /m.test(after.out), 'the diff shows which lines changed, so the blast radius is visible');
} finally { restore(); }
ok(fs.readFileSync(WF, 'utf8') === orig, 'the workflow source is restored after the falsification');
ok(run(['--runs', RUNS_DIR, '--check', snap]).code === 0, 'and the snapshot checks clean again');

console.log('\n# the lift refuses to run against a source it cannot locate');
const stripped = orig.replace('// <<REPLAY:pc_gate>>', '// (markers removed)');
try {
  fs.writeFileSync(WF, stripped);
  const noMark = run(['--runs', RUNS_DIR]);
  ok(noMark.code === 1 && /markers are missing/.test(noMark.out),
     'without the markers the harness stops instead of replaying a stale copy of the arithmetic');
} finally { fs.writeFileSync(WF, orig); }

// ---------------------------------------------------------------- the schema engineers actually write
// This harness was written against one artifact shape and silently read ZERO runs out of the other,
// printing `runs=0 void=0` for a corpus holding a complete 6-pair control. A replay that reports
// nothing to re-decide is indistinguishable from a replay that re-decided nothing, and the second is
// what a clean --check gets read as. So: the alternative shape must be ingested, and an artifact this
// script genuinely cannot use must be LOUD.
console.log('\n# the wave-7 artifact shape is ingested, not silently skipped');
const alt = path.join(tmp, 'runs2', 'team_y', 'kern');
fs.mkdirSync(alt, { recursive: true });
const altRecs = [];
for (let i = 0; i < 6; i++) {
  const b = 4.60 * (1 + 0.004 * Math.sin(i));
  // No `arms` block, no `base`, no `rep`, no `max`: arms are inferred from the records, the base is
  // the arm whose switch reads zero, and the k-th occurrence of each arm is pair k.
  altRecs.push({ arm: 'ctrl_off', tokens: 8192, route: 'uniform', guard: '8192_uniform', tree: '/w/ctrl', env: { SPIN: 0 }, rc: 0, e2e_max_ms: b, idx: 2 * i });
  altRecs.push({ arm: 'ctrl_on', tokens: 8192, route: 'uniform', guard: '8192_uniform', tree: '/w/ctrl', env: { SPIN: 25 }, rc: 0, e2e_max_ms: b * 1.023, idx: 2 * i + 1 });
}
fs.writeFileSync(path.join(alt, 'setup_ab_control.json'), JSON.stringify({
  records: altRecs, measured_pct: -2.3, null_arm_pct: -0.04, null_pairs_pct: [0.2, -0.04, -0.35, 0.23, -0.11],
}, null, 1));
const RUNS2 = path.join(tmp, 'runs2');
let r2 = run(['--runs', RUNS2]);
ok(r2.code === 0, `the alternative shape replays (${r2.code})`);
ok(/arms=\[ctrl_off, ctrl_on\]/.test(r2.out),
   'arms are inferred from the records when the artifact declares no arms block');
ok(/ctrl_on vs ctrl_off \(same-workspace\)/.test(r2.out),
   'the base arm is the one whose switch reads ZERO — an env of {SPIN:0} is the switch off, not a second candidate');
const m2 = r2.out.match(/measured=(-[\d.]+)%\s+\((\d+) pairs/);
ok(!!m2 && Number(m2[2]) === 6,
   `all six interleaved pairs are recovered without a rep field (${m2 ? m2[2] : 'none'})`);
ok(!!m2 && Math.abs(Number(m2[1]) + 2.25) < 0.15,
   `the injected slowdown is recovered from e2e_max_ms (${m2 ? m2[1] : '?'}%)`);
ok(/null=-0\.04% \(5, recorded\)/.test(r2.out),
   'a null arm reported as a summary rather than as records is used AND labelled recorded, not dropped to NaN');

console.log('\n# an artifact that cannot be replayed is loud, and an empty corpus is not a pass');
const junk = path.join(tmp, 'runs3', 'team_z', 'kern');
fs.mkdirSync(junk, { recursive: true });
fs.writeFileSync(path.join(junk, 'setup_ab_weird.json'), JSON.stringify({ measurements: [{ who: 'a', ms: 1 }] }));
const r3 = run(['--runs', path.join(tmp, 'runs3')]);
ok(/!! unusable/.test(r3.out) && /not replayable/.test(r3.out),
   'an artifact with no arm, guard or timing is reported as unusable rather than counted as zero runs');
ok(/Expected per record/.test(r3.out), 'and it says what shape it wanted, so the artifact can be fixed');
const snap3 = path.join(tmp, 'snap3.txt');
ok(run(['--runs', path.join(tmp, 'runs3'), '--snapshot', snap3]).code === 1,
   'a corpus that yields no gate verdict FAILS a snapshot instead of recording a snapshot of nothing');
// A single-arm baseline is a legitimate artifact, not a broken one. Flagging it every run is how a
// reader learns to skim past the `!!` lines, which is how the real one gets missed.
const solo = path.join(tmp, 'runs4', 'team_w', 'kern');
fs.mkdirSync(solo, { recursive: true });
fs.writeFileSync(path.join(solo, 'setup_ab_base.json'), JSON.stringify({
  records: [{ arm: 'base', guard: 'g1', rc: 0, e2e_max_ms: 4.6 }, { arm: 'base', guard: 'g1', rc: 0, e2e_max_ms: 4.61 }],
}));
const r4 = run(['--runs', path.join(tmp, 'runs4')]);
ok(/-- .*setup_ab_base\.json .* a baseline, not an A\/B/.test(r4.out) && !/!! unusable.*setup_ab_base/.test(r4.out),
   'a one-arm baseline is reported as a baseline, not as an error');

// ---------------------------------------------------------------- constructed vs recorded controls
// The branch this section exists for aborted one run. A synthetic slowdown aimed at ~3.4% measured
// 2.30%, 6/6 pairs negative, against a null arm whose worst pair was 0.35pp — and a band written as
// -5..-2.5 killed the run over 0.2pp of KNOB-SIZING error. Whether that is a harness failure or a
// sizing miss depends entirely on whether the band was a measurement or a target, so the gate must
// decide it differently in the two cases, and that difference has to be tested on records rather
// than asserted in prose.
console.log('\n# a constructed control may undershoot its target; a recorded one may not');
function gateOn(pcExtra, bandExtra) {
  // Drive the LIVE gate through the replay's own lift, so this test cannot drift from the workflow.
  const src = fs.readFileSync(WF, 'utf8');
  const mm = src.match(/\/\/ <<REPLAY:pc_gate>>[\s\S]*?\n([\s\S]*?)\n\s*\/\/ <<\/REPLAY:pc_gate>>/);
  const fn = new Function('pc', 'POSITIVE_CONTROL', `
    const lo = Number(POSITIVE_CONTROL.expected_pct_lo);
    const hi = Number(POSITIVE_CONTROL.expected_pct_hi);
    const got = Number(pc.measured_pct);
${mm[1]}
    return { ok, tooSmall, undershoot, resolved, signUnanimous };
  `);
  return fn(
    { ran: true, measured_pct: -2.30, null_arm_pct: -0.04,
      null_pairs_pct: [0.2, -0.04, -0.35, 0.23, -0.11],
      control_pairs_pct: [-2.42, -3.07, -2.37, -1.58, -2.23, -2.07], ...pcExtra },
    { expected_pct_lo: -5.0, expected_pct_hi: -2.5, implausible_pct: 15.0, ...bandExtra });
}
ok(gateOn({}, { magnitude: 'constructed' }).ok === true,
   'one run\'s control PASSES when its band is declared a target: -2.30% is ~7x the worst null pair and 6/6 pairs agree');
ok(gateOn({}, { magnitude: 'constructed' }).undershoot === true,
   'and it passes as an UNDERSHOOT, so the caveat travels into the report rather than vanishing');
ok(gateOn({}, {}).ok === false,
   'the same numbers FAIL under the default: a recorded effect that reads short IS the instrument\'s fault');
ok(gateOn({}, { magnitude: 'recorded' }).ok === false, 'and `recorded` is the explicit form of that default');
ok(gateOn({ measured_pct: -0.9, control_pairs_pct: [-0.9, -1.0, -0.8, -0.9, -0.9, -1.0] },
          { magnitude: 'constructed' }).ok === false,
   'an injection reading under HALF its target still aborts — that is a switch that never took effect, not a sizing miss');
ok(gateOn({ control_pairs_pct: [-2.42, 3.07, -2.37, 1.58, -2.23, -2.07] }, { magnitude: 'constructed' }).ok === false,
   'pairs that disagree in sign abort: a median out of a disagreement is drift, not an effect');
ok(gateOn({ null_pairs_pct: [1.9, -1.7, 1.8, -1.6, 1.5] }, { magnitude: 'constructed' }).ok === false,
   'a LOUD null aborts even at the same measured value — the effect must beat the interleave, not the median');
ok(gateOn({ null_arm_pct: -0.04, null_pairs_pct: undefined }, { magnitude: 'constructed' }).ok === true,
   'with no per-pair nulls recorded the median is used as a fallback rather than failing the control outright');
ok(gateOn({ measured_pct: 2.30, control_pairs_pct: [2.4, 3.1, 2.4, 1.6, 2.2, 2.1] },
          { magnitude: 'constructed' }).ok === false,
   'the undershoot latitude does NOT extend to the wrong sign: a slowdown that reads as a speedup is a wiring error');

console.log(
  failures === 0
    ? '\nPASS: finished runs can be re-decided, and a gate change is caught without a GPU.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
