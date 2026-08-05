#!/usr/bin/env node
// Regression guard for the PORT-SHAPE budget defaults (no GPU, no model needed).
//
// A port (mode=author: a fresh seed, or a plain -> Gluon/TileLang/HIP transcription) lands BELOW the
// frozen comparator by construction and climbs back. kernel_workflow's defaults are tuned for
// mode=optimize, where a candidate under the baseline is worthless, and applied unchanged to a port
// they delete its recovery phase entirely: the transcription round yields no candidate, so no patch is
// saved, no verify runs, `winner` is null, and the loop stops two rounds in.
//
// Two invariants under test, and the second matters as much as the first:
//   1. on mode=author the four knobs take their port values, so a port gets enough rounds AND its
//      recovery round is representable;
//   2. on mode=optimize EVERY value is the historical one, so an ordinary run -- and every e2e path
//      that dispatches `mode: 'optimize'` -- is byte-identical to the pre-feature build.
//
// The knob expressions are extracted from the ACTUAL script source and re-evaluated, so this fails if
// someone changes a default without changing this file.
//
// Run:  node e2e_workflow/scripts/test_port_budget_defaults.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const KW = path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js');
const E2E = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');
const src = fs.readFileSync(KW, 'utf8');

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail != null ? `-- ${detail}` : ''); failures++; }
  else console.log('  ok:', msg);
};

// --- rebuild the knob block for a given `A` (the CLI args) -------------------
// Pull the real expressions out of the source rather than restating them here.
function knobs(A) {
  const grab = (re, name) => {
    const m = src.match(re);
    if (!m) throw new Error(`could not extract ${name} from kernel_workflow.js`);
    return m[0];
  };
  const body = [
    grab(/const MODE = String\(A\.mode[^\n]*\n/, 'MODE'),
    grab(/const PORT_SHAPE = [^\n]*\n/, 'PORT_SHAPE'),
    grab(/const BUDGET = parseInt\(A\.budget[^\n]*\n/, 'BUDGET'),
    grab(/const MIN_IMPROVE = \(\(\) => \{[\s\S]*?\n\}\)\(\);\n/, 'MIN_IMPROVE'),
    grab(/const CANDIDATE_FLOOR = \(\(\) => \{[\s\S]*?\n\}\)\(\);\n/, 'CANDIDATE_FLOOR'),
    grab(/const PROGRESS_DELTA = \(\(\) => \{[\s\S]*?\n\}\)\(\);\n/, 'PROGRESS_DELTA'),
    grab(/const DEEP_COST = \(\(\) => \{[\s\S]*?\n\}\)\(\);\n/, 'DEEP_COST'),
    grab(/const MAX_NO_IMPROVE = Math\.max\(1, parseInt\([\s\S]*?\n?[^\n]*\)\);\n/, 'MAX_NO_IMPROVE'),
  ].join('\n');
  return new Function('A', `${body}\nreturn { MODE, PORT_SHAPE, BUDGET, MIN_IMPROVE, CANDIDATE_FLOOR, PROGRESS_DELTA, DEEP_COST, MAX_NO_IMPROVE };`)(A);
}

console.log('\n# mode=optimize is UNCHANGED (the historical defaults)');
const o = knobs({});
ok(o.PORT_SHAPE === false, 'default mode is not the port shape');
ok(o.BUDGET === 6, 'BUDGET default 6', o.BUDGET);
ok(o.CANDIDATE_FLOOR === 1.0, 'CANDIDATE_FLOOR default 1.0', o.CANDIDATE_FLOOR);
ok(o.MAX_NO_IMPROVE === 2, 'MAX_NO_IMPROVE default 2', o.MAX_NO_IMPROVE);
ok(o.MIN_IMPROVE === 0.02, 'MIN_IMPROVE default 0.02', o.MIN_IMPROVE);
ok(o.PROGRESS_DELTA === o.MIN_IMPROVE,
   'PROGRESS_DELTA equals MIN_IMPROVE, i.e. the historical progress test', o.PROGRESS_DELTA);
ok(o.DEEP_COST === 2, 'DEEP_COST default 2', o.DEEP_COST);

console.log('\n# mode=author takes the port defaults');
const p = knobs({ mode: 'author' });
ok(p.PORT_SHAPE === true, 'mode=author is the port shape');
ok(p.BUDGET === 20, 'BUDGET default 20', p.BUDGET);
ok(Math.floor(p.BUDGET / p.DEEP_COST) >= 10,
   'that is >= 10 dedicated deep_explore rounds', Math.floor(p.BUDGET / p.DEEP_COST));
ok(p.CANDIDATE_FLOOR < 1.0,
   'CANDIDATE_FLOOR is below 1.0, so the recovery phase is representable', p.CANDIDATE_FLOOR);
ok(p.PROGRESS_DELTA < 0,
   'PROGRESS_DELTA is NEGATIVE, so a round that gives ground still counts as advancing',
   p.PROGRESS_DELTA);
ok(p.MAX_NO_IMPROVE > o.MAX_NO_IMPROVE, 'MAX_NO_IMPROVE is relaxed', p.MAX_NO_IMPROVE);

console.log('\n# every knob is still overridable, in both modes');
ok(knobs({ mode: 'author', budget: 4 }).BUDGET === 4, 'explicit budget wins on author');
ok(knobs({ mode: 'author', candidate_floor: 0.9 }).CANDIDATE_FLOOR === 0.9,
   'explicit candidate_floor wins on author');
ok(knobs({ candidate_floor: 0.5 }).CANDIDATE_FLOOR === 0.5,
   'explicit candidate_floor still works on optimize');
ok(knobs({ mode: 'author', progress_delta: 0 }).PROGRESS_DELTA === 0,
   'explicit progress_delta wins');
ok(knobs({ mode: 'author', max_no_improve: 1 }).MAX_NO_IMPROVE === 1,
   'explicit max_no_improve wins');

console.log('\n# the commit gate is NOT loosened by any of this');
ok(p.MIN_IMPROVE === o.MIN_IMPROVE,
   'MIN_IMPROVE is identical in both modes -- banking still requires beating cumulative by 2%');
ok(/const improved = !!\(winner && winner\.geomean > cumulative \* \(1 \+ MIN_IMPROVE\)\)/.test(src),
   'the commit gate still reads MIN_IMPROVE against cumulative');
ok(/const madeProgress = !!\(winner && winner\.geomean > bestSeen \* \(1 \+ PROGRESS_DELTA\)\)/.test(src),
   'the progress signal reads PROGRESS_DELTA against bestSeen');
ok(/A round with NO candidate at\n\s*\/\/ all[\s\S]{0,200}never progress/.test(src),
   'a round with no candidate at all is still never progress');

console.log('\n# e2e dispatches a port lane with a port-sized wave budget');
const e2e = fs.readFileSync(E2E, 'utf8');
ok(/const DEEP_WAVE_BUDGET_PORT = parseInt\(/.test(e2e), 'DEEP_WAVE_BUDGET_PORT is defined');
ok(/budget: l\.mode === 'author' \? DEEP_WAVE_BUDGET_PORT : DEEP_WAVE_BUDGET/.test(e2e),
   'author lanes get DEEP_WAVE_BUDGET_PORT, optimize lanes are unchanged');
ok(/\.\.\.\(l\.mode === 'author' \? \{\} : \{ max_no_improve: DEEP_WAVE_BUDGET \}\)/.test(e2e),
   'author lanes do NOT override max_no_improve, so the port default applies');

// --- behavioral replay: the loop must survive the landing ---------------------
// Mirrors the real round loop: candidate filter, commit gate, bestSeen progress, DEEP_COST.
function replay({ budget, deepCost, maxNoImprove, floor, progressDelta, minImprove }, traj) {
  let dispatched = 0, round = 0, noImprove = 0, cumulative = 1.0, bestSeen = 0, commits = 0;
  while (dispatched < budget && noImprove < maxNoImprove && round < traj.length) {
    round++; dispatched += deepCost;
    const raw = traj[round - 1];
    const winner = raw > floor ? raw : null;
    const improved = winner !== null && winner > cumulative * (1 + minImprove);
    const progress = winner !== null && winner > bestSeen * (1 + progressDelta);
    if (improved) { cumulative = winner; commits++; }
    if (winner !== null && winner > bestSeen) bestSeen = winner;
    noImprove = (progress || improved) ? 0 : noImprove + 1;
  }
  return { rounds: round, commits, final: cumulative };
}

console.log('\n# behavioral replay of a real port trajectory');
// transcribe lands at 0.75, pipeline recovery reaches 1.02, then optimization in small steps --
// deliberately including a round that GIVES GROUND (1.05 -> 1.01), which is what exploration does.
const PORT_TRAJ = [0.75, 1.02, 1.05, 1.01, 1.06, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14];
const before = replay({ budget: 6, deepCost: 2, maxNoImprove: 2, floor: 1.0,
                        progressDelta: 0.02, minImprove: 0.02 }, PORT_TRAJ);
const after = replay({ budget: p.BUDGET, deepCost: p.DEEP_COST, maxNoImprove: p.MAX_NO_IMPROVE,
                       floor: p.CANDIDATE_FLOOR, progressDelta: p.PROGRESS_DELTA,
                       minImprove: p.MIN_IMPROVE }, PORT_TRAJ);
console.log(`    old defaults: ${before.rounds} round(s), ${before.commits} commit(s), ${before.final.toFixed(2)}x`);
console.log(`    port defaults: ${after.rounds} round(s), ${after.commits} commit(s), ${after.final.toFixed(2)}x`);
ok(after.rounds >= 10, 'a port now gets >= 10 rounds', after.rounds);
ok(after.rounds > before.rounds, 'and strictly more than the optimize-tuned defaults gave it');
ok(after.final > before.final, 'and lands higher', `${before.final.toFixed(2)} -> ${after.final.toFixed(2)}`);

// The control that matters: an ORDINARY run must be unaffected.
const ORD_TRAJ = [1.06, 1.03, 1.01, 1.12, 1.15];
const ordBefore = replay({ budget: 6, deepCost: 1, maxNoImprove: 2, floor: 1.0,
                           progressDelta: 0.02, minImprove: 0.02 }, ORD_TRAJ);
const ordAfter = replay({ budget: o.BUDGET, deepCost: 1, maxNoImprove: o.MAX_NO_IMPROVE,
                          floor: o.CANDIDATE_FLOOR, progressDelta: o.PROGRESS_DELTA,
                          minImprove: o.MIN_IMPROVE }, ORD_TRAJ);
ok(JSON.stringify(ordBefore) === JSON.stringify(ordAfter),
   'an ordinary optimize run replays IDENTICALLY under the new defaults',
   `${JSON.stringify(ordBefore)} vs ${JSON.stringify(ordAfter)}`);

console.log(failures
  ? `\nFAIL: ${failures} check(s) failed.`
  : '\nPASS: port-shape defaults give a port its rounds; mode=optimize is unchanged.');
process.exit(failures ? 1 : 0);
