#!/usr/bin/env node
// Regression guard for the head-budget REFUND (no GPU, no model needed).
//
// Why this exists. The head track caps itself at HEAD_BUDGET recursive authoring runs. On gfx1151 the
// 2026-09-21 run spent all three of them on lm_head / gate_up_proj / down_proj -- every one a decode
// GEMM against the part's 233 GB/s LPDDR5X wall (lm_head measured 233.4 GB/s, i.e. 100% of roofline).
// All three bake-offs came back with a measured Amdahl ceiling of 0.0%, so 2h06m of budget bought three
// proofs of physics while ops the kernel lane has already beaten 2.21x sat below the 5% head bar and
// were never reached. The refund returns such a slot to the queue.
//
// What must stay true, and is pinned below:
//   1. The refund is OFF by default on every non-RDNA part -- the CDNA path is byte-for-byte unchanged.
//   2. An ABSENT ceiling never refunds. "We did not measure" is not "we measured zero", and conflating
//      them retires an op that was never timed.
//   3. A harness fault never refunds, for the same reason: a probe that could not measure has said
//      nothing about the op.
//   4. A ceiling ABOVE the noise band never refunds -- that op can still pay.
//   5. The refund is BOUNDED (<= HEAD_REFUND_MAX) and cannot run past the end of the queue.
//
// The predicate is LIFTED OUT OF THE SHIPPED SOURCE rather than restated here: a restatement would keep
// passing while e2e_workflow.js drifted, which is the entire failure mode this file is meant to catch.
//
// Run:  node e2e_workflow/scripts/test_head_budget_refund.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const WORKFLOW = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');
const src = fs.readFileSync(WORKFLOW, 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

// ---------------------------------------------------------------- lift the shipped expressions
// (a) the arch-defaulted cap
const capRe = /A\.head_refund_max != null \? A\.head_refund_max : \(([^)]*\([^)]*\)[^)]*|[^)]*)\)/;
const capM = src.match(capRe);
ok(!!capM, 'HEAD_REFUND_MAX default expression located in e2e_workflow.js');
// (b) how the ceiling is read off the bake-off return
const ceilM = src.match(/const ceiling = bake && typeof bake\.amdahl_ceiling_e2e_pct === 'number'\s*\n?\s*\? bake\.amdahl_ceiling_e2e_pct : null;/);
ok(!!ceilM, 'ceiling extraction expression located');
// (c) the refund guard itself
const guardM = src.match(/if \((REFUND_LIVE && headRefundsUsed < HEAD_REFUND_MAX[\s\S]*?nextIdx < headQueue\.length)\) \{/);
ok(!!guardM, 'refund guard condition located');
if (failures) { console.error('\ncould not lift the shipped expressions -- did the refund block move?'); process.exit(1); }

const archDefault = new Function('archFamily', 'HEAD_BUDGET',
  `return (${capM[1]});`);
const readCeiling = new Function('bake',
  ceilM[0].replace(/^const ceiling = /, 'return ').replace(/;$/, ';'));
const refunds = new Function(
  'REFUND_LIVE', 'headRefundsUsed', 'HEAD_REFUND_MAX', 'harness', 'ceiling', 'NOISE_BAND',
  'nextIdx', 'headQueue', `return !!(${guardM[1]});`);

// ---------------------------------------------------------------- 1. arch default
console.log('arch default:');
ok(archDefault(() => 'cdna', 3) === 0, 'cdna defaults to 0 refunds -- CDNA behaviour unchanged');
ok(archDefault(() => '', 3) === 0, 'unknown arch defaults to 0 refunds -- conservative');
ok(archDefault(() => 'rdna', 3) === 3, 'rdna defaults to HEAD_BUDGET refunds (queue bounded at 2x budget)');

// ---------------------------------------------------------------- 2. reading the ceiling
console.log('ceiling extraction:');
ok(readCeiling({ amdahl_ceiling_e2e_pct: 0.0 }) === 0, 'a measured 0.0 reads back as 0, not null');
ok(readCeiling({ amdahl_ceiling_e2e_pct: 4.2 }) === 4.2, 'a measured value reads back verbatim');
ok(readCeiling({}) === null, 'an omitted key reads back as null (unknown)');
ok(readCeiling({ amdahl_ceiling_e2e_pct: null }) === null, 'an explicit null reads back as null, not 0');
ok(readCeiling(null) === null, 'a null bake-off reads back as null');

// ---------------------------------------------------------------- 3. the guard, one axis at a time
console.log('refund guard:');
const Q = new Array(9).fill(0).map((_, i) => ({ short_name: `op${i}` }));
const base = { LIVE: true, used: 0, max: 3, harness: false, ceiling: 0.0, noise: 0.5, next: 3 };
const fire = (o) => { const c = { ...base, ...o };
  return refunds(c.LIVE, c.used, c.max, c.harness, c.ceiling, c.noise, c.next, Q); };

ok(fire({}) === true, 'measured 0.0 ceiling under a 0.5% band, budget left, queue left -> refund');
ok(fire({ ceiling: 0.5 }) === true, 'a ceiling exactly AT the band refunds (it cannot clear the gate)');
ok(fire({ ceiling: 0.51 }) === false, 'a ceiling above the band does NOT refund -- that op can still pay');
ok(fire({ ceiling: null }) === false, 'an UNMEASURED ceiling never refunds');
ok(fire({ harness: true }) === false, 'a harness fault never refunds -- it measured nothing');
ok(fire({ LIVE: false }) === false, 'inert outside the serial head loop (deep/fast prepare up front)');
ok(fire({ max: 0 }) === false, 'max 0 (the non-RDNA default) never refunds');
ok(fire({ used: 3, max: 3 }) === false, 'the refund is bounded -- exhausted budget stops it');
ok(fire({ next: Q.length }) === false, 'cannot refund past the end of the queue');

// ---------------------------------------------------------------- 4. the loop this actually changes
// Replays the serial head loop's admission arithmetic with the shipped guard: the same three
// bandwidth-wall heads, once with the shipped CDNA default and once with the RDNA default.
console.log('serial-loop replay (three zero-ceiling heads, 9-op queue):');
const replay = (max) => {
  const HEAD_BUDGET = 3;
  const heads = Q.slice(0, HEAD_BUDGET).map((c) => c.short_name);
  let used = 0;
  for (let i = 0; i < heads.length; i++) {          // for..of over a live Array, as shipped
    const ceiling = readCeiling({ amdahl_ceiling_e2e_pct: 0.0 });   // every head hits the memory wall
    const nextIdx = HEAD_BUDGET + used;
    if (refunds(max > 0, used, max, false, ceiling, 0.5, nextIdx, Q)) { used++; heads.push(Q[nextIdx].short_name); }
  }
  return heads;
};
const cdna = replay(0);
ok(cdna.length === 3 && cdna.join(',') === 'op0,op1,op2',
  'cap 0 (CDNA/unknown): exactly the first 3 ops are visited -- identical to every prior run');
const rdna = replay(3);
ok(rdna.length === 6 && rdna.join(',') === 'op0,op1,op2,op3,op4,op5',
  'cap 3 (RDNA): the three wall-bound heads hand their slots to op3/op4/op5');
ok(rdna.slice(0, 3).join(',') === cdna.join(','),
  'the refund only ADDS heads -- it never drops or reorders one the old path would have run');

// ---------------------------------------------------------------- 5. structural: the number is carried
console.log('plumbing:');
ok(/amdahl_ceiling_e2e_pct: \{ type: 'number' \}/.test(src),
  'OPBENCH_SCHEMA carries amdahl_ceiling_e2e_pct up from the bake-off');
ok(!/required.*amdahl_ceiling/.test(src),
  'it is OPTIONAL -- a bake-off that could not measure is not forced to invent a number');
// Split rather than a global regex: a lazy [\s\S]* match can run past one call site into the next
// and silently under-count, which would make this assertion pass with a site left unpatched.
const bakeoffCalls = src.split(/roleAgent\('op_benchmarker', 'bakeoff'/).slice(1)
  .map((tail) => tail.slice(0, 500));
ok(bakeoffCalls.length === 3, `all three bakeoff call sites found (${bakeoffCalls.length})`);
ok(bakeoffCalls.every((c) => /NOISE_BAND_PCT: NOISE_BAND/.test(c)),
  'every bakeoff call passes NOISE_BAND_PCT -- op_benchmarker.md judges the ceiling against it');
ok(/head_refunds: headRefunds/.test(src), 'refunds are carried in state and reported, not silent');

const role = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'roles', 'op_benchmarker.md'), 'utf8');
ok(/`NOISE_BAND_PCT`/.test(role.split('PHASE=bakeoff')[1] || ''),
  'op_benchmarker.md declares NOISE_BAND_PCT as a bakeoff input');
ok(/"amdahl_ceiling_e2e_pct":/.test(role), 'op_benchmarker.md declares the field in its return JSON');
ok(/OMIT the\n\s*key entirely when the bake-off did not measure/.test(role),
  'op_benchmarker.md tells the role to OMIT rather than guess 0.0');

console.log(failures ? `\n${failures} FAILED` : '\nall good');
process.exit(failures ? 1 : 0);
