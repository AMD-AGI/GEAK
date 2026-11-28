#!/usr/bin/env node
// Cross-round memory: the loop's largest source of wasted budget, and until now the one diagnosed
// cause with no code behind it.
//
// The bug was a single assignment, `history.insights = mem.insights`. `update_memory` asks an agent
// to distil durable insights; an agent looking at one round's results returns that round's insights;
// the assignment discarded everything earlier it happened not to restate. So the board silently
// shrank to the most recent round, and engineers in round 3 re-proposed directions round 1 had
// already disproved — paying twice out of a budget counted in 8-card leases.
//
// These assertions run the real merge, lifted from between the `<<REPLAY:memory_merge>>` markers in
// kernel_lane.js, on inputs shaped like the ones that caused the loss. They are behavioural: if
// someone restores the assignment, or quietly re-introduces silent eviction, they fail.

const fs = require('fs');
const path = require('path');
const megaRole = require('./mega_roles');

const ROOT = path.resolve(__dirname, '..');
const WF = path.join(ROOT, 'kernel_lane.js');
const src = require('./mega_source')();

let failures = 0;
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}

const region = src.match(/\/\/ <<REPLAY:memory_merge>>\n([\s\S]*?)\n\/\/ <<\/REPLAY:memory_merge>>/);
ok(!!region, 'the memory-merge region can be located in kernel_lane.js');
if (!region) { process.exit(1); }
const M = new Function(`${region[1]}\nreturn { mergeInsights, renderInsights, normInsight };`)();

console.log('\n# an insight the summariser stops repeating SURVIVES');
let book = M.mergeInsights([], ['LDS is the binding constraint at 1 WG/CU'], 1, false).book;
book = M.mergeInsights(book, ['the barrier dominates under skew'], 2, false).book;
book = M.mergeInsights(book, ['payload transfer is not free'], 3, false).book;
ok(book.length === 3, `all three rounds are on the board (got ${book.length})`);
ok(book.some((e) => /LDS is the binding constraint/.test(e.text)),
   'round 1\'s finding is still there in round 3, though nobody restated it — the exact loss the old assignment caused');

console.log('\n# provenance survives re-summarisation');
const rendered = M.renderInsights(book);
ok(/^\[r1\] LDS is the binding constraint/.test(rendered[0]),
   'each rendered insight carries the round it first appeared in');
// An agent that echoes the rendered board back must not create a duplicate with a new origin round.
const echoed = M.mergeInsights(book, rendered, 4, false).book;
ok(echoed.length === 3, `echoing the whole board back adds nothing (got ${echoed.length})`);
ok(echoed[0].first_round === 1 && echoed[0].last_round === 4,
   'a restated insight keeps its ORIGIN round and updates only its last-seen round');
ok(echoed[0].restated === 1, 'restatement is counted, so a repeatedly-confirmed finding can outrank a stale one');
ok(!/\[r1\]\s*\[r1\]/.test(M.renderInsights(echoed)[0]), 'the provenance tag is not doubled on the round trip');

console.log('\n# a round that measured nothing cannot launder its claims into measurements');
let voidBook = M.mergeInsights([], ['stage2 branchless epilog is a dead end'], 2, true).book;
ok(/FROM-VOID-ROUND/.test(M.renderInsights(voidBook)[0]),
   'an insight distilled from an all-INACTIVE round is tagged as such');
voidBook = M.mergeInsights(voidBook, ['stage2 branchless epilog is a dead end'], 3, true).book;
ok(/FROM-VOID-ROUND/.test(M.renderInsights(voidBook)[0]),
   'restating it in another evidence-free round does not upgrade it');

console.log('\n# eviction is bounded AND reported');
let big = [];
for (let i = 1; i <= 12; i++) big = M.mergeInsights(big, [`finding ${i}`], i, false).book;
const merged = M.mergeInsights(big, ['finding 13'], 13, false, 5);
ok(merged.book.length === 5, `the board is capped (got ${merged.book.length})`);
ok(merged.evicted.length === 8, `everything dropped is returned to the caller (got ${merged.evicted.length})`);
ok(merged.book.some((e) => e.text === 'finding 13') && merged.book.some((e) => e.text === 'finding 12'),
   'the most recent findings are the ones kept');
ok(merged.evicted.some((e) => e.text === 'finding 1'), 'the oldest is the one aged out');
// The caller has to actually print them; a returned list nobody logs is still a silent drop.

console.log('\n# blank and malformed entries do not consume the board');
const junk = M.mergeInsights([], ['', '   ', null, undefined, 'real one'], 1, false).book;
ok(junk.length === 1 && junk[0].text === 'real one', `only real content is stored (got ${junk.length})`);

console.log('\n# the tech lead keeps the board');
const lead = megaRole('tech_lead');
ok(/The board is merged, not replaced/.test(lead),
   'the tech lead is told not to re-list history defensively');
ok(/Restating an existing insight is a signal, not padding/.test(lead),
   'and is told what restating now means, since it has a mechanical effect');

console.log('\n# the assignment that caused the loss is gone');
const codeLines = src.split('\n').filter((l) => !/^\s*(\/\/|\*)/.test(l));
ok(!codeLines.some((l) => /history\.insights = mem\.insights/.test(l)),
   'the wholesale replacement is gone from the code (the comment recording it may stay — it is why the merge exists)');

console.log(
  failures === 0
    ? '\nPASS: the blackboard accumulates, ages loudly, and tells engineers what was never tried.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
