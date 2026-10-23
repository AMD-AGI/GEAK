#!/usr/bin/env node
// The candidate shelf: verified non-winners kept as offers instead of collapsing to a number.
//
// Before it, a round remembered exactly one thing — the winner. A direction that PASSED independent
// verification at 1.03x and lost to a 1.09x survived only as `{id, claimed, verified, status}` in
// the round record, so round 5 could not combine with it, could not build on it, and would happily
// re-derive it. On this task the budget is 8 rounds of an 8-GPU collective with one lease per round,
// which makes re-deriving a finished result the most expensive mistake available.
//
// The whole difficulty is deciding whether a shelved patch STILL APPLIES. A patch is a diff against
// the CANONICAL that existed when it was cut, and every committed winner moves CANONICAL out from
// under everything on the shelf. So the test is mechanical — file sets — and this suite exists
// mostly to pin the two ways a mechanical test can quietly become a rubber stamp:
//
//   1. An empty file set read as orthogonality. "Touches nothing" is the maximally-combinable
//      answer, and it is exactly what an agent that forgot the field produces. It must read as
//      UNKNOWN and be withheld.
//   2. Aging that depends on the integrator saying which patches it used. If a superseded candidate
//      stays on offer whenever `best.patches` is omitted, the offer is unsound by default.
//
// It lifts the pure region out of kernel_mega.js the way replay_runs.js does, so what is tested
// is the shipped code rather than a copy of it.
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

const m = src.match(/\/\/ <<REPLAY:candidate_shelf>>([\s\S]*?)\/\/ <<\/REPLAY:candidate_shelf>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:candidate_shelf>> region — nothing to test'); process.exit(1); }
const { shelfFile, shelfFiles, shelfAdd, shelfEligible } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { shelfFile, shelfFiles, shelfAdd, shelfEligible };`)();

const cand = (id, geomean, files, extra) =>
  ({ id, title: id, specialty: 'x', geomean, patch: `/p/${id}.diff`, touched_files: files, ...extra });

console.log('\n# paths are normalised before they are compared');
ok(shelfFile('./a/b.py') === 'a/b.py' && shelfFile('/a/b.py') === 'a/b.py' && shelfFile(' a/b.py ') === 'a/b.py',
   'a leading ./ or / and stray whitespace do not make two spellings of one file look disjoint');
ok(shelfFiles({ touched_files: ['a.py', './a.py', 'a.py'] }).length === 1,
   'duplicates collapse, so a repeated path cannot inflate a footprint');
ok(shelfFiles({}).length === 0 && shelfFiles({ touched_files: 'a.py' }).length === 0,
   'a missing or non-array touched_files yields no files rather than throwing');

console.log('\n# an UNKNOWN footprint is never treated as orthogonality');
{
  const { shelf } = shelfAdd([], [cand('r1_d0', 1.03, []), cand('r1_d1', 1.02, ['k.py'])], 1, 10);
  ok(shelf.find((e) => e.id === 'r1_d0').footprint === 'unknown',
     'a candidate that reported no touched_files is marked footprint=unknown, not empty-and-safe');
  const pick = shelfEligible(shelf, {}, 5);
  ok(pick.offer.length === 1 && pick.offer[0].id === 'r1_d1',
     'and it is NOT offered, even with room to spare — unknown is not orthogonal');
  ok(pick.unknown.length === 1 && pick.unknown[0].id === 'r1_d0',
     'it is returned in `unknown` so the caller can NAME it; a silent drop looks like an empty shelf');
}

console.log('\n# a later winner ages out the patches it collides with, and only those');
{
  const { shelf } = shelfAdd([], [
    cand('r1_a', 1.05, ['stage1.py']),
    cand('r1_b', 1.04, ['combine.py']),
  ], 1, 10);
  // Round 3 commits a winner that rewrites stage1.py.
  const pick = shelfEligible(shelf, { 3: ['stage1.py'] }, 5);
  ok(pick.eligible.length === 1 && pick.eligible[0].id === 'r1_b',
     'the patch whose file the winner rewrote is withheld');
  ok(pick.stale.length === 1 && pick.stale[0].clash.join() === 'stage1.py',
     'and the withheld one names the exact file that collided, so the log can say why');
  ok(shelfEligible(shelf, { 1: ['stage1.py'] }, 5).eligible.length === 2,
     'an absorption from the SAME round the patch was cut in does not age it — it was cut against that');
  // What a resume does: carried entries are rebased to round 0 (older than anything this wave will
  // commit) and the previous wave's absorptions are collapsed to round 0.5, so they still age.
  const carried = shelf.map((e) => ({ ...e, base_round: 0 }));
  ok(shelfEligible(carried, { 0.5: ['combine.py'] }, 5).eligible.map((e) => e.id).join() === 'r1_a',
     'across a wave boundary, a prior wave\'s absorptions (round 0.5) still age a carried entry (round 0)');
}

console.log('\n# a fusion chain does not age its own halves off the shelf');
{
  // The consumer half of chain D3 is on the shelf, touching combine.py. Round 3 commits the producer
  // half of the SAME chain (also combine.py, by construction). Under a file-set-only test this would
  // stale the consumer and the fusion could never be closed. The chain tag is what prevents it.
  const { shelf } = shelfAdd([], [cand('r2_consumer', 0.97, ['combine.py'], { chain_id: 'D3' })], 2, 10);
  const sameChain = shelfEligible(shelf, { 3: { files: ['combine.py'], chain_id: 'D3' } }, 5);
  ok(sameChain.eligible.length === 1 && sameChain.eligible[0].id === 'r2_consumer',
     'a commit on the SAME chain (D3) that touches the shared file does NOT age the consumer half — ' +
     'the two halves editing one file is the intended relationship, not a conflict');
  // An UNRELATED chain (or a standalone winner) touching the same file still ages it.
  const otherChain = shelfEligible(shelf, { 3: { files: ['combine.py'], chain_id: 'D9' } }, 5);
  ok(otherChain.eligible.length === 0 && otherChain.stale.length === 1,
     'a commit on a DIFFERENT chain touching the same file still stales it — chain-exemption is not a blanket pass');
  const legacy = shelfEligible(shelf, { 3: ['combine.py'] }, 5);
  ok(legacy.eligible.length === 0,
     'a legacy array-form absorption (no chain) still ages the entry, so old callers behave unchanged');
  // An entry with no chain_id is aged by everything, chain-tagged or not.
  const { shelf: plain } = shelfAdd([], [cand('r2_plain', 1.02, ['combine.py'])], 2, 10);
  ok(shelfEligible(plain, { 3: { files: ['combine.py'], chain_id: 'D3' } }, 5).eligible.length === 0,
     'a standalone entry (no chain_id) is aged even by a chain-tagged absorption — exemption needs a tag on BOTH sides');
}

console.log('\n# aging does not depend on anyone reporting what the merge used');
{
  // The integrator took r1_a and said nothing about it. The winner's own file list is what ages the
  // shelf, so the absorbed candidate stops being offered regardless. If this ever inverts, an
  // omitted `best.patches` leaves superseded work on offer, which is the unsound direction.
  const { shelf } = shelfAdd([], [cand('r1_a', 1.05, ['stage1.py'])], 1, 10);
  const pick = shelfEligible(shelf, { 2: ['stage1.py', 'stage2.py'] }, 5);
  ok(pick.offer.length === 0,
     'a candidate whose files the committed winner contains is withheld with no `patches` list needed');
}

console.log('\n# the shelf is bounded, and it evicts the weakest rather than the oldest');
{
  let shelf = [];
  for (let r = 1; r <= 4; r++) {
    const add = shelfAdd(shelf, [cand(`r${r}`, 1.0 + r / 100, [`f${r}.py`])], r, 3);
    shelf = add.shelf;
    if (r === 4) ok(add.evicted.length === 1 && add.evicted[0].id === 'r1',
                    'past the cap the LOWEST verified speedup is evicted, and the eviction is reported');
  }
  ok(shelf.length === 3 && shelf[0].id === 'r4',
     'the shelf stays at the cap, best first');
  ok(shelfEligible(shelf, {}, 2).offer.length === 2,
     'K bounds the offer independently of how much is shelved — one lease per round is the reason');
  ok(shelfEligible(shelf, {}, 0).offer.length === 0 && shelfEligible(shelf, {}, 0).eligible.length === 3,
     'K=0 turns the offer off without hiding that the shelf still holds applicable work');
}

console.log('\n# re-shelving the same id updates it instead of duplicating it');
{
  let { shelf } = shelfAdd([], [cand('r1_a', 1.05, ['x.py'])], 1, 10);
  ({ shelf } = shelfAdd(shelf, [cand('r1_a', 1.11, ['x.py', 'y.py'])], 3, 10));
  ok(shelf.filter((e) => e.id === 'r1_a').length === 1, 'one row per id');
  ok(shelf[0].geomean === 1.11 && shelf[0].base_round === 3,
     'and it is rebased to the round it was re-verified in, or it would be aged against absorptions it already survived');
}

console.log('\n# an absorbed entry is gone for good');
{
  const { shelf } = shelfAdd([], [cand('r1_a', 1.05, ['x.py'])], 1, 10);
  shelf[0].absorbed = true;
  const pick = shelfEligible(shelf, {}, 5);
  ok(pick.offer.length === 0 && pick.eligible.length === 0 && pick.unknown.length === 0,
     'it is not offered, and it is not reported as withheld either — it is in CANONICAL, not pending');
}

console.log(failures === 0
  ? '\nPASS: the shelf offers what still applies and withholds what it cannot vouch for.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
