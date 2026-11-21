// Unit tests for the MAP-Elites archive + crossover (pure).
// Run on the rig (node not required locally):  node tests/archive.test.mjs

import assert from 'node:assert/strict';
import {
  cellKey, buildArchive, selectElites, crossoverPairs, archiveStats,
  selectMergeGroup, mergeImproves,
} from '../lib/archive.mjs';

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`ok   - ${name}`); }
  catch (e) { console.error(`FAIL - ${name}\n      ${e.message}`); process.exitCode = 1; }
}

const lineage = [
  { id: 'v0', ref: 'avo-v0', speedup: 1.0, strategy: 'baseline', regime: 'unknown' },
  { id: 'v1', ref: 'avo-v1', speedup: 1.4, strategy: 'tiling', regime: 'memory-bound' },
  { id: 'v2', ref: 'avo-v2', speedup: 1.9, strategy: 'tiling', regime: 'memory-bound' }, // best of cell tiling|memory
  { id: 'v3', ref: 'avo-v3', speedup: 1.6, strategy: 'mfma', regime: 'compute-bound' },
  { id: 'v4', ref: 'avo-v4', speedup: 1.7, strategy: 'occupancy', regime: 'basement' },
];

// --- cellKey -------------------------------------------------------------
test('cellKey defaults to strategy|regime (normalized)', () => {
  assert.equal(cellKey({ strategy: 'Tiling', regime: 'Memory-Bound' }), 'tiling|memory-bound');
  assert.equal(cellKey({}), '?|?');
});

// --- buildArchive --------------------------------------------------------
test('archive keeps the best node per cell', () => {
  const a = buildArchive(lineage);
  assert.equal(a.get('tiling|memory-bound').ref, 'avo-v2');   // v2 (1.9) beats v1 (1.4)
  // distinct cells: baseline|unknown, tiling|memory-bound, mfma|compute-bound, occupancy|basement
  assert.equal(a.size, 4);
});

// --- selectElites --------------------------------------------------------
test('selectElites: best first, one per cell, baseline excluded', () => {
  const refs = selectElites(lineage, { k: 4 });
  assert.equal(refs[0], 'avo-v2');                 // global best first
  assert.ok(!refs.includes('avo-v0'));             // baseline excluded
  assert.ok(!refs.includes('avo-v1'));             // dominated within its cell by v2
  assert.deepEqual(refs, ['avo-v2', 'avo-v4', 'avo-v3']); // ranked by speedup: 1.9,1.7,1.6
});
test('selectElites k=1 -> just best', () => {
  assert.deepEqual(selectElites(lineage, { k: 1 }), ['avo-v2']);
});
test('selectElites empty -> []', () => {
  assert.deepEqual(selectElites([], { k: 3 }), []);
});

// --- crossoverPairs ------------------------------------------------------
test('crossover pairs come from DIFFERENT regimes, ranked by combined speedup', () => {
  const pairs = crossoverPairs(lineage, { limit: 3 });
  // regime reps: memory=v2(1.9), compute=v3(1.6), basement=v4(1.7)
  // best combined: v2+v4=3.6, then v2+v3=3.5, then v4+v3=3.3
  assert.equal(pairs.length, 3);
  assert.deepEqual([pairs[0].a.ref, pairs[0].b.ref].sort(), ['avo-v2', 'avo-v4']);
  assert.ok(pairs[0].regimes[0] !== pairs[0].regimes[1]);
});
test('crossover needs >=2 regimes', () => {
  const single = [{ id: 'v1', ref: 'avo-v1', speedup: 1.4, regime: 'memory-bound' }];
  assert.deepEqual(crossoverPairs(single), []);
});

// --- archiveStats --------------------------------------------------------
test('archiveStats reports cells/regimes/total', () => {
  const s = archiveStats(lineage);
  assert.equal(s.total, 5);
  assert.equal(s.regimes, 4);   // unknown, memory-bound, compute-bound, basement
});

// --- selectMergeGroup (Integrator trigger) ------------------------------
test('selectMergeGroup: same base + different regimes, best-per-regime', () => {
  const passers = [
    { ref: 'a', startRef: 'avo-v2', regime: 'memory-bound', speedup: 1.8 },
    { ref: 'b', startRef: 'avo-v2', regime: 'memory-bound', speedup: 1.6 }, // dominated in cell
    { ref: 'c', startRef: 'avo-v2', regime: 'compute-bound', speedup: 1.7 },
  ];
  const g = selectMergeGroup(passers);
  assert.equal(g.base, 'avo-v2');
  assert.equal(g.members.length, 2);                       // one per regime, best kept
  assert.deepEqual(g.pair.map(p => p.ref).sort(), ['a', 'c']);
});
test('selectMergeGroup: null when only one regime', () => {
  assert.equal(selectMergeGroup([
    { startRef: 'avo-v2', regime: 'memory-bound', speedup: 1.8 },
    { startRef: 'avo-v2', regime: 'memory-bound', speedup: 1.6 },
  ]), null);
});
test('selectMergeGroup: null when complementary winners are on DIFFERENT bases', () => {
  assert.equal(selectMergeGroup([
    { startRef: 'avo-v2', regime: 'memory-bound', speedup: 1.8 },
    { startRef: 'avo-v3', regime: 'compute-bound', speedup: 1.7 },
  ]), null);                                                // cannot stack across bases
});
test('selectMergeGroup: picks the highest combined same-base group', () => {
  const g = selectMergeGroup([
    { ref: 'a', startRef: 'b1', regime: 'memory-bound', speedup: 1.2 },
    { ref: 'b', startRef: 'b1', regime: 'compute-bound', speedup: 1.2 },   // combined 2.4
    { ref: 'c', startRef: 'b2', regime: 'memory-bound', speedup: 1.9 },
    { ref: 'd', startRef: 'b2', regime: 'basement', speedup: 1.8 },        // combined 3.7
  ]);
  assert.equal(g.base, 'b2');
});
test('selectMergeGroup: ignores unknown-regime passers', () => {
  assert.equal(selectMergeGroup([
    { startRef: 'avo-v2', regime: 'unknown', speedup: 1.8 },
    { startRef: 'avo-v2', speedup: 1.7 },
  ]), null);
});

// --- mergeImproves (merge must beat the best winner it combined) ---------
test('mergeImproves: true only when merged beats the best member', () => {
  const members = [{ speedup: 1.8 }, { speedup: 1.7 }];
  assert.equal(mergeImproves(1.95, members), true);
  assert.equal(mergeImproves(1.80, members), false);   // ties the best -> not an improvement
  assert.equal(mergeImproves(1.50, members), false);
});
test('mergeImproves: margin + invalid inputs', () => {
  assert.equal(mergeImproves(1.81, [{ speedup: 1.8 }], { margin: 0.02 }), false); // < 1.8*1.02
  assert.equal(mergeImproves(0, [{ speedup: 1.8 }]), false);
  assert.equal(mergeImproves(2.0, []), true);          // no bar -> any positive improves
});

console.log(`\n${passed} passed${process.exitCode ? ' (with failures)' : ''}`);
