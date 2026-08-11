#!/usr/bin/env node
/**
 * test_learned_index.js — guards the generated learned/INDEX.md contract.
 *
 * Pure node (fs/os/path) on a throwaway tmp dir: no GPU, no agent, no repo mutation. Asserts that the
 * index is DERIVED from card frontmatter (so a lost/racing append can never drop a card), that grouping
 * and ordering are deterministic, that a non-active card leaves the index, and that --check is honest.
 */
'use strict';
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { main, render, collect } = require('./build_learned_index.js');

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'learned-'));
const card = (name, fm, body = '- lever: x\n') =>
  fs.writeFileSync(path.join(dir, `${name}.md`), `---\n${fm}\n---\n# ${name}\n${body}`);

card('split-k-gemm-gfx942', [
  'name: split-k-gemm-gfx942',
  'description: split-K + LDS re-tiling on skinny-M GEMM → 1.34x isolated',
  'keywords: [split-k, lds-tiling, skinny-m]',
  'kernels: [_gemm_a8w8_kernel]',
  'platforms: [gfx942]', 'kernel_class: dense_gemm', 'regime: decode',
  'confidence: ★★★', 'lifecycle: active',
].join('\n'));
card('launch-overhead-gfx942', [
  'name: launch-overhead-gfx942',
  'description: fuse the wrapper prologue → +12% on launch-bound decode',
  'keywords: [host, launch-overhead]',
  'platforms: [gfx942]', 'kernel_class: dense_gemm', 'regime: decode',
  'confidence: ★★', 'lifecycle: active',
].join('\n'));
card('method-parity-first', [
  'name: method-parity-first',
  'description: re-run oracle parity after a hand-merged patch',
  'keywords: [parity, integrate]',
  'kernel_class: method', 'confidence: ★★', 'lifecycle: active',
].join('\n'));
card('retired-lever', [
  'name: retired-lever', 'description: refuted', 'kernel_class: dense_gemm',
  'confidence: ★', 'lifecycle: archived',
].join('\n'));
// Not a card: no frontmatter at all. Must be skipped, not guessed at.
fs.writeFileSync(path.join(dir, 'stray-notes.md'), '# just some notes\n');

assert.strictEqual(main([dir]), 0);
const idx = fs.readFileSync(path.join(dir, 'INDEX.md'), 'utf8');

assert.ok(idx.includes('GENERATED FILE'), 'index announces it is generated');
assert.ok(idx.includes('split-K + LDS re-tiling'), 'card description reaches the index');
assert.ok(idx.includes('kernels: _gemm_a8w8_kernel'), 'kernel names are indexed');
assert.ok(idx.includes('kw: split-k, lds-tiling, skinny-m'), 'keywords are indexed');
assert.ok(idx.includes('[gfx942 · decode]'), 'platform + regime scope is rendered');
assert.ok(!idx.includes('retired-lever'), 'a non-active card leaves the index');
assert.ok(!idx.includes('stray-notes'), 'a file without frontmatter is not indexed');
console.log('  ok: index is derived from card discovery frontmatter');

// Ordering: op groups before `method`; inside a group, higher confidence first.
const gGemm = idx.indexOf('## dense_gemm'), gMethod = idx.indexOf('## method');
assert.ok(gGemm !== -1 && gMethod !== -1 && gGemm < gMethod, 'method group sorts last');
assert.ok(idx.indexOf('split-K + LDS') < idx.indexOf('fuse the wrapper prologue'),
  '★★★ sorts above ★★ within a group');
console.log('  ok: grouping + confidence ordering are deterministic');

// The race fix: a card written by another lane appears on the next regen without anyone appending.
assert.strictEqual(main([dir, '--check']), 0, 'freshly built index is up to date');
card('concurrent-lane-card', [
  'name: concurrent-lane-card', 'description: written by a parallel lane',
  'platforms: [gfx950]', 'kernel_class: attention_decode', 'confidence: ★★', 'lifecycle: active',
].join('\n'));
assert.strictEqual(main([dir, '--check']), 1, '--check reports a stale index');
assert.strictEqual(main([dir]), 0);
assert.ok(fs.readFileSync(path.join(dir, 'INDEX.md'), 'utf8').includes('written by a parallel lane'),
  'the other lane\'s card is published by a plain regen — no append, nothing lost');
console.log('  ok: regen publishes every card on disk (no append race)');

// Keyword hygiene: mechanical spelling differences are normalized away, and the ones that survive are
// REPORTED so a curator merges them — never silently rewritten.
card('drifty-card', [
  'name: drifty-card', 'description: written with sloppy keywords',
  'keywords: [Split_K, LDS Tiling, splitk]',
  'platforms: [gfx942]', 'kernel_class: dense_gemm', 'confidence: ★★', 'lifecycle: active',
].join('\n'));
main([dir]);
const drift = fs.readFileSync(path.join(dir, 'INDEX.md'), 'utf8');
assert.ok(drift.includes('kw: split-k, lds-tiling, splitk'),
  'Split_K/LDS Tiling normalize to split-k/lds-tiling (case, underscore, space)');
assert.ok(drift.includes('## keyword vocabulary'), 'the controlled vocabulary is published');
assert.ok(/split-k\(2\)/.test(drift), 'vocabulary counts show which term is already established');
assert.ok(drift.includes('Near-duplicate keywords') && /split-k \/ splitk|splitk \/ split-k/.test(drift),
  'a synonym that survives normalization is flagged, not merged behind the curator\'s back');
console.log('  ok: keywords normalize; residual drift is reported for a human/curator call');
fs.unlinkSync(path.join(dir, 'drifty-card.md'));
main([dir]);

// Idempotent: same cards in, byte-identical index out.
const a = fs.readFileSync(path.join(dir, 'INDEX.md'), 'utf8');
main([dir]);
assert.strictEqual(fs.readFileSync(path.join(dir, 'INDEX.md'), 'utf8'), a, 'regen is idempotent');
console.log('  ok: regen is idempotent');

// The cap is reported, not silently exceeded.
for (let i = 0; i < 45; i++) {
  card(`bulk-${String(i).padStart(2, '0')}`, [
    `name: bulk-${i}`, `description: bulk card ${i}`, 'kernel_class: dense_gemm',
    'confidence: ★', 'lifecycle: active',
  ].join('\n'));
}
main([dir]);
assert.ok(fs.readFileSync(path.join(dir, 'INDEX.md'), 'utf8').includes('Over the 40-card cap'),
  'overflow is flagged in the file itself');
console.log('  ok: the ≤40-card cap is surfaced, not silently blown');

fs.rmSync(dir, { recursive: true, force: true });
console.log('\nPASS: learned/INDEX.md is generated from card frontmatter, ordered, capped, race-free.');
