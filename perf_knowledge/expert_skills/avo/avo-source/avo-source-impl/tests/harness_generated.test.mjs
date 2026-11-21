import assert from 'node:assert/strict';
import { isHarnessGeneratedPath } from '../lib/harness_generated.mjs';

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`ok   - ${name}`); }
  catch (e) { console.error(`FAIL - ${name}\n      ${e.message}`); process.exitCode = 1; }
}

test('overlay trees and priv overlay variants', () => {
  assert.equal(isHarnessGeneratedPath('_cand_overlay'), true);
  assert.equal(isHarnessGeneratedPath('_cand_overlay/sitecustomize.py'), true);
  assert.equal(isHarnessGeneratedPath('_priv_overlay_abc/x'), true);
});

test('GLM harness append log is generated', () => {
  assert.equal(isHarnessGeneratedPath('_runs.jsonl'), true);
  assert.equal(isHarnessGeneratedPath('./_runs.jsonl'), true);
});

test('kernel source is not generated', () => {
  assert.equal(isHarnessGeneratedPath('kernel_src/sparse_mla_seam.py'), false);
  assert.equal(isHarnessGeneratedPath('unittest.py'), false);
});

if (!process.exitCode) console.log(`\n${passed} tests passed`);
