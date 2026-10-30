#!/usr/bin/env node
// Regression guard for the HeadKernel selection gate (kernelSelectionVerified) on the q8pd9 case:
// the Extractor returned seam_candidates keyed `target` with no device_kernels, and the live target
// was aiter's backward-compat alias. No GPU, no model, no agent needed.
//
// The helpers are EXTRACTED from the shipped source rather than reimplemented here, so this cannot
// pass while e2e_workflow.js drifts.
//
// Run:  node e2e_workflow/scripts/test_selection_gate.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const eq = (got, want, msg) => {
  const a = JSON.stringify(got), b = JSON.stringify(want);
  if (a !== b) { console.error('  FAIL:', msg, '\n    got: ', a, '\n    want:', b); failures++; }
  else console.log('  ok:', msg);
};
const slice = (from, to) => {
  const s = src.indexOf(from), e = src.indexOf(to, s);
  if (s < 0 || e < 0) { console.error(`FAIL: block ${from} not found in e2e_workflow.js`); process.exit(1); }
  return src.slice(s, e);
};
const gate = new Function(
  `${slice('const CALLABLE_SPEC_RX =', 'const PRE_FLAGGED_HEADS')}\n` +
  'return { kernelSelectionVerified, selectionCandidatesForHead, candidateSpec };')();

const KERNEL = '_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_' +
  'BLOCK_SIZE_K_128_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_24_cache_modifier_CG';
const SHIM = 'aiter.ops.triton.gemm_a8w8_blockscale:gemm_a8w8_blockscale';
const CANON = 'aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale:gemm_a8w8_blockscale';
const OPSEAM = 'sglang.srt.layers.quantization.fp8_utils:triton_gemm_a8w8_blockscale';
const head = { entity_kind: 'gpu_kernel', device_kernel: KERNEL, op_kind: 'gemm', seam_candidates: [] };
const verdict = {
  contract: 'kernel_selection', ok: true, target_callable: SHIM, device_kernel: KERNEL,
  total_calls_observed: 185472, target_marker_calls: 256, matched_kernel_calls: 64,
  candidate_targets_tested: [CANON, SHIM, OPSEAM], live_candidate_targets: [SHIM, OPSEAM],
  deeper_live_candidates: [], installed_but_never_live_candidates: [CANON], deepest_verified: true,
};
const ext = (over = {}) => ({
  target_callable: SHIM, device_kernel: KERNEL, selection_validation: { ...verdict },
  seam_candidates: [
    { target: OPSEAM, role: 'op_seam', depth: 1, live: true },
    { target: SHIM, role: 'inner_launcher', depth: 2, live: true, selected: true },
    { target: CANON, role: 'inner_launcher', depth: 2, live: false },
  ],
  ...over,
});

console.log('A. candidate spec keys');
eq(gate.candidateSpec({ target: SHIM }), SHIM, '`target` is read as the spec');
eq(gate.candidateSpec({ target_callable: CANON, target: SHIM }), CANON, 'target_callable still wins');

console.log('B. q8pd9 extractor return passes the gate');
{
  const r = gate.kernelSelectionVerified(head, ext());
  eq(r.ok, true, `shim alias with runtime-live evidence is deployable (${r.why})`);
}

console.log('C. the relaxation is tied to runtime evidence');
{
  const noLive = ext({ selection_validation: { ...verdict, live_candidate_targets: [OPSEAM] } });
  eq(gate.kernelSelectionVerified(head, noLive).ok, false, 'not live in the verdict -> still refused');
  const otherKernel = ext({ selection_validation: { ...verdict, device_kernel: '_some_other_kernel' } });
  eq(gate.kernelSelectionVerified(head, otherKernel).ok, false, 'verdict kernel differs -> still refused');
  const noMatch = ext({ selection_validation: { ...verdict, matched_kernel_calls: 0 } });
  eq(gate.kernelSelectionVerified(head, noMatch).ok, false, 'no matched kernel calls -> still refused');
  const listed = ext();
  listed.seam_candidates[1].device_kernels = ['_unrelated_kernel'];
  eq(gate.selectionCandidatesForHead(head, listed).map(gate.candidateSpec).includes(SHIM), false,
    'a candidate that LISTS non-matching kernels is not rescued by the verdict');
}

(async () => {
console.log('D. the gate reads the verdict file, not the agent\'s abridged echo of it');
{
  // The Workflow sandbox has no require: the file is read by an agent running the command the
  // workflow hands it. This reader runs that command for real; `reply` can override its answer.
  const { execSync } = require('child_process');
  let reply = null;
  const prompts = [];
  const safeAgent = async (prompt) => {
    prompts.push(prompt);
    if (reply) return reply;
    const cmd = prompt.split('```bash\n')[1].split('\n```')[0];
    try { return { stdout: execSync(cmd, { encoding: 'utf8', shell: '/bin/bash', stdio: 'pipe' }) }; }
    catch (e) { return { stdout: `${e.stdout || ''}${e.stderr || ''}` }; }
  };
  const obj = (properties, required) => ({ type: 'object', properties, required });
  const withDiskSelection = new Function('require', 'log', 'safeAgent', 'obj',
    `${slice('const SELECTION_FIELDS', 'async function extractWithBaseline')}\n` +
    'return withDiskSelection;')(undefined, () => {}, safeAgent, obj);
  const os = require('os');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'geak_sel_'));
  // What the q8pd10 extractor returned: ok and counts only, no target_callable/device_kernel.
  const abridged = { contract: 'kernel_selection', ok: true, deepest_verified: true, target_marker_calls: 256,
    matched_kernel_calls: 64, failed: [], candidate_targets_tested: [OPSEAM, SHIM, CANON] };
  const returned = ext({ task_dir: dir, selection_validation: abridged });
  eq(gate.kernelSelectionVerified(head, returned).ok, false, 'abridged echo alone is refused');
  eq(await withDiskSelection(returned), returned, 'no verdict file -> the return is left as is');
  // The real verdict carries far more than the gate reads (per-call traces); the reader projects it.
  fs.writeFileSync(path.join(dir, 'selection_validation.json'),
    JSON.stringify({ ...verdict, calls: Array(2000).fill({ trace: 'x'.repeat(64) }) }));
  const read = await withDiskSelection(returned);
  const r = gate.kernelSelectionVerified(head, read);
  eq(r.ok, true, `verdict file on disk passes the gate (${r.why})`);
  eq('calls' in read.selection_validation, false, 'only the fields the gate reads come back');
  fs.writeFileSync(path.join(dir, 'selection_validation.json'), JSON.stringify({ ...verdict, ok: false,
    failed: ['candidate_marker_not_installed'] }));
  eq(gate.kernelSelectionVerified(head, await withDiskSelection(ext({ task_dir: dir }))).ok, false,
    'a failing verdict file overrides an agent that claims ok');
  fs.writeFileSync(path.join(dir, 'selection_validation.json'), '{truncated');
  eq((await withDiskSelection(ext({ task_dir: dir }))).selection_validation.failed,
    ['selection_validation.json unreadable'], 'a corrupt verdict file fails');
  reply = { stdout: 'I could not find the command.' };
  eq(gate.kernelSelectionVerified(head, await withDiskSelection(ext({ task_dir: dir }))).ok, false,
    'a reader that answers prose instead of the output fails');
  eq(prompts.every((p) => p.includes(`${dir}/selection_validation.json`)), true, 'the reader is given the task dir file');
  fs.rmSync(dir, { recursive: true, force: true });
}

console.log('E. the extract sites wait for the disk read');
const calls = (src.match(/withDiskSelection\(/g) || []).length - 1;
eq((src.match(/= await withDiskSelection\(await safeAgent\(/g) || []).length, calls,
  `every call site (${calls}) awaits it`);

if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
console.log('\nall selection-gate checks passed');
})();
