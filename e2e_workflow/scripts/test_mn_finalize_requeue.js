#!/usr/bin/env node
// Finalize-gate's disk scan must not re-queue a head candidate whose A/B already completed, on
// multinode; on a single node the scan behaves exactly as on main. The statements are EXTRACTED
// from the shipped source.
//
// Case (dsv4f2c5ax_r1, AgentX, 1P1D): HeadKernel A/B'd ck (58.305) and triton (39.098) and banked
// ck. The scan read the CAND_TAG overlay dirs (they hold no integrate_result.json) as incomplete,
// re-queued both, and the last one banked (triton) replaced ck as the final overlay.
//
// Run:  node e2e_workflow/scripts/test_mn_finalize_requeue.js
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
const line = (needle) => {
  const lines = src.split('\n').filter((l) => l.includes(needle));
  if (lines.length !== 1) { console.error(`FAIL: expected one line with ${needle}, found ${lines.length}`); process.exit(1); }
  return lines[0].trim();
};

const DECL = line('const MN_DONE = new Set();');
const RECORD = line('MN_DONE.add(');
const FILTER = line('known.has(it.short_name)');
const ABC = line('const abc = !!(integ');
// RECORD sits in the HeadKernel candidate loop, the only place CAND_TAG is set.
const loop = src.slice(src.indexOf('for (let ci = 0; ci < headCands.length; ci++)'), src.indexOf('if (bestPick) {'));
if (!loop.includes(RECORD) || !loop.includes(ABC)) {
  console.error('FAIL: MN_DONE is not recorded in the HeadKernel candidate loop'); process.exit(1);
}

// Run the integrate results through RECORD, then the scan output through FILTER; return what is queued.
const run = (MN, integs, scan) => new Function('MN', 'integs', 'scan', `
  ${DECL}
  for (const integ of integs) { ${ABC} ${RECORD} }
  const known = new Set(), queued = [];
  for (const it of scan) {
    ${FILTER}
    known.add(it.short_name); queued.push(it.short_name);
  }
  return queued;`)(MN, integs, scan);

const OV = '/workhome/geak/exp/dsv4f2c5ax_r1/geak/e2e_X/overlay';
const integs = [
  { gate: 'accepted', ab_complete: true, accepted_overlay: `${OV}/cand_gemm_a8w8_blockscale_c0_ck`, e2e_throughput_tok_s: 58.305 },
  { gate: 'accepted', ab_complete: true, accepted_overlay: `${OV}/cand_gemm_a8w8_blockscale_c1_triton`, e2e_throughput_tok_s: 39.098 },
];
const scan = [
  { short_name: 'gemm_a8w8_blockscale_c0_ck', overlay_dir: `${OV}/cand_gemm_a8w8_blockscale_c0_ck` },
  { short_name: 'gemm_a8w8_blockscale_c1_triton', overlay_dir: `${OV}/cand_gemm_a8w8_blockscale_c1_triton` },
];
const MN = { nodes: [{ host: 'a' }], topology: 'pd_disagg' };

console.log('A. multinode');
eq(run(MN, integs, scan), [], 'completed CAND_TAG A/Bs are not re-queued');
eq(run(MN, [{ ...integs[0], gate: 'incomplete' }, { ...integs[1], ab_complete: false }], scan),
  scan.map((s) => s.short_name), 'an A/B that did not complete is still re-queued to finish');
eq(run(MN, [integs[0]], [...scan, { short_name: 'other_op', overlay_dir: `${OV}/cand_other_op` }]),
  ['gemm_a8w8_blockscale_c1_triton', 'other_op'], 'only the completed overlay is skipped');
eq(run(MN, [{ ...integs[0], accepted_overlay: undefined }], scan), scan.map((s) => s.short_name),
  'an A/B that names no overlay skips nothing');

console.log('B. single node is main');
eq(run(null, integs, scan), scan.map((s) => s.short_name), 'without MN nothing is recorded, the scan queues as on main');
eq(RECORD, 'if (MN && abc) MN_DONE.add(integ.accepted_overlay);', 'recording is gated on MN');
eq(FILTER.replace(' || MN_DONE.has(it.overlay_dir)', ''),
  'if (!it || !it.short_name || known.has(it.short_name)) continue;', 'the filter is main\'s plus the MN-only set test');

if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
console.log('\nall finalize re-queue checks passed');
