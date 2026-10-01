#!/usr/bin/env node
// Regression guard for the extraction smoke-status contract (no GPU, no model needed).
//
// Every extraction gate in e2e_workflow.js tests `smoke === 'pass'` / `unittest_smoke === 'pass'`.
// When those fields were free strings, agents answered "PASS (exit 0) on GPU 1 ..." and a unittest
// that had PASSED was recorded as "op extraction failed" — the head's bake-off and kernel lane were
// skipped outright (0924/0925 runs: MoE sorting, topk gating, CK blockscale / MoE GEMM, ...).
// The fix pins the fields to enum ['pass','fail'] so StructuredOutput rejects anything else and the
// agent must re-answer. This test pins that: the enum is in the shipped schemas, the strings actually
// seen in those runs are REJECTED by the schema (never silently read as a failed extraction), and the
// role file tells the agent the same contract.
//
// The schemas are EXTRACTED from the shipped source rather than reimplemented here, so this cannot
// pass while e2e_workflow.js drifts.
//
// Run:  node e2e_workflow/scripts/test_smoke_status_contract.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');
const role = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'roles', 'kernel_extractor.md'), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg);
};

// Pull one `const NAME = ...;` statement (single- or multi-line) out of the source.
const grab = (name) => {
  const start = src.indexOf(`const ${name} =`);
  if (start < 0) { console.error(`FAIL: ${name} not found in e2e_workflow.js`); process.exit(1); }
  const end = src.indexOf(';\n', start);
  return src.slice(start, end + 1);
};
const { EXTRACT_OP_SCHEMA, EXTRACT_SCHEMA } = new Function(
  ['obj', 'arrObj', 'arrStr', 'SMOKE_STATUS', 'EXTRACT_OP_SCHEMA', 'EXTRACT_SCHEMA']
    .map(grab).join('\n') + '\nreturn { EXTRACT_OP_SCHEMA, EXTRACT_SCHEMA };')();

// Just enough JSON-schema to judge the gate field: required + type + enum of the top-level props.
const validate = (schema, v) => {
  for (const k of schema.required || []) if (!(k in v)) return false;
  for (const [k, p] of Object.entries(schema.properties)) {
    if (!(k in v)) continue;
    if (p.type === 'string' && typeof v[k] !== 'string') return false;
    if (p.enum && !p.enum.includes(v[k])) return false;
  }
  return true;
};

// Verbatim prefixes of what agents returned in the runs that lost these heads.
const REAL_WORLD = [
  'pass. gpu_lock.sh 0 python3 unittest.py exited 0 with RESULT=PASS',
  'PASS (exit 0) on GPU 1',
  'PASS (rc=0). ',
  'PASS (exit 0). Correctness ran under _cand_overlay',
  'PASS (exit 0). Command: cd $TASK && ...',
  'PASS (exit 0, GPU 0): the full oracle',
  'PASS (rc=0, GPU 1). All 17 eager oracle cases',
];
const OTHER_BAD = ['PASS', 'Pass', ' pass', 'pass ', 'passed', 'FAIL', 'failed', 'ok', ''];

for (const [name, schema, field, base] of [
  ['EXTRACT_OP_SCHEMA', EXTRACT_OP_SCHEMA, 'smoke', { op_kind: 'gemm', task_dir: '/t' }],
  ['EXTRACT_SCHEMA', EXTRACT_SCHEMA, 'unittest_smoke', { editable: true, task_dir: '/t' }],
]) {
  console.log(`${name}.${field}`);
  const p = schema.properties[field];
  ok(p && JSON.stringify(p.enum) === JSON.stringify(['pass', 'fail']), `${field} is enum ['pass','fail']`);
  ok((schema.required || []).includes(field), `${field} is required`);
  ok(schema.properties.smoke_detail && schema.properties.smoke_detail.type === 'string',
    'smoke_detail exists as the home for free text');
  ok(validate(schema, { ...base, [field]: 'pass' }), '"pass" accepted');
  ok(validate(schema, { ...base, [field]: 'fail' }), '"fail" accepted');
  ok(validate(schema, { ...base, [field]: 'pass', smoke_detail: REAL_WORLD[1] }),
    '"pass" + prose in smoke_detail accepted');
  for (const s of [...REAL_WORLD, ...OTHER_BAD]) {
    ok(!validate(schema, { ...base, [field]: s }), `rejected (agent must re-answer): ${JSON.stringify(s)}`);
  }
}

// The gates compare against exactly the value the enum allows — so a schema-valid "pass" can never
// again be read as a failed extraction.
console.log('gates');
const gates = src.match(/(?:smoke|unittest_smoke) [!=]== '[^']*'/g) || [];
ok(gates.length >= 5, `found ${gates.length} smoke gates`);
ok(gates.every((g) => /'pass'$/.test(g)), 'every gate compares against lowercase \'pass\'');

// An extraction failure's reason must survive the enum: the status field can only say "fail", so the
// prose lives in smoke_detail. Every failure site reads it through extractFailWhy (review of #491: a
// gpu_lock refusal only in smoke_detail, with unrelated notes, collapsed to a bare "fail").
console.log('failure reasons');
const helperStart = src.indexOf('const extractFailWhy = (');
ok(helperStart >= 0, 'extractFailWhy is defined');
const extractFailWhy = new Function(
  src.slice(helperStart, src.indexOf(';\n', helperStart) + 1) + '\nreturn extractFailWhy;')();
const GPU_LOCK = "Exit code 1: GPU 2 had foreign work running above the lock's busy/VRAM threshold";
ok(extractFailWhy({ unittest_smoke: 'fail', smoke_detail: GPU_LOCK,
  notes: 'The task uses synthesized fixed-seed inputs and has no reference_io.pt.' }).startsWith(GPU_LOCK),
  'agent fail: the smoke_detail cause leads even when notes is unrelated (gpu_lock case)');
// The shape extractWithBaseline returns when it overrides an agent "pass".
const forced = { smoke: 'fail', unittest_smoke: 'fail', selection_failed: true,
  smoke_detail: 'workflow forced fail: kernel selection failed: deeper_live_candidate_exists',
  notes: 'kernel selection failed: deeper_live_candidate_exists — agent notes' };
ok(extractFailWhy(forced).startsWith('workflow forced fail:'), 'forced fail: the workflow reason leads');
ok(extractFailWhy({ smoke: 'fail', notes: 'capture OOM' }) === 'capture OOM', 'notes alone is kept');
ok(extractFailWhy({ smoke: 'fail' }) === 'fail' && extractFailWhy({ unittest_smoke: 'fail' }) === 'fail',
  'bare status when there is no prose');
ok(extractFailWhy(null) === 'none', '"none" when there is no result');
const forcedReturns = src.match(/smoke: 'fail', unittest_smoke: 'fail'[^}]*?smoke_detail: `workflow forced fail: /g) || [];
ok(forcedReturns.length === 2, `both workflow-forced fails overwrite smoke_detail (${forcedReturns.length}/2)`);
ok(!/notes \|\| (?:p\.)?ext\.(?:unittest_)?smoke\b/.test(src), 'no failure reason reads only notes || status');
const sites = src.match(/extractFailWhy\((?:p\.)?ext\)/g) || [];
ok(sites.length === 4, `all four extraction-failure sites use extractFailWhy (${sites.length}/4)`);

console.log('role file');
ok(/Smoke status contract/.test(role), 'kernel_extractor.md states the smoke status contract');
ok(/lowercase string `"pass"` or\s+`"fail"`/.test(role), 'contract names exactly lowercase "pass"/"fail"');
ok(!/"(unittest_)?smoke": "pass\|fail"/.test(role), 'no ambiguous "pass|fail" placeholder left in the output examples');
ok((role.match(/"smoke_detail":/g) || []).length === 2, 'both output examples show smoke_detail');
const contract = role.slice(role.indexOf('Smoke status contract'), role.indexOf('## PHASE=extract'));
ok(/exit 3/.test(contract) && /UT_HARNESS_INCOMPLETE/.test(contract) && /regenerate the UT first/.test(contract),
  'contract defers exit 3 / UT_HARNESS_INCOMPLETE to the regenerate rule');
ok(!/anything else is `"fail"`/.test(contract), 'contract no longer says "anything else is fail"');
ok(!/reason="harness_incomplete_unrecoverable"/.test(role) && /smoke_detail` with `harness_incomplete_unrecoverable`/.test(role),
  'exit-3 give-up reason goes to smoke_detail, which the workflow reads');

if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
console.log('\nall smoke status contract checks passed');
