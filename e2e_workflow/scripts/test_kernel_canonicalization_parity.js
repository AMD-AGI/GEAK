#!/usr/bin/env node
// Regression guard for kernel symbol canonicalization (no GPU, no model needed).
//
// Invariant under test: canonicalDeviceKernel / kernelIdentitiesMatch in e2e_workflow.js agree with
// canonical_kernel_name / kernel_matches in scripts/kernel_selection.py. The JS gate and the Python
// verdict compare the SAME two symbols, so any drift between them lets a kernel pass one side and be
// refused by the other. Both sides are pinned to tests/kernel_symbols.json; the Python half of this
// is TestTheSharedFixtureHoldsOnThisSide in tests/test_kernel_selection.py.
//
// The functions are extracted from the real workflow source rather than reimplemented here -- a copy
// would pass this test while the shipped code drifted, which is the failure it exists to catch.
//
// Run:  node e2e_workflow/scripts/test_kernel_canonicalization_parity.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const WORKFLOW = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');
const FIXTURE = path.join(ROOT, 'e2e_workflow', 'scripts', 'tests', 'kernel_symbols.json');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const src = fs.readFileSync(WORKFLOW, 'utf8');
const start = src.indexOf('const SHORT_NAME_LIMIT');
const end = src.indexOf('function requiredDeviceKernel');
ok(start !== -1 && end !== -1 && start < end, 'canonicalization block located in e2e_workflow.js');
if (failures) process.exit(1);

const build = new Function(
  `${src.slice(start, end)}\nreturn { canonicalDeviceKernel, kernelIdentitiesMatch, SHORT_NAME_LIMIT };`);
const { canonicalDeviceKernel, kernelIdentitiesMatch, SHORT_NAME_LIMIT } = build();

const fixture = JSON.parse(fs.readFileSync(FIXTURE, 'utf8'));

console.log('\n# canonical tokens');
for (const c of fixture.canonical) {
  const got = canonicalDeviceKernel(c.symbol);
  ok(got === c.token, `${JSON.stringify(c.symbol.slice(0, 56))} -> ${JSON.stringify(c.token)}` +
    (got === c.token ? '' : ` (got ${JSON.stringify(got)})`));
}

console.log('\n# match verdicts');
for (const c of fixture.matches) {
  const forward = kernelIdentitiesMatch(c.a, c.b), back = kernelIdentitiesMatch(c.b, c.a);
  ok(forward === c.match && back === c.match,
    `${c.why} (got ${forward}/${back}, want ${c.match})`);
}

console.log('\n# constants shared with the Python side');
ok(SHORT_NAME_LIMIT === 60, 'SHORT_NAME_LIMIT matches parse_profile.SHORT_NAME_LIMIT');

// The fixture is only worth anything while it still carries symbols that real captures produced.
const real = fixture.canonical.filter((c) => c.real).length;
ok(real >= 5, `fixture keeps ${real} symbols taken verbatim from ROCm captures`);

console.log(failures ? `\nFAILED (${failures})` : '\nPASS');
process.exit(failures ? 1 : 0);
