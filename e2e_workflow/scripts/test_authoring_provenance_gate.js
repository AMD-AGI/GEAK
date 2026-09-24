#!/usr/bin/env node
// Deterministic regression test for the authoring promotion provenance hard gate.
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');
const start = src.indexOf('const SHA256_RX =');
const end = src.indexOf('// Fold the env/flags', start);
if (start < 0 || end < 0) throw new Error('provenance gate helpers not found');
const helpers = src.slice(start, end);
const gate = new Function(
  `${helpers}\nreturn { provenanceAllowsPromotion, integAccepted };`
)();

let failures = 0;
function ok(value, message) {
  if (!value) {
    failures++;
    console.error('FAIL:', message);
  } else {
    console.log('ok:', message);
  }
}

const shaA = 'a'.repeat(64);
const shaB = 'b'.repeat(64);
const bind = { kind: 'rebind', target: 'pkg.live:op', file: 'kernel_src/op.py' };
const kr = {
  winner_kind: 'patch',
  code_patch: '/run/final.patch',
  reference_io_sha256: shaB,
  target_callable: 'pkg.live:op',
  candidate_bind: bind,
};
const accepted = {
  gate: 'accepted',
  parity_kind: 'byte_exact',
  accepted_overlay: '/run/overlay/candidate',
  accepted_env: 'GEAK_FIXTURE_ENABLE=1',
  accepted_flags: '',
  engagement_evidence: 'GEAK_FIXTURE_HITS=12',
  provenance_ok: true,
  provenance: {
    authored_patch_sha256: shaA,
    oracle_sha256: shaB,
    target_seam: 'pkg.live:op',
    candidate_binding: bind,
  },
};

// integAccepted refers to this global helper in the shipped source.
global.isImplausibleSpeedup = () => false;
const acceptedWithGuard = new Function(
  'isImplausibleSpeedup',
  `${helpers}\nreturn integAccepted;`
)(() => false);

ok(gate.provenanceAllowsPromotion(accepted, kr), 'complete authored provenance passes');
ok(acceptedWithGuard(accepted, 20, 1.2, kr), 'complete authored candidate can be promoted');
const reordered = JSON.parse(JSON.stringify(accepted));
reordered.provenance.candidate_binding = {
  file: 'kernel_src/op.py', target: 'pkg.live:op', kind: 'rebind',
};
ok(gate.provenanceAllowsPromotion(reordered, kr),
  'candidate binding comparison is key-order independent');

for (const [name, mutate] of [
  ['provenance_ok', x => { x.provenance_ok = false; }],
  ['overlay', x => { x.accepted_overlay = ''; }],
  ['env field', x => { delete x.accepted_env; }],
  ['flags field', x => { delete x.accepted_flags; }],
  ['engagement', x => { x.engagement_evidence = ''; }],
  ['patch digest', x => { x.provenance.authored_patch_sha256 = 'not-a-sha'; }],
  ['oracle digest', x => { x.provenance.oracle_sha256 = shaA; }],
  ['target seam', x => { x.provenance.target_seam = 'pkg.other:op'; }],
  ['candidate binding', x => { x.provenance.candidate_binding = { kind: 'module' }; }],
]) {
  const candidate = JSON.parse(JSON.stringify(accepted));
  mutate(candidate);
  ok(!gate.provenanceAllowsPromotion(candidate, kr), `rejects missing/mismatched ${name}`);
}

ok(gate.provenanceAllowsPromotion(
  { gate: 'accepted' }, { winner_kind: 'env' }),
  'non-authoring config promotion remains backward compatible');
ok(gate.provenanceAllowsPromotion(
  { gate: 'accepted' }, { winner_kind: 'patch', code_patch: '/stored.patch',
    provenance: 'knowledge_base_replay' }),
  'foreign KB replay remains governed by its fresh parity gate');

// Every shipped acceptance call must pass the exact KERNEL_RESULT it is about to bank.
const calls = [...src.matchAll(/integAccepted\(([^;\n]+)\)/g)].map(m => m[1]);
ok(calls.length >= 7, 'found all authoring acceptance sites');
ok(calls.every(call => call.split(',').length >= 4),
  'every acceptance site supplies candidate provenance context');

console.log(failures ? `FAILED (${failures})` : 'PASS');
process.exit(failures ? 1 : 0);
