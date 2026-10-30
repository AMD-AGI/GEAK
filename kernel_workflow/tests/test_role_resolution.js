#!/usr/bin/env node
// Every role name kernel_mega.js passes to roleAgent() must resolve to a file under roles/.
//
// roleAgent() emits "First Read <WORKFLOW_DIR>/roles/<role>.md and follow its instructions for
// PHASE=<phase>". A name with no file behind it does not fail loudly — the agent finds nothing,
// improvises a procedure, and the orchestrator cannot tell the difference. One run ran the claim-
// recovery path against `optimize_engineer`, which has never existed, in three consecutive rounds;
// recovery is the one path whose entire purpose is to NOT improvise (no GPU, no re-measurement),
// so improvising it is the worst possible place for this bug to land.
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const src = fs.readFileSync(path.join(WF, 'kernel_mega.js'), 'utf8');

console.log('\n# 1. every roleAgent() role name has a role file');
{
  // roleAgent(<quoted role>, ...) — the role is always a string literal in this script.
  const roles = new Set();
  for (const m of src.matchAll(/\broleAgent\(\s*['"]([a-z_]+)['"]/g)) roles.add(m[1]);
  ok(roles.size > 0, `found ${roles.size} distinct role name(s) passed to roleAgent()`);
  for (const r of [...roles].sort()) {
    ok(fs.existsSync(path.join(WF, 'roles', 'mega', `${r}.md`)), `roles/${r}.md exists (used by roleAgent)`);
  }
  ok(!roles.has('optimize_engineer'),
     'the never-existent `optimize_engineer` is not referenced (wave-13 regression)');
}

console.log('\n# 2. multi-phase roles document each phase they are called with');
{
  // Scoped to roles roleAgent() calls with MORE THAN ONE distinct phase. A single-phase role file
  // is unambiguous — the whole file *is* the procedure for that phase, so demanding it name the
  // phase would only be demanding a heading. The failure mode being guarded is specific to
  // multi-phase roles: the agent reads a file whose body describes the OTHER phase, finds a
  // complete and confident procedure, and follows it. That is how `engineer`/`recover` (no GPU, no
  // re-measurement) would have been run as an optimization round, and how `benchmark_engineer`/
  // `recover` would have re-measured a baseline that was already on disk.
  const pairs = [...src.matchAll(/\broleAgent\(\s*['"]([a-z_]+)['"]\s*,\s*['"]([a-z_]+)['"]/g)]
    .map((m) => [m[1], m[2]]);
  ok(pairs.length > 0, `found ${pairs.length} (role, phase) pair(s)`);
  const phasesOf = new Map();
  for (const [r, p] of pairs) {
    if (!phasesOf.has(r)) phasesOf.set(r, new Set());
    phasesOf.get(r).add(p);
  }
  const multi = [...phasesOf].filter(([, ps]) => ps.size > 1).sort();
  ok(multi.length > 0, `${multi.length} role(s) are called with more than one phase`);
  for (const [r, ps] of multi) {
    const f = path.join(WF, 'roles', 'mega', `${r}.md`);
    if (!fs.existsSync(f)) continue;   // already failed above
    const body = fs.readFileSync(f, 'utf8');
    for (const p of [...ps].sort()) {
      ok(new RegExp(`PHASE=${p}\\b`).test(body),
         `roles/${r}.md has an explicit PHASE=${p} section (role has ${ps.size} phases)`);
    }
  }
}

console.log('\n# 3. the recover contract still forbids what makes recovery meaningless');
{
  const eng = fs.readFileSync(path.join(WF, 'roles', 'mega', 'engineer.md'), 'utf8').replace(/\s+/g, ' ');
  ok(/PHASE=recover/.test(eng), 'engineer.md has an explicit PHASE=recover section');
  ok(/No GPU command\. No lease\. No re-measurement\./.test(eng),
     'recovery forbids GPU, lease and re-measurement in so many words');
  ok(/exactly as recorded/i.test(eng) && /UNRESOLVED/.test(eng),
     'and requires per_case verbatim, UNRESOLVED guards included');
  ok(/per_case: \[\]/.test(eng),
     'an empty recovery is defined as the correct output, not a failure');

  // The benchmark side of recovery has the same prohibitions and one extra: the positive control
  // already on disk is the control result. Re-running it is not a safer answer, it is a different
  // experiment silently substituted for the one the round is divided by.
  const bench = fs.readFileSync(path.join(WF, 'roles', 'mega', 'benchmark_engineer.md'), 'utf8')
    .replace(/\s+/g, ' ');
  ok(/Do not run any GPU command and do not take a lease/i.test(bench),
     'benchmark_engineer.md forbids GPU and lease in PHASE=recover');
  ok(/claim_complete: true.*is.*the positive-control result/i.test(bench),
     'a completed control on disk IS the control result — not a prompt to re-run it');
  ok(/baseline_per_case: \[\]/.test(bench),
     'an empty baseline recovery is defined as correct, and fabrication as the worst outcome');
}

console.log(
  failures === 0
    ? '\nPASS: every role/phase roleAgent() asks for resolves to a documented file.'
    : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
