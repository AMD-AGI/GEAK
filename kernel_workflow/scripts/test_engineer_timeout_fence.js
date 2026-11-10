#!/usr/bin/env node
// Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Regression guard: an engineer whose return never came back is stopped, and the lane merges exactly
// the patch its verify measured (no GPU, no model needed).
//
// agentT() resolves a call that exceeds AGENT_TIMEOUT_MS to null, but agent() cannot be cancelled, so
// the engineer kept running beside its own verify. Verify and the later integrate/commit each re-read
// the PATH ${out_dir}/best_patch.diff; when the engineer rewrote it in between, the round verified one
// patch and merged another (observed: a patch rewritten 5 min after its verify finished and 2 min
// before the integrator read it). Under test, on the REAL Verify-stage callback of kernel_lane.js:
//   - verify is handed FROZEN_PATCH, and the stage returns that frozen path as the candidate's patch
//     (which is what integrate and commit read), never best_patch.diff;
//   - STOP_ENGINEER is passed exactly when the engineer's return is missing, and not for an engineer
//     that returned (even with status "failed": it has finished);
//   - a trustworthy below-floor return still skips verify;
//   - the role files carry the matching instructions.
//
// Run:  node kernel_workflow/scripts/test_engineer_timeout_fence.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const KW = path.join(ROOT, 'kernel_workflow');
const src = fs.readFileSync(path.join(KW, 'kernel_lane.js'), 'utf8');

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail != null ? `-- ${detail}` : ''); failures++; }
  else console.log('  ok:', msg);
};

// Pull the Verify-stage callback (the 2nd pipeline stage of the Optimize -> Verify round) out of the
// source, so the test exercises the code that ships rather than a restatement of it.
function verifyStage() {
  const start = src.indexOf('    (prev) => {\n      const { d, eng } = prev;\n');
  const end = src.indexOf('\n    }\n  );\n\n  const clean = results.filter(Boolean);', start);
  if (start < 0 || end < 0) throw new Error('could not locate the Verify stage in kernel_lane.js');
  const body = src.slice(start, end + '\n    }'.length);
  const calls = [];
  const env = {
    round: 3,
    CANDIDATE_FLOOR: 1.0,
    primSpeedup: (r) => (r && typeof r.speedup_geomean === 'number' ? r.speedup_geomean : 0),
    log: () => {},
    roleAgent: (role, phase, intro, inputs) => ({ role, phase, inputs }),
    agentT: async (p, o) => { calls.push({ prompt: p, opts: o }); return { status: 'verified' }; },
    CANONICAL: '/eval/workspace', EVAL_DIR: '/eval', WORKFLOW_DIR: '/wf', COMMANDMENT: '/eval/COMMANDMENT.md',
    BASELINE_PER_CASE: [], HARNESS_ADDENDUM: '', REQUIRE_GRAPH_CAPTURE: false, VERIFY_SCHEMA: {},
  };
  const names = Object.keys(env);
  const fn = new Function(...names, `return (${body});`)(...names.map((k) => env[k]));
  return { fn, calls };
}

(async () => {
  const d = { id: 'r3_d0', out_dir: '/eval/round_3/engineer_0', gpu_id: 'pool' };
  const bestPatch = `${d.out_dir}/best_patch.diff`;
  const frozen = `${d.out_dir}/verify/candidate.diff`;

  console.log('engineer return missing (agentT timed out / gave up):');
  {
    const { fn, calls } = verifyStage();
    const r = await fn({ d, eng: null });
    const inp = calls.length === 1 ? calls[0].prompt.inputs : {};
    ok(calls.length === 1, 'verify is dispatched');
    ok(inp.PATCH === bestPatch, 'verify reads the engineer patch from best_patch.diff', inp.PATCH);
    ok(inp.FROZEN_PATCH === frozen, 'verify is told where to freeze it', inp.FROZEN_PATCH);
    ok(inp.STOP_ENGINEER === d.out_dir, 'verify is told to stop the engineer first', inp.STOP_ENGINEER);
    ok(r && r.patch === frozen, 'the candidate carries the FROZEN patch to integrate/commit', r && r.patch);
  }

  console.log('engineer returned status "failed" (finished, nothing to stop):');
  {
    const { fn, calls } = verifyStage();
    const r = await fn({ d, eng: { status: 'failed', speedup_geomean: 0 } });
    const inp = calls.length === 1 ? calls[0].prompt.inputs : {};
    ok(calls.length === 1, 'verify is dispatched (recovery path)');
    ok(!('STOP_ENGINEER' in inp), 'no STOP_ENGINEER for an engineer that returned');
    ok(inp.FROZEN_PATCH === frozen && r && r.patch === frozen, 'still verifies and merges the frozen copy');
  }

  console.log('engineer returned an above-floor win:');
  {
    const { fn, calls } = verifyStage();
    const r = await fn({ d, eng: { status: 'ok', speedup_geomean: 1.4, measurement_valid: true } });
    const inp = calls.length === 1 ? calls[0].prompt.inputs : {};
    ok(!('STOP_ENGINEER' in inp), 'no STOP_ENGINEER');
    ok(r && r.patch === frozen, 'candidate patch is the frozen copy');
    ok(calls.length === 1 && calls[0].opts.label === 'verify r3_d0', 'label unchanged', calls[0] && calls[0].opts.label);
  }

  console.log('engineer returned a trustworthy below-floor result:');
  {
    const { fn, calls } = verifyStage();
    const r = await fn({ d, eng: { status: 'ok', speedup_geomean: 0.9, measurement_valid: true } });
    ok(calls.length === 0 && r && r.ver === null, 'verify is still skipped');
  }

  console.log('downstream reads the stage result, not best_patch.diff:');
  {
    const after = src.slice(src.indexOf('  const clean = results.filter(Boolean);'));
    const roundEnd = after.indexOf('// --- update cross-round memory');
    const merge = after.slice(0, roundEnd);
    ok(/per_case: r\.ver\.per_case \|\| \[\], patch: r\.patch,/.test(merge), 'candidates take r.patch');
    ok(/files: r\.d\.focus_files \|\| \[\], patch: r\.patch \}/.test(merge), 'integrator takes r.patch');
    ok(/git apply \$\{winner\.patch\}/.test(merge), 'commit applies winner.patch');
    ok(!merge.includes('best_patch.diff'), 'no step after verify names best_patch.diff');
  }

  console.log('role contracts:');
  {
    const ver = fs.readFileSync(path.join(KW, 'roles', 'verify_engineer.md'), 'utf8');
    ok(ver.includes('scripts/stop_engineer.sh" "$STOP_ENGINEER"'), 'verify runs stop_engineer.sh when STOP_ENGINEER is set');
    ok(ver.includes('cp "$PATCH" "$FROZEN_PATCH"'), 'verify freezes the patch');
    ok(ver.indexOf('stop_engineer.sh') < ver.indexOf('cp "$PATCH" "$FROZEN_PATCH"'), 'stop happens before the copy');
    ok(ver.indexOf('cp "$PATCH" "$FROZEN_PATCH"') < ver.indexOf('\n0. '), 'freeze happens before step 0');
    ok(ver.includes('git apply "$FROZEN_PATCH"') && !ver.includes('git apply "$PATCH"'), 'verify applies only the frozen copy');
    for (const role of ['engineer.md', 'deep_engineer.md']) {
      const s = fs.readFileSync(path.join(KW, 'roles', role), 'utf8');
      ok(s.includes('test ! -e $OUTPUT_DIR/STOP &&'), `${role} honours $OUTPUT_DIR/STOP`);
    }
    ok(fs.existsSync(path.join(KW, 'scripts', 'stop_engineer.sh')), 'stop_engineer.sh exists');
  }

  if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
  console.log('\nall checks passed');
})().catch((e) => { console.error(e); process.exit(1); });
