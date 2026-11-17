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
//     (which is what integrate and commit read), never best_patch.diff; with EVO on it returns EVO's
//     immutable object, from which the frozen copy is made;
//   - STOP_ENGINEER is passed exactly when the engineer's return is missing, and not for an engineer
//     that returned (even with status "failed": it has finished);
//   - a verify that could not stop the engineer (status "stop_failed") is logged and never verified,
//     and routing books it as no_result;
//   - a trustworthy below-floor return still skips verify;
//   - engineers are told to save through save_patch.sh (atomic), and the role files carry the
//     matching stop / freeze / stop_failed instructions.
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
function verifyStage(opts = {}) {
  const start = src.indexOf('    async (prev) => {\n      const { d, eng, route } = prev;\n');
  const end = src.indexOf('\n    }\n  );\n\n  const clean = results.filter(Boolean);', start);
  if (start < 0 || end < 0) throw new Error('could not locate the Verify stage in kernel_lane.js');
  const body = src.slice(start, end + '\n    }'.length);
  const calls = [];
  const logs = [];
  const env = {
    round: 3,
    CANDIDATE_FLOOR: 1.0,
    primSpeedup: (r) => (r && typeof r.speedup_geomean === 'number' ? r.speedup_geomean : 0),
    says: (v, w) => String(v || '').toLowerCase() === w,
    log: (m) => logs.push(m),
    roleAgent: (role, phase, intro, inputs) => ({ role, phase, inputs }),
    agentT: async (p, o) => { calls.push({ prompt: p, opts: o }); return opts.ver || { status: 'verified' }; },
    CANONICAL: '/eval/workspace', EVAL_DIR: '/eval', WORKFLOW_DIR: '/wf', COMMANDMENT: '/eval/COMMANDMENT.md',
    BASELINE_PER_CASE: [], HARNESS_ADDENDUM: '', REQUIRE_GRAPH_CAPTURE: false, VERIFY_SCHEMA: {},
    // Default-off subsystems from main; EVO is switched on per case below.
    EVO: opts.evo || null, ROUTING_ON: false,
    __p1WorkerOutcome: () => {}, __routeScope: (x) => x,
  };
  const names = Object.keys(env);
  const fn = new Function(...names, `return (${body});`)(...names.map((k) => env[k]));
  return { fn, calls, logs };
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

  console.log('EVO on (patch already an immutable archive object):');
  {
    const obj = '/eval/evo/objects/' + 'a'.repeat(64);
    const evo = { checkPatch: async () => ({ patch_object_path: obj }) };
    const { fn, calls } = verifyStage({ evo });
    const r = await fn({ d, eng: null });
    const inp = calls.length === 1 ? calls[0].prompt.inputs : {};
    ok(inp.PATCH === obj, 'verify reads EVO\'s object', inp.PATCH);
    ok(inp.FROZEN_PATCH === frozen && inp.STOP_ENGINEER === d.out_dir, 'and still stops + freezes');
    ok(r && r.patch === obj, 'EVO keeps receiving its own object path', r && r.patch);
  }

  console.log('verify could not stop the engineer:');
  {
    const ver = { status: 'stop_failed', correctness: 'not_checked', notes: 'STOP_FAILED: 1 process(es) still running' };
    const { fn, logs } = verifyStage({ ver });
    const r = await fn({ d, eng: null });
    ok(r && r.ver === ver, 'stage returns the stop_failed verdict');
    ok(logs.some((m) => /could not stop the engineer/.test(m) && /STOP_FAILED/.test(m)), 'it is logged with the reason');
    const verifiedFilter = src.match(/const verified = clean\.filter\(r => r\.ver && says\(r\.ver\.status, 'verified'\)/);
    ok(!!verifiedFilter, 'only status "verified" becomes a candidate (stop_failed never does)');
    ok(/says\(r\.ver\.status, 'stop_failed'\) \? 'no_result'/.test(src), 'routing books stop_failed as no_result');
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

  console.log('engineers save atomically:');
  {
    ok(/Save best_patch\.diff via \\`bash \$\{WORKFLOW_DIR\}\/scripts\/save_patch\.sh <KERNEL_PATH> \$\{d\.out_dir\}\\`/.test(src),
      'the Optimize prompt saves through save_patch.sh');
    ok(!/git diff > \$\{d\.out_dir\}\/best_patch\.diff/.test(src), 'the Optimize prompt no longer writes it with git diff >');
    for (const role of ['engineer.md', 'deep_engineer.md']) {
      const s = fs.readFileSync(path.join(KW, 'roles', role), 'utf8');
      ok(s.includes('bash $SKILL_DIR/scripts/save_patch.sh $KERNEL_PATH $OUTPUT_DIR'), `${role} saves through save_patch.sh`);
      ok(!/git diff > \$OUTPUT_DIR\/best_patch\.diff/.test(s), `${role} has no in-place git diff > best_patch.diff`);
      ok(s.includes('exit 87, `GEAK_ENGINEER_STOPPED`'), `${role} explains the stop refusals`);
    }
  }

  console.log('verify role contract:');
  {
    const ver = fs.readFileSync(path.join(KW, 'roles', 'verify_engineer.md'), 'utf8');
    const stopAt = ver.indexOf('scripts/stop_engineer.sh" "$STOP_ENGINEER" || FENCE=STOP_FAILED');
    const freezeAt = ver.indexOf('scripts/freeze_patch.sh" "$PATCH" "$FROZEN_PATCH" || FENCE=FREEZE_FAILED');
    ok(stopAt > 0, 'a failed stop is caught (FENCE=STOP_FAILED)');
    ok(freezeAt > 0, 'the patch is frozen by freeze_patch.sh, and a failed freeze is caught');
    ok(stopAt < freezeAt && freezeAt < ver.indexOf('\n0. '), 'stop, then freeze, both before step 0');
    ok(/\[ "\$FENCE" = ok \] && \{ bash/.test(ver), 'no freeze after a failed stop');
    ok(/STOP HERE: do not build, apply or measure/.test(ver) && ver.includes('status:"stop_failed"'),
      'on a fence failure verify measures nothing and returns stop_failed');
    ok(!ver.includes('cp "$PATCH"'), 'no unchecked copy of the live patch is left');
    ok(ver.includes('git apply "$FROZEN_PATCH"') && !ver.includes('git apply "$PATCH"'), 'verify applies only the frozen copy');
    for (const f of ['stop_engineer.sh', 'freeze_patch.sh', 'save_patch.sh']) {
      ok(fs.existsSync(path.join(KW, 'scripts', f)), `${f} exists`);
    }
  }

  if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
  console.log('\nall checks passed');
})().catch((e) => { console.error(e); process.exit(1); });
