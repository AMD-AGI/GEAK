#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';
// CPU-only fixture: GEAK (greedy) always-on source custody and host delivery.
// The real lane runs with real git/cp/chmod/sha256sum host commands on temporary repositories.
// Agents are scripted; the commit agent executes the exact bash block of its prompt. No model/GPU.
// Property: only the measured tree is exported. Verifiers receive disposable inputs; mutations
// cannot alter custody or lend the old result to changed bytes. A valid fallback still exports.
// Usage: node test_geak_pin_custody.js [--lane <kernel_lane.js>]   (exit 1 on any failure)
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const {execFileSync, spawnSync} = require('node:child_process');

const lanePath = process.argv.includes('--lane')
  ? path.resolve(process.argv[process.argv.indexOf('--lane') + 1])
  : path.resolve(__dirname, '../kernel_lane.js');
const laneSource = fs.readFileSync(lanePath, 'utf8');
const body = laneSource.replace(/^export const meta/m, 'const meta');
const sha = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const total = 2000000000;
const perCase = (s) => [{name: 'c1', baseline_ms: 8, optimized_ms: 8 / s, speedup: s}];
const patchFor = (after) => `diff --git a/k.py b/k.py\n--- a/k.py\n+++ b/k.py\n@@ -1 +1 @@\n-baseline\n+${after}\n`;
const field = (prompt, key) => {
  const line = prompt.split('\n').find(l => l.startsWith(`- ${key}: `));
  if (!line) return undefined;
  const raw = line.slice(key.length + 4);
  try { return JSON.parse(raw); } catch (error) { return raw; }
};
// Rewrite a read-only snapshot the way a shell agent can: make it writable, or replace it by rename.
function tamper(file, how) {
  if (how === 'rename') {
    fs.writeFileSync(`${file}.tmp`, patchFor('tampered'));
    fs.renameSync(`${file}.tmp`, file);
  } else {
    fs.chmodSync(file, 0o644);
    fs.writeFileSync(file, patchFor('tampered'));
  }
}

async function lane(sc) {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'geak-pin-custody-'));
  try {
    const canonical = path.join(temp, 'canonical'), evalDir = path.join(temp, 'eval');
    fs.mkdirSync(canonical); fs.mkdirSync(evalDir);
    const git = (...args) => execFileSync('git', ['-C', canonical, ...args], {encoding: 'utf8'}).trim();
    git('init', '-q');
    fs.writeFileSync(path.join(canonical, 'k.py'), 'baseline\n');
    git('add', '.');
    git('-c', 'user.name=fixture', '-c', 'user.email=fixture@localhost', 'commit', '-qm', 'baseline');
    if (sc.newFiles) {
      fs.writeFileSync(path.join(canonical, 'test_harness.py'), 'unrelated harness\n');
      if (sc.obstruction) fs.writeFileSync(path.join(canonical, 'backend.hip'), 'pre-existing unrelated source\n');
    }
    let activeDeadline = null, now = total - 7200, cutoffThrown = false;
    const setProcessDeadline = epoch => { activeDeadline = epoch; };
    setProcessDeadline.secondsLeft = () => activeDeadline - now;
    const obs = {verified: {}, custodyBeforeVerify: {}, commits: [], unhandled: [], history: null, cumulative: null};
    const g = {
      args: {kernel_path: '/original', workflow_dir: '/workflow', budget: sc.speeds.length, max_no_improve: 2,
        editable_source_paths: sc.newFiles ? ['k.py', 'backend.hip', 'impl.s', 'module.py'] : ['k.py'],
        agent_timeout_ms: 45000, agent_retries: 1, warm_start: 'off', update_experience: 'off', llm_stats: 'false',
        deadline_epoch: total, search_reserve_s: 5100, final_validation_reserve_s: 1800, final_cleanup_reserve_s: 300,
        ...(sc.delivery === false ? {} : {delivery_reserve_s: sc.delivery === 0 ? 0 : 1000})},
      phase: () => {}, log: () => {}, setProcessDeadline,
      workflow: async () => { throw new Error('unexpected nested workflow'); },
      budget: {total: null, spent: () => 0, remaining: () => Infinity},
      parallel: async ts => Promise.all(ts.map(t => t())),
      pipeline: async (items, ...stages) => Promise.all(items.map(async item => {
        for (const stage of stages) item = await stage(item);
        return item;
      })),
      runProcess: async options => {
        if (sc.cutoffAtPinCheck && !cutoffThrown && obs.commits.length && options.cmd === 'sha256sum') {
          cutoffThrown = true; now = total - 5100; // The search cutoff arrives as the commit returns.
          throw Object.assign(new Error('fixture search cutoff'), {code: 'GEAK_PROCESS_DEADLINE'});
        }
        const r = spawnSync(options.cmd, options.args, {input: options.input, encoding: 'utf8', timeout: 10000});
        if (r.error) throw r.error;
        return {status: 'completed', code: r.status, stdout: r.stdout, stderr: r.stderr,
          strays_killed: false, group_cleared: true, stdio_held: false};
      },
      agent: async (prompt, options) => {
        const label = options.label;
        if (label === 'director:setup') return {eval_dir: evalDir, workspace: canonical, kernel_name: 'fixture',
          device_gfx: 'gfx950', device_target: 'unknown', physical_cu_count: 256, baseline_frozen: true,
          source_files: ['k.py']};
        if (label === 'tech_lead:analyze') return {kernel_type: 'gemm', modifiable_files: ['k.py']};
        if (label === 'benchmark_engineer') return {baseline_per_case: perCase(1), baseline_geomean_ms: 8,
          commandment_path: '/eval/COMMANDMENT.md', correctness_cmd: 'fixture-correctness',
          benchmark_cmd: 'fixture-benchmark'};
        if (label.includes('profile')) return {bottleneck: 'memory', top_opportunities: []};
        if (label.startsWith('tech_lead:plan')) return {stop: false,
          directions: sc.speeds.map((s, i) => ({id: `d${i}`, title: `d${i}`, specialty: 'memory', prompt: 'p'}))};
        if (label.startsWith('eng ')) {
          const out = field(prompt, 'OUTPUT_DIR'), i = Number(out.match(/engineer_(\d+)$/)[1]);
          fs.mkdirSync(out, {recursive: true});
          if (sc.newFiles) {
            const ws = path.join(out, 'workspace');
            execFileSync('git', ['clone', '-q', canonical, ws]);
            fs.writeFileSync(path.join(ws, 'k.py'), `measured ${i}\n`);
            fs.writeFileSync(path.join(ws, 'backend.hip'), 'qualified HIP source\n');
            fs.writeFileSync(path.join(ws, 'impl.s'), 'qualified assembly source\n');
            fs.writeFileSync(path.join(ws, 'module.py'), 'qualified module source\n');
            fs.writeFileSync(path.join(ws, 'test_harness.py'), 'unrelated engineer harness\n');
            execFileSync('bash', [path.join(__dirname, 'save_candidate_patch.sh'), ws,
              path.join(out, 'best_patch.diff'), 'HEAD', 'k.py', 'backend.hip', 'impl.s', 'module.py']);
            assert.ok(prompt.includes('save_candidate_patch.sh'), 'native engineer receives the capture tool');
          } else if (!sc.omitPatch) fs.writeFileSync(path.join(out, 'best_patch.diff'), patchFor(`measured ${i}`));
          if (sc.workspaceMutation) {
            fs.mkdirSync(path.join(out, 'workspace'), {recursive: true});
            fs.writeFileSync(path.join(out, 'workspace/k.py'), `measured ${i}\n`);
          }
          return sc.engineerReturn || {status: 'ok', measurement_valid: true, speedup_geomean: sc.speeds[i]};
        }
        if (label.startsWith('verify ')) {
          const measured = field(prompt, 'PATCH');
          const i = Number(String(field(prompt, 'ENGINEER_WORKSPACE')).match(/engineer_(\d+)\/workspace$/)[1]);
          obs.verified[i] = measured;
          const pin = path.join(evalDir, `delivery/sources/round_1_engineer_${i}.diff`);
          obs.custodyBeforeVerify[i] = {path: pin, sha256: sha(fs.readFileSync(pin))};
          assert.notEqual(measured, pin, 'the verifier never receives a custody file');
          assert.equal(sha(fs.readFileSync(measured)), obs.custodyBeforeVerify[i].sha256,
            'the disposable input begins with the pinned bytes');
          if (sc.engineerReturn?.measurement_valid === false) {
            const receipt = JSON.parse(fs.readFileSync(`${pin}.json`, 'utf8'));
            assert.equal(receipt.evidence, null, 'unmeasured source has no fitness before verification');
            assert.equal(receipt.eng.status, 'partial');
            assert.ok(!label.includes('(delivery)'), 'partial source uses ordinary round verification');
          }
          if (sc.workspaceMutation) {
            const workspace = field(prompt, 'ENGINEER_WORKSPACE');
            fs.writeFileSync(path.join(workspace, 'k.py'), 'later unmeasured iteration\n');
            fs.writeFileSync(path.join(path.dirname(workspace), 'best_patch.diff'), patchFor('later unmeasured iteration'));
            assert.equal(sha(fs.readFileSync(pin)), obs.custodyBeforeVerify[i].sha256,
              'later worker changes cannot overwrite the host source pin');
          }
          if (sc.newFiles) {
            const ws = path.join(temp, `verify_${i}`);
            execFileSync('git', ['clone', '-q', canonical, ws]);
            execFileSync('git', ['-C', ws, 'apply', '--', measured]);
            assert.equal(fs.readFileSync(path.join(ws, 'backend.hip'), 'utf8'), 'qualified HIP source\n');
            assert.equal(fs.readFileSync(path.join(ws, 'impl.s'), 'utf8'), 'qualified assembly source\n');
            assert.equal(fs.readFileSync(path.join(ws, 'module.py'), 'utf8'), 'qualified module source\n');
            assert.equal(fs.existsSync(path.join(ws, 'test_harness.py')), false);
          }
          if (sc.tamper && sc.tamper.startsWith('verifier') && i === 0) tamper(measured, sc.tamper.split('-')[1]);
          return {status: 'verified', correctness: 'pass', verified_geomean: sc.speeds[i], per_case: perCase(sc.speeds[i])};
        }
        if (label.startsWith('integrate ')) {
          if (sc.tamper === 'integrator') tamper(field(prompt, 'PATCHES').find(p => p.id === 'd0').patch, 'chmod');
          return {attempted: true, conclusion: 'no_improvement', notes: 'fixture'};
        }
        if (label.startsWith('commit ')) {
          const block = prompt.match(/```bash\n([\s\S]*?)\n```/)[1];
          if (sc.tamper === 'commit-agent') tamper(block.match(/git apply (?:--index -- )?(\S+)/)[1], 'chmod');
          const r = spawnSync('bash', ['-e', '-c', block], {encoding: 'utf8'});
          obs.commits.push({code: r.status, stderr: r.stderr.trim()});
          if (sc.tamper === 'commit-tree' && r.status === 0) {
            fs.writeFileSync(path.join(canonical, 'k.py'), 'hand-merged but unmeasured\n');
            git('add', 'k.py');
            git('-c', 'user.name=fixture', '-c', 'user.email=fixture@localhost', 'commit', '-qm', 'unmeasured hand-merge');
          }
          return {committed: r.status === 0, note: 'fixture executed the prompt block'};
        }
        if (label.startsWith('tech_lead:memory')) return {insights: [], ledger: []};
        if (label.startsWith('storage:') || label.startsWith('kb:')) return {ok: true};
        if (label === 'tech_lead:report') {
          obs.history = field(prompt, 'HISTORY');
          obs.cumulative = field(prompt, 'CUMULATIVE_SPEEDUP');
          return {final_speedup_geomean: obs.cumulative, final_patch: '/final.diff', report_path: '/report', rounds: 1};
        }
        if (label === 'director:validate') return {validation_status: 'accepted',
          director_verified_speedup_geomean: field(prompt, 'TECH_LEAD_REPORTED_GEOMEAN')};
        obs.unhandled.push(label);
        return null;
      },
    };
    g.agent.processDeadlineActive = true;
    const value = await new Function(...Object.keys(g), `return (async () => {${body}})();`)(...Object.values(g));
    const pins = path.join(evalDir, 'delivery/sources');
    assert.equal(value.process_validated, true);
    assert.equal(value.final_patch, value.delivered_patch);
    assert.equal(fs.readFileSync(value.delivered_patch, 'utf8').trim(),
      git('diff', '--binary', git('rev-list', '--max-parents=0', 'HEAD'), 'HEAD', '--', '.'),
      'the host export equals the final canonical tree against ORIGINAL');
    if (sc.newFiles) {
      const exported = fs.readFileSync(value.delivered_patch, 'utf8');
      for (const name of ['backend.hip', 'impl.s', 'module.py']) assert.ok(exported.includes(`b/${name}`));
      assert.ok(!exported.includes('test_harness.py'));
      assert.equal(git('show', 'HEAD:backend.hip'), 'qualified HIP source');
      assert.equal(git('show', 'HEAD:impl.s'), 'qualified assembly source');
      assert.equal(git('show', 'HEAD:module.py'), 'qualified module source');
      assert.equal(fs.readFileSync(path.join(canonical, 'test_harness.py'), 'utf8'), 'unrelated harness\n');
      assert.equal(git('ls-files', '--', 'test_harness.py'), '');
      if (sc.obstruction) assert.equal(fs.readFileSync(path.join(evalDir,
        'delivery/obstructions/round_1/backend.hip'), 'utf8'), 'pre-existing unrelated source\n');
    }
    return {value, obs, committed: git('show', 'HEAD:k.py') + '\n', worktree: fs.readFileSync(path.join(canonical, 'k.py'), 'utf8'),
      status: git('status', '--porcelain', '--untracked-files=no'), incomplete: git('for-each-ref', '--format=%(refname)', 'refs/geak/'),
      pins: fs.existsSync(pins) ? fs.readdirSync(pins).sort() : [],
      snapshots: Object.fromEntries(Object.entries(obs.custodyBeforeVerify).map(([i, pin]) => [i, {
        sha256: sha(fs.readFileSync(pin.path)),
        receipt: JSON.parse(fs.readFileSync(`${pin.path}.json`, 'utf8')),
        verifier_sha256: sha(fs.readFileSync(obs.verified[i])),
      }]))};
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
}

const delivery = r => (r.obs.history && r.obs.history.delivery) || {candidates: [], delivered: null};
const snapshotNames = ['round_1_engineer_0.diff', 'round_1_engineer_0.diff.json'];
function checkVerifierIsolation(r) {
  assert.deepEqual(r.pins, snapshotNames, 'custody and its receipt exist independently of verification reserve');
  assert.match(r.obs.verified[0], /\/engineer_0\/verification_input\.diff$/);
  assert.equal(r.snapshots[0].sha256, r.obs.custodyBeforeVerify[0].sha256,
    'verifier mutation leaves the custody bytes unchanged');
  assert.equal(r.snapshots[0].receipt.source.sha256, r.snapshots[0].sha256,
    'the saved custody receipt still matches the source');
  assert.equal(r.snapshots[0].receipt.ver, null, 'changed verification input is not authoritative evidence');
  assert.equal(r.snapshots[0].receipt.evidence, null, 'the changed result cannot be promoted using its old score');
  assert.notEqual(r.snapshots[0].verifier_sha256, r.snapshots[0].sha256);
  assert.equal(r.committed, 'baseline\n');
  assert.equal(r.worktree, 'baseline\n');
  assert.equal(r.status, '');
  assert.equal(r.obs.cumulative, 1);
  assert.equal(r.value.warm_start.rounds_committed, 0);
  assert.equal(delivery(r).delivered, null);
}
const cases = [
  {name: 'pre-latch partial source uses ordinary verification without inherited fitness', sourceRecovery: true,
    speeds: [2], engineerReturn: {status: 'partial', measurement_valid: false, speedup_geomean: 0, per_case: []},
    check: r => {
      assert.equal(r.committed, 'measured 0\n');
      assert.equal(r.obs.cumulative, 2);
      assert.equal(r.snapshots[0].receipt.evidence, 2, 'only the independent verifier assigns fitness');
      assert.equal(r.snapshots[0].receipt.eng.measurement_valid, false);
    }},
  {name: 'pre-latch partial return without a patch cannot win', sourceRecovery: true,
    speeds: [2], omitPatch: true,
    engineerReturn: {status: 'partial', measurement_valid: false, speedup_geomean: 0, per_case: []},
    check: r => {
      assert.equal(r.committed, 'baseline\n');
      assert.equal(r.obs.cumulative, 1);
      assert.deepEqual(r.obs.verified, {}, 'missing source never reaches a verifier model call');
      assert.deepEqual(r.pins, []);
      assert.equal(r.value.warm_start.rounds_committed, 0);
    }},
  {name: 'immutable measured pin survives later unmeasured workspace and return-file changes', sourceRecovery: true,
    speeds: [2], workspaceMutation: true,
    check: r => {
      assert.equal(r.committed, 'measured 0\n');
      assert.equal(r.obs.cumulative, 2);
      assert.equal(r.snapshots[0].sha256, r.obs.custodyBeforeVerify[0].sha256);
      assert.equal(r.snapshots[0].receipt.source.sha256, r.snapshots[0].sha256);
    }},
  ...[false, true].map(obstruction => ({name: `new source capture → pin → verifier → export; obstruction=${obstruction}`,
    speeds: [2], newFiles: true, obstruction, check: r => {
      assert.equal(r.committed, 'measured 0\n'); assert.equal(r.obs.cumulative, 2);
      assert.equal(r.value.final_speedup, 2); assert.equal(r.value.warm_start.rounds_committed, 1);
      assert.equal(r.obs.commits[0].code, 0);
    }})),
  {name: 'control: an unchanged pin is committed as measured', speeds: [2],
    check: r => {
      assert.equal(r.committed, 'measured 0\n', 'the pinned measured bytes are committed');
      assert.equal(r.obs.cumulative, 2);
      assert.equal(r.value.warm_start.rounds_committed, 1);
      assert.equal(r.value.final_speedup, 2);
      assert.deepEqual(r.pins, snapshotNames, 'one snapshot and its durable evidence receipt');
      assert.match(r.obs.verified[0], /\/engineer_0\/verification_input\.diff$/, 'the verifier measures a disposable copy');
      assert.equal(r.snapshots[0].sha256, r.obs.custodyBeforeVerify[0].sha256);
      assert.equal(r.snapshots[0].receipt.source.sha256, r.snapshots[0].sha256);
      assert.equal(r.snapshots[0].receipt.ver.status, 'verified');
      assert.equal(r.snapshots[0].receipt.evidence, 2);
    }},
  ...['chmod', 'rename'].map(how => ({name: `verifier rewrites its disposable input (${how}) and returns verified`,
    speeds: [2], tamper: `verifier-${how}`,
    check: checkVerifierIsolation})),
  {name: 'integrator rewrites the winner pin after verification (post-verify check alone misses this)',
    speeds: [2, 1.5], tamper: 'integrator',
    check: r => {
      assert.notEqual(r.committed, 'tampered\n', 'changed bytes are not committed under the old verification');
      assert.notEqual(r.obs.cumulative, 2, 'the changed winner is not promoted');
      assert.equal(r.committed, 'measured 1\n', 'the unchanged qualified source is delivered by its pinned bytes');
      assert.equal(r.obs.cumulative, 1.5);
      assert.equal(r.status, '');
      assert.equal(delivery(r).delivered, 'round_1_engineer_1');
      assert.ok(r.value.delivery_incomplete.some(row => row.key === 'round_1_engineer_0'
        && row.status === 'source_changed' && row.evidence === 2),
      'the stronger undeliverable result remains visible while the fallback is exported');
    }},
  {name: 'commit agent rewrites the pin before applying it (pre-commit check alone misses this)',
    speeds: [2], tamper: 'commit-agent',
    check: r => {
      assert.notEqual(r.committed, 'tampered\n', 'changed bytes are not committed under the old verification');
      assert.equal(r.committed, 'baseline\n');
      assert.equal(r.worktree, 'baseline\n');
      assert.equal(r.status, '');
      assert.equal(r.obs.cumulative, 1);
      assert.ok(r.value.delivery_incomplete.some(row => row.status === 'source_changed'));
    }},
  {name: 'a successful but unmeasured hand-merge is rolled back before pinned-source delivery',
    speeds: [2], tamper: 'commit-tree',
    check: r => {
      assert.equal(r.committed, 'measured 0\n');
      assert.equal(r.worktree, 'measured 0\n');
      assert.equal(r.status, '');
      assert.equal(r.obs.cumulative, 2);
      assert.ok(r.obs.history.rounds.some(row => row.status === 'commit_failed'));
      assert.equal(delivery(r).delivered, 'round_1_engineer_0');
      assert.match(r.incomplete, /refs\/geak\/incomplete-round-1/,
        'the rejected hand-merge remains reachable as incomplete evidence');
    }},
  {name: 'a host check that cannot run at the search cutoff still ends with the exact pinned bytes', speeds: [2],
    cutoffAtPinCheck: true,
    check: r => {
      assert.equal(r.committed, 'measured 0\n', 'the pinned measured bytes are the committed source');
      assert.equal(r.worktree, 'measured 0\n');
      assert.equal(r.status, '');
      assert.equal(r.obs.cumulative, 2);
    }},
  ...[false, 0].map(reserve => ({name: `${reserve === false ? 'omitted' : 'zero'} delivery reserve preserves custody and verifier isolation`,
    speeds: [2], tamper: 'verifier-chmod', delivery: reserve, check: checkVerifierIsolation})),
];

(async () => {
  console.log(`lane: ${lanePath} (sha256 ${sha(laneSource)})`);
  let failed = 0;
  const selectedCases = cases.filter(sc => (!process.argv.includes('--new-source-only') || sc.newFiles)
    && (!process.argv.includes('--source-recovery-only') || sc.sourceRecovery));
  for (const sc of selectedCases) {
    const r = await lane(sc);
    const summary = {committed: r.committed.trim(), cumulative: r.obs.cumulative,
      rounds_committed: r.value.warm_start.rounds_committed, final_speedup: r.value.final_speedup,
      round_status: r.obs.history ? r.obs.history.rounds.map(x => x.status || (x.improved ? 'committed' : 'not_committed')) : null,
      delivery: delivery(r).candidates.map(row => `${row.key}:${row.status}`), delivered: delivery(r).delivered,
      incomplete_refs: r.incomplete ? r.incomplete.split('\n') : [], commit_exits: r.obs.commits.map(c => c.code),
      unhandled: r.obs.unhandled};
    try {
      sc.check(r);
      console.log(`  ok: ${sc.name} ${JSON.stringify(summary)}`);
    } catch (error) {
      failed++;
      console.log(`  FAIL: ${sc.name}: ${error.message.split('\n')[0]} ${JSON.stringify(summary)}`);
    }
  }
  if (failed) {
    console.log(`FAIL: ${failed}/${cases.length} GEAK pin-custody case(s). CPU fixture only; no performance claim.`);
    process.exitCode = 1;
  } else {
    console.log(`PASS: ${selectedCases.length} GEAK pin-custody cases. CPU fixture only; no performance claim.`);
  }
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
