#!/usr/bin/env node
// Faithful affected-path fixture: real lane, real Git, exact commit prompt block,
// and the runtime's grace+settle admission floor. No model or GPU execution.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync, execFileSync} = require('node:child_process');
const arg = name => process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : null;
const lanePath = path.resolve(arg('--lane') || path.join(__dirname, '../kernel_lane.js'));
const source = fs.readFileSync(lanePath, 'utf8');
const body = source.replace(/^export const meta/m, 'const meta');
const total = 2000000000;
const field = (prompt, key) => {
  const line = prompt.split('\n').find(s => s.startsWith(`- ${key}: `));
  if (!line) return undefined;
  const raw = line.slice(key.length + 4);
  try { return JSON.parse(raw); } catch (_) { return raw; }
};
const cases = [
  {name: 'two_rounds_with_untracked_harness', rounds: 2},
  {name: 'complete_control', rounds: 1},
  {name: 'report_returns_at_three_seconds', rounds: 1, cut: 'report'},
  {name: 'director_returns_at_three_seconds', rounds: 1, cut: 'director'},
  {name: 'director_tampers_then_returns_at_three_seconds', rounds: 1, cut: 'director', tamper: true},
  {name: 'director_tampers_with_time_for_restoration', rounds: 1, tamper: true},
  {name: 'first_export_failure_stays_explicit', rounds: 1, firstExportFails: true},
];

async function runCase(sc) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'geak-delivery-mf-'));
  try {
    const canonical = path.join(tmp, 'canonical'), evalDir = path.join(tmp, 'eval');
    fs.mkdirSync(canonical); fs.mkdirSync(evalDir);
    const git = (...args) => execFileSync('git', ['-C', canonical, ...args], {encoding: 'utf8'}).trim();
    git('init', '-q');
    fs.writeFileSync(path.join(canonical, 'k.py'), 'baseline\n');
    git('add', 'k.py');
    git('-c', 'user.name=fixture', '-c', 'user.email=fixture@localhost', 'commit', '-qm', 'baseline');
    // benchmark_engineer can create this after the baseline commit in a real run.
    fs.writeFileSync(path.join(canonical, 'test_harness.py'), 'untracked evaluation harness\n');
    const seen = [], commits = [], calls = [], deadlines = [], hostRefusals = [];
    let now = total - 7200, deadline, round = 0, preDirectorBytes = null;
    const setProcessDeadline = epoch => { deadline = epoch; deadlines.push(epoch); };
    setProcessDeadline.secondsLeft = () => deadline - now;
    const perCase = speedup => [{name: 'c1', baseline_ms: 8, optimized_ms: 8 / speedup, speedup}];
    const g = {
      args: {kernel_path: '/original', workflow_dir: '/workflow', budget: sc.rounds, max_no_improve: 2,
        editable_source_paths: ['k.py'],
        agent_timeout_ms: 45000, agent_retries: 1, warm_start: 'off', update_experience: 'off',
        llm_stats: 'false', deadline_epoch: total, search_reserve_s: 5100,
        final_validation_reserve_s: 1800, final_cleanup_reserve_s: 300},
      phase: () => {}, log: () => {}, setProcessDeadline,
      workflow: async () => { throw new Error('unexpected nested workflow'); },
      budget: {total: null, spent: () => 0, remaining: () => Infinity},
      parallel: async items => Promise.all(items.map(f => f())),
      pipeline: async (items, ...stages) => Promise.all(items.map(async item => {
        for (const stage of stages) item = await stage(item);
        return item;
      })),
      runProcess: async options => {
        // Matches createRuntime.scriptRunProcess/boundedTimeout: no command can
        // start when its remaining time is <= its grace + settle reservation.
        const cleanupMs = (options.graceMs ?? 5000) + (options.settleMs ?? 2000);
        if ((deadline - now) * 1000 <= cleanupMs) {
          hostRefusals.push({cmd: options.cmd, seconds_left: deadline - now, cleanup_ms: cleanupMs});
          throw Object.assign(new Error('workflow process deadline reached'), {code: 'GEAK_PROCESS_DEADLINE'});
        }
        if (sc.firstExportFails && options.cmd === 'sh'
            && options.args.at(-1) === path.join(evalDir, 'host_final_patch.diff')) {
          return {status: 'completed', code: 1, stdout: '', stderr: 'fixture write failure',
            strays_killed: false, group_cleared: true, stdio_held: false};
        }
        const r = spawnSync(options.cmd, options.args, {input: options.input, cwd: options.cwd,
          env: options.env ? {...process.env, ...options.env} : process.env,
          encoding: 'utf8', timeout: 10000});
        if (r.error) throw r.error;
        return {status: 'completed', code: r.status, stdout: r.stdout, stderr: r.stderr,
          strays_killed: false, group_cleared: true, stdio_held: false};
      },
      agent: async (prompt, options) => {
        const label = options.label; calls.push(label);
        if (label === 'director:setup') return {eval_dir: evalDir, workspace: canonical,
          kernel_name: 'fixture', device_gfx: 'gfx950', device_target: 'unknown',
          physical_cu_count: 256, baseline_frozen: true, source_files: ['k.py']};
        if (label === 'tech_lead:analyze') return {kernel_type: 'gemm', modifiable_files: ['k.py']};
        if (label === 'benchmark_engineer') return {baseline_per_case: perCase(1), baseline_geomean_ms: 8,
          commandment_path: '/eval/COMMANDMENT.md', correctness_cmd: 'fixture-correctness', benchmark_cmd: 'fixture-benchmark'};
        if (label.includes('profile')) return {bottleneck: 'memory', top_opportunities: []};
        if (label.startsWith('tech_lead:plan')) {
          round++;
          return {stop: false, directions: [{id: 'd0', title: 'candidate', specialty: 'memory', prompt: 'p'}]};
        }
        if (label.startsWith('eng ')) {
          const out = field(prompt, 'OUTPUT_DIR'), before = fs.readFileSync(path.join(canonical, 'k.py'), 'utf8');
          seen.push(before.trim()); fs.mkdirSync(out, {recursive: true});
          fs.writeFileSync(path.join(out, 'best_patch.diff'),
            `diff --git a/k.py b/k.py\n--- a/k.py\n+++ b/k.py\n@@ -1 +1 @@\n-${before}+candidate ${round}\n`);
          return {status: 'ok', measurement_valid: true, speedup_geomean: round + 1};
        }
        if (label.startsWith('verify ')) return {status: 'verified', correctness: 'pass',
          verified_geomean: round + 1, per_case: perCase(round + 1)};
        if (label.startsWith('commit ')) {
          const block = prompt.match(/```bash\n([\s\S]*?)\n```/)[1];
          const r = spawnSync('bash', ['-e', '-c', block], {encoding: 'utf8'});
          commits.push({code: r.status, add_all: /\badd -A\b/.test(block),
            harness_tracked: git('ls-files', '--', 'test_harness.py') !== ''});
          return {committed: r.status === 0};
        }
        if (label.startsWith('tech_lead:memory')) return {insights: [], ledger: []};
        if (label.startsWith('storage:') || label.startsWith('kb:')) return {};
        if (label === 'tech_lead:report') {
          if (sc.cut === 'report') { now = deadline - 3; return null; }
          return {final_speedup_geomean: field(prompt, 'CUMULATIVE_SPEEDUP'),
            final_patch: '/ignored.diff', report_path: '/report', rounds: sc.rounds};
        }
        if (label === 'director:validate') {
          const patch = field(prompt, 'FINAL_PATCH'); preDirectorBytes = fs.readFileSync(patch, 'utf8');
          if (sc.tamper) fs.writeFileSync(patch, 'changed by Director\n');
          if (sc.cut === 'director') { now = deadline - 3; return null; }
          return {validation_status: 'accepted', director_verified_speedup_geomean: field(prompt, 'TECH_LEAD_REPORTED_GEOMEAN')};
        }
        throw new Error('unhandled agent ' + label);
      },
    };
    g.agent.processDeadlineActive = true;
    let value = null, error = null;
    try { value = await new Function(...Object.keys(g), `return (async () => {${body}})();`)(...Object.values(g)); }
    catch (e) { error = {code: e.code, message: e.message}; }
    const exported = path.join(evalDir, 'host_final_patch.diff');
    const output = {case: sc.name, seen, commits, host_refusals: hostRefusals,
      error, validation_status: value?.validation_status, process_validated: value?.process_validated,
      final_speedup: value?.final_speedup, final_patch_withheld: value?.final_patch === '',
      delivered_pointer_retained: value?.delivered_patch === exported,
      incomplete: value?.delivery_incomplete, rounds_committed: value?.warm_start.rounds_committed,
      exported_matches_pre_director: preDirectorBytes !== null && fs.readFileSync(exported, 'utf8') === preDirectorBytes,
      harness_tracked_at_end: git('ls-files', '--', 'test_harness.py') !== '',
      kb_write_called: calls.includes('kb:write')};
    try {
      assert.equal(g.args.deadline_epoch, total);
      assert.ok(deadlines.every(d => [total - 5100, total - 1800, total - 300].includes(d)), 'only original phase cutoffs');
      if (sc.firstExportFails) {
        assert.equal(error?.code, 'GEAK_NATIVE_DELIVERY_INCOMPLETE');
      } else {
        assert.equal(error, null, 'role cutoff must return an explicit scoped result');
        assert.equal(value.delivered_patch, exported);
        assert.equal(output.harness_tracked_at_end, false);
        if (sc.rounds === 2) {
          assert.deepEqual(seen, ['baseline', 'candidate 1'], 'round two compounds on the measured round-one source');
          assert.equal(value.warm_start.rounds_committed, 2);
          assert.ok(commits.every(c => !c.add_all && !c.harness_tracked));
        }
        if (sc.cut === 'director') {
          assert.equal(value.validation_status, 'unavailable');
          assert.equal(value.final_speedup, null); assert.equal(value.final_patch, '');
          assert.equal(value.process_validated, false); assert.equal(output.kb_write_called, false);
          assert.ok(value.delivery_incomplete.some(r => r.status === 'export_unverified'));
          assert.equal(output.exported_matches_pre_director, !sc.tamper,
            'a cutoff retains the pointer without pretending that a tampered file is intact');
        } else if (sc.tamper) {
          assert.equal(value.validation_status, 'source_changed');
          assert.equal(value.final_patch, ''); assert.equal(output.exported_matches_pre_director, true);
        } else {
          assert.equal(value.process_validated, true); assert.equal(value.final_speedup, sc.rounds + 1);
          assert.equal(output.exported_matches_pre_director, true);
        }
      }
      return {...output, pass: true};
    } catch (e) { return {...output, pass: false, assertion: e.message}; }
  } finally { fs.rmSync(tmp, {recursive: true, force: true}); }
}
(async () => {
  const results = [];
  for (const sc of cases) results.push(await runCase(sc));
  const receipt = {lane: lanePath, sha256: crypto.createHash('sha256').update(source).digest('hex'),
    scope: 'CPU fixture only; inert roles, actual prompt shell block and runtime host-admission floor', results};
  if (arg('--json')) fs.writeFileSync(arg('--json'), JSON.stringify(receipt, null, 2) + '\n');
  for (const r of results) console.log(`${r.pass ? 'PASS' : 'FAIL'} ${r.case}: ${JSON.stringify(r)}`);
  if (results.some(r => !r.pass)) process.exitCode = 1;
})().catch(error => {console.error(error); process.exitCode = 1;});
