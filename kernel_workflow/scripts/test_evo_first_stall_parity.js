#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CPU-only parity check for EVO's schedule: the real lane runs greedy and evo with identical
// args and scripted answers. Every agent call must be byte-identical until a round neither
// improves nor advances; from the next round EVO develops an archive rooted at that incumbent.
// Run after scripts/build_workflows.mjs: node kernel_workflow/scripts/test_evo_first_stall_parity.js
// CPU-only: run the installed kernel_lane.js bundle (read-only) under stubbed runtime globals in
// greedy and evo modes with IDENTICAL args and IDENTICAL scripted agent answers, then record every
// agent prompt as the runtime would dispatch it (prompt + schemaInstruction, codex-neutralized).
// Writes only under OUT. No GPU, no network, no model.
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {spawnSync} = require('child_process');

const ROOT = path.resolve(__dirname, '../..');
const LANE = path.join(ROOT, 'kernel_workflow/kernel_lane.js');
const WORKFLOW = path.join(ROOT, 'kernel_workflow');
const OUT = fs.mkdtempSync(path.join(require('os').tmpdir(), 'geak-evo-first-stall-'));
const TMP = path.join(OUT, 'tmp');
fs.mkdirSync(TMP, {recursive: true});

const sha = s => crypto.createHash('sha256').update(String(s)).digest('hex');
const ENV = {...process.env, PYTHONDONTWRITEBYTECODE: '1', TMPDIR: TMP,
  GIT_AUTHOR_NAME: 'fixture', GIT_COMMITTER_NAME: 'fixture',
  GIT_AUTHOR_EMAIL: 'fixture@localhost', GIT_COMMITTER_EMAIL: 'fixture@localhost',
  GIT_AUTHOR_DATE: '2000-01-01T00:00:00Z', GIT_COMMITTER_DATE: '2000-01-01T00:00:00Z',
  GIT_CONFIG_NOSYSTEM: '1', HOME: TMP};
delete ENV.GEAK_GPU_USE_LOG;
function command(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, {encoding: 'utf8', timeout: 20000, env: ENV, ...opts});
  if (r.status !== 0) throw new Error(`${cmd} ${args.join(' ')} failed: ${r.error || r.stderr}`);
  return r.stdout;
}
const git = (dir, ...a) => command('git', ['-C', dir, ...a]);
const field = (prompt, key) => {
  const line = prompt.split('\n').find(l => l.startsWith(`- ${key}: `));
  if (!line) return undefined;
  const raw = line.slice(key.length + 4);
  try { return JSON.parse(raw); } catch (e) { return raw; }
};
const perCase = s => [{name: 'c1', baseline_ms: 2, optimized_ms: +(2 / s).toFixed(4), speedup: s, weight: 1},
                      {name: 'c2', baseline_ms: 4, optimized_ms: +(4 / s).toFixed(4), speedup: s, weight: 1}];
const verified = s => ({status: s <= 1 ? 'regression' : 'verified', correctness: 'pass', verified_geomean: s,
  verified_weighted: s, verified_arithmetic: s, per_case: perCase(s), variance_note: 'fixture'});
const engOk = s => ({status: 'success', measurement_valid: true, speedup_geomean: s, speedup_weighted: s,
  speedup_arithmetic: s, per_case: perCase(s), strategy: `fixture strategy ${s}`, notes: 'fixture notes'});

function copySources(src, dst) {
  fs.mkdirSync(dst, {recursive: true});
  for (const n of fs.readdirSync(src)) if (n.endsWith('.py')) fs.copyFileSync(path.join(src, n), path.join(dst, n));
}

async function runMode(mode, scenario, template) {
  const {schemaInstruction} = await import(path.join(ROOT, 'interface/runtime/engine/schema.mjs'));
  const {neutralizeForBackend} = await import(path.join(ROOT, 'interface/runtime/engine/config.mjs'));
  const root = path.join(OUT, 'runs', scenario.name, mode);
  fs.rmSync(root, {recursive: true, force: true});
  fs.mkdirSync(root, {recursive: true});
  const task = path.join(root, 'task'), evalDir = path.join(root, 'eval'), canonical = path.join(evalDir, 'workspace');
  fs.mkdirSync(task); fs.writeFileSync(path.join(task, 'k.py'), 'A = 0\nB = 0\n');
  git(task, 'init', '-q'); git(task, 'add', '-A'); git(task, 'commit', '-qm', 'ORIGINAL');
  fs.mkdirSync(evalDir);
  command('git', ['clone', '-q', task, canonical]);
  // Clock: deterministic fake time. Runtime semantics: initial process deadline = outer - cleanup.
  const T0 = 1000000;
  let now = T0;
  const args0 = JSON.parse(JSON.stringify(template));
  const OUTER = T0 + args0.preparation_allowance_s + args0.search_duration_s + args0.search_reserve_s + 600;
  let processDeadline = OUTER - (args0.final_cleanup_reserve_s || 0);
  const setProcessDeadline = (epoch) => {
    if (!Number.isFinite(epoch) || epoch > OUTER - (args0.final_cleanup_reserve_s || 0)) throw new Error('bad deadline');
    processDeadline = epoch;
  };
  setProcessDeadline.secondsLeft = () => processDeadline - now;
  const subst = v => typeof v === 'string' ? v.replace(/^\/arm\//, root + '/arm/').replace(/^\/input\/task$/, task)
    : Array.isArray(v) ? v.map(subst) : v && typeof v === 'object'
      ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, subst(x)])) : v;
  const args = {...subst(args0), optimizer: mode,
    kernel_path: task, eval_dir: evalDir, exp_root: path.join(root, 'exp'), workflow_dir: WORKFLOW,
    // This fixture creates its canonical repository above; it does not exercise fresh Setup.
    canonical_selection_resume: true,
    deadline_epoch: OUTER, settlement_deadline_epoch: OUTER, editable_source_paths: ['k.py'],
    population_lineage: {...args0.population_lineage, id: 'lane', root: path.join(root, 'population')},
    ...(scenario.args || {})};
  const rec = {mode, scenario: scenario.name, calls: [], logs: [], phases: [], host: []};
  const norm = s => String(s).split(root).join('<RUN>').split(WORKFLOW).join('<WORKFLOW>');
  const step = (label) => scenario.stepFor ? scenario.stepFor(label) : 60;
  const answer = async (label, prompt, opts) => {
    if (label === 'director:setup') return {eval_dir: evalDir, workspace: canonical, kernel_name: 'fix_kernel',
      device_gfx: 'gfx950', device_target: 'mi350x', device_name: 'AMD Instinct MI350X', physical_cu_count: 256,
      baseline_frozen: true, source_files: ['k.py']};
    if (label === 'tech_lead:analyze') return {kernel_type: 'triton', kernel_file: 'k.py', entry_point: 'f',
      modifiable_files: ['k.py'], bottleneck_guess: 'memory', roadmap_summary: 'fixture roadmap',
      candidate_directions: [], kk_operator: null, kk_language: 'triton', kk_refs: []};
    if (label === 'benchmark_engineer') return {commandment_path: path.join(evalDir, 'COMMANDMENT.md'),
      baseline_per_case: perCase(1), baseline_geomean_ms: 2.83, correctness_cmd: 'c', benchmark_cmd: 'b'};
    if (label === 'profile_engineer:baseline') return {bottleneck: 'memory', top_opportunities: [], summary_path: evalDir + '/p0.md'};
    if (label.startsWith('tech_lead:plan ') || label.startsWith('tech_lead:replan ')) {
      const key = label.split(' ')[1];
      const p = scenario.plans[key];
      if (scenario.onPlan) { const t = scenario.onPlan(key, {now, processDeadline}); if (t != null) now = t; }
      return p || {stop: true, reasoning: 'fixture stop', directions: []};
    }
    if (label.startsWith('eng ') || label.startsWith('deep ')) {
      const id = label.split(' ')[1].split(':')[0];
      const spec = scenario.eng[id] || {speed: 1.1};
      const ws = field(prompt, 'KERNEL_PATH'), out = field(prompt, 'OUTPUT_DIR');
      let anchor = 'HEAD';
      if (fs.existsSync(path.join(ws, '.git'))) {
        const evolution = field(prompt, 'EVOLUTION'); // EVO: host already prepared the parent
        anchor = evolution ? evolution.workspace_base_commit : git(ws, 'rev-list', '--max-parents=0', 'HEAD').trim();
      } else {
        const block = prompt.match(/```bash\n([\s\S]*?)\n```/);        // GEAK: execute the role's own setup block
        command('bash', ['-e', '-c', block[1]]);
      }
      fs.writeFileSync(path.join(ws, 'k.py'), `A = ${JSON.stringify(id)}\nB = ${spec.speed}\n`);
      fs.writeFileSync(path.join(out, 'best_patch.diff'), git(ws, 'diff', '--binary', anchor, '--', '.'));
      return spec.engineer || engOk(spec.speed);
    }
    if (label.startsWith('verify ')) {
      const id = label.slice(7).split(' ')[0];
      const s = (scenario.verify || {})[id];
      return s == null ? {status: 'apply_failed', correctness: 'not_checked', verified_geomean: 0} : verified(s);
    }
    if (label.startsWith('integrate ')) {
      const r = label.slice(10);
      const spec = (scenario.integrate || {})[r];
      if (!spec) return {attempted: true, conclusion: 'no_improvement', notes: 'fixture'};
      const ws = path.join(field(prompt, 'INTEGRATE_DIR'), 'ws');
      copySources(field(prompt, 'CANONICAL'), ws);
      git(ws, 'init', '-q'); git(ws, 'add', '-A'); git(ws, 'commit', '-qm', 'integration base');
      fs.writeFileSync(path.join(ws, 'k.py'), `A = "integrated ${r}"\nB = ${spec.speed}\n`);
      const patch = path.join(field(prompt, 'INTEGRATE_DIR'), 'integrated_patch.diff');
      fs.writeFileSync(patch, git(ws, 'diff', '--binary', 'HEAD', '--', '.'));
      return {attempted: true, conclusion: spec.conclusion, best: {patches: [], geomean: spec.speed,
        arithmetic: spec.speed, weighted: spec.speed, per_case: perCase(spec.speed), patch_file: patch}};
    }
    if (label.startsWith('commit ')) {
      const block = prompt.match(/```bash\n([\s\S]*?)\n```/);
      command('bash', ['-e', '-c', block[1]]);
      return {committed: true, current_best_diff: evalDir + '/current_best.diff'};
    }
    if (label.startsWith('reprofile ')) return {bottleneck: 'compute', top_opportunities: [],
      summary_path: `${evalDir}/p_${label.slice(10)}.md`, shift_note: 'fixture shift'};
    if (label.startsWith('tech_lead:memory ')) return {insights: ['fixture insight'],
      ledger: [{direction: label.slice(17), verdict: 'partial'}], bottleneck_now: 'compute', suggest_next: 'fixture next'};
    if (label.startsWith('storage:reclaim')) return {ok: true, note: 'reclaimed'};
    if (label === 'tech_lead:report') return {final_speedup_geomean: field(prompt, 'CUMULATIVE_SPEEDUP'),
      final_speedup_weighted: field(prompt, 'CUMULATIVE_SPEEDUP'),
      report_path: evalDir + '/tech_lead_report.md', final_patch: evalDir + '/final_patch.diff',
      rounds: field(prompt, 'HISTORY').rounds.length};
    if (label === 'director:validate') return {director_verified_speedup_geomean: field(prompt, 'TECH_LEAD_REPORTED_GEOMEAN'),
      director_verified_speedup_weighted: field(prompt, 'TECH_LEAD_REPORTED_WEIGHTED') ?? field(prompt, 'TECH_LEAD_REPORTED_GEOMEAN'),
      validation_status: 'accepted', timing_basis: 'clean'};
    if (label === 'kb:cite') return {filed: 0};
    throw new Error(`unanswered label ${label}`);
  };
  const agent = async (p, o = {}) => {
    const label = o.label || '';
    const dispatched = neutralizeForBackend(o.schema ? p + schemaInstruction(o.schema) : p, 'codex');
    rec.calls.push({seq: rec.calls.length, label, phase: o.phase || '', at: now - T0,
      process_deadline_minus_now: processDeadline - now,
      prompt: norm(dispatched), prompt_sha256: sha(norm(dispatched))});
    const r = await answer(label, p, o);
    now += step(label);
    return r === undefined ? null : JSON.parse(JSON.stringify(r));
  };
  Object.defineProperty(agent, 'processTimeoutSupported', {value: true});
  Object.defineProperty(agent, 'processDeadlineActive', {get: () => true});
  const runProcess = async (req) => {
    rec.host.push(norm([req.cmd, ...(req.args || [])].join(' ')).slice(0, 300));
    const r = spawnSync(req.cmd, req.args, {encoding: 'utf8', timeout: 20000, input: req.input,
      cwd: req.cwd, env: {...ENV, ...(req.env || {})}});
    return {status: r.error ? 'failed' : 'completed', code: r.status, stdout: r.stdout || '',
      stderr: r.stderr || '', group_cleared: true, strays_killed: false, stdio_held: false};
  };
  const parallel = async (thunks) => Promise.all(thunks.map(t => Promise.resolve().then(t).catch(() => null)));
  // Same semantics as run_workflow.mjs pipeline(): items concurrent, stages sequential per item.
  const pipeline = async (items, ...stages) => Promise.all((items || []).map(async (item, idx) => {
    let prev = item;
    for (const st of stages) {
      try { prev = await st(prev, item, idx); }
      catch (e) { rec.logs.push(`pipeline item ${idx} failed: ${e && e.message}`); return null; }
    }
    return prev;
  }));
  const body = fs.readFileSync(LANE, 'utf8').replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  const fn = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', 'runProcess',
    'setTimeout', 'clearTimeout', 'setProcessDeadline', body);
  let result = null, error = null;
  try {
    result = await fn(agent, parallel, pipeline, t => rec.phases.push(t), m => rec.logs.push(norm(m)),
      async () => { throw new Error('no nesting'); }, args, {total: null, spent: () => 0, remaining: () => Infinity},
      runProcess, setTimeout, clearTimeout, setProcessDeadline);
  } catch (e) { error = String(e && e.stack || e); }
  rec.result = result ? JSON.parse(norm(JSON.stringify({...result, llm_timeline: undefined}))) : null;
  rec.error = error ? norm(error) : null;
  const readArchive = name => fs.existsSync(path.join(evalDir, name, 'archive.json'))
    ? JSON.parse(norm(fs.readFileSync(path.join(evalDir, name, 'archive.json'), 'utf8'))) : null;
  rec.archive = readArchive('evolution');
  rec.continuation_archive = readArchive('evolution_continuation');
  return rec;
}


const assert = require('assert/strict');
const TEMPLATE = {agent_retries: 1, agent_timeout_ms: 7200000, budget: 96, delivery_facts: true,
  delivery_reserve_s: 0, dra_enabled: false, engineer_reserve_s: 0, eval_dir: '/arm/eval', exp_root: '/arm/exp',
  expected_gfx: 'gfx950', expected_target: 'mi350x', final_cleanup_reserve_s: 180,
  final_validation_reserve_s: 2400, gpu_ids: '0', gpu_mode: 'pool', kb_artifacts_dir: '/arm/local_kb_artifacts',
  kb_mode: 'local', kb_remote: 'off', kb_store_dir: '/arm/local_kb_store', kernel_path: '/input/task',
  learned_dir: '/arm/local_learned', max_forced_replans: 96, max_no_improve: 96, min_improve: 0.02,
  mode: 'optimize', no_stop_s: 900, population_lineage: {id: 'lane', imports: [], root: '/arm/population'},
  preparation_allowance_s: 10800, progress_delta: 0.02, required_gpu_count: 1, search_duration_s: 28800,
  search_reserve_s: 3600, target_language: 'triton', task: 'Optimize the fixture kernel.',
  update_experience: 'off', use_expert_skills: false, use_learned_kb: false,
  wait_for_process_settlement: true, warm_start: 'off'};
const d = (id, specialty, extra = {}) => ({id, title: `T ${id}`, specialty, focus_files: ['k.py'],
  expected_speedup: 1.3, prompt: `fixture prompt for ${id}`, ...extra});
const cutAt = stopAt => (k, c) => (k === stopAt ? c.processDeadline - 600 : null);
const SCENARIOS = [
  {name: 'improving', divergeAt: null, plans: {
      r1: {stop: false, reasoning: 'r1', directions: [d('r1_d0', 'memory'), d('r1_d1', 'algorithm')]},
      r2: {stop: false, reasoning: 'r2', directions: [d('r2_d0', 'compute')]}},
    onPlan: cutAt('r3'), eng: {r1_d0: {speed: 1.30}, r1_d1: {speed: 0.95}, r2_d0: {speed: 1.40}},
    verify: {r1_d0: 1.30, r1_d1: 0.95, r2_d0: 1.40}, integrate: {r1: {conclusion: 'no_improvement', speed: 1.25}}},
  {name: 'stall_after_r2', divergeAt: 'tech_lead:plan r3', incumbent: 1.30, plans: {
      r1: {stop: false, reasoning: 'r1', directions: [d('r1_d0', 'memory')]},
      r2: {stop: false, reasoning: 'r2', directions: [d('r2_d0', 'compute')]},
      r3: {stop: false, reasoning: 'r3', directions: [d('r3_d0', 'algorithm')]},
      r4: {stop: false, reasoning: 'r4', directions: [d('r4_d0', 'memory', {parent_id: 'r3_d0', approach: 'alt'})]}},
    onPlan: cutAt('r5'), eng: {r1_d0: {speed: 1.30}, r2_d0: {speed: 1.29}, r3_d0: {speed: 1.20}, r4_d0: {speed: 1.45}},
    verify: {r1_d0: 1.30, r2_d0: 1.29, r3_d0: 1.20, r4_d0: 1.45}},
  {name: 'stall_in_r1', divergeAt: 'tech_lead:plan r2', incumbent: 1, plans: {
      r1: {stop: false, reasoning: 'r1', directions: [d('r1_d0', 'memory')]},
      r2: {stop: false, reasoning: 'r2', directions: [d('r2_d0', 'compute')]}},
    onPlan: cutAt('r3'), eng: {r1_d0: {speed: 0.90}, r2_d0: {speed: 1.25}}, verify: {r1_d0: 0.90, r2_d0: 1.25}},
  // Default-off factor: after the stall an engineer returns its unmeasured final implementation.
  {name: 'development_fallback', divergeAt: 'tech_lead:plan r3', incumbent: 1.30,
    args: {evolution_return_development: true}, plans: {
      r1: {stop: false, reasoning: 'r1', directions: [d('r1_d0', 'memory')]},
      r2: {stop: false, reasoning: 'r2', directions: [d('r2_d0', 'compute')]},
      r3: {stop: false, reasoning: 'r3', directions: [d('r3_d0', 'algorithm')]},
      r4: {stop: false, reasoning: 'r4', directions: [d('r4_d0', 'memory', {parent_id: 'r3_d0', approach: 'continue'})]}},
    onPlan: cutAt('r5'), eng: {r1_d0: {speed: 1.30}, r2_d0: {speed: 1.29}, r4_d0: {speed: 1.45},
      r3_d0: {speed: 1.20, engineer: {status: 'partial', measurement_valid: false, speedup_geomean: 0,
        speedup_weighted: 0, speedup_arithmetic: 0, per_case: [], strategy: 'fixture unmeasured attempt',
        notes: 'unmeasured development fallback; zeros are required sentinels, not measurements'}}},
    verify: {r1_d0: 1.30, r2_d0: 1.29, r4_d0: 1.45}},
];
const FALLBACK = 'Development fallback';
(async () => {
  for (const sc of SCENARIOS) {
    const g = await runMode('greedy', sc, TEMPLATE), e = await runMode('evo', sc, TEMPLATE);
    assert.equal(g.error, null, `${sc.name} greedy: ${g.error}`);
    assert.equal(e.error, null, `${sc.name} evo: ${e.error}`);
    let first = null;
    for (let i = 0; i < Math.max(g.calls.length, e.calls.length); i++) {
      const a = g.calls[i], b = e.calls[i];
      if (!a || !b || a.label !== b.label || a.prompt_sha256 !== b.prompt_sha256) { first = i; break; }
    }
    if (sc.divergeAt === null) {
      assert.equal(first, null, `${sc.name}: evo diverged at call ${first} (${e.calls[first]?.label})`);
      assert.equal(e.continuation_archive, null, `${sc.name}: no archive while every round improves`);
    } else {
      assert.notEqual(first, null, `${sc.name}: evo never diverged`);
      assert.equal(e.calls[first].label, sc.divergeAt, `${sc.name}: first divergence`);
      assert.equal(g.calls[first].label, sc.divergeAt, `${sc.name}: greedy label at divergence`);
      const archive = e.continuation_archive;
      assert.ok(archive, `${sc.name}: evolution archive created at the stall`);
      assert.equal(archive.baseline.origin, 'greedy_incumbent');
      assert.equal(archive.baseline.speedup, sc.incumbent);
      const firstEvoRound = Number(sc.divergeAt.split(' r')[1]);
      assert.ok(archive.entries.length > 0 && archive.entries.every(r => Number(r.id.slice(1).split('_')[0]) >= firstEvoRound),
        `${sc.name}: the archive holds only rounds developed after the stall`);
      assert.equal(field(e.calls[first].prompt, 'HISTORY').evolution_start.first_round, firstEvoRound,
        `${sc.name}: the first differentiated round is recorded`);
    }
    if (sc.args?.evolution_return_development !== true) {
      assert.ok(![...g.calls, ...e.calls].some(c => c.prompt.includes(FALLBACK)), `${sc.name}: fallback is off by default`);
    }
    console.log(`  ok: ${sc.name} (${g.calls.length}/${e.calls.length} calls, first divergence ${first === null ? 'none' : e.calls[first].label})`);
  }
  {
    const sc = SCENARIOS.find(s => s.name === 'development_fallback');
    const g = await runMode('greedy', sc, TEMPLATE), e = await runMode('evo', sc, TEMPLATE);
    assert.ok(!g.calls.some(c => c.prompt.includes(FALLBACK)), 'greedy never receives the fallback');
    const engineer = id => e.calls.find(c => /^(eng|deep) /.test(c.label) && c.label.split(' ')[1].startsWith(`${id}:`));
    for (const id of ['r1_d0', 'r2_d0']) assert.ok(!engineer(id).prompt.includes(FALLBACK), `${id}: no fallback before the stall`);
    for (const id of ['r3_d0', 'r4_d0']) assert.ok(engineer(id).prompt.includes(FALLBACK), `${id}: fallback after the stall`);
    assert.match(engineer('r3_d0').prompt, /never justifies skipping an available evaluation/);
    assert.match(engineer('r3_d0').prompt, /a correct measured but slower source stays under the existing return rules/);
    assert.match(engineer('r3_d0').prompt, /a re-measured unchanged starting program is not such development/);
    assert.match(engineer('r3_d0').prompt, /Never edit the canonical incumbent/);
    assert.match(engineer('r3_d0').prompt, /native reproducible-gain requirements for qualified export and delivery remain unchanged/);
    const plan = r => e.calls.find(c => c.label === `tech_lead:plan ${r}`).prompt;
    assert.doesNotMatch(plan('r3'), /roles\/evolution\.md/, 'empty archive: the first post-stall plan has no archive policy');
    assert.match(plan('r4'), /roles\/evolution\.md/, 'the returned attempt makes the next plan archive-aware');
    assert.ok(e.calls.some(c => c.label === 'verify r3_d0'), 'an unmeasured return is still sent to the ordinary verifier');
    const rows = Object.fromEntries(e.continuation_archive.entries.map(r => [r.id, r]));
    const attempt = rows.r3_d0;
    assert.ok(attempt.patch_ref && attempt.patch_ref.sha256 !== sha(''), 'the actual attempt is a retained non-empty source');
    assert.equal(attempt.retention_error, null);
    assert.equal(attempt.verified, false);
    assert.equal(attempt.status, 'unverified');
    assert.equal(attempt.speedup, null, 'no fitness without verification');
    assert.equal(attempt.engineer.measurement_valid, false);
    const child = rows.r4_d0;
    assert.equal(child.parent_id, 'r3_d0');
    assert.equal(child.evolution_context.initialization.parent_patch_ref.sha256, attempt.patch_ref.sha256,
      'the unverified attempt is materialized as the selected parent');
    assert.equal(child.verified, true);
    assert.equal(e.result.final_speedup, 1.45, 'only the verified descendant is delivered');
    console.log('  ok: development_fallback returns the unmeasured attempt as an unverified selectable row');
  }
  const evo = (await runMode('evo', SCENARIOS[1], TEMPLATE));
  const rows = Object.fromEntries(evo.continuation_archive.entries.map(r => [r.id, r]));
  assert.equal(rows.r4_d0.parent_id, 'r3_d0', 'a slower retained alternative is developed as a parent');
  assert.equal(rows.r3_d0.status, 'correct');
  assert.equal(evo.result.final_speedup, 1.45, 'delivery keeps the developed improvement');
  fs.rmSync(OUT, {recursive: true, force: true});
  console.log('PASS: EVO stays byte-identical to greedy until its first non-improving round, then evolves from that incumbent. CPU fixture only.');
})().catch(error => { console.error(error); process.exit(1); });
