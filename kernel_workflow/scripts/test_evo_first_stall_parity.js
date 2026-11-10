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
  const T0 = scenario.nativeRuntime ? Math.floor(Date.now() / 1000) : 1000000;
  let now = T0;
  const args0 = JSON.parse(JSON.stringify(template));
  const OUTER = T0 + args0.preparation_allowance_s + args0.search_duration_s + args0.search_reserve_s + 600;
  let processDeadline = OUTER - (args0.final_cleanup_reserve_s || 0);
  const setProcessDeadline = (epoch) => {
    if (!Number.isFinite(epoch) || epoch > OUTER - (args0.final_cleanup_reserve_s || 0)) throw new Error('bad deadline');
    if (scenario.onDeadline) scenario.onDeadline(epoch, {now, processDeadline});
    processDeadline = epoch;
  };
  setProcessDeadline.secondsLeft = () => processDeadline - now;
  const subst = v => typeof v === 'string' ? v.replace(/^\/arm\//, root + '/arm/').replace(/^\/input\/task$/, task)
    : Array.isArray(v) ? v.map(subst) : v && typeof v === 'object'
      ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, subst(x)])) : v;
  const args = {...subst(args0), optimizer: mode, evolution_return_development: false,
    kernel_path: task, eval_dir: evalDir, exp_root: path.join(root, 'exp'), workflow_dir: WORKFLOW,
    // This fixture creates its canonical repository above; it does not exercise fresh Setup.
    canonical_selection_resume: true,
    deadline_epoch: OUTER, settlement_deadline_epoch: OUTER, editable_source_paths: ['k.py'],
    population_lineage: {...args0.population_lineage, id: 'lane', root: path.join(root, 'population')},
    ...(scenario.args || {})};
  const rec = {mode, scenario: scenario.name, calls: [], logs: [], phases: [], host: [], planner_observations: []};
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
      baseline_per_case: perCase(1), baseline_geomean_ms: 2.83, correctness_cmd: 'c', benchmark_cmd: 'b',
      reliable: true, ...(scenario.bench || {})};
    if (label === 'profile_engineer:baseline') {
      fs.writeFileSync(evalDir + '/p0.md', 'fixture profile before engineering\n');
      return Object.hasOwn(scenario, 'profile') ? scenario.profile
        : {bottleneck: 'memory', top_opportunities: [], summary_path: evalDir + '/p0.md'};
    }
    if (label.startsWith('tech_lead:plan ') || label.startsWith('tech_lead:replan ')) {
      const key = label.split(' ')[1];
      if (scenario.failPlan === key) throw new Error('fixture planner failure');
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
      if (scenario.beforeEngineerReturn) await scenario.beforeEngineerReturn(id, {ws, out, rec});
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
      if (scenario.commitFail === label.slice(7)) return {committed: false, notes: 'fixture commit failure'};
      const block = prompt.match(/```bash\n([\s\S]*?)\n```/);
      command('bash', ['-e', '-c', block[1]]);
      return {committed: true, current_best_diff: evalDir + '/current_best.diff'};
    }
    if (label.startsWith('reprofile ')) {
      const file = `${evalDir}/p_${label.slice(10)}.md`;
      fs.writeFileSync(file, 'fixture profile after ' + label + '\n');
      return {bottleneck: 'compute', top_opportunities: [], summary_path: file, shift_note: 'fixture shift'};
    }
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
    if (o.plannerObservation) rec.planner_observations.push(JSON.parse(norm(JSON.stringify({
      ...o.plannerObservation, label}))));
    const dispatched = neutralizeForBackend(o.schema ? p + schemaInstruction(o.schema) : p, 'codex');
    rec.calls.push({seq: rec.calls.length, label, phase: o.phase || '', at: now - T0,
      process_deadline_minus_now: processDeadline - now,
      prompt: norm(dispatched), prompt_sha256: sha(norm(dispatched)), role_prompt: norm(p)});
    if (scenario.onCall) await scenario.onCall(label, p, {rec, root, evalDir, canonical, args, command, git, field});
    const r = await answer(label, p, o);
    if (scenario.onResult) await scenario.onResult(label, r, {rec, root, evalDir, canonical, args, setNow: value => {now = value;}, processDeadline});
    if (!scenario.nativeRuntime) now += step(label);
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
  const source = scenario.laneSource || fs.readFileSync(LANE, 'utf8');
  const body = source.replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  const fn = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', 'runProcess',
    'setTimeout', 'clearTimeout', 'setProcessDeadline', body);
  let result = null, error = null;
  try {
    if (scenario.nativeRuntime) {
      const {createRuntime} = await import(path.join(ROOT, 'interface/runtime/engine/run_workflow.mjs'));
      const complete = (schema, value) => {
        if (schema.type === 'object') {
          const out = {...value};
          for (const key of schema.required || []) if (out[key] === undefined)
            out[key] = complete(schema.properties[key], undefined);
          return out;
        }
        if (value !== undefined) return value;
        if (schema.enum) return schema.enum[0];
        return schema.type === 'array' ? [] : schema.type === 'boolean' ? false
          : schema.type === 'number' || schema.type === 'integer' ? 0 : '';
      };
      const rt = createRuntime({concurrency: scenario.concurrency || 8, schemaRetries: 0,
        log: message => rec.logs.push(norm(message)), backend: {name: 'codex', async runAgent({prompt, label}) {
          const match = prompt.match(/The JSON MUST validate against this JSON Schema:\n\n```json\n([\s\S]*?)\n```/);
          const schema = match ? JSON.parse(match[1]) : null;
          const value = await agent(prompt, {label});
          return {text: JSON.stringify(schema ? complete(schema, value) : value)};
        }}});
      const fixture = path.join(root, 'native_lane.js');
      fs.writeFileSync(fixture, source);
      const keys = ['GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL'];
      const saved = Object.fromEntries(keys.map(key => [key, process.env[key]]));
      for (const key of keys) process.env[key] = ENV[key];
      try { result = await rt.runScript(fixture, args, 0); }
      finally { for (const key of keys) saved[key] === undefined ? delete process.env[key] : process.env[key] = saved[key]; }
      rec.runtime_state = rt.state;
    } else {
      result = await fn(agent, parallel, pipeline, t => rec.phases.push(t), m => rec.logs.push(norm(m)),
        async () => { throw new Error('no nesting'); }, args, {total: null, spent: () => 0, remaining: () => Infinity},
        runProcess, setTimeout, clearTimeout, setProcessDeadline);
    }
  } catch (e) { error = String(e && e.stack || e); }
  rec.result = result ? JSON.parse(norm(JSON.stringify({...result, llm_timeline: undefined}))) : null;
  rec.error = error ? norm(error) : null;
  const readArchive = name => fs.existsSync(path.join(evalDir, name, 'archive.json'))
    ? JSON.parse(norm(fs.readFileSync(path.join(evalDir, name, 'archive.json'), 'utf8'))) : null;
  rec.archive = readArchive('evolution');
  rec.continuation_archive = readArchive('evolution_continuation');
  const overlapFile = path.join(evalDir, 'bounded_overlap.json');
  rec.bounded_overlap = fs.existsSync(overlapFile) ? JSON.parse(norm(fs.readFileSync(overlapFile, 'utf8'))) : null;
  rec.fixture_root = root;
  assert.ok(!fs.readdirSync(evalDir).some(n => /_inputs\.json$/.test(n)), 'observer writes no EVAL input artifact');
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
if (require.main === module) (async () => {
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
      assert.ok(![...g.calls, ...e.calls].some(c => c.prompt.includes(FALLBACK)), `${sc.name}: explicit development-return opt-out is honored`);
    }
    console.log(`  ok: ${sc.name} (${g.calls.length}/${e.calls.length} calls, first divergence ${first === null ? 'none' : e.calls[first].label})`);
  }
  {
    const sc = SCENARIOS.find(s => s.name === 'development_fallback');
    const g = await runMode('greedy', sc, TEMPLATE), e = await runMode('evo', sc, TEMPLATE);
    assert.ok(!g.calls.some(c => c.prompt.includes(FALLBACK)), 'greedy never receives the fallback');
    const defaultEvo = await runMode('evo', {...sc, args: {...sc.args, evolution_return_development: undefined}}, TEMPLATE);
    assert.deepEqual(defaultEvo.calls, e.calls, 'omitted development-return matches explicit true');
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
  {
    // Default-off stall-source import: the stall round's retained sources become archive rows.
    const plans = {r1: {stop: false, reasoning: 'r1', directions: [d('r1_d0', 'memory')]},
      r2: {stop: false, reasoning: 'r2', directions: [d('r2_d0', 'compute'), d('custom_alt', 'algorithm')]},
      r3: {stop: false, reasoning: 'r3', directions: [d('r3_d0', 'memory', {parent_id: 'r2_d0', approach: 'requalify'})]}};
    // r2_d0 reports a valid local 1.40 but its verifier fails; custom_alt's 1.31 claim verifies at 0.8.
    const sc = {name: 'stall_sources', plans, onPlan: cutAt('r4'),
      eng: {r1_d0: {speed: 1.30}, r2_d0: {speed: 1.40}, custom_alt: {speed: 1.31}, r3_d0: {speed: 1.45}},
      verify: {r1_d0: 1.30, custom_alt: 0.8, r3_d0: 1.45}};
    const lr = {...sc, name: 'stall_sources_on', args: {evolution_import_stall_sources: true}};
    const g = await runMode('greedy', sc, TEMPLATE), off = await runMode('evo', sc, TEMPLATE);
    const on = await runMode('evo', lr, TEMPLATE);
    for (const r of [g, off, on]) assert.equal(r.error, null, r.error);
    const divergence = (a, b) => {
      for (let i = 0; i < Math.max(a.calls.length, b.calls.length); i++) {
        if (a.calls[i]?.label !== b.calls[i]?.label || a.calls[i]?.prompt_sha256 !== b.calls[i]?.prompt_sha256) return b.calls[i]?.label;
      }
      return null;
    };
    assert.equal(divergence(g, off), 'tech_lead:plan r3');
    assert.equal(divergence(off, on), 'tech_lead:plan r3', 'the import changes nothing before the first post-stall plan');
    const plan = (r, k) => r.calls.find(c => c.label === `tech_lead:plan ${k}`).prompt;
    assert.doesNotMatch(plan(off, 'r3'), /roles\/evolution\.md/, 'off: the first post-stall archive is empty');
    assert.match(plan(on, 'r3'), /roles\/evolution\.md/, 'on: the first post-stall plan sees the stall sources');
    const rowsOf = r => Object.fromEntries(r.continuation_archive.entries.map(x => [x.id, x]));
    const rows = rowsOf(on), offRows = rowsOf(off);
    assert.ok(!Object.keys(rows).some(id => id.startsWith('r1_')), 'older rounds have another base and stay out');
    const pin = fs.readFileSync(path.join(OUT, 'runs', 'stall_sources_on', 'evo', 'eval', 'delivery', 'sources',
      'round_2_engineer_0.diff'));
    assert.equal(rows.r2_d0.parent_id, 'ORIGINAL');
    assert.equal(rows.r2_d0.verified, false);
    assert.equal(rows.r2_d0.status, 'unverified');
    assert.equal(rows.r2_d0.speedup, null, 'a failed verification gives no fitness');
    assert.equal(rows.r2_d0.engineer.measurement_valid, true, 'the engineer report travels with the row');
    assert.equal(rows.r2_d0.patch_ref.sha256, sha(pin), 'the pinned stall-round source, not the mutable workspace patch');
    assert.match(rows.r2_d0.original_patch, /delivery\/sources\/round_2_engineer_0\.diff$/);
    assert.equal(rows.r2_d1.title, 'T custom_alt', 'a planner-chosen id is renamed by its engineer slot');
    assert.equal(rows.r2_d1.verified, true);
    assert.equal(rows.r2_d1.speedup, 0.8, 'settled regression verification keeps its fitness');
    assert.equal(rows.r2_d1.status, 'slower');
    assert.equal(rows.r2_d1.verifier.status, 'regression');
    assert.equal(on.continuation_archive.best_id, 'r3_d0', 'imports never take the best or default-parent rank');
    assert.equal(rows.r3_d0.parent_id, 'r2_d0');
    assert.equal(rows.r3_d0.evolution_context.initialization.parent_patch_ref.sha256, rows.r2_d0.patch_ref.sha256,
      'the unverified stall source is materialized as the selected parent');
    assert.equal(offRows.r3_d0.parent_id, 'ORIGINAL', 'off: the same request falls back to the incumbent');
    assert.ok(offRows.r3_d0.evolution_context.reference_errors.length > 0);
    assert.ok(!Object.keys(offRows).some(id => id.startsWith('r2_')));
    assert.ok(on.logs.some(m => m.includes('stall-round sources on the archive base')));
    assert.ok(!off.logs.some(m => m.includes('stall-round sources on the archive base')));
    for (const r of [g, off, on]) assert.equal(r.result.final_speedup, 1.45);
    // A verified stall source above the incumbent is still delivered, now through its archive row.
    const dv = {name: 'stall_sources_delivery', onPlan: cutAt('r4'), plans: {
        r1: plans.r1, r2: {stop: false, reasoning: 'r2', directions: [d('r2_d0', 'compute')]},
        r3: {stop: false, reasoning: 'r3', directions: [d('r3_d0', 'algorithm')]}},
      eng: {r1_d0: {speed: 1.30}, r2_d0: {speed: 1.31}, r3_d0: {speed: 1.20}},
      verify: {r1_d0: 1.30, r2_d0: 1.31, r3_d0: 1.20}};
    const dOff = await runMode('evo', dv, TEMPLATE);
    const dOn = await runMode('evo', {...dv, name: 'stall_sources_delivery_on',
      args: {evolution_import_stall_sources: true}}, TEMPLATE);
    for (const r of [dOff, dOn]) {
      assert.equal(r.error, null, r.error);
      assert.equal(r.result.final_speedup, 1.31, 'end-of-run delivery keeps the verified stall source');
      assert.equal(r.result.delivery.delivered, 'round_2_engineer_0');
      assert.ok(r.result.delivery.candidates.every(c => c.archive_record_status !== 'failed'
        && c.archive_selection_status !== 'failed'), 'delivery records and selects the same row');
      assert.equal(r.continuation_archive.best_id, 'r2_d0');
    }
    assert.equal(rowsOf(dOn).r2_d0.parent_id, 'ORIGINAL');
    // The switch can be the last event before the search deadline: no later round adopts the archive.
    const late = {name: 'stall_sources_late', plans: {r1: plans.r1,
        r2: {stop: false, reasoning: 'r2', directions: [d('r2_d0', 'compute')]}},
      onPlan: (k, c) => (k === 'r2' ? c.processDeadline - 960 : null),
      stepFor: label => (label === 'storage:reclaim r2' ? 1000 : 60),
      eng: {r1_d0: {speed: 1.30}, r2_d0: {speed: 1.31}}, verify: {r1_d0: 1.30, r2_d0: 1.31}};
    for (const args of [{}, {evolution_import_stall_sources: true}]) {
      const r = await runMode('evo', {...late, name: `stall_sources_late_${Object.keys(args).length}`, args}, TEMPLATE);
      assert.equal(r.error, null, r.error);
      assert.ok(r.continuation_archive, 'the switch happened');
      assert.ok(!r.calls.some(c => c.label === 'tech_lead:plan r3'), 'no round starts after the switch');
      assert.equal(r.result.final_speedup, 1.31, 'delivery still uses the pinned stall source');
    }
    // A failed commit leaves round 2 on the same base, and its progress defers the switch to round 3:
    // only the stall round itself is imported.
    const prior = {name: 'stall_sources_prior', onPlan: cutAt('r4'), commitFail: 'r2', plans: {
        r1: plans.r1, r2: {stop: false, reasoning: 'r2', directions: [d('r2_d0', 'compute')]},
        r3: {stop: false, reasoning: 'r3', directions: [d('r3_d0', 'algorithm')]}},
      eng: {r1_d0: {speed: 1.30}, r2_d0: {speed: 1.40}, r3_d0: {speed: 1.31}},
      verify: {r1_d0: 1.30, r2_d0: 1.40, r3_d0: 1.31}};
    const pOn = await runMode('evo', {...prior, name: 'stall_sources_prior_on',
      args: {evolution_import_stall_sources: true}}, TEMPLATE);
    assert.equal(pOn.error, null, pOn.error);
    assert.ok(pOn.logs.some(m => m.includes('Round 2: commit failed')), 'round 2 kept the base');
    const r4 = pOn.calls.find(c => c.label === 'tech_lead:plan r4').prompt;
    assert.equal(field(r4, 'HISTORY').evolution_start.after_round, 3, 'the switch follows round 3');
    assert.deepEqual(field(r4, 'EVOLUTION_ENTRIES').map(x => x.id), ['r3_d0'], 'only the stall round is imported');
    assert.ok(pOn.logs.some(m => m.includes('round_3_engineer_0 -> r3_d0: imported')));
    assert.ok(!pOn.logs.some(m => m.includes('round_2_engineer_0 ->')), 'the same-base earlier round stays delivery-only');
    assert.equal(pOn.result.final_speedup, 1.40, 'delivery still finds the earlier verified source');
    const bad = await runMode('greedy', {...sc, name: 'stall_sources_greedy', args: {evolution_import_stall_sources: true}}, TEMPLATE);
    assert.match(bad.error || '', /evolution_import_stall_sources requires optimizer=evo/);
    console.log('  ok: stall-round sources enter the first-stall archive only when enabled; delivery stays consistent');
  }
  {
    // Common preparation contract: search refuses an empty or unreliable native baseline, as
    // preparation already does, and a supplied prepared reference also reaches the Director.
    for (const bench of [{reliable: false}, {baseline_per_case: []}, {reliable: undefined}]) {
      const r = await runMode('greedy', {...SCENARIOS[0], name: 'bench_contract', bench}, TEMPLATE);
      assert.match(r.error || '', /no reliable non-empty baseline recorded/, JSON.stringify(bench));
      assert.ok(!r.calls.some(c => c.label.startsWith('tech_lead:plan')), 'no search starts');
    }
    const ref = {path: '/prepared/original.json', sha256: 'a'.repeat(64)};
    const withRef = await runMode('greedy', {...SCENARIOS[0], name: 'prepared_ref', args: {prepared_original_ref: ref}}, TEMPLATE);
    const setupPrompt = c => c.calls.find(x => x.label === 'director:setup').prompt;
    assert.deepEqual(field(setupPrompt(withRef), 'PREPARED_ORIGINAL_REF'), ref, 'the Director receives the supplied reference');
    const plain = await runMode('greedy', {...SCENARIOS[0], name: 'prepared_ref_off'}, TEMPLATE);
    assert.equal(field(setupPrompt(plain), 'PREPARED_ORIGINAL_REF'), undefined, 'no reference, no input');
    console.log('  ok: baseline contract and prepared-reference Setup input');
  }
  const evo = (await runMode('evo', SCENARIOS[1], TEMPLATE));
  const rows = Object.fromEntries(evo.continuation_archive.entries.map(r => [r.id, r]));
  assert.equal(rows.r4_d0.parent_id, 'r3_d0', 'a slower retained alternative is developed as a parent');
  assert.equal(rows.r3_d0.status, 'correct');
  assert.equal(evo.result.final_speedup, 1.45, 'delivery keeps the developed improvement');
  {
    const sc = {...SCENARIOS[0], name: 'planner_inputs', failPlan: 'r1', profile: null,
      args: {task: 'Optimize the fixture kernel λ🧪.'}, plans: {
        ...SCENARIOS[0].plans, 'r1#1': SCENARIOS[0].plans.r1}};
    const off = await runMode('greedy', sc, TEMPLATE);
    const on = await runMode('greedy', {...sc, args: {...sc.args, record_planner_inputs: true}}, TEMPLATE);
    for (const actual of [off, on]) assert.equal(actual.error, null, actual.error);
    assert.deepEqual(on.calls, off.calls, 'observer ON/OFF preserves every dispatched prompt and call');
    assert.deepEqual(on.host, off.host, 'observer adds no host subprocess');
    assert.deepEqual(on.result, off.result, 'observer ON/OFF preserves native outcomes');
    assert.deepEqual(off.planner_observations, [], 'default OFF attaches no metadata');
    const first = on.planner_observations.find(r => r.round === 1 && r.logical_attempt === 'initial');
    const replan = on.planner_observations.find(r => r.round === 1 && r.logical_attempt === 'replan1');
    const later = on.planner_observations.find(r => r.round === 2 && r.logical_attempt === 'initial');
    assert.ok(first && replan && later, 'failed initial call, forced replan and later call retain logical identity');
    for (const row of on.planner_observations) {
      assert.equal(row.eval_dir, '<RUN>/eval');
      assert.equal(JSON.parse(row.components.role_prompt.serialized), on.calls.find(c => c.label === row.label).role_prompt);
    }
    const c = first.components;
    assert.equal(c.prepared_original_ref.state, 'absent');
    assert.equal(c['inputs.STOP_NOT_PERMITTED'].state, 'absent');
    assert.equal(c['inputs.PROFILE_SUMMARY'].state, 'null');
    assert.equal(c['inputs.KK_OPERATOR'].state, 'empty');
    assert.equal(c['inputs.KK_REFS'].state, 'empty');
    assert.equal(c.TASK.state, 'present');
    assert.equal(JSON.parse(c['inputs.HISTORY'].serialized).rounds.length, 0, 'snapshot survives later history mutation');
    assert.ok(JSON.parse(later.components['inputs.HISTORY'].serialized).rounds.length > 0);
    assert.equal(replan.components['inputs.STOP_NOT_PERMITTED'].state, 'present');
    console.log('  ok: planner observer ON/OFF prompt/call parity, frozen states and eval/logical scope');
  }
  fs.rmSync(OUT, {recursive: true, force: true});
  console.log('PASS: EVO stays byte-identical to greedy until its first non-improving round, then evolves from that incumbent. CPU fixture only.');
})().catch(error => { console.error(error); process.exit(1); });

module.exports = {runMode, TEMPLATE, SCENARIOS, d, OUT, field};
