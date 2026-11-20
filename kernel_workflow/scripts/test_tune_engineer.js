#!/usr/bin/env node
// Regression for the `tune` engineer specialty (no GPU, no model needed).
//
// `tune` is an ordinary specialty: the TechLead plans it, it is charged against the budget, the
// engineer reads roles/engineer.md (+ knowledge/tuning_playbook.md), and its patch, history rows and
// direction_entropy go through exactly the same structures as every other direction. Checks:
//   - every list of specialties agrees (PLAN/RESEARCH schema, Eikos validator + registry, roles);
//   - the knowledge the tune specialty is told to read exists;
//   - a planned tune direction runs through the REAL kernel_lane.js (in a vm, deterministic agent
//     replies) like any other direction, with no special-case code path.
//
// Run:  node kernel_workflow/scripts/test_tune_engineer.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const ROOT = path.resolve(__dirname, '..', '..');
const LANE = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'), 'utf8');
const read = (...p) => fs.readFileSync(path.join(ROOT, ...p), 'utf8');
const clone = (v) => JSON.parse(JSON.stringify(v));

const SPECIALISTS = (r) => [
  { id: `r${r}_d0`, title: 'a', specialty: 'algorithm', focus_files: ['src/a.py'], prompt: 'do a' },
  { id: `r${r}_d1`, title: 'b', specialty: 'memory', focus_files: ['src/b.py'], prompt: 'do b' },
];
const WITH_TUNE = (r) => SPECIALISTS(r).concat([
  { id: `r${r}_d2`, title: 'retune', specialty: 'tune', focus_files: ['src/config.py'], prompt: 'sweep tiles' }]);
const DEEP = (r) => ({ stop: false, directions: [{ id: `r${r}_d0`, title: 'deep', specialty: 'deep_explore', prompt: 'go' }] });

async function run(source, options = {}) {
  const body = source.replace('export const meta =', 'const meta =');
  const labels = [], prompts = [], logs = [];
  let timerId = 0;
  const baseline = [{ name: 'small', latency_ms: 1 }, { name: 'large', latency_ms: 1 }];
  const perCase = (s) => baseline.map((row) => ({ name: row.name, baseline_ms: row.latency_ms,
    optimized_ms: row.latency_ms / s, speedup: s }));
  // score(id) -> engineer-claimed == verified speedup of that direction id
  const score = options.score || ((id) => (id.endsWith('_d2') ? 1.15 : id.endsWith('_d0') ? 1.2 : 1.1));
  const plan = options.plan || ((r) => ({ stop: false, directions: SPECIALISTS(r) }));
  let cumulative = 1;
  const sandbox = {
    args: { kernel_path: '/task/op', workflow_dir: '/repo/kernel_workflow', budget: 6, gpu_ids: '0,1',
      max_no_improve: 3, min_improve: 0.01, warm_start: 'off', use_learned_kb: 'false',
      update_experience: 'off', expected_gfx: 'gfx942', ...options.args },
    phase() {}, log(m) { logs.push(String(m)); },
    setTimeout() { return timerId++; }, clearTimeout() {},
    async pipeline(items, start, finish) {
      return Promise.all(items.map(async (it) => { const r = await start(it); return finish ? finish(r) : r; }));
    },
    async parallel(thunks) { return Promise.all(thunks.map((t) => t())); },
    async workflow() { throw new Error('nested workflow() not expected in a lane run'); },
    async agent(prompt, opts) {
      const label = String(opts && opts.label);
      labels.push(label);
      prompts.push({ label, prompt, schema: clone(opts.schema || null), phase: opts.phase });
      const reply = options.reply && options.reply(label, prompt);
      if (reply !== undefined) return reply;
      if (label === 'director:setup') {
        return { eval_dir: '/eval', workspace: '/eval/workspace', kernel_name: 'op', baseline_frozen: true,
          device_gfx: 'gfx942', device_target: 'unknown', physical_cu_count: 304, device_name: 'MI300X' };
      }
      if (label === 'tech_lead:analyze') return { kernel_type: 'gemm', roadmap_summary: 'analysis' };
      if (label === 'benchmark_engineer') {
        return { commandment_path: '/eval/COMMANDMENT.md', baseline_per_case: baseline, baseline_geomean_ms: 1 };
      }
      if (label.startsWith('profile_engineer') || label.startsWith('reprofile')) {
        return { device: 'gfx942', bottleneck: 'memory', top_opportunities: [], summary_path: '/eval/profile.md' };
      }
      if (/^tech_lead:(plan|replan) r/.test(label)) return plan(Number(label.match(/ r(\d+)/)[1]));
      if (/^(eng|deep) r/.test(label)) {
        const id = label.split(' ')[1].split(':')[0];
        const s = score(id);
        return { status: 'success', measurement_valid: true, speedup_geomean: s, speedup_arithmetic: s,
          per_case: perCase(s), strategy: `x-${id}`, notes: 'ok' };
      }
      if (label.startsWith('verify r')) {
        const id = label.split(' ')[1];
        const s = score(id);
        return { status: 'verified', correctness: 'pass', verified_geomean: s, verified_arithmetic: s,
          per_case: perCase(s) };
      }
      if (label.startsWith('integrate r')) {
        return options.integrate ? options.integrate(prompt) : { attempted: true, conclusion: 'no_improvement' };
      }
      if (label.startsWith('commit r')) {
        const m = prompt.match(/\((\d+\.\d+)x\)/);
        if (m) cumulative = Math.max(cumulative, Number(m[1]));
        return { committed: true, current_best_diff: '/eval/current_best.diff', note: 'committed' };
      }
      if (label.startsWith('tech_lead:memory')) {
        return { insights: [], ledger: [], bottleneck_now: 'memory', suggest_next: 'x' };
      }
      if (label.startsWith('storage:reclaim')) return { ok: true, note: 'reclaimed' };
      if (label === 'tech_lead:report') {
        return { final_speedup_geomean: cumulative, final_patch: '/eval/final_patch.diff', report_path: '/eval/report.md' };
      }
      if (label === 'director:validate') {
        return { director_verified_speedup_geomean: cumulative, validation_status: 'accepted',
          timing_basis: 'not_applicable', correctness: 'pass' };
      }
      return null;
    },
  };
  const result = await vm.runInNewContext(`(async()=>{${body}})()`, sandbox);
  return clone({ result, labels, prompts, logs });
}

let failures = 0;
const cases = [];
const test = (name, fn) => cases.push([name, fn]);
const promptOf = (t, label) => {
  const p = t.prompts.find((x) => x.label === label);
  assert(p, `no prompt for ${label}`);
  return p.prompt;
};
const plansTune = { plan: (r) => ({ stop: false, directions: WITH_TUNE(r) }) };
const sorted = (xs) => Array.from(xs).sort();

test('every list of specialties includes tune and they agree', async () => {
  const enums = [...LANE.matchAll(/specialty: \{ type: 'string', enum: \[([^\]]*)\] \}/g)]
    .map((m) => sorted(m[1].split(',').map((x) => x.trim().replace(/'/g, ''))));
  assert(enums.length >= 2, 'PLAN_SCHEMA and RESEARCH_SCHEMA specialty enums not found');
  const want = enums[0];
  assert(want.includes('tune'), 'PLAN_SCHEMA enum lacks tune');
  for (const e of enums) assert.deepStrictEqual(e, want);
  const py = read('e2e_workflow', 'scripts', 'eikos_decide.py').match(/^SPECIALTIES = \{([^}]*)\}/m);
  assert(py, 'eikos_decide.SPECIALTIES not found');
  assert.deepStrictEqual(sorted(py[1].split(',').map((x) => x.trim().replace(/"/g, ''))), want,
    'Eikos validator would reject a dispatched specialty');
  const reg = JSON.parse(read('e2e_workflow', 'scripts', 'eikos_questions', 'decision_registry.json'));
  const k2 = (reg.decisions || []).find((q) => q.id === 'K2');
  assert(k2, 'registry K2 not found');
  assert.deepStrictEqual(sorted(k2.options), want, 'registry K2 options drift from PLAN_SCHEMA');
  const engLine = read('kernel_workflow', 'roles', 'engineer.md').match(/- `SPECIALTY` — one of `([^`]*)`/);
  assert(engLine, 'engineer.md SPECIALTY line');
  assert.deepStrictEqual(sorted(engLine[1].split('|').map((x) => x.trim())),
    want.filter((x) => x !== 'deep_explore'), 'engineer.md SPECIALTY list');
  assert(read('kernel_workflow', 'roles', 'tech_lead.md').includes('- **tune** —'), 'tech_lead.md specialty entry');
  assert(read('kernel_workflow', 'roles', 'researcher.md').includes('| tune |'), 'researcher.md specialty list');
});

test('the knowledge the tune specialty reads exists', async () => {
  const eng = read('kernel_workflow', 'roles', 'engineer.md');
  assert(/- tune +→ `tuning_playbook\.md`/.test(eng), 'engineer.md knowledge map for tune');
  assert(fs.existsSync(path.join(ROOT, 'kernel_workflow', 'knowledge', 'tuning_playbook.md')));
  for (const d of ['tuning-core', 'tuning-flydsl', 'tuning-hip', 'tuning-ck', 'tuning-triton']) {
    assert(fs.existsSync(path.join(ROOT, 'perf_knowledge', 'expert_skills', 'tuning', d, 'SKILL.md')), d);
  }
  assert(read('kernel_workflow', 'roles', 'tech_lead.md').includes('2t. **When to plan a `tune` direction'));
});

test('no special-case code path for tune in the lane', async () => {
  for (const s of ['tune_engineer', 'TUNE_ON', 'TUNE_ENGINEER', 'engineer_tune', "=== 'tune'"]) {
    assert(!LANE.includes(s), `kernel_lane.js still special-cases tune: ${s}`);
  }
  const a = await run(LANE, plansTune);
  assert(!('tune_engineer' in a.result));
});

test('a planned tune direction is charged and dispatched like any engineer', async () => {
  const a = await run(LANE, plansTune);
  const b = await run(LANE, {});
  assert.strictEqual(a.result.budget_used, 6);
  // 3 directions per round on a budget of 6 => 2 rounds; 2 per round => 3 rounds.
  assert.strictEqual(a.labels.filter((l) => /^tech_lead:plan r/.test(l)).length, 2);
  assert.strictEqual(b.labels.filter((l) => /^tech_lead:plan r/.test(l)).length, 3);
  const tp = promptOf(a, 'eng r1_d2:tune');
  for (const s of ['You are Engineer r1_d2 (specialty=tune) for round 1.', '/repo/kernel_workflow/roles/engineer.md',
    '- SPECIALTY: tune', '/eval/round_1/engineer_2/workspace', 'src/config.py']) assert(tp.includes(s), s);
  // Same prompt template as the other specialists: only the direction fields differ.
  const norm = (p) => p.replace(/r1_d\d/g, 'ID').replace(/engineer_\d/g, 'E')
    .replace(/specialty=\w+/g, 'S').replace(/- SPECIALTY: \w+/g, 'S').replace(/- DIRECTION: [^\n]*/, 'D');
  assert.strictEqual(norm(tp), norm(promptOf(a, 'eng r1_d0:algorithm')));
  assert.deepStrictEqual(a.prompts.find((p) => p.label === 'eng r1_d2:tune').schema,
    a.prompts.find((p) => p.label === 'eng r1_d0:algorithm').schema);
  assert(a.labels.includes('verify r1_d2'));
  const integ = promptOf(a, 'integrate r1');
  assert(integ.includes('/eval/round_1/engineer_2/best_patch.diff') && integ.includes('"specialty":"tune"'));
});

test('tune lands in the existing history structures', async () => {
  const a = await run(LANE, plansTune);
  const p2 = promptOf(a, 'tech_lead:plan r2');
  assert(p2.includes('{"id":"r1_d2","title":"retune","specialty":"tune"'), 'history.rounds[].directions');
  assert(/"results":\[[^\]]*\{"id":"r1_d2","claimed":1\.15,"verified":1\.15/.test(p2), 'history.rounds[].results');
  assert(!p2.includes('"tune":{'), 'no separate tune record');
  assert.deepStrictEqual(a.result.direction_entropy, { distinct: 3, issued: 6, ratio: 0.5 });
});

test('a tune-only win is committed alone', async () => {
  const a = await run(LANE, { ...plansTune, score: (id) => (id.endsWith('_d2') ? 1.1 : 0.2) });
  assert(/round 1 winner: engineer r1_d2 \(1\.10x\)/.test(promptOf(a, 'commit r1')));
  assert(!a.labels.includes('integrate r1'));
});

(async () => {
  for (const [name, fn] of cases) {
    try { await fn(); console.log('  ok:', name); } catch (e) { failures++; console.error('  FAIL:', name, '--', e && e.stack || e); }
  }
  console.log(failures === 0 ? `\nPASS: ${cases.length} tune-engineer checks.` : `\nFAILED: ${failures} check(s).`);
  process.exit(failures === 0 ? 0 : 1);
})();
