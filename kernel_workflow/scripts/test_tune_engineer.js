#!/usr/bin/env node
// Mock-lane regression for the opt-in tune engineer (args.tune_engineer, no GPU, no model needed).
//
// Runs the REAL kernel_lane.js in a vm with deterministic agent replies and checks:
//   - off (the default) adds nothing: no tune dispatch, no TUNE_ENGINEER plan input, no result key;
//   - on: every non-deep_explore round adds ONE uncharged `r<N>_tune` engineer whose patch goes through
//     the unchanged verify -> integrate -> commit path, while the planned engineers' prompts, the budget
//     accounting and direction_entropy are exactly what the same run gets with the switch off;
//   - a deep_explore round gets no tune engineer; a tune-only win is committed alone.
// BASE_LANE=/path/to/kernel_lane.js (optional) additionally checks that `off` replays a pre-feature
// lane byte for byte.
//
// Run:  node kernel_workflow/scripts/test_tune_engineer.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const ROOT = path.resolve(__dirname, '..', '..');
const LANE = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'), 'utf8');
const ROLE = path.join(ROOT, 'kernel_workflow', 'roles', 'tune_engineer.md');
const clone = (v) => JSON.parse(JSON.stringify(v));

const SPECIALISTS = (r) => [
  { id: `r${r}_d0`, title: 'a', specialty: 'algorithm', focus_files: ['src/a.py'], prompt: 'do a' },
  { id: `r${r}_d1`, title: 'b', specialty: 'memory', focus_files: ['src/b.py'], prompt: 'do b' },
];
const DEEP = (r) => ({ stop: false, directions: [{ id: `r${r}_d0`, title: 'deep', specialty: 'deep_explore', prompt: 'go' }] });

async function run(source, options = {}) {
  const body = source.replace('export const meta =', 'const meta =');
  const labels = [], prompts = [], logs = [];
  let timerId = 0;
  const baseline = [{ name: 'small', latency_ms: 1 }, { name: 'large', latency_ms: 1 }];
  const perCase = (s) => baseline.map((row) => ({ name: row.name, baseline_ms: row.latency_ms,
    optimized_ms: row.latency_ms / s, speedup: s }));
  // score(id) -> engineer-claimed == verified speedup of that direction id
  const score = options.score || ((id) => (id.endsWith('_tune') ? 1.15 : id.endsWith('_d0') ? 1.2 : 1.1));
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
const isTune = (l) => /r\d+_tune\b/.test(l);
const TUNE_KEY_RE = /\n- TUNE_ENGINEER: [^\n]*/;
const ON = { tune_engineer: 'on' };

test('the role file exists and states the parameter boundary', async () => {
  const role = fs.readFileSync(ROLE, 'utf8');
  for (const s of ['parameter boundary', 'Per-shape config dispatch', 'FORBIDDEN', 'measurement_valid',
    'invalid_measurement', 'A B A B', '"specialty": "tune"']) assert(role.includes(s), s);
});

test('off is the default and adds nothing', async () => {
  for (const args of [{}, { tune_engineer: 'off' }, { tune_engineer: 'nonsense' }]) {
    const a = await run(LANE, { args });
    assert(!a.labels.some(isTune), `tune dispatch with ${JSON.stringify(args)}`);
    assert(!a.prompts.some((p) => TUNE_KEY_RE.test(p.prompt) || p.prompt.includes('tune_engineer.md')));
    assert(!('tune_engineer' in a.result), 'result.tune_engineer must be absent when off');
    assert(!a.logs.some((l) => /tune engineer/.test(l)));
  }
});

test('on: one uncharged r<N>_tune per round; planned engineers unchanged', async () => {
  const a = await run(LANE, { args: ON });
  const b = await run(LANE, {});
  assert.strictEqual(a.result.budget_used, b.result.budget_used, 'tune must not be charged');
  assert.strictEqual(a.result.rounds, b.result.rounds);
  const rounds = a.labels.filter((l) => /^tech_lead:plan r/.test(l)).length;
  assert(rounds >= 2, `expected >= 2 rounds, got ${rounds}`);
  for (let r = 1; r <= rounds; r++) {
    assert(a.labels.includes(`eng r${r}_tune:tune`), `tune missing in round ${r}`);
    assert(a.labels.includes(`verify r${r}_tune`), `tune not verified in round ${r}`);
    for (const [d, sp] of [['d0', 'algorithm'], ['d1', 'memory']]) {
      assert.strictEqual(promptOf(a, `eng r${r}_${d}:${sp}`), promptOf(b, `eng r${r}_${d}:${sp}`),
        `planned engineer r${r}_${d} prompt changed`);
    }
  }
  // Plan prompt: only the TUNE_ENGINEER input is added (round 1 has identical history).
  const p1 = promptOf(a, 'tech_lead:plan r1');
  assert(TUNE_KEY_RE.test(p1) && p1.includes('r1_tune') && p1.includes('NOT charged'));
  assert.strictEqual(p1.replace(TUNE_KEY_RE, ''), promptOf(b, 'tech_lead:plan r1'));
  // Tune prompt: tune role + tuning skillset, own out_dir, the ordinary engineer schema.
  const tp = a.prompts.find((p) => p.label === 'eng r1_tune:tune');
  for (const s of ['/repo/kernel_workflow/roles/tune_engineer.md', '/repo/kernel_workflow/knowledge/self_monitoring.md',
    '/repo/perf_knowledge/expert_skills/tuning/', 'PARAMETERS ONLY', '/eval/round_1/engineer_tune/workspace',
    'You are Engineer r1_tune (specialty=tune) for round 1.', '- SPECIALTY: tune']) {
    assert(tp.prompt.includes(s), s);
  }
  assert(!tp.prompt.includes('roles/engineer.md'));
  assert.deepStrictEqual(tp.schema, a.prompts.find((p) => p.label === 'eng r1_d0:algorithm').schema);
  // The tune patch enters the round's integrate like any verified patch.
  const integ = promptOf(a, 'integrate r1');
  assert(integ.includes('/eval/round_1/engineer_tune/best_patch.diff') && integ.includes('"id":"r1_tune"'));
  // direction_entropy sees the TechLead's plan only.
  assert.deepStrictEqual(a.result.direction_entropy, b.result.direction_entropy);
  const te = a.result.tune_engineer;
  assert.strictEqual(te.rounds_run, rounds);
  assert.strictEqual(te.passed_verify, rounds);
  assert.strictEqual(te.sent_to_integrate, rounds);
  assert(te.per_round.every((t) => t.claimed === 1.15 && t.verified === 1.15 && !t.committed_alone));
});

test('deep_explore round: no tune engineer that round; the others still get one', async () => {
  const a = await run(LANE, { args: ON,
    plan: (r) => (r === 2 ? DEEP(r) : { stop: false, directions: SPECIALISTS(r) }) });
  assert(a.labels.includes('eng r1_tune:tune'));
  assert(!a.labels.some((l) => /r2_tune/.test(l)));
  assert(a.logs.includes('Round 2: tune engineer skipped (dedicated deep_explore round)'));
  assert(a.labels.includes('eng r3_tune:tune'));
  assert(!a.result.tune_engineer.per_round.some((t) => t.round === 2));
});

test('a tune-only win is committed alone', async () => {
  const a = await run(LANE, { args: ON, score: (id) => (id.endsWith('_tune') ? 1.1 : 0.2) });
  assert(/round 1 winner: engineer r1_tune \(1\.10x\)/.test(promptOf(a, 'commit r1')));
  assert(!a.labels.includes('integrate r1'), 'one verified patch must not trigger integrate');
  const t1 = a.result.tune_engineer.per_round[0];
  assert(t1.round_winner && t1.committed_alone && !t1.sent_to_integrate);
  assert.strictEqual(a.result.tune_engineer.committed_alone, 1);
});

test('the TechLead sees the tune record in HISTORY', async () => {
  const a = await run(LANE, { args: ON });
  assert(promptOf(a, 'tech_lead:plan r2').includes('"tune":{"id":"r1_tune"'));
});

test('a planned engineer that fails does not take the tune engineer down', async () => {
  const a = await run(LANE, { args: ON, score: (id) => (id.endsWith('_tune') ? 1.1 : 0.2),
    reply: (label) => (/^eng r1_d0:/.test(label) ? null : undefined) });
  assert(a.labels.includes('verify r1_tune'));
  assert(a.result.tune_engineer.per_round[0].passed_verify);
});

if (process.env.BASE_LANE) {
  test('off replays the pre-feature lane byte for byte (BASE_LANE)', async () => {
    const base = fs.readFileSync(process.env.BASE_LANE, 'utf8');
    for (const opts of [{}, { score: (id) => (id.endsWith('_d1') ? 1.3 : 0.2) },
      { plan: (r) => (r === 2 ? DEEP(r) : { stop: false, directions: SPECIALISTS(r) }) }]) {
      assert.deepStrictEqual(await run(LANE, opts), await run(base, opts));
    }
  });
}

(async () => {
  for (const [name, fn] of cases) {
    try { await fn(); console.log('  ok:', name); } catch (e) { failures++; console.error('  FAIL:', name, '--', e && e.stack || e); }
  }
  console.log(failures === 0 ? `\nPASS: ${cases.length} tune-engineer checks.` : `\nFAILED: ${failures} check(s).`);
  process.exit(failures === 0 ? 0 : 1);
})();
