#!/usr/bin/env node
// Parity guard for the greedy Optimize loop in kernel_lane.js (no GPU, no agent, no network).
//
// Six deterministic scenarios run kernel_lane.js under stubbed Workflow runtime globals. Each records
//   - every agent call in dispatch order: label, phase, sha256(prompt), sha256(opts)
//   - the phase() and log() sequences and the lane's return value
//   - the loop-state variables the Report/Validate postlude reads, captured twice: on loop entry and
//     right before phase('Report') (round, dispatched, cumulative, bestPerCase, finalWinner,
//     roundsCommitted, history, citations, profileSummary, deadlineHit, forcedReplans, kbCapBound)
// and compares its SHA256 digest with test_greedy_parity.golden.json, derived from records produced by
// the ordinary loop at the `evo` base (3619cd04). Reviewed instruction and product-status
// changes update prompt hashes only; calls, options, state and results remain checked.
// The two captures are a read-only __probe() call
// inserted before the loop's `while` line and before `phase('Report');`; nothing else is changed.
//
//   (a) improve_then_budget        integrate + three commits (incl. a deep_explore round), storage reclaim, budget end
//   (b) warmstart_then_no_improve  warm-start adopt feeds the loop's start state; two stalls -> no_improve
//   (c) planner_filter_stop        gfx1201/unknown peaks: roofline directions filtered -> planner stop reason
//   (d) deadline_mid_loop          a committed round, then the clock passes the deadline before round 2
//   (e) forced_replan_failures     refused stops, failed/missing/invalid candidates, replan cap -> tech_lead_stop
//   (f) citations_kb_cap           learned-card citations, KB cap strip + forced cold direction, kb:cite
//
// Run:            node kernel_workflow/scripts/test_greedy_parity.js [--lane <kernel_lane.js>]
// Review candidate: node kernel_workflow/scripts/test_greedy_parity.js --lane <reviewed kernel_lane.js> --write-golden
// Candidates and observed mismatch records go to a temporary directory; tracked goldens are never overwritten.
'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const GOLDEN = path.join(__dirname, 'test_greedy_parity.golden.json');

const argv = process.argv.slice(2);
const opt = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
const LANE = path.resolve(opt('--lane') || path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'));
const WRITE = argv.includes('--write-golden');

const sha = (s) => crypto.createHash('sha256').update(String(s)).digest('hex');
const clone = (v) => (v === undefined ? null : JSON.parse(JSON.stringify(v)));

// --- instrumentation: read-only probes at loop entry and right before the Report phase -------------
const PROBE_KEYS = ['round', 'dispatched', 'cumulative', 'bestPerCase', 'finalWinner', 'roundsCommitted',
  'history', 'citations', 'profileSummary', 'deadlineHit', 'forcedReplans', 'kbCapBound'];
const LOOP_LINE = 'while (!skipLoop && dispatched < BUDGET && noImprove < MAX_NO_IMPROVE) {';
const REPORT_LINE = "phase('Report');";
function instrument(src) {
  let body = src.replace(/^export const meta/m, 'const meta');
  for (const marker of [LOOP_LINE, REPORT_LINE]) {
    const n = body.split(marker).length - 1;
    if (n !== 1) throw new Error(`expected exactly one '${marker}' in ${LANE}, found ${n}`);
  }
  const probe = (tag) => `__probe(${JSON.stringify(tag)}, { ${PROBE_KEYS.join(', ')} });\n`;
  return body.replace(LOOP_LINE, () => probe('loop_entry') + LOOP_LINE)
    .replace(REPORT_LINE, () => probe('after_loop') + REPORT_LINE);
}

function build(body, argsObj, answer) {
  const rec = { calls: [], phases: [], logs: [], loop_entry: null, after_loop: null };
  const g = {
    args: argsObj,
    phase: (t) => rec.phases.push(t),
    log: (m) => rec.logs.push(String(m)),
    workflow: async () => { throw new Error('kernel_lane must not nest workflows'); },
    agent: async (p, o) => {
      const label = (o && o.label) || '';
      rec.calls.push({ label, phase: (o && o.phase) || '', prompt_sha256: sha(p), opts_sha256: sha(JSON.stringify(o)) });
      return clone(answer(label, String(p)));
    },
    parallel: async (thunks) => Promise.all(thunks.map((t) => t().catch(() => null))),
    pipeline: async (items, ...stages) => Promise.all(items.map(async (it, i) => {
      let v = it;
      for (const st of stages) v = await st(v, it, i);
      return v;
    })),
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
    __probe: (tag, state) => {
      if (rec[tag]) throw new Error(`probe ${tag} hit twice`);
      rec[tag] = clone(state);
    },
  };
  const fn = new Function(...Object.keys(g), `return (async () => { ${body} })();`);
  return { run: () => fn(...Object.values(g)), rec };
}

// --- fixture --------------------------------------------------------------------------------------
// Paths are fixed strings (the lane never touches the filesystem), so prompt hashes do not depend on
// where the repository is checked out.
const EVAL = '/cpu-fixture/eval';
const BASE_ARGS = { kernel_path: '/cpu-fixture/task', workflow_dir: '/cpu-fixture/kernel_workflow',
  agent_timeout_ms: 0, agent_retries: 1 };
const perCase = (s) => [{ name: 'c1', baseline_ms: 2, optimized_ms: +(2 / s).toFixed(4), speedup: s },
                        { name: 'c2', baseline_ms: 4, optimized_ms: +(4 / s).toFixed(4), speedup: s }];
const verified = (v) => ({ status: 'verified', correctness: 'pass', verified_geomean: v,
  verified_arithmetic: +(v + 0.01).toFixed(4), per_case: perCase(v) });
const ENG_OK = { status: 'ok', speedup_geomean: 1.5, measurement_valid: true };
const ENG_BELOW = { status: 'ok', speedup_geomean: 0.95, measurement_valid: true };   // trusted, no verify
const dir = (id, specialty, extra = {}) => ({ id, title: `T ${id}`, specialty, prompt: `do ${id}`, ...extra });
const STOP = { stop: true, directions: [], reasoning: 'converged' };
// `- KEY: value` line of a cfg() block, parsed back (numbers are JSON, strings raw).
const field = (prompt, key) => {
  const line = prompt.split('\n').find((l) => l.startsWith(`- ${key}: `));
  if (!line) return undefined;
  const raw = line.slice(key.length + 4);
  try { return JSON.parse(raw); } catch (e) { return raw; }
};

function answers(sc) {
  const idOf = (label) => label.split(' ')[1].split(':')[0];
  return (label, prompt) => {
    if (label === 'director:setup') return {
      eval_dir: EVAL, workspace: EVAL + '/workspace', kernel_name: 'fix_kernel',
      device_gfx: sc.gfx || 'gfx950', device_target: 'unknown', physical_cu_count: 256,
      baseline_frozen: true, source_files: ['k.py'],
    };
    if (label === 'tech_lead:analyze') return { kernel_type: 'gemm', roadmap_summary: 'r', modifiable_files: ['k.py'] };
    if (label === 'benchmark_engineer') return {
      commandment_path: EVAL + '/COMMANDMENT.md', baseline_per_case: perCase(1), baseline_geomean_ms: 2.83,
      correctness_cmd: 'python3 t.py --correctness', benchmark_cmd: 'python3 t.py --benchmark',
    };
    if (label === 'profile_engineer:baseline') return { bottleneck: 'memory', top_opportunities: [], summary_path: EVAL + '/p0.md' };
    if (label === 'warm_start:resolve') return sc.resolve || { read_reason: 'empty', candidates: [] };
    if (label.startsWith('warm_start:verify')) return sc.warmVerify;
    if (label.startsWith('warm_start:adopt')) return { committed: true };
    if (label === 'warm_start:reprofile') return { bottleneck: 'compute', summary_path: EVAL + '/pws.md', shift_note: 'ws' };
    if (label === 'kb:attest') return { ran: 1 };
    if (label.startsWith('clock ')) return { epoch: sc.clock ? sc.clock(label.slice(6)) : 0 };
    if (label.startsWith('tech_lead:plan ') || label.startsWith('tech_lead:replan ')) {
      const key = label.split(' ')[1];
      return key in (sc.plans || {}) ? sc.plans[key] : STOP;
    }
    if (label.startsWith('eng ') || label.startsWith('deep ')) {
      const id = idOf(label);
      return id in (sc.eng || {}) ? sc.eng[id] : ENG_OK;
    }
    if (label.startsWith('verify ')) {
      const key = label.slice(7);
      const v = (sc.verify || {})[key];
      return typeof v === 'number' ? verified(v) : (v === undefined ? { status: 'failed', verified_geomean: 0 } : v);
    }
    if (label.startsWith('integrate ')) {
      return (sc.integrate || {})[label.slice(10)] || { attempted: true, conclusion: 'no_gain' };
    }
    if (label.startsWith('commit ')) return { committed: true };
    if (label.startsWith('reprofile ')) {
      const r = label.slice(10);
      return { bottleneck: `after_${r}`, top_opportunities: [], summary_path: `${EVAL}/p_${r}.md`, shift_note: `shift_${r}` };
    }
    if (label.startsWith('tech_lead:memory ')) {
      const r = label.slice(17);
      return { insights: [`insight_${r}`], ledger: [{ h: `h_${r}` }], bottleneck_now: `bn_${r}`, suggest_next: `next_${r}` };
    }
    if (label.startsWith('storage:reclaim')) return { ok: true, note: 'reclaimed' };
    // Postlude: echo what the loop handed over, so the downstream numbers depend on it.
    if (label === 'tech_lead:report') return { final_speedup_geomean: field(prompt, 'CUMULATIVE_SPEEDUP'),
      report_path: EVAL + '/report.md', final_patch: EVAL + '/final_patch.diff', rounds: field(prompt, 'HISTORY').rounds.length };
    if (label === 'director:validate') return { director_verified_speedup_geomean: field(prompt, 'TECH_LEAD_REPORTED_GEOMEAN'),
      validation_status: 'accepted', timing_basis: 'clean' };
    if (label === 'update_experience') return { action: 'created', card_path: '/learned/c.md' };
    if (label === 'kb:cite') return { filed: 1 };
    if (label === 'kb:write') return { written: true, slug: 's', dir: '/d' };
    throw new Error(`unanswered agent label: ${label}`);
  };
}

const SCENARIOS = [
  { name: 'a_improve_then_budget', args: { budget: 4 },
    plans: {
      r1: { stop: false, directions: [dir('a1', 'memory', { focus_files: ['k.py'] }), dir('b1', 'compute')] },
      r2: { stop: false, directions: [dir('a2', 'algorithm')] },
      r3: { stop: false, directions: [dir('a3', 'deep_explore', { expected_speedup: 2.5 })] },
    },
    verify: { a1: 1.3, b1: 1.2, a2: 1.8, a3: 2.2 },
    integrate: { r1: { attempted: true, conclusion: 'improved',
      best: { geomean: 1.5, arithmetic: 1.52, per_case: perCase(1.5), patch_file: EVAL + '/round_1/integrate/best.diff' } } } },

  // The adopt sets cumulative/bestSeen/bestPerCase/finalWinner/profileSummary before the loop. Round 2's
  // 1.15 clears round 1's 1.1 by more than PROGRESS_DELTA, so only a carried-over bestSeen=1.3 keeps it a stall.
  { name: 'b_warmstart_then_no_improve', args: {},
    resolve: { read_reason: 'read', slug: 'fix_kernel', candidates: [{ rank: 1, slug: 'ws-slug', speedup: 1.35,
      direction: 'tile', patch_path: '/kb/p.diff', session_id: 'sess1', exp_dir: '/kb/exp1' }] },
    warmVerify: verified(1.3),
    plans: { r1: { stop: false, directions: [dir('a1', 'memory')] },
             r2: { stop: false, directions: [dir('a2', 'compute')] } },
    verify: { a1: 1.1, a2: 1.15 } },

  { name: 'c_planner_filter_stop', args: {}, gfx: 'gfx1201',
    plans: {
      r1: { stop: false, directions: [dir('a1', 'memory'), dir('b1', 'memory', { title: 'reach the bandwidth roofline' })] },
      r2: { stop: false, directions: [dir('a2', 'compute', { prompt: 'hit 90% of peak flops' })] },
    },
    verify: { a1: 1.25 } },

  { name: 'd_deadline_mid_loop', args: { deadline_epoch: 10000, budget: 6 },
    clock: (tag) => (tag === 'pre-r1' ? 5000 : 10050),
    plans: { r1: { stop: false, directions: [dir('a1', 'memory'), dir('b1', 'compute')] } },
    eng: { b1: ENG_BELOW },
    verify: { a1: 1.4 } },

  { name: 'e_forced_replan_failures', args: { deadline_epoch: 100000, max_forced_replans: 2 },
    clock: () => 1000,
    plans: {
      r1: STOP,
      'r1#1': { stop: false, directions: [dir('a1', 'memory'), dir('b1', 'compute'), dir('c1', 'algorithm'), dir('d1', 'host_runtime')] },
      r2: STOP, 'r2#2': STOP,
    },
    eng: { a1: { status: 'failed', notes: 'crashed' }, b1: null,
           c1: { status: 'invalid_measurement', speedup_geomean: 1.6, measurement_valid: false } },
    verify: { 'a1 (recovered)': { status: 'failed', correctness: 'fail', verified_geomean: 0 },
              'b1 (recovered)': { status: 'apply_failed', notes: 'empty patch' },
              c1: { status: 'invalid_measurement', correctness: 'pass', verified_geomean: 1.6 },
              d1: 1.2 } },

  { name: 'f_citations_kb_cap', args: { budget: 8 },
    plans: {
      r1: { stop: false, directions: [dir('a1', 'memory', { learned_refs: ['card1.md'] }),
        dir('b1', 'compute', { learned_refs: ['card2.md', 'card3.md'] }), dir('c1', 'algorithm')] },
      r2: { stop: false, directions: [dir('a2', 'memory', { learned_refs: ['card4.md'] }), dir('b2', 'compute')] },
      r3: { stop: false, directions: [dir('a3', 'algorithm', { learned_refs: ['card5.md'] })] },
    },
    eng: { c1: ENG_BELOW, a2: ENG_BELOW },
    verify: { a1: 1.4, b1: 1.2, b2: 1.3, a3: 1.5 } },
];

async function runScenario(body, sc) {
  const { run, rec } = build(body, { ...BASE_ARGS, ...sc.args }, answers(sc));
  const ret = await run();
  if (!rec.loop_entry || !rec.after_loop) throw new Error(`${sc.name}: a probe was not reached`);
  const result = { ...ret, llm_timeline: ret.llm_timeline ? sha(JSON.stringify(ret.llm_timeline)) : null };
  const changed = PROBE_KEYS.filter((k) => JSON.stringify(rec.loop_entry[k]) !== JSON.stringify(rec.after_loop[k]));
  return { stop_reason: ret.stopped_by, globals_changed_by_loop: changed, ...rec, result };
}

(async () => {
  const src = fs.readFileSync(LANE, 'utf8');
  const body = instrument(src);
  console.log(`lane: ${LANE} (sha256 ${sha(src).slice(0, 16)})`);
  const got = {};
  for (const sc of SCENARIOS) got[sc.name] = await runScenario(body, sc);

  const digests = Object.fromEntries(Object.entries(got).map(([name, record]) => [name, sha(JSON.stringify(record))]));
  let artifactDir;
  const save = (name, value) => {
    artifactDir ||= fs.mkdtempSync(path.join(os.tmpdir(), 'geak-greedy-parity-'));
    const file = path.join(artifactDir, name);
    fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n');
    return file;
  };
  if (WRITE) {
    const candidate = { comment: 'SHA256 of JSON.stringify(record) for each reviewed greedy scenario; includes calls, prompts, options, phases, logs, loop state and result.',
      source_lane_sha256: sha(src), scenarios: digests };
    console.log(`Review candidate: ${save('test_greedy_parity.golden.json', candidate)}`);
    console.log(`Full observed records: ${save('observed.json', got)}`);
    return;
  }

  const golden = JSON.parse(fs.readFileSync(GOLDEN, 'utf8')).scenarios;
  let failures = 0;
  for (const name of new Set([...Object.keys(golden), ...Object.keys(got)])) {
    const r = got[name];
    if (golden[name] && digests[name] === golden[name]) {
      console.log(`  ok: ${name}: ${r.calls.length} calls, stopped_by=${r.stop_reason}, full record digest matches`);
      continue;
    }
    failures++;
    console.error(`  FAIL: ${name}: expected ${golden[name] || '(no golden)'}, observed ${digests[name] || '(scenario missing)'}`);
    if (r) console.error(`    Full observed record: ${save(name + '.json', r)}`);
  }
  console.log(failures === 0 ? '\nPASS: greedy loop matches the reviewed workflow snapshots.'
    : `\nFAILED: ${failures} scenario(s) differ from the golden.`);
  process.exit(failures === 0 ? 0 : 1);
})().catch((e) => { console.error(e && e.stack || e); process.exit(1); });
