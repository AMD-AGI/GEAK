// Runs workflow.js with synthetic agent responses: no agents, no GPU, no filesystem side effects.
// Every agent call is answered by `respond(label, prompt)`; anything it leaves undefined falls back to
// a well-formed default, so a test only states the responses its scenario is about.
'use strict';

const fs = require('fs');
const path = require('path');

const PKG = path.resolve(__dirname, '..');
const ROOT = path.resolve(PKG, '..', '..', '..', '..');
const WF_DIR = path.join(ROOT, 'kernel_workflow');
const SKILL = path.join(ROOT, 'perf_knowledge', 'expert_skills', 'skills', 'gluon_authoring');
const SRC = fs.readFileSync(path.join(PKG, 'workflow.js'), 'utf8')
  .replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

// Every required reference the prompt lists, in the form the engineer would echo back.
const refsIn = (prompt) => [...prompt.matchAll(/^\d+\. (\S+) — /gm)].map(m => m[1]);

const goodG3 = { command: 'python3 kt/probe.py measure --dir ir/r --arch gfx942', exit_code: 0,
  ir_dir: '/x/ir', output_path: '/x/g3.txt', vgpr_wg_per_cu: 2, lds_wg_per_cu: 4, binding: 'registers' };
const goodG4 = { command: 'python3 kt/ab_bench.py --module a.py --permute --json /x/ab.json', exit_code: 0,
  json_path: '/x/ab.json', delta_pct: 5, noise_band_pct: 1, resolved: true, valid: true };
const goodAtt = { status: 'captured', tool: 'capture.sh', command: 'bash kt/capture.sh r /x/att --kernel k --source k.py -- python run.py',
  exit_code: 0, output_dir: '/x/att', state: 'ok' };
const goodProfileCmd = 'python3 skill/scripts/gluon_profile.py /x/prof --kernel k --source k.py --gpu 6 --compare /x/prev -- python run.py';
const goodEdge = { command: 'python3 skill/scripts/edge_shapes.py --module a.py --dims M=64:16 --json /x/edge.json', exit_code: 0,
  json_path: '/x/edge.json', baseline: 'champion' };
const gate = (exit) => ({ command: 'python3 skill/scripts/parity_gate.py --champion-ms 1 --anchor-ms 1.2', exit_code: exit,
  ratio: exit === 0 ? 0.97 : 0.8, suspects: [], json_path: '/x/parity.json', mode: 'gate' });

// Escape every regex metacharacter, backslash included, before building a pattern from a label.
const escapeRe = (s) => String(s).replace(/[\\^$.*+?()[\]{}|]/g, '\\$&');
const layerRefsIn = (prompt, layer) => {
  const m = prompt.match(new RegExp(`^- layer ${escapeRe(layer)}: (.*)$`, 'm'));
  return m ? m[1].split(', ') : [];
};

// A complete engineer return for the stage in the prompt (round tag read from the prompt).
function engineerReturn(prompt, over) {
  const stage = (prompt.match(/PHASE=(\w+)/) || [])[1];
  const tag = (prompt.match(/ROUND_TAG: (r\d+(?:_\w+)?)/) || [])[1];
  const layer = (over && over.layer) || (stage === 'recover' || stage === 'climb' ? '3' : '');
  const reads = [...refsIn(prompt), ...(layer ? layerRefsIn(prompt, layer) : [])];
  const base = { stage, references_read: reads, track_tag: tag, patch_file: `/x/${tag}.diff`, correctness: 'pass' };
  if (stage === 'transcribe') Object.assign(base, { g3_occupancy: goodG3, edge_shapes: goodEdge,
    equivalence: { layout_diff: 'PASS', numeric: 'pass', determinism: 'pass', asm_parity: 'pass', output_path: '/x/eq' } });
  if (stage === 'author') Object.assign(base, { g3_occupancy: goodG3, edge_shapes: goodEdge });
  if (stage === 'recover') Object.assign(base, { suspect: 'lost_layout', layer: '3', g3_occupancy: goodG3, g4_screen: goodG4, parity_gate: gate(0), claim: 'recovery' });
  if (stage === 'climb') Object.assign(base, { layer: '3', profile_dir: '/x/prof', profile_command: goodProfileCmd, g3_occupancy: goodG3, g4_screen: goodG4, claim: 'win' });
  return Object.assign(base, over || {});
}

// A tech_lead plan: one direction per entry of `dirs` (default one layer-3 direction; a recover round
// repays lost_layout).
function planReturn(prompt, dirs) {
  const recover = /STAGE: recover/.test(prompt);
  const list = dirs || [recover ? { specialty: 'memory', layer: '3', suspect: 'lost_layout' } : { specialty: 'memory', layer: '3' }];
  return { stop: false, reasoning: 'r', directions: list.map((d, i) => ({ id: `p${i}`, title: `t${i}`, prompt: 'do it', ...d })) };
}

// A complete integrate return for the prompt's ROUND_TAG.
function integrateReturn(prompt, over) {
  const tag = (prompt.match(/ROUND_TAG: (r\d+_int)/) || [])[1];
  const base = { attempted: true, conclusion: 'improved', combos_tried: [{ patches: ['a', 'b'], relation: 'orthogonal' }],
    references_read: refsIn(prompt), track_tag: tag, patch_file: `/x/${tag}.diff`, correctness: 'pass', g4_screen: goodG4 };
  if (/STAGE: recover/.test(prompt)) base.parity_gate = gate(0);
  return Object.assign(base, over || {});
}

function defaults(label, prompt, ctx) {
  if (label === 'director:setup') return { eval_dir: '/tmp/eval', workspace: '/tmp/eval/workspace', kernel_name: 'k',
    device_gfx: ctx.gfx || 'gfx942', device_target: 'unknown', physical_cu_count: 80, baseline_frozen: true };
  if (label === 'benchmark_engineer') return { commandment_path: '/tmp/eval/COMMANDMENT.md',
    baseline_per_case: [{ name: 'c0', latency_ms: 1 }], baseline_geomean_ms: 1 };
  if (label.startsWith('gluon_profiler:baseline')) return { bottleneck: 'latency', summary_path: '/tmp/eval/profile.md',
    command: 'python3 skill/scripts/gluon_profile.py /tmp/eval/gluon/profile_baseline --kernel k --source k.py --gpu 6 -- python run.py',
    exit_code: 0, profile_dir: '/tmp/eval/gluon/profile_baseline', references_read: refsIn(prompt),
    layers: { trace: 'ok', pmc: 'ok', att: 'ok', ir: 'ok' } };
  if (label.startsWith('gluon_lead:entry')) return { entry_mode: ctx.entry || 'port', references_read: refsIn(prompt),
    champion_gate: { command: 'python3 champion_gate.py --champion b.json', exit_code: 0, output_path: '/x/g1.txt', flags: [] },
    track_ready: true, champion_bundle: '/x/plain_champion.json', source_is_anchor: false, champion_patch: '' };
  if (label.startsWith('gluon_lead:budget')) return { command: 'python3 kt/hw_budget.py --sku MI308X --json', exit_code: 0,
    budget_json: '/x/budget.json', references_read: refsIn(prompt) };
  if (label.startsWith('gluon_lead:close')) return { report_path: '/x/report.md', final_patch: '/tmp/eval/final_patch.diff',
    closure_review_path: '/x/closure_review.md', references_read: refsIn(prompt) };
  if (label.startsWith('gluon_engineer:')) return engineerReturn(prompt);
  if (label === 'tech_lead:analyze') return { kernel_type: 'gemm', roadmap_summary: 'r', kk_operator: 'gemm_op', kk_language: 'gluon',
    kk_refs: ['operators/gemm_op/tuning.md'], modifiable_files: ['source/k.py'] };
  if (label === 'warm_start:resolve') return { read_reason: 'kernel_page_not_found', candidates: [] };
  if (label.startsWith('storage:reclaim')) return { ok: true, note: 'reclaimed' };
  if (label === 'update_experience') return { action: 'created', card_path: '/x/card.md' };
  if (label === 'kb:cite') return { filed: 1 };
  if (label === 'kb:write') return { written: true, slug: 's' };
  if (label === 'kb:attest') return { ran: 1 };
  if (/^tech_lead:(re)?plan /.test(label)) return planReturn(prompt);
  if (label.startsWith('tech_lead:memory')) return { insights: ['an insight'], ledger: [] };
  if (label.startsWith('integrate r')) return integrateReturn(prompt);
  if (label.startsWith('gluon_profiler:reprofile')) return { bottleneck: 'latency', command: 'python3 s/gluon_profile.py /x/rp --compare /x/prev -- python run.py',
    exit_code: 0, profile_dir: '/x/reprofile', layers: { trace: 'ok', pmc: 'ok', att: 'ok', ir: 'ok' }, references_read: refsIn(prompt) };
  if (label.startsWith('evidence_audit')) return { ok: true, findings: [], checked: ['all'] };
  if (label === 'verify champion') return { status: 'verified', correctness: 'pass', verified_geomean: 1.0 };
  if (label.startsWith('verify r')) return { status: 'verified', correctness: 'pass', verified_geomean: 1.1 };
  if (label === 'director:validate') return { validation_status: 'accepted', director_verified_speedup_geomean: 1.1 };
  if (label.startsWith('clock ')) return { epoch: 1000 };
  return null;
}

async function launch(args, respond, ctx) {
  const calls = [];
  const logs = [];
  const agent = async (prompt, opts) => {
    calls.push({ label: opts.label, prompt, opts });
    const own = respond ? respond(opts.label, prompt, calls) : undefined;
    return own !== undefined ? own : defaults(opts.label, prompt, ctx || {});
  };
  const parallel = (ts) => Promise.all(ts.map((t) => Promise.resolve().then(t).catch(() => null)));
  const fn = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', SRC);
  let result = null;
  let error = null;
  try {
    result = await fn(agent, parallel, async () => [], () => {}, (m) => logs.push(String(m)), async () => null,
      { kernel_path: '/tmp/task', workflow_dir: WF_DIR, gpu_ids: '6,7', agent_timeout_ms: 0, engineer_timeout_ms: 0, ...args },
      (ctx && ctx.tokens) || { total: null, spent: () => 0, remaining: () => Infinity });
  } catch (e) { error = String(e && e.message || e); }
  return { result, error, calls, logs, labels: calls.map(c => c.label) };
}

let failures = 0;
const ok = (cond, msg, got) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg, got === undefined ? '' : `(got ${typeof got === 'string' ? got : JSON.stringify(got)})`); failures++; }
};
const done = (name) => {
  if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
  console.log(`\nall ${name} checks passed`);
};
// True when `sub` appears in `labels` in this relative order.
const inOrder = (labels, sub) => {
  let at = -1;
  for (const l of sub) { const i = labels.indexOf(l, at + 1); if (i < 0) return false; at = i; }
  return true;
};

module.exports = { goodEdge, escapeRe, PKG, ROOT, WF_DIR, SKILL, launch, engineerReturn, planReturn, integrateReturn, refsIn, layerRefsIn, gate, goodAtt, goodG3, goodG4, goodProfileCmd, ok, done, inOrder };
