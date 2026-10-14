#!/usr/bin/env node
// End-to-end regression for a STANDALONE AgentX trace-replay run, on CPU: no GPU, no model, no
// network. It follows one run from the workflow's own declaration to the number Setup grades:
//
//   e2e_workflow.js's declaration -> the bench_env.sh body it hands the Director
//   -> the eval dir laid out as roles/director.md step 3 lays it out
//   -> bench_e2e.sh under each lifecycle the run uses (warm_server, isolated_server, legacy)
//   -> adapters/clients/agentx.sh -> aiperf -> the vendored map_aiperf.py -> bench_summarize.py
//   -> the Setup check in e2e_workflow.js that refuses a baseline measured on another axis.
//
// Every stage above is the shipped file, extracted or copied, never re-implemented. Two things are
// stand-ins: the serving backend (a sleep is the server) and aiperf (it records its argv and writes
// the profile export a finished replay leaves behind). That is what lets this check the joins no
// unit test sees: that the declared knobs reach the client that measures, that the graded number is
// the axis the workflow declared, and that each lifecycle takes the sample count it asks for.
//
// Run:  node e2e_workflow/scripts/test_agentx_standalone_e2e.js
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..', '..');
const SKILL_DIR = path.join(ROOT, 'e2e_workflow');
const src = fs.readFileSync(path.join(SKILL_DIR, 'e2e_workflow.js'), 'utf8');
const director = fs.readFileSync(path.join(SKILL_DIR, 'roles', 'director.md'), 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };
const near = (a, b) => typeof a === 'number' && Math.abs(a - b) < 1e-3;
const bail = (msg) => { console.error(`\nFAILED: ${msg}`); process.exit(1); };

// ── The code under test, sliced from the real workflow source ───────────────────────────────────
const declStart = src.indexOf('const WORKLOAD_SPEC =');
const declEnd = src.indexOf("].join('\\n');", declStart);
const setupStart = src.indexOf('// ── The trace-replay declaration must have TAKEN EFFECT');
const setupEnd = src.lastIndexOf('\n  // =====', src.indexOf('// MODULE A', setupStart));
if (declStart < 0 || declEnd < 0 || setupStart < 0 || setupEnd < setupStart) {
  bail('cannot locate the declaration or the Setup check in e2e_workflow.js');
}
const decl = src.slice(declStart, declEnd + "].join('\\n');".length);
const setupCheck = src.slice(setupStart, setupEnd);

// The declaration reads an exported E2E_METRIC; this run must not inherit one from the shell.
const INHERITED_E2E_METRIC = process.env.E2E_METRIC;
delete process.env.E2E_METRIC;

// Evaluates the declaration and, given an eval dir, the Setup check after it -- in one scope, so
// the shape Setup installs lands in the same WORKLOAD the roles read.
function workflow(args, evalDir) {
  const logs = [];
  const body = `${decl}\n${evalDir ? setupCheck : ''}\n` +
    'return { AGENTX_ENV, AGENTX_METRIC_BASIS, AGENTX_E2E_METRIC, WORKLOAD, ' +
    'WORKLOAD_SHAPE_PROVENANCE, SHAPE_IS_MEASURED };';
  const out = new Function('A', 'ISL_DECLARED', 'OSL_DECLARED', 'EVAL_DIR', 'log', 'require', body)(
    args, null, null, evalDir || '', (m) => logs.push(String(m)), require);
  return Object.assign(out, { logs });
}
function setupError(args, evalDir) {
  try { workflow(args, evalDir); return ''; } catch (e) { return String(e && e.message); }
}

// ── Stand-ins: the server and aiperf ────────────────────────────────────────────────────────────
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'agentx_e2e_'));
const FIX = path.join(TMP, 'fixtures');
fs.mkdirSync(path.join(FIX, 'bin'), { recursive: true });
const EVENTS = path.join(FIX, 'events.jsonl');

const FAKE_BACKEND = path.join(FIX, 'fake_backend.sh');
fs.writeFileSync(FAKE_BACKEND, `
adapter_default_port() { echo 18731; }
adapter_launch() {
  printf '{"event":"launch"}\\n' >> "$E2E_EVENTS"
  \${SERVER_LAUNCH_PREFIX:-} sleep 120 &
  SERVER_PID=$!
}
adapter_health() { return 0; }
# The synthetic client, which the agentx client replaces whenever bench_env.sh selects it.
adapter_bench() {
  local n
  n=$(( $(cat "$E2E_EVENTS.native" 2>/dev/null || echo 0) + 1 ))
  printf '%s' "$n" > "$E2E_EVENTS.native"
  printf '{"event":"native_bench","call":%s}\\n' "$n" >> "$E2E_EVENTS"
  printf '{"output_throughput":%s}\\n' "$((n * 100))" >> "$RESULT_JSONL"
}
`);

// Call n reports P90 ITL = 20+n ms, so a test can tell exactly which replay a summary reports.
const OUT_TPUT = 150.0;
const IN_TPUT = 21000.0;
const ISL_AVG = 52500;
const OSL_AVG = 375;
const p90For = (call) => 1000 / (20 + call);
const AIPERF = path.join(FIX, 'bin', 'aiperf');
fs.writeFileSync(AIPERF, `#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
events = os.environ["E2E_EVENTS"]
n = 1
if os.path.exists(events):
    n += sum('"aiperf"' in line for line in open(events))
with open(events, "a") as fh:
    fh.write(json.dumps({"event": "aiperf", "call": n, "argv": argv}) + "\\n")
art = argv[argv.index("--artifact-dir") + 1]
os.makedirs(art, exist_ok=True)
rc = 400
export = {
    "metadata": {"submission_valid": True},
    "metrics": {
        "request_count": rc,
        "output_token_throughput": {"avg": ${OUT_TPUT}},
        "input_token_throughput": {"avg": ${IN_TPUT}},
        "input_sequence_length": {"avg": ${ISL_AVG}},
        "total_output_tokens": rc * ${OSL_AVG},
        "time_to_first_token": {"avg": 900.0, "p50": 850.0, "p90": 2400.0, "p99": 5000.0},
        "inter_token_latency": {"avg": 18.0, "p50": 17.0, "p90": 20.0 + n, "p99": 60.0},
        "e2e_output_token_throughput": {"p10": 9.0, "p50": 30.0},
    },
}
json.dump(export, open(os.path.join(art, "profile_export_aiperf.json"), "w"))
`);
fs.chmodSync(AIPERF, 0o755);

const events = () => !fs.existsSync(EVENTS) ? []
  : fs.readFileSync(EVENTS, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const resetEvents = () => { for (const f of [EVENTS, `${EVENTS}.native`]) fs.rmSync(f, { force: true }); };
const flag = (argv, name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
const readJson = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
const rows = (p) => fs.readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));

// ── The eval dir, staged from director.md step 3 itself ─────────────────────────────────────────
const step3 = director.slice(director.indexOf('3. Build the layout'), director.indexOf('\n4. '));
const block = (step3.match(/```bash\n([\s\S]*?)```/) || [])[1] || '';
const layout = ((block.match(/mkdir -p "\$EVAL_DIR"\/\{([^}]+)\}/) || [])[1] || '').split(',');
const copies = [...block.matchAll(/^\s*cp (-r )?"\$SKILL_DIR\/scripts\/([^"]+)" "\$EVAL_DIR\/([^"]+)"/gm)];
const staged = copies.map((m) => m[3]);
if (!layout.includes('baseline') || !staged.includes('bench_e2e.sh')) {
  bail('cannot read the eval-dir layout out of roles/director.md step 3');
}
function stage(name, benchEnv) {
  const evalDir = path.join(TMP, name);
  for (const d of layout) fs.mkdirSync(path.join(evalDir, d), { recursive: true });
  for (const [, recursive, from, to] of copies) {
    const s = path.join(SKILL_DIR, 'scripts', from);
    if (recursive) fs.cpSync(s, path.join(evalDir, to), { recursive: true });
    else fs.copyFileSync(s, path.join(evalDir, to));
  }
  if (benchEnv) fs.writeFileSync(path.join(evalDir, 'bench_env.sh'), benchEnv);
  return evalDir;
}

// A bench run as the Director issues it (step 5), in an environment holding nothing this test did
// not put there: an inherited BENCH_CLIENT/E2E_METRIC/AGENTX_* would outrank bench_env.sh.
const SCRUB = /^(BENCH_|E2E_|AGENTX_|GEAK_|MEASUREMENT_|AIPERF_|INFERENCEX_PATH$|REPEATS$|REPLICAS$|CONC$|ISL$|OSL$|PROFILE$|WEKA_)/;
function bench(evalDir, workload, extra) {
  const env = {};
  for (const [k, v] of Object.entries(process.env)) if (!SCRUB.test(k)) env[k] = v;
  Object.assign(env, {
    PATH: `${path.join(FIX, 'bin')}:${process.env.PATH}`,
    E2E_EVENTS: EVENTS,
    ADAPTER: FAKE_BACKEND,
    SERVING_GPU_LOCK_DISABLE: '1',
    SERVER_STOP_GRACE_S: '0',
    BACKEND: 'fake', GPU: '0', TP: '1', MODEL: path.join(TMP, 'model'),
    ISL: String(workload.isl), OSL: String(workload.osl), CONC: String(workload.conc),
    PROFILE: '0', GEAK_REPEAT_MODE: 'warm_server', MEASUREMENT_PURPOSE: 'parity', REPLICAS: '1',
    EFFECTIVE_CONFIG_DIGEST: '', OVERLAY_PYTHONPATH: '', EXTRA_SERVER_ARGS: '', EXTRA_ENV: '',
    OUT_DIR: path.join(evalDir, 'baseline'),
  }, extra || {});
  for (const [k, v] of Object.entries(extra || {})) if (v === null) delete env[k];
  resetEvents();
  const proc = spawnSync('bash', [path.join(evalDir, 'bench_e2e.sh')],
    { env, encoding: 'utf8', timeout: 120000 });
  const summaryPath = path.join(env.OUT_DIR, 'bench_summary.json');
  return {
    proc, out: env.OUT_DIR, events: events(),
    summary: fs.existsSync(summaryPath) ? readJson(summaryPath) : null,
  };
}
const aiperfCalls = (run) => run.events.filter((e) => e.event === 'aiperf');
const launches = (run) => run.events.filter((e) => e.event === 'launch').length;
const why = (run) => `rc=${run.proc.status}\n${String(run.proc.stdout).slice(-1500)}\n${String(run.proc.stderr).slice(-1500)}`;

try {
  // ── 1. The declaration the Director is handed ─────────────────────────────────────────────────
  // The shorthand, a spec override of the concurrency, and the synthetic args.conc a caller may
  // still pass: the declared concurrency has to be the one every bench line carries.
  console.log('\n# the standalone declaration');
  const ARGS = { workload_kind: 'agentx_trace_replay', workload_spec: { concurrency: 4 }, conc: 64 };
  const wf = workflow(ARGS);
  ok(wf.AGENTX_ENV.length > 0, 'a trace replay declares a bench_env.sh body');
  ok(wf.AGENTX_METRIC_BASIS === 'p90_intvty_inferencex',
    'graded on InferenceX P90 interactivity when nothing else is declared');
  ok(wf.WORKLOAD.conc === 4, 'the declared concurrency is the one the Director puts on bench lines');

  // ── 2. The eval dir, laid out as director.md step 3 says ──────────────────────────────────────
  console.log('\n# the eval dir, staged from director.md step 3');
  for (const need of ['bench_e2e.sh', 'bench_replica.sh', 'server_teardown.sh', 'bench_summarize.py', 'adapters']) {
    ok(staged.includes(need), `step 3 stages ${need}`);
  }
  const EVAL_DIR = stage('eval_agentx', wf.AGENTX_ENV);
  ok(fs.statSync(path.join(EVAL_DIR, 'bench_env.sh')).size > 0, 'bench_env.sh is written verbatim before any bench');
  ok(fs.existsSync(path.join(EVAL_DIR, 'adapters', 'clients', 'map_aiperf.py')),
    'the vendored mapper travels with the copied adapters, so no InferenceX checkout is needed');

  // ── 3. The baseline: the Director's own command line ──────────────────────────────────────────
  console.log('\n# the baseline: warm_server parity, exactly as the Director issues it');
  const base = bench(EVAL_DIR, wf.WORKLOAD);
  ok(base.proc.status === 0, `the baseline bench exits 0 (${base.proc.status === 0 ? 'ok' : why(base)})`);
  const s = base.summary || {};
  const calls = aiperfCalls(base);
  ok(launches(base) === 1, 'one server for the whole leg');
  ok(calls.length === 2, 'one discarded full replay to warm the cache, then the timed replay');
  ok(s.metric_basis === wf.AGENTX_METRIC_BASIS, `the summary is on the declared axis (${s.metric_basis})`);
  ok(near(s.throughput_tok_s_median, p90For(2)),
    `the graded number is 1000 / P90 ITL of the TIMED replay (${s.throughput_tok_s_median}), not the warmup's`);
  ok(near(s.guard_aggregate_output_tok_s_median, OUT_TPUT), 'output tok/s rides beside it as the guard');
  ok(!('guard_total_tok_s_median' in s) && !('guard_basis' in s),
    'the removed total-throughput guard fields stay absent');
  ok(near(s.observed_isl, ISL_AVG) && near(s.observed_osl, OSL_AVG), 'the served request shape is reported');
  for (const c of calls) {
    ok(flag(c.argv, '--concurrency') === '4', `replay ${c.call} runs at the declared concurrency, not args.conc`);
    ok(flag(c.argv, '--benchmark-duration') === '3600', `replay ${c.call} runs the canonical 3600s window`);
    ok(flag(c.argv, '--public-dataset') === 'semianalysis_cc_traces_weka_062126'
      && flag(c.argv, '--num-dataset-entries') === '393', `replay ${c.call} replays the canonical corpus`);
    ok(flag(c.argv, '--failed-request-threshold') === '0.01', `replay ${c.call} tolerates no crashed tail`);
    ok(!c.argv.includes('--unsafe-override'), `replay ${c.call} needs no override`);
  }
  const baseRow = rows(path.join(base.out, 'bench_runs.jsonl'));
  ok(baseRow.length === 1 && baseRow[0].submission_valid === true
    && baseRow[0].submission_invalid_reasons.length === 0,
  'only the timed row is kept, and a canonical replay stays submittable');

  // ── 4. Setup's own check accepts it and installs the served shape ─────────────────────────────
  console.log("\n# the workflow's Setup check, run against what the baseline wrote");
  const setup = workflow(ARGS, EVAL_DIR);
  ok(setup.logs.some((l) => /trace-replay declaration verified/.test(l)), 'Setup verifies the declaration took effect');
  ok(setup.WORKLOAD.isl === ISL_AVG && setup.WORKLOAD.osl === OSL_AVG
    && setup.WORKLOAD_SHAPE_PROVENANCE === 'agentx_measured_this_run',
  'and kernels are targeted at the shape the baseline served, not the 1024/1024 placeholder');

  // ── 5. A search leg runs the scenario floor and can never be kept ─────────────────────────────
  console.log('\n# a search leg');
  const search = bench(EVAL_DIR, wf.WORKLOAD,
    { MEASUREMENT_PURPOSE: 'search', OUT_DIR: path.join(EVAL_DIR, 'search_leg') });
  ok(search.proc.status === 0, `the search leg exits 0 (${search.proc.status === 0 ? 'ok' : why(search)})`);
  const sc = aiperfCalls(search);
  ok(sc.length === 2 && sc.every((c) => flag(c.argv, '--benchmark-duration') === '900'
    && c.argv.includes('--unsafe-override') && flag(c.argv, '--failed-request-threshold') === '0.10'),
  'it replays the 900s floor, which needs the override, at the exploring tolerance');
  const searchRow = rows(path.join(search.out, 'bench_runs.jsonl'))[0] || {};
  ok(searchRow.submission_valid === false
    && (searchRow.submission_invalid_reasons || []).some((r) => /duration=900s\(canonical 3600s\)/.test(r)),
  'and its number is stamped non-canonical, so it can never be the one that is KEPT');
  ok((search.summary || {}).metric_basis === wf.AGENTX_METRIC_BASIS, 'on the same axis as the baseline');

  // ── 5b. Overriding the corpus does not move the reference it is judged against ────────────────
  console.log('\n# a corpus override');
  const SIBLING = 'semianalysis_cc_traces_weka_062126_256k';
  const OVERRIDE = { workload_kind: 'agentx_trace_replay', workload_spec: { concurrency: 4, corpus: SIBLING } };
  const ovWf = workflow(OVERRIDE);
  const ov = bench(stage('eval_corpus_override', ovWf.AGENTX_ENV), ovWf.WORKLOAD);
  ok(ov.proc.status === 0 && aiperfCalls(ov).every((c) => flag(c.argv, '--public-dataset') === SIBLING),
    'the declared corpus is the one replayed');
  const ovRow = rows(path.join(ov.out, 'bench_runs.jsonl'))[0] || {};
  ok(ovRow.submission_valid === false && (ovRow.submission_invalid_reasons || []).some((r) =>
    r === `corpus=${SIBLING}(canonical semianalysis_cc_traces_weka_062126)`),
  'and even on the canonical window it is stamped non-canonical, so it cannot pass for a leaderboard number');

  // ── 6. Isolated validation takes the replicas it asks for ─────────────────────────────────────
  console.log('\n# isolated validation');
  const iso = bench(EVAL_DIR, wf.WORKLOAD, {
    GEAK_REPEAT_MODE: 'isolated_server', MEASUREMENT_PURPOSE: 'validation', REPLICAS: '3',
    OUT_DIR: path.join(EVAL_DIR, 'validation'),
  });
  ok(iso.proc.status === 0, `the validation leg exits 0 (${iso.proc.status === 0 ? 'ok' : why(iso)})`);
  const v = iso.summary || {};
  ok(launches(iso) === 3 && aiperfCalls(iso).length === 3,
    'REPLICAS=3 is three fresh servers with one replay each -- bench_env.sh does not collapse it to one');
  ok(v.requested_replicas === 3 && v.successful_replicas === 3 && v.status === 'complete',
    'and the aggregate reports all three');
  ok(near(v.throughput_tok_s_median, p90For(2)), 'its number is the median replica, on the declared axis');
  ok(v.metric_basis === wf.AGENTX_METRIC_BASIS && near(v.guard_aggregate_output_tok_s_median, OUT_TPUT),
    'with the guard carried through the aggregate');

  // ── 7. Legacy gives a trace replay one timed round unless asked for more ──────────────────────
  // The lifecycle PROFILE=1 / REPEATS=0 fall back to; each round is a full measured window.
  console.log('\n# the legacy lifecycle');
  const legacyLeg = (extra) => bench(EVAL_DIR, wf.WORKLOAD, Object.assign({
    GEAK_REPEAT_MODE: 'legacy', MEASUREMENT_PURPOSE: 'search', REPLICAS: null,
    OUT_DIR: path.join(EVAL_DIR, `legacy_${Object.keys(extra || {}).join('_') || 'default'}`),
  }, extra || {}));
  const leg = legacyLeg();
  ok(leg.proc.status === 0 && aiperfCalls(leg).length === 2 && (leg.summary || {}).runs === 1,
    'the replay gets its short warmup and ONE timed round, not three');
  const leg2 = legacyLeg({ REPEATS: '2' });
  ok(leg2.proc.status === 0 && aiperfCalls(leg2).length === 3 && (leg2.summary || {}).runs === 2,
    'an explicit REPEATS is still honoured');
  const SYN_DIR = stage('eval_synthetic', '');
  const syn = bench(SYN_DIR, { isl: 1024, osl: 1024, conc: 64 },
    { GEAK_REPEAT_MODE: 'legacy', MEASUREMENT_PURPOSE: 'search', REPLICAS: null });
  ok(syn.proc.status === 0 && aiperfCalls(syn).length === 0
    && syn.events.filter((e) => e.event === 'native_bench').length === 4 && (syn.summary || {}).runs === 3,
  'with no bench_env.sh the synthetic client keeps its warmup + three rounds');

  // ── 8. A baseline on the wrong axis is stopped at Setup, not discovered at the end ────────────
  console.log('\n# what Setup refuses');
  const inheritedDir = stage('eval_inherited', wf.AGENTX_ENV);
  const inherited = bench(inheritedDir, wf.WORKLOAD, { E2E_METRIC: 'total' });
  ok((inherited.summary || {}).metric_basis === 'aggregate_total_token_tok_s',
    'an E2E_METRIC the bench inherited outranks bench_env.sh ...');
  ok(/not comparable/.test(setupError(ARGS, inheritedDir)), '... and Setup refuses that baseline');
  const forgotDir = stage('eval_forgot_env', '');
  bench(forgotDir, wf.WORKLOAD);
  ok(/bench_env\.sh was never\s+written/.test(setupError(ARGS, forgotDir)),
    'a Director that skipped bench_env.sh is caught before the synthetic baseline is used');
} finally {
  fs.rmSync(TMP, { recursive: true, force: true });
  if (INHERITED_E2E_METRIC !== undefined) process.env.E2E_METRIC = INHERITED_E2E_METRIC;
}

console.log(failures === 0
  ? '\nPASS: a standalone AgentX run measures and grades what it declared, end to end.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
