#!/usr/bin/env node
// Regression guard for the standalone workload declaration (no GPU, no model needed).
//
// GEAK runs both under an orchestrator and on its own. Under Hyperloom,
// interface/run_e2e.py exports BENCH_CLIENT / E2E_METRIC / AGENTX_* into the environment GEAK
// inherits. Standalone there is no such parent, so e2e_workflow.js declares the workload itself:
// it composes a bench_env.sh body (AGENTX_ENV) that the Director drops beside the copied bench
// script, and injects a workload-identity block into every role prompt.
//
// Two invariants are under test, and the first is the important one:
//
//   1. FIXED ISL/OSL IS UNTOUCHED. With no workload declaration, AGENTX_ENV is '' (so no
//      bench_env.sh is ever written and bench_e2e.sh's source is a no-op), workloadIdentityBlock()
//      is '' (so every role prompt is byte-identical), and the WORKLOAD object keeps exactly the
//      isl/osl/conc it always had. The synthetic path must not be able to notice this feature.
//
//   2. THE DECLARATION IS AUTHORITY-ORDERED. Every line the env body emits assigns only when the
//      name is unset or empty, so a real exported value always outranks the file. That is what keeps
//      an orchestrated run — which exports these same names for its own reasons — behaving exactly as
//      it did before, and what lets an operator override one knob without editing anything.
//      The form is asserted too, because the obvious shorter spelling is silently wrong: inside a
//      `${K:='v'}` expansion the quotes are literal characters of the value, not shell quoting, so
//      that version assigns the value with its quotes attached and every consumer sees `'agentx'`.
//
// The functions are extracted from the REAL workflow source and executed, rather than re-implemented
// here, so drift in the shipped code fails this test instead of hiding behind a copy.
//
// Run:  node e2e_workflow/scripts/test_agentx_declaration.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const WF = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');
const src = fs.readFileSync(WF, 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

// ── Rebuild the declaration block + the prompt injector with controlled inputs ──────────────────
// Slice the contiguous region that defines WORKLOAD_SPEC..AGENTX_ENV, plus the standalone
// workloadIdentityBlock() function, and evaluate them with args/ISL/OSL/CONC injected.
const declStart = src.indexOf('const WORKLOAD_SPEC =');
const declEnd = src.indexOf("].join('\\n');", declStart);
ok(declStart > 0 && declEnd > declStart, 'workload declaration block located in the workflow source');
const decl = src.slice(declStart, declEnd + "].join('\\n');".length);

const blkMatch = src.match(/function workloadIdentityBlock\(\) \{[\s\S]*?\n\}/);
ok(!!blkMatch, 'workloadIdentityBlock() defined');
if (declStart < 0 || !blkMatch) { console.log('\nFAILED: cannot locate the code under test.'); process.exit(1); }

// The injector must short-circuit on its FIRST statement; a later return would already have
// interpolated agentx text into a synthetic prompt.
ok(/function workloadIdentityBlock\(\) \{\s*\n\s*if \(!IS_AGENTX\) return '';/.test(src),
  "workloadIdentityBlock() opens with `if (!IS_AGENTX) return '';`");

// The sliced region now also RESOLVES the kernel-targeting shape, so ISL/OSL and
// the provenance are outputs of the code under test rather than inputs we inject.
// Only the three raw arg derivations are mirrored here; the assertions below pin
// that mirroring to the source so a change there fails this test.
ok(/const ISL_DECLARED = A\.isl != null \? parseInt\(A\.isl, 10\) : null;/.test(src),
  'ISL_DECLARED is derived straight from args.isl (mirrored by this harness)');
ok(/const OSL_DECLARED = A\.osl != null \? parseInt\(A\.osl, 10\) : null;/.test(src),
  'OSL_DECLARED is derived straight from args.osl (mirrored by this harness)');

function build(args) {
  const islDeclared = args.isl != null ? parseInt(args.isl, 10) : null;
  const oslDeclared = args.osl != null ? parseInt(args.osl, 10) : null;
  const conc = parseInt(args.conc != null ? args.conc : 64, 10);
  return new Function('A', 'ISL_DECLARED', 'OSL_DECLARED', 'CONC',
    `${decl}\n${blkMatch[0]}\nreturn { WORKLOAD_KIND, IS_AGENTX, AGENTX, AGENTX_ENV, AGENTX_METRIC_BASIS, AGENTX_E2E_METRIC, ISL, OSL, WORKLOAD, WORKLOAD_SHAPE_PROVENANCE, SHAPE_IS_MEASURED, workloadIdentityBlock };`
  )(args, islDeclared, oslDeclared, conc);
}

// ── 1. Synthetic (no declaration) is completely inert ───────────────────────────────────────────
console.log('\n# no declaration => the fixed ISL/OSL path cannot notice this feature');
for (const [label, args] of [
  ['args = {}', {}],
  ['workload_kind absent, unrelated args', { isl: 4096, osl: 512, conc: 8 }],
  ['workload_spec = {} (empty object)', { workload_spec: {} }],
  ['workload_kind = "synthetic"', { workload_kind: 'synthetic' }],
  ['a non-agentx workload_spec.kind', { workload_spec: { kind: 'fixed_isl_osl' } }],
]) {
  const m = build(args);
  ok(m.AGENTX_ENV === '', `${label}: AGENTX_ENV is '' (no bench_env.sh is written)`);
  ok(m.AGENTX === null, `${label}: AGENTX is null`);
  ok(m.IS_AGENTX === false, `${label}: IS_AGENTX false`);
  ok(m.workloadIdentityBlock() === '', `${label}: role prompts get '' (byte-identical)`);
}

// A malformed workload_spec must not switch the workload on, and must not throw either.
for (const bad of [null, 'agentx_trace_replay', 42, []]) {
  let m;
  try { m = build({ workload_spec: bad }); } catch (e) { m = { threw: e.message }; }
  ok(m && !m.threw && m.AGENTX_ENV === '',
    `workload_spec=${JSON.stringify(bad)} is ignored, not fatal, and stays synthetic`);
}

// ── 2. The shorthand alone yields the canonical AgentX setup ────────────────────────────────────
console.log('\n# workload_kind shorthand => canonical setup, no other args required');
const sc = build({ workload_kind: 'agentx_trace_replay', conc: 8 });
ok(sc.IS_AGENTX === true, 'shorthand switches the workload on');
ok(sc.AGENTX.num_entries === 393, 'canonical corpus size 393');
ok(sc.AGENTX.duration_s === 3600, 'canonical window 3600s');
ok(sc.AGENTX.geak_loop_duration_s === 900, 'search legs run the 900s scenario floor');
ok(sc.AGENTX.scenario === 'inferencex-agentx-mvp', 'canonical scenario');
// Pin the exact name, not a prefix. A prefix match accepts the _256k sibling,
// which drops every request over 256k tokens (98.8k -> 68.3k) and is meant for
// ~256k-context servers; the campaign serves at 1048576, so replaying it would
// measure a lighter workload and produce a baseline nothing can be compared to.
ok(sc.AGENTX.corpus === 'semianalysis_cc_traces_weka_062126', 'canonical full-context weka corpus');
ok(sc.AGENTX_METRIC_BASIS === 'aggregate_total_token_tok_s', 'graded on total tokens by default');
ok(sc.AGENTX_E2E_METRIC === 'total', "metric basis maps to E2E_METRIC=total");

// nested spec spelling must be equivalent to the shorthand
const nested = build({ workload_spec: { kind: 'agentx_trace_replay' }, conc: 8 });
ok(nested.AGENTX_ENV === sc.AGENTX_ENV, 'workload_spec.kind and workload_kind produce the same env');

// ── 3. Authority order: the file may only ever assign when the name is empty ────────────────────
console.log('\n# every assignment defers to an inherited value (orchestrated runs keep their behaviour)');
const lines = sc.AGENTX_ENV.split('\n').filter((l) => l && !l.startsWith('#') && !l.startsWith('export '));
ok(lines.length > 0, 'the env body assigns something');
const GUARD = /^\[ -n "\$\{([A-Z_][A-Z0-9_]*):-\}" \] \|\| \1=/;
const bad = lines.filter((l) => !GUARD.test(l));
ok(bad.length === 0, `every assignment is guarded on the same name (offenders: ${JSON.stringify(bad)})`);
ok(!/(^|\n)[A-Z_]+=/.test(sc.AGENTX_ENV), 'no unguarded VAR= line can clobber an exported value');
// The `${K:='v'}` trap: quotes inside that expansion are literal, so a value would arrive quoted.
ok(!/:=/.test(sc.AGENTX_ENV), 'the body avoids ${K:=...}, whose quotes would become part of the value');
ok(lines.every((l) => /=(''|'(?:[^']|'\\'')*')$/.test(l)),
  'every value is wrapped in real shell single quotes');
ok(/^export /m.test(sc.AGENTX_ENV), 'the declared names are exported to child processes');
ok(sc.AGENTX_ENV.startsWith('#!/usr/bin/env bash'), 'the body is a sourceable bash file');

// ── 4. The knobs the measurement actually depends on ────────────────────────────────────────────
console.log('\n# the declaration sets what bench_e2e.sh and the client adapter read');
const has = (name, val) => new RegExp(`\\[ -n "\\$\\{${name}:-\\}" \\] \\|\\| ${name}='${val}'$`, 'm').test(sc.AGENTX_ENV);
ok(has('BENCH_CLIENT', 'agentx'), 'BENCH_CLIENT=agentx (the trace-replay client, not a synthetic sweep)');
ok(has('E2E_METRIC', 'total'), 'E2E_METRIC=total (the axis bench_e2e.sh medians)');
ok(has('GEAK_METRIC_BASIS', 'aggregate_total_token_tok_s'), 'GEAK_METRIC_BASIS records the declared basis');
ok(has('GEAK_ISL_OSL_INACTIVE', '1'), 'GEAK_ISL_OSL_INACTIVE=1 (adapters refuse to treat isl/osl as the load)');
ok(has('GEAK_WORKLOAD_KIND', 'agentx_trace_replay'), 'GEAK_WORKLOAD_KIND names the workload');
ok(has('REPEATS', '1'), 'REPEATS=1 (a duration-bounded window is not repeated 3x)');
ok(has('CONC', '8'), 'CONC comes from the declaration');
ok(has('AGENTX_NUM_ENTRIES', '393') && has('GEAK_AGENTX_DURATION_S', '3600')
  && has('GEAK_AGENTX_LOOP_DURATION_S', '900'), 'corpus size and both durations are declared');
ok(has('AGENTX_WARMUP_REQUESTS_PER_LANE', '10') && has('AGENTX_WARMUP_GRACE_PERIOD', '1800'),
  'warmup policy is declared');

// Optional knobs appear only when asked for: an empty INFERENCEX_PATH must not be declared at all,
// or it would shadow a real one inherited from the environment.
ok(!/INFERENCEX_PATH/.test(sc.AGENTX_ENV), 'INFERENCEX_PATH is absent unless the caller supplies it');
ok(!/AIPERF_BIN/.test(sc.AGENTX_ENV), 'AIPERF_BIN is absent unless the caller supplies it');
const withOpt = build({ workload_kind: 'agentx_trace_replay',
  workload_spec: { kind: 'agentx_trace_replay', inferencex_path: '/ix', aiperf_bin: '/b/aiperf',
    profile_warmup_s: 120, profile_window_s: 25 } });
ok(/\|\| INFERENCEX_PATH='\/ix'$/m.test(withOpt.AGENTX_ENV), 'a supplied INFERENCEX_PATH is declared');
ok(/\|\| AGENTX_PROFILE_WARMUP_S='120'$/m.test(withOpt.AGENTX_ENV), 'profile window placement is declarable');

// The failed-request tolerance is one of those optional knobs, and for a specific reason: pinning
// it here would freeze one tolerance across the whole run, and the client adapter is the only
// place that knows whether the leg it is about to run is exploring (a bad tail costs one probe) or
// producing a number that will be compared (where failures are the long trajectories, so dropping
// them flatters whichever leg crashed). Declaring it must still win, for a run that needs one
// tolerance end to end.
ok(!/AGENTX_FAILED_REQUEST_THRESHOLD/.test(sc.AGENTX_ENV),
  'no failed-request threshold is pinned by default, so the client can set it per measurement purpose');
const withThresh = build({ workload_spec: { kind: 'agentx_trace_replay',
  failed_request_threshold: 0.02 } });
ok(/\|\| AGENTX_FAILED_REQUEST_THRESHOLD='0.02'$/m.test(withThresh.AGENTX_ENV),
  'a declared failed-request threshold is still pinned for every leg');

// ── 5. workload_spec overrides the canonical defaults, field by field ───────────────────────────
console.log('\n# an explicit spec overrides the defaults it names, and only those');
const ov = build({ workload_spec: {
  kind: 'agentx_trace_replay', num_entries: 50, geak_loop_duration_s: 300,
  metric_basis: 'aggregate_output_tok_s', concurrency: 16, corpus: 'my_corpus',
} });
ok(ov.AGENTX.num_entries === 50 && ov.AGENTX.geak_loop_duration_s === 300, 'named fields override');
ok(ov.AGENTX.duration_s === 3600, 'unnamed fields keep the canonical value');
ok(ov.AGENTX_E2E_METRIC === 'output', 'an output basis maps to E2E_METRIC=output');
ok(/\|\| E2E_METRIC='output'$/m.test(ov.AGENTX_ENV), 'the output basis reaches the env body');
ok(/\|\| CONC='16'$/m.test(ov.AGENTX_ENV), 'spec concurrency wins over args.conc');
ok(/\|\| AGENTX_CANONICAL_DATASET='my_corpus'$/m.test(ov.AGENTX_ENV),
  'canonical dataset defaults to the pinned corpus when not stated separately');

// Values are shell-quoted, so a corpus name containing a quote cannot break the sourced file.
const nasty = build({ workload_spec: { kind: 'agentx_trace_replay', corpus: "a'b" } });
ok(/\|\| AGENTX_DATASET='a'\\''b'/.test(nasty.AGENTX_ENV), 'values are single-quote escaped for the shell');

// ── 6. The prompt block states the traps this workload sets ─────────────────────────────────────
console.log('\n# every role is told what it is measuring');
const blk = sc.workloadIdentityBlock();
ok(blk !== '', 'declared run => non-empty block');
ok(/WORKLOAD IDENTITY/.test(blk), 'block is headed as the workload identity');
ok(/KERNEL-TARGETING shape, not a benchmark/.test(blk),
  'isl/osl are labelled a kernel-targeting shape, not a benchmark');
ok(/NEVER build a synthetic sweep/.test(blk), 'synthetic sweeps are forbidden explicitly');
ok(/bench_env\.sh/.test(blk), 'the block points at the file that owns client/metric selection');
ok(/Do NOT set/.test(blk) && /BENCH_CLIENT/.test(blk), 'roles are told not to set the client themselves');
ok(/ISL=<isl> OSL=<osl> CONC=<conc>/.test(blk),
  'roles are told to keep their existing bench line unchanged (no role-file churn needed)');
ok(/throughput_tok_s_median/.test(blk), 'the metric-neutral summary key is named');
ok(blk.includes(String(sc.AGENTX.geak_loop_duration_s)) && blk.includes(String(sc.AGENTX.duration_s)),
  'both measurement windows are stated so phases can budget');
ok(/non-canonical/.test(blk) && /submission_valid=false/.test(blk),
  'the non-canonical search-leg stamp is explained rather than left to surprise a role');

// ── 7. The kernel-targeting shape: a real regime, or an honest placeholder ─────────────────────
// Nothing is MEASURED at isl/osl here -- aiperf owns the sequence lengths. But roles read them when
// they size tiles and synthesize GEMM inputs, so the old 1024/1024 default aimed the whole kernel
// search ~110x below the real load while a provenance string claimed 'handoff_workload' even
// standalone, where no handoff exists. The measurement was never affected; every kernel choice was.
console.log('\n# isl/osl resolve to a real regime for kernel work, or say plainly that they did not');

// No corpus average is compiled into the workflow: the corpus is declarable, the
// numbers move with tokenizer and window, and a stale constant would be read as
// fact by every role. So before anything is measured the shape is openly pending.
ok(!/112729|112,729/.test(src),
  'the workflow hardcodes no corpus request-shape average anywhere');
const shape = build({ workload_kind: 'agentx_trace_replay' });
ok(shape.WORKLOAD_SHAPE_PROVENANCE === 'agentx_pending_baseline',
  'before the baseline runs the shape is marked pending, not asserted');
ok(shape.SHAPE_IS_MEASURED === false, 'a pending shape is not marked measured');
ok(/function adoptMeasuredShape\(/.test(src)
  && /WORKLOAD_SHAPE_PROVENANCE = 'agentx_measured_this_run'/.test(src),
  'the run installs the shape it measured (adoptMeasuredShape)');
ok(/res\['observed_isl'\] = _num\(_s\.get\('observed_isl'\)\)/.test(src),
  "the post-Setup probe reads the baseline's served shape out of bench_summary.json");
ok(/adoptMeasuredShape\(verdict\.observed_isl, verdict\.observed_osl/.test(src),
  'and feeds it into the kernel-targeting shape');

const declShape = build({ workload_kind: 'agentx_trace_replay', isl: 104757, osl: 742 });
ok(declShape.ISL === 104757 && declShape.OSL === 742,
  'an explicit isl/osl on an agentx run wins over the corpus default');
ok(declShape.WORKLOAD_SHAPE_PROVENANCE === 'agentx_declared_args',
  'a caller-supplied shape is labelled as declared, not passed off as a corpus measurement');

// An explicit observed_* is someone stating what they measured on THIS stack, so
// it outranks our table and takes the same label the Hyperloom path uses.
const obs = build({ workload_spec: { kind: 'agentx_trace_replay',
  observed_isl: 90000, observed_osl: 600 } });
ok(obs.ISL === 90000 && obs.WORKLOAD_SHAPE_PROVENANCE === 'agentx_observed',
  'workload_spec.observed_isl/observed_osl outrank the measured-shape table');

// Hyperloom resolves this upstream in interface/run_e2e.py::_targeting_shape.
const hl = build({ workload_kind: 'agentx_trace_replay', isl: 5, osl: 6,
  workload_shape_provenance: 'agentx_observed' });
ok(hl.ISL === 5 && hl.WORKLOAD_SHAPE_PROVENANCE === 'agentx_observed',
  'an upstream-resolved shape and its label survive verbatim (one resolution, two entry points)');

// Whatever corpus is declared -- a new AgentX release, a capped variant, another
// tokenizer -- the pre-baseline state is the same honest "not yet measured". No
// corpus can inherit another's average, because none is stored.
for (const corpus of ['semianalysis_cc_traces_weka_062126_256k', 'some_future_corpus_v9']) {
  const other = build({ workload_spec: { kind: 'agentx_trace_replay', corpus } });
  ok(other.ISL === 1024 && other.OSL === 1024,
    `${corpus}: no shape is invented before the baseline`);
  ok(other.WORKLOAD_SHAPE_PROVENANCE === 'agentx_pending_baseline',
    `${corpus}: the pending state is reported, not a borrowed measurement`);
  ok(other.AGENTX.corpus === corpus && other.AGENTX_ENV !== '',
    `${corpus}: still fully runnable; only the shape claim waits`);
}
const fb = build({ workload_spec: { kind: 'agentx_trace_replay', corpus: 'some_future_corpus_v9' } });

const fbBlk = fb.workloadIdentityBlock();
ok(/A PLACEHOLDER, NOT THIS CORPUS/.test(fbBlk), 'the prompt labels a placeholder shape as one');
ok(/Do NOT size tiles/.test(fbBlk), 'and tells roles not to size kernels from it');
ok(/observed_isl\/observed_osl/.test(fbBlk),
  'and points a role at the baseline summary for the real served shape');
// A shape that IS known (declared here; measured-this-run at runtime) reads as usable.
const usableBlk = declShape.workloadIdentityBlock();
ok(/right regime to size tiles/.test(usableBlk) && !/A PLACEHOLDER/.test(usableBlk),
  'a known shape is instead presented as usable for kernel sizing');
// The prompt must not hardcode one corpus's distribution: the corpus is
// declarable and AgentX keeps changing it, so a baked-in average goes stale
// silently and would be read as fact by every role.
for (const stale of [/112\.?7k/, /~?89k/, /500k at p99/]) {
  ok(!stale.test(fbBlk) && !stale.test(shape.workloadIdentityBlock()),
    `the block states no hardcoded corpus distribution (${stale})`);
}

// The synthetic path keeps its shape, and stops claiming a handoff it never had.
ok(build({}).ISL === 1024 && build({}).OSL === 1024,
  'a bare synthetic run keeps the 1024/1024 default shape exactly as before');
ok(build({}).WORKLOAD_SHAPE_PROVENANCE === 'synthetic_default',
  'a bare synthetic run reports synthetic_default, not a fabricated handoff_workload');
ok(build({ isl: 4096, osl: 512 }).WORKLOAD_SHAPE_PROVENANCE === 'declared_args',
  'an explicit synthetic isl/osl reports declared_args');
ok(build({ isl: 4096, osl: 512 }).ISL === 4096,
  'and is used verbatim, so the fixed ISL/OSL path is unchanged');

// ── 8. The interpolation site keeps a synthetic prompt whitespace-identical ─────────────────────
console.log('\n# the injection site adds nothing when the block is empty');
ok(/keep the two separate\.\n\$\{workloadIdentityBlock\(\)\}\n## Inputs/.test(src),
  'block is interpolated on its own line between the serving invariant and ## Inputs, ' +
  'so an empty block leaves the original blank line exactly as it was');

console.log(failures === 0
  ? '\nPASS: no declaration => fixed ISL/OSL is byte-identical; a declaration is authority-ordered.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
