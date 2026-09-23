#!/usr/bin/env node
// Regression guard: the env/flags an ACCEPTED win needs in order to BIND must reach the running
// configuration (no GPU, no model, no KB needed).
//
// The bug this exists for: the banking sites folded `cand.apply_env` only when
// `cand.winner_kind === 'env'`. An authored kernel is winner_kind 'patch', so a patch whose overlay
// needs env to bind — a tuned-table path, a backend selector — fell through BOTH branches. The win
// was banked and the overlay carried forward; the one thing that makes it engage was not. Measured
// on gfx1151: an authoring run banked +10.40% e2e and the binding `GEAK_TUNED_GEMM_TABLE=...` never
// reached the launcher. Accepted, with no way to start it.
//
// That silence had two layers, so this file tests both:
//   (a) the folding itself, and
//   (b) that every banking site actually CALLS it. A correct helper nobody calls reproduces the
//       original bug exactly, and (a) alone would still pass.
//
// The function is EXTRACTED from the shipped source rather than reimplemented here, so this cannot
// pass while e2e_workflow.js drifts.
//
// Run:  node e2e_workflow/scripts/test_accepted_config_binds.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const SRC_PATH = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');
const src = fs.readFileSync(SRC_PATH, 'utf8');

let failures = 0;
const eq = (got, want, msg) => {
  const a = JSON.stringify(got), b = JSON.stringify(want);
  if (a !== b) { console.error('  FAIL:', msg, '\n    got: ', a, '\n    want:', b); failures++; }
  else console.log('  ok:', msg);
};

// ---------------------------------------------------------------- extract the shipped function
const mStart = src.indexOf('function bindAcceptedConfig(');
if (mStart < 0) { console.error('FAIL: bindAcceptedConfig not found in e2e_workflow.js'); process.exit(1); }
// the function body ends at the first line that is exactly '}' at column 0
const mEnd = src.indexOf('\n}\n', mStart);
if (mEnd < 0) { console.error('FAIL: could not delimit bindAcceptedConfig'); process.exit(1); }
const fnSrc = src.slice(mStart, mEnd + 3);

// It mutates the module-scope `curEnv` / `curFlags`, so give it those as locals and hand back an
// accessor. Fresh instance per case: these are accumulators and cross-case bleed would hide a bug.
const mk = (env = '', flags = '') => new Function(
  `let curEnv = ${JSON.stringify(env)}, curFlags = ${JSON.stringify(flags)};\n${fnSrc}\n` +
  `return { bind: bindAcceptedConfig, read: () => ({ env: curEnv, flags: curFlags }) };`)();

// ------------------------------------------------------------------ the drop this exists for
console.log('a patch-kind win whose overlay needs env');
{
  // Verbatim shape of the gfx1151 artifact: gate accepted, winner is a code patch, and the binding
  // env is reported on the integrate result — NOT as a candidate-level `apply_env` lever.
  const h = mk('', '--max-num-seqs 256');
  h.bind({ gate: 'accepted', accepted_overlay: '/x/overlay',
           accepted_env: 'GEAK_TUNED_GEMM_TABLE=/x/config/tuned_gemm_table.json' });
  eq(h.read().env, 'GEAK_TUNED_GEMM_TABLE=/x/config/tuned_gemm_table.json',
     'binding env reaches curEnv even though winner_kind is not "env"');
  eq(h.read().flags, '--max-num-seqs 256', 'unrelated inherited flags untouched');
}

console.log('accumulation onto an existing config');
{
  const h = mk('VLLM_USE_V1=1', '--tp 1');
  h.bind({ accepted_env: 'GEAK_TUNED_GEMM_TABLE=/x/t.json', accepted_flags: '--enable-chunked-prefill' });
  eq(h.read().env, 'VLLM_USE_V1=1 GEAK_TUNED_GEMM_TABLE=/x/t.json', 'env appends, does not replace');
  eq(h.read().flags, '--tp 1 --enable-chunked-prefill', 'flags append, do not replace');
}

console.log('a stacked gate re-reporting the same binding');
{
  // A stacked gate accepts repeatedly and re-reports the binding each time. A flat KEY=VAL list that
  // accumulates duplicates is not merely ugly: `env` resolves last-wins, so a later stale duplicate
  // would silently override the fresh value.
  const h = mk();
  h.bind({ accepted_env: 'GEAK_TUNED_GEMM_TABLE=/x/t.json' });
  h.bind({ accepted_env: 'GEAK_TUNED_GEMM_TABLE=/x/t.json' });
  h.bind({ accepted_env: 'GEAK_TUNED_GEMM_TABLE=/x/t.json VLLM_USE_V1=1' });
  eq(h.read().env, 'GEAK_TUNED_GEMM_TABLE=/x/t.json VLLM_USE_V1=1',
     'repeated binding is deduped; a genuinely new token still lands');
}

console.log('additive for every pre-existing run');
{
  // The fields did not exist in INTEGRATE_SCHEMA before this change, so every CDNA run to date
  // reports neither. Those must come out byte-identical — this is an adaptation, not a behaviour change.
  for (const integ of [null, undefined, {}, { gate: 'accepted', accepted_overlay: '/x/o' },
                       { accepted_env: '' }, { accepted_env: '   ' }, { accepted_flags: '' }]) {
    const h = mk('VLLM_USE_V1=1', '--tp 1');
    h.bind(integ);
    eq(h.read(), { env: 'VLLM_USE_V1=1', flags: '--tp 1' },
       `no-op for ${JSON.stringify(integ) || String(integ)}`);
  }
}

// ------------------------------------------------- (b) a helper nobody calls IS the original bug
console.log('every banking site calls it');
{
  // The original defect was not a wrong fold — it was an absent one. Pin the wiring: every line that
  // banks an accepted overlay must be followed by the bind. Grepping the shipped source is the only
  // way to catch a NEW banking site added later without one, which is precisely how this regresses.
  const lines = src.split('\n');
  const bankRx = /curOverlay = (?:[A-Za-z_$][\w$.]*\.)?accepted_overlay \|\| curOverlay/;
  const missing = [];
  let banks = 0;
  lines.forEach((l, i) => {
    if (!bankRx.test(l)) return;
    banks++;
    // the bind may sit on the same line (semicolon-joined sites) or the next one
    const near = l + '\n' + (lines[i + 1] || '');
    if (!/bindAcceptedConfig\(/.test(near)) missing.push(i + 1);
  });
  eq(missing, [], `all ${banks} accepted-overlay banking sites bind their config`);
  if (banks < 10) { console.error(`  FAIL: only ${banks} banking sites found — regex drifted?`); failures++; }
  else console.log(`  ok: found ${banks} banking sites`);
}

console.log(failures ? `\nFAILED (${failures})` : '\nPASS');
process.exit(failures ? 1 : 0);
