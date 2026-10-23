#!/usr/bin/env node
// mega_fast_test: a full-workflow fast-test mode that REUSES a prior wave's expensive front-matter
// measurements (positive-control calibration) instead of re-running them, so a
// scorable full-workflow iteration costs minutes not ~40 min.
//
// The three properties this test pins, because each is a way the feature could silently go wrong:
//
//   1. DEFAULT-OFF, BYTE-IDENTICAL. When the flag is off the cache helpers make NO agent() call and
//      return null/undefined, and the Benchmark/Profile dispatch falls through to the SAME agentT()
//      call it always made. If that were not true, every existing search-lane wave (and its resume
//      cache key, which is keyed on the agent() call sequence) would change under a default flag.
//
//   2. HONESTY GATE. Reuse is allowed ONLY when a validity key recomputed from disk NOW matches the
//      key stored beside the cached artifacts. A cache that is trusted without the key check would
//      reuse a calibration taken against a different base/instrument — silent metric-relaxation.
//
//   3. THE SCORING DENOMINATOR IS NEVER CACHED. Candidate/finalist scores are median(base/cand) from
//      the per-candidate paired A/B, which re-pins the scattered baseline fresh every turn. The cache
//      only feeds advisory context + the instrument-validity gate. This test asserts the cache never
//      touches pairedGuardReadout / BASELINE_PER_CASE's role as the paired-A/B rail.
'use strict';

const fs = require('fs');
const path = require('path');
const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_mega.js'), 'utf8');
const bench = fs.readFileSync(path.join(WF, 'roles', 'mega', 'benchmark_engineer.md'), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

console.log('\n# 1. the flag is default-off and inert without a STATE_DIR to persist across waves');
ok(/const MEGA_FAST_TEST = MODE === 'mega' && !!STATE_DIR &&\s*\n\s*String\(A\.mega_fast_test != null \? A\.mega_fast_test : 'false'\) === 'true';/.test(src),
   'MEGA_FAST_TEST defaults to false and requires mode=mega + a STATE_DIR');

console.log('\n# 2. when off, the cache helpers make no agent() call (byte-identical dispatch)');
ok(/async function fastTestCacheLoad\(kind[^)]*\)\s*\{\s*\n\s*if \(!MEGA_FAST_TEST\) return null;/.test(src),
   'fastTestCacheLoad returns null before any agent() when the flag is off');
ok(/async function fastTestCachePublish\(\)\s*\{\s*\n\s*if \(!MEGA_FAST_TEST\) return;/.test(src),
   'fastTestCachePublish returns before any agent() when the flag is off');

console.log('\n# 3. benchmark cache miss measures; Mega pre-candidate profile stays executable');
ok(/const benchCache = noGpuFrontMatter \? null : await fastTestCacheLoad\('bench'\);\s*\nconst bench = noGpuFrontMatter \? structuralBench : benchCache \? benchCache\.bench : await agentT\(/.test(src),
   'the real benchmark agentT() call is the else-branch of the bench cache (unchanged prompt/opts when off)');
ok(/const profileCache = MODE === 'mega' \? null : await fastTestCacheLoad\('profile'\);/.test(src) &&
   /if \(MODE === 'mega'\)[\s\S]{0,900}profiler_used: 'benchmark-only'/.test(src) &&
   !/roleAgent\('profile_engineer', 'mega_analysis'/.test(src),
   'Mega does not run fused-only flags against a pre-candidate frozen baseline');
ok(/if \(!noGpuFrontMatter && MEGA_FAST_TEST && !benchCache\) await fastTestCachePublish\(\);/.test(src),
   'a fresh benchmark still publishes reusable calibration artifacts');

console.log('\n# 4. reuse is gated on a validity key recomputed from disk (honesty gate)');
ok(/key_valid: \{ type: 'boolean' \}/.test(src) &&
   /res\.key_valid !== true/.test(src),
   'the load result carries key_valid and the script treats anything but a true key as a MISS');
ok(/frozen_rev/.test(bench) && /bench_sha/.test(bench) &&
   /control_sha/.test(bench) && /guards_sha/.test(bench),
   'the key hashes the frozen base rev + bench harness + control spec + guards');
ok(/python3 "\$FAST_TEST_KEY_TOOL"/.test(bench) &&
   /geak-fast-test-key-v1/.test(bench) &&
   /do not reimplement or infer its\s+serialization/i.test(bench),
   'both phases delegate key serialization to one checked-in deterministic tool');
ok(/Recompute `current_key` from disk NOW/.test(bench) &&
   /never fabricate a number/.test(bench),
   'the role recomputes the key at load time and refuses to fabricate a missing cached number');
ok(/NO GPU, NO lease/.test(bench),
   'both cache phases forbid a GPU lease — they copy/hash files, they do not measure');

console.log('\n# 5. the scoring denominator is never served from cache');
ok(/pairedGuardReadout/.test(src) && !/fastTestCache\w*\([^)]*\)[\s\S]{0,200}pairedGuardReadout/.test(src),
   'scores stay median(base/cand) from the per-candidate paired A/B, independent of the cache');
ok(/const bench = noGpuFrontMatter \? structuralBench : benchCache \? benchCache\.bench/.test(src) &&
   /const BASELINE_PER_CASE = benchR\.baseline_per_case;/.test(src),
   'a cached BASELINE_PER_CASE only re-enters as advisory context, not as a paired-A/B reading');
ok(/commandment_materialized/.test(src) &&
   /commandment_current_identity/.test(src) &&
   /String\(res\.bench && res\.bench\.commandment_path \|\| ''\) === COMMANDMENT/.test(src),
  'a cache hit is rejected unless it materializes the current EVAL_DIR COMMANDMENT');
ok(/tools\/rebase_commandment\.py/.test(bench) &&
   /normalized[\s\S]*bytes are identical/.test(bench) &&
   /rejects[\s\S]*fixed candidate paths/.test(bench) &&
   /current path, never the cache or an older run/.test(bench),
  'the cache loader permits only a normalized EVAL_DIR rebase');
ok(/setup_ab_control_evidence_manifest\.json/.test(bench),
  'cached calibration evidence is self-contained instead of pointing at an old run');

console.log('\n# 6. every fast-test input the role declares is actually threaded (contract mirror)');
for (const name of [
  'CACHE_DIR', 'CACHE_KIND', 'FROZEN_KERNEL_PATH', 'BENCH_HARNESS',
  'MEGA_ANALYSIS_DIR', 'FAST_TEST_KEY_TOOL', 'CONTROL_JSON', 'GUARDS_JSON',
]) {
  const threaded = src.includes(name + ':') ||
    new RegExp(`\\b${name}\\s*,`).test(src);
  ok(bench.includes('`' + name + '`') && threaded,
     `${name} is declared in benchmark_engineer.md Inputs and passed by a helper`);
}
ok(/roleAgent\('benchmark_engineer', 'fast_test_load'[\s\S]{0,600}SKILL_DIR: WORKFLOW_DIR/.test(src) &&
   /roleAgent\('benchmark_engineer', 'fast_test_publish'[\s\S]{0,600}SKILL_DIR: WORKFLOW_DIR/.test(src),
   'both new call sites pass SKILL_DIR (the input_contract soundness invariant)');

console.log(failures
  ? `\nFAILED: ${failures} assertion(s).`
  : '\nPASS: mega_fast_test is default-off byte-identical, key-gated, and never caches the denominator.');
process.exit(failures ? 1 : 0);
