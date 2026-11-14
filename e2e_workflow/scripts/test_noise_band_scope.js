// Tests how the acceptance noise band is resolved on each workload.
//
// Run: node e2e_workflow/scripts/test_noise_band_scope.js
//
// NOISE_BAND is the band an A/B's delta has to clear to be called a win. For a fixed ISL/OSL sweep a
// replica is a fixed amount of identical work and 0.5% is defensible. The agentx trace replay moves
// more: the 20260912 run measured two surviving replicas of the same baseline config on one box at a
// 3.43% spread on total tok/s (24054.0 / 23255.9) and 3.86% on InferenceX P90 interactivity
// (29.52 / 28.40). That spread is not its default band all the same: one kernel's A/B rarely moves
// e2e that far, so a band that wide keeps most single-kernel wins from reaching Validate, and
// run_e2e.py holds the final pair to the KEEP rule on the canonical window.
//
// An explicit band has to SURVIVE Setup. The Director's role file returns noise_band_pct=0.5 on every
// run, and `setup.noise_band_pct || default` kept that 0.5 over a band the caller passed.
//
// The code is extracted from the real workflow source and evaluated, rather than re-implemented
// here, so the test fails if the source stops matching the shape it asserts.

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

// ── Extract the default + the post-Setup resolver and evaluate them with IS_AGENTX / args ──────
const start = src.indexOf('const NOISE_BAND_DEFAULT');
const fnStart = src.indexOf('function gateNoiseBand(', start);
const end = src.indexOf('\n}\n', fnStart);
ok(start !== -1 && fnStart !== -1 && end !== -1,
  'the noise band default and its post-Setup resolver are located in e2e_workflow.js');
if (start === -1 || fnStart === -1 || end === -1) { process.exit(1); }

const expr = src.slice(start, end + 2);
const band = (isAgentx, args) =>
  new Function('IS_AGENTX', 'A',
    `${expr}\nreturn { NOISE_BAND_DEFAULT, gateNoiseBand };`)(isAgentx, args || {});

// ── 1. the fixed-shape default is untouched ────────────────────────────────────────────────────
console.log('\n# a fixed ISL/OSL workload keeps the band it always had');
const syn = band(false);
ok(syn.NOISE_BAND_DEFAULT === 0.5, 'a non-agentx run still defaults to 0.5%');
ok(syn.gateNoiseBand(0.5) === 0.5 && syn.gateNoiseBand(1.5) === 1.5 && syn.gateNoiseBand(0.3) === 0.3,
  "a non-agentx run gates on the Director's band verbatim, as before");
ok(syn.gateNoiseBand(undefined) === 0.5 && syn.gateNoiseBand(0) === 0.5,
  'and falls back to the default when the Director returned none');
ok(band(false, { noise_band_pct: 2 }).gateNoiseBand(0.5) === 0.5,
  "and the Director still outranks an explicit band there (the pre-existing precedence)");

// ── 2. the trace replay has no wider default ───────────────────────────────────────────────────
console.log("\n# a trace replay gates each A/B on the Director's band too");
const agentx = band(true);
ok(agentx.NOISE_BAND_DEFAULT === 0.5, 'an agentx run defaults to 0.5%, as a fixed-shape one does');
ok(agentx.gateNoiseBand(0.5) === 0.5, "the Director's default 0.5 is the band");
ok(agentx.gateNoiseBand(undefined) === 0.5 && agentx.gateNoiseBand(NaN) === 0.5,
  'no Director band => the default');
ok(agentx.gateNoiseBand(6) === 6, 'a Director that measured a wider spread still widens it');

// ── 3. an explicit band survives Setup ─────────────────────────────────────────────────────────
console.log("\n# on a trace replay the Director's 0.5 can widen an explicit band, never replace it");
ok(band(true, { noise_band_pct: 4 }).gateNoiseBand(0.5) === 4,
  'noise_band_pct=4 still gives a 4% band');
ok(band(true, { noise_band_pct: 2 }).gateNoiseBand(0.5) === 2,
  "an explicit band is not replaced by the Director's 0.5");
ok(band(true, { noise_band_pct: 1.25 }).gateNoiseBand(3) === 3,
  'but the Director may widen an explicit band');
ok(band(true, { noise_band_pct: '0.75' }).gateNoiseBand(undefined) === 0.75,
  'a string-valued band is parsed, as it arrives from the command line');

// ── 4. both places the band is (re)loaded go through the resolver ──────────────────────────────
console.log('\n# Setup and a resumed phase resolve the band the same way');
ok(/NOISE_BAND = gateNoiseBand\(setup\.noise_band_pct\);/.test(src), 'Setup resolves the band');
ok(/NOISE_BAND = gateNoiseBand\(ST\.noise_band_pct\);/.test(src),
  'a resumed phase re-resolves the band it carried');
ok(!/NOISE_BAND = (setup|ST)\.noise_band_pct \|\|/.test(src),
  'no assignment bypasses the resolver');

console.log(failures === 0
  ? '\nPASS: the noise band resolves the same on every workload, and an explicit band is kept.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
