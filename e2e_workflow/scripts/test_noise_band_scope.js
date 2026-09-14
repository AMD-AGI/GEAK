// Tests that the acceptance noise band is scoped to the workload it was measured on.
//
// Run: node e2e_workflow/scripts/test_noise_band_scope.js
//
// NOISE_BAND is the band a candidate has to clear to be called a win, so it is only meaningful if
// it reflects how much the measurement moves when NOTHING changes. For a fixed ISL/OSL sweep a
// replica is a fixed amount of identical work and 0.5% is defensible. For the agentx trace replay
// it is not: the 20260912 run measured three replicas of the same baseline config on one box at a
// 3.43% spread (24054.0 high, 23255.9 low), because the client paces whole trajectories of very
// unequal size against a prefix cache at 85-96% hits. A 0.5% band there certifies noise as a win.
//
// The expression is extracted from the real workflow source and evaluated, rather than
// re-implemented here, so the test fails if the source stops matching the shape it asserts.

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

// ── Extract the two constants and evaluate them with IS_AGENTX / args injected ─────────────────
const start = src.indexOf('const NOISE_BAND_AGENTX');
const marker = 'A.noise_band_pct != null ? A.noise_band_pct : (IS_AGENTX ? NOISE_BAND_AGENTX : 0.5));';
const end = src.indexOf(marker, start);
ok(start !== -1 && end !== -1,
  'the noise band default is still a single workload-aware expression in e2e_workflow.js');
if (start === -1 || end === -1) { process.exit(1); }

const expr = src.slice(start, end + marker.length);
const band = (isAgentx, args) =>
  new Function('IS_AGENTX', 'A', `${expr}\nreturn { NOISE_BAND_DEFAULT, NOISE_BAND_AGENTX };`)(isAgentx, args || {});

// ── 1. the fixed-shape default is untouched ────────────────────────────────────────────────────
console.log('\n# a fixed ISL/OSL workload keeps the band it always had');
ok(band(false).NOISE_BAND_DEFAULT === 0.5,
  'a non-agentx run still defaults to 0.5%');

// ── 2. the trace replay carries its own floor ──────────────────────────────────────────────────
console.log('\n# the trace replay carries the band its own replicas measured');
const agentx = band(true);
ok(agentx.NOISE_BAND_DEFAULT === agentx.NOISE_BAND_AGENTX,
  'an agentx run defaults to the agentx band, not the synthetic one');
ok(agentx.NOISE_BAND_DEFAULT > 0.5,
  'the agentx band is wider than the synthetic default');
ok(agentx.NOISE_BAND_DEFAULT >= 3.43,
  'the agentx band is at least the 3.43% replica spread measured on 20260912; a band under the ' +
  'observed same-config spread would accept noise as a win');

// ── 3. an explicit band still wins, for both ───────────────────────────────────────────────────
// Setup is expected to supply a measured band eventually (e2e_workflow.js reads
// setup.noise_band_pct ahead of this default); the constant must not get in its way.
console.log('\n# an explicitly supplied band overrides the default either way');
ok(band(true, { noise_band_pct: 1.25 }).NOISE_BAND_DEFAULT === 1.25,
  'an explicit band overrides the agentx default');
ok(band(false, { noise_band_pct: 2 }).NOISE_BAND_DEFAULT === 2,
  'an explicit band overrides the synthetic default');
ok(band(true, { noise_band_pct: '0.75' }).NOISE_BAND_DEFAULT === 0.75,
  'a string-valued band is parsed, as it arrives from the command line');

// ── 4. the measured band still takes precedence over any default ───────────────────────────────
console.log('\n# the hook for a measured band is still in place');
ok(/NOISE_BAND = setup\.noise_band_pct \|\| NOISE_BAND_DEFAULT;/.test(src),
  'Setup can still override the default with a band it measured');

console.log(failures === 0
  ? '\nPASS: the noise band is scoped to the workload it was measured on.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
