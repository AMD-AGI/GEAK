#!/usr/bin/env node
// A launch-structure change needs attribution, but the task decides which metric creates credit.
// changed_kernel rejects a win outside the changed body; operator_e2e retains the operator result
// and withholds only the mechanism claim. Independently rank-reduced stage timers must not silently
// replace the operator span.
//
// What went wrong, concretely: a fused candidate ran 4878us against the 4774us of the two kernels
// it replaced -- 2.18% SLOWER -- and was promoted to current best on a +4.24% end-to-end reading.
// The whole 223us was in the gaps between kernels (residual moved +0.0799 -> -0.2178 ms), almost
// certainly because the fused kernel's grid-wide join incidentally aligned every rank's consumer
// start and tightened the next kernel's arrival window. That is a real effect and a barrier would
// buy it far more cheaply, but it is not a fact about the kernel under test, and two subsequent
// rounds were spent looking for a mechanism inside a kernel that had got slower.
//
// So this gate, unlike the overlap gate next to it, REJECTS. Each assertion below is a way that
// decision could silently go wrong:
//
//   - the field is absent and the candidate sails through as if attributed (defect class B: the
//     schema is additionalProperties:true, so an omitted `attribution` validates)
//   - the sign convention is inverted and a regression reads as a win
//   - a flat kernel with an end-to-end win is waved through as "well, it didn't get worse"
//   - the rejection happens silently, so the report shows no candidate rather than a refused one
//   - a run whose engineers never report the field changes behaviour (it must not)
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_mega.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const m = src.match(/\/\/ <<REPLAY:attribution_gate>>([\s\S]*?)\/\/ <<\/REPLAY:attribution_gate>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:attribution_gate>> region — nothing to test'); process.exit(1); }
const { attributionVerdict } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { attributionVerdict };`)();

const WON = { geomean: 1.0424 };   // the real reading from the wave this gate was written for
const LOST = { geomean: 0.99 };

console.log('\n# the wave-15 round-3 case: an end-to-end win on a kernel that got slower');
{
  const v = attributionVerdict({ attribution: {
    changed_us: 4878.0, replaced_sum_us: 4774.0, guard: '8192_rank-mixed-skew',
    residual_ms_base: 0.0799, residual_ms_cand: -0.2178,
  } }, WON);
  ok(v.state === 'gap_win', 'a +4.24% end-to-end on a 2.18%-slower kernel is a GAP_WIN');
  ok(v.reject === true, 'and it is rejected, not merely caveated');
  ok(/4878\.0us against 4774\.0us/.test(v.caveat), 'the caveat quotes both kernel times so the reader can check it');
  ok(/-2\.18%/.test(v.caveat), 'and states the kernel-time delta with the correct sign');
  ok(/298us of the claim is in the gaps/.test(v.caveat),
     'and converts the residual swing into the microseconds it accounts for');
  ok(/changed.kernel|outside the declared score/.test(v.caveat),
     'and says why it is outside a changed-kernel score rather than guessing a cheaper implementation');
}

console.log('\n# operator-e2e keeps the real operator result while withholding the mechanism claim');
{
  const v = attributionVerdict({ attribution: {
    changed_us: 4878.0, replaced_sum_us: 4774.0,
    residual_ms_base: 0.0799, residual_ms_cand: -0.2178,
  } }, { ...WON, metric: 'operator_e2e' });
  ok(v.state === 'mechanism_mismatch' && v.reject === false,
     'the same reading is not rejected when the declared score owns the complete operator span');
  ok(/mechanism claim withheld/.test(v.caveat),
     'but it cannot be reported as overlap or as a faster fused body');
}

console.log('\n# the sign convention is not inverted');
{
  const v = attributionVerdict({ attribution: { changed_us: 4600, replaced_sum_us: 4774 } }, WON);
  ok(v.state === 'attributed' && v.reject === false,
     'a kernel FASTER than what it replaced, with an end-to-end win, is attributed and kept');
}
{
  const v = attributionVerdict({ attribution: { changed_us: 4774, replaced_sum_us: 4600 } }, WON);
  ok(v.state === 'gap_win' && v.reject === true,
     'and swapping the two fields flips it to a rejection — the gate is reading them, not guessing');
}

console.log('\n# a flat kernel cannot produce an end-to-end win');
{
  const v = attributionVerdict({ attribution: { changed_us: 4774.2, replaced_sum_us: 4774.0 } }, WON);
  ok(v.state === 'gap_win' && v.reject === true,
     'kernel time inside the noise band with a +4.24% e2e claim is rejected, not waved through');
  ok(/flat/.test(v.caveat), 'and the caveat says flat rather than implying a regression it did not measure');
}

console.log('\n# a missing field is a hole, never a pass — and never a rejection either');
{
  const v = attributionVerdict({}, WON);
  ok(v.state === 'unattributed', 'no attribution block at all reads as UNATTRIBUTED');
  ok(v.reject === false,
     'but it does NOT reject: a run whose engineers do not report the field must behave exactly as before');
  ok(/Report attribution\.changed_us/.test(v.caveat),
     'and the caveat names the exact fields to report, so the next round can close the hole');
}
{
  ok(attributionVerdict({ attribution: { changed_us: 4878 } }, WON).state === 'unattributed',
     'half the pair is not a measurement');
  ok(attributionVerdict({ attribution: { changed_us: 4878, replaced_sum_us: 0 } }, WON).state === 'unattributed',
     'a zero denominator is refused rather than turned into Infinity%');
  ok(attributionVerdict({ attribution: { changed_us: 'fast', replaced_sum_us: 4774 } }, WON).state === 'unattributed',
     'a non-numeric value falls to unattributed rather than coercing to NaN and comparing false');
}
{
  const v = attributionVerdict({}, LOST);
  ok(v.state === 'not_applicable' && !v.caveat,
     'a candidate that did not win needs no attribution and is not nagged for one');
}

console.log('\n# the opposite failure: the kernel got faster and the operator did not');
{
  const v = attributionVerdict({ attribution: {
    changed_us: 4500, replaced_sum_us: 4774, residual_ms_base: 0.08, residual_ms_cand: 0.36,
  } }, LOST);
  ok(v.state === 'kernel_win_e2e_loss', 'a real kernel win with no end-to-end win is its own state');
  ok(v.reject === false,
     'and is NOT rejected — the kernel result stands, the regression is a separate fact about where the cost went');
  ok(/moved rather than disappeared/.test(v.caveat), 'and the caveat says the cost moved, not that the win was fake');
}

console.log('\n# the rejection is wired into the run, and it is loud');
{
  ok(/!r\.attribution_rejected/.test(src),
     'the verified filter excludes attribution-rejected candidates, so they cannot reach the candidate list');
  ok(/ATTRIBUTION \$\{r\.attribution_rejected\.toUpperCase\(\)\}/.test(src),
     'and every rejection is logged with its id — exclusion alone reads as "it did not work"');
  ok(/const ATTRIBUTION_CAVEATS = \[\]/.test(src) && /ATTRIBUTION_CAVEATS \} : \{\}\)/.test(src),
     'and travels to the report, so a refused candidate is distinguishable from no candidate');
  const gateIdx = src.indexOf('const at = attributionVerdict');
  const filtIdx = src.indexOf('const verified = clean.filter');
  ok(gateIdx > 0 && filtIdx > gateIdx,
     'attribution is decided BEFORE the verified filter — a gate that can reject must run before the selection it feeds');
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
