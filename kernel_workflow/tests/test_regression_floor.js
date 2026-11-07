#!/usr/bin/env node
// args.regression_floor_by_guard lowers the no-regression floor of a named guard only.
'use strict';

const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'kernel_mega.js'), 'utf8');
let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};
const region = (name) => {
  const m = src.match(new RegExp(`// <<REPLAY:${name}>>([\\s\\S]*?)// <</REPLAY:${name}>>`));
  if (!m) throw new Error(`missing REPLAY region ${name}`);
  return m[1];
};
const load = (prelude) => new Function(`${prelude}\n${region('bimodal_split')}\nreturn guardContract;`)();

const result = { per_case: [
  { name: 't', speedup: 1.15 },
  { name: 'fb', speedup: 0.99 },
  { name: 'r', speedup: 0.99 },
] };

console.log('\n# default floor stays 1.0');
const plain = load('')(result, ['t'], ['fb'], 0);
ok(plain.regression_pass === false && plain.regressed.includes('fb'),
  'without RF a 0.99x regression guard regresses');
const empty = load('const RF = Object(undefined);')(result, ['t'], ['fb'], 0);
ok(empty.regression_pass === false, 'an absent arg keeps the strict 1.0 floor');

console.log('\n# a declared floor applies to its guard only');
const gc = load("const RF = { fb: 0.98 };");
ok(gc(result, ['t'], ['fb'], 0).regression_pass === true, 'fb at 0.99x clears its 0.98 floor');
ok(gc(result, ['t'], ['fb', 'r'], 0).regressed.join() === 'r', 'an undeclared guard keeps 1.0');
ok(gc({ per_case: [{ name: 't', speedup: 1.1 }, { name: 'fb', speedup: 0.97 }] },
  ['t'], ['fb'], 0).regression_pass === false, 'below the declared floor still regresses');

console.log('\n# final selection uses the same per-guard floor');
const sel = src.slice(src.indexOf('function megaFinalSelectionVerdict('));
const loop = sel.slice(sel.indexOf('for (const guard of o.regressionGuards || [])'));
ok(/const fl = typeof RF > 'u' \? 1 : RF\[guard\] \|\| 1;/.test(loop) &&
   /Number\(readout\.score\) >= fl/.test(loop) && /perCase\.get\(String\(guard\)\) >= fl/.test(loop),
  'paired median and per-case speedup are both compared with the guard floor');

if (failures) { console.error(`FAILED: ${failures} assertion(s).`); process.exit(1); }
console.log('\nall regression-floor checks passed');
