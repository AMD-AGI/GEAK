#!/usr/bin/env node
// Regression guard for a caller-pinned eval_dir (no GPU, no model needed).
//
// run_e2e pins EVAL_DIR_OVERRIDE and writes into it before the workflow starts, so the directory always
// exists and is non-empty. The Director read the "append _${RANDOM} until fresh" rule as applying to the
// override too, built a sibling directory, and the run's work landed where run_e2e never reads.
//
// Run:  node e2e_workflow/scripts/test_pinned_eval_dir.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow.js'), 'utf8');
const start = src.indexOf('function evalDirDivergence');
const end = src.indexOf('\n}\n', start);
ok(start !== -1 && end > start, 'evalDirDivergence located in e2e_workflow.js');
if (failures) process.exit(1);
const evalDirDivergence = new Function(`${src.slice(start, end + 2)}\nreturn evalDirDivergence;`)();

console.log('\n# a pinned eval_dir must be used as is');
ok(evalDirDivergence('/x/geak/e2e_cycle0', '/x/geak/e2e_cycle0') === '', 'the pinned path is accepted');
ok(evalDirDivergence('/x/geak/e2e_cycle0', '/x/geak/e2e_cycle0/') === '', 'a trailing slash is ignored');
const sibling = evalDirDivergence('/x/geak/e2e_cycle0', '/x/geak/e2e_cycle0_14495');
ok(sibling.includes('e2e_cycle0_14495') && sibling.includes('pinned /x/geak/e2e_cycle0'),
  'a suffixed sibling is refused and both paths are named');
ok(evalDirDivergence('/x/geak/e2e_cycle0', '') !== '', 'a missing eval_dir is refused when one was pinned');
ok(evalDirDivergence('/', '/tmp/different') !== '', 'a pinned root directory still counts as a pin');
ok(evalDirDivergence('/', '/') === '', 'the pinned root directory is accepted');

console.log('\n# without a pin the Director chooses');
ok(evalDirDivergence('', '/x/exp/e2e_model_20261005_1_2') === '', 'no override accepts the generated path');

console.log('\n# the setup gate uses it before adopting the Director answer');
const gate = src.indexOf('evalDirDivergence(EVAL_DIR_OVERRIDE, setup.eval_dir)');
ok(gate !== -1 && gate < src.indexOf('EVAL_DIR = setup.eval_dir;'), 'setup refuses a divergent eval_dir first');

console.log('\n# the Director prompt keeps the suffix rule to generated paths');
const director = fs.readFileSync(path.join(ROOT, 'roles', 'director.md'), 'utf8');
ok(/EVAL_DIR_OVERRIDE` is set, `EVAL_DIR` is exactly that path/.test(director),
  'step 2 says the override is used exactly');
ok(/if that generated path exists/.test(director), 'the _${RANDOM} rule applies only to the generated path');

console.log(failures ? `\n${failures} failure(s)` : '\nall passed');
process.exit(failures ? 1 : 0);
