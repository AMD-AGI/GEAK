#!/usr/bin/env node
// Regression guard for the head author plan on a source-delivered prepared UT (no GPU, no model).
//
// Such a UT fails a candidate whose kernels it cannot attribute to the task's own source. Without
// candidate gates only a rewrite of the extracted leaves can pass, so author entries are dropped while a
// rewrite is planned; with them, author entries in its `launchers` languages are kept.
//
// Run:  node e2e_workflow/scripts/test_head_plan_leaf_gates.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const startMarker = 'function headAuthorPlan(h, authorPlan) {';
const start = src.indexOf(startMarker);
const end = src.indexOf('\n}\n', start);
ok(start !== -1 && end > start, 'headAuthorPlan located in e2e_workflow.js');
const block = src.slice(start, end + 2);

const callSites = src.split('headAuthorPlan(h, bake.author_plan)').length - 1;
ok(src.includes('let plan = headAuthorPlan(h, bake.author_plan);') && callSites === 2,
  'the head track and the fast-mode author fan-out both build their lanes through headAuthorPlan');

const planFor = (h, authorPlan, max = 3) => {
  const logs = [];
  const plan = new Function('log', 'HEAD_AUTHOR_MAX', `${block}\nreturn headAuthorPlan;`)(
    (m) => logs.push(m), max)(h, authorPlan);
  return { plan, logs };
};
const author = (language) => ({ language, route: 'author', rationale: '' });
const rewrite = (language) => ({ language, route: 'rewrite', rationale: '' });
const delivered = { short_name: 'fused_op', source_delivery_required: true };

let r = planFor(delivered, [author('triton'), rewrite('hip')]);
ok(r.plan.length === 1 && r.plan[0].route === 'rewrite' && r.plan[0].language === 'hip',
  'a source-delivered head keeps only the rewrite of its leaves');
ok(r.logs.some((l) => /dropped author lane\(s\) triton/.test(l)), 'the dropped author lane is logged');

r = planFor(delivered, [author('triton'), author('hip')]);
ok(r.plan.length === 2 && r.logs.length === 0, 'an all-author plan is kept (the last lanes are never dropped)');

r = planFor({ short_name: 'extracted_op' }, [author('triton'), rewrite('hip')]);
ok(r.plan.length === 2 && r.logs.length === 0, 'a head without delivered leaves keeps its author lanes');

r = planFor(delivered, [author('triton'), rewrite('hip'), rewrite('ck')], 2);
ok(r.plan.length === 2 && r.plan.every((ap) => ap.route === 'rewrite'),
  'HEAD_AUTHOR_MAX applies after the drop, so a second rewrite takes the freed slot');

r = planFor(delivered, []);
ok(Array.isArray(r.plan) && r.plan.length === 0 && r.logs.length === 0, 'an empty plan stays empty');

r = planFor(delivered, undefined);
ok(Array.isArray(r.plan) && r.plan.length === 0, 'a missing plan is an empty one');

const gated = (launchers) => ({ ...delivered, prepared_ut: { result: { candidate_gates: { version: 1, launchers } } } });
r = planFor(gated(['triton']), [author('triton'), author('hip'), rewrite('flydsl')]);
ok(r.plan.map((ap) => `${ap.route}:${ap.language}`).join(',') === 'author:triton,rewrite:flydsl',
  'a UT with candidate gates keeps the author lanes in its launcher languages and the rewrite');
ok(r.logs.some((l) => /dropped author lane\(s\) hip: the prepared UT attributes only triton kernels/.test(l)),
  'the author lane outside its launchers is dropped and logged');

r = planFor(gated(['Triton', 'HIP']), [author('triton'), author('hip'), rewrite('flydsl')]);
ok(r.plan.length === 3 && r.logs.length === 0, 'launcher names match the plan languages case-insensitively');

for (const gates of [gated([]), gated('triton'), { ...delivered, prepared_ut: { result: {} } }, { ...delivered, prepared_ut: {} }]) {
  r = planFor(gates, [author('triton'), rewrite('hip')]);
  ok(r.plan.length === 1 && r.plan[0].route === 'rewrite' && r.logs.some((l) => /gates on its own editable leaves/.test(l)),
    `no usable launchers keep the leaf-gate rule (${JSON.stringify(gates.prepared_ut)})`);
}

console.log(failures === 0 ? '\nPASS: head plans stay within what the prepared UT can accept.' : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
