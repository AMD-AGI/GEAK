// A seam that launched GPU work but whose DECLARED kernel name matches nothing it launched is a
// NAME defect, not a seam defect. The extract retry loop used to treat every selection failure the
// same way: tell the agent the seam is wrong, forbid reusing that target (ATTEMPTED_TARGET_CALLABLES),
// and re-capture. Against a name typo that is unfixable by construction -- the one correct seam is
// banned after attempt 1 and the search has no terminating condition. One gemm head spent 13.6h and
// seven captures there; the seam had been right since the first attempt.
//
// e2e_workflow.js is a workflow script (top-level `export const meta`), so it cannot be require()d.
// These tests read the source and EXECUTE the extracted predicate, so the assertion tracks the real
// definition instead of a copy that can drift away from it.
const fs = require('fs');
const path = require('path');
const assert = require('assert');

const SRC = fs.readFileSync(path.join(__dirname, '..', 'e2e_workflow.js'), 'utf8');
let failures = 0;
function test(name, fn) {
  try { fn(); console.log(`  ok: ${name}`); }
  catch (err) { failures++; console.log(`  FAIL: ${name}\n    ${err && err.message}`); }
}

// Pull the predicate out of the source and make it callable, so a behavioural change to the guard
// has to be made in e2e_workflow.js to pass -- editing the test alone cannot satisfy it.
function extract(startsWith, endsWith) {
  const start = SRC.indexOf(startsWith);
  assert.ok(start >= 0, `source no longer contains \`${startsWith}\``);
  const end = SRC.indexOf(endsWith, start);
  assert.ok(end >= 0, `no \`${endsWith}\` after \`${startsWith}\``);
  return SRC.slice(start, end + endsWith.length);
}
const NAME_ONLY_SRC = extract('const nameCodes =', "'device_kernel_not_under_target');");
// eslint-disable-next-line no-new-func
const nameOnly = new Function('selection', `${NAME_ONLY_SRC}\nreturn nameOnly;`);

console.log('selection failures distinguish a wrong NAME from a wrong SEAM');

test('a live seam with an unmatched declared name is a name defect', () => {
  assert.strictEqual(nameOnly({ ok: false, codes: ['device_kernel_name_mismatch'] }), true);
});

test('a name rejected against the profile before any capture is a name defect', () => {
  assert.strictEqual(nameOnly({ ok: false, codes: ['device_kernel_not_in_profile'] }), true);
});

test('a marker that launched nothing is a SEAM defect, and must still descend', () => {
  assert.strictEqual(nameOnly({ ok: false, codes: ['device_kernel_not_under_target'] }), false);
});

test('a seam defect stays a seam defect even when a name code rides along', () => {
  // Both present: the marker launched nothing under at least one rank, so descending is still the
  // repair. Fail towards the old behaviour, which is merely slow, not towards skipping a real fix.
  assert.strictEqual(nameOnly({
    ok: false,
    codes: ['device_kernel_name_mismatch', 'device_kernel_not_under_target'],
  }), false);
});

test('unrelated selection failures are unaffected', () => {
  assert.strictEqual(nameOnly({ ok: false, codes: ['deepest_not_verified'] }), false);
  assert.strictEqual(nameOnly({ ok: false, codes: [] }), false);
});

test('a passing selection is never a name defect', () => {
  assert.strictEqual(nameOnly({ ok: true }), false);
  assert.strictEqual(nameOnly({ ok: true, codes: ['device_kernel_name_mismatch'] }), false);
});

test('a verdict with no codes at all does not throw', () => {
  assert.strictEqual(nameOnly({ ok: false }), false);
});

console.log('the retry loop acts on the distinction');

test('a name defect does not ban the seam it was reported against', () => {
  // ATTEMPTED_TARGET_CALLABLES is handed to the agent with "Do not return any value again". Pushing
  // a correct seam into it is what made the 13.6h hunt unrecoverable.
  const guard = extract('if (priorTarget && !nameOnly', 'attemptedTargets.push(priorTarget);');
  assert.ok(/!nameOnly/.test(guard), 'attemptedTargets is still populated on a name-only failure');
});

test('the name corrective tells the agent to keep the target, not to descend', () => {
  const corrective = extract('SELECTED A LIVE SEAM BUT DECLARED THE WRONG GPU KERNEL NAME',
    'as selection_validation.');
  assert.ok(/DO NOT descend to another callable/.test(corrective), 'does not forbid descending');
  assert.ok(/keep this target_callable/.test(corrective), 'does not tell the agent to keep the seam');
  assert.ok(!/Do not return any ATTEMPTED_TARGET_CALLABLES/.test(corrective),
    'still bans previously attempted targets');
  // The names are the whole point: the agent cannot copy a spelling it was not shown.
  assert.ok(/kernels_under_target/.test(SRC), 'verdict launch list is never surfaced');
  assert.ok(/profile_kernel_candidates/.test(SRC), 'profile candidate names are never surfaced');
});

test('the seam corrective still descends', () => {
  const corrective = extract('PRIOR ATTEMPT DID NOT SELECT THE PROFILED GPU KERNEL',
    'Do not return any ATTEMPTED_TARGET_CALLABLES value again.');
  assert.ok(/Select the deepest live inner_launcher\/op_seam/.test(corrective),
    'the genuine wrong-seam path lost its descend instruction');
});

test('every extract site is given the profile to check the declared name against', () => {
  // Counting two independent literals proves nothing (a site can omit PROFILE_TOPN while some other
  // role's Inputs keeps the global count up). Every extractor site must build its Inputs through the
  // ONE helper that carries it, so the check cannot be lost by retyping an object literal.
  const sites = SRC.match(/'kernel_extractor', 'extract(_op)?'[^]{0,200}/g) || [];
  assert.ok(sites.length > 0, 'no kernel_extractor call sites found');
  for (const site of sites) {
    assert.ok(/extract(or|Op)Inputs\(/.test(site),
      `an extractor call site builds its Inputs inline instead of via extractorInputs(); without ` +
      `PROFILE_TOPN it cannot run the pre-capture name check and pays for a typo with a full capture:\n` +
      site.split('\n').slice(0, 3).join('\n'));
  }
  const helper = extract('function extractorInputs(', '\n}');
  assert.ok(/PROFILE_TOPN: profile \? profile\.profile_topN_json : ''/.test(helper),
    'extractorInputs no longer passes PROFILE_TOPN, so no extract site can run the name check');
  // The op track must still reach the same helper rather than forking its own literal.
  assert.ok(/const extractOpInputs = \([^)]*\) => extractorInputs\(/.test(SRC),
    'extractOpInputs no longer delegates to extractorInputs');
});

console.log(failures ? `\nFAIL (${failures})` : '\nPASS');
process.exit(failures ? 1 : 0);
