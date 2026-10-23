#!/usr/bin/env node
// Every input a role file says it is given must actually be passed to it.
//
// WHY THIS EXISTS — and why it is a different test from the other gates. Three separate defects,
// found three separate times by three separate waves failing, turned out to be one defect:
//
//   * Analyze wrote a four-rung roadmap ending in the fused two-launch shape the program exists to
//     reach. plan_round was handed neither roadmap.md nor candidate_directions. The top rung was
//     never proposed.
//   * Analyze produced `task_graph` and `resource_timeline`. pipeOccupancyGate rejected directions
//     whose claims exceeded their pipe's idle fraction — judging the planner against a table the
//     planner had never seen. The graph, the one artifact that says which work has NO EDGE between
//     it and other work, went only to the end-of-wave Report.
//   * Analyze produced `modifiable_files`. verify_engineer.md step 5 says to diff the patch's file
//     list against `MODIFIABLE_FILES` and reject anything outside it. Nothing ever passed it, so
//     verify compared against a list it did not have.
//
// In all three the artifact existed, was well-formed, was named in the consuming role's own
// contract, and was never threaded. Each was found by a wave burning a lease and coming back with
// nothing. Each fix was one line. That ratio is why this test exists: the class is cheap to check
// statically and expensive to find dynamically.
//
// The check is: for every `roleAgent(role, phase)` call, collect the keys actually passed; for every
// role file, collect the ALLCAPS names it declares as inputs; assert the second is a subset of the
// first. EXEMPT below is the escape hatch, and every entry carries the reason it is not a defect.
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

// --- what the harness actually passes ----------------------------------------------------------
// Balanced-paren scan, not a fixed window: the calls differ in size by 20x and a window either
// truncates the big ones or bleeds the next call into the small ones. The key regex uses a
// LOOKAHEAD for the trailing delimiter, because shorthand properties (`COMMANDMENT, PATCH,`) share
// their commas — consuming one hides every second key, which is how an earlier hand-audit of this
// same question produced two phantom findings and missed a real one.
function balanced(s, i) {
  let d = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === '(') d++;
    else if (s[j] === ')') { d--; if (d === 0) return j; }
  }
  return -1;
}
const passedByRole = new Map();   // role -> Set(all keys across its phases)
const passedByPhase = new Map();  // "role:phase" -> Set(keys)
let megaRuntimeVerifyKeys = new Set();
const KEY_RE = /(?<=[\n{,])\s*([A-Z][A-Z0-9_]{2,})\s*(?=[:,\n}])/g;
// Conditional input bundles are spread in as `...RESUME_INPUT` / `...KB_INPUTS`, so their keys are
// not textually inside the call. Resolve each spread identifier to its `const NAME = ... {...}`
// definition and fold those keys in. Without this the test reports a wired input as missing, and a
// test that cries wolf gets its findings dismissed — which is the failure mode it exists to prevent.
const spreadKeys = (name) => {
  const d = src.match(new RegExp(`const ${name}\\s*=([\\s\\S]{0,400}?);\\n`));
  return d ? [...d[1].matchAll(KEY_RE)].map((k) => k[1]) : [];
};
for (const m of src.matchAll(/roleAgent\(\s*'([a-z_]+)',\s*'([a-z_]+)'/g)) {
  const [, role, phase] = m;
  const seg = src.slice(m.index, balanced(src, src.indexOf('(', m.index)));
  const keys = new Set([...seg.matchAll(KEY_RE)].map((k) => k[1]));
  for (const s of seg.matchAll(/\.\.\.([A-Z][A-Z0-9_]{2,})\b/g)) for (const k of spreadKeys(s[1])) keys.add(k);
  passedByPhase.set(`${role}:${phase}`, keys);
  if (!passedByRole.has(role)) passedByRole.set(role, new Set());
  for (const k of keys) passedByRole.get(role).add(k);
}
// Mega runtime Verify shares one input object with its timeout-recovery retry.
{
  const start = src.indexOf('const verifyInputs = {');
  const end = src.indexOf('\n    };', start);
  const seg = src.slice(start, end);
  const keys = new Set([...seg.matchAll(KEY_RE)].map((match) => match[1]));
  const prior = passedByPhase.get('verify_engineer:verify') || new Set();
  for (const key of keys) prior.add(key);
  passedByPhase.set('verify_engineer:verify', prior);
  if (!passedByRole.has('verify_engineer')) passedByRole.set('verify_engineer', new Set());
  for (const key of keys) passedByRole.get('verify_engineer').add(key);
  megaRuntimeVerifyKeys = keys;
}
// engineer/deep_engineer's OPTIMIZE dispatch is a hand-built prompt, not a roleAgent call. Attribute
// its cfg keys to both roles, or their Inputs sections are checked against the recover phase's four
// keys and the test either cries wolf or (worse) gets an exemption written for it.
{
  const at = src.indexOf('You are Engineer ${d.id}');
  const seg = src.slice(at, balanced(src, src.lastIndexOf('agentT(', at) + 6));
  const keys = [...seg.matchAll(KEY_RE)].map((k) => k[1]);
  for (const s of seg.matchAll(/\.\.\.([A-Z][A-Z0-9_]{2,})\b/g)) keys.push(...spreadKeys(s[1]));
  for (const role of ['engineer', 'deep_engineer']) {
    if (!passedByRole.has(role)) passedByRole.set(role, new Set());
    for (const k of keys) passedByRole.get(role).add(k);
  }
  ok(keys.includes('DIRECTION') && keys.includes('KERNEL_PATH'),
     `the hand-built engineer prompt was located and parsed (${keys.length} keys)`);
}
// Mega candidate lanes dispatch through a role variable whose value is the common `engineer`, so the
// literal roleAgent scanner cannot attribute it.
{
  const at = src.indexOf('Advance candidate lane ${candidateId}');
  const start = src.lastIndexOf('eng = await agentT(', at);
  const seg = src.slice(start, at + 3200);
  const keys = [...seg.matchAll(KEY_RE)].map((k) => k[1]);
  for (const role of ['engineer']) {
    if (!passedByRole.has(role)) passedByRole.set(role, new Set());
    for (const k of keys) passedByRole.get(role).add(k);
  }
  ok(keys.includes('CANDIDATE_ID') && keys.includes('BASE_TREE') &&
     keys.includes('PRIOR_CANDIDATE'),
  `the Mega candidate prompt was located and parsed (${keys.length} keys)`);
}
// Mega Analyze selects a role with a mode-gated expression. Attribute the identical input object to
// both possible roles; the static literal scanner above intentionally cannot parse this form.
{
  const needle = "roleAgent(MODE === 'mega' ? 'mega_search_lead' : 'tech_lead'";
  let at = src.indexOf(needle);
  while (at >= 0) {
    const seg = src.slice(at, balanced(src, src.indexOf('(', at)));
    const keys = new Set([...seg.matchAll(KEY_RE)].map((k) => k[1]));
    for (const s of seg.matchAll(/\.\.\.([A-Z][A-Z0-9_]{2,})\b/g)) {
      for (const k of spreadKeys(s[1])) keys.add(k);
    }
    if (seg.includes('RESUME_INPUT')) {
      for (const k of spreadKeys('RESUME_INPUT')) keys.add(k);
    }
    for (const role of ['tech_lead', 'mega_search_lead']) {
      if (!passedByRole.has(role)) passedByRole.set(role, new Set());
      for (const k of keys) passedByRole.get(role).add(k);
      passedByPhase.set(`${role}:analyze`, keys);
    }
    at = src.indexOf(needle, at + needle.length);
  }
}
ok(passedByPhase.size >= 15, `found ${passedByPhase.size} roleAgent call sites to check`);
ok(passedByPhase.has('analysis_engineer:analyze_profile'),
   'including the multi-line call form — a regex that needs `(\'role\', \'phase\'` on one line misses it');
ok([...passedByPhase.values()].every((s) => s.has('SKILL_DIR')),
   'the parse is sound: every call site passes SKILL_DIR, so no call was mis-scanned as empty');
ok(passedByPhase.get('verify_engineer:verify').has('BASELINE_PER_CASE')
   && passedByPhase.get('verify_engineer:verify').has('PATCH'),
   'and consecutive shorthand keys are both seen (the lookahead does its job)');
// --- names a role file may declare without the harness passing them ----------------------------
const EXEMPT = new Map(Object.entries({
  HIP_VISIBLE_DEVICES: 'environment variable the agent sets itself, not an input',
  GEAK_GPU_RUN_TIMEOUT: 'environment variable read by the lease wrapper',
  GEAK_WEIGHTED_SPEEDUP: 'environment variable read by the benchmark harness',
  PYBIND11_MODULE: 'a C macro named in a code example',
  // COMMANDMENT.md entry names. benchmark_engineer WRITES these; they are sections of a file it
  // authors, not values handed to it.
  SETUP: 'COMMANDMENT.md entry name authored by benchmark_engineer',
  BENCHMARK: 'COMMANDMENT.md entry name authored by benchmark_engineer',
  FULL_BENCHMARK: 'COMMANDMENT.md entry name authored by benchmark_engineer',
  CORRECTNESS: 'COMMANDMENT.md entry name authored by benchmark_engineer',
  PROFILE: 'COMMANDMENT.md entry name authored by benchmark_engineer',
  METRIC: 'COMMANDMENT.md entry name authored by benchmark_engineer',
  PARSE: 'COMMANDMENT.md entry name authored by benchmark_engineer',
}));

// Prose words that happen to be capitalised, and JSON field names quoted in examples.
const NOT_AN_INPUT = new Set(['JSON', 'TODO', 'NOTE', 'GPU', 'CPU', 'LDS', 'VOID', 'ISA', 'DAG',
  'HEAD', 'MANDATORY', 'UNDECLARED', 'UNRESOLVED', 'UNRECORDED', 'NOT_YET_ACTUALLY_TESTED',
  'MI300X', 'MI355X', 'ALL', 'ANY', 'AND', 'NOT', 'OFF']);

console.log('\n# 1. every input a role file declares is passed to that role');
for (const f of fs.readdirSync(path.join(WF, 'roles', 'mega')).filter((x) => x.endsWith('.md'))) {
  const role = f.replace(/\.md$/, '');
  const body = fs.readFileSync(path.join(WF, 'roles', 'mega', f), 'utf8');
  // Declared inputs = backticked ALLCAPS inside an `## Inputs` block or an `Inputs:` prose line.
  const blocks = [];
  // The heading is `## Inputs` in some files and `## Inputs (in your prompt)` in others. Matching
  // only the bare form silently parsed two role files as declaring nothing, which reported as a
  // clean pass. A check that finds nothing must not look like a check that found nothing wrong.
  for (const m of body.matchAll(/^## Inputs.*\n([\s\S]*?)(?=\n## |\n$)/gm)) blocks.push(m[1]);
  for (const m of body.matchAll(/^Inputs: ([\s\S]*?)\n\n/gm)) blocks.push(m[1]);
  const declared = new Set();
  for (const b of blocks) for (const m of b.matchAll(/`([A-Z][A-Z0-9_]{2,})`/g)) declared.add(m[1]);
  const passed = passedByRole.get(role);
  if (!passed) { console.log(`  -- ${role}: not dispatched by this script; nothing to check`); continue; }
  if (declared.size === 0) { ok(false, `${role}: NO parseable Inputs section — the role is unchecked`); continue; }
  const missing = [...declared].filter((n) => !passed.has(n) && !EXEMPT.has(n) && !NOT_AN_INPUT.has(n));
  ok(missing.length === 0,
     `${role}: declares ${declared.size} inputs, all threaded${missing.length ? ` — MISSING ${missing.join(', ')}` : ''}`);
}

console.log('\n# 2. the three artifacts that were produced and never consumed');
{
  // Pinned by name. A generic subset check goes green the moment somebody deletes the sentence from
  // the role file, which is the wrong way to make it pass.
  const plan = passedByPhase.get('tech_lead:plan_round');
  ok(plan.has('ROADMAP_LADDER') && plan.has('LADDER_DISPATCHED'),
     'plan_round gets the ladder and what has been taken off it');
  ok(plan.has('TASK_GRAPH'),
     'plan_round gets the dependency graph — overlap cannot be planned from a profile alone');
  ok(plan.has('RESOURCE_TIMELINE'),
     'plan_round gets the pipe table it will be priced against');
  ok(passedByPhase.get('verify_engineer:verify').has('MODIFIABLE_FILES'),
     'verify gets the file whitelist its step 5 rejects patches against');
  ok(megaRuntimeVerifyKeys.has('CANDIDATE_IMPORT_MODULES') &&
     megaRuntimeVerifyKeys.has('EXPECTED_STRUCTURAL_TREE_DIGEST') &&
     megaRuntimeVerifyKeys.has('EXPERT_SKILL_CONTRACT_TOOL') &&
     megaRuntimeVerifyKeys.has('LANE_LOCK'),
     'Mega runtime verify receives import, tree-digest, contract, and lane-lock identity inputs');
  ok(passedByPhase.get('tech_lead:update_memory').has('ROADMAP_LADDER'),
     'update_memory gets the ladder — it writes the memory the NEXT wave reads');
}

console.log('\n# 3. the gate and the planner must see the same table');
{
  // The specific inversion that made this a bug rather than an omission: pipeOccupancyGate rejects
  // directions using resource_timeline. If the gate reads an artifact the planner does not get, the
  // planner is being scored on hidden information.
  const gateReads = /pipeOccupancyGate\(analysis && analysis\.resource_timeline, directions\)/.test(src);
  ok(gateReads, 'the per-round pipe gate still prices directions against resource_timeline');
  ok(passedByPhase.get('tech_lead:plan_round').has('RESOURCE_TIMELINE'),
     'and the planner it scores is handed that same table');
  const tl = fs.readFileSync(path.join(WF, 'roles', 'mega', 'tech_lead.md'), 'utf8').replace(/\s+/g, ' ');
  ok(/no \*\*edge\*\*|no edge/.test(tl),
     'tech_lead.md tells the planner what the graph is FOR: work with no edge between it can overlap');
  ok(/say in `reasoning` why not/.test(tl),
     'and an unordered pair left unproposed must be explained, not silently dropped');
}

console.log('\n# 4. an absent whitelist is not an absent check');
{
  ok(/MODIFIABLE_FILES: \(analysis[\s\S]{0,220}: 'UNDECLARED'/.test(src),
     'MODIFIABLE_FILES is passed as UNDECLARED when Analyze declared none, never omitted');
  const ve = fs.readFileSync(path.join(WF, 'roles', 'mega', 'verify_engineer.md'), 'utf8').replace(/\s+/g, ' ');
  ok(/do \*\*not\*\* treat that as "no check to do"/i.test(ve),
     'and verify is told that UNDECLARED means judge it yourself, not skip it');
}

console.log('\n# 5. the ledger is handed the column it is asked to fill');
{
  ok(/roadmap_rung: r\.d\.roadmap_rung/.test(src),
     'ROUND_RESULTS carries each direction\'s rung, so the ledger does not have to guess it');
  const tl = fs.readFileSync(path.join(WF, 'roles', 'mega', 'tech_lead.md'), 'utf8').replace(/\s+/g, ' ');
  ok(/rung that is merely absent reads to the next wave as a rung that was tried and dropped/.test(tl),
     'and unspent rungs must be written down as unresolved, or the next wave will not re-propose them');
}

console.log(failures
  ? `\nFAILED: ${failures} assertion(s).`
  : '\nPASS: no artifact is produced, gated on, and withheld from the role that must act on it.');
process.exit(failures ? 1 : 0);
