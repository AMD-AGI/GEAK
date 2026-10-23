#!/usr/bin/env node
// Every field a gate reads must be DECLARED in the schema, not only described in a comment.
//
// This is the mirror of test_input_contract.js. That one checks inputs: does the role receive what
// its contract says it receives. This one checks outputs: when a gate reads `x.foo`, was the agent
// that produced `x` ever actually required to return `foo`?
//
// WHY IT IS A DEFECT AND NOT A STYLE POINT. `obj()` sets `additionalProperties: true`, so a schema
// entry of `{ type: 'object', additionalProperties: true }` accepts ANY object. Four of the most
// load-bearing artifacts in this workflow were declared that way, with their real shape written in
// the comment above them. In every case the gate's handling of a MISSING field is a default, and in
// every case the default reads as the benign outcome:
//
//   positive_control  — `pc.ran !== false` is TRUE for a missing `ran`; a missing `switch_present`
//                       makes the pre-flight inert; a missing `null_arm_pct` silently fails the
//                       overshoot quiet-check. A renamed field validates and reads as a control.
//   task_graph        — the gate counts edges with `enforced_by === 'launch_boundary'`. A graph that
//                       names that column anything else reports "0 edges enforced only by a launch
//                       boundary", which reads as the FINDING "nothing to unfuse" rather than "the
//                       column was never filled in". The shipped gate's own comment says these two
//                       must not look alike; the schema was what let them.
//   resource_timeline — the gate DROPS any pipe row whose `utilization_pct` is not finite. A
//                       five-pipe table with the field misnamed arrives as an empty table and is
//                       reported as PIPE TABLE MISSING.
//   candidate_directions — the ladder. Omit `gated_on` and every rung looks unconditional: the gate
//                       finds no unmet prerequisite and reports OK on precisely the run whose
//                       ordering was lost.
//   ledger            — memoryMerge keys rows by `id`. A row without one merges as a new row every
//                       round, which is indistinguishable from normal ledger growth.
//
// The check below is mechanical: lift each `<<REPLAY:*>>` gate, collect the property names it reads,
// and assert each is declared somewhere in the schemas. It is deliberately a whole-file search
// rather than a per-schema one — pinning which schema owns which gate would encode today's wiring
// and break on a rename, and the failure this catches (the name is nowhere) survives that looseness.
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

const strip = (s) => s.replace(/\/\/[^\n]*/g, '');
const region = (name) => {
  const m = src.match(new RegExp(`// <<REPLAY:${name}>>([\\s\\S]*?)// <</REPLAY:${name}>>`));
  return m ? strip(m[1]) : null;
};

// Everything declared as a property anywhere in the schema block, at any nesting depth. A gate's
// field is satisfied by being declared ANYWHERE — see the header note on why this is deliberate.
const schemaBlock = src.slice(src.indexOf('const perCase = {'), src.indexOf('// ---', src.indexOf('const VALIDATE_SCHEMA')));
const declared = new Set([...strip(schemaBlock).matchAll(/(?:^|[\s{,])([a-z_][a-z0-9_]*)\s*:\s*(?:\{|obj\(|perCase)/gm)]
  .map((m) => m[1]));
ok(declared.size > 60, `parsed ${declared.size} declared schema properties`);
ok(declared.has('utilization_pct') && declared.has('enforced_by'),
   'the parser reaches nested properties, not just top-level schema keys');

// JS built-ins and local-variable members that are not artifact fields.
const NOT_A_FIELD = new Set(['map', 'filter', 'length', 'slice', 'forEach', 'find', 'includes',
  'push', 'join', 'toFixed', 'some', 'every', 'sort', 'flat', 'concat', 'indexOf', 'reduce', 'trim',
  'split', 'replace', 'match', 'keys', 'test', 'has', 'add', 'size', 'abs', 'min', 'max', 'sign',
  'round', 'floor', 'isFinite', 'isArray', 'from', 'set', 'get', 'toLowerCase', 'toUpperCase',
  'startsWith', 'endsWith', 'flatMap', 'entries', 'values', 'toString', 'padEnd', 'padStart']);

// Only reads off an AGENT-RETURNED artifact are schema-governed. A gate also dereferences its own
// locals, the run's input args, and bookkeeping the script itself attaches to a result after the
// fact — none of those are things an agent was ever asked to return, and folding them in would bury
// the real findings in noise. So each gate declares which root identifiers hold agent output.
const ARTIFACT_ROOTS = {
  task_graph_gate: ['tg', 'e', 'n'],            // the graph, plus its edge/node elements
  pipe_occupancy_gate: ['rt', 'p', 'd'],        // the timeline, its pipe rows, and plan directions
  roadmap_ladder_gate: ['r', 'd'],              // ladder rungs and dispatched directions
  pc_gate: ['pc'],                              // the control result. NOT POSITIVE_CONTROL, which is
                                                // the run's INPUT args (magnitude, implausible_pct).
  // memory_merge mostly manipulates the insight BOOK, whose text/restated/void_round/first_round
  // fields this region writes itself — those are its outputs, not anything an agent was asked for,
  // and checking them against the schema would be checking the script against itself. The one
  // schema-governed thing it reads is the direction id it keys dead-ends by, which comes from
  // PLAN_SCHEMA.
  memory_merge: ['d'],
  claim_boundary: ['eng', 'ver'],               // engineer + verify results
  // The shelf's rows are built by the script, but the two things it decides on are not: `c` is a
  // verified candidate (VERIFY_SCHEMA.touched_files) and `e` carries those fields forward across
  // waves through SETUP_SCHEMA.prior_state.shelf. Deciding whether a patch still applies off a
  // field nobody was required to return is exactly the failure this file exists to catch.
  candidate_shelf: ['c', 'e'],
  // `ver` is the verification result and `o` is its `overlap` sub-object, whose whole point is that
  // a renamed or dropped field must not read as "measured and fine". This is the field group most
  // at risk of drifting: the meter does not exist yet, so the schema is the only thing pinning the
  // names the engineer will be asked to fill in.
  overlap_gate: ['ver', 'o'],
  // evidence_stop is deliberately absent for the same reason as memory_merge's outputs: the three
  // fields it decides on (`inactive`, `unbacked`, `same_artifact`) are marks the round attaches to
  // a result AFTER verification, not things any agent was asked to return. The one agent-returned
  // field it touches, `verified_geomean`, it reaches through rungOutcomeOf, which is checked where
  // that region is.
  //
  // bimodal_split is deliberately absent. It reads nothing off an agent artifact — its inputs are
  // raw latency readings the benchmark collected, passed in as plain numbers. Listing it here would
  // require inventing schema fields for `p.base`/`p.cand` and would check the script against itself.
};
console.log('\n# 1. every property a gate reads off an agent artifact is declared in a schema');
for (const [g, roots] of Object.entries(ARTIFACT_ROOTS)) {
  const body = region(g);
  if (!body) { ok(false, `${g}: REPLAY markers missing — the gate cannot be checked`); continue; }
  const re = new RegExp(`\\b(?:${roots.join('|')})\\.([a-z_][a-z0-9_]*)\\b`, 'g');
  const reads = new Set([...body.matchAll(re)].map((m) => m[1]));
  const undeclared = [...reads].filter((f) => !declared.has(f) && !NOT_A_FIELD.has(f)).sort();
  ok(reads.size > 0, `${g}: found ${reads.size} artifact property reads to check`);
  ok(undeclared.length === 0,
     `${g}: all declared${undeclared.length ? ` — UNDECLARED ${undeclared.join(', ')}` : ''}`);
}

console.log('\n# 2. the artifacts that were opaque, and the default each one hid');
{
  const opaque = (field) => new RegExp(`${field}: \\{ type: \\[?'object`).test(src)
    || new RegExp(`${field}: \\{ type: 'array', items: \\{ type: 'object', additionalProperties: true \\} \\}`).test(src);
  for (const [f, hid] of [
    ['positive_control', 'a missing `ran` reads as "it ran"'],
    ['task_graph', 'a renamed enforcement column reads as "nothing to unfuse"'],
    ['resource_timeline', 'a dropped utilization_pct reads as "PIPE TABLE MISSING"'],
    ['candidate_directions', 'a missing gated_on reads as "every rung is unconditional"'],
    ['ledger', 'a row without an id reads as normal ledger growth'],
    // The sixth, found while wiring the candidate shelf: the script reads patch_file/geomean/
    // weighted/arithmetic/per_case out of INTEGRATE_SCHEMA.best and none of them were declared.
    // A missing patch_file makes the commit step apply nothing; a missing geomean scores the
    // integration at 0 and the merge quietly loses to an individual patch.
    ['best', 'a missing patch_file commits nothing and a missing geomean scores the merge at 0'],
  ]) ok(!opaque(f), `${f} is no longer an opaque object — ${hid}`);
}

console.log('\n# 3. required means required only where absence is indistinguishable from success');
{
  // Over-requiring is its own failure: a structured-output schema that demands a field the agent
  // legitimately cannot know forces it to invent one. So the required lists are short and each entry
  // has to earn its place by having a dangerous default.
  const req = (name, fields) => {
    // Two shapes in the tree: `foo: obj({...}, [req])` and `foo: { type: 'array', items: obj({...},
    // [req]) }`. Matching only the first reported "required: none" for both array-valued schemas.
    let at = src.indexOf(`${name}: obj({`);
    if (at < 0) at = src.indexOf(`${name}: { type: 'array', items: obj({`);
    const seg = at < 0 ? '' : src.slice(at, at + 2000);
    const m = seg.match(/\}, \[([^\]]*)\]\)/);
    const got = m ? m[1].replace(/['\s]/g, '').split(',').filter(Boolean) : [];
    ok(fields.every((f) => got.includes(f)),
       `${name} requires ${fields.join(', ')} (got: ${got.join(', ') || 'none'})`);
  };
  req('positive_control', ['ran', 'switch_present', 'measured_pct']);
  req('ledger', ['id', 'verdict']);
  req('candidate_directions', ['id', 'gated_on']);
  // `gated_on: []` is a real answer. Requiring the key forces the planner to SAY nothing gates this
  // rung, instead of the gate having to read absence as that claim.
  ok(/gated_on: \{ type: 'array'/.test(src),
     'and gated_on is an array, so "[]" is expressible — required is not the same as non-empty');
}

console.log('\n# 4. the enums that carry a verdict are closed');
{
  // An open string field for a verdict is how `unresolved` came to exist only in the role prose.
  ok(/enum: \['confirmed', 'partial', 'unresolved', 'dead_end'\]/.test(src),
     'the ledger verdict enum includes unresolved, so a mis-graded rung cannot be spelled freely');
  ok(/enum: \['measured', 'assumed'\]/.test(src),
     'a task-graph node duration is measured or assumed — there is no third, vaguer option');
  ok(/enum: \['throughput_bound', 'latency_bound', 'launch_bound', 'mixed'\]/.test(src),
     'and the pipe class the gate derives is checked against a closed stated one');
}

console.log(failures
  ? `\nFAILED: ${failures} assertion(s).`
  : '\nPASS: no gate reads a field the schema never asked for.');
process.exit(failures ? 1 : 0);
