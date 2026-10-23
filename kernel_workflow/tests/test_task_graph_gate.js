#!/usr/bin/env node
// Behavioral test for the Analyze-phase task-graph gate (args.require_task_graph).
//
// WHY THIS IS NOT A SOURCE-GREP TEST. The house rule for changes to kernel_mega.js is that they
// are validated by BEHAVIOR, not by asserting that a phrase appears in the source — a grep test
// passes just as happily on a comment as on working code, and it goes green the moment someone
// writes the words while the logic says something else. So this file LIFTS the gate verbatim out of
// kernel_mega.js between its `<<REPLAY:task_graph_gate>>` markers and runs it, exactly as
// scripts/replay_runs.js does for the positive-control gate. If the shipped function changes, this
// test runs the changed function.
//
// WHAT THE GATE IS FOR. The failure it catches is an Analyze phase that asserts "the stages are
// serialized, so fuse them" and is believed, because nothing downstream can tell an analysis from a
// restatement of the launch count. Requiring the graph as an object makes the claim checkable. The
// gate does not abort — an honest partial graph beats a retry loop that eventually returns a
// complete-looking invention — it emits a caveat that travels to the report.
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

// --- lift ------------------------------------------------------------------------------------
const m = src.match(
  /\/\/ <<REPLAY:task_graph_gate>>[\s\S]*?\n([\s\S]*?)\n\s*\/\/ <<\/REPLAY:task_graph_gate>>/);
if (!m) {
  console.error('FAILED: the <<REPLAY:task_graph_gate>> markers are missing from kernel_mega.js.');
  console.error('They are not decoration: without them this test cannot reach the shipped gate and');
  console.error('would silently start testing nothing. Restore the markers around taskGraphGate().');
  process.exit(1);
}
// eslint-disable-next-line no-new-func
const taskGraphGate = new Function(`${m[1]}\nreturn taskGraphGate;`)();
ok(typeof taskGraphGate === 'function', 'gate lifted from kernel_mega.js and is callable');

// A lifted region that closes over run state would throw here rather than in production, which is
// the cheap way to find out. Purity is a property this test enforces, not one it assumes.
ok((() => { try { taskGraphGate(null); return true; } catch { return false; } })(),
   'the lifted region is pure (no closure over run state)');

const G = (over) => Object.assign({
  nodes: [
    { id: 'a', stage: 's1', tile: 'e0', duration_us: 10, source: 'profile' },
    { id: 'b', stage: 's2', tile: 'e0', duration_us: 20, source: 'profile' },
  ],
  edges: [{ from: 'a', to: 'b', scope: 'l2', enforced_by: 'launch_boundary', bytes: 1024 }],
  critical_path: ['a', 'b'], critical_path_us: 30, measured_e2e_us: 50,
  zero_slack_nodes: ['a', 'b'], false_edges: [], unknowns: [],
}, over || {});

// --- 1. absence is reported, and reported as an epistemic problem ------------------------------
console.log('\n# a missing graph');
for (const [what, tg] of [
  ['null', null], ['undefined', undefined], ['empty object', {}],
  ['empty arrays', { nodes: [], edges: [] }],
  ['prose instead of a graph', { summary: 'the stages are serialized, so we should fuse them' }],
]) {
  const g = taskGraphGate(tg);
  ok(g.verdict === 'MISSING', `${what} -> MISSING`);
  ok(/asserted, not derived/.test(g.caveat), `${what} caveat says the directions are unsupported`);
}
// The last case is the one that matters most: a run CAN return a `task_graph` key containing a
// sentence. That must not read as "the graph was provided" just because the key exists.

// --- 2. a real graph passes and is summarised in numbers ---------------------------------------
console.log('\n# a usable graph');
{
  const g = taskGraphGate(G());
  ok(g.verdict === 'OK', 'a tile-level graph with a narrower-scope edge passes');
  ok(g.caveat === '', 'no caveat on a usable graph');
  ok(/2 nodes, 1 edges/.test(g.summary), 'summary counts nodes and edges');
  ok(/1 enforced only by a launch boundary/.test(g.summary),
     'summary reports the launch-boundary edge count -- the size of the opportunity');
  ok(/dependence explains 60\.0% of e2e/.test(g.summary),
     'summary reports what share of e2e the dependence chain explains');
  ok(/40\.0% is resource-bound work plus bubbles/.test(g.summary),
     'the residual is attributed to resources+bubbles, not offered as recoverable headroom');
  // Regression guard for a real miscalibration. The gate used to print the residual as "addressable
  // ceiling", which is false whenever the operator is throughput-bound: once a run returned
  // cp=130µs / e2e=4580.8µs and had to add a prose note explaining that ~97% of the gap was arithmetic
  // no reordering can delete. A headline number the graph must argue against is worse than none,
  // because the number is what travels into the report.
  ok(!/addressable ceiling/.test(g.summary),
     'the residual is NOT labelled an addressable ceiling');
}
{
  const g = taskGraphGate(G({ critical_path_us: undefined, measured_e2e_us: undefined }));
  ok(g.verdict === 'OK' && /NOT quantified/.test(g.summary),
     'an unquantified critical path is stated, not silently omitted');
}
{
  const g = taskGraphGate(G({
    nodes: [{ id: 'a', duration_us: 10, source: 'assumed' },
            { id: 'b', duration_us: 20, source: 'profile' }],
  }));
  ok(/1 node duration\(s\) assumed/.test(g.summary),
     'assumed durations are counted -- an all-assumed graph is a guess wearing a schema');
}
{
  const g = taskGraphGate(G({ unknowns: [{ what: 'scope of a->b' }] }));
  ok(/1 declared unknown/.test(g.summary), 'declared unknowns are surfaced, not penalised');
  ok(g.verdict === 'OK', 'declaring an unknown does not fail the gate -- honesty must stay cheap');
}

// --- 3. the arithmetic self-check ---------------------------------------------------------------
console.log('\n# internal consistency');
{
  const g = taskGraphGate(G({ critical_path_us: 80, measured_e2e_us: 50 }));
  ok(g.verdict === 'INCONSISTENT', 'critical path longer than e2e is caught');
  ok(/cannot be longer than the thing it is a path through/.test(g.caveat),
     'the caveat explains why it is impossible rather than just flagging it');
}
{
  const g = taskGraphGate(G({ critical_path_us: 50, measured_e2e_us: 50 }));
  ok(g.verdict === 'OK', 'critical path EQUAL to e2e is legal (a fully serial operator)');
}

// --- 4. the kernel-granularity tell -------------------------------------------------------------
// This is the failure mode the whole artifact exists to prevent, so it gets its own verdict: a
// "graph" whose nodes are kernels. Every edge is a launch boundary and no edge has a narrow scope,
// because the scope column was never really answered. Such a graph cannot separate "this ordering
// is required by the data" from "this ordering is what the code happens to do" -- which is the only
// question it was built to answer -- so a fusion ranked from it is unsupported.
console.log('\n# kernel-granular graphs are called out');
{
  const g = taskGraphGate({
    nodes: [{ id: 'k1' }, { id: 'k2' }, { id: 'k3' }],
    edges: [
      { from: 'k1', to: 'k2', scope: 'hbm', enforced_by: 'launch_boundary' },
      { from: 'k2', to: 'k3', scope: 'hbm', enforced_by: 'launch_boundary' },
    ],
    critical_path_us: 40, measured_e2e_us: 50,
  });
  ok(g.verdict === 'KERNEL_GRANULARITY', 'all-launch-boundary, all-coarse-scope is flagged');
  ok(/nodes are KERNELS/.test(g.caveat), 'the caveat names the actual mistake');
}
{
  // Discrimination: the same shape with ONE narrower-scope edge is a real tile graph that happens
  // to contain coarse edges. Flagging that would train readers to ignore the verdict.
  const g = taskGraphGate({
    nodes: [{ id: 'a' }, { id: 'b' }, { id: 'c' }],
    edges: [
      { from: 'a', to: 'b', scope: 'hbm', enforced_by: 'launch_boundary' },
      { from: 'b', to: 'c', scope: 'lds', enforced_by: 'launch_boundary' },
    ],
    critical_path_us: 40, measured_e2e_us: 50,
  });
  ok(g.verdict === 'OK', 'one narrow-scope edge is enough to show the column was answered');
}
{
  // And a graph where the edges are already fine-grained is a finding, not a failure: there is
  // nothing left to unfuse, and the run should be free to say so.
  const g = taskGraphGate(G({
    edges: [{ from: 'a', to: 'b', scope: 'l2', enforced_by: 'fence_flag' }],
  }));
  ok(g.verdict === 'OK' && /0 enforced only by a launch boundary/.test(g.summary),
     'zero launch-boundary edges passes and says so -- "nothing to unfuse" is a valid result');
}

// --- 4b. the operands the edge does not carry -----------------------------------------------------
// The graph's nodes are output tiles. Everything the consumer loads that the producer does not
// produce -- the second weight matrix, its scales, the descriptors, the index arrays -- therefore
// has no node, no edge, and no way to be ranked. On the operator this was written for that omission
// hid the cheapest lever on the table: GEMM2's weights depend on nothing and both stages measured
// ~30% HBM, so they could have been pulled into L2 during GEMM1. Six nodes, five edges, two correct
// fusable edges found, and prefetch never mentioned once.
console.log('\n# what the consumer loads that the producer never sends');
{
  const g = taskGraphGate(G({
    edges: [{ from: 'a', to: 'b', scope: 'l2', enforced_by: 'launch_boundary', fan_in: 12 }],
  }));
  ok(g.verdict === 'OK' && /DID NOT SAY WHAT THE CONSUMER LOADS ANYWAY/.test(g.caveat),
     'a fan-in edge that never says what else its consumer needs is called out');
  ok(/a->b/.test(g.caveat), 'and the edge is named, so the answer is a one-line addition');
}
{
  const g = taskGraphGate(G({
    edges: [{ from: 'a', to: 'b', scope: 'l2', enforced_by: 'launch_boundary', fan_in: 12,
              producer_independent_operands: ['w2', 'w2_scale', 'descriptors'] }],
  }));
  ok(g.caveat === '', 'answering it clears the note');
}
{
  const g = taskGraphGate(G({
    edges: [{ from: 'a', to: 'b', scope: 'l2', enforced_by: 'launch_boundary', fan_in: 12,
              producer_independent_operands: [] }],
  }));
  ok(g.caveat === '',
     'and an EMPTY list clears it too — "the consumer needs nothing else" is a real answer, and a ' +
     'gate that cannot accept it teaches the role to invent operands');
}
{
  const g = taskGraphGate(G());   // the base fixture's edge has no fan_in
  ok(g.caveat === '',
     'an edge with no fan-in is not asked: the question is about a consumer that gathers, and ' +
     'asking it everywhere turns a real finding into a form field');
}
ok(/advisory|Advisory/.test(src.slice(src.indexOf('WHAT THE CONSUMER NEEDS'), src.indexOf('WHAT THE CONSUMER NEEDS') + 1800)),
   'and it is advisory by construction — a missing operand list makes prefetch unrankable, it does ' +
   'not make the graph wrong');

// --- 5. wiring ----------------------------------------------------------------------------------
// The gate can be perfect and still never run. These four are the connections between it and the
// rest of the workflow, and each has been an independent way for a feature to be inert.
console.log('\n# wiring');
ok(/A\.require_task_graph/.test(src), 'the gate is driven by args.require_task_graph');
ok(/REQUIRE_TASK_GRAPH: '1'/.test(src), 'the requirement is passed through to the tech_lead agent');
ok(/task_graph:\s*\{\s*type:\s*\['object',\s*'null'\]/.test(src),
   'ANALYZE_SCHEMA accepts task_graph (without this the model cannot return one)');
ok(/TASK_GRAPH_CAVEAT \? \{ TASK_GRAPH_CAVEAT \}/.test(src),
   'the caveat is threaded into the report, not just logged');
const lead = fs.readFileSync(path.join(WF, 'roles', 'mega', 'tech_lead.md'), 'utf8');
ok(/REQUIRE_TASK_GRAPH/.test(lead), 'tech_lead.md tells the role when the graph is required');
ok(/enforced_by/.test(lead), 'tech_lead.md specifies the enforced_by vocabulary');
ok(/tile_task_graph\.md/.test(lead), 'tech_lead.md routes to the derivation method card');
ok(/fusion_preconditions\.md/.test(lead),
   'tech_lead.md requires the fusion precondition test before ranking a fusion');

// Default-off is a property worth asserting: turning this on for every kernel would produce
// filled-in forms, and a filled-in form is worse than no artifact because it looks like evidence.
ok(!/require_task_graph\s*[|?]{2}\s*true/.test(src), 'the requirement is opt-in, not defaulted on');

console.log(failures
  ? `\nFAILED: ${failures} assertion(s).`
  : '\nPASS: the task-graph gate discriminates and is wired end-to-end.');
process.exit(failures ? 1 : 0);
