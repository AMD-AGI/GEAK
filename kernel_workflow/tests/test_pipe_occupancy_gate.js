#!/usr/bin/env node
// Behavioral test for the Analyze-phase pipe-occupancy gate.
//
// Same house rule as test_task_graph_gate.js: this LIFTS the shipped gate verbatim out of
// kernel_mega.js between its `<<REPLAY:pipe_occupancy_gate>>` markers and runs it, rather than
// grepping the source for a phrase. A grep test goes green the moment somebody writes the words.
//
// WHAT THE GATE IS FOR. The dependency graph answers whether two pieces of work MAY overlap. That
// question is necessary and not sufficient, and a program that only asks it will spend wave after
// wave on directions that are perfectly derivable from a correct graph and foreclosed by a counter
// nobody collected. That is not hypothetical: one program ran four waves of occupancy, launch-count
// and communication-overlap arms, then took hardware counters for the first time and found no pipe
// above 41%, HBM at 7-8%, an inter-kernel gap of 0.00us over 72 boundaries, and occupancy already
// kill-gated at both ends. Every lever it had bought was dead before it was bought.
//
// So the gate does three things, and the tests below are one per thing: it prints the table, it
// DERIVES the class from the numbers instead of trusting the stated one, and it prices each
// direction against the pipe it claims to fill.
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
  /\/\/ <<REPLAY:pipe_occupancy_gate>>[\s\S]*?\n([\s\S]*?)\n\s*\/\/ <<\/REPLAY:pipe_occupancy_gate>>/);
if (!m) {
  console.error('FAILED: the <<REPLAY:pipe_occupancy_gate>> markers are missing from kernel_mega.js.');
  console.error('Without them this test cannot reach the shipped gate and would silently test nothing.');
  process.exit(1);
}
// eslint-disable-next-line no-new-func
const pipeOccupancyGate = new Function(`${m[1]}\nreturn pipeOccupancyGate;`)();
ok(typeof pipeOccupancyGate === 'function', 'gate lifted from kernel_mega.js and is callable');
ok((() => { try { pipeOccupancyGate(null, null); return true; } catch { return false; } })(),
   'the lifted region is pure (no closure over run state)');

// The real measured shape from the program this came from: nothing saturated, gap zero.
const LATENT = (over) => Object.assign({
  pipes: [
    { stage: 's1', pipe: 'valu', utilization_pct: 19.3, source: 'SQ_INSTS_VALU/SQ_BUSY_CYCLES' },
    { stage: 's1', pipe: 'mfma', utilization_pct: 41.4, source: 'SQ_INSTS_MFMA/SQ_BUSY_CYCLES' },
    { stage: 's1', pipe: 'lds', utilization_pct: 15.5, source: 'SQ_LDS_IDX_ACTIVE/SQ_BUSY_CYCLES' },
    { stage: 's1', pipe: 'hbm', utilization_pct: 8.5, source: 'bytes/peak_bw' },
  ],
  interkernel_gap_us: { median: 0.0, max: 0.002, n_boundaries: 72 },
  class: 'latency_bound',
  stall_reason: [{ stage: 's1', waiting_on: 'B-operand load', counter: 'SQ_WAIT_ANY' }],
  idle_pipe_opportunities: [
    { stage: 's1', idle_pipe: 'hbm', candidate_work: 'GEMM2 B weights + scales',
      dag_edge_status: 'no edge: w2 is constructor-set, not produced by s1',
      blocked_by: 'launch_boundary' },
  ],
  closed_axes: [{ axis: 'occupancy', ruled_out_by: 'nw16 2->4 waves/SIMD = -11.2%, 4/4' }],
  unknowns: [],
}, over || {});

console.log('\n# a missing table is an epistemic problem, not a zero');
{
  const r = pipeOccupancyGate(null, []);
  ok(r.verdict === 'MISSING', 'no table at all -> MISSING');
  ok(/unbounded above/.test(r.caveat),
     'the caveat says what is now unknown — every expected_speedup is unbounded');
  ok(/occupancy/i.test(r.caveat) && /fuse launches/i.test(r.caveat),
     'and names the two reflexes that can no longer be ruled out cheaply');
  // An empty pipes array must not read as "measured, all zero".
  ok(pipeOccupancyGate({ pipes: [] }, []).verdict === 'MISSING',
     'an empty pipes array is MISSING, not a table of zeroes');
}

console.log('\n# the class is DERIVED from the numbers, not taken on trust');
{
  const r = pipeOccupancyGate(LATENT(), []);
  ok(r.class_derived === 'latency_bound', 'all pipes low + zero gap -> latency_bound');
  ok(/max=41\.4%/.test(r.summary) && /0\.000us/.test(r.summary),
     'the summary carries the numbers a reader needs to recompute the verdict');
  // The failure mode worth catching: a stated class the table does not support.
  const lying = pipeOccupancyGate(LATENT({ class: 'throughput_bound' }), []);
  ok(lying.verdict === 'INCONSISTENT' && /CLASS CONTRADICTS THE TABLE/.test(lying.caveat),
     'a stated class the numbers do not support is flagged, because the class picks the levers');
  // And the honest cases still classify.
  const sat = pipeOccupancyGate(LATENT({
    pipes: [{ stage: 's1', pipe: 'mfma', utilization_pct: 93.0, source: 'x' }],
    class: 'throughput_bound',
  }), []);
  ok(sat.class_derived === 'throughput_bound', 'a saturated pipe -> throughput_bound');
  const crossEngine = pipeOccupancyGate(LATENT({
    pipes: [
      { stage: 'producer', pipe: 'mfma', utilization_pct: 93.0, source: 'x' },
      { stage: 'consumer', pipe: 'interconnect', utilization_pct: 18.0, source: 'y' },
    ],
    class: 'mixed',
  }), []);
  ok(crossEngine.class_derived === 'mixed',
     'a saturated MFMA engine plus idle interconnect is mixed, not globally throughput-bound');
  const hostb = pipeOccupancyGate(LATENT({
    interkernel_gap_us: { median: 12.0, max: 20, n_boundaries: 72 }, class: 'launch_bound',
  }), []);
  ok(hostb.class_derived === 'launch_bound', 'idle pipes + a real launch gap -> launch_bound');
}

console.log('\n# latency-bound closes the two reflexes, by name and with the mechanism');
{
  const r = pipeOccupancyGate(LATENT(), []);
  ok(/OCCUPANCY/.test(r.caveat) && /adding waves adds\s*\n?\s*stalled waves/.test(r.caveat),
     'occupancy is closed AND the mechanism is given, so it is not read as a style preference');
  ok(/registers and LDS that software pipelining needs/.test(r.caveat),
     'the harm direction is stated — occupancy can buy out the actual fix');
  ok(/LAUNCH\s*\n?\s*OVERHEAD \(the gap is already zero\)/.test(r.caveat),
     'launch-count fusion is closed against the measured gap, not against an opinion');
  // The crucial non-obvious half: a zero gap does NOT kill fusion, it changes what fusion is for.
  ok(/inexpressible/.test(r.caveat) && /grid-wide barrier plus a\s*\n?\s*pipeline drain/.test(r.caveat),
     'and it says what fusion IS still for — the boundary makes next-stage work inexpressible');
  // A table saying the pipes are idle while naming nothing to put in them has not answered the
  // question it was collected to answer.
  const noOpp = pipeOccupancyGate(LATENT({ idle_pipe_opportunities: [] }), []);
  ok(/NO IDLE-PIPE OPPORTUNITIES LISTED/.test(noOpp.caveat),
     'idle pipes with no candidate work listed is itself flagged');
}

console.log('\n# directions are priced against the pipe they claim');
{
  // 1/0.085 ~= 11.8x is the perfect-fill bound for a pipe at 8.5%; 1.05x is fine.
  const sane = pipeOccupancyGate(LATENT(), [
    { id: 'd0', title: 'prefetch w2', fills_pipe: 'hbm', pipe_util_pct: 8.5, expected_speedup: 1.05 },
  ]);
  ok(!/OVERCLAIMS/.test(sane.caveat), 'a claim inside its pipe headroom passes silently');

  const greedy = pipeOccupancyGate(LATENT(), [
    { id: 'd1', title: 'moon', fills_pipe: 'mfma', pipe_util_pct: 93.0, expected_speedup: 2.0 },
  ]);
  ok(greedy.verdict === 'INCONSISTENT' && /OVERCLAIMS ITS PIPE/.test(greedy.caveat),
     'a claim above the perfect-fill bound of its own pipe is rejectable by arithmetic');
  ok(/without a run/.test(greedy.caveat),
     'and the caveat says so — the point is not spending the lease to find out');

  const vague = pipeOccupancyGate(LATENT(), [
    { id: 'd2', title: 'make it faster', specialty: 'compute', expected_speedup: 1.3 },
  ]);
  ok(/NO PIPE NAMED/.test(vague.caveat),
     'a direction that names no pipe is flagged as underived rather than silently ranked');
  // deep_explore has a broad mandate by construction and must not be forced to name one unit.
  const deep = pipeOccupancyGate(LATENT(), [
    { id: 'd3', title: 'rewrite', specialty: 'deep_explore', expected_speedup: 1.3 },
  ]);
  ok(!/NO PIPE NAMED/.test(deep.caveat),
     'deep_explore is exempt — its mandate is explicitly not one pipe');
}

// A hole Analyze located, measured, and put a number on, that no direction then claims. The
// per-direction checks above cannot see this: the evidence of the miss is in a list they never read.
// Measured cost of not having it: a Stage2 tail round at 18.75% occupancy, 115us, 2.5% of e2e -- one
// of two quantified holes in the whole operator -- was assumed to be collected as a side effect of a
// fusion rung. That rung never ran, and the 2.5% sat unclaimed for four waves without ever being
// declined.
console.log('\n# a sized hole that nobody claimed');
const SIZED = (over) => LATENT(Object.assign({
  idle_pipe_opportunities: [
    { id: 'H1', window: 'the combine kernel', idle_pipe: 'mfma', recoverable_us: 140, pct_of_e2e: 3.0 },
    { id: 'H2', window: 'stage2 final round', idle_pipe: 'all', recoverable_us: 115, pct_of_e2e: 2.5 },
  ],
}, over || {}));
const DIR = (o) => Object.assign({ id: 'D1', fills_pipe: 'mfma', pipe_util_pct: 24.7, expected_speedup: 1.03 }, o);
{
  const r = pipeOccupancyGate(SIZED(), [DIR({ fills_hole: 'H1' })]);
  ok(/SIZED HOLE WITH NO DIRECTION \(1\)/.test(r.caveat) && /H2/.test(r.caveat) && !/H1\b/.test(r.caveat.split('SIZED HOLE')[1]),
     'the claimed hole drops out and the unclaimed one is named');
  ok(/115us/.test(r.caveat) && /2\.5% of e2e/.test(r.caveat),
     'with the size Analyze already measured, so the cost of leaving it is in the sentence');
}
{
  const r = pipeOccupancyGate(SIZED(), [DIR({ graph_refs: ['H1', 'H2'] })]);
  ok(!/SIZED HOLE/.test(r.caveat),
     'graph_refs counts as a claim — a direction should not have to name the same hole twice');
}
{
  const r = pipeOccupancyGate(SIZED({
    idle_pipe_opportunities: [{ id: 'H2', recoverable_us: 115, pct_of_e2e: 2.5, rides_on: 'D2' }],
  }), [DIR({})]);
  ok(!/SIZED HOLE/.test(r.caveat),
     '`rides_on` is a legitimate answer: a hole another rung absorbs is planned for, and now the ' +
     'claim is on the record where it can be checked when that rung reports');
}
{
  const r = pipeOccupancyGate(SIZED({
    idle_pipe_opportunities: [{ id: 'H3', window: 'dispatch recv wait', recoverable_us: null, pct_of_e2e: null }],
  }), [DIR({})]);
  ok(!/SIZED HOLE/.test(r.caveat),
     'an UNSIZED hole is not demanded: Analyze explicitly refusing to put a number on one is the ' +
     'honest state, and charging for it would push the role into inventing the number');
}
{
  const r = pipeOccupancyGate(SIZED(), []);
  ok(!/SIZED HOLE/.test(r.caveat),
     'and the post-Analyze call, which has no directions yet, does not report every hole as unowned');
}

console.log('\n# the knowledge and the roles actually route to it');
{
  const card = fs.readFileSync(path.join(WF, 'knowledge/pipe_occupancy.md'), 'utf8');
  const lead = fs.readFileSync(path.join(WF, 'roles/mega/tech_lead.md'), 'utf8');
  ok(/Raising occupancy can buy out the fix/.test(card),
     'the card states the counter-intuitive harm, not just that occupancy "may not help"');
  ok(/SQ counters aggregate every shader engine/.test(card) && /units of 2/.test(card),
     'the two collection traps that each cost a day are carried with the method');
  ok(/inexpressible/.test(card),
     'the card carries the barrier-dissolution argument that survives a zero launch gap');
  ok(/A direction that cannot name a pipe is not yet an optimization/.test(card),
     'and the pricing rule that makes the table load-bearing');
  ok(/pipe_occupancy\.md/.test(lead), 'tech_lead points at the card');
  ok(/"resource_timeline"/.test(lead), 'the ANALYZE contract asks for the artifact');
  ok(/`resource_timeline` is required whenever `task_graph` is/.test(lead),
     'and requires it on the same trigger as the graph, so one cannot be filed without the other');
  ok(/Every GPU direction names the pipe it fills/.test(lead),
     'plan_round binds the lease allocation to the table');
  ok(/pipeOccupancyGate\(analysis && analysis\.resource_timeline, directions\)/.test(src),
     'and the orchestrator actually re-runs the gate against the round\'s directions');
}

console.log(
  failures === 0
    ? '\nPASS: the pipe table is collected, derived from, and priced against.'
    : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
