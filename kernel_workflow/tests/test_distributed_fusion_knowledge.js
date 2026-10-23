#!/usr/bin/env node
// Wiring guard for the `distributed` specialty and its knowledge file.
//
// The knowledge is only a capability if it is REACHABLE: the orchestrator must accept the
// specialty, the roles that consume knowledge must route to the file, and the file must carry the
// invariants that make a fused multi-rank kernel correct rather than merely fast. A plain
// documentation drop passes none of these.
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const WF = path.join(ROOT, 'kernel_workflow');
const read = (...p) => fs.readFileSync(path.join(WF, ...p), 'utf8');

let failures = 0;
const ok = (condition, message) => {
  if (condition) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# distributed specialty wiring');

// 1. Orchestrator accepts it. Without this the TechLead cannot emit the direction at all:
//    PLAN_SCHEMA validation rejects the round and the knowledge is unreachable.
const src = read('kernel_mega.js');
const enumLine = src.match(/specialty:\s*\{\s*type:\s*'string',\s*enum:\s*\[([^\]]+)\]/);
ok(!!enumLine, 'PLAN_SCHEMA specialty enum found');
if (enumLine) {
  const values = enumLine[1].split(',').map((s) => s.trim().replace(/'/g, ''));
  ok(values.includes('distributed'), 'enum accepts `distributed`');
  for (const pre of ['algorithm', 'memory', 'compute', 'host_runtime', 'deep_explore']) {
    ok(values.includes(pre), `pre-existing specialty preserved: ${pre}`);
  }
}

// 2. Dispatch stays generic. Only `deep_explore` may be special-cased; if `distributed` needed
//    bespoke orchestration the enum change alone would silently do nothing.
ok(
  !/specialty === 'distributed'/.test(src),
  'dispatch treats `distributed` generically (no special-casing needed)',
);

// 3. The file exists and the consuming roles route to it.
const KNOWLEDGE = 'distributed_fusion.md';
ok(fs.existsSync(path.join(WF, 'knowledge', KNOWLEDGE)), `knowledge/${KNOWLEDGE} exists`);

const engineer = read('roles', 'mega', 'engineer.md');
ok(/`SPECIALTY`[^\n]*distributed/.test(engineer), 'engineer.md declares the specialty');
ok(
  new RegExp(`- distributed\\s+→[\\s\\S]{0,200}${KNOWLEDGE}`).test(engineer),
  'engineer.md routes `distributed` to the knowledge file',
);

const deep = read('roles', 'mega', 'deep_engineer.md');
ok(deep.includes(KNOWLEDGE), 'deep_engineer.md lists the knowledge file');
ok(
  /distributed_fusion\.md[\s\S]{0,240}(ONLY when|Skip entirely)/.test(deep),
  'deep_engineer.md gates the read so single-GPU kernels do not load it',
);

const lead = read('roles', 'mega', 'tech_lead.md');
ok(/candidate_directions[\s\S]{0,400}distributed/.test(lead), 'tech_lead.md may emit the specialty');
ok(lead.includes(KNOWLEDGE), 'tech_lead.md points at the knowledge file');
ok(
  /no-payload control/.test(lead),
  'tech_lead.md requires the no-payload control before spending a round',
);

ok(read('README.md').includes('distributed_fusion'), 'README knowledge list mentions it');

// 3b. The two gates the doctrine assumes. Knowledge that says "you must clear a liveness gate" is
//     inert unless verify actually applies one, and an interleaving rule is inert unless the
//     COMMANDMENT is built to interleave.
ok(
  /\.\.\.\(d\.specialty \? \{ SPECIALTY: d\.specialty \} : \{\}\)/.test(src),
  'verify_engineer receives the direction specialty',
);
ok(/liveness: \{ type: 'string' \}/.test(src), 'VERIFY_SCHEMA carries a liveness verdict');

const verify = read('roles', 'mega', 'verify_engineer.md');
ok(/`SPECIALTY` is `distributed`/.test(verify), 'verify gates the liveness stress on the specialty');
ok(/REQUIRED_REPLAYS/.test(verify) && /1000 by default/.test(verify) && /replay_count/.test(verify),
   'liveness stress is quantified and the completed count is returned');
ok(/two different problem sizes/.test(verify), 'liveness varies problem size (residency changes with it)');
ok(/timeout here is a FAILURE, never a skip/.test(verify), 'a hang cannot be silently skipped');
ok(/liveness.*only when SPECIALTY=distributed/.test(verify), 'return contract documents the field');

const bench = read('roles', 'mega', 'benchmark_engineer.md');
ok(/RESOLUTION FLOOR/.test(bench), 'baseline drift is recorded as a resolution floor');
ok(/alternately inside ONE process invocation/.test(bench), 'COMMANDMENT interleaves when drift is large');
ok(/paired\*\* delta/.test(bench), 'the reported delta is paired, not two independent medians');
ok(/RANK-MAX, not rank-mean/.test(bench), 'multi-rank metric is rank-max');

// The gates must stay OFF by default: a normal single-GPU run has no specialty match and no drift,
// so neither addition may change its behavior.
ok(/ONLY if `SPECIALTY` is `distributed`/.test(verify), 'liveness gate is opt-in, not always-on');
ok(/If `drift` exceeds/.test(bench), 'interleaving is conditional on measured drift');

// 3c. Author mode (0→1). The rank count must actually reach the two roles that branch on it, and
//     the author must be told the seed is the SCATTERED form — a from-scratch fused megakernel is
//     the single most likely way to burn the whole budget on a hang.
const plumbed = src.match(/GPUS_PER_JOB: String\(GPU_RESOURCE\.gpusPerJob\)/g) || [];
ok(plumbed.length >= 2, 'GPUS_PER_JOB reaches both author_engineer and tech_lead analyze');

const author = read('roles', 'mega', 'author_engineer.md');
ok(/`GPUS_PER_JOB`/.test(author), 'author_engineer documents GPUS_PER_JOB as an input');
ok(/GPUS_PER_JOB` > 1/.test(author), 'the distributed section is gated on rank count > 1');
ok(
  /NAIVE, SCATTERED, MULTI-LAUNCH form — never a fused megakernel/.test(author),
  'author seeds the scattered form and leaves fusion to the optimize specialty',
);
ok(/shape the seed to be\s*\*fusable\*/.test(author), 'the seed is required to stay legible to the optimizer');
ok(/timeout 600 bash/.test(author), 'every distributed GPU command carries a wall-clock timeout');
ok(/return `authored:false`/.test(author), 'a hang is a clean failure, not a stall');
ok(/launch all N ranks/.test(author), 'the correctness loop launches every rank');
ok(/distributed_fusion\.md` Levers 7–9/.test(author), 'author cross-references the deadlock levers');
ok(
  /`GPUS_PER_JOB == "1"` disqualifies the specialty outright/.test(lead),
  'tech_lead keys distributed eligibility off the resolved rank count',
);

// 4. Doctrine content. These are the load-bearing claims: each one, if absent, turns a fusion
//    that merely underperforms into one that hangs or silently reads stale data.
console.log(`\n# knowledge/${KNOWLEDGE} content`);
const doc = read('knowledge', KNOWLEDGE);
const required = [
  [/removes a \*wait\*, not[\s\S]{0,40}\*launch\*/, 'Lever 3: fusion pays for waits, not launches'],
  [/CTA-local phase pipeline/i, 'same-engine stages retain the CTA-local pipeline mechanism'],
  [/my shard is empty[\s\S]{0,160}never evidence that the whole grid drained/i,
    'local shard exhaustion cannot become a global drain'],
  [/static `max\(A\/f, B\/\(1-f\)\)`[\s\S]{0,180}cannot reject/i,
    'the static same-engine bound does not reject phase-staggered scheduling'],
  [/acquire fence/i, 'every wait pairs with an acquire fence'],
  [/rank-local edge[\s\S]{0,100}agent scope/i, 'rank-local publication uses agent scope'],
  [/peer-rank edge[\s\S]{0,100}system scope/i, 'cross-rank publication uses system scope'],
  [/write-through/i, 'write-through stores as the cheap alternative to a release fence'],
  [/every participant index must land inside/i, 'residency invariant is stated as an invariant'],
  [/coordinator → workers → coordinator/, 'acyclicity rule names the cycle'],
  [/Never clear an arrival counter/i, 'reset-free counter rule'],
  [/1000 CUDA-Graph replays/, 'liveness gate is quantified'],
  [/[Ii]nterleave the arms/, 'interleaved A/B is mandated'],
  [/spread/i, 'per-rep spread is required alongside the median'],
  [/rank-max/i, 'rank-max is named as the metric that matters'],
  [/no measured overlap change is suspicious/, 'unexplained latency wins are rejected'],
];
for (const [re, what] of required) ok(re.test(doc), what);
const partition = read('knowledge', 'resource_partition.md');
ok(/Same engine, static split/.test(partition) &&
   /does \*\*not\*\* disprove a\s+phase-staggered CTA pipeline/i.test(partition),
   'resource-partition same-engine math is scoped to static roles');
const megaLead = read('roles', 'mega', 'mega_search_lead.md');
ok(/without assuming that sharing a\s+compute engine makes overlap impossible/i.test(megaLead),
  'Mega search planning cannot dismiss local overlap from engine labels alone');

// Anti-patterns must stay measured, not folkloric: each carries a number so a future round can
// tell "we tried it and it cost X" apart from "someone thought this was a bad idea".
const antipatterns = [
  [/~89 ns/, 'work-stealing claim queue cost'],
  [/4\.4535[\s\S]{0,40}4\.4476/, 'grid-stride unroll null result'],
  [/\+2\.33 ms/, 'LDS-reduction dead end'],
  [/\+0\.411 ms/, 'release-fence cost'],
];
// NOTE: `876.6 µs` used to be asserted here as an anti-pattern measurement. It was
// removed from the card, and removing it is the point of the change, not a regression. That number
// is this operator's *diagnosis* — it appeared verbatim in both the knowledge card and the task
// roadmap, so a run could recite it without ever having measured anything. The distinction the
// remaining entries respect: a cost datum about a HARDWARE MECHANISM (what a release fence lowers
// to, what a claim queue costs per item) transfers to the next operator and is method. A latency
// measured on THIS operator's edges is its answer. See the "what this file deliberately does not
// contain" block in the card. The absence assertions below enforce that it stays out.
for (const [re, what] of antipatterns) ok(re.test(doc), `anti-pattern keeps its measurement: ${what}`);

// 5. Lessons from the completed fusion. Each of these cost a wrong number or a wasted
//    milestone to learn, and each is the kind of thing a fresh round re-derives the hard way.
console.log('\n# post-mortem doctrine');
const postmortem = [
  [/launch count is not an objective/i, 'launch count is explicitly rejected as a target'],
  [
    /who is the natural producer of this data/i,
    'operator-boundary check before fusing a phase in',
  ],
  [
    /state\s+the specific mechanism you are copying/i,
    'reference-design citations must name the mechanism, not "it is one kernel"',
  ],
  [
    /same tree with only the fusion flag toggled/i,
    'fusion attribution uses a same-tree flag toggle, not the frozen baseline',
  ],
  [
    /deletes the instrument that measured it/i,
    'fusing a phase deletes its instrument; a 0 reading means removal',
  ],
  [
    /aggregation unit/i,
    'in-kernel timers must declare their aggregation unit',
  ],
  [
    /exceeding the kernel's own wall time is an aggregation error/i,
    'the wait-vs-kernel-duration self-check is stated',
  ],
  [
    /falls back silently|fails its predicate falls back/i,
    'silently-falling-back opt-in paths are called out',
  ],
  [/refuse to report a number without that marker|void, not zero/i, 'path marker is required'],
  [
    /residency-side/i,
    'the overlap proof ceiling names the instrument that would close it',
  ],
];
for (const [re, what] of postmortem) ok(re.test(doc), what);

// 5b. Both of these were added because a planning role misapplied the doctrine above to close a
//     direction on evidence that could not support the conclusion. The prose is the fix; these
//     assertions are what stop the prose from being edited back out.
console.log('\n# decision-test guards');

// A small inter-kernel gap says the HOST issued the launches back-to-back. It says nothing about
// whether the GPU sat waiting. Rejecting a fusion on the gap inverts Lever 3.
ok(
  /small inter-kernel gap is NOT grounds to reject/i.test(doc),
  'the launch-count rule is explicitly stated to run in both directions',
);
ok(/12\.4 µs/.test(doc), 'the misapplication keeps the gap number that was wrongly used');
ok(
  /evidence about the host, not about whether the GPU is idle waiting/.test(doc),
  'the gap is characterised as host-side evidence',
);
ok(
  /no-payload control[\s\S]{0,600}exposed wait/.test(doc),
  'the replacement decision test names the no-payload control AND the exposed wait',
);

// A screen that cannot observe the dynamic schedule cannot refute a claim about it.
ok(
  /static-ISA screen is a screen for RESOURCE hypotheses only/i.test(doc),
  'the static-ISA screen declares its scope',
);
ok(
  /cannot refute a latency-hiding/i.test(doc),
  'the screen is explicitly barred from refuting scheduling-class directions',
);
// NB: match across a hard wrap — this file is wrapped at 100 cols and a plain-space regex here has
// silently broken once already when a paragraph was re-flowed.
ok(
  /4 of 4\s+paired reps/.test(doc),
  'the counterexample keeps its paired-rep evidence',
);
ok(
  /never as the\s+sole grounds to close a scheduling-class direction/.test(doc),
  'the permitted use (ordering, compile-legality) is separated from the barred use',
);

// ---------------------------------------------------------------------------------------------
// Answer containment.
//
// This block replaces two assertions that required the outturn TABLE — the per-guard fusion gains
// this operator actually achieved. Keeping it was defensible while the card was a post-mortem;
// it is indefensible while the open question is *whether the workflow can derive the fusion
// unaided*. A candidate that reproduces a result the knowledge card handed it demonstrates
// retrieval, not derivation, and the wave that produces it cannot be graded.
//
// The band ("2-5% is a good outcome") is retained as calibration and asserted below: it bounds
// effort without naming which edge, which guard, or how much. The line the card now draws:
//   KEEP  — the shape of the effect, the failure modes, the cost of a hardware mechanism
//   DROP  — which edges were missing here, what they measured, what the fix scored per guard
//
// These are ABSENCE assertions and they are deliberately brittle. If a future edit re-adds one of
// these numbers "for context", this test fails and the reviewer has to justify it in the open.
// ---------------------------------------------------------------------------------------------
const leaked = [
  [/\+4\.71%/, 'per-guard fusion outturn (large-uniform gain)'],
  [/876\.6/, "this operator's instrumented skew peer-wait"],
  [/\b191 µs\b/, "this operator's instrumented uniform peer-wait"],
  [/124\.7|651\.3/, "this operator's per-rank wait split"],
  [/2\.5205|1\.5568/, "this operator's no-payload control absolute timings"],
  [/GEMM1\s*(→|->)\s*GEMM2/, "this operator's named intra-rank readiness edge"],
  [/p2p_scatter_epilog/, "this operator's named cross-rank publish site"],
];
for (const [re, what] of leaked) {
  ok(!re.test(doc), `card does NOT pre-supply the answer: ${what}`);
}

// The band survives, because bounding effort is method and does not name the mechanism.
ok(/\b2[–-]5%/.test(doc), 'the 2-5% expectation band is retained as effort calibration');

// And the removal has to be self-documenting, or the next author re-adds the numbers in good faith.
ok(
  /deliberately does not contain/i.test(doc),
  'the card explains WHY the operator-specific results were removed',
);

// The method must actually replace what was subtracted -- otherwise this is deletion, not a
// redesign, and the Analyze phase is left with less than it had.
console.log('\n# method cards exist and are reachable');
const METHOD_CARDS = {
  'tile_task_graph.md': [
    [/output region overlaps/i, 'states the edge rule'],
    [/enforced_by|what enforces this edge/i, 'requires the "what enforces this edge today" column'],
    [/critical path/i, 'requires a critical path'],
    [/slack/i, 'requires slack'],
    [/launch_boundary/i, 'names the launch-boundary edge class'],
  ],
  'fusion_preconditions.md': [
    [/single wave|one wave/i, 'carries the one-wave disqualifier'],
    [/idle hardware/i, 'carries the idle-hardware condition'],
    [/smaller granularity|partial handover/i, 'carries the partial-handover condition'],
    [/DOES NOT PAY/, 'makes "does not pay" an emittable verdict'],
    [/ladder/i, 'requires cheaper levers to be ruled out first'],
  ],
  'resource_partition.md': [
    [/wave[- ]quantization|waves = ceil/i, 'covers the wave-quantization tail'],
    [/static/i, 'covers static vs dynamic scheduling'],
    [/critical path/i, 'requires predicting where the critical path moves'],
  ],
};
for (const [file, checks] of Object.entries(METHOD_CARDS)) {
  let text = '';
  try { text = read('knowledge', file); } catch { /* reported by the first check */ }
  ok(text.length > 0, `knowledge/${file} exists`);
  for (const [re, what] of checks) ok(re.test(text), `knowledge/${file} ${what}`);
  // A method card that quietly re-imports the answer defeats the whole exercise.
  ok(!/876\.6|4\.71%|p2p_scatter_epilog/.test(text), `knowledge/${file} carries no operator answer`);
}

// Reachability: the fusion card must route to them, or an engineer reading only the lever list
// never learns the graph exists.
for (const card of Object.keys(METHOD_CARDS)) {
  ok(doc.includes(card), `distributed_fusion.md points at ${card}`);
}

// The path-marker gate has to be enforced by the role, not just described in knowledge.
ok(/PATH MARKER/.test(bench), 'benchmark_engineer enforces the path marker for opt-in candidates');
ok(/void, not zero/.test(bench), 'a missing marker voids the result rather than scoring it');

ok(
  /Priority/.test(doc) && /no-payload control/.test(doc),
  'file ends with an actionable priority ordering',
);

console.log(
  failures === 0
    ? '\nPASS: `distributed` is reachable end-to-end and its doctrine carries its evidence.'
    : `\nFAILED: ${failures} assertion(s).`,
);
process.exit(failures === 0 ? 0 : 1);
