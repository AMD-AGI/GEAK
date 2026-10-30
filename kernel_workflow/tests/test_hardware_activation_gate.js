#!/usr/bin/env node
// ON-HARDWARE ACTIVATION GATE.
//
// The failure this closes, in full: a fused MegaMoE arm (the D1/D2 producer-readiness + combine
// consumer rewrite) was authored behind a default-OFF env switch and carried across eight rounds. It
// was green on every static screen the workflow knows how to run — py_compile, an OFF-path compile
// check, a CPU dry-run, and a COMPILE_ONLY ISA-distinctness hash that satisfied `artifact_distinct`.
// Its ON path had never once traced or launched on a card. The first time it was run on hardware, it
// crashed at JIT-trace time ("cannot evaluate dynamic 'Boolean' as Python bool during tracing"), a
// fault no static screen can see because the ON kernel variant only compiles when the switch is set at
// run time. The wave banked eight rounds of scaffolding whose perf-bearing path did not exist.
//
// Two things let that happen and both are fixed here:
//   1. `activation_confirmed:"yes"` (a host marker) and `artifact_distinct:"yes"` (a static ISA hash)
//      are both satisfiable WITHOUT a card. The commit/enabling gate trusted them. Now, on a wave that
//      sets `require_hardware_activation`, a committable/enabling candidate must ALSO report
//      `activation_on_hardware:"yes"` — the switched path traced+launched on a device this round.
//   2. working_kernel exempts itself from the no-improve AND no-evidence stops (a debug round is
//      non-improving by construction; a crashing-on-device terminal round is real progress). It must
//      NOT exempt itself from "nothing reached a card at all". A separate `noHardware` counter, live
//      under every objective, hard-stops the wave after MAX_NO_HARDWARE such rounds and surfaces it.
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_mega.js'), 'utf8');
const read = (...p) => fs.readFileSync(path.join(WF, ...p), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

// Execute the real functionalAcceptance / runsCleanly out of their REPLAY regions (same technique as
// test_objective_working_kernel.js), so the behaviour is tested, not just the source text.
const og = src.match(/\/\/ <<REPLAY:objective_gate>>([\s\S]*?)\/\/ <<\/REPLAY:objective_gate>>/);
const es = src.match(/\/\/ <<REPLAY:enabling_step>>([\s\S]*?)\/\/ <<\/REPLAY:enabling_step>>/);
if (!og || !es) { console.error('  FAIL: could not extract objective_gate / enabling_step regions'); process.exit(1); }
const { functionalAcceptance, runsCleanly } =
  // eslint-disable-next-line no-new-func
  new Function(`${og[1]}\n${es[1]}\nreturn { functionalAcceptance, runsCleanly };`)();

const good = { status: 'verified', correctness: 'pass', activation_confirmed: 'yes', liveness: 'pass' };

console.log('\n# behaviour: a switched candidate must have run on a card when the wave asks for it');
{
  const req = { requireHardwareActivation: true };
  ok(functionalAcceptance({ ...good, activation_on_hardware: 'yes' }, req).pass,
     'on-hardware yes + otherwise-clean passes the gate');
  const noHw = functionalAcceptance({ ...good, activation_on_hardware: 'no' }, req);
  ok(!noHw.pass && noHw.missing.some((m) => /HARDWARE/.test(m)),
     'on-hardware no is REJECTED, and the reason names hardware — a compile-only proof does not count');
  ok(!functionalAcceptance({ ...good, activation_on_hardware: 'unknown' }, req).pass,
     'unknown is rejected too — "we could not tell it ran on a card" is not "it ran on a card"');
  ok(!functionalAcceptance({ ...good }, req).pass,
     'a MISSING activation_on_hardware fails under the requirement — silence is not a device run, which ' +
     'is exactly the state the eight banked rounds were in');
  ok(!functionalAcceptance({ ...good, activation_on_hardware: 'n/a' }, req).pass,
     'n/a does not satisfy the requirement either: this task is all switched/JIT fusion, so "no switched ' +
     'path" is not an available answer for a candidate being committed on it');
}

console.log('\n# backward compatibility: the gate is inert unless the wave opts in');
{
  ok(functionalAcceptance({ ...good, activation_on_hardware: 'no' }, {}).pass,
     'with no requirement set, activation_on_hardware is ignored — a wave that did not ask for it is ' +
     'byte-identical to before');
  ok(runsCleanly(good),
     'runsCleanly with default (no requirements) still passes on the legacy fields — the existing ' +
     'working_kernel commit test keeps holding');
}

console.log('\n# wiring: the requirement is enforced in code, threaded, and not waived at commit');
{
  ok(/const REQUIRE_HW_ACTIVATION = String\(\s*A\.require_hardware_activation/.test(src),
     'REQUIRE_HW_ACTIVATION is parsed from the launch arg, default OFF');
  ok(/requireHardwareActivation: REQUIRE_HW_ACTIVATION/.test(src),
     'functionalRequirementsFor threads it — and, per its comment, NOT gated on strictPath and NOT ' +
     'emptied by runtimeCommit, so a still-slow terminal commit cannot waive it');
  ok(/if \(req\.requireHardwareActivation && s\(ver\.activation_on_hardware\) !== 'yes'\)/.test(src),
     'functionalAcceptance enforces it: yes is the only accepted value under the requirement');
  ok(/activation_on_hardware: \{ type: 'string' \}/.test(src),
     'the verify schema carries activation_on_hardware so the verifier is asked for it');
}

console.log('\n# wiring: the no-hardware stop is orthogonal, live under working_kernel, and hard-stops');
{
  ok(/let noHardware = 0;/.test(src) &&
     /const MAX_NO_HARDWARE = [^;]*A\.max_no_hardware[^;]*MAX_NO_IMPROVE[^;]*;/.test(src),
     'a dedicated noHardware counter exists (separate from noImprove/noEvidence); its cap defaults to ' +
     'MAX_NO_IMPROVE but is independently overridable via max_no_hardware (no-hardware cap decoupling)');
  ok(/const hardwareWasDue = !!shouldVerify;/.test(src) &&
     /if \(reachedHardware\) noHardware = 0;\s*\n\s*else if \(hardwareWasDue\) noHardware \+= 1;/.test(src),
     'it counts a miss only after a complete candidate reaches Verify; source-only authoring checkpoints are exempt');
  ok(/if \(noHardware >= MAX_NO_HARDWARE\) \{[\s\S]*?break;\s*\n\s*\}/.test(src),
     'at the cap it sets stopReason and BREAKS the loop — the operator chose hard-stop, not warn-and-continue');
  ok(/while \(dispatched < BUDGET &&\s*\(MODE === 'mega' \|\| WORKING_KERNEL \|\|/.test(src),
     'working_kernel still short-circuits SPEED/EVIDENCE stops; Mega also does so because one stalled ' +
     'candidate lane must not terminate the independent portfolio');
}

console.log('\n# the roles tell the agents to produce and demand on-hardware execution');
{
  const verify = read('roles', 'mega', 'verify_engineer.md');
  const eng = read('roles', 'mega', 'engineer.md');
  const lead = read('roles', 'mega', 'tech_lead.md');
  ok(/activation_on_hardware/.test(verify) && /ON HARDWARE/.test(verify),
     'verify_engineer.md defines activation_on_hardware and what makes it yes vs no');
  ok(/COMPILE_ONLY/.test(verify) && /trace-time/.test(verify),
     'and names the two things that fake it — a host marker and a COMPILE_ONLY ISA hash — and the ' +
     'trace-time fault they hide');
  ok(/ran ON A CARD this round/.test(eng),
     'engineer.md tells the enabling-step author its path marker must mean ran-on-a-card, not py_compiled');
  ok(/banking is PREPARATION, not progress/.test(lead),
     'tech_lead.md keeps lease-free authoring legal but says it is not progress until it runs on hardware');
  ok(/candidate `mega_e2e` vs baseline/.test(lead) || /candidate `mega_e2e` against the baseline/.test(lead),
     'and flips the first freed lease to the candidate-e2e win question, before attribution/overlap coverage');
}

console.log(failures === 0
  ? '\nPASS: a switched candidate that never traced+launched on a card cannot be committed, cannot count ' +
    'as on-device progress, and three such rounds hard-stop the wave — while a wave that did not opt in ' +
    'is unchanged.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
