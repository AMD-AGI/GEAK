#!/usr/bin/env node
// Regression guard for KernelFusion decode coverage (no GPU, no model needed).
//
// Incident: a vLLM run passed semantics_shape_capture=false to skip the sglang-only Shape replay.
// That same switch gated PHASE=complete_table, which on vLLM is the CUDA-graph-off layer-boundary
// donor, so the fusion tables held prefill only and the run said so in one log line.
//
// Invariants under test, on the ACTUAL code extracted from the workflow script:
//   (a) complete_table is gated per backend: semantics_shape_capture on sglang, semantics_layer_donor
//       on vllm, and neither switch reaches the other backend;
//   (b) the complete_table call site uses that per-backend gate;
//   (c) semanticsPhaseCoverage never reports "covered" without evidence: a missing decode is false,
//       an unreported coverage is null (unknown), a failed table covers nothing;
//   (d) the coverage verdict is folded into the fusion disposition as coverage_gaps.
//
// Run:  node e2e_workflow/scripts/test_fusion_decode_coverage_switch.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');            // .../GEAK
const FILE = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

console.log(`\n# ${path.relative(ROOT, FILE)}`);
const src = fs.readFileSync(FILE, 'utf8');

// 1) The switch blocks, probed with controlled module-scope deps.
const shape = src.match(/const SEMANTICS_SHAPE_CAPTURE_ON = [\s\S]*?const FUSION_SHAPE_CAPTURE_ON = [\s\S]*?=== 'true';/);
const donor = src.match(/const FUSION_LAYER_DONOR_ON = [\s\S]*?const FUSION_COMPLETE_TABLE_SWITCH = [^\n]*\n/);
const helper = src.match(/function semanticsPhaseCoverage\([\s\S]*?\n\}\n/);
ok(!!shape, 'semantics_shape_capture switch block found');
ok(!!donor, 'per-backend complete_table switch block found');
ok(!!helper, 'semanticsPhaseCoverage helper found');

if (shape && donor) {
  const make = new Function('A', 'BACKEND',
    shape[0] + '\n' + donor[0] +
    '\nreturn { FUSION_COMPLETE_TABLE_ON, FUSION_COMPLETE_TABLE_SWITCH };');
  const gate = (backend, a) => make(a, backend);

  ok(gate('vllm', {}).FUSION_COMPLETE_TABLE_ON === true, 'vllm: decode donor ON by default');
  ok(gate('vllm', { semantics_shape_capture: 'false' }).FUSION_COMPLETE_TABLE_ON === true,
    'vllm: semantics_shape_capture=false does NOT turn the donor off (the incident)');
  ok(gate('vllm', { semantics_layer_donor: 'false' }).FUSION_COMPLETE_TABLE_ON === false,
    'vllm: semantics_layer_donor=false turns the donor off');
  ok(gate('vllm', {}).FUSION_COMPLETE_TABLE_SWITCH === 'semantics_layer_donor',
    'vllm: the switch named in warnings is semantics_layer_donor');

  ok(gate('sglang', {}).FUSION_COMPLETE_TABLE_ON === true, 'sglang: Shape replay ON by default');
  ok(gate('sglang', { semantics_shape_capture: 'false' }).FUSION_COMPLETE_TABLE_ON === false,
    'sglang: semantics_shape_capture=false turns the replay off');
  ok(gate('sglang', { semantics_layer_donor: 'false' }).FUSION_COMPLETE_TABLE_ON === true,
    'sglang: semantics_layer_donor has no effect');
  ok(gate('sglang', {}).FUSION_COMPLETE_TABLE_SWITCH === 'semantics_shape_capture',
    'sglang: the switch named in warnings is semantics_shape_capture');
}

// 2) The complete_table call site uses the per-backend gate, not the sglang switch.
const site = src.match(/if \((\w+) && semantics &&\s*\n\s*semantics\.status !== 'failed'[\s\S]{0,400}?'complete_table'/);
ok(!!site, 'complete_table call site found');
if (site) ok(site[1] === 'FUSION_COMPLETE_TABLE_ON', `complete_table gated by FUSION_COMPLETE_TABLE_ON (found ${site[1]})`);

// 3) Coverage verdicts.
if (helper) {
  const cov = new Function('FUSION_COMPLETE_TABLE_SWITCH',
    helper[0] + '\nreturn semanticsPhaseCoverage;')('semantics_layer_donor');
  const table = { semantic_table_json: '/t.json' };

  const both = cov({ ...table, status: 'pass', phases_in_tables: ['decode', 'prefill'], missing_required_phases: [] }, true);
  ok(both.decode_covered === true, 'pass with decode in tables -> covered');

  const prefillOnly = cov({ ...table, status: 'partial', phases_in_tables: ['prefill'], missing_required_phases: ['decode'] }, false);
  ok(prefillOnly.decode_covered === false, 'partial with decode missing -> NOT covered');
  ok(prefillOnly.decode_completion_enabled === false && prefillOnly.decode_completion_switch === 'semantics_layer_donor',
    'records whether completion ran and which switch controls it');

  const unreported = cov({ ...table, status: 'pass' }, true);
  ok(unreported.decode_covered === null, 'coverage not reported -> unknown (null), never covered');

  const tablesOnly = cov({ ...table, status: 'pass', phases_in_tables: ['prefill'] }, true);
  ok(tablesOnly.decode_covered === false, 'phases_in_tables without decode -> NOT covered');

  ok(cov({ status: 'failed', notes: 'x' }, true).decode_covered === false, 'failed table -> covers nothing');
  ok(cov(null, true).decode_covered === false && cov(null, true).semantics_status === 'missing',
    'no mapper result -> covers nothing');
}

// 3b) complete_table is not cut off by the generic 120-minute hung-agent guard: the donor/replay
//     is a full model load plus a capture window, and a timed-out agent keeps running unread.
//     The no-guard default is vLLM-only; sglang keeps its original call (no timeoutMs key at all).
const tmo = src.match(/const FUSION_COMPLETE_TABLE_TIMEOUT_MS = [\s\S]*?;\n/);
ok(!!tmo, 'complete_table timeout block found');
if (tmo) {
  const t = (backend, a) => new Function('A', 'BACKEND', tmo[0] + 'return FUSION_COMPLETE_TABLE_TIMEOUT_MS;')(a, backend);
  ok(t('vllm', {}) === 3 * 3600 * 1000, 'vllm: complete_table defaults to 3h, above the 120min generic guard');
  ok(t('vllm', { semantics_completion_timeout_ms: 0 }) === 0, 'vllm: 0 still selects the outer workflow bound only');
  ok(t('sglang', {}) === null, 'sglang: no default (generic guard, call unchanged)');
  ok(t('sglang', { semantics_completion_timeout_ms: '0' }) === 0, 'sglang: explicit arg still honoured');
  ok(t('vllm', { semantics_completion_timeout_ms: 7200000 }) === 7200000, 'vllm: explicit arg overrides the default');
}
ok(/label: 'semantics-mapper:shape-completion', schema: SEMANTICS_SCHEMA,\s*\n\s*\.\.\.\(FUSION_COMPLETE_TABLE_TIMEOUT_MS != null \? \{ timeoutMs: FUSION_COMPLETE_TABLE_TIMEOUT_MS \} : \{\}\) \}/.test(src),
  'complete_table call adds timeoutMs only when one is set');
ok(/fusionCompletionOutcome = completed \? 'returned' : 'no_result'/.test(src) &&
   /fusionCoverage\.decode_completion_outcome = fusionCompletionOutcome/.test(src),
  'a complete_table that returned nothing is recorded as no_result in the coverage verdict');
const bounded = src.match(/function agentBounded[\s\S]*?\n\}\n/);
if (bounded) {
  // No time budget unless a probe below passes one.
  const mk = (wps, AT, atf, ag, lg, st, TB = null, FPS = false, rem = () => Infinity, FR = 0) => new Function(
    'withProcessSafety', 'AGENT_TIMEOUT_MS', 'agentTimeoutFor', 'agent', 'log', 'setTimeout',
    'TIME_BUDGET_MS', 'FINAL_PHASE_STARTED', 'remainingMs', 'FINAL_RESERVE_MS',
    bounded[0] + '\nreturn agentBounded;')(wps, AT, atf, ag, lg, st, TB, FPS, rem, FR);
  let armed = 0;
  const ab = mk((p) => p, 7200000, () => 7200000, async () => 'r', () => {}, () => { armed++; return 0; });
  ab('p', { timeoutMs: 0 });
  ok(armed === 0, 'timeoutMs: 0 arms no hung-agent guard');
  ab('p', {});
  ok(armed === 1, 'default call still arms the generic guard');
  // A longer explicit bound must be armed as given, not clipped back to AGENT_TIMEOUT_MS.
  let armedMs = null;
  const ab2 = mk((p) => p, 7200000, () => 7200000, async () => 'r', () => {},
    (fn, ms) => { armedMs = ms; return 0; });
  ab2('p', { timeoutMs: 3 * 3600 * 1000 });
  ok(armedMs === 3 * 3600 * 1000, `timeoutMs above AGENT_TIMEOUT_MS is armed as given (got ${armedMs})`);
  ab2('p', {});
  ok(armedMs === 7200000, 'no timeoutMs -> generic 120min guard');
  // Under time_budget_s a longer bound still yields to remaining-minus-reserve.
  const ab3 = mk((p) => p, 7200000, () => 1800000, async () => 'r', () => {},
    (fn, ms) => { armedMs = ms; return 0; }, 12 * 3600 * 1000, false, () => 90 * 60000, 60 * 60000);
  ab3('p', { timeoutMs: 3 * 3600 * 1000 });
  ok(armedMs === 30 * 60000, `with 90min left and a 60min reserve, a 3h request is held to 30min (got ${armedMs})`);
}

// 4) The verdict reaches the disposition (and therefore the report).
ok(/fusionCoverage = semanticsPhaseCoverage\(semantics, FUSION_COMPLETE_TABLE_ON\)/.test(src),
  'KernelFusion computes coverage from the final semantics result');
ok(/phase_coverage: fusionCoverage,\s*\n\s*coverage_gaps:/.test(src),
  'coverage is folded into fusionDisposition as phase_coverage + coverage_gaps');
ok(/phases_in_tables: arrStr, missing_required_phases: arrStr/.test(src),
  'SEMANTICS_SCHEMA accepts the mapper coverage fields');

console.log(failures ? `\n${failures} FAILURE(S)` : '\nALL OK');
process.exit(failures ? 1 : 0);
