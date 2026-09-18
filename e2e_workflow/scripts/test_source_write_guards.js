#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Execute the shipped write blocks with fake agents. Source-bearing runs must
// reach no shared writer, even when every ordinary success gate is satisfied.
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '../..');
const e2e = fs.readFileSync(path.join(root, 'e2e_workflow/e2e_workflow.js'), 'utf8');
const lane = fs.readFileSync(path.join(root, 'kernel_workflow/kernel_lane.js'), 'utf8');
const dispatcher = fs.readFileSync(path.join(root, 'kernel_workflow/kernel_workflow.js'), 'utf8');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

function between(source, start, end) {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first + start.length);
  assert(first >= 0 && last > first, `missing production block: ${start}`);
  return source.slice(first, last);
}

function block(source, start, indent = '') {
  return between(source, start, '\n' + indent + '}\n') + '\n' + indent + '}\n';
}

function context(request) {
  const calls = [];
  const logs = [];
  const agent = async (prompt, options) => {
    calls.push({ prompt, label: options.label });
    return { results: [], rungs: [], insights: ['local-insight'] };
  };
  const op = { op: 'example', backend: 'triton', isolated_speedup: 1.2,
    engaged: true, artifact: '/run/table.json', session_id: 'prior-record', source: 'recall' };
  return {
    calls, logs, BASELINE_SOURCE_REQUEST: request,
    safeAgent: agent, agentT: agent, roleAgent: (_role, _phase, intro) => intro,
    log: message => logs.push(message), shq: JSON.stringify, obj: value => value,
    arrObj: {}, arrStr: {}, want: () => true,
    E2E_STORE_SCRIPT: '/store.py', KB_ENV_PRELUDE: '', kbIdentityFlags: () => '',
    kbPlaneFlags: () => '', describeOverrides: () => '',
    KB_ATTEST_OUTCOME: { adopted: 'validated' }, BASELINE_TPUT: 100,
    verdicts: [{ session_id: 'prior-record', outcome: 'adopted', measured_tok_s: 110 }],
    KB_DIMS: { gfx: 'gfx942', precision: 'bf16', framework_version: 'test' },
    tunedOps: [op], KERNEL_WF_DIR: '/kernel', E2E_KB_PLANE: 'both',
    E2E_KB_STORE_DIR: '/shared/store', KB_ARTIFACTS_DIR: '/shared/artifacts',
    tuning: { gate: 'accepted', apply_env: 'TUNING=1' }, tuneOk: true,
    BACKEND: 'sglang', EVAL_DIR: '/run', WORKFLOW_DIR: '/workflow', MODEL_NAME: 'model',
    milestone: 1, history: { ledger: [], insights: [], milestones: [] }, cands: [op],
    profile: {}, ANALYSIS_SKILL_INPUTS: {}, EXPERIENCE_SCHEMA: {},
    allAccepted: [op], acceptedHeads: [op], acceptedKernels: [],
    validatedOk: true, validation: { director_verified_throughput_tok_s: 110,
      validation_status: 'accepted', output_parity: 'pass' }, finalTput: 110, finalSpeedup: 1.1,
    wfReturn: { validation_status: 'validated_win', throughput_speedup: 1.1, final_throughput_tok_s: 110 },
    E2E_WARM_START_ON: true, curFlags: 'accepted', INIT_FLAGS: '', curEnv: '', INIT_ENV: '',
    FAST_MODE: false, DEEP_MODE: false,
    warm_start: { candidates: [{ outcome: 'validated', session_id: 'record',
      rank: 1, verified_speedup: 1.2, status: 'adopted' }], plane: 'remote' },
    KB_STORE_DIR: '/shared/store', EXPERIENCE_STORE: '/experience.py',
    KERNEL_NAME: 'kernel', TARGET_LANGUAGE: 'triton', GFX: 'gfx942', KB_VERSION_FLAG: '',
    kbGate: '', UPDATE_EXPERIENCE_ON: true, kbAccepted: true, finalPrimary: 1.2,
    learned_card: null, LEARNED_DIR: '/shared/learned', KERNEL_KNOWLEDGE_DIR: '/knowledge',
    MODE: 'optimize', analysis: {}, profileSummary: {}, report: {},
    citations: [{ card: 'prior-card', outcome: 'validated' }], HELD_OUT: false,
    UPDATE_EXPERIENCE_SCHEMA: {}, KB_ROOT_OK: true, kebab: String, HAS_WORKLOAD: false,
    bestPerCase: [], KB_REMOTE: 'auto', KB_MODE: 'store', BASELINE_GEOMEAN_MS: 1,
    OP_SPEC: {}, WARMSTART_WRITE_SCHEMA: {}, winner: { speedup: 1.2 }, laneRows: [], rep: null,
  };
}

const cases = [
  ['e2e attestation', between(e2e, 'const configHalfOnly =', '      const allVerdicts ='), 'kb:attest'],
  ['tuning attestation', between(e2e, 'const tuningAttestable =', '\n  if (tuneOk) {'), 'kernel-kb:attest-tuned'],
  ['tuning write', between(e2e, 'const kernelKbOps =', '    await requireE2EValidationCheckpoint'), 'kernel-kb:write-tuned'],
  ['milestone learned card', between(e2e, 'const exp = BASELINE_SOURCE_REQUEST', '  history.milestones.push'), 'architect:experience m1'],
  ['final learned card', block(e2e, 'if (!BASELINE_SOURCE_REQUEST && allAccepted.length)', '  '), 'architect:experience final'],
  ['deployment write', between(e2e, 'const kbNoWinVerdict =', '\nreturn wfReturn;'), 'kb:write'],
  ['lane attestation', 'const benched = warm_start.candidates;\n' +
    block(lane, 'if (!BASELINE_SOURCE_REQUEST && benched.length)', '      '), 'kb:attest'],
  ['lane learned card', block(lane, 'if (!BASELINE_SOURCE_REQUEST && !kbGate'), 'update_experience'],
  ['lane citation ledger', block(lane, 'if (!BASELINE_SOURCE_REQUEST && citations.length'), 'kb:cite'],
  ['lane write', lane.match(/const KB_WRITE_OK = [^\n]+/)[0] + '\n' +
    between(lane, 'let kb_written = null;', '// finalPrimary is the total'), 'kb:write'],
  ['bakeoff learned card', block(dispatcher, 'if (!BASELINE_SOURCE_REQUEST && winner && winner.speedup'), 'update_experience'],
];

async function main() {
  for (const [name, code, label] of cases) {
    for (const request of ['', '/run/source_requests/accepted.json']) {
      const env = context(request);
      const run = new AsyncFunction(...Object.keys(env), code);
      await run(...Object.values(env));
      assert.deepStrictEqual(env.calls.map(call => call.label), request ? [] : [label],
        `${name}: ${request ? 'source-bound write leaked' : 'ordinary write changed'}; ${env.logs.join('; ')}`);
      assert.strictEqual(env.wfReturn.final_throughput_tok_s, 110, 'local result retained');
      assert.strictEqual(env.tunedOps.length, 1, 'local tuned op retained');
      assert.strictEqual(env.citations.length, 1, 'local citations retained');
    }
    console.log(`PASS ${name}: source-bound skipped, ordinary path unchanged`);
  }

  const argsCode = between(e2e, 'const laneArgs = (wfArgs) =>', '// EXP_ROOT =');
  const makeArgs = new Function('BASELINE_SOURCE_REQUEST', 'LANE_USE_LEARNED_KB', argsCode + '\nreturn laneArgs;');
  const input = { kernel_path: '/task', warm_start: 'reference', kb_store_dir: '/shared/store' };
  assert.deepStrictEqual(makeArgs('', 'false')(input), { use_learned_kb: 'false', ...input });
  const staged = '/run/source_requests/exact.json';
  const child = makeArgs(staged, 'false')({ ...input, baseline_source_request_path: 'stale' });
  assert.strictEqual(child.baseline_source_request_path, staged);
  assert.strictEqual(child.warm_start, input.warm_start);
  assert.strictEqual(child.kb_store_dir, input.kb_store_dir);
  const spread = dispatcher.match(/\.\.\.\(BASELINE_SOURCE_REQUEST \? \{ baseline_source_request_path: BASELINE_SOURCE_REQUEST \} : \{\}\)/);
  assert(spread, 'bakeoff lanes must inherit the same source binding');
  const forward = new Function('BASELINE_SOURCE_REQUEST', `return ({ ${spread[0]} });`);
  assert.deepStrictEqual(forward(''), {});
  assert.deepStrictEqual(forward(staged), { baseline_source_request_path: staged });
  assert(dispatcher.includes('{ ...A, workflow_dir: WORKFLOW_DIR }'), 'single-language passthrough keeps source binding');
  console.log('PASS source binding reaches nested lanes without changing warm-start read settings');

  for (const source of [e2e, lane, dispatcher]) {
    new AsyncFunction(source.replace(/^export const meta/m, 'const meta'));
    const helper = source.match(/function sourceContractBlock\(role\) \{[\s\S]*?\n\}/)[0];
    const makeBlock = new Function('BASELINE_SOURCE_REQUEST', 'EVAL_DIR_OVERRIDE', helper + '\nreturn sourceContractBlock;');
    assert.strictEqual(makeBlock('', '/run')('director'), '');
    assert(makeBlock(staged, '/run')('director').includes('learned cards'));
  }
  console.log('PASS all workflow scripts compile; source-free prompts remain unchanged');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
