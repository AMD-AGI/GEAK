// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';

// Render selected declarations and task expressions. No workflow or shell runs.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const vm = require('node:vm');

function region(source, start, end) {
  const begin = source.indexOf(start);
  if (begin < 0 || source.indexOf(start, begin + start.length) >= 0) throw new Error('ambiguous source start');
  const finish = source.indexOf(end, begin + start.length);
  if (finish < 0) throw new Error('source end missing');
  return source.slice(begin, finish);
}

function declaration(source, name) {
  const matches = source.match(new RegExp('^const ' + name + ' = .+;$', 'gm')) || [];
  if (matches.length !== 1) throw new Error('declaration missing');
  return matches[0];
}

function evaluate(program, data) {
  const context = vm.createContext({DATA: data}, {codeGeneration: {strings: false, wasm: false}});
  return new vm.Script(program).runInContext(context, {timeout: 1000});
}

async function main(input) {
  const sources = {};
  for (const name of ['kernel_workflow.js', 'kernel_lane.js']) {
    const raw = fs.readFileSync(path.join(input.source_root, name));
    if (crypto.createHash('sha256').update(raw).digest('hex') !== input.source_hashes[name]) {
      throw new Error('source changed');
    }
    sources[name] = raw.toString('utf8');
  }
  const dispatcher = sources['kernel_workflow.js'];
  const rootDeclarations = region(dispatcher, 'const A = args || {};',
    '\n// ===========================================================================\n// SINGLE-LANGUAGE PASS-THROUGH');
  const passthrough = region(dispatcher, "if (MODE === 'optimize' || MODE === 'author') {",
    '\n// ===========================================================================\n// From here on:');
  const laneRequest = await evaluate('(async () => { const args = DATA.root_args; let calls = 0;'
    + ' const phase = () => {}; const log = () => {};'
    + ' const workflow = async (script, args) => { if (++calls !== 1) throw new Error("multiple lanes"); return {...script, args}; };'
    + rootDeclarations + '\n' + passthrough + '\nthrow new Error("unsupported dispatcher mode"); })()', input);
  if (!laneRequest || laneRequest.scriptPath !== path.join(input.source_root, 'kernel_lane.js')) {
    throw new Error('alternate lane source');
  }

  const source = sources['kernel_lane.js'];
  const defaults = region(source, 'const A = args || {};', '\n// ---------------------------------------------------------------------------\n// DEEP-MODE');
  const schemaCode = declaration(source, 'obj') + '\n'
    + region(source, 'const WARMSTART_RESOLVE_SCHEMA = obj({', '\n// ---------------------------------------------------------------------------\n// Prompt helpers.');
  const clockCode = region(source, 'const DEADLINE_EPOCH = (() => {', '\nlet deadlineHit');
  const heldOut = declaration(source, 'HELD_OUT');
  const parts = {
    clock: region(source, '  const r = await agentT(\n    `Run EXACTLY this command and nothing else:',
      '\n  if (!r || !Number.isFinite(r.epoch))'),
    resolver: region(source, '    const localResolveCmd = KB_MODE', '\n    warm_start.read_reason = resolved.read_reason'),
    storage: region(source, '  const reclaimCmd =', '\n  // END STORAGE RECLAIM'),
    citation: region(source, '    await agentT(\n      `Run EXACTLY this command and nothing else. Do NOT edit any file.',
      '\n    log(`[kb] citation ledger filed'),
    writer: region(source, '  const kernelClass = (analysis && analysis.kernel_type)', '\n  const remoteWrite = (kb_written && kb_written.remote)'),
  };
  const prefix = 'const args = DATA.lane_args;\n' + defaults + '\n' + schemaCode + '\n' + clockCode + '\n' + heldOut + '\n';
  const data = {...input, lane_args: laneRequest.args};
  if (input.operation === 'inspect') {
    return evaluate('(() => {' + prefix + '\nreturn {'
      + 'mode: MODE, workflow_dir: WORKFLOW_DIR, exp_root: EXP_ROOT, kernel_path: KERNEL_PATH_ORIG,'
      + 'budget: BUDGET, language: TARGET_LANGUAGE, kb_artifacts: KB_ARTIFACTS_DIR, kb_store: KB_STORE_DIR,'
      + 'kb_mode: KB_MODE, kb_remote: KB_REMOTE, version: KB_FRAMEWORK_VERSION, warm_start_enabled: WARM_START_ON, writer_enabled: KB_WRITE_OK,'
      + 'deadline_enabled: DEADLINE_EPOCH > 0, held_out: HELD_OUT, has_workload: HAS_WORKLOAD,'
      + 'dtype: OP_SPEC.dtype ? String(OP_SPEC.dtype) : "", lane_request: DATA.lane_request}; })()', {...data, lane_request: laneRequest});
  }
  if (!Object.hasOwn(parts, input.site)) throw new Error('unknown helper site');
  const dynamic = input.dynamic || {};
  const context = 'const setup = DATA.setup; const D = DATA.dynamic;'
    + ' const EVAL_DIR = setup.eval_dir;'
    + " const KERNEL_NAME = String(setup.kernel_name || '').replace(/_task$/, '') || setup.kernel_name;"
    + ' const GFX = D.gfx; const round = D.round; const tag = D.tag;'
    + ' const citations = D.citations; const analysis = {kernel_type: D.kernel_class};'
    + ' const finalPrimary = D.speedup; const BASELINE_GEOMEAN_MS = D.baseline_ms;'
    + ' const report = {final_patch: D.patch, report_path: D.report};'
    + ' const warm_start = {direction: D.direction, exp_dir: D.parent};'
    + ' const bestPerCase = (D.case_names || []).map(name => ({name}));'
    + ' let captured; let kb_written;'
    + ' const agentT = async (task, options) => { if (captured) throw new Error("multiple helper calls");'
    + ' captured = {task, options}; return captured; };';
  return await evaluate('(async () => {' + prefix + context + '\n' + parts[input.site]
    + '\nif (!captured) throw new Error("helper call missing"); return captured; })()', {...data, dynamic});
}

main(JSON.parse(fs.readFileSync(0, 'utf8'))).then(result => {
  process.stdout.write(JSON.stringify(result));
}).catch(() => {
  process.stderr.write('The public workflow source layout is unsupported.\n');
  process.exitCode = 1;
});
