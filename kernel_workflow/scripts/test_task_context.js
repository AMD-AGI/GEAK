#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// CPU-only prompt capture: execute the generated GEAK lane with stubbed agents and GPU work.
// Fixture references are strings only; no evidence files, credentials, or live results are read.
'use strict';

const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const lane = path.resolve(__dirname, '..', 'kernel_lane.js');
const body = fs.readFileSync(lane, 'utf8').replace(/^export const meta/m, 'const meta');
const hash = (text) => crypto.createHash('sha256').update(text).digest('hex');
const evidence = ['hardware', 'roofline', 'profile', 'compiler-ir', 'assembly'].map((name) =>
  `/cpu-fixture/evidence/${name}.md sha256=${hash(`fake ${name} evidence`)}`);
const skill = `/cpu-fixture/skills/kernel-hardware-evidence/SKILL.md sha256=${hash('fake skill')}`;
const task = [
  'Optimize the fixture kernel while preserving the public contract.',
  'COMMON_HARDWARE_CONTEXT_BEGIN',
  'Read the following evidence and skill before analysis, planning, or implementation.',
  ...evidence,
  skill,
  'COMMON_HARDWARE_CONTEXT_END',
].join('\n');

const evalDir = '/cpu-fixture/eval';
const perCase = (speedup) => [{ name: 'fixture', baseline_ms: 2, optimized_ms: 2 / speedup, speedup }];
const direction = (id, specialty) => ({ id, specialty, title: `Investigate ${id}`,
  prompt: `Measure ${id}`, focus_files: ['kernel.py'] });
const stop = { stop: true, directions: [], reasoning: 'fixture requests forced replan' };

function answer(label) {
  if (label === 'director:setup') return {
    eval_dir: evalDir, workspace: `${evalDir}/workspace`, kernel_name: 'fixture',
    device_gfx: 'gfx950', device_target: 'mi350x', physical_cu_count: 256,
    baseline_frozen: true, source_files: ['kernel.py'],
  };
  if (label === 'tech_lead:analyze') return { kernel_type: 'gemm', modifiable_files: ['kernel.py'] };
  if (label === 'benchmark_engineer') return {
    commandment_path: `${evalDir}/COMMANDMENT.md`, baseline_per_case: perCase(1),
    baseline_geomean_ms: 2, correctness_cmd: 'fixture correctness', benchmark_cmd: 'fixture benchmark',
  };
  if (label === 'profile_engineer:baseline' || label.startsWith('reprofile ')) {
    return { bottleneck: 'memory', top_opportunities: [], summary_path: `${evalDir}/profile.md` };
  }
  if (label === 'warm_start:resolve') return { read_reason: 'empty', candidates: [] };
  if (label.startsWith('clock ')) return { epoch: 1000 };
  if (label === 'tech_lead:plan r1') return stop;
  if (label === 'tech_lead:replan r1#1') return {
    stop: false, directions: [direction('memory', 'memory'), direction('compute', 'compute')],
  };
  if (label === 'tech_lead:plan r2') return {
    stop: false, directions: [direction('rewrite', 'deep_explore')],
  };
  if (/^(eng |deep )/.test(label)) return { status: 'ok', speedup_geomean: 1.5, measurement_valid: true };
  if (label.startsWith('verify ')) {
    const speedup = label === 'verify rewrite' ? 1.8 : 1.3;
    return { status: 'verified', correctness: 'pass', verified_geomean: speedup,
      verified_arithmetic: speedup, per_case: perCase(speedup) };
  }
  if (label === 'integrate r1') return { attempted: true, conclusion: 'no_gain' };
  if (label.startsWith('commit ')) return { committed: true };
  if (label.startsWith('tech_lead:memory ')) return { insights: [], ledger: [], bottleneck_now: 'memory' };
  if (label.startsWith('storage:reclaim')) return { ok: true };
  if (label === 'tech_lead:report') return { final_speedup_geomean: 1.8,
    report_path: `${evalDir}/report.md`, final_patch: `${evalDir}/final_patch.diff` };
  if (label === 'director:validate') return { director_verified_speedup_geomean: 1.8,
    validation_status: 'accepted', timing_basis: 'fixture' };
  if (label === 'update_experience') return { action: 'skipped', note: 'fixture' };
  if (label === 'kb:attest') return { ran: 0 };
  if (label === 'kb:write') return { written: false };
  throw new Error(`Unexpected agent call: ${label}`);
}

async function capture(source, extraArgs = {}) {
  const calls = [];
  const globals = {
    args: { kernel_path: '/cpu-fixture/task', workflow_dir: '/cpu-fixture/kernel_workflow',
      budget: 4, deadline_epoch: 100000, max_forced_replans: 1,
      agent_timeout_ms: 0, agent_retries: 1, ...extraArgs },
    phase: () => {}, log: () => {},
    workflow: async () => { throw new Error('Unexpected nested workflow'); },
    agent: async (prompt, options) => {
      calls.push({ label: options.label, prompt: String(prompt) });
      return JSON.parse(JSON.stringify(answer(options.label)));
    },
    parallel: async (thunks) => Promise.all(thunks.map((thunk) => thunk())),
    // Exercise the real engineer -> verifier stages; an empty pipeline would miss both roles.
    pipeline: async (items, ...stages) => Promise.all(items.map(async (item, index) => {
      let value = item;
      for (const stage of stages) value = await stage(value, item, index);
      return value;
    })),
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
  };
  const run = new Function(...Object.keys(globals), `return (async () => { ${source} })();`);
  const result = await run(...Object.values(globals));
  assert.equal(result.stopped_by, 'budget', 'fixture must exercise both optimization rounds');
  return calls;
}

(async () => {
  const calls = await capture(body, { task });
  const required = [
    'director:setup', 'tech_lead:analyze', 'benchmark_engineer', 'profile_engineer:baseline',
    'tech_lead:plan r1', 'tech_lead:replan r1#1', 'tech_lead:plan r2',
    'eng memory:memory', 'eng compute:compute', 'deep rewrite:deep_explore',
    'verify memory', 'verify compute', 'verify rewrite', 'integrate r1',
    'reprofile r1', 'reprofile r2', 'tech_lead:memory r1', 'tech_lead:report', 'director:validate',
  ];
  const recipients = calls.filter(({ prompt }) =>
    /^You are the \w+\. PHASE=/.test(prompt) || /^You are Engineer /.test(prompt));
  for (const label of required) {
    assert(recipients.some((call) => call.label === label), `Role was not captured: ${label}`);
  }
  for (const { label, prompt } of recipients) {
    assert.equal(prompt.split(`- TASK: ${task}`).length - 1, 1, `${label}: TASK must appear intact once`);
    for (const ref of [...evidence, skill]) {
      assert.equal(prompt.split(ref).length - 1, 1, `${label}: reference/hash missing or duplicated`);
    }
  }

  const absent = await capture(body);
  const empty = await capture(body, { task: '' });
  assert.deepEqual(empty, absent, 'Empty and absent task must produce identical prompt bytes');

  // test_greedy_parity.js separately checks absent-task prompts against the reviewed goldens.
  console.log(`PASS: intact TASK evidence/skill reaches ${recipients.length} captured role calls; absent/empty prompts match.`);
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
