#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Deterministic kernel-lane harness: runs kernel_lane.js with synthetic role responses (no agents/GPU).
'use strict';

const fs = require('fs');
const path = require('path');

const wfDir = path.resolve(__dirname, '..');
const geakRoot = path.resolve(wfDir, '..');
const body = fs.readFileSync(path.join(wfDir, 'kernel_lane.js'), 'utf8')
  .replace(/^export const meta/m, 'const meta');

async function run(deviceGfx, expectedGfx = '', extras = {}) {
  const prompts = [];
  const logs = [];
  const phases = [];
  const {
    deviceName = '',
    deviceTarget,
    physicalCuCount,
    omitGfx = false,
    omitTarget = false,
    expectedTarget = '',
    expectedDeviceName = '',
    expectedPhysicalCuCount = 0,
    budget = 0,
    planDirections = null,
  } = extras;
  const globals = {
    args: {
      kernel_path: '/tmp/kernel',
      state: extras.state,
      mode: extras.mode || 'optimize',
      workflow_dir: wfDir,
      budget,
      expected_gfx: expectedGfx,
      expected_target: expectedTarget,
      expected_device_name: expectedDeviceName,
      expected_physical_cu_count: expectedPhysicalCuCount,
      use_expert_skills: 'false',
      ...(extras.language ? {target_language: extras.language} : {}),
      ...(extras.useGluon === undefined ? {} : {use_gluon_skill: extras.useGluon}),
      use_learned_kb: 'true',
    },
    phase: (name) => phases.push(name),
    log: (message) => logs.push(String(message)),
    workflow: async () => null,
    parallel: async (items) => Promise.all(items.map((item) => item())),
    pipeline: async (items, ...stages) => Promise.all(items.map(async (item) => {
      let result = item;
      for (const stage of stages) result = await stage(result);
      return result;
    })),
    budget: { total: null, spent: () => 0, remaining: () => budget },
    agent: async (prompt, options) => {
      prompts.push({ label: options.label, prompt, schema: options.schema });
      if (extras.agentResponses && Object.prototype.hasOwnProperty.call(extras.agentResponses, options.label)) {
        return extras.agentResponses[options.label];
      }
      if (extras.agentResponse) {
        const value = extras.agentResponse(options.label, prompt);
        if (value !== undefined) return value;
      }
      if (options.label === 'director:setup') {
        const setup = {
          eval_dir: '/tmp/eval',
          workspace: '/tmp/eval/workspace',
          baseline_dir: '/tmp/eval/baseline',
          kernel_name: 'kernel',
          device_name: deviceName || 'fixture',
          baseline_frozen: true,
          source_files: ['kernel.py'],
        };
        if (!omitGfx) setup.device_gfx = deviceGfx;
        if (!omitTarget) {
          setup.device_target = deviceTarget != null
            ? deviceTarget
            : 'unknown';
        }
        setup.physical_cu_count = physicalCuCount != null
          ? physicalCuCount
          : (deviceGfx === 'gfx1201' ? 64 : 256);
        return setup;
      }
      if (options.label === 'tech_lead:analyze') {
        return {
          kernel_type: 'triton',
          modifiable_files: ['kernel.py'],
          kk_operator: 'dense_gemm',
          kk_language: extras.sourceLanguage || 'triton',
          kk_refs: ['optimization/mfma_scheduling.md'],
        };
      }
      if (options.label === 'benchmark_engineer') {
        return {
          commandment_path: '/tmp/eval/COMMANDMENT.md',
          baseline_per_case: [{ name: 'x', latency_ms: 1 }],
          baseline_geomean_ms: 1,
        };
      }
      if (options.label === 'profile_engineer:baseline') {
        return { bottleneck: 'unknown', device: deviceGfx, summary_path: '' };
      }
      if (options.label.startsWith('tech_lead:plan')) {
        return { stop: false, directions: planDirections || [] };
      }
      if (options.label === 'update_experience') {
        return { action: 'skipped', card_path: '', key: '', note: 'test' };
      }
      return null;
    },
  };
  const fn = new Function(
    ...Object.keys(globals),
    `return (async () => { ${body} })();`,
  );
  return { result: await fn(...Object.values(globals)), prompts, logs, phases };
}

module.exports = {run};
