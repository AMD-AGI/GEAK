#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';
// A supplied workload must be in a recognized form; otherwise the run refuses before any agent
// starts instead of silently ranking by unweighted geomean.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const text = fs.readFileSync(path.join(__dirname, '../src/kernel_lane/01b_args.js'), 'utf8');
const start = text.indexOf('const WORKLOAD_SPEC_PATH');
const end = text.indexOf('const EXTERNAL_OBJECTIVE');
assert.ok(start >= 0 && end > start, 'workload block located');
const block = text.slice(start, end);
const parse = (A, OP_SPEC = null) => new Function('A', 'OP_SPEC', block + '\nreturn HAS_WORKLOAD;')(A, OP_SPEC);

assert.equal(parse({}), false, 'no workload stays unweighted');
assert.equal(parse({}, {workload: []}), false, 'an empty list still means no workload');
assert.equal(parse({}, {workload: [{name: 'c1'}, {name: 'c2'}]}), true, 'a cases list is a workload');
assert.equal(parse({}, {workload: {kernels: [{name: 'k', cases: [{name: 'c1'}]}]}}), true,
  'a workload-v1 object with kernels[] is a workload');
assert.equal(parse({workload_spec_path: '/w.json'}), true, 'a workload path is a workload');
assert.equal(parse({}, {workload_path: '/w.json'}), true, 'an op_spec workload path is a workload');
assert.equal(parse({workload: [{name: 'c1'}]}), true, 'a top-level cases list is a workload');
assert.throws(() => parse({}, {workload: {kind: 'captured', cases: [{name: 'c1'}]}}),
  /unsupported workload shape/, 'a cases object without kernels[] is refused, not treated as unweighted');
assert.throws(() => parse({}, {workload: {kernels: []}}), /unsupported workload shape/,
  'an empty kernels[] is refused');
assert.throws(() => parse({workload: {cases: []}}), /unsupported workload shape/);
console.log('PASS: supplied workloads are recognized or refused; absent and empty lists stay unweighted.');
