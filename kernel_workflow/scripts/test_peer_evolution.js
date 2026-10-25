#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Real archive, Git materializer and native source helper; no model or GPU calls.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const ROOT = path.resolve(__dirname, '../..');
const source = fs.readFileSync(path.join(ROOT, 'kernel_workflow/src/kernel_lane/14a_evolution.js'), 'utf8');

(async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'geak-peer-evolution-'));
  try {
    const fixture = spawnSync('python3', ['-B', path.join(__dirname, 'test_peer_sources.py'), '--fixture', temp],
      {encoding: 'utf8', timeout: 30000});
    assert.equal(fixture.status, 0, fixture.stderr);
    const setup = JSON.parse(fixture.stdout);
    const evalDir = path.join(temp, 'eval');
    fs.mkdirSync(evalDir);
    const calls = [], logs = [], start = {cumulative: 1, history: {rounds: []}, deadline_epoch: 12345};
    const original = JSON.stringify(start), canonical = fs.readFileSync(path.join(setup.base, 'kernel_src/module.py'));
    const globals = {
      EVAL_DIR: evalDir, CANONICAL: setup.base, WORKFLOW_DIR: path.join(ROOT, 'kernel_workflow'),
      A: {external_result_command: ['python3', 'native-reader.py', '--task-root', setup.task_root, 'read']},
      EVOLUTION_PEERS: [setup.peer], EXTERNAL_OBJECTIVE: {kind: 'external_primary'},
      primSpeedup: value => value?.primary_speedup, says: (value, expected) => value === expected,
      log: message => logs.push(message),
      hostRun: async (cmd, args, options = {}) => {
        calls.push([cmd, ...args]);
        const result = spawnSync(cmd, args, {input: options.input, encoding: 'utf8', timeout: 30000,
          env: {...process.env, PYTHONDONTWRITEBYTECODE: '1'}});
        return result.status === 0 ? result.stdout : null;
      },
    };
    const make = new Function(...Object.keys(globals), source + '\nreturn createEvolution;')(...Object.values(globals));
    const evo = await make(start);
    await evo.snapshotPeers(1);
    const first = evo.planInputs(), offer = first.EVOLUTION_PEER_OFFERS[0];
    assert.equal(first.EVOLUTION_PEER_OFFERS.length, 2);
    assert.ok(offer && first.EVOLUTION_PEER_SNAPSHOT_SHA256);
    assert.equal(fs.readdirSync(path.join(evalDir, 'evolution/patches')).length, 0, 'snapshot copies no patches');
    assert.equal(evo.entries.length, 0);
    // A replan sees the exact frozen archive, even if the publisher updates it.
    const peerArchive = path.join(setup.peer.root, 'archive.json');
    const published = fs.readFileSync(peerArchive);
    fs.writeFileSync(peerArchive, '{partial next version');
    fs.unlinkSync(path.join(setup.peer.root, 'patches/r1_d1.diff'));
    assert.deepEqual(evo.planInputs(), first);
    const [direction] = await evo.prepare([{parent_id: offer.id, donor_ids: [offer.id],
      out_dir: path.join(evalDir, 'round_1/engineer_0')}], 1);
    assert.equal(direction.parent_id, offer.id);
    assert.deepEqual(direction.donor_ids, [offer.id]);
    assert.equal(evo.entries.length, 1);
    assert.equal(fs.readdirSync(path.join(evalDir, 'evolution/patches')).length, 1, 'unselected peer patch stays remote');
    const imported = evo.entries[0];
    assert.equal(imported.verified, false);
    assert.equal(imported.speedup, null);
    assert.equal(imported.origin.id, 'r1_d0');
    assert.equal(await evo.selectBest(offer.id), false, 'foreign speedup never promotes an import');
    assert.equal(evo.planInputs().EVOLUTION_BEST_ID, 'ORIGINAL');
    await evo.prepareWorkspace(direction);
    const workspace = path.join(direction.out_dir, 'workspace/kernel_src/module.py');
    assert.equal(fs.readFileSync(workspace, 'utf8'), '# peer implementation; never imported\n');
    assert.equal(fs.statSync(workspace).mode & 0o111, 0o111, 'native patch retains executable mode');
    const imports = () => calls.filter(row => row[0] === 'python3' && row[2] === 'import').length;
    const before = imports();
    await evo.prepare([{parent_id: offer.id}], 2);
    assert.equal(imports(), before, 'retained import is reused privately');
    // Invalid selected parents and donors fail their directions without fallback.
    for (const selection of [{parent_id: first.EVOLUTION_PEER_OFFERS[1].id},
      {parent_id: 'peer_missing'}, {parent_id: 'ORIGINAL', donor_ids: ['peer_missing']}]) {
      const [bad] = await evo.prepare([selection], 2);
      assert.match(bad.evolution_context.peer_source_error, /selected peer source/);
      if (selection.parent_id === 'peer_missing') assert.equal(bad.parent_id, 'peer_missing');
      const count = calls.length;
      await assert.rejects(evo.prepareWorkspace(bad), /selected peer source/);
      assert.equal(calls.length, count, 'failed import launches no workspace or engineer command');
    }
    const importedBytes = fs.readFileSync(imported.patch_ref.path);
    fs.appendFileSync(imported.patch_ref.path, 'changed private donor');
    const [changed] = await evo.prepare([{parent_id: 'ORIGINAL', donor_ids: [offer.id]}], 2);
    await assert.rejects(evo.prepareWorkspace(changed), /private imported peer patch changed/);
    fs.writeFileSync(imported.patch_ref.path, importedBytes);
    assert.equal(evo.entries.length, 1, 'failed second import preserves the first retained source');
    assert.equal(evo.planInputs().EVOLUTION_BEST_ID, 'ORIGINAL');
    await evo.snapshotPeers(2);
    assert.equal(evo.planInputs().EVOLUTION_PEER_OFFERS.length, 0);
    assert.equal(evo.planInputs().EVOLUTION_PEER_ERRORS.length, 1);
    const [local] = await evo.prepare([{parent_id: 'ORIGINAL'}], 3);
    assert.equal(local.parent_id, 'ORIGINAL', 'missing peer snapshot leaves local search available');
    // The usual local verification path remains necessary for a descendant.
    await evo.record(direction, null, null, imported.patch_ref.path);
    assert.equal(await evo.selectBest(direction.id), false);
    await evo.record(direction, null, setup.local_verifier, imported.patch_ref.path);
    assert.equal(await evo.selectBest(direction.id), true);
    assert.equal(evo.planInputs().EVOLUTION_BEST_ID, direction.id);
    assert.deepEqual(fs.readFileSync(path.join(setup.base, 'kernel_src/module.py')), canonical);
    assert.equal(JSON.stringify(start), original, 'imports leave canonical bytes, history, incumbent and clock inputs alone');
    const offDir = path.join(temp, 'off');
    fs.mkdirSync(offDir);
    const offGlobals = {...globals, EVAL_DIR: offDir, EVOLUTION_PEERS: []};
    const off = await new Function(...Object.keys(offGlobals), source + '\nreturn createEvolution;')(...Object.values(offGlobals))(start);
    const beforeOff = JSON.stringify(off.planInputs()), beforeCalls = calls.length;
    await off.snapshotPeers(1);
    assert.equal(JSON.stringify(off.planInputs()), beforeOff, 'OFF adds no prompt fields or archive changes');
    assert.equal(calls.length, beforeCalls, 'OFF performs no peer host calls');
    assert.equal(fs.existsSync(path.join(offDir, 'round_1')), false, 'OFF writes no snapshot artifacts');
    const [unknown] = await off.prepare([{parent_id: 'peer_missing', donor_ids: ['peer_missing']}], 1);
    assert.equal(unknown.parent_id, 'ORIGINAL', 'OFF preserves existing unknown-parent fallback');
    assert.deepEqual(unknown.donor_ids, []);
    assert.equal(unknown.evolution_context.peer_source_error, undefined);
    assert.equal(calls.length, beforeCalls, 'OFF unknown IDs perform no peer I/O');
    fs.writeFileSync(peerArchive, published);
    console.log('PASS: selected peer import, frozen planning inputs, explicit failures and local-only promotion.');
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
