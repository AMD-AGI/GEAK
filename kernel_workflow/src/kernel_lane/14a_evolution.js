// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Retained population for the stock GEAK loop. No model calls or separate runner.
async function createEvolution(start) {
  const path = `${EVAL_DIR}/evolution`, base = `${path}/base`, archive = `${path}/archive.json`;
  const population = A.population_lineage || null;
  if (population && (typeof population.root !== 'string' || !population.root.startsWith('/')
      || /[\0\r\n]/.test(population.root) || population.root.split('/').includes('..')
      || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(population.id)
      || !Array.isArray(population.imports)
      || population.imports.some(id => !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(id) || id === population.id)
      || (OPTIMIZER !== 'evo' && population.imports.length))) {
    throw new Error('invalid population lineage configuration');
  }
  const entries = [];
  let baseCommit, baseTree, baseSourceId = null, sourcePaths = null, bestId = 'ORIGINAL', queue = Promise.resolve();
  let baseline = {id: 'ORIGINAL', source: base, speedup: start.cumulative, origin: 'optimize_start'};
  const run = async (cmd, args, opts) => {
    const result = await hostRun(cmd, args, opts);
    if (result === null) throw new Error(`evolution ${cmd} failed`);
    return result;
  };
  const serial = (work) => {
    const pending = queue.then(work);
    queue = pending.catch(() => {});
    return pending;
  };
  const snapshot = (rows, best) => ({schema: 'geak.evolution/1', base, base_commit: baseCommit,
    ...(population ? {base_tree: baseTree, base_source_id: baseSourceId, lineage_id: population.id} : {}),
    baseline, best_id: best, entries: rows});
  const save = async (rows, best = bestId) => {
    await run('sh', ['-c',
      'set -e; tmp=$(mktemp "$1/.archive.XXXXXX"); trap \'rm -f -- "$tmp"\' EXIT; cat > "$tmp"; mv -f -- "$tmp" "$2"',
      'sh', path, archive], {input: JSON.stringify(snapshot(rows, best), null, 2) + '\n'});
    entries.splice(0, entries.length, ...rows);
    bestId = best;
  };
  // Each run starts fresh. Atomic directory creation refuses reuse without a
  // separate resume protocol or any mutation of an earlier archive.
  await run('mkdir', [path]);
  await run('mkdir', [`${path}/patches`]);
  baseCommit = (await run('git', ['-C', CANONICAL, 'rev-parse', '--verify', 'HEAD^{commit}'])).trim();
  if (!/^[0-9a-f]{40,64}$/.test(baseCommit)
      || (await run('git', ['-C', CANONICAL, 'status', '--porcelain', '--untracked-files=no'])).trim()) {
    throw new Error('evolution base requires the exact clean canonical Git commit');
  }
  await run('mkdir', [base]);
  await run('bash', [`${WORKFLOW_DIR}/scripts/materialize_workspace.sh`, '--src', CANONICAL,
    '--dst', base, '--shared-root', `${EVAL_DIR}/_shared`, '--link-aiter']);
  if (population) {
    if (EXTERNAL_OBJECTIVE) {
      // The owner already hashed the editable source closure. Fixed fixtures
      // and per-lineage shared-input symlinks are not a parent identity.
      const initial = await checkedExternalMeasurement(start.finalWinner || bench, base);
      if (initial?.external_measurement_valid !== true || !/^[0-9a-f]{64}$/.test(initial.source_id)
          || !Array.isArray(initial.source_paths) || !initial.source_paths.length) {
        throw new Error('population base requires the native editable-source identity');
      }
      baseSourceId = initial.source_id;
      sourcePaths = initial.source_paths;
      baseTree = (await run('git', ['-C', CANONICAL, 'rev-parse', `${baseCommit}^{tree}`])).trim();
    } else {
      // Legacy tasks retain a private index once; queries reuse its objects.
      await run('git', ['-C', base, 'init', '-q']);
      await run('git', ['-C', base, 'add', '-A']);
      baseTree = (await run('git', ['-C', base, 'write-tree'])).trim();
    }
    await run('mkdir', ['-p', `${population.root}/${population.id}`]);
  }
  await save([], 'ORIGINAL');
  const retained = (id) => entries.find(row => row.id === id && row.patch_ref && !row.retention_error);
  const defaultParent = () => retained(bestId)?.verified ? bestId : 'ORIGINAL';
  const evidence = (value) => {
    if (!value) return null;
    const keys = ['status', 'correctness', 'strategy', 'strategies_tried', 'measurement_valid', 'external_measurement_valid',
      'external_evaluation', 'primary_regime', 'primary_speedup', 'development_geomean_speedup',
      'verified_geomean', 'verified_weighted', 'verified_arithmetic', 'speedup_geomean',
      'per_case', 'native_reproducible_gain', 'native_reproducible_regression', 'notes'];
    return Object.fromEntries(keys.filter(key => value[key] !== undefined).map(key => [key, value[key]]));
  };
  const digest = async (file) => {
    const hash = (await run('sha256sum', ['--', file])).match(/^[0-9a-f]{64}/);
    if (!hash) throw new Error('retained patch has no source hash');
    return hash[0];
  };
  const metric = EXTERNAL_OBJECTIVE || {kind: HAS_WORKLOAD ? 'time_weighted' : 'geomean'};
  const immutable = (file, contents) => run('sh', ['-c', 'set -Ce; cat > "$1"; chmod a-w -- "$1"',
    'sh', file], {input: contents});
  const checkpointTree = async (winner, round) => {
    if (!population) return null;
    try {
      if (winner?.source_tree) return winner.source_tree;
      if (!winner) return (await run('git', ['-C', CANONICAL, 'rev-parse', 'HEAD^{tree}'])).trim();
      const hash = await digest(winner.patch);
      const indexed = (...args) => run('env', [`GIT_INDEX_FILE=${path}/checkpoint-${round}.index`,
        'git', '-C', CANONICAL, ...args]);
      await indexed('read-tree', 'HEAD');
      await indexed('apply', '--cached', '--', winner.patch);
      const tree = (await indexed('write-tree')).trim();
      if (await digest(winner.patch) !== hash) throw new Error('winner patch changed');
      return tree;
    } catch (error) { log(`[population] checkpoint source unavailable: ${String(error)}`); return null; }
  };
  const publishCheckpoint = (round, winner, expectedTree) => serial(async () => {
    if (!population || !expectedTree) return false;
    try {
      const tree = (await run('git', ['-C', CANONICAL, 'rev-parse', 'HEAD^{tree}'])).trim();
      if (tree !== expectedTree
          || (await run('git', ['-C', CANONICAL, 'status', '--porcelain', '--untracked-files=no'])).trim()) {
        throw new Error('committed source differs from the verified winner');
      }
      const prefix = `${population.root}/${population.id}/r${round}`;
      const patch = `${prefix}.diff`;
      // Ordinary patches are relative to the previous local incumbent. Always
      // export a cumulative patch from this lineage's fixed initial source.
      const diff = await run('git', ['-C', CANONICAL, 'diff', '--binary', '--no-color',
        '--no-ext-diff', '--src-prefix=a/', '--dst-prefix=b/', baseCommit, 'HEAD', '--', ...(sourcePaths || ['.'])]);
      await immutable(patch, diff);
      const hash = await digest(patch);
      const checked = EXTERNAL_OBJECTIVE ? await checkedExternalMeasurement(winner, base, patch) : winner;
      const speedup = EXTERNAL_OBJECTIVE ? primSpeedup(checked) : round === 0 ? start.cumulative : winner.geomean;
      if (!Number.isFinite(speedup) || speedup <= 0
          || (EXTERNAL_OBJECTIVE && (checked?.external_measurement_valid !== true
            || checked.patch_sha256 !== hash))) throw new Error('committed checkpoint has no exact source-bound evidence');
      let retainedTree = tree;
      if (!EXTERNAL_OBJECTIVE) {
        // Materialization can add an ignore file. Name the exact retained tree
        // separately from the canonical commit checked above.
        const indexed = (...args) => run('env', [`GIT_INDEX_FILE=${path}/published-${round}.index`,
          'git', '-C', base, ...args]);
        await indexed('read-tree', baseTree);
        if (diff) await indexed('apply', '--cached', '--', patch);
        retainedTree = (await indexed('write-tree')).trim();
      }
      let deliveryRef = null;
      try {
        const originalCommit = (await run('git', ['-C', CANONICAL, 'rev-list', '--max-parents=0', baseCommit])).trim();
        if (!/^[0-9a-f]{40,64}$/.test(originalCommit)) throw new Error('no unique ORIGINAL commit');
        const delivery = `${prefix}.delivery.diff`;
        await immutable(delivery, await run('git', ['-C', CANONICAL, 'diff', '--binary', '--no-color',
          '--no-ext-diff', '--src-prefix=a/', '--dst-prefix=b/', originalCommit, 'HEAD', '--', ...(sourcePaths || ['.'])]));
        const deliveryHash = await digest(delivery);
        const delivered = EXTERNAL_OBJECTIVE ? await checkedExternalMeasurement(winner, KERNEL_PATH_ORIG, delivery) : checked;
        if (EXTERNAL_OBJECTIVE && (delivered?.external_measurement_valid !== true
            || delivered.patch_sha256 !== deliveryHash)) throw new Error('delivery patch does not reconstruct measured source from ORIGINAL');
        deliveryRef = {path: delivery, sha256: deliveryHash, base: 'ORIGINAL'};
      } catch (error) { log(`[population] checkpoint r${round} retained without an ORIGINAL delivery patch: ${String(error)}`); }
      const row = {schema: 'geak.population.checkpoint/1', lineage_id: population.id,
        optimizer: OPTIMIZER, round, base, base_commit: baseCommit, base_tree: baseTree,
        base_source_id: baseSourceId, source_id: checked?.source_id || null,
        source_tree: retainedTree, canonical_tree: tree,
        patch_ref: {path: patch, sha256: hash, base_commit: baseCommit, base_tree: baseTree},
        delivery_patch_ref: deliveryRef,
        metric, speedup, verified: true, qualification: EXTERNAL_OBJECTIVE ? 'native_source_bound' : 'legacy_verifier',
        external_evaluation: checked?.external_evaluation || null,
        evidence: EXTERNAL_OBJECTIVE ? externalFields(checked) : {
          status: 'verified', correctness: 'pass', verified_geomean: checked?.geomean_unweighted ?? speedup,
          verified_weighted: checked?.weighted ?? null, per_case: checked?.per_case || BASELINE_PER_CASE}};
      await immutable(`${prefix}.json`, JSON.stringify(row, null, 2) + '\n');
      const pointer = {schema: 'geak.population.pointer/1', lineage_id: population.id,
        checkpoint: `${prefix}.json`, sha256: await digest(`${prefix}.json`)};
      await run('sh', ['-c',
        'set -e; tmp=$(mktemp "$1/.latest.XXXXXX"); trap \'rm -f -- "$tmp"\' EXIT; cat > "$tmp"; mv -f -- "$tmp" "$1/latest.json"',
        'sh', `${population.root}/${population.id}`], {input: JSON.stringify(pointer) + '\n'});
      log(`[population] ${population.id} published committed checkpoint r${round} (${speedup.toFixed(3)}x)`);
      return true;
    } catch (error) {
      log(`[population] checkpoint r${round} not published: ${String(error)}`);
      return false;
    }
  });
  const refreshPeers = () => serial(async () => {
    if (!population?.imports.length) return;
    try {
      const result = JSON.parse(await run('python3', [`${WORKFLOW_DIR}/scripts/query_population.py`,
        '--root', population.root, '--checkpoints-only',
        ...(baseSourceId ? ['--base-source-id', baseSourceId] : ['--base-tree', baseTree]),
        ...population.imports.flatMap(id => ['--lineage', id])]));
      const next = [...entries];
      for (const item of result.records || []) {
        if (!['qualified', 'no_improvement', 'legacy_verified'].includes(item.status)) continue;
        const peer = item.checkpoint;
        if (!peer || !population.imports.includes(peer.lineage_id)
            || (baseSourceId ? peer.base_source_id !== baseSourceId : peer.base_tree !== baseTree)
            || JSON.stringify(peer.metric) !== JSON.stringify(metric)
            || !Number.isSafeInteger(peer.round) || peer.round < 1) continue;
        const id = `peer_${peer.lineage_id}_r${peer.round}`;
        if (next.some(row => row.id === id)) continue;
        const target = `${path}/patches/${id}.diff`;
        if (await digest(peer.patch_ref.path) !== peer.patch_ref.sha256) throw new Error('peer patch changed before copy');
        await run('cp', ['--no-clobber', '--', peer.patch_ref.path, target]);
        if (await digest(target) !== peer.patch_ref.sha256) throw new Error('copied peer patch changed');
        const verified = item.evidence || peer.evidence;
        const speedup = EXTERNAL_OBJECTIVE ? primSpeedup(verified) : peer.speedup;
        if (!Number.isFinite(speedup) || speedup <= 0) continue;
        next.push({id, parent_id: 'ORIGINAL', donor_ids: [], title: `${peer.lineage_id} committed round ${peer.round}`,
          approach: null, specialty: 'peer_checkpoint', base_commit: baseCommit,
          patch_ref: {path: target, sha256: peer.patch_ref.sha256, base_commit: baseCommit, base_tree: baseTree},
          verified: true, speedup, status: speedup <= 1 ? 'slower' : 'correct',
          engineer: null, verifier: evidence(verified), retention_error: null,
          peer_checkpoint: {lineage_id: peer.lineage_id, round: peer.round,
            path: item.checkpoint_path, sha256: item.checkpoint_sha256, source_tree: peer.source_tree}});
      }
      if (next.length !== entries.length) await save(next); // Preserve this member's local best ID.
    } catch (error) { log(`[population] peer refresh failed: ${String(error)}`); }
  });
  return {
    base, baseCommit, path: archive, entries, checkpointTree, publishCheckpoint, refreshPeers,
    engineerContext(direction) {
      const context = direction.evolution_context;
      if (A.evolution_evidence_handoff !== true) return context;
      const selected = (id) => {
        const row = entries.find(entry => entry.id === id);
        return {id, patch_ref: row?.patch_ref || null, status: row?.status || null,
          verified: row?.verified ?? null, retention_error: row?.retention_error ?? null,
          last_record_error: row?.last_record_error ?? null,
          engineer: row?.engineer || null, verifier: row?.verifier || null};
      };
      // Prompt-only view: never retain an expanded context inside archive rows.
      const selectedEvidence = {
        parent: selected(direction.parent_id),
        donors: (direction.donor_ids || []).map(selected)};
      let bytes = 0;
      for (const char of JSON.stringify(selectedEvidence)) {
        const point = char.codePointAt(0);
        bytes += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4;
      }
      return bytes > 16384
        ? {...context, selected_evidence_omitted: 'over_16384_utf8_bytes'}
        : {...context, selected_evidence: selectedEvidence};
    },
    planInputs() {
      const summary = entries.slice(-12);
      const best = entries.find(row => row.id === bestId);
      if (best && !summary.includes(best)) summary.unshift(best);
      return {EVOLUTION_ARCHIVE: archive, EVOLUTION_BASE: base, EVOLUTION_BASE_COMMIT: baseCommit,
        EVOLUTION_BEST_ID: bestId, EVOLUTION_ENTRY_COUNT: entries.length,
        EVOLUTION_ENTRIES: summary.map(row => ({id: row.id,
          title: row.title, approach: row.approach, specialty: row.specialty, parent_id: row.parent_id, donor_ids: row.donor_ids,
          status: row.status, speedup: row.speedup, patch_ref: row.patch_ref,
          retention_error: row.retention_error, last_record_error: row.last_record_error || null,
          strategy: row.engineer && row.engineer.strategy}))};
    },
    memoryInputs(directions) {
      return {EVOLUTION_ARCHIVE: archive, EVOLUTION_ROUND: directions.map(direction => {
        const row = entries.find(entry => entry.id === direction.id);
        return {id: direction.id, parent_id: direction.parent_id, donor_ids: direction.donor_ids || [],
          approach: direction.approach || null, patch_ref: row?.patch_ref || null,
          status: row?.status || null, verified: row?.verified ?? null,
          retention_error: row?.retention_error || null, last_record_error: row?.last_record_error || null,
          report_path: `${direction.out_dir}/report.md`};
      })};
    },
    prepare(directions, round) {
      if (!Number.isSafeInteger(round) || round < 1) throw new Error('positive evolution round required');
      return directions.map((direction, index) => {
        const errors = [], requested = direction.parent_id;
        let parent = requested, resolution = 'explicit';
        if (!parent || (parent !== 'ORIGINAL' && !retained(parent))) {
          if (parent) errors.push(`unknown or unavailable parent_id: ${String(parent)}`);
          parent = defaultParent();
          resolution = `${requested ? 'invalid_reference_' : ''}default_${parent === 'ORIGINAL' ? 'original' : 'best_verified'}`;
        }
        const donors = [];
        if (direction.donor_ids != null && !Array.isArray(direction.donor_ids)) errors.push('donor_ids must be an array');
        for (const donor of Array.isArray(direction.donor_ids) ? direction.donor_ids : []) {
          if (donor !== 'ORIGINAL' && !retained(donor)) errors.push(`unknown or unavailable donor_id: ${String(donor)}`);
          else if (!donors.includes(donor)) donors.push(donor);
        }
        return {...direction, id: `r${round}_d${index}`, requested_id: direction.id || null,
          parent_id: parent, donor_ids: donors, evolution_context: {requested_parent_id: requested || null,
            parent_id: parent,
            parent_resolution: resolution, reference_errors: errors,
            donors: donors.map(id => ({id, patch_ref: id === 'ORIGINAL' ? null : retained(id).patch_ref}))}};
      });
    },
    async prepareWorkspace(direction) {
      const workspace = `${direction.out_dir}/workspace`;
      const parent = direction.parent_id === 'ORIGINAL' ? null : retained(direction.parent_id);
      if (direction.parent_id !== 'ORIGINAL' && !parent) throw new Error('selected evolution parent is unavailable');
      const parentRef = parent ? {...parent.patch_ref} : null;
      if (parentRef && await digest(parentRef.path) !== parentRef.sha256) throw new Error('selected parent patch changed');
      await run('bash', [`${WORKFLOW_DIR}/scripts/materialize_workspace.sh`, '--src', base,
        '--dst', workspace, '--shared-root', `${EVAL_DIR}/_shared`, '--link-aiter']);
      await run('git', ['-C', workspace, 'init', '-q']);
      await run('git', ['-C', workspace, 'add', '-A']);
      await run('git', ['-C', workspace, 'commit', '-qm', 'Candidate baseline']);
      const workspaceBase = (await run('git', ['-C', workspace, 'rev-parse', 'HEAD'])).trim();
      if (parentRef) {
        await run('git', ['-C', workspace, 'apply', '--', parentRef.path]);
        if (await digest(parentRef.path) !== parentRef.sha256) throw new Error('selected parent patch changed during preparation');
      }
      await run('git', ['-C', workspace, 'add', '-A']);
      const initialTree = (await run('git', ['-C', workspace, 'write-tree'])).trim();
      if (parentRef) await run('git', ['-C', workspace, 'commit', '--allow-empty', '-qm', 'Selected evolution parent']);
      const initialCommit = (await run('git', ['-C', workspace, 'rev-parse', 'HEAD'])).trim();
      direction.evolution_context = {...direction.evolution_context,
        workspace_base_commit: workspaceBase,
        initialization: {status: 'ready', workspace, base_commit: baseCommit,
          workspace_base_commit: workspaceBase, initial_tree: initialTree, initial_commit: initialCommit,
          parent_id: direction.parent_id, parent_patch_ref: parentRef}};
      return direction.evolution_context.initialization;
    },
    record(direction, eng, ver, patch) {
      return serial(async () => {
        const previous = entries.find(row => row.id === direction?.id);
        const failure = (reason) => {
          log(`[evolution] ${direction?.id}: ${reason}`);
          return {id: direction?.id || null, status: 'unverified', verified: false, speedup: null,
            patch_ref: null, ...previous, retention_error: reason, last_record_error: reason};
        };
        if (!/^r[1-9][0-9]*_(?:d[0-9]+|integrated)$/.test(direction?.id)) {
          return failure(`invalid evolution id: ${direction?.id}`);
        }
        if (previous && (previous.parent_id !== direction.parent_id
            || JSON.stringify(previous.donor_ids) !== JSON.stringify(direction.donor_ids || []))) {
          return failure(`evolution lineage changed for ${direction.id}`);
        }
        const speedup = primSpeedup(ver);
        const verified = !!(ver && (says(ver.status, 'verified') || says(ver.status, 'regression'))
          && says(ver.correctness, 'pass') && Number.isFinite(speedup) && speedup > 0
          && (!EXTERNAL_OBJECTIVE || ver.external_measurement_valid === true));
        const row = {id: direction.id, parent_id: direction.parent_id, donor_ids: direction.donor_ids || [],
          title: direction.title || '', approach: direction.approach || null, specialty: direction.specialty || '',
          base_commit: baseCommit, patch_ref: null, original_patch: patch || null,
          verified, speedup: verified ? speedup : null,
          status: verified ? (speedup <= 1 ? 'slower' : 'correct')
            : ver && says(ver.correctness, 'fail') ? 'incorrect' : 'unverified',
          engineer: evidence(eng) || previous?.engineer || null, verifier: evidence(ver),
          evolution_context: direction.evolution_context || null, retention_error: null};
        try {
          if (typeof patch !== 'string' || !patch) throw new Error('no patch path supplied');
          const target = `${path}/patches/${direction.id}.diff`, hash = await digest(patch);
          if (previous?.patch_ref && previous.patch_ref.sha256 !== hash) {
            throw Object.assign(new Error('retained patch changed before evidence update'), {sourceChanged: true});
          }
          if (!previous?.patch_ref) await run('cp', ['--no-clobber', '--', patch, target]);
          if (await digest(target) !== hash) throw new Error('existing retained patch differs');
          row.patch_ref = {path: target, sha256: hash, base_commit: baseCommit};
        } catch (error) {
          if (error.sourceChanged) return failure(String(error));
          row.retention_error = `patch retention failed: ${String(error)}`;
          if (previous?.patch_ref) {
            const failure = row.retention_error;
            Object.assign(row, previous, {last_record_error: failure});
          }
        }
        const next = previous ? entries.map(entry => entry.id === row.id ? row : entry) : [...entries, row];
        try { await save(next); }
        catch (error) {
          row.retention_error = `${row.retention_error ? row.retention_error + '; ' : ''}archive checkpoint failed: ${String(error)}`;
        }
        if (row.retention_error || row.last_record_error) {
          log(`[evolution] ${row.id}: ${row.retention_error || row.last_record_error}`);
        }
        return row;
      });
    },
    selectBest(id) {
      return serial(async () => {
        const selected = retained(id);
        if (id !== 'ORIGINAL' && (!selected || !selected.verified)) {
          log(`[evolution] best checkpoint unchanged: ${id} has no retained verified source`); return false;
        }
        try { await save([...entries], id); return true; }
        catch (error) { log(`[evolution] best checkpoint failed: ${String(error)}`); return false; }
      });
    },
  };
}
