// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// These reports describe their source; custody hashes do not validate an explanation.
function reportedDetails(value) {
  if (!value) return null;
  const keys = ['strategy', 'strategies_tried', 'notes', 'evidence_context'];
  const useful = v => typeof v === 'string' ? v.trim().length > 0
    : Array.isArray(v) ? v.length > 0 : v && typeof v === 'object' ? Object.keys(v).length > 0 : false;
  const fields = keys.filter(key => useful(value[key])).map(key => [key, key === 'evidence_context'
    ? Object.fromEntries(Object.entries(value[key]).filter(([, v]) => useful(v))) : value[key]])
    .filter(([, value]) => useful(value));
  return fields.length ? Object.fromEntries(fields) : null;
}
// Prior reports are supplied only for the same retained candidate/source continuation.
// Parents and donors stay separately quoted; an empty default cannot erase useful context.
function mergeReported(prior, value) {
  const update = reportedDetails(value);
  if (!update) return prior || null;
  return {...prior, ...update, ...(prior?.evidence_context || update.evidence_context
    ? {evidence_context: {...prior?.evidence_context, ...update.evidence_context}} : {})};
}
function reportedEvidence(direction, eng, ver, patch, contributors = []) {
  const prior = direction?.reported_context;
  return {schema: 'geak.reported_context/1', authority: 'author_reported_interpretation_only',
    source_binding: {candidate_id: direction?.id || null, source_id: ver?.source_id || prior?.source_binding?.source_id || null,
      patch_ref: {path: patch || null, sha256: ver?.patch_sha256 || null,
        identity_status: ver?.patch_sha256 ? 'reported source hash available' : 'hash unavailable until archive or publication binding'}},
    approach: direction?.approach || prior?.approach || null,
    engineer: mergeReported(prior?.engineer, eng),
    verifier: mergeReported(prior?.verifier, ver?.reported_by === 'integrator' ? null : ver),
    ...(prior?.integrator || ver?.reported_by === 'integrator'
      ? {integrator: mergeReported(prior?.integrator, ver?.reported_by === 'integrator' ? ver : null)} : {}),
    ...(contributors.length || prior?.contributors ? {
      contributor_scope: 'Input references are not proof of inclusion. Their observations apply to their own source only; they do not verify this combined source.',
      contributors: contributors.length ? contributors : prior.contributors} : {})};
}
function contextBytes(value) {
  let bytes = 0;
  for (const char of JSON.stringify(value ?? null)) {
    const point = char.codePointAt(0);
    bytes += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4;
  }
  return bytes;
}
// Prompt-only reduction. The full archive/checkpoint artifact is retained separately.
function boundedContext(value, fullRef, limit = 16384) {
  if (CASE_MEASUREMENT_VIEW !== undefined) {
    let omitted = false;
    const prose = v => Array.isArray(v) ? v.map(prose)
      : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).filter(([key]) => {
        if (['per_case', 'frozen_reference_per_case', 'reported_per_case'].includes(key)) {
          omitted = true; return false;
        }
        return true;
      }).map(([key, child]) => [key, prose(child)])) : v;
    const reduced = prose(value);
    if (omitted) value = {...reduced, case_tables_omitted_from_prose: true, full_context_ref: fullRef};
  }
  if (contextBytes(value) <= limit) return value;
  const compact = (v, depth = 0) => typeof v === 'string'
    ? [...v].slice(0, 192).join('') + ([...v].length > 192 ? ' [text omitted]' : '')
    : Array.isArray(v) ? v.slice(0, 4).map(x => compact(x, depth + 1))
      : v && typeof v === 'object' ? depth > 7 ? '[nested context omitted]'
        : Object.fromEntries(Object.entries(v).slice(0, 24).map(([k, x]) => [k, compact(x, depth + 1)])) : v;
  const result = {context_omitted: 'over_utf8_budget; summary may omit text, array items and nested context',
    full_context_ref: fullRef, summary: compact(value)};
  if (contextBytes(result) > limit) result.summary = {preview_json: [...JSON.stringify(result.summary)].slice(0, Math.floor(limit / 8)).join('')};
  return result;
}

// Only reported measurement metadata/rows; never manufacture host reader authority.
function compactCaseMeasurementContext(value) {
  const context = value?.evidence_context;
  if (!context) return null;
  const m = context.case_measurement;
  return {case_measurement: m ? Object.fromEntries(['basis', 'cohort_ref', 'reduction', 'weight_source']
    .filter(key => m[key] !== undefined).map(key => [key, m[key]])) : null,
    frozen_reference_per_case: context.frozen_reference_per_case ?? null,
    evidence_refs: context.evidence_refs ?? null};
}
// Both experimental views require matching supplied tables, not expected task coverage. This
// switches prompt values only; canonical rows, scalar scores and acceptance stay intact.
function measurementCaseView(value, fullRef = null) {
  const native = Array.isArray(value?.per_case) ? value.per_case : [];
  if (CASE_MEASUREMENT_VIEW === undefined) return {per_case: value?.per_case ?? null};
  const context = compactCaseMeasurementContext(value), m = context?.case_measurement;
  const frozen = context?.frozen_reference_per_case;
  const validRows = rows => Array.isArray(rows) && rows.length > 0
    && rows.every(row => row && typeof row.name === 'string' && row.name.length > 0
      && ['baseline_ms', 'optimized_ms', 'speedup'].every(key => Number.isFinite(row[key]) && row[key] > 0))
    && new Set(rows.map(row => row.name)).size === rows.length;
  const available = m?.basis === 'native' && ['cohort_ref', 'reduction', 'weight_source']
    .every(key => typeof m[key] === 'string' && m[key].trim())
    && validRows(native) && validRows(frozen) && native.length === frozen.length
    && native.every(row => frozen.some(other => other.name === row.name && other.optimized_ms === row.optimized_ms));
  const selected = !available ? {per_case: [], case_measurement_view_unavailable: 'native_and_matched_frozen_tables_unavailable'}
    : CASE_MEASUREMENT_VIEW === 'current_mixed' ? {per_case: frozen}
      : {per_case: native, case_measurement: {...m, authority: 'author_reported; no host validation'},
        frozen_reference_diagnostics: {basis: 'frozen_reference', case_rows: frozen.length,
          full_context_ref: fullRef, evidence_refs: context.evidence_refs}};
  const view = {...selected, case_measurement_refs: {full_context_ref: fullRef,
    evidence_refs: context?.evidence_refs ?? null}};
  log(`[case_measurement_view] ${JSON.stringify({mode: CASE_MEASUREMENT_VIEW,
    available: !!available, native_rows: native.length, frozen_rows: Array.isArray(frozen) ? frozen.length : null,
    selected_unbounded_rows: view.per_case.length, per_view_unbounded_bytes: contextBytes(view),
    full_context_ref: fullRef, evidence_refs: context?.evidence_refs ?? null})}`);
  return view;
}
function currentBestCaseInputs(rows, winner, fullRef) {
  if (CASE_MEASUREMENT_VIEW !== undefined && !winner && rows === BASELINE_PER_CASE) {
    const inputs = {CURRENT_BEST_PER_CASE: rows, CURRENT_BEST_CASE_MEASUREMENT: {
      source: 'ORIGINAL', basis: 'initial_benchmark', authority: 'benchmark_author_reported; no host validation',
      full_context_ref: {path: `${EVAL_DIR}/baseline_timing.json`}}};
    log(`[case_measurement_current_best_inline] ${JSON.stringify({mode: CASE_MEASUREMENT_VIEW,
      initial_original: true, inline_rows: Array.isArray(rows) ? rows.length : null,
      bytes: contextBytes(inputs), full_context_ref: inputs.CURRENT_BEST_CASE_MEASUREMENT.full_context_ref})}`);
    return inputs;
  }
  const value = {per_case: rows, evidence_context: winner?.evidence_context
    || winner?.reported_context?.verifier?.evidence_context};
  const view = measurementCaseView(value, fullRef);
  const inputs = {CURRENT_BEST_PER_CASE: view.per_case,
    ...(CASE_MEASUREMENT_VIEW === undefined ? {} : {CURRENT_BEST_CASE_MEASUREMENT:
      Object.fromEntries(Object.entries(view).filter(([key]) => key !== 'per_case'))})};
  if (CASE_MEASUREMENT_VIEW !== undefined) log(`[case_measurement_current_best_inline] ${JSON.stringify({
    mode: CASE_MEASUREMENT_VIEW, inline_rows: view.per_case.length, bytes: contextBytes(inputs), full_context_ref: fullRef})}`);
  return inputs;
}
// Prompt-only numerical view. Keep every supplied case and its extra fields;
// row counts describe the report, not expected task coverage or qualification.
function caseEvidenceReport(value, fullRef = null) {
  if (!value) return null;
  const view = measurementCaseView(value, fullRef);
  const keys = ['status', 'correctness', 'measurement_valid', 'reported_status', 'reported_validation_status',
    'external_measurement_valid',
    'external_evaluation', 'source_identity', 'primary_regime', 'primary_speedup', 'development_geomean_speedup',
    'verified_geomean', 'verified_weighted', 'verified_arithmetic', 'speedup_geomean',
    'speedup_weighted', 'speedup_arithmetic', 'native_reproducible_gain', 'native_reproducible_regression',
    'variance_note', 'graph_safe'];
  return {...Object.fromEntries(keys.filter(key => value[key] !== undefined).map(key => [key, value[key]])),
    ...view,
    case_rows: Array.isArray(view.per_case) ? view.per_case.length : null,
    ...(value.observed_evidence !== undefined ? {observed_evidence: value.observed_evidence,
      reported_per_case: value.reported_per_case ?? null} : {}),
    evidence_refs: value.evidence_context?.evidence_refs ?? null};
}

// Numeric handoffs are complete or explicitly reference-only. Prose reduction
// must not turn the first four case rows into a seemingly complete comparison.
function boundedCaseEvidence(rows, fullRef, limit = 16384) {
  const bytes = contextBytes(rows), overflow = bytes > limit;
  const result = !overflow ? {entries: rows, full_context_ref: fullRef}
    : {entries: null, case_evidence_omitted: 'over_utf8_budget; read complete cases at full_context_ref',
    full_context_ref: fullRef, row_counts: rows.map(row => ({entry_id: row.archive_ref.entry_id,
      engineer: row.engineer?.case_rows ?? null, verifier: row.verifier?.case_rows ?? null}))};
  if (CASE_MEASUREMENT_VIEW !== undefined) log(`[case_measurement_view_bound] ${JSON.stringify({
    mode: CASE_MEASUREMENT_VIEW, input_bytes: bytes, output_bytes: contextBytes(result), limit, overflow,
    inline_rows: overflow ? 0 : rows.reduce((n, row) => n + (row.engineer?.case_rows || 0) + (row.verifier?.case_rows || 0), 0),
    full_context_ref: fullRef})}`);
  return result;
}

// Retained population for the stock GEAK loop. No model calls or separate runner.
// `name` and `lineage` let EVO root a second archive at a later incumbent without publishing it.
async function createEvolution(start, {name = 'evolution', lineage = A.population_lineage || null,
  origin = 'optimize_start'} = {}) {
  const path = `${EVAL_DIR}/${name}`, base = `${path}/base`, archive = `${path}/archive.json`;
  const population = lineage;
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
  let baseline = {id: 'ORIGINAL', source: base, speedup: start.cumulative, origin};
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
    const keys = ['status', 'correctness', 'strategy', 'strategies_tried', 'measurement_valid',
      'reported_status', 'reported_validation_status', 'external_measurement_valid',
      'external_evaluation', 'source_identity', 'primary_regime', 'primary_speedup', 'development_geomean_speedup',
      'verified_geomean', 'verified_weighted', 'verified_arithmetic', 'speedup_geomean',
      'speedup_weighted', 'speedup_arithmetic', 'variance_note', 'graph_safe',
      'per_case', ...(EXTERNAL_OBJECTIVE ? ['observed_evidence', 'reported_per_case'] : []), 'native_reproducible_gain',
      'native_reproducible_regression', 'notes', 'evidence_context'];
    return Object.fromEntries(keys.filter(key => value[key] !== undefined).map(key => [key, value[key]]));
  };
  const digest = async (file) => {
    const hash = (await run('sha256sum', ['--', file])).match(/^[0-9a-f]{64}/);
    if (!hash) throw new Error('retained patch has no source hash');
    return hash[0];
  };
  const metric = EXTERNAL_OBJECTIVE || {kind: HAS_WORKLOAD ? 'time_weighted' : 'geomean'};
  const caseEvidence = (row) => ({
    archive_ref: {path: archive, entry_id: row.id},
    patch_ref: row.patch_ref || null, entry_verified: row.verified ?? null, metric,
    reported_context_ref: row.reported_context_ref || null, peer_checkpoint: row.peer_checkpoint || null,
    authority: 'Retained evidence only; this view performs no new verification. Reference paths may be unavailable.',
    engineer: caseEvidenceReport(row.engineer, {path: archive, entry_id: row.id, role: 'engineer'}),
    verifier: caseEvidenceReport(row.verifier, {path: archive, entry_id: row.id, role: 'verifier'})});
  const selectedCases = ids => boundedCaseEvidence(ids.map(id => caseEvidence(
    entries.find(row => row.id === id) || {id})), {path: archive, entry_ids: ids});
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
      let reported = null, reportedRef = null;
      if (winner?.reported_context) {
        reported = {...winner.reported_context,
          published_source_binding: {lineage_id: population.id, round, source_id: checked?.source_id || null,
            source_tree: retainedTree, patch_ref: {path: patch, sha256: hash}},
          evidence_ref_scope: 'Paths quoted inside author reports belong to the origin lineage. They are unavailable locally unless separately shared; only this shared context artifact and published patch have shared custody.'};
        const contextPath = `${prefix}.context.json`;
        try {
          await immutable(contextPath, JSON.stringify(reported, null, 2) + '\n');
          reportedRef = {path: contextPath, sha256: await digest(contextPath), authority: 'reported_interpretation_only'};
        } catch (error) {
          reportedRef = {unavailable: 'full reported context artifact could not be retained', expected_path: contextPath};
          log(`[population] checkpoint r${round} context unavailable; source qualification unchanged`);
        }
      }
      const row = {schema: 'geak.population.checkpoint/1', lineage_id: population.id,
        reported_context: reported ? boundedContext(reported, reportedRef, 8192) : null,
        reported_context_ref: reportedRef,
        optimizer: OPTIMIZER, round, base, base_commit: baseCommit, base_tree: baseTree,
        base_source_id: baseSourceId, source_id: checked?.source_id || null,
        source_tree: retainedTree, canonical_tree: tree,
        patch_ref: {path: patch, sha256: hash, base_commit: baseCommit, base_tree: baseTree},
        delivery_patch_ref: deliveryRef,
        metric, speedup, verified: true, qualification: EXTERNAL_OBJECTIVE ? 'native_source_bound' : 'legacy_verifier',
        external_evaluation: checked?.external_evaluation || null,
        // Native checkpoints retain the role report, without upgrading its legacy authority.
        evidence: EXTERNAL_OBJECTIVE ? externalFields(checked) : checked?.verifier_report || {
          status: 'verified', correctness: 'pass', verified_geomean: checked?.geomean_unweighted ?? (HAS_WORKLOAD ? null : speedup),
          verified_weighted: checked?.weighted ?? null,
          per_case: Array.isArray(checked?.per_case) ? checked.per_case
            : (round === 0 && !start.finalWinner ? BASELINE_PER_CASE : []),
          evidence_context: compactCaseMeasurementContext(checked)}};
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
        let reportedRef = peer.reported_context_ref || null;
        if (reportedRef) {
          try {
            if (reportedRef.path !== `${population.root}/${peer.lineage_id}/r${peer.round}.context.json`
                || !/^[0-9a-f]{64}$/.test(reportedRef.sha256)
                || await digest(reportedRef.path) !== reportedRef.sha256) throw new Error('context identity unavailable');
            reportedRef = {...reportedRef, custody: 'hash_checked_at_import'};
          } catch (error) {
            reportedRef = {unavailable: 'full context missing or identity not checked; inline checkpoint quote remains reported only',
              expected_path: reportedRef.path, expected_sha256: reportedRef.sha256};
          }
        }
        next.push({id, parent_id: 'ORIGINAL', donor_ids: [], title: `${peer.lineage_id} committed round ${peer.round}`,
          approach: peer.reported_context?.approach || peer.reported_context?.summary?.approach || null,
          reported_context: peer.reported_context || null, reported_context_ref: reportedRef,
          specialty: 'peer_checkpoint', base_commit: baseCommit,
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
    caseInputs: selectedCases,
    engineerContext(direction) {
      const context = direction.evolution_context;
      if (A.evolution_evidence_handoff !== true) return context;
      const proseReport = value => value ? Object.fromEntries(Object.entries(value)
        .filter(([key]) => !['per_case', 'reported_per_case', 'observed_evidence'].includes(key))) : null;
      const selected = (id) => {
        const row = entries.find(entry => entry.id === id);
        return {id, patch_ref: row?.patch_ref || null, status: row?.status || null,
          verified: row?.verified ?? null, retention_error: row?.retention_error ?? null,
          last_record_error: row?.last_record_error ?? null,
          engineer: proseReport(row?.engineer), verifier: proseReport(row?.verifier),
          reported_context: row?.reported_context || null,
          reported_context_ref: row?.reported_context_ref || {path: archive, entry_id: id},
          peer_checkpoint: row?.peer_checkpoint || null};
      };
      // Prompt-only view: never retain an expanded context inside archive rows.
      const selectedEvidence = {
        parent: selected(direction.parent_id),
        donors: (direction.donor_ids || []).map(selected)};
      return {...context, selected_evidence: boundedContext(selectedEvidence,
        {path: archive, parent_id: direction.parent_id, donor_ids: direction.donor_ids || []}),
        selected_case_evidence: selectedCases([...new Set([direction.parent_id, ...(direction.donor_ids || [])])])};
    },
    planInputs() {
      // Override only this recipient; memory, integration and engineer handoff keep their own controls.
      const plannerCaseEvidence = A.evolution_planner_case_evidence_view === undefined
        ? A.evolution_case_evidence_view === true : A.evolution_planner_case_evidence_view;
      const summary = entries.slice(-12);
      const best = entries.find(row => row.id === bestId);
      if (best && !summary.includes(best)) summary.unshift(best);
      const inputs = {EVOLUTION_ARCHIVE: archive, EVOLUTION_BASE: base, EVOLUTION_BASE_COMMIT: baseCommit,
        EVOLUTION_BEST_ID: bestId, EVOLUTION_ENTRY_COUNT: entries.length,
        EVOLUTION_ENTRIES: summary.map(row => ({id: row.id,
          title: row.title, approach: row.approach, specialty: row.specialty, parent_id: row.parent_id, donor_ids: row.donor_ids,
          status: row.status, speedup: row.speedup, patch_ref: row.patch_ref,
          retention_error: row.retention_error, last_record_error: row.last_record_error || null,
          ...(plannerCaseEvidence ? {case_evidence: caseEvidence(row)} : {}),
          strategy: row.engineer && row.engineer.strategy,
          reported_context: boundedContext(row.reported_context,
            row.reported_context_ref || {path: archive, entry_id: row.id}, 2048)}))};
      if (CASE_MEASUREMENT_VIEW !== undefined) log(`[case_measurement_planner_inline] ${JSON.stringify({
        mode: CASE_MEASUREMENT_VIEW, bytes: contextBytes(inputs), row_counts: inputs.EVOLUTION_ENTRIES.map(row => ({
          id: row.id, engineer: row.case_evidence?.engineer?.case_rows || 0,
          verifier: row.case_evidence?.verifier?.case_rows || 0}))})}`);
      return inputs;
    },
    memoryInputs(directions) {
      const inputs = {EVOLUTION_ARCHIVE: archive, EVOLUTION_ROUND: directions.map(direction => {
        const row = entries.find(entry => entry.id === direction.id);
        return {id: direction.id, parent_id: direction.parent_id, donor_ids: direction.donor_ids || [],
          approach: direction.approach || null, patch_ref: row?.patch_ref || null,
          status: row?.status || null, verified: row?.verified ?? null,
          retention_error: row?.retention_error || null, last_record_error: row?.last_record_error || null,
          ...(A.evolution_case_evidence_view === true ? {case_evidence: caseEvidence(row || {id: direction.id})} : {}),
          report_path: `${direction.out_dir}/report.md`};
      })};
      if (CASE_MEASUREMENT_VIEW !== undefined) log(`[case_measurement_memory_inline] ${JSON.stringify({
        mode: CASE_MEASUREMENT_VIEW, bytes: contextBytes(inputs), row_counts: inputs.EVOLUTION_ROUND.map(row => ({
          id: row.id, engineer: row.case_evidence?.engineer?.case_rows || 0,
          verifier: row.case_evidence?.verifier?.case_rows || 0}))})}`);
      return inputs;
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
        // A zero-byte cumulative diff is the already-materialized archive base.
        if (parentRef.sha256 !== 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855') {
          await run('git', ['-C', workspace, 'apply', '--', parentRef.path]);
        }
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
          base_commit: baseCommit, patch_ref: null, original_patch: previous?.original_patch || patch || null,
          verified, speedup: verified ? speedup : null,
          status: verified ? (speedup <= 1 ? 'slower' : 'correct')
            : ver && says(ver.correctness, 'fail') ? 'incorrect' : 'unverified',
          engineer: evidence(eng) || previous?.engineer || null, verifier: evidence(ver),
          evolution_context: direction.evolution_context || null, retention_error: null,
          reported_context: reportedEvidence({...direction, reported_context: direction.reported_context || previous?.reported_context}, eng, ver, patch)};
        try {
          if (typeof patch !== 'string' || !patch) throw new Error('no patch path supplied');
          const target = `${path}/patches/${direction.id}.diff`, hash = await digest(patch);
          if (previous?.patch_ref && previous.patch_ref.sha256 !== hash) {
            throw Object.assign(new Error('retained patch changed before evidence update'), {sourceChanged: true});
          }
          if (!previous?.patch_ref) await run('cp', ['--no-clobber', '--', patch, target]);
          if (await digest(target) !== hash) throw new Error('existing retained patch differs');
          row.patch_ref = {path: target, sha256: hash, base_commit: baseCommit};
          row.reported_context.source_binding.patch_ref = row.patch_ref;
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
