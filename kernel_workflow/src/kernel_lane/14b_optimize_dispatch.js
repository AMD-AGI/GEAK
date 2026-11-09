// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host helpers shared by native measurement and the evolution workspace policy.
// One host command; null when it did not run cleanly to exit 0.
async function hostRun(cmd, args, opts = {}) {
  const r = await runProcess({ cmd, args, timeoutMs: 300000, graceMs: 2000, ...opts });
  if (PROCESS_BOUNDED && r.status === 'timeout') {
    throw Object.assign(new Error('external host command timed out'), { code: 'GEAK_HOST_TIMEOUT' });
  }
  return processClean(r) && r.code === 0 ? r.stdout : null;
}
// Finished by itself, and nothing it started can still be running.
const processClean = (r) => r.status === 'completed' && !r.strays_killed && r.group_cleared && !r.stdio_held;
const git = (dir, ...args) => hostRun('git', ['-C', dir, ...args]);
const writeFile = (path, text) => hostRun('sh', ['-c', 'cat > "$1"', 'sh', path], { input: text });

const optimizeStart = { cumulative, bestSeen, bestPerCase, finalWinner, profileSummary, history };
applyOptimizeResult(await nativeDeliveryBoundary(async () => {
  // Publication owns an archive. EVO runs the greedy rounds unchanged until one neither improves nor
  // advances, then roots its own evolution archive at the incumbent those rounds reached.
  // evolution_start=search_start keeps the archive from the first round (the earlier EVO schedule).
  const populationArchive = A.population_lineage ? await createEvolution(optimizeStart) : null;
  const fromStart = OPTIMIZER === 'evo' && A.evolution_start === 'search_start';
  if (OPTIMIZER === 'evo' && ![undefined, 'first_stall', 'search_start'].includes(A.evolution_start)) {
    throw new Error('evolution_start must be first_stall or search_start');
  }
  if (A.evolution_import_stall_sources === true && (OPTIMIZER !== 'evo' || fromStart)) {
    throw new Error('evolution_import_stall_sources requires optimizer=evo with evolution_start=first_stall');
  }
  return await greedyOptimize(optimizeStart, {
    archive: fromStart ? populationArchive || await createEvolution(optimizeStart) : null,
    start: OPTIMIZER === 'evo' && !fromStart ? (state) => createEvolution(state,
      {name: 'evolution_continuation', lineage: null, origin: 'greedy_incumbent'}) : null,
  }, populationArchive);
}));
