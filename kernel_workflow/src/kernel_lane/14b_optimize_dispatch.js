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
// A population member owns its archive even when it uses the ordinary optimizer.
// Only EVO receives archive planning inputs; publication cannot change greedy's policy.
const populationArchive = A.population_lineage ? await createEvolution(optimizeStart) : null;
applyOptimizeResult(OPTIMIZER === 'evo'
  ? await greedyOptimize(optimizeStart, populationArchive || await createEvolution(optimizeStart), populationArchive)
  : await greedyOptimize(optimizeStart, null, populationArchive));
