import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const HERE = path.dirname(new URL(import.meta.url).pathname);
const WORKFLOW_DIR = path.resolve(HERE, '..');
const SCRIPT = path.join(WORKFLOW_DIR, 'avo_workflow.js');

function transformWorkflow(source) {
  const scriptDir = path.dirname(SCRIPT);
  let body = source.replace(
    /import\(\s*(['"])(\.[^'"]+)\1\s*\)/g,
    (_match, _quote, spec) => `import(${JSON.stringify(
      pathToFileURL(path.resolve(scriptDir, spec)).href)})`);
  body = body.replace(/^([ \t]*)export\s+/gm, '$1');
  return body;
}

async function runSmoke(operator, {
  techLeadStop = false, variation = null, verify = null, extraArgs = {},
  setupResult = null, seedState = null, directions = null, agentNull = null,
  techLeadReplanStops = false, benchmarkReport = false, practicalRoofline = false,
  supervisorDirective = null, abbreviateReplies = false,
} = {}) {
  const root = mkdtempSync(path.join(tmpdir(), `avo-phase-${operator}-`));
  const kernel = path.join(root, 'kernel');
  const evalDir = path.join(root, 'eval');
  const repo = path.join(evalDir, 'avo_repo');
  mkdirSync(kernel, { recursive: true });
  mkdirSync(repo, { recursive: true });
  if (seedState) seedState({ evalDir, repo, root });
  const phases = [];
  const labels = [];
  const prompts = [];
  const logs = [];
  const heartbeats = [];
  const prunes = [];

  const reply = async (prompt, options = {}) => {
    const label = options.label || '';
    labels.push(label);
    prompts.push({ label, prompt: String(prompt || '') });
    // Mirrors the shim contract: agent() resolves to null once it gives up, which
    // is how an unreachable backend reaches the workflow.
    if (agentNull && agentNull(label)) return null;
    if (label === 'director:setup') {
      if (setupResult) return setupResult({ evalDir, repo });
      return { resume: false, eval_dir: evalDir, avo_repo: repo,
        baseline_dir: path.join(evalDir, 'baseline'), kernel_name: 'fixture',
        source_files: [] };
    }
    if (label === 'benchmark_engineer') {
      return { commandment_path: path.join(evalDir, 'COMMANDMENT.md'),
        correctness_cmd: 'true', benchmark_cmd: 'true', profile_cmd: 'true',
        baseline_per_case: [{ name: 'case', baseline_ms: 1, optimized_ms: 1, speedup: 1 }],
        baseline_geomean_ms: 1, num_test_cases: 1, reliable: true };
    }
    if (label === 'lineage_engineer:seed') return { seeded: true, tag: 'avo-v0' };
    if (label.startsWith('tech_lead:replan')) {
      // The one re-prompt a refused stop earns. Default: TechLead complies and plans for
      // real; techLeadReplanStops models the one that keeps refusing.
      if (techLeadReplanStops) return { stop: true, reasoning: 'still exhausted', directions: [] };
      return { stop: false, reasoning: 'replanned under a refused stop', directions: [{
        id: 'r0', title: 'replanned direction', specialty: 'algorithm',
        focus_files: ['kernel.py'], expected_speedup: 1.02, prompt: 'replanned fixture probe',
      }] };
    }
    if (label.startsWith('tech_lead:plan')) {
      return { stop: techLeadStop, reasoning: techLeadStop ? 'search exhausted' : 'fixture',
        directions: techLeadStop ? [] : (directions || [{
        id: 'd0', title: 'fixture direction', specialty: 'algorithm',
        focus_files: ['kernel.py'], expected_speedup: 1.01, prompt: 'no-op fixture',
      }]) };
    }
    if (label.startsWith('supervisor c')) {
      if (supervisorDirective) return supervisorDirective;
      return {
        diagnosis: 'fixture stall',
        mark_failed: [],
        new_directions: [{
          name: 'supervisor escape', expected: 'exercise ESCALATE health gates',
          prompt: 'run one deterministic rescue attempt',
        }],
        converge: false,
      };
    }
    if (label.startsWith('variation') || label.startsWith('escalate_')) {
      if (variation) return variation(label);
      return { status: 'no_change', strategy: 'fixture', patch_file: '',
        patch_hash: '', self_speedup: 1, per_case: [], rationale: 'fixture' };
    }
    if (label.startsWith('verify') && verify) return verify(label);
    if (label === 'tech_lead:report') {
      writeFileSync(path.join(evalDir, 'report.md'), '# Fixture report\n');
      writeFileSync(path.join(evalDir, 'final_patch.diff'), 'diff --git a/a b/a\n');
      const roofline_evidence_path = path.join(evalDir, 'roofline_bound.md');
      if (practicalRoofline) {
        writeFileSync(roofline_evidence_path, 'Final best: 99.4% of measured practical ceiling.\n');
      }
      return { final_speedup_geomean: 1, final_speedup_arithmetic: 1,
        steps: 1, commits: 0, report_path: path.join(evalDir, 'report.md'),
        final_patch: path.join(evalDir, 'final_patch.diff'),
        per_case: [{ name: 'case', baseline_ms: 1, optimized_ms: 1, speedup: 1 }],
        ...(practicalRoofline ? {
          practical_roofline_fraction: 0.994,
          practical_roofline_regime: 'mixed-streaming',
          roofline_evidence_path,
        } : {}) };
    }
    if (label === 'director:validate') {
      let benchmark_report = '';
      if (benchmarkReport) {
        benchmark_report = path.join(evalDir, 'validation_workspace', 'performance_report.json');
        mkdirSync(path.dirname(benchmark_report), { recursive: true });
        writeFileSync(benchmark_report, '{"runs":[1,2,3,4,5]}\n');
      }
      return { kernel_name: 'fixture', director_verified_speedup_geomean: 1,
        director_verified_speedup_arithmetic: 1, reported_speedup_geomean: 1,
        validation_status: 'accepted', correctness: 'pass',
        formal_status: 'not_applicable', infrastructure_status: 'healthy',
        per_case: [{ name: 'case', baseline_ms: 1, optimized_ms: 1, speedup: 1 }],
        applied_to_original: 'false', final_patch: path.join(evalDir, 'final_patch.diff'),
        benchmark_report };
    }
    if (label === 'director:cleanup') return { cleaned: true };
    throw new Error(`unexpected agent label: ${label}`);
  };
  // Measuring roles hand their result over as a report file (REPORT_PATH, or
  // VERIFY_DIR/verify_report.json); the workflow reads that file, not the reply.
  const agent = async (prompt, options = {}) => {
    const result = await reply(prompt, options);
    const input = (key) => (String(prompt).match(new RegExp(`^- ${key}: (.+)$`, 'm')) || [])[1];
    const reportPath = input('REPORT_PATH')
      || (String(options.label || '').startsWith('verify') && input('VERIFY_DIR')
        ? path.join(input('VERIFY_DIR'), 'verify_report.json') : '');
    if (result && reportPath) {
      mkdirSync(path.dirname(reportPath), { recursive: true });
      writeFileSync(reportPath, JSON.stringify(result));
      // A long per_case list can come back retyped as a summary string in the reply.
      if (abbreviateReplies && Array.isArray(result.per_case)) {
        return { ...result, per_case: `${result.per_case.length} cases in ${reportPath}` };
      }
    }
    return result;
  };

  const pipeline = async (items, ...stages) => Promise.all(
    items.map((item) => stages.reduce(
      (promise, stage) => promise.then(stage), Promise.resolve(item))));
  const args = {
    kernel_path: kernel,
    workflow_dir: WORKFLOW_DIR,
    exp_root: path.join(root, 'exp'),
    budget_s: 60,
    final_phase_reserve_s: 0,
    max_steps: 1,
    min_commits: 0,
    use_profiling: 'false',
    operator,
    operator_directions: 1,
    gpu_ids: '0',
    ...extraArgs,
  };

  // Mirror the real signature — addWorktree(avoRepo, wsPath, ref). Swapping the
  // last two made the mock mkdir the *ref* as a relative path, which littered the
  // checkout with empty avo-v{N} directories and handed roles a WORKSPACE of
  // "avo-v0" instead of a path, so a mis-wired workspace could not be caught here.
  globalThis.addWorktree = (_repo, wsPath, _ref) => {
    mkdirSync(wsPath, { recursive: true });
    return wsPath;
  };
  globalThis.pruneWorktrees = (...callArgs) => {
    prunes.push(callArgs);
    return 0;
  };
  globalThis.recordAttempt = () => true;
  globalThis.syncAvoHeartbeat = (_stateDir, fields) => {
    heartbeats.push({ ...fields });
    return true;
  };

  try {
    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
    const fn = new AsyncFunction('args', 'agent', 'phase', 'log', 'pipeline',
      transformWorkflow(readFileSync(SCRIPT, 'utf8')));
    const result = await fn(args, agent, (name) => phases.push(name),
      (msg) => logs.push(String(msg)), pipeline);
    const artifactSnapshot = (result.output_artifacts || []).map((artifact) => ({
      artifact,
      exists: existsSync(artifact),
      contents: existsSync(artifact) ? readFileSync(artifact, 'utf8') : '',
    }));
    return { result, phases, labels, prompts, logs, heartbeats, prunes, artifactSnapshot };
  } finally {
    delete globalThis.addWorktree;
    delete globalThis.pruneWorktrees;
    delete globalThis.recordAttempt;
    delete globalThis.syncAvoHeartbeat;
    rmSync(root, { recursive: true, force: true });
  }
}

test('measured results come from the report file when the reply abbreviates per_case', async () => {
  // Real failure: the pre-commit Director wrote all 38 cases to its report but replied
  // per_case as "38 cases in <file>", and summarizeRoofline threw on the string.
  const patchDir = mkdtempSync(path.join(tmpdir(), 'avo-patch-'));
  const patch = path.join(patchDir, 'candidate.patch');
  writeFileSync(patch, 'diff --git a/new.txt b/new.txt\nnew file mode 100644\n'
    + '--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+x\n');
  const perCase = Array.from({ length: 38 }, (_, i) => ({
    name: `case${i}`, baseline_ms: 2, optimized_ms: 1, speedup: 2 }));
  try {
    const { result, labels, logs } = await runSmoke('team', {
      abbreviateReplies: true,
      variation: () => ({ status: 'ok', strategy: 'fixture', patch_file: patch,
        patch_hash: 'abc', self_speedup: 2, per_case: [], rationale: 'fixture' }),
      verify: () => ({ status: 'ok', correctness: 'pass', verified_geomean: 2,
        verified_arithmetic: 2, infrastructure_status: 'healthy', per_case: perCase }),
    });
    assert.ok(labels.some((label) => label.startsWith('verify')));
    assert.ok(labels.includes('director:validate'));
    assert.equal(logs.some((line) => line.includes('is not a function')), false);
    assert.ok(result);
  } finally {
    rmSync(patchDir, { recursive: true, force: true });
  }
});

test('source result materializes durable artifacts before cleanup', async () => {
  const { result, artifactSnapshot } = await runSmoke('single', { benchmarkReport: true });
  assert.ok(result.output_bundle.endsWith('/final_artifacts'));
  assert.ok(result.output_artifacts.length >= 4);
  assert.ok(artifactSnapshot.every((item) => item.exists));
  assert.ok(artifactSnapshot.some((item) =>
    item.artifact.endsWith('/benchmark_report.json')
    && item.contents.includes('"runs":[1,2,3,4,5]')));
  assert.ok(result.formal_report_path.includes('/avo_state/final_validation/'));
  assert.equal(result.formal_report_path.includes('/validation_workspace/'), false);
});

test('heartbeat advances through terminal workflow phases', async () => {
  const { heartbeats } = await runSmoke('single');
  const phases = heartbeats.map((heartbeat) => heartbeat.phase);
  for (const required of ['report', 'validate', 'cleanup', 'complete']) {
    assert.ok(phases.includes(required), `missing ${required} heartbeat`);
  }
  assert.equal(phases.at(-1), 'complete');
});

test('measured practical roofline is returned with explicit provenance', async () => {
  const { result, artifactSnapshot } = await runSmoke('single', { practicalRoofline: true });
  assert.equal(result.roofline_fraction, 0.994);
  assert.equal(result.roofline_regime, 'mixed-streaming');
  assert.equal(result.roofline_basis, 'measured_practical');
  assert.ok(result.roofline_evidence_path.endsWith('/roofline_bound.md'));
  assert.ok(artifactSnapshot.some((item) =>
    item.artifact.endsWith('/roofline_evidence.md') && item.contents.includes('99.4%')));
});

for (const operator of ['single', 'team']) {
  test(`workflow phases complete with mocked ${operator} operator`, async () => {
    const { result, phases, labels } = await runSmoke(operator);
    assert.deepEqual(phases, ['Setup', 'Benchmark', 'Seed', 'Evolve', 'Report', 'Validate', 'Cleanup']);
    assert.equal(result.validation_status, 'accepted');
    assert.equal(result.steps, 1);
    assert.ok(labels.some((label) => label.startsWith('variation')));
    if (operator === 'team') {
      assert.ok(labels.some((label) => label.startsWith('tech_lead:plan')));
    }
  });
}

test('omitted operator defaults to team', async () => {
  const { labels } = await runSmoke(undefined);
  assert.ok(labels.some((label) => label.startsWith('tech_lead:plan')),
    'default operator must execute Team planning');
});

test('TechLead is told the wall-clock budget, not just the direction cap', async () => {
  // tech_lead.md rule 1 conditions its stop gate on whether budget remains, so the budget
  // has to reach the role. It used to receive only BUDGET_REMAINING, which carried the
  // per-round direction cap and no time at all — leaving the gate unevaluable and the stop
  // decided on the agent's confidence alone.
  const { prompts } = await runSmoke('team');
  const plan = prompts.find((p) => p.label.startsWith('tech_lead:plan'));
  assert.ok(plan, 'tech_lead:plan must run');
  for (const key of ['BUDGET_TOTAL_S', 'BUDGET_REMAINING_S', 'BUDGET_REMAINING_PCT',
    'ROUNDS_AFFORDABLE', 'AVG_ROUND_S', 'DIRECTION_CAP', 'MAX_STEPS']) {
    assert.ok(plan.prompt.includes(key), `plan prompt must carry ${key}`);
  }
  // The old name has to be gone, not merely supplemented: a direction count sitting under
  // a name that reads like a budget is exactly what misled the role.
  assert.equal(/\bBUDGET_REMAINING\b/.test(plan.prompt), false,
    'the ambiguous BUDGET_REMAINING must no longer be passed');
});

test('team operator honors TechLead stop once early_stop opts into it', async () => {
  const { result, labels } = await runSmoke('team', {
    techLeadStop: true, extraArgs: { early_stop: 'true' },
  });
  assert.equal(result.validation_status, 'accepted');
  assert.ok(labels.some((label) => label.startsWith('tech_lead:plan')));
  assert.equal(labels.some((label) => label.startsWith('variation')), false);
});

test('a confirmed TechLead stop cannot end the run while the budget still fits a round', async () => {
  // The regression this guards: with early_stop off the run is contracted to the full
  // budget, but a TechLead convinced it had converged re-issued the stop until the
  // confirmation streak was met, ending a 12h run at round 7 with ~10h unspent. Meeting
  // the streak must not be sufficient on its own — the budget has to be spent too.
  const { result, labels, logs } = await runSmoke('team', {
    techLeadStop: true, extraArgs: { team_stop_confirmations: 1 },
  });
  assert.equal(result.validation_status, 'accepted');
  assert.ok(labels.some((label) => label.startsWith('variation')));
  assert.ok(logs.some((line) => line.includes('REFUSED [budget_remains]')));
});

test('a refused stop is re-planned rather than forfeited to a generic escape probe', async () => {
  // Holding the budget only guarantees the round runs; this is what makes the round worth
  // running. Without the re-prompt a stopping TechLead returns no directions and the round
  // falls through to one unfocused team_escape probe.
  const { labels, prompts, logs } = await runSmoke('team', {
    techLeadStop: true, extraArgs: { team_stop_confirmations: 1 },
  });
  assert.ok(labels.some((label) => label.startsWith('tech_lead:replan')));
  assert.ok(logs.some((line) => line.includes('re-planning once')));
  // The re-prompt has to state the refusal and the budget behind it, and ask for a count
  // that reads correctly at this run's direction cap of 1.
  const replan = prompts.find((p) => p.label.startsWith('tech_lead:replan'));
  assert.ok(replan.prompt.includes('STOP_REFUSED'));
  assert.ok(replan.prompt.includes('a materially new direction'));
  const worker = prompts.find((p) => p.label.startsWith('variation'));
  assert.ok(worker, 'the round must still dispatch a worker');
  assert.ok(worker.prompt.includes('replanned fixture probe'));
  assert.equal(worker.prompt.includes('team_escape'), false);
});

test('a TechLead that will not re-plan still leaves the round an escape direction', async () => {
  const { labels, prompts } = await runSmoke('team', {
    techLeadStop: true, techLeadReplanStops: true,
    extraArgs: { team_stop_confirmations: 1 },
  });
  assert.ok(labels.some((label) => label.startsWith('tech_lead:replan')));
  const worker = prompts.find((p) => p.label.startsWith('variation'));
  assert.ok(worker, 'the round must still dispatch a worker');
  assert.ok(worker.prompt.includes('team_escape'));
});

test('a stop that will be honored is not re-planned', async () => {
  // early_stop on means the stop ends the run, so spending an extra planning call on it
  // would be pure waste.
  const { labels } = await runSmoke('team', {
    techLeadStop: true, extraArgs: { early_stop: 'true' },
  });
  assert.equal(labels.some((label) => label.startsWith('tech_lead:replan')), false);
});

test('team operator refuses a lone TechLead stop and still spends the round', async () => {
  // early_stop defaults off, which promises the full budget. One pessimistic
  // planning round must not end the run, and the refused round has to carry real
  // work rather than spin empty.
  const { result, labels, logs } = await runSmoke('team', { techLeadStop: true });
  assert.equal(result.validation_status, 'accepted');
  assert.ok(labels.some((label) => label.startsWith('variation')));
  assert.ok(logs.some((line) => line.includes('REFUSED [awaiting_confirmations]')));
});

// A patch that really applies: the workflow skips verify for a candidate whose
// patch fails `git apply --check`, so a missing file would hide every case below.
const FIXTURE_PATCH = path.join(mkdtempSync(path.join(tmpdir(), 'avo-patch-')), 'candidate.patch');
writeFileSync(FIXTURE_PATCH, 'diff --git a/new.txt b/new.txt\nnew file mode 100644\n'
  + 'index 0000000..ce01362\n--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+hello\n');
process.on('exit', () => rmSync(path.dirname(FIXTURE_PATCH), { recursive: true, force: true }));

const okVariation = () => ({
  status: 'ok', strategy: 'fixture', patch_file: FIXTURE_PATCH,
  patch_hash: 'abc', self_speedup: 1.2, per_case: [], rationale: 'fixture',
});

test('a harness that cannot measure aborts quickly instead of burning the budget', async () => {
  // The rig itself reports it could not measure. Retrying is pointless, so the
  // run must stop at the infra threshold rather than spend all 8 steps logging
  // "no improvement".
  const { labels } = await runSmoke('single', {
    extraArgs: { max_steps: 8, max_infra_failures: 2 },
    variation: okVariation,
    verify: () => ({
      status: 'failed', correctness: 'pass', verified_geomean: 0,
      verified_arithmetic: 0, infrastructure_status: 'infrastructure_unhealthy',
      per_case: [],
    }),
  });
  assert.equal(labels.filter((label) => label.startsWith('verify s')).length, 2);
});

test('workers returning unusable evidence get a looser budget than a dead rig', async () => {
  // A rejected report set is usually the worker's mistake (wrong session id,
  // wrong repeat count), not a broken harness, so the next worker can still
  // succeed. It must not trip the low infra threshold — but it is still bounded.
  const { labels } = await runSmoke('single', {
    extraArgs: { max_steps: 20, max_infra_failures: 2 },
    variation: okVariation,
    verify: () => ({
      status: 'failed', correctness: 'pass', verified_geomean: 0,
      verified_arithmetic: 0, infrastructure_status: 'infrastructure_unhealthy',
      reported_infrastructure_status: 'healthy',
      evidence_rejection: 'screening report missing', per_case: [],
    }),
  });
  const verifies = labels.filter((label) => label.startsWith('verify s')).length;
  assert.ok(verifies > 2, `expected more than the infra threshold, got ${verifies}`);
  assert.equal(verifies, 6);
});

for (const operator of ['single', 'team']) {
  test(`a capture that moves mid-run stops the ${operator} run at once`, async () => {
    // The capture is shared, so once it drifts every worker's evidence is void —
    // including evidence already banked. Charging that to the per-round evidence
    // budget spends six rounds and hours of GPU time to conclude "something moved".
    const { labels, logs } = await runSmoke(operator, {
      extraArgs: { max_steps: 20, max_infra_failures: 2 },
      variation: okVariation,
      verify: () => ({
        status: 'failed', correctness: 'pass', verified_geomean: 0,
        verified_arithmetic: 0, infrastructure_status: 'infrastructure_unhealthy',
        reported_infrastructure_status: 'healthy',
        evidence_rejection: 'formal report artifact identity does not match '
          + 'workspace files: capture /caps/c now hashes aaa but the report recorded bbb',
        evidence_rejection_component: 'capture',
        per_case: [],
      }),
    });
    assert.equal(labels.filter((label) => label.startsWith('verify')).length, 1,
      'the run kept dispatching workers after the capture moved');
    assert.ok(logs.some((line) => /capture changed during the run/.test(line)),
      `no capture-drift abort in ${JSON.stringify(logs.slice(-4))}`);
  });
}

test('a direction measured below the commit gate is not retired on first sight', async () => {
  // "Correct but not fast enough" is the normal outcome of most attempts.
  // Retiring on the first occurrence drains the TechLead's plan within a couple
  // of rounds and collapses every later round into one generic escape worker.
  const { prompts } = await runSmoke('team', {
    extraArgs: { max_steps: 2, max_infra_failures: 0 },
    variation: () => ({
      status: 'ok', strategy: 'fixture', patch_file: FIXTURE_PATCH,
      patch_hash: 'abc', self_speedup: 1.2, per_case: [], rationale: 'fixture',
    }),
    verify: () => ({
      status: 'ok', correctness: 'pass', verified_geomean: 1.0,
      verified_arithmetic: 1.0, infrastructure_status: 'healthy', per_case: [],
    }),
  });
  const worker = prompts.filter((entry) => entry.label.startsWith('variation'));
  assert.ok(worker.length >= 2);
  assert.equal(
    worker.every((entry) => !entry.prompt.includes('team_escape')), true);
});

test('one round cannot retire a direction just by handing it to several workers', async () => {
  // A round can aim several workers at one direction name — the planner repeats a
  // title, or more start points than directions make the pairing cycle the same
  // direction. That produces several "below the gate" reports for one name in a
  // single round. Retirement means "failed again in a LATER round", so the round
  // that first tried it must leave it alive.
  const { prompts } = await runSmoke('team', {
    extraArgs: { max_steps: 2, operator_directions: 2 },
    directions: [
      { id: 'd0', title: 'shared title', specialty: 'algorithm',
        focus_files: ['kernel.py'], expected_speedup: 1.01, prompt: 'worker a' },
      { id: 'd1', title: 'shared title', specialty: 'memory',
        focus_files: ['kernel.py'], expected_speedup: 1.01, prompt: 'worker b' },
    ],
    variation: okVariation,
    verify: () => ({
      status: 'ok', correctness: 'pass', verified_geomean: 1.0,
      verified_arithmetic: 1.0, infrastructure_status: 'healthy', per_case: [],
    }),
  });
  // Round 2's plan carries the failure memory built by round 1.
  const planned = prompts.filter((entry) => entry.label.startsWith('tech_lead:plan'));
  assert.ok(planned.length >= 2, `expected two planning rounds, got ${planned.length}`);
  const failedInRound2 = JSON.parse(planned[1].prompt.match(/"failed":\s*(\[[^\]]*\])/)[1]);
  assert.deepEqual(failedInRound2, [],
    'the direction was retired inside the round that first tried it');
});

test('resume recomputes the commit-gate base instead of trusting the Director', async () => {
  // Every field below is transcribed by the Director LLM from the same files
  // avo_lineage.py reads. A low `cumulative` would let a regression clear the
  // gate and be recorded as a new best, so the deterministic value must win.
  const { logs } = await runSmoke('single', {
    extraArgs: { max_steps: 1 },
    seedState: ({ evalDir }) => {
      const state = path.join(evalDir, 'avo_state');
      mkdirSync(state, { recursive: true });
      writeFileSync(path.join(state, 'lineage.json'), JSON.stringify({
        best_id: 'v3',
        committed: [
          { id: 'v0', git_ref: 'avo-v0', strategy: 'baseline', score: { speedup: 1.0 } },
          { id: 'v3', git_ref: 'avo-v3', strategy: 'tiling', score: { speedup: 1.42 } },
        ],
      }));
      writeFileSync(path.join(state, 'baseline.json'), JSON.stringify({
        per_case: [{ name: 'case', baseline_ms: 2, optimized_ms: 2, speedup: 1 }],
        geomean_ms: 2,
      }));
    },
    setupResult: ({ evalDir, repo }) => ({
      resume: true, eval_dir: evalDir, avo_repo: repo,
      baseline_dir: path.join(evalDir, 'baseline'), kernel_name: 'fixture',
      source_files: [],
      // The Director under-reports the standing best.
      cumulative: 1.05, best_version: 'avo-v1', commits: 1,
      baseline_per_case: [{ name: 'case', baseline_ms: 9, optimized_ms: 9, speedup: 1 }],
      baseline_geomean_ms: 9,
    }),
    variation: okVariation,
    verify: () => ({
      status: 'ok', correctness: 'pass', verified_geomean: 1.1,
      verified_arithmetic: 1.1, infrastructure_status: 'healthy', per_case: [],
    }),
  });
  const resumeLine = logs.find((line) => line.startsWith('RESUME: best='));
  assert.ok(resumeLine, `no RESUME line in ${JSON.stringify(logs.slice(0, 12))}`);
  assert.match(resumeLine, /best=avo-v3 \(1\.4200x\)/);
  assert.ok(logs.some((line) => /corrected cumulative 1\.05 -> 1\.42/.test(line)),
    'the Director/lineage disagreement was not reported');
});

// An unreachable agent backend is not a broken rig. Charging it to the rig budget
// ended a 19.5h run: one API blip returned four null workers, which read as "every
// worker hit a broken harness" and tripped the deliberately-low infra threshold.
test('a dead agent backend is not charged to the rig budget', async () => {
  const { logs } = await runSmoke('team', {
    extraArgs: { max_steps: 2, max_infra_failures: 2, max_agent_outages: 5 },
    agentNull: (label) => label.startsWith('variation'),
  });
  assert.ok(logs.some((line) => line.includes('AGENT OUTAGE')));
  assert.equal(logs.some((line) => line.includes('INFRA FAILURE')), false);
  assert.equal(logs.some((line) => line.includes('persistent infra failure')), false);
});

test('a dead agent backend still aborts once its own budget is spent', async () => {
  const { logs } = await runSmoke('team', {
    extraArgs: { max_steps: 6, max_agent_outages: 2 },
    agentNull: (label) => label.startsWith('variation'),
  });
  assert.ok(logs.some((line) => line.includes('AGENT OUTAGE (2/2)')));
  assert.ok(logs.some((line) => line.includes('agent backend stayed unreachable')));
});

test('a team verifier outage is charged to the agent budget, not the rig', async () => {
  const { logs } = await runSmoke('team', {
    extraArgs: { max_steps: 4, max_agent_outages: 2 },
    variation: okVariation,
    agentNull: (label) => label.startsWith('verify variation_'),
  });
  assert.ok(logs.some((line) => line.includes('AGENT OUTAGE (2/2)')));
  assert.ok(logs.some((line) => line.includes('agent backend stayed unreachable')));
  assert.equal(logs.some((line) => line.includes('INFRA FAILURE')), false);
});

test('ESCALATE applies the same capture-drift abort as a normal team round', async () => {
  const { labels, logs } = await runSmoke('single', {
    extraArgs: {
      max_steps: 8,
      stall_no_improve: 1,
      supervisor_cycles_to_escalate: 1,
      escalate_workers: 1,
    },
    variation: (label) => label.startsWith('escalate_')
      ? okVariation()
      : {
        status: 'no_change', strategy: 'fixture', patch_file: '',
        patch_hash: '', self_speedup: 1, per_case: [], rationale: 'force supervisor',
      },
    verify: (label) => label.startsWith('verify escalate_') ? ({
      status: 'failed', correctness: 'pass', verified_geomean: 0,
      verified_arithmetic: 0, infrastructure_status: 'infrastructure_unhealthy',
      reported_infrastructure_status: 'healthy',
      evidence_rejection: 'formal report artifact identity does not match '
        + 'workspace files: capture changed during ESCALATE',
      evidence_rejection_component: 'capture',
      per_case: [],
    }) : null,
  });
  assert.equal(
    labels.filter((label) => label.startsWith('verify escalate_')).length, 1);
  assert.ok(logs.some((line) => /capture changed during the run/.test(line)),
    `no ESCALATE capture-drift abort in ${JSON.stringify(logs.slice(-6))}`);
  assert.equal(
    labels.filter((label) => label.startsWith('variation s')).length, 1,
    'the run continued after ESCALATE detected shared capture drift');
});

test('ESCALATE verifier outages consume the agent budget, not the rig budget', async () => {
  const { logs } = await runSmoke('single', {
    extraArgs: {
      max_steps: 8,
      stall_no_improve: 1,
      supervisor_cycles_to_escalate: 1,
      escalate_workers: 1,
      max_agent_outages: 1,
    },
    variation: (label) => label.startsWith('escalate_')
      ? okVariation()
      : {
        status: 'no_change', strategy: 'fixture', patch_file: '',
        patch_hash: '', self_speedup: 1, per_case: [], rationale: 'force supervisor',
      },
    agentNull: (label) => label.startsWith('verify escalate_'),
  });
  assert.ok(logs.some((line) => line.includes('AGENT OUTAGE (1/1)')));
  assert.ok(logs.some((line) => line.includes('agent backend stayed unreachable')));
  assert.equal(logs.some((line) => line.includes('INFRA FAILURE')), false);
});

// Seed a real two-tag lineage so avo-v0..avo-v1 is a genuine git diff.
const seedGitLineage = ({ evalDir, repo }) => {
  const git = (...a) => execFileSync('git', ['-C', repo, ...a], { encoding: 'utf8' });
  execFileSync('git', ['init', '-q', repo]);
  git('config', 'user.email', 'avo@test');
  git('config', 'user.name', 'avo');
  writeFileSync(path.join(repo, 'kernel.s'), 'v_mov_b32 v0, 0\n');
  git('add', '-A'); git('commit', '-qm', 'baseline'); git('tag', 'avo-v0');
  writeFileSync(path.join(repo, 'kernel.s'), 'v_mov_b32 v0, 0 nt\n');
  git('add', '-A'); git('commit', '-qm', 'avo-v1'); git('tag', 'avo-v1');
  const state = path.join(evalDir, 'avo_state');
  mkdirSync(state, { recursive: true });
  writeFileSync(path.join(state, 'lineage.json'), JSON.stringify({
    best_id: 'v1',
    committed: [
      { id: 'v0', git_ref: 'avo-v0', strategy: 'baseline', score: { speedup: 1.0 } },
      { id: 'v1', git_ref: 'avo-v1', strategy: 'nt', score: { speedup: 1.3122 } },
    ],
  }));
  writeFileSync(path.join(state, 'baseline.json'), JSON.stringify({
    per_case: [{ name: 'case', baseline_ms: 2, optimized_ms: 2, speedup: 1 }],
    geomean_ms: 2,
  }));
};
const resumeAtV1 = ({ evalDir, repo }) => ({
  resume: true, eval_dir: evalDir, avo_repo: repo,
  baseline_dir: path.join(evalDir, 'baseline'), kernel_name: 'fixture',
  source_files: [], cumulative: 1.3122, best_version: 'avo-v1', commits: 1,
  baseline_per_case: [{ name: 'case', baseline_ms: 2, optimized_ms: 2, speedup: 1 }],
  baseline_geomean_ms: 2,
});

// The cumulative patch is a diff between two lineage tags, so an unreachable
// reporter must not discard a run whose best is already committed and tagged.
test('an unreachable reporter falls back to the deterministic lineage diff', async () => {
  const { result, logs } = await runSmoke('single', {
    extraArgs: { max_steps: 1 },
    seedState: seedGitLineage, setupResult: resumeAtV1,
    agentNull: (label) => label === 'tech_lead:report',
  });
  // The line is only logged after a non-empty avo-v0..avo-v1 diff was written, so it
  // also proves the synthesis ran (runSmoke removes the tmp tree before returning).
  const synthesized = logs.find((line) => line.includes('synthesized avo-v0..avo-v1'));
  assert.ok(synthesized, `no synthesis line in ${JSON.stringify(logs.slice(-8))}`);
  assert.ok(result.final_patch.endsWith('final_patch.diff'));
  assert.match(synthesized, /deterministically at .*final_patch\.diff/);
});

test('an unreachable Director reports the committed lineage as unvalidated', async () => {
  const { result, logs, labels, heartbeats, artifactSnapshot } = await runSmoke('single', {
    extraArgs: { max_steps: 1 },
    seedState: seedGitLineage, setupResult: resumeAtV1,
    agentNull: (label) => label === 'director:validate',
  });
  assert.equal(result.validation_status, 'validation_unavailable');
  assert.equal(result.final_geomean, 0);
  // The durable facts of the run must survive the outage.
  assert.equal(result.best_version, 'avo-v1');
  assert.equal(result.num_commits, 1);
  assert.ok(result.final_patch);
  assert.ok(result.output_bundle.endsWith('/final_artifacts'));
  assert.ok(artifactSnapshot.every((item) => item.exists));
  assert.ok(labels.includes('director:cleanup'));
  assert.ok(heartbeats.some((heartbeat) => heartbeat.phase === 'cleanup'));
  assert.ok(logs.some((line) => line.includes('UNVALIDATED')));
});

test('supervisor backtrack changes one start ref without lowering the global gate', async () => {
  const { labels, prompts, logs } = await runSmoke('team', {
    extraArgs: { max_steps: 2, stall_no_improve: 1, start_k: 1 },
    seedState: seedGitLineage,
    setupResult: resumeAtV1,
    supervisorDirective: {
      diagnosis: 'probe the baseline branch',
      mark_failed: [],
      new_directions: [{ name: 'baseline probe', expected: 'orthogonal',
        prompt: 'try a source change from the older base' }],
      backtrack_to: 'avo-v0',
      converge: false,
    },
  });
  assert.ok(labels.some((label) =>
    label.startsWith('variation_0002') && label.endsWith('@avo-v0')));
  const secondWorker = prompts.find((entry) => entry.label.startsWith('variation_0002'));
  assert.match(secondWorker.prompt, /CUMULATIVE_SPEEDUP[\s\S]{0,40}1\.3122/);
  assert.ok(logs.some((line) =>
    line.includes('global champion/gate remains avo-v1 @ 1.3122x')));
});

test('campaign budget exposes an explicit final-phase reserve', async () => {
  const { logs } = await runSmoke('single', {
    extraArgs: { final_phase_reserve_s: 12 },
  });
  assert.ok(logs.some((line) =>
    line.includes('Campaign budget=60s, final-phase reserve=12s')));
});
