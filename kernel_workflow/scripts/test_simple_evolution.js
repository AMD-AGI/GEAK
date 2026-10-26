#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CPU-only integration checks: the real lane, archive, materializer and Git patches.
// Agent answers are scripted; no model, GPU, network or installed workflow is used.
// Run after scripts/build_workflows.mjs: node kernel_workflow/scripts/test_simple_evolution.js
'use strict';
const assert = require('assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const {spawnSync} = require('child_process');
const ROOT = path.resolve(__dirname, '../..');
const lanePath = process.argv.includes('--lane')
  ? path.resolve(process.argv[process.argv.indexOf('--lane') + 1])
  : path.join(ROOT, 'kernel_workflow/kernel_lane.js');
const laneSource = fs.readFileSync(lanePath, 'utf8');
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const original = {'k.py': 'A = 0\nB = 0\n'};
const source = (a, b, extra = {}) => ({'k.py': `A = ${a}\nB = ${b}\n`, ...extra});
let validateSchema;

function command(cmd, args, options = {}) {
  const r = spawnSync(cmd, args, {encoding: 'utf8', timeout: 10000, ...options});
  assert.equal(r.status, 0, `${cmd} ${args.join(' ')}: ${r.error || r.stderr}`);
  return r.stdout;
}
const git = (dir, ...args) => command('git', ['-C', dir, ...args]);
function writeSources(dir, files) {
  fs.mkdirSync(dir, {recursive: true});
  for (const [name, contents] of Object.entries(files)) fs.writeFileSync(path.join(dir, name), contents);
}
function readSources(dir) {
  return Object.fromEntries(fs.readdirSync(dir).filter(name => /\.(py|bin)$/.test(name)).sort()
    .map(name => [name, fs.readFileSync(path.join(dir, name), name.endsWith('.bin') ? null : 'utf8')]));
}
function initRepo(dir) {
  git(dir, 'init', '-q');
  git(dir, 'config', 'user.name', 'Saptarshi Majumder');
  git(dir, 'config', 'user.email', '32464366+iraj465@users.noreply.github.com');
  git(dir, 'add', '-A');
  git(dir, 'commit', '-qm', 'CPU fixture baseline');
}

// Reuse the existing greedy harness without editing it or its golden. Inject only
// host runtime dependencies; its agent call/probe/phase recording remains unchanged.
function harness(hostGlobals) {
  const file = path.join(__dirname, 'test_greedy_parity.js');
  const src = fs.readFileSync(file, 'utf8');
  const marker = '\n(async () => {\n  const src = fs.readFileSync(LANE';
  assert.equal(src.split(marker).length, 2, 'greedy harness entrypoint changed');
  const prefix = src.split(marker)[0].replace(/^#![^\n]*\n/, '')
    .replace('    args: argsObj,', '    ...hostGlobals, args: argsObj,')
    .replace('  const fn = new Function',
      '  g.agent.processDeadlineActive = !!hostGlobals.setProcessDeadline;\n  const fn = new Function');
  return new Function('require', '__dirname', 'process', 'hostGlobals',
    `${prefix}\nreturn {instrument, build, answers, field, perCase, verified, BASE_ARGS, dir};`)
    (require, __dirname, {argv: []}, hostGlobals);
}

async function runCase(sc) {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'geak-evolution-test-'));
  const task = path.join(temp, 'task'), evalDir = path.join(temp, 'eval');
  const populationRoot = path.join(temp, 'population');
  const canonical = path.join(evalDir, 'workspace');
  const hostFinalPatch = path.join(evalDir, 'host_final_patch.diff');
  const finalFiles = sc.finalFiles || sc.expected;
  const finalSpeed = sc.finalSpeed ?? sc.best;
  writeSources(task, original);
  initRepo(task);
  fs.mkdirSync(evalDir);
  command('git', ['clone', '-q', task, canonical]);
  git(canonical, 'config', 'user.name', 'Saptarshi Majumder');
  git(canonical, 'config', 'user.email', '32464366+iraj465@users.noreply.github.com');
  const originalCommit = git(canonical, 'rev-parse', 'HEAD').trim();
  if (sc.population) {
    fs.mkdirSync(populationRoot);
    fs.writeFileSync(path.join(populationRoot, 'manifest.json'), JSON.stringify({
      schema: 'geak.population/1', root: populationRoot, kernel_path: task,
      objective: {kind: 'geomean'}, lineages: [sc.population.id, ...sc.population.imports].map(id => ({
        id, optimizer: id === sc.population.id ? sc.optimizer || 'evo' : 'greedy',
        imports: [], eval_dir: id === sc.population.id ? evalDir : path.join(temp, id, 'eval')})),
    }));
  }
  let now = 1000, deadline = Infinity;
  const cutoff = epoch => { deadline = epoch; };
  cutoff.secondsLeft = () => deadline - now;
  const hostCalls = [], events = [], errors = [], schemas = new Map(), authoredGenomes = new Map();
  const externalRecords = new Map(), externalReads = [];
  const objective = {kind: 'external_primary', primary_regime: 'cpu_fixture'};
  const sourceIdentity = files => hash(JSON.stringify(Object.entries(files).sort()));
  let finalMeasured = false;
  const registerExternal = (id, workspace, speed, development) => {
    const files = readSources(workspace), candidate = sourceIdentity(files);
    const ref = {candidate_id: candidate, evaluation_id: id};
    externalRecords.set(id, {files, value: {...h.verified(development), objective,
      source_id: candidate, source_paths: Object.keys(files),
      external_evaluation: ref, external_measurement_valid: true,
      primary_regime: objective.primary_regime, primary_speedup: speed,
      development_geomean_speedup: development, native_reproducible_gain: speed > 1,
      native_reproducible_regression: speed < 1, timing_basis: 'cpu_fixture',
      status: speed <= 1 ? 'regression' : 'verified', validation_status: 'accepted'}});
    return ref;
  };
  const hostEnv = {...process.env,
    GIT_AUTHOR_NAME: 'Saptarshi Majumder', GIT_COMMITTER_NAME: 'Saptarshi Majumder',
    GIT_AUTHOR_EMAIL: '32464366+iraj465@users.noreply.github.com',
    GIT_COMMITTER_EMAIL: '32464366+iraj465@users.noreply.github.com'};
  const h = harness({
    runProcess: async request => {
      hostCalls.push([request.cmd, ...request.args]);
      if (sc.external && request.cmd.startsWith('/cpu-fixture/external-')) {
        try {
          const field = key => request.args[request.args.indexOf(key) + 1];
          const workspace = field('--workspace');
          const completed = value => ({status: 'completed', code: 0, group_cleared: true,
            strays_killed: false, stdio_held: false, stdout: JSON.stringify(value)});
          if (request.args.includes('--capture-only')) {
            assert.equal(request.cmd, '/cpu-fixture/external-evaluate');
            if (sc.finalFiles) {
              // Model the last edit already present when native capture takes its snapshot.
              writeSources(workspace, sc.finalFiles);
              if (sc.stagedBinary) {
                git(workspace, 'add', '--', sc.stagedBinary);
                assert.match(git(workspace, 'diff', '--cached', '--binary'), /GIT binary patch/,
                  'final capture includes a newly staged binary source');
              }
              assert.equal(git(workspace, 'show', 'HEAD:k.py'), sc.expected['k.py'],
                'final edit is uncommitted and differs from the retained winner HEAD');
              assert.notEqual(git(workspace, 'diff', '--binary', 'HEAD'), '');
            }
            assert.deepEqual(readSources(workspace), finalFiles, 'capture exact final worktree source');
            const ref = registerExternal('final', workspace, finalSpeed, sc.finalDev ?? sc.bestDev);
            // A race after source capture can change the following Git diff. The final
            // reader must reject that patch even if the captured source is later restored.
            if (sc.patchSourceMismatch) writeSources(workspace, sc.patchSourceMismatch);
            return completed({schema: 'geak-native-measurement-identity-v1', capture_only: true,
              ...ref, source_id: ref.candidate_id, source_paths: Object.keys(finalFiles)});
          }
          const id = field('--evaluation-id'), record = externalRecords.get(id);
          assert.ok(record, 'reader/evaluator requires an independently retained measurement');
          assert.equal(field('--candidate-id'), sourceIdentity(record.files), 'identity binds measured bytes');
          const patch = request.args.includes('--patch') ? field('--patch') : null;
          if (patch) assert.deepEqual(readSources(workspace), original, 'trusted patch reader starts at ORIGINAL');
          const files = patch ? copyAndApply(workspace, patch,
            path.join(temp, `external-replay-${externalReads.length}`)) : readSources(workspace);
          if (id === 'final' && patch) {
            assert.equal(patch, hostFinalPatch, 'final trusted reader always receives the host patch');
            if (sc.patchSourceMismatch) {
              assert.equal(finalMeasured, true, 'mismatch is checked after host measurement');
              assert.deepEqual(files, sc.patchSourceMismatch);
              assert.notDeepEqual(files, record.files, 'replayed patch differs from captured source');
              externalReads.push({id, workspace, patch, source_id: sourceIdentity(files), rejected: true});
              return {status: 'completed', code: 1, group_cleared: true,
                stdout: '', stderr: 'patch does not reconstruct the captured source'};
            }
          }
          assert.deepEqual(files, record.files, 'trusted reader reconstructs exact measured source bytes');
          assert.equal(sourceIdentity(files), record.value.external_evaluation.candidate_id);
          if (request.cmd === '/cpu-fixture/external-evaluate') {
            assert.equal(id, 'final');
            assert.equal(events.filter(e => e.label === 'director:validate').length, 0,
              'final host evaluation precedes Director review');
            finalMeasured = true;
            return completed({completed: true});
          }
          assert.equal(request.cmd, '/cpu-fixture/external-read');
          if (id === 'final') assert.equal(finalMeasured, true, 'final read requires completed host evaluation');
          externalReads.push({id, workspace, patch, source_id: sourceIdentity(files)});
          return completed({...record.value, patch_sha256: patch ? hash(fs.readFileSync(patch)) : null});
        } catch (error) { errors.push(error); throw error; }
      }
      // run_workflow.mjs also bounds host commands, not just agent processes.
      const cleanupMs = (request.graceMs ?? 5000) + (request.settleMs ?? 2000);
      if (sc.bounded && (deadline - now) * 1000 <= cleanupMs) {
        throw Object.assign(new Error('CPU fixture host cutoff'), {code: 'GEAK_PROCESS_DEADLINE'});
      }
      const r = spawnSync(request.cmd, request.args, {
        encoding: 'utf8', timeout: 10000, input: request.input,
        cwd: request.cwd, env: {...hostEnv, ...(request.env || {})},
      });
      return {status: r.error ? 'failed' : 'completed', code: r.status,
        stdout: r.stdout || '', stderr: r.stderr || '', group_cleared: true,
        strays_killed: false, stdio_held: false};
    },
    ...(sc.bounded ? {setProcessDeadline: cutoff} : {}),
  });
  const fallback = h.answers(sc);
  const copyAndApply = (base, patch, destination) => {
    fs.mkdirSync(destination, {recursive: true});
    for (const [name, contents] of Object.entries(readSources(base))) {
      fs.writeFileSync(path.join(destination, name), contents);
    }
    initRepo(destination);
    if (fs.readFileSync(patch).length) git(destination, 'apply', '--', patch);
    return readSources(destination);
  };
  const bashBlock = prompt => {
    const block = prompt.match(/```bash\n([\s\S]*?)\n```/);
    assert.ok(block, 'actual agent prompt must contain setup/commit commands');
    command('bash', ['-e', '-c', block[1]]);
  };
  const answer = (label, prompt) => {
    try {
      if (sc.bounded && deadline - now <= 7) {
        throw Object.assign(new Error('CPU fixture process cutoff'), {code: 'GEAK_PROCESS_DEADLINE'});
      }
      events.push({label, prompt}); // Calls admitted by the modeled runtime cutoff.
      if (label === 'director:setup') return {...fallback(label, prompt),
        eval_dir: evalDir, workspace: canonical, source_files: ['k.py']};
      if (label === 'tech_lead:analyze') return {...fallback(label, prompt),
        modifiable_files: sc.modifiableFiles || ['k.py', 'inherited.py']};
      if (label === 'benchmark_engineer') return {...fallback(label, prompt),
        commandment_path: path.join(evalDir, 'COMMANDMENT.md'),
        ...(sc.external ? {external_evaluation: registerExternal('original', canonical, 1, 1)} : {})};
      if (label.startsWith('tech_lead:plan ') || label.startsWith('tech_lead:replan ')) {
        const round = label.split(' ')[1];
        if (sc.expireAfterPlan === round) now = deadline;
        if (sc.inspectPlan) sc.inspectPlan({label, prompt, h, temp, populationRoot, canonical});
        return fallback(label, prompt);
      }
      if (label.startsWith('tech_lead:memory ') && sc.memory) return sc.memory({label, prompt, h});
      if (label.startsWith('eng ') || label.startsWith('deep ')) {
        const id = label.split(' ')[1].split(':')[0], spec = sc.candidates[id];
        assert.ok(spec, `unexpected engineer ${id}`);
        const base = h.field(prompt, 'CANONICAL');
        const baselineFiles = sc.optimizer === 'greedy' ? readSources(canonical) : original;
        assert.deepEqual(readSources(base), baselineFiles, 'engineer receives its optimizer baseline');
        assert.deepEqual(h.field(prompt, 'baseline_per_case'), h.perCase(1));
        // Deliberately execute no bootstrap instructions: the host must already
        // have materialized the selected source before this model call starts.
        const workspace = h.field(prompt, 'KERNEL_PATH');
        if (sc.optimizer === 'greedy') bashBlock(prompt);
        assert.deepEqual(readSources(workspace), spec.inherited || baselineFiles, `${id} inherited source`);
        const evolution = h.field(prompt, 'EVOLUTION');
        assert.equal(h.field(prompt, 'CONTINUING_LINEAGE'), sc.population && sc.optimizer !== 'greedy' ? true : undefined);
        if (sc.optimizer !== 'greedy') {
        assert.equal(Object.hasOwn(evolution, 'selected_evidence'),
          sc.args?.evolution_evidence_handoff === true && !spec.handoffOverLimit,
          'only explicit true with an in-budget payload enables the prompt handoff');
        assert.equal(Object.hasOwn(evolution, 'selected_evidence_omitted'),
          sc.args?.evolution_evidence_handoff === true && !!spec.handoffOverLimit,
          'over-budget delivery is explicit');
        }
        const anchor = sc.optimizer === 'greedy' ? git(workspace, 'rev-parse', 'HEAD').trim()
          : evolution.workspace_base_commit;
        assert.match(anchor, /^[0-9a-f]{40,64}$/, 'host supplies the fixed cumulative-patch anchor');
        assert.equal(git(workspace, 'show', `${anchor}:k.py`), baselineFiles['k.py'], 'patch anchor matches optimizer baseline');
        assert.equal(git(workspace, 'show', 'HEAD:k.py'), (spec.inherited || baselineFiles)['k.py'],
          'selected parent is the committed starting HEAD');
        if (spec.inspect) spec.inspect({h, prompt, canonical,
          replay: patch => copyAndApply(task, patch, path.join(temp, `donor-replay-${id}`))});
        if (spec.commitDuringEngineering) {
          writeSources(workspace, spec.commitDuringEngineering);
          git(workspace, 'add', '-A');
          command('git', ['-C', workspace, 'commit', '-qm', 'CPU fixture intermediate engineering commit'],
            {env: hostEnv});
          assert.notEqual(git(workspace, 'rev-parse', 'HEAD').trim(), anchor);
        }
        writeSources(workspace, spec.files);
        const patch = path.join(h.field(prompt, 'OUTPUT_DIR'), 'best_patch.diff');
        if (!spec.noPatch) {
          // The ordinary engineer's save instruction includes new editable sources.
          git(workspace, 'add', '-N', '--', ...Object.keys(spec.files));
          fs.writeFileSync(patch, git(workspace, 'diff', '--binary',
            ...(spec.patchOptions || []), anchor, '--', '.'));
        }
        if (spec.engineerObservationFiles) writeSources(workspace, spec.engineerObservationFiles);
        if (sc.expireAfterEngineer === id) now = deadline;
        const author = spec.engineer || {status: 'ok', measurement_valid: true, speedup_geomean: spec.speed,
          ...(sc.external ? {external_evaluation: registerExternal(`eng-${id}`, workspace,
            spec.speed, spec.development)} : {})};
        if (!Object.hasOwn(spec, 'genome')) return author;
        const designGenome = typeof spec.genome === 'function'
          ? spec.genome({id, patch, workspace, anchor, h, prompt, temp, authoredGenomes}) : spec.genome;
        authoredGenomes.set(id, designGenome);
        const response = {...author, design_genome: designGenome};
        assert.deepEqual(validateSchema(response, schemas.get(label)), {ok: true, errors: []},
          'optional author JSON passes the actual runtime schema validator');
        return response;
      }
      if (label.startsWith('verify ')) {
        const id = label.split(' ')[1], spec = sc.candidates[id];
        assert.ok(spec, `unexpected verifier ${id}`);
        assert.deepEqual(readSources(h.field(prompt, 'CANONICAL')),
          sc.optimizer === 'greedy' ? readSources(canonical) : original, 'verifier uses optimizer baseline');
        assert.deepEqual(h.field(prompt, 'BASELINE_PER_CASE'), h.perCase(1));
        if (spec.noPatch) return {status: 'apply_failed', notes: 'patch absent'};
        assert.deepEqual(copyAndApply(h.field(prompt, 'CANONICAL'), h.field(prompt, 'PATCH'),
          h.field(prompt, 'VERIFY_DIR')), spec.files, `${id} cumulative patch reconstructs exact source`);
        if (spec.mutateAfterVerification) {
          const replacement = path.join(temp, `replaced-patch-${id}`);
          writeSources(replacement, original);
          initRepo(replacement);
          writeSources(replacement, spec.mutateAfterVerification);
          fs.writeFileSync(h.field(prompt, 'PATCH'), git(replacement, 'diff', '--binary'));
        }
        return spec.verifier || {...h.verified(spec.speed), status: spec.speed <= 1 ? 'regression' : 'verified',
          ...(sc.external ? {external_evaluation: registerExternal(`verify-${id}`,
            h.field(prompt, 'VERIFY_DIR'), spec.speed, spec.development)} : {})};
      }
      if (label.startsWith('commit ')) {
        const round = label.split(' ')[1];
        if (sc.badCommit === round) {
          writeSources(canonical, source(999, 999, {'unfinished.py': 'WRONG = True\n'}));
          git(canonical, 'add', '-A');
          git(canonical, 'commit', '-qm', 'Deliberately wrong CPU fixture commit');
          return {committed: true}; // Real tree check must reject the false receipt.
        }
        bashBlock(prompt);
        return {committed: true};
      }
      if (label.startsWith('integrate ') && sc.integration) {
        const base = h.field(prompt, 'CANONICAL');
        assert.deepEqual(readSources(base), original, 'integrator uses the frozen base');
        const workspace = path.join(h.field(prompt, 'INTEGRATE_DIR'), 'workspace');
        writeSources(workspace, original);
        initRepo(workspace);
        writeSources(workspace, sc.integration.files);
        git(workspace, 'add', '-N', '--', ...Object.keys(sc.integration.files));
        const patch = path.join(h.field(prompt, 'INTEGRATE_DIR'), 'best.diff');
        fs.writeFileSync(patch, git(workspace, 'diff', '--binary'));
        const designGenome = typeof sc.integration.genome === 'function'
          ? sc.integration.genome({patch, workspace, h, prompt, temp, authoredGenomes}) : sc.integration.genome;
        if (Object.hasOwn(sc.integration, 'genome')) authoredGenomes.set('integrated', designGenome);
        const response = {attempted: true, conclusion: 'improved', best: {
          geomean: sc.integration.speed, arithmetic: sc.integration.speed,
          per_case: h.perCase(sc.integration.speed), patch_file: patch,
          ...(sc.external ? {external_evaluation: registerExternal(label, workspace,
            sc.integration.speed, sc.integration.development)} : {}),
          ...(Object.hasOwn(sc.integration, 'genome') ? {design_genome: designGenome} : {})}};
        assert.deepEqual(validateSchema(response, schemas.get(label)), {ok: true, errors: []},
          'integrated optional author JSON passes the actual runtime schema validator');
        return response;
      }
      if (label.startsWith('storage:reclaim')) {
        const keep = Number(label.match(/r(\d+)$/)[1]);
        for (let round = 1; round < keep; round++) {
          fs.rmSync(path.join(evalDir, `round_${round}`), {recursive: true, force: true});
        }
        return {ok: true, note: 'old fixture rounds removed'};
      }
      if (label === 'tech_lead:report') {
        if (sc.external) assert.ok(fs.existsSync(hostFinalPatch), 'host saves final patch before Report');
        if (sc.patchSourceMismatch) writeSources(canonical, finalFiles);
        if (sc.report === 'null') return null;
        const patch = path.join(evalDir, 'final_patch.diff');
        if (sc.report !== 'missing') fs.writeFileSync(patch,
          sc.report === 'wrong' ? 'not a patch\n' : git(canonical, 'diff', '--binary', originalCommit, 'HEAD'));
        if (sc.report === 'wrong') fs.writeFileSync(hostFinalPatch, 'Report overwrote the host patch\n');
        const stale = sc.report === 'stale' ? JSON.parse(fs.readFileSync(
          path.join(evalDir, 'evolution/archive.json'), 'utf8')).entries[0].patch_ref.path : null;
        return {final_speedup_geomean: h.field(prompt, 'CUMULATIVE_SPEEDUP'),
          ...(sc.report === 'missing' ? {} : {final_patch: stale || patch}),
          report_path: path.join(evalDir, 'report.md'),
          rounds: h.field(prompt, 'HISTORY').rounds.length};
      }
      if (label === 'director:validate') {
        if (sc.external) assert.equal(h.field(prompt, 'FINAL_PATCH'), hostFinalPatch,
          'Director reviews the host path independently of the Report return');
        assert.deepEqual(copyAndApply(task, h.field(prompt, 'FINAL_PATCH'), path.join(temp, 'final-replay')),
          sc.patchSourceMismatch || finalFiles, 'Director receives the patch frozen before Report');
        if (sc.report === 'wrong') fs.writeFileSync(hostFinalPatch, 'Director overwrote the host patch\n');
        return {director_verified_speedup_geomean: h.field(prompt, 'TECH_LEAD_REPORTED_GEOMEAN'),
          validation_status: 'accepted', timing_basis: 'cpu_fixture'};
      }
      return fallback(label, prompt);
    } catch (error) {
      if (error.code !== 'GEAK_PROCESS_DEADLINE') errors.push(error);
      throw error;
    }
  };
  let observed, passed = false;
  try {
    const body = h.instrument(laneSource);
    const args = {...h.BASE_ARGS, optimizer: sc.optimizer || 'evo', kernel_path: task,
      workflow_dir: path.join(ROOT, 'kernel_workflow'), ...sc.args,
      ...(sc.population ? {eval_dir: evalDir, population_lineage: {...sc.population, root: populationRoot},
        ...(sc.optimizer !== 'greedy' ? {continuing_lineage: true} : {})} : {}),
      ...(sc.external ? {objective, external_measurement_command: ['/cpu-fixture/external-evaluate'],
        external_result_command: ['/cpu-fixture/external-read']} : {}),
      ...(sc.bounded ? {deadline_epoch: 10000, search_reserve_s: 200,
        final_validation_reserve_s: 100, final_cleanup_reserve_s: 0} : {})};
    const {run, rec} = h.build(body, args, answer, (label, prompt, options) => {
      schemas.set(label, options.schema);
    });
    observed = rec;
    let result;
    try { result = await run(); } catch (error) { throw errors[0] || error; }
    if (errors.length) throw errors[0];
    const archive = JSON.parse(fs.readFileSync(path.join(evalDir, 'evolution/archive.json'), 'utf8'));
    assert.deepEqual(readSources(task), original, 'task source unchanged');
    assert.deepEqual(readSources(archive.base), original, 'archive base unchanged');
    assert.deepEqual(readSources(canonical), finalFiles, 'canonical holds the final captured source');
    assert.equal(result.final_speedup, sc.patchSourceMismatch ? null : finalSpeed,
      'final score is bound to the captured source relative to ORIGINAL');
    assert.equal(rec.after_loop.cumulative, sc.best);
    assert.deepEqual(rec.after_loop.bestPerCase, h.perCase(sc.bestDev || sc.best));
    assert.equal(events.filter(e => e.label === 'tech_lead:report').length, 1, 'one report owner');
    assert.equal(events.filter(e => e.label === 'director:validate').length, 1, 'one validation owner');
    assert.equal(archive.best_id, sc.bestId);
    if (sc.optimizer !== 'greedy') {
    assert.equal(rec.after_loop.finalWinner.id, archive.best_id, 'result and archive select the same source');
    const bestRow = archive.entries.find(row => row.id === archive.best_id);
    assert.deepEqual(copyAndApply(task, bestRow.patch_ref.path, path.join(temp, 'selected-source-replay')),
      sc.expected, 'selected archive identity reconstructs the final best independently of Report');
    }
    for (const row of archive.entries) if (row.patch_ref) {
      assert.equal(hash(fs.readFileSync(row.patch_ref.path)), row.patch_ref.sha256, 'retained patch identity');
      assert.equal(row.patch_ref.base_commit, originalCommit, 'all patches share immutable commit');
    }
    assert.ok(archive.entries.every(row => !row.evolution_context ||
      !Object.keys(row.evolution_context).some(key => key.startsWith('selected_evidence')
        || key.startsWith('selected_design_genomes'))),
      'prompt evidence never becomes recursive archive context');
    assert.ok(events.filter(e => e.label.startsWith('tech_lead:plan '))
      .every(e => !e.prompt.includes('"selected_evidence"')), 'planner inputs remain unchanged');
    sc.check({archive, result, rec, events, hostCalls, h, temp, externalReads, finalMeasured,
      schemas, authoredGenomes, scenario: sc});
    passed = true;
    console.log(`  ok: ${sc.name}`);
  } finally {
    if (passed) fs.rmSync(temp, {recursive: true, force: true});
    else console.error(JSON.stringify({fixture: temp, logs: observed && observed.logs}, null, 2));
  }
}

const direction = (id, extra = {}) => ({id, title: id, specialty: 'memory', prompt: `Develop ${id}`, ...extra});
const plan = (...directions) => ({stop: false, directions});
const slow = source(1, 0, {'inherited.py': 'P = 1\n'}), fast = source(0, 2);
const child = source(1, 1, {'inherited.py': 'P = 1\n'});
const approachMemoryCase =
  {name: 'approach evidence reaches memory, forced replanning and the next planner',
    args: {budget: 3, deadline_epoch: 100000}, clock: () => 1000,
    best: 1.8, bestId: 'r2_d0', expected: child,
    plans: {r1: {stop: true, directions: []},
      'r1#1': plan(direction('a', {approach: 'approach-a'}), direction('b', {approach: 'approach-b'})),
      r2: plan(direction('continue-a', {parent_id: 'r1_d0', donor_ids: ['r1_d1'], approach: 'approach-a',
        prompt: 'Resolve question-a using the retained source and report.'}))},
    candidates: {
      r1_d0: {files: slow, speed: 0.8, engineer: {status: 'ok', measurement_valid: true,
        speedup_geomean: 0.8, strategies_tried: ['realized-a', 'unresolved-a']}},
      r1_d1: {files: fast, speed: 1.4},
      r2_d0: {files: child, inherited: slow, speed: 1.8, inspect: ({h, prompt}) => {
        assert.match(h.field(prompt, 'DIRECTION').prompt, /question-a/,
          'the planner carries the continuation question in the existing engineer assignment');
      }},
    },
    memory: ({label, prompt, h}) => {
      const saved = JSON.parse(fs.readFileSync(h.field(prompt, 'EVOLUTION_ARCHIVE'), 'utf8'));
      const attempted = h.field(prompt, 'EVOLUTION_ROUND');
      const round = Number(label.match(/r(\d+)$/)[1]);
      assert.deepEqual(attempted.map(row => row.id), round === 1 ? ['r1_d0', 'r1_d1'] : ['r2_d0']);
      for (const item of attempted) {
        const retained = saved.entries.find(row => row.id === item.id);
        for (const key of ['parent_id', 'donor_ids', 'approach', 'patch_ref', 'status', 'verified',
          'retention_error']) assert.deepEqual(item[key], retained[key]);
        assert.equal(item.last_record_error, retained.last_record_error || null);
        assert.equal(item.report_path, path.join(h.field(prompt, 'EVAL_DIR'),
          `round_${round}`, `engineer_${item.id.split('_d')[1]}`, 'report.md'));
      }
      if (round !== 1) return {insights: [], ledger: []};
      assert.deepEqual(saved.entries[0].engineer.strategies_tried, ['realized-a', 'unresolved-a']);
      return {insights: ['question-a remains open'], ledger: [{approach: 'approach-a',
        direction: 'r1_d0', patch_ref: attempted[0].patch_ref, report_path: attempted[0].report_path,
        next_question: 'question-a', verdict: 'partial'}]};
    },
    inspectPlan: ({label, prompt, h}) => {
      if (label !== 'tech_lead:plan r2') return;
      const ledger = h.field(prompt, 'HISTORY').ledger;
      assert.equal(ledger.length, 1, 'existing memory ledger reaches the next planning call');
      assert.equal(ledger[0].approach, 'approach-a');
      assert.equal(ledger[0].next_question, 'question-a');
      assert.equal(hash(fs.readFileSync(ledger[0].patch_ref.path)), ledger[0].patch_ref.sha256);
      assert.deepEqual(h.field(prompt, 'HISTORY').insights, ['question-a remains open']);
    },
    check: ({events, archive}) => {
      assert.ok(events.some(event => event.label === 'tech_lead:replan r1#1'));
      for (const event of events.filter(event => /^tech_lead:(?:plan|replan|memory) /.test(event.label))) {
        assert.ok(event.prompt.includes(`Read ${path.join(ROOT, 'kernel_workflow/roles/evolution.md')} for the retained-approach policy.`),
          `${event.label} receives the same EVO policy`);
      }
      assert.equal(archive.entries.find(row => row.id === 'r2_d0').parent_id, 'r1_d0');
    }};
const cases = [
  {name: 'slower-parent, donor identity, failures, cumulative new-file patch and retained source',
    args: {budget: 6, max_no_improve: 6}, best: 1.8, bestId: 'r4_d0', expected: child,
    plans: {r1: plan(direction('slow'), direction('fast')),
      r2: plan(direction('missing')), r3: plan(direction('unverified')),
      r4: plan(direction('child', {parent_id: 'r1_d0', donor_ids: ['r1_d1']})),
      r5: plan(direction('incorrect'))},
    candidates: {
      r1_d0: {files: slow, speed: 0.8, engineer: {status: 'ok', measurement_valid: true,
        speedup_geomean: 0.8, notes: 'Failed premise; corrected next question. é😀\nRaw: /cpu-fixture/evidence.json'}},
      r1_d1: {files: fast, speed: 1.4},
      r2_d0: {files: source(5, 5), inherited: fast, noPatch: true, engineer: {status: 'failed'}},
      r3_d0: {files: source(6, 6), inherited: fast, speed: 50,
        verifier: {status: 'invalid_measurement', correctness: 'pass', verified_geomean: 50}},
      r4_d0: {files: child, inherited: slow, speed: 1.8, inspect: ({h, prompt, canonical, replay}) => {
        assert.deepEqual(readSources(canonical), fast, 'child develops a nonbest, nonlatest parent');
        const context = h.field(prompt, 'EVOLUTION');
        assert.equal(context.requested_parent_id, 'r1_d0');
        assert.equal(context.donors[0].id, 'r1_d1');
        const saved = JSON.parse(fs.readFileSync(h.field(prompt, 'EVOLUTION_ARCHIVE'), 'utf8'));
        if (context.selected_evidence) {
          const {parent, donors} = context.selected_evidence;
          assert.equal(parent.id, 'r1_d0');
          assert.deepEqual(donors.map(row => row.id), ['r1_d1']);
          for (const evidence of [parent, ...donors]) {
            const row = saved.entries.find(entry => entry.id === evidence.id);
            assert.deepEqual(Object.keys(evidence).sort(), ['id', 'patch_ref', 'status', 'verified',
              'retention_error', 'last_record_error', 'engineer', 'verifier'].sort());
            for (const key of Object.keys(evidence)) assert.deepEqual(evidence[key], row[key] ?? null);
          }
          assert.match(parent.engineer.notes, /\nRaw: \/cpu-fixture\/evidence\.json$/);
        }
        assert.deepEqual(context.donors[0].patch_ref, saved.entries.find(row => row.id === 'r1_d1').patch_ref,
          'donor ID is joined to that retained source');
        assert.deepEqual(replay(context.donors[0].patch_ref.path), fast, 'donor bytes match the fast lineage');
      }},
      r5_d0: {files: source(7, 7, {'inherited.py': 'P = 1\n'}), inherited: child, speed: 90,
        verifier: {status: 'correctness_failed', correctness: 'fail', verified_geomean: 90}},
    },
    check: ({archive, result, events, h}) => {
      const rows = Object.fromEntries(archive.entries.map(row => [row.id, row]));
      assert.equal(archive.entries.length, 6, 'one retained record per attempt');
      assert.equal(rows.r1_d0.status, 'slower');
      assert.equal(rows.r1_d0.verified, true);
      assert.equal(rows.r1_d0.speedup, 0.8);
      assert.equal(rows.r2_d0.status, 'unverified');
      assert.ok(rows.r2_d0.retention_error);
      assert.equal(rows.r3_d0.status, 'unverified');
      assert.equal(rows.r3_d0.speedup, null, 'engineer claim cannot validate a source');
      assert.equal(rows.r5_d0.status, 'incorrect');
      assert.equal(rows.r4_d0.parent_id, 'r1_d0');
      assert.deepEqual(rows.r4_d0.donor_ids, ['r1_d1']);
      assert.equal(result.stopped_by, 'budget');
      assert.ok(events.some(e => e.label === 'verify r1_d0'), 'below-floor source independently verified');
      const lastPlan = events.find(e => e.label === 'tech_lead:plan r4');
      assert.ok(h.field(lastPlan.prompt, 'EVOLUTION_ENTRIES').some(row => row.id === 'r1_d0'));
      const missing = h.field(events.find(e => e.label === 'tech_lead:memory r2').prompt, 'EVOLUTION_ROUND')[0];
      assert.equal(missing.patch_ref, null, 'memory must not imply that a missing patch was retained');
      assert.equal(missing.verified, false);
      assert.ok(missing.retention_error);
      const unverified = h.field(events.find(e => e.label === 'tech_lead:memory r3').prompt, 'EVOLUTION_ROUND')[0];
      assert.ok(unverified.patch_ref, 'retained source remains available despite invalid measurement');
      assert.equal(unverified.verified, false, 'retention is not verification');
    }},
  {name: 'false successful commit preserves prior best bytes, patch and measurements',
    args: {budget: 2}, best: 1.4, bestId: 'r1_d0', expected: fast, badCommit: 'r2',
    plans: {r1: plan(direction('best')), r2: plan(direction('bad-commit'))},
    candidates: {r1_d0: {files: fast, speed: 1.4},
      r2_d0: {files: source(2, 2), inherited: fast, speed: 1.9}},
    check: ({archive, rec}) => {
      assert.equal(archive.entries[1].verified, true, 'failed promotion keeps verified candidate');
      assert.equal(rec.after_loop.roundsCommitted, 1);
      assert.equal(rec.after_loop.finalWinner.id, 'r1_d0');
      assert.equal(rec.after_loop.history.rounds[1].status, 'commit_failed');
    }},
  {name: 'deadline after committed round exports best without another engineer',
    args: {budget: 5, deadline_epoch: 10000}, best: 1.4, bestId: 'r1_d0', expected: fast,
    clock: tag => tag === 'pre-r1' ? 1000 : 10001,
    plans: {r1: plan(direction('best')), r2: plan(direction('late'))},
    candidates: {r1_d0: {files: fast, speed: 1.4}},
    check: ({result, events}) => {
      assert.equal(result.stopped_by, 'deadline');
      assert.equal(events.filter(e => e.label.startsWith('eng ')).length, 1);
      assert.ok(!events.some(e => e.label === 'tech_lead:plan r2'));
    }},
  {name: 'process cutoff after planning admits no further engineer and exports best',
    bounded: true, expireAfterPlan: 'r2', args: {budget: 5},
    best: 1.4, bestId: 'r1_d0', expected: fast,
    plans: {r1: plan(direction('best')), r2: plan(direction('late'))},
    candidates: {r1_d0: {files: fast, speed: 1.4}},
    check: ({result, events}) => {
      assert.equal(result.stopped_by, 'deadline');
      assert.equal(events.filter(e => e.label.startsWith('eng ')).length, 1);
      assert.equal(events.filter(e => e.label.startsWith('verify ')).length, 1);
    }},
  {name: 'cutoff leaves a saved unverified attempt distinct from the validated best',
    bounded: true, expireAfterEngineer: 'r2_d0', args: {budget: 5},
    best: 1.4, bestId: 'r1_d0', expected: fast,
    plans: {r1: plan(direction('best')), r2: plan(direction('unfinished'))},
    candidates: {r1_d0: {files: fast, speed: 1.4},
      r2_d0: {files: source(4, 4), inherited: fast, speed: 50}},
    check: ({archive, result, events}) => {
      const unfinished = archive.entries.find(row => row.id === 'r2_d0');
      assert.ok(unfinished, 'saved attempt must remain in the archive after the host cutoff');
      assert.equal(unfinished.status, 'unverified');
      assert.equal(unfinished.verified, false);
      assert.equal(unfinished.speedup, null);
      assert.ok(unfinished.patch_ref, 'unfinished source retained for later investigation');
      assert.ok(!events.some(e => e.label === 'verify r2_d0'));
      assert.equal(result.stopped_by, 'deadline');
    }},
  {name: 'integrated candidate retains contributor IDs and remains a selectable source',
    args: {budget: 3}, best: 1.8, bestId: 'r1_integrated', expected: source(1, 2),
    plans: {r1: plan(direction('a'), direction('b')),
      r2: plan(direction('continued', {parent_id: 'r1_integrated'}))},
    integration: {files: source(1, 2), speed: 1.8},
    candidates: {r1_d0: {files: source(1, 0), speed: 1.2},
      r1_d1: {files: fast, speed: 1.4},
      r2_d0: {files: source(1, 3), inherited: source(1, 2), speed: 1.6}},
    check: ({archive}) => {
      const integrated = archive.entries.find(row => row.id === 'r1_integrated');
      assert.equal(integrated.verified, true);
      assert.deepEqual(integrated.donor_ids, ['r1_d0', 'r1_d1']);
      assert.equal(archive.entries.find(row => row.id === 'r2_d0').parent_id, 'r1_integrated');
    }},
  {name: 'source changed after measurement cannot validate or promote the wrong patch',
    args: {budget: 2}, best: 1.4, bestId: 'r1_d0', expected: fast,
    plans: {r1: plan(direction('best')), r2: plan(direction('changed-source'))},
    candidates: {r1_d0: {files: fast, speed: 1.4},
      r2_d0: {files: source(2, 2), inherited: fast, speed: 1.9,
        mutateAfterVerification: source(3, 3)}},
    check: ({archive, rec}) => {
      const changed = archive.entries.find(row => row.id === 'r2_d0');
      assert.equal(changed.verified, false, 'measurement cannot be joined to a replaced source');
      assert.equal(changed.speedup, null);
      assert.ok(changed.last_record_error || changed.retention_error || rec.logs.some(line =>
        line.includes('r2_d0') && /retained patch changed/.test(line)), 'source mismatch remains explicit');
      assert.equal(rec.after_loop.roundsCommitted, 1);
    }},
  {name: 'parallel deep proposals charge their real cost without exceeding the remaining budget',
    args: {budget: 3, deep_cost: 2}, best: 1.4, bestId: 'r1_d0', expected: fast,
    plans: {r1: plan(...['deep-a', 'deep-b', 'deep-c'].map(id =>
      direction(id, {specialty: 'deep_explore'})))},
    candidates: {r1_d0: {files: fast, speed: 1.4}},
    check: ({result, events, h}) => {
      assert.equal(events.filter(e => e.label.startsWith('deep ')).length, 1,
        'only one cost-two direction fits the budget of three');
      assert.equal(result.budget_used, 2);
      assert.ok(result.budget_used <= result.budget_total);
      assert.equal(h.field(events.find(e => e.label === 'tech_lead:plan r1').prompt, 'DEEP_COST'), 2);
    }},
  {name: 'empty analyzer file list and intermediate commits do not break valid cumulative promotion',
    args: {budget: 2}, modifiableFiles: [], best: 1.8, bestId: 'r2_d0', expected: source(3, 2),
    plans: {r1: plan(direction('best')), r2: plan(direction('committing-engineer'))},
    candidates: {r1_d0: {files: fast, speed: 1.4},
      r2_d0: {files: source(3, 2), inherited: fast, speed: 1.8,
        commitDuringEngineering: source(2, 2)}},
    check: ({archive, rec}) => {
      assert.equal(rec.after_loop.roundsCommitted, 2, 'analyzer output cannot veto measured promotion');
      assert.equal(archive.entries.find(row => row.id === 'r2_d0').parent_id, 'r1_d0');
    }},
  {name: 'external primary accepts exact cumulative child source from a retained non-ORIGINAL parent',
    external: true, args: {budget: 2}, best: 1.8, bestDev: 1.2, bestId: 'r2_d0', expected: child,
    plans: {r1: plan(direction('retained-parent')),
      r2: plan(direction('child', {parent_id: 'r1_d0'}))},
    candidates: {r1_d0: {files: slow, speed: 0.8, development: 1.5},
      r2_d0: {files: child, inherited: slow, speed: 1.8, development: 1.2,
        inspect: ({canonical}) => assert.deepEqual(readSources(canonical), original,
          'native regression is retained without promoting its better diagnostic geomean')}},
    check: ({archive, result, rec, externalReads, finalMeasured, scenario}) => {
      const parent = archive.entries.find(row => row.id === 'r1_d0');
      const retainedChild = archive.entries.find(row => row.id === 'r2_d0');
      assert.equal(parent.verified, true);
      assert.equal(parent.status, 'slower');
      assert.equal(parent.speedup, 0.8);
      assert.equal(retainedChild.parent_id, parent.id);
      assert.equal(retainedChild.verified, true);
      assert.equal(retainedChild.speedup, 1.8);
      assert.equal(rec.after_loop.roundsCommitted, 1);
      assert.ok(externalReads.some(row => row.id === 'verify-r2_d0' && row.patch),
        'trusted reader checked child cumulative patch against the fixed base');
      assert.equal(finalMeasured, true);
      assert.equal(externalReads.filter(row => row.id === 'final' && row.patch).length, 1,
        'final trusted reader checks host patch against ORIGINAL after host evaluation');
      assert.equal(result.validation_status, scenario.patchSourceMismatch ? 'invalid_measurement' : 'accepted');
      assert.equal(result.final_speedup, scenario.patchSourceMismatch ? null : scenario.finalSpeed ?? 1.8);
      if (scenario.patchSourceMismatch) {
        assert.equal(result.outcome, 'invalid_measurement');
        assert.equal(result.final_patch, '', 'wrong captured-source patch is never exported');
        assert.equal(externalReads.filter(row => row.rejected).length, 1);
      } else {
        assert.equal(path.basename(result.final_patch), 'host_final_patch.diff');
        assert.equal(externalReads.find(row => row.id === 'final').patch, result.final_patch,
          'export uses exactly the patch checked by the trusted reader');
      }
    }},
];
const externalLineage = cases[cases.length - 1];
cases.push(approachMemoryCase);
cases.push(...[true, false, 'true'].map(evolution_evidence_handoff => ({...cases[0],
  name: `selected evidence handoff with ${JSON.stringify(evolution_evidence_handoff)}`,
  args: {...cases[0].args, evolution_evidence_handoff}})));
const unicodeNotes = '😀'.repeat(4096);
cases.push({...cases[0], name: 'UTF-8 budget omits the whole payload and preserves archive access',
  args: {...cases[0].args, evolution_evidence_handoff: true}, candidates: {...cases[0].candidates,
    r1_d0: {...cases[0].candidates.r1_d0,
      engineer: {...cases[0].candidates.r1_d0.engineer, notes: unicodeNotes}},
    r4_d0: {...cases[0].candidates.r4_d0, handoffOverLimit: true, inspect: info => {
      cases[0].candidates.r4_d0.inspect(info);
      const context = info.h.field(info.prompt, 'EVOLUTION');
      assert.equal(context.selected_evidence_omitted, 'over_16384_utf8_bytes');
      const saved = JSON.parse(fs.readFileSync(info.h.field(info.prompt, 'EVOLUTION_ARCHIVE'), 'utf8'));
      assert.equal(saved.entries.find(row => row.id === 'r1_d0').engineer.notes, unicodeNotes);
      assert.ok(unicodeNotes.length < 16384 && Buffer.byteLength(unicodeNotes, 'utf8') === 16384,
        'payload overhead crosses the byte limit despite fewer UTF-16 code units');
    }}}});
cases.push(...['null', 'missing', 'stale', 'wrong'].map(report => ({...externalLineage, report,
  name: `external cumulative parent and child survive ${report} Report patch`})),
  {...externalLineage, name: 'host patch binds unstaged final edit and staged binary instead of HEAD',
    finalFiles: source(2, 1, {'inherited.py': 'P = 1\n',
      'payload.bin': Buffer.from([0, 255, 128, 10, 0, 1])}),
    stagedBinary: 'payload.bin', finalSpeed: 2, finalDev: 1.3},
  {...externalLineage, name: 'final trusted reader rejects source drift between native capture and Git diff',
    patchSourceMismatch: source(99, 99, {'inherited.py': 'P = 1\n'})});

// Publish a peer through the same on-disk contract the real query verifies.
// Git commit timestamps may differ; the shared source tree is the compatibility key.
function peerCheckpoint({temp, populationRoot}, id, files, speed,
  {differentBase = false, tamper = false, genome} = {}) {
  const evalDir = path.join(temp, id, 'eval'), base = path.join(evalDir, 'evolution/base');
  const initial = differentBase ? source(77, 77) : original;
  const initialDir = path.join(evalDir, 'initial');
  writeSources(initialDir, initial);
  command('bash', [path.join(ROOT, 'kernel_workflow/scripts/materialize_workspace.sh'),
    '--src', initialDir, '--dst', base]);
  initRepo(base);
  const baseCommit = git(base, 'rev-parse', 'HEAD').trim();
  const baseTree = git(base, 'rev-parse', 'HEAD^{tree}').trim();
  const workspace = path.join(evalDir, 'candidate');
  command('bash', [path.join(ROOT, 'kernel_workflow/scripts/materialize_workspace.sh'),
    '--src', base, '--dst', workspace]);
  initRepo(workspace);
  writeSources(workspace, files);
  git(workspace, 'add', '-A');
  const sourceTree = git(workspace, 'write-tree').trim();
  const directory = path.join(populationRoot, id);
  fs.mkdirSync(directory);
  const patch = path.join(directory, 'r1.diff');
  fs.writeFileSync(patch, git(workspace, 'diff', '--cached', '--binary', 'HEAD'));
  const checkpoint = {schema: 'geak.population.checkpoint/1', lineage_id: id, optimizer: 'greedy', round: 1,
    base, base_commit: baseCommit, base_tree: baseTree, source_tree: sourceTree,
    patch_ref: {path: patch, sha256: hash(fs.readFileSync(patch)), base_commit: baseCommit, base_tree: baseTree},
    delivery_patch_ref: differentBase ? null : {path: patch, sha256: hash(fs.readFileSync(patch)), base: 'ORIGINAL'},
    metric: {kind: 'geomean'}, speedup: speed, verified: true, qualification: 'legacy_verifier',
    external_evaluation: null, evidence: {status: 'verified', correctness: 'pass', verified_geomean: speed}};
  if (genome !== undefined) checkpoint.design_genome = typeof genome === 'function'
    ? genome({patch, baseCommit, checkpoint}) : genome;
  const checkpointPath = path.join(directory, 'r1.json');
  fs.writeFileSync(checkpointPath, JSON.stringify(checkpoint));
  fs.writeFileSync(path.join(directory, 'latest.json'), JSON.stringify({schema: 'geak.population.pointer/1',
    lineage_id: id, checkpoint: checkpointPath, sha256: hash(fs.readFileSync(checkpointPath))}));
  if (tamper) fs.appendFileSync(patch, 'tampered source bytes\n');
  return checkpoint;
}

cases.push({...externalLineage, name: 'declared native lineage publishes exact cumulative committed checkpoints',
  population: {id: 'native', imports: []}, expected: source(1, 1),
  candidates: {r1_d0: {...externalLineage.candidates.r1_d0, files: source(1, 0)},
    r2_d0: {...externalLineage.candidates.r2_d0, files: source(1, 1), inherited: source(1, 0)}}, check(info) {
    externalLineage.check(info);
    const root = path.join(info.temp, 'population/native');
    const pointer = JSON.parse(fs.readFileSync(path.join(root, 'latest.json')));
    const row = JSON.parse(fs.readFileSync(pointer.checkpoint));
    assert.equal(row.round, 2);
    assert.equal(row.qualification, 'native_source_bound');
    assert.equal(row.speedup, info.scenario.best);
    assert.equal(row.external_evaluation.evaluation_id, 'verify-r2_d0');
    assert.equal(row.patch_ref.sha256, hash(fs.readFileSync(row.patch_ref.path)));
    assert.ok(info.externalReads.some(read => read.id === 'verify-r2_d0' && read.patch === row.patch_ref.path),
      'publisher replays cumulative committed source through the existing trusted reader');
    assert.equal(hash(fs.readFileSync(pointer.checkpoint)), pointer.sha256);
    assert.equal(fs.statSync(pointer.checkpoint).mode & 0o222, 0, 'published row has no write bits');
    assert.ok(fs.existsSync(path.join(root, 'r0.json')), 'initial verified source is independently recoverable');
  }},
  {name: 'ordinary lineage exports cumulative checkpoints without EVO planner context', optimizer: 'greedy',
    population: {id: 'ordinary', imports: []}, args: {budget: 2},
    best: 1.8, bestId: 'ORIGINAL', expected: child,
    plans: {r1: plan(direction('r1_d0')), r2: plan(direction('r2_d0'))},
    candidates: {r1_d0: {files: slow, speed: 1.4}, r2_d0: {files: child, inherited: slow, speed: 1.8}},
    check({archive, events, temp}) {
      assert.equal(archive.entries.length, 0, 'ordinary optimizer retains its own unchanged planning model');
      for (const event of events.filter(row => row.label.startsWith('tech_lead:plan '))) {
        assert.ok(!event.prompt.includes('EVOLUTION_ARCHIVE') && !event.prompt.includes('roles/evolution.md'));
      }
      const result = JSON.parse(command('python3', [path.join(ROOT, 'kernel_workflow/scripts/query_population.py'),
        '--root', path.join(temp, 'population'), '--checkpoints-only']));
      assert.equal(result.selected, null, 'legacy checkpoints preserve source without a native-style final-delivery claim');
      assert.ok(result.records.every(row => row.status === 'legacy_verified'));
      const row = result.records.find(row => row.checkpoint.round === 2).checkpoint;
      assert.equal(row.speedup, 1.8);
      assert.equal(row.patch_ref.base_tree, archive.base_tree);
      assert.equal(row.qualification, 'legacy_verifier', 'legacy evidence is honestly distinguished from native qualification');
    }},
  {name: 'explicit peer import is namespaced and never replaces the local default incumbent',
    population: {id: 'local', imports: ['peer', 'different', 'tampered']}, args: {budget: 3},
    best: 1.8, bestId: 'r2_d1', expected: source(0, 3),
    plans: {r1: plan(direction('local')), r2: plan(direction('peer-child', {parent_id: 'peer_peer_r1'}), direction('local-child'))},
    candidates: {r1_d0: {files: fast, speed: 1.4},
      r2_d0: {files: child, inherited: slow, speed: 1.1},
      r2_d1: {files: source(0, 3), inherited: fast, speed: 1.8}},
    inspectPlan(info) {
      if (info.label.endsWith('r1')) {
        peerCheckpoint(info, 'peer', slow, 2.0);
        peerCheckpoint(info, 'different', slow, 3.0, {differentBase: true});
        peerCheckpoint(info, 'tampered', slow, 4.0, {tamper: true});
      } else if (info.label.endsWith('r2')) {
        const saved = JSON.parse(fs.readFileSync(info.h.field(info.prompt, 'EVOLUTION_ARCHIVE')));
        assert.equal(saved.best_id, 'r1_d0', 'faster peer evidence does not reset the local incumbent');
        assert.equal(saved.entries.find(row => row.id === 'peer_peer_r1').speedup, 2.0);
        assert.ok(!saved.entries.some(row => row.id.includes('different') || row.id.includes('tampered')),
          'incompatible base and changed peer patch never become selectable references');
      }
    },
    check({archive}) {
      const peer = archive.entries.find(row => row.id === 'peer_peer_r1');
      assert.equal(peer.peer_checkpoint.lineage_id, 'peer');
      assert.ok(peer.patch_ref.path.includes('/evolution/patches/peer_peer_r1.diff'));
      assert.equal(archive.entries.find(row => row.id === 'r2_d0').parent_id, peer.id);
      assert.equal(archive.entries.find(row => row.id === 'r2_d1').parent_id, 'r1_d0');
    }});

// Narratives are optional author claims. All cases below still exercise actual
// source retention, verification, promotion and (where enabled) publication.
const authorGenome = text => ({patch}) => ({patch_sha256: hash(fs.readFileSync(patch)), text});
const claimText = 'Hypothesis: combine adjacent loads. Realization: partial. '
  + 'Observed: the oracle passed; this does not establish the mechanism. '
  + 'Gap: redundant address work remains. Rival: register pressure. Next repair: inspect the loop.';
function assertGenome(row, text, native = false) {
  assert.equal(row.design_genome.kind, 'author_claims');
  assert.equal(row.design_genome.text, text);
  assert.equal(row.design_genome.base_commit, row.patch_ref.base_commit);
  assert.equal(row.design_genome.patch_sha256, row.patch_ref.sha256);
  assert.equal(Object.hasOwn(row.design_genome, 'source_id'), native);
  assert.equal(Object.hasOwn(row.design_genome, 'external_evaluation'), native);
}
function assertGenomeCalls({events, schemas}, enabled) {
  const writer = /^(?:eng |deep |integrate |tech_lead:(?:plan|replan|memory) )/;
  for (const event of events.filter(item => writer.test(item.label))) {
    assert.equal(event.prompt.includes('/roles/design_genome.md'), enabled,
      `${event.label} optional writer guidance follows the effective EVO mode`);
    const schema = schemas.get(event.label);
    const properties = /^(?:eng |deep )/.test(event.label) ? schema.properties
      : event.label.startsWith('integrate ') ? schema.properties.best.properties : null;
    if (!properties) continue;
    assert.equal(Object.hasOwn(properties, 'design_genome'), enabled);
    if (enabled) assert.deepEqual(properties.design_genome, {},
      'arbitrary optional JSON reaches host normalization without invalidating the worker return');
    const required = event.label.startsWith('integrate ') ? schema.properties.best.required : schema.required;
    assert.ok(!required.includes('design_genome'), 'the narrative is never required');
  }
}
const genomeModeObservations = new Map();
const genomeModeContracts = new Map();
const genomeModes = [undefined, 'off', 'record', 'handoff'];
for (const mode of genomeModes) cases.push({
  name: `design genome ${mode === undefined ? 'absent' : mode}: writer parity and selective handoff`,
  args: {budget: 3, ...(mode === undefined ? {} : {design_genome_mode: mode})},
  best: 1.8, bestId: 'r1_integrated', expected: source(1, 2),
  plans: {r1: plan(direction('a'), direction('b')),
    r2: plan(direction('child', {parent_id: 'r1_d0', donor_ids: ['r1_d1']}))},
  integration: {files: source(1, 2), speed: 1.8},
  candidates: {
    r1_d0: {files: source(1, 0), speed: 1.2, genome: authorGenome(claimText)},
    r1_d1: {files: fast, speed: 1.4, genome: authorGenome('A distinct donor hypothesis, with unknown realization.')},
    r2_d0: {files: source(1, 3), inherited: source(1, 0), speed: 1.6},
  },
  check(info) {
    const {archive, events, h, rec, result} = info;
    const enabled = mode === 'record' || mode === 'handoff';
    assertGenomeCalls(info, enabled);
    const parent = archive.entries.find(row => row.id === 'r1_d0');
    if (enabled) assertGenome(parent, claimText);
    else assert.ok(archive.entries.every(row => !Object.hasOwn(row, 'design_genome')));
    for (const id of ['r1_integrated', 'r2_d0']) {
      assert.ok(!Object.hasOwn(archive.entries.find(row => row.id === id), 'design_genome'),
        'new children and integrated candidates cannot inherit contributor narratives');
    }
    const planned = h.field(events.find(event => event.label === 'tech_lead:plan r2').prompt, 'EVOLUTION_ENTRIES');
    assert.equal(Object.hasOwn(planned.find(row => row.id === parent.id), 'design_genome'), enabled);
    const patches = h.field(events.find(event => event.label === 'integrate r1').prompt, 'PATCHES');
    for (const patch of patches) {
      assert.equal(Object.hasOwn(patch, 'design_genome'), enabled);
      if (enabled) assert.deepEqual(patch.design_genome,
        archive.entries.find(row => row.id === patch.id).design_genome);
    }
    for (const event of events.filter(event => event.label.startsWith('eng '))) {
      const context = h.field(event.prompt, 'EVOLUTION');
      const hasParentGenome = !event.label.includes('r1_');
      assert.equal(Object.hasOwn(context, 'selected_design_genomes'), enabled && hasParentGenome);
      if (!enabled || !hasParentGenome) continue;
      const selected = context.selected_design_genomes;
      assert.deepEqual(selected.parent, {id: parent.id, design_genome: parent.design_genome});
      assert.deepEqual(selected.donors, [{id: 'r1_d1',
        design_genome: archive.entries.find(row => row.id === 'r1_d1').design_genome}]);
    }
    const observation = {calls: events.map(event => event.label), phases: rec.phases,
      budget: result.budget_used, best: archive.best_id};
    genomeModeObservations.set(mode, observation);
    assert.deepEqual(observation, genomeModeObservations.get(undefined),
      'the mode changes no calls, phases, budget or selected source');
    if (enabled) {
      const contract = {schemas: [...info.schemas], guidance: events.map(event => ({label: event.label,
        instruction: event.prompt.match(/Read \S+\/roles\/design_genome\.md[^.]*\./)?.[0] || null}))};
      genomeModeContracts.set(mode, contract);
      if (mode === 'handoff') assert.deepEqual(contract, genomeModeContracts.get('record'),
        'record and handoff use the same writer guidance and complete call schemas');
    }
  },
});

const greedyGenomeBase = cases.find(sc => sc.optimizer === 'greedy');
for (const mode of ['record', 'handoff']) cases.push({...greedyGenomeBase,
  name: `greedy ignores design genome mode ${mode}`,
  args: {...greedyGenomeBase.args, design_genome_mode: mode},
  candidates: Object.fromEntries(Object.entries(greedyGenomeBase.candidates).map(([id, spec]) =>
    [id, {...spec, genome: authorGenome(claimText)}])),
  check(info) { greedyGenomeBase.check(info); assertGenomeCalls(info, false); },
});

const unicodeGenome = '😀'.repeat(1024); // Exactly 4096 UTF-8 bytes.
const malformedGenomes = [null, 'raw prose', [], {text: 5, patch_sha256: '0'.repeat(64)},
  {text: claimText, patch_sha256: '0'.repeat(64)}, authorGenome('   '),
  authorGenome(`${unicodeGenome}a`), ({authoredGenomes}) => authoredGenomes.get('r1_d0')];
cases.push({name: 'malformed, oversized and swapped genomes never disqualify valid source',
  args: {budget: malformedGenomes.length + 3, design_genome_mode: 'handoff'},
  best: 1.8, bestId: `r2_d${malformedGenomes.length + 1}`, expected: source(44, 44),
  plans: {r1: plan(direction('parent')),
    r2: plan(...Array.from({length: malformedGenomes.length + 2}, (_, i) => direction(`attempt-${i}`)))},
  candidates: {
    r1_d0: {files: fast, speed: 1.4, genome: authorGenome(claimText)},
    ...Object.fromEntries(malformedGenomes.map((genome, i) => [`r2_d${i}`,
      {files: source(i + 10, i + 10), inherited: fast, speed: 1.5, genome}])),
    [`r2_d${malformedGenomes.length}`]: {files: source(43, 43), inherited: fast, speed: 50,
      genome: authorGenome('Unverified implementation; realization unknown.'),
      verifier: {status: 'invalid_measurement', correctness: 'pass', verified_geomean: 50}},
    [`r2_d${malformedGenomes.length + 1}`]: {files: source(44, 44), inherited: fast, speed: 1.8,
      genome: authorGenome(unicodeGenome)},
  },
  check({archive}) {
    for (let i = 0; i < malformedGenomes.length; i++) {
      const row = archive.entries.find(entry => entry.id === `r2_d${i}`);
      assert.equal(row.verified, true);
      assert.equal(row.speedup, 1.5);
      assert.equal(row.retention_error, null, 'bad optional prose cannot become a retention failure');
      assert.ok(!Object.hasOwn(row, 'design_genome'));
    }
    const unverified = archive.entries.find(row => row.id === `r2_d${malformedGenomes.length}`);
    assert.equal(unverified.verified, false);
    assertGenome(unverified, 'Unverified implementation; realization unknown.');
    assertGenome(archive.entries.find(row => row.id === archive.best_id), unicodeGenome);
  },
});

cases.push({name: 'design genome and evidence handoff obey the combined UTF-8 packet budget',
  args: {budget: 3, design_genome_mode: 'handoff', evolution_evidence_handoff: true},
  best: 1.8, bestId: 'r2_d0', expected: child,
  plans: {r1: plan(direction('a'), direction('b')),
    r2: plan(direction('continued', {parent_id: 'r1_d0', donor_ids: ['r1_d1']}))},
  candidates: {
    r1_d0: {files: slow, speed: 0.8, genome: authorGenome(unicodeGenome),
      engineer: {status: 'ok', measurement_valid: true, speedup_geomean: 0.8, notes: 'E'.repeat(8000)}},
    r1_d1: {files: fast, speed: 1.4, genome: authorGenome(unicodeGenome)},
    r2_d0: {files: child, inherited: slow, speed: 1.8, inspect({h, prompt}) {
      const context = h.field(prompt, 'EVOLUTION');
      assert.ok(context.selected_evidence, 'existing evidence fits its own limit');
      assert.equal(context.selected_design_genomes_omitted, 'over_16384_utf8_bytes');
      assert.ok(!Object.hasOwn(context, 'selected_design_genomes'));
    }},
  },
  check({archive}) { assertGenome(archive.entries[0], unicodeGenome); },
});

const integratedGenomeBase = cases.find(sc => sc.name.startsWith('integrated candidate'));
cases.push({...integratedGenomeBase, name: 'integrator retains only its own freshly bound author genome',
  args: {...integratedGenomeBase.args, design_genome_mode: 'handoff'},
  integration: {...integratedGenomeBase.integration, genome: authorGenome(claimText)},
  candidates: {...integratedGenomeBase.candidates,
    r1_d0: {...integratedGenomeBase.candidates.r1_d0, genome: authorGenome('Donor A hypothesis.')},
    r1_d1: {...integratedGenomeBase.candidates.r1_d1, genome: authorGenome('Donor B hypothesis.')}},
  check(info) {
    integratedGenomeBase.check(info);
    const integrated = info.archive.entries.find(row => row.id === 'r1_integrated');
    assertGenome(integrated, claimText);
    const context = info.h.field(info.events.find(event => event.label.startsWith('eng r2_d0')).prompt, 'EVOLUTION');
    assert.deepEqual(context.selected_design_genomes.parent, {id: integrated.id, design_genome: integrated.design_genome});
    assert.ok(!Object.hasOwn(info.archive.entries.find(row => row.id === 'r2_d0'), 'design_genome'));
  },
});
cases.push({...integratedGenomeBase, name: 'integrator cannot relabel a donor genome as its own measured source',
  args: {...integratedGenomeBase.args, design_genome_mode: 'record'},
  integration: {...integratedGenomeBase.integration,
    genome: ({authoredGenomes}) => authoredGenomes.get('r1_d0')},
  candidates: {...integratedGenomeBase.candidates,
    r1_d0: {...integratedGenomeBase.candidates.r1_d0, genome: authorGenome(claimText)}},
  check(info) {
    integratedGenomeBase.check(info);
    const integrated = info.archive.entries.find(row => row.id === 'r1_integrated');
    assert.equal(integrated.retention_error, null);
    assert.ok(!Object.hasOwn(integrated, 'design_genome'));
    assert.equal(info.archive.best_id, integrated.id, 'valid integrated fitness survives invalid optional prose');
  },
});

const nativeGenomeBase = cases.find(sc => sc.name.startsWith('declared native lineage'));
for (const mismatch of [false, true]) cases.push({...nativeGenomeBase,
  name: mismatch ? 'checked engineer source A cannot attach claims to checked verifier source B'
    : 'native author genome persists through cumulative checkpoint publication',
  args: {...nativeGenomeBase.args, design_genome_mode: 'handoff'},
  candidates: Object.fromEntries(Object.entries(nativeGenomeBase.candidates).map(([id, spec]) => [id,
    {...spec, genome: authorGenome(claimText), ...(!mismatch ? {patchOptions: ['--full-index']} : {}),
      ...(mismatch && id === 'r2_d0'
      ? {engineerObservationFiles: source(88, 88)} : {})}])),
  check(info) {
    nativeGenomeBase.check(info);
    const parent = info.archive.entries.find(row => row.id === 'r1_d0');
    const selected = info.archive.entries.find(row => row.id === 'r2_d0');
    assertGenome(parent, claimText, true);
    assert.equal(parent.design_genome.external_evaluation.evaluation_id, 'verify-r1_d0');
    const pointer = JSON.parse(fs.readFileSync(path.join(info.temp, 'population/native/latest.json')));
    const checkpoint = JSON.parse(fs.readFileSync(pointer.checkpoint));
    if (mismatch) {
      assert.equal(selected.verified, true, 'independently verified B remains eligible');
      assert.equal(selected.retention_error, null);
      assert.ok(!Object.hasOwn(selected, 'design_genome'));
      assert.ok(!Object.hasOwn(checkpoint, 'design_genome'));
      const authorRead = info.externalReads.find(row => row.id === 'eng-r2_d0');
      const verifierRead = info.externalReads.find(row => row.id === 'verify-r2_d0');
      assert.notEqual(authorRead.source_id, verifierRead.source_id, 'both independent readers checked different sources');
    } else {
      assertGenome(selected, claimText, true);
      assertGenome(checkpoint, claimText, true);
      assert.notEqual(checkpoint.design_genome.patch_sha256, selected.design_genome.patch_sha256,
        'native publication rebinds an equivalent diff representation only after source identity is checked');
      assert.equal(checkpoint.design_genome.source_id, selected.design_genome.source_id);
      assert.deepEqual(checkpoint.design_genome.external_evaluation, checkpoint.external_evaluation);
    }
  },
});
cases.push({name: 'legacy publication omits claims when equivalent diff bytes change',
  population: {id: 'legacy', imports: []}, args: {budget: 1, design_genome_mode: 'record'},
  best: 1.4, bestId: 'r1_d0', expected: fast, plans: {r1: plan(direction('legacy'))},
  candidates: {r1_d0: {files: fast, speed: 1.4, genome: authorGenome(claimText), patchOptions: ['--full-index']}},
  check({archive, temp}) {
    assertGenome(archive.entries[0], claimText);
    const pointer = JSON.parse(fs.readFileSync(path.join(temp, 'population/legacy/latest.json')));
    const checkpoint = JSON.parse(fs.readFileSync(pointer.checkpoint));
    assert.equal(checkpoint.verified, true);
    assert.equal(checkpoint.speedup, 1.4);
    assert.notEqual(checkpoint.patch_ref.sha256, archive.entries[0].patch_ref.sha256);
    assert.ok(!Object.hasOwn(checkpoint, 'design_genome'),
      'legacy evidence cannot independently rebind narrative to a different patch representation');
  },
});

const peerGenomeBase = cases.find(sc => sc.name.startsWith('explicit peer import'));
for (const mode of ['record', 'handoff']) cases.push({...peerGenomeBase,
  name: `peer import in ${mode} controls automatic peer narrative retention and exposure`,
  args: {...peerGenomeBase.args, design_genome_mode: mode},
  inspectPlan(info) {
    if (info.label.endsWith('r1')) {
      peerCheckpoint(info, 'peer', slow, 2, {genome: ({patch, baseCommit}) => ({
        kind: 'author_claims', text: claimText, base_commit: baseCommit,
        patch_sha256: hash(fs.readFileSync(patch))})});
      peerCheckpoint(info, 'different', slow, 3, {differentBase: true});
      peerCheckpoint(info, 'tampered', slow, 4, {tamper: true});
    } else peerGenomeBase.inspectPlan(info);
  },
  check(info) {
    peerGenomeBase.check(info);
    const peer = info.archive.entries.find(row => row.id === 'peer_peer_r1');
    const checkpoint = JSON.parse(fs.readFileSync(peer.peer_checkpoint.path));
    assert.equal(checkpoint.design_genome.text, claimText, 'published peer artifact retains its narrative in both modes');
    if (mode === 'handoff') assert.deepEqual(peer.design_genome, checkpoint.design_genome,
      'peer narrative retains its own original binding');
    else assert.ok(!Object.hasOwn(peer, 'design_genome'), 'record does not import peer narrative into the local archive');
    const planned = info.h.field(info.events.find(event => event.label === 'tech_lead:plan r2').prompt, 'EVOLUTION_ENTRIES');
    assert.equal(Object.hasOwn(planned.find(row => row.id === peer.id), 'design_genome'), mode === 'handoff');
    const context = info.h.field(info.events.find(event => event.label.startsWith('eng r2_d0')).prompt, 'EVOLUTION');
    if (mode === 'handoff') assert.deepEqual(context.selected_design_genomes.parent,
      {id: peer.id, design_genome: peer.design_genome, peer_checkpoint: peer.peer_checkpoint});
    else {
      assert.ok(!Object.hasOwn(context, 'selected_design_genomes'));
      assert.ok(info.events.every(event => !event.prompt.includes(claimText)),
        'record does not automatically expose the published peer narrative through another prompt');
    }
    assert.ok(!Object.hasOwn(info.archive.entries.find(row => row.id === 'r2_d0'), 'design_genome'));
  },
});

(async () => {
  ({validate: validateSchema} = await import(path.join(ROOT, 'interface/runtime/engine/schema.mjs')));
  console.log(`lane: ${lanePath} (sha256 ${hash(laneSource)})`);
  const h = harness({});
  const invalidArgs = [
    [{optimizer: 'external'}, /unknown optimizer: external; use greedy or evo/],
    [{optimizer: 'evo', optimizer_command: ['python3', 'retired.py']}, /optimizer_command is retired/],
    [{optimizer: 'greedy', optimizer_command: null}, /optimizer_command is retired/],
    [{optimizer: 'evo', optimizer_task_ref: {path: '/old/task.json', sha256: '0'.repeat(64)}},
      /optimizer_task_ref is retired/],
    [{optimizer_reserve_s: 0}, /optimizer_reserve_s is retired/],
  ];
  for (const [args, message] of invalidArgs) {
    const unsupported = h.build(h.instrument(laneSource), {...h.BASE_ARGS, ...args},
      () => assert.fail('unsupported optimizer arguments must not start an agent'));
    await assert.rejects(unsupported.run(), message);
    assert.equal(unsupported.rec.calls.length, 0);
  }
  const wrongSetup = h.build(h.instrument(laneSource), {...h.BASE_ARGS, eval_dir: '/assigned',
    population_lineage: {id: 'isolated', imports: [], root: '/population'}}, h.answers({}));
  await assert.rejects(wrongSetup.run(), /assigned private eval_dir\/workspace/);
  assert.equal(wrongSetup.rec.calls.length, 1, 'a misplaced population Setup cannot reach Analyze');
  for (const sc of cases) await runCase(sc);
  console.log('PASS: real EVO workflow CPU integration checks. No performance claim.');
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
