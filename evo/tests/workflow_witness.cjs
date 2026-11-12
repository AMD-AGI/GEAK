const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const childProcess = require('node:child_process');

const root = path.resolve(__dirname, '../..');
const fixture = process.argv[2];
const scenario = process.argv[3];
const calls = [];
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const laneSource = fs.readFileSync(path.join(root, 'kernel_workflow/kernel_lane.js'), 'utf8');
const factoryMarker = '\nfunction createEvolution() {';
assert.equal(laneSource.split(factoryMarker).length, 2, 'Shared EVO factory must occur exactly once');
const factorySource = laneSource.slice(laneSource.indexOf(factoryMarker));

function command(program, values, cwd) {
  return childProcess.execFileSync(program, values, { cwd, encoding: 'utf8', timeout: 30000 });
}

async function helperAgent(prompt, options) {
  calls.push({ label: options.label, prompt });
  const responses = {
    'false-receipt': { exit_code: 0, stdout: JSON.stringify({ ok: false, command: 'init' }) },
    'structured-error': { exit_code: 2, stderr: 'command failed', stdout: JSON.stringify({
      ok: false, error: { code: 'init_conflict', message: 'Initialization differs from captured founder' },
      verification: 'not_performed',
    }) },
    'missing-result': null,
    'missing-stdout': { exit_code: 0 },
    'non-string-stdout': { exit_code: 0, stdout: {} },
    'malformed-stdout': { exit_code: 0, stdout: 'not JSON' },
    'empty-stdout': { exit_code: 0, stdout: '' },
    'null-receipt': { exit_code: 0, stdout: 'null' },
    'nonzero-malformed-stdout': { exit_code: 2, stdout: 'not JSON', stderr: 'helper crashed' },
    'nonzero-missing-stdout': { exit_code: 2, stderr: 'helper crashed' },
    'nonzero-invalid-diagnostic': { exit_code: 2, stderr: 'helper crashed',
      stdout: JSON.stringify({ ok: false, error: { code: 123, message: 'invalid diagnostic' } }) },
    'nonzero-success-receipt': { exit_code: 2, stdout: JSON.stringify({
      ok: true, command: 'init', archive: path.join(fixture, 'eval', 'evolution_archive'),
    }) },
  };
  if (Object.hasOwn(responses, scenario)) return responses[scenario];
  const script = prompt.split('```bash\n')[1].split('\n```')[0];
  const result = childProcess.spawnSync('bash', ['-c', script], { encoding: 'utf8', timeout: 30000 });
  if (scenario === 'helper-error' && options.label === 'evo:query') {
    assert.equal(result.status, 2);
    assert.equal(result.stderr, '');
    const receipt = JSON.parse(result.stdout);
    assert.equal(receipt.ok, false);
    assert.deepEqual(receipt.error, { code: 'unknown_source', message: 'Unknown entry: missing-parent' });
  }
  return { exit_code: result.status, stdout: result.stdout, stderr: result.stderr };
}

const exercise = new AsyncFunction('A', 'WORKFLOW_DIR', 'CANONICAL', 'EVAL_DIR', 'agentT', 'obj', 'log',
  'fs', 'path', 'assert', 'command', 'scenario', 'analysis',
  'let round = 1;\n' + factorySource + `
  const evolution = createEvolution();
  assert.deepEqual(Object.keys(evolution).sort(),
    ['initialize', 'plannerInput', 'prepare', 'engineerInput', 'recordBatch', 'checkPatch', 'run', 'result'].sort());
  await evolution.initialize();
  if (scenario === 'helper-error') return evolution.run('query', ['--entry', 'missing-parent']);
  assert.equal(evolution.result().archive, path.join(EVAL_DIR, 'evolution_archive'));
  assert.deepEqual(evolution.plannerInput().source_paths, ['kernel.py']);
  const direction = { id: 'first', idx: 0, title: 'Source custody witness', out_dir: path.join(EVAL_DIR, 'round_1', 'engineer_0') };
  const prepared = await evolution.prepare(direction);
  assert.equal(prepared.receipt.requested_parent_id, 'INCUMBENT');
  const workspace = path.join(direction.out_dir, 'workspace');
  fs.writeFileSync(path.join(workspace, 'kernel.py'), 'value = 3\\n');
  const patch = path.join(direction.out_dir, 'best_patch.diff');
  fs.writeFileSync(patch, command('git', ['diff', '--binary'], workspace));
  const custody = await evolution.checkPatch(direction, patch);
  assert.ok(custody);
  assert.equal(custody.patch_object_path,
    path.join(evolution.result().archive, 'objects', custody.patch_sha256));
  assert.equal(fs.readFileSync(custody.patch_object_path, 'utf8'), fs.readFileSync(patch, 'utf8'));
  fs.writeFileSync(path.join(workspace, 'kernel.py'), 'value = 4\\n');
  await evolution.recordBatch([{ d: direction, patch: custody.patch_object_path,
    eng: { status: 'ok', speedup_geomean: 1.2 }, ver: { status: 'verified' } }]);
  assert.equal(evolution.result().errors.length, 0);
  fs.writeFileSync(path.join(CANONICAL, 'kernel.py'), 'value = 3\\n');
  command('git', ['add', 'kernel.py'], CANONICAL);
  round = 2;
  const successor = { id: 'second', idx: 0, title: 'Reuse older work', parent_id: 'r1_d0-development',
    donor_ids: ['r1_d0-best'], out_dir: path.join(EVAL_DIR, 'round_2', 'engineer_0') };
  fs.rmSync(path.join(EVAL_DIR, 'round_1'), { recursive: true });
  successor.evo_prepared = await evolution.prepare(successor);
  const successorWorkspace = path.join(successor.out_dir, 'workspace');
  assert.equal(fs.readFileSync(path.join(successorWorkspace, 'kernel.py'), 'utf8'), 'value = 4\\n');
  assert.equal(command('git', ['show', ':kernel.py'], successorWorkspace), 'value = 3\\n');
  assert.equal(fs.readFileSync(path.join(CANONICAL, 'kernel.py'), 'utf8'), 'value = 3\\n');
  assert.equal(fs.readFileSync(path.join(successorWorkspace, 'oracle.py'), 'utf8'), 'immutable oracle\\n');
  assert.equal(evolution.engineerInput(successor).parent_id, 'r1_d0-development');
  assert.deepEqual(evolution.engineerInput(successor).donor_ids, ['r1_d0-best']);
  fs.writeFileSync(path.join(successorWorkspace, 'kernel.py'), 'value = 5\\n');
  fs.writeFileSync(path.join(successor.out_dir, 'best_patch.diff'), command('git', ['diff', '--binary'], successorWorkspace));
  await evolution.recordBatch([{ d: successor, eng: { status: 'failed' }, ver: null }]);
  assert.equal(evolution.result().errors.length, 0);
  const index = await evolution.run('query', []);
  assert.equal(index.index.original_benchmark.status, 'external_not_captured');
  assert.equal(index.index.entries['r2_d0-development'].verification.independently_verified, false);
  assert.equal(index.index.entries['r2_d0-development'].parent_binding.requested_parent_id, 'r1_d0-development');
  return { ok: true, ...evolution.result(), entries: Object.keys(index.index.entries) };
`);

exercise({ optimizer: 'evo', evolution_source_paths: ['kernel.py'] }, path.join(root, 'kernel_workflow'),
  path.join(fixture, 'canonical'), path.join(fixture, 'eval'), helperAgent,
  (properties, required) => ({ type: 'object', properties, required }), () => {},
  fs, path, assert, command, scenario, { modifiable_files: ['kernel.py'] }).then(result => {
    process.stdout.write(JSON.stringify({ ...result, calls: calls.map(call => call.label) }) + '\n');
  }).catch(error => {
    process.stderr.write(String(error.stack || error) + '\n');
    process.exitCode = 1;
  });
