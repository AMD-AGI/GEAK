const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const childProcess = require('node:child_process');

const root = path.resolve(__dirname, '../..');
const fixture = process.argv[2];
const scenario = process.argv[3];
const calls = [];
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

function command(program, values, cwd) {
  return childProcess.execFileSync(program, values, { cwd, encoding: 'utf8', timeout: 30000 });
}

async function helperAgent(prompt, options) {
  calls.push({ label: options.label, prompt });
  if (scenario === 'false-receipt') {
    return { exit_code: 0, stdout: JSON.stringify({ ok: false, command: 'init' }) };
  }
  const script = prompt.split('```bash\n')[1].split('\n```')[0];
  const result = childProcess.spawnSync('bash', ['-c', script], { encoding: 'utf8', timeout: 30000 });
  return { exit_code: result.status, stdout: result.stdout, stderr: result.stderr };
}

const exercise = new AsyncFunction('A', 'WORKFLOW_DIR', 'CANONICAL', 'EVAL_DIR', 'agentT', 'obj', 'log',
  'fs', 'path', 'assert', 'command', 'scenario',
  'let round = 1;\n' + fs.readFileSync(path.join(root, 'evo/extension.js'), 'utf8') + `
  await evoInitialize();
  const direction = { id: 'first', idx: 0, title: 'Source custody witness', out_dir: path.join(EVAL_DIR, 'round_1', 'engineer_0') };
  const prepared = await evoPrepare(direction);
  assert.equal(prepared.receipt.requested_parent_id, 'INCUMBENT');
  const workspace = path.join(direction.out_dir, 'workspace');
  fs.writeFileSync(path.join(workspace, 'kernel.py'), 'value = 3\\n');
  const patch = path.join(direction.out_dir, 'best_patch.diff');
  fs.writeFileSync(patch, command('git', ['diff', '--binary'], workspace));
  assert.ok(await evoCheckPatch(direction, patch));
  fs.writeFileSync(path.join(workspace, 'kernel.py'), 'value = 4\\n');
  await evoRecordBatch([{ d: direction, eng: { status: 'ok', speedup_geomean: 1.2 }, ver: { status: 'verified' } }]);
  assert.equal(EVO_ERRORS.length, 0);
  fs.writeFileSync(path.join(CANONICAL, 'kernel.py'), 'value = 3\\n');
  command('git', ['add', 'kernel.py'], CANONICAL);
  round = 2;
  const successor = { id: 'second', idx: 0, title: 'Reuse older work', parent_id: 'r1_d0-development',
    donor_ids: ['r1_d0-best'], out_dir: path.join(EVAL_DIR, 'round_2', 'engineer_0') };
  fs.rmSync(path.join(EVAL_DIR, 'round_1'), { recursive: true });
  successor.evo_prepared = await evoPrepare(successor);
  const successorWorkspace = path.join(successor.out_dir, 'workspace');
  assert.equal(fs.readFileSync(path.join(successorWorkspace, 'kernel.py'), 'utf8'), 'value = 4\\n');
  assert.equal(command('git', ['show', ':kernel.py'], successorWorkspace), 'value = 3\\n');
  assert.equal(fs.readFileSync(path.join(CANONICAL, 'kernel.py'), 'utf8'), 'value = 3\\n');
  assert.equal(fs.readFileSync(path.join(successorWorkspace, 'oracle.py'), 'utf8'), 'immutable oracle\\n');
  assert.equal(evoEngineerInput(successor).parent_id, 'r1_d0-development');
  fs.writeFileSync(path.join(successorWorkspace, 'kernel.py'), 'value = 5\\n');
  fs.writeFileSync(path.join(successor.out_dir, 'best_patch.diff'), command('git', ['diff', '--binary'], successorWorkspace));
  await evoRecordBatch([{ d: successor, eng: { status: 'failed' }, ver: null }]);
  assert.equal(EVO_ERRORS.length, 0);
  const index = await evoRun('query', []);
  assert.equal(index.index.original_benchmark.status, 'external_not_captured');
  assert.equal(index.index.entries['r2_d0-development'].verification.independently_verified, false);
  assert.equal(index.index.entries['r2_d0-development'].parent_binding.requested_parent_id, 'r1_d0-development');
  return { ok: true, errors: EVO_ERRORS, entries: Object.keys(index.index.entries) };
`);

exercise({ optimizer: 'evo', evolution_source_paths: ['kernel.py'] }, path.join(root, 'kernel_workflow'),
  path.join(fixture, 'canonical'), path.join(fixture, 'eval'), helperAgent,
  (properties, required) => ({ type: 'object', properties, required }), () => {},
  fs, path, assert, command, scenario).then(result => {
    process.stdout.write(JSON.stringify({ ...result, calls: calls.map(call => call.label) }) + '\n');
  }).catch(error => {
    process.stderr.write(String(error.stack || error) + '\n');
    process.exitCode = 1;
  });
