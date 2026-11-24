import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const HERE = path.dirname(new URL(import.meta.url).pathname);
const WORKFLOW_DIR = path.resolve(HERE, '..');
const SHIM = path.join(WORKFLOW_DIR, 'run_avo.mjs');
const SCRIPT = path.join(WORKFLOW_DIR, 'avo_workflow.js');

test('validate-only accepts a full config without launching agents', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'avo-shim-check-'));
  try {
    const kernel = path.join(root, 'kernel');
    mkdirSync(kernel);
    const config = path.join(root, 'config.json');
    writeFileSync(config, JSON.stringify({
      scriptPath: SCRIPT,
      args: {
        kernel_path: kernel,
        workflow_dir: WORKFLOW_DIR,
        exp_root: path.join(root, 'exp'),
      },
    }));
    const result = JSON.parse(execFileSync(
      process.execPath, [SHIM, config, '--validate-only'], { encoding: 'utf8' }));
    assert.equal(result.status, 'ok');
    assert.equal(result.entrypoint, 'run_avo.mjs');
    assert.equal(result.operator, 'team');
    assert.equal(result.start_policy, 'diverse');
    assert.equal(result.launched, false);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('validate-only rejects low-level targets before launch', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'avo-shim-check-'));
  try {
    const kernel = path.join(root, 'kernel');
    mkdirSync(kernel);
    const config = path.join(root, 'config.json');
    writeFileSync(config, JSON.stringify({
      scriptPath: SCRIPT,
      args: {
        kernel_path: kernel,
        workflow_dir: WORKFLOW_DIR,
        target_language: 'llvm-ir',
      },
    }));
    const result = spawnSync(
      process.execPath, [SHIM, config, '--validate-only'], { encoding: 'utf8' });
    assert.equal(result.status, 2);
    assert.match(result.stderr, /source-level optimization only/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
