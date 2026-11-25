import {
  closeSync, existsSync, openSync, readFileSync, readSync, readdirSync, statSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, relative, resolve, sep } from 'node:path';

const lexical = (a, b) => (a < b ? -1 : (a > b ? 1 : 0));

export function sha256File(filePath) {
  return createHash('sha256').update(readFileSync(filePath)).digest('hex');
}

/** Match asm-workflow/backend/geak_asm_harness.py::_capture_identity exactly. */
export function captureIdentity(rootPath) {
  const root = resolve(rootPath);
  const digest = createHash('sha256');
  const files = [];
  const walk = (directory) => {
    const entries = readdirSync(directory, { withFileTypes: true });
    const regularFiles = [];
    const directories = [];
    for (const entry of entries) {
      const path = join(directory, entry.name);
      const stat = statSync(path, { bigint: true });
      if (entry.isDirectory()) directories.push({ path, name: entry.name });
      else if (stat.isFile()) regularFiles.push({ path, stat, name: entry.name });
    }
    regularFiles.sort((a, b) => lexical(a.name, b.name));
    files.push(...regularFiles);
    directories.sort((a, b) => lexical(a.name, b.name));
    for (const entry of directories) walk(entry.path);
  };
  walk(root);
  // Sort on path *components*, not the flat string: Python's pathlib compares
  // component-wise, so `module_variables/a.bin` precedes `module_variables.json`
  // ('.'(0x2E) > '\0'). Sorting the joined strings would invert that pair and
  // yield a digest that never matches geak_asm_harness.py::_capture_identity.
  const sortKey = (p) => relative(root, p).split(sep).join('\0');
  files.sort((a, b) => lexical(sortKey(a.path), sortKey(b.path)));
  for (const { path, stat } of files) {
    const rel = relative(root, path);
    if (rel === 'output' || rel.startsWith('output/')) continue;
    digest.update(`${rel}\0${stat.size}\n`);
    const fd = openSync(path, 'r');
    try {
      const chunk = Buffer.alloc(1024 * 1024);
      let position = 0;
      while (position < Number(stat.size)) {
        const count = readSync(fd, chunk, 0, chunk.length, position);
        if (count === 0) break;
        digest.update(chunk.subarray(0, count));
        position += count;
      }
    } finally {
      closeSync(fd);
    }
  }
  return digest.digest('hex');
}

export function verifyFormalArtifacts(artifact, reportPath = '',
                                      expectedPolicySha256 = undefined) {
  const candidateRelative = reportPath && artifact && artifact.candidate_path
    ? relative(dirname(resolve(reportPath)), resolve(artifact.candidate_path))
    : '';
  // One generic message for every way this can fail costs hours: a rebuilt
  // candidate and a capture that moved read identically in the log. Name the
  // artifact that disagreed, and tag the error so the caller can tell a
  // per-worker problem from a capture that drifted under the whole run.
  const fail = (detail, component = '') => {
    const error = new Error(
      `formal report artifact identity does not match workspace files: ${detail}`);
    error.artifactComponent = component;
    throw error;
  };
  if (!artifact) fail('the report carries no artifact block');
  for (const key of ['baseline_path', 'candidate_path', 'capture_path']) {
    if (!artifact[key]) fail(`the artifact block has no ${key}`);
  }
  if (reportPath
      && (candidateRelative === '..' || candidateRelative.startsWith(`..${sep}`))) {
    fail(`candidate ${artifact.candidate_path} lies outside the report directory `
      + `${dirname(resolve(reportPath))}`, 'candidate');
  }
  const components = [
    ['baseline', artifact.baseline_path, artifact.baseline_sha256,
      () => sha256File(artifact.baseline_path)],
    ['candidate', artifact.candidate_path, artifact.candidate_sha256,
      () => sha256File(artifact.candidate_path)],
    ['capture', artifact.capture_path, artifact.capture_identity_sha256,
      () => captureIdentity(artifact.capture_path)],
  ];
  for (const [name, path, expected, compute] of components) {
    let actual;
    try {
      actual = compute();
    } catch (error) {
      fail(`${name} ${path} is unreadable (${error.code || error.message})`, name);
    }
    if (actual !== expected) {
      fail(`${name} ${path} now hashes to ${actual}, but the report recorded `
        + `${expected}`, name);
    }
  }
  let validationPolicySha256 = '';
  if (expectedPolicySha256 !== undefined) {
    const policyPath = join(artifact.capture_path, 'validation_policy.json');
    validationPolicySha256 = existsSync(policyPath) ? sha256File(policyPath) : '';
    if (validationPolicySha256 !== String(expectedPolicySha256 || '')) {
      throw new Error(
        'formal report validation policy identity does not match capture');
    }
  }
  return {
    ...artifact,
    disk_verified: true,
    ...(expectedPolicySha256 !== undefined
      ? { validation_policy_sha256: validationPolicySha256 } : {}),
  };
}
