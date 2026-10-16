#!/usr/bin/env node
// Build the single-file workflow scripts from their source parts.
//
// Both workflow runners (interface/runtime/engine/run_workflow.mjs and Claude Code's
// native Workflow tool) load exactly ONE script file with no import/require, so the
// committed kernel_workflow/*.js and e2e_workflow/*.js stay single files. They are
// generated: edit the parts, then run this script. Each target is the concatenation
// of the parts listed for it in build_workflows.json, in order, with a one-line
// "generated" notice after the first part. The first part holds `export const meta`,
// which the native Workflow tool requires at the very top of the file. A part may be
// listed under several targets (shared helpers).
//
//   node scripts/build_workflows.mjs           rewrite every target
//   node scripts/build_workflows.mjs --check   exit 1 if any committed target differs
//                                              from a fresh build (CI), or if a part
//                                              file is not listed in the manifest
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(readFileSync(join(ROOT, 'scripts/build_workflows.json'), 'utf8'));
const check = process.argv.includes('--check');
const problems = [];

const listed = new Set(Object.values(manifest).flat());
for (const dir of new Set([...listed].map((p) => dirname(p)))) {
  for (const name of readdirSync(join(ROOT, dir)).filter((n) => n.endsWith('.js')).sort()) {
    if (!listed.has(`${dir}/${name}`)) problems.push(`${dir}/${name}: part not listed in build_workflows.json`);
  }
}

for (const [target, parts] of Object.entries(manifest)) {
  const notice = Buffer.from(`// GENERATED from ${dirname(parts[1])}/ by scripts/build_workflows.mjs. ` +
    'Edit the parts there, then rebuild.\n');
  const [first, ...rest] = parts.map((p) => readFileSync(join(ROOT, p)));
  const built = Buffer.concat([first, notice, ...rest]);
  const path = join(ROOT, target);
  if (check) {
    let current = null;
    try { current = readFileSync(path); } catch { /* missing target is reported below */ }
    if (!current || !current.equals(built)) {
      problems.push(`${target}: differs from a fresh build of its ${parts.length} parts ` +
        '(edit the parts, then run node scripts/build_workflows.mjs)');
    }
  } else {
    writeFileSync(path, built);
    console.log(`built ${target} from ${parts.length} parts (${built.length} bytes)`);
  }
}

if (problems.length) {
  for (const p of problems) console.error(`build_workflows: ${p}`);
  process.exit(1);
}
if (check) console.log(`build_workflows: ${Object.keys(manifest).length} target(s) up to date`);
