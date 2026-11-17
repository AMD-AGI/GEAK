// Warns when a change grows the workflow scripts by more than WARN_BYTES. Advisory: always exits 0.
//
// Run: node e2e_workflow/scripts/check_workflow_growth.js [BASE]
//
// BASE defaults to HEAD^1, the base parent of the merge commit a pull_request run checks out (the job
// needs fetch-depth: 2); locally pass e.g. origin/main. The working tree is compared against BASE for
// the scripts test_workflow_script_size.js checks, which is the hard gate on the workflow tool's limit.
// A warning annotation is emitted when one script, or all of them together, grow by more than
// WARN_BYTES net (env WORKFLOW_GROWTH_WARN_BYTES, default 10 KiB); bytes inserted are reported too.

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..', '..');
const DIRS = ['e2e_workflow', 'kernel_workflow'];
const CEILING_BYTES = 524288 - 4096;
const WARN_BYTES = parseInt(process.env.WORKFLOW_GROWTH_WARN_BYTES, 10) > 0
  ? parseInt(process.env.WORKFLOW_GROWTH_WARN_BYTES, 10) : 10240;
const BASE = process.argv[2] || 'HEAD^1';

const git = (args) => execFileSync('git', ['-C', ROOT, ...args],
  { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe'] });
// Workflow commands: % and newlines must be escaped in messages; , and : also in property values.
const esc = (s) => String(s).replace(/%/g, '%25').replace(/\r/g, '%0D').replace(/\n/g, '%0A');
const escProp = (s) => esc(s).replace(/:/g, '%3A').replace(/,/g, '%2C');
const size = (n) => `${(Math.abs(n) / 1024).toFixed(1)} KB`;
const kb = (n) => `${n < 0 ? '-' : n > 0 ? '+' : ''}${size(n)}`;
const num = (n) => n.toLocaleString('en-US');

function baseSizes() {
  const sizes = new Map();
  const out = git(['ls-tree', '-l', BASE, '--', ...DIRS.map((d) => `${d}/`)]);
  for (const line of out.split('\n')) {
    const m = line.match(/^\d+ blob [0-9a-f]+ +(\d+)\t(.+)$/);
    if (m && m[2].endsWith('.js') && m[2].split('/').length === 2) sizes.set(m[2], Number(m[1]));
  }
  return sizes;
}

function currentSizes() {
  const sizes = new Map();
  for (const dir of DIRS) {
    for (const f of fs.readdirSync(path.join(ROOT, dir)).filter((x) => x.endsWith('.js'))) {
      sizes.set(`${dir}/${f}`, fs.statSync(path.join(ROOT, dir, f)).size);
    }
  }
  return sizes;
}

// Bytes on added lines of the diff from BASE to the working tree (a rewrite counts in full).
function insertedBytes(rel) {
  let inHunk = false;
  let bytes = 0;
  for (const line of git(['diff', '--no-color', '--no-ext-diff', '-U0', BASE, '--', rel]).split('\n')) {
    if (line.startsWith('diff --git ')) inHunk = false;
    else if (line.startsWith('@@')) inHunk = true;
    else if (inHunk && line.startsWith('+')) bytes += Buffer.byteLength(line, 'utf8');
  }
  return bytes;
}

function main() {
  let base;
  try {
    if (BASE.startsWith('-')) throw new Error(`not a revision: ${BASE}`);
    base = baseSizes();
  } catch (e) {
    console.log(`::warning title=Workflow script growth::check skipped, cannot read ${esc(BASE)}: ` +
      esc((e.stderr || e.message || '').trim()));
    return;
  }
  const cur = currentSizes();
  const rows = [...new Set([...base.keys(), ...cur.keys()])].sort().map((rel) => {
    const before = base.get(rel) || 0;
    const after = cur.get(rel) || 0;
    return { rel, before, after, delta: after - before, inserted: base.has(rel) ? insertedBytes(rel) : after };
  });
  const total = rows.reduce((s, r) => s + r.delta, 0);
  const hint = `PRs that add over ${size(WARN_BYTES)} to the workflow scripts may not be mergeable as is; ` +
    'condense comments or move code out of the script.';

  const lines = [`workflow script growth vs ${BASE} (warn above ${num(WARN_BYTES)} bytes net):`];
  for (const r of rows) {
    lines.push(`  ${r.rel}: ${num(r.before)} -> ${num(r.after)} bytes (${kb(r.delta)}, ${num(r.inserted)} inserted)` +
      `, ${num(CEILING_BYTES - r.after)} under the ${num(CEILING_BYTES)} ceiling`);
  }
  lines.push(`  total: ${kb(total)} net`);
  console.log(lines.join('\n'));

  const grown = rows.filter((x) => x.delta > WARN_BYTES);
  for (const r of grown) {
    console.log(`::warning file=${escProp(r.rel)},title=Workflow script growth::${esc(r.rel)} grows by ` +
      `${kb(r.delta)} (${num(r.before)} -> ${num(r.after)} bytes, ${num(CEILING_BYTES - r.after)} left under ` +
      `the ${num(CEILING_BYTES)}-byte ceiling). ${hint}`);
  }
  if (!grown.length && total > WARN_BYTES) {
    console.log(`::warning title=Workflow script growth::the workflow scripts grow by ${kb(total)} in total. ${hint}`);
  }
  const warned = grown.length > 0 || total > WARN_BYTES;
  if (!warned) console.log(`ok: no workflow script growth over ${num(WARN_BYTES)} bytes`);

  if (process.env.GITHUB_STEP_SUMMARY) {
    const md = [
      `### Workflow script growth vs \`${BASE}\``, '',
      warned ? `**Warning:** over ${size(WARN_BYTES)} net growth. ${hint}` : 'No growth over the warning threshold.',
      '', '| Script | Base | This change | Net | Inserted | Under ceiling |', '|---|---:|---:|---:|---:|---:|',
      ...rows.map((r) => `| \`${r.rel}\` | ${num(r.before)} | ${num(r.after)} | ${kb(r.delta)} | ` +
        `${num(r.inserted)} | ${num(CEILING_BYTES - r.after)} |`),
      `| total | | | ${kb(total)} | | |`, '',
    ];
    fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, md.join('\n') + '\n');
  }
}

try {
  main();
} catch (e) {
  console.log(`::warning title=Workflow script growth::check failed: ${esc(e.message)}`);
}
