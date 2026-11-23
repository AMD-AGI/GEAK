// Tests check_workflow_growth.js: warns on net growth over the threshold, never fails.
//
// Run: node e2e_workflow/scripts/test_check_workflow_growth.js
//
// Each case copies the check into a scratch git repo (it resolves the repo from its own path), so
// nothing here reads or changes this checkout. Needs node and git only.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync, spawnSync } = require('child_process');

const CHECK = path.join(__dirname, 'check_workflow_growth.js');
const LANE = 'kernel_workflow/kernel_lane.js';
const E2E = 'e2e_workflow/e2e_workflow.js';

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const fill = (n) => `${'// x\n'.repeat(Math.floor(n / 5))}${'y'.repeat(n % 5)}`;
const scratch = [];
process.on('exit', () => scratch.forEach((d) => fs.rmSync(d, { recursive: true, force: true })));

function repo() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wf-growth-'));
  scratch.push(dir);
  const git = (...args) => execFileSync('git', ['-C', dir, '-c', 'user.name=t', '-c', 'user.email=t@t',
    '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', ...args], { stdio: 'pipe', encoding: 'utf8' });
  git('init', '-q');
  git('checkout', '-q', '-b', 'main');
  const write = (rel, text) => {
    fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true });
    fs.writeFileSync(path.join(dir, rel), text);
  };
  write(E2E, fill(30000));
  write(LANE, fill(20000));
  write('e2e_workflow/scripts/check_workflow_growth.js', fs.readFileSync(CHECK));
  git('add', '-A');
  git('commit', '-q', '-m', 'base');
  const run = (args = [], env = {}) => {
    const r = spawnSync(process.execPath, [path.join(dir, 'e2e_workflow/scripts/check_workflow_growth.js'), ...args],
      { encoding: 'utf8', env: { ...process.env, GITHUB_STEP_SUMMARY: '', ...env } });
    return { status: r.status, out: r.stdout + r.stderr };
  };
  return { dir, git, write, run };
}

const fileWarning = (rel) => `::warning file=${rel},title=Workflow script growth::`;
const totalWarning = '::warning title=Workflow script growth::the workflow scripts grow by';

{
  const r = repo();
  const res = r.run(['HEAD']);
  ok(res.status === 0 && res.out.includes('ok: no workflow script growth') && !res.out.includes('::warning'),
    'no change: no warning');
}
{
  const r = repo();
  r.write(LANE, fill(20000 + 11 * 1024));
  const res = r.run(['HEAD']);
  ok(res.status === 0 && res.out.includes(fileWarning(LANE)), 'one script grows 11 KB: warning on that file, exit 0');
  ok(!res.out.includes(totalWarning), 'a per-file warning is not repeated as a total');
}
{
  const r = repo();
  r.write(LANE, fill(20000 + 6 * 1024));
  r.write(E2E, fill(30000 + 6 * 1024));
  const res = r.run(['HEAD']);
  ok(res.status === 0 && res.out.includes(totalWarning) && !res.out.includes('::warning file='),
    'two scripts grow 6 KB each: one total warning');
}
{
  const r = repo();
  r.write(LANE, 'z\n'.repeat(6000));
  const res = r.run(['HEAD']);
  ok(res.status === 0 && !res.out.includes('::warning') && /kernel_lane\.js: [^\n]*\(-7\.8 KB, 12,000 inserted\)/.test(res.out),
    'a rewrite that shrinks the script: no warning, inserted bytes still reported');
}
{
  const r = repo();
  r.git('checkout', '-q', '-b', 'pr');
  r.write(LANE, fill(20000 + 11 * 1024));
  r.git('commit', '-q', '-am', 'grow');
  r.git('checkout', '-q', 'main');
  r.git('merge', '-q', '--no-ff', '-m', 'merge', 'pr');
  const res = r.run();
  ok(res.status === 0 && res.out.includes(fileWarning(LANE)), 'default base HEAD^1 of a merge commit sees the PR growth');
}
{
  const r = repo();
  const res = r.run(['HEAD~5']);
  ok(res.status === 0 && res.out.includes('::warning title=Workflow script growth::check skipped'),
    'missing base: skipped with a warning, exit 0');
  const dash = r.run(['--output=x']);
  ok(dash.status === 0 && dash.out.includes('check skipped') && !fs.existsSync(path.join(r.dir, 'x')),
    'an option-like base is refused, not passed to git');
}
{
  const r = repo();
  r.write(LANE, fill(20200));
  const res = r.run(['HEAD'], { WORKFLOW_GROWTH_WARN_BYTES: '100' });
  ok(res.status === 0 && res.out.includes(fileWarning(LANE)), 'WORKFLOW_GROWTH_WARN_BYTES lowers the threshold');
}
{
  const r = repo();
  r.write(LANE, fill(20000 + 11 * 1024));
  const summary = path.join(r.dir, 'summary.md');
  r.run(['HEAD'], { GITHUB_STEP_SUMMARY: summary });
  const md = fs.existsSync(summary) ? fs.readFileSync(summary, 'utf8') : '';
  ok(md.includes(`| \`${LANE}\` | 20,000 | 31,264 | +11.0 KB |`) && md.includes('**Warning:**'),
    'step summary lists the per-script sizes');
}

console.log(failures === 0
  ? '\nPASS: the workflow growth check warns without failing.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
