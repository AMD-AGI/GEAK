#!/usr/bin/env node
// Orchestration checks for tools/hip_ep_e2e/hip_ep_e2e_workflow.js (no GPU, no agents).
// The script runs in a stub sandbox where agent() answers by label. Checks the phase order, what
// reaches the tools and the report, the refusals, and that the long-command template works when an
// agent runs it as written.
//
// Run:  node tools/hip_ep_e2e/tests/test_workflow.js
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const HERE = path.join(__dirname, '..');
const SRC = fs.readFileSync(path.join(HERE, 'hip_ep_e2e_workflow.js'), 'utf8');
ok(!SRC.includes('import('), 'no import( (the native Workflow sandbox refuses it at launch)');
const titles = [...SRC.matchAll(/\{ title: '([^']+)'/g)].map((m) => m[1]);
const used = [...SRC.matchAll(/\bphase\('([^']+)'\)/g)].map((m) => m[1]);
ok(used.every((t) => titles.includes(t)), 'every phase() title is declared in meta.phases');

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const body = SRC.replace('export const meta =', 'const meta =');
const run = new AsyncFunction('args', 'agent', 'phase', 'log', 'setTimeout', body);

const ARGS = { model_path: '/m/q27b', hipep_pkg: '/pkg/rel', tools_dir: '/geak/tools/hip_ep_e2e',
  exp_root: '/exp', therock: '/tr', sysroot: '/sys', py314: '/py/bin/python', hipep_src: '/src',
  hipep_build: '/build' };

function sandbox(answers) {
  const calls = { phases: [], agents: [], prompts: {}, schemas: {} };
  const agent = async (prompt, opts) => {
    calls.agents.push(opts.label);
    calls.prompts[opts.label] = prompt;
    calls.schemas[opts.label] = opts.schema;
    const a = answers[opts.label];
    return typeof a === 'function' ? a(prompt) : a;
  };
  return { calls, agent, phase: (t) => calls.phases.push(t), log: () => {} };
}
const go = (sb, args = ARGS) => run(args, sb.agent, sb.phase, sb.log, setTimeout);
const refusal = async (answers, args = ARGS) => {
  try { await go(sandbox(answers), args); } catch (e) { return String(e); }
  return '';
};

const SETUP = { ok: true, error: '', eval_dir: '/exp/e2e_q27b', device_gfx: 'gfx1151',
  baseline_json: '/b.json', noise_from: '/aa/ab.json', aa_parity_exact: true,
  baseline: { 1024: { ttft_us: 2.05e6 } } };
const ANSWERS = {
  'director:setup': SETUP,
  'profiler:profile': { heads_json: '/h.json',
    prefill: [{ base: 'MatMulNBitsWMMA_NoZP', share: 0.82 }],
    decode: [{ base: 'matmul_nbits_gemv_dp4a_kernel', share: 0.93 }] },
  'validator:validate': { report_md: '# report\n' },
  persist: { written: true },
};

(async () => {
  console.log('\n# a run');
  const sb = sandbox(ANSWERS);
  const ret = await go(sb);
  ok(JSON.stringify(sb.calls.phases) === JSON.stringify(['Setup', 'Profile', 'Validate']), 'phases run in order');
  ok(/DEVICE_GFX: gfx1151/.test(sb.calls.prompts['validator:validate']) &&
    /AA_PARITY_EXACT: true/.test(sb.calls.prompts['validator:validate']) &&
    /HEADS_JSON: \/h\.json/.test(sb.calls.prompts['validator:validate']) &&
    ret.aa_parity_exact === true && ret.device_gfx === 'gfx1151', "Setup's and Profile's findings reach the report and the return");
  ok(ret.report_md === '# report\n' && /report\.md/.test(sb.calls.prompts.persist),
    'the report comes back as text and is saved next to workflow_return.json');
  const toolEnv = JSON.parse(/TOOL_ENV: (\{.*\})/.exec(sb.calls.prompts['director:setup'])[1]);
  const pathsBlock = /^PATHS = \{([\s\S]*?)^\}/m.exec(fs.readFileSync(path.join(HERE, 'hipep_env.py'), 'utf8'))[1];
  const needed = [...pathsBlock.matchAll(/"(HIPEP_\w+)":/g)].map((m) => m[1]);
  ok(needed.length === 6 && needed.every((k) => toolEnv[k]), `TOOL_ENV sets every path the tools read (${needed.join(', ')})`);
  ok(toolEnv.HIPEP_PKG === '/pkg/rel', 'HIPEP_PKG is the release package');

  console.log('\n# refusals');
  ok(await refusal({ ...ANSWERS, 'director:setup': { ...SETUP, device_gfx: 'gfx1201' } }) === '',
    'any device the package runs on is accepted; the arch is recorded, not checked');
  ok(/Setup failed: build failed; see .*kbuild/.test(await refusal({ ...ANSWERS, 'director:setup':
    { ...SETUP, ok: false, error: 'build failed; see /exp/e2e_q27b/kbuild/build.log', baseline_json: '' } })),
  "a failed Setup stops the run with the director's reason");
  ok(/Setup failed: no A\/A result/.test(await refusal({ ...ANSWERS, 'director:setup': { ...SETUP, noise_from: '' } })),
    'an A/A that produced nothing stops the run');
  ok(await refusal({ ...ANSWERS, 'director:setup': { ...SETUP, noise_from: '' } }, { ...ARGS, aa: 'false' }) === '',
    'with aa=false no A/A is expected');
  ok(/Setup failed: the director returned no result/.test(await refusal({ ...ANSWERS, 'director:setup': null })),
    'a silent director stops the run');
  for (const k of ['tools_dir', 'therock', 'hipep_src']) {
    ok(new RegExp(`args.${k} is required`).test(await refusal(ANSWERS, { ...ARGS, [k]: '' })), `args.${k} is required`);
  }

  console.log('\n# the A/A candidate is the rebuild Setup makes');
  const director = fs.readFileSync(path.join(HERE, 'roles', 'director.md'), 'utf8');
  ok(/build_kernels\.sh - EVAL_DIR\/kbuild/.test(director) &&
    /make_pkg\.sh BASE_PKG EVAL_DIR\/kbuild\/libcustom_kernels_\*\.so EVAL_DIR\/pkg\/aa/.test(director),
  'the A/A packages the library Setup rebuilt, not a copy of the shipped one');

  console.log('\n# the long-command template, run as an agent would run it');
  const prompt = sandbox(ANSWERS);
  await go(prompt);
  const lines = prompt.calls.prompts['director:setup'].split('\n').map((l) => l.trim());
  const prep = lines.find((l) => l.startsWith('mkdir -p DIR'));
  const start = lines.find((l) => l.startsWith('setsid nohup'));
  const dir = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'hipep-long-')), 'new', 'dir');
  const fill = (l) => l.replace(/DIR/g, dir).replace(/NAME/g, 'job').replace('<the command>', 'sh -c "sleep 2; exit 3"');
  execFileSync('bash', ['-c', fill(prep)]);
  const t0 = Date.now();
  // The agent's Bash tool runs with job control on, which makes setsid fork: `set -m` does the same.
  execFileSync('bash', ['-c', `set -m; ${fill(start)}`], { stdio: ['ignore', 'pipe', 'pipe'] });
  ok(Date.now() - t0 < 1500, 'starting a long step returns at once, into a dir that did not exist');
  let pid = 0;
  for (let i = 0; i < 50 && !pid; i++) {
    execFileSync('sleep', ['0.1']);
    pid = fs.existsSync(path.join(dir, 'job.pid')) ? parseInt(fs.readFileSync(path.join(dir, 'job.pid'), 'utf8'), 10) : 0;
  }
  let sid = NaN;
  try { sid = parseInt(execFileSync('ps', ['-o', 'sid=', '-p', String(pid)]).toString(), 10); } catch (e) { /* gone */ }
  ok(sid === pid, 'the pid file names the running step itself, a session leader, even with job control on');
  let rc = '';
  for (let i = 0; i < 100 && !rc; i++) {
    execFileSync('sleep', ['0.1']);
    rc = fs.existsSync(path.join(dir, 'job.rc')) ? fs.readFileSync(path.join(dir, 'job.rc'), 'utf8').trim() : '';
  }
  ok(rc === '3', "the .rc file holds the command's exit code");

  console.log(failures ? `\n${failures} failure(s)` : '\nall passed');
  process.exit(failures ? 1 : 0);
})();
