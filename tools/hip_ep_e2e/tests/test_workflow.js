#!/usr/bin/env node
// Orchestration checks for tools/hip_ep_e2e/hip_ep_e2e_workflow.js (no GPU, no agents).
// The script runs in a stub sandbox: agent() answers by label, workflow() records the nested
// kernel_workflow call. Checks the phase order, when the lanes run, how their verdicts move the
// current package, what reaches the tools, the kernel lane and the validator, the refusals, and that
// the long-command template works when an agent runs it as written.
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
const run = new AsyncFunction('args', 'agent', 'phase', 'log', 'workflow', 'setTimeout', body);

const ARGS = { model_path: '/m/q27b', hipep_pkg: '/pkg/rel', tools_dir: '/geak/tools/hip_ep_e2e',
  kernel_workflow_dir: '/geak/kernel_workflow', exp_root: '/exp', therock: '/tr', sysroot: '/sys', py314: '/py/bin/python', hipep_src: '/src',
  hipep_build: '/build' };

function sandbox(answers) {
  const calls = { phases: [], agents: [], prompts: {}, schemas: {}, workflows: [] };
  const agent = async (prompt, opts) => {
    calls.agents.push(opts.label);
    calls.prompts[opts.label] = prompt;
    calls.schemas[opts.label] = opts.schema;
    const a = answers[opts.label];
    return typeof a === 'function' ? a(prompt) : a;
  };
  const workflow = async (spec, a) => { calls.workflows.push({ spec, args: a }); return answers.kernel; };
  return { calls, agent, workflow, phase: (t) => calls.phases.push(t), log: () => {} };
}
const go = (sb, args = ARGS) => run(args, sb.agent, sb.phase, sb.log, sb.workflow, setTimeout);
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
    prefill: [{ base: 'MatMulNBitsWMMA_NoZP', share: 0.15 }, { base: 'MatMulNBitsWMMAX_NoZP', share: 0.67 },
      { base: 'gatedDeltaNetRecurrentKernel', share: 0.077 }],
    decode: [{ base: 'matmul_nbits_gemv_dp4a_kernel', share: 0.93 }] },
  'tuner:tuning': { accepted: true, reason: 'TTFT -9%', pkg: '/pkg/lut1', overlay: '/o/lut',
    ab_json: '/t/ab.json', parity_json: '/t/p.json' },
  'extractor:extract': { workload_json: '/task/workload.json', status: 'PASS - 12/12 shapes, ALL PASS',
    steer: 'prefill WMMA' },
  clock: { epoch: 1000 },
  kernel: { final_speedup: 1.03, final_patch: '/k/final_patch.diff', eval_dir: '/k',
    validation_status: 'accepted' },
  'integrator:integrate': { accepted: false, reason: 'inside noise', pkg: '/pkg/k1',
    overlay: '/o/k1', ab_json: '/i/ab.json', parity_json: '/i/p.json' },
  'validator:validate': { validation_status: 'validated', report_md: '# report\n', per_len: {} },
  persist: { written: true },
};

(async () => {
  console.log('\n# tuning accepted, kernel candidate rejected at the gate');
  let sb = sandbox(ANSWERS);
  let ret = await go(sb);
  ok(JSON.stringify(sb.calls.phases) === JSON.stringify(
    ['Setup', 'Profile', 'Strategize', 'Tuning', 'HeadKernel', 'Integrate', 'Validate']), 'phases run in order');
  ok(/BASE_PKG: \/pkg\/rel/.test(sb.calls.prompts['tuner:tuning']), 'the candidate is gated against the release');
  ok(ret.final_pkg === '/pkg/lut1' && ret.final_overlay === '/o/lut' && ret.final_label === 'tuning',
    'the validated candidate becomes final');
  ok(/FINAL_PKG: \/pkg\/lut1/.test(sb.calls.prompts['validator:validate']) &&
    /BASE_PKG: \/pkg\/rel/.test(sb.calls.prompts['validator:validate']),
  'the validator re-measures the final package against the release');
  ok(/DEVICE_GFX: gfx1151/.test(sb.calls.prompts['validator:validate']) &&
    /AA_PARITY_EXACT: true/.test(sb.calls.prompts['validator:validate']) &&
    ret.aa_parity_exact === true && ret.device_gfx === 'gfx1151', "Setup's findings reach the report and the return");
  ok(ret.report_md === '# report\n' && /report\.md/.test(sb.calls.prompts.persist),
    'the report comes back as text and is saved next to workflow_return.json');
  ok(Math.abs(ret.lanes[0].shares.matmul_nbits - 0.82) < 1e-9, 'the matmul_nbits share sums every MatMulNBitsWMMA variant');
  ok(/OPS: matmul_nbits\n/.test(sb.calls.prompts['tuner:tuning']) && /ARCH: gfx1151/.test(sb.calls.prompts['tuner:tuning']),
    'the tuner refreshes only the ops that carry the run, for the device the run is on');
  ok(sb.calls.schemas['tuner:tuning'].required.includes('overlay') &&
    sb.calls.schemas['integrator:integrate'].required.includes('overlay'), 'an accepted candidate must name its overlay');
  ok(sb.calls.schemas['validator:validate'].properties.validation_status.enum.includes('validated'),
    "the validator's status is one of the values the return reads");
  const toolEnv = JSON.parse(/TOOL_ENV: (\{.*\})/.exec(sb.calls.prompts['tuner:tuning'])[1]);
  const pathsBlock = /^PATHS = \{([\s\S]*?)^\}/m.exec(fs.readFileSync(path.join(HERE, 'hipep_env.py'), 'utf8'))[1];
  const needed = [...pathsBlock.matchAll(/"(HIPEP_\w+)":/g)].map((m) => m[1]);
  ok(needed.length === 6 && needed.every((k) => toolEnv[k]), `TOOL_ENV sets every path the tools read (${needed.join(', ')})`);
  ok(toolEnv.HIPEP_PKG === '/pkg/rel', 'HIPEP_PKG is the release package');

  console.log('\n# the kernel lane');
  ok(sb.calls.workflows.length === 1, 'kernel_workflow is called once');
  const kw = sb.calls.workflows[0];
  ok(kw.spec.scriptPath === '/geak/kernel_workflow/kernel_lane.js',
    'the kernel lane worker is called directly (the dispatcher would nest a third level)');
  ok(kw.args.kernel_path === '/exp/e2e_q27b/lanes/1_kernel/matmul_nbits_u4_prefill_task' &&
    /TASK_DIR: \/exp\/e2e_q27b\/lanes\/1_kernel\/matmul_nbits_u4_prefill_task/.test(sb.calls.prompts['extractor:extract']),
  'the task dir is named after the op, which names the kernel in the experience store');
  ok(kw.args.target_language === 'hip' && kw.args.mode === 'optimize', 'the lane optimizes HIP');
  ok(kw.args.workload_spec_path === '/task/workload.json' && kw.args.task === 'prefill WMMA',
    'the kernel lane gets the workload and the steer');
  ok(kw.args.deadline_epoch === 1000 + 6 * 3600 && kw.args.no_stop_s > 6 * 3600,
    'the window is a cap: a deadline, and the TechLead may stop early');
  ok(kw.args.exp_root === '/exp/e2e_q27b/kernel' && !('gpu_ids' in kw.args), "the lane works under this run's eval dir");
  ok(/CUR_OVERLAY: \/o\/lut/.test(sb.calls.prompts['extractor:extract']), 'the kernel task starts from the accepted overlay');
  ok(/CUR_PKG: \/pkg\/lut1/.test(sb.calls.prompts['integrator:integrate']) &&
    /CUR_OVERLAY: \/o\/lut/.test(sb.calls.prompts['integrator:integrate']) &&
    /FINAL_PATCH: \/k\/final_patch\.diff/.test(sb.calls.prompts['integrator:integrate']),
  'the kernel candidate builds on the accepted tuning package and its overlay');
  ok(/RETUNE_OPS: matmul_nbits\n/.test(sb.calls.prompts['integrator:integrate']) &&
    /ARCH: gfx1151/.test(sb.calls.prompts['integrator:integrate']),
  "the integrator re-sweeps the tables of the kernel lane's op, for the device");
  ok(kw.args && ret.lanes[1].deadline_epoch === 1000 + 6 * 3600, 'the lane records its deadline');
  const integrator = fs.readFileSync(path.join(HERE, 'roles', 'integrator.md'), 'utf8');
  ok(/parity\.py --model MODEL_PATH --base BASE_PKG --cand LANE_DIR\/pkg \\\n\s+--out LANE_DIR\/parity --lens LENS/.test(integrator),
    'kernel candidates are checked for parity against the release, at the timed lengths');
  ok(ret.lanes.length === 2 && ret.lanes[1].result.reason === 'inside noise' && ret.final_label === 'tuning',
    'a rejected kernel candidate leaves the tuning package final');

  sb = sandbox({ ...ANSWERS, 'integrator:integrate': { ...ANSWERS['integrator:integrate'], accepted: true, reason: 'TTFT -3%' } });
  ret = await go(sb);
  ok(ret.final_pkg === '/pkg/k1' && ret.final_overlay === '/o/k1' && ret.final_label === 'kernel:matmul_nbits_u4_prefill',
    'an accepted kernel candidate becomes final');
  sb = sandbox({ ...ANSWERS, 'tuner:tuning': { ...ANSWERS['tuner:tuning'], accepted: false, reason: 'inside noise' } });
  ret = await go(sb);
  ok(/CUR_PKG: \/pkg\/rel/.test(sb.calls.prompts['integrator:integrate']),
    "a rejected tuning candidate leaves the release as the kernel lane's base");
  sb = sandbox({ ...ANSWERS, 'integrator:integrate': null });
  ret = await go(sb);
  ok(ret.lanes[1].result.reason === 'the integrator returned no result', 'a silent integrator is recorded with a reason');
  sb = sandbox({ ...ANSWERS, 'extractor:extract': { ...ANSWERS['extractor:extract'],
    status: 'fail: performance: M2048_K5120_N34816_gs32_noz failed' } });
  ret = await go(sb);
  ok(sb.calls.workflows.length === 0 && /^the task failed at baseline: fail: performance/.test(ret.lanes[1].result.reason),
    'a task that fails at baseline (any step) runs no kernel lane');
  ok(/LENS: 1024,2048/.test(sb.calls.prompts['extractor:extract']) &&
    /make_task\.py --op OP --arch ARCH --model MODEL_PATH --m LENS/.test(fs.readFileSync(path.join(HERE, 'roles', 'extractor.md'), 'utf8')),
  'the task times the lengths the A/B gate times');
  for (const status of ['flagged', 'unknown']) {
    sb = sandbox({ ...ANSWERS, kernel: { ...ANSWERS.kernel, validation_status: status } });
    ret = await go(sb);
    ok(!sb.calls.agents.includes('integrator:integrate') && new RegExp(`\\(1\\.03x, ${status}\\)`).test(ret.lanes[1].result.reason),
      `a ${status} lane result is not integrated`);
  }
  sb = sandbox({ ...ANSWERS, kernel: { final_speedup: 0, final_patch: '', eval_dir: '/k', validation_status: 'no_baseline',
    reason: 'no frozen baseline for matmul_nbits_u4_prefill' } });
  ret = await go(sb);
  ok(/no frozen baseline for matmul_nbits_u4_prefill/.test(ret.lanes[1].result.reason), "the lane's own reason is kept");
  sb = sandbox({ ...ANSWERS, kernel: { ...ANSWERS.kernel, final_patch: '' } });
  ret = await go(sb);
  ok(!sb.calls.agents.includes('integrator:integrate'), 'no patch, no integration');
  sb = sandbox({ ...ANSWERS, clock: null });
  ret = await go(sb);
  ok(!('deadline_epoch' in sb.calls.workflows[0].args) && ret.lanes[1].deadline_epoch === null,
    'an unreadable clock runs the lane uncapped, and says so');
  sb = sandbox({ ...ANSWERS, kernel: { final_speedup: 0.98, final_patch: '/k/p.diff', eval_dir: '/k',
    validation_status: 'accepted' } });
  ret = await go(sb);
  ok(!sb.calls.agents.includes('integrator:integrate') &&
    /^no verified isolated speedup \(0\.98x/.test(ret.lanes[1].result.reason), 'no integration without an isolated speedup');
  sb = sandbox(ANSWERS);
  sb.workflow = async () => { throw new Error('nesting depth exceeded'); };
  ret = await go(sb);
  ok(/^kernel lane did not run: Error: nesting depth exceeded/.test(ret.lanes[1].result.reason) &&
    !sb.calls.agents.includes('integrator:integrate'), 'a lane that throws is reported as not run');

  console.log('\n# tuning lane rejected, silent, or not reproduced');
  sb = sandbox({ ...ANSWERS, 'tuner:tuning': { ...ANSWERS['tuner:tuning'], accepted: false,
    reason: 'TTFT@2048 -0.6% inside noise 1.0%' } });
  ret = await go(sb);
  ok(ret.final_pkg === '/pkg/rel' && ret.final_label === 'release' && ret.candidate === null,
    'a rejected candidate leaves the release final');
  ok(ret.lanes[0].result.reason === 'TTFT@2048 -0.6% inside noise 1.0%', 'the gate reason is recorded');
  sb = sandbox({ ...ANSWERS, 'tuner:tuning': null });
  ret = await go(sb);
  ok(ret.final_pkg === '/pkg/rel' && ret.lanes[0].result.reason === 'the tuner returned no result',
    'a tuner with no result is recorded with a reason');
  sb = sandbox({ ...ANSWERS, 'validator:validate': { ...ANSWERS['validator:validate'], validation_status: 'not_reproduced' } });
  ret = await go(sb);
  ok(ret.final_pkg === '/pkg/rel' && ret.candidate && ret.candidate.pkg === '/pkg/lut1' &&
    ret.validation_status === 'not_reproduced', 'a candidate the validator does not reproduce is not returned as final');

  console.log('\n# the lane threshold');
  sb = sandbox({ ...ANSWERS, 'profiler:profile': { ...ANSWERS['profiler:profile'],
    prefill: [{ base: 'MatMulNBitsWMMA_NoZP', share: 0.12 }, { base: 'gqa_flash_prefill_v6x_kernel', share: 0.07 }],
    decode: [{ base: 'matmul_nbits_gemv_dp4a_kernel', share: 0.9 }, { base: 'gqa_flash_decode_kernel', share: 0.05 }] } });
  ret = await go(sb);
  ok(!sb.calls.phases.includes('Tuning') && !sb.calls.phases.includes('HeadKernel') && sb.calls.workflows.length === 0,
    'under 20% in every phase: no lane runs');
  ok(ret.lanes.length === 2 && /^not run: no tunable op holds 20\.0% of a phase/.test(ret.lanes[0].result.reason) &&
    /^not run: matmul_nbits_u4_prefill is 12\.0% of its phase, under 20\.0%/.test(ret.lanes[1].result.reason),
  'each lane carries its reason to the report');
  ok(sb.calls.agents.includes('validator:validate') && ret.final_pkg === '/pkg/rel', 'the run still validates and reports');
  sb = sandbox({ ...ANSWERS, 'profiler:profile': { ...ANSWERS['profiler:profile'],
    prefill: [{ base: 'MatMulNBitsWMMA_NoZP', share: 0.56 }, { base: 'gqa_flash_prefill_v6x_kernel', share: 0.07 }],
    decode: [{ base: 'gqa_flash_decode_kernel', share: 0.17 }, { base: 'gqa_flash_decode_reduce_kernel', share: 0.03 }] } });
  ret = await go(sb);
  ok(/OPS: matmul_nbits,gqa\n/.test(sb.calls.prompts['tuner:tuning']) && Math.abs(ret.lanes[0].shares.gqa - 0.2) < 1e-9,
    "an op counts its largest phase: gqa's 20% of a decode step tunes it, at exactly the threshold");
  ok(sb.calls.workflows.length === 1 && /ARCH: gfx1151/.test(sb.calls.prompts['extractor:extract']),
    'the kernel lane runs on its own op\'s share, for the device the run is on');
  sb = sandbox({ ...ANSWERS, 'profiler:profile': { ...ANSWERS['profiler:profile'],
    prefill: [{ base: 'MatMulNBitsWMMA_NoZP', share: 0.1 }], decode: [{ base: 'gqa_flash_decode_kernel', share: 0.3 }] } });
  ret = await go(sb);
  ok(/OPS: gqa\n/.test(sb.calls.prompts['tuner:tuning']) && sb.calls.workflows.length === 0 &&
    /^not run: matmul_nbits_u4_prefill is 10\.0%/.test(ret.lanes[1].result.reason),
  "each lane goes by its own ops: gqa's decode share tunes gqa, and no kernel lane runs on a 10% GEMM");
  sb = sandbox({ ...ANSWERS, 'profiler:profile': { ...ANSWERS['profiler:profile'],
    prefill: [{ base: 'MatMulNBitsFp16GEMM', share: 0.9 }] } });
  await go(sb);
  ok(!sb.calls.agents.includes('tuner:tuning') && sb.calls.workflows.length === 0,
    'a kernel no table and no task covers does not count');

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
  for (const k of ['tools_dir', 'kernel_workflow_dir', 'therock', 'hipep_src']) {
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
  const lines = prompt.calls.prompts['tuner:tuning'].split('\n').map((l) => l.trim());
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
