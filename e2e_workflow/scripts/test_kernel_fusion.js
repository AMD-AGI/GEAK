#!/usr/bin/env node
// KernelFusion in e2e_workflow.js (kernel_fusion=true; default OFF), on CPU: no GPU, no model, no LLM.
// kernel_fusion_mockrun.mjs runs the WHOLE workflow on GEAK's own runtime with scripted agents and stubbed
// kernel_lane.js runs, so these checks see what a real run would send and return:
//   - switched off (unset or "false"), no prompt, log, lane or return field carries any trace of it;
//   - switched on, the fusion profile reads Profile's own trace and leaves Profile as it is, Strategize routes
//     its regions as kernel candidates, and the Milestone loop builds, optimizes (kernel_lane.js) and
//     measures each one with GEAK's own extract, integrate A/B, Finalize-gate and Validate;
//   - a region on a head's own kernel is one more candidate in that head's HeadKernel slot instead, and goes
//     back to the Milestone loop when the head never reaches its A/B;
//   - every outcome (accepted, pending, overlay rejected, unknown region, failed or unconfigured profile,
//     fast/deep mode, a given topk, AgentX, a resumed run) ends cleanly and the run goes on to Validate.
//
// Run:  node e2e_workflow/scripts/test_kernel_fusion.js
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const SCRIPTS = __dirname;
const WF_DIR = path.resolve(SCRIPTS, '..');
const src = fs.readFileSync(path.join(WF_DIR, 'e2e_workflow.js'), 'utf8');
const EV = '/tmp/kf_mock/eval';

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };
const runs = {};
const run = (scenario, env = {}) => {
  if (runs[scenario]) return runs[scenario];
  const p = spawnSync(process.execPath, [path.join(SCRIPTS, 'kernel_fusion_mockrun.mjs'), scenario],
    { encoding: 'utf8', timeout: 180000, maxBuffer: 256 * 1024 * 1024, env: { ...process.env, ...env } });
  if (p.status !== 0) throw new Error(`mock run ${scenario} exited ${p.status}: ${p.stderr}`);
  const r = JSON.parse(p.stdout.trim().split('\n').pop());
  if (!r.ok) throw new Error(`mock run ${scenario} threw: ${r.error}`);
  runs[scenario] = r;
  return r;
};
const labels = (r) => r.calls.map((c) => c.label);
const kfOf = (r) => (r.result && r.result.kernel_fusion) || {};
const promptOf = (r, label) => (r.calls.find((c) => c.label === label) || {}).prompt || '';
const banked = (r) => ((r.result && r.result.accepted_kernels) || []);
const names = (r) => banked(r).map((k) => k.short_name);
const before = (r, a, b) => labels(r).indexOf(a) >= 0 && labels(r).indexOf(a) < labels(r).indexOf(b);
const laneOf = (r, short) => r.lanes.find((l) => String(l.kernel_path || '').endsWith(`/${short}`)) || {};
// What a run carries of KernelFusion, wherever it would show: none of these appears in a run switched off.
const KF_MARKS = /FUSION_STRATEGY|FUSION_CANDIDATES|FUSION_SCOPE|FUSION REGION|fusion_analyst|fusion_ut_contract|fused_kernel_task|fusion_overlay_check|"op_kind": ?"fusion"|\bkf_c\d|\[kf\]/;
// A failed fusion profile must leave the run as off_kernels': same calls (but the analyst's own, and the
// final persists, which carry the return), same logs (agent counters aside), same result (but the record).
const END = /^(persist-workflow-return|file_writer:persist_llm_stats)/;
const bareCalls = (r) => r.calls.filter((c) => !c.label.startsWith('fusion_analyst') && !END.test(c.label))
  .map((c) => `${c.label}\n${c.prompt}`);
const bareLogs = (r) => r.logs.filter((l) => !l.startsWith('[kf]') && !l.includes('fusion_analyst'))
  .map((l) => l.replace(/\d+ spawned/, 'N spawned').replace(/\d+ agent attempts/, 'N agent attempts'));
const bareResult = (r) => {
  const o = JSON.parse(JSON.stringify(r.result));
  delete o.kernel_fusion;
  if (o.state) delete o.state.kernel_fusion;
  return o;
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

console.log('## A. structure: no phase, hooks into the stages GEAK already has');
const meta = (src.match(/phases:\s*\[[\s\S]*?\n {2}\],/) || [''])[0];
ok(meta.length > 0 && !/KernelFusion|Fusion/.test(meta), 'meta.phases is GEAK\'s own: KernelFusion adds no phase');
ok(!/phase\('KernelFusion'\)|want\('fusion'\)/.test(src), 'no KernelFusion phase() or phase selector');
ok(/const KF_ON = String\(A\.kernel_fusion != null \? A\.kernel_fusion : 'false'\) === 'true';/.test(src),
  'the switch defaults to off and only the string "true" turns it on');
ok(/label: 'profiler:baseline'[^\n]*\n\s*log\(`Baseline profiled\.[^\n]*\n\s*if \(KF_ON\) await kfProfile\('Profile'\);/.test(src),
  'fixed load: the fusion profile follows profiler:baseline, in the Profile step');
ok(/profile = strategy;\s*if \(KF_ON\) \{\s*await kfProfile\('KernelExtract'\);\s*strategy = await kfPrepareRegions\(/.test(src),
  'AgentX: the fusion profile follows the baseline preparation, then the region UTs join the inventory');
ok((src.match(/\.\.\.\(KF_ON && KF_INPUTS\),/g) || []).length === 4,
  'its regions reach Strategize, both re-strategizes and plan_milestone (KF_INPUTS, {} until there are regions)');
ok(['...(KF_ON ? kfExpand(c) : c), idx: i',
  "(KF_ON && kfExtractIntro(c)) || 'Capture shapes + oracle; emit an immutable unittest task dir.'",
  '...(KF_ON && kfExtractInputs(c)),', "task: (KF_ON && kfLaneTask(c)) || 'Compare candidate backends '",
  '...(KF_ON && kfResultFields(c)) },', '...(KF_ON && kfAccuracyFor(c)),'].every((s) => src.split(s).length === 2),
  'the Milestone loop treats a region as a kernel candidate: one hook each in plan, extract, lane and integrate');
const count = (s) => src.split(s).length - 1;
ok(count('if (KF_ON) headCands.push(...(await kfHeadCandidates(h)));') === 1 &&
  count('...(KF_ON && kfHeadResult(cand)) },') === 1 && count('...(KF_ON && kfAccuracyFor(cand.kf)),') === 1 &&
  count('(KF_ON && kfHeadPct(cand)) || h.pct_gpu_time') === 4 && count('...(KF_ON && kfHeadSpec(cand)),') === 2,
  'HeadKernel takes a region as one more candidate: its A/B inputs, gate share, pending entry and correctives');
// Outside its own blocks every KernelFusion reference sits behind KF_ON, so a GEAK test that slices one
// function out of the workflow needs only KF_ON = false for it.
const cut = (s, from, to) => {
  const a = s.indexOf(from), b = s.indexOf(to, a);
  return a < 0 || b < 0 ? '' : s.slice(0, a) + s.slice(b);
};
const outside = [['// ---- KERNEL FUSION (opt-in switch', '// Kernel-layer warm start:'],
  ['// ---- KernelFusion (kernel_fusion=true)', "if (want('setup')) {"],
  ['// KernelFusion record (result["kernel_fusion"]', '// State to carry into the NEXT phase invocation']]
  .reduce((s, [from, to]) => s && cut(s, from, to), src);
const KF_NAME = /\bkf[A-Z]\w*|\bKF_(?!ON\b)[A-Z_]+/;
let guardIndent = null;   // inside an `if (KF_ON) {` block
const kfLines = [], unguarded = [];
for (const l of outside.split('\n')) {
  const indent = l.search(/\S/);
  if (guardIndent != null && l.trim() === '}' && indent === guardIndent) { guardIndent = null; continue; }
  if (guardIndent == null && /\bif \(KF_ON\) \{$/.test(l.trim())) { guardIndent = indent; continue; }
  if (!KF_NAME.test(l)) continue;
  kfLines.push(l);
  const g = l.search(/\bKF_ON\b/);
  if (guardIndent == null && !(g >= 0 && g < l.search(KF_NAME))) unguarded.push(l);
}
ok(outside.length > 0 && kfLines.length >= 15 && unguarded.length === 0,
  `outside its own blocks, all ${kfLines.length} lines naming KernelFusion read KF_ON first` +
  (unguarded.length ? `: ${unguarded.map((l) => l.trim()).join(' | ')}` : ''));
ok(/integ = await reviewSpeedup\(integ, withTuning, label, phaseName\);\s*if \(KF_ON\) integ = await kfOverlayGate\(integ, withTuning, phaseName\);/.test(src),
  'every integrate A/B (Milestone and Finalize-gate alike) runs the overlay check on an accepted region');
ok(/if \(KF_ON && k\.op_kind === 'fusion'\) kfBanked\(list\[list\.length - 1\]\);/.test(src) &&
  !/\bkind: 'fusion'/.test(src), 'a banked region stays kind "patch" (the KB replay reads it) with an op_kind tag');
ok(/fusion_analyst:analyze', schema: KF_OPPORTUNITY_SCHEMA \}, 2, kfTimeout\(3600000\)\)/.test(src) &&
  (src.match(/1, kfTimeout\(Infinity\)\)/g) || []).length === 2,
  'what runs ahead of the search is bounded: the analysis 2 x <= 1 h, each region UT call one attempt');
const block = src.slice(src.indexOf('const KF_SKILL_DIR ='), src.indexOf("if (want('setup')) {"));
ok(block.length > 10000 && !/Date\.now|require\(|process\./.test(block),
  'the KernelFusion helpers use no construct the workflow sandbox forbids');
ok(!fs.existsSync(path.join(WF_DIR, 'kernel_fusion')) && !/kf_workflow\.js/.test(src),
  'no standalone KernelFusion workflow or module dir');
const role = fs.readFileSync(path.join(WF_DIR, 'roles', 'fusion_analyst.md'), 'utf8');
ok(role.includes('## PHASE=analyze') && (role.match(/## PHASE=/g) || []).length === 1,
  'roles/fusion_analyst.md: one phase, the fusion profile (the reviews moved to the architect)');
const strat = fs.readFileSync(path.join(WF_DIR, 'roles', '_fragments', 'fusion_strategy.md'), 'utf8');
ok(/Feasibility Review/.test(strat) && /never `head_candidates`/.test(strat) && /exclusive_with/.test(strat) &&
  /route at most one of them at a time/.test(strat), 'roles/_fragments/fusion_strategy.md: the routing rules');
ok(/stays in `head_candidates` exactly as it would without the\s+region/.test(strat) && /`<head short_name>:kf_<id>`/.test(strat),
  'roles/_fragments/fusion_strategy.md: a head op keeps its head route, and HISTORY names a region tried in its slot');
const arch = fs.readFileSync(path.join(WF_DIR, 'roles', 'system_architect.md'), 'utf8');
ok(/1e\. \*\*Fusion regions \(only if `FUSION_STRATEGY` is in your inputs; otherwise skip this step entirely/.test(arch),
  'the architect routes fusion regions only when FUSION_STRATEGY is in its inputs');

console.log('\n## B. switched off: no trace of it anywhere');
for (const s of ['off', 'off_false', 'off_kernels', 'off_agentx', 'off_head', 'off_agentx_head']) {
  const r = run(s);
  const hits = [...r.calls.map((c) => c.prompt), ...r.logs, ...r.lanes.map((l) => JSON.stringify(l))]
    .filter((t) => KF_MARKS.test(t));
  ok(hits.length === 0, `${s}: no prompt, log or lane mentions KernelFusion`);
  ok(!('kernel_fusion' in r.result) && !('kernel_fusion' in (r.result.state || {})), `${s}: no kernel_fusion return or state`);
}
ok(same(run('off').calls, run('off_false').calls) && same(run('off').logs, run('off_false').logs) &&
  same(run('off').result, run('off_false').result), 'kernel_fusion "false" is exactly unset');
ok(labels(run('off_kernels')).includes('finish-integrate mk_gemm'), 'off_kernels: GEAK\'s own Finalize-gate replay ran');
ok(labels(run('off_head')).includes('integrate h_gemm:aiter') && labels(run('off_agentx_head')).includes('integrate h_gemm:aiter'),
  'off_head, off_agentx_head: GEAK\'s own head track ran');

console.log('\n## C. switched on: profile, route, optimize, measure');
const on = run('on');
const offk = run('off_kernels');
ok(before(on, 'profiler:baseline', 'fusion_analyst:analyze') && before(on, 'fusion_analyst:analyze', 'architect:strategize'),
  'profiler:baseline -> fusion_analyst:analyze -> architect:strategize');
ok(promptOf(on, 'profiler:baseline') === promptOf(offk, 'profiler:baseline'), 'the Profile round is the one GEAK runs anyway');
const an = promptOf(on, 'fusion_analyst:analyze');
ok(an.includes(`${EV}/profile/round_0/profile`) && an.includes(`${EV}/profile/round_0/profile_topN.json`) &&
  an.includes(`${EV}/kernel_fusion/analysis`), 'the analysis reads Profile\'s own trace and Top-N, and writes under EVAL_DIR');
const st = promptOf(on, 'architect:strategize');
ok(['FUSION_STRATEGY', 'fusion_strategy.md', 'FUSION_CANDIDATES', 'FUSION_RECOMMENDED', 'FUSION_TOPK_JSON',
  'IMPLEMENTABLE_LEVELS', 'UNSUPPORTED_PATTERNS'].every((k) => st.includes(k)), 'Strategize gets the fusion profile');
ok(!st.includes('fused_design') && !st.includes('one pass: normalize'), 'the architect routes on the slim region facts');
const ex = promptOf(on, 'extract kf_c2');
ok(/FUSION REGION/.test(ex) && ex.includes('fusion_ut_contract.md') && ex.includes('pkg.norm:fused_norm_quant') &&
  ex.includes('quant_kernel'), 'extract kf_c2: the region UT under the fusion UT contract, with the region\'s facts');
ok(promptOf(on, 'extract mk_gemm') === promptOf(offk, 'extract mk_gemm'), 'a single kernel beside it is extracted as before');
const lane = laneOf(on, 'kf_c2');
ok(/kernel_workflow\/kernel_lane\.js$/.test(lane.script) && /^FUSION REGION c2 \(L1\): write ONE fused triton kernel/.test(lane.task) &&
  lane.task.includes('fused_kernel_task.md') && lane.task.includes('one pass: normalize'),
  'the fused kernel is optimized by GEAK\'s kernel_lane.js with the region task');
ok(/^Compare candidate backends/.test(laneOf(on, 'mk_gemm').task), 'the single kernel\'s lane task is GEAK\'s own');
const ik = promptOf(on, 'integrate kf_c2');
ok(/"op_kind": ?"fusion"/.test(ik) && /"fusion_id": ?"c2"/.test(ik) && ik.includes('live_call_seam') &&
  /ACCURACY_GATE/.test(ik), 'integrate kf_c2: GEAK\'s A/B with the region in KERNEL_RESULT and the accuracy gate');
ok(before(on, 'integrate kf_c2', 'overlay-check kf_c2') && before(on, 'overlay-check kf_c2', 'integrate mk_gemm') &&
  promptOf(on, 'overlay-check kf_c2').includes(`--overlay "${EV}/overlay/cand_kf_c2"`),
  'the accepted overlay is checked on disk before anything else runs on it');
const kc2 = banked(on).find((k) => k.short_name === 'kf_c2') || {};
ok(kc2.kind === 'patch' && kc2.op_kind === 'fusion' && kc2.fusion_id === 'c2' && names(on).includes('mk_gemm'),
  'banked: kf_c2 (kind patch, op_kind fusion) and mk_gemm');
ok(/ACCURACY_GATE/.test(promptOf(on, 'integrate mk_gemm')) && /ACCURACY_GATE/.test(promptOf(on, 'director:validate')) &&
  !/ACCURACY_GATE/.test(promptOf(offk, 'integrate mk_gemm')) && !/ACCURACY_GATE/.test(promptOf(offk, 'director:validate')),
  'once a fused kernel is in, every later A/B and Validate hold the run to task accuracy');
ok(promptOf(on, 'architect:plan m2').includes('FUSION_STRATEGY'), 'later milestone plans see the fusion profile too');
const k = kfOf(on);
ok(k.status === 'accepted' && same(k.routed, ['kf_c2']) && k.accepted[0].fusion_id === 'c2' && k.soft_parity === true &&
  same(k.outcomes.map((o) => [o.short_name, o.verdict]), [['kf_c2', 'confirmed']]) &&
  k.trace_dir === `${EV}/profile/round_0/profile` && on.result.state.kernel_fusion.status === 'accepted',
  'the record: profile, routed regions, accepts, verdicts (history.ledger); carried in state');
ok(labels(on).includes('director:validate'), 'the run goes on to Validate');

console.log('\n## D. every outcome ends cleanly');
const pend = run('on_pending');
const fin = promptOf(pend, 'finish-integrate kf_c2');
ok(/"op_kind": ?"fusion"/.test(fin) && /ACCURACY_GATE/.test(fin) && before(pend, 'finish-integrate kf_c2', 'overlay-check kf_c2'),
  'pending: the Finalize-gate replay keeps the region and the accuracy gate, then checks the overlay');
ok(names(pend).includes('kf_c2') && !/ACCURACY_GATE/.test(promptOf(pend, 'integrate mk_gemm')) &&
  /ACCURACY_GATE/.test(promptOf(pend, 'director:validate')),
  'pending: banked at the Finalize-gate; the gate applies from then on, not before');
const bad = run('on_bad_overlay');
ok(!names(bad).includes('kf_c2') && names(bad).includes('mk_gemm') && kfOf(bad).soft_parity === false &&
  !/ACCURACY_GATE/.test(promptOf(bad, 'integrate mk_gemm')) && !/ACCURACY_GATE/.test(promptOf(bad, 'director:validate')),
  'overlay rejected: the region is not banked and the rest of the run is not held to task accuracy');
ok(kfOf(bad).outcomes[0].verdict === 'dead_end' && /^overlay: accepted overlay failed validation/.test(kfOf(bad).outcomes[0].lesson) &&
  !labels(bad).some((l) => /kernel_surgeon|corrective/.test(l)), 'overlay rejected: a final verdict, no corrective re-author');
const unk = run('on_unknown');
ok(!/FUSION REGION/.test(promptOf(unk, 'extract k_x')) && /^Compare candidate backends/.test(laneOf(unk, 'k_x').task) &&
  !/"op_kind": ?"fusion"/.test(promptOf(unk, 'integrate k_x')) && same(kfOf(unk).routed, []),
  'a fusion_id the profile does not have is an ordinary kernel candidate');
const err = run('on_error');
ok(kfOf(err).status === 'error' && /returned nothing/.test(kfOf(err).reason) &&
  labels(err).filter((l) => l === 'fusion_analyst:analyze').length === 2, 'failed profile: two attempts, recorded');
ok(same(bareCalls(err), bareCalls(offk)) && same(bareLogs(err), bareLogs(offk)) && same(bareResult(err), bareResult(offk)),
  'failed profile: the rest of the run is off_kernels\' run, call for call');
const cfg = run('on_config_error');
ok(kfOf(cfg).status === 'error' && /^config: kf_trace_dir does not exist/.test(kfOf(cfg).reason) &&
  !labels(cfg).includes('fusion_analyst:analyze') && same(bareCalls(cfg), bareCalls(offk)) && same(bareResult(cfg), bareResult(offk)),
  'unusable knob: no fusion profile, and otherwise off_kernels\' run');
for (const s of ['on_fast', 'on_deep']) {
  const r = run(s);
  ok(kfOf(r).status === 'skipped' && /runs no Milestone loop/.test(kfOf(r).reason) &&
    !labels(r).includes('fusion_analyst:analyze') && labels(r).includes('director:validate'),
  `${s}: skipped (no Milestone loop to optimize a region in)`);
}
const tk = run('on_topk');
ok(!labels(tk).includes('fusion_analyst:analyze') && promptOf(tk, 'architect:strategize').includes('FUSION_CANDIDATES') &&
  promptOf(tk, 'extract kf_c2').includes('pkg.norm:fused_norm_quant') && kfOf(tk).topk_json === '/data/topk.json' &&
  names(tk).includes('kf_c2'), 'a given topk is reused, not re-analyzed (a bare-string seam is read as the seam)');

console.log('\n## E. AgentX: the region UTs join the prepared inventory');
const ax = run('on_agentx');
ok(before(ax, 'kernel_extractor:prepare baseline', 'fusion_analyst:analyze') &&
  before(ax, 'fusion_analyst:analyze', 'kernel_extractor:prepare kf_c1') &&
  promptOf(ax, 'fusion_analyst:analyze').includes(`${EV}/kernel_extraction/round_baseline`),
  'the analysis reads the baseline preparation\'s captures (KX_EVIDENCE_DIR)');
const pk = promptOf(ax, 'kernel_extractor:prepare kf_c2');
ok(/FUSION_SCOPE: \[\{"task":"kf_c2","fusion_level":"L1"/.test(pk) && pk.includes('fusion_ut_contract.md') &&
  pk.includes('"device_kernels":["rms_norm_kernel","quant_kernel"]'),
  'each recommended region gets one kernel-extract invocation scoped to it');
ok(same(kfOf(ax).unprepared, [{ id: 'c3', reason: 'no seam.target_callable' }]) &&
  !labels(ax).includes('kernel_extractor:prepare kf_c3'), 'a region without a seam gets no UT, and says why');
ok(!labels(ax).some((l) => /^extract /.test(l)) &&
  same(ax.lanes.map((l) => l.kernel_path.split('/').pop()), ['kf_c2', 'k_norm', 'kf_c1']) &&
  /^FUSION REGION c1/.test(laneOf(ax, 'kf_c1').task) && !/FUSION REGION/.test(laneOf(ax, 'k_norm').task),
  'the prepared UTs are reused, in GPU-time order with the single kernel');
ok(same(names(ax), ['kf_c2', 'k_norm', 'kf_c1']) && labels(ax).includes('overlay-check kf_c1') &&
  !labels(ax).includes('overlay-check k_norm') && kfOf(ax).accepted.length === 2, 'banked all three; only regions get the overlay check');

console.log('\n## F. phase by phase: a resumed run keeps the fusion profile');
const setup = run('on_setup');
ok(kfOf(setup).status === 'profiled' && labels(setup).filter((l) => !END.test(l)).slice(-1)[0] === 'architect:strategize' &&
  setup.result.state.kernel_fusion.candidates.length === 4, 'setup: profiled and carried in state');
const stateFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'kf_state_')), 'state.json');
fs.writeFileSync(stateFile, JSON.stringify(setup.result.state));
const res = run('on_resume', { KF_MOCK_STATE: stateFile });
ok(!labels(res).includes('fusion_analyst:analyze') && /FUSION REGION/.test(promptOf(res, 'extract kf_c2')) &&
  promptOf(res, 'architect:plan m2').includes('FUSION_STRATEGY') && names(res).includes('kf_c2') &&
  kfOf(res).status === 'accepted', 'resume: the carried regions are routed, optimized and banked without a new profile');

console.log('\n## G. HeadKernel: a region on a head\'s own kernel is one more candidate in that head\'s slot');
const krIn = (r, label) => {
  const p = promptOf(r, label), i = p.indexOf('- KERNEL_RESULT: ');
  return i < 0 ? {} : JSON.parse(p.slice(i + 17, p.indexOf('\n', i)));
};
const heads = (r) => (r.result && r.result.accepted_heads) || [];
const once = (r, label) => labels(r).filter((l) => l === label).length === 1;
const hd = run('on_head');
ok(before(hd, 'bakeoff h_gemm', 'extract kf_c4') && before(hd, 'extract kf_c4', 'integrate h_gemm:kf_c4') &&
  once(hd, 'extract kf_c4') && !labels(hd).includes('integrate kf_c4') &&
  /FUSION REGION/.test(promptOf(hd, 'extract kf_c4')) && promptOf(hd, 'extract kf_c4').includes('pkg.mlp:MLP.forward'),
  'on_head: the region UT is built in the head\'s slot after its bake-off, and the Milestone loop never gets the region');
ok(/^FUSION REGION c4 \(L2\)/.test(laneOf(hd, 'kf_c4').task) && /^Author\+optimize a triton/.test(laneOf(hd, 'h_gemm').task),
  'on_head: one fused kernel lane beside the head\'s own author lane');
const fk = krIn(hd, 'integrate h_gemm:kf_c4'), ak = krIn(hd, 'integrate h_gemm:aiter');
ok(fk.short_name === 'h_gemm' && fk.task_dir === `${EV}/tasks/kf_c4` && fk.op_kind === 'fusion' && fk.fusion_id === 'c4' &&
  fk.winner_kind === 'patch' && fk.pct_gpu_time === 22 && fk.live_call_seam === 'pkg.mlp:MLP.forward' &&
  /ACCURACY_GATE/.test(promptOf(hd, 'integrate h_gemm:kf_c4')),
  'on_head: the fused candidate\'s A/B carries the region\'s UT, seam, GPU-time share and accuracy gate');
ok(ak.task_dir === `${EV}/tasks/h_gemm` && ak.op_kind === 'gemm' && ak.pct_gpu_time === 20 && !('fusion_id' in ak) &&
  /REUSE_REF: true/.test(promptOf(hd, 'integrate h_gemm:aiter')) && !/ACCURACY_GATE/.test(promptOf(hd, 'integrate h_gemm:aiter')),
  'on_head: the head\'s own candidates are A/B\'d as before, against the reference leg the fused one measured');
ok(before(hd, 'integrate h_gemm:kf_c4', 'overlay-check h_gemm') && before(hd, 'overlay-check h_gemm', 'integrate h_gemm:aiter'),
  'on_head: the fused candidate\'s accepted overlay is checked on disk before the next candidate runs');
const hk = heads(hd)[0] || {};
ok(heads(hd).length === 1 && hk.backend === 'kf_c4' && hk.kind === 'patch' && hk.op_kind === 'fusion' && hk.fusion_id === 'c4' &&
  same(kfOf(hd).routed, ['kf_c4', 'kf_c2']) && kfOf(hd).accepted.some((a) => a.short_name === 'h_gemm' && a.fusion_id === 'c4') &&
  kfOf(hd).outcomes.some((o) => o.short_name === 'h_gemm:kf_c4' && o.verdict === 'candidate_passed'),
  'on_head: the best measured candidate, the fused one, is banked as the head\'s win; the record says where it ran');
const lose = run('on_head_loses');
const lk = heads(lose)[0] || {};
ok(lk.backend === 'aiter' && lk.op_kind === 'gemm' && !('fusion_id' in lk) && !kfOf(lose).accepted.some((a) => a.fusion_id === 'c4') &&
  kfOf(lose).outcomes.some((o) => o.short_name === 'h_gemm:kf_c4' && o.verdict === 'candidate_passed') &&
  once(lose, 'extract kf_c4') && !labels(lose).includes('integrate kf_c4'),
  'on_head_loses: a fused candidate measured below another is not banked, and the region is not tried again');
const unrun = run('on_head_unrun');
ok(!labels(unrun).includes('bakeoff h_gemm') && labels(unrun).includes('integrate kf_c4') &&
  /^FUSION REGION c4/.test(laneOf(unrun, 'kf_c4').task) && (banked(unrun).find((x) => x.short_name === 'kf_c4') || {}).op_kind === 'fusion' &&
  kfOf(unrun).outcomes.some((o) => o.short_name === 'kf_c4' && o.verdict === 'confirmed'),
  'on_head_unrun: a head that never reaches its A/B leaves its region to the Milestone loop');
const hp = run('on_head_pending');
const pr = krIn(hp, 'finish-integrate h_gemm');
ok(pr.op_kind === 'fusion' && pr.task_dir === `${EV}/tasks/kf_c4` && /ACCURACY_GATE/.test(promptOf(hp, 'finish-integrate h_gemm')) &&
  before(hp, 'finish-integrate h_gemm', 'overlay-check h_gemm') && (heads(hp)[0] || {}).fusion_id === 'c4',
  'on_head_pending: the Finalize-gate replays the fused A/B as it was, checks the overlay, and banks it');
const fx = run('on_head_fix');
const fixLane = fx.lanes.find((l) => /^CORRECTIVE FIX/.test(String(l.task))) || {};
const fr = krIn(fx, 'integrate h_gemm corrective');
ok(String(fixLane.kernel_path).endsWith('/_exp/kf_c4') && (fixLane.op_spec || {}).op_kind !== 'gemm' &&
  fr.op_kind === 'fusion' && fr.task_dir === `${EV}/tasks/kf_c4` && fr.pct_gpu_time === 22 &&
  /ACCURACY_GATE/.test(promptOf(fx, 'integrate h_gemm corrective')) && before(fx, 'integrate h_gemm corrective', 'overlay-check h_gemm') &&
  (heads(fx)[0] || {}).fusion_id === 'c4' && (heads(fx)[0] || {}).corrective === true,
  'on_head_fix: a fixable live failure of the fused kernel gets the head\'s corrective re-author, on the fused kernel');
const axh = run('on_agentx_head');
ok(!labels(axh).some((l) => /^extract/.test(l)) && labels(axh).includes('kernel_extractor:prepare kf_c4') &&
  axh.lanes.filter((l) => l.kernel_path.endsWith('/kf_c4')).length === 1 && before(axh, 'integrate h_gemm:kf_c4', 'integrate kf_c2') &&
  !labels(axh).includes('integrate kf_c4') && (heads(axh)[0] || {}).fusion_id === 'c4',
  'AgentX: the head takes the region\'s prepared UT, so nothing is extracted, and the Milestone loop skips the region');

console.log(failures === 0 ? '\nPASS: KernelFusion runs inside GEAK\'s own stages; off, it leaves no trace.'
  : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
