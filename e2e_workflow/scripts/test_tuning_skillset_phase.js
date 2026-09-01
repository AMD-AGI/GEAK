#!/usr/bin/env node
// Regression guard for the tuning skillset, now folded INTO the head-kernel track (no GPU, no model,
// no agent needed).
//
// It used to be a standalone phase between ConfigSweep and HeadKernel. It is not one any more: tuning a
// GEMM in a phase of its own, and then optimizing that same GEMM in HeadKernel an hour later, meant two
// agents searching the same op with no shared oracle and no shared ledger. The skillset is now read PER
// HEAD OP by the two agents that already own that op — `quick_tune` (the cheap rung) and
// `op_benchmarker` (Tier A/B) — and the per-op outcomes are folded back into one `tuning` object by
// finalizeHeadTuning() so every downstream consumer is unchanged.
//
// Six invariants, none obvious from reading one file:
//
//   A. WHOLE — the skillset is vendored as one intact tree with its own entry points, and its method is
//      NOT copied/scattered into GEAK's own knowledge/ files. It is validated standalone, so GEAK must
//      run the copy that was validated; a scattered paraphrase silently voids that validation.
//   B. IN THE HEAD TRACK — no TuningSkillset phase, no tuning_specialist role; the skillset reaches
//      exactly the two head-op agents; `quick_tune` runs BEFORE every bake-off site; and
//      finalizeHeadTuning() runs at the end of the head track, before the post-head re-profile.
//   C. ADDITIVE OFF — `tuning_skillset:"false"` injects nothing anywhere, so the run is byte-identical
//      to a build without the feature.
//   D. THIN ADAPTER — roles/quick_tune.md routes into the skillset instead of paraphrasing its method.
//   E. REACHES PRODUCTION — the win travels through the EXISTING deliverable handles (final_patch.diff,
//      final_launch.sh) rather than dying in the eval dir. GEAK's overlay is a PYTHONPATH mechanism for
//      code, but a tuned artifact is usually data plus a cache invalidation, so without this the run
//      reports a gain that the shipped bundle does not reproduce. result.json gains its block additively.
//   F. HONEST ATTRIBUTION — the fold takes no server A/B of its own, so `tuning_delta_pct`,
//      `tuning_speedup` and `share_of_total_gain_pct` are NULL (= not measured), never 0. The evidence
//      that replaces them is per-op `ops_tuned[].isolated_speedup`, measured on each op's oracle.
//
// Run:  node e2e_workflow/scripts/test_tuning_skillset_phase.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');            // .../GEAK
const FILE = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');
const ROLE = path.join(ROOT, 'e2e_workflow', 'roles', 'quick_tune.md');
const OLD_ROLE = path.join(ROOT, 'e2e_workflow', 'roles', 'tuning_specialist.md');
// Under expert_skills/, so tuning skills share the hierarchy, index and maintenance model of every
// other expert skill — while the tree itself stays vendored and pinned as one unit.
const SKILLSET_REL = 'perf_knowledge/expert_skills/tuning';
const SKILLSET = path.join(ROOT, SKILLSET_REL);

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

console.log(`\n# ${path.relative(ROOT, FILE)}`);
const src = fs.readFileSync(FILE, 'utf8');

// ---------------------------------------------------------------------------
// A. The skillset is vendored WHOLE, with its own entry points intact.
// ---------------------------------------------------------------------------
console.log('\n## A. vendored whole');
ok(fs.existsSync(SKILLSET), `${SKILLSET_REL}/ is vendored into the repo`);
for (const rel of ['README.md', 'tuning-core/SKILL.md', 'validate/claims.py', 'tuning-kb/README.md']) {
  ok(fs.existsSync(path.join(SKILLSET, rel)), `vendored tree keeps its own entry point: ${rel}`);
}
// What git COMMITS must equal what the manifest HASHES, both ways. --verify cannot see this: it compares
// the manifest against the working tree, so it is blind to a file that exists on the syncer's disk but
// never made it into the commit (`*.log` swallowed six tuning-kb evidence logs; `build/`, `lib/`, `dist/`
// and friends would each swallow a subdirectory), and equally blind to committed cache junk, since it
// applies the same EXCLUDE_DIRS when scanning. Both directions are checked against the git index here.
{
  const { execFileSync } = require('child_process');
  const git = (args, input) => {
    try {
      return execFileSync('git', ['-C', ROOT, ...args],
        { input, encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'] });
    } catch (e) { return (e && e.stdout) || ''; }   // check-ignore exits 1 when nothing matches
  };
  const manifestPath = path.join(ROOT, 'e2e_workflow', 'knowledge', 'tuning_skillset.manifest.sha256');
  const inManifest = fs.readFileSync(manifestPath, 'utf8').split('\n')
    .filter((l) => l && !l.startsWith('#'))
    .map((l) => SKILLSET_REL + '/' + l.split('  ')[1]);

  // check-ignore takes pathnames, not globs, so feed it the manifest itself. --no-index is what makes
  // this assertion mean anything: by default check-ignore consults the index and never calls a TRACKED
  // file ignored, so once the tree is committed the check would pass no matter what the rules say.
  const ignored = git(['check-ignore', '--no-index', '--stdin'], inManifest.join('\n'))
    .split('\n').filter(Boolean);
  ok(ignored.length === 0,
    `no manifest-tracked file is git-ignored (checked ${inManifest.length})` +
    (ignored.length ? ` -- e.g. ${ignored.slice(0, 3).join(', ')}` : ''));

  // A repo with no git index (tarball export) cannot answer this half; skip rather than fail.
  const indexed = git(['ls-files', '--', SKILLSET_REL]).split('\n').filter(Boolean);
  if (indexed.length) {
    const manifestSet = new Set(inManifest);
    const indexedSet = new Set(indexed);
    const unhashed = indexed.filter((p) => !manifestSet.has(p));
    const uncommitted = inManifest.filter((p) => !indexedSet.has(p));
    ok(unhashed.length === 0,
      `git tracks nothing the manifest does not hash (${indexed.length} tracked)` +
      (unhashed.length ? ` -- e.g. ${unhashed.slice(0, 3).join(', ')}` : ''));
    ok(uncommitted.length === 0,
      `every manifest entry is in the git index` +
      (uncommitted.length ? ` -- e.g. ${uncommitted.slice(0, 3).join(', ')}` : ''));
  }
}
// Every skill directory keeps its SKILL.md — the unit of invocation stays whole.
const skillDirs = fs.readdirSync(SKILLSET, { withFileTypes: true })
  .filter((d) => d.isDirectory() && d.name.startsWith('tuning-') && d.name !== 'tuning-kb')
  .map((d) => d.name);
ok(skillDirs.length >= 8, `vendored tree keeps its per-backend skills (${skillDirs.length} found)`);
for (const d of skillDirs) {
  ok(fs.existsSync(path.join(SKILLSET, d, 'SKILL.md')), `${d}/SKILL.md present (independently invocable)`);
}
// NOT scattered: no SKILL.md from the skillset was copied into GEAK's own knowledge tree.
const knowledgeDir = path.join(ROOT, 'e2e_workflow', 'knowledge');
const walk = (dir, out = []) => {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out); else out.push(p);
  }
  return out;
};
const scattered = walk(knowledgeDir).filter((p) => /tuning-(core|aiter|triton|ck|hip|hipblaslt|flydsl|in-vllm|in-sglang)/.test(path.basename(p)));
ok(scattered.length === 0,
  `skillset method is NOT copied into e2e_workflow/knowledge/ (found ${scattered.length} stray copies) ` +
  `-> the standalone validation still describes what GEAK runs`);
// The integrity manifest is the enforcement point for "whole and unmodified".
ok(fs.existsSync(path.join(knowledgeDir, 'tuning_skillset.manifest.sha256')), 'vendored tree is hash-pinned by a manifest');
ok(fs.existsSync(path.join(ROOT, 'e2e_workflow', 'scripts', 'tuning_skillset_sync.py')), 'a verify/re-sync tool ships with it');

// ---------------------------------------------------------------------------
// B. No standalone phase — the skillset is read per head op, and the outcomes are folded back.
// ---------------------------------------------------------------------------
console.log('\n## B. folded into the head-kernel track');
const phaseList = src.match(/phases:\s*\[[\s\S]*?\n  \],/);
ok(!!phaseList, 'meta.phases list found');
if (phaseList) {
  ok(!/TuningSkillset/.test(phaseList[0]), "meta.phases no longer declares a 'TuningSkillset' phase");
  ok(/HeadKernel[\s\S]*?quick-tune[\s\S]*?tuning skillset/.test(phaseList[0]),
    'HeadKernel\'s detail names the ladder that replaced it (quick-tune -> bake-off w/ the skillset)');
}
const at = (re) => src.search(re);
const iHead = at(/if \(want\('head'\)/);
ok(iHead > 0, "the HeadKernel block exists");
ok(!/phase\('TuningSkillset'\)/.test(src), 'no block announces itself as a TuningSkillset phase');
ok(!/roleAgent\('tuning_specialist'/.test(src), 'the tuning_specialist role is not dispatched anywhere');
ok(!fs.existsSync(OLD_ROLE), 'roles/tuning_specialist.md is gone (its work moved into the head track)');

// The skillset must reach the head-op agents and NOBODY else. Spraying it across roles is the scattering
// failure mode this design has always guarded; the fold narrows the allowed set from one role to two,
// it does not open it. Walk every agent call site — roleAgent() for op_benchmarker, and the BARE agent
// prompt quick_tune uses — and check who receives the skillset inputs.
const consumers = new Set();
for (let i = src.indexOf("roleAgent('"); i !== -1; i = src.indexOf("roleAgent('", i + 1)) {
  const end = src.indexOf('{ phase:', i);
  const call = src.slice(i, end === -1 ? i + 4000 : end);
  const role = (call.match(/roleAgent\('([a-z_]+)'/) || [])[1] || '?';
  if (/headTuningInputs\(\)|TUNING_SKILLSET_DIR/.test(call)) consumers.add(role);
}
ok(consumers.size === 1 && consumers.has('op_benchmarker'),
  `among roleAgent() call sites the skillset reaches exactly op_benchmarker (got: ` +
  `${[...consumers].join(', ') || 'none'})`);
// quick_tune is the other consumer, and it is deliberately NOT a roleAgent: it never launches a server,
// so roleAgent's serving-config preamble is pure cost to the cheapest rung on the ladder.
const qtFn = src.slice(src.indexOf('async function quickTune('), src.indexOf('const quickTuneInputs'));
ok(qtFn.length > 200, 'quickTune() is defined');
ok(/roles\/quick_tune\.md/.test(qtFn) && !/roleAgent\(/.test(qtFn),
  'quick_tune runs as a BARE agent pointed at roles/quick_tune.md (no serving-config preamble)');
ok(/\.\.\.headTuningInputs\(\)/.test(qtFn), 'quick_tune receives the skillset inputs');
ok(/schema: QUICKTUNE_SCHEMA/.test(qtFn), 'its return is schema-forced');
ok(/QUICK_TUNE_KINDS/.test(qtFn) && /return null/.test(qtFn),
  'it is scoped to GEMM/attention kinds and returns null (ladder unchanged) for anything else');

// Ordering, per bake-off site: the cheap rung runs BEFORE the expensive one, and hands it what it
// learned. All three op_benchmarker bake-off sites must be covered — a site that skips quick_tune
// re-derives the obvious at full price.
const qtCalls = (src.match(/const qt = await quickTune\(/g) || []).length;
const qtSpreads = (src.match(/\.\.\.quickTuneInputs\(qt\)/g) || []).length;
const bakeoffs = (src.match(/roleAgent\('op_benchmarker', 'bakeoff'/g) || []).length;
ok(bakeoffs === 3, `three op_benchmarker bake-off sites (found ${bakeoffs})`);
ok(qtCalls === bakeoffs, `every bake-off site is preceded by a quickTune() call (${qtCalls}/${bakeoffs})`);
ok(qtSpreads === bakeoffs, `every bake-off site is handed QUICK_TUNE (${qtSpreads}/${bakeoffs})`);
for (const m of src.matchAll(/roleAgent\('op_benchmarker', 'bakeoff'/g)) {
  const before = src.slice(Math.max(0, m.index - 4000), m.index);
  ok(before.lastIndexOf('const qt = await quickTune(') > -1,
    'the quickTune() call sits above its bake-off, not after it');
}

// The fold. Per-op outcomes are aggregated into the single `tuning` object every downstream consumer
// already reads, and the accepted config is carried forward — the same two jobs the old phase did.
ok(/function finalizeHeadTuning\(\)/.test(src), 'finalizeHeadTuning() aggregates the per-op outcomes');
const iFinalize = at(/\n  finalizeHeadTuning\(\);/);
const iEndHead = src.indexOf('} // end serial head track');
const iPostProfile = src.indexOf("label: 'profiler:post-head'");
ok(iEndHead > 0 && iFinalize > iEndHead, 'it runs at the END of the serial head track');
ok(iPostProfile > 0 && iFinalize < iPostProfile,
  'it runs BEFORE the post-head re-profile (a banked tuning changes which kernels are hot, so ' +
  'profiling the untuned stack would hand Milestone a stale ranking)');
ok(/curEnv = \(curEnv \? curEnv \+ ' ' : ''\) \+ applyEnv/.test(src),
  'an accepted deploy folds its required env into the carried config (downstream phases inherit it)');
ok(/if \(tuning\.apply_overlay\) curOverlay = tuning\.apply_overlay;/.test(src),
  '...and its overlay (the code half is not dropped at the fold)');
// The three bars, enforced by the orchestrator rather than trusted from the agent. `engagement_verified`
// is the skillset's own central claim; the other two are what make the number mean anything.
ok(/t\.gate === 'accepted' && t\.engagement_verified === true &&\s*\n?\s*Number\(t\.isolated_speedup\) > 1\.0/.test(src),
  'only ops that ACCEPTED, PROVED engagement, and beat 1.0x on their oracle are banked');
ok(/ops_tuned: banked\.map/.test(src) && /engaged: t\.engagement_verified === true/.test(src)
  && /artifact: \(t\.artifacts \|\| \[\]\)\[0\]/.test(src),
  'ops_tuned keeps `engaged` + `artifact` — the two fields run_e2e.py\'s KB write-back gates on');
ok(/bankAccepted\(acceptedKernels/.test(src.slice(src.indexOf('function finalizeHeadTuning'))),
  'a banked tuning is registered as an accepted kernel (it shows up in the run ledger like any win)');

// The knowledge a tuning run produces outlives the run. Reviewer comment #2 asked for one KB rather
// than two, and this is the half that could not simply be pointed at the e2e store: that store is keyed
// on the whole deployment and gated on the run's FINAL throughput, so a proven tuned table in a run that
// ended flat was discarded with it. Filing per op in the kernel store — which ranks on isolated speedup,
// the number a tuned op actually has — is what makes tuning a producer instead of a dead end. The write
// itself lives in run_e2e.py (its own suite guards it); what this file guards is that the fold still
// emits the shape that write gates on, asserted just above.
const runE2eSrc = fs.readFileSync(path.join(ROOT, 'interface', 'run_e2e.py'), 'utf8');
ok(/--carrier", "tuned_artifact/.test(runE2eSrc),
  'the tuned table is filed under the tuned_artifact carrier (data, and no diff can express it)');
ok(/o\.get\("engaged"\) is True\s*\n\s*and _as_float\(o\.get\("isolated_speedup"\)\) > 1\.0/.test(runE2eSrc),
  'per-op gate on the write side matches the fold: engaged AND >1.0x');
// The read is PLANE-addressed, not directory-addressed: the write goes to the shared service, and a
// directory read would look at this run's own checkout, which is created empty and deleted with the run
// — a miss indistinguishable from a genuinely empty page. Name the keys the role actually reads.
const roleSrc = fs.existsSync(ROLE) ? fs.readFileSync(ROLE, 'utf8') : '';
for (const key of ['TUNED_KB_PLANE', 'TUNED_KB_STORE', 'TUNED_KB_GFX', 'TUNED_KB_PRECISION',
                   'TUNED_KB_SCRIPT', 'TUNED_KB_ENV_PRELUDE']) {
  ok(new RegExp(key + ':').test(src) && new RegExp('\\$' + key).test(roleSrc),
    `${key} is handed to the role AND read by it (a key only one side knows is dead plumbing)`);
}
ok(/TUNING_KB_ENABLED && KB_DIMS && KB_DIMS\.gfx \?/.test(src),
  'the store handles sit behind the same blind-eval switch the write does');

// ---------------------------------------------------------------------------
// C. OFF is additive-free.
// ---------------------------------------------------------------------------
console.log('\n## C. additive when off');
const gate = src.match(/const TUNING_SKILLSET_ENABLED = [\s\S]*?const TUNING_KB_ENABLED = [^\n]*\n/);
ok(!!gate, 'TUNING_* gating block found');
if (gate) {
  const make = new Function('A', 'WORKFLOW_DIR',
    gate[0] + '\nreturn { TUNING_SKILLSET_ENABLED, TUNING_SKILLSET_DIR, TUNING_KB_ENABLED };');
  const off = make({ tuning_skillset: 'false' }, '/repo/e2e_workflow');
  ok(off.TUNING_SKILLSET_ENABLED === false, 'tuning_skillset:"false" disables the whole track');
  const on = make({}, '/repo/e2e_workflow');
  ok(on.TUNING_SKILLSET_ENABLED === true, 'default is ON');
  ok(on.TUNING_SKILLSET_DIR === '/repo/perf_knowledge/expert_skills/tuning',
    'skillset dir defaults to the vendored tree beside the workflow dir');
  ok(make({ tuning_skillset_dir: '/elsewhere/skillset/' }, '/repo/e2e_workflow').TUNING_SKILLSET_DIR === '/elsewhere/skillset',
    'the vendored tree can be overridden (e.g. point at an upstream checkout to re-verify standalone)');
  ok(on.TUNING_KB_ENABLED === true && make({ tuning_kb: 'false' }, '/wf').TUNING_KB_ENABLED === false,
    'tuning-kb (the answer key) is ON by default and gateable for blind evaluation runs');
}
// The tuning search is uncapped by op count — it rides the head track's own op budget now. What IS
// bounded is the cheap rung's wall clock, and setting it to 0 must switch that rung off cleanly.
ok(!/TUNING_BUDGET/.test(src), 'no separate op budget caps tuning (it inherits the head track\'s)');
const qtGate = src.match(/const QUICK_TUNE_MINUTES = [\s\S]*?const QUICK_TUNE_ENABLED = [^\n]*\n/);
ok(!!qtGate, 'QUICK_TUNE_* gating block found');
if (qtGate) {
  const mk = new Function('A', 'TUNING_SKILLSET_ENABLED',
    qtGate[0] + '\nreturn { QUICK_TUNE_MINUTES, QUICK_TUNE_ENABLED };');
  ok(mk({}, true).QUICK_TUNE_MINUTES === 25, 'quick_tune defaults to a 25-minute budget');
  ok(mk({ quick_tune_minutes: '0' }, true).QUICK_TUNE_ENABLED === false,
    'quick_tune_minutes:0 switches the cheap rung off (the bake-off ladder is unchanged without it)');
  ok(mk({}, false).QUICK_TUNE_ENABLED === false,
    'tuning_skillset:"false" also closes the cheap rung — one flag cannot half-apply');
}
// The report/finalize inputs must be an EMPTY spread when off, so those prompts are unchanged. These are
// FUNCTIONS, not consts, and that is load-bearing after the fold: `tuning` is not settled until
// finalizeHeadTuning() runs at the end of the head track, which is BELOW this definition. A const here
// would capture `null` and silently drop every banked win out of the report and the deploy bundle.
ok(/const tuningReportInputs = \(\) => \(\(TUNING_SKILLSET_ENABLED && tuning\) \? \{ TUNING_RESULT: tuning \} : \{\}\);/.test(src),
  'report inputs are a FUNCTION returning {} when tuning is off/absent (Report prompt byte-identical)');
ok((src.match(/\.\.\.tuningReportInputs\(\)/g) || []).length === 1,
  'the report spread appears exactly once (Report phase only)');
ok(/const tuningFinalizeInputs = \(\) =>/.test(src), 'finalize inputs are a function for the same reason');
// Fast mode's contract is HeadKernel-only. 'tune' stays in the skip set purely so an older
// `phases=tune` caller still resolves false rather than crashing on an unknown token.
ok(/FAST_SKIP = FAST_MODE \? new Set\(\['config', 'tune', 'kernel'\]\)/.test(src),
  "fast mode's skip set is unchanged ('tune' survives as a legacy token)");

// ---------------------------------------------------------------------------
// D. The role delegates to the skillset instead of restating it.
// ---------------------------------------------------------------------------
console.log('\n## D. role is a thin adapter');
ok(fs.existsSync(ROLE), 'roles/quick_tune.md exists');
const role = fs.existsSync(ROLE) ? fs.readFileSync(ROLE, 'utf8') : '';
if (role) {
  ok(/## PHASE=quick_tune/.test(role), 'role defines PHASE=quick_tune');
  for (const key of ['TUNING_SKILLSET_DIR', 'TUNING_KB_ENABLED', 'OP_TASK_DIR', 'CURRENT_OVERLAY']) {
    ok(role.includes(key), `role consumes ${key}`);
  }
  ok(/never edit anything inside it/i.test(role), 'role forbids editing the vendored tree');
  // The rung is defined by what it does NOT do. If it starts launching servers it stops being cheap,
  // and the ladder loses the reason it exists.
  ok(/no serving stack, no server launch, no e2e benchmark/i.test(role),
    'role states it runs BARE — no server, no e2e bench (that is what makes it the cheap rung)');
  ok(/op_bench\.py/.test(role) && /reference_io\.pt/.test(role),
    'its oracle is the extracted op unittest, scored by op_bench.py against the pinned reference IO');
  ok(/engagement/i.test(role) && /A recall is not an accept/.test(role),
    'engagement must be PROVEN, and a KB recall still has to earn its accept on this box');
  // The point of vendoring whole is that the METHOD stays in the skillset. The role must route into it
  // and must not grow into a paraphrase of the loop, which is the failure mode this guards.
  ok(/[Rr]ead them and use them/.test(role),
    'role routes into the skillset rather than restating it');
  ok(!/TUNING_BUDGET/.test(role), 'role imposes no op budget');
  // What this guards is the role growing into a PARAPHRASE OF THE METHOD — the thing that is supposed
  // to stay in the vendored skillset. Two sections are not method and are measured separately: "Prior
  // tuning knowledge" is the contract for reading a store (which plane, which carrier, what a silent
  // miss looks like), and "The deliverable" is the contract for the deploy bundle (which file goes
  // where, what the integrator will look for). Neither has anywhere else to live, and measuring them
  // against the method budget made a healthy role look like a bloated one. Both are still capped by
  // the total, so being exempt from the first budget is not a licence to grow without limit.
  const KB_SECTION = /### Prior tuning knowledge[\s\S]*?\n---\n/;
  const DELIVERABLE = /### The deliverable[\s\S]*?(?=\n### )/;
  const method = role.replace(KB_SECTION, '').replace(DELIVERABLE, '').split(/\s+/).length;
  const total = role.split(/\s+/).length;
  ok(KB_SECTION.test(role), 'the KB contract section is still findable by the split above');
  ok(DELIVERABLE.test(role), 'the deploy-bundle contract section is still findable by the split above');
  ok(method < 1600, `the METHOD prose stays a thin adapter, not a manual (${method} words < 1600)`);
  ok(total < 2600, `the role as a whole stays bounded (${total} words < 2600)`);
}
// The other consumer is a rung on the bake-off ladder, not a role of its own — op_benchmarker must be
// told that Tier B owns the skillset, and must be told to read what quick_tune already measured. A
// rung that re-derives the cheap levers wastes the whole point of putting a cheap rung below it.
const opb = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'roles', 'op_benchmarker.md'), 'utf8');
ok(/TUNING_SKILLSET_DIR/.test(opb) && /never edit anything/.test(opb),
  'op_benchmarker is handed the vendored skillset under the same never-edit rule');
ok(/QUICK_TUNE/.test(opb) && /[Rr]ead `QUICK_TUNE` before you plan anything/.test(opb),
  'op_benchmarker is told to read QUICK_TUNE first (its levers_tried are measurements, not advice)');

// ---------------------------------------------------------------------------
// E. The win reaches production. A tuned DATA artifact cannot ride the PYTHONPATH overlay, so it must
//    travel through final_patch.diff + final_launch.sh or the reported gain will not reproduce.
// ---------------------------------------------------------------------------
console.log('\n## E. tuning reaches the final bundle');
ok(/deploy_bundle: \{ type: 'string' \}/.test(src), 'QUICKTUNE_SCHEMA accepts a deploy bundle');
ok(/deploy_bundle: `\$\{EVAL_DIR\}\/tuning\/deploy`/.test(src),
  'the fold names one bundle for the whole run — per-op bundles merge into it, they do not each ship');
ok(/const tuningFinalizeInputs = \(\) => \(\(TUNING_SKILLSET_ENABLED && tuning && tuning\.gate === 'accepted'\)/.test(src),
  'the deploy bundle is handed to Finalize only when tuning actually banked a win');
ok((src.match(/\.\.\.tuningFinalizeInputs\(\)/g) || []).length === 1,
  'the finalize spread appears exactly once (Finalize phase only, empty otherwise)');
ok(/tuning_in_bundle: \{ type: 'boolean' \}/.test(src),
  'Finalize reports whether the tuning made it into the shipped bundle');
const integrator = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'roles', 'e2e_integrator.md'), 'utf8');
ok(/TUNING_DEPLOY_BUNDLE/.test(integrator), 'the integrator knows how to fold the deploy bundle in');
ok(/final_patch\.diff/.test(integrator) && /deploy\.sh/.test(integrator),
  'the bundle travels through the EXISTING handles: concatenated into final_patch.diff, run from final_launch.sh');
ok(/before[\s\S]{0,40}the server launch/i.test(integrator),
  'deploy.sh runs BEFORE launch (a config applied to a running server does nothing)');
// The live-tree carve-out. GEAK forbids editing installed packages and asserts the tree is pristine
// before/after every A/B leg, but an accepted tuning deploy legitimately writes into it. Without the
// carve-out the head track cleans the tuning away mid-run and every later delta is measured against a
// reference leg that quietly lost it.
ok(/live_tree_files: arrStr/.test(src), 'QUICKTUNE_SCHEMA carries the live-tree paths the deploy owns');
ok(/live_tree_files: \[\.\.\.new Set\(banked\.flatMap\(\(t\) => t\.live_tree_files \|\| \[\]\)\)\]/.test(src),
  'the fold unions every banked op\'s live-tree paths (a carve-out that lists only one op\'s files ' +
  'lets the integrator clean the others away mid-run)');
ok(/function tuningIntegrateInputs\(\)/.test(src)
  && /if \(!\(TUNING_SKILLSET_ENABLED && tuning && tuning\.gate === 'accepted'\)\) return \{\};/.test(src),
  'the carve-out is empty unless an accept was banked (integrate prompt unchanged without tuning)');
ok((src.match(/\.\.\.tuningIntegrateInputs\(\)/g) || []).length === 2,
  'every integrate path gets the carve-out: runIntegrateBothLegs (covers head/milestone/corrective) + the deep lane');
ok(/TUNING_LIVE_TREE_FILES/.test(integrator),
  'the integrator honours the carve-out in its never-mutate-site-packages rule');
ok(/any OTHER dirty path is still a hard failure/.test(integrator),
  'the carve-out is scoped to the declared list — it does not blunt the rule');
ok(/live_tree_files/.test(role), 'the role is told to declare every live-tree path it writes');
ok(/covers \*\*data only\*\*/.test(integrator),
  'the carve-out is data-only — a .py in the live tree cannot be varied per A/B leg');
// Tuning-enabling CODE changes are in scope (a tuned table behind an unrouted seam binds to nothing),
// and they travel as a reversible overlay, never as a live-tree source edit.
ok(/apply_overlay: \{ type: 'string' \}/.test(src), 'the rung can return a routing/dispatch overlay');
ok(/if \(tuning\.apply_overlay\) curOverlay = tuning\.apply_overlay;/.test(src),
  'an accepted tuning overlay is carried forward (the code half is not dropped at the fold)');
ok(/OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,/.test(src),
  'the post-head re-profile runs WITH the carried overlay, not an empty one');
ok(/reversible \*\*overlay\*\*/.test(role) && /Never edit a `\.py`\s+in the installed tree/.test(role),
  'the role routes code through the overlay and forbids live-tree source edits');
// result.json must gain the tuning block WITHOUT any existing key changing.
const runE2e = fs.readFileSync(path.join(ROOT, 'interface', 'run_e2e.py'), 'utf8');
ok(/def _tuning_skillset_section\(/.test(runE2e), 'run_e2e.py builds an additive tuning_skillset block');
ok(/if tuning_section is not None:\n\s+result\["tuning_skillset"\] = tuning_section/.test(runE2e),
  'the block is appended after the result dict is complete, so no existing key is touched');
ok(/reaches_production_via/.test(runE2e),
  'result.json tells the caller that final_patch/final_launch_script now carry the tuning');

// ---------------------------------------------------------------------------
// G. the tuning track READS the same KB it writes
//
// Before this, tuning was write-only: a run banked a tuned table and the next run on the identical
// deployment searched for it again from scratch, for hours, with the answer sitting in the store the
// whole time. The loop only closes if the role is a warm-start consumer AND the reference paths reach
// its Inputs — the prompt block alone is not enough, since it is empty whenever the read found
// nothing, which is also when a reader most needs to know the store exists.
// ---------------------------------------------------------------------------
console.log('\n## G. the tuning track reads the KB it writes');
ok(/const WARM_START_ROLES = new Set\(\[[^\]]*'quick_tune'/.test(src),
  'quick_tune is a warm-start consumer (the prompt block reaches it)');
const hti = src.slice(src.indexOf('const headTuningInputs'), src.indexOf('function finalizeHeadTuning'));
ok(/\.\.\.\(TUNING_KB_ENABLED \? KB_REF_INPUTS : \{\}\)/.test(hti),
  'the always-fires Inputs channel reaches every head-tuning consumer');
ok(/KB_CACHE_DIR/.test(hti),
  'the agents are told where the recalled artifacts were materialized, not just that they exist');
// One switch, both stores. `tuning_kb=false` is the blind-evaluation control; when the tuning
// knowledge moved into the shared KB, a block gated only on KB_REF_DIR would have kept feeding the
// role priors in exactly the runs designed to have none.
ok(/if \(role === 'quick_tune' && !TUNING_KB_ENABLED\) return '';/.test(src),
  'blind eval stays blind: TUNING_KB_ENABLED=false closes the KB block as well as tuning-kb/');
ok(/\.\.\.\(TUNING_KB_ENABLED && KB_CACHE_DIR \? \{ KB_CACHE_DIR \} : \{\}\)/.test(hti),
  '...and closes the Inputs channel with it — one flag cannot half-apply');
ok(at(/if \(E2E_WARM_START_ON\) \{/) < iHead,
  'the warm start runs BEFORE the head track, so KB_REF_DIR is armed by the time the rungs read it');
ok(/tuning_source: \/kb\|recall\|knowledge\/i\.test\(String\(t\.mode \|\| ''\)\) \? 'recall' : 'search'/.test(src),
  'a banked op records whether it was searched or recalled (a recall must not re-bank as a discovery)');
ok(/from_tuning_skillset: true/.test(src) && /tuning_artifact: \(t\.artifacts/.test(src),
  'the banked record carries the LEVER (which artifact, applied how) — without it the KB entry says a ' +
  'kernel got faster and nothing about why');
ok(/Prior tuning knowledge/.test(role) && /KB_REFERENCE_DIR/.test(role),
  'the role file tells the rung to check the KB before searching');
ok(/prove engagement/i.test(role) && /A recall is not an accept/.test(role),
  'a recalled artifact still has to earn its accept on this box');

// ---------------------------------------------------------------------------
// F. Attribution is NULL, not zero.
//
// The standalone phase took its own interleaved pre/post SERVER A/B, so it could say "tuning was 12.4%
// of a 31% run gain". The in-head fold does not take that measurement — deliberately: a per-op pass
// scored on a per-op oracle is what makes the cheap rung cheap, and a whole-server A/B per op would
// cost more than the tuning it is attributing. So the three attribution keys are gone as numbers and
// present as nulls. This distinction is the whole risk of the refactor: a consumer that renders null
// as 0.0% turns "we did not measure this" into "this contributed nothing", which is a different and
// false claim, and it is the claim that would get a real win thrown away.
// ---------------------------------------------------------------------------
console.log('\n## F. unmeasured attribution is null, never 0');
const fold = src.slice(src.indexOf('function finalizeHeadTuning'), src.indexOf('// The CHEAP standalone pass'));
ok(/tuning_delta_pct: null/.test(fold), 'the fold emits tuning_delta_pct as null');
ok(!/tuning_delta_pct: 0/.test(fold), 'and never as 0');
const tr = src.slice(src.indexOf('function tuningReturn'), src.indexOf('function tuningReturn') + 4000);
ok(/tuning_delta_pct: null/.test(tr) && /tuning_speedup: null/.test(tr)
  && /share_of_total_gain_pct: null/.test(tr),
  'all three attribution keys leave tuningReturn() as null');
ok(/source: 'head_track'/.test(tr) && /attempts: headTuning\.length/.test(tr),
  'what replaces them is provenance: which track measured this, and how many ops it tried');
// The old pre/post legs are gone. Leaving one of their keys behind would be worse than removing the
// feature: a reader would take a stale field for a fresh measurement.
for (const dead of ['pre_tune_throughput_tok_s', 'post_tune_throughput_tok_s', 'ab_interleaved']) {
  ok(!new RegExp(dead).test(src), `the standalone A/B's \`${dead}\` is gone, not left stale`);
}
// The rendering side. run_e2e.py must state the per-op evidence and must NOT print a number for the
// share it did not measure.
ok(/isolated_speedup/.test(runE2eSrc) && /not measured/.test(runE2eSrc),
  'run_e2e.py reports per-op isolated speedups and says the share is NOT MEASURED');
const roles = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'roles', 'system_architect.md'), 'utf8');
ok(/not measured/i.test(roles) && /per op/i.test(roles),
  'the report role is told to attribute tuning per op and not to print null as 0%');

console.log(failures === 0
  ? '\nPASS: the tuning skillset is vendored whole and runs per head op inside the HeadKernel track.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
