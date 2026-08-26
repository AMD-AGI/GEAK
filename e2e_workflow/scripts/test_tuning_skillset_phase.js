#!/usr/bin/env node
// Regression guard for the standalone TuningSkillset phase (no GPU, no model, no agent needed).
//
// Five invariants, all of which the integration depends on and none of which is obvious from reading
// one file:
//
//   A. WHOLE — the skillset is vendored as one intact tree with its own entry points, and its method is
//      NOT copied/scattered into GEAK's own knowledge/ files. It is validated standalone, so GEAK must
//      run the copy that was validated; a scattered paraphrase silently voids that validation.
//   B. STANDALONE + BEFORE HeadKernel — it is its own gated phase (`want('tune')`) whose block appears
//      after ConfigSweep and before HeadKernel in BOTH the declared phase list and the source order.
//      This is what lets it run its complete loop and own an attributable pre/post A/B.
//   C. ADDITIVE OFF — `tuning_skillset:"false"` injects nothing anywhere, so the run is byte-identical
//      to a build without the feature.
//   D. THIN ADAPTER — the role routes into the skillset instead of paraphrasing its method.
//   E. REACHES PRODUCTION — the win travels through the EXISTING deliverable handles (final_patch.diff,
//      final_launch.sh) rather than dying in the eval dir. GEAK's overlay is a PYTHONPATH mechanism for
//      code, but a tuned artifact is usually data plus a cache invalidation, so without this the run
//      reports a gain that the shipped bundle does not reproduce. result.json gains its block additively.
//
// Run:  node e2e_workflow/scripts/test_tuning_skillset_phase.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');            // .../GEAK
const FILE = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');
const ROLE = path.join(ROOT, 'e2e_workflow', 'roles', 'tuning_specialist.md');
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
// B. Standalone phase, positioned after ConfigSweep and before HeadKernel.
// ---------------------------------------------------------------------------
console.log('\n## B. standalone, before HeadKernel');
const phaseList = src.match(/phases:\s*\[[\s\S]*?\n  \],/);
ok(!!phaseList, 'meta.phases list found');
if (phaseList) {
  const order = (title) => phaseList[0].indexOf(`title: '${title}'`);
  ok(order('TuningSkillset') > 0, "meta.phases declares 'TuningSkillset'");
  ok(order('ConfigSweep') < order('TuningSkillset'), 'declared AFTER ConfigSweep');
  ok(order('TuningSkillset') < order('HeadKernel'), 'declared BEFORE HeadKernel');
}
// Source order is what actually runs — assert the gated blocks are in the same sequence.
const at = (re) => src.search(re);
const iConfig = at(/if \(want\('config'\)/);
const iTune = at(/if \(want\('tune'\) && TUNING_SKILLSET_ENABLED\)/);
const iHead = at(/if \(want\('head'\)/);
ok(iTune > 0, "a gated `want('tune')` block exists");
ok(iConfig > 0 && iConfig < iTune, 'the tune block runs AFTER the ConfigSweep block');
ok(iTune < iHead, 'the tune block runs BEFORE the HeadKernel block');
ok(/phase\('TuningSkillset'\)/.test(src), 'the block announces itself as its own phase');
ok(/roleAgent\('tuning_specialist', 'tune'/.test(src), 'it dispatches ONE dedicated role (not a fragment on another role)');
// It must NOT be injected into any OTHER role's prompt — spraying it around is the scattering failure
// mode this whole design avoids. Walk every roleAgent() call site (each ends at its `{ phase:` options
// object) and assert the skillset inputs appear in exactly one: tuning_specialist's.
const consumers = [];
for (let i = src.indexOf("roleAgent('"); i !== -1; i = src.indexOf("roleAgent('", i + 1)) {
  const end = src.indexOf('{ phase:', i);
  const call = src.slice(i, end === -1 ? i + 2000 : end);
  const role = (call.match(/roleAgent\('([a-z_]+)'/) || [])[1] || '?';
  if (/TUNING_(SKILLSET_DIR|KB_ENABLED|BUDGET|TARGETS)/.test(call)) consumers.push(role);
}
ok(consumers.length === 1 && consumers[0] === 'tuning_specialist',
  `skillset inputs reach exactly ONE role prompt — tuning_specialist (got: ${consumers.join(', ') || 'none'}) ` +
  `-> it is run whole by its owner, never sprayed across other roles`);

// Attribution: it takes its own pre/post legs and folds the accept into the carried config.
ok(/pre_tune_throughput_tok_s/.test(src) && /post_tune_throughput_tok_s/.test(src),
  'the phase reports its OWN pre/post throughput (attributable delta)');
ok(/share_of_total_gain_pct/.test(src), "the return exposes tuning's share of the run's total gain");
ok(/curEnv = \(curEnv \? curEnv \+ ' ' : ''\) \+ tuning\.apply_env/.test(src),
  'an accepted deploy folds its required env into the carried config (downstream phases inherit the tuning)');
ok(/profiler', 'reprofile'[\s\S]{0,400}ROUND: 'tuning'/.test(src),
  'an accepted tuning triggers a re-profile (tuning reshapes which kernels dominate)');
// The skillset's own thesis, enforced by the orchestrator rather than trusted from the agent.
ok(/engagement_verified === true/.test(src),
  'an accept is refused unless engagement was PROVEN (the skillset\'s central claim, enforced in code)');

// The knowledge a tuning run produces outlives the run. Reviewer comment #2 asked for one KB rather
// than two, and this is the half that could not simply be pointed at the e2e store: that store is keyed
// on the whole deployment and gated on the run's FINAL throughput, so a proven tuned table in a run that
// ended flat was discarded with it. Filing per op in the kernel store — which ranks on isolated speedup,
// the number a tuned op actually has — is what makes the tuning phase a producer instead of a dead end.
ok(/experience_store\.py/.test(src),
  "accepted tuning is written into the kernel lane's experience store");
ok(/--carrier tuned_artifact/.test(src),
  'it is filed under the tuned_artifact carrier (a tuned table is data, and no diff can express it)');
ok(/Number\(o\.isolated_speedup\) > 1\.0 && o\.engaged === true/.test(src),
  'per-op gate: only ops that beat 1.0x AND were proven engaged are filed');
// The whole point of writing here rather than there: the run's own e2e verdict must not gate it.
const wb = src.slice(src.indexOf('const kernelKbOps'), src.indexOf('const kernelKbOps') + 3000);
ok(!/win_gate|throughput_speedup|kbNoWinVerdict/.test(wb),
  "the tuned-op write does NOT consult the run's e2e verdict (that coupling is what it exists to break)");
ok(/kernel-kb\] tuned-op write failed \(NON-FATAL/.test(src),
  'a failed KB write loses a record, never a measurement — it cannot fail the run');
ok(/TUNED_KB_ROOT: KB_ARTIFACTS_DIR/.test(src) && /TUNING_KB_ENABLED && KB_DIMS && KB_DIMS\.gfx \?/.test(src),
  'the same store is handed BACK to the role to read, under the same blind-eval switch');

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
  ok(off.TUNING_SKILLSET_ENABLED === false, 'tuning_skillset:"false" disables the phase');
  const on = make({}, '/repo/e2e_workflow');
  ok(on.TUNING_SKILLSET_ENABLED === true, 'default is ON');
  ok(on.TUNING_SKILLSET_DIR === '/repo/perf_knowledge/expert_skills/tuning',
    'skillset dir defaults to the vendored tree beside the workflow dir');
  ok(make({ tuning_skillset_dir: '/elsewhere/skillset/' }, '/repo/e2e_workflow').TUNING_SKILLSET_DIR === '/elsewhere/skillset',
    'the vendored tree can be overridden (e.g. point at an upstream checkout to re-verify standalone)');
  ok(on.TUNING_KB_ENABLED === true && make({ tuning_kb: 'false' }, '/wf').TUNING_KB_ENABLED === false,
    'tuning-kb (the answer key) is ON by default and gateable for blind evaluation runs');
}
// The tuning loop is uncapped by design: tuning ops are cheap and cumulative, unlike head ops.
ok(!/TUNING_BUDGET/.test(src), 'no op budget caps the tuning loop');
// The report inputs must be an EMPTY spread when off, so the Report prompt is unchanged.
ok(/const TUNING_REPORT_INPUTS = \(TUNING_SKILLSET_ENABLED && tuning\) \? \{ TUNING_RESULT: tuning \} : \{\};/.test(src),
  'report inputs are an empty object when the phase is off/absent (Report prompt byte-identical)');
ok((src.match(/\.\.\.TUNING_REPORT_INPUTS/g) || []).length === 1,
  'the report spread appears exactly once (Report phase only)');
// Fast mode's contract is HeadKernel-only, so 'tune' must join its skip set.
ok(/FAST_SKIP = FAST_MODE \? new Set\(\['config', 'tune', 'kernel'\]\)/.test(src),
  "fast mode skips 'tune' (its budget is reserved for the head track)");

// ---------------------------------------------------------------------------
// D. The role delegates to the skillset instead of restating it.
// ---------------------------------------------------------------------------
console.log('\n## D. role is a thin adapter');
ok(fs.existsSync(ROLE), 'roles/tuning_specialist.md exists');
const role = fs.existsSync(ROLE) ? fs.readFileSync(ROLE, 'utf8') : '';
if (role) {
  ok(/## PHASE=tune/.test(role), 'role defines PHASE=tune');
  for (const key of ['TUNING_SKILLSET_DIR', 'TUNING_KB_ENABLED', 'CURRENT_THROUGHPUT', 'CURRENT_OVERLAY']) {
    ok(role.includes(key), `role consumes ${key}`);
  }
  ok(/never edit anything inside it/i.test(role), 'role forbids editing the vendored tree');
  ok(/engagement/i.test(role) && /isolated-server A\/B/i.test(role),
    'role carries engagement proof and the isolated-server A/B contract');
  // The point of vendoring whole is that the METHOD stays in the skillset. The role must route into it
  // and must not grow into a paraphrase of the loop, which is the failure mode this guards.
  ok(/[Rr]ead them and use them/.test(role),
    'role routes into the skillset rather than restating it');
  ok(!/TUNING_BUDGET/.test(role), 'role imposes no op budget');
  const words = role.split(/\s+/).length;
  ok(words < 1600, `role stays a thin adapter, not a manual (${words} words < 1600)`);
}

// ---------------------------------------------------------------------------
// E. The win reaches production. A tuned DATA artifact cannot ride the PYTHONPATH overlay, so it must
//    travel through final_patch.diff + final_launch.sh or the reported gain will not reproduce.
// ---------------------------------------------------------------------------
console.log('\n## E. tuning reaches the final bundle');
ok(/deploy_bundle: \{ type: 'string' \}/.test(src), 'TUNING_SCHEMA accepts a deploy bundle');
ok(/const TUNING_FINALIZE_INPUTS = \(TUNING_SKILLSET_ENABLED && tuning && tuning\.gate === 'accepted'\)/.test(src),
  'the deploy bundle is handed to Finalize only when tuning actually banked a win');
ok(/\.\.\.TUNING_FINALIZE_INPUTS,/.test(src)
  && (src.match(/\.\.\.TUNING_FINALIZE_INPUTS/g) || []).length === 1,
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
ok(/live_tree_files: arrStr/.test(src), 'TUNING_SCHEMA carries the live-tree paths the deploy owns');
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
ok(/apply_overlay: \{ type: 'string' \}/.test(src), 'the phase can return a routing/dispatch overlay');
ok(/if \(tuning\.apply_overlay\) curOverlay = tuning\.apply_overlay;/.test(src),
  'an accepted tuning overlay is carried forward (the code half is not dropped at the phase boundary)');
ok(/OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,/.test(src),
  'the post-tuning re-profile runs WITH the carried overlay, not an empty one');
ok(/reversible \*\*overlay\*\*/.test(role) && /Never edit a `\.py` in/.test(role),
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
ok(/const WARM_START_ROLES = new Set\(\[[^\]]*'tuning_specialist'/.test(src),
  'tuning_specialist is a warm-start consumer (the prompt block reaches it)');
const tunePhase = src.slice(iTune, src.indexOf('schema: TUNING_SCHEMA', iTune));
ok(/\.\.\.\(TUNING_KB_ENABLED \? KB_REF_INPUTS : \{\}\)/.test(tunePhase),
  'the always-fires Inputs channel reaches the tuning phase too');
ok(/KB_CACHE_DIR/.test(tunePhase),
  'the role is told where the recalled artifacts were materialized, not just that they exist');
// One switch, both stores. `tuning_kb=false` is the blind-evaluation control; when the tuning
// knowledge moved into the shared KB, a block gated only on KB_REF_DIR would have kept feeding the
// role priors in exactly the runs designed to have none.
ok(/if \(role === 'tuning_specialist' && !TUNING_KB_ENABLED\) return '';/.test(src),
  'blind eval stays blind: TUNING_KB_ENABLED=false closes the KB block as well as tuning-kb/');
ok(/\.\.\.\(TUNING_KB_ENABLED && KB_CACHE_DIR \? \{ KB_CACHE_DIR \} : \{\}\)/.test(tunePhase),
  '...and closes the Inputs channel with it — one flag cannot half-apply');
ok(iTune > at(/if \(E2E_WARM_START_ON\) \{/),
  'the warm start runs BEFORE the tuning phase, so KB_REF_DIR is armed by the time the role reads it');
ok(/tuning_source: String\(o\.source \|\| o\.origin \|\| ''\)/.test(src),
  'a banked op records whether it was searched or recalled (a recall must not re-bank as a discovery)');
ok(/Prior tuning knowledge/.test(role) && /KB_REFERENCE_DIR/.test(role),
  'the role file tells the specialist to check the KB before searching');
ok(/prove engagement/i.test(role) && /A recall is not an accept/.test(role),
  'a recalled artifact still has to earn its accept on this box');

console.log(failures === 0
  ? '\nPASS: tuning skillset is vendored whole and runs standalone before HeadKernel.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
