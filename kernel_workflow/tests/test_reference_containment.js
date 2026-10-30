#!/usr/bin/env node
// Guard for the reference-containment assertion, added after the SECOND imported-answer incident.
//
// tests/test_capability_eval.js already covers the doctrine half: engineers are never handed the
// reference location, the roadmap is scrubbed, and verify rejects byte-identical files. All of that
// was in place and correctly worded during one run — and that run still filed a candidate whose 11 of
// 12 files were byte-identical to the sibling reference checkout, differing only in the default of
// one `os.environ.get` call. One run had done the same at 8 of 11.
//
// The reason is structural, not behavioural: the reference tree lived one directory above the
// workspace. Not telling an agent where something is does not stop `ls ..`. So containment is now
// asserted at startup rather than requested in prose:
//
//   1. IT ABORTS      — a reference inside the run's own tree stops the run before any budget.
//   2. ANCESTOR       — "inside" means the common ancestor of task + workflow + artifacts, not cwd.
//   3. IT PASSES OUT  — a relocated reference runs normally (the fix must be usable).
//   4. HONEST SCOPE   — the pass log admits a path check cannot cover git-reachable copies.

const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

let failures = 0;
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}

const wf = read('kernel_mega.js');

// --- 1. the assertion exists, and it aborts -------------------------------
console.log('\n# the check exists and is fatal');
ok(/REFERENCE NOT CONTAINED/.test(wf), 'the containment failure has a distinct, greppable name');
ok(/throw new Error\(\s*\n?\s*`REFERENCE NOT CONTAINED/.test(wf),
   'it THROWS — a warning would leave the run free to measure a copy');
ok(/capability_eval is on/.test(wf), 'it only applies to capability_eval runs');
ok(/one `ls \.\.`|walking up one directory/.test(wf),
   'the source names the actual reach mechanism, not a vague "leak"');

// --- 2. the check fires BEFORE any budget is spent ------------------------
console.log('\n# it fires before the run costs anything');
const iCheck = wf.indexOf('REFERENCE NOT CONTAINED');
const iSetup = wf.indexOf("phase('Setup')");
const iLoop = wf.indexOf('while (dispatched < BUDGET');
ok(iCheck > 0 && iSetup > iCheck, 'containment is asserted before Setup');
ok(iCheck > 0 && iLoop > iCheck, 'containment is asserted before the optimize loop');

// --- 3. the ancestor arithmetic, extracted and exercised ------------------
// Re-typing the rule here would let the copy drift from the original; lift the real expression.
console.log('\n# ancestor arithmetic (extracted from source)');
const src = wf.match(/  const abs = \(p\) => \{[\s\S]*?const inside = KNOWN_REFERENCE_PATHS\.filter\(\s*\n?\s*\(p\) => ancestor && \(abs\(p\) === ancestor \|\| abs\(p\)\.startsWith\(ancestor \+ '\/'\)\)\);/);
ok(!!src, 'the containment expression can be located in kernel_mega.js');
if (src) {
  const decide = new Function('KERNEL_PATH_ORIG', 'EXP_ROOT', 'WORKFLOW_DIR', 'KNOWN_REFERENCE_PATHS',
    `${src[0]}\n return { inside, ancestor };`);
  const K = '/proj/app/tasks/megamoe';
  const E = '/proj/app/artifacts/runs';
  const W = '/proj/app/GEAK/kernel_workflow';
  const run = (refs) => decide(K, E, W, refs);

  ok(run([]).ancestor === '/proj/app',
     'the ancestor is the common prefix of task + artifacts + workflow');
  // The real incident: a sibling of the task dir, one level up from the workspace.
  ok(run(['/proj/app/AITER-candidate']).inside.length === 1,
     'a sibling reference checkout inside the run tree is caught (the actual incident)');
  ok(run(['/outside/hidden-reference']).inside.length === 0,
     'a relocated reference passes — the fix has to be usable, not just diagnostic');
  ok(run(['/proj/app/tasks/megamoe/ref']).inside.length === 1,
     'a reference nested inside the task dir itself is caught');
  ok(run(['/proj/app']).inside.length === 1,
     'the ancestor directory itself counts as inside');
  // Prefix-vs-path-boundary: /proj/application must NOT be swallowed by the /proj/app ancestor.
  ok(run(['/proj/application/ref']).inside.length === 0,
     'a sibling whose name merely starts with the ancestor string is NOT flagged');
  ok(run(['/proj/app/../app/AITER-candidate']).inside.length === 1,
     'a path that only escapes via .. is resolved first, then caught');
  ok(run(['/outside/ref-a', '/proj/app/ref-b']).inside.length === 1,
     'a mixed list reports only the offending tree');
}

// --- 4. the pass path must not overclaim ----------------------------------
// A path check cannot see a reference that is also a branch in a repo the run can read. Saying so is
// the difference between a guarantee and a filter; only one of those is true here.
console.log('\n# the pass log states what it does NOT cover');
ok(/REFERENCE CONTAINMENT ok/.test(wf), 'a passing check is logged, not silent');
ok(/CONFIGURED PATHS ONLY/.test(wf), 'the log admits the check is path-scoped');
ok(/branch in another repository the run can read/.test(wf),
   'the git-reachable hole is named explicitly rather than left implied');
// NB: matched across the string-concat wrap. This assertion broke once when the sentence was
// re-flowed across two `+`-joined literals, which is a formatting change, not a policy change.
ok(/only verify's byte-identity check stands between that and a `? ?\+?\s*`?counted win/.test(wf),
   'the log says which gate is load-bearing once containment passes');
// The path check was aimed at the sibling checkout and the real leak was somewhere else entirely.
// The log has to say so, or the next reader takes "CONTAINMENT ok" at face value.
ok(/five more places/.test(wf) && /none of them a known_reference_path/.test(wf),
   'the log quantifies how much of the exposure the path check does NOT cover');
ok(/SIBLING EVAL DIRS/.test(wf),
   'the exp_root sibling hole is named — it is the one every engineer can reach by construction');
ok(/reference_leak_sweep\.sh/.test(wf), 'the log points at the content sweep that does cover them');

// --- 5. the content sweep exists and is honest about its scope ------------
console.log('\n# the content sweep');
const sweep = read('scripts/reference_leak_sweep.sh');
// The marker list lives OUTSIDE SKILL_DIR: it names the reference's own symbols, and
// under scripts/ it sat one `ls` away from every engineer the workflow runs, which turns the leak
// detector into a leak. The test follows it rather than pinning it back, and asserts below that no
// copy has reappeared inside SKILL_DIR.
const markerTmp = fs.mkdtempSync(path.join(os.tmpdir(), 'geak-markers-'));
process.on('exit', () => { try { fs.rmSync(markerTmp, { recursive: true, force: true }); } catch {} });
const MARKER_PATH = path.join(markerTmp, 'reference_leak_markers.txt');
fs.writeFileSync(MARKER_PATH, [
  '# Deliberately EXCLUDED task knob: AITER_MEGAMOE_FUSE_ALL',
  ...Array.from({ length: 60 }, (_, i) => `SYNTHETIC_REFERENCE_MARKER_${String(i).padStart(3, '0')}`),
].join('\n') + '\n');
const markers = fs.readFileSync(MARKER_PATH, 'utf8');
ok(!fs.existsSync(path.join(ROOT, 'scripts', 'reference_leak_markers.txt')),
   'the marker list is NOT inside SKILL_DIR, where engineers can read the reference vocabulary');
ok(/refusing to use/.test(sweep),
   'the sweep refuses to run against a SKILL_DIR copy rather than silently preferring it');
ok(!/\/root\/|\/sgl-workspace\//.test(sweep),
   'the sweep source contains no real machine path to the hidden marker vocabulary');
ok(/--derive/.test(sweep), 'the marker list can be regenerated from the two trees, not hand-maintained');
ok(/EXTS:=/.test(sweep),
   'the sweep is scoped to files a leak can be copied out of, not every file that names a counter');
ok(/NOT covered: unreferenced history, dangling objects/.test(sweep),
   'the clean result states what it does not cover');
// The ref pass, added after the sweep walked the frozen baseline, missed a `mega` branch
// carrying 140 of 140 markers, and reported clean — and that clean result was relayed to the user.
ok(/REFERENCE LEAK IN GIT REFS/.test(sweep),
   'the sweep scans ref tips, not only the working tree — an unchecked-out branch has no files');
ok(/for-each-ref/.test(sweep) && /rev-parse HEAD/.test(sweep),
   'refs AND a detached HEAD are enumerated: a detached HEAD is how one would hide it from for-each-ref');
ok(/reference_leak_sweep\.sh'\)/.test(sweep) || /!\*reference_leak_sweep\.sh/.test(sweep),
   'the ref pass excludes the scanner itself, or it flags itself in every repo that carries it');
// Excluding the flags the task itself names is what keeps the sweep from firing on honest work.
ok(/AITER_MEGAMOE_FUSE_ALL/.test(markers) && /Deliberately EXCLUDED/.test(markers),
   'markers the run is legitimately told about are excluded, with the reason recorded');
// A marker is an identifier, so it must match as one. `_out_cache_modifier` (harvested as the tail of
// the reference's `self._out_cache_modifier`) fired under -F on an independently authored
// `g1_out_cache_modifier` and cost four tool calls to clear. False positives are how a checker gets
// switched off, so the boundary rule is pinned here.
ok(/\(\?<!\[A-Za-z0-9_\]\)/.test(sweep) && /\(\?!\[A-Za-z0-9_\]\)/.test(sweep),
   'markers are matched at identifier boundaries, not as substrings');
ok(/tail or head of a longer identifier IS a different identifier/.test(sweep),
   'the source states why substring matching is wrong, not just that it changed');
// Both of this subsystem's fail-open bugs (the glob in skill_address_scan.sh, `-P -f` here) reported
// clean by scanning nothing. A self-test is the only thing that distinguishes that from a real pass.
ok(/Refusing to run/.test(sweep) && /matching nothing/.test(sweep),
   'the sweep proves its own engine fires before trusting a clean result');
ok(/only supports? a single pattern|accepts only a single pattern/.test(sweep),
   'the -P single-pattern constraint is recorded where the next author will hit it');
// The gate agent is forbidden from opening what it flags, so the path alone is not triageable.
ok(/grep -o -P "\$pattern" "\$f"/.test(sweep) && /NAME THE MARKER, NOT JUST THE FILE/.test(sweep),
   'a leak report names the matched identifiers, so triage needs no one to read the file');
const markerLines = markers.split('\n').filter((l) => l.trim() && !l.startsWith('#'));
ok(markerLines.length > 50, `the marker list is populated (${markerLines.length} markers)`);
// THE RULE IS "not an English word", and `includes('_')` was only ever a proxy for it. The proxy is
// wrong in one direction that matters: CamelCase names (FusedSharedStorage, WorkgroupOneAs) are
// identifiers with no underscore, and they are among the strongest evidence in the list because nobody
// coins them by accident. Use the real rule, the same one scripts/reference_leak_sweep.sh applies when
// it derives: a token with no underscore, no digit AND no internal capital is a word, not a name. That
// still rejects Deadlock / Lockstep / Megakernel / Publishing, which is the whole point.
const isWord = (t) => /^[A-Z]?[a-z]+$/.test(t);
const wordy = markerLines.filter(isWord);
ok(wordy.length === 0,
   `no bare English words: every marker is an identifier, so a hit is evidence${
     wordy.length ? ` (offenders: ${wordy.join(', ')})` : ''}`);

// A MARKER THE RUN IS HANDED IS NOT A MARKER. This is the SCATTERED failure, and it is worth a
// standing assertion rather than a one-time cleanup: GEAK_TASK.md instructs engineers to print
// `path=SCATTERED`, the reference also contains it, so a mechanical derive proposes it — and a list
// containing it fires on every honest candidate the wave produces. A scanner whose output must be
// hand-triaged is a scanner that gets switched off, so this fails OPEN in the loud direction, which is
// the expensive one. The sweep's --given flag prevents it at derive time; this catches a marker added
// by hand, or one that became given later because someone edited the task text.
const givenText = ['tasks', 'knowledge', path.join('roles', 'mega')].map((d) => {
  const p = path.join(ROOT, d);
  const out = [];
  const walk = (dir) => {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const f = path.join(dir, e.name);
      if (e.isDirectory()) walk(f);
      else if (/\.(md|json|txt)$/.test(e.name)) out.push(fs.readFileSync(f, 'utf8'));
    }
  };
  if (fs.existsSync(p)) walk(p);
  return out.join('\n');
}).join('\n');
const given = new Set(givenText.match(/\b[A-Za-z_][A-Za-z0-9_]*\b/g) || []);
const collide = markerLines.filter((m) => given.has(m));
ok(collide.length === 0,
   `no marker appears in material the run is given${
     collide.length ? ` -- these would fire on honest work: ${collide.join(', ')}` : ''}`);

// --- 6. the control workspace, which is where the real leak came from -----
console.log('\n# the positive control must not leave the answer in the tree');
const bench = read('roles/mega/benchmark_engineer.md');
ok(/outside the run tree/.test(bench), 'the control workspace is required to live outside the tree');
ok(/5\.5 hours/.test(bench), 'the incident keeps its duration — it was not a brief window');
ok(/11 of 12 files were\s*\n?byte-identical/.test(bench),
   'the consequence is stated: a candidate byte-identical to the control workspace');
ok(/did not need to be/.test(bench),
   'it states the engineer needed no reference disclosure — the leak did the work');
ok(/`mv`, not `rm`/.test(bench), 'cleanup avoids rm, which prompts and blocks background runs');
// /tmp is the right place to BUILD the control and the wrong place to LEAVE it: it is outside the run
// tree (so no path check or tree sweep covers it) and it is a directory every agent greps. A wave that
// renamed its finished control workspace inside /tmp was found by the next wave's analyze agent.
ok(/out of `\/tmp` too, not renamed inside it/.test(bench),
   'retirement moves the control workspace out of /tmp, not to another name within it');
ok(/FOUND EXISTING IMPLEMENTATION/.test(bench) && /do not spend that twice/.test(bench),
   'the incident is recorded: doctrine caught it once, which is not a reason to re-create it');
ok(/outside the project root and\s*\n?\s*outside `\/tmp`/.test(bench),
   'the retirement target is specified, not left as "aside"');

// --- 7. the gate that actually RUNS the scanners --------------------------
// Sections 5 and 6 assert the scanners exist and the doctrine is written down. Both were true of the
// startup path check too, and it still missed every real leak. What closed the loop is that the
// scanners are now executed by the run itself, after Setup, with the power to abort.
console.log('\n# the post-Setup containment gate');
ok(/CONTAINMENT GATE FAILED/.test(wf), 'the gate has a distinct, greppable failure name');
ok(/throw new Error\(\s*\n?\s*`CONTAINMENT GATE FAILED/.test(wf),
   'a leak found after Setup ABORTS — the startup check aborts, so this must too');
const iGate = wf.indexOf('CONTAINMENT GATE FAILED');
ok(iGate > wf.indexOf("log(`Setup done"),
   'the gate runs AFTER Setup — before Setup the eval dir does not exist and the sweep passes vacuously');
ok(iGate > 0 && iGate < wf.indexOf('while (dispatched < BUDGET'),
   'the gate runs BEFORE the optimize loop spends budget');
ok(/reference_leak_sweep\.sh --tree/.test(wf) && /skill_address_scan\.sh --skills-dir/.test(wf),
   'the gate invokes both scanners, not just the one that existed first');
ok(/'clean', 'leak', 'skipped', 'unknown'/.test(wf),
   'the verdict keeps "could not run" distinct from "found nothing"');
ok(/STRICT_AUTONOMY \|\| verdict === 'leak'/.test(wf) &&
   /strict\/capability evidence requires an explicit clean/.test(wf),
   'a dead/skipped strict gate aborts rather than failing open');
// Wrap-tolerant: the sentence spans two `+`-joined literals (same reason as the section-4 note).
ok(/do not\s*`? ?\+?\s*`?\s*read any file they flag/.test(wf),
   'the gate agent is told not to read what it finds — the checker must not become the next leak');

// --- 8. the address scanner tests RESOLVABILITY, not citation -------------
// A hex-token blacklist would fire on every honest Sources line and get switched off inside a week.
console.log('\n# the skill-card address scanner');
const addr = read('scripts/skill_address_scan.sh');
ok(/cat-file -t/.test(addr),
   'addresses are tested by resolving them in a real repo, not by matching hex');
ok(/citation, not a door|resolves nowhere passes/.test(addr),
   'the source states why citation alone is not a finding');
ok(/AITER_JIT_DIR/.test(addr),
   'the repo GEAK_TASK points every engineer at is included in the search set');
ok(/refs\/remotes\//.test(addr),
   'remote-tracking refs are checked too — deleting the local branch left refs/remotes/yz/mega live');
ok(/PASS by absence, not by checking/.test(addr),
   'zero reachable repos is reported as an untested pass, not a clean one');
// Matched on the tail only: the sentence is split across two `echo` lines in the script.
ok(/be address-free and still be a full specification/.test(addr),
   'the clean path admits the scanner does not measure how much the card gives away');
// The bug this caught in its own first hour: `for pat in $INCLUDES` globs *.md against the cwd, so
// the include list silently became the cwd's markdown files and the scan reported clean by scanning
// nothing. A checker that fails open is worse than no checker, so the fix is pinned by a test.
ok(/read -r -a inc_pats/.test(addr) && /would GLOB/.test(addr),
   'the include list is split without globbing, and the reason is recorded');
ok(/resolve each DISTINCT token once|one git call per unique address/.test(addr),
   'tokens are deduped before resolution — the naive loop timed out at two minutes on one tree');
// 7-digit cycle counts in ATT dumps resolve as abbreviated commits in a 555 MB repo. Dropping
// all-digit tokens costs ~1.5% of real SHAs; the script has to say so rather than filter quietly.
ok(/tok ~ \/\^\[0-9\]\+\$\/\) next/.test(addr) && /1\.5% of the time/.test(addr),
   'all-digit tokens are dropped, with the miss rate stated');
ok(/not a tree audit/.test(addr),
   'the header scopes the scanner to injected knowledge and points elsewhere for tree-level leaks');

// --- 8b. the same rule applied to filesystem paths, tested by RUNNING it --
// A git sha was never the only kind of address. `/outside/hidden-reference/reference.patch` in a card
// is the same door with a shorter walk, and pass 1 cannot see it. These assertions execute the
// scanner against a synthetic card, because the failure being guarded is a match that silently
// stopped happening — which is exactly what a source-grep assertion cannot distinguish from a pass.
console.log('\n# the scanner resolves filesystem paths, not just git addresses');
const { execFileSync } = require('child_process');
const SCAN = path.join(ROOT, 'scripts', 'skill_address_scan.sh');
const box = fs.mkdtempSync(path.join(os.tmpdir(), 'gk_addr_'));
process.on('exit', () => { try { fs.rmSync(box, { recursive: true, force: true }); } catch { /* best effort */ } });
const skills = path.join(box, 'skills');
fs.mkdirSync(skills, { recursive: true });
fs.mkdirSync(path.join(box, 'secret'), { recursive: true });
const answer = path.join(box, 'secret', 'reference.patch');
fs.writeFileSync(answer, 'the answer\n');
const card = (body) => { fs.writeFileSync(path.join(skills, 'card.md'), body); };
const scan = (extra = []) => {
  try { return { code: 0, out: execFileSync('bash', [SCAN, '--skills-dir', skills, '--scan-root', box, ...extra], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) }; }
  catch (e) { return { code: e.status == null ? -1 : e.status, out: `${e.stdout || ''}${e.stderr || ''}` }; }
};

card(`Reference implementation: ${answer}\n`);
let s = scan();
ok(s.code === 1 && /RESOLVABLE PATH/.test(s.out),
   'a card naming a path that opens on this machine FAILS, the same way a resolvable sha does');
ok(new RegExp(`at\\s+${skills.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/card\\.md:1`).test(s.out),
   'the finding names the card and line, so it can be redacted without opening the target');

card('Runtime: /opt/rocm/lib/libhsa.so\nEdit aiter/ops/flydsl/mega_moe_stage2.py\nSee /no/such/tree/here.md\n');
s = scan();
ok(s.code === 0, `a card with only system, relative and dead paths is clean (exit ${s.code})`);
ok(!/\/opt\/rocm/.test(s.out), 'a system prefix is not a finding — every box has it and it carries nothing');
ok(!/mega_moe_stage2/.test(s.out),
   'a RELATIVE path is not a finding: it resolves against the engineer\'s own workspace, which is the tree the task asks them to edit');
ok(!/no\/such\/tree/.test(s.out),
   'a path that names another machine\'s layout is a citation, not a door — same rule as an unresolvable sha');

card(`Reference implementation: ${answer}\n`);
ok(scan(['--allow-prefix', path.join(box, 'secret')]).code === 0,
   '--allow-prefix exempts a tree the cards may legitimately name');

// The scanner is where an assumption about containment gets tested instead of trusted: the plan was
// to chmod the quarantine roots unreadable, and under uid 0 that does nothing at all.
ok(/permission bits are not containment/.test(addr) && /move it off this machine or delete it/.test(addr),
   'running as root, the scan says chmod-based quarantine is inert rather than reporting it as contained');
// Third time this subsystem would have failed open. The first two (a glob that ate the include list,
// a grep flag pair that matched nothing) both reported clean by scanning nothing.
ok(/did not fire on its own probe/.test(addr),
   'the path extractor proves it still matches before any clean result is believed');
ok(/Not an exit: the filesystem pass below needs no repository/.test(addr) &&
   !/could resolve against[\s\S]{0,400}\bexit 0\b/.test(addr),
   'zero reachable repos no longer exits early — the git-only version would have skipped the path pass entirely');

// --- 9. the ref pass is RUN, not just grepped for ------------------------
//
// Every assertion above this line reads source text, and source text can describe a scanner that
// does nothing. The failure being guarded here is precisely that shape: the sweep DID run, DID walk
// the frozen baseline, and DID print "clean" while a branch in that same repo carried the whole
// answer. So build the offending shape — clean worktree, marker on an unchecked-out branch — and
// require a non-zero exit. This is the one test in this file that would have caught the incident.
console.log('\n# the ref pass, executed against a real branch-only leak');
{
  const { execFileSync } = require('child_process');
  const marker = fs.readFileSync(MARKER_PATH, 'utf8')
    .split('\n').map(s => s.trim()).filter(s => s && !s.startsWith('#'))[0];
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'refleak-'));
  const repo = path.join(tmp, 'repo');
  const sh = (cmd) => execFileSync('bash', ['-c', cmd], { cwd: repo, stdio: 'pipe' });
  try {
    fs.mkdirSync(repo);
    sh('git init -q . && git config user.email t@t && git config user.name t');
    fs.writeFileSync(path.join(repo, 'k.py'), 'def clean(): pass\n');
    sh('git add -A && git commit -qm base && git checkout -q -b leakbranch');
    fs.writeFileSync(path.join(repo, 'k.py'), `def f():\n    x = ${marker}\n`);
    sh('git add -A && git commit -qm ref && git checkout -q -');
    ok(!fs.readFileSync(path.join(repo, 'k.py'), 'utf8').includes(marker),
       'precondition: the working tree is clean, so a file-only sweep has nothing to find');
    let code = 0;
    try {
      execFileSync('bash', [path.join(ROOT, 'scripts/reference_leak_sweep.sh'), '--tree', tmp],
                   { stdio: 'pipe', env: { ...process.env, MARKER_FILE: MARKER_PATH } });
    } catch (e) { code = e.status; }
    ok(code === 1,
       `a marker reachable only via \`git checkout\` is a LEAK (exit ${code}, want 1)`);
  } catch (e) {
    ok(false, `ref-pass execution test could not run: ${e.message}`);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

console.log(
  failures === 0
    ? '\nPASS: a reference inside the run tree stops the run; a relocated one does not.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
