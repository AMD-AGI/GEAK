#!/usr/bin/env node
// Every private workspace must be a git repo, and every patch must be produced by diffing it.
//
// WHY. Five places in this workflow build a private workspace with the same tar-pipe, and all five
// exclude `.git` deliberately: the source history may carry an optimized or reference-derived tree,
// and an engineer who can `git show` it is one command away from copying the answer. That exclusion
// is correct and stays.
//
// What was missing is the other half. Director's two copies (workspace, validation_workspace) run
// `git init` + one commit afterwards, so HEAD is exactly the baseline and `git diff` regenerates the
// patch. The engineer, integrator and verify copies did not. In those, `cd $KERNEL_PATH && git diff`
// either fails or — worse — silently reports against whatever ancestor repo sits above the
// workspace. A wave's TechLead recorded the consequence verbatim: "Diffs there are hand-maintained
// and must be re-verified with `git apply --check`." A hand-maintained diff is a second source of
// truth for a change that already has one, and it drifts.
//
// The fix is `git init` in the fresh copy, NOT copying `.git` in. A fresh one-commit repo carries no
// history, so it adds nothing to the reference-containment sweep surface and leaks nothing.
//
// This test reads the shipped recipes as text because they ARE text — the workflow hands these
// snippets to agents to run. There is no function to lift.
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(WF, p), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const SITES = [
  ['kernel_mega.js', 'engineer/deep_engineer round workspace'],
  ['roles/mega/integrator.md', 'integrator merge workspace'],
  ['roles/mega/verify_engineer.md', 'verify workspace'],
  ['roles/mega/director.md', 'director baseline/workspace + validation_workspace'],
];

console.log('\n# 1. .git is still excluded everywhere — the containment half must not regress');
for (const [f, what] of SITES) {
  const s = read(f);
  ok(/--exclude=['"]?\.?\/?\.git['"]?/.test(s) && /--exclude='\*\/\.git'/.test(s),
     `${what}: the tar-copy still excludes .git (source history stays unreadable)`);
}

console.log('\n# 2. ...and every one of them re-creates a repo, so HEAD is the baseline');
for (const [f, what] of SITES) {
  const s = read(f);
  ok(/git init -q/.test(s), `${what}: runs \`git init\` in the fresh copy`);
  ok(/git -c user\.email=\S+ -c user\.name=\S+ commit -q -m/.test(s),
     `${what}: and commits, with an explicit identity (the box may have no global git user)`);
  ok(/GIT_TERMINAL_PROMPT=0/.test(s) && /GIT_PAGER=cat/.test(s),
     `${what}: with pager and prompts disabled — a git that blocks hangs an autonomous agent`);
  ok(/\.gitignore/.test(s),
     `${what}: writes .gitignore, so \`git add -A\` cannot stage build artifacts into a patch`);
}

console.log('\n# 3. patches are staged before diffing, or created files vanish from them');
{
  // `git diff` with no argument shows tracked modifications only. A new .py file an engineer adds is
  // untracked, so it is absent from the patch; the patch then applies cleanly and fails at import,
  // which reads as a flaky candidate rather than a broken patch.
  const producers = [
    ['kernel_mega.js', 'the engineer prompt'],
    ['roles/mega/engineer.md', 'engineer step 5'],
    ['roles/mega/deep_engineer.md', 'deep_engineer step 3'],
    ['roles/mega/integrator.md', 'integrator output'],
  ];
  for (const [f, what] of producers) {
    const s = read(f);
    ok(/git add -A && git diff HEAD/.test(s), `${what}: stages then diffs against HEAD`);
    ok(!/(?<!add -A && )git diff >/.test(s),
       `${what}: no bare \`git diff >\` survives anywhere in the file`);
  }
}

console.log('\n# 4. the reason is written down where the agent reads it, not only here');
{
  const eng = read('roles/mega/engineer.md').replace(/\s+/g, ' ');
  ok(/fresh one-commit git repo/i.test(eng),
     'engineer.md tells the engineer its workspace HEAD is the baseline it started from');
  ok(/omits files you CREATED/i.test(eng),
     'and says what a bare `git diff` loses, so the instruction is followable rather than magic');
  ok(/rather than hand-editing it/i.test(eng),
     'and forbids the hand-maintained diff that this whole defect produced');
}

console.log(failures
  ? `\nFAILED: ${failures} assertion(s).`
  : '\nPASS: every private workspace is a fresh repo, and every patch is diffed out of one.');
process.exit(failures ? 1 : 0);
