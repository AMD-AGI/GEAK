#!/usr/bin/env node
// workflow.js is launched directly, so its argument checks, default paths and every file it names
// must hold on their own: no dispatcher fills them in.
'use strict';

const fs = require('fs');
const path = require('path');
const { PKG, ROOT, WF_DIR, SKILL, launch, ok, done } = require('./harness');

const src = fs.readFileSync(path.join(PKG, 'workflow.js'), 'utf8');

(async () => {
  console.log('\n# 1. argument checks happen before any agent runs');
  for (const [args, re, what] of [
    [{ kernel_path: '' }, /kernel_path is required/, 'kernel_path required'],
    [{ workflow_dir: '' }, /workflow_dir is required/, 'workflow_dir required'],
    [{ entry_mode: 'sideways' }, /entry_mode must be/, 'entry_mode validated'],
    [{ target_arch: 'gfx90a' }, /target_arch must be gfx942 or gfx950/, 'target_arch validated'],
    [{ gpu_ids: '6,x' }, /gpu_ids/, 'gpu_ids validated'],
    [{ gpus_per_job: 2 }, /single-GPU/, 'multi-GPU jobs refused'],
  ]) {
    const x = await launch(args);
    ok(x.calls.length === 0 && re.test(x.error || ''), what, x.error);
  }

  console.log('\n# 2. the detected GPU must be gfx942/gfx950 and match target_arch');
  {
    const x = await launch({}, undefined, { gfx: 'gfx1201' });
    ok(/supports gfx942\/gfx950 only/.test(x.error || '') && x.labels.join() === 'director:setup', 'RDNA refused after Setup', x.error);
    const y = await launch({ target_arch: 'gfx950' }, undefined, { gfx: 'gfx942' });
    ok(/architecture mismatch/.test(y.error || ''), 'target_arch mismatch refused', y.error);
    const setup = y.calls[0].prompt;
    ok(/EXPECTED_GFX: gfx950/.test(setup), 'target_arch is passed to the director as EXPECTED_GFX');
  }

  console.log('\n# 3. defaults resolve from workflow_dir alone');
  {
    const x = await launch({ budget: 1, entry_mode: 'incumbent' }, undefined, { entry: 'incumbent' });
    const eng = x.calls.find(c => /^gluon_engineer:/.test(c.label)).prompt;
    ok(eng.includes(`SKILL: ${SKILL}`) && fs.existsSync(path.join(SKILL, 'skill.md')), 'SKILL defaults to the in-tree pack');
    ok(eng.includes(`KT: ${path.join(WF_DIR, 'scripts', 'kernel_tools')}`), 'KT is the shared kernel_tools dir');
    ok(/GPU_ID: 6,7/.test(eng), 'GPU pool passed through to gpu_lock.sh');
    ok(eng.includes(`bash ${WF_DIR}/scripts/gpu_lock.sh`), 'GPU access is through kernel_workflow/scripts/gpu_lock.sh');
    ok(x.calls.find(c => c.label === 'director:setup').prompt.includes(`EXP_ROOT: ${path.join(ROOT, 'exp')}`), 'exp_root defaults to <repo>/exp');
    for (const c of x.calls) {
      const role = (c.prompt.match(/Then Read (\S+\.md)/) || [])[1];
      if (role) ok(fs.existsSync(role), `${c.label}: role file exists (${path.relative(ROOT, role)})`);
    }
  }

  console.log('\n# 4. every reference the stage machine requires exists in the skill pack');
  {
    const tables = src.match(/const STAGE_REFS = \{[\s\S]*?\n\};\nconst LAYER_REFS = \{[\s\S]*?\n\};/);
    ok(!!tables, 'STAGE_REFS / LAYER_REFS found');
    const rels = [...new Set([...tables[0].matchAll(/'((?:references\/)[^']+\.md|skill\.md)'/g)].map(m => m[1]))];
    ok(rels.length >= 20, `${rels.length} references listed`);
    for (const rel of rels) ok(fs.existsSync(path.join(SKILL, rel)), `exists: ${rel}`);
  }

  console.log('\n# 4b. every workload page the archetype router names exists');
  {
    const block = src.match(/const WORKLOAD_DOCS = \{[\s\S]*?\n\};/);
    const rels = [...new Set([...block[0].matchAll(/'(references\/workloads\/[^']+\.md)'/g)].map(m => m[1]))];
    ok(rels.length >= 6, `${rels.length} workload pages`);
    for (const rel of rels) ok(fs.existsSync(path.join(SKILL, rel)), `exists: ${rel}`);
  }

  console.log('\n# 5. the script fits the workflow tool and kernel/e2e workflows do not refer to it');
  {
    const bytes = fs.statSync(path.join(PKG, 'workflow.js')).size;
    ok(bytes <= 524288 - 4096, `workflow.js is ${bytes} bytes`);
    for (const rel of ['kernel_workflow/kernel_lane.js', 'kernel_workflow/kernel_workflow.js', 'e2e_workflow/e2e_workflow.js', 'interface/run_e2e.py']) {
      const text = fs.readFileSync(path.join(ROOT, rel), 'utf8');
      ok(!/gluon_authoring|use_gluon_skill|gluonSkillBlock/.test(text), `${rel} has no Gluon method injection`);
    }
  }
  done('launch-contract');
})();
