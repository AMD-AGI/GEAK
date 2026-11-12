#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const fs = require('fs');
const assert = require('assert');
const path = require('path');
const geakRoot = path.resolve(__dirname, '../..');
const {run} = require('./kernel_lane_test_harness');
module.exports = {run};

if (require.main === module) (async () => {
  const marker = '## Triton/Gluon method (automatic language reference)';
  for (const gfx of ['gfx942', 'gfx950', 'gfx1201', 'gfx1151']) {
    for (const language of ['triton', 'gluon', 'hip', 'flydsl']) {
      for (const useGluon of [undefined, 'false', 'auto']) {
        const {prompts} = await run(gfx, '', {language, useGluon});
        const expected = ['gfx942','gfx950'].includes(gfx) && ['triton','gluon'].includes(language) && useGluon === 'auto';
        for (const label of ['tech_lead:analyze', 'profile_engineer:baseline']) {
          const prompt = prompts.find(x => x.label === label).prompt;
          assert.strictEqual(prompt.includes(marker), expected, `${gfx}/${language}/${useGluon}/${label}`);
          if (expected) {
            assert(prompt.includes('draft means unvalidated'));
            assert(prompt.includes('kernel-roofline'));
            assert(prompt.includes(language === 'triton' ? 'Keep the requested Triton backend' : 'incumbent entry'));
          }
        }
        assert(!prompts.find(x => x.label === 'benchmark_engineer').prompt.includes(marker));
      }
    }
  }
  const incumbent = await run('gfx950', '', {sourceLanguage:'gluon', useGluon:'auto'});
  assert(incumbent.prompts.find(x => x.label === 'profile_engineer:baseline').prompt.includes('incumbent entry'));
  const deep = await run('gfx950', '', {language:'gluon', useGluon:'auto', budget:1, planDirections:[{
    id:'gluon', specialty:'deep_explore', mode:'deep_explore', hypothesis:'explicit pipeline',
    out_dir:'/tmp/eval/deep', gpu_id:'0'
  }]});
  // The exact direction schema is owned by the lane; ensure any dispatched worker carries the method.
  const workers = deep.prompts.filter(x => /^(deep|eng) /.test(x.label));
  assert(workers.length > 0, 'a real Optimize worker must be dispatched');
  assert(workers.every(x => x.prompt.includes(marker)));
  // Invoke the actual e2e central lane-argument builder with controlled dependencies.
  const e2e = fs.readFileSync(path.join(geakRoot, 'e2e_workflow/e2e_workflow.js'), 'utf8');
  const laneArgsSource = e2e.match(/const laneArgs = \(wfArgs\) => \{[\s\S]*?^\};/m)[0];
  for (const setting of ['auto', 'false']) {
    const deps = {LANE_USE_LEARNED_KB:'false', USE_GLUON_SKILL:setting, A:{}, RDNA4_ISOLATE:false,
      EXPECTED_GFX:'', EXPECTED_TARGET:'', EXPECTED_DEVICE_NAME:'', EXPECTED_PHYSICAL_CU_COUNT:0,
      ablLaneArgs:()=>({})};
    const laneArgs = new Function(...Object.keys(deps), laneArgsSource+'; return laneArgs;')(...Object.values(deps));
    assert.strictEqual(laneArgs({target_language:'gluon'}).use_gluon_skill, setting);
  }
  console.log('PASS: actual lane prompts route Triton/Gluon guidance by target/GPU, default off, preserve draft and opt-in, including Optimize.');
})().catch(e => { console.error(e); process.exit(1); });
