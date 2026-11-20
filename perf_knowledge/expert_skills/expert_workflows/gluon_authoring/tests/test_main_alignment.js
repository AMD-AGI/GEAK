#!/usr/bin/env node
// The search outside the Gluon method is kernel_lane.js's. This reads kernel_lane.js and checks that the
// defaults, the planner's direction schema and the deep_explore rule still agree, so a change on main
// fails here instead of silently drifting the two workflows apart.
'use strict';

const fs = require('fs');
const path = require('path');
const { PKG, WF_DIR, launch, ok, done, planReturn } = require('./harness');

const lane = fs.readFileSync(path.join(WF_DIR, 'kernel_lane.js'), 'utf8');
const wf = fs.readFileSync(path.join(PKG, 'workflow.js'), 'utf8');

// kernel_lane.js writes a default as `A.<key> != null ? A.<key> : <value>`; workflow.js as `num(A.<key>, <value>`.
const laneDefault = (key) => (lane.match(new RegExp(`A\\.${key} != null \\? A\\.${key} : ([\\w.]+)`)) || [])[1];
const wfDefault = (key) => (wf.match(new RegExp(`num\\(A\\.${key}, ([^)]+)\\)`)) || [])[1];
const schemaBlock = (src, name) => (src.match(new RegExp(`const ${name} = obj\\(\\{[\\s\\S]*?\\n\\}, \\[([^\\]]*)\\]\\);`)) || []);
const enumOf = (block) => ((block.match(/specialty: \{ type: 'string', enum: (\[[^\]]*\]|SPECIALTIES) \}/) || [])[1] || '');

(async () => {
  console.log('\n# 1. numeric defaults');
  for (const [key, wfExpect] of [['budget', '6'], ['deep_cost', '2'], ['min_improve', '0.02'], ['no_stop_s', '900'], ['max_forced_replans', '6']]) {
    const l = laneDefault(key);
    ok(l === wfExpect, `kernel_lane.js ${key} default is ${wfExpect}`, l);
    ok(wfDefault(key) === wfExpect, `workflow.js ${key} default is ${wfExpect}`, wfDefault(key));
  }
  ok(laneDefault('max_no_improve') === '2' && /max_no_improve, ENTRY === 'port' \? 6 : 3\)/.test(wf),
    'max_no_improve: kernel_lane.js 2, here 3 on purpose (documented; an A/B passes one value to both arms; a port keeps 6)');
  for (const [key, val] of [['use_learned_kb', "'true'"], ['kb_dir_cap', '1'], ['kb_cold_direction', "'true'"]]) {
    const re = (src) => new RegExp(`A\\.${key} != null \\? A\\.${key} : ${val.replace(/'/g, "\\'")}`).test(src) ||
      new RegExp(`num\\(A\\.${key}, ${val}\\)`).test(src);
    ok(re(lane) && re(wf), `${key} defaults to ${val} in both`);
  }
  ok(laneDefault('progress_delta') === 'MIN_IMPROVE' && /progress_delta, ENTRY === 'port' \? -0\.05 : MIN_IMPROVE\)/.test(wf),
    'progress_delta: MIN_IMPROVE as in kernel_lane.js (a port gives ground)');

  console.log('\n# 2. the planner, memory and integrate schemas');
  const [lanePlan, laneReq] = schemaBlock(lane, 'PLAN_SCHEMA');
  const [wfPlan] = schemaBlock(wf, 'PLAN_SCHEMA');
  ok(!!lanePlan && !!wfPlan, 'PLAN_SCHEMA found in both');
  const laneEnum = enumOf(lanePlan || '');
  const wfEnum = (wf.match(/const SPECIALTIES = (\[[^\]]*\]);/) || [])[1];
  ok(laneEnum && laneEnum === wfEnum, 'the same specialties', [laneEnum, wfEnum]);
  for (const f of ['id', 'title', 'specialty', 'focus_files', 'expected_speedup', 'prompt', 'kk_refs', 'learned_refs']) {
    ok(new RegExp(`\\b${f}: `).test(lanePlan || '') && new RegExp(`\\b${f}: `).test(wfPlan || ''), `direction field ${f} in both`);
  }
  const itemReq = (src) => ((src || '').match(/\}, \['id', 'title', 'specialty', 'prompt'\]\)/) || [])[0];
  ok(itemReq(lanePlan) && itemReq(wfPlan), 'the same required direction fields');
  ok(laneReq === wfPlan.match(/\n\}, \[([^\]]*)\]\);$/)[1], 'the same required plan fields', laneReq);
  for (const name of ['MEMORY_SCHEMA', 'INTEGRATE_SCHEMA']) {
    const l = schemaBlock(lane, name)[1];
    const w = schemaBlock(wf, name)[1];
    ok(l && l === w, `${name}: the same required fields (${l})`, w);
  }

  console.log('\n# 3. the rules the script enforces');
  ok(/if \(deepDir\) directions = \[deepDir\];/.test(lane) && /if \(deep\) ds = \[deep\];/.test(wf), 'deep_explore runs alone in both');
  ok(/plan\.directions\.slice\(0, remaining\)/.test(lane) && /plan\.directions\.filter\([^)]*\)\.slice\(0, cap\)/.test(wf), 'the plan is cut to the remaining budget in both');
  ok(/verified\.length >= 2/.test(lane) && /integrable\.length >= 2/.test(wf), 'integration needs >= 2 candidates in both');
  ok(/if \(madeProgress \|\| installed\) \{ noImprove = 0; \} else \{ noImprove\+\+; \}/.test(lane) &&
    /if \(madeProgress \|\| improved\) noImprove = 0; else noImprove\+\+;/.test(wf), 'the same stall rule');
  for (const phase of ['plan_round', 'update_memory']) {
    ok(fs.readFileSync(path.join(WF_DIR, 'roles', 'tech_lead.md'), 'utf8').includes(`## PHASE=${phase}`), `tech_lead.md still has PHASE=${phase}`);
  }
  ok(fs.existsSync(path.join(WF_DIR, 'roles', 'integrator.md')), 'integrator.md exists');

  console.log('\n# 3b. the knowledge loop');
  for (const [key, val] of [['warm_start', "'on'"], ['update_experience', "'on'"], ['box_quiet', "'true'"], ['held_out', "'false'"]]) {
    const re = new RegExp(`A\\.${key} != null \\? A\\.${key} : ${val.replace(/'/g, "\\'")}`);
    ok(re.test(lane) && re.test(wf), `${key} defaults to ${val} in both`);
  }
  for (const frag of ["A.kb_mode || 'store'", "A.kb_remote || 'auto'", "parseFloat(A.warm_start_min_speedup)) ? parseFloat(A.warm_start_min_speedup) : 1.05",
    "'/kb_store_local'", '/scripts/experience_store.py']) {
    const norm = (t) => t.replace(/\s+/g, ' ');
    ok(norm(lane).includes(norm(frag)) && norm(wf).includes(norm(frag)), `same: ${frag}`);
  }
  const prelude = (src) => (src.match(/const KB_ENV_PRELUDE =\n([\s\S]*?);\n/) || [])[1];
  ok(prelude(lane) && prelude(lane) === prelude(wf), 'the KB credential prelude is kernel_lane.js\'s, verbatim');
  for (const cmd of ['sync-local --root', 'resolve-remote --plane local --store', 'resolve-remote --plane remote', ' remap --patch',
    ' attest ${planeFlags}', 'write-remote --plane both --store', 'scripts/kb.py --kb-dir ${LEARNED_DIR} cite']) {
    ok(lane.includes(cmd) && wf.includes(cmd), `both run: ${cmd.trim()}`);
  }
  ok(schemaBlock(lane, 'ANALYZE_SCHEMA')[1] === schemaBlock(wf, 'ANALYZE_SCHEMA')[1], 'ANALYZE_SCHEMA: the same required fields');
  for (const role of ['tech_lead', 'update_experience', 'verify_engineer']) {
    ok(new RegExp(`geakRole\\('${role}'\\)`).test(wf) && fs.existsSync(path.join(WF_DIR, 'roles', `${role}.md`)), `${role} is kernel_workflow's role file`);
  }
  ok(fs.readFileSync(path.join(WF_DIR, 'roles', 'tech_lead.md'), 'utf8').includes('## PHASE=analyze'), 'tech_lead.md still has PHASE=analyze');
  ok(fs.existsSync(path.join(WF_DIR, 'scripts', 'reclaim_eval_artifacts.sh')) && wf.includes('reclaim_eval_artifacts.sh'), 'storage reclaim script shared');

  console.log('\n# 4. behaviour: with no budget given, 6 directions as in kernel_lane.js');
  {
    const x = await launch({ entry_mode: 'incumbent', max_no_improve: 99 }, (label, prompt) =>
      /^tech_lead:plan/.test(label) ? planReturn(prompt, [{ specialty: 'memory', layer: '2' }, { specialty: 'compute', layer: '7' }]) : undefined,
    { entry: 'incumbent' });
    ok(x.result.budget === 6 && x.result.budget_used === 6 && x.result.rounds_used === 3, '3 rounds of 2 directions', x.result);
  }
  done('main-alignment');
})();
