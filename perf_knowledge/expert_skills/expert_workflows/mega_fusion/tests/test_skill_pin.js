#!/usr/bin/env node
// expert_skill_id alone pins a MegaMoE skill: the run reads the skill's status and digests once
// (skill:pin, before Setup) and then behaves exactly like a run that passed them. A run that passes
// them, or does not name a skill, never calls skill:pin.
'use strict';

const fs = require('fs');
const path = require('path');

const PKG = path.resolve(__dirname, '..');
const ROOT = path.resolve(PKG, '..', '..', '..', '..');
let failures = 0;
const ok = (cond, msg, got) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg, got === undefined ? '' : `(got ${got})`); failures++; }
};

const src = fs.readFileSync(path.join(PKG, 'workflow.js'), 'utf8')
  .replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

const PIN = {
  validation_status: 'experimental', bundle_sha256: 'a'.repeat(64),
  planner_extension_sha256: 'b'.repeat(64), contract_sha256: 'c'.repeat(64),
};

// Runs workflow.js; skill:pin gets `pin`, every other agent answers null (so Setup stops the run).
async function launch(args, pin = PIN) {
  const calls = [];
  const agent = async (prompt, opts = {}) => {
    calls.push({ label: opts.label, prompt });
    return opts.label === 'skill:pin' ? pin : null;
  };
  const parallel = (ts) => Promise.all(ts.map((t) => Promise.resolve().then(t).catch(() => null)));
  const fn = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', src);
  let error = null;
  try {
    await fn(agent, parallel, async () => [], () => {}, () => {}, async () => null,
      { agent_timeout_ms: 0, ...args }, {});
  } catch (e) { error = String(e && e.message || e); }
  return { calls, labels: calls.map((c) => c.label), error };
}

(async () => {
  const base = {
    kernel_path: '/tmp/op_task', workflow_dir: path.join(ROOT, 'kernel_workflow'),
    gpu_ids: '0,1', gpus_per_job: 2, required_accuracy_cases: ['128'],
  };
  // The fully pinned run spells out the skill-validation gates an id-only run gets by default.
  const gates = { require_graph_capture: 'true', direct_graph_accuracy: 'true',
    require_hardware_activation: 'true', require_artifact_distinct: 'true', require_attribution: 'true',
    allow_partial_fusion: 'false' };
  const pinned = { ...base, ...gates, use_expert_skills: 'true', expert_skill_id: 'megamoe_small_mtpr_fusion',
    expert_skill_validation_status: 'experimental', expert_skill_usage: 'candidate_validation',
    expert_skill_bundle_sha256: PIN.bundle_sha256,
    expert_skill_planner_extension_sha256: PIN.planner_extension_sha256,
    expert_skill_contract_sha256: PIN.contract_sha256 };

  console.log('\n# 1. expert_skill_id alone: skill:pin first, then the same Setup as a fully pinned run');
  {
    const idOnly = await launch({ ...base, expert_skill_id: 'megamoe_small_mtpr_fusion' });
    const full = await launch(pinned);
    ok(idOnly.labels[0] === 'skill:pin', 'first agent is skill:pin', idOnly.labels[0]);
    ok(/--emit-bundle/.test(idOnly.calls[0].prompt) &&
      idOnly.calls[0].prompt.includes('_contribute/validate_skill.py megamoe_small_mtpr_fusion'),
      'skill:pin runs validate_skill.py <id> --emit-bundle');
    ok(idOnly.labels[1] === 'director:setup', 'Setup follows', idOnly.labels[1]);
    ok(JSON.stringify(idOnly.calls.slice(1)) === JSON.stringify(full.calls),
      'every later prompt equals the fully pinned run');
    ok(idOnly.error === full.error, 'same outcome as the fully pinned run', idOnly.error);
  }

  console.log('\n# 2. a fully pinned run and a generic run never call skill:pin');
  {
    const full = await launch(pinned);
    ok(!full.labels.includes('skill:pin'), 'fully pinned: no skill:pin', full.labels.join(','));
    const generic = await launch(base);
    ok(!generic.labels.includes('skill:pin'), 'generic: no skill:pin', generic.labels.join(','));
    const off = await launch({ ...base, use_expert_skills: 'false', expert_skill_id: 'megamoe_small_mtpr_fusion' });
    ok(!off.labels.includes('skill:pin'), 'use_expert_skills=false wins over expert_skill_id');
    ok(JSON.stringify(off.calls) === JSON.stringify(generic.calls), 'and runs exactly like generic');
  }

  console.log('\n# 3. a skill that cannot be pinned stops before Setup');
  {
    const bad = await launch({ ...base, expert_skill_id: 'megamoe_small_mtpr_fusion' }, { error: 'unknown skill' });
    ok(bad.labels.join(',') === 'skill:pin', 'only skill:pin ran', bad.labels.join(','));
    ok(/--emit-bundle failed: unknown skill/.test(bad.error || ''), 'error names the failure', bad.error);
    const draft = await launch({ ...base, expert_skill_id: 'megamoe_small_mtpr_fusion' },
      { ...PIN, validation_status: 'draft' });
    ok(draft.labels.join(',') === 'skill:pin' && /validated or explicitly pinned experimental/.test(draft.error || ''),
      'a draft skill is refused by the existing status check', draft.error);
  }

  console.log('\n# 4. a generic run keeps the old gate defaults');
  {
    const gate = (name) => (src.match(new RegExp(`const ${name} =\\s*([\\s\\S]*?);\\n`)) || [])[1];
    const evalGate = (name, A, SKILL_GATES) =>
      // eslint-disable-next-line no-new-func
      new Function('A', 'SKILL_GATES', 'OP_SPEC', 'gateArg',
        `return ${gate(name)};`)(A, SKILL_GATES, {},
        (v) => String(v != null ? v : SKILL_GATES) === 'true');
    for (const [name, generic, skill] of [
      ['REQUIRE_GRAPH_CAPTURE', false, true], ['DIRECT_GRAPH_ACCURACY', false, true],
      ['REQUIRE_ATTRIBUTION', false, true], ['REQUIRE_ARTIFACT_DISTINCT', false, true],
      ['REQUIRE_HW_ACTIVATION', false, true], ['ALLOW_PARTIAL_FUSION', true, false]]) {
      ok(evalGate(name, {}, false) === generic && evalGate(name, {}, true) === skill,
        `${name}: generic ${generic}, skill run ${skill}`);
    }
    ok(evalGate('DIRECT_GRAPH_ACCURACY', { direct_graph_accuracy: 'false' }, true) === false,
      'an explicit arg still wins on a skill run');
  }

  console.log('\n# 5. expert_skill_usage defaults by run kind');
  {
    const m = src.match(/const EXPERT_SKILL_USAGE = ([\s\S]*?);\n/);
    ok(!!m, 'EXPERT_SKILL_USAGE is defined');
    const usage = (A, PINNED_MEGA_SKILL, MEGA_STRUCTURAL_ONLY) =>
      // eslint-disable-next-line no-new-func
      new Function('A', 'PINNED_MEGA_SKILL', 'MEGA_STRUCTURAL_ONLY', `return ${m[1]};`)(A, PINNED_MEGA_SKILL, MEGA_STRUCTURAL_ONLY);
    ok(usage({}, true, false) === 'candidate_validation', 'pinned on-card run -> candidate_validation');
    ok(usage({}, true, true) === 'authoring', 'pinned structural-only run -> authoring');
    ok(usage({ expert_skill_usage: 'authoring' }, true, true) === 'authoring', 'an explicit value wins');
    ok(usage({}, false, false) === '', 'no pinned skill -> empty');
  }

  if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
  console.log('\nall skill-pin checks passed');
})();
