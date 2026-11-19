#!/usr/bin/env node
// kernel_lane.js `skill_gates` (opt-in hard gates an expert skill's caller declares): off by default
// and guarded at every use site; args validated; a gate passes only on a declared exit code; a gate
// runner that returns nothing fails closed; placeholders are filled. No GPU, no model.
// Run: node kernel_workflow/scripts/tests/test_skill_gates.js
'use strict';
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.resolve(__dirname, '..', '..', 'kernel_lane.js'), 'utf8');
let failures = 0;
const ok = (c, m) => { if (c) console.log('  ok:', m); else { console.error('  FAIL:', m); failures++; } };

// 1. Parser: extract the IIFE and evaluate it with controlled args.
const parser = src.match(/const SKILL_GATES = \(\(\) => \{[\s\S]*?\n\}\)\(\);/)[0];
const parse = (A) => new Function('A', parser + '\nreturn SKILL_GATES;')(A);
ok(Array.isArray(parse({})) && parse({}).length === 0, 'no skill_gates -> none (default off)');
ok(parse({ skill_gates: '' }).length === 0, "'' -> none");
const g = parse({ skill_gates: JSON.stringify([{ id: 'a', run: 'true' }, { id: 'b', run: 'x', pass: [0, 3], setup: 's' }]) });
ok(g.length === 2 && g[0].pass[0] === 0 && g[1].pass.join() === '0,3' && g[1].setup === 's', 'JSON string parsed, pass defaults to [0]');
let threw = false; try { parse({ skill_gates: [{ id: 'a' }] }); } catch (e) { threw = /skill_gates/.test(e.message); }
ok(threw, 'a gate without run is refused');

// 2. Every use is guarded by SKILL_GATES.length, so the default run never dispatches a gate agent.
const calls = [...src.matchAll(/await skillGateCheck\(/g)].map(m => m.index);
ok(calls.length === 2 && calls.every(i => src.slice(Math.max(0, i - 400), i).includes('SKILL_GATES.length')),
  `both call sites guarded by SKILL_GATES.length (${calls.length} calls)`);
ok(!/gluon/i.test(parser) && !/gluon/i.test(src.match(/async function skillGateCheck[\s\S]*?\n\}/)[0]), 'the mechanism names no skill');

// 3. skillGateCheck: pass only on declared exit codes; null runner fails closed; placeholders filled.
const fnSrc = src.match(/async function skillGateCheck\(id, patch, dir\) \{[\s\S]*?\n\}/)[0];
const make = (gates, reply) => {
  const prompts = [];
  const agentT = async (p) => { prompts.push(p); return reply; };
  const f = new Function('SKILL_GATES', 'agentT', 'log', 'CANONICAL', 'EVAL_DIR', 'GPU_POOL', 'WORKFLOW_DIR', 'obj', 'SKILL_GATE_SCHEMA',
    fnSrc + '\nreturn skillGateCheck;')(gates, agentT, () => {}, '/c', '/e', '6,7', '/wf', (p) => p, {});
  return { f, prompts };
};
(async () => {
  const gates = [{ id: 'edge', run: 'check {CANDIDATE} vs {BASELINE} in {EVAL_DIR} on {GPU_ID}', pass: [0, 3], setup: '' }];
  let m = make(gates, { results: [{ id: 'edge', exit_code: 3 }] });
  let r = await m.f('d0', '/p.diff', '/e/r1/d0/skill_gates');
  ok(r.pass, 'exit 3 passes when declared');
  ok(m.prompts[0].includes('check /e/r1/d0/skill_gates/ws vs /c in /e on 6,7'), 'placeholders filled');
  m = make(gates, { results: [{ id: 'edge', exit_code: 1 }] });
  ok(!(await m.f('d0', '/p', '/d')).pass, 'exit 1 fails');
  m = make(gates, { results: [] });
  ok(!(await m.f('d0', '/p', '/d')).pass, 'a gate with no row fails');
  m = make(gates, null);
  r = await m.f('d0', '/p', '/d');
  ok(!r.pass && /fail closed/.test(r.note), 'a runner that returns nothing fails closed');
  if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
  console.log('\nall skill-gates checks passed');
})();
