#!/usr/bin/env node
// Regression guard for AgentX window lanes (no GPU, no model).
//
// A lane batch with nothing queued behind it is the run's last search work: its lanes run rounds
// until the search window less one A/B per batch member and one round, with a direction budget and
// plateau cap that do not bind first and TechLead stops refused until one round is left. Any other
// lane keeps the ordinary deadline (one A/B and one round before the final reserve, stops allowed).
//
// Run:  node e2e_workflow/scripts/test_agentx_window_lanes.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const start = src.indexOf('async function laneDeadlineArgs(');
const end = src.indexOf('\n}\n', start);
ok(start !== -1 && end > start, 'laneDeadlineArgs located in e2e_workflow.js');
const fnSrc = src.slice(start, end + 3);
ok(/\bfastBoundedWorkflow\(ref, wfArgs, label, windowAbs = 0\)/.test(src)
  && /laneDeadlineArgs\(label, 'HeadKernel', windowAbs\)/.test(src)
  && /\breserveBoundedWorkflow\(ref, wfArgs, label, windowAbs = 0\)/.test(src)
  && /laneDeadlineArgs\(label, 'Milestone', windowAbs\)/.test(src),
  'both lane wrappers hand their windowAbs to laneDeadlineArgs');

const H = 3600, MIN = 60000, NOW = 1700000000;
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
function make({ windowH, guards = true, clock = true }) {
  const logs = [];
  const f = new AsyncFunction('env', `
    const { agentxGuardsOn, searchWindowMs, agentxAbMs, agentxLaneTailMs, AGENTX_LANE_ROUND_S,
            epochNow, KERNEL_BUDGET, log } = env;
    ${fnSrc}
    return laneDeadlineArgs;`);
  const abMs = (legs) => legs * 30 * MIN;           // a 30min leg
  return f({
    agentxGuardsOn: () => guards, searchWindowMs: () => windowH * H * 1000,
    agentxAbMs: abMs, agentxLaneTailMs: () => abMs(2) + H * 1000, AGENTX_LANE_ROUND_S: H,
    epochNow: async () => (clock ? NOW : null), KERNEL_BUDGET: 6, log: (m) => logs.push(m),
  }).then((fn) => ({ fn, logs }));
}

(async () => {
  let { fn, logs } = await make({ windowH: 10 });
  let r = await fn('h0:triton', 'HeadKernel', 0);
  ok(r.deadline_epoch === NOW + 8 * H && r.no_stop_s === 8 * H + 86400 && !('budget' in r) && !('max_no_improve' in r),
    'an ordinary lane stops one A/B and one round before the reserve and may stop early');

  r = await fn('h0:triton', 'HeadKernel', 2);
  ok(r.deadline_epoch === NOW + 7 * H, 'a window lane owing 2 A/Bs runs until the window less 2 A/Bs and one round');
  ok(r.no_stop_s === H, 'its TechLead stops are refused until one round is left');
  ok(r.budget === 7 * 6 && r.max_no_improve === 7, 'its direction budget and plateau cap are sized to the rounds that fit');
  ok(logs.some((m) => /window lane, rounds until ~420min \(2 A\/B\(s\)/.test(m)), 'the window lane is logged');

  ({ fn } = await make({ windowH: 4.5 }));
  r = await fn('m1', 'Milestone', 3);
  ok(r.deadline_epoch === NOW + 2.5 * H && !('budget' in r), 'under two rounds of window, a window lane is an ordinary lane');

  ({ fn } = await make({ windowH: 10, guards: false }));
  ok(Object.keys(await fn('h0:triton', 'HeadKernel', 2)).length === 0, 'off AgentX or without a time budget, nothing is set');
  ({ fn } = await make({ windowH: 10, clock: false }));
  ok(Object.keys(await fn('h0:triton', 'HeadKernel', 2)).length === 0, 'an unreadable clock sets no deadline');

  const mile = src.indexOf('const mileWindowAbs = ');
  ok(mile !== -1 && src.slice(mile, src.indexOf('\n', mile)).includes('KERNEL_EXTRACT_ON && !kernelQueue.length ? cands.length : 0')
    && src.includes("String(c.short_name || ext.op_kind || 'milestone'), mileWindowAbs)"),
  'a milestone batch that empties the prepared queue runs window lanes owing one A/B per candidate');

  console.log(failures === 0 ? '\nPASS: the last lanes spend the window, every other lane is unchanged.' : `\nFAILED: ${failures} assertion(s).`);
  process.exit(failures === 0 ? 0 : 1);
})();
