#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_mega.js'), 'utf8');
const role = fs.readFileSync(
  path.resolve(__dirname, '..', 'roles', 'mega', 'mega_search_lead.md'), 'utf8',
);
const replay = src.match(
  /\/\/ <<REPLAY:analyze_resume_fallback>>([\s\S]*?)\/\/ <<\/REPLAY:analyze_resume_fallback>>/,
);
if (!replay) throw new Error('missing analyze replay region');
// eslint-disable-next-line no-new-func
const { megaPlanIRVerdict } = new Function(
  `${replay[1]}\nreturn { megaPlanIRVerdict };`,
)();

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const base = {
  plan_version: 'mega-plan-v2',
  target: {
    launch_count: 2,
    required_regions: ['producer'],
    required_queues: ['work'],
  },
  work_domains: [{ id: 'items' }],
  regions: [{ id: 'producer', work_domain: 'items' }],
  buffers: [{ id: 'payload', producers: ['producer'], consumers: [] }],
  counters: [{ id: 'ready', producers: ['producer'], consumers: ['producer'] }],
  queues: [{ id: 'work', work_domain: 'items' }],
  events: [{
    id: 'ready_event', producer: 'producer', consumer: 'producer', counter: 'ready',
  }],
  resources: {
    arch: 'gfx950',
    workgroup: { wave_size: 64, thread_count: 256 },
    local_memory: { total_bytes: 1, limit_bytes: 2 },
  },
  schedule: {
    primary_loop: {
      kind: 'unified_g1_g2_then_combine',
      carried_state: ['g2_next_unit', 'g2_units_remaining'],
      queue_selection: 'continuation_then_gated_g2_then_g1_then_blocking_g2',
    },
  },
  abi: { entry_point: 'kernel', arguments: [{ id: 'payload' }] },
  compiler_constraints: [{ id: 'shape' }],
  evidence_requirements: { accuracy: 'required' },
  known_unknowns: [],
};

console.log('\n# truthful queue selection remains backward compatible');
const carried = [...base.schedule.primary_loop.carried_state];
ok(megaPlanIRVerdict(base).pass,
  'a truthful scalar queue_selection with the exact two carried scalars passes');
ok(JSON.stringify(base.schedule.primary_loop.carried_state) === JSON.stringify(carried),
  'validation does not rename or expand carried state');
const legacy = JSON.parse(JSON.stringify(base));
delete legacy.schedule.primary_loop.queue_selection;
legacy.schedule.primary_loop.queue_priority = ['work'];
const legacyBytes = JSON.stringify(legacy);
ok(megaPlanIRVerdict(legacy).pass,
  'legacy queue_priority plans continue to pass');
ok(JSON.stringify(legacy) === legacyBytes,
  'legacy queue_priority plans are not renamed or mutated');
legacy.schedule.primary_loop.queue_priority = [];
ok(megaPlanIRVerdict(legacy).pass,
  'legacy empty queue_priority arrays retain their prior schema behavior');
const missing = JSON.parse(JSON.stringify(base));
delete missing.schedule.primary_loop.queue_selection;
ok(!megaPlanIRVerdict(missing).pass,
  'omitting both queue selectors fails closed');
ok(/queue_selection: \{ type: 'string' \}/.test(src) &&
   !/\}, \['kind', 'carried_state', 'queue_priority', 'progress_invariants'\]\)/.test(src),
  'StructuredOutput declares queue_selection without requiring invented priority');
ok(/preserve it and its `carried_state`[\s\S]*do not invent a global `queue_priority`/.test(role),
  'Analyze is explicitly told to preserve truthful extension vocabulary');

console.log(failures
  ? `\nFAIL: ${failures} queue-selection assertion(s)`
  : '\nPASS: queue_selection is optional and legacy-compatible.');
process.exit(failures ? 1 : 0);
