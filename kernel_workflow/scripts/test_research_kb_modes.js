#!/usr/bin/env node
// Regression guard for online/offline Researcher-KB routing (no GPU, model, web, or filesystem writes).
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const WF_DIR = path.join(ROOT, 'kernel_workflow');
const BODY = fs.readFileSync(path.join(WF_DIR, 'kernel_lane.js'), 'utf8')
  .replace(/^export const meta/m, 'const meta');

let failures = 0;
const ok = (condition, message, detail) => {
  if (condition) console.log('  ok:', message);
  else {
    console.error('  FAIL:', message, detail ? `-> ${detail}` : '');
    failures++;
  }
};

function build(extraArgs) {
  const trace = { phases: [], labels: [], prompts: new Map(), logs: [] };
  const args = {
    kernel_path: '/tmp/kernel',
    workflow_dir: WF_DIR,
    perf_knowledge_dir: '/tmp/perf_knowledge',
    budget: 1,
    ...extraArgs,
  };
  const agent = async (prompt, options) => {
    const label = (options && options.label) || '';
    trace.labels.push(label);
    trace.prompts.set(label, prompt);
    if (label === 'director:setup') {
      return {
        eval_dir: '/tmp/eval',
        workspace: '/tmp/eval/workspace',
        kernel_name: 'demo',
        baseline_frozen: true,
      };
    }
    if (label === 'tech_lead:analyze') {
      return {
        kernel_type: 'hip',
        kernel_file: 'kernel.hip',
        entry_point: 'run',
        modifiable_files: ['kernel.hip'],
        bottleneck_guess: 'latency',
        roadmap_summary: 'test',
        candidate_directions: [],
        kk_operator: 'reduction',
        kk_language: 'hip',
        kk_refs: [],
      };
    }
    if (label === 'benchmark_engineer') {
      return {
        commandment_path: '/tmp/eval/COMMANDMENT.md',
        baseline_per_case: [{ name: 'case', baseline_ms: 1 }],
        baseline_geomean_ms: 1,
        num_test_cases: 1,
        reliable: true,
      };
    }
    if (label === 'profile_engineer:baseline') {
      return {
        bottleneck: 'latency',
        device: 'MI350X / gfx950',
        dispatch_count: 1,
        top_opportunities: [],
        summary_path: '/tmp/eval/profiling_summary.md',
      };
    }
    if (label === 'researcher:plan') {
      return { facts: { bottleneck_type: 'latency' }, questions: [] };
    }
    if (label === 'researcher:synthesize') {
      return {
        num_questions: 0,
        num_directions: 1,
        brief_path: '/tmp/eval/deep_search_brief.md',
        directions: [],
      };
    }
    if (label === 'research_kb:ingest') {
      return {
        ok: true,
        mode: 'ingest',
        snapshot_id: 'research-online',
        cards_created: 1,
        cards_merged: 0,
        card_ids: ['research-reduction-demo'],
      };
    }
    if (label === 'research_kb:retrieve') {
      return {
        ok: true,
        mode: 'retrieve',
        snapshot_id: 'research-offline',
        brief_path: '/tmp/eval/deep_search_brief.offline.md',
        cards_retrieved: 1,
        card_ids: ['research-reduction-demo'],
      };
    }
    if (label === 'research_kb:validate') {
      return {
        ok: true,
        mode: 'validate',
        snapshot_id: extraArgs.dra_mode === 'offline'
          ? 'research-offline'
          : 'research-online',
        card_ids: ['research-reduction-demo'],
        validation_event_id: 'validation-demo',
        validation_recorded: true,
      };
    }
    if (label.startsWith('tech_lead:plan')) return { stop: true, directions: [] };
    if (label === 'tech_lead:report') {
      return {
        final_speedup_geomean: 1,
        final_speedup_arithmetic: 1,
        rounds: 0,
        budget_used: 0,
        report_path: '/tmp/eval/report.md',
        final_patch: '/tmp/eval/final.patch',
        per_case: [],
      };
    }
    if (label === 'director:validate') {
      return {
        kernel_name: 'demo',
        director_verified_speedup_geomean: 1,
        director_verified_speedup_arithmetic: 1,
        validation_status: 'pass',
        correctness: 'pass',
      };
    }
    return null;
  };
  const globals = {
    args,
    phase: (name) => trace.phases.push(name),
    log: (message) => trace.logs.push(message),
    workflow: async () => null,
    agent,
    parallel: async (thunks) => Promise.all(thunks.map((thunk) => thunk())),
    pipeline: async (items, ...stages) => Promise.all(items.map(async (item, index) => {
      let value = item;
      for (const stage of stages) value = await stage(value, item, index);
      return value;
    })),
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
  };
  const fn = new Function(
    ...Object.keys(globals),
    `return (async () => { ${BODY} })();`,
  );
  return { run: () => fn(...Object.values(globals)), trace };
}

(async () => {
  console.log('\n# historical off mode');
  {
    const { run, trace } = build({});
    const result = await run();
    ok(result.dra_mode === 'off', 'dra_mode defaults to off', result.dra_mode);
    ok(!trace.labels.some((label) => label.startsWith('researcher:')),
      'off mode invokes no Researcher');
    ok(!trace.labels.some((label) => label.startsWith('research_kb:')),
      'off mode invokes no Researcher KB manager');
  }

  console.log('\n# backward-compatible online mode');
  {
    const { run, trace } = build({ dra_enabled: 'true' });
    const result = await run();
    ok(result.dra_mode === 'online', 'dra_enabled=true maps to online', result.dra_mode);
    ok(trace.labels.includes('researcher:plan') && trace.labels.includes('researcher:synthesize'),
      'online mode preserves the existing Researcher phases', trace.labels.join(','));
    ok(trace.labels.includes('research_kb:ingest') && !trace.labels.includes('research_kb:retrieve'),
      'online mode ingests but never re-retrieves its fresh findings', trace.labels.join(','));
    ok(trace.labels.includes('research_kb:validate'),
      'online outcome is recorded separately after Director validation', trace.labels.join(','));
    ok(result.research_brief_path === '/tmp/eval/deep_search_brief.md',
      'online run returns the fresh brief path', result.research_brief_path);
    ok(result.research_kb_snapshot === 'research-online',
      'online run surfaces the written snapshot', result.research_kb_snapshot);
    ok(result.research_kb_validation_event === 'validation-demo',
      'online run surfaces the validation event', result.research_kb_validation_event);
    const planPrompt = trace.prompts.get('tech_lead:plan r1') || '';
    ok(planPrompt.includes('/tmp/eval/deep_search_brief.md'),
      'TechLead receives the fresh online brief directly');
  }

  console.log('\n# offline mode');
  {
    const { run, trace } = build({ dra_mode: 'offline' });
    const result = await run();
    ok(!trace.labels.some((label) => label.startsWith('researcher:')),
      'offline mode invokes no Researcher or web-research phase', trace.labels.join(','));
    ok(trace.labels.includes('research_kb:retrieve') && !trace.labels.includes('research_kb:ingest'),
      'offline mode only retrieves from the KB', trace.labels.join(','));
    ok(trace.labels.includes('research_kb:validate'),
      'offline outcome is recorded against retrieved card IDs', trace.labels.join(','));
    ok(result.research_brief_path === '/tmp/eval/deep_search_brief.offline.md',
      'offline run returns the materialized brief path', result.research_brief_path);
    ok(JSON.stringify(result.research_kb_card_ids) === '["research-reduction-demo"]',
      'offline run exposes retrieved card provenance', JSON.stringify(result.research_kb_card_ids));
    const planPrompt = trace.prompts.get('tech_lead:plan r1') || '';
    ok(planPrompt.includes('/tmp/eval/deep_search_brief.offline.md'),
      'TechLead receives the offline brief through the unchanged handoff');
  }

  console.log('\n# invalid explicit mode');
  {
    const { run } = build({ dra_mode: 'sometimes' });
    let error = null;
    try { await run(); } catch (caught) { error = caught; }
    ok(error && /unknown dra_mode/.test(error.message),
      'unknown dra_mode throws instead of silently selecting a path', error && error.message);
  }

  console.log(failures
    ? `\nFAIL: ${failures} Researcher-KB routing check(s) failed.`
    : '\nPASS: online uses the fresh brief + immediate ingest; offline retrieves without Researcher.');
  process.exit(failures ? 1 : 0);
})();
