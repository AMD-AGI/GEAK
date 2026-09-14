#!/usr/bin/env node
// Controlled (agent × model) comparison experiment runner.
//
// Runs the SAME workflow + task + args through every (agent, model) combo, N
// repeats each, holding task/budget/gpu fixed (they live in --args, shared by
// all combos) so the only thing that varies is the axis you sweep. Collects
// per-run metrics (NO token/cost by design) and writes a comparison table.
//
//   node experiment.mjs \
//     --script ../../kernel_workflow/kernel_workflow.js \
//     --args '{"kernel_path":"/abs/knn","workflow_dir":"/abs/kernel_workflow","budget":6}' \
//     --agents claude,qwen,codex --models default --repeats 3 \
//     --out ./exp_compare
//
// Axes:
//   --agents a,b,c   CLI agents to compare (names in registry.agents)
//   --models m,n     models to compare (names in registry.models; "default" = the
//                    agent's own configured model / env)
//   --repeats N      repeats per combo (default 1) — capture LLM nondeterminism
//   --sequential     run combos one at a time (DEFAULT; GPU is shared). Pass
//                    --parallel-combos K to overlap K at once (only safe if the
//                    task does not contend on the same GPU).
//
// Outputs under --out:
//   results.jsonl    one line per run (combo, repeat, speedup, ok, wall, …)
//   summary.md       aggregated comparison table (mean speedup, success rate, …)
//   summary.csv      same, machine-readable

import { spawn } from 'node:child_process';
import { mkdir, writeFile, appendFile, readFile } from 'node:fs/promises';
import { resolve as resolvePath, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const RUN_WORKFLOW = resolvePath(HERE, 'run_workflow.mjs');

// Speedup lives under different keys in kernel vs e2e returns — try in order.
const SPEEDUP_KEYS = [
  'throughput_speedup', 'final_speedup', 'final_weighted', 'final_geomean',
  'speedup', 'director_verified_speedup',
];

function parseArgv(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('--')) out[key] = true;
      else { out[key] = next; i++; }
    } else out._.push(a);
  }
  return out;
}

const list = (s) => String(s || '').split(',').map((x) => x.trim()).filter(Boolean);

function extractSpeedup(result) {
  if (!result || typeof result !== 'object') return null;
  for (const k of SPEEDUP_KEYS) {
    const v = result[k];
    if (typeof v === 'number' && isFinite(v)) return v;
  }
  return null;
}

function runOne({ script, argsJson, agent, model, rep, outDir, timeoutMs }) {
  return new Promise((resolve) => {
    const tag = `${agent}__${model}__r${rep}`;
    const resultFile = join(outDir, `${tag}.result.json`);
    const metricsFile = join(outDir, `${tag}.metrics.json`);
    const cliArgs = [
      RUN_WORKFLOW, script,
      '--args', argsJson,
      '--agent', agent,
      '--result-file', resultFile,
      '--metrics-file', metricsFile,
    ];
    if (model && model !== 'default') cliArgs.push('--model', model);

    const started = Date.now();
    const child = spawn('node', cliArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    let killer = null;
    if (timeoutMs > 0) killer = setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, timeoutMs);
    child.stdout.on('data', (d) => { out += d.toString(); process.stdout.write(`[${tag}] ${d}`); });
    child.stderr.on('data', (d) => { err += d.toString(); });
    child.on('close', async (code) => {
      if (killer) clearTimeout(killer);
      let result = null;
      let metrics = null;
      try { result = JSON.parse(await readFile(resultFile, 'utf8')); } catch {}
      try { metrics = JSON.parse(await readFile(metricsFile, 'utf8')); } catch {}
      const speedup = extractSpeedup(result);
      const rec = {
        agent, model, rep,
        ok: !!(metrics && metrics.ok) && code === 0,
        speedup,
        wall_ms: (metrics && metrics.wall_ms) != null ? metrics.wall_ms : (Date.now() - started),
        agents_spawned: metrics ? metrics.agents_spawned : null,
        schema_failures: metrics ? metrics.schema_failures : null,
        exit_code: code,
        error: metrics ? metrics.error : (err.slice(-300) || null),
      };
      resolve(rec);
    });
  });
}

function mean(xs) { const a = xs.filter((x) => typeof x === 'number' && isFinite(x)); return a.length ? a.reduce((s, x) => s + x, 0) / a.length : null; }
function fmt(x, d = 3) { return (typeof x === 'number' && isFinite(x)) ? x.toFixed(d) : '—'; }

function aggregate(records) {
  const byCombo = new Map();
  for (const r of records) {
    const key = `${r.agent}/${r.model}`;
    if (!byCombo.has(key)) byCombo.set(key, []);
    byCombo.get(key).push(r);
  }
  const rows = [];
  for (const [combo, rs] of byCombo) {
    const oks = rs.filter((r) => r.ok);
    const sp = oks.map((r) => r.speedup);
    rows.push({
      combo,
      runs: rs.length,
      success_rate: rs.length ? oks.length / rs.length : 0,
      speedup_mean: mean(sp),
      speedup_min: sp.length ? Math.min(...sp.filter((x) => x != null)) : null,
      speedup_max: sp.length ? Math.max(...sp.filter((x) => x != null)) : null,
      wall_s_mean: mean(rs.map((r) => r.wall_ms != null ? r.wall_ms / 1000 : null)),
      agents_mean: mean(rs.map((r) => r.agents_spawned)),
      schema_fail_mean: mean(rs.map((r) => r.schema_failures)),
    });
  }
  rows.sort((a, b) => (b.speedup_mean || 0) - (a.speedup_mean || 0));
  return rows;
}

function toMarkdown(rows) {
  const h = '| combo | runs | success | speedup(mean) | speedup(min–max) | wall(s) | agents | schema_fail |';
  const sep = '|---|---|---|---|---|---|---|---|';
  const body = rows.map((r) =>
    `| ${r.combo} | ${r.runs} | ${(r.success_rate * 100).toFixed(0)}% | ${fmt(r.speedup_mean)} | ${fmt(r.speedup_min)}–${fmt(r.speedup_max)} | ${fmt(r.wall_s_mean, 0)} | ${fmt(r.agents_mean, 1)} | ${fmt(r.schema_fail_mean, 1)} |`
  );
  return `# (agent × model) comparison\n\n${h}\n${sep}\n${body.join('\n')}\n\n> speedup 从 workflow 返回值提取(${SPEEDUP_KEYS.join(' / ')});无 token/成本指标(按设计)。\n`;
}

function toCsv(rows) {
  const head = 'combo,runs,success_rate,speedup_mean,speedup_min,speedup_max,wall_s_mean,agents_mean,schema_fail_mean';
  const lines = rows.map((r) => [
    r.combo, r.runs, fmt(r.success_rate, 3), fmt(r.speedup_mean), fmt(r.speedup_min), fmt(r.speedup_max),
    fmt(r.wall_s_mean, 1), fmt(r.agents_mean, 1), fmt(r.schema_fail_mean, 1),
  ].join(','));
  return [head, ...lines].join('\n') + '\n';
}

// bounded-concurrency map
async function pool(items, k, fn) {
  const out = new Array(items.length);
  let idx = 0;
  const workers = Array.from({ length: Math.max(1, k) }, async () => {
    while (true) {
      const i = idx++;
      if (i >= items.length) break;
      out[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return out;
}

async function main() {
  const opts = parseArgv(process.argv.slice(2));
  const script = opts.script && opts.script !== true ? resolvePath(opts.script) : null;
  const argsJson = opts.args && opts.args !== true ? opts.args : '{}';
  const agents = list(opts.agents);
  const models = list(opts.models).length ? list(opts.models) : ['default'];
  const repeats = opts.repeats && opts.repeats !== true ? parseInt(opts.repeats, 10) : 1;
  const outDir = resolvePath(opts.out && opts.out !== true ? opts.out : './exp_compare');
  const parallelCombos = opts['parallel-combos'] && opts['parallel-combos'] !== true
    ? parseInt(opts['parallel-combos'], 10) : 1;
  const timeoutMs = opts['run-timeout-ms'] && opts['run-timeout-ms'] !== true
    ? parseInt(opts['run-timeout-ms'], 10) : 0;

  if (!script || !agents.length) {
    console.error('usage: experiment.mjs --script <path> --args <json> --agents a,b[,c] ' +
      '[--models m,n] [--repeats N] [--out dir] [--parallel-combos K] [--run-timeout-ms N]');
    process.exit(2);
  }

  await mkdir(outDir, { recursive: true });
  const jsonl = join(outDir, 'results.jsonl');
  await writeFile(jsonl, '');

  // Build the full run list (matrix × repeats).
  const runs = [];
  for (const agent of agents)
    for (const model of models)
      for (let rep = 1; rep <= repeats; rep++)
        runs.push({ script, argsJson, agent, model, rep, outDir, timeoutMs });

  console.error(`[experiment] ${agents.length} agents × ${models.length} models × ${repeats} repeats = ${runs.length} runs; parallel-combos=${parallelCombos}`);

  const records = await pool(runs, parallelCombos, async (r) => {
    console.error(`[experiment] START ${r.agent}/${r.model} rep${r.rep}`);
    const rec = await runOne(r);
    await appendFile(jsonl, JSON.stringify(rec) + '\n');
    console.error(`[experiment] DONE  ${r.agent}/${r.model} rep${r.rep}: ok=${rec.ok} speedup=${rec.speedup} wall=${Math.round((rec.wall_ms || 0) / 1000)}s`);
    return rec;
  });

  const rows = aggregate(records);
  await writeFile(join(outDir, 'summary.md'), toMarkdown(rows));
  await writeFile(join(outDir, 'summary.csv'), toCsv(rows));
  console.error(`\n[experiment] wrote ${jsonl}\n[experiment] wrote ${join(outDir, 'summary.md')}\n[experiment] wrote ${join(outDir, 'summary.csv')}\n`);
  process.stdout.write(toMarkdown(rows) + '\n');
}

main().catch((e) => { console.error('EXPERIMENT CRASH:', e && e.stack || e); process.exit(1); });
