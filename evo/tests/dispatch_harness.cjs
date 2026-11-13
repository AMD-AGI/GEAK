'use strict';

const fs = require('node:fs');
const crypto = require('node:crypto');
const path = require('node:path');
const vm = require('node:vm');

const clone = value => value === undefined ? null : JSON.parse(JSON.stringify(value));

function agentReply(label, args, request, prompt) {
  const evalDir = '/fixture/eval';
  const perCase = [{ name: 'fixture', speedup: 1.25 }];
  if (Object.hasOwn(request.replies || {}, label)) return clone(request.replies[label]);
  if (label === 'director:setup') return {
    eval_dir: evalDir, workspace: `${evalDir}/canonical`, baseline_dir: `${evalDir}/original`,
    kernel_name: 'fixture', baseline_frozen: true,
    device_gfx: args.expected_gfx || 'gfx950', device_target: args.expected_target || 'unknown',
    device_name: args.expected_device_name || 'CPU fixture identity',
    physical_cu_count: args.expected_physical_cu_count || 256,
    source_files: ['kernel.py'], gpu_requirement: { kind: 'single' },
  };
  if (label.startsWith('author:')) return { authored: true, correctness: 'pass', baseline_ms: 1 };
  if (label === 'tech_lead:analyze') return {
    kernel_type: 'fixture', roadmap_summary: 'Synthetic deterministic test', modifiable_files: ['kernel.py'],
  };
  if (label === 'benchmark_engineer') return {
    commandment_path: `${evalDir}/COMMANDMENT.md`, baseline_per_case: perCase,
    baseline_geomean_ms: 1, reliable: true,
  };
  if (label.startsWith('profile_engineer:') || label.startsWith('reprofile ')) return {
    bottleneck: 'memory', top_opportunities: ['fixture'], dispatch_count: 1,
  };
  if (label.startsWith('tech_lead:plan ')) return {
    stop: false, directions: ['memory', 'compute'].map((specialty, index) => ({
      id: `fixture_${index}`, title: `Synthetic ${specialty}`, specialty,
      focus_files: ['kernel.py'], prompt: 'Synthetic source candidate',
      ...(request.lineage || {}),
    })),
  };
  if (label.startsWith('eng ') || label.startsWith('deep ')) return {
    status: 'success', measurement_valid: true, speedup_geomean: 1.2,
    speedup_weighted: 1.2, per_case: perCase,
  };
  if (label.startsWith('verify ')) return {
    status: 'verified', correctness: 'pass', verified_geomean: 1.2,
    verified_weighted: 1.2, per_case: perCase, graph_safe: 'pass',
  };
  if (label.startsWith('integrate ')) return {
    attempted: true, conclusion: 'improved', best: {
      geomean: 1.25, weighted: 1.25, per_case: perCase, patch_file: `${evalDir}/integrated.diff`,
    },
  };
  if (label.startsWith('commit ')) return { committed: true, current_best_diff: `${evalDir}/current_best.diff` };
  if (label.startsWith('tech_lead:memory ')) return { insights: ['Synthetic insight'], ledger: [] };
  if (label.startsWith('storage:reclaim ')) return { ok: true, note: 'synthetic; no command executed' };
  if (label === 'tech_lead:report') return {
    final_speedup_geomean: 1.25, final_speedup_weighted: 1.25, rounds: 1,
    report_path: `${evalDir}/report.md`, final_patch: `${evalDir}/final.diff`,
  };
  if (label === 'director:validate') return {
    director_verified_speedup_geomean: 1.25, director_verified_speedup_weighted: 1.25,
    validation_status: 'accepted', correctness: 'pass', timing_basis: 'synthetic',
  };
  if (label === 'warm_start:resolve') return { candidates: [], reason: 'synthetic empty archive' };
  if (label === 'update_experience') return { action: 'skipped', note: 'synthetic' };
  if (label === 'kb:write') return { stored: false, reason: 'synthetic' };
  if (label === 'file_writer:persist_llm_stats') return { written: true, ok: true };
  if (label === 'evo:check-patch') {
    const match = prompt.match(/'--patch' '([^']+)'/);
    if (!match) throw new Error('Mock cannot identify the check-patch input');
    const digest = crypto.createHash('sha256').update(match[1]).digest('hex');
    return { exit_code: 0, stdout: JSON.stringify({
      fixture: true, ok: true, command: 'check-patch', archive: `${evalDir}/evolution_archive`, patch_sha256: digest,
      patch_object_path: `${evalDir}/evolution_archive/objects/${digest}`,
    }), stderr: '' };
  }
  if (label === 'evo:prepare') {
    const workspace = prompt.match(/'--workspace' '([^']+)'/)[1];
    const parent = prompt.match(/'--parent' '([^']+)'/)[1];
    const donors = [...prompt.matchAll(/'--donor' '([^']+)'/g)].map(match => ({ id: match[1] }));
    return { exit_code: 0, stdout: JSON.stringify({ fixture: true, ok: true, command: 'prepare',
      archive: `${evalDir}/evolution_archive`, receipt: { workspace, requested_parent_id: parent, donors,
        materialized_canonical_source_id: 'a'.repeat(64) } }), stderr: '' };
  }
  if (label.startsWith('evo:')) return {
    exit_code: 0, stdout: JSON.stringify({ fixture: true, ok: true, command: label.slice(4),
      archive: `${evalDir}/evolution_archive` }), stderr: '',
  };
  throw new Error(`Unscripted mock agent: ${label}`);
}

function inspectSource(source, inspection) {
  if (inspection === 'e2e') {
    const marker = '// ---- SINGLE-KERNEL PASS-THROUGH: delegate straight to the kernel layer ----';
    if (source.split(marker).length !== 2) throw new Error('E2E inspection seam changed');
    return source.slice(0, source.indexOf(marker)) + `
      const reference = { scriptPath: KERNEL_WF_SCRIPT };
      const calls = [];
      calls.push(await workflow(reference, laneArgs(probeArgs)));
      calls.push(await fastBoundedWorkflow(reference, probeArgs, 'fixture fast'));
      calls.push(await deepBoundedWorkflow(reference, probeArgs, 'fixture deep'));
      calls.push(await reserveBoundedWorkflow(reference, probeArgs, 'fixture reserve'));
      return { scriptPath: KERNEL_WF_SCRIPT, laneArgs: laneArgs(probeArgs), calls };
    `;
  }
  if (inspection === 'schemas') {
    const marker = "phase('Setup');";
    const offset = source.indexOf(marker);
    if (offset < 0) throw new Error('Lane schema inspection seam changed');
    const prefix = source.slice(0, offset);
    const names = [...prefix.matchAll(/const ([A-Z_]+_SCHEMA)\s*=/g)].map(match => match[1]);
    if (names.length < 15) throw new Error('Lane schema capture is incomplete');
    return prefix + `return { ${names.join(', ')} };`;
  }
  return source;
}

async function run(request) {
  const trace = { agents: [], workflows: [], phases: [], logs: [], pipelines: [], mockErrors: [], maxDepth: 0 };
  const timers = new Set();
  const root = path.posix.resolve(request.root);
  const sourcePaths = new Map(Object.entries(request.sources).map(([filename, source]) => [
    path.posix.resolve(root, filename), source,
  ]));

  async function execute(filename, args, depth, inspection) {
    trace.maxDepth = Math.max(trace.maxDepth, depth);
    if (depth > 2) throw new Error('Mock nesting limit exceeded: more than two workflow levels');
    let source = sourcePaths.get(path.posix.resolve(filename));
    if (source === undefined) throw new Error(`Unscripted workflow path: ${filename}`);
    source = inspectSource(source, inspection).replace(/^export const meta/m, 'const meta');
    const context = vm.createContext({
      args: clone(args), probeArgs: clone(request.probeArgs || {}),
      phase: title => trace.phases.push({ depth, title }),
      log: message => trace.logs.push({ depth, message: String(message) }),
      console: { log() {}, warn() {}, error() {} },
      budget: { total: null, spent: () => 0, remaining: () => Infinity },
      setTimeout: (callback, delay) => {
        const timer = setTimeout(callback, delay);
        timers.add(timer);
        return timer;
      },
      clearTimeout: timer => { clearTimeout(timer); timers.delete(timer); },
      agent: async (prompt, options) => {
        trace.agents.push({ depth, prompt, options: clone(options) });
        try { return agentReply(options.label, args, request, prompt); }
        catch (error) { trace.mockErrors.push(error.message); throw error; }
      },
      workflow: async (reference, childArgs) => {
        trace.workflows.push({ depth, scriptPath: reference.scriptPath, args: clone(childArgs) });
        if (request.stubWorkflows) return { validation_status: 'accepted', final_speedup: 1.25 };
        return execute(reference.scriptPath, childArgs, depth + 1);
      },
      parallel: async thunks => Promise.all(thunks.map(thunk => thunk())),
      pipeline: async (items, ...stages) => {
        trace.pipelines.push({ depth, items: clone(items), stages: stages.length });
        return Promise.all(items.map(async (item, index) => {
          let value = item;
          for (const stage of stages) value = await stage(value, item, index);
          return value;
        }));
      },
    });
    return new vm.Script(`(async () => {\n${source}\n})()`, { filename }).runInContext(context, { timeout: 3000 });
  }

  try {
    const result = await execute(path.posix.resolve(root, request.entry), request.args, request.initialDepth || 1, request.inspect);
    return { result: clone(result), error: null, trace };
  } catch (error) {
    return { result: null, error: { name: error.name, message: error.message }, trace };
  } finally {
    for (const timer of timers) clearTimeout(timer);
  }
}

async function main() {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const requests = Array.isArray(input) ? input : [input];
  const results = [];
  for (const request of requests) results.push(await run(request));
  process.stdout.write(JSON.stringify(Array.isArray(input) ? results : results[0]));
}

main().catch(error => { process.stderr.write(error.stack); process.exitCode = 1; });
