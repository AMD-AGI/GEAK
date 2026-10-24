#!/usr/bin/env node
// Standalone GEAK workflow runtime.
//
// WHY THIS EXISTS
// GEAK's orchestration (kernel_workflow.js / e2e_workflow.js) is plain JS that
// depends on globals injected ONLY by Claude Code's `Workflow` tool:
//   agent() parallel() pipeline() phase() log() workflow() args budget
// (plus runProcess(), which only this runtime provides: one command in its own
// process group, see backends/base.mjs; under the Workflow tool it is undefined).
// That runtime is Claude-Code-only, so the workflows cannot run under any other
// agent. This module re-implements those globals as an ordinary Node process and
// dispatches each agent() call to a PLUGGABLE backend (claude | codex | ...).
// The parallelism and one-level nesting live HERE, so the agent CLI does not
// need to support either — which is what makes a plain one-shot CLI usable.
//
// The workflow .js files are executed UNMODIFIED.
//
// USAGE
//   node run_workflow.mjs <script.js> [--args '<json>'] [--args-file <path>]
//        [--profile name | --agent name --model name] [--registry path]
//        [--result-file <path>] [--metrics-file <path>] [--concurrency N]
//        [--agent-timeout-ms N]
//   Selection also from env: GEAK_AGENT_PROFILE / GEAK_AGENT_BACKEND (=agent) /
//   GEAK_MODEL. Agents/models/profiles are defined in registry.json.
//   The final workflow return value is written to --result-file (if given) and
//   printed to stdout as "WORKFLOW_RESULT <json>"; run metrics (no token/cost) as
//   "WORKFLOW_METRICS <json>" and to --metrics-file.

import { readFile, writeFile } from 'node:fs/promises';
import { resolve as resolvePath } from 'node:path';
import { cpus } from 'node:os';
import { extractJson, validate, schemaInstruction } from './schema.mjs';
import { defaultConcurrency, runProcess, runtimeFault } from './backends/base.mjs';
import { parseArgv, loadRegistry, resolveSelection, neutralizeForBackend } from './config.mjs';
import { makeGenericBackend } from './backends/generic.mjs';

// ---------------------------------------------------------------------------
// Counting semaphore — bounds concurrent agent subprocesses. Shared across the
// top-level workflow AND any nested workflow() call, matching the Workflow tool.
// ---------------------------------------------------------------------------
class Semaphore {
  constructor(n) { this.free = n; this.waiters = []; }
  acquire() {
    return new Promise((res) => {
      if (this.free > 0) { this.free--; res(); }
      else this.waiters.push(res);
    });
  }
  release() {
    const w = this.waiters.shift();
    if (w) w();          // hand the slot straight to a waiter
    else this.free++;
  }
}

// ---------------------------------------------------------------------------
// Script loader. Strip the single top-level `export ` on meta/const/etc. so the
// module body can run inside an AsyncFunction (which also makes the script's
// top-level `return {...}` the workflow result). The regex only removes
// `export ` when immediately followed by a JS declaration keyword, so a bash
// `export FOO=bar` inside a prompt heredoc string is left untouched.
// ---------------------------------------------------------------------------
function toRunnableBody(src) {
  return src
    .replace(/^﻿/, '')
    .replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
}

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

export function compileWorkflowBody(src) {
  return new AsyncFunction(
    'agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', 'runProcess',
    'setTimeout', 'clearTimeout', 'setProcessDeadline',
    toRunnableBody(src)
  );
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------
const MAX_TOTAL_AGENTS = 1000;         // lifetime backstop, matches Workflow tool
const MAX_NESTING = 1;                 // workflow() may nest exactly one level
// Native Claude Code imposes NO per-agent timeout — GEAK owns the FUNCTIONAL
// timeout via its own hang-guards (kernel agentT ~60min, e2e agentBounded ~120min).
// To match native, the runtime's spawn timeout is only a GENEROUS BACKSTOP, kept
// strictly longer than GEAK's largest wrapper so it NEVER preempts a legitimately-
// long agent — it only reaps a subprocess that GEAK's Promise.race abandoned (its
// hang-guard resolves null but the OS child keeps holding a semaphore slot). The
// previous 60min default was SHORTER than e2e's 120min agents and silently killed
// them (spawning duplicate exp dirs). 4h. Set to 0 to disable the backstop.
const DEFAULT_AGENT_TIMEOUT_MS = 14400000;

function nowStamp() {
  // Date.now() is intentionally avoided inside scripts; here (host side) it is fine.
  return new Date().toISOString().slice(11, 19);
}

// ---------------------------------------------------------------------------
// createRuntime — builds the Workflow-tool globals over a pluggable backend.
// Exported so the primitives are unit-testable with a fake backend (selftest.mjs)
// without needing a real agent CLI / network / GPU.
//   backend: { name, async runAgent({prompt,label,cwd,env,model,timeoutMs}) -> {text} }
// Returns { agent, parallel, pipeline, phase, log, makeWorkflow, runScript, state }.
// ---------------------------------------------------------------------------
export function createRuntime({
  backend,
  concurrency = 8,
  agentTimeoutMs = DEFAULT_AGENT_TIMEOUT_MS,
  schemaRetries = 2,
  log = (msg) => process.stderr.write(`[${nowStamp()}] ${msg}\n`),
} = {}) {
  const SCHEMA_RETRIES = schemaRetries;
  const sem = new Semaphore(concurrency);
  // Anchor epoch cutoffs once. Later wall-clock adjustments cannot extend them.
  const epochStartedMs = Date.now();
  const monotonicStarted = process.hrtime.bigint();
  const elapsedMs = () => Number(process.hrtime.bigint() - monotonicStarted) / 1e6;
  const msUntil = (epoch) => epoch * 1000 - epochStartedMs - elapsedMs();
  const deadlineError = () => Object.assign(new Error('workflow process deadline reached'),
    { code: 'GEAK_PROCESS_DEADLINE' });
  const roleTimeoutError = () => Object.assign(new Error('workflow role timeout reached'),
    { code: 'GEAK_AGENT_TIMEOUT' });
  const boundedTimeout = (epoch, requested, cleanupMs) => {
    if (!Number.isFinite(epoch) || epoch <= 0) throw new Error('positive process deadline required');
    const remaining = Math.floor(msUntil(epoch) - cleanupMs);
    // runProcess interprets zero as unlimited, so never pass an expired bound.
    if (remaining <= 0) throw deadlineError();
    return requested > 0 ? Math.min(requested, remaining) : remaining;
  };
  // Metrics (no token/cost by design): agent-call count + structured-output
  // parse failures. The experiment runner aggregates these across combos.
  const state = { spawned: 0, schemaFails: 0 };

  let currentPhase = '';
  const phase = (title) => { currentPhase = title; log(`=== PHASE: ${title} ===`); };

  // budget stub — the workflows never read budget.total/spent/remaining, but the
  // global must exist. total=null => remaining()=Infinity (loops fall to their
  // non-budget path), matching "no target set".
  const budget = { total: null, spent: () => 0, remaining: () => Infinity };

  // ---- agent() -----------------------------------------------------------
  async function agent(prompt, agentOpts = {}) {
    const label = agentOpts.label || 'agent';
    const ph = agentOpts.phase || currentPhase;
    const schema = agentOpts.schema;
    // A role's existing hang limit covers queueing and all schema attempts.
    const roleStopMs = agentOpts.processDeadlineEpoch != null && agentOpts.processTimeoutMs > 0
      ? elapsedMs() + Number(agentOpts.processTimeoutMs) : Infinity;

    if (++state.spawned > MAX_TOTAL_AGENTS) {
      throw new Error(`agent cap ${MAX_TOTAL_AGENTS} exceeded (runaway-loop backstop)`);
    }

    const withSchema = schema ? prompt + schemaInstruction(schema) : prompt;
    // Strip Claude-specific wording (e.g. "a StructuredOutput tool is forced")
    // for non-claude backends — roles/*.md and the .js are used unmodified, so
    // this compensates at the runtime layer. No-op for claude.
    const fullPrompt = neutralizeForBackend(withSchema, backend.name);

    await sem.acquire();
    try {
      log(`  -> [${ph}] ${label} (backend=${backend.name}, ${state.spawned} spawned)`);
      let lastErr;
      const attempts = schema ? SCHEMA_RETRIES + 1 : 1;
      for (let i = 0; i < attempts; i++) {
        let timeoutMs = agentTimeoutMs;
        if (agentOpts.processDeadlineEpoch != null) {
          const roleRemaining = Math.floor(roleStopMs - elapsedMs());
          if (roleRemaining <= 0) throw roleTimeoutError();
          const requested = timeoutMs > 0 ? Math.min(timeoutMs, roleRemaining) : roleRemaining;
          // Recompute after the semaphore wait and before every schema retry.
          timeoutMs = boundedTimeout(agentOpts.processDeadlineEpoch, requested, 7000);
        }
        // Describe this attempt's existing bound after queue/phase/retry clipping.
        const timeGuidance = timeoutMs > 0
          ? `\n\nBackend call allowance at dispatch: ${timeoutMs} ms. Backend cutoff (UTC): ` +
            `${new Date(epochStartedMs + elapsedMs() + timeoutMs).toISOString()}. ` +
            `This includes startup, tool work and your final response. Finish tool work early enough to return ` +
            `the requested result within this bound and any earlier workflow deadline; report incomplete work if needed.`
          : '';
        let text;
        try {
          ({ text } = await backend.runAgent({
            prompt: fullPrompt + timeGuidance, label, cwd: agentOpts.cwd, env: agentOpts.env,
            model: agentOpts.model, timeoutMs,
          }));
        } catch (error) {
          if (agentOpts.processDeadlineEpoch != null && msUntil(agentOpts.processDeadlineEpoch) <= 7000) {
            throw deadlineError();
          }
          if (elapsedMs() >= roleStopMs - 1) throw roleTimeoutError();
          throw error;
        }
        if (!schema) return text;
        try {
          const obj = extractJson(text);
          const v = validate(obj, schema);
          if (!v.ok) throw new Error(`schema mismatch: ${v.errors.slice(0, 3).join('; ')}`);
          return obj;
        } catch (e) {
          lastErr = e;
          state.schemaFails++;
          log(`  !! [${ph}] ${label} structured-output parse failed (attempt ${i + 1}/${attempts}): ${e.message}`);
        }
      }
      // Exhausted internal retries — throw so the script's agentT retries/degrades.
      throw lastErr || new Error(`[${label}] structured output unusable`);
    } finally {
      sem.release();
    }
  }

  // ---- parallel() : barrier; a throwing thunk resolves to null ------------
  async function parallel(thunks) {
    return Promise.all(
      (thunks || []).map((t) => Promise.resolve().then(t).catch((e) => {
        log(`  xx parallel task failed -> null: ${String(e && e.message || e).slice(0, 160)}`);
        return null;
      }))
    );
  }

  // ---- pipeline() : per-item, no barrier between stages -------------------
  // Each stage receives (prevResult, originalItem, index). Stage 1's prevResult
  // IS the item. A throwing stage drops that item to null and skips the rest.
  async function pipeline(items, ...stages) {
    return Promise.all((items || []).map(async (item, idx) => {
      let prev = item;
      for (const st of stages) {
        try { prev = await st(prev, item, idx); }
        catch (e) {
          log(`  xx pipeline item ${idx} failed -> null: ${String(e && e.message || e).slice(0, 160)}`);
          return null;
        }
      }
      return prev;
    }));
  }

  // ---- workflow() : run another script inline, one level of nesting -------
  function makeWorkflow(depth) {
    return async function workflow(ref, wfArgs) {
      if (depth >= MAX_NESTING) {
        throw new Error('workflow() nesting is one level only');
      }
      const path = typeof ref === 'string'
        ? ref
        : (ref && ref.scriptPath);
      if (!path) throw new Error('workflow(ref): ref must be {scriptPath} or a path string');
      log(`  >> nested workflow: ${path}`);
      return runScript(resolvePath(path), wfArgs || {}, depth + 1);
    };
  }

  async function runScript(absScriptPath, scriptArgs, depth) {
    const src = await readFile(absScriptPath, 'utf8');
    let fn;
    try {
      fn = compileWorkflowBody(src);
    } catch (e) {
      throw new Error(`failed to compile workflow ${absScriptPath}: ${e.message}`);
    }
    // Deadline flags may outlive an early return or failure. Keep them active
    // while this script runs, then clear only timers created by this invocation.
    const timers = new Set();
    let closed = false;
    const scriptSetTimeout = (...args) => {
      const timer = setTimeout(...args);
      // Late continuations must not keep the process alive after script return.
      if (closed) timer.unref();
      else timers.add(timer);
      return timer;
    };
    const scriptClearTimeout = (timer) => {
      timers.delete(timer);
      clearTimeout(timer);
    };
    let processDeadlineEpoch = null;
    let maximumDeadlineEpoch = null;
    if (scriptArgs.search_reserve_s != null) {
      const total = Number(scriptArgs.deadline_epoch);
      const reserve = Number(scriptArgs.search_reserve_s);
      const cleanup = Number(scriptArgs.final_cleanup_reserve_s || 0);
      if (!Number.isFinite(total) || total <= 0 || !Number.isFinite(reserve) || reserve <= 0
          || !Number.isFinite(cleanup) || cleanup < 0 || cleanup >= reserve) {
        throw new Error('process-bounded workflow requires a deadline and positive search reserve');
      }
      maximumDeadlineEpoch = total - cleanup;
      processDeadlineEpoch = maximumDeadlineEpoch;
    }
    const setProcessDeadline = (epoch) => {
      if (maximumDeadlineEpoch == null || !Number.isFinite(epoch) || epoch <= 0
          || epoch > maximumDeadlineEpoch) throw new Error('process deadline exceeds original workflow allowance');
      processDeadlineEpoch = epoch;
    };
    setProcessDeadline.secondsLeft = () => processDeadlineEpoch == null ? Infinity : msUntil(processDeadlineEpoch) / 1000;
    const scriptAgent = (prompt, options = {}) => agent(prompt, processDeadlineEpoch == null ? options : {
      ...options, processDeadlineEpoch,
      processTimeoutMs: options.processTimeoutMs ?? Number(scriptArgs.agent_timeout_ms || 0),
    });
    Object.defineProperty(scriptAgent, 'processDeadlineActive', { get: () => processDeadlineEpoch != null });
    const scriptRunProcess = (options) => {
      if (processDeadlineEpoch == null) return runProcess(options);
      const cleanupMs = (options.graceMs ?? 5000) + (options.settleMs ?? 2000);
      return runProcess({ ...options,
        timeoutMs: boundedTimeout(processDeadlineEpoch, options.timeoutMs || 0, cleanupMs) });
    };
    try {
      return await fn(scriptAgent, parallel, pipeline, phase, log, makeWorkflow(depth), scriptArgs, budget,
        scriptRunProcess, scriptSetTimeout, scriptClearTimeout, setProcessDeadline);
    } finally {
      closed = true;
      for (const timer of timers) clearTimeout(timer);
      timers.clear();
    }
  }

  return { agent, parallel, pipeline, phase, log, budget, makeWorkflow, runScript, state };
}

// ---------------------------------------------------------------------------
// CLI entry
// ---------------------------------------------------------------------------
// Resolve the agent backend from CLI flags / env via the config registry.
// Precedence: --profile/--agent/--model flag > GEAK_AGENT_PROFILE/-AGENT/-MODEL
// (GEAK_AGENT_BACKEND kept as a back-compat alias for --agent) > registry default.
// A hand-written backends/<agent>.mjs, if present, overrides the generic runner.
export async function selectBackend(opts = {}) {
  const flag = (k) => (opts[k] && opts[k] !== true) ? opts[k] : undefined;
  const sel = {
    profile: flag('profile') || process.env.GEAK_AGENT_PROFILE,
    agent: flag('agent') || process.env.GEAK_AGENT_BACKEND,
    model: flag('model') || process.env.GEAK_MODEL,
    // Lets a provider key select its own CLI when nothing was named explicitly,
    // so a single-kernel run needs no more setup than an e2e run does.
    env: process.env,
  };
  const registryPath = flag('registry') || process.env.GEAK_REGISTRY;
  const registry = await loadRegistry(registryPath || undefined);
  const resolved = resolveSelection(registry, sel);

  // Escape hatch: a custom backends/<agent>.mjs takes precedence over generic.
  try {
    const custom = await import(`./backends/${resolved.agentName}.mjs`);
    if (custom && typeof custom.runAgent === 'function') {
      return { name: custom.name || resolved.agentName, runAgent: custom.runAgent, resolved };
    }
  } catch { /* no custom module — use generic */ }

  const backend = makeGenericBackend(resolved);
  return { ...backend, resolved };
}

async function main() {
  const opts = parseArgv(process.argv.slice(2));
  const scriptPath = opts._[0];
  if (!scriptPath) {
    console.error('usage: run_workflow.mjs <script.js> [--args json] [--args-file path] ' +
      '[--profile name | --agent name --model name] [--result-file path] ' +
      '[--metrics-file path] [--concurrency N]');
    process.exit(2);
  }

  let topArgs = {};
  if (opts['args-file']) topArgs = JSON.parse(await readFile(opts['args-file'], 'utf8'));
  else if (opts.args && opts.args !== true) topArgs = JSON.parse(opts.args);

  const backend = await selectBackend(opts);

  const concurrency = opts.concurrency && opts.concurrency !== true
    ? parseInt(opts.concurrency, 10)
    : (parseInt(process.env.GEAK_CONCURRENCY || '', 10) || defaultConcurrency(cpus().length));

  // Honor an explicit 0 (disable backstop) from CLI or env: use Number.isFinite,
  // NOT `|| DEFAULT`, since 0 is falsy and would otherwise be silently overridden.
  const cliTimeout = (opts['agent-timeout-ms'] && opts['agent-timeout-ms'] !== true)
    ? parseInt(opts['agent-timeout-ms'], 10) : NaN;
  const envTimeout = parseInt(process.env.GEAK_AGENT_TIMEOUT_MS || '', 10);
  const agentTimeoutMs = Number.isFinite(cliTimeout) ? cliTimeout
    : Number.isFinite(envTimeout) ? envTimeout
    : DEFAULT_AGENT_TIMEOUT_MS;

  // schema-extraction retries WITHIN a single agent() call (mirrors the harness
  // "model retries on mismatch"); the script's agentT adds an OUTER retry layer.
  const schemaRetries = parseInt(process.env.GEAK_SCHEMA_RETRIES || '2', 10);

  const rt = createRuntime({ backend, concurrency, agentTimeoutMs, schemaRetries });

  const startedMs = Date.now();
  const combo = `${backend.resolved.agentName}${backend.resolved.modelName ? '/' + backend.resolved.modelName : ''}`;
  rt.log(`runtime start: script=${scriptPath} combo=${combo} concurrency=${concurrency}`);

  let result = null;
  let ok = false;
  let errMsg = null;
  try {
    result = await rt.runScript(resolvePath(scriptPath), topArgs, 0);
    ok = true;
  } catch (e) {
    errMsg = String(e && e.message || e);
    rt.log(`runtime ERROR: ${errMsg}`);
  }
  // Agent failures degrade to null inside the workflow; this one must still fail the run.
  const fault = runtimeFault();
  if (fault) {
    ok = false;
    errMsg = errMsg ? `${errMsg}; ${fault}` : fault;
    rt.log(`runtime ERROR: ${fault}`);
  }
  const wallMs = Date.now() - startedMs;

  const resultJson = JSON.stringify(result ?? null);
  if (opts['result-file'] && opts['result-file'] !== true) {
    await writeFile(opts['result-file'], resultJson);
    rt.log(`result written to ${opts['result-file']}`);
  }

  // Machine-readable metrics (no token/cost by design) for the experiment runner.
  const metrics = {
    agent: backend.resolved.agentName,
    model: backend.resolved.modelName,
    ok,
    error: errMsg,
    wall_ms: wallMs,
    agents_spawned: rt.state.spawned,
    schema_failures: rt.state.schemaFails,
    script: scriptPath,
  };
  if (opts['metrics-file'] && opts['metrics-file'] !== true) {
    await writeFile(opts['metrics-file'], JSON.stringify(metrics, null, 2));
    rt.log(`metrics written to ${opts['metrics-file']}`);
  }

  process.stdout.write(`WORKFLOW_RESULT ${resultJson}\n`);
  process.stdout.write(`WORKFLOW_METRICS ${JSON.stringify(metrics)}\n`);
  rt.log(`runtime done: combo=${combo} ok=${ok} ${rt.state.spawned} agent(s), ${rt.state.schemaFails} schema-fail, ${Math.round(wallMs / 1000)}s`);
  if (!ok) process.exitCode = 1;
}

// Only run as CLI when invoked directly (not when imported by selftest).
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => {
    console.error(`FATAL: ${e && e.stack || e}`);
    process.exit(1);
  });
}
