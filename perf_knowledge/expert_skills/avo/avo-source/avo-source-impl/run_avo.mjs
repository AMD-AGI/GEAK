// ===========================================================================
// run_avo.mjs — standalone runtime shim for PerfSkills Workflows.
//
// **Canonical copy:** lives next to `avo_workflow.js` under
// `PerfSkills/avo-workflow/` (versioned with the engine). A thin forwarder may
// exist at the monorepo root (`avo_projects/run_avo.mjs`) for legacy launch scripts.
//
// PerfSkills' avo_workflow.js / team_workflow.js are NOT plain Node scripts:
// they rely on runtime-injected globals (args, agent, phase, log, pipeline)
// that the IDE "Workflow tool" normally provides. This shim supplies those
// primitives so the UNMODIFIED workflow runs from a normal terminal, with the
// agent() backend wired to the local headless Claude Code CLI (`claude -p`).
//
// Usage:
//   node run_avo.mjs <config.json>
// where config.json = { "scriptPath": "<abs avo_workflow.js>", "args": { ... } }
//
// Everything the workflow writes is governed by args.exp_root (logs land OUTSIDE
// PerfSkills). Per-agent raw transcripts are saved under <exp_root>/_shim_logs/.
// ===========================================================================

import fs from 'fs';
import { spawn, execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

import { finalAnswer } from './lib/agent_contract.mjs';
import {
  AGENT_TAG_VAR, pidsWithTag, reapTagged, waitTagGone,
} from './lib/agent_reaper.mjs';
import {
  isAgentTimeout, isTransportFailure, transportBackoffMs,
} from './lib/agent_failure.mjs';

const { readFileSync, writeFileSync, appendFileSync, mkdirSync, rmSync, existsSync } = fs;

const NODE_MAJOR = Number(String(process.versions.node || '0').split('.')[0]);
if (!Number.isFinite(NODE_MAJOR) || NODE_MAJOR < 18) {
  console.error(`run_avo.mjs requires Node.js >= 18 (found ${process.versions.node || 'unknown'})`);
  process.exit(2);
}

// --- config ----------------------------------------------------------------
const cfgPath = process.argv[2];
const VALIDATE_ONLY = process.argv.includes('--validate-only');
if (!cfgPath) {
  console.error('usage: node run_avo.mjs <config.json> [--validate-only]');
  process.exit(2);
}
const CFG = JSON.parse(readFileSync(cfgPath, 'utf8'));
if (!CFG.scriptPath || !CFG.args) { console.error('config needs {scriptPath, args}'); process.exit(2); }
for (const [name, value] of [
  ['scriptPath', CFG.scriptPath],
  ['args.kernel_path', CFG.args.kernel_path],
  ['args.workflow_dir', CFG.args.workflow_dir],
]) {
  if (!value || !path.isAbsolute(String(value)) || !existsSync(String(value))) {
    console.error(`${name} must be an existing absolute path`);
    process.exit(2);
  }
}
const REQUESTED_LEVEL = String(
  CFG.args.avo_language_level || CFG.args.language_level || 'source').toLowerCase();
const TARGET_LANGUAGE = String(
  CFG.args.target_language || CFG.args.kernel_language || '').toLowerCase();
const LOW_LEVEL_TARGETS = new Set([
  'asm', 'amdgcn', 'gcn', 'ir', 'llvm-ir', 'llvm_ir', 'll', 'ttgir',
]);
const LOW_LEVEL_FLAG = ['asm_mode', 'ir_mode', 'ir_feedback']
  .some((key) => String(CFG.args[key] ?? '').toLowerCase() === 'true');
if (REQUESTED_LEVEL !== 'source' || LOW_LEVEL_TARGETS.has(TARGET_LANGUAGE) || LOW_LEVEL_FLAG) {
  console.error('integrated AVO supports source-level optimization only');
  process.exit(2);
}

// <GEAK>/perf_knowledge/expert_skills/avo/avo-source/avo-source-impl -> <GEAK>
const DEFAULT_ROOT = path.resolve(path.dirname(CFG.scriptPath), '../../../../..');
const CLAUDE_HOME = CFG.claude_home || process.env.HOME || DEFAULT_ROOT;
const CLAUDE_BIN = CFG.claude_bin || 'claude';
const CLAUDE_MODEL = CFG.model || '';                       // '' = user's configured default
const AGENT_TIMEOUT_MS = parseInt(CFG.agent_timeout_ms || 2400000, 10); // 40 min/agent
const AGENT_RETRIES = parseInt(CFG.agent_retries != null ? CFG.agent_retries : 1, 10);
// An unreachable backend gets its own retry budget and backs off, so a transient API
// blip costs minutes instead of the whole run. Classification lives in lib so it can
// be unit-tested; see lib/agent_failure.mjs for why only some channels are consulted.
const AGENT_TRANSPORT_RETRIES = parseInt(
  CFG.agent_transport_retries != null ? CFG.agent_transport_retries : 6, 10);
const AGENT_TRANSPORT_BACKOFF_MS = parseInt(
  CFG.agent_transport_backoff_ms != null ? CFG.agent_transport_backoff_ms : 20000, 10);
const AGENT_TRANSPORT_BACKOFF_CAP_MS = parseInt(
  CFG.agent_transport_backoff_cap_ms != null ? CFG.agent_transport_backoff_cap_ms : 300000, 10);
// An agent that has returned its final answer normally exits within a second. One
// that still holds a background task or a scheduled wakeup does not exit at all;
// after this grace it is stopped instead of being waited on until the timeout.
const FINAL_ANSWER_GRACE_MS = 3000;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// Default to this checkout instead of a site-specific mount. Set root_dir when
// the kernel/capture intentionally lives outside the GEAK repository.
const ROOT_DIR = CFG.root_dir || CFG.args.root_dir || DEFAULT_ROOT; // cwd + --add-dir scope

const EXP_ROOT = String(CFG.args.exp_root || `${DEFAULT_ROOT}-runs/avo-source`).replace(/\/+$/, '');
if (VALIDATE_ONLY) {
  console.log(JSON.stringify({
    status: 'ok',
    entrypoint: 'run_avo.mjs',
    scriptPath: CFG.scriptPath,
    kernel_path: CFG.args.kernel_path,
    workflow_dir: CFG.args.workflow_dir,
    exp_root: EXP_ROOT,
    operator: CFG.args.operator || 'team',
    start_policy: CFG.args.start_policy || 'diverse',
    launched: false,
  }, null, 2));
  process.exit(0);
}
const SHIM_LOGS = `${EXP_ROOT}/_shim_logs`;
mkdirSync(SHIM_LOGS, { recursive: true });

const startedAt = Date.now();
const ts = () => new Date().toISOString().replace(/[:.]/g, '-');
const slug = (s) => String(s || 'agent').replace(/[^a-zA-Z0-9_.-]+/g, '_').slice(0, 80);

// --- injected primitives ----------------------------------------------------
globalThis.args = CFG.args;
globalThis.log = (m) => console.log(`[avo +${((Date.now() - startedAt) / 1000).toFixed(0)}s] ${m}`);
globalThis.phase = (n) => console.log(`\n========== PHASE: ${n} ==========`);
// Why the last agent(label) call returned null, so the workflow can tell a dead
// backend from a rig that cannot measure. Keyed by label because the Evolve fan-out
// runs several agents concurrently and a single global would race between them.
const AGENT_FAILURES = new Map();
globalThis.agentFailure = (label) => AGENT_FAILURES.get(label) || null;

// pipeline(items, ...stages): run each item's stages sequentially, all items
// concurrently (GPU contention is serialized by the roles' gpu_lock.sh).
// Every item is awaited to completion even when one rejects: the caller's step
// teardown prunes ALL non-main worktrees, so returning while siblings are still
// building would delete the directories they are working in.
globalThis.pipeline = async (items, ...stages) => {
  const settled = await Promise.allSettled((items || []).map((it) =>
    stages.reduce((p, fn) => p.then((v) => fn(v)), Promise.resolve(it))));
  const failed = settled.find((r) => r.status === 'rejected');
  if (failed) throw failed.reason;
  return settled.map((r) => r.value);
};

// recordAttempt(stateDir, rec): DETERMINISTIC backstop for the AVO attempt ledger.
// The workflow records EVERY attempt (incl. failures) here so the full search
// trajectory is durable WITHOUT depending on an LLM role running a shell command.
// The workflow guards every call with `typeof recordAttempt === 'function'`, so it
// is inert under a Workflow runtime that does not provide this primitive (there the
// variation_engineer role self-records instead). Best-effort: never throws.
globalThis.recordAttempt = (stateDir, rec) => {
  try {
    if (!stateDir) return false;
    mkdirSync(stateDir, { recursive: true });
    const line = JSON.stringify({ ts: Date.now() / 1000, source: 'runtime', ...(rec || {}) });
    appendFileSync(path.join(stateDir, 'attempts.jsonl'), line + '\n');
    return true;
  } catch { return false; }
};

// recordSupervision(stateDir, directive): DETERMINISTIC supervisor audit log. The
// orchestrator owns this append so the `ts` is a REAL wall-clock stamp (the LLM-self-
// appended variant sometimes emitted a hallucinated epoch — observed a 2026-06-09 line
// inside a 2026-06-17 run). Same {ts, directive} shape avo_lineage.py resume expects.
// Best-effort: never throws. Inert when absent (then the supervisor role self-appends).
globalThis.recordSupervision = (stateDir, directive) => {
  try {
    if (!stateDir) return false;
    mkdirSync(stateDir, { recursive: true });
    const line = JSON.stringify({ ts: Date.now() / 1000, source: 'runtime', directive: directive || {} });
    appendFileSync(path.join(stateDir, 'supervisor_log.jsonl'), line + '\n');
    return true;
  } catch { return false; }
};

// syncAvoHeartbeat(stateDir, fields): DETERMINISTIC merge into avo_state/heartbeat.json.
// The orchestrator calls this at the end of each Evolve loop iteration (and on supervisor
// / escalate) so `step` matches the JS loop counter — authoritative vs per-role bash
// heartbeat that could fire mid-step or use a mismatched step index. Inert when absent.
globalThis.syncAvoHeartbeat = (stateDir, fields) => {
  try {
    if (!stateDir || !fields || typeof fields !== 'object') return false;
    mkdirSync(stateDir, { recursive: true });
    const p = path.join(stateDir, 'heartbeat.json');
    let cur = {};
    try {
      cur = JSON.parse(readFileSync(p, 'utf8'));
    } catch { cur = {}; }
    const merged = { ...cur, ...fields, ts: Date.now() / 1000, iso: new Date().toISOString() };
    writeFileSync(p, JSON.stringify(merged, null, 2), 'utf8');
    return true;
  } catch {
    return false;
  }
};

// pruneWorktrees(avoRepo, under, opts): DETERMINISTIC per-step worktree teardown. The
// orchestrator owns the worktree lifecycle here, so a step's scratch worktrees are
// reclaimed even when an LLM role forgot to (or errored before) removing its own —
// without depending on the LLM running a shell command. The main avo_repo is NEVER
// touched. Two modes:
//   • opts.all === true  → remove EVERY registered worktree except the main repo. Safe
//     because the evolve loop is strictly sequential (team fan-out is awaited within one
//     iteration), so at a step's teardown every non-main worktree is this-step garbage —
//     INCLUDING ones an LLM mis-created at the wrong path (git registers them by absolute
//     path, so they are caught no matter where they landed).
//   • otherwise          → remove only worktrees registered under the `under` prefix(es).
// Best-effort: never throws. Returns the count of worktrees removed.
globalThis.pruneWorktrees = (avoRepo, under, opts = {}) => {
  let removed = 0;
  try {
    if (!avoRepo) return 0;
    const all = !!(opts && opts.all);
    const prefixes = (Array.isArray(under) ? under : [under]).filter(Boolean).map(String);
    if (!all && !prefixes.length) return 0;
    const git = (a) => execFileSync(
      'git', ['-c', `safe.directory=${avoRepo}`, '-C', avoRepo, ...a],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
    try { git(['worktree', 'prune']); } catch { /* */ }
    let list = '';
    try { list = git(['worktree', 'list', '--porcelain']); } catch { return 0; }
    for (const line of list.split('\n')) {
      if (!line.startsWith('worktree ')) continue;
      const wt = line.slice('worktree '.length).trim();
      if (!wt || wt === avoRepo) continue;                       // never remove the main repo
      if (!all && !prefixes.some((p) => wt === p || wt.startsWith(p))) continue;
      try { git(['worktree', 'remove', '--force', wt]); } catch { /* */ }
      try { rmSync(wt, { recursive: true, force: true }); } catch { /* */ }
      removed++;
    }
    try { git(['worktree', 'prune']); } catch { /* */ }
  } catch { /* best-effort */ }
  return removed;
};

// addWorktree(avoRepo, wsPath, ref): DETERMINISTIC workspace creation (Layer 3). The
// orchestrator pre-creates each step's primary workspace at a FIXED absolute path and
// the ROLE just `cd`s in — so the LLM never runs `git worktree add` for the primary
// workspace and can no longer mis-place it (the wrong-path failure mode). Clears any
// stale dir first, falls back to a plain copy+checkout if `worktree add` fails (e.g.
// shared object-store quirks), and strips ephemeral build dirs. Best-effort: returns
// the absolute workspace path on success, or null (then the role self-creates).
globalThis.addWorktree = (avoRepo, wsPath, ref) => {
  try {
    if (!avoRepo || !wsPath || !ref) return null;
    const git = (a) => execFileSync(
      'git', ['-c', `safe.directory=${avoRepo}`, '-C', avoRepo, ...a],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
    try { git(['worktree', 'remove', '--force', wsPath]); } catch { /* */ }
    try { rmSync(wsPath, { recursive: true, force: true }); } catch { /* */ }
    mkdirSync(path.dirname(wsPath), { recursive: true });
    try {
      git(['worktree', 'add', '-f', '--detach', wsPath, ref]);
    } catch {
      mkdirSync(wsPath, { recursive: true });
      execFileSync('cp', ['-r', `${avoRepo}/.`, wsPath], { stdio: 'ignore' });
      execFileSync('git', ['-c', `safe.directory=${wsPath}`,
        '-C', wsPath, 'checkout', '-f', ref], { stdio: 'ignore' });
    }
    for (const d of ['build', '__pycache__', '.torch_ext']) {
      try { rmSync(path.join(wsPath, d), { recursive: true, force: true }); } catch { /* */ }
    }
    return wsPath;
  } catch { return null; }
};

// --- JSON extraction from a possibly-chatty final message -------------------
function extractJson(text) {
  if (text == null) return null;
  const s = String(text).trim();
  try { return JSON.parse(s); } catch { /* fall through */ }
  // strip ```json fences
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) { try { return JSON.parse(fence[1].trim()); } catch { /* */ } }
  // balanced-brace scan for the last top-level {...}
  let depth = 0, start = -1, inStr = false, esc = false, best = null;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inStr) { if (esc) esc = false; else if (c === '\\') esc = true; else if (c === '"') inStr = false; continue; }
    if (c === '"') inStr = true;
    else if (c === '{') { if (depth === 0) start = i; depth++; }
    else if (c === '}') { depth--; if (depth === 0 && start >= 0) { const cand = s.slice(start, i + 1); try { best = JSON.parse(cand); } catch { /* */ } } }
  }
  return best;
}

// One-line summary of a tool_use event (fine-grained progress without the noise).
function fmtToolUse(name, input) {
  try {
    if (name === 'Bash' && input && input.command) {
      const c = String(input.command).replace(/\s+/g, ' ').trim();
      return `Bash: ${c.slice(0, 90)}${c.length > 90 ? '…' : ''}`;
    }
    if (input && input.file_path) return `${name}: …/${String(input.file_path).split('/').slice(-2).join('/')}`;
    if (name === 'Grep' && input && input.pattern) return `Grep: ${String(input.pattern).slice(0, 60)}`;
    return name || 'tool';
  } catch { return name || 'tool'; }
}

// Agents run in their own process groups (see `detached` below), so they no longer
// inherit the terminal's Ctrl-C. Track them and tear the groups down ourselves.
const LIVE_AGENT_PGIDS = new Set();
const LIVE_AGENT_TAGS = new Set();
let agentTagSeq = 0;
let shuttingDown = false;
const shutdownAgents = (signal) => {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const pgid of LIVE_AGENT_PGIDS) {
    try { process.kill(-pgid, signal); } catch { /* already gone */ }
  }
  // Ctrl-C must not leave a worker's background loop holding a GPU. This runs
  // inside an 'exit' handler too, so it signals once rather than escalating later.
  for (const tag of LIVE_AGENT_TAGS) {
    for (const pid of pidsWithTag(tag)) {
      try { process.kill(pid, signal); } catch { /* already gone */ }
    }
  }
};
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(sig, () => { shutdownAgents('SIGKILL'); process.exit(130); });
}
process.on('exit', () => shutdownAgents('SIGKILL'));

// Streaming spawn: parse the NDJSON event stream so the round shows EACH tool call
// live + a periodic liveness ticker (instead of one line per finished agent). Falls
// back gracefully: if no `result` event arrives, the raw stream is still captured.
// `required` = the keys of the role's schema; a result that carries them all is the
// final answer, and the call ends on it even if the process stays alive.
function spawnClaude(prompt, label, timeoutMs = AGENT_TIMEOUT_MS, required = []) {
  return new Promise((resolve) => {
    const a = ['--print', '--output-format', 'stream-json', '--verbose',
      '--dangerously-skip-permissions', '--add-dir', ROOT_DIR, '--add-dir', '/tmp'];
    if (CLAUDE_MODEL) a.push('--model', CLAUDE_MODEL);
    const tag = `${process.pid}-${++agentTagSeq}-${Math.random().toString(36).slice(2, 10)}`;
    LIVE_AGENT_TAGS.add(tag);
    const child = spawn(CLAUDE_BIN, a, {
      cwd: ROOT_DIR,
      // IS_SANDBOX=1 lets --dangerously-skip-permissions run as root (this dev box).
      env: {
        ...process.env, HOME: CLAUDE_HOME, IS_SANDBOX: '1', [AGENT_TAG_VAR]: tag,
      },
      stdio: ['pipe', 'pipe', 'pipe'],
      // Own process group: on timeout the agent's GPU grandchildren (benchmark
      // binaries, rocprof) must die with it, or they contaminate every later
      // measurement and keep holding the GPU lock.
      detached: true,
    });
    LIVE_AGENT_PGIDS.add(child.pid);
    const t0 = Date.now();
    let raw = '', err = '', buf = '';
    let resultText = null, costUsd = null, isError = false, nTools = 0, done = false;
    let childExited = false;
    let graceTimer = null;   // armed by the first final answer
    let early = false;       // the call is being ended on the final answer
    const el = () => ((Date.now() - startedAt) / 1000).toFixed(0);
    const since = () => ((Date.now() - t0) / 1000).toFixed(0);
    const prog = (m) => console.log(`[avo +${el()}s]   · ${label}: ${m}`);

    const handleEvent = (ev) => {
      // Once the call is being ended on the final answer, late events must not
      // replace it.
      if (early || !ev || typeof ev !== 'object') return;
      if (ev.type === 'assistant' && ev.message && Array.isArray(ev.message.content)) {
        for (const b of ev.message.content) {
          if (b.type === 'tool_use') { nTools++; prog(`⛏ ${fmtToolUse(b.name, b.input)}`); }
          else if (b.type === 'text' && b.text && b.text.trim()) {
            const t = b.text.replace(/\s+/g, ' ').trim();
            prog(`💭 ${t.slice(0, 120)}${t.length > 120 ? '…' : ''}`);
          }
        }
      } else if (ev.type === 'result') {
        if (typeof ev.result === 'string') resultText = ev.result;
        if (typeof ev.total_cost_usd === 'number') costUsd = ev.total_cost_usd;
        isError = !!ev.is_error;
        if (!isError && !graceTimer && finalAnswer(ev.result, required)) {
          graceTimer = setTimeout(finishOnAnswer, FINAL_ANSWER_GRACE_MS);
        }
      }
    };

    child.stdout.on('data', (d) => {
      raw += d; buf += d;
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) { try { handleEvent(JSON.parse(line)); } catch { /* partial/non-json */ } }
      }
    });
    child.stderr.on('data', (d) => { err += d; });

    const ticker = setInterval(() => prog(`… running (${since()}s, ${nTools} tool calls)`), 60000);
    const finish = (res) => {
      if (done) return;
      done = true;
      LIVE_AGENT_PGIDS.delete(child.pid);
      const escaped = reapTagged(tag);
      LIVE_AGENT_TAGS.delete(tag);
      if (escaped) prog(`⚠ reaped ${escaped} background process(es) the agent left running`);
      clearInterval(ticker); clearTimeout(timer); clearTimeout(graceTimer); resolve(res);
    };
    // Only ever signal the group while the child is known to be alive: once Node
    // reaps it the pid (== pgid) can be recycled by an unrelated process.
    const killTree = (signal) => {
      if (childExited) return;
      try { process.kill(-child.pid, signal); } catch { /* group already gone */ }
    };
    const timer = setTimeout(() => {
      killTree('SIGTERM');
      setTimeout(() => killTree('SIGKILL'), 5000).unref();
      finish({ ok: false, raw, err: err + '\n[shim] TIMEOUT', resultText, costUsd, nTools });
    }, timeoutMs);
    // The final answer arrived and the process is still alive after the grace: stop
    // the agent's own group and everything that carries its tag, and return only
    // when all of it is gone, so nothing overlaps the next step's measurement.
    function finishOnAnswer() {
      if (done) return;
      early = true;
      clearTimeout(timer);
      killTree('SIGTERM');
      setTimeout(() => killTree('SIGKILL'), 5000).unref();
      const stopping = reapTagged(tag);
      prog(`✂ final answer received; process still alive, stopping ${stopping} process(es)`);
      waitTagGone(tag).then((left) => {
        if (left) prog(`⚠ ${left} process(es) still alive after SIGKILL`);
        finish({ ok: true, code: null, raw, err, resultText, costUsd, nTools, early: true });
      });
    }
    child.on('exit', () => { childExited = true; });
    child.on('error', (e) => {
      if (early) return;
      finish({ ok: false, raw, err: String(e), resultText, costUsd, nTools });
    });
    child.on('close', (code) => {
      if (early) return;
      finish({ ok: code === 0 && !isError, code, raw, err, resultText, costUsd, nTools });
    });
    child.stdin.write(prompt); child.stdin.end();
  });
}

// agent(prompt, {phase, label, schema}) -> parsed JSON object (or null).
globalThis.agent = async (prompt, opts = {}) => {
  const label = opts.label || opts.phase || 'agent';
  const requestedTimeout = Number(opts.timeout_ms);
  const timeoutMs = Number.isFinite(requestedTimeout) && requestedTimeout > 0
    ? Math.min(AGENT_TIMEOUT_MS, Math.max(1000, Math.floor(requestedTimeout)))
    : AGENT_TIMEOUT_MS;
  const required = (opts.schema && Array.isArray(opts.schema.required)) ? opts.schema.required : [];
  const contract = `

## OUTPUT CONTRACT (headless automation — read carefully)
You are running UNATTENDED in an automated workflow. Perform ALL work described
above yourself using your Bash / Read / Write / Edit tools (build, run on GPU,
measure, write files). Do not ask for confirmation.
When completely finished, your FINAL message MUST be EXACTLY ONE JSON object and
NOTHING else: no markdown code fences, no prose, no explanation before or after.
${required.length ? `It MUST include at least these keys: ${required.join(', ')}.` : ''}
Conform to the JSON shape the role file specifies.`;

  let contractAttempts = 0;   // reachable agent, but the output contract was broken
  let outageAttempts = 0;     // the agent CLI never reached the backend
  for (let attempt = 0; ; attempt++) {
    const t0 = Date.now();
    const fullPrompt = prompt + contract + (contractAttempts > 0
      ? `\n\n[RETRY ${contractAttempts}] Your previous output was not a single valid JSON object with the required keys. Output ONLY the JSON object now.`
      : '');
    const r = await spawnClaude(fullPrompt, label, timeoutMs, required);
    const dt = ((Date.now() - t0) / 1000).toFixed(0);

    // stream-json: the final text is the `result` event (captured in spawnClaude);
    // fall back to the raw stream only if no result event arrived.
    const resultText = r.resultText != null ? r.resultText : r.raw;
    const parsed = extractJson(resultText);

    const logFile = `${SHIM_LOGS}/${ts()}__${slug(label)}__try${attempt}.json`;
    try {
      writeFileSync(logFile, JSON.stringify({
        label, attempt, ok: r.ok, code: r.code, seconds: Number(dt),
        cost_usd: r.costUsd, tool_calls: r.nTools, parsed_ok: !!parsed,
        early_finish: !!r.early,
        result_text: resultText, raw_tail: (r.raw || '').slice(-4000),
        stderr_tail: (r.err || '').slice(-2000),
      }, null, 2));
    } catch { /* */ }

    const missing = required.filter((k) => !parsed || !(k in parsed));
    if (parsed && missing.length === 0) {
      globalThis.log(`✓ ${label} (${dt}s, ${r.nTools} tools${typeof r.costUsd === 'number' ? `, $${r.costUsd.toFixed(3)}` : ''})`);
      AGENT_FAILURES.delete(label);
      return parsed;
    }
    const why = !parsed ? 'no JSON parsed' : `missing keys: ${missing.join(',')}`;
    const outage = isTransportFailure(r, parsed);
    const timeout = isAgentTimeout(r);
    if (outage && outageAttempts < AGENT_TRANSPORT_RETRIES) {
      outageAttempts++;
      const wait = transportBackoffMs(
        outageAttempts, AGENT_TRANSPORT_BACKOFF_MS, AGENT_TRANSPORT_BACKOFF_CAP_MS);
      globalThis.log(`✗ ${label} (${dt}s, ${r.nTools} tools) — agent backend unreachable; `
        + `waiting ${Math.round(wait / 1000)}s then retry ${outageAttempts}/${AGENT_TRANSPORT_RETRIES} (see ${logFile})`);
      await sleep(wait);
      continue;
    }
    if (!outage && contractAttempts < AGENT_RETRIES) {
      contractAttempts++;
      globalThis.log(`✗ ${label} (${dt}s, ${r.nTools} tools) — ${
        timeout ? 'agent wall-clock timeout' : why}; retrying (see ${logFile})`);
      continue;
    }
    globalThis.log(`✗ ${label} (${dt}s, ${r.nTools} tools) — ${
      timeout ? 'agent wall-clock timeout' : why}; giving up`
      + `${outage ? ` after ${outageAttempts} backoff retries (agent backend unreachable)` : ''} (see ${logFile})`);
    AGENT_FAILURES.set(label, {
      kind: outage ? 'transport' : (timeout ? 'timeout' : 'contract'),
      detail: timeout ? `agent exceeded ${timeoutMs}ms` : why,
      at: Date.now(),
    });
    return null;
  }
};

// --- transform the workflow source into an injectable async-function body ---
// The PerfSkills Workflow runtime wraps the script body in an async function with
// (args, agent, phase, log, pipeline) as params — that is why the file legally has
// BOTH top-level `import`/`export` AND a top-level `return`. We reproduce that: turn
// the 4 relative `import { ... } from './lib/x'` into `const { ... } = await
// import('file://abs/lib/x')`, strip `export `, then run the rest in an AsyncFunction.
function transformWorkflow(src, scriptPath) {
  const scriptDir = path.dirname(scriptPath);
  let s = src.replace(
    /^[ \t]*import\s+\{([^}]*)\}\s+from\s+['"](\.[^'"]+)['"];?[ \t]*$/gm,
    (_m, names, spec) => {
      const destructure = names.replace(/\bas\b/g, ':');          // `x as y` -> `x : y`
      const url = pathToFileURL(path.resolve(scriptDir, spec)).href;
      return `const {${destructure}} = await import(${JSON.stringify(url)});`;
    });
  // Native Workflow metadata is first, so newer workflow
  // sources load their sibling helpers with dynamic import() after meta. Make
  // those specifiers absolute before injecting the source into AsyncFunction.
  s = s.replace(
    /import\(\s*(['"])(\.[^'"]+)\1\s*\)/g,
    (_m, _quote, spec) => `import(${JSON.stringify(
      pathToFileURL(path.resolve(scriptDir, spec)).href)})`);
  s = s.replace(/^([ \t]*)export\s+/gm, '$1');                     // `export const meta` -> `const meta`
  return s;
}

async function runWorkflow() {
  console.log(`[shim] launching ${CFG.scriptPath}`);
  console.log(`[shim] exp_root=${EXP_ROOT}  shim_logs=${SHIM_LOGS}`);
  console.log(`[shim] claude=${CLAUDE_BIN} HOME=${CLAUDE_HOME} model=${CLAUDE_MODEL || '(default)'}`);
  try {
    const src = readFileSync(CFG.scriptPath, 'utf8');
    const body = transformWorkflow(src, CFG.scriptPath);
    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
    const fn = new AsyncFunction('args', 'agent', 'phase', 'log', 'pipeline', body);
    const result = await fn(globalThis.args, globalThis.agent, globalThis.phase, globalThis.log, globalThis.pipeline);
    const persistedResult = {
      ...(result && typeof result === 'object' ? result : {}),
      _shim_run_id: CFG.run_id || '',
    };
    const summaryPath = `${SHIM_LOGS}/result.json`;
    try { writeFileSync(summaryPath, JSON.stringify(persistedResult, null, 2)); } catch { /* */ }
    console.log('\n[shim] workflow finished. Result contract:');
    console.log(JSON.stringify(persistedResult, null, 2));
    console.log(`[shim] (also saved to ${summaryPath})`);
  } catch (e) {
    console.error(`\n[shim] workflow threw: ${e && e.stack ? e.stack : e}`);
    const summaryPath = `${SHIM_LOGS}/result.json`;
    const failure = {
      engine: 'avo-workflow',
      validation_status: 'workflow_error',
      error: String(e && e.stack ? e.stack : e),
      failed_at: new Date().toISOString(),
      _shim_run_id: CFG.run_id || '',
    };
    try { writeFileSync(summaryPath, JSON.stringify(failure, null, 2)); } catch { /* */ }
    process.exitCode = 1;
  }
}
runWorkflow();
