#!/usr/bin/env node
// Backend CONFORMANCE test — "does this CLI backend actually support GEAK?"
//
// selftest.mjs tests the RUNTIME against a FAKE backend (no CLI needed).
// conformance.mjs tests a REAL backend (codex / cursor / qwen / …) against the
// small set of capabilities GEAK genuinely requires. If every REQUIRED probe
// passes, the backend can drive GEAK: it can run headless one-shot, emit valid
// structured output (incl. enum), execute Bash, and Read/Write files outside cwd.
//
//   node conformance.mjs --profile codex
//   node conformance.mjs --agent cursor --model default
//   node conformance.mjs --fake            # self-check the harness (no real CLI)
//   node conformance.mjs --profile qwen --quick   # skip the concurrency probe
//
// Each probe maps to a COMPAT_findings R-item so a failure is actionable.
// Exit code 0 = CONFORMS, 1 = one or more required probes failed, 2 = usage.

import { mkdtemp, writeFile, readFile, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname, resolve as resolvePath } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRuntime, selectBackend } from './run_workflow.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const GEAK_ROOT = resolvePath(HERE, '..', '..');   // interface/runtime -> repo root

// ---------------------------------------------------------------------------
// CLI parsing (same convention as run_workflow.mjs)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// A perfectly-conformant FAKE backend for --fake: it performs the real tool
// side-effects (reads/writes the referenced files) and emits the requested JSON.
// This lets the conformance harness itself be exercised with no CLI/network, and
// doubles as a regression check that the probes + assertions are self-consistent.
// It keys off explicit markers embedded in the prompts (READ_PATH=/WRITE_PATH=…)
// that a real model reads as ordinary text.
// ---------------------------------------------------------------------------
function makeFakeConformantBackend() {
  return {
    name: 'fake',
    async runAgent({ prompt }) {
      const echo = prompt.match(/reply with exactly this token:\s*(\S+)/);
      if (echo) return { text: echo[1] };

      const w = prompt.match(/WRITE_PATH=(\S+)[\s\S]*?WRITE_CONTENT=(\S+)/);
      if (w) { await writeFile(w[1], w[2]); return { text: '```json\n{"written":true}\n```' }; }

      const r = prompt.match(/READ_PATH=(\S+)/);
      if (r) { const c = await readFile(r[1], 'utf8'); return { text: '```json\n' + JSON.stringify({ content: c }) + '\n```' }; }

      if (prompt.includes('STATUS_PROBE')) {
        const cm = prompt.match(/"count"\s*=\s*(\d+)/);   // P5 requests count=<i>; P2 has no "=" -> 42
        const count = cm ? parseInt(cm[1], 10) : 42;
        return { text: `\`\`\`json\n{"status":"ready","count":${count}}\n\`\`\`` };
      }
      return { text: 'ok' };
    },
  };
}

// ---------------------------------------------------------------------------
// Probes. Each returns { ok, detail }. `required` probes gate the verdict.
// ---------------------------------------------------------------------------
let nc = 0;
const nonce = (tag) => `GEAK_${tag}_${process.pid}_${nc++}`;

const STATUS_SCHEMA = {
  type: 'object', required: ['status', 'count'],
  properties: { status: { type: 'string', enum: ['ready', 'error'] }, count: { type: 'integer' } },
};
const CONTENT_SCHEMA = { type: 'object', required: ['content'], properties: { content: { type: 'string' } } };
const WRITTEN_SCHEMA = { type: 'object', required: ['written'], properties: { written: { type: 'boolean' } } };

function buildProbes(rt, ctx) {
  return [
    {
      id: 'P1-headless-text', rItem: 'R2', required: true,
      what: 'headless one-shot run + clean stdout',
      run: async () => {
        const tok = nonce('ECHO');
        const text = await rt.agent(
          `This is a conformance probe. Do NOT use any tools. reply with exactly this token: ${tok} and nothing else.`,
          { label: 'P1' });
        const ok = typeof text === 'string' && text.includes(tok);
        return { ok, detail: ok ? 'echoed token' : `expected token not found in: ${String(text).slice(0, 80)}` };
      },
    },
    {
      id: 'P2-structured-enum', rItem: 'R1', required: true,
      what: 'structured output valid + enum + integer',
      run: async () => {
        const obj = await rt.agent(
          `STATUS_PROBE. Return a JSON object with "status" set to the literal "ready" and "count" set to the integer 42.`,
          { label: 'P2', schema: STATUS_SCHEMA });
        const ok = obj && obj.status === 'ready' && obj.count === 42;
        return { ok, detail: ok ? 'status=ready count=42' : `got ${JSON.stringify(obj)}` };
      },
    },
    {
      id: 'P3-bash-read', rItem: 'R2/R3', required: true,
      what: 'executes Bash + reads a nonce it cannot guess (not hallucinating)',
      run: async () => {
        const secret = nonce('READ');
        const path = join(ctx.dir, 'probe_read.txt');
        await writeFile(path, secret);   // secret is NOT in the prompt
        const obj = await rt.agent(
          `Use the Bash tool to run \`cat\` on the file below and return its exact stdout in the JSON field "content". ` +
          `READ_PATH=${path}`,
          { label: 'P3', schema: CONTENT_SCHEMA });
        const ok = obj && String(obj.content).trim() === secret;
        return { ok, detail: ok ? 'round-tripped the secret via Bash' : `got ${JSON.stringify(obj)} want ${secret}` };
      },
    },
    {
      id: 'P4-write-cwd-external', rItem: 'R3/R7', required: true,
      what: 'Write tool + write outside cwd (sandbox permits)',
      run: async () => {
        const secret = nonce('WRITE');
        const path = join(ctx.dir, 'probe_write.txt');
        await rt.agent(
          `Use the Write tool to create a file, then return JSON {"written": true}. ` +
          `WRITE_PATH=${path} WRITE_CONTENT=${secret} — write EXACTLY that content to EXACTLY that path.`,
          { label: 'P4', schema: WRITTEN_SCHEMA });
        let got = null; try { got = (await readFile(path, 'utf8')).trim(); } catch { /* not written */ }
        const ok = got === secret;
        return { ok, detail: ok ? 'file written with expected content' : `file content: ${JSON.stringify(got)}` };
      },
    },
    {
      id: 'P5-concurrent-schema', rItem: 'concurrency', required: !ctx.quick,
      what: '3 structured probes under parallel() (real-CLI concurrency + retry)',
      skip: ctx.quick,
      run: async () => {
        const res = await rt.parallel(Array.from({ length: 3 }, (_, i) => async () => {
          const obj = await rt.agent(
            `STATUS_PROBE ${i}. Return JSON with "status"="ready" and "count"=${i}.`,
            { label: `P5#${i}`, schema: STATUS_SCHEMA });
          return obj && obj.status === 'ready' && obj.count === i;
        }));
        const good = res.filter((x) => x === true).length;
        const ok = good === 3;
        return { ok, detail: `${good}/3 concurrent schema probes valid` };
      },
    },
  ];
}

// ===========================================================================
// CONTRACT AUDIT (static, no CLI) — the "detect drift" half of conformance.
//
// The capability probes above prove the backend can do what GEAK needs TODAY.
// But a "pass" is only honest if GEAK hasn't grown a NEW requirement the probes
// don't cover. These static checks read the actual GEAK sources and fail when
// GEAK's contract drifts beyond what this runtime + probe set support:
//   A-primitive : a new injected global (new Workflow primitive) the runtime
//                 doesn't implement — would ReferenceError on a real run.
//   A-tools     : a role now asks for an unmapped Claude Code tool
//                 (WebFetch/WebSearch/MCP/…) — backend may lack it; no probe.
//   A-forbidden : a script uses Date.now/Math.random/new Date/process
//                 (native forbids these; runtime allows -> silent divergence).
//   A-wording   : a NEW Claude-specific phrase not covered by neutralization
//                 (would mislead a non-claude model) — reported as WARN.
// When one fires, the fix is: handle the new capability AND update the baseline
// constant here so the check goes green again on purpose (never by accident).
// ===========================================================================

const PRIMITIVES = ['agent', 'parallel', 'pipeline', 'workflow', 'phase', 'log', 'require']; // callable injected globals
// Unambiguous Claude Code tool tokens (do NOT collide with English prose, unlike
// "Edit"/"Skill"/"Task"). `mcp__` prefix flags any MCP tool.
const CLAUDE_TOOLS = ['WebFetch', 'WebSearch', 'MultiEdit', 'NotebookEdit', 'TodoWrite', 'Glob', 'SlashCommand', 'ExitPlanMode'];
const MAPPED_TOOLS = new Set(['WebFetch', 'WebSearch']);
// Called identifiers that are undeclared but are JS/Node builtins or control
// keywords — NOT injected primitives. Keeps A-primitive precise.
const BUILTINS = new Set([
  'if', 'for', 'while', 'switch', 'catch', 'return', 'function', 'typeof', 'await', 'new', 'else',
  'do', 'with', 'yield', 'void', 'delete', 'case', 'throw', 'instanceof', 'in', 'of', 'super',
  'JSON', 'Math', 'Object', 'Array', 'Number', 'String', 'Boolean', 'Promise', 'Set', 'Map', 'Date',
  'Error', 'RegExp', 'Symbol', 'BigInt', 'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'Infinity',
  'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'encodeURIComponent', 'decodeURIComponent',
  'require', 'import', 'structuredClone', 'queueMicrotask', 'atob', 'btoa', 'fetch',
  'async', 'get', 'set', 'static', 'constructor', 'as', 'from', 'export', 'default', 'let', 'const', 'var',
]);
// Reviewed today, NOT new primitives — GEAK-internal locals / callback params /
// enum values that the undeclared-call heuristic can't tie back to a declaration
// (higher-order params, string artifacts). Confirmed none is a Workflow primitive
// (the only primitives are agent/parallel/pipeline/workflow/phase/log/require/args/budget).
// Grow this ONLY after eyeballing the source and confirming the name is a local.
const ACK_NONPRIMITIVE = new Set(['fn', 'moe', 'group', 'head', 'rejected', 'incomplete', 'parse', 'not', 'unknown']);
// Claude-specific phrases present today (comments / neutralized). A NEW phrase
// outside this set is surfaced as drift.
const ACK_WORDING = ['structuredoutput', 'invoke the workflow tool', 'background task', 'subagent'];
const WORDING_WATCH = ['structuredoutput', 'ultracode', 'background task', 'invoke the workflow tool', 'subagent', 'effort tier'];

async function listFiles(dir, ext) {
  try { return (await readdir(dir)).filter((f) => f.endsWith(ext)).map((f) => join(dir, f)); }
  catch { return []; }
}

// Strip comments + string/template literals so heuristics see CODE, not prose.
function stripCommentsAndStrings(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')                 // block comments
    .replace(/(^|[^:])\/\/[^\n]*/g, '$1 ')             // line comments (keep http://)
    .replace(/`(?:\\[\s\S]|[^`\\])*`/g, ' `` ')        // template literals (drops prompt prose)
    .replace(/'(?:\\.|[^'\\])*'/g, ' "" ')             // single-quoted
    .replace(/"(?:\\.|[^"\\])*"/g, ' "" ');            // double-quoted
}

function collectDeclared(code) {
  const names = new Set();
  let m;
  const add = (raw) => {
    let n = raw.trim().replace(/^\.\.\./, '');
    n = n.split('=')[0].split(':').pop().trim();       // strip default value / destructure rename
    if (/^[A-Za-z_$][\w$]*$/.test(n)) names.add(n);
  };
  // named declarations
  for (const re of [
    /\bfunction\s+([A-Za-z_$][\w$]*)/g,
    /\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)/g,
    /\bclass\s+([A-Za-z_$][\w$]*)/g,
    /\bcatch\s*\(\s*([A-Za-z_$][\w$]*)/g,
  ]) while ((m = re.exec(code))) names.add(m[1]);
  // destructuring: const {a,b}= / const [a,b]=
  for (const re of [/\b(?:const|let|var)\s*\{([^}]*)\}/g, /\b(?:const|let|var)\s*\[([^\]]*)\]/g])
    while ((m = re.exec(code))) for (const p of m[1].split(',')) if (p.trim()) add(p);
  // arrow params (parenthesized): (a, b) =>
  const ap = /\(([^()]*)\)\s*=>/g;
  while ((m = ap.exec(code))) for (const p of m[1].split(',')) if (p.trim()) add(p);
  // function params: function name?(a, b)
  const fp = /\bfunction\b[^(]*\(([^()]*)\)/g;
  while ((m = fp.exec(code))) for (const p of m[1].split(',')) if (p.trim()) add(p);
  // method / object-shorthand / class-method definitions: NAME(params) {
  const md = /(?:^|[;{},\n])\s*(?:async\s+)?(?:get\s+|set\s+|static\s+)?([A-Za-z_$][\w$]*)\s*\([^()]*\)\s*\{/g;
  while ((m = md.exec(code))) names.add(m[1]);
  // single-arg arrows without parens: x =>
  const sa = /(?:^|[(,=:?&|]|\breturn\b|=>)\s*([A-Za-z_$][\w$]*)\s*=>/g;
  while ((m = sa.exec(code))) names.add(m[1]);
  return names;
}

function collectCalled(code) {
  const called = new Set();
  const re = /(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(/g;   // NAME( not preceded by . or word char
  let m; while ((m = re.exec(code))) called.add(m[1]);
  return called;
}

async function auditContract(root) {
  const wfDirs = ['kernel_workflow', 'e2e_workflow'].map((d) => join(root, d));
  const scriptFiles = [];
  for (const d of wfDirs) scriptFiles.push(...await listFiles(d, '.js'));
  const roleFiles = [];
  for (const d of wfDirs) roleFiles.push(...await listFiles(join(d, 'roles'), '.md'));

  if (!scriptFiles.length) {
    return [{ id: 'A0-sources', required: false, ok: null, rItem: '-', detail: `no GEAK sources under ${root} (skipped)` }];
  }

  const scriptSrc = (await Promise.all(scriptFiles.map((f) => readFile(f, 'utf8')))).join('\n');
  const roleSrc = (await Promise.all(roleFiles.map((f) => readFile(f, 'utf8')))).join('\n');
  const code = stripCommentsAndStrings(scriptSrc);
  const rows = [];

  // A-primitive: undeclared called identifiers that aren't builtins or known
  // primitives. Declarations are collected from the RAW source (so fragile
  // string-stripping can never make us "lose" a real declaration); calls are
  // collected from the STRIPPED source (so prose inside prompt strings isn't
  // mistaken for a call).
  const declared = collectDeclared(scriptSrc);
  const called = collectCalled(code);
  const novel = [...called].filter((n) =>
    !declared.has(n) && !BUILTINS.has(n) && !PRIMITIVES.includes(n) && !ACK_NONPRIMITIVE.has(n)
    && !n.includes('_'));   // Workflow primitives are lowerCamelCase; snake_case = a skill/path/local, not a primitive
  rows.push({
    id: 'A-primitive', required: true, rItem: 'runtime',
    ok: novel.length === 0,
    detail: novel.length === 0 ? `only known primitives called (${PRIMITIVES.join(',')})`
      : `NEW injected global(s) not implemented by runtime: ${novel.join(', ')} — implement in run_workflow.mjs + add a probe`,
  });

  // A-tools: unambiguous Claude tool tokens in role prompts (raw, incl. prose is fine —
  // these tokens don't appear as prose) + any mcp__ tool.
  const toolHits = new Set();
  for (const t of CLAUDE_TOOLS) if (!MAPPED_TOOLS.has(t) && (new RegExp(`\\b${t}\\b`).test(roleSrc) || new RegExp(`\\b${t}\\b`).test(scriptSrc))) toolHits.add(t);
  if (/\bmcp__/.test(roleSrc) || /\bmcp__/.test(scriptSrc)) toolHits.add('mcp__*');
  rows.push({
    id: 'A-tools', required: true, rItem: 'R3/cap',
    ok: toolHits.size === 0,
    detail: toolHits.size === 0 ? 'all role tools are native or explicitly mapped'
      : `role(s) now require tool(s) beyond Read/Write/Bash: ${[...toolHits].join(', ')} — ensure the backend has it + add a probe`,
  });

  // A-forbidden: sandbox-forbidden constructs in CODE (comments/strings stripped)
  const forbidden = [];
  for (const [name, re] of [['Date.now', /\bDate\.now\b/], ['Math.random', /\bMath\.random\b/],
    ['new Date', /\bnew\s+Date\b/], ['process.', /\bprocess\./]]) {
    if (re.test(code)) forbidden.push(name);
  }
  rows.push({
    id: 'A-forbidden', required: true, rItem: 'parity',
    ok: forbidden.length === 0,
    detail: forbidden.length === 0 ? 'no Date.now/Math.random/new Date/process in script code'
      : `script uses native-forbidden construct(s): ${forbidden.join(', ')} — breaks native parity/resume`,
  });

  // A-wording: NEW Claude-specific phrase not acknowledged (WARN, not required)
  const lc = (scriptSrc + '\n' + roleSrc).toLowerCase();
  const newWording = WORDING_WATCH.filter((w) => lc.includes(w) && !ACK_WORDING.includes(w));
  rows.push({
    id: 'A-wording', required: false, rItem: 'neutralize',
    ok: newWording.length === 0,
    detail: newWording.length === 0 ? 'no new Claude-specific wording'
      : `NEW Claude wording (add a NEUTRALIZE_RULES entry): ${newWording.join(', ')}`,
  });

  return rows;
}

// ---------------------------------------------------------------------------
async function main() {
  const opts = parseArgv(process.argv.slice(2));
  const useFake = !!opts.fake;
  const quick = !!opts.quick;
  const auditOnly = !!opts['audit-only'];
  const geakRoot = opts['geak-root'] && opts['geak-root'] !== true ? resolvePath(opts['geak-root']) : GEAK_ROOT;
  const perProbeTimeoutMs = opts['agent-timeout-ms'] && opts['agent-timeout-ms'] !== true
    ? parseInt(opts['agent-timeout-ms'], 10) : 300000;   // 5min/probe — trivial tasks
  const log = (m) => process.stderr.write(`[conformance] ${m}\n`);

  const rows = [];

  // --- Contract audit (static, always runs, no CLI needed) ----------------
  log(`contract audit @ ${geakRoot}`);
  rows.push(...await auditContract(geakRoot));

  // --- Capability probes (need a real/fake backend; skipped in --audit-only)
  let comboLabel = 'audit-only';
  if (!auditOnly) {
    let backend;
    if (useFake) {
      backend = makeFakeConformantBackend();
      comboLabel = 'fake';
    } else {
      if (!opts.profile && !opts.agent && !process.env.GEAK_AGENT_PROFILE && !process.env.GEAK_AGENT_BACKEND) {
        console.error('usage: conformance.mjs (--profile <name> | --agent <name> [--model <name>] | --fake | --audit-only) [--quick] [--geak-root DIR] [--agent-timeout-ms N]');
        process.exit(2);
      }
      backend = await selectBackend(opts);
      comboLabel = `${backend.resolved.agentName}${backend.resolved.modelName ? '/' + backend.resolved.modelName : ''}`;
    }

    const dir = await mkdtemp(join(tmpdir(), 'geak-conformance-'));
    const rt = createRuntime({ backend, concurrency: 4, agentTimeoutMs: perProbeTimeoutMs, log });
    log(`backend=${comboLabel} tmp=${dir}${quick ? ' (quick)' : ''}`);
    for (const p of buildProbes(rt, { dir, quick })) {
      if (p.skip) { rows.push({ ...p, ok: null, detail: 'skipped (--quick)' }); log(`SKIP ${p.id}`); continue; }
      log(`RUN  ${p.id} — ${p.what}`);
      let ok = false, detail = '';
      try { ({ ok, detail } = await p.run()); }
      catch (e) { ok = false; detail = `threw: ${String(e && e.message || e).slice(0, 160)}`; }
      rows.push({ ...p, ok, detail });
      log(`${ok ? 'PASS' : 'FAIL'} ${p.id}: ${detail}`);
    }
    await rm(dir, { recursive: true, force: true });
  }

  // Report
  const pad = (s, n) => String(s).padEnd(n);
  console.log(`\n=== GEAK backend conformance: ${comboLabel} ===`);
  console.log(`${pad('probe', 22)} ${pad('req', 4)} ${pad('R-item', 10)} result  detail`);
  for (const r of rows) {
    const res = r.ok === null ? 'SKIP' : r.ok ? 'PASS' : 'FAIL';
    console.log(`${pad(r.id, 22)} ${pad(r.required ? 'yes' : 'no', 4)} ${pad(r.rItem, 10)} ${pad(res, 6)}  ${r.detail}`);
  }

  const warns = rows.filter((r) => !r.required && r.ok === false);
  if (warns.length) console.log(`\nWARN: ${warns.map((w) => `${w.id} (${w.detail})`).join('; ')}`);

  const failed = rows.filter((r) => r.required && r.ok === false);
  if (failed.length === 0) {
    console.log(`\nCONFORMS ✅  — ${comboLabel}: all required checks passed (contract audit + capability probes).`);
    process.exit(0);
  } else {
    console.log(`\nDOES NOT CONFORM ❌  — ${failed.length} required check(s) failed: ${failed.map((f) => `${f.id}[${f.rItem}]`).join(', ')}`);
    console.log(`Handle the new capability, add a probe, then update the reviewed baseline in conformance.mjs.`);
    process.exit(1);
  }
}

main().catch((e) => { console.error(`CONFORMANCE CRASH: ${e && e.stack || e}`); process.exit(1); });
