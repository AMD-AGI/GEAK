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

import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRuntime, selectBackend } from './run_workflow.mjs';

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

// ---------------------------------------------------------------------------
async function main() {
  const opts = parseArgv(process.argv.slice(2));
  const useFake = !!opts.fake;
  const quick = !!opts.quick;
  const perProbeTimeoutMs = opts['agent-timeout-ms'] && opts['agent-timeout-ms'] !== true
    ? parseInt(opts['agent-timeout-ms'], 10) : 300000;   // 5min/probe — trivial tasks

  let backend, comboLabel;
  if (useFake) {
    backend = makeFakeConformantBackend();
    comboLabel = 'fake';
  } else {
    if (!opts.profile && !opts.agent && !process.env.GEAK_AGENT_PROFILE && !process.env.GEAK_AGENT_BACKEND) {
      console.error('usage: conformance.mjs (--profile <name> | --agent <name> [--model <name>]) [--quick] [--fake] [--agent-timeout-ms N]');
      process.exit(2);
    }
    backend = await selectBackend(opts);
    comboLabel = `${backend.resolved.agentName}${backend.resolved.modelName ? '/' + backend.resolved.modelName : ''}`;
  }

  const dir = await mkdtemp(join(tmpdir(), 'geak-conformance-'));
  const log = (m) => process.stderr.write(`[conformance] ${m}\n`);
  const rt = createRuntime({ backend, concurrency: 4, agentTimeoutMs: perProbeTimeoutMs, log });

  log(`backend=${comboLabel} tmp=${dir}${quick ? ' (quick)' : ''}`);
  const probes = buildProbes(rt, { dir, quick });
  const rows = [];
  for (const p of probes) {
    if (p.skip) { rows.push({ ...p, ok: null, detail: 'skipped (--quick)' }); log(`SKIP ${p.id}`); continue; }
    log(`RUN  ${p.id} — ${p.what}`);
    let ok = false, detail = '';
    try { ({ ok, detail } = await p.run()); }
    catch (e) { ok = false; detail = `threw: ${String(e && e.message || e).slice(0, 160)}`; }
    rows.push({ ...p, ok, detail });
    log(`${ok ? 'PASS' : 'FAIL'} ${p.id}: ${detail}`);
  }

  await rm(dir, { recursive: true, force: true });

  // Report
  const pad = (s, n) => String(s).padEnd(n);
  console.log(`\n=== GEAK backend conformance: ${comboLabel} ===`);
  console.log(`${pad('probe', 22)} ${pad('req', 4)} ${pad('R-item', 10)} result  detail`);
  for (const r of rows) {
    const res = r.ok === null ? 'SKIP' : r.ok ? 'PASS' : 'FAIL';
    console.log(`${pad(r.id, 22)} ${pad(r.required ? 'yes' : 'no', 4)} ${pad(r.rItem, 10)} ${pad(res, 6)}  ${r.detail}`);
  }

  const failed = rows.filter((r) => r.required && r.ok === false);
  if (failed.length === 0) {
    console.log(`\nCONFORMS ✅  — ${comboLabel} can drive GEAK (all required probes passed).`);
    process.exit(0);
  } else {
    console.log(`\nDOES NOT CONFORM ❌  — ${failed.length} required probe(s) failed: ${failed.map((f) => `${f.id}[${f.rItem}]`).join(', ')}`);
    console.log(`See interface/runtime/COMPAT_findings.md for the referenced R-items.`);
    process.exit(1);
  }
}

main().catch((e) => { console.error(`CONFORMANCE CRASH: ${e && e.stack || e}`); process.exit(1); });
