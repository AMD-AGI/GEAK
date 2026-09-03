#!/usr/bin/env node
// Self-test for the standalone runtime primitives + schema emulation.
// Runs WITHOUT any real agent CLI, network, or GPU — uses a fake backend.
//
//   node interface/runtime/selftest.mjs
//
// Exits non-zero on the first failed assertion.

import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRuntime } from './run_workflow.mjs';
import { extractJson, validate } from './schema.mjs';
import { resolveSelection, buildInvocation, neutralizeForBackend, loadRegistry } from './config.mjs';

let passed = 0;
const fails = [];
function ok(cond, msg) {
  if (cond) { passed++; }
  else { fails.push(msg); console.error(`  FAIL: ${msg}`); }
}
const eq = (a, b, msg) => ok(JSON.stringify(a) === JSON.stringify(b), `${msg} (got ${JSON.stringify(a)}, want ${JSON.stringify(b)})`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const silent = () => {};

// A fake backend whose response is driven by a directive embedded in the prompt.
// It also tracks max observed concurrency so we can assert the semaphore cap.
function makeFakeBackend() {
  const b = { name: 'fake', inFlight: 0, maxInFlight: 0, calls: 0 };
  b.runAgent = async ({ prompt }) => {
    b.calls++;
    b.inFlight++;
    b.maxInFlight = Math.max(b.maxInFlight, b.inFlight);
    try {
      await sleep(15);
      // Directives: FAKE_TEXT:<s> | FAKE_JSON:<json> | FAKE_BADJSON | FAKE_THROW
      if (prompt.includes('FAKE_THROW')) throw new Error('boom');
      if (prompt.includes('FAKE_BADJSON')) return { text: 'no json here at all' };
      const jm = prompt.match(/FAKE_JSON:(\{.*?\})\s*(?:\n|$)/s);
      if (jm) return { text: 'blah blah\n```json\n' + jm[1] + '\n```\ndone' };
      const tm = prompt.match(/FAKE_TEXT:(.*)$/m);
      return { text: tm ? tm[1] : 'ok' };
    } finally {
      b.inFlight--;
    }
  };
  return b;
}

async function testSchemaUnit() {
  eq(extractJson('x ```json\n{"a":1}\n``` y'), { a: 1 }, 'extractJson fenced');
  eq(extractJson('prefix {"a":{"b":2}} suffix'), { a: { b: 2 } }, 'extractJson balanced');
  eq(extractJson('```json\n{"a":1}\n```\n```json\n{"a":2}\n```'), { a: 2 }, 'extractJson takes LAST fenced');
  let threw = false; try { extractJson('nothing'); } catch { threw = true; }
  ok(threw, 'extractJson throws on no-json');

  const sch = { type: 'object', required: ['id', 'items'], properties: { id: { type: 'string' }, items: { type: 'array', items: { type: 'object', required: ['x'], properties: { x: { type: 'number' } } } } } };
  ok(validate({ id: 'a', items: [{ x: 1 }] }, sch).ok, 'validate ok');
  ok(!validate({ items: [] }, sch).ok, 'validate missing required');
  ok(!validate({ id: 5, items: [] }, sch).ok, 'validate wrong type');
  ok(!validate({ id: 'a', items: [{ y: 1 }] }, sch).ok, 'validate nested required');
  ok(validate(2, { type: 'integer' }).ok, 'validate integer accepts whole number');
  ok(!validate(1.5, { type: 'integer' }).ok, 'validate integer rejects fractional number');

  // enum: GEAK branches on exact enum strings — out-of-enum must be rejected
  // (parity with native's forced StructuredOutput tool).
  const esch = { type: 'object', required: ['outcome'], properties: { outcome: { type: 'string', enum: ['have_winner', 'no_win', 'tamper'] } } };
  ok(validate({ outcome: 'have_winner' }, esch).ok, 'validate enum accepts allowed value');
  ok(!validate({ outcome: 'won' }, esch).ok, 'validate enum rejects out-of-enum value');
  ok(validate('memory', { type: 'string', enum: ['memory', 'compute'] }).ok, 'validate enum at top level ok');
  ok(!validate('gpu', { type: 'string', enum: ['memory', 'compute'] }).ok, 'validate enum at top level reject');
}

async function testParallel() {
  const b = makeFakeBackend();
  const rt = createRuntime({ backend: b, concurrency: 3, log: silent });
  const res = await rt.parallel([
    () => rt.agent('FAKE_TEXT:one'),
    () => rt.agent('FAKE_THROW'),           // -> null, does not reject the batch
    () => rt.agent('FAKE_TEXT:three'),
  ]);
  eq(res, ['one', null, 'three'], 'parallel returns per-thunk, throw->null');
  ok(b.maxInFlight <= 3, `semaphore cap respected (max ${b.maxInFlight} <= 3)`);
}

async function testConcurrencyCap() {
  const b = makeFakeBackend();
  const rt = createRuntime({ backend: b, concurrency: 2, log: silent });
  await rt.parallel(Array.from({ length: 8 }, (_, i) => () => rt.agent(`FAKE_TEXT:${i}`)));
  ok(b.maxInFlight <= 2, `concurrency cap=2 respected (max ${b.maxInFlight})`);
  eq(b.calls, 8, 'all 8 agents ran under cap');
}

async function testPipeline() {
  const b = makeFakeBackend();
  const rt = createRuntime({ backend: b, concurrency: 4, log: silent });
  const seenArgs = [];
  const res = await rt.pipeline(
    ['a', 'b', 'c'],
    (prev, item, idx) => { seenArgs.push([prev, item, idx]); return prev === 'b' ? Promise.reject(new Error('drop b')) : prev + '1'; },
    (prev, item, idx) => `${prev}-${item}-${idx}`,
  );
  eq(res, ['a1-a-0', null, 'c1-c-2'], 'pipeline per-item, stage throw drops item to null');
  // stage1 prevResult === item for a,b,c ; stage2 skipped for b
  ok(seenArgs.some(([p, it]) => p === 'a' && it === 'a'), 'pipeline stage1 prev===item');
}

async function testSchemaAgent() {
  const b = makeFakeBackend();
  const rt = createRuntime({ backend: b, concurrency: 2, schemaRetries: 2, log: silent });
  const sch = { type: 'object', required: ['ok'], properties: { ok: { type: 'boolean' } } };
  const good = await rt.agent('FAKE_JSON:{"ok":true}\n', { schema: sch });
  eq(good, { ok: true }, 'agent schema returns parsed object');

  let threw = false;
  const before = b.calls;
  try { await rt.agent('FAKE_BADJSON', { schema: sch }); } catch { threw = true; }
  ok(threw, 'agent schema throws after retries exhausted');
  eq(b.calls - before, 3, 'schema retried schemaRetries+1 = 3 times');
}

async function testRunScriptAndNesting() {
  const dir = await mkdtemp(join(tmpdir(), 'geak-selftest-'));
  const child = join(dir, 'child.js');
  const parent = join(dir, 'parent.js');
  await writeFile(child, [
    "export const meta = { name: 'child', description: 'c', phases: [] };",
    "const A = args || {};",
    "const t = await agent('FAKE_TEXT:child-saw-' + A.tag);",
    "return { child: t };",
  ].join('\n'));
  await writeFile(parent, [
    "export const meta = { name: 'parent', description: 'p', phases: [] };",
    "phase('Work');",
    "const path = require('node:path');",
    "const r = await workflow({ scriptPath: '" + child.replace(/\\/g, '\\\\') + "' }, { tag: 'X' });",
    "let nestedBlocked = false;",
    // A nested workflow() call inside the child would throw; prove one level works
    // and that the parent's own second-level guard triggers if it recursed again.
    "return { fromChild: r, required: path.basename('/a/b'), ok: true };",
  ].join('\n'));

  const b = makeFakeBackend();
  const rt = createRuntime({ backend: b, concurrency: 4, log: silent });
  const out = await rt.runScript(parent, {}, 0);
  eq(out, { fromChild: { child: 'child-saw-X' }, required: 'b', ok: true }, 'runScript loads (export-stripped) + scoped require + top-level return + one-level workflow() nesting');

  // Second-level nesting must throw: call workflow() at depth 1.
  let nestThrew = false;
  try { await rt.makeWorkflow(1)({ scriptPath: child }, {}); } catch { nestThrew = true; }
  ok(nestThrew, 'workflow() nesting beyond one level throws');

  await rm(dir, { recursive: true, force: true });
}

async function testAgentCap() {
  // Not exhaustively (1000 spawns is slow); just confirm the counter increments.
  const b = makeFakeBackend();
  const rt = createRuntime({ backend: b, concurrency: 8, log: silent });
  await rt.parallel(Array.from({ length: 5 }, (_, i) => () => rt.agent(`FAKE_TEXT:${i}`)));
  eq(rt.state.spawned, 5, 'agent spawn counter increments');
}

async function testConfig() {
  const reg = {
    default_profile: 'claude',
    agents: {
      claude: { bin: 'claude', prompt: 'stdin', args: ['-p'], model_flag: '--model', base_url_env: 'ANTHROPIC_BASE_URL', env: { IS_SANDBOX: '1' }, dialect: 'anthropic' },
      codex: { bin: 'codex', prompt: 'arg', args: ['exec', '--sandbox', 'workspace-write', '--ephemeral', '--ignore-rules'], approve: '', model_flag: '-m', base_url_env: 'OPENAI_BASE_URL', dialect: 'openai', provider_autoconfig: 'codex', extra_args_env: 'GEAK_CODEX_EXTRA_ARGS',
        provider_autoselect: [
          { trigger_env: 'OPENAI_API_KEY', base_url: 'https://api.openai.com/v1', key_env: 'OPENAI_API_KEY' },
        ] },
      hermes: { bin: 'hermes', prompt: 'arg', args: ['--safe-mode', '--toolsets', 'terminal,file,web'], prompt_flag: '-z', model_flag: '-m', external_container_env: 'GEAK_HERMES_EXTERNAL_SANDBOX', dialect: 'openai' },
      qwen: { bin: 'qwen', prompt: 'stdin', args: ['-p'], approve: '--yolo', model_flag: '-m', base_url_env: 'OPENAI_BASE_URL', dialect: 'openai' },
    },
    models: { qc: { id: 'Qwen3-Coder', base_url: 'http://ep:8000/v1', key_env: 'OPENAI_API_KEY' } },
    profiles: { claude: { agent: 'claude' }, qwen: { agent: 'qwen', model: 'qc' } },
  };

  // resolveSelection precedence
  eq(resolveSelection(reg, {}).agentName, 'claude', 'resolve default_profile');
  eq(resolveSelection(reg, { profile: 'qwen' }).modelName, 'qc', 'resolve profile supplies model');
  eq(resolveSelection(reg, { profile: 'qwen', model: null, agent: 'codex' }).agentName, 'codex', 'agent overrides profile agent');
  let threw = false; try { resolveSelection(reg, { agent: 'nope' }); } catch { threw = true; }
  ok(threw, 'resolveSelection throws on unknown agent');

  // buildInvocation: stdin agent with model endpoint
  const q = resolveSelection(reg, { profile: 'qwen' });
  const invQ = buildInvocation(q.agent, q.model, 'PROMPT', { env: {} });
  eq(invQ.cmd, 'qwen', 'buildInvocation cmd');
  ok(invQ.args.includes('--yolo'), 'buildInvocation approve flag');
  ok(invQ.args.includes('-m') && invQ.args.includes('Qwen3-Coder'), 'buildInvocation model flag+id');
  eq(invQ.env.OPENAI_BASE_URL, 'http://ep:8000/v1', 'buildInvocation routes base_url');
  eq(invQ.promptOnStdin, true, 'buildInvocation stdin delivery');

  // buildInvocation: arg-delivery agent puts prompt last, stdin off
  const c = resolveSelection(reg, { agent: 'codex' });
  const invC = buildInvocation(c.agent, null, 'PROMPT_TEXT', { env: {} });
  eq(invC.promptOnStdin, false, 'codex prompt via arg');
  eq(invC.args[invC.args.length - 1], 'PROMPT_TEXT', 'codex prompt appended as last arg');

  // Hermes writable tools require both an explicit declaration and concrete
  // container detection. Neither half may authorize the other by itself.
  const h = resolveSelection(reg, { agent: 'hermes' });
  let missingDeclaration = false;
  try { buildInvocation(h.agent, null, 'P', { env: {}, containerDetected: true }); }
  catch { missingDeclaration = true; }
  ok(missingDeclaration, 'Hermes rejects container without external-sandbox declaration');
  let missingContainer = false;
  try { buildInvocation(h.agent, null, 'P', { env: { GEAK_HERMES_EXTERNAL_SANDBOX: '1' }, containerDetected: false }); }
  catch { missingContainer = true; }
  ok(missingContainer, 'Hermes rejects declaration without container detection');
  const invH = buildInvocation(h.agent, null, 'P', { env: { GEAK_HERMES_EXTERNAL_SANDBOX: '1' }, containerDetected: true });
  ok(invH.args.includes('--safe-mode'), 'Hermes uses safe mode');
  ok(invH.args.includes('terminal,file,web'), 'Hermes exposes the workflow-required web capability');
  ok(!invH.args.includes('--yolo'), 'Hermes never uses yolo');
  eq(invH.args[invH.args.length - 2], '-z', 'Hermes prompt flag precedes prompt');
  eq(invH.args[invH.args.length - 1], 'P', 'Hermes prompt appended as final arg');

  // codex provider auto-config: helper reads the value of a `-c key=value` override
  const cval = (args, key) => {
    for (let i = 0; i < args.length - 1; i++) {
      if (args[i] === '-c' && args[i + 1].startsWith(key + '=')) return args[i + 1].slice(key.length + 1);
    }
    return undefined;
  };
  // (a) base_url-driven: explicit OPENAI_BASE_URL -> geak_auto provider + that base_url
  const invAuto = buildInvocation(c.agent, null, 'P', { env: { OPENAI_BASE_URL: 'https://api.openai.com/v1' } });
  eq(cval(invAuto.args, 'model_provider'), '"geak_auto"', 'autoconfig sets model_provider');
  eq(cval(invAuto.args, 'model_providers.geak_auto.base_url'), '"https://api.openai.com/v1"', 'autoconfig base_url from OPENAI_BASE_URL');
  eq(cval(invAuto.args, 'model_providers.geak_auto.wire_api'), '"responses"', 'autoconfig wire_api=responses');
  // (b) a localhost OpenAI-compatible endpoint is configured like any other explicit endpoint
  const invLocal = buildInvocation(c.agent, null, 'P', { env: { OPENAI_BASE_URL: 'http://127.0.0.1:8791/v1' } });
  eq(cval(invLocal.args, 'model_providers.geak_auto.base_url'), '"http://127.0.0.1:8791/v1"', 'autoconfig accepts local OpenAI-compatible endpoint');
  // (c) key-driven auto-select: OPENAI_API_KEY -> OpenAI official
  const invOai = buildInvocation(c.agent, null, 'P', { env: { OPENAI_API_KEY: 'sk-x' } });
  eq(cval(invOai.args, 'model_providers.geak_auto.base_url'), '"https://api.openai.com/v1"', 'auto-select OPENAI_API_KEY -> OpenAI official');
  // (d) disabled via GEAK_CODEX_AUTOCONFIG=0
  const invOff = buildInvocation(c.agent, null, 'P', { env: { OPENAI_BASE_URL: 'https://api.openai.com/v1', GEAK_CODEX_AUTOCONFIG: '0' } });
  eq(cval(invOff.args, 'model_provider'), undefined, 'autoconfig disabled by GEAK_CODEX_AUTOCONFIG=0');
  // (e) caller pins model_provider via extra args -> skip autoconfig
  const invPin = buildInvocation(c.agent, null, 'P', { env: { OPENAI_BASE_URL: 'https://api.openai.com/v1', GEAK_CODEX_EXTRA_ARGS: '-c model_provider=pinned_provider' } });
  eq(cval(invPin.args, 'model_provider'), 'pinned_provider', 'extra_args model_provider wins over autoconfig');
  // codex thinking level: default max, GEAK_CODEX_EFFORT override, extra_args pin not double-emitted
  eq(cval(invOai.args, 'model_reasoning_effort'), 'max', 'codex effort defaults to max');
  const invEff = buildInvocation(c.agent, null, 'P', { env: { OPENAI_API_KEY: 'sk-x', GEAK_CODEX_EFFORT: 'high' } });
  eq(cval(invEff.args, 'model_reasoning_effort'), 'high', 'GEAK_CODEX_EFFORT overrides default');
  const invEffPin = buildInvocation(c.agent, null, 'P', { env: { OPENAI_API_KEY: 'sk-x', GEAK_CODEX_EXTRA_ARGS: '-c model_reasoning_effort=low' } });
  eq(invEffPin.args.filter((a) => a.startsWith('model_reasoning_effort=')).length, 1, 'effort not double-emitted when pinned via extra_args');
  eq(cval(invEffPin.args, 'model_reasoning_effort'), 'low', 'extra_args effort wins');

  // neutralizeForBackend
  const p = 'Return ONLY the structured JSON (a StructuredOutput tool is forced).';
  ok(neutralizeForBackend(p, 'claude') === p, 'neutralize no-op for claude');
  ok(!/StructuredOutput tool is forced/.test(neutralizeForBackend(p, 'qwen')), 'neutralize strips Claude wording for non-claude');
  const webPrompt = 'Use WebSearch, then WebFetch.';
  ok(!/WebSearch|WebFetch/.test(neutralizeForBackend(webPrompt, 'hermes')), 'neutralize maps Claude web tool names');

  // real registry.json loads + resolves the shipped profiles
  const real = await loadRegistry();
  eq(resolveSelection(real, { profile: 'qwen' }).agentName, 'qwen', 'shipped registry: qwen profile');
  eq(resolveSelection(real, { agent: 'codex' }).agentName, 'codex', 'shipped registry: codex agent');
  eq(resolveSelection(real, { agent: 'kimi' }).agentName, 'kimi', 'shipped registry: kimi agent');
  const publicCodex = resolveSelection(real, { profile: 'codex-openai' });
  const publicCodexInvocation = buildInvocation(publicCodex.agent, publicCodex.model, 'P', {
    env: { OPENAI_API_KEY: 'test-only', GEAK_CODEX_MODEL: 'gpt-public' },
  });
  ok(publicCodexInvocation.args.includes('gpt-public'), 'GEAK_CODEX_MODEL outranks profile placeholder');
  ok(!publicCodexInvocation.args.includes('<your-openai-model-id>'), 'profile placeholder never outranks env model');
}

async function main() {
  await testSchemaUnit();
  await testConfig();
  await testParallel();
  await testConcurrencyCap();
  await testPipeline();
  await testSchemaAgent();
  await testRunScriptAndNesting();
  await testAgentCap();

  console.log(`\n${passed} checks passed, ${fails.length} failed.`);
  if (fails.length) process.exit(1);
}

main().catch((e) => { console.error('SELFTEST CRASH:', e); process.exit(1); });
