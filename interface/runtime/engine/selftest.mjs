#!/usr/bin/env node
// Self-test for the standalone runtime primitives + schema emulation.
// Runs WITHOUT any real agent CLI, network, or GPU — uses a fake backend.
//
//   node interface/runtime/engine/selftest.mjs
//
// Exits non-zero on the first failed assertion.

import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRuntime } from './run_workflow.mjs';
import { extractJson, validate } from './schema.mjs';
import { resolveSelection, buildInvocation, neutralizeForBackend, loadRegistry, deriveAgentFromEnv, codexEffort } from './config.mjs';

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
    "const r = await workflow({ scriptPath: '" + child.replace(/\\/g, '\\\\') + "' }, { tag: 'X' });",
    "let nestedBlocked = false;",
    // A nested workflow() call inside the child would throw; prove one level works
    // and that the parent's own second-level guard triggers if it recursed again.
    "return { fromChild: r, ok: true };",
  ].join('\n'));

  const b = makeFakeBackend();
  const rt = createRuntime({ backend: b, concurrency: 4, log: silent });
  const out = await rt.runScript(parent, {}, 0);
  eq(out, { fromChild: { child: 'child-saw-X' }, ok: true }, 'runScript loads (export-stripped) + top-level return + one-level workflow() nesting');

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
      codex: { bin: 'codex', prompt: 'arg', args: ['exec', '--dangerously-bypass-approvals-and-sandbox'], approve: '', model_flag: '-m', base_url_env: 'OPENAI_BASE_URL', dialect: 'openai', provider_autoconfig: 'codex', extra_args_env: 'GEAK_CODEX_EXTRA_ARGS',
        provider_autoselect: [
          // Synthetic header-carrying provider: the real registry ships none, but
          // env_http_headers stays supported (also reachable via OPENAI_CUSTOM_HEADERS).
          { trigger_env: 'FIXTURE_GW_KEY', base_url: 'https://gw.example.com/v1', key_env: 'FIXTURE_GW_KEY', env_http_headers: { 'X-Gw-Subscription-Key': 'FIXTURE_GW_KEY' } },
          { trigger_env: 'FIXTURE_PLAIN_KEY', base_url: 'https://plain.example.com/v1', key_env: 'FIXTURE_PLAIN_KEY' },
          { trigger_env: 'OPENAI_API_KEY', base_url: 'https://api.openai.com/v1', key_env: 'OPENAI_API_KEY' },
        ] },
      // A SYNTHETIC agent, deliberately not any shipped one: it exercises the
      // config-driven fields the two real agents happen not to combine (stdin
      // delivery + an approve flag + a model endpoint). Keeping it nameless-by-
      // design means trimming the registry never silently deletes coverage.
      stub: { bin: 'stub-cli', prompt: 'stdin', args: ['-p'], approve: '--yolo', model_flag: '-m', base_url_env: 'OPENAI_BASE_URL', dialect: 'openai' },
    },
    models: { sm: { id: 'Stub-Model-1', base_url: 'http://ep:8000/v1', key_env: 'OPENAI_API_KEY' } },
    profiles: { claude: { agent: 'claude' }, stub: { agent: 'stub', model: 'sm' } },
  };

  // resolveSelection precedence
  eq(resolveSelection(reg, {}).agentName, 'claude', 'resolve default_profile');
  eq(resolveSelection(reg, { profile: 'stub' }).modelName, 'sm', 'resolve profile supplies model');
  eq(resolveSelection(reg, { profile: 'stub', model: null, agent: 'codex' }).agentName, 'codex', 'agent overrides profile agent');
  let threw = false; try { resolveSelection(reg, { agent: 'nope' }); } catch { threw = true; }
  ok(threw, 'resolveSelection throws on unknown agent');

  // buildInvocation: stdin agent with model endpoint
  const q = resolveSelection(reg, { profile: 'stub' });
  const invQ = buildInvocation(q.agent, q.model, 'PROMPT', { env: {} });
  eq(invQ.cmd, 'stub-cli', 'buildInvocation cmd');
  ok(invQ.args.includes('--yolo'), 'buildInvocation approve flag');
  ok(invQ.args.includes('-m') && invQ.args.includes('Stub-Model-1'), 'buildInvocation model flag+id');
  eq(invQ.env.OPENAI_BASE_URL, 'http://ep:8000/v1', 'buildInvocation routes base_url');
  eq(invQ.promptOnStdin, true, 'buildInvocation stdin delivery');

  // buildInvocation: arg-delivery agent puts prompt last, stdin off
  const c = resolveSelection(reg, { agent: 'codex' });
  const invC = buildInvocation(c.agent, null, 'PROMPT_TEXT', { env: {} });
  eq(invC.promptOnStdin, false, 'codex prompt via arg');
  eq(invC.args[invC.args.length - 1], 'PROMPT_TEXT', 'codex prompt appended as last arg');

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
  // (b) a localhost base_url is autoconfigured like any other: point OPENAI_BASE_URL
  //     at a local OpenAI-compatible server and it just works, no config.toml needed
  const invLocal = buildInvocation(c.agent, null, 'P', { env: { OPENAI_BASE_URL: 'http://127.0.0.1:8000/v1' } });
  eq(cval(invLocal.args, 'model_providers.geak_auto.base_url'), '"http://127.0.0.1:8000/v1"', 'autoconfig also covers a localhost base_url');
  // (c) key-driven auto-select: a header-carrying provider (no base_url) supplies
  //     its own endpoint, key env AND custom auth header
  const invGw = buildInvocation(c.agent, null, 'P', { env: { FIXTURE_GW_KEY: 'x' } });
  eq(cval(invGw.args, 'model_providers.geak_auto.base_url'), '"https://gw.example.com/v1"', 'auto-select by trigger_env -> that provider base_url');
  eq(cval(invGw.args, 'model_providers.geak_auto.env_key'), '"FIXTURE_GW_KEY"', 'auto-select carries its key_env');
  eq(cval(invGw.args, 'model_providers.geak_auto.env_http_headers.X-Gw-Subscription-Key'), '"FIXTURE_GW_KEY"', 'auto-select carries env_http_headers');
  // (d) key-driven auto-select: only OPENAI_API_KEY -> OpenAI official
  const invOai = buildInvocation(c.agent, null, 'P', { env: { OPENAI_API_KEY: 'sk-x' } });
  eq(cval(invOai.args, 'model_providers.geak_auto.base_url'), '"https://api.openai.com/v1"', 'auto-select OPENAI_API_KEY -> OpenAI official');
  // (e) disabled via GEAK_CODEX_AUTOCONFIG=0
  const invOff = buildInvocation(c.agent, null, 'P', { env: { OPENAI_BASE_URL: 'https://api.openai.com/v1', GEAK_CODEX_AUTOCONFIG: '0' } });
  eq(cval(invOff.args, 'model_provider'), undefined, 'autoconfig disabled by GEAK_CODEX_AUTOCONFIG=0');
  // (f) caller pins model_provider via extra args -> skip autoconfig
  const invPin = buildInvocation(c.agent, null, 'P', { env: { OPENAI_BASE_URL: 'https://api.openai.com/v1', GEAK_CODEX_EXTRA_ARGS: '-c model_provider=my_provider' } });
  eq(cval(invPin.args, 'model_provider'), 'my_provider', 'extra_args model_provider wins over autoconfig');
  // codex thinking level: default xhigh (its true maximum — it has no 'max'),
  // GEAK_CODEX_EFFORT override, extra_args pin not double-emitted
  eq(cval(invOai.args, 'model_reasoning_effort'), 'xhigh', 'codex effort defaults to xhigh');
  const invEff = buildInvocation(c.agent, null, 'P', { env: { OPENAI_API_KEY: 'sk-x', GEAK_CODEX_EFFORT: 'high' } });
  eq(cval(invEff.args, 'model_reasoning_effort'), 'high', 'GEAK_CODEX_EFFORT overrides default');
  const invEffPin = buildInvocation(c.agent, null, 'P', { env: { OPENAI_API_KEY: 'sk-x', GEAK_CODEX_EXTRA_ARGS: '-c model_reasoning_effort=low' } });
  eq(invEffPin.args.filter((a) => a.startsWith('model_reasoning_effort=')).length, 1, 'effort not double-emitted when pinned via extra_args');
  eq(cval(invEffPin.args, 'model_reasoning_effort'), 'low', 'extra_args effort wins');

  // neutralizeForBackend
  const p = 'Return ONLY the structured JSON (a StructuredOutput tool is forced).';
  ok(neutralizeForBackend(p, 'claude') === p, 'neutralize no-op for claude');
  ok(!/StructuredOutput tool is forced/.test(neutralizeForBackend(p, 'codex')), 'neutralize strips Claude wording for non-claude');

  // real registry.json loads + resolves the shipped profiles. The registry ships
  // exactly two agents; an unknown name must fail loudly rather than fall back,
  // so a stale `--agent qwen` in someone's script cannot silently run on claude.
  const real = await loadRegistry();
  eq(resolveSelection(real, { profile: 'codex-gpt56' }).modelName, 'openai_gpt56', 'shipped registry: codex-gpt56 profile pins its model');
  eq(resolveSelection(real, { agent: 'codex' }).agentName, 'codex', 'shipped registry: codex agent');
  let gone = false; try { resolveSelection(real, { agent: 'qwen' }); } catch { gone = true; }
  ok(gone, 'shipped registry: a retired agent name is rejected, not silently ignored');
}

// Configuring a provider key must be enough to land on that key's CLI with no
// flags at all — that is the whole point of the credential-derived path. The
// trigger names must come from the registry's provider_autoselect (so removing a
// provider removes its trigger), and any explicit selection must still win.
async function testCredentialDerivedAgent() {
  const real = await loadRegistry();
  const pick = (env, sel = {}) => resolveSelection(real, { ...sel, env }).agentName;

  eq(pick({}), 'claude', 'no keys: registry default_profile still applies');
  eq(pick({ ANTHROPIC_API_KEY: 'sk-ant-x' }), 'claude', 'an Anthropic-only setup is unchanged');
  eq(pick({ OPENAI_API_KEY: 'sk-x' }), 'codex', 'an OpenAI key alone selects codex — no flags');
  eq(pick({ GEAK_AMDKEY: 'x' }), 'codex', 'an AMD gateway key alone selects codex — no flags');
  eq(pick({ SAFE_API_KEY: 'ak-x' }), 'claude', 'a de-listed key triggers nothing — removing a provider removes its trigger');
  eq(pick({ OPENAI_API_KEY: 'sk-x', GEAK_AGENT_AUTO: '0' }), 'claude', 'GEAK_AGENT_AUTO=0 opts out');
  eq(pick({ OPENAI_API_KEY: 'sk-x' }, { agent: 'claude' }), 'claude', 'an explicit agent outranks the key');
  eq(pick({ OPENAI_API_KEY: 'sk-x' }, { profile: 'claude' }), 'claude', 'an explicit profile outranks the key');
  eq(deriveAgentFromEnv(real, { OPENAI_API_KEY: '   ' }), '', 'a blank key derives nothing');

  // Triggers are registry-driven, not a hardcoded list: a synthetic agent with
  // its own provider_autoselect is selected by its own trigger_env.
  const synthetic = {
    default_profile: 'claude',
    profiles: { claude: { agent: 'claude' } },
    agents: {
      claude: { bin: 'claude' },
      mycli: { bin: 'mycli', provider_autoselect: [{ trigger_env: 'MYCLI_KEY', base_url: 'https://x/v1' }] },
    },
  };
  eq(deriveAgentFromEnv(synthetic, { MYCLI_KEY: 'k' }), 'mycli', 'triggers come from the registry, not a fixed list');
  eq(deriveAgentFromEnv(synthetic, { OPENAI_API_KEY: 'sk-x' }), '', 'a key no agent claims derives nothing');

  // Selection is decided by the SHAPE of the credential environment, not by one
  // key's presence: a trigger only wins when its side is the only one configured.
  // These four shapes are the ones hyperloom pins in test_specialist_codex_backend
  // (openai_only -> codex; anthropic_only / both / unconfigured -> claude).
  const shapes = [
    ['openai_only', { OPENAI_API_KEY: 'sk-x' }, 'codex'],
    ['gateway_only', { GEAK_AMDKEY: 'x' }, 'codex'],
    ['anthropic_only', { ANTHROPIC_API_KEY: 'sk-ant-x' }, 'claude'],
    ['both_configured', { OPENAI_API_KEY: 'sk-x', ANTHROPIC_API_KEY: 'sk-ant-x' }, 'claude'],
    ['gateway_plus_anthropic', { GEAK_AMDKEY: 'x', ANTHROPIC_API_KEY: 'sk-ant-x' }, 'claude'],
    ['unconfigured', {}, 'claude'],
  ];
  for (const [shapeName, env, expected] of shapes) {
    eq(pick(env), expected, `credential shape ${shapeName} runs ${expected}`);
  }
  // Every Anthropic-side variable counts, not just the API key — a subscription
  // token or a bare base_url is just as much a configured Claude deployment.
  for (const v of ['ANTHROPIC_BASE_URL', 'ANTHROPIC_AUTH_TOKEN', 'CLAUDE_CODE_OAUTH_TOKEN']) {
    eq(pick({ GEAK_AMDKEY: 'x', [v]: 'set' }), 'claude', `${v} alone is enough to block the hijack`);
  }
  // The exclusion must not be a hardcoded Anthropic special case.
  const three = {
    default_profile: 'claude',
    profiles: { claude: { agent: 'claude' } },
    agents: {
      claude: { bin: 'claude', credential_env: ['SIDE_A_KEY'] },
      mycli: { bin: 'mycli', provider_autoselect: [{ trigger_env: 'MYCLI_KEY' }] },
    },
  };
  eq(deriveAgentFromEnv(three, { MYCLI_KEY: 'k' }), 'mycli', 'an unambiguous trigger still selects');
  eq(deriveAgentFromEnv(three, { MYCLI_KEY: 'k', SIDE_A_KEY: 'a' }), '',
    'credential_env is read generically, so any agent can block an ambiguous auto-select');
  // Ambiguity blocks only the GUESS; asking for codex explicitly still works.
  eq(pick({ GEAK_AMDKEY: 'x', ANTHROPIC_API_KEY: 'y' }, { agent: 'codex' }), 'codex',
    'an explicit --agent overrides the ambiguity, so the escape hatch stays open');
}

// A gateway serves its own catalog, which codex knows nothing about, so an
// auto-selected provider supplies the model id too. It must stay welded to that
// provider's endpoint: a gateway id 404s anywhere else.
async function testProviderDefaultModel() {
  const real = await loadRegistry();
  const codex = resolveSelection(real, { agent: 'codex' }).agent;
  const modelOf = (env) => {
    const a = buildInvocation(codex, null, 'P', { env }).args;
    const i = a.indexOf('-m');
    return i >= 0 ? a[i + 1] : '';
  };
  const provider = (trigger) => (real.agents.codex.provider_autoselect || [])
    .find((p) => p.trigger_env === trigger);

  eq(modelOf({ GEAK_AMDKEY: 'x' }), provider('GEAK_AMDKEY').default_model,
    "a gateway key alone yields that gateway's default model");
  eq(modelOf({ OPENAI_API_KEY: 'sk-x' }), provider('OPENAI_API_KEY').default_model,
    'the official endpoint supplies its own default too');
  ok((real.agents.codex.provider_autoselect || []).every((p) => p.default_model),
    'every auto-selectable provider declares a default — otherwise codex runs with no -m');
  eq(modelOf({ GEAK_AMDKEY: 'x', GEAK_CODEX_MODEL: 'mine' }), 'mine',
    'GEAK_CODEX_MODEL outranks the provider default');
  eq(modelOf({ GEAK_AMDKEY: 'x', OPENAI_BASE_URL: 'https://other/v1' }), '',
    'an explicitly chosen endpoint gets no inherited default (it would 404)');
  eq(modelOf({ GEAK_AMDKEY: 'x', GEAK_CODEX_AUTOCONFIG: '0' }), '',
    'no auto-config means no auto model');
  eq(modelOf({ SAFE_API_KEY: 'ak-x' }), '',
    'a de-listed key supplies no endpoint and no model');

  // The AMD gateway authenticates on Ocp-Apim-Subscription-Key ONLY: a request
  // carrying just the Bearer is a 401, while the APIM header alone is a 200 on
  // both /responses and /chat/completions (measured 2026-09-09). env_key is still
  // emitted because codex requires one and the extra Bearer is accepted, but the
  // header is the credential that matters — dropping it breaks every request.
  // A distinctive sentinel: a one-character key would collide with ordinary argv
  // text (an '=x' substring also matches '=xhigh'), making the leak check below
  // both flaky and far weaker than it looks.
  const SECRET = 'sk-sentinel-must-not-appear-in-argv';
  const amd = buildInvocation(codex, null, 'P', { env: { GEAK_AMDKEY: SECRET } }).args;
  const cval = (args, k) => {
    for (let i = 0; i < args.length - 1; i++) {
      if (args[i] === '-c' && args[i + 1].startsWith(k + '=')) return args[i + 1].slice(k.length + 1);
    }
    return undefined;
  };
  eq(cval(amd, 'model_providers.geak_auto.env_http_headers.Ocp-Apim-Subscription-Key'), '"GEAK_AMDKEY"',
    'AMD provider carries the APIM header — the only credential the gateway accepts');
  eq(cval(amd, 'model_providers.geak_auto.env_key'), '"GEAK_AMDKEY"', 'AMD provider still declares env_key (codex needs one)');
  // -c values land in argv, which is world-readable via `ps`, so they must name
  // the env var rather than inline the secret.
  ok(!amd.some((a) => String(a).includes(SECRET)), 'the key value never reaches argv');

  // A pinned registry model still wins, and brings its own endpoint. Pinning must
  // beat a live trigger, or a stray key would silently redirect the run.
  const pinned = resolveSelection(real, { agent: 'codex', model: 'openai_official' });
  const argsPinned = buildInvocation(pinned.agent, pinned.model, 'P', { env: { GEAK_AMDKEY: 'x' } }).args;
  eq(argsPinned[argsPinned.indexOf('-m') + 1], pinned.model.id, 'a pinned model outranks the provider default');
  eq(cval(argsPinned, 'model_providers.geak_auto.base_url'), `"${pinned.model.base_url}"`,
    "a pinned model keeps its OWN endpoint even while a gateway key is set");

  // Official OpenAI's suffixless gpt-5.6. Ids are endpoint-local — this one 404s on
  // the AMD gateway and AMD's -sol/-terra/-luna 404 here — so the entry has to drag
  // its own endpoint and Bearer key along instead of inheriting the gateway's.
  const g56 = resolveSelection(real, { profile: 'codex-gpt56' });
  eq(g56.agentName, 'codex', 'codex-gpt56 runs on codex');
  const args56 = buildInvocation(g56.agent, g56.model, 'P', { env: { GEAK_AMDKEY: 'x' } }).args;
  eq(args56[args56.indexOf('-m') + 1], 'gpt-5.6', 'codex-gpt56 pins the suffixless official id');
  eq(cval(args56, 'model_providers.geak_auto.base_url'), '"https://api.openai.com/v1"',
    'gpt-5.6 resolves to api.openai.com, not to whichever gateway key is set');
  eq(cval(args56, 'model_providers.geak_auto.env_key'), '"OPENAI_API_KEY"',
    'the official endpoint authenticates on the Bearer key, not GEAK_AMDKEY');
  eq(cval(args56, 'model_providers.geak_auto.env_http_headers.Ocp-Apim-Subscription-Key'), undefined,
    'no APIM header leaks onto the official endpoint');
  // The mirror image of that locality: a suffixless id as the AMD default would 404
  // on every run — the trap that got the old gpt56 entry deleted.
  ok(!/^gpt-5\.\d+$/.test(provider('GEAK_AMDKEY').default_model),
    'the AMD provider default stays a gateway-local id');

  // codex's scale is none|low|medium|high|xhigh. It has no 'max', so emitting one
  // would hand the CLI a config it rejects; xhigh is the real maximum and 'max'
  // must be translated to it (hyperloom resolve_codex_reasoning_effort does the same).
  eq(codexEffort(undefined), 'xhigh', 'the default thinking level is codex\'s true maximum');
  eq(codexEffort('max'), 'xhigh', "'max' is translated, never passed through");
  eq(codexEffort('MAX'), 'xhigh', 'the alias is case-insensitive');
  eq(codexEffort('medium'), 'medium', 'a level codex knows survives untouched');
  let badEffort = false;
  try { codexEffort('ultra'); } catch { badEffort = true; }
  ok(badEffort, 'an unknown level fails loudly here instead of inside codex');
  const effortOf = (env) => {
    const a = buildInvocation(codex, null, 'P', { env }).args;
    for (let i = 0; i < a.length - 1; i++) {
      if (a[i] === '-c' && a[i + 1].startsWith('model_reasoning_effort=')) return a[i + 1].split('=')[1];
    }
    return undefined;
  };
  eq(effortOf({ GEAK_AMDKEY: 'x' }), 'xhigh', 'argv carries xhigh, not the unsupported max');
  eq(effortOf({ GEAK_AMDKEY: 'x', GEAK_CODEX_EFFORT: 'max' }), 'xhigh', 'GEAK_CODEX_EFFORT=max still means xhigh');
  eq(effortOf({ GEAK_AMDKEY: 'x', GEAK_CODEX_EFFORT: 'low' }), 'low', 'GEAK_CODEX_EFFORT picks a lower level');
  const pinnedEffortArgs = buildInvocation(codex, null, 'P',
    { env: { GEAK_AMDKEY: 'x', GEAK_CODEX_EXTRA_ARGS: '-c model_reasoning_effort=high' } }).args;
  eq(pinnedEffortArgs.filter((a) => String(a).startsWith('model_reasoning_effort=')).length, 1,
    'an effort pinned in extra args is not double-emitted');
  eq(effortOf({ GEAK_AMDKEY: 'x', GEAK_CODEX_EXTRA_ARGS: '-c model_reasoning_effort=high' }), 'high',
    'and the pinned value is the one that survives');
}

async function main() {
  await testSchemaUnit();
  await testConfig();
  await testCredentialDerivedAgent();
  await testProviderDefaultModel();
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
