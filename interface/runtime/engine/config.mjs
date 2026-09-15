// Config-driven agent/model resolution for the standalone runtime.
//
// Two orthogonal axes (see registry.json):
//   agents[]  — how to drive a code-agent CLI (bin, flags, prompt delivery, env). Model-independent.
//   models[]  — an endpoint (id + base_url + key env). CLI-independent.
//   profiles[]— a named (agent, model) combo you can pin.
//
// This module is intentionally pure/deterministic (except loadRegistry, which
// reads the JSON file) so the resolution + invocation logic is unit-testable
// (selftest.mjs) without spawning anything.

import { readFile } from 'node:fs/promises';
import { resolve as resolvePath, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
export const DEFAULT_REGISTRY_PATH = resolvePath(HERE, 'registry.json');

export async function loadRegistry(path = DEFAULT_REGISTRY_PATH) {
  const raw = JSON.parse(await readFile(path, 'utf8'));
  return raw;
}

// Credential-derived agent selection, so that configuring a key is by itself
// enough to run on the CLI that key belongs to — no GEAK_AGENT_BACKEND needed.
// Without this, a key-only setup falls through to registry.default_profile
// (claude), which cannot authenticate against an OpenAI-side deployment at all.
//
// The trigger names are read straight out of each agent's provider_autoselect,
// so the keys that SELECT an agent are exactly the keys that CONFIGURE its
// provider — one list in registry.json, no second copy to drift. Adding or
// removing a provider therefore changes the trigger set with zero code.
//
// Consulted ONLY when nothing was selected explicitly, so --profile / --agent /
// GEAK_AGENT_BACKEND always win. GEAK_AGENT_AUTO=0 opts out entirely.
const AUTO_OFF = new Set(['0', 'false', 'no']);

// Every env var that means "this agent's provider side is configured": its
// declared credential_env plus the trigger names it auto-selects on.
function sideEnvNames(agent) {
  return [
    ...(agent.credential_env || []),
    ...(agent.provider_autoselect || []).map((p) => p && p.trigger_env).filter(Boolean),
  ];
}

const isSet = (env, name) => Boolean(String(env[name] || '').trim());

// Which agent does the credential environment point at, if any?
//
// A trigger key selects its own agent ONLY when no other agent's side is also
// configured. Without that exclusion, exporting a gateway key beside a working
// Claude setup would silently move every run onto codex; ambiguity instead falls
// through to the registry default, which is also what a CLI authenticated by
// other means (a logged-in claude, Bedrock) needs. Mirrors hyperloom's
// is_openai_only()/is_anthropic_only() shape test in common/llm_config.py.
export function deriveAgentFromEnv(registry, env = {}) {
  if (AUTO_OFF.has(String(env.GEAK_AGENT_AUTO ?? '1').trim().toLowerCase())) return '';

  const agents = Object.entries(registry.agents || {});
  const configured = agents
    .filter(([, agent]) => sideEnvNames(agent).some((name) => isSet(env, name)))
    .map(([name]) => name);

  const triggered = agents.find(([, agent]) => (agent.provider_autoselect || [])
    .some((p) => p && p.trigger_env && isSet(env, p.trigger_env)));
  if (!triggered) return '';

  return configured.every((name) => name === triggered[0]) ? triggered[0] : '';
}

// Resolve a selection into a concrete { agentName, agent, modelName, model }.
// Precedence for what to run (highest first) is applied by the CALLER (CLI flag
// > env > registry default); this function just resolves the chosen names.
//
//   sel = { profile?, agent?, model?, env? }
// - profile: name in registry.profiles (supplies agent + optional model)
// - agent:   overrides the profile's agent (or stands alone)
// - model:   overrides the profile's model (or stands alone)
// - env:     process.env-like map, consulted only to derive an agent when none
//            was selected at all (see deriveAgentFromEnv)
export function resolveSelection(registry, sel = {}) {
  const profiles = registry.profiles || {};
  const agents = registry.agents || {};
  const models = registry.models || {};

  let agentName = sel.agent;
  let modelName = sel.model;

  if (sel.profile) {
    const p = profiles[sel.profile];
    if (!p) throw new Error(`unknown profile "${sel.profile}" (have: ${Object.keys(profiles).join(', ') || 'none'})`);
    if (!agentName) agentName = p.agent;
    if (!modelName) modelName = p.model;
  }

  if (!agentName && !sel.profile) agentName = deriveAgentFromEnv(registry, sel.env || {}) || undefined;

  if (!agentName) agentName = registry.default_profile
    ? (profiles[registry.default_profile] || {}).agent
    : undefined;
  if (!agentName) throw new Error('no agent selected (pass --profile/--agent or set registry.default_profile)');

  const agent = agents[agentName];
  if (!agent) throw new Error(`unknown agent "${agentName}" (have: ${Object.keys(agents).join(', ')})`);

  let model = null;
  if (modelName) {
    model = models[modelName];
    if (!model) throw new Error(`unknown model "${modelName}" (have: ${Object.keys(models).join(', ')})`);
  }

  return { agentName, agent, modelName: modelName || null, model };
}

// codex's model_reasoning_effort accepts exactly these; anything else makes the
// CLI reject its own config, so 'max' (the level people ask for, and the one
// claude does accept) has to be translated rather than passed through.
const CODEX_EFFORTS = new Set(['none', 'low', 'medium', 'high', 'xhigh']);

export function codexEffort(requested) {
  const want = String(requested ?? '').trim().toLowerCase();
  if (!want) return 'xhigh';
  if (want === 'max') return 'xhigh';
  if (!CODEX_EFFORTS.has(want)) {
    throw new Error(
      `unsupported codex reasoning effort ${JSON.stringify(String(requested))} `
      + `(GEAK_CODEX_EFFORT must be one of: ${[...CODEX_EFFORTS].join(', ')}, or max=xhigh)`);
  }
  return want;
}

// Build the concrete subprocess invocation for one agent() call.
//   agent: a resolved agent recipe (registry.agents[name])
//   model: resolved model object or null
//   prompt: the (already schema-instructed + neutralized) prompt text
//   opts.modelOverride: explicit model id string (beats model.id)
//   opts.env: process.env-like map used to read *_env overrides (defaults to {})
// Returns { cmd, args, promptOnStdin, env } for backends/base.mjs spawnAgent.
export function buildInvocation(agent, model, prompt, opts = {}) {
  const penv = opts.env || {};
  const cmd = (agent.bin_env && penv[agent.bin_env]) || agent.bin;

  const args = [...(agent.args || [])];

  // auto-approve flag (env override wins; empty string disables)
  let approve = agent.approve;
  if (agent.approve_env && penv[agent.approve_env] !== undefined) approve = penv[agent.approve_env];
  if (approve) args.push(...String(approve).split(/\s+/).filter(Boolean));

  // Which provider_autoselect entry this run resolves to, if any. Resolved here
  // because it supplies BOTH the provider block below AND the default model id:
  // an endpoint and the ids it actually serves are one fact, so both come from
  // one registry entry and a gateway id can never leak onto another endpoint.
  // Deliberately skipped when a base_url is supplied explicitly — an endpoint
  // chosen by hand needs its model named by hand too.
  const autoProvider = (() => {
    if (agent.provider_autoconfig !== 'codex') return null;
    if (String(penv.GEAK_CODEX_AUTOCONFIG ?? '1') === '0') return null;
    if (String(penv.OPENAI_BASE_URL || (model && model.base_url) || '').trim()) return null;
    for (const p of agent.provider_autoselect || []) {
      if (p && p.trigger_env && String(penv[p.trigger_env] || '').trim()) return p;
    }
    return null;
  })();

  // model id: explicit override > pinned model > agent's model_env > the
  // auto-selected provider's own default (absent on providers whose CLI default
  // already works, e.g. official OpenAI)
  const modelId = opts.modelOverride
    || (model && model.id)
    || (agent.model_env && penv[agent.model_env])
    || (autoProvider && autoProvider.default_model)
    || '';
  if (modelId && agent.model_flag) args.push(agent.model_flag, modelId);

  // codex provider auto-config (Hyperloom-style): emit `-c model_providers.geak_auto.*`
  // overrides so NO hand-written config.toml and NO `-c model_provider=` selection are
  // needed. Two resolution modes (first wins), gated on agent.provider_autoconfig==='codex':
  //   (1) base_url-driven — an explicit OPENAI_BASE_URL (or the selected model's
  //       base_url) is used directly. key_env defaults to OPENAI_API_KEY.
  //   (2) key-driven auto-SELECT — when no base_url is available, walk
  //       agent.provider_autoselect in order and pick the first provider whose
  //       trigger_env is set: GEAK_AMDKEY->AMD gateway, OPENAI_API_KEY->OpenAI official.
  //       Each entry carries base_url / key_env / default_model and optional
  //       env_http_headers, so "give the AMD key -> AMD, the OpenAI key ->
  //       OpenAI". The same list drives deriveAgentFromEnv, which picks the agent
  //       itself from the key.
  // Emitted BEFORE extra_args so GEAK_CODEX_EXTRA_ARGS still wins. Skipped when
  // disabled (GEAK_CODEX_AUTOCONFIG=0) or when the caller already pins
  // model_provider via extra_args. OPENAI_CUSTOM_HEADERS (JSON
  // {"Header":"ENV_VAR_NAME"}) overrides a selected provider's headers.
  if (agent.provider_autoconfig === 'codex'
      && String(penv.GEAK_CODEX_AUTOCONFIG ?? '1') !== '0') {
    let baseUrl = String(penv.OPENAI_BASE_URL || (model && model.base_url) || '').trim();
    let keyEnv = (model && model.key_env) || 'OPENAI_API_KEY';
    let headers = null;
    if (autoProvider) {
      baseUrl = String(autoProvider.base_url || '').trim();
      keyEnv = autoProvider.key_env || keyEnv;
      headers = autoProvider.env_http_headers || null;
    }
    // explicit OPENAI_CUSTOM_HEADERS wins over an auto-selected provider's headers
    try { const h = JSON.parse(penv.OPENAI_CUSTOM_HEADERS || 'null'); if (h && typeof h === 'object') headers = h; } catch { /* ignore */ }

    const extra = (agent.extra_args_env && penv[agent.extra_args_env]) || '';
    const providerPinned = /model_provider\s*=/.test(extra);
    if (baseUrl && !providerPinned) {
      const P = 'geak_auto';
      const ts = (s) => '"' + String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
      args.push('-c', `model_provider=${ts(P)}`);
      args.push('-c', `model_providers.${P}.name=${ts(P)}`);
      args.push('-c', `model_providers.${P}.base_url=${ts(baseUrl)}`);
      args.push('-c', `model_providers.${P}.env_key=${ts(keyEnv)}`);
      args.push('-c', `model_providers.${P}.wire_api=${ts('responses')}`);
      if (headers && typeof headers === 'object') {
        for (const [h, v] of Object.entries(headers)) {
          if (h && v) args.push('-c', `model_providers.${P}.env_http_headers.${h}=${ts(v)}`);
        }
      }
    }
    // Fail fast when autoconfig is ON, nothing is pinned, and yet no endpoint
    // resolved: that means no key matched any provider_autoselect entry and no
    // base_url was given, so the block above emitted NOTHING and codex silently
    // falls back to its own ~/.codex/config.toml -- which does not exist in the
    // containers GEAK runs in. The run then dies at the FIRST agent call, hours
    // into a workflow, with an error that names neither the key nor the config.
    // This is the exact shape hyperloom's .env filter produces when the key name
    // is not allowlisted: GEAK_AGENT_BACKEND=codex gets through, the key does
    // not. Raising here costs one line of startup and names the fix.
    if (!baseUrl && !providerPinned) {
      const names = (agent.provider_autoselect || [])
        .map((p) => p && p.key_env).filter(Boolean);
      throw new Error(
        `codex selected but no provider resolved: set one of ${names.join(' / ')}, `
        + 'or OPENAI_BASE_URL, or disable auto-config with GEAK_CODEX_AUTOCONFIG=0 '
        + "to use codex's own config.toml");
    }
  }

  // codex thinking level. codex only knows none|low|medium|high|xhigh — 'max' is
  // NOT one of them, so the generic "maximum thinking" has to be translated onto
  // xhigh (same mapping as hyperloom's resolve_codex_reasoning_effort). Default is
  // therefore xhigh; override via GEAK_CODEX_EFFORT, or pin through
  // GEAK_CODEX_EXTRA_ARGS (appended after, so it wins). Emitted only when
  // extra_args does not already set it.
  if (agent.provider_autoconfig === 'codex') {
    const extra0 = (agent.extra_args_env && penv[agent.extra_args_env]) || '';
    if (!/model_reasoning_effort\s*=/.test(extra0)) {
      const effort = codexEffort(penv.GEAK_CODEX_EFFORT);
      if (effort) args.push('-c', `model_reasoning_effort=${effort}`);
    }
  }

  // extra args (env only) — an escape hatch for build-specific flags
  if (agent.extra_args_env && penv[agent.extra_args_env]) {
    args.push(...String(penv[agent.extra_args_env]).split(/\s+/).filter(Boolean));
  }

  // env: agent.env + model endpoint (base_url routed to the agent's dialect env)
  const env = { ...(agent.env || {}) };
  if (model && model.base_url && agent.base_url_env) {
    env[agent.base_url_env] = model.base_url;
  }

  const promptOnStdin = (agent.prompt || 'stdin') === 'stdin';
  if (!promptOnStdin) args.push(prompt);   // prompt delivered as the final arg

  return { cmd, args, promptOnStdin, env };
}

// -- Prompt neutralization --------------------------------------------------
// GEAK's role prompts + the JS roleAgent() base carry Claude-Code-specific
// wording ("a StructuredOutput tool is forced"). For non-claude backends this
// references a tool that does not exist and can confuse the agent. We replace it
// with a backend-agnostic instruction — WITHOUT editing roles/*.md or the .js
// (those are used unmodified). schema.mjs still appends the authoritative
// fenced-JSON contract; this only removes the misleading phrase.
const NEUTRALIZE_RULES = [
  [/a StructuredOutput tool is forced/gi, 'return your result as a single ```json fenced code block'],
  [/the script forces a StructuredOutput tool/gi, 'return your result as a single ```json fenced code block'],
  [/StructuredOutput tool/gi, 'JSON output'],
  [/\bas StructuredOutput\b/gi, 'as a single ```json fenced code block'],
];

export function neutralizeForBackend(prompt, agentName) {
  if (agentName === 'claude') return prompt;   // native wording is correct for claude
  let out = prompt;
  for (const [re, rep] of NEUTRALIZE_RULES) out = out.replace(re, rep);
  return out;
}
