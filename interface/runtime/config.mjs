// Config-driven agent/model resolution for the standalone runtime.
//
// Two orthogonal axes (see registry.json):
//   agents[]  — how to drive a code-agent CLI (bin, flags, prompt delivery, env). Model-independent.
//   models[]  — an endpoint (id + base_url + key env). CLI-independent.
//   profiles[]— a named (agent, model) combo you can pin.
//
// This module is deterministic except loadRegistry and the fail-closed outer-
// container check. Tests inject containerDetected so invocation logic remains unit-testable
// (selftest.mjs) without spawning anything.

import { existsSync, readFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve as resolvePath, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
export const DEFAULT_REGISTRY_PATH = resolvePath(HERE, 'registry.json');

export function runningInContainer() {
  if (existsSync('/.dockerenv') || existsSync('/run/.containerenv')) return true;
  try {
    const cgroup = readFileSync('/proc/1/cgroup', 'utf8').toLowerCase();
    return ['docker', 'kubepods', 'containerd', 'podman', 'lxc'].some((marker) => cgroup.includes(marker));
  } catch {
    return false;
  }
}

function truthy(value) {
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').trim().toLowerCase());
}

export async function loadRegistry(path = DEFAULT_REGISTRY_PATH) {
  const raw = JSON.parse(await readFile(path, 'utf8'));
  return raw;
}

// Resolve a selection into a concrete { agentName, agent, modelName, model }.
// Precedence for what to run (highest first) is applied by the CALLER (CLI flag
// > env > registry default); this function just resolves the chosen names.
//
//   sel = { profile?, agent?, model? }
// - profile: name in registry.profiles (supplies agent + optional model)
// - agent:   overrides the profile's agent (or stands alone)
// - model:   overrides the profile's model (or stands alone)
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

  if (agent.external_container_env) {
    const declared = truthy(penv[agent.external_container_env]);
    const detected = opts.containerDetected === undefined ? runningInContainer() : opts.containerDetected === true;
    if (!declared || !detected) {
      throw new Error(
        `${agent.external_container_env}=1 and a verifiable outer container are required for this agent backend`,
      );
    }
  }

  const profile = (agent.profile_env && penv[agent.profile_env]) || agent.profile || '';
  if (profile && agent.profile_flag) args.push(agent.profile_flag, profile);

  const provider = (agent.provider_env && penv[agent.provider_env]) || agent.provider || '';
  if (provider && agent.provider_flag) args.push(agent.provider_flag, provider);

  // auto-approve flag (env override wins; empty string disables)
  let approve = agent.approve;
  if (agent.approve_env && penv[agent.approve_env] !== undefined) approve = penv[agent.approve_env];
  if (approve) args.push(...String(approve).split(/\s+/).filter(Boolean));

  // model id: explicit override > agent's environment override > profile model
  const modelId = opts.modelOverride
    || (agent.model_env && penv[agent.model_env])
    || (model && model.id)
    || '';
  if (modelId && agent.model_flag) args.push(agent.model_flag, modelId);

  // codex provider auto-config (Hyperloom-style): emit `-c model_providers.geak_auto.*`
  // overrides so NO hand-written config.toml and NO `-c model_provider=` selection are
  // needed. Two public resolution modes (first wins), gated on agent.provider_autoconfig==='codex':
  //   (1) base_url-driven — an explicit OPENAI_BASE_URL (or the selected model's
  //       base_url) is used directly. key_env defaults to OPENAI_API_KEY.
  //   (2) key-driven auto-SELECT — when no base_url is available, walk
  //       agent.provider_autoselect in order and pick the first provider whose
  //       trigger_env is set; the shipped entry maps OPENAI_API_KEY to the
  //       official OpenAI endpoint. Entries may carry base_url / key_env /
  //       optional env_http_headers.
  // Emitted BEFORE extra_args so GEAK_CODEX_EXTRA_ARGS still wins. Skipped when
  // disabled (GEAK_CODEX_AUTOCONFIG=0) or the caller already pins model_provider
  // via extra_args. OPENAI_CUSTOM_HEADERS (JSON
  // {"Header":"ENV_VAR_NAME"}) overrides a selected provider's headers.
  if (agent.provider_autoconfig === 'codex'
      && String(penv.GEAK_CODEX_AUTOCONFIG ?? '1') !== '0') {
    let baseUrl = String(penv.OPENAI_BASE_URL || (model && model.base_url) || '').trim();
    let keyEnv = (model && model.key_env) || 'OPENAI_API_KEY';
    let headers = null;
    if (!baseUrl && Array.isArray(agent.provider_autoselect)) {
      for (const p of agent.provider_autoselect) {
        if (p && p.trigger_env && String(penv[p.trigger_env] || '').trim()) {
          baseUrl = String(p.base_url || '').trim();
          keyEnv = p.key_env || keyEnv;
          headers = p.env_http_headers || null;
          break;
        }
      }
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
  }

  // codex thinking level (reasoning effort). Default 'max'; override via
  // GEAK_CODEX_EFFORT, or pin through GEAK_CODEX_EXTRA_ARGS (appended after, so it
  // wins). Applies to the codex agent regardless of provider source (auto-config
  // or an operator-supplied endpoint). Emitted only when not already set in extra_args.
  if (agent.provider_autoconfig === 'codex') {
    const extra0 = (agent.extra_args_env && penv[agent.extra_args_env]) || '';
    if (!/model_reasoning_effort\s*=/.test(extra0)) {
      const effort = String(penv.GEAK_CODEX_EFFORT ?? 'max').trim();
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
  if (!promptOnStdin) {
    if (agent.prompt_flag) args.push(agent.prompt_flag);
    args.push(prompt);   // prompt delivered as the final arg
  }

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
  [/\bWebSearch\b/g, 'the available web-search capability'],
  [/\bWebFetch\b/g, 'the available web-page extraction capability'],
];

export function neutralizeForBackend(prompt, agentName) {
  if (agentName === 'claude') return prompt;   // native wording is correct for claude
  let out = prompt;
  for (const [re, rep] of NEUTRALIZE_RULES) out = out.replace(re, rep);
  return out;
}
