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

  // auto-approve flag (env override wins; empty string disables)
  let approve = agent.approve;
  if (agent.approve_env && penv[agent.approve_env] !== undefined) approve = penv[agent.approve_env];
  if (approve) args.push(...String(approve).split(/\s+/).filter(Boolean));

  // model id: explicit override > model.id > agent's model_env
  const modelId = opts.modelOverride
    || (model && model.id)
    || (agent.model_env && penv[agent.model_env])
    || '';
  if (modelId && agent.model_flag) args.push(agent.model_flag, modelId);

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
