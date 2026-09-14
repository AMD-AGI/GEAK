// Generic, config-driven backend: drives ANY code-agent CLI from a resolved
// registry recipe (see config.mjs + registry.json). This is the single backend
// for claude / qwen / codex / kimi — the differences live in registry.json data,
// not code. Adding a new agent is a registry entry, zero code.
//
// Escape hatch: run_workflow.mjs prefers a hand-written backends/<name>.mjs if
// one exists, so a CLI with behavior the registry can't express can still be
// supported by dropping in a custom module.

import { spawnAgent } from './base.mjs';
import { buildInvocation } from '../config.mjs';

// Factory: bind a resolved agent recipe + model into a backend object with the
// standard { name, runAgent } shape the runtime expects.
export function makeGenericBackend({ agentName, agent, model }) {
  return {
    name: agentName,
    agent,
    model,
    async runAgent({ prompt, label, cwd, env, model: modelOverride, timeoutMs }) {
      const inv = buildInvocation(agent, model, prompt, {
        modelOverride,
        env: process.env,
      });
      const { stdout, stderr, code } = await spawnAgent({
        cmd: inv.cmd,
        args: inv.args,
        prompt,
        promptOnStdin: inv.promptOnStdin,
        cwd,
        env: { ...inv.env, ...(env || {}) },
        timeoutMs,
      });
      if (code !== 0) {
        throw new Error(`[${label || 'agent'}] ${inv.cmd} exited ${code}: ${String(stderr).slice(-400)}`);
      }
      return { text: stdout };
    },
  };
}
