// Backend adapter contract + shared subprocess helper.
//
// A backend turns one agent() call into ONE one-shot coding-agent subprocess
// (claude -p / codex exec). It knows nothing about parallelism, phases, budgets
// or nesting — the runtime (run_workflow.mjs) owns all of that. This is exactly
// why a CLI that cannot itself orchestrate parallel or nested subagents is still
// usable: the runtime spawns N independent one-shot processes and the backend
// only has to run one prompt to completion.
//
// A backend module must export:
//   name: string
//   async runAgent({ prompt, label, cwd, env, model, timeoutMs }) -> { text }
//     - resolves with the agent's final stdout text (schema parsing is the
//       runtime's job, not the backend's)
//     - throws on non-zero exit / spawn error / timeout, so the runtime's retry
//       + degrade-to-null path handles transient failures.

import { spawn } from 'node:child_process';

// Spawn a subprocess, feed `prompt` on stdin, collect stdout/stderr, enforce a
// hard timeout. Resolves { stdout, stderr, code }; rejects on spawn error or
// timeout. A non-zero exit is NOT rejected here — the caller decides (some CLIs
// exit non-zero on benign conditions), but by default runAgent treats it as an
// error.
export function spawnAgent({ cmd, args = [], prompt = '', cwd, env, timeoutMs = 3600000, promptOnStdin = true }) {
  return new Promise((resolve, reject) => {
    let child;
    try {
      child = spawn(cmd, args, {
        cwd: cwd || process.cwd(),
        env: { ...process.env, ...(env || {}) },
        stdio: ['pipe', 'pipe', 'pipe'],
      });
    } catch (e) {
      reject(new Error(`spawn ${cmd} failed: ${e.message}`));
      return;
    }

    let stdout = '';
    let stderr = '';
    let done = false;

    const timer = timeoutMs > 0 ? setTimeout(() => {
      if (done) return;
      done = true;
      try { child.kill('SIGKILL'); } catch { /* ignore */ }
      reject(new Error(`${cmd} timed out after ${Math.round(timeoutMs / 1000)}s`));
    }, timeoutMs) : null;

    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });

    child.on('error', (e) => {
      if (done) return;
      done = true;
      if (timer) clearTimeout(timer);
      reject(new Error(`${cmd} spawn error: ${e.message}`));
    });

    child.on('close', (code) => {
      if (done) return;
      done = true;
      if (timer) clearTimeout(timer);
      resolve({ stdout, stderr, code });
    });

    // A CLI that dies before reading its prompt (bad key, unloadable config,
    // wrong version) leaves us writing to a closed pipe. That EPIPE arrives as
    // an ASYNCHRONOUS 'error' event on the stdin STREAM -- `child.on('error')`
    // above is a different emitter and the try/catch below cannot see it
    // either, so with no listener here it is an uncaught exception that kills
    // the runtime and replaces the CLI's own diagnostic with our stack trace.
    // Swallow it: 'close' still fires, and settles with the child's exit code
    // and stderr, which is the message the user actually needs.
    child.stdin.on('error', () => { /* child exited first; 'close' reports why */ });

    // Feed the prompt on stdin so we never hit ARG_MAX with large prompts.
    // When the CLI takes the prompt as a positional arg instead (promptOnStdin
    // false), it is already in `args`; just close stdin.
    try {
      if (promptOnStdin) child.stdin.write(prompt);
      child.stdin.end();
    } catch (e) {
      // Synchronous throws only (e.g. ERR_STREAM_WRITE_AFTER_END); the async
      // EPIPE is handled by the listener above.
    }
  });
}

// Resolve the concurrency cap the runtime uses, matching the Workflow tool:
// min(16, cpus - 2), floored at 1.
export function defaultConcurrency(cpuCount) {
  return Math.max(1, Math.min(16, (cpuCount || 1) - 2));
}
