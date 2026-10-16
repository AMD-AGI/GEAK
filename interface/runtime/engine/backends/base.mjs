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

// Run one command to completion in its own process group and wait until that group is gone.
//
// The child is the leader of a new process group, so everything it starts (compilers, benchmark
// servers, a shell's background jobs) can be signalled together:
//   - on timeout the whole group gets SIGTERM, then SIGKILL after `graceMs`;
//   - when the leader exits, group members that are still alive are strays: they get the same
//     TERM -> KILL and are reported in `strays_killed`.
// The promise settles once the stdio pipes have closed and the group is empty, and in any case at
// most `graceMs + settleMs` after the leader exits or is signalled. It never rejects:
//   { status: 'completed' | 'timeout' | 'spawn_error', code, signal, stdout, stderr, duration_ms,
//     strays_killed, group_cleared, stdio_held, error? }
// group_cleared false: a member of the group survived SIGKILL (e.g. stuck in the kernel).
// stdio_held true: something outside the group (a process that called setsid()) still held our
// stdout/stderr when we stopped waiting. Such a process is out of reach here; only a cgroup bounds it.
// Only completed + group_cleared + !stdio_held means nothing the command started is still running.
// So that a retry never runs beside what is left, later calls return spawn_error without starting
// anything: while a group that survived SIGKILL is alive, and for the rest of this process once an
// escaped process was seen holding the pipes (it cannot be tracked). runtimeFault() reports either
// state, and run_workflow then fails the run even if the workflow itself returned.
// An escape that does not hold the pipes is not seen at all.
export function runProcess({ cmd, args = [], cwd, env, input, timeoutMs = 0, graceMs = 5000, settleMs = 2000 }) {
  return new Promise((resolve) => {
    const started = Date.now();
    const out = { status: 'completed', code: null, signal: null, stdout: '', stderr: '',
      duration_ms: 0, strays_killed: false, group_cleared: true, stdio_held: false };
    if (escapeSeen) {
      resolve({ ...out, status: 'spawn_error', error: `not starting another command: ${escapeSeen}` });
      return;
    }
    for (const g of unresolvedGroups) if (!groupAlive(g)) { unresolvedGroups.delete(g); liveGroups.delete(g); }
    if (unresolvedGroups.size) {
      resolve({ ...out, status: 'spawn_error',
        error: `process group(s) ${[...unresolvedGroups].join(',')} survived SIGKILL; not starting another command` });
      return;
    }
    let child;
    try {
      child = spawn(cmd, args, {
        cwd: cwd || process.cwd(),
        env: { ...process.env, ...(env || {}) },
        stdio: ['pipe', 'pipe', 'pipe'],
        detached: true,
      });
    } catch (e) {
      resolve({ ...out, status: 'spawn_error', error: e.message });
      return;
    }
    const pgid = child.pid;
    let settled = false;
    let terminating = false;
    let closed = false;
    let watchdog = null;
    const timers = [];
    if (pgid) trackGroup(pgid);

    const finish = () => {
      if (settled) return;
      settled = true;
      timers.forEach(clearTimeout);
      clearTimeout(watchdog);
      if (pgid && out.group_cleared) liveGroups.delete(pgid);
      else if (pgid) unresolvedGroups.add(pgid);
      if (out.stdio_held) escapeSeen = `an escaped process of '${cmd}' held its pipes`;
      resolve({ ...out, duration_ms: Date.now() - started });
    };
    // Stop waiting `ms` from now, whatever still holds the pipes or the group.
    const giveUpAfter = (ms) => {
      clearTimeout(watchdog);
      watchdog = setTimeout(() => {
        signalGroup(pgid, 'SIGKILL');
        out.group_cleared = !groupAlive(pgid);
        out.stdio_held = !closed;
        child.stdout.destroy();
        child.stderr.destroy();
        finish();
      }, ms);
    };
    // Signal the whole group: TERM now, KILL after the grace period.
    const terminate = () => {
      if (terminating || !pgid) return;
      terminating = true;
      signalGroup(pgid, 'SIGTERM');
      timers.push(setTimeout(() => signalGroup(pgid, 'SIGKILL'), graceMs));
      giveUpAfter(graceMs + settleMs);
    };
    // After the pipes close, wait until no member of the group is left (the watchdog bounds it).
    const settle = () => {
      closed = true;
      const poll = () => (groupAlive(pgid) ? timers.push(setTimeout(poll, 20)) : finish());
      poll();
    };

    const deadline = timeoutMs > 0 ? setTimeout(() => {
      out.status = 'timeout';
      terminate();
    }, timeoutMs) : null;
    timers.push(deadline);
    child.stdout.on('data', (d) => { out.stdout += d.toString(); });
    child.stderr.on('data', (d) => { out.stderr += d.toString(); });
    child.on('error', (e) => {
      // ENOENT/EACCES arrive here, before any pid exists.
      if (!pgid) { out.status = 'spawn_error'; out.error = e.message; finish(); }
    });
    child.on('exit', (code, signal) => {
      clearTimeout(deadline);
      out.code = code;
      out.signal = signal;
      if (terminating) return;
      if (groupAlive(pgid)) {
        out.strays_killed = true;
        terminate();
      } else {
        giveUpAfter(settleMs);   // the group is gone; only an escaped process can hold the pipes
      }
    });
    child.on('close', () => { if (pgid) settle(); });

    // A child that exits before reading stdin turns our write into an async EPIPE on the stdin
    // stream; 'exit'/'close' still report why it ended.
    child.stdin.on('error', () => {});
    try {
      if (input) child.stdin.write(input);
      child.stdin.end();
    } catch { /* synchronous write-after-end only */ }
  });
}

// Why this runtime may no longer be trusted to have stopped what it started, or null. The runner
// checks it at the end, so a run cannot report success while such a process may still be running.
export function runtimeFault() {
  for (const g of unresolvedGroups) if (!groupAlive(g)) { unresolvedGroups.delete(g); liveGroups.delete(g); }
  if (escapeSeen) return escapeSeen;
  return unresolvedGroups.size ? `process group(s) ${[...unresolvedGroups].join(',')} survived SIGKILL` : null;
}

const liveGroups = new Set();
const unresolvedGroups = new Set();   // groups that survived SIGKILL (also kept in liveGroups)
let escapeSeen = null;                // set once, never cleared
function groupAlive(pgid) {
  try { process.kill(-pgid, 0); return true; } catch (e) { return e.code === 'EPERM'; }
}
function signalGroup(pgid, sig) {
  try { process.kill(-pgid, sig); } catch { /* already gone */ }
}
// A detached group does not receive the terminal's Ctrl-C, so running groups go down with us. The
// handlers are installed on first use and re-raise the signal, keeping its default effect.
function trackGroup(pgid) {
  if (!trackGroup.installed) {
    trackGroup.installed = true;
    const killAll = () => { for (const g of liveGroups) signalGroup(g, 'SIGKILL'); };
    process.on('exit', killAll);
    for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
      process.once(sig, () => { killAll(); process.kill(process.pid, sig); });
    }
  }
  liveGroups.add(pgid);
}

// One agent CLI call: the prompt goes on stdin (no ARG_MAX limit) unless the CLI takes it as a
// positional argument already in `args`. Resolves { stdout, stderr, code }; a non-zero exit is NOT
// rejected here, the caller decides. Rejects on spawn error, on timeout, and when something the CLI
// started could not be shown to be gone (a survivor in its group, or an escaped process holding its
// pipes). Either also blocks later runProcess calls (see runProcess), so the runtime's retry
// cannot run next to it. Children it left behind and that were killed do not
// reject: the CLI's own answer is complete.
export async function spawnAgent({ cmd, args = [], prompt = '', cwd, env, timeoutMs = 3600000, promptOnStdin = true }) {
  const r = await runProcess({ cmd, args, cwd, env, input: promptOnStdin ? prompt : '', timeoutMs });
  if (r.status === 'spawn_error') throw new Error(`${cmd} spawn error: ${r.error}`);
  if (r.status === 'timeout') throw new Error(`${cmd} timed out after ${Math.round(timeoutMs / 1000)}s`);
  if (!r.group_cleared || r.stdio_held) {
    throw new Error(`${cmd} left processes that could not be stopped (${r.group_cleared ? 'pipes held' : 'group not cleared'})`);
  }
  return { stdout: r.stdout, stderr: r.stderr, code: r.code };
}

// Resolve the concurrency cap the runtime uses, matching the Workflow tool:
// min(16, cpus - 2), floored at 1.
export function defaultConcurrency(cpuCount) {
  return Math.max(1, Math.min(16, (cpuCount || 1) - 2));
}
