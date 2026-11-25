import { readFileSync, readdirSync } from 'node:fs';

// Killing an agent's process group is not enough to stop the agent's work. Every
// Bash tool call runs in a process group of its own, so anything a worker leaves
// running in the background (`nohup … &`, a self-written driver loop) sits in a
// group the orchestrator never learns about. One such loop outlived a full
// teardown and immediately relaunched a benchmark on a GPU the run had given up.
//
// The environment, unlike the process group, survives setsid and reparenting, so a
// tag planted on the agent is carried by every descendant however it detached.
export const AGENT_TAG_VAR = 'AVO_AGENT_TAG';

export function pidsWithTag(tag, { proc = '/proc', self = process.pid } = {}) {
  if (!tag) return [];
  const needle = `${AGENT_TAG_VAR}=${tag}`;
  const found = [];
  let entries;
  try { entries = readdirSync(proc); } catch { return found; }
  for (const entry of entries) {
    if (!/^\d+$/.test(entry)) continue;
    const pid = Number(entry);
    if (pid === self) continue;
    let environ;
    // A process can exit between the listing and the read, and another user's
    // environ is unreadable; neither means anything is wrong.
    try { environ = readFileSync(`${proc}/${entry}/environ`, 'latin1'); } catch { continue; }
    if (environ.split('\0').includes(needle)) found.push(pid);
  }
  return found;
}

// Called once the agent has returned its verdict: whatever still carries its tag is
// unaccounted-for work that would contend for the GPU and skew later measurements.
// Returns how many stragglers were signalled, so the caller can surface it.
export function reapTagged(tag, options = {}) {
  const { kill = process.kill.bind(process), graceMs = 5000, schedule = setTimeout } = options;
  const stragglers = pidsWithTag(tag, options);
  for (const pid of stragglers) {
    try { kill(pid, 'SIGTERM'); } catch { /* already gone */ }
  }
  if (stragglers.length) {
    const timer = schedule(() => {
      for (const pid of pidsWithTag(tag, options)) {
        try { kill(pid, 'SIGKILL'); } catch { /* already gone */ }
      }
    }, graceMs);
    if (timer && typeof timer.unref === 'function') timer.unref();
  }
  return stragglers.length;
}

// After the stragglers were signalled the caller must not hand the GPU to the next
// step until they are really gone: a process that ignores SIGTERM lives until the
// SIGKILL of reapTagged's grace period. Resolves with how many tagged processes
// are still alive: 0 once all are gone, or the remaining count at the deadline.
export async function waitTagGone(tag, options = {}) {
  const { timeoutMs = 7000, pollMs = 100 } = options;
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const left = pidsWithTag(tag, options).length;
    if (left === 0 || Date.now() >= deadline) return left;
    await new Promise((done) => { setTimeout(done, pollMs); });
  }
}
