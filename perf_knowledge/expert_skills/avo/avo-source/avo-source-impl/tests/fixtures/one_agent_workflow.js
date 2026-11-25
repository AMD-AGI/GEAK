// Minimal workflow for tests/agent_early_finish.test.mjs: one agent() call.
// Returns the answer, how long the call took, and whether the fake agent's
// background process was still alive at the moment agent() returned.
const fs = await import('node:fs');

const started = Date.now();
const answer = await agent(args.probe_prompt || 'probe', {
  label: 'probe', schema: { required: args.probe_required || [] },
});
const seconds = (Date.now() - started) / 1000;

let stragglerAlive = null;
if (args.straggler_pidfile) {
  const pid = Number(fs.readFileSync(args.straggler_pidfile, 'utf8'));
  // A killed process whose parent never reaps it (a container without an init)
  // stays in the process table as a zombie; signal 0 still succeeds on it. It
  // holds no GPU and runs nothing, so only a non-zombie counts as alive.
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
    stragglerAlive = stat.slice(stat.lastIndexOf(')') + 2, stat.lastIndexOf(')') + 3) !== 'Z';
  } catch { stragglerAlive = false; }
}
return { answer, seconds, straggler_alive: stragglerAlive };
