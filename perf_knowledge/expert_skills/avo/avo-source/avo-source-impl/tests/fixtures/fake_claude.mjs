// A stand-in for the `claude` CLI, for tests/agent_early_finish.test.mjs.
//
// It replays `result` events on a schedule and then stays alive until `exit_at`
// (or forever), which is how the real CLI behaves while the agent still holds a
// background task or a scheduled wakeup. The plan is a JSON file named by
// FAKE_CLAUDE_PLAN: { events: [{t, result, is_error?}], exit_at?, straggler_pidfile? }
// (times in seconds from start).
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';

const plan = JSON.parse(readFileSync(process.env.FAKE_CLAUDE_PLAN, 'utf8'));
process.stdin.resume();                      // the prompt arrives on stdin; ignore it

if (plan.straggler_pidfile) {
  // A background command in its own process group that ignores SIGTERM, like a
  // benchmark loop a worker left behind. It inherits the agent tag from our env.
  const straggler = spawn(process.execPath, ['-e',
    "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000);"],
  { detached: true, stdio: 'ignore' });
  straggler.unref();
  writeFileSync(plan.straggler_pidfile, String(straggler.pid));
}

for (const ev of plan.events || []) {
  setTimeout(() => {
    process.stdout.write(`${JSON.stringify({
      type: 'result', subtype: ev.is_error ? 'error' : 'success',
      is_error: !!ev.is_error, result: ev.result, total_cost_usd: 0.01,
    })}\n`);
  }, ev.t * 1000);
}
if (typeof plan.exit_at === 'number') {
  setTimeout(() => process.exit(0), plan.exit_at * 1000);
} else {
  setInterval(() => {}, 1000);               // never exits on its own
}
