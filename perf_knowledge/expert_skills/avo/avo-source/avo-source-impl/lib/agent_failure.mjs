// Pure classification of a failed agent() call (no I/O, no runtime globals).
//
// Single source of truth for telling an unreachable agent backend from a reachable
// agent that broke the output contract, used by run_avo.mjs and unit-tested by
// tests/agent_failure.test.mjs.
//
// The two need opposite responses. Re-prompting an agent that answered in prose is
// exactly right; re-prompting a refused connection just burns the retry budget at
// wire speed. Conflating them ended a 19.5h run: four workers x ConnectionRefused x
// 0 tool calls inside 190s each, read as "every worker hit a broken harness".

/** Failure strings worth waiting out rather than re-prompting. */
export const TRANSPORT_FAILURE_RE = new RegExp([
  'unable to connect to api', 'connection ?refused', 'econnrefused', 'econnreset',
  'enotfound', 'eai_again', 'etimedout', 'socket hang up', 'network error',
  'bad gateway', 'service unavailable', 'gateway time-?out', 'overloaded',
  'too many requests', 'rate.?limit', '\\b50[234]\\b', '\\b429\\b',
].join('|'), 'i');

/**
 * Did this agent invocation fail because the backend could not be reached?
 *
 * Only the CLI's own failure channels are consulted: its stderr, plus the final
 * result text when the session either failed or never ran a tool. The raw event
 * stream is deliberately excluded — it carries tool output and file contents, where
 * an unrelated "503" in some payload would fake an outage and spend the backoff
 * budget on a healthy agent. Once JSON came back the backend was plainly reachable,
 * so a parsed result is never an outage however it reads.
 *
 * @param {object} result  spawn result: {ok, nTools, resultText, err}
 * @param {object|null} parsed  JSON extracted from the agent's final message
 * @returns {boolean}
 */
export function isTransportFailure(result, parsed) {
  if (parsed || !result) return false;
  const suspect = !result.ok || Number(result.nTools) === 0;
  const channels = `${result.err || ''}\n${suspect ? (result.resultText || '') : ''}`;
  return TRANSPORT_FAILURE_RE.test(channels);
}

/**
 * Did the runtime kill this agent at its wall-clock deadline?
 *
 * This is an outage of the verifier agent, not evidence that the candidate or
 * measurement rig failed. Keep it separate from transport errors because retry
 * backoff cannot make a too-short invocation deadline longer.
 */
export function isAgentTimeout(result) {
  return !!result && /\[shim\]\s*TIMEOUT/i.test(String(result.err || ''));
}

/**
 * Exponential backoff for outage retry `attempt` (1-based), capped.
 *
 * @param {number} attempt  1 for the first retry
 * @param {number} baseMs   wait before the first retry
 * @param {number} capMs    upper bound on any single wait
 * @returns {number} milliseconds to wait
 */
export function transportBackoffMs(attempt, baseMs, capMs) {
  const n = Math.max(1, Math.floor(Number(attempt) || 1));
  return Math.min(capMs, baseMs * (2 ** (n - 1)));
}
