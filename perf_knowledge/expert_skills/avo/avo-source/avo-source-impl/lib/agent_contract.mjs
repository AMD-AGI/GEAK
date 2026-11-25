// Pure check of an agent's output contract (no I/O, no runtime globals).
//
// `claude --print` emits one `result` event per turn, so a session that waits on a
// background task or a scheduled wakeup emits several. Earlier ones are progress
// text ("still waiting ..."); only one of them is the answer. run_avo.mjs uses this
// check to tell them apart, and tests/agent_contract.test.mjs pins the rules.

/**
 * Is this result text the agent's final answer?
 *
 * The whole text must be one JSON object (a single surrounding code fence is
 * accepted) that carries every required key. Prose with a JSON object embedded in
 * it does not qualify: a progress message may quote a draft.
 *
 * @param {unknown} text        result text of one turn
 * @param {string[]} required   keys the role's schema requires
 * @returns {object|null} the parsed answer, or null
 */
export function finalAnswer(text, required = []) {
  if (typeof text !== 'string') return null;
  let s = text.trim();
  const fence = s.match(/^```(?:json)?\s*([\s\S]*?)```$/i);
  if (fence) s = fence[1].trim();
  let value;
  try { value = JSON.parse(s); } catch { return null; }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  return required.every((key) => key in value) ? value : null;
}
