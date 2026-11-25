/**
 * Normalize the supervisor REDIRECT JSON so downstream logic and audit logs always
 * see a consistent shape (name + expected + prompt on every direction; strip junk).
 * Pure — unit-tested in tests/supervisor_directive.test.mjs.
 *
 * @param {object|null|undefined} raw
 * @param {{ step?: number, failed?: string[] }} ctx
 * @returns {{ directive: object, repairs: string[] }}
 */
export function normalizeSupervisorDirective(raw, ctx = {}) {
  const repairs = [];
  const failed = Array.isArray(ctx.failed) ? ctx.failed : [];
  const step = ctx.step != null ? ctx.step : 0;

  if (!raw || typeof raw !== 'object') {
    repairs.push('supervisor returned null/non-object; synthesizing minimal directive');
    return {
      directive: {
        diagnosis: '(no supervisor output)',
        mark_failed: [],
        new_directions: [{
          name: `escape_null_supervisor_${step}`,
          expected: '',
          prompt: 'Re-read lineage.json and attempts.jsonl; propose one orthogonal lever from ROOFLINE.levers consistent with the binding regime.',
        }],
        backtrack_to: '',
        next_direction: '',
        notes: '',
        converge: false,
      },
      repairs,
    };
  }

  let diagnosis = raw.diagnosis != null ? String(raw.diagnosis).trim() : '';
  if (!diagnosis) {
    repairs.push('empty diagnosis replaced with placeholder');
    diagnosis = '(no diagnosis text)';
  }

  const mark_failed = [];
  for (const f of raw.mark_failed || []) {
    const s = f != null ? String(f).trim() : '';
    if (s && !mark_failed.includes(s)) mark_failed.push(s);
  }

  const rawDirs = raw.new_directions;
  const list = Array.isArray(rawDirs) ? rawDirs : (rawDirs != null ? [rawDirs] : []);
  if (!Array.isArray(raw.new_directions)) {
    repairs.push('new_directions was not an array; coerced or cleared');
  }

  const new_directions = [];
  for (const item of list) {
    if (item == null) continue;
    const o = typeof item === 'string' ? { name: item } : item;
    const name = o.name != null ? String(o.name).trim() : '';
    if (!name) {
      repairs.push('dropped new_directions entry with empty name');
      continue;
    }
    let expected = o.expected != null ? String(o.expected).trim() : '';
    let prompt = o.prompt != null ? String(o.prompt).trim() : '';
    if (!expected && !prompt) {
      repairs.push(`direction "${name}": filled default expected/prompt`);
      expected = expected || 'improve geomean vs baseline without regressing correctness';
      prompt = prompt || `Execute one concrete change for "${name}" aligned with ROOFLINE.regime and levers; cite evidence from profile_summary or attempts.jsonl.`;
    } else if (!prompt) {
      repairs.push(`direction "${name}": filled default prompt`);
      prompt = `Deliver the expected outcome: ${expected || '(see name)'}. One focused edit; measure with the COMMANDMENT benchmark.`;
    } else if (!expected) {
      expected = 'see prompt';
    }
    new_directions.push({ name, expected, prompt });
  }

  if (new_directions.length === 0) {
    repairs.push('no valid new_directions after normalization; injected escape_stall');
    new_directions.push({
      name: `escape_stall_${step}`,
      expected: '',
      prompt: 'Pick one lever from ROOFLINE.levers for the current regime; avoid FAILED_DIRECTIONS names; single measurable step.',
    });
  }

  const backtrack_to = raw.backtrack_to != null ? String(raw.backtrack_to).trim() : '';
  const next_direction = raw.next_direction != null ? String(raw.next_direction).trim() : '';
  const notes = raw.notes != null ? String(raw.notes).trim() : '';
  const converge = raw.converge === true;

  const directive = {
    diagnosis,
    mark_failed,
    new_directions,
    backtrack_to,
    next_direction,
    notes,
    converge,
  };
  return { directive, repairs };
}
