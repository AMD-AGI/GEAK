// Structured-output emulation for the standalone workflow runtime.
//
// Claude Code's Workflow `agent(prompt, {schema})` forces a StructuredOutput
// tool call and returns a validated object. A generic coding-agent CLI (claude
// -p / qwen -p) has no such forced tool, so we emulate it: append a JSON-Schema
// instruction to the prompt, then extract + lightly validate the JSON the agent
// prints. Parsing/validation failure is thrown by the caller so the script's
// agentT() retry/degrade-to-null path (kernel_workflow.js) tolerates it exactly
// as it tolerates a real StructuredOutput miss.

// Compact, deterministic pretty-print of a JSON Schema for the prompt.
function schemaToText(schema) {
  try {
    return JSON.stringify(schema, null, 2);
  } catch {
    return String(schema);
  }
}

// The instruction appended to a schema-bearing agent prompt. Mirrors the
// harness contract ("return ONLY the structured JSON ... a StructuredOutput
// tool is forced") but for a plain-text CLI: demand a single fenced JSON block
// as the LAST thing printed, so extraction is unambiguous.
export function schemaInstruction(schema) {
  return (
    `\n\n## OUTPUT CONTRACT (STRICT)\n` +
    `Do all your work first (Bash/Read/Write). Then, as the VERY LAST thing you print, ` +
    `output your result as a SINGLE JSON object inside one \`\`\`json fenced code block, and NOTHING after it.\n` +
    `The JSON MUST validate against this JSON Schema:\n\n\`\`\`json\n${schemaToText(schema)}\n\`\`\`\n\n` +
    `Rules: emit exactly one \`\`\`json block; no comments, no trailing text, no prose after the closing fence; ` +
    `include every required field; use the exact field names from the schema.`
  );
}

// Walk a string and return the substring of the first balanced {...} or [...]
// starting at `start` (respecting strings/escapes). Null if unbalanced.
function balancedSlice(s, start) {
  const open = s[start];
  const close = open === '{' ? '}' : ']';
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === '\\') esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === open) depth++;
    else if (c === close) {
      depth--;
      if (depth === 0) return s.slice(start, i + 1);
    }
  }
  return null;
}

// Extract the intended JSON object/array from free-form agent stdout.
// Priority: (1) last ```json fenced block, (2) last ``` fenced block that
// parses, (3) last top-level balanced {...} / [...] that parses.
export function extractJson(text) {
  if (typeof text !== 'string' || !text.trim()) {
    throw new Error('empty agent output');
  }

  // (1) fenced ```json blocks — take the LAST one (the contract asks for it last).
  const fenceRe = /```(?:json)?\s*([\s\S]*?)```/gi;
  const fenced = [];
  let m;
  while ((m = fenceRe.exec(text)) !== null) fenced.push(m[1].trim());
  for (let i = fenced.length - 1; i >= 0; i--) {
    try { return JSON.parse(fenced[i]); } catch { /* try next */ }
  }

  // (2)/(3) scan for balanced brackets, preferring the LAST parseable candidate.
  const candidates = [];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '{' || text[i] === '[') {
      const slice = balancedSlice(text, i);
      if (slice) { candidates.push(slice); i += slice.length - 1; }
    }
  }
  for (let i = candidates.length - 1; i >= 0; i--) {
    try { return JSON.parse(candidates[i]); } catch { /* try next */ }
  }

  throw new Error('no parseable JSON object found in agent output');
}

// Lightweight JSON-Schema check: enough to catch a wrong-shape response (the
// common failure), not a full validator. Verifies top-level type + required
// keys, and recurses into `properties` for nested objects that are present.
export function validate(obj, schema) {
  const errors = [];
  check(obj, schema, '$', errors);
  return { ok: errors.length === 0, errors };
}

function typeOf(v) {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  return typeof v; // object | string | number | boolean
}

function check(v, schema, path, errors) {
  if (!schema || typeof schema !== 'object') return;

  if (schema.type) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    const actual = typeOf(v);
    // JSON Schema "integer" maps onto JS number.
    const ok = types.some((t) => t === actual || (t === 'integer' && actual === 'number'));
    if (!ok) {
      errors.push(`${path}: expected type ${types.join('|')}, got ${actual}`);
      return; // shape wrong; deeper checks are noise
    }
  }

  // enum: the value must be one of the allowed literals. GEAK branches on these
  // exact strings (director specialty, e2e outcome/status), so an out-of-enum
  // value slipping through would silently mis-route logic. Native's forced
  // StructuredOutput tool enforces this; emulate it here (deep-equal for safety).
  if (Array.isArray(schema.enum)) {
    const inEnum = schema.enum.some((e) => JSON.stringify(e) === JSON.stringify(v));
    if (!inEnum) {
      errors.push(`${path}: value ${JSON.stringify(v)} not in enum [${schema.enum.map((e) => JSON.stringify(e)).join(', ')}]`);
    }
  }

  if ((typeOf(v) === 'object') && schema.properties) {
    if (Array.isArray(schema.required)) {
      for (const key of schema.required) {
        if (!(key in v)) errors.push(`${path}.${key}: missing required field`);
      }
    }
    for (const [key, sub] of Object.entries(schema.properties)) {
      if (key in v) check(v[key], sub, `${path}.${key}`, errors);
    }
  }

  if (typeOf(v) === 'array' && schema.items) {
    v.forEach((el, i) => check(el, schema.items, `${path}[${i}]`, errors));
  }
}
