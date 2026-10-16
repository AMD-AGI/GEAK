// ---------------------------------------------------------------------------
// Small helpers every workflow uses: the "## Inputs" list in role prompts and
// JSON-schema builders for structured agent output.
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const arrStr = { type: 'array', items: { type: 'string' } };
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };
