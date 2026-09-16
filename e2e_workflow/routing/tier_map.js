// GEAK Expt-3 — complexity-based model routing: the pure tier map, validators, and the
// escalation core.
//
// WHAT THIS IS. Pure, side-effect-free decision logic for the agent() seam:
//   1. decideFor(opts) — "for THIS scope, which cheap model + which validator?" (static
//      per-scope ALLOWLIST keyed on opts.phase + the static prefix of opts.label). Returns
//      null for un-mapped scopes -> they stay on the pinned strong model.
//   2. checkVerbatimWrite(prompt, result, readFile, expected) — the deterministic artifact
//      oracle for the "Write EXACTLY this content ... verbatim" helper family. It compares the
//      ON-DISK bytes against the authoritative expected content. `readFile(path)->string|null`
//      is injected (fs on Path B / the host on Path A). Per Astra: if no deterministic verifier
//      is available (no readFile) the result is UNVERIFIED == FAILURE, never a pass. `expected`
//      (the host-held {path, content}) is authoritative when supplied; prompt-parsing is only a
//      fallback and is fence-length aware.
//   3. escalate(prompt, opts, decision, run, deps) — cheap attempt -> deterministic gate -> ONE
//      strong fallback that BYPASSES the cheap map, recording EVERY attempt. `run` is injected.
//
// WHY A SEPARATE FILE. Reviewable, unit-tested source of truth. Native Workflow scripts get no
// require()/fs, so e2e_workflow.js / kernel_lane.js INLINE these bodies; routing_dryrun.js
// executes the shipped inline copies and asserts they have not drifted from this file.
//
// REVERSIBILITY. OFF unless enabled -> decideFor() returns null for every scope -> the seam
// sets no opts.model -> byte-identical run. ON -> only allowlisted scopes route.

'use strict';

const MODEL_STRONG = 'claude-opus-4-8';   // pinned default; reached by falling through, never set as override
const MODEL_CHEAP = 'claude-sonnet-5';    // the routed model

// Scope-key separator. Built textually via fromCharCode so the SOURCE FILE contains no literal
// NUL byte (ordinary rg/git review sees text); at runtime it is U+0000, which cannot occur in a
// phase name or a static label prefix.
const SCOPE_SEP = String.fromCharCode(0);

function labelPrefix(label) {
  const s = String(label == null ? '' : label);
  const sp = s.indexOf(' ');            // colon is part of the static identity; a space starts dynamic text
  return sp >= 0 ? s.slice(0, sp) : s;
}

function scopeKey(phase, label) {
  return String(phase == null ? '' : phase) + SCOPE_SEP + labelPrefix(label);
}

// Allowlist, built programmatically so no key literal carries a NUL. Value = { tier, kind }
// where kind selects the validator. Every entry confirmed against source to be a pure
// verbatim-write helper. Seeded narrowest with TWO e2e entries.
const TIER_MAP = Object.freeze((() => {
  const m = {};
  // e2e_workflow.js:5240 — persists canonical workflow_return.json (non-fatal; run_e2e recovers).
  m[scopeKey('Validate', 'file_writer:persist:workflow-return')] = { tier: 'cheap', kind: 'verbatim_write' };
  // e2e_workflow.js:3085 — writes measured_on_this_box.md, verbatim markdown table.
  m[scopeKey('WarmStart', 'warm_start:record-measurements')] = { tier: 'cheap', kind: 'verbatim_write' };
  return m;
})());

const TIER_MODEL = Object.freeze({ cheap: MODEL_CHEAP, strong: MODEL_STRONG });

function routingEnabledFromEnv(env) {
  env = env || (typeof process !== 'undefined' ? process.env : {}) || {};
  return String(env.GEAK_ROUTING || '').trim() === '1';
}

// THE DECISION. Pure. Returns { model, kind, validate } for a mapped cheap scope, or null.
function decideFor(opts, cfg) {
  cfg = cfg || {};
  const enabled = cfg.enabled != null ? cfg.enabled : routingEnabledFromEnv(cfg.env);
  if (!enabled) return null;
  const map = cfg.map || TIER_MAP;
  const tierModel = cfg.tierModel || TIER_MODEL;
  const validators = cfg.validators || VALIDATORS;
  const entry = map[scopeKey(opts && opts.phase, opts && opts.label)];
  if (!entry) return null;                                  // un-mapped -> pinned (fall through)
  const model = tierModel[entry.tier];
  if (!model || model === MODEL_STRONG) return null;        // strong tier == no override
  return { model, kind: entry.kind, validate: validators[entry.kind] || null };
}

// Back-compat convenience: model-only decision.
function routeFor(opts, cfg) {
  const d = decideFor(opts, cfg);
  return d ? d.model : undefined;
}

// ---- Artifact oracle for the verbatim-write family --------------------------------------
const _PATH_RE = /create the file\s+"([^"]+)"/i;
// Fence-length aware: capture the FIRST run of >=3 backticks and match the SAME-length closing
// run on its own line, so a ````markdown outer block is not truncated by an inner ``` block.
const _FENCE_RE = /(`{3,})[a-zA-Z0-9]*\r?\n([\s\S]*?)\r?\n\1(?:\r?\n|$)/;

function extractVerbatimIntent(prompt) {
  const p = String(prompt == null ? '' : prompt);
  const pm = _PATH_RE.exec(p);
  const fm = _FENCE_RE.exec(p);
  return { path: pm ? pm[1] : null, block: fm ? fm[2] : null };
}

// Deterministic artifact gate. `expected` (host-held {path, content}) is authoritative when
// supplied; otherwise fall back to parsing the prompt. `readFile(path)->string|null` opens the
// real artifact. NO readFile -> UNVERIFIED == FAILURE (never a pass). Normalization is declared,
// not "strict byte equality": the on-disk content must equal the expected block, optionally with
// exactly one trailing newline the Write tool may append. Returns { ok, verified, reason, path }.
function checkVerbatimWrite(prompt, result, readFile, expected) {
  const want = expected && expected.content != null
    ? { path: expected.path, block: String(expected.content) }
    : extractVerbatimIntent(prompt);
  if (!want.path || want.block == null) {
    return { ok: false, verified: false, reason: 'no authoritative expected path/content', path: want.path };
  }
  // A receipt claiming a different path is an immediate miss (checked before opening the file).
  if (result && result.path && String(result.path) !== want.path) {
    return { ok: false, verified: false, reason: 'receipt path != expected path', path: want.path };
  }
  if (typeof readFile !== 'function') {
    // No deterministic verifier available (e.g. Path A, no fs, no host verifier bound). Per Astra
    // this MUST NOT pass the gate — represent as unverified failure so acceptance escalates.
    return { ok: false, verified: false, reason: 'no deterministic verifier available (no readFile)',
             path: want.path, weak: true };
  }
  let onDisk;
  try { onDisk = readFile(want.path); } catch (e) { onDisk = null; }
  if (onDisk == null) return { ok: false, verified: true, reason: 'artifact absent on disk', path: want.path };
  const got = String(onDisk);
  const ok = got === want.block || got === want.block + '\n';   // declared trailing-newline tolerance
  return { ok, verified: true,
           reason: ok ? 'artifact matches expected bytes (±1 trailing newline)' : 'artifact bytes differ from expected',
           path: want.path };
}

const VALIDATORS = { verbatim_write: checkVerbatimWrite };

// ---- Escalation core --------------------------------------------------------------------
// Cheap attempt -> deterministic gate -> ONE strong fallback that BYPASSES the cheap map. Fixed
// cap of 2 attempts (1 cheap + 1 strong). EVERY attempt recorded via deps.record. `run(prompt,
// opts)->Promise<result>` is injected. deps.expected (optional host-held {path,content}) is
// passed to the validator as authoritative. Returns { result, accepted, attempts }.
async function escalate(prompt, opts, decision, run, deps) {
  deps = deps || {};
  const readFile = deps.readFile || null;
  const expected = deps.expected || null;
  const log = deps.log || function () {};
  const record = deps.record || function () {};
  const lbl = (opts && opts.label) || 'agent';
  const ph = (opts && opts.phase) || '';
  const attempts = [];

  const cheapOpts = Object.assign({}, opts, { model: decision.model });
  log(`  [route] ${lbl} @${ph} -> ${decision.model} (cheap attempt 1/2)`);
  const r1 = await run(prompt, cheapOpts);
  const v1 = decision.validate ? decision.validate(prompt, r1, readFile, expected)
                               : { ok: !!r1, verified: false, reason: r1 ? 'nonempty (no validator)' : 'empty' };
  const a1 = { label: lbl, phase: ph, model: decision.model, attempt: 1, ok: !!v1.ok, verified: !!v1.verified, reason: v1.reason };
  attempts.push(a1); record(a1);
  if (v1.ok) return { result: r1, accepted: 'cheap', attempts };

  const strongOpts = Object.assign({}, opts, { model: MODEL_STRONG });
  log(`  [route] ${lbl} cheap failed gate (${v1.reason}) — escalating to ${MODEL_STRONG} (attempt 2/2, bypassing cheap map)`);
  const r2 = await run(prompt, strongOpts);
  const v2 = decision.validate ? decision.validate(prompt, r2, readFile, expected)
                               : { ok: !!r2, verified: false, reason: r2 ? 'nonempty (no validator)' : 'empty' };
  const a2 = { label: lbl, phase: ph, model: MODEL_STRONG, attempt: 2, ok: !!v2.ok, verified: !!v2.verified, reason: v2.reason, escalated: true };
  attempts.push(a2); record(a2);
  return { result: r2, accepted: v2.ok ? 'strong' : 'strong-unverified', attempts };
}

module.exports = {
  MODEL_STRONG, MODEL_CHEAP, SCOPE_SEP, TIER_MAP, TIER_MODEL, VALIDATORS,
  labelPrefix, scopeKey, routingEnabledFromEnv, decideFor, routeFor,
  extractVerbatimIntent, checkVerbatimWrite, escalate,
};
