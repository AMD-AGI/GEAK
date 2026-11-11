// ---- MERGING A RECOVERED CONFIGURATION INTO THE ONE THIS RUN WAS HANDED ----
// A stored e2e record holds a WHOLE launch configuration. The launcher's flag/env expansion is
// last-wins, so a key both sides pin would apply in join order; merge here instead, naming every key
// once and reporting what changed.

// Flags that may legitimately repeat (deduping `--lora-path` would drop adapters); every other flag
// here is a scalar. Grow it only with evidence.
const REPEATABLE_FLAGS = new Set(['--lora-path', '--lora-paths']);

const _looksLikeFlag = (t) => /^--?[A-Za-z]/.test(t);

/** `--a 1 --b=2 --c` -> [{name,value,eq}]. Unrecognized tokens ride along under name=null. */
function parseFlagString(s) {
  const toks = String(s || '').trim().split(/\s+/).filter(Boolean);
  const items = [];
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    if (!_looksLikeFlag(t)) { items.push({ name: null, value: t, eq: false }); continue; }
    const eq = t.indexOf('=');
    if (eq > 0) { items.push({ name: t.slice(0, eq), value: t.slice(eq + 1), eq: true }); continue; }
    // Everything up to the next flag is this flag's value; `--foo a b` stays one item so a
    // multi-valued flag is overridden as a unit rather than half-overridden.
    const vals = [];
    while (i + 1 < toks.length && !_looksLikeFlag(toks[i + 1])) vals.push(toks[++i]);
    items.push({ name: t, value: vals.length ? vals.join(' ') : null, eq: false });
  }
  return items;
}

const renderFlagItems = (items) => items.map(
  (it) => (it.name == null ? it.value
    : it.value == null ? it.name
      : it.eq ? `${it.name}=${it.value}` : `${it.name} ${it.value}`)).join(' ').trim();

/** `K=V K2=V2` -> [{name,value}]. A token that is not an assignment rides along under name=null. */
function parseEnvString(s) {
  return String(s || '').trim().split(/\s+/).filter(Boolean).map((t) => {
    const m = /^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/.exec(t);
    return m ? { name: m[1], value: m[2] } : { name: null, value: t };
  });
}

const renderEnvItems = (items) => items.map(
  (it) => (it.name == null ? it.value : `${it.name}=${it.value}`)).join(' ').trim();

/**
 * Key-by-key merge of a recovered configuration onto this run's baseline. The recovered value wins
 * a collision IN PLACE (it is what is being tested), so baseline order survives and each key appears
 * once. Returns `{ merged, overrides, added, unchanged }`; `overrides` lists the knobs where this box
 * disagreed with the record.
 */
function mergeConfig(baseline, recovered, { parse, render, repeatable }) {
  const out = parse(baseline).map((it) => ({ ...it }));
  const at = new Map();
  out.forEach((it, i) => { if (it.name && !repeatable.has(it.name)) at.set(it.name, i); });
  const overrides = [], added = [], unchanged = [];
  for (const it of parse(recovered)) {
    if (it.name == null || repeatable.has(it.name) || !at.has(it.name)) {
      out.push({ ...it });
      if (it.name != null) {
        added.push(it.name);
        if (!repeatable.has(it.name)) at.set(it.name, out.length - 1);
      }
      continue;
    }
    const i = at.get(it.name), prev = out[i];
    if (String(prev.value == null ? '' : prev.value) === String(it.value == null ? '' : it.value)) {
      unchanged.push(it.name);
      continue;
    }
    overrides.push({ flag: it.name, baseline_value: prev.value, recalled_value: it.value });
    out[i] = { ...it };
  }
  return { merged: render(out), overrides, added, unchanged };
}

const mergeFlags = (baseFlags, storedFlags) => mergeConfig(baseFlags, storedFlags,
  { parse: parseFlagString, render: renderFlagItems, repeatable: REPEATABLE_FLAGS });
const mergeEnv = (baseEnv, storedEnv) => mergeConfig(baseEnv, storedEnv,
  { parse: parseEnvString, render: renderEnvItems, repeatable: new Set() });

/**
 * Which trial actually ran, and what it measured: the LAST trial with a positive throughput (after a
 * repair, trials[0] is the failed verbatim replay), and `best_throughput_tok_s` only when it differs
 * from the baseline (the role returns the baseline there when nothing was kept).
 */
function measuredTrialOf(sweep, baselineTput) {
  const trials = (sweep && Array.isArray(sweep.trials)) ? sweep.trials : [];
  const ran = trials.filter((t) => t && Number(t.throughput_tok_s) > 0);
  const trial = ran.length ? ran[ran.length - 1] : (trials[0] || {});
  const best = Number(sweep && sweep.best_throughput_tok_s) || 0;
  const measured = ran.length ? Number(trial.throughput_tok_s)
    : (best && best !== baselineTput ? best : 0);
  return { trial, measured };
}

/**
 * Knobs the merged configuration asked for that the trial that ran does not carry, recovered by
 * subtraction (`repair.dropped_flags` is optional and often empty), so `applied_partial` is right.
 */
function droppedByDiff(mergedFlags, mergedEnv, applied) {
  const app = String(applied || '').trim();
  if (!app) return [];
  const keyOf = (t) => {
    const f = /^(--?[A-Za-z][^=\s]*)/.exec(t);
    if (f) return f[1];
    const e = /^([A-Za-z_][A-Za-z0-9_]*)=/.exec(t);
    return e ? e[1] : null;
  };
  const have = new Set();
  app.split(/\s+/).forEach((t) => { const k = keyOf(t); if (k) have.add(k); });
  const want = [];
  parseFlagString(mergedFlags).forEach((it) => { if (it.name) want.push(it.name); });
  parseEnvString(mergedEnv).forEach((it) => { if (it.name) want.push(it.name); });
  return want.filter((k) => !have.has(k));
}

/** `--context-length 11264 -> 13312; MAX_MODEL_LEN 13312 -> 16384`, or "". For prompts and logs. */
const describeOverrides = (overrides) => (overrides || []).map(
  (o) => `${o.flag} ${o.baseline_value == null ? '(unset)' : o.baseline_value} -> ` +
    `${o.recalled_value == null ? '(set)' : o.recalled_value}`).join('; ');

