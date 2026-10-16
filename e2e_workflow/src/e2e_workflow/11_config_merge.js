// ===========================================================================
// MERGING A RECOVERED CONFIGURATION INTO THE ONE THIS RUN WAS HANDED
// ===========================================================================
// A stored e2e record holds a WHOLE launch configuration, not a delta — the only form replayable on
// a box whose baseline nobody recorded. Replaying it means combining it with the baseline THIS run
// was handed, which under Hyperloom it does not own (interface/run_e2e.py seeds
// `initial_extra_server_args` from EXPLORE and sets `config_tune: "false"`, so no later sweep
// corrects anything either).
//
// That combination used to be a string concatenation performed by an agent, and it was wrong in a
// way nothing could see. `adapters/sglang.sh` expands `$EXTRA_SERVER_ARGS`/`$EXTRA_ENV` straight
// into argparse and `env`, both last-wins. So when the baseline and the record each pin
// `--context-length`, which one applies is decided by the order the agent happened to write them
// in. Lose that coin flip and the server runs the baseline twice: ~0% delta, no complaint in any
// log — the flag WAS honoured, just not with the recorded value — and a record that wins is filed
// as `rejected`, the one outcome the whole warm start exists to avoid.
//
// So the merge is arithmetic, done here, and it reports what it did — the same rule the launcher
// already applies to one flag by hand (`adapters/sglang.sh` and `--watchdog-timeout`), generalized
// and moved to where the string is built. The output names every key once, so last-wins never votes.

// Flags that legitimately appear more than once. Deliberately tiny: `--lora-path` is a list and
// deduping it would silently drop adapters, while everything else in these configs is a scalar
// where a second occurrence is a mistake we are here to remove. Grow it only with evidence.
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
 * Key-by-key merge of a recovered configuration onto this run's baseline.
 *
 * The recovered value WINS on a collision — it is the thing being tested, and applying the
 * baseline's value instead would test nothing. It wins IN PLACE, so the baseline's ordering
 * survives and the output contains each key once.
 *
 * Returns `{ merged, overrides, added, unchanged }`. `overrides` is the part that has to reach a
 * human and the tuner: it is the list of knobs where this box disagreed with the record, and it is
 * the first thing to look at when a replay comes back neutral.
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
 * Which trial actually ran, and what it measured. Once a repair pass exists, `trials[0]` and
 * `best_throughput_tok_s` are both the wrong place to read it from:
 *   * after a repair `trials[0]` is the corpse of the verbatim replay and the LAST trial is the one
 *     that came up. Taking `kept`/`parity`/notes from the corpse makes a repaired config unadoptable
 *     by construction and feeds its launch error to the `inert` regex, which then reads a successful
 *     repair as "the config never took effect".
 *   * the role returns THE BASELINE in `best_throughput_tok_s` when nothing was kept. Filing that as
 *     `measured_tok_s` writes a perfect 0.00% delta onto a record that was never measured.
 *
 * So: the last trial carrying a positive throughput, and `best` only when it is a number the
 * baseline cannot be mistaken for.
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
 * Knobs the merged configuration asked for that the thing which actually ran does not carry.
 *
 * `repair.dropped_flags` is optional and the role often expresses the repair as a second trial and
 * leaves it empty. The string it benched is on the trial either way, so the drop list is recoverable
 * by subtraction — and without it `applied_partial` reads false on a run that dropped something,
 * which is the difference between "the record lost" and "most of the record lost".
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

