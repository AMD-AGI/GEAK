#!/usr/bin/env node
/*
 * replay_runs.js — re-decide finished runs with today's workflow, on no GPU.
 *
 * Gate changes in kernel_mega.js are otherwise validated only by launching a wave. A finished run
 * tree already holds the inputs: setup_ab*.json records every benchmark invocation (arm, guard,
 * rep, rank-mean, rank-max, exit code, path markers). This script recomputes the paired effects and
 * runs them through the current gate arithmetic, lifted verbatim from kernel_mega.js between its
 * `<<REPLAY:pc_gate>>` markers. The output is a deterministic snapshot: record it, change the
 * workflow, re-run with --check and diff.
 *
 *   node scripts/replay_runs.js --runs <dir> [--runs <dir> ...] [--json] \
 *        [--snapshot <file>] [--check <file>]
 *
 * Exit codes: 0 ok / snapshot matches, 1 usage or corpus error, 4 --check found a difference.
 *
 * Agent judgement is not replayed (it is recorded, not reproducible); this covers the deterministic
 * aggregation and gating layer.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const WF = path.join(ROOT, 'kernel_mega.js');

// ---------------------------------------------------------------- args
const argv = process.argv.slice(2);
const roots = [];
let asJson = false, snapshotOut = null, checkAgainst = null;
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--runs') roots.push(argv[++i]);
  else if (a === '--json') asJson = true;
  else if (a === '--snapshot') snapshotOut = argv[++i];
  else if (a === '--check') checkAgainst = argv[++i];
  else if (a === '-h' || a === '--help') { console.log(fs.readFileSync(__filename, 'utf8').split('*/')[0]); process.exit(0); }
  else { console.error(`replay_runs: unknown argument: ${a}`); process.exit(1); }
}
if (!roots.length) { console.error('replay_runs: at least one --runs <dir> is required'); process.exit(1); }

// ---------------------------------------------------------------- lift the live gate
// Verbatim, between markers. Missing markers stop the replay rather than falling back to a stale copy.
function liftGate() {
  const src = fs.readFileSync(WF, 'utf8');
  const m = src.match(/\/\/ <<REPLAY:pc_gate>>[\s\S]*?\n([\s\S]*?)\n\s*\/\/ <<\/REPLAY:pc_gate>>/);
  if (!m) {
    console.error('replay_runs: the <<REPLAY:pc_gate>> markers are missing from kernel_mega.js.');
    console.error('  The gate arithmetic is lifted from between them; without them this script would');
    console.error('  be replaying a copy that can drift. Restore the markers around the block that');
    console.error('  computes mLo/mHi/wantSign/tooSmall/absurd/overshoot/ok.');
    process.exit(1);
  }
  return new Function('pc', 'POSITIVE_CONTROL', `
    const lo = Number(POSITIVE_CONTROL.expected_pct_lo);
    const hi = Number(POSITIVE_CONTROL.expected_pct_hi);
    const got = Number(pc.measured_pct);
${m[1]}
    return { ok, ran, tooSmall, absurd, overshoot, undershoot, nullQuiet, resolved, signUnanimous, mLo, mHi, wantSign };
  `);
}
const gate = liftGate();

// ---------------------------------------------------------------- corpus
function walk(dir, out = []) {
  let ents;
  try { ents = fs.readdirSync(dir, { withFileTypes: true }); } catch { return out; }
  for (const e of ents) {
    const p = path.join(dir, e.name);
    // Skip workspaces (full AITER checkouts with thousands of config files).
    if (e.isDirectory()) {
      if (/^(workspace|baseline|baseline_copy|optimized|validation_workspace|inherited_assets|\.git|__pycache__)$/.test(e.name)) continue;
      walk(p, out);
    } else if (/^setup_ab.*\.json$/.test(e.name)) out.push(p);
  }
  return out;
}

// ---------------------------------------------------------------- schema normalisation
// Accept both artifact shapes ({arms, base, guards, runs} and {records}); an artifact that parses but
// yields no usable pair is a loud error, never a quiet zero. --check fails on an empty corpus too.
const NUMS = ['max', 'e2e_max_ms', 'rank_max_ms', 'latency_max_ms'];
const MEANS = ['mean', 'e2e_mean_ms', 'rank_mean_ms'];

// An arm is the base of its pair when its switch is off; an env whose every value is falsy (e.g. a
// switch set to 0) counts as off.
function envInert(env) {
  const e = env || {};
  const keys = Object.keys(e);
  if (!keys.length) return true;
  return keys.every((k) => e[k] === 0 || e[k] === '0' || e[k] === '' || e[k] === false || e[k] === null);
}

function normalise(doc) {
  const raw = Array.isArray(doc.runs) ? doc.runs
    : Array.isArray(doc.records) ? doc.records : [];
  // Without `rep`, derive it from per-arm occurrence order: in an interleaved A/B the k-th occurrence
  // of each arm is pair k.
  const seenPerArm = new Map();
  const runs = raw.map((r) => {
    const arm = r.arm;
    const n = (seenPerArm.get(arm) || 0); seenPerArm.set(arm, n + 1);
    const max = NUMS.map((k) => r[k]).find((v) => Number.isFinite(Number(v)));
    const mean = MEANS.map((k) => r[k]).find((v) => Number.isFinite(Number(v)));
    return {
      ...r, arm,
      guard: r.guard || (r.tokens != null && r.route ? `${r.tokens}_${r.route}` : undefined),
      rep: r.rep != null ? r.rep : n,
      max: Number.isFinite(Number(max)) ? Number(max) : Number(mean),
    };
  });
  // Arms come from the declaration if present, otherwise from the records.
  let armMeta = new Map((Array.isArray(doc.arms) ? doc.arms : [])
    .map((a) => [a.name, { env: a.env || {}, ws: String(a.ws || a.tree || '') }]));
  if (!armMeta.size) {
    armMeta = new Map();
    for (const r of runs) {
      if (!r.arm || armMeta.has(r.arm)) continue;
      armMeta.set(r.arm, { env: r.env || {}, ws: String(r.ws || r.tree || '') });
    }
  }
  const arms = [...armMeta.keys()];
  const guards = Array.isArray(doc.guards) && doc.guards.length
    ? doc.guards : [...new Set(runs.map((r) => r.guard).filter(Boolean))];
  const base = doc.base && arms.includes(doc.base)
    ? doc.base
    : (arms.find((a) => envInert(armMeta.get(a).env)) || arms[0]);
  return { runs, arms, armMeta, guards, base };
}

// A run is usable only if it completed, reported a latency and printed the path marker; otherwise
// it is void.
function voidReason(r) {
  if (r.rc !== 0) return `rc=${r.rc}`;
  if (r.ok === false) return 'ok=false';
  if (!Number.isFinite(Number(r.max))) return 'no rank-max';
  if (r.marker_ok === false) return 'path marker absent';
  return null;
}

// Paired, never pooled: use the within-rep difference. A rep with a void arm drops whole.
function pairedPct(runs, guard, armA, armB) {
  const byRep = new Map();
  for (const r of runs) {
    if (r.guard !== guard) continue;
    if (r.arm !== armA && r.arm !== armB) continue;
    const k = `${r.rep}`;
    if (!byRep.has(k)) byRep.set(k, {});
    // Reps re-run under the same index: last write wins, as the benchmark engineer reports them.
    byRep.get(k)[r.arm] = r;
  }
  const deltas = []; const dropped = [];
  for (const [rep, pair] of [...byRep.entries()].sort()) {
    const a = pair[armA], b = pair[armB];
    if (!a || !b) { dropped.push(`rep${rep}:missing-arm`); continue; }
    const va = voidReason(a), vb = voidReason(b);
    if (va || vb) { dropped.push(`rep${rep}:${va ? armA + '(' + va + ')' : ''}${va && vb ? '+' : ''}${vb ? armB + '(' + vb + ')' : ''}`); continue; }
    // Positive = armB faster than armA on rank-max (a collective is gated by its slowest rank).
    deltas.push((Number(a.max) - Number(b.max)) / Number(a.max) * 100);
  }
  if (!deltas.length) return { pct: NaN, n: 0, dropped, deltas, spread: NaN };
  const sorted = [...deltas].sort((x, y) => x - y);
  const med = sorted.length % 2 ? sorted[(sorted.length - 1) / 2]
    : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
  return { pct: med, n: deltas.length, dropped, deltas, spread: sorted[sorted.length - 1] - sorted[0] };
}

// Two bands: the historical speedup control and the packaged synthetic slowdown, to show the gate is
// sign-agnostic on real data.
const BANDS = {
  'speedup_3.55..4.93': { expected_pct_lo: 3.55, expected_pct_hi: 4.93 },
  'slowdown_-5.0..-2.5': { expected_pct_lo: -5.0, expected_pct_hi: -2.5, implausible_pct: 15.0 },
  // The same slowdown band declared as a target: magnitude must change the verdict only on an
  // under-read.
  'slowdown_-5.0..-2.5_constructed': {
    expected_pct_lo: -5.0, expected_pct_hi: -2.5, implausible_pct: 15.0, magnitude: 'constructed' },
};

const records = [];
for (const root of roots) {
  const files = walk(path.resolve(root)).sort();
  if (!files.length) records.push({ kind: 'corpus', root, note: 'no setup_ab*.json found' });
  for (const f of files) {
    let doc;
    try { doc = JSON.parse(fs.readFileSync(f, 'utf8')); }
    catch (e) { records.push({ kind: 'unreadable', file: f, note: String(e.message) }); continue; }
    const { runs, arms, armMeta, guards, base: declaredBase } = normalise(doc);
    // Path is recorded relative to the corpus root so the snapshot is diffable across machines.
    const rel = path.relative(path.resolve(root), f);
    // A setup_ab*.json with no arm, guard or timing is reported loudly. A single-arm file (a seeded
    // baseline) is not an error and is reported separately.
    if (runs.length && arms.length === 1 && guards.length && runs.some((r) => Number.isFinite(r.max))) {
      records.push({
        kind: 'single_arm', root: path.basename(path.resolve(root)), file: rel,
        note: `${runs.length} run(s) of one arm (${arms[0]}) over ${guards.length} guard(s) — a baseline, not an A/B; no pair to re-decide.`,
      });
      continue;
    }
    if (!runs.length || arms.length < 2 || !guards.length || !runs.some((r) => Number.isFinite(r.max))) {
      records.push({
        kind: 'unusable', root: path.basename(path.resolve(root)), file: rel,
        note: `parsed but not replayable: ${runs.length} record(s), ${arms.length} arm(s), ` +
              `${guards.length} guard(s), ${runs.filter((r) => Number.isFinite(r.max)).length} with a rank-max. ` +
              `Expected per record: arm, guard (or tokens+route), and one of ${NUMS.join('/')}.`,
      });
      continue;
    }

    const voids = runs.map((r) => ({ arm: r.arm, guard: r.guard, rep: r.rep, why: voidReason(r) }))
      .filter((v) => v.why);
    records.push({
      kind: 'ab', root: path.basename(path.resolve(root)), file: rel,
      arms, guards, n_runs: runs.length, n_void: voids.length,
      voids: voids.slice(0, 8),
    });

    // Which two arms: a control's arms differ only by the switch (same workspace, different env), so pair
    // within a workspace when it holds two arms, and fall back to the declared base otherwise.
    const wsOf = (n) => String((armMeta.get(n) || {}).ws || '');
    const nullArm = arms.find((a) => /null/i.test(a));
    const pairs = [];
    const byWs = new Map();
    for (const a of arms) { if (a === nullArm) continue; const k = wsOf(a); if (!byWs.has(k)) byWs.set(k, []); byWs.get(k).push(a); }
    for (const [, group] of byWs) {
      if (group.length < 2) continue;
      // Within a workspace the base is the arm with no env override.
      const off = group.find((a) => envInert((armMeta.get(a) || {}).env)) || group[0];
      for (const on of group) if (on !== off) pairs.push({ cand: on, base: off, how: 'same-workspace' });
    }
    if (!pairs.length) {
      for (const a of arms) if (a !== declaredBase && a !== nullArm) pairs.push({ cand: a, base: declaredBase, how: 'declared-base' });
    }
    for (const guard of guards) {
      for (const { cand, base, how } of pairs) {
        const eff = pairedPct(runs, guard, base, cand);
        // The null arm is a copy of the run's own baseline, so it is always paired against that baseline.
        let nul = nullArm ? pairedPct(runs, guard, declaredBase, nullArm) : { pct: NaN, n: 0 };
        // A null arm reported only as a summary is used and labelled as recorded (weaker than recomputed,
        // but better than NaN).
        let nullSrc = nullArm ? 'recomputed' : 'absent';
        if (!nul.n && Number.isFinite(Number(doc.null_arm_pct))) {
          nul = { pct: Number(doc.null_arm_pct), n: Array.isArray(doc.null_pairs_pct) ? doc.null_pairs_pct.length : 0 };
          nullSrc = 'recorded';
        }
        // The gate reads individual pairs (sign agreement, worst null pair), so supply them as the live
        // workflow does.
        const pc = {
          measured_pct: eff.pct, null_arm_pct: nul.pct, ran: eff.n > 0,
          control_pairs_pct: eff.deltas, null_pairs_pct: nul.deltas || doc.null_pairs_pct,
        };
        // Replay against the band the packaged task ships: the snapshot answers what today's workflow would
        // decide.
        for (const [bandName, band] of Object.entries(BANDS)) {
          const v = gate(pc, band);
          v.wrongSign = Number.isFinite(eff.pct) && Math.sign(eff.pct) !== v.wantSign;
          records.push({
            kind: 'gate', root: path.basename(path.resolve(root)), file: rel, guard,
            pair: `${cand} vs ${base} (${how})`, band: bandName,
            measured_pct: round2(eff.pct), pairs: eff.n, spread_pp: round2(eff.spread),
            null_pct: round2(nul.pct), null_pairs: nul.n, null_src: nullSrc,
            dropped: eff.dropped.length ? eff.dropped.slice(0, 4) : undefined,
            verdict: verdictOf(v),
          });
        }
      }
    }
  }
}

function round2(x) { return Number.isFinite(x) ? Number(x.toFixed(2)) : null; }
function verdictOf(v) {
  if (!v.ran) return 'NOT_RUN';
  if (v.absurd) return 'FAIL_ABSURD';
  if (v.tooSmall) return v.wrongSign ? 'FAIL_WRONG_SIGN' : 'FAIL_INSENSITIVE';
  if (v.overshoot) return v.nullQuiet ? 'PASS_OVERSHOOT' : 'FAIL_OVERSHOOT_LOUD_NULL';
  if (v.undershoot) return 'PASS_UNDERSHOOT';
  return 'PASS';
}

// ---------------------------------------------------------------- report
const snap = { corpus: roots.map((r) => path.basename(path.resolve(r))).sort(), records };
const text = asJson ? JSON.stringify(snap, null, 1) : render(snap);

function render(s) {
  const L = [];
  L.push(`replay corpus: ${s.corpus.join(', ')}`);
  const ab = s.records.filter((r) => r.kind === 'ab');
  const gates = s.records.filter((r) => r.kind === 'gate');
  const bad = s.records.filter((r) => r.kind === 'unreadable' || r.kind === 'corpus' || r.kind === 'unusable');
  for (const r of bad) L.push(`  !! ${r.kind}: ${r.file || r.root} — ${r.note}`);
  for (const r of s.records.filter((x) => x.kind === 'single_arm')) L.push(`  -- ${r.file} — ${r.note}`);
  L.push('');
  L.push('A/B artifacts');
  for (const r of ab) {
    L.push(`  ${r.root}/${r.file}`);
    L.push(`    arms=[${r.arms.join(', ')}] guards=[${r.guards.join(', ')}] runs=${r.n_runs} void=${r.n_void}`);
    for (const v of r.voids) L.push(`      VOID ${v.arm}/${v.guard}/rep${v.rep}: ${v.why}`);
  }
  L.push('');
  L.push('gate verdicts (today\'s arithmetic, recorded data)');
  for (const g of gates) {
    L.push(`  ${g.root}/${g.file} [${g.guard}] ${g.pair}`);
    L.push(`    band=${g.band} measured=${g.measured_pct}% (${g.pairs} pairs, spread ${g.spread_pp}pp) ` +
           `null=${g.null_pct}% (${g.null_pairs}, ${g.null_src}) -> ${g.verdict}` +
           (g.dropped ? `  dropped: ${g.dropped.join(' ')}` : ''));
  }
  return L.join('\n') + '\n';
}

// An empty corpus is not a pass: --check needs gate verdicts to compare.
const nGates = records.filter((r) => r.kind === 'gate').length;
const nUnusable = records.filter((r) => r.kind === 'unusable' || r.kind === 'unreadable').length;
if (!nGates) {
  console.error(`replay_runs: the corpus produced NO gate verdicts` +
    (nUnusable ? ` (${nUnusable} artifact(s) parsed but were not replayable — see above)` : '') + '.');
  console.error('  Nothing was re-decided, so this run cannot validate a workflow change. Point --runs');
  console.error('  at a tree containing a completed A/B, or fix the artifact schema it reports above.');
  if (checkAgainst || snapshotOut) { process.stdout.write(text); process.exit(1); }
}

if (snapshotOut) { fs.writeFileSync(snapshotOut, text); console.log(`snapshot written: ${snapshotOut}`); }
if (checkAgainst) {
  const want = fs.existsSync(checkAgainst) ? fs.readFileSync(checkAgainst, 'utf8') : null;
  if (want === null) { console.error(`replay_runs: no snapshot at ${checkAgainst} to check against`); process.exit(1); }
  if (want === text) { console.log(`replay matches ${path.basename(checkAgainst)} — no historical verdict changed.`); process.exit(0); }
  const a = want.split('\n'), b = text.split('\n');
  console.error(`replay DIFFERS from ${path.basename(checkAgainst)} — a change to the workflow re-decided a finished run:`);
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if (a[i] !== b[i]) { console.error(`  - ${a[i] === undefined ? '(eof)' : a[i]}`); console.error(`  + ${b[i] === undefined ? '(eof)' : b[i]}`); }
  }
  console.error('\nIf that is what the change was for, re-record with --snapshot. If it is not, the change');
  console.error('has a side effect on decisions you did not intend, and the diff above is where.');
  process.exit(4);
}
if (!snapshotOut) process.stdout.write(text);
