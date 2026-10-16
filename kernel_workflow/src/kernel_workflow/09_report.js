// ===========================================================================
// PHASE: Report — rank ALL candidates on the SAME frozen baseline; write table + winner.
// Candidates are THREE classes, all scored against the one frozen input-kernel baseline:
//   (a) each kernel_lane lane (input-language optimize + per-language author) — a source patch;
//   (b) the tuned env backend from Discover (aiter/CK offline tune) — an env/config winner (no patch).
// ===========================================================================
phase('Report');
// (a) lane candidates
const cands = results.map(x => ({
  lang: x.lane.lang, mode: x.lane.mode, kind: 'lane', speedup: x.speedup,
  validation_status: x.r ? x.r.validation_status : 'failed',
  eval_dir: x.r ? x.r.eval_dir : '', patch: x.r ? x.r.final_patch : '', apply_env: '', tuning_artifact: '',
}));
// (b) tuned env backend candidate (only if it beat the frozen baseline)
const tunedSpeedup = Number(bake.tuned_speedup);
if (Number.isFinite(tunedSpeedup) && tunedSpeedup > 1.0 && bake.winner_backend && bake.winner_backend !== 'none') {
  cands.push({
    lang: bake.winner_backend, mode: 'env-tune', kind: 'env', speedup: tunedSpeedup,
    validation_status: 'env', eval_dir: EVAL_DIR, patch: '',
    apply_env: bake.apply_env || '', tuning_artifact: bake.tuning_artifact || '',
  });
  log(`tuned env backend ${bake.winner_backend} -> ${tunedSpeedup.toFixed(2)}x (vs frozen baseline)`);
}
// A candidate only WINS if it actually BEAT the frozen baseline (speedup > 1.0) — the SAME guard the
// tuned env candidate uses above. Slower-than-baseline lanes remain in the table (laneRows) for full
// transparency but must never win "by default"; winner=null => validation_status 'no_winner' => keep
// the original kernel. Without this, a lane that is SLOWER than baseline (e.g. the only non-failed lane
// at 0.17x) would be mislabeled the winner and mislead downstream automation reading .winner.
// ...and only if the director ACCEPTED it. Ranking on speed alone let a lane whose validation came
// back `flagged` (patch did not install, correctness failed, contended box) win the bake-off, be
// applied to the original kernel, and be curated into the KB — a number with a known reason not to
// be believed, promoted by every downstream step. Reported in review of #411. `laneRows` still
// carries every lane for transparency; only eligibility to WIN is tightened.
const ACCEPTED = (c) => String(c.validation_status || '').toLowerCase() === 'accepted';
const ranked = cands.filter(c => c.speedup > 1.0 && (c.kind !== 'lane' || ACCEPTED(c)))
  .sort((a, b) => b.speedup - a.speedup);
const rejectedByGate = cands.filter(c => c.speedup > 1.0 && c.kind === 'lane' && !ACCEPTED(c));
if (rejectedByGate.length) {
  log(`bake-off: ${rejectedByGate.length} lane(s) beat the baseline but are NOT eligible to win ` +
      `(validation_status != accepted): ${rejectedByGate.map(c => `${c.lang}=${c.validation_status}`).join(', ')}`);
}
const winner = ranked[0] || null;
const laneRows = cands;
const bestSpeedup = cands.reduce((m, c) => Math.max(m, Number(c.speedup) || 0), 0);

// Apply instruction depends on the winner class: a lane winner is a source patch (git apply); an
// env-tune winner is a config/env deploy (no source change — record apply_env + tuning_artifact).
const applyStep = (APPLY_TO_ORIGINAL !== 'true' || !winner)
  ? `Do NOT modify the original kernel (report only). Set applied_to_original to "".`
  : (winner.kind === 'env')
    ? `The winner is an ENV/CONFIG tune (backend ${winner.lang}), not a source patch. Do NOT edit kernel ` +
      `source. Record in the report the deploy: apply_env='${winner.apply_env}' and tuning_artifact=` +
      `'${winner.tuning_artifact}'. Set applied_to_original to "${winner.lang} (env: ${winner.apply_env})".`
    : (winner.patch
        ? `APPLY the winning lane patch: cd ${KERNEL_PATH_ORIG} && (git apply ${winner.patch} || git apply --3way ${winner.patch}); ` +
          `if it does not apply cleanly, apply it manually to match intent. Set applied_to_original to the winning language. Do NOT delete anything.`
        : `The winner has no patch file; report only and set applied_to_original to "".`);

const rep = await agentT(
  `You are the bake-off reporter. Write a concise markdown comparison report and, if requested, apply the winner.

## Task
1. Write ${EVAL_DIR}/bakeoff_report.md containing:
   - The op: ${oracle.op_kind} from ${KERNEL_PATH_ORIG} (task_dir ${oracle.task_dir}).
   - The FROZEN baseline (the ONE denominator every candidate shares): the input kernel itself
     (baseline_ms = ${bake.baseline_ms != null ? bake.baseline_ms : 'unknown'} ms; op_bench best_known_ms = ${bake.best_known_ms != null ? bake.best_known_ms : 'unknown'} ms).
   - A candidate table sorted fastest-first with columns: language | mode | kind (lane|env) | speedup vs frozen baseline | validation_status | eval_dir | patch/env.
   - The WINNER (fastest verified) and a one-paragraph rationale. All speedups are directly comparable
     because every candidate — the input-language optimize lane, each authored language, AND the tuned
     env backend — was scored against the SAME frozen input kernel.
     NOTE: WINNER may be null — this means NO candidate beat the frozen baseline (every speedup <= 1.0x,
     e.g. all lanes failed or were slower than the original). In that case state clearly that no
     candidate beat the baseline and the ORIGINAL kernel is RETAINED (do not present a slower-than-1.0x
     candidate as a "winner by default"); still list every candidate in the table for transparency.
2. ${applyStep}

## Inputs
${cfg({
    OP_KIND: oracle.op_kind, KERNEL_PATH: KERNEL_PATH_ORIG, TASK_DIR: oracle.task_dir,
    BASELINE_MS: bake.baseline_ms != null ? bake.baseline_ms : '',
    BEST_KNOWN_MS: bake.best_known_ms != null ? bake.best_known_ms : '',
    CANDIDATES: laneRows,
    WINNER: winner,
    APPLY_TO_ORIGINAL,
  })}

Return ONLY {report_path, applied_to_original, note} as StructuredOutput.`,
  { phase: 'Report', label: 'bakeoff:report', schema: REPORT_SCHEMA });

log(winner
  ? `Bake-off COMPLETE. winner=${winner.lang}:${winner.mode} ${winner.speedup.toFixed(2)}x. Results in ${EVAL_DIR}`
  : `Bake-off COMPLETE. NO candidate beat the frozen baseline (best ${bestSpeedup.toFixed(2)}x <= 1.0x across ${cands.length} candidate(s)) — keeping the ORIGINAL kernel. Results in ${EVAL_DIR}`);

