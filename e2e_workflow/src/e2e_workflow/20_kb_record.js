// ===========================================================================
// MODULE B — record this run in the deployment knowledge base.
//
// STRICTLY AFTER the workflow_return.json persist above, and that ordering is not stylistic.
// workflow_return.json is the pinned contract run_e2e.py depends on, documented as the workflow's
// final act precisely because this process gets SIGKILLed. Module B is the one operation in the
// whole workflow that waits on a NETWORK rather than a GPU, and putting it in front of the persist
// would trade the run's canonical artifact for a knowledge-base entry. Feeding the KB that same file
// as --result has a second benefit: the record cannot drift from the report, because they are the
// same bytes.
//
// EVERY --apply IS PERMANENT. The service exposes no DELETE for /v1/kb/*, so a wrong record cannot
// be cleaned up, only outranked. Hence the gates below are conservative by design.
// ===========================================================================
// The Director's verdict, not the raw ratio, decides whether a same-session A/B is a real win: a
// declared no-win can still carry a ratio ABOVE 1.0 when a low outlier in the base leg depresses its
// median, so the two mechanically-identical legs' ratio squeaks over 1.0 while the run bought
// nothing (see the 20260822 gemma-4-26B-A4B run: 1.0215x raw same-session, but 0.9453x vs the
// provided baseline, validation_status=validated_no_win, empty overlay + empty patch). Keying the
// write on the ratio alone wrote that BELOW-baseline number in as the exact-id champion and buried
// the real framework-layer win beneath it. A KB write is PERMANENT (no delete), so a verdict the
// Director already computed to mean "no win" must never mint a champion record, whatever the ratio
// rounded to. `startsWith` also catches the `..._no_number_used_carried_ab` fallback spelling.
const kbNoWinVerdict = ['validated_no_win', 'recovered_no_gain']
  .some((s) => String(wfReturn.validation_status || '').startsWith(s));
if (E2E_WARM_START_ON && KB_DIMS && KB_DIMS.gfx && want('final') && EVAL_DIR &&
    !A.kb_defer_to_interface &&
    wfReturn.throughput_speedup > 1.0 && wfReturn.final_throughput_tok_s > 0 && !kbNoWinVerdict) {
  // Computed HERE, deterministically, from facts this script already holds — never asked of an
  // agent. `direction` is inside _content_digest, so a label that varies between two runs of the
  // same configuration mints a second session instead of replacing the first, and the page fills
  // with near-identical entries that outrank each other by noise. '' (unlabeled) is the honest
  // answer when nothing distinct happened; the store collapses unlabeled records with nothing.
  const kbDirection = [
    (curFlags !== INIT_FLAGS || curEnv !== INIT_ENV) ? 'config' : '',
    (acceptedKernels.length || acceptedHeads.length) ? 'kernels' : '',
    FAST_MODE ? 'fast' : (DEEP_MODE ? 'deep' : ''),
  ].filter(Boolean).join('+');
  const writeCmd =
    KB_ENV_PRELUDE +
    `python3 ${shq(E2E_STORE_SCRIPT)} write ${kbIdentityFlags()} ` +
    `${kbPlaneFlags()} --result ${shq(EVAL_DIR + '/workflow_return.json')} ` +
    // --require-win re-asks, inside the writer, the same question the `if` above already answered.
    // Not redundancy for its own sake: run_e2e.py's salvage path is a SECOND writer, and the two
    // must refuse the same results. One gate implementation (e2e_store.win_gate), two callers —
    // the arrangement that stops the next gate fix from reaching only half the write paths.
    `--require-win --direction ${shq(kbDirection)} ` +
    `--measured-by ${shq('e2e_workflow:' + BACKEND)} --apply`;
  try {
    const written = await safeAgent(
      `You are the e2e knowledge-base writer. Run EXACTLY this command and return its JSON stdout ` +
      `verbatim as StructuredOutput. It pretty-prints over several lines; return the whole object. ` +
      `Do NOT edit the command, do NOT re-run it with different flags, and do NOT retry it on ` +
      `failure — every write it performs is PERMANENT (the store has no delete), so a second ` +
      `attempt with adjusted arguments creates a second permanent record rather than fixing the ` +
      `first. If it fails, return what it printed along with the error.\n` +
      '```bash\n' + writeCmd + ` | tee ${shq(EVAL_DIR + '/kb_write.json')}\n` + '```',
      { phase: 'Validate', label: 'kb:write',
        schema: obj({ ok: { type: 'boolean' }, applied: { type: 'boolean' },
          session_id: { type: 'string' }, files: arrStr, rungs: arrObj }, []) },
      1);
    wfReturn.kb_written = written || { ok: false, error: 'writer agent returned nothing' };
    const rungs = (written && written.rungs) || [];
    log(`[kb] wrote this run: session=${(written && written.session_id) || '?'} ` +
      `direction='${kbDirection}' rungs=${rungs.filter(r => r.written).length}/${rungs.length} ` +
      `promoted=${rungs.filter(r => r.promoted).length}` +
      `${rungs.filter(r => r.error).map(r => ` [${r.tier}: ${r.error}]`).join('')}`);
  } catch (e) {
    // Non-fatal, exactly like the persist above. The run's result is already on disk and already
    // returned; failing to record it is a lost opportunity, not a lost measurement.
    wfReturn.kb_written = { ok: false, error: String(e).slice(0, 200) };
    log(`[kb] write failed (NON-FATAL — the run's own artifacts are unaffected): ${String(e)}`);
  }
} else if (E2E_WARM_START_ON && KB_DIMS) {
  const why = !KB_DIMS.gfx ? 'no gfx (an arch-less record is permanent and unattributable)'
    : !want('final') ? 'this is a phase-partial run, so the number is not final'
    : !(wfReturn.throughput_speedup > 1.0) ? `no win to record (${wfReturn.throughput_speedup}x)`
    : kbNoWinVerdict ? `Director declared no win (${wfReturn.validation_status}) — the ${wfReturn.throughput_speedup}x same-session ratio is box-drift, not a gain`
    : 'no final throughput measured';
  log(`[kb] not recording this run: ${why}.`);
}

// Persist the agent timeline so the ledger can attribute tokens/time to the right role, then render
// the run report (ledger -> clickable role-execution-tree HTML + MD twin) as the very last step. The
// script has no filesystem access, so a tiny agent writes the file and runs the report driver.
// Entirely best-effort: accounting must never fail a run that produced a real speedup.
if (EVAL_DIR && LLM_STATS) {
  try {
    // `instance` = this run's eval dir: a stable per-run identity so the parser can dedupe a
    // nested timeline reached twice (parent-merge + glob) WITHOUT collapsing two distinct lanes
    // that happen to share a shape. Unique per run, and available with no Date/random.
    const tlJson = JSON.stringify({ ...LLM_TL, instance: EVAL_DIR });
    await safeAgent(persistLlmStatsPrompt(EVAL_DIR, tlJson),
      { phase: 'Validate', label: 'file_writer:persist_llm_stats',
        schema: obj({ written: { type: 'boolean' }, path: { type: 'string' }, ok: { type: 'boolean' } }, []) },
      2);
    const nestedNote = LLM_TL.nested.length ? ' plus ' + LLM_TL.nested.length + ' nested kernel run(s)' : '';
    log(`LLM token+time+cost ledger -> ${EVAL_DIR}/reports/trace/ and run report (HTML+MD) -> ` +
        `${EVAL_DIR}/report/. ${LLM_TL.events.length} agent attempts recorded${nestedNote}.`);
  } catch (e) {
    log(`LLM stats emit failed (NON-FATAL — the run is unaffected): ${String(e)}`);
  }
}

return wfReturn;
