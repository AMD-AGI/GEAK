// ===========================================================================
// MODULE B — record this run in the deployment knowledge base.
// Strictly after the workflow_return.json persist (Module B can be SIGKILLed while on the network);
// that same file is the --result, so record and report match. EVERY --apply IS PERMANENT (no
// DELETE for /v1/kb/*), hence the conservative gates.
// ===========================================================================
// The Director's verdict, not the raw ratio, decides a real win (a low base-leg outlier can lift a
// no-win above 1.0); `startsWith` also catches the `..._no_number_used_carried_ab` spelling.
const kbNoWinVerdict = ['validated_no_win', 'recovered_no_gain']
  .some((s) => String(wfReturn.validation_status || '').startsWith(s));
if (E2E_WARM_START_ON && KB_DIMS && KB_DIMS.gfx && want('final') && EVAL_DIR &&
    !A.kb_defer_to_interface &&
    wfReturn.throughput_speedup > 1.0 && wfReturn.final_throughput_tok_s > 0 && !kbNoWinVerdict) {
  // Computed deterministically here, never by an agent: `direction` is in _content_digest, so an
  // unstable label would mint duplicate sessions. '' (unlabeled) when nothing distinct happened.
  const kbDirection = [
    (curFlags !== INIT_FLAGS || curEnv !== INIT_ENV) ? 'config' : '',
    (acceptedKernels.length || acceptedHeads.length) ? 'kernels' : '',
    FAST_MODE ? 'fast' : (DEEP_MODE ? 'deep' : ''),
  ].filter(Boolean).join('+');
  const writeCmd =
    KB_ENV_PRELUDE +
    `python3 ${shq(E2E_STORE_SCRIPT)} write ${kbIdentityFlags()} ` +
    `${kbPlaneFlags()} --result ${shq(EVAL_DIR + '/workflow_return.json')} ` +
    // --require-win repeats the `if` above inside the writer: run_e2e.py's salvage path is a second
    // writer, and both must refuse the same results via one gate (e2e_store.win_gate).
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
    // Non-fatal, like the persist above: the result is already on disk and returned.
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

// Last step, best-effort: an agent persists the agent timeline (per-role token/time attribution) and
// renders the run report (role-execution-tree HTML + MD).
if (EVAL_DIR && LLM_STATS) {
  try {
    // `instance` (this run's eval dir) lets the parser dedupe a nested timeline reached twice
    // without collapsing distinct lanes of the same shape; needs no Date/random.
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
