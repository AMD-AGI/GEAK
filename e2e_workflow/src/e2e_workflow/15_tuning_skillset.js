// ===========================================================================
// PHASE: TuningSkillset — the VENDORED tuning skillset, run WHOLE and STANDALONE, BEFORE HeadKernel.
// TUNING_SKILLSET_DIR is one hash-pinned tree (scripts/tuning_skillset_sync.py). tuning_specialist
// runs its full loop with its own pre/post A/B, so the head track then works on what is still hot,
// against a tuned reference. An accept folds into curFlags/curEnv and the profile is re-taken.
// ===========================================================================
if (want('tune') && TUNING_SKILLSET_ENABLED && !searchClosed('TuningSkillset') && !tuningWindowShort()) {
  phase('TuningSkillset');
  log(`Tuning skillset: ${TUNING_SKILLSET_DIR} (whole, standalone, pre-HeadKernel); ` +
    `no op cap, tuning-kb ${TUNING_KB_ENABLED ? 'ENABLED' : 'DISABLED (blind eval)'}.`);
  tuning = await safeAgent(
    roleAgent('tuning_specialist', 'tune',
      'Tune the live stack with the skillset, measure your OWN in-session isolated-server pre/post A/B, ' +
      'prove engagement, and hand back a deploy bundle that reaches production through EVAL_DIR/final/.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD,
      BASELINE_THROUGHPUT: BASELINE_TPUT, CURRENT_THROUGHPUT: curTput,
      CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_OVERLAY: curOverlay,
      MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
      NOISE_BAND_PCT: NOISE_BAND, ACCURACY_GATE, ...ACCURACY_INPUTS,
      PROFILE_TOPN: profile ? profile.profile_topN_json : '',
      TUNING_TARGETS: (strategy && strategy.head_candidates) || headQueue || [],
      TUNING_SKILLSET_DIR, TUNING_KB_ENABLED, SKILL_DIR: WORKFLOW_DIR,
      // The always-fires channel (as for the Architect and config_tuner), gated like warmStartBlock()
      // so a blind eval stays blind through both channels.
      ...(TUNING_KB_ENABLED ? KB_REF_INPUTS : {}),
      ...(TUNING_KB_ENABLED && KB_CACHE_DIR ? { KB_CACHE_DIR } : {}),
      // The read side of the tuned-table write below (kernel store, keyed per op/backend/gfx); the
      // role queries per op itself. Read and write must name the same PLANE; a read takes exactly
      // one (kbResolveScript), so `both` reads `remote` with the local mirror as fallback.
      ...(TUNING_KB_ENABLED && KB_DIMS && KB_DIMS.gfx ? {
        TUNED_KB_PLANE: E2E_KB_PLANE === 'both' ? 'remote' : E2E_KB_PLANE,
        TUNED_KB_STORE: E2E_KB_STORE_DIR,
        TUNED_KB_GFX: KB_DIMS.gfx,
        // Pages are keyed on arch and op, not dtype; this filters the read to one dtype's rows
        // (empty offers every dtype).
        TUNED_KB_PRECISION: KB_DIMS.precision || '',
        TUNED_KB_SCRIPT: KERNEL_WF_DIR + '/scripts/experience_store.py',
        TUNED_KB_ENV_PRELUDE: KB_ENV_PRELUDE,
      } : {}),
    }),
    { phase: 'TuningSkillset', label: 'tuning_specialist:tune', schema: TUNING_SCHEMA });

  // An accept must clear every bar the skillset sets: proven engagement (an artifact never proven
  // live proves nothing) and a completed both-leg A/B (a post-only number is not a measurement).
  const tuned = tuning && tuning.gate === 'accepted';
  const tuneOk = tuned && tuning.engagement_verified === true && tuning.ab_complete !== false &&
    tuning.correctness_gate !== 'fail' && tuning.post_tune_throughput_tok_s > 0 &&
    tuning.post_tune_throughput_tok_s > (tuning.pre_tune_throughput_tok_s || 0);

  // Every op the skillset named: read by the banking (gated on `tuneOk`) and by the attestation
  // (not gated). `tuning &&` because safeAgent can return null.
  const tunedOps = ((tuning && tuning.ops_tuned) || [])
    .filter((o) => String(o.op || o.short_name || '').trim());

  // Attest recalled tuned ops (the write below files only wins). Outside `tuneOk`: a per-op verdict
  // stands even when the phase banks nothing. Before the write, which carries the ledger across its
  // rewrite (experience_store.py `_carry_remote`). Only recalled ops that reached the GPU (artifact,
  // engagement or a measurement) count: `recalls` counts attempts on hardware (kb/attest.py).
  const tuningAttestable = KB_DIMS && KB_DIMS.gfx ? tunedOps.filter((o) =>
    String(o.session_id || '').trim() &&
    /recall|kb|knowledge/i.test(String(o.source || o.origin || ''))) : [];
  const tuningRecalls = tuningAttestable.filter((o) =>
    o.engaged === true || Number(o.isolated_speedup) > 0 || String(o.artifact || '').trim());
  const tuningUnattempted = tuningAttestable.length - tuningRecalls.length;
  if (tuningUnattempted) {
    log(`[kernel-kb] ${tuningUnattempted} recalled op(s) NOT attested: no artifact, engagement, or ` +
      `measurement, so they never reached the GPU — an offer nobody benched is not evidence.`);
  }
  if (tuningRecalls.length) {
    const storeScript = KERNEL_WF_DIR + '/scripts/experience_store.py';
    // One plane per verdict (`both` would count one attempt twice): the plane that answered, as the
    // op reports it (the role retries a remote miss locally), else the requested one.
    const askedRemote = E2E_KB_PLANE !== 'local';
    const planeOf = (o) => {
      const said = String(o.read_plane || o.recall_plane || o.plane || '').trim().toLowerCase();
      return said === 'local' || said === 'remote' ? said : (askedRemote ? 'remote' : 'local');
    };
    const cmds = tuningRecalls.map((o) => {
      const sp = Number(o.isolated_speedup) || 0;
      const outcome = o.engaged !== true ? 'not_reproduced' : sp > 1.0 ? 'validated' : 'failed';
      const plane = planeOf(o);
      return `python3 ${shq(storeScript)} attest --plane ${plane} ` +
        (plane === 'local' && E2E_KB_STORE_DIR ? `--store ${shq(E2E_KB_STORE_DIR)} ` : '') +
        `--session-id ${shq(String(o.session_id).trim())} ` +
        `--kernel-name ${shq(String(o.op || o.short_name).trim())} ` +
        `--language ${shq(String(o.backend || 'tuned').trim())} --gfx ${shq(KB_DIMS.gfx)} ` +
        (KB_DIMS.framework_version ? `--framework-version ${shq(KB_DIMS.framework_version)} ` : '') +
        `--outcome ${outcome} --measured-speedup ${sp} ` +
        // `claimed_gate`: the skillset's own claim (an unproven accept is downgraded below this);
        // `banked` is recorded beside the outcome, not as a gate on it.
        `--note ${shq(`e2e tuning recall: engaged=${o.engaged === true}; banked=${!!tuneOk}` +
          `; claimed_gate=${tuning.gate}` +
          `${o.note ? '; ' + String(o.note) : ''}`.slice(0, 300))} ` +
        `--measured-by ${shq('e2e_workflow:tuning:' + BACKEND)} --apply || true`;
    });
    // The prelude authenticates the service; needed if ANY op attests remotely.
    const anyRemote = tuningRecalls.some((o) => planeOf(o) === 'remote');
    try {
      await safeAgent(
        `You are the tuning knowledge-base attestor. Run EXACTLY these commands in order and ` +
        `return {"ran": <how many you ran>, "note": "<anything that failed>"}. Each records what ` +
        `this box saw when it installed a RECALLED tuned artifact. Do NOT edit them, do NOT add ` +
        `or drop any, and do NOT retry a failure — a repeat would double-count the attempt.\n` +
        '```bash\n' + (anyRemote ? KB_ENV_PRELUDE + '\n' : '') + cmds.join('\n') + '\n```',
        { phase: 'TuningSkillset', label: 'kernel-kb:attest-tuned',
          schema: obj({ ran: { type: 'number' }, note: { type: 'string' } }, []) },
        1);
      const planes = tuningRecalls.map(planeOf);
      log(`[kernel-kb] attested ${tuningRecalls.length} recalled tuned op(s) ` +
        `(${planes.filter((p) => p === 'remote').length} remote, ` +
        `${planes.filter((p) => p === 'local').length} local; claimed gate=${tuning.gate}` +
        `${tuneOk ? '' : ', not banked by this phase'}).`);
    } catch (e) {
      log(`[kernel-kb] tuned attest failed (NON-FATAL): ${String(e).slice(0, 200)}`);
    }
  }

  if (tuneOk) {
    const tuningBaselineConfig = { flags: curFlags, env: curEnv };
    if (tuning.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + tuning.apply_env;
    if (tuning.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + tuning.apply_flags;
    // Carry the routing/enabling overlay forward as an accepted head patch does; otherwise the code
    // half of a tuning win is dropped and its tuned data binds to nothing.
    if (tuning.apply_overlay) curOverlay = tuning.apply_overlay;
    curTput = tuning.post_tune_throughput_tok_s;
    log(`Tuning skillset ACCEPTED: ${tuning.pre_tune_throughput_tok_s} -> ${curTput} tok/s ` +
      `(+${(tuning.tuning_delta_pct || 0).toFixed(2)}% attributable to tuning; ` +
      `${(tuning.ops_tuned || []).length} op(s), engagement proven` +
      `${tuning.apply_overlay ? `, carrying routing overlay ${tuning.apply_overlay}` : ''}). Re-profiling.`);
    // An unverified deploy bundle does not withhold the accept (the in-run numbers stay valid); it
    // risks DELIVERY (final_launch.sh reproducing less). Finalize re-checks; warn early here too.
    if (tuning.deploy_verified !== true) {
      log(`WARNING: tuning accepted WITHOUT a verified deploy bundle (deploy_bundle=` +
        `${tuning.deploy_bundle || '<none>'}). The in-run measurements stand, but this win may not ` +
        `reproduce from EVAL_DIR/final/ — Finalize will re-check and report tuning_in_bundle.`);
    }
    history.ledger.push({
      direction: 'tuning_skillset', verdict: 'confirmed',
      e2e_delta_pct: tuning.tuning_delta_pct || 0,
      lesson: `tuning skillset (standalone, pre-head): ${tuning.summary || 'accepted'}`,
    });

    // Bank each tuned op as an accepted kernel, deterministically here rather than via Finalize's
    // accepted_kernels, so the KB record carries the lever itself and not just the launch script.
    for (const o of tunedOps) {
      bankAccepted(acceptedKernels, {
        short_name: String(o.op || o.short_name).trim(),
        backend: o.backend || '',
        // A tuned table is applied through env, a first-class winner_kind in the store.
        kind: 'env',
        isolated: Number(o.isolated_speedup) || 0,
        // One A/B covers all ops: the delta is per-op only when there is one op; otherwise 0 (the
        // phase figure is tuning_skillset.tuning_delta_pct) so consumers that sum do not double-count.
        e2e_delta_pct: tunedOps.length === 1 ? (tuning.tuning_delta_pct || 0) : 0,
        apply_env: tuning.apply_env || '',
        apply_flags: tuning.apply_flags || '',
        patch: '',
        engaged: o.engaged === true,
        from_tuning_skillset: true,
        tuning_artifact: o.artifact || '',
        // SEARCHED here or RECALLED: a recall re-banked as a discovery would look like an
        // independent confirmation to anyone counting records.
        tuning_source: String(o.source || o.origin || '').trim() || 'search',
      }, {});
    }
    if (tunedOps.length) {
      const recalled = tunedOps.filter((o) => /recall|kb|knowledge/i.test(String(o.source || o.origin || ''))).length;
      log(`Banked ${tunedOps.length} tuned op(s) as accepted kernels (kind=env) so the KB record ` +
        `carries the lever: ${tunedOps.map((o) => o.op || o.short_name).join(', ')}` +
        `${recalled ? ` (${recalled} recalled from the KB rather than searched)` : ''}.`);
    }

    // ---- Write each tuned op into the KERNEL lane's experience store, one entry per op ----
    // Unlike the end-of-run e2e record (whole deployment, gated on final throughput), this ranks on
    // isolated per-op speedup per (kernel, language, gfx), so the tuned table survives the run's
    // verdict. Filed only if tuneOk, and per op only if `isolated_speedup > 1.0` and `engaged`.
    // Best-effort: a failure loses a record, never a measurement.
    const kernelKbOps = KB_DIMS && KB_DIMS.gfx ? tunedOps.filter((o) =>
      Number(o.isolated_speedup) > 1.0 && o.engaged === true && String(o.artifact || '').trim()) : [];
    if (kernelKbOps.length) {
      const storeScript = KERNEL_WF_DIR + '/scripts/experience_store.py';
      // `both` mirrors to the shared service; without a store dir it degrades to the plain write
      // (as at the kernel lane's own write site).
      const remoteOn = E2E_KB_PLANE !== 'local' && !!E2E_KB_STORE_DIR;
      // `--precision` and `--serving-*` are recorded in `value.upstream`, not keyed; precision lets
      // the reader's `--precision` filter drop another dtype's table before it skews the ranking.
      const cmds = kernelKbOps.map((o) => {
        const op = String(o.op || o.short_name).trim();
        return `python3 ${shq(storeScript)} ${remoteOn
            ? `write-remote --plane both --store ${shq(E2E_KB_STORE_DIR)}`
            : 'write'} \\
  --root ${shq(KB_ARTIFACTS_DIR)} --kernel-name ${shq(op)} \\
  --language ${shq(String(o.backend || 'tuned').trim())} --gfx ${shq(KB_DIMS.gfx)} \\
  --kernel-class tuning --speedup ${Number(o.isolated_speedup)} \\
  --carrier tuned_artifact --artifact ${shq(String(o.artifact).trim())} \\
  --tuner ${shq(String(o.tuner || '').trim())} --apply-env ${shq(tuning.apply_env || '')} \\
  --cache-invalidation ${shq((tuning.cache_invalidation || []).join(' && '))} \\
  --metric-kind tuning_isolated --case-names ${shq(String(o.shapes || '').replace(/,/g, ';'))} \\
  --precision ${shq(KB_DIMS.precision || '')} --serving-framework ${shq(BACKEND)} \\
  --serving-framework-version ${shq(KB_DIMS.framework_version || '')} \\
  --direction ${shq('tuning-' + (String(o.backend || 'op').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-')))} \\
  --eval-dir ${shq(EVAL_DIR)}${tuning.report_path ? ` --report ${shq(tuning.report_path)}` : ''}`;
      });
      try {
        const kw = await safeAgent(
          `You are the tuning experience writer. Run EACH of the commands below EXACTLY as written, ` +
          `in order. Each applies its own gates and prints one line of JSON; collect those lines in ` +
          `order into \`results\`. Do NOT edit a command, do NOT add or drop flags, and do NOT retry ` +
          `one that fails` + (remoteOn
            ? ` — these may reach the shared KB Store service, and every write it accepts is ` +
              `PERMANENT (it exposes no delete), so a retry creates a second permanent record ` +
              `instead of fixing the first`
            : '') + `. For a command that errors, put ` +
          `{"written": false, "reason": "io_error"} in its place so the list stays aligned with the ` +
          `command order.\n\n` +
          `THEN, last, write \`{"results": [...]}\` — the same list — to ` +
          `\`${EVAL_DIR}/tuning/kb_write_tuned.json\`. That file is the receipt saying this filing ` +
          `already happened: run_e2e.py refiles these ops from disk when the workflow is killed ` +
          `before reaching this step, and it skips when the receipt is present. Without it, a run ` +
          `that dies AFTER this step gets every op written a second time, and the store counts the ` +
          `copy as an independent reproduction — which can promote a candidate on one measurement ` +
          `seen twice. Write the receipt even if every command failed.\n` +
          '```bash\n' + (remoteOn ? KB_ENV_PRELUDE + '\n' : '') + cmds.join('\n\n') + '\n```',
          { phase: 'TuningSkillset', label: 'kernel-kb:write-tuned',
            schema: obj({ results: arrObj }, ['results']) },
          1);
        const rows = (kw && kw.results) || [];
        const wrote = rows.filter((r) => r && r.written).length;
        const dup = rows.filter((r) => r && !r.written && /duplicate/.test(String(r.reason || ''))).length;
        log(`[kernel-kb] tuned ops filed as isolated wins: ${wrote}/${kernelKbOps.length} written` +
          `${dup ? `, ${dup} already known (counted as reproductions)` : ''}` +
          `${rows.filter((r) => r && !r.written && !/duplicate/.test(String(r.reason || '')))
            .map((r) => ` [${r.reason}]`).join('')}. These survive the run's own e2e verdict.`);
      } catch (e) {
        log(`[kernel-kb] tuned-op write failed (NON-FATAL — the tuning result itself is unaffected): ` +
          `${String(e).slice(0, 200)}`);
      }
    } else if (tunedOps.length && (!KB_DIMS || !KB_DIMS.gfx)) {
      log(`[kernel-kb] tuned ops NOT filed: no gfx established, and an arch-less entry is ` +
        `unattributable (a tuned table is valid for exactly one arch).`);
    }
    await requireE2EValidationCheckpoint('tuning/e2e_validation.json', {
      phase: 'TuningSkillset', validation_level: 'tuning_skillset', gate: 'accepted',
      validation_status: 'accepted_tuning',
      baseline_throughput_tok_s: tuning.pre_tune_throughput_tok_s,
      final_throughput_tok_s: tuning.post_tune_throughput_tok_s,
      throughput_speedup: tuning.tuning_speedup ||
        (tuning.post_tune_throughput_tok_s / tuning.pre_tune_throughput_tok_s),
      baseline_config: tuningBaselineConfig,
      accepted_config: { flags: curFlags, env: curEnv, effective_config_digest: EFFECTIVE_CONFIG_DIGEST },
      accepted_kernels: tunedOps.map((o) => ({
        kernel_id: o.kernel_id || o.op || o.short_name || '',
        kernel_slot: o.kernel_slot || o.target_callable || o.target_file || o.op || o.short_name || '',
        short_name: o.op || o.short_name || '', backend: o.backend || 'geak',
        from_tuning_skillset: true, isolated: o.isolated_speedup || 0,
      })),
      accepted_heads: [], final_patch: [], final_overlay: { path: curOverlay },
      final_launch_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
        workload: WORKLOAD, source_artifact: `${EVAL_DIR}/tuning/ab/ab_summary.json`,
        acceptance: { gain_exceeds_noise: true, correctness_passed: tuning.correctness_gate !== 'fail',
          noise_floor_pct: tuning.noise_floor_pct || NOISE_BAND } },
      stack: { kernel_slots: tunedOps.map((o) => ({
        kernel_slot: o.kernel_slot || o.target_callable || o.target_file || o.op || o.short_name || '',
        selected: true,
      })), deployment_layers: [tuning.deploy_bundle].filter(Boolean) },
      replay: { asset_paths: [tuning.deploy_bundle, ...(tuning.artifacts || [])].filter(Boolean),
        cache_invalidation: tuning.cache_invalidation || [], requires_server_restart: true },
      integrity: { checkpoint_assets: [] },
      tuning_skillset: tuning,
    });

    // Tuning changed which kernels dominate: re-profile + re-strategize, as after ConfigSweep.
    if (!searchClosed('post-tuning re-profile and re-strategize')) {
      profile = await safeAgent(
        roleAgent('profiler', 'reprofile', 'Re-profile after the tuning skillset deployed its artifacts.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 'tuning',
          // curOverlay, not '': without a banked routing overlay the tuned artifact binds to nothing.
          OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Profile', label: 'profiler:post-tuning', schema: PROFILE_SCHEMA });
      const retune = await safeAgent(
        roleAgent('system_architect', 'strategize', 'Re-route after tuning changed the landscape.', {
          EVAL_DIR, PROFILE_TOPN: profile ? profile.profile_topN_json : '', BASELINE_THROUGHPUT: curTput,
          WORKLOAD, BUDGET, HEAD_THRESHOLD_PCT, CONFIG_TUNE_ENABLED: false, SKILL_DIR: WORKFLOW_DIR,
          ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Strategize', label: 'architect:post-tuning-strategize', schema: STRATEGY_SCHEMA });
      if (retune && retune.kernel_candidates) kernelQueue = retune.kernel_candidates.slice();
      if (retune && retune.head_candidates)
        headQueue = applyOpIdentityGuard(retune.head_candidates.slice(), 'post-tuning re-strategize');
      if (retune) await ensureFlydslGate();
    }
  } else if (tuned) {
    // Claimed accepted but failed a hard bar: not banked (an unproven artifact poisons later A/Bs).
    log(`Tuning skillset claimed ACCEPTED but did not clear the bar ` +
      `(engagement_verified=${tuning.engagement_verified}, ab_complete=${tuning.ab_complete}, ` +
      `correctness=${tuning.correctness_gate}, pre=${tuning.pre_tune_throughput_tok_s}, ` +
      `post=${tuning.post_tune_throughput_tok_s}) — NOT banked; continuing on the pre-tuning config.`);
    history.ledger.push({ direction: 'tuning_skillset', verdict: 'flagged',
      lesson: 'tuning claimed a win it could not prove (engagement/A-B/correctness) — not banked' });
    tuning = { ...tuning, gate: 'no_win', reason: (tuning.reason || '') + ' [orchestrator: accept withheld — unproven]' };
  } else {
    log(`Tuning skillset produced no banked win (gate=${tuning ? tuning.gate : 'null/degraded'}` +
      `${tuning && tuning.reason ? `: ${tuning.reason}` : ''}).`);
    history.ledger.push({ direction: 'tuning_skillset', verdict: tuning && tuning.gate === 'skipped' ? 'skipped' : 'dead_end',
      lesson: (tuning && (tuning.reason || tuning.summary)) || 'tuning phase produced no result' });
  }
} else if (want('tune') && !TUNING_SKILLSET_ENABLED) {
  log('Tuning skillset DISABLED (tuning_skillset=false) — skipping the phase entirely.');
}
// Report inputs for the tuning phase; {} when off/absent, so the Report prompt is unchanged.
const TUNING_REPORT_INPUTS = (TUNING_SKILLSET_ENABLED && tuning) ? { TUNING_RESULT: tuning } : {};
// Finalize inputs: a tuned DATA artifact cannot ride the PYTHONPATH overlay, so Finalize folds this
// bundle into EVAL_DIR/final/ (diff, files, deploy.sh before server start). {} unless tuning banked.
const TUNING_FINALIZE_INPUTS = (TUNING_SKILLSET_ENABLED && tuning && tuning.gate === 'accepted')
  ? {
    TUNING_DEPLOY_BUNDLE: tuning.deploy_bundle || `${EVAL_DIR}/tuning/deploy`,
    TUNING_APPLY_ENV: tuning.apply_env || '',
    TUNING_CACHE_INVALIDATION: tuning.cache_invalidation || [],
    TUNING_ARTIFACTS: tuning.artifacts || [],
    TUNING_LIVE_TREE_FILES: tuning.live_tree_files || [],
    TUNING_OVERLAY: tuning.apply_overlay || '',
  }
  : {};

// The live-tree carve-out for EVERY integrate leg (roles/e2e_integrator.md, "never mutate
// site-packages"): an accepted tuning deploy legitimately writes into the installed tree, so these
// paths are exempt from the clean-tree check (otherwise a leg "restores" them and drops the tuning).
// A function, so it reflects a downgrade to no_win and a resumed run's carried state; {} otherwise.
function tuningIntegrateInputs() {
  if (!(TUNING_SKILLSET_ENABLED && tuning && tuning.gate === 'accepted')) return {};
  return {
    TUNING_LIVE_TREE_FILES: tuning.live_tree_files || [],
    TUNING_DEPLOY_BUNDLE: tuning.deploy_bundle || `${EVAL_DIR}/tuning/deploy`,
  };
}

