// ===========================================================================
// PHASE: TuningSkillset — the VENDORED tuning skillset, run WHOLE and STANDALONE, BEFORE HeadKernel.
//
// The skillset lives at TUNING_SKILLSET_DIR as one intact, hash-pinned tree (it is validated standalone,
// so GEAK must run the copy that was validated — see e2e_workflow/scripts/tuning_skillset_sync.py). It is
// NOT decomposed into knowledge/ files and NOT injected as an advisory fragment into other roles: ONE role
// (tuning_specialist) runs its complete six-step loop as ONE phase.
//
// Position is deliberate — after ConfigSweep, before HeadKernel:
//   * BEFORE HeadKernel, because tuning is the cheap lever that reshapes which kernels dominate. Running
//     it first means the head track spends its (expensive) budget on the ops that are still hot AFTER
//     tuning, and every head A/B measures against a reference leg that already contains the tuning.
//   * STANDALONE, because a tuning loop folded into the head bake-off never runs to completion (it
//     collapses into "one more candidate config") and its contribution becomes unattributable. With its
//     own in-session pre/post isolated-server A/B, the final report can state tuning's share of the gain.
//
// An accept is folded into curFlags/curEnv (the deploy's required env IS the engagement mechanism), then
// the profile is re-taken exactly as it is after a config win, because tuning changes the landscape too.
// ===========================================================================
if (want('tune') && TUNING_SKILLSET_ENABLED && !searchClosed('TuningSkillset')) {
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
      // The always-fires channel, same as the Architect's and the config_tuner's. The prompt BLOCK
      // above can be empty (no candidates, or the read never ran); these Inputs entries are how the
      // role learns the store exists at all. Gated by TUNING_KB_ENABLED for the same reason
      // warmStartBlock() is: blind eval has to stay blind through BOTH channels or through neither.
      ...(TUNING_KB_ENABLED ? KB_REF_INPUTS : {}),
      ...(TUNING_KB_ENABLED && KB_CACHE_DIR ? { KB_CACHE_DIR } : {}),
      // The other half of the write below: the kernel store is where THIS phase files its tuned
      // tables, keyed per (op, backend, gfx), so it is also where a later run finds them — even when
      // the run that produced them ended below its own e2e baseline and wrote no deployment record
      // at all. Handing over the root and the arch rather than a candidate list is deliberate: the
      // role knows which ops it is about to work on and can ask per op, which no read done up here
      // before the profile could do. Same TUNING_KB_ENABLED switch as everything else.
      //
      // PLANE, not root. The write below is `write-remote --plane both` whenever a service is
      // configured, so the record lands behind a key on the shared store; a directory read against
      // KB_ARTIFACTS_DIR would look at this run's own checkout, which HL creates empty per run, and
      // would report `kernel_page_not_found` — indistinguishable from a page that has nothing. Read
      // and write must name the same plane. A read takes exactly ONE (see kbResolveScript), so
      // `both` resolves to `remote` and the local mirror is the named fallback.
      ...(TUNING_KB_ENABLED && KB_DIMS && KB_DIMS.gfx ? {
        TUNED_KB_PLANE: E2E_KB_PLANE === 'both' ? 'remote' : E2E_KB_PLANE,
        TUNED_KB_STORE: E2E_KB_STORE_DIR,
        TUNED_KB_GFX: KB_DIMS.gfx,
        // The page is keyed on arch and op, NOT on dtype — one `fused_moe` page holds the bf16 and
        // the fp8_w8a8 tables side by side, ranked against each other on speedup. Passing this makes
        // the read drop the other dtype's rows instead of ranking them; leaving it empty is the old
        // behaviour (everything offered), which is why an unstated precision is never an error.
        TUNED_KB_PRECISION: KB_DIMS.precision || '',
        TUNED_KB_SCRIPT: KERNEL_WF_DIR + '/scripts/experience_store.py',
        TUNED_KB_ENV_PRELUDE: KB_ENV_PRELUDE,
      } : {}),
    }),
    { phase: 'TuningSkillset', label: 'tuning_specialist:tune', schema: TUNING_SCHEMA });

  // An accept must clear EVERY bar the skillset itself sets. Engagement is the load-bearing one: its
  // stated thesis is that a speedup whose artifact was never proven live has proven nothing, and that
  // failure mode is silent (the artifact lands where nothing reads it and the timing still moves). A
  // completed BOTH-leg A/B is required for the same reason it is on the head track — a post-only number
  // is not a measurement.
  const tuned = tuning && tuning.gate === 'accepted';
  const tuneOk = tuned && tuning.engagement_verified === true && tuning.ab_complete !== false &&
    tuning.correctness_gate !== 'fail' && tuning.post_tune_throughput_tok_s > 0 &&
    tuning.post_tune_throughput_tok_s > (tuning.pre_tune_throughput_tok_s || 0);

  // Every op the skillset named, read twice below: by the accepted-kernel banking (gated on
  // `tuneOk`) and by the attestation under it (deliberately not). `tuning &&`, not just
  // `.ops_tuned ||`: `safeAgent` returns null by DESIGN, and this read sits outside the `tuneOk`
  // short-circuit that used to screen the null off.
  const tunedOps = ((tuning && tuning.ops_tuned) || [])
    .filter((o) => String(o.op || o.short_name || '').trim());

  // The other half of the recall loop: the write below only files wins, so without this the ledger
  // grows in one direction. OUTSIDE `tuneOk` on purpose — a verdict on a stored record is a per-op
  // fact, while the PHASE gate is an aggregate with different inputs; one op can reproduce
  // perfectly inside a run that banks nothing. Before the write, since `write-remote` carries the
  // ledger across its rewrite (experience_store.py `_carry_remote`).
  //
  // Searched ops are excluded — no prior record to be evidence about. So are recalled ops that
  // never reached the GPU: `recalls` counts ATTEMPTS ON HARDWARE (kb/attest.py), proven by an
  // installed `artifact`, engagement, or a measurement.
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
    // One plane per verdict — `both` would count one attempt twice on two ledgers a curation pass
    // then compares. The plane that ANSWERED is not the one ASKED FOR (the tuning role retries a
    // remote miss locally), so the op names its own plane; the requested one is the fallback.
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
        // `claimed_gate`, not `gate`: the orchestrator's downgrade of an unproven accept happens
        // below this, so what is readable here is the skillset's claim. `banked` sits beside the
        // outcome rather than gating it — an auditor wants both facts, not one instead of the other.
        `--note ${shq(`e2e tuning recall: engaged=${o.engaged === true}; banked=${!!tuneOk}` +
          `; claimed_gate=${tuning.gate}` +
          `${o.note ? '; ' + String(o.note) : ''}`.slice(0, 300))} ` +
        `--measured-by ${shq('e2e_workflow:tuning:' + BACKEND)} --apply || true`;
    });
    // The prelude authenticates the service, needed when ANY op attests remotely — the set is
    // mixed by construction.
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
    // Carry the routing/enabling overlay forward exactly as an accepted head patch does. Without this
    // the code half of a tuning win is silently dropped at the phase boundary and the data half is left
    // bound to nothing — the failure the skillset spends its longest section on.
    if (tuning.apply_overlay) curOverlay = tuning.apply_overlay;
    curTput = tuning.post_tune_throughput_tok_s;
    log(`Tuning skillset ACCEPTED: ${tuning.pre_tune_throughput_tok_s} -> ${curTput} tok/s ` +
      `(+${(tuning.tuning_delta_pct || 0).toFixed(2)}% attributable to tuning; ` +
      `${(tuning.ops_tuned || []).length} op(s), engagement proven` +
      `${tuning.apply_overlay ? `, carrying routing overlay ${tuning.apply_overlay}` : ''}). Re-profiling.`);
    // A missing/unverified deploy bundle does NOT withhold the accept: the gain was measured and proven
    // live, and the artifacts are already installed in this container, so every in-run number stays
    // valid. What breaks is DELIVERY — final_launch.sh would reproduce less than the headline. Finalize
    // re-checks and reports it, but warn here too, because this is the point at which it is still cheap
    // to fix and the easiest failure in this phase to not notice.
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

    // Bank each tuned op as an accepted kernel, so the KB record carries the LEVER and not just the
    // launch script that happens to reference it. This is deliberately done HERE, deterministically,
    // rather than left to Finalize's own accepted_kernels: the DeepSeek-V4-Pro 20260823 run banked a
    // 3.29x tuned a8w8 blockscale table, returned accepted_kernels:[], and wrote a KB record whose
    // files were [final.patch, launch.sh, report.md] — the lever itself was never recorded, and the
    // next run at that canonical id recalls a configuration it cannot reproduce. Gated on tuneOk, so
    // an unproven tuning claim banks nothing, exactly as it folds nothing into curEnv/curFlags.
    // (`tunedOps` is computed above the branch: the attestation needs it on every path, banking only
    // on this one.)
    for (const o of tunedOps) {
      bankAccepted(acceptedKernels, {
        short_name: String(o.op || o.short_name).trim(),
        backend: o.backend || '',
        // A tuned table is applied through env (the deploy's env var IS the engagement mechanism),
        // which is already a first-class winner_kind in the store alongside patch/authored.
        kind: 'env',
        isolated: Number(o.isolated_speedup) || 0,
        // The phase measured ONE pre/post A/B covering all of its ops, so its delta is attributable
        // to a single op and only to a single op. With several, the per-op number is left at 0 and
        // the phase-level figure stands alone in tuning_skillset.tuning_delta_pct — stamping the
        // whole delta onto each op would double-count it in any consumer that sums the list.
        e2e_delta_pct: tunedOps.length === 1 ? (tuning.tuning_delta_pct || 0) : 0,
        apply_env: tuning.apply_env || '',
        apply_flags: tuning.apply_flags || '',
        patch: '',
        engaged: o.engaged === true,
        from_tuning_skillset: true,
        tuning_artifact: o.artifact || '',
        // Whether this op was SEARCHED here or RECALLED from a prior record. Now that the tuning
        // track reads the store it also writes, the two are indistinguishable downstream without
        // this field — and a recall re-banked as a discovery would make one original search look
        // like N independent confirmations to anyone counting records.
        tuning_source: String(o.source || o.origin || '').trim() || 'search',
      }, {});
    }
    if (tunedOps.length) {
      const recalled = tunedOps.filter((o) => /recall|kb|knowledge/i.test(String(o.source || o.origin || ''))).length;
      log(`Banked ${tunedOps.length} tuned op(s) as accepted kernels (kind=env) so the KB record ` +
        `carries the lever: ${tunedOps.map((o) => o.op || o.short_name).join(', ')}` +
        `${recalled ? ` (${recalled} recalled from the KB rather than searched)` : ''}.`);
    }

    // ---------------------------------------------------------------------------------------
    // Write each tuned op into the KERNEL lane's experience store, one entry per op.
    //
    // Not a duplicate of the e2e KB write at the end of the run: that record is keyed on the whole
    // deployment and gated on the run's FINAL throughput_speedup, so a tuning win a later phase
    // erases — or a run that ends below its own baseline for unrelated reasons — takes the tuned
    // table down with it, and the next run at the same identity searches it again from scratch. The
    // kernel store has no such coupling: it ranks on ISOLATED per-op speedup and is addressed per
    // (kernel, language, gfx), which is the granularity at which a tuned table is reusable.
    //
    // Gated on tuneOk, so nothing unproven is filed. Per-op: `isolated_speedup > 1.0` and `engaged`,
    // because an op the phase could not prove live is not knowledge about that op whatever the
    // phase-level A/B measured. Best-effort — a failure here loses a record, never a measurement.
    const kernelKbOps = KB_DIMS && KB_DIMS.gfx ? tunedOps.filter((o) =>
      Number(o.isolated_speedup) > 1.0 && o.engaged === true && String(o.artifact || '').trim()) : [];
    if (kernelKbOps.length) {
      const storeScript = KERNEL_WF_DIR + '/scripts/experience_store.py';
      // `both` mirrors to the shared service; without a store dir there is nothing to mirror INTO,
      // so it degrades to the plain directory write rather than erroring. Same selection the kernel
      // lane makes at its own write site.
      const remoteOn = E2E_KB_PLANE !== 'local' && !!E2E_KB_STORE_DIR;
      // `--precision` and the two `--serving-*` flags are RECORDED, not keyed — they land in
      // `value.upstream` and never move the entry's address. A tuned table's destination filename
      // encodes its dtype (`...dtype=fp8_w8a8.json`), so a table measured under one precision offered
      // to a run on another installs under a name that runtime never looks up: wasted, not wrong. What
      // it does corrupt is RANKING, by putting an unusable table above a usable one on the same page.
      // Stating precision here is what lets the reader's `--precision` filter drop it first.
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

    // Tuning changed which kernels dominate — re-profile + re-strategize so the head track works the
    // POST-tuning landscape, not the pre-tuning one. Same contract as the post-ConfigSweep re-profile.
    if (!searchClosed('post-tuning re-profile and re-strategize')) {
      profile = await safeAgent(
        roleAgent('profiler', 'reprofile', 'Re-profile after the tuning skillset deployed its artifacts.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 'tuning',
          // curOverlay, not '': tuning may have banked a routing overlay, and profiling without it would
          // measure a stack where the tuned artifact binds to nothing, then hand the head track a Top-N
          // for the wrong landscape.
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
    // Claimed accepted but failed a hard bar. Do NOT bank it — an unproven artifact silently poisons
    // every downstream A/B, which is the exact failure the skillset exists to prevent.
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
// Report inputs for the tuning phase. Empty object when the phase is off/absent, so the Report prompt is
// byte-identical to a build without this feature.
const TUNING_REPORT_INPUTS = (TUNING_SKILLSET_ENABLED && tuning) ? { TUNING_RESULT: tuning } : {};
// Finalize inputs. A tuned DATA artifact cannot ride the PYTHONPATH overlay, so the ONLY way it reaches
// production is for Finalize to fold this bundle into EVAL_DIR/final/ (concatenate its diff into
// final_patch.diff, copy its files, and invoke its deploy.sh from final_launch.sh before the server
// starts). Passed only when the tuning phase actually banked a win, so an unaccepted/absent tuning
// leaves the Finalize prompt byte-identical.
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

// The live-tree carve-out, handed to EVERY integrate leg (see roles/e2e_integrator.md, "never mutate
// site-packages"). That rule asserts `git status --porcelain` is EMPTY in the installed tree before and
// after every gate leg — but an accepted tuning deploy legitimately writes into that tree, because a
// config table only takes effect from inside the package that reads it. Without this list the head track
// sees a dirty tree and either fails the assertion or "restores" it, silently deleting the banked tuning
// out from under the reference leg of every later A/B. The rule's rationale does not apply to these
// paths: they are recorded, reproducible from the deploy bundle, and INTENDED to be in both legs, exactly
// like accepted env. A function, not a const, so it reflects a downgrade-to-no_win and so a resumed run
// picks the carve-out up from carried state. Returns {} otherwise => prompt byte-identical without tuning.
function tuningIntegrateInputs() {
  if (!(TUNING_SKILLSET_ENABLED && tuning && tuning.gate === 'accepted')) return {};
  return {
    TUNING_LIVE_TREE_FILES: tuning.live_tree_files || [],
    TUNING_DEPLOY_BUNDLE: tuning.deploy_bundle || `${EVAL_DIR}/tuning/deploy`,
  };
}

