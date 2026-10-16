// ===========================================================================
// PHASE: HeadKernel — the highest-pct_gpu_time ops (GEMM / attention), optimized
// regardless of edit flag, via the bake-off ladder. This is the lever the old
// design missed for GEMM (~78% of GPU time). Runs BEFORE the editable-kernel loop.
// The op-identity guard (see Strategize) has already forced fused/monolithic heads to op_kind=moe with
// dense-GEMM synth OFF + the live seam preserved, so each head is optimized AS the op the live kernel
// dispatches (backend-swap / tune / author-fused) — never decomposed into an un-integrable dense GEMM.
// ===========================================================================
if (want('head') && headQueue.length && HEAD_BUDGET > 0) {
  phase('HeadKernel');
  log(`Head-kernel track: ${headQueue.length} candidate op(s), head_budget=${HEAD_BUDGET}, threshold=${HEAD_THRESHOLD_PCT}%.`);
  // Head ops are taken in the Architect's Amdahl-ranked order — no forced GEMM-first reordering.
  const heads = headQueue.slice(0, HEAD_BUDGET).map((c, i) => ({
    ...c, idx: i, gpu_id: GPU_LIST[i % GPU_LIST.length],
    short_name: c.short_name || `${c.op_kind || 'op'}${i}`,
  }));
  if (DEEP) {
    // ============ DEEP-MODE v2: GLOBAL cross-kernel × cross-backend co-optimization ====================
    // One global lane pool over ALL (head op × backend) lanes (kernels + backends optimize concurrently);
    // GPU-elastic serial e2e gate overlapping co-opt; full-backend roster with ceiling-aware patience +
    // revive; per-op SHARED_KB + run-global GLOBAL_KB; budget-driven EV scheduling toward a +50% e2e goal.
    const ROOFLINE_SCHEMA = { type: 'object', additionalProperties: true, properties: { roofline_note: { type: 'string' }, target_geomean: { type: 'number' } }, required: ['roofline_note'] };
    const OKV = { type: 'object', additionalProperties: true, properties: { ok: { type: 'boolean' }, summary: { type: 'string' }, feedback_path: { type: 'string' }, addendum_path: { type: 'string' } }, required: ['ok'] };
    const HARVEST_SCHEMA = { type: 'object', additionalProperties: true, properties: { lanes: { type: 'array', items: { type: 'object', additionalProperties: true, properties: { uid: { type: 'string' }, has_state: { type: 'boolean' }, cumulative: { type: 'number' }, best_ms: { type: 'number' }, vs_live: { type: 'number' }, eval_dir: { type: 'string' }, patch: { type: 'string' } }, required: ['uid'] } } }, required: ['lanes'] };

    // ---- GPU partition (N-adaptive, elastic; serial gate) -------------------------------------------
    // Co-opt runs on cards NOT needed by the serving slot ('main', never paused). The serving cards form
    // a second co-opt pool ('serve') used only in waves with NO due gate; a gate wave runs the e2e A/B on
    // the serving slot CONCURRENTLY with co-opt on the dedicated cards (overlap, no idle). With exactly TP
    // cards (no dedicated), 'main' is empty and a gate wave runs the gate alone (graceful time-slice).
    const servingCards = SERVING_GPU.split(',').map(s => s.trim()).filter(Boolean);
    const servingSet = new Set(servingCards);
    const dedicatedCoopt = GPU_LIST.filter(g => !servingSet.has(g));
    const haveSpare = dedicatedCoopt.length > 0;
    const cooptMain = haveSpare ? makeSem(dedicatedCoopt) : null;          // dedicated cards (never paused)
    const cooptServe = makeSem(haveSpare ? servingCards : GPU_LIST);       // serving cards (idle during a gate wave)
    const mainSlots = haveSpare ? dedicatedCoopt.length : 0;
    const serveSlots = haveSpare ? servingCards.length : GPU_LIST.length;
    log(`[deep] partition: serving {${servingCards.join(',')}} TP=${SERVING_TP}; dedicated co-opt {${dedicatedCoopt.join(',') || '(none)'}}; mode=${haveSpare ? 'OVERLAP gate||co-opt' : 'TIME-SLICE (N==TP)'}; budget ${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h; e2e target ×${DEEP_E2E_TARGET} (${Math.round(BASELINE_TPUT * DEEP_E2E_TARGET)} tok/s).`);

    // ---- ceiling prior: GENERAL (by lane ROLE, not model/shape) -------------------------------------
    const ceilingPrior = (lang, mode) => (mode === 'author' ? 2.2 : (lang === 'triton' && mode === 'optimize' ? 1.6 : 1.8));

    // ---- per-head prep: extract + bake-off + roofline + lane roster (cheap agents on GPU_LIST[0]) ----
    const GLOBAL_KB = `${EVAL_DIR}/deep_head/GLOBAL_KB.md`;
    const prepHead = async (h) => {
      const ext = await extractWithBaseline(
        'kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, KERNEL: h, GEMM_SYNTH: gemmSynthFor(h),
          ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
          CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          REQUIRE_DECODE_BUCKET: true, DECODE_M_BUCKETS: [1, CONC],
          PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
        },
        { phase: 'HeadKernel', label: `extract_op ${h.short_name}`, schema: EXTRACT_OP_SCHEMA });
      const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
      if (!ext || ext.smoke !== 'pass' || !ext.task_dir) {
        const why = extractFailWhy(ext);
        log(`  [deep] ${h.short_name}: op extraction failed (${why})${isDominant ? ' [DOMINANT — flagged]' : ''}; skipping.`);
        if (isDominant) flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'extract', gate: 'extract_failed', reason: why });
        history.ledger.push({ direction: h.short_name, verdict: isDominant ? 'flagged' : 'dead_end', lesson: `op extraction failed (${why})` });
        return null;
      }
      const bake = await safeAgent(
        roleAgent('op_benchmarker', 'bakeoff', 'DISCOVER existing impls, tune cheap levers, DECIDE author_plan.', {
          EVAL_DIR, OP_TASK_DIR: ext.task_dir, OP_KIND: ext.op_kind, PCT_GPU_TIME: h.pct_gpu_time,
          CANDIDATE_BACKENDS: ext.candidate_backends || h.candidate_backends || [],
          GPU_ID: GPU_LIST[0], ENABLE_FP8, KERNEL_WF_DIR, KERNEL_BUDGET: DEEP_WAVE_BUDGET, SKILL_DIR: WORKFLOW_DIR,
        }),
        { phase: 'HeadKernel', label: `bakeoff ${h.short_name}`, schema: OPBENCH_SCHEMA });
      // Lane roster: ALWAYS tune the live editable kernel + author EVERY backend the bake-off proposes
      // (no backend dropped a priori — unbiased), then fill remaining diversity with distinct authoring
      // directions. Steers are DIRECTIONS (generic), never hard-coded magic numbers.
      let lanesSpec = [];
      if (DEEP_BACKENDS_OVERRIDE) {
        lanesSpec = DEEP_BACKENDS_OVERRIDE.split(',').map(s => s.trim()).filter(Boolean).map(s => {
          const [lang, mode] = s.split(':'); const L = (lang || '').trim();
          return { key: L, lang: L, mode: (mode || 'author').trim(), steer: '' };
        }).filter(b => b.lang);
      } else {
        const planLangs = (bake && Array.isArray(bake.author_plan) ? bake.author_plan : [])
          .map(ap => ({ lang: (ap.language || '').trim().toLowerCase(), mode: ap.route === 'rewrite' ? 'optimize' : 'author' }))
          .filter(x => x.lang);
        const liveLang = (ext.live_backend || 'triton').toLowerCase();
        const otherLangs = [...new Set(planLangs.map(x => x.lang).filter(l => l && l !== liveLang))];
        lanesSpec = [{ key: `${liveLang}-opt`, lang: liveLang, mode: 'optimize',
          steer: ` DIRECTION=tile-tune: per-shape AUTOTUNE the live ${liveLang} kernel (block sizes, warps, stages, scheduling, grid swizzle; split-K for small-M decode). Stay graph-capture-safe.` }];
        for (const l of otherLangs) lanesSpec.push({ key: l, lang: l, mode: (planLangs.find(x => x.lang === l) || {}).mode || 'author',
          steer: ` AUTHOR a ${l} implementation that beats the LIVE kernel (not just your own first port); read SHARED_KB + GLOBAL_KB and borrow the winning decomposition other lanes/kernels found.` });
        const extra = [
          { key: `${liveLang}-fused`, lang: liveLang, mode: 'author', steer: ` DIRECTION=fused-author: author a fresh single-pass FUSED kernel (fold pre/post ops + scaling into the main ${MATRIX_CORE_NAME} core; epilogue-fuse activation). Beat the LIVE kernel.` },
          { key: `${liveLang}-splitk`, lang: liveLang, mode: 'author', steer: ' DIRECTION=split-K: author a split-K + accumulate variant for the large-M prefill shapes, with a per-shape launch selector that uses the non-split path for small-M decode.' },
          { key: `${liveLang}-deep`, lang: liveLang, mode: 'optimize', steer: ' DIRECTION=deep-explore: combine persistent kernel + epilogue fusion + grid swizzle + aggressive tiling in one coherent rewrite; push toward the roofline SOTA bar.' },
        ];
        for (const t of extra) lanesSpec.push(t);   // global pool — no per-card truncation here
      }
      const liveBaselineMs = (bake && Number.isFinite(bake.best_known_ms) && bake.best_known_ms > 0) ? bake.best_known_ms : 0;
      const deepDir = `${EVAL_DIR}/deep_head/${h.short_name}`;
      const sharedKb = `${deepDir}/SHARED_KB.md`;
      const opSpec = { op_kind: ext.op_kind, shapes: ext.shapes || {}, dtype: ext.dtype || 'bf16', regime: h.regime || 'both', cuda_graph_safe: true, ...(ext.workload_path ? { workload_path: ext.workload_path } : {}) };
      const anchor = await safeAgent(
        `You are the ROOFLINE ANCHOR + shared-KB bootstrapper for DEEP cross-backend optimization of head op ${h.short_name} (${ext.op_kind}). ` +
        `Inputs: OP_TASK_DIR=${ext.task_dir}; shapes=${JSON.stringify(ext.shapes || {})}; dtype=${ext.dtype || '?'}; read ${EVAL_DIR}/env_report.json for the on-box device peak (FLOP/s + HBM bandwidth). ` +
        `DO: (a) mkdir -p ${deepDir}; (b) compute the ROOFLINE ceiling per case (compute- vs memory-bound, target ms/case + an overall SOTA geomean ~80-90% of roofline); ` +
        `(c) bootstrap ${sharedKb} (markdown) with sections: Roofline target; Current best per backend (table backend|best geomean|technique|wave — empty now); Techniques that WORK (technique -> measured effect -> source); Dead-ends (scoped, evidence); Cross-backend assignments (borrow); Open hypotheses. ` +
        (E2E_LEARNED_KB_ENABLED
          ? `Cite relevant ${KERNEL_KNOWLEDGE_DIR} cards (read INDEX in knowledge/learned/ first) for ${ext.op_kind}. `
          : `E2E_LEARNED_KB=off: do not read or cite knowledge/learned/; derive the bootstrap from this run's measurements only. `) +
        `Return {roofline_note, target_geomean}.`,
        { phase: 'HeadKernel', label: `roofline ${h.short_name}`, schema: ROOFLINE_SCHEMA });
      const rooflineTarget = anchor && Number.isFinite(anchor.target_geomean) ? anchor.target_geomean : 0;
      const lanes = lanesSpec.map((b) => ({
        uid: `${h.short_name}::${b.key || b.lang}`, key: b.key || b.lang, lang: b.lang, mode: b.mode, steer: b.steer || '',
        state_dir: `${deepDir}/state/${b.key || b.lang}`, best: 1.0, noImprove: 0, active: true, ran: 0, lastEval: '', patch: '',
        ceiling: Math.max(ceilingPrior(b.lang, b.mode), rooflineTarget || 0),
        head: h, ext, deepDir, sharedKb, opSpec, liveBaselineMs, rooflineTarget,
      }));
      log(`[deep] ${h.short_name} (${(h.pct_gpu_time || 0).toFixed(1)}% GPU): ${lanes.length} lanes [${lanes.map(l => l.key + ':' + l.mode).join(', ')}]; roofline×${(rooflineTarget || 0).toFixed(2)}.`);
      return { h, ext, lanes, deepDir, sharedKb, liveBaselineMs };
    };

    const dHeads = heads.slice().sort((a, b) => (b.pct_gpu_time || 0) - (a.pct_gpu_time || 0));   // dominant-Amdahl first
    log(`[deep] cross-kernel × cross-backend co-opt over ${dHeads.length} head op(s); GPU pool {${GPU_LIST.join(',')}}.`);
    headDispatched += dHeads.length;
    const preps = [];
    for (const h of dHeads) { if (DEEP_DEADLINE_HIT) break; const p = await prepHead(h); if (p) preps.push(p); }
    const allLanes = [];
    for (const p of preps) for (const l of p.lanes) allLanes.push(l);
    if (!allLanes.length) { log('[deep] no viable head lanes; nothing to optimize.'); }

    // ---- EV: Amdahl mass × remaining ceiling gap × recent improvement (with exploration floor) -------
    const evOf = (l) => {
      const amdahl = Math.max(0.01, (l.head.pct_gpu_time || 1) / 100);
      const gap = Math.max(0.02, (l.ceiling || 1.5) - l.best);
      const rate = l.ran === 0 ? 0.6 : Math.max(0.03, (l.lastGain || 0));   // unrun lanes get an exploration bonus
      return amdahl * gap * rate;
    };

    // ---- batched HARVEST of a set of lanes from DISK truth (immune to a nulled burst) ----------------
    const harvestLanes = async (set, tag) => {
      if (!set.length) return;
      const byHead = {}; for (const l of set) (byHead[l.head.short_name] = byHead[l.head.short_name] || []).push(l);
      for (const sn of Object.keys(byHead)) {
        const ls = byHead[sn]; const base = ls[0].liveBaselineMs;
        const harvest = await safeAgent(
          `Deep-v2 co-opt harvest [${tag}] for head ${sn} (${ls[0].ext.op_kind}). LIVE baseline geomean = ${base || '?'} ms (the bar; a lane's own "cumulative" is self-relative and NOT comparable — compute speedup vs THIS baseline). Read DISK, do not guess. Lanes:\n` +
          ls.map(l => `- uid=${l.uid}: STATE.json=${l.state_dir}/STATE.json; cumulative-best workspace=${l.state_dir}/best; newest finished run under ${l.deepDir}/runs/${l.key}/team_*/*/ with a final_patch.diff (+ baseline_timing.json / tech_lead_report.md for the optimized per-case ms)`).join('\n') + '\n' +
          `For EACH uid return: uid; has_state (STATE.json AND best/kernel_src exist); best_ms (ABSOLUTE geomean ms of the lane's cumulative-best across cases); vs_live (= ${base || 0}/best_ms when both>0, else 1.0); cumulative (self-relative, ref); eval_dir (newest runs/<key>/team_*/<op> with non-empty final_patch.diff, else ""); patch (that diff path, else ""). Return {lanes:[{uid,has_state,best_ms,vs_live,cumulative,eval_dir,patch}]}.`,
          { phase: 'HeadKernel', label: `harvest ${sn} ${tag}`, schema: HARVEST_SCHEMA });
        const hmap = {}; for (const e of (harvest && Array.isArray(harvest.lanes) ? harvest.lanes : [])) hmap[e.uid] = e;
        const anyState = Object.values(hmap).some(e => e && e.has_state);
        for (const l of ls) {
          const e = hmap[l.uid]; if (!e) continue;
          const g = Number.isFinite(e.vs_live) && e.vs_live > 0 ? e.vs_live : (Number.isFinite(e.cumulative) && !base ? e.cumulative : null);
          if (g != null && g > l.best * 1.001) { l.lastGain = g - l.best; l.best = g; l.noImprove = 0; }
          else if (l.ran > 0) { l.lastGain = 0; l.noImprove++; }
          if (e.eval_dir) l.lastEval = e.eval_dir;
          if (e.patch) l.patch = e.patch;
          if (e.has_state) l.noStateStreak = 0;   // P3: produced a result -> not infeasible
          // ceiling-aware patience: lanes still far from their ceiling get MORE waves before parking.
          const farFromCeiling = (l.ceiling - l.best) > 0.3 * Math.max(l.ceiling, 1e-9);
          const streakCap = farFromCeiling ? DEEP_PLATEAU_STREAK_HIGH : DEEP_PLATEAU_STREAK;
          if (e.has_state === false && anyState && l.ran > 0) {
            l.active = false; l.noStateStreak = (l.noStateStreak || 0) + 1;
            // P3: 2 consecutive bursts with NO persisted result while peers produced => structurally
            // infeasible backend\u00d7op (the backend has no primitive for this op, e.g. flydsl on grouped-MoE).
            // Mark dead so reseed/revive never waste budget re-trying it.
            if (l.noStateStreak >= 2) { l.dead = true; log(`  [deep] DEAD ${l.uid} \u2014 infeasible (no result ${l.noStateStreak}x while peers produced); won't reseed/revive.`); }
            else log(`  [deep] park ${l.uid} \u2014 no persisted result while peers produced.`);
          }
          else if (l.noImprove >= streakCap) { l.active = false; log(`  [deep] park ${l.uid} (plateau ${l.best.toFixed(3)}x, ceiling ${l.ceiling.toFixed(2)}x).`); }
        }
      }
    };

    // ---- CURATE per-op SHARED_KB + run-global GLOBAL_KB, and REVIVE high-ceiling parked lanes ---------
    const curateAndRevive = async (tag) => {
      for (const p of preps) {
        const ls = allLanes.filter(l => l.head.short_name === p.h.short_name && (l.lastEval || l.ran > 0));
        if (!ls.length) continue;
        await safeAgent(
          `KB CURATOR [deep ${tag}] for ${p.h.short_name} (${p.lanes[0] ? p.lanes[0].ext.op_kind : ''}). Per-lane vs-live best: ${JSON.stringify(ls.map(l => ({ uid: l.uid, lang: l.lang, vs_live: l.best, active: l.active, eval_dir: l.lastEval })))}. ` +
          `For each lane with an eval_dir, READ its insight_log.md + tech_lead_report.md; extract what WORKED (measured), what FAILED (scoped dead-end), and any technique another lane should borrow. ` +
          `REWRITE ${p.sharedKb} (keep sections; "Current best per backend" one row per lane; every technique a MEASURED effect + SOURCE; disproven -> Dead-ends; fill "Cross-backend assignments (borrow)" with concrete "lane X found Y -> lane Z try W"). ` +
          `ALSO append cross-KERNEL transferable techniques to ${GLOBAL_KB} (techniques that should generalize to OTHER head ops/backends in this run; one line each, with the source uid). Keep both concise/high-signal. Return {ok,summary}.`,
          { phase: 'HeadKernel', label: `curate ${p.h.short_name} ${tag}`, schema: OKV });
      }
      // revive: a parked lane that is still HIGH-ceiling (big remaining gap) gets one more shot with the
      // freshly-curated cross-backend borrows — so an initially-poor high-ceiling backend is not abandoned.
      let revived = 0;
      for (const l of allLanes) {
        if (l.active || l.dead || (l.revives || 0) >= 2) continue;
        if ((l.ceiling - l.best) > 0.4 * Math.max(l.ceiling, 1e-9)) { l.active = true; l.noImprove = 0; l.revives = (l.revives || 0) + 1; revived++; }
      }
      if (revived) log(`  [deep] revived ${revived} high-ceiling parked lane(s) with fresh borrows.`);
    };

    // ---- BATCHED serial e2e GATE on the serving slot (runs concurrently with co-opt on dedicated cards)
    const runGate = async (opts = {}) => {
      // P4: at FINALIZE, sweep ALL patched lanes (incl. isolated~1.0) onto the CUMULATIVE overlay \u2014 the
      // e2e gate is the true arbiter, so a host-level / oracle-invisible win (isolated~1.0 but real e2e
      // gain, e.g. fused_moe big-M coarsen) still gets banked. Per-wave gates keep the cheap top-2-by-iso.
      const cands = opts.final
        ? allLanes.filter(l => (l.lastEval || l.patch)).sort((a, b) => b.best - a.best)
        : allLanes.filter(l => (l.lastEval || l.patch) && l.best > 1.0).sort((a, b) => b.best - a.best).slice(0, 2);
      if (!cands.length) return;
      const bankedHeads = new Set();   // P4: one kernel per head in the combined overlay (same-head lanes edit the SAME module -> cannot stack; keep the first lane that converts e2e)
      if (opts.final) log(`[deep] FINALIZE gate: sweeping ${cands.length} patched lane(s) onto the cumulative overlay (combined cross-kernel deliverable).`);
      e2eGateCount++;
      log(`[deep] E2E GATE #${e2eGateCount} on serving {${SERVING_GPU}} TP=${SERVING_TP}: [${cands.map(c => c.uid + ' ' + c.best.toFixed(3) + 'x').join(', ')}] (overlapping co-opt on dedicated cards).`);
      for (const c of cands) {
        if (opts.final && bankedHeads.has(c.head.short_name)) { log(`  [deep] FINALIZE: skip ${c.uid} -- head ${c.head.short_name} already banked (same module, cannot stack).`); continue; }
        const deepInputs = {
          EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
          KERNEL_RESULT: {
            short_name: c.head.short_name, task_dir: c.ext.task_dir, op_kind: c.ext.op_kind, lane: c.key,
            winner_kind: 'patch', winner_backend: c.lang,
            target_callable: c.ext.target_callable || c.head.target_callable || '',
            authored_language: c.lang, authored_kernel_eval_dir: c.lastEval,
            apply_env: '', apply_flags: '', code_patch: c.patch || (c.lastEval ? `${c.lastEval}/final_patch.diff` : ''), tuning_artifact: '',
            verified_isolated_speedup: c.best, pct_gpu_time: c.head.pct_gpu_time, parity_note: 'expected_close',
          },
          CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
          CURRENT_THROUGHPUT: curTput,
          BASELINE_FLAGS, BASELINE_ENV, BASELINE_OVERLAY,
          MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
          SKILL_DIR: WORKFLOW_DIR, DEEP_FEEDBACK: true,
          ...tuningIntegrateInputs(), ...ACCURACY_INPUTS,
          ...(opts.final && ACCURACY_GATE !== 'none' ? { ACCURACY_LIMIT: DEEP_FINAL_ACCURACY_LIMIT } : {}),   // de-noise the finalize accuracy decision
        };
        const integ = await safeAgent(
          roleAgent('e2e_integrator', 'integrate', 'Apply a deep head candidate; gate on e2e throughput; report engagement/cudagraph/mem/decode for feedback.', deepInputs),
          { phase: 'HeadKernel', label: `integrate ${c.uid} g${e2eGateCount}`, schema: INTEGRATE_SCHEMA });
        if (integ && integ.output_parity === 'fail') {
          log(`  [deep] ${c.uid}: REJECTED — output_parity=fail vs true baseline.`);
          history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'dead_end', lesson: 'parity fail vs true baseline' });
        } else if (integAccepted(integ, c.head.pct_gpu_time, c.best) && integ.e2e_throughput_tok_s > curTput) {
          curOverlay = integ.accepted_overlay || curOverlay; curTput = integ.e2e_throughput_tok_s; bankedHeads.add(c.head.short_name);
          bankAccepted(acceptedHeads, { short_name: c.head.short_name, op_kind: c.ext.op_kind, backend: c.lang, lane: c.key, kind: 'patch', ...e2eFrom(integ), isolated: c.best }, krOf(deepInputs));
          log(`  [deep] ${c.uid}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%); target ${Math.round(BASELINE_TPUT * DEEP_E2E_TARGET)} tok/s.`);
          history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
        } else {
          // gateRejectReason converts an implausible "pass" into a corruption reject so it routes to the
          // correctness corrective (the deep site already calls the corrective helper for every reject).
          const dreason = gateRejectReason(integ, c.head.pct_gpu_time, c.best);
          const dcorr = await tryCorrectiveReauthor({
            short_name: c.head.short_name, op_kind: c.ext.op_kind, shapes: c.ext.shapes, dtype: c.ext.dtype, regime: c.head.regime,
            gpu_id: SERVING_GPU, kernel_eval_dir: c.lastEval, task_dir: c.ext.task_dir, language: c.lang,
            isolated: c.best, reason: dreason, fix_class: rejectClass(dreason), pct_gpu_time: c.head.pct_gpu_time,
            base_inputs: {
              EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
              KERNEL_RESULT: {
                short_name: c.head.short_name, task_dir: c.ext.task_dir, op_kind: c.ext.op_kind, lane: c.key,
                winner_kind: 'patch', winner_backend: c.lang,
                target_callable: c.ext.target_callable || c.head.target_callable || '',
                authored_language: c.lang, authored_kernel_eval_dir: c.lastEval, apply_env: '', apply_flags: '',
                code_patch: c.patch || '', tuning_artifact: '', verified_isolated_speedup: c.best,
                pct_gpu_time: c.head.pct_gpu_time, parity_note: 'expected_close' },
              SKILL_DIR: WORKFLOW_DIR, ...ACCURACY_INPUTS,
            },
            cur: { overlay: curOverlay, flags: curFlags, env: curEnv, tput: curTput },
          });
          if (dcorr.banked) {
            curOverlay = dcorr.integ.accepted_overlay || curOverlay; curTput = dcorr.integ.e2e_throughput_tok_s; bankedHeads.add(c.head.short_name);
            bankAccepted(acceptedHeads, { short_name: c.head.short_name, op_kind: c.ext.op_kind, backend: c.lang, lane: c.key, kind: 'patch', ...e2eFrom(dcorr.integ), isolated: dcorr.isolated, corrective: true, patch: dcorr.patch || '' }, krOf(deepInputs));
            log(`  [deep] ${c.uid}: ACCEPTED after corrective re-author. e2e now ${curTput} tok/s (+${dcorr.integ.e2e_delta_pct}%).`);
            history.ledger.push({ direction: c.uid, isolated_speedup: dcorr.isolated, e2e_delta_pct: dcorr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed: ${dreason}` });
          } else {
            log(`  [deep] ${c.uid}: e2e gate ${integ ? integ.gate : 'none'} (${integ ? integ.reason || '' : 'integrate failed'}).`);
            history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ ? integ.e2e_delta_pct : 0, verdict: 'dead_end', lesson: integ ? integ.reason || 'no e2e gain' : 'integrate failed' });
          }
        }
      }
      const fb = await safeAgent(
        `You are the e2e FEEDBACK + HARNESS refiner [deep g${e2eGateCount}]. For the candidates just gated, write ${EVAL_DIR}/deep_head/e2e_feedback.md (per candidate: e2e delta; ENGAGED-live vs eager-fell-back under cudagraph; cudagraph/memory/decode behavior; parity/accuracy; ROOT CAUSE of any isolated->e2e gap). ` +
        `Then refresh ${EVAL_DIR}/deep_head/HARNESS_ADDENDUM.md so the isolated target ALIGNS with e2e WITHOUT touching the frozen oracle: (a) e2e-critical decode M-buckets to weight; (b) whether a cudagraph capture/replay measurement wrapper is needed; (c) hard gates (decode-no-regress, memory cap, graph-safe) so an isolated "win" that is all-NUL/eager under graph is caught EARLY. Return {ok, feedback_path, addendum_path}.`,
        { phase: 'HeadKernel', label: `feedback g${e2eGateCount}`, schema: OKV });
      gateFeedbackPath = (fb && fb.feedback_path) || `${EVAL_DIR}/deep_head/e2e_feedback.md`;
      gateHarnessPath = (fb && fb.addendum_path) || `${EVAL_DIR}/deep_head/HARNESS_ADDENDUM.md`;
      // re-profile if the stack moved enough — chase the new dominant bottleneck (Amdahl shifted).
      if (want('profile') && curTput > lastReprofileTput * (1 + DEEP_REPROFILE_GAIN)) {
        log(`[deep] e2e +${((curTput / lastReprofileTput - 1) * 100).toFixed(1)}% since last profile — re-profiling to chase the moving bottleneck.`);
        const rp = await safeAgent(
          roleAgent('profiler', 'reprofile', 'Re-profile the CURRENT overlaid server; return refreshed head pct_gpu_time so EV re-weights toward the new bottleneck.', {
            EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          }),
          { phase: 'HeadKernel', label: `reprofile g${e2eGateCount}`, schema: { type: 'object', additionalProperties: true, properties: { heads: { type: 'array', items: { type: 'object', additionalProperties: true } } } } });
        if (rp && Array.isArray(rp.heads)) {
          for (const nh of rp.heads) {
            const tgt = allLanes.filter(l => l.head.short_name === (nh.short_name || nh.name));
            for (const l of tgt) if (Number.isFinite(nh.pct_gpu_time)) l.head.pct_gpu_time = nh.pct_gpu_time;
          }
          log(`[deep] re-profile updated head Amdahl weights.`);
        }
        lastReprofileTput = curTput;
      }
    };

    // ---- DEPTH: fresh authoring directions to RE-SEED a plateaued lane (so it keeps going DEEPER) ----
    // When a lane plateaus we don't abandon it (and its share of the budget) — we hand it a NEW direction
    // it hasn't tried, biased to the dominant-Amdahl head, so the search compounds depth instead of exiting.
    const DEEP_STEERS = {
      triton: [
        ` DIRECTION=persistent-kernel: a persistent / grid-stride kernel that keeps tiles resident and overlaps global load with ${MATRIX_CORE_NAME}.`,
        ` DIRECTION=warp-specialization: split warps into a producer (async global->LDS copy) and a consumer (${MATRIX_CORE_NAME}) for software pipelining.`,
        ' DIRECTION=epilogue-fusion: fuse the scale/activation/cast epilogue into the GEMM to remove a memory round-trip.',
        RDNA4_ISOLATE
          ? ' DIRECTION=wmma-layout: re-tune WMMA tile shape / LDS swizzle / num_warps / waves_per_eu / GROUP_SIZE_M for this exact (N,K,M-bucket).'
          : ' DIRECTION=mfma-layout: re-tune matrix_instr_nonkdim / kpack / LDS swizzle / waves_per_eu / GROUP_SIZE_M for this exact (N,K,M-bucket).',
        ' DIRECTION=double-buffer: deepen num_stages and LDS double-buffering to hide HBM latency on the K loop.',
        ' DIRECTION=split-K-atomic: split the K reduction across CUs with atomic accumulate for the large-M prefill shapes; non-split for small-M decode.',
        ' DIRECTION=fresh-rewrite: abandon the current tiling and try a fundamentally different decomposition than your best so far.',
      ],
      _default: [
        ' DIRECTION=new-decomposition: try a fundamentally different tiling/decomposition than your current best.',
        ' DIRECTION=fuse-prologue-epilogue: fold the pre/post ops + scaling into the main compute kernel.',
        ' DIRECTION=retune-shapes: per (N,K,M-bucket) re-search the launch-config space from scratch.',
        ' DIRECTION=pipeline: add software pipelining / double buffering across the reduction loop.',
      ],
    };
    const nextSteer = (l) => { const lib = DEEP_STEERS[l.lang] || DEEP_STEERS._default; const s = lib[(l.steerIdx || 0) % lib.length]; l.steerIdx = (l.steerIdx || 0) + 1; return s; };
    // RE-SEED when all lanes parked but budget remains: revive lanes (dominant head first) with a FRESH
    // direction so deep optimization uses the FULL budget instead of exiting at the first global plateau.
    const reseedForDepth = () => {
      let n = 0;
      const order = allLanes.slice().sort((a, b) => (b.head.pct_gpu_time || 0) - (a.head.pct_gpu_time || 0) || b.best - a.best);
      for (const l of order) {
        if (l.dead || (l.reseeds || 0) >= DEEP_MAX_RESEEDS) continue;
        l.active = true; l.noImprove = 0; l.reseeds = (l.reseeds || 0) + 1; l.steer = nextSteer(l); n++;
      }
      if (n) log(`[deep] global plateau but budget remains -> RE-SEED ${n} lane(s) with fresh DEEP directions (depth pass; dominant head first). Keeps compounding until the ${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h budget.`);
      return n > 0;
    };

    // ---- GLOBAL WAVE LOOP (runs until the budget; re-seeds on plateau for depth) ---------------------
    let wave = 0, e2eGateCount = 0, lastE2eIsoBest = 1.0, lastReprofileTput = curTput, gateFeedbackPath = '', gateHarnessPath = '', e2eIntervalHit = false;
    const armInterval = () => { if (typeof setTimeout === 'function' && DEEP_E2E_MAX_INTERVAL_MS > 0) setTimeout(() => { e2eIntervalHit = true; }, DEEP_E2E_MAX_INTERVAL_MS); };
    armInterval();
    let convergeStreak = 0, deepBurstsSpent = 0;
    try {
    // Honour TIME_DEADLINE_HIT as well as DEEP_DEADLINE_HIT (both armed off the same deadline) so the deep
    // wave loop can never become the one dispatcher that runs past the reserve.
    while (!DEEP_DEADLINE_HIT && !TIME_DEADLINE_HIT) {
      if (convergeStreak >= DEEP_CONVERGE_STREAK) { log(`[deep] CONVERGED \u2014 ${convergeStreak} consecutive zero-gain waves; stopping to finalize.`); break; }
      const projAgents = (deepBurstsSpent + mainSlots + serveSlots) * DEEP_AGENTS_PER_BURST + wave * 6;   // project a FULL next wave of bursts + per-wave overhead (harvest/curate/gate)
      if (projAgents >= DEEP_AGENT_BUDGET) { log(`[deep] agent budget reached (proj ~${projAgents}/${DEEP_AGENT_BUDGET}, ${deepBurstsSpent} bursts); stopping to finalize with margin for the cap.`); break; }
      if (!allLanes.some(l => l.active)) { if (!reseedForDepth()) break; continue; }   // depth: don't exit on plateau while budget remains
      wave++;
      const prevTput = curTput;
      const globalIsoBest = Math.max(1.0, ...allLanes.map(l => l.best));
      const gained = globalIsoBest / Math.max(lastE2eIsoBest, 1e-9) - 1;
      const haveCand = allLanes.some(l => (l.lastEval || l.patch) && l.best > 1.0);
      const gateDue = haveCand && (e2eGateCount === 0 || gained >= DEEP_E2E_GAIN_TRIGGER || e2eIntervalHit);
      // pick ready lanes by EV; a gate wave reserves the serving cards (only dedicated cards co-opt).
      const slots = gateDue ? mainSlots : (mainSlots + serveSlots);
      const ready = allLanes.filter(l => l.active).sort((a, b) => evOf(b) - evOf(a)).slice(0, Math.max(0, slots));
      const onMain = ready.slice(0, mainSlots);
      const onServe = gateDue ? [] : ready.slice(mainSlots);
      log(`[deep] WAVE ${wave}: ${allLanes.filter(l => l.active).length} active; running ${ready.length} [${ready.map(l => l.uid).join(', ') || '-'}]${gateDue ? ' + E2E GATE (overlap)' : ''}; e2e ${curTput} tok/s.`);
      const runBurst = (l, pool) => pool.with(1, async (g) => {
        l.ran++;
        await deepBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
          kernel_path: l.ext.task_dir, workflow_dir: KERNEL_WF_DIR, mode: l.mode, target_language: l.lang, op_spec: l.opSpec,
          perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR, use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
          ...KB_ARGS,
          budget: DEEP_WAVE_BUDGET, max_no_improve: DEEP_WAVE_BUDGET, gpu_ids: g[0],
          state_dir: l.state_dir, shared_kb: l.sharedKb, global_kb: GLOBAL_KB,
          incremental_analyze: l.ran > 1 ? 'true' : 'false',   // P2: 2nd+ burst of a lane = continuation -> skip cold re-analysis
          ...(gateFeedbackPath ? { e2e_feedback: gateFeedbackPath } : {}),
          ...(gateHarnessPath ? { harness_addendum: gateHarnessPath } : {}),
          exp_root: `${l.deepDir}/runs/${l.key}`, apply_to_original: 'false',
          task: `deep lane '${l.key}' of ${l.head.short_name} (${l.ext.op_kind}), backend=${l.lang}, mode=${l.mode}.${l.steer} Build STRICTLY beyond this lane's cumulative best (vs-live ${l.best.toFixed(3)}x); roofline SOTA ~${(l.rooflineTarget || 0).toFixed(2)}x. Beat the LIVE kernel, not just your own first port. Read SHARED_KB + GLOBAL_KB and BORROW transferable techniques (incl. from OTHER kernels); write findings back.` + GRAPH_REQ + (TASK || ''),
        }, l.uid);
        return null;
      });
      await Promise.all([
        gateDue ? runGate() : Promise.resolve(),
        parallel(onMain.map(l => () => runBurst(l, cooptMain))),
        parallel(onServe.map(l => () => runBurst(l, cooptServe))),
      ]);
      if (gateDue) { lastE2eIsoBest = globalIsoBest; e2eIntervalHit = false; armInterval(); }   // e2eGateCount is bumped inside runGate()
      deepBurstsSpent += ready.length;
      await harvestLanes(ready, `w${wave}`);
      await curateAndRevive(`w${wave}`);
      const newIsoBest = Math.max(1.0, ...allLanes.map(l => l.best));   // P1 convergence: did this wave move e2e OR any lane's isolated best?
      convergeStreak = (curTput > prevTput * 1.001 || newIsoBest > globalIsoBest * 1.001) ? 0 : convergeStreak + 1;
    }
    } catch (e) { log(`[deep] wave loop aborted (${(e && e.message) || e}); proceeding to finalize so progress is still banked + reported.`); }
    // FINALIZE \u2014 always runs (convergence / agent-budget / deadline / throw all land here). Wrapped so
    // that even if it hits the runtime cap mid-sweep, the overlay banked so far (curOverlay/curTput) holds.
    try {
      await harvestLanes(allLanes.filter(l => l.lastEval || l.ran > 0), 'final');
      await runGate({ final: true });   // P4 FINALIZE: every patched lane -> cumulative overlay (the combined cross-kernel deliverable; banks oracle-invisible host/e2e wins like the manual +21.8%)
    } catch (e) { log(`[deep] finalize partial (${(e && e.message) || e}); best banked overlay so far: e2e ${curTput} tok/s.`); }
    log(`[deep] done after ${wave} wave(s), ${e2eGateCount} gate(s). e2e ${curTput} tok/s (${(curTput / Math.max(BASELINE_TPUT, 1e-9)).toFixed(3)}× baseline; target ×${DEEP_E2E_TARGET}). per-lane vs-live: ${allLanes.map(l => l.uid + '=' + l.best.toFixed(2) + 'x').join(', ')}.`);
  } else if (FAST_MODE && GPU_LIST.length > 1) {
    // ========================= FAST-MODE PARALLEL HEAD TRACK (fast-mode only) =========================
    // The default behavior is the byte-identical serial `else` branch below — this whole block only runs
    // when FAST_MODE is on AND there is more than one card. Design (FAST_PLAN §4, STRICT timing):
    //   opt-A  parallel: per-head extract + bake-off, each leasing ONE card exclusively
    //   opt-B  parallel: ALL (operator × direction) author jobs in one pool, each leasing ONE card
    //   BARRIER: every isolated job has released its card -> ISO pool fully idle
    //   integrate SERIAL on the fixed serving slot {SERVING_GPU} (TP=SERVING_TP), one op at a time
    // Why this satisfies the three requirements:
    //   (1) operators AND optimization directions both fan out (flattened (head,language) job pool);
    //   (2) the GPU semaphore gives every op-bench / kernel-layer job an EXCLUSIVE card, so no two
    //       speed measurements ever share a GPU -> no timing contention while optimizing;
    //   (3) integration happens only AFTER the barrier and SERIALLY on the serving slot (which itself
    //       spans all TP cards), so no isolated work can preempt the e2e A/B -> the gate number is clean.
    const ISO = makeSem(GPU_LIST);
    log(`[fast-mode] PARALLEL head track: ISO lanes={${GPU_LIST.join(',')}} (${GPU_LIST.length}); ` +
      `serving slot={${SERVING_GPU}} TP=${SERVING_TP} reserved for the serial e2e gate (hard barrier between).`);

    // ---- opt-A: per-head extract + bake-off, parallel, exclusive 1-card lease each ----
    const prepared = await parallel(heads.map((h) => async () => {
      if (FAST_DEADLINE_HIT) return { h, dead: 'deadline' };
      return ISO.with(1, async (g) => {
        const gpu = g[0];
        const ext = await extractWithBaseline(
          'kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
            EVAL_DIR, MODEL_PATH, GPU_ID: gpu, WORKLOAD, KERNEL: h, GEMM_SYNTH: gemmSynthFor(h),
            ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
            CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
            REQUIRE_DECODE_BUCKET: true, DECODE_M_BUCKETS: [1, CONC],
            PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
          },
          { phase: 'HeadKernel', label: `extract_op ${h.short_name}`, schema: EXTRACT_OP_SCHEMA });
        if (!ext || ext.smoke !== 'pass' || !ext.task_dir) return { h, gpu, ext, dead: 'extract' };
        const bake = await safeAgent(
          roleAgent('op_benchmarker', 'bakeoff', 'DISCOVER existing impls, tune cheap levers, DECIDE author_plan.', {
            EVAL_DIR, OP_TASK_DIR: ext.task_dir, OP_KIND: ext.op_kind, PCT_GPU_TIME: h.pct_gpu_time,
            CANDIDATE_BACKENDS: ext.candidate_backends || h.candidate_backends || [],
            GPU_ID: gpu, ENABLE_FP8, KERNEL_WF_DIR, KERNEL_BUDGET, SKILL_DIR: WORKFLOW_DIR,
          }),
          { phase: 'HeadKernel', label: `bakeoff ${h.short_name}`, schema: OPBENCH_SCHEMA });
        return { h, gpu, ext, bake };
      });
    }));

    // ---- process opt-A: dominant-head flagging (never silently skip), seed direct_light + author jobs ----
    const headState = new Map();   // short_name -> { h, ext, cands: [] }
    const authorJobs = [];         // flattened (operator × direction) author directions
    for (const p of prepared) {
      if (!p) continue;
      const h = p.h;
      const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
      if (p.dead === 'deadline') { log(`  [fast-mode] ${h.short_name}: skipped (dispatch deadline).`); continue;
      }
      if (p.dead === 'extract' || !p.ext || !p.ext.task_dir) {
        const why = extractFailWhy(p.ext);
        if (isDominant) { log(`  ⚠️ FLAG ${h.short_name}: DOMINANT head op extraction FAILED (${why}) — flagged, NOT skipped.`);
          flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'extract', gate: 'extract_failed', reason: why }); }
        else log(`  ${h.short_name}: op extraction failed (${why}); skipping.`);
        history.ledger.push({ direction: h.short_name, verdict: isDominant ? 'flagged' : 'dead_end', lesson: `op extraction failed (${why})` });
        continue;
      }
      const ext = p.ext, bake = p.bake;
      const harness = !!(bake && (bake.gate === 'harness_error' || bake.harness_suspect));
      const hasPlan = !!(bake && Array.isArray(bake.author_plan) && bake.author_plan.length);
      if (!bake || (bake.gate !== 'have_winner' && bake.gate !== 'author_recommended')) {
        if (isDominant || harness) {
          log(`  ⚠️ FLAG ${h.short_name}: bake-off gate=${bake ? bake.gate : 'null'}${harness ? ' (HARNESS ERROR — not a real no-win)' : ''}.${hasPlan ? ' Proceeding to author route.' : ''}`);
          flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'bakeoff', gate: bake ? bake.gate : 'null', harness_error: harness, had_author_plan: hasPlan, reason: bake ? bake.reason || bake.gate : 'bakeoff null' });
          history.ledger.push({ direction: h.short_name, isolated_speedup: bake ? bake.isolated_speedup : 0, verdict: harness ? 'harness_error' : 'flagged', lesson: bake ? bake.reason || bake.gate : 'bakeoff null' });
          if (!hasPlan) continue;
        } else {
          log(`  ${h.short_name}: no win and nothing worth authoring (${bake ? bake.reason || bake.gate : 'none'}); skipping.`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: bake ? bake.isolated_speedup : 0, verdict: 'dead_end', lesson: bake ? bake.reason || 'no op win' : 'bakeoff failed' });
          continue;
        }
      }
      const st = { h, ext, cands: [] };
      headState.set(h.short_name, st);
      if (bake && bake.gate === 'have_winner' && bake.isolated_speedup > 1.0)
        st.cands.push({ kind: 'direct_light', source: bake.winner_backend, winner_kind: bake.winner_kind,
          apply_env: bake.apply_env || '', apply_flags: bake.apply_flags || '', code_patch: bake.code_patch || '',
          tuning_artifact: bake.tuning_artifact || '', isolated: bake.isolated_speedup, parity_note: bake.parity_note || 'expected_close' });
      for (const ap of (bake && bake.author_plan ? bake.author_plan.slice(0, HEAD_AUTHOR_MAX) : []))
        authorJobs.push({ short_name: h.short_name, h, ext, ap, best_known_ms: bake.best_known_ms });
    }

    // ---- opt-B: ALL (operator × direction) author jobs in ONE parallel pool, exclusive 1-card lease ----
    log(`[fast-mode] author fan-out: ${authorJobs.length} (operator × direction) job(s) across ${GPU_LIST.length} GPU lanes.`);
    const authored = await parallel(authorJobs.map((j) => async () => {
      if (FAST_DEADLINE_HIT) return null;
      return ISO.with(1, async (g) => {
        const lang = j.ap.language || 'triton';
        let al;
        try {
          al = await fastBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
            kernel_path: j.ext.task_dir, workflow_dir: KERNEL_WF_DIR,
            mode: j.ap.route === 'rewrite' ? 'optimize' : 'author', target_language: lang,
            op_spec: { op_kind: j.ext.op_kind, shapes: j.ext.shapes || {}, dtype: j.ext.dtype || 'bf16', regime: j.h.regime || '', cuda_graph_safe: true, ...(j.ext.workload_path ? { workload_path: j.ext.workload_path } : {}) },
            perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
            use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
            ...KB_ARGS,
            budget: KERNEL_BUDGET, gpu_ids: g[0], exp_root: `${EVAL_DIR}/kernels/_exp`,
            task: `Author+optimize a ${lang} implementation of this op vs the immutable oracle (beat ${j.best_known_ms || '?'} ms). ` +
              `This kernel will be overlaid onto the LIVE decode path (CUDA-graph captured): its STEADY-STATE hot path MUST be ` +
              `host-sync-free (NO .item()/.cpu()/.tolist()/.sum().item()/torch.cuda.synchronize(), no Python branch on a GPU scalar). ` +
              `Cache any weight prep (transpose/requant/preshuffle) by weight.data_ptr() done ONCE, not per call. ` +
              `MEMORY FOOTPRINT IS A HARD CONSTRAINT: use the FUSED fp8 path (fold the block-scale into the operand scale, one fp8 ${MATRIX_CORE_NAME} ` +
              `GEMM) and cache only COMPACT fp8/preshuffled weights (never a bf16 expansion); the integrated kernel MUST fit at the ` +
              `accepted config's mem-fraction. ` + GRAPH_REQ + (TASK || ''),
            apply_to_original: 'false',
          }, `${j.short_name}:${lang}`);
        } catch (e) { al = { authored: false, validation_status: 'error', reason: String(e) }; }
        return { j, al };
      });
    }));
    // ---- BARRIER: all isolated optimize done; ISO pool idle; serving slot now contention-free ----
    for (const r of authored) {
      if (!r || !r.al) continue;
      const j = r.j, al = r.al, lang = j.ap.language || 'triton';
      const st = headState.get(j.short_name); if (!st) continue;
      // Rank candidates on the WORKLOAD-WEIGHTED speedup (kernel layer's PRIMARY metric, = the env/
      // direct_light candidate's bake.isolated_speedup), NOT the unweighted geomean. In a decode-bound
      // serving regime the geomean is dominated by zero/low-weight prefill buckets and buries the decode
      // bucket that governs throughput — ranking by geomean can drop the truly-best authored kernel.
      const alIso = al.final_weighted != null ? al.final_weighted : al.final_geomean;
      if (al.authored !== false && alIso > 1.0 && al.final_patch) {
        st.cands.push({ kind: 'authored', source: lang, winner_kind: 'authored', language: lang,
          final_patch: al.final_patch, kernel_eval_dir: al.eval_dir, isolated: alIso });
        log(`  ${j.short_name}: authored ${lang} ${alIso.toFixed(2)}x weighted (vs the frozen online kernel, not the seed).`);
      } else {
        log(`  ${j.short_name}: author ${lang} produced no usable kernel (${al ? al.reason || al.validation_status : 'none'}).`);
        history.ledger.push({ direction: `${j.short_name}:${lang}`, verdict: 'dead_end', lesson: al ? al.reason || 'author no speedup' : 'author failed' });
      }
    }

    // ---- integrate SERIAL on the fixed serving slot, in head order, ISO quiesced (no GPU preemption) ----
    for (const h of heads) {
      const st = headState.get(h.short_name); if (!st) continue;
      const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
      if (!st.cands.length) {
        if (isDominant) { log(`  ⚠️ FLAG ${h.short_name}: DOMINANT head produced NO candidate — flagged, NOT skipped.`);
          if (!flaggedHeads.some((f) => f.short_name === h.short_name)) flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'no_candidate', gate: 'no_candidate', reason: 'bake-off + author route both empty' });
          history.ledger.push({ direction: h.short_name, verdict: 'flagged', lesson: 'DOMINANT head: no candidate to integrate' }); }
        else log(`  ${h.short_name}: no candidate to integrate; skipping.`);
        continue;
      }
      st.cands.sort((a, b) => (b.isolated || 0) - (a.isolated || 0));
      const cand = st.cands[0];
      log(`  ${h.short_name}: best candidate=${cand.source} (${(cand.isolated || 0).toFixed(2)}x, ${cand.kind}). Integrating to e2e (serial, slot {${SERVING_GPU}}).`);
      const headWinnerInputs = {
        EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
        KERNEL_RESULT: { short_name: h.short_name, task_dir: st.ext.task_dir, op_kind: st.ext.op_kind,
          winner_kind: cand.winner_kind, winner_backend: cand.source,
          target_callable: st.ext.target_callable || h.target_callable || '',
          authored_language: cand.language || '', authored_kernel_eval_dir: cand.kernel_eval_dir || '',
          apply_env: cand.apply_env || '', apply_flags: cand.apply_flags || '',
          code_patch: cand.code_patch || cand.final_patch || '', tuning_artifact: cand.tuning_artifact || '',
          verified_isolated_speedup: cand.isolated || 0, pct_gpu_time: h.pct_gpu_time,
          // Pass the Architect's live seam + a concrete engagement assertion so the Integrator can
          // VERIFY the overlay actually binds on the live path BEFORE spending a full e2e A/B — an
          // unreachable lever is then rejected in minutes (no_engagement), not hours.
          live_call_seam: h.live_call_seam || '', engagement_check: h.engagement_check || '',
          parity_note: cand.parity_note || 'expected_close' },
        CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
        CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR,
        ENGAGEMENT_CHECK: h.engagement_check || '',
      };
      const integ = await runIntegrateBothLegs(
        'Apply the head-op winner; gate on e2e throughput.', headWinnerInputs,
        `integrate ${h.short_name}`, 'HeadKernel');
      if (integAccepted(integ, h.pct_gpu_time, cand.isolated) && integ.e2e_throughput_tok_s > curTput) {
        curOverlay = integ.accepted_overlay || curOverlay;
        if (cand.winner_kind === 'env' && cand.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + cand.apply_env;
        if (cand.winner_kind === 'flag' && cand.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + cand.apply_flags;
        curTput = integ.e2e_throughput_tok_s;
        bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: st.ext.op_kind, backend: cand.source, kind: cand.winner_kind, ...e2eFrom(integ), isolated: cand.isolated }, krOf(headWinnerInputs));
        log(`  ${h.short_name}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
        history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
      } else {
        // gateRejectReason injects an implausible_speedup verdict when the gate "passed" but the delta is
        // impossible (corruption) — so a fake win routes to the correctness corrective instead of banking.
        const reason = gateRejectReason(integ, h.pct_gpu_time, cand.isolated);
        const corr = (cand.kind === 'authored' && rejectClass(reason) !== '')
          ? await tryCorrectiveReauthor({
              short_name: h.short_name, op_kind: st.ext.op_kind, shapes: st.ext.shapes, dtype: st.ext.dtype, regime: h.regime,
              gpu_id: SERVING_GPU, kernel_eval_dir: cand.kernel_eval_dir, task_dir: st.ext.task_dir, language: cand.language,
              isolated: cand.isolated, reason, fix_class: rejectClass(reason), pct_gpu_time: h.pct_gpu_time, phase_name: 'HeadKernel',
              base_inputs: {
                EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
                KERNEL_RESULT: { short_name: h.short_name, task_dir: st.ext.task_dir, op_kind: st.ext.op_kind,
                  winner_kind: cand.winner_kind, winner_backend: cand.source,
                  target_callable: st.ext.target_callable || h.target_callable || '',
                  authored_language: cand.language || '', authored_kernel_eval_dir: cand.kernel_eval_dir || '',
                  apply_env: cand.apply_env || '', apply_flags: cand.apply_flags || '',
                  code_patch: cand.code_patch || cand.final_patch || '', tuning_artifact: cand.tuning_artifact || '',
                  verified_isolated_speedup: cand.isolated || 0, pct_gpu_time: h.pct_gpu_time, parity_note: 'expected_close' },
                SKILL_DIR: WORKFLOW_DIR,
              },
              cur: { overlay: curOverlay, flags: curFlags, env: curEnv, tput: curTput },
            })
          : { banked: false };
        if (corr.banked) {
          curOverlay = corr.integ.accepted_overlay || curOverlay; curTput = corr.integ.e2e_throughput_tok_s;
          bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: st.ext.op_kind, backend: cand.source, kind: 'authored', ...e2eFrom(corr.integ), isolated: corr.isolated, corrective: true, patch: corr.patch || '' }, krOf(headWinnerInputs));
          log(`  ${h.short_name}: ACCEPTED after corrective re-author (${reason}). e2e now ${curTput} tok/s (+${corr.integ.e2e_delta_pct}%).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: corr.isolated, e2e_delta_pct: corr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed: ${reason}` });
        } else {
          log(`  ${h.short_name}: REJECTED at e2e gate (${reason}).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ ? integ.e2e_delta_pct : 0, verdict: 'dead_end', lesson: reason || 'no e2e gain' });
        }
      }
    }
  } else {
  for (const h of heads) {
    // Budget guard: stop STARTING new head ops once a dispatch deadline has fired, so the in-flight work +
    // Finalize/Validate still land inside the wall-clock budget. FAST_DEADLINE_HIT covers fast mode;
    // TIME_DEADLINE_HIT covers ALL modes when time_budget_s was passed (inert otherwise => no-op default).
    if ((FAST_MODE && FAST_DEADLINE_HIT) || TIME_DEADLINE_HIT) {
      log(`[budget] dispatch deadline reached — stopping head dispatch before ${h.short_name} (${headDispatched}/${heads.length} heads done).`);
      break;
    }
    headDispatched++;
    // (h1) Extract the op into a standalone immutable unittest. The op-identity guard already forced a
    // fused/monolithic head to op_kind=moe with GEMM_SYNTH off (gemmSynthFor) so it is extracted as the
    // fused op bound at its live seam — never decomposed into a standalone dense GEMM. Nothing is skipped.
    const ext = await extractWithBaseline(
      'kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: h.gpu_id, WORKLOAD, KERNEL: h, GEMM_SYNTH: gemmSynthFor(h),
        ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
        CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        // The unittest MUST span BOTH regimes. Steady-state serving is decode/TPOT-bound, so a
        // head GEMM tuned only on GPU-time-dominant prefill M regresses decode and loses e2e.
        // Pass the decode M explicitly (= running batch ≈ conc) so it is never dropped, plus a
        // per-step M=1. See kernel_extractor.md "Shapes must span BOTH regimes".
        REQUIRE_DECODE_BUCKET: true,
        DECODE_M_BUCKETS: [1, CONC],
        PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
      },
      { phase: 'HeadKernel', label: `extract_op ${h.short_name}`, schema: EXTRACT_OP_SCHEMA });
    const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
    if (!ext || ext.smoke !== 'pass' || !ext.task_dir) {
      const why = extractFailWhy(ext);
      if (isDominant) {
        log(`  ⚠️ FLAG ${h.short_name}: DOMINANT head (${(h.pct_gpu_time || 0).toFixed(1)}% GPU) op extraction FAILED (${why}) — flagged, NOT silently skipped.`);
        flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'extract', gate: 'extract_failed', reason: why });
        history.ledger.push({ direction: h.short_name, verdict: 'flagged', lesson: `DOMINANT head extraction failed (${why})` });
      } else {
        log(`  ${h.short_name}: op extraction failed (${why}); skipping.`);
        history.ledger.push({ direction: h.short_name, verdict: 'dead_end', lesson: 'op extraction failed' });
      }
      continue;
    }
    // (h2) DISCOVER existing impls + tune cheap levers + DECIDE an author_plan.
    const bake = await safeAgent(
      roleAgent('op_benchmarker', 'bakeoff', 'DISCOVER existing impls, tune cheap levers, DECIDE author_plan.', {
        EVAL_DIR, OP_TASK_DIR: ext.task_dir, OP_KIND: ext.op_kind, PCT_GPU_TIME: h.pct_gpu_time,
        CANDIDATE_BACKENDS: ext.candidate_backends || h.candidate_backends || [],
        GPU_ID: h.gpu_id, ENABLE_FP8, KERNEL_WF_DIR, KERNEL_BUDGET, SKILL_DIR: WORKFLOW_DIR,
      }),
      { phase: 'HeadKernel', label: `bakeoff ${h.short_name}`, schema: OPBENCH_SCHEMA });
    if (!bake || (bake.gate !== 'have_winner' && bake.gate !== 'author_recommended')) {
      const gate = bake ? bake.gate : 'null';
      const harness = !!(bake && (bake.gate === 'harness_error' || bake.harness_suspect));
      const hasPlan = !!(bake && Array.isArray(bake.author_plan) && bake.author_plan.length);
      // A DOMINANT head, or a HARNESS fault (not a real no-win), must NEVER be silently skipped.
      // Flag it loudly; and if there is an author_plan, STILL try the author route (it is judged by the
      // immutable unittest, independent of the broken bake-off probe) — so fall through instead of skip.
      if (isDominant || harness) {
        log(`  ⚠️ FLAG ${h.short_name}: ${isDominant ? `DOMINANT head (${(h.pct_gpu_time || 0).toFixed(1)}% GPU)` : 'head'} bake-off gate=${gate}${harness ? ' (HARNESS ERROR — bake-off could not measure; NOT a real no-win)' : ''}. ${hasPlan ? 'Proceeding to author route anyway.' : 'No author_plan to fall back on.'}`);
        flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'bakeoff', gate, harness_error: harness, had_author_plan: hasPlan, reason: bake ? bake.reason || gate : 'bakeoff returned null' });
        history.ledger.push({ direction: h.short_name, isolated_speedup: bake ? bake.isolated_speedup : 0, verdict: harness ? 'harness_error' : 'flagged', lesson: bake ? bake.reason || gate : 'bakeoff null' });
        if (!hasPlan) continue;       // can't author -> FLAGGED (surfaced in report), not a silent skip
        // else: fall through to the author route below (do NOT continue)
      } else {
        log(`  ${h.short_name}: no win and nothing worth authoring (${bake ? bake.reason || gate : 'none'}); skipping.`);
        history.ledger.push({ direction: h.short_name, isolated_speedup: bake ? bake.isolated_speedup : 0, verdict: 'dead_end', lesson: bake ? bake.reason || 'no op win' : 'bakeoff failed' });
        continue;
      }
    }

    // Build the candidate list: the cheap direct_light winner (if any) + any authored implementations.
    const headCands = [];
    if (bake.gate === 'have_winner' && bake.isolated_speedup > 1.0) {
      headCands.push({ kind: 'direct_light', source: bake.winner_backend, winner_kind: bake.winner_kind,
        apply_env: bake.apply_env || '', apply_flags: bake.apply_flags || '', code_patch: bake.code_patch || '',
        tuning_artifact: bake.tuning_artifact || '', isolated: bake.isolated_speedup,
        parity_note: bake.parity_note || 'expected_close' });
    }
    // Author/rewrite route: write (+optimize) a fresh impl per planned language via the recursive kernel
    // layer. mode=author writes a from-scratch baseline then optimizes it; mode=optimize rewrites an
    // existing editable impl. The immutable oracle in ext.task_dir is the judge for both.
    let plan = (bake.author_plan || []).slice(0, HEAD_AUTHOR_MAX);
    for (const ap of plan) {
      const lang = ap.language || 'triton';
      let al;
      // Retry the nested author on a TRANSIENT/early failure (threw, or returned with no real
      // optimization: no final_geomean) — a transient nested-workflow death must NOT silently drop a
      // language (it dropped FlyDSL in the 2026-06-12 run). Do NOT retry a COMPLETED no-speedup
      // (final_geomean present but <=1.0) — that's a real result, retrying just wastes budget.
      const AUTHOR_TRIES = parseInt(A.head_author_tries != null ? A.head_author_tries : (FAST_MODE ? 1 : 2), 10);
      for (let attempt = 1; attempt <= AUTHOR_TRIES; attempt++) {
        try {
          al = await fastBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
            kernel_path: ext.task_dir, workflow_dir: KERNEL_WF_DIR,
            mode: ap.route === 'rewrite' ? 'optimize' : 'author', target_language: lang,
            op_spec: { op_kind: ext.op_kind, shapes: ext.shapes || {}, dtype: ext.dtype || 'bf16', regime: h.regime || '', cuda_graph_safe: true, ...(ext.workload_path ? { workload_path: ext.workload_path } : {}) },
            perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
            use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
            ...KB_ARGS,
            budget: KERNEL_BUDGET, gpu_ids: h.gpu_id, exp_root: `${EVAL_DIR}/kernels/_exp`,
            task: `Author+optimize a ${lang} implementation of this op vs the immutable oracle (beat ${bake.best_known_ms || '?'} ms). ` +
              `This kernel will be overlaid onto the LIVE sglang decode path, which is CUDA-graph captured: its STEADY-STATE hot ` +
              `path (2nd call onward) MUST be host-sync-free — NO .item()/.cpu()/.tolist()/.sum().item()/torch.cuda.synchronize() ` +
              `and no Python branch on a GPU scalar (a host sync DEADLOCKS graph capture → 0 live forwards → e2e rejected). ` +
              `Cache any weight prep (transpose/requant/preshuffle) by weight.data_ptr() done ONCE, not per call. ` +
              `MEMORY FOOTPRINT IS A HARD CONSTRAINT: the persistent weight cache is kept for ALL layers at once, so do NOT ` +
              `re-materialize full bf16 weights (raw+preshuffled bf16 across every layer = tens of GB → forces mem-fraction ` +
              `down → starves the KV-cache pool → net e2e REGRESSION even when the GEMM is faster). Use the FUSED fp8 path ` +
              `(fold the block-scale into the operand scale, run ONE fp8 ${MATRIX_CORE_NAME} GEMM — the "kill the dequant" lever) and cache ` +
              `only COMPACT fp8/preshuffled weights (~the model's own fp8 weight size), never a bf16 expansion. The integrated ` +
              `kernel MUST fit at the same mem-fraction the accepted config uses. ` + GRAPH_REQ + (TASK || ''),
            apply_to_original: 'false',
          }, `${h.short_name}:${lang}`);
        } catch (e) { al = { authored: false, validation_status: 'error', reason: String(e) }; }
        const transient = !al || al.validation_status === 'error' || (al.authored === false && al.final_geomean == null);
        if (!transient || attempt === AUTHOR_TRIES) break;
        log(`  ${h.short_name}: author ${lang} attempt ${attempt}/${AUTHOR_TRIES} died transiently (${al ? al.reason || al.validation_status : 'null'}) — retrying so this language isn't dropped.`);
      }
      // Rank on the WORKLOAD-WEIGHTED speedup (kernel layer's PRIMARY metric, same basis as the
      // env/direct_light candidate's bake.isolated_speedup) — NOT the unweighted geomean, which in a
      // decode-bound regime is dragged down by low/zero-weight prefill buckets and can bury the decode
      // bucket that actually governs serving throughput. Fall back to geomean if no weighted number.
      const alIso = al && (al.final_weighted != null ? al.final_weighted : al.final_geomean);
      if (al && al.authored !== false && alIso > 1.0 && al.final_patch) {
        headCands.push({ kind: 'authored', source: lang, winner_kind: 'authored', language: lang,
          final_patch: al.final_patch, kernel_eval_dir: al.eval_dir, isolated: alIso });
        log(`  ${h.short_name}: authored ${lang} ${alIso.toFixed(2)}x weighted (vs the frozen online kernel, not the seed).`);
      } else {
        log(`  ${h.short_name}: author ${lang} produced no usable kernel (${al ? al.reason || al.validation_status : 'none'}).`);
        history.ledger.push({ direction: `${h.short_name}:${lang}`, verdict: 'dead_end', lesson: al ? al.reason || 'author no speedup' : 'author failed' });
      }
    }
    if (!headCands.length) {
      if (isDominant) {
        log(`  ⚠️ FLAG ${h.short_name}: DOMINANT head (${(h.pct_gpu_time || 0).toFixed(1)}% GPU) produced NO candidate (bake-off + author route both empty) — flagged, NOT silently skipped.`);
        if (!flaggedHeads.some(f => f.short_name === h.short_name)) {
          flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'no_candidate', gate: 'no_candidate', reason: 'bake-off harness/no-win and author route produced no usable kernel' });
        }
        history.ledger.push({ direction: h.short_name, verdict: 'flagged', lesson: 'DOMINANT head: no candidate to integrate' });
      } else {
        log(`  ${h.short_name}: no candidate to integrate; skipping.`);
      }
      continue;
    }
    headCands.sort((a, b) => (b.isolated || 0) - (a.isolated || 0));
    log(`  ${h.short_name}: ${headCands.length} candidate(s) [${headCands.map(c => `${c.source} ${(c.isolated || 0).toFixed(2)}x`).join(', ')}] — e2e-testing EACH; reference leg measured ONCE and reused.`);

    // (h3) e2e gate. Test EVERY candidate, not just the top-iso one: isolated speedup is a weak predictor
    // of e2e (a decode-bound run had an iso-1.055 env tune deliver +14.5% e2e), so the truly-best candidate
    // can only be found by measuring each on the live server. All of a head's candidates A/B against the
    // SAME current config, so the REFERENCE leg is identical for all of them — measure it ONCE (first
    // candidate) and reuse it for the rest via REUSE_REF/SHARED_REF_MED (the integrator then skips the
    // reference server launch and only benches the candidate leg). CAND_TAG keeps each candidate's cand
    // bench/overlay in its own dir so they don't clobber each other. Pick the candidate with the highest
    // MEASURED e2e that clears the gate; only if NONE clears do we fall back to corrective/pending/reject on
    // the top candidate. Inputs are built per-candidate so Fix C can re-issue the SAME A/B at Finalize.
    const mkIntegrateInputs = (cand, ci, sharedRefMed) => ({
      EVAL_DIR, MODEL_PATH, GPU_ID: h.gpu_id, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
      KERNEL_RESULT: { short_name: h.short_name, task_dir: ext.task_dir, op_kind: ext.op_kind,
        winner_kind: cand.winner_kind, winner_backend: cand.source,
        target_callable: ext.target_callable || h.target_callable || '',
        authored_language: cand.language || '', authored_kernel_eval_dir: cand.kernel_eval_dir || '',
        apply_env: cand.apply_env || '', apply_flags: cand.apply_flags || '',
        code_patch: cand.code_patch || cand.final_patch || '', tuning_artifact: cand.tuning_artifact || '',
        verified_isolated_speedup: cand.isolated || 0, pct_gpu_time: h.pct_gpu_time,
        parity_note: cand.parity_note || 'expected_close' },
      CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
      CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR,
      CAND_TAG: `c${ci}_${cand.source}`,
      ...(sharedRefMed != null ? { REUSE_REF: true, SHARED_REF_MED: sharedRefMed } : {}),
    });

    let sharedRefMed = null;      // reference leg measured once per head, then reused by later candidates
    let bestPick = null;          // { cand, integ } with the highest MEASURED e2e that cleared the gate
    let top = null;               // top-iso candidate's {cand, inputs, integ, abc} — the corrective fallback target
    for (let ci = 0; ci < headCands.length; ci++) {
      const cand = headCands[ci];
      const inputs = mkIntegrateInputs(cand, ci, sharedRefMed);
      const integ = await runIntegrateBothLegs(
        'Apply a head-op candidate; gate on e2e throughput (the reference config is shared across this head\'s candidates — reuse it when REUSE_REF/SHARED_REF_MED are set).',
        inputs, `integrate ${h.short_name}:${cand.source}`, 'HeadKernel');
      // A/B completion: gate:'incomplete' / ab_complete:false means a leg is still missing (NOT a rejection).
      const abc = !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);
      if (sharedRefMed == null && abc && integ.ref_med) sharedRefMed = integ.ref_med;   // lock the shared ref
      if (ci === 0) top = { cand, inputs, integ, abc };
      const passed = abc && integAccepted(integ, h.pct_gpu_time, cand.isolated) && integ.e2e_throughput_tok_s > curTput;
      // Persist EVERY candidate's MEASURED e2e (one ledger row per backend, keyed short_name:backend) so the
      // report + experience library keep the full per-backend picture — not just the winning backend.
      history.ledger.push({ direction: `${h.short_name}:${cand.source}`, isolated_speedup: cand.isolated || 0,
        e2e_delta_pct: integ ? (integ.e2e_delta_pct || 0) : 0,
        verdict: passed ? 'candidate_passed' : (abc ? 'candidate_rejected' : 'candidate_incomplete'),
        lesson: `${cand.winner_kind} ${cand.source}: ${integ && integ.e2e_throughput_tok_s ? `${integ.e2e_throughput_tok_s.toFixed(0)} tok/s (${(integ.e2e_delta_pct || 0).toFixed(2)}%)` : (integ ? integ.reason || integ.gate : 'null/timeout')}` });
      if (passed) {
        if (!bestPick || integ.e2e_throughput_tok_s > bestPick.integ.e2e_throughput_tok_s) bestPick = { cand, integ, inputs };
        log(`  ${h.short_name}: candidate ${cand.source} PASSED e2e gate (${integ.e2e_throughput_tok_s} tok/s, +${integ.e2e_delta_pct}%).`);
      } else {
        log(`  ${h.short_name}: candidate ${cand.source} ${abc ? `rejected (${gateRejectReason(integ, h.pct_gpu_time, cand.isolated)})` : `A/B incomplete (${integ ? integ.reason || integ.gate : 'null/timeout'})`}.`);
      }
    }

    if (bestPick) {
      // Best MEASURED e2e among all candidates that cleared the gate. A head winner may be carried as an
      // overlay (authored/patch) AND/OR config (env/flag) — capture both.
      const cand = bestPick.cand, integ = bestPick.integ;
      curOverlay = integ.accepted_overlay || curOverlay;
      if (cand.winner_kind === 'env' && cand.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + cand.apply_env;
      if (cand.winner_kind === 'flag' && cand.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + cand.apply_flags;
      curTput = integ.e2e_throughput_tok_s;
      bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: cand.winner_kind, ...e2eFrom(integ), isolated: cand.isolated }, krOf(bestPick.inputs));
      log(`  ${h.short_name}: ACCEPTED best candidate=${cand.source} (${(cand.isolated || 0).toFixed(2)}x iso). e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
    } else {
      // No candidate cleared the gate. Operate on the TOP (highest-weighted) candidate: corrective re-author
      // for a fixable reject/crash, else keep an incomplete A/B as PENDING, else a plain reject. Same
      // three-state handling as before, now applied to the best candidate after testing them all.
      const cand = top.cand, integ = top.integ, headIntegrateInputs = top.inputs;
      if (!top.abc) {
        // A FIXABLE crash-during-warmup (cand server died -> ZERO A/B samples) is the most common corrective
        // case (cuda_graph_capture_unsafe / host-sync / NO_BINARY_FOR_GPU). Try the corrective re-author
        // first; only keep PENDING if it is not fixable (a real transient timeout/hang) or the fix didn't land.
        const reason = integ ? (integ.reason || integ.gate || '') : '';
        const corr = (cand.kind === 'authored' && FIXABLE_REJECT_RX.test(reason))
          ? await tryCorrectiveReauthor({
              short_name: h.short_name, op_kind: ext.op_kind, shapes: ext.shapes, dtype: ext.dtype, regime: h.regime,
              gpu_id: h.gpu_id, kernel_eval_dir: cand.kernel_eval_dir, task_dir: ext.task_dir, language: cand.language,
              isolated: cand.isolated, base_inputs: headIntegrateInputs, reason,
              cur: { overlay: curOverlay, flags: curFlags, env: curEnv, tput: curTput },
            })
          : { banked: false };
        if (corr.banked) {
          curOverlay = corr.integ.accepted_overlay || curOverlay; curTput = corr.integ.e2e_throughput_tok_s;
          bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: 'authored', ...e2eFrom(corr.integ), isolated: corr.isolated, corrective: true, patch: corr.patch || '' }, krOf(headIntegrateInputs));
          log(`  ${h.short_name}: ACCEPTED after corrective re-author (was crash/incomplete: ${reason}). e2e now ${curTput} tok/s (+${corr.integ.e2e_delta_pct}%).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: corr.isolated, e2e_delta_pct: corr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed crash: ${reason}` });
        } else {
          pendingIntegrations.push({ track: 'head', short_name: h.short_name, isolated: cand.isolated || 0,
            pct_gpu_time: h.pct_gpu_time, inputs: headIntegrateInputs,
            winner_kind: cand.winner_kind, apply_env: cand.apply_env || '', apply_flags: cand.apply_flags || '',
            op_kind: ext.op_kind, backend: cand.source,
            partial: integ ? { gate: integ.gate, ref_med: integ.ref_med, cand_med: integ.cand_med, reason: integ.reason } : null });
          log(`  ${h.short_name}: INTEGRATE INCOMPLETE — A/B not finished (${integ ? integ.reason || integ.gate : 'null/timeout'}); kept as PENDING (not a rejection).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, verdict: 'incomplete', lesson: integ ? integ.reason || 'A/B not finished' : 'integrate timed out/null before A/B completed' });
        }
      } else {
        // gateRejectReason injects an implausible_speedup verdict when the gate "passed" but the delta is
        // impossible (corruption); rejectClass then routes parity/accuracy/implausible rejects to the
        // correctness corrective and JIT/capture/host-sync rejects to the integration corrective.
        const reason = gateRejectReason(integ, h.pct_gpu_time, cand.isolated);
        const corr = (cand.kind === 'authored')
          ? await tryCorrectiveReauthor({
              short_name: h.short_name, op_kind: ext.op_kind, shapes: ext.shapes, dtype: ext.dtype, regime: h.regime,
              gpu_id: h.gpu_id, kernel_eval_dir: cand.kernel_eval_dir, task_dir: ext.task_dir, language: cand.language,
              isolated: cand.isolated, base_inputs: headIntegrateInputs, reason, fix_class: rejectClass(reason), pct_gpu_time: h.pct_gpu_time,
              cur: { overlay: curOverlay, flags: curFlags, env: curEnv, tput: curTput },
            })
          : { banked: false };
        if (corr.banked) {
          curOverlay = corr.integ.accepted_overlay || curOverlay; curTput = corr.integ.e2e_throughput_tok_s;
          bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: 'authored', ...e2eFrom(corr.integ), isolated: corr.isolated, corrective: true, patch: corr.patch || '' }, krOf(headIntegrateInputs));
          log(`  ${h.short_name}: ACCEPTED after corrective re-author. e2e now ${curTput} tok/s (+${corr.integ.e2e_delta_pct}%).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: corr.isolated, e2e_delta_pct: corr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed: ${reason}` });
        } else {
          log(`  ${h.short_name}: REJECTED at e2e gate (${reason}).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct || 0, verdict: 'dead_end', lesson: reason || 'no e2e gain' });
        }
      }
    }
  }
  } // end serial head track (default path; runs for normal mode and fast-mode-single-GPU)
  // Head wins reshape the profile massively (GEMM mass shrinks) — re-profile before the kernel loop.
  if (acceptedHeads.length && !searchClosed('post-head re-profile')) {
    profile = await safeAgent(
      roleAgent('profiler', 'reprofile', 'Re-profile after head-kernel wins.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 'head',
        OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        ...ANALYSIS_SKILL_INPUTS,
      }),
      { phase: 'Profile', label: 'profiler:post-head', schema: PROFILE_SCHEMA });
  }
  log(`Head-kernel track done. ${acceptedHeads.length} accepted, throughput ${curTput} tok/s (${(curTput / BASELINE_TPUT).toFixed(3)}x).`);
  if (flaggedHeads.length) {
    log(`⚠️ ${flaggedHeads.length} DOMINANT head(s) FLAGGED (not optimized, NOT silently skipped): ` +
      flaggedHeads.map(f => `${f.short_name} [${(f.pct_gpu_time || 0).toFixed(1)}% GPU, ${f.gate}${f.harness_error ? '/harness' : ''}]`).join('; ') +
      `. These carry the most headroom — see the report's FLAGGED section.`);
  }
}

