// ===========================================================================
// PHASE: UpdateExperience — distill ONE reusable card into this workflow's knowledge/learned/
// (see knowledge/learned/README.md). Runs ONCE, at the very end: never per round, because only
// the final director-verified speedup is evidence a card may cite. `validation` is required for
// the same reason — without it `finalPrimary` falls back to the TechLead's own unverified
// `cumulative`. ADD-only, so a skipped or failing step leaves the run byte-neutral.
// ===========================================================================
let learned_card = null;
// Two runs must never become a card, and neither is a failure. A run that shared its GPU measured
// contention, not the kernel. A run on a HELD-OUT kernel is the instrument for measuring whether the
// KB helps at all — distil it and the next A/B over that kernel reads back its own answer and looks
// spectacular. Defaults are permissive, so the CALLER is the one who has to declare a busy box.
const BOX_QUIET = String(A.box_quiet != null ? A.box_quiet : 'true') === 'true';
const HELD_OUT = String(A.held_out != null ? A.held_out : 'false') === 'true';
const kbGate = !BOX_QUIET ? 'the box was contended — the number measured neighbours, not the kernel'
  : HELD_OUT ? 'this kernel is in the held-out split and is the measuring instrument'
  : '';
if (kbGate) log(`[kb] not distilling: ${kbGate}.`);
// `validation.validation_status === 'accepted'` and not merely "a validation object exists": a
// `flagged` result means the director found a reason not to believe the number (patch install
// no-op, correctness fail, contended box), and curating from it teaches the next run a lesson this
// run did not earn. Reported in review of #411.
const kbAccepted = String((validation && validation.validation_status) || '').toLowerCase() === 'accepted';
if (!kbGate && UPDATE_EXPERIENCE_ON && kbAccepted && nativeGainEligible(validation) && Number.isFinite(finalPrimary) && finalPrimary > 1.0) {
  try {
    learned_card = await agentT(
      roleAgent('update_experience', 'Validate',
        'Curate one distilled learned card from this lane\'s verified win (ADD-only, measured evidence, ' +
        'ratios not wall-clock; record the pitfalls hit; total-then-per-direction for a stacked win).', {
          SCOPE: 'lane', LEARNED_DIR, SKILL_DIR: WORKFLOW_DIR, EVAL_DIR,
          PERF_KNOWLEDGE_DIR: KERNEL_KNOWLEDGE_DIR,
          CURATION_ISOLATE: RDNA4_ISOLATE ? 'true' : 'false',
          REQUIRED_PLATFORMS: RDNA4_ISOLATE ? ['gfx1201'] : [],
          WINNER: {
            kernel: KERNEL_NAME, language: TARGET_LANGUAGE, mode: MODE, gfx: GFX,
            device_target: DEVICE_TARGET, physical_cu_count: PHYSICAL_CU_COUNT,
            kernel_class: (analysis && analysis.kernel_type) || '',
            speedup: finalPrimary, validation_status: validation ? validation.validation_status : '',
            bottleneck: profileSummary ? profileSummary.bottleneck : '',
          },
          // `rounds[]` (per-round directions + per-candidate claimed/verified/status + running
          // `cumulative`) is what makes the card's `stack:` attribution and `pitfall:` lines derivable.
          HISTORY: history,
          // What this run's plan cited and what the verifier then measured. CONTEXT ONLY — the
          // counters are applied by `kb.py drain` from the ledger filed below, never by this agent.
          // The comment that used to sit here said the curator merges them, which stopped being
          // true when that arithmetic moved into code.
          CITATIONS: citations,
          PROFILE: profileSummary,
          REPORT_PATH: report && report.report_path ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
        }),
      { phase: 'Validate', label: 'update_experience', schema: UPDATE_EXPERIENCE_SCHEMA });
    if (learned_card && learned_card.card_path) {
      log(`[kb] learned card ${learned_card.action || 'written'}: ${learned_card.card_path}`);
    }
  } catch (e) {
    log(`[kb] update_experience skipped: ${e && e.message ? e.message : e}`);
  }
}

// FILE THE CITATION LEDGER — OUTSIDE the curation gate, and deliberately so.
// This is the loss half of the loop. It first shipped INSIDE the `finalPrimary > 1.0` branch above,
// which is the exact bias it exists to remove: the runs that produce losses are the ones that did
// not beat the baseline, were flagged, or ran on a contended box, and every one of them was
// filtered out before it could file anything. Selecting successful runs into the feedback data is
// worse than no feedback, because the resulting counters look earned. Reported in review of #411.
// Curation (writing a NEW card) still requires a verified win — a run with nothing to teach must
// not teach — but the ledger of what the cards ALREADY in the tree did is owed for every run.
// `held_out` is the one exception: that split is the measuring instrument, so it neither writes
// cards nor votes on them.
// A workflow script has no filesystem, hence the one-line agent; the JSON is assembled by
// `kb.py cite` rather than by the agent, because a proposal hand-written by a model is one more
// place for the schema to drift.
if (citations.length && !HELD_OUT) {
  try {
    await agentT(
      `Run EXACTLY this command and nothing else. Do NOT edit any file.
\`\`\`bash
cat <<'CITEJSON' | python3 ${WORKFLOW_DIR}/scripts/kb.py --kb-dir ${LEARNED_DIR} cite \\
--run-id ${JSON.stringify(KERNEL_NAME)} \\
--kernel ${JSON.stringify(KERNEL_NAME)} --citations -
${JSON.stringify(citations)}
CITEJSON
\`\`\`
Return {"filed": <the "citations" number the command printed, or 0>}.`,
      { phase: 'Validate', label: 'kb:cite', effort: 'low',
        schema: { type: 'object', properties: { filed: { type: 'number' } },
                  required: ['filed'], additionalProperties: true } });
    log(`[kb] citation ledger filed: ${citations.length} row(s) for the next drain`);
  } catch (e) {
    log(`[kb] citation ledger NOT filed: ${e && e.message ? e.message : e}`);
  }
}
