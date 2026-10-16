
// ===========================================================================
// SINGLE-KERNEL PASS-THROUGH: delegate straight to the unchanged kernel layer.
// ===========================================================================
if (!MODEL_PATH && KERNEL_PATH) {
  phase('Setup');
  log(`Single-kernel pass-through -> ${KERNEL_WF_SCRIPT} on ${KERNEL_PATH}`);
  // Recurse into the UNCHANGED kernel layer via the native workflow() primitive (one allowed level of
  // nesting). kernel_workflow.js returns {eval_dir, final_geomean, final_patch, validation_status, ...}.
  let passthru;
  try {
    const r = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, laneArgs({
      kernel_path: KERNEL_PATH, workflow_dir: KERNEL_WF_DIR,
      // The lane defaults to triton; a pass-through caller naming another backend must reach that
      // backend's own warm-start page (the slug is per-language), not triton's.
      ...(A.target_language ? { target_language: String(A.target_language) } : {}),
      use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
      ...KB_ARGS,
      budget: KERNEL_BUDGET, gpu_ids: GPU_IDS, task: TASK, exp_root: EXP_ROOT,
      apply_to_original: APPLY_TO_ORIGINAL,
    }));
    noteKernelKB(r, 'single-kernel pass-through');
    passthru = { ran: true, kernel_eval_dir: r.eval_dir, final_patch: r.final_patch,
      final_geomean: r.final_geomean, validation_status: r.validation_status,
      note: (r.winner && r.winner.source) || '' };
  } catch (e) {
    passthru = { ran: false, kernel_eval_dir: '', final_patch: '', final_geomean: 0,
      validation_status: 'error', note: String(e) };
  }
  log(`Single-kernel done. geomean=${passthru ? passthru.final_geomean : '?'}x`);
  return { mode: 'single_kernel', kernel_path: KERNEL_PATH, ...(passthru || {}) };
}

// OP-IDENTITY GUARD — a fused-MoE / grouped-expert GEMM must be optimized AS the fused op at its live
// dispatcher seam, never decomposed into standalone dense GEMMs (a dense candidate has no live call site
// → no_rebind_seam). So force op_kind='moe' (the grouped-GEMM branch; gemmSynthFor keys on this to keep
// dense synth OFF). The head is never SKIPPED — editability is irrelevant, since a non-editable fused
// kernel is still backend-swapped at its (editable) dispatcher. GENERIC: detects via the Architect's
// is_fused_kernel OR the profile class/name; never keys on a backend name.
// Module-scoped so EVERY strategize/re-strategize call site applies it identically — the initial
// Strategize, the post-config re-strategize, and the post-tuning one — instead of only the first.
// Admission runs here too, so a call site cannot tag identity and then forget to admit.
const _isFusedOp = (c) => (c && c.is_fused_kernel === true) ||
  /(?:^|[^a-z])moe(?:[^a-z]|$)|group(?:ed)?[_ ]?gemm|ck_moe|expert|fused[_ ]?moe|fmoe|asm_moe|fused_custom/i
    .test(`${(c && c.op_kind) || ''} ${(c && c.short_name) || ''} ${(c && c.name) || ''} ${(c && c.classification) || ''} ${(c && c.class) || ''} ${(c && c.backend) || ''}`);
function applyOpIdentityGuard(queue, stage) {
  let tagged = 0;
  for (const c of (queue || [])) {
    if (!_isFusedOp(c)) continue;
    c.op_kind = 'moe';                                                                // grouped-GEMM branch (gemmSynthFor → no dense synth)
    tagged++;
  }
  const admitted = admitHeads(queue, stage);
  if (tagged) log(`[op-identity] ${tagged} fused/grouped head(s): op_kind=moe (never dense-GEMM), bound at live seam — optimized as the fused op, never skipped.`);
  return admitted;
}

