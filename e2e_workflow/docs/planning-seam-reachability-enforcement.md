# Planning: op-identity fidelity (optimize the op the LIVE kernel actually is)

## Problem
Planning must dispatch a task that MATCHES the real running kernel. The failure: a **fused / monolithic
kernel** (fused-MoE, grouped-expert GEMM, an aiter/CK/asm library kernel) was DECOMPOSED into its
constituent standalone **dense GEMMs** and optimized as a dense `A·Bᵀ`. But the live kernel never
dispatches those standalone GEMMs, so the candidate has **no live call site** and dies at integration
(`no_rebind_seam` / 0 engagement) — ~hours wasted. Observed on both the aiter-routed CK MoE and the
Triton `fused_moe_kernel`.

This is **NOT about editability.** A non-editable library fused kernel can still be **backend-swapped** at
its dispatcher seam (the vLLM `fused_moe`/`fused_experts` dispatcher is editable Python even when the
underlying kernel is a `.so`). The one rule: **optimize the op the live kernel actually is, at the seam it
is actually called from** — never a different op.

## Earlier (wrong) attempt — corrected here
A prior version added a `headIntegrationRoute` gate that **routed fused heads to config and SKIPPED
optimization** (only a tune-hook). Too blunt — it gave up on the biggest lever (a 40–77% GPU fused MoE)
instead of trying other backends. **Removed.**

## Fix (this change) — op-identity guard. Generic, no backend names.
1. **`_isFusedOp(c)`** (Strategize route-guard): a head is fused/monolithic if the Architect marked
   `is_fused_kernel`, OR the profile class/backend/name matches a fused/grouped/expert/MoE/`fused_custom`
   signature. For every such head:
   - force `op_kind = 'moe'` → the grouped-GEMM branch (never the dense-GEMM ladder),
   - set `_forbid_gemm_synth` → the extractor gets `GEMM_SYNTH=false` (no dense decomposition),
   - preserve the live dispatcher seam as `target_callable`.
   The head **stays in the optimization track** — nothing is skipped on editability grounds.
2. **`gemmSynthFor(h)`** threads that per-head flag into all 3 head loops' `extract_op` calls.
3. **Removed** the `headIntegrationRoute` skip-gates (3 loops) and the `editable=false` skip backstop — a
   non-editable head is no longer skipped; it is optimized via backend-swap at the dispatcher seam.
4. **Roles:**
   - `kernel_extractor.md` — op-identity rule: a standalone LIBRARY op → STOP (config); a FUSED op →
     extract the FUSED op, bind `target_callable` to `LIVE_CALL_SEAM` (the editable dispatcher), report
     `editable=true` (rebindable). Never a dense-GEMM proxy for a fused op.
   - `op_benchmarker.md` — for a fused op, **try other fused BACKENDS first** (aiter fused-MoE / live
     Triton / flydsl fused), then author a fused replacement; match the candidate signature to the
     dispatcher; a non-editable underlying kernel is a reason to prefer backend-swap, not to skip.
   - `system_architect.md` — the seam-reachability contract (`is_fused_kernel`/`live_call_seam`/
     `integration_lever`/`engagement_check`) that feeds `_isFusedOp` (from 858e1ea4).
5. **Engagement pre-gate** (kept): the Integrator verifies `ENGAGEMENT_CHECK` on the candidate server BEFORE
   the timed A/B, so any residual signature/seam mismatch dies in minutes, not hours.

## Effect
- Fused MoE (Triton OR aiter/CK) → optimized AS a fused op: backend-swap / tune / author-fused-replacement,
  all bound at the live dispatcher → integrable. No more dense-GEMM `no_rebind_seam` waste, and the fused
  head is **no longer skipped**.
- Real standalone dense GEMMs (qkv/o/lm_head) and attention kernels are unaffected (detection is precise).

## Backward-compat / safety
Non-fused heads: byte-identical behavior. Delimiter balance verified; `_isFusedOp`/`gemmSynthFor`
unit-checked. No `node` in this env — run `node --check e2e_workflow/e2e_workflow.js` before merge.
