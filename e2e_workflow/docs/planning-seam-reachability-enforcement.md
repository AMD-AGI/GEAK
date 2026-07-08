# Planning: enforce seam-reachability (stop authoring un-integrable fused/library heads)

## Problem
When the Config sweep swaps a hot op onto a **library/fused backend** (e.g. aiter fused-MoE → CK
`kernel_moe_gemm_2lds`, or a hipBLASLt GEMM), the re-profile makes those **non-editable library kernels**
the top heads. The System Architect then nominated them as **source-rewrite heads**, so the kernel layer
spent ~hours authoring a standalone/dense replacement that has **no live call site to bind to** →
every candidate failed `no_rebind_seam` / 0 live engagement. Observed on the aiter-routed Mixtral MoE
(the profiler correctly marked them `class=library_gemm, edit=N`, but nothing acted on it).

An earlier commit (`858e1ea4`) added the **planning contract** to `roles/system_architect.md`
(`is_fused_kernel`, `live_call_seam`, `integration_lever`, `engagement_check`) — but **no code consumed
those fields**, so the LLM emitted them and the orchestrator ignored them.

## Fix (this change) — wire the code to consume the planning contract. Generic, no backend names.

1. **`headIntegrationRoute(h)`** (e2e_workflow.js): returns `'config'` for a head whose only lever is a
   config/tune-hook — `integration_lever ∈ {fused-op-tune-hook, dense-linear-env-overlay}`, or
   `is_fused_kernel && integration_lever !== 'author-fused-replacement'`. Otherwise `'author'`. Keys ONLY
   on the Architect's flags, never on a backend name. Absent fields → `'author'` (byte-identical to before).

2. **Seam gate in all 3 head loops** (default serial, fast-parallel, deep): a `'config'`-routed head is
   **flagged + recorded as a config direction and its source-author is skipped** (dominant heads are never
   silently dropped). This removes the ~hours wasted on `no_rebind_seam`.

3. **Second backstop (default loop)**: honor the Extractor's own honest signal — if it reports
   `editable=false` with no `target_callable`, route to config too (catches the case where the Architect
   missed `is_fused_kernel`). Fires only on an EXPLICIT `editable===false`.

4. **Engagement pre-gate**: `live_call_seam` + `engagement_check` are threaded into the integrate inputs;
   `roles/e2e_integrator.md` now verifies the `ENGAGEMENT_CHECK` assertion **before** the timed A/B and
   rejects `no_engagement`/`no_rebind_seam` in minutes if the overlay never bound — the catch-all for when
   an authored candidate slips through the planning gate.

5. **`author-fused-replacement` path**: `LIVE_CALL_SEAM`/`INTEGRATION_LEVER` are passed to the extractor;
   `roles/kernel_extractor.md` now: (a) reports `editable=false, target_callable=""` for a fused/library op
   instead of synthesizing a standalone-GEMM proxy, and (b) for `author-fused-replacement`, extracts the
   FUSED op and binds `target_callable` to `LIVE_CALL_SEAM` so the authored kernel is integrable.

## Defense in depth
Architect flag (1) → Extractor honest signal (3) → Integrator engagement pre-gate (4). A fused/library
head is caught at planning (no wasted author); if it slips through, it dies in minutes at the pre-gate,
not hours at a full A/B.

## Backward-compat
Every gate is a no-op when the new fields are absent / `editable` is not explicitly false → a strategy
that doesn't emit the planning contract behaves exactly as before. No `node` in this env; delimiter
balance verified and `headIntegrationRoute` unit-checked — run `node --check e2e_workflow/e2e_workflow.js`
before merge.
