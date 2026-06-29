---
key: int4_w4a16 FlyDSL MoE GEMM2-leg levers (inside landed shim) · gfx942/MI300X · vLLM
type: lever
confidence: ★★
effect: AFTER the FlyDSL int4 MoE apply-back is accepted, the GEMM2 (down-proj) leg becomes the new #1 head (43–47% eff GPU). TWO distinct levers exist. (1) TILE-TUNE: stage-2 prefill tile_m 32→64 (tile_n=256, tile_k=128; GEMM1/decode untouched) lands +10.35% e2e VERIFIED (single-run non-overlapping, 12/12 byte-exact); SAME-tile re-tune is now exhausted (2 back-to-back nulls). (2) FUSED-XROWS (the structurally-new angle the tile-tune card called for): GEMM2-leg fused-SwiGLU + un-replicated-X (`FLYDSL_MOE_FUSED_XROWS=1`) lands +6.49% e2e STACK (provisional — engaged, parity-safe, clearly non-negative, but distributions OVERLAP at 2 repeats so NOT yet standalone-acceptable). Both are parity-neutral / ZERO extra HBM.
last_seen: 2026-06-28 (ROUND 2 of the 0628 eval: fused-xrows lever ENGAGED x4, +6.49% STACK — see source below. Tile-tune lever banked & exhausted as of 0626 R4.)
---
# Tile-tune the FlyDSL GEMM2 down-proj leg at the LIVE _pick_tiles seam (follow-on after apply-back)
- lever: once `apply_flydsl_moe_to_vllm` is accepted, re-profile — the FlyDSL `moe_gemm2_0` down-proj leg
  is the new dominant head (47.15% eff GPU, raw 28.9%, skew 1.7× healthy). Tune ITS prefill tile: stage-2
  prefill `tile_m 32→64` at `tile_n=256, tile_k=128`. Tile-size-only ⇒ parity-neutral, ZERO extra HBM.
- apply: the win must be wired into the LIVE selection path `flydsl_fused_experts_impl → _get_exe →
  _pick_tiles` — make `_pick_tiles` STAGE-AWARE (stage-2 prefill M>64 ⇒ tile_m=64), leave GEMM1 + decode
  untouched. Bundle as a patched shim in the carried overlay, engage via `FLYDSL_SHIM_DIR=.../flydsl_shim_g2tune`.
- verify: grep the bench server for the live `ENGAGED stage2 prefill tile_m=64` hits on all 4 TP workers
  (this run: 45600 in CAND, 0 in REF stock shim); KV unchanged (19.9 GiB both legs, memory-neutral); then
  the e2e gate (non-overlap A/B) + parity vs the ACCEPTED FlyDSL ref (12/12 byte-exact here).
- caution: **an isolated GEMM-leg patch that only ADDS a harness-shaped entry point does NOT auto-engage.**
  The given final_patch added a standalone `grouped_gemm2_candidate` the live path never calls. Always port
  the verified tile into the real call site (`_pick_tiles`/`_get_exe`), stage-scoped, and PROVE engagement
  with a live counter — iso win ≠ e2e win until the live tile selection actually picks the new tile.
- caution: **keep GEMM1 + decode tiles untouched** — the win is prefill-M-bucket-specific (M=2048 −26%,
  M=4096 −7.5%, decode unchanged in the kernel-layer measure); a global tile_m bump can regress decode.
- caution: **a byte-identical follow-on tune banks no further win** — once tile_m=64 is in the live shim,
  a re-attempt that ships an empty patch (final_patch.diff 0 bytes / cand shim md5-identical to accepted)
  is e2e=0% by construction and fails the engagement gate (no new kernel on the live path). Don't burn a
  TP=4 A/B on an overlay vs a copy of itself; the GEMM2 prefill win is already banked — move to the next
  M-bucket / tile_n,tile_k sweep or the `_batched_gemm_a8w8` subkernel for a NEW lever. (ROUND 3 null:
  empty/byte-identical; ROUND 4 null: a re-tune produced no isolated speedup at all.) **Two consecutive
  re-tune nulls on the SAME prefill tile ⇒ this exact lever is exhausted** — also verify any next GEMM2
  idea changes the LIVE-path bytes AND clears the kernel-layer iso gate before spending a TP=4 A/B.
- source: 2026-06-26 e2e run (eval e2e_moonshotai-Kimi-K2.6_20260626T103907Z, ROUND 2, gfx942 MI300X TP4
  GPU 0,1,2,3, vLLM 0.19.0, Kimi-K2.6 int4-W4A16 g32; on top of the accepted FlyDSL apply-back stack).
  Tight 2-launch A/B (ISL8192/OSL1024/cc64, 2 reps): REF med 531.741 (513.5/549.9) → CAND med 586.749
  (566.9/606.6) = +10.35%, non-overlapping (cand_min 566.94 > ref_max 549.94), TPOT 105.9→94.4 ms.
  Kernel-layer iso: prefill tile_m 32→64 gave M=2048 −26% / M=4096 −7.5% (Director-verified, correct).
  Companion `[[flydsl-moe-applyback-gfx942]]` (the apply-back this builds on).
  ROUND 3 (same eval): GEMM2 stays #1 head but de-inflated 47.15→42.69% eff (avg 285.7→233.5µs, −18%)
  after the accepted h1; a re-tune attempt shipped an empty/byte-identical patch → e2e=0% null (see the
  byte-identical-follow-on caution). Win remains banked at 586.749 tok/s.
  ROUND 4 (same eval): GEMM2 still #1 head (~42.69% eff); a second re-tune attempt produced no isolated
  speedup → another dead_end. The two back-to-back nulls confirm the prefill tile_m=64 win is fully
  banked and the same-tile lever is tapped out; the next GEMM2 lever must be structurally new.

# Lever (2): GEMM2-leg fused-SwiGLU + un-replicated-X (the structurally-new angle) — FLYDSL_MOE_FUSED_XROWS=1
- lever: the genuinely-new GEMM2 angle the tile-tune card asked for. Instead of re-tiling, FUSE the SwiGLU
  activation into the GEMM2-leg epilogue AND stop replicating X across the topk fan-out (un-replicated /
  compact "xrows" input). New impl `flydsl_fused_experts_impl_fused_xrows` + module `moe_gemm_2stage_xrows.py`.
- apply: build a candidate overlay = accepted FlyDSL shim + patched `flydsl_moe_shim.py` exposing the new
  fused-xrows impl + the new `moe_gemm_2stage_xrows.py`, wired into the LIVE path via an env-gated dispatch
  (`FLYDSL_MOE_FUSED_XROWS=1`) INSIDE `flydsl_fused_experts_impl`. Off ⇒ falls back to the accepted shim.
- verify: prove engagement on ALL 4 TP workers — `fused_xrows ENGAGED x4`, convert x240, non-fused impl
  count=0, 0 shim errors, healthy cudagraph capture (no host-sync hang, decode ~940 tok/s). Memory parity:
  same mem-util 0.9, KV 20.13 GiB / 307,566 tok, no OOM. Then the e2e A/B + isolated correctness gate.
- caution: **at 2 repeats this lever scored +6.49% MEDIAN but the run distributions OVERLAP** (cand_min
  548.88 < ref_max 556.93), so it is a STACK (provisional, parity-safe, engaged, non-negative) — NOT a
  standalone 'accepted'. Also verify with MORE repeats + the Director's gsm8k accuracy gate before promoting
  to ★★/★★★; it is an epilogue-fusion QUANT change, so the Director's FINAL combined validation is the
  authoritative headline gate. Topology shift to re-check next: fused-xrows SPLITS the single moe_gemm2_0
  into TWO engaged FlyDSL kernels — moe_gemm1_0 (stage1 gate_up w/ in-epilogue SwiGLU) becomes the bigger
  #1 (31.49% eff), moe_gemm2_0 down-proj #2 (14.91% eff); the next GEMM lever should target gemm1.
- source: 2026-06-28 e2e run (eval e2e_moonshotai-Kimi-K2.6_20260628T122336Z, ROUND 2, gfx942 MI300X TP4
  GPU 0,1,2,3, vLLM, Kimi-K2.6 int4-W4A16 g32; stacked on the accepted FlyDSL apply-back shim, provenance
  OK — unittest.py sha256 == meta.json). Tight 2-launch A/B, E2E_REPEATS=2, same serving config both legs:
  REF med 535.53 (514.12/556.93) → CAND med 570.26 (548.88/591.64) = +6.49% (>> 0.5% noise, cand_med ≥
  ref_med) but OVERLAPPING ⇒ STACK. Isolated correctness Director-verified (all M-buckets PASS within
  2e-2/0.20 gate). Companion `[[flydsl-moe-applyback-gfx942]]` (the apply-back both GEMM2 levers build on).
