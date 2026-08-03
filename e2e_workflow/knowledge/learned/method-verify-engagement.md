---
key: engagement verification · any gfx · any backend
type: method
confidence: ★★★
effect: turns "did my kernel actually run live?" from a guess into proof
confirms: 5
last_seen: 2026-08-02
---
# Prove the optimized kernel ran on the LIVE serving path (don't infer from an e2e wiggle)
- lever: instrument the candidate kernel with a one-shot stderr banner and grep the server log — this
  PROVES engagement on both the bench and parity legs, instead of inferring it from a throughput delta
  (which can move for unrelated reasons).
- apply: emit `[overlay-mark] <kernel> OPTIMIZED kernel CALLED` (once) from inside the candidate; for an
  overlay rebind also look for `[overlay] injected module <path>` (N hits = N workers) and `[OVERLAY_ENGAGED]`.
- cheap pre-gate (save the A/B legs): if the candidate's `final_patch.diff` is EMPTY / kernel_src is
  empty / the kernel-layer Director FLAGGED it as `candidate == frozen baseline` (a no-op), then the
  cand config is byte-identical to the current overlay → it can NEVER bind a new kernel on the live path
  (`no_engagement` is guaranteed). Reject WITHOUT running the throughput A/B — two identical configs only
  measure noise. (Seen: add_rmsnorm_quant hidden=5120, empty 0-byte patch, iso ~1.0089 = noise-around-1.)
- verify: ≥1 banner per worker on the live run = engaged. ZERO banners but server healthy = the seam
  missed (wrong rebind target / a self-capturing wrapper fell back to eager — see
  [[method-cudagraph-safe-integration]]). For cudagraph paths, verify engagement INSIDE the captured
  region, not just at module-injection time.
- caution: setattr SUCCESS ≠ live dispatch. A rebind whose overlay applied cleanly (import banner printed,
  `setattr` returned, no error) can still be DEAD if the seam sits on a branch the model never takes.
  Also verify the branch is actually reached at runtime, not just that the attr was replaced. Seen: aiter
  `add_rmsnorm_quant` rebound onto `communicator._fused_rmsnorm_fp8_per_token_quant`, but Qwen3-14B's
  `prepare_attn/prepare_mlp` never pass `quant_format` (defaults to '') so the `fp8_per_token` branch
  (communicator.py L574/L622) is never taken → stock RMSNorm ran; 18 decode batches served, 0
  authored-kernel invocations (ARQ_ENGAGE_MARKER file never flushed) despite the successful setattr. A
  flushed per-invocation MARKER FILE (not just the import banner) is the reliable dead-branch probe.
- source: exp/e2e_*Qwen3.5-27B*/ FLA overlay runs 2026-06-07 / 06-09;
  /wekafs/test_results/Qwen3_14B_20260802 round_2 (add_rmsnorm_quant dead-branch: setattr OK, 0 live hits → REJECT)
