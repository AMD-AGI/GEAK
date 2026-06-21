---
key: engagement verification · any gfx · any backend
type: method
confidence: ★★★
effect: turns "did my kernel actually run live?" from a guess into proof
confirms: 3
last_seen: 2026-06-09
---
# Prove the optimized kernel ran on the LIVE serving path (don't infer from an e2e wiggle)
- lever: instrument the candidate kernel with a one-shot stderr banner and grep the server log — this
  PROVES engagement on both the bench and parity legs, instead of inferring it from a throughput delta
  (which can move for unrelated reasons).
- apply: emit `[overlay-mark] <kernel> OPTIMIZED kernel CALLED` (once) from inside the candidate; for an
  overlay rebind also look for `[overlay] injected module <path>` (N hits = N workers) and `[OVERLAY_ENGAGED]`.
- verify: ≥1 banner per worker on the live run = engaged. ZERO banners but server healthy = the seam
  missed (wrong rebind target / a self-capturing wrapper fell back to eager — see
  [[method-cudagraph-safe-integration]]). For cudagraph paths, verify engagement INSIDE the captured
  region, not just at module-injection time.
- source: exp/e2e_*Qwen3.5-27B*/ FLA overlay runs 2026-06-07 / 06-09
