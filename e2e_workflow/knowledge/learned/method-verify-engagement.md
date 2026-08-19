---
key: engagement verification · any gfx · any backend
type: method
confidence: ★★★
confirms: 3
effect: turns "did my kernel actually run live?" from a guess into proof
last_seen: 2026-08-16
---
# Prove the optimized kernel ran on the LIVE serving path (don't infer it from an e2e wiggle)

- path: instrument the candidate with a one-shot stderr banner and grep the server log. This PROVES
  engagement on both the bench and the parity leg, instead of inferring it from a throughput delta
  (which moves for unrelated reasons).
- apply: emit `[overlay-mark] <kernel> OPTIMIZED kernel CALLED` once from inside the candidate; for an
  overlay rebind also grep `[overlay] injected module <path>` (N hits = N workers) and
  `[OVERLAY_ENGAGED]`.
- verify: ≥1 banner per worker on the live run = engaged. Zero banners with a healthy server = the seam
  missed — wrong rebind target, or a self-capturing wrapper fell back to eager (see
  [[method-cudagraph-safe-integration]]). On cudagraph paths, verify engagement INSIDE the captured
  region, not just at module-injection time.
- caution: an in-place fast-path patch (a runtime-gated `apply()` that switches to a native path, rather
  than an overlay rebind) can be missing its banner and still compile and pass. There, exact-zero parity
  (`max_rel_err == 0.0`) against a LIVE baseline is itself the reliable tell of a silent fallback. Make
  the fail-closed engagement assert UNCONDITIONAL — an assert gated on an env var the verify harness
  never sets protects nothing (two rounds shipped a no-op that passed correctness).
- source: exp/e2e_*Qwen3.5-27B*/ FLA overlay runs 2026-06-07 / 06-09; exp/e2e_*MXFP4*/ 2026-08-16
  (native-mxfp4 fast path fell back twice on a TP4 shard).
