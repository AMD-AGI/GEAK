---
key: engagement verification · any gfx · any backend
type: method
confidence: ★★★
confirms: 5
effect: turns "did my kernel actually run live?" from a guess into proof — and, when the banner also prints the TUNED VALUES, turns a null e2e delta into a one-A/B corrective fix (a -0.222% reject became a +1.2249% accept)
last_seen: 2026-08-24
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
- caution: also verify the overlay FORK you inherited implements the manifest entry you need. A "lazy"
  sitecustomize variant (a meta-path finder that applies rebinds at real-import time — written to dodge a
  startup import that blows the TP rendezvous) may implement ONLY `rebinds`, with no add-module and no
  capture support. Anything seeded `--from` that overlay silently drops its add-module/capture entry and
  BOTH legs resolve to the same installed file, i.e. you measure baseline vs baseline. Two independent
  bugs from one gap in a single run (the capture leg, then the candidate leg). Two defences: keep an
  `assert_legs_differ`-style check that resolves each leg's module file before any timing, and when you
  add a lazy variant, port every manifest kind (rebinds AND modules AND captures) — the missing kinds are
  no-ops for the baseline leg, so adding them is safe.
- **caution (also print the tuned VALUES, not just "ENGAGED" — module-level engagement is not value-level
  engagement).** A candidate can inject in every worker, print its banner, and still execute NONE of the
  constants it was tuned for, because the upstream launcher picks them from a runtime branch (measured:
  `context_attention_fwd` chooses tiles from `is_pow2(kv_page_size)`, so a rewrite's tuned
  `BLOCK_M=128/BLOCK_N=64` silently became 32x32 on a hybrid model). The A/B is clean, the delta is null,
  and nothing looks broken. Make the banner emit the ACTUAL launch parameters plus the runtime quantity
  the branch keys on (`... BLOCK_M=%d BLOCK_N=%d warps=%d real_block_size=%d`) and read it off the live
  worker log BEFORE the timed leg. Corollary for triage: a rejected candidate whose banner shows the
  tuned values did not bind is a CORRECTIVE candidate, not a dead end — re-binding them and re-running one
  A/B turned -0.222% into **+1.2249% accepted, disjoint**, on the identical patch. Generalize to any
  config-table / constant-tuned lever (per-shape GEMM DBs, MoE config JSONs): assert from the live server
  that the tuned VALUES are the ones executing, not merely that your file was loaded.
- source: exp/e2e_*Qwen3.5-27B*/ FLA overlay runs 2026-06-07 / 06-09; exp/e2e_*MXFP4*/ 2026-08-16
  (native-mxfp4 fast path fell back twice on a TP4 shard); exp/e2e_*Qwen3.5-27B-FP8*/ 2026-08-22
  (lazy-overlay manifest gap, TP4 gfx950 vLLM); exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-23..08-24
  (TP2 gfx950 vLLM 0.26: tile-branch value-binding failure and its one-round corrective).
