---
key: e2e A/B measurement · any gfx · sglang/vllm
type: method
confidence: ★★★
effect: stops false wins — a positive median inside the noise band is a NULL, not a win; and picks the right correctness gate when byte parity is unavailable
confirms: 4
last_seen: 2026-09-09
---
# Honest e2e A/B: tight interleave + non-overlap gate (not just a positive median)
- lever: run a tight INTERLEAVED A/B (REF, CAND, REF, CAND, …) on a SINGLE GPU with a PINNED port, then
  gate on BOTH `delta_med > noise_band` AND non-overlapping distributions (`cand_min > ref_max`). The
  ~0.5% noise band is real: clean ref/cand medians overlap routinely, so a sub-band delta with
  overlapping [min,max] is a NULL.
- apply: ≥5–7 repeats/leg, back-to-back, same GPU. Combine an accepted-config stack and gate the SUM vs
  the TRUE baseline (small real wins only count when stacked).
- verify: sglang derives `grpc_port = port + 10000` and rejects >65535 → an OS ephemeral port >55535
  crashes launch; ALWAYS pin PORT to a low value. Budget for grpc-port-flake retries.
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-07 / 06-09
- caution (also verify): **byte parity is not always an available gate.** On gfx950 fp8 serving, two
  FRESH no-overlay TRUE-baseline servers with identical flags/env at greedy `temp=0 seed=0` produced
  **0/12 byte-exact completions** — the baseline itself is non-deterministic across launches, so a
  candidate's byte divergence proves nothing. When that is the case, gate on TASK ACCURACY instead:
  a fixed seed-pinned subset, same n for both legs, and McNemar on the discordant pairs rather than a
  raw accuracy delta (e.g. gsm8k n=800 0.9113 vs 0.9025, 21 vs 14 flips, p=0.31 → unchanged). Prove
  the baseline's non-determinism first; do not merely assert it.
- caution (also verify): a delta below the run's own MEASURED across-restart floor is not bankable even
  when the two legs do not overlap — measure that floor from repeated reference-leg server lifetimes,
  never from the configured noise band (a baseline of n=1 reports a degenerate 0.0 spread).
- source: exp/e2e_*Qwen3-14B-FP8*/ 2026-09-09 (fp8 parity non-determinism + accuracy gate)

