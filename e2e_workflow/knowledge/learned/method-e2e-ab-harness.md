---
key: e2e A/B measurement · any gfx · sglang/vllm
type: method
confidence: ★★★
effect: stops false wins — a positive median inside the noise band is a NULL, not a win
confirms: 3
last_seen: 2026-06-09
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
