---
key: e2e A/B measurement · any gfx · sglang/vllm
type: method
confidence: ★★★
effect: stops false wins — a positive median inside the noise band is a NULL, not a win
confirms: 4
last_seen: 2026-08-14
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
- caution (**box drift is bigger than most wins — also re-measure the REF in the SAME session**,
  gfx942/vLLM Qwen3-14B-FP8, 2026-08-14): the SAME accepted config medianed 4800.5 tok/s in one session
  and 4593.6 tok/s hours later (−4.3%), and within one session the ref leg alone spread 4388–4645
  (5.8%) with one faulted repeat at 1541 tok/s / 45 ms TPOT. So (a) NEVER quote a delta against a
  throughput recorded in an earlier session, (b) pool ≥2 ref blocks and report min/max, and (c) drop or
  flag a faulted repeat by its TPOT outlier rather than letting it inflate the delta (+1.19% vs +0.59%
  here). A +0.6% median with `cand_min < ref_max` on a 5.8%-spread session is a NULL for the solo gate —
  stack it and let the Director's combined validation decide.
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-07 / 06-09;
  /wekafs/test_results/Qwen3_14B_20260813/e2e_Qwen3-14B-FP8_20260813_031549_2866199_26474/overlay/cand_fused_quant_prologue/integrate_result.json
