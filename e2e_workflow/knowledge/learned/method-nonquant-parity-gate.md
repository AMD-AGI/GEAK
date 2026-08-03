---
key: non-quant bf16 kernel rewrite · any gfx · greedy-parity serving
type: method
confidence: ★★
effect: catches "faster-but-wrong" servers — a bf16 kernel that is numerically correct vs fp32 can still FLIP greedy argmaxes; e2e gains here were sub-noise anyway, so the parity fail is decisive
last_seen: 2026-07-30
confirms: 3
---
# Non-quant kernel rewrites must pass BYTE-EXACT greedy parity, not just an fp32 tolerance
- lever: when an authored bf16 kernel replaces a stock non-quant op (rope, rmsnorm/qk-norm, silu
  act_and_mul, scatter), gate it on byte-exact greedy (temp=0, fixed-seed) output vs a FRESH true
  no-overlay baseline server — NOT on maxabs-vs-fp32. The correct bar is byte-exact UNLESS an
  ACCURACY_GATE is configured (then a task-metric tolerance applies).
- apply: run ~10 fixed prompts twice on the baseline first to PROVE it is deterministic (base==base2,
  N/N byte-exact), then run the candidate against it; count prompts that diverge.
- verify: candidate must match the deterministic baseline on all prompts. A different reduction order
  (a correct bf16 kernel) still changes rounding → flips argmaxes → real token/reasoning-path changes
  (one case degenerated into a repetition loop). This is NOT FP run-to-run noise (baseline was
  bit-stable), so it is a REAL model-output change → REJECT and keep previous config.
- caution: also re-check Amdahl BEFORE spending parity budget — these ops were 0.5-1.5% GPU time, so
  even the passing ones (store_kvcache byte-exact) only cleared ~0.01% e2e (distributions overlapped).
  A parity-passing but sub-noise op is a STACK candidate, never a standalone accept; never stack a
  parity FAIL for a sub-noise gain.
- source: /wekafs/test_results/Qwen3_14B_20260730 (rope HIP, rmsnorm qk-norm, silu act_and_mul all
  diverged 6-7/10 vs deterministic baseline; store_kvcache passed 10/10 and was stacked)
