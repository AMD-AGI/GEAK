---
key: isolated-vs-e2e ranking · any gfx · decode-bound serving
type: method
confidence: ★★
effect: on ONE head, iso 1.53–1.70× siblings spanned +19.35% … −32.54% e2e — iso rank ≠ e2e rank
last_seen: 2026-08-13
confirms: 1
---
# A big isolated win can be a −30% e2e loss: A/B every sibling candidate, and read TPOT
- lever: treat the op-bench / oracle "serving-weighted speedup" as a **screen that admits candidates**,
  not as a ranking. Carry the top 2–3 siblings to the e2e gate; the winner is decided there.
- apply: same head, same seam, 4 candidates (vLLM fp8 blockscale GEMM, gfx942, conc=64):
  iso 1.6955→**+19.35%**, 1.6797→+13.65%, 1.6488→**−32.51%**, 1.5346 (vendor bpreshuffle)→**−32.54%**.
  All four proved engaged (BIND + monotonic CALL counter), all four parity-clean.
- verify: **split the e2e delta into TTFT vs TPOT.** Both losers showed TTFT ≈ ref but TPOT
  15.53 → 23.27 ms — i.e. the authored/vendor lane is fast on the oracle's captured decode rows yet slow
  at the LIVE decode batch shapes (cuda-graph M buckets, padded M, per-call layout fixups such as an
  x_scale transpose). TPOT is the discriminator; throughput alone hides which lane broke.
- caution: also verify the candidate's config table COVERS the live graph-capture M buckets, not just the
  oracle's captured M (tuning at M∈{1,64} while the server replays a padded M=128 graph is the observed
  failure mode). And when a candidate carries extra HBM (a duplicated shuffled weight, +12.3 GiB here),
  record the KV-pool shrink — but do not assume it is the cause; measure TPOT before blaming memory.
- source: /wekafs/test_results/Qwen3_14B_20260813/e2e_Qwen3-14B-FP8_20260813_031549_2866199_26474/
  overlay/cand_kernel_gemm_xdl_cshuffle_v3/integrate_result*.json
