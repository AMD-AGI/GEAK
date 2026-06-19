---
key: fp8_a8w8_blockscale dense GEMM · gfx942 · sglang Triton live path
type: lever
confidence: ★★★
effect: iso ~1.06–1.10× prefill (up/gate head); e2e ceiling ~+4–5% on a 57%-GPU head
confirms: 11
last_seen: 2026-06-15
---
# fp8 a8w8 blockscale GEMM → per-(N,K) M-bucketed Triton config-JSON overlay
- lever: the live seam is `aiter.ops.triton.gemm_a8w8_blockscale`. Stock has NO per-(N,K) config for
  the model's large shapes → it uses a generic tile = headroom. Win = drop a tuned
  `gfx942-GEMM-A8W8_BLOCKSCALE-N=<N>-K=<K>.json` into `aiter/ops/triton/configs/gemm/`
  (winner_kind=**patch**; `AITER_TRITON_CONFIGS_PATH` is `__file__`-fixed, not env-overridable).
- apply: **M-bucket the config** — tile shape depends on (N,K):
  · wide-N up/gate (N=34816,K=5120): prefill BM=256/BN=128/GROUP_M=4/nw=8.
  · K-heavy/narrow-N down (N=5120,K=17408): prefill **BM=128/BN=256**/GM=4/nw=8 (widen BN, keep BM).
  · square-ish qkv/o (N=5120,K=6144): prefill BM=256/BN=128/GM=4/nw=8 (small clean win).
  · decode (M≤1024) MUST stay generic BM=128. Integrator must rebind BOTH `sglang...fp8_utils` globals
    `triton_gemm_a8w8_blockscale` + `gemm_a8w8_blockscale_bpreshuffle`.
- verify: honest in-process `config=` kwarg A/B, same synth fp8 operands held fixed, interleaved
  min-of-N; confirm engagement via live `_get_config(M,N,K)` (returns a (dict,use_persistent) tuple → [0]).
- dead-end: a FLAT overlay (BM=256 for all M) tanks decode 0.6–0.7× — decode MUST stay generic.
- dead-end: BN=256 + BM=256 together = LDS spill (0.29×) — widen only one dim.
- dead-end: on the **vLLM CK live path** (not Triton) this overlay does NOT apply — live is CK
  xdl-cshuffle; the lever there is env `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<csv>`, but it yields ~1.00×
  (CK default heuristic already picks the optimal `256x128x128 intrawave_v3`). ALWAYS check which live
  path (CK vs Triton) is engaged BEFORE choosing the lever.
- source: exp/e2e_*Qwen3.5-27B-FP8*/ runs 06-08 … 06-15 (11 consistent re-confirms)
