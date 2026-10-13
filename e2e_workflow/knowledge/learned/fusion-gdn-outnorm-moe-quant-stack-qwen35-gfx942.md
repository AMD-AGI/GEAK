<!-- discovery: key=kernel fusion (GDN out-norm+quant stacked with MoE router/sort/quant) · gfx942 · sglang Qwen3.5-397B GDN-hybrid fp8 decode TP8, cudagraph; confidence=★★; last_seen=2026-10-11 -->
# fusion · GDN output-norm+quant + MoE router/sort/quant stack · gfx942 · Qwen3.5-397B GDN-hybrid fp8 decode TP8

lever: on a GDN(linear-attn)/MoE hybrid, the accepted marginal decode stack is the GDN **output**
tail + the MoE prologue quant — NOT the GDN conv path. Two overlays stacked on the router fusion
(e04) cleared the band together where each was in-noise alone.
- e03: GDN output gated-RMSNorm + per-group-128 fp8 quant, kept the **vendor aiter HIP**
  `gated_rmsnorm_fp8_group_quant` (self-authored Triton tied numerically, slightly slower).
- e06: MoE router prologue + `moe_sorting` + per-group-128 fp8 quant, authored single-launch Triton
  group-quant that folds the standalone `dynamic_per_group_scaled_quant` kernel (#6).

apply: e03 seam = `sglang.srt.models.qwen3_5 Qwen3_5GatedDeltaNet.forward` (output norm+quant);
e06 seam = `aiter.fused_moe.get_quant` (per_1x128 → authored quant) + a `CudaGraphRunner.capture`
warm hook (decode graphs capture before any eager forward). Stack via a combined runpy loader.

verify: `[overlay-e03-gdn-outnorm-quant] ENGAGED` and `[overlay-e06-moe-sort-quant] ENGAGED` on all
8 ranks, 0 scheduler exceptions. Win: **+1.773% e2e** (ref med 296.04 → cand med 301.29,
non-overlapping, spread 0.65%), **TPOT -1.62%**, TTFT -2.07%; gsm8k full-1319 0.5527 vs 0.4246 base
(accuracy_gate PASS, drop -0.128, McNemar p=1.0). 2026-10-11, single non-overlap run.

caution — **ALSO VERIFY the GDN causal_conv1d overlays' STACKED e2e before banking them.** On this
model the two GDN conv overlays are DESTRUCTIVE in a stack even though each is unit-correct and
in-noise solo: e02 (prefill `causal_conv1d` split-free, conv relMAE 1.5e-3, 1.47-1.59×) and e05
(decode `conv_update`+packed gated-delta recurrence, 2.09×, conv_state bit-exact). Every stack
containing e02 or e05 regressed **~-9% e2e** (5-stack -9.59%, 4-stack -8.99%, 3-stack -9.05%);
dropping BOTH (the floor {e03,e06}) recovered +1.773%. Don't assume unit-correct + in-noise-solo
fusions stack additively — gate the stack. Also: e06 and e08 (AR+quant) fold the IDENTICAL kernel
#6 at the same `get_quant` seam and are mutually exclusive — with e08 present it wraps e06 and makes
e06 inert (0 ENGAGED). And two overlays that both arm a `sys.meta_path` post-import finder on the
SAME module need a per-instance re-entrancy guard or they mutually recurse → RecursionError at
import (hit here on `gdn_backend` with e02+e05 stacked).

caution — **the MoE sort+quant rung (here e06) does NOT win SOLO, and n=200 no-thinking gsm8k
cannot gate it.** Re-tested solo in a later run (977c28e6Z, its e02): the single-launch Triton
per-1x128 fp8 group-quant at `aiter.fused_moe.get_quant` (folding kernel #6 + its `shuffle_scale`
transpose into one launch) was unit-EXACT vs the vendor HIP quant (identical MSE, bias ~5e-7),
1.33× unit, ENGAGED 8/8 ranks, cudagraph-safe — yet the serving A/B was **-0.01% (in-noise)**: the
~48µs prefill-quant saving is below the e2e floor (the 4 CK `moe_sorting` kernels dominating the chain
are vendor CK, not foldable). So it only mattered STACKED with e03 (as this card's win shows), never
solo. Also: at n=200 **no-thinking** gsm8k is unusable to gate a numerically-inert quant — two
IDENTICAL no-overlay baseline boots scored 0.555 vs 0.465 (base-vs-base McNemar "fails", p=0.041,
**48% of questions flip**); gate accuracy with thinking-on or full-1319, and never read a single
small-n no-thinking drop as a kernel regression.

source: exp/Qwen35/.../e2e_Qwen3.5-397B-A17B-FP8_...55bdd857Z (apply_stack, 2026-10-11);
exp/Qwen35/.../e2e_Qwen3.5-397B-A17B-FP8_...977c28e6Z (apply_one e02, solo re-test, 2026-10-11).
