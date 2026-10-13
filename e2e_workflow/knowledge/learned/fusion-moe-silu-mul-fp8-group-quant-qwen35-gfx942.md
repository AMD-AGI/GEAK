# fusion · MoE expert silu_and_mul + per-group fp8 quant · gfx942 · sglang Qwen3.5-397B MoE fp8 decode TP8, cudagraph

reuse key: kernel fusion (MoE expert intermediate silu(gate)*up + per-1x128 fp8 quant) · gfx942 · sglang Qwen3.5 MoE fp8 decode TP8, cudagraph

lever: In the aiter MoE expert path, the SiLU(gate)*up activation and the a2 (expert-input) per-group
  fp8 quant between the two expert GEMMs run as TWO separate decode kernel launches, not one. Fuse
  them into ONE self-authored Triton kernel `fused_silu_mul_fp8_quant(x_fp32[M,2H]) -> (fp8[M,H], scale[M,1])`
  (silu in fp32, quantize once). Here inter_dim H=128 == group_size 128, so per-token == per-1x128.

apply (seam): the experts run `aiter.fused_moe.fused_moe -> fused_moe_` (REGISTERED torch_compile_guard
  custom op) `-> fused_moe_2stages`; for per-1x128 fp8 the CK stage1 takes the split-K path and emits a
  FP32 interleaved gate||up buffer, so `ck_moe_stage1` calls `aiter.silu_and_mul(out, valid_out_fp32)` and
  `fused_moe_2stages` then calls `quant_func(a2,...)` (= `get_quant(per_1x128)` = `per_group_quant_hip(128)`).
  Overlay: rebind `aiter.silu_and_mul` — on the fp32-input/bf16-out/last==2*out seam run the fused kernel,
  stash (fp8,scale) keyed by `out.data_ptr()`, SKIP the stock bf16 silu (its only consumer is the quant we
  intercept); and wrap `aiter.fused_moe.get_quant` so the returned quant_func returns the stashed fp8 on a
  data_ptr match. 2 launches -> 1. Lazy sys.meta_path post-import hooks on `aiter` / `aiter.fused_moe`.

verify (engagement): server banner `[overlay-e04-silu-mul-fp8-quant] ENGAGED` + `ENGAGED-consume` on ALL
  8 ranks (silu=8 consume=8 fallback=0), coexisting with e01/e02/e06; the fused silu+quant captured in the
  decode graph. Under cudagraph replay Python counters read 0 — the startup banner is the proof.

caution: Both calls execute in the SAME `fused_moe_` capture frame, so the a2 buffer's data_ptr identity
  is preserved silu.out -> quant.input — this is WHY a plain stash/consume works here, UNLIKE e05 whose
  fold-OUTSIDE / consume-INSIDE the fused_moe_ boundary lost data_ptr identity (moe=0). `fp32 input` is
  exclusive to the MoE stage1 activation (CK/g1u1/cktile), always followed by the a2 quant, so dense/shared
  MLP silu (bf16 input, or sglang's own `SiluAndMul`) is untouched — guard on it. ALSO VERIFY: unlike the
  expert-INPUT quant fold (e05, in-noise -0.164% amortized by the MoE decode cudagraph), fusing the
  expert-INTERMEDIATE silu+quant removes a real activation kernel + bf16 round-trip on the critical path and
  WINS. self-authored (iso 1.636x) > existing `act_mul_and_fp8_group_quant` (1.323x) -> keep self.

source: Qwen3.5-397B-A17B-FP8 20261010 e04 (2026-10-12): LIVE warm_server TP8 GPU0-7 A/B
  ref=[319.453,317.588,319.138] med 319.138 vs cand=[322.426,323.491,321.114] med 322.426 -> +1.030%
  non-overlapping (cand_min 321.114 > ref_max 319.453), TPOT 11.768->11.638 ms (-1.10%), TTFT flat;
  accuracy unit-equivalence PASS (calibration mode=unit_equivalence_only). confidence ★★ (single-run
  non-overlapping A/B). last_seen 2026-10-12.
