---
key: paged decode attention (GQA, page_size=1, bf16 KV) · gfx950 · sglang + aiter
type: routing
confidence: ★★
effect: no op-level backend win — live aiter asm beats the sglang Triton decode kernel 1.27x (M=1) / 1.70x (M=64); head 24.4% GPU, so Tier-C author is the only remaining lever
confirms: 1
last_seen: 2026-08-16
---
# sglang decode paged-attn on gfx950: the aiter prebuilt kernel is the fastest available backend
- setting: Qwen3-14B-FP8, TP=1, `attention_backend=aiter`, page_size=1, GQA 40q/8kv, D=128, bf16 KV,
  decode-only seam `AiterAttnBackend.forward_decode -> paged_attention_ragged`
  (`paged_attention_ll4mi_QKV_mfma16_kernel`, prebuilt `module_aiter_core.so` = NOT editable).
- measured (immutable-oracle operands, cold-flush cuda-event device time under HIP-graph replay):
  aiter 0.0233 ms (M=1) / 0.1408 ms (M=64, ctx=1536); sglang Triton `decode_attention_fwd`
  0.0258 / 0.2386 ms at its BEST `max_kv_splits` (swept 4/8/16/32 — splits are a flat knob here:
  M=64 stays 0.239-0.249 ms). Both legs correct vs the fp32 GQA oracle (rel ~3-4e-3).
- consequence: `--attention-backend triton` / `--decode-attention-backend triton` is a **decode
  REGRESSION** on this box — do not spend a Config-Tuner e2e slot on it for a decode-bound run.
  Tier-A/B yield nothing; the only op-level lever left is Tier-C **route=author** (fresh Triton paged
  decode against the immutable oracle), competing with a hand-tuned MFMA asm kernel at 65% of roofline.
- how to bench it: `op_bench.py bench_attn` deliberately does NOT do a cross-backend attn bake-off
  (returns one `current` entry, `harness_suspect=false` — expected, not a fault). Write a driver that
  reuses the task's `opbuild.make_args` + `harness_lib.time_op(graph=True)` and calls
  `sglang...triton_ops.decode_attention:decode_attention_fwd` directly. Gotchas: `k_scale`/`v_scale`
  must be python floats (1.0), NOT tensors and NOT None (tensor -> triton `pointer<fp32> * f32`
  compile error; None -> `float * NoneType`); k/v buffers are `[pool, HKV, D]` (squeeze the page dim).
- source: exp/e2e_qwen3_14_fp8_20260816 (Qwen3-14B-FP8, gfx950, sglang 0.5.15, head h1 24.39% GPU)
