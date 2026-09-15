---
key: fused_custom_attention (dense non-causal MHA, no KV cache) · gfx950 · sglang-diffusion DiT
type: lever + dead-end
confidence: ★★
effect: isolated 1.074x weighted (bit-exact) from calling aiter.fmha_v3_fwd directly; Amdahl ceiling
  only ~0.73% e2e at 10.56% GPU time -> stack-only. NO backend swap wins: triton FA 2.0x SLOWER,
  torch SDPA 2.5x slower than the aiter asm kernel.
confirms: 1
last_seen: 2026-07-30
---
# FLUX.1-dev DiT joint attention (`aiter::fmha_fwd_hd128_bf16`) on MI350X

Shape: B=1, S=4608 (4096 image + 512 T5 text tokens), **H=3 heads/rank** (24 heads / Ulysses SP=8),
D=128, bf16, non-causal, dropout 0, no mask, no KV cache. 57 calls per DiT forward x 28 denoise
steps = 1596 calls/image; 10.56% of profiled denoise GPU time.

- **dead-end — Tier-A backend swap.** Measured on the immutable oracle (cuda-event device ms,
  cold-flushed, S=4608): aiter `flash_attn_func` (live) **0.1070**, aiter `mha_fwd` (CK entry)
  0.1606, aiter **Triton** FA (`aiter.ops.triton.attention.mha`) 0.2122, torch SDPA
  (default/flash/mem-efficient all identical) **0.267**, SDPA math 0.812. The hand-written asm
  kernel wins by 2-2.5x. Do NOT spend a config slot on `--attention-backend {torch_sdpa,fa}` for
  a FLUX DiT; the aiter default is already the fastest available implementation.
- **lever (small, free, bit-exact) — skip the `FlashAttnFunc` wrapper.** `AITerImpl.forward` calls
  `aiter.flash_attn_func(..., return_lse=True)` and **discards the LSE**. Calling
  `aiter.fmha_v3_fwd(q,k,v, 0.0, D**-0.5, False, -1, -1, False, False, 0)` runs the SAME asm
  kernel with `return_softmax_lse=False` and is **1.074x weighted / 1.07x on the live shape,
  1.78x on S=64, 2.07x on S=1** (short buckets are wrapper-dominated). Output is **bit-identical**
  (max_rel_err 0.0 vs the frozen oracle), so image parity is free. Passing `return_lse=False` to
  `flash_attn_func` does NOT help — the cost is in the autograd-Function/dispatcher wrapper, not
  the LSE write. `how_v3_bf16_cvt` in {0,1,2} is within noise (0.0976/0.0980/0.0978 ms).
- **caution:** the Amdahl ceiling of that lever is ~0.73% e2e, BELOW this box's 1.92% baseline
  spread. It cannot be banked alone — stack it, or verify with a many-repeat A/B.
- **where the real headroom is (author target).** The kernel reaches only ~326 TFLOP/s
  (4*B*H*S^2*D = 32.6 GFLOP in 0.100 ms) = **~14% of the MI350X bf16 peak**, because Ulysses SP=8
  leaves just **H=3** heads per rank: the grid is B*H*ceil(S/BLOCK_M) = 108 workgroups at BM=128
  (216 at BM=64) against **256 CUs** — the GPU is structurally under-filled, and no amount of
  backend swapping fixes that. A split-KV / flash-decoding-style two-pass Triton FA (split the
  4608-long K/V across workgroups, then combine with the running LSE) is the lever that can add
  parallelism. This is the Tier-C author case; note the seed can start from aiter's editable
  Triton FA but must close a 2.0x gap to the asm kernel before it wins anything.
- **rebind seam:** `AITerImpl.forward` (BF16 path) in
  `multimodal_gen/runtime/layers/attention/backends/aiter.py`. Overlay the FILE (not the class
  attribute): `USPAttention.__init__` setattr's a DEBUG-wrapped BOUND copy of `forward` onto the
  instance at construction, so a post-construction class monkeypatch never runs.
- source: exp/e2e_FLUX-1-dev_20260729_132053_288743_23791 (h1 bake-off, 2026-07-30); isolated only,
  e2e gate not yet run.
