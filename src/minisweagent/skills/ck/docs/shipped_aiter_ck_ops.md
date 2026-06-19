# Shipped aiter CK ops (try these FIRST)

These are the real, codegen-tuned CK ops shipped in the on-box aiter. Wire the
matching one before authoring anything. All signatures below were verified on
this box (aiter at `/usr/local/lib/python3.12/dist-packages/aiter/`).

## Discovery

```bash
python3 -c "import aiter; print([x for x in dir(aiter) if 'ck' in x.lower() or 'CK' in x])"
```

Observed CK-related symbols include:
`ck_moe_stage1`, `ck_moe_stage2`, `ck_moe_stage1_fwd`, `ck_moe_stage2_fwd`,
`batched_gemm_bf16_CK`, `batched_gemm_a8w8_CK`, `gemm_a8w8_CK`,
`gemm_a8w8_blockscale`, `gemm_a8w8_blockscale_bpreshuffle`,
`gemm_a8w8_bpreshuffle_cktile`, `gemm_a8w8_blockscale_cktile`,
`flatmm_a8w8_blockscale_ASM`, `fmoe_fp8_blockscale_g1u1`, `gemm_a4w4_blockscale`,
`moe_cktile2stages_gemm1`, `moe_cktile2stages_gemm2`, `rmsnorm2d_fwd_ck`, ...

> Note: most ops are wrapped by `@compile_ops`, so `inspect.signature(aiter.<op>)`
> may report `(*args, **kwargs)`. To get the real signature, read the def in the
> source module (paths below) or call the high-level Python wrapper.

## Op-class → op mapping

| target kernel | op to try FIRST |
|---|---|
| MoE / fused_moe / grouped-expert | `aiter.fused_moe` (high-level) or `aiter.ck_moe_stage1` + `aiter.ck_moe_stage2` |
| dense bf16/fp16 GEMM (batched) | `aiter.batched_gemm_bf16_CK` |
| fp8/a8 dense GEMM | `aiter.gemm_a8w8_CK` / `aiter.batched_gemm_a8w8_CK` |
| fp8 per-block-scale GEMM | `aiter.gemm_a8w8_blockscale`, `aiter.gemm_a8w8_blockscale_bpreshuffle` |
| fp8 block-scale flat/preshuffle (ASM) | `aiter.flatmm_a8w8_blockscale_ASM` |
| fp4 (mxfp4) block-scale GEMM | `aiter.gemm_a4w4_blockscale` |

---

## MoE

### High-level (preferred) — `aiter.fused_moe`
Source: `aiter/fused_moe.py:88`. Does sorting + both GEMM stages + activation +
topk-weight in one call. Map your kernel's tensors onto:

```python
fused_moe(
    hidden_states,                 # [tokens, model_dim]
    w1,                            # [expert, inter_dim*2, model_dim]  (gate+up, N,K)
    w2,                            # [expert, model_dim, inter_dim]
    topk_weight, topk_ids,         # [tokens, topk] from gating
    expert_mask=None,              # EP
    activation=ActivationType.Silu,
    quant_type=QuantType.No,       # No / per_Token / per_1x128 / per_1x32(mxfp4)
    doweight_stage1=False,
    w1_scale=None, w2_scale=None,  # quant weight scales
    a1_scale=None, a2_scale=None,  # quant activation scales
    block_size_M=None,             # tuning knob (None -> -1 = auto)
    dtype=None, splitk=0, ...,
)
```

### Low-level 2-stage — `aiter.ck_moe_stage1` / `ck_moe_stage2`
Source: `aiter/ops/moe_op.py:290` (`ck_moe_stage1_fwd`) and `:313`
(`ck_moe_stage2_fwd`); convenience wrappers in `aiter/fused_moe.py:1465`. Use
when you need explicit control over the 2 stages (e.g. custom intermediate or
split-K). Requires pre-sorted token/expert ids (from `aiter.moe_sorting` /
`moe_align_block_size`):

```python
ck_moe_stage1_fwd(
    hidden_states, w1, w2,
    sorted_token_ids, sorted_expert_ids, num_valid_ids,
    out, topk,
    kernelName=None,               # pin a specific tuned instance (else heuristic)
    w1_scale=None, a1_scale=None,
    block_m=32,
    sorted_weights=None,
    quant_type=0, activation=0, splitk=1,
    use_non_temporal_load=False, dst_type=None, is_shuffled=True,
)

ck_moe_stage2_fwd(
    inter_states, w1, w2,
    sorted_token_ids, sorted_expert_ids, num_valid_ids,
    out, topk,
    kernelName=None,
    w2_scale=None, a2_scale=None,
    block_m=32, sorted_weights=None,
    quant_type=0, activation=0, splitk=1,
    use_non_temporal_load=False, dst_type=None, is_shuffled=True,
)
```

`is_shuffled=True` expects **pre-shuffled** expert weights
(`aiter.shuffle_weight`). `quant_type`: 0=No, per_Token, per_1x128 (blockscale),
per_1x32 (mxfp4, gfx950). `activation`: 0=gelu, 1=silu. There is also a ck_tile
variant pair `moe_cktile2stages_gemm1` / `_gemm2` (`aiter/ops/moe_op.py:358`).

---

## Dense GEMM

### `aiter.batched_gemm_bf16_CK` (bf16/fp16, batched)
```python
batched_gemm_bf16_CK(
    XQ, WQ,                        # A [B,M,K], B [B,N,K]
    bias=None, dtype=torch.bfloat16, splitK=None,
)
```

### `aiter.gemm_a8w8_CK` (fp8/int8 A8W8, non-batched)
```python
gemm_a8w8_CK(
    XQ, WQ,                        # quantized A [M,K], B [N,K]
    x_scale, w_scale,              # dequant scales
    bias=None, dtype=torch.bfloat16, splitK=None,
) -> Tensor
```
Picks the tuned instance via `get_CKGEMM_config((cu_num, padded_M, N, K))` from
`a8w8_tuned_gemm.csv` (see `instance_tuning.md`).

### `aiter.batched_gemm_a8w8_CK` (fp8/int8 A8W8, batched)
```python
batched_gemm_a8w8_CK(
    XQ, WQ, x_scale, w_scale,
    bias=None, dtype=torch.bfloat16, splitK=None,
)
```

---

## fp8 block-scale / preshuffle GEMM

`@compile_ops`-wrapped (signatures show `(*args, **kwargs)`; read the def in
`aiter/ops/gemm_op_a8w8.py` for exact params). Typical arg order mirrors the
A8W8 ops (`XQ, WQ, x_scale, w_scale, ...`):

- `aiter.gemm_a8w8_blockscale` — per-1x128 block-scale fp8 GEMM.
- `aiter.gemm_a8w8_blockscale_bpreshuffle` — block-scale + B preshuffle.
- `aiter.gemm_a8w8_bpreshuffle_cktile` / `gemm_a8w8_blockscale_cktile` — ck_tile
  variants.
- `aiter.flatmm_a8w8_blockscale_ASM(XQ, WQ, x_scale, w_scale, dtype=torch.float16)`
  — ASM flat-mm fast path for block-scale fp8.
- `aiter.gemm_a4w4_blockscale` — mxfp4 (gfx950).

---

## Norm / quant (CK)

- `aiter.rmsnorm2d_fwd_ck`, `aiter.rmsnorm2d_fwd_with_add_ck`,
  `aiter.rmsnorm2d_fwd_with_dynamicquant_ck`,
  `aiter.rmsnorm2d_fwd_with_add_dynamicquant_ck` — CK rmsnorm + fused add/quant.

---

## Wiring checklist

- Preserve the kernel's external interface (signature, output shape & dtype,
  `get_inputs()` / `get_init_inputs()`).
- Map existing args → op params using the verified signature; do not guess.
- One launch per logical op — no Python per-expert / per-batch loops.
- Pre-shuffle weights with `aiter.shuffle_weight` when the op expects
  `is_shuffled=True` / preshuffle layout.
- Validate with `save_and_test` after each change; CK fp8/blockscale paths only
  need to match within tolerance.
