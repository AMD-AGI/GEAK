---
title: causal_conv1d — numerics
kind: operator_overview
operator: causal_conv1d
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/causal_conv1d_update.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:op_tests/triton_tests/test_causal_conv1d.py
  - https://github.com/Dao-AILab/causal-conv1d
---

# causal_conv1d — numerics

## The contract
A ≤4-tap depthwise MAC plus optional SiLU. The accumulation is **fp32** regardless of input dtype: the
HIP kernel reads `x`/`W`/`bias` and promotes each to `float`, accumulates `out_val += weight_vals[j] *
x_vals[j]` in fp32, applies SiLU in fp32 (`out_val / (1 + expf(-out_val))`), then casts to the input
dtype on store (`out[...] = input_t(out_val)`). The Triton kernel does the same (loads, fp32 MAC, fp32
SiLU, store). So accuracy is dominated by the **single output rounding**, not by accumulation order —
there are at most 4 products, so no catastrophic cancellation and no long-reduction drift.

## Parity vs reference
The oracle is `F.conv1d(x, W.unsqueeze(1), bias, padding=width-1, groups=dim)[..., :seqlen]` then SiLU,
all in `weight.dtype` then cast back (aiter `causal_conv1d_ref`). Because both the kernel and a sane
reference accumulate in fp32 over the same ≤4 taps, parity is **tight**:
- bf16 in/out: expect `atol≈2e-2, rtol≈1e-2` (bf16's 8-bit mantissa is the floor; the conv adds almost
  no extra error).
- fp16: tighter (`atol≈1e-3`). fp32: near-exact.
Use a relative-error / allclose gate at these bands; this is a **same-math** kernel (no quant), so byte
parity is not expected but the tolerance is small and stable.

## State correctness — the real numerical risk is *indexing*, not arithmetic
The subtle bugs in causal_conv1d are **state-management**, not rounding:
- **Causal padding direction**: padding is `width-1` zeros on the **left only** (past), 0 on the right.
  Getting this backwards silently leaks future tokens → looks fine on a parity test with `initial_states=
  None` but corrupts autoregressive decode. The kernel encodes this in the sliding-window init
  (`x_vals` seeded from the last `width-1` state entries) and the left-shift after each step.
- **conv_state shift/circular update**: decode updates `conv_state` **in place** — either linear
  shift-left (non-circular) or modular index (`cache_seqlens % state_len`, circular). An off-by-one in
  `update_idx = cache_seqlen - (width-1)` produces a *plausible-looking* but wrong window. Always verify
  multi-step decode against a rolling reference, not just one step.
- **Continuous batching / pad slots**: `conv_state_indices[i] == pad_slot_id` ⇒ the kernel **skips** that
  row (early return). If the output buffer is `empty_like` rather than `zeros_like`, padded rows contain
  garbage (the docstring explicitly warns: initialize `out` with `torch.zeros_like` when using padding).
- **`state_len ≥ width-1`** is asserted; a smaller state silently can't hold the causal context.

## dtype support
fp16, bf16, fp32 for both x and weight (independently dispatched in the HIP kernel:
`DISPATCH_ITYPE_*` × `DISPATCH_WTYPE_*`). **No fp8/quant path** — this op is too cheap and too
accuracy-sensitive (it feeds the recurrent state) to quantize; quantization wins live in the
neighboring GEMMs, not here. Widths **2, 3, 4 only** (Mamba `d_conv=4` and below); other widths are
not compiled.

## Tie-breaks / determinism
No reduction across blocks, no atomics → the op is **deterministic** and order-independent (each output
is one channel's ≤4-tap dot). No argmax/tie-break concerns. The only nondeterminism risk is if a caller
aliases `conv_state` across concurrent streams — the in-place update is not guarded.

## Verify
```python
# parity (prefill), aiter op_tests harness style:
y = causal_conv1d_fn(x, W, bias, conv_states, qsl, seq_lens, activation="silu")
y_ref = causal_conv1d_ref(x_bcl, W, bias, activation="silu")   # F.conv1d groups=dim
torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=1e-2)     # bf16 band
# decode multi-step: run N update() steps, compare to a rolling F.conv1d over the same N tokens.
```

## Sources
- fp32 MAC + fp32 SiLU + single output rounding, width 2/3/4, fp16/bf16/fp32 dispatch: `ROCm/aiter@a6bb49937:csrc/kernels/causal_conv1d_update.cu`.
- Reference (`F.conv1d` groups=dim, padding=width-1) + parity harness: `ROCm/aiter@a6bb49937:op_tests/triton_tests/test_causal_conv1d.py`.
- `zeros_like` output for pad slots, `state_len ≥ width-1`, circular vs linear state: `ROCm/aiter@a6bb49937:aiter/ops/causal_conv1d.py` (docstring) + `causal_conv1d_update.cu`.
- Causal (left-pad) depthwise definition: https://github.com/Dao-AILab/causal-conv1d
