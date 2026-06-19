# CK pitfalls

Common failure modes when rewriting/tuning a kernel into CK on MI300X (gfx942).

## Process pitfalls (most common cause of a bad rewrite)

- **Authoring before trying the shipped op.** A from-scratch CK rewrite that
  ignores an applicable shipped aiter CK op (`fused_moe`, `ck_moe_stage1/2`,
  `gemm_a8w8_CK`, `batched_gemm_bf16_CK`, ...) is a failure. Do STEP 0 first.
- **Writing raw CK template metaprogramming from a blank file.** CK is a codegen
  library — copy the nearest `example/ck_tile/` template, never start empty.
- **Python per-expert / per-batch loops.** One launch per logical op. Looping in
  Python over experts/batches defeats the point and is slow.
- **Guessing signatures.** Most aiter CK ops are `@compile_ops`-wrapped, so
  `inspect.signature` reports `(*args, **kwargs)`. Read the real def in the
  source module (e.g. `aiter/ops/moe_op.py`, `aiter/ops/gemm_op_a8w8.py`,
  `aiter/fused_moe.py`) instead of guessing arg order.

## Instance / dispatch mismatch

- **Shape not in the tuned CSV.** The shipped op looks up
  `(cu_num, padded_M, N, K)` in the tuned CSV and *silently falls back to a
  default* (it only warns). If perf is bad, check whether your shape is actually
  tuned — add it to the untuned CSV and run the tuner (see `instance_tuning.md`).
- **cu_num / arch mismatch.** Tuned CSVs are keyed by compute-unit count. A CSV
  tuned on a different GPU won't match; retune on the target box.
- **Stale build after retuning.** If kernels were built before you added a shape,
  rebuild with `AITER_REBUILD=1` (and clear `aiter/jit` `build`/`*.so` if you
  changed instances), or the old instance is used.
- **Pinning a wrong `kernelName`.** `ck_moe_stage1/2` accept an explicit
  `kernelName`; an instance whose tile/quant/dtype doesn't match the problem will
  fail to build or mis-dispatch. Leave it `None` to use the heuristic unless you
  have a tuner-verified name.

## Host (`.cu`/`.cpp`) vs device (`.hpp`) split

- ck_tile separates the **host dispatcher** (arg parse, grid/block, launch) in
  the `.cpp`/`.cu` from the **device kernel template** in the `.hpp`. Putting
  device template code in the host file (or vice versa) breaks the parallel
  codegen build and compile-time guarantees. Keep the kernel body in the `.hpp`.
- Instances are emitted one-per-file by `generate.py` / `gen_instances.py`; if you
  add an instance, regenerate rather than hand-editing the dispatch table.

## Block-scale / preshuffle layout

- **Preshuffle expected but not applied.** Ops with `is_shuffled=True` /
  `*_bpreshuffle` / weight-preshuffle pipelines expect B pre-shuffled via
  `aiter.shuffle_weight`. Passing un-shuffled weights gives wrong results.
- **Block-scale granularity mismatch.** `per_1x128` (blockscale) tiles weights
  into 128x128 blocks with a permute before quant; the scale tensor shape and
  layout must match the op's expectation. `per_1x32` (mxfp4) is gfx950-only.
- **Scale dtype / shape.** Activation/weight scales are typically fp32 with
  specific shapes (`w_scale` `[N,1]` or per-block, `a_scale` per-token). Wrong
  scale shape → silent garbage or a shape error.

## Numerics / parity

- **fp16 gate+up overflow in MoE.** gate+up in fp16 easily exceeds fp16 max
  (65504) → INF. Use **bf16** for gate+up (per the 15_fused_moe README).
- **fp8 / block-scale tolerance.** Quantized CK paths won't bit-match a bf16
  reference — validate within tolerance (e.g. `--errRatio` in the tuner, a
  relative tolerance in `save_and_test`), not for exact equality.
- **Atomic-accumulation MoE stage 2.** The 2nd MoE GEMM accumulates with atomics
  into a buffer that the sorting step zeroes. If you bypass the fused zeroing,
  zero the output yourself or you'll accumulate into garbage.
- **Activation / epilogue mismatch.** `activation` 0=gelu vs 1=silu, and
  `doweight_stage1` (apply topk-weight in stage 1 vs 2) must match the reference
  semantics or results drift.

## Validate-then-benchmark discipline

- Always `save_and_test` for parity **before** trusting a speedup number.
- A correct, simple CK wiring that beats baseline is better than a clever
  hand-authored ck_tile kernel that fails correctness.
- Don't create shims/mocks to make parity "pass" — that's a failure.
