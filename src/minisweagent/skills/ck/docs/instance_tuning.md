# Tuning the CK instance

The shipped CK ops do not contain a single kernel — they dispatch to a
**codegen-generated set of instances** and pick the best one for a shape from a
**tuned CSV** at call time. So "tuning a CK op" means: add your shape, run the
on-box codegen tuner to sweep instances, and let it write the winning instance
into the CSV the op reads.

## How dispatch works (so you know what tuning changes)

For A8W8 dense GEMM, `aiter.gemm_a8w8_CK` calls
`get_CKGEMM_config((cu_num, padded_M, N, K))`
(`aiter/ops/gemm_op_a8w8.py:278`) which looks up `a8w8_tuned_gemm.csv`. If the
shape is present it uses the tuned `kernelName`/`kernelId`; if not, it warns and
falls back to a default. MoE 2-stage uses the same idea via
`aiter/configs/tuned_fmoe.csv` plus a heuristic dispatch. **Therefore the lever
is: get your (M,N,K)/expert shape into the tuned CSV with the best instance.**

## On-box tuners

(aiter codegen lives under
`/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/`; in a developed aiter
checkout these are `csrc/...`.)

| op class | tuner | untuned CSV | tuned CSV |
|---|---|---|---|
| MoE 2-stage | `ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` | `aiter/configs/untuned_fmoe.csv` | `aiter/configs/tuned_fmoe.csv` |
| fp8 dense GEMM | `ck_gemm_a8w8/gemm_a8w8_tune.py` | `aiter/configs/a8w8_untuned_gemm.csv` | `aiter/configs/a8w8_tuned_gemm.csv` |
| batched bf16 GEMM | `ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` | `aiter/configs/bf16_untuned_batched_gemm.csv` | `aiter/configs/bf16_tuned_batched_gemm.csv` |

Each tuner dir also has `gen_instances.py` (emits the per-instance source files)
and a common module describing the instance set.

## MoE 2-stage tuning workflow

1. Add the shape to `aiter/configs/untuned_fmoe.csv` — columns:
   `token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a,
   q_dtype_w, q_type, use_g1u1, doweight_stage1`. Example row:
   `1024,4096,14336,8,2,ActivationType.Silu,dtypes.bf16,dtypes.fp8,dtypes.fp8,QuantType.per_Token,True,True`
2. Run the tuner (builds instances via JIT, takes minutes):
   ```bash
   python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
       -i aiter/configs/untuned_fmoe.csv \
       -o aiter/configs/tuned_fmoe.csv
   ```
   Useful flags: `--last` (only tune newly added shapes), `-k/--splitK` (enable
   split-K), `--errRatio 0.01` (tighten parity), `--mp N` (parallel GPUs),
   `-o2 profile.csv` (dump all candidates, not just the winner), `--iters`,
   `--warmup`, `-v`.
3. The tuned CSV gets per-shape winners with `block_m`, `ksplit`, `us1`,
   `kernelName1`, `us2`, `kernelName2`, `run_1stage`, `tflops`, `bw`.
4. Test: `python3 op_tests/test_moe_2stage.py` (use `AITER_REBUILD=1` if kernels
   were built before adding the shape).

Notes (from the codegen README): only G1U1 (gate+up fused) MoE is tunable; quant
types supported are `per_Token`, `per_1x128` (blockscale), `per_1x32` (mxfp4,
gfx950 only).

## Dense GEMM tuning workflow (A8W8 example)

1. Add `M,N,K` rows to `aiter/configs/a8w8_untuned_gemm.csv`.
2. Run:
   ```bash
   python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py \
       -i aiter/configs/a8w8_untuned_gemm.csv \
       -o aiter/configs/a8w8_tuned_gemm.csv [-k]
   ```
   Output columns: `cu_num, M, N, K, kernelId, splitK, us, kernelName, tflops,
   bw, errRatio`.
3. Test via `op_tests/test_gemm_a8w8.py` (`AITER_REBUILD=1` to rebuild from the
   tuned CSV). Batched bf16 uses `batched_gemm_bf16_tune.py` with `B,M,N,K`.

## The instance knob surface (what the sweep varies)

The MoE 2-stage instance is a `kernelInstanceGEMM1` / `kernelInstanceGEMM2`
dataclass (`gemm_moe_ck2stages_common.py`). The fields = the knobs:

| knob | field | meaning |
|---|---|---|
| block size | `BLOCK_SIZE` | threads per block (e.g. 128, 256) |
| tile M/N/K | `MPerBlock`, `NPerBlock`, `KPerBlock` | the per-block tile shape — the dominant perf knob |
| warp layout | `MWaves`, `NWaves` | warps along M / N |
| pipeline | `GemmPipelineVersion` | `1` (basic) vs `3` (compute V3) — the memory/compute schedule |
| swizzle | `Nswizzle` | N-dim swizzle for store coalescing |
| topk-weight fuse | `MulRoutedWeight` | fuse topk-weight multiply into the GEMM |
| activation | `ActOP` | silu vs gelu fused in epilogue |
| epilogue op | `CDEElementOp` | e.g. `TypeCast` |
| quant | `QuantType` | per_Tensor / per_Token / per_1x128 / per_1x32 |
| dtypes | `Adtype`, `Bdtype`, `Cdtype` | A/B/C element types |

The candidate lists are keyed by dtype and arch, e.g. `a16w16_gemm1_kernels_list`
(bf16/fp16), `a8w8_gemm1_kernels_list` (fp8/int8), plus `*_gfx950` variants. A
representative fp8 stage-1 instance set varies tile shapes such as
`32x64x256`, `64x64x128`, `128x128x128`, `256x128x128` with pipeline v1 vs v3.
To add a custom instance, extend the kernel list and re-run `gen_instances.py`;
to just pick the best existing one for your shape, the tuner does it for you.

## Picking the right instance for a shape regime

- **small M (decode / few tokens)** → smaller `MPerBlock` (16/32), more N tiling;
  often split-K helps when K is large and M,N small.
- **large M (prefill / many tokens)** → larger `MPerBlock` (128/256), pipeline v3.
- **block-scale fp8** → use the `per_1x128` quant instances; weight is usually
  preshuffled.
- Don't hand-pick blindly — run the tuner with `-o2` to see all candidates, then
  read off the winner for your `(M,N,K)`/experts.
