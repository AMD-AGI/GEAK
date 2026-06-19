# Authoring a ck_tile program (last resort)

Only do this if STEP 0 (shipped op) and STEP 1 (instance tuning) don't win.
**Never write CK from a blank file.** Copy the closest `example/ck_tile/`
template and adapt it. Locate the on-box composable_kernel examples first:
```bash
python3 - <<'PY'
import pathlib
for root in ("/usr/local/lib/python3.12/dist-packages/aiter_meta",
             "/usr/local/lib/python3.12/dist-packages/aiter"):
    for p in pathlib.Path(root).rglob("example/ck_tile"):
        print(p); break
PY
# or: find / -type d -path '*composable_kernel/example/ck_tile' 2>/dev/null | head
```
(aiter vendors composable_kernel under `3rdparty/`/`csrc/`; the upstream repo
is github.com/ROCm/composable_kernel `example/ck_tile/` if you need to compare.)
The example layout below is captured here so you can author from it even if the
example tree isn't reachable on this box.

## Pick the nearest example

| target kernel | start from |
|---|---|
| dense GEMM (bf16/fp16/fp8/int8/int4-B) | `03_gemm` |
| batched GEMM | `16_batched_gemm` |
| grouped GEMM | `17_grouped_gemm` |
| preshuffled / flat GEMM | `18_flatmm`, `52_cshuffle_lds` |
| block-scale fp8 GEMM | `38_block_scale_gemm`, `42_mx_gemm` (mxfp4) |
| multi-D / multi-ABD GEMM (fused bias/elementwise) | `19_gemm_multi_d`, `22_gemm_multi_abd` |
| split-K / stream-K GEMM | `40_streamk_gemm`, `03_gemm` splitk variants |
| MoE / fused-MoE | `15_fused_moe` (+ `13_moe_sorting`, `09_topk_softmax`, `14_moe_smoothquant`) |
| attention | `01_fmha` |
| rmsnorm / layernorm | `10_rmsnorm2d`, `02_layernorm2d` |
| norm + residual + quant | `11_add_rmsnorm2d_rdquant`, `12_smoothquant` |
| reduce / transpose / permute | `05_reduce`, `37_transpose`, `06_permute` |

## Example layout (using 03_gemm)

A ck_tile example is split into **host launcher** (`.cpp`) and **device template**
(`.hpp`), with a config struct and an invoker. From `03_gemm`:

- `gemm_basic.cpp` — host entry: parses args (`-m -n -k -prec -a_layout ...`),
  selects `GemmConfig` + `Invoker`, dispatches on dtype
  (`run_gemm_example_prec_type<GemmConfig, Invoker, fp16/bf16/fp8/...>`), launches.
- `gemm_utils.hpp` — `GemmConfig*` structs (tile dims, warp layout, pipeline) and
  helpers. **This is where the tile config lives** — your main knob surface.
- `*_invoker.hpp` — `BasicInvoker` / `universal_gemm_invoker.hpp` /
  `gemm_weight_preshuffle_invoker.hpp`: builds the kernel args, computes
  grid/block, launches the device template.
- `run_gemm_example.inc`, `run_gemm_example_common.hpp` — shared run scaffolding.
- `CMakeLists.txt`, `script/cmake-ck-dev.sh` — build (targets like
  `tile_example_gemm_basic`, `tile_example_gemm_universal`,
  `tile_example_gemm_weight_preshuffle`).

Pipelines available in 03_gemm: **basic** (`gemm_basic`), **memory-bound /
universal** (`universal_gemm`), **weight-preshuffle** (best for inference GEMM,
bypasses LDS by pre-shuffling B in warp layout). Precisions: fp16/bf16/fp8/bf8/
int8/pk_int4 (B). Split-K is exposed via `-split_k` and dedicated splitk
variants (`gemm_splitk_two_stage.cpp`).

## The pieces you fill in (the ck_tile model)

When adapting, you are parameterizing these (see `overview.md`):

1. **Problem** — A/B/C dtypes, accumulator dtype, layouts (R/C-major),
   element-wise/epilogue ops.
2. **Config / Traits** — `MPerBlock`/`NPerBlock`/`KPerBlock`, warp tile dims,
   `MWaves`/`NWaves`, block size. Match these to your shape regime (see the tile
   guidance in `instance_tuning.md`).
3. **Pipeline** — pick basic vs compute-v3 vs preshuffle; swapping the pipeline
   type is the cheapest structural change.
4. **Epilogue** — fuse activation / type cast / quant / bias / topk-weight here.
5. **Host launcher** — keep arg parsing + grid/block + launch in the `.cpp`/`.cu`;
   keep the kernel body in the `.hpp` (parallel-build friendly, see `pitfalls.md`).

## MoE specifics (15_fused_moe)

The fused-MoE example fuses **moe-sorting → group-GEMM → activation → topk-weight
multiply → scatter** into a back-to-back 2-GEMM kernel:
- `fused_moesorting.hpp` rearranges tokens so each workgroup serves one expert
  (token-by-token → expert-by-expert), and **zeroes the output buffer** (the 2nd
  GEMM accumulates with atomics, so no separate `torch.zeros`).
- `fused_moegemm.hpp` is the group-GEMM; B (expert weights) is **pre-shuffled**
  for coalesced loads, activation stays `[tokens, hidden]`.
- Kernels are instantiated per-config under `instances/` via `generate.py`.
- WARNING (from the example): gate+up in fp16 easily overflows fp16 max (65504)
  and yields INF — use **bf16** for gate+up.

But before authoring MoE from this example, you should already have tried
`aiter.fused_moe` / `ck_moe_stage1/2` (STEP 0) and the 2-stage tuner (STEP 1).

## FMHA specifics (01_fmha)

Two template params drive the kernel (`fmha_fwd_kernel.hpp`): a `FmhaPipeline`
(swappable block-tile pipeline, the perf-critical piece) and an
`EpiloguePipeline` (store + post-fusion). Instances are codegen'd by
`generate.py` (see `FMHA_FWD_KERNEL_BODY`). Supports batch/group mode, GQA/MQA
(`-h_k`), variable seqlen.

## Build & validate

Each example builds with `script/cmake-ck-dev.sh <src> <arch>` (use `gfx942` for
MI300X) then `make tile_example_<target>` / `ninja`. Validate against CPU/GPU
reference with the example's `-v` flag, then port the adapted kernel into the
harness and confirm parity with `save_and_test` before benchmarking.
