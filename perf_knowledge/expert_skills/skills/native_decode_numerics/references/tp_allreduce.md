# TP all-reduce: native numerics facts

Operators: `allreduce` (the tensor-parallel sum of the per-rank bf16 partials of a row-parallel linear) and
`fused_allreduce_rmsnorm` (aiter's fused all-reduce + residual add + RMSNorm). Backends: the vLLM ROCm
communicator of the `tp` group, that is aiter custom all-reduce, vLLM custom all-reduce and RCCL through PyNccl.
Not here: SGLang's communicators (`sglang_backends.md`), the unfused RMSNorm itself (`norm_and_quant.md`),
the QuickReduce codecs and the batch-invariant path (neither is modelled). A task that runs on one rank has no
all-reduce: skip this file.

**Versions.** Line numbers are valid at vLLM commit `f09d5e30c`, aiter tag `v0.1.26` (rows marked
`@285579b6` were read at that aiter commit) and, for SGLang, commit `22e1cb6d98`; in any other version
locate the code by symbol.

**Status.** `capture` = verified on a model capture (a captured decode step of a served model was reproduced
with this fact; "2 models" when two served models agree). `probe` = verified by a real-op probe (the installed
native op was run on synthetic inputs and compared with an emulation of this fact, usually bit for bit).
`source only` = read from source, never run against the native op: treat it as a suspect and confirm it with
the flip test in `parity_method.md` before relying on it.

In this file no fact rests on a model capture. Every `probe` ran the installed communicators on **two ranks**
of gfx942 with synthetic bf16 partials and compared bit for bit: five message sizes from 1 to 128 tokens of
hidden 4096 for the plain all-reduce (at two ranks that exercises aiter's LL, LL128 and 1-stage kernels and
vLLM's 1-stage kernel), and 1, 8 and 32 tokens with eps 1e-6 for the fused kernel. Nothing was measured on
more than two ranks; summation orders for four and eight ranks were simulated on CPU only.

## Which backend is running

Read these from the process that produced the task's oracle (its recorded environment and launch command, or
the server log kept with it), not from the machine you happen to work on. The number of ranks is
`tensor_parallel_size`; the message size is rows x hidden x 2 bytes for bf16.

| Backend / provider | Selected when | Where to read it |
|---|---|---|
| aiter custom all-reduce (`AiterCustomAllreduce`) | `tp` group on gfx94x / gfx95x, custom all-reduce not disabled, `VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_CUSTOM_AR` on, not batch-invariant, message inside the size gate (TPA-04, TPA-05) | process environment (the first flag defaults to off, the second to on); server log line `Using [...] all-reduce backends (in dispatch order) for group 'tp:...'` |
| vLLM custom all-reduce (`CustomAllreduce`) | same platform conditions with aiter off (the two are exclusive: vLLM's is built only when aiter's is absent); message under 8 MiB | same log line; `--disable-custom-all-reduce` in the launch command turns both custom kernels off |
| RCCL through PyNccl | whatever the custom kernels refuse: custom all-reduce disabled, more than two ranks that are not fully connected, message over the size gate, byte size not a multiple of 16 | same log line (last entry); launch command |
| QuickReduce | only when `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` is not `NONE` (default `NONE`) and the message reaches its minimum size (TPA-06) | process environment; a decode message is a few KiB per token, far below the default minimum |
| batch-invariant path | `VLLM_BATCH_INVARIANT` set | process environment; another kernel, not covered here |
| aiter fused all-reduce + RMSNorm | compiled graph only: optimization level 2 or 3 (2 is the default), aiter on, more than one rank, not batch-invariant, aiter custom all-reduce present, norm eps 1e-5 or 1e-6 (TPA-14, TPA-15); never under `--enforce-eager` | launch command (`--enforce-eager`, the optimization level, `compilation_config.pass_config.fuse_allreduce_rms`); at debug log level the fusion pass prints how many patterns it replaced |

## Input of the reduce

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| TPA-01 | Each rank's partial is already rounded to the layer dtype (bf16) before the reduce: the row-parallel GEMM stores per rank, then the all-reduce sums the stored values. The native result is not the rounded exact sum of unrounded partials. | vLLM · `RowParallelLinear` then `tensor_model_parallel_all_reduce` · bf16 | source only | not quantified for the kernel; a tolerance derived from the exact sum of unrounded partials was about 2x too tight for the native itself | `vllm/model_executor/layers/linear.py:1810-1820` `RowParallelLinear.forward` |
| TPA-02 | A row-parallel linear adds its bias on rank 0 only, before the all-reduce. | vLLM · `RowParallelLinear` with a bias | source only | bias on every rank: about 5.5x to 6.7x the stage tolerance (negative control of a reference built from this rule, not a native run) | `vllm/model_executor/layers/linear.py:1814` |

## Dispatch and size gates

A wrong answer in TPA-03 to TPA-06 selects the wrong store rounding (TPA-09 to TPA-12), so those rows carry no
figure of their own.

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| TPA-03 | Dispatch order of one call: NCCL symmetric memory (`VLLM_USE_NCCL_SYMM_MEM`, default off), QuickReduce, the FlashInfer paths (CUDA only, absent on ROCm), aiter custom, vLLM custom, the batch-invariant path, torch symmetric memory (CUDA only), then PyNccl (RCCL). The first whose size test admits the message runs. | vLLM · ROCm · `tp` group | source only | see TPA-09 | `vllm/distributed/device_communicators/cuda_communicator.py:362-446` `CudaCommunicator.all_reduce`; `:56-88`, `:150-197` `__init__` |
| TPA-04 | The custom kernels exist only for the `tp` group, on gfx94x / gfx95x, without `--disable-custom-all-reduce`. aiter's replaces vLLM's when `VLLM_ROCM_USE_AITER` (default off) and `VLLM_ROCM_USE_AITER_CUSTOM_AR` (default on) are set and the run is not batch-invariant. | vLLM · ROCm · gfx942 / gfx950 | source only | see TPA-09 | `vllm/platforms/rocm.py:1271-1273` `use_custom_allreduce`; `vllm/config/parallel.py:1167-1168`; `vllm/envs.py:1271-1278` |
| TPA-05 | Size gates. aiter custom: 0 < bytes <= 64 MiB (`AITER_CUSTOM_AR_MIN_SIZE`, `AITER_CUSTOM_AR_MAX_SIZE`), bytes a multiple of 16, weakly contiguous input, two ranks or a fully connected mesh. vLLM custom: bytes < 8 MiB under the same conditions. Everything else goes to RCCL. | vLLM · aiter custom / vLLM custom | source only | see TPA-09 | `vllm/distributed/device_communicators/custom_all_reduce.py:496-511` `should_custom_ar`; `aiter: aiter/dist/device_communicators/custom_all_reduce.py:1607-1637` `_fits_custom_ar_size`, `:170-186` |
| TPA-06 | QuickReduce is off unless `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` is not `NONE`. When on it takes only messages at or above a minimum per (dtype, rank count, codec) between 1 MiB and 2 GiB (`VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB` can lower it), and it quantises the message. | vLLM · QuickReduce · bf16 / fp16 · 2, 4 or 8 ranks | source only | not quantified (lossy codec, not modelled) | `vllm/distributed/device_communicators/quick_all_reduce.py:349-367` `should_quick_allreduce`; `:46-63`, `:186-192`; `vllm/envs.py:1412-1416` |
| TPA-07 | aiter custom picks its kernel by rank count and element count of a 2-byte dtype. Two ranks: LL up to 32768 elements, LL128 up to 458752, then 1-stage. Four ranks: LL up to 28672, LL128 up to 32768, then 1-stage below 160 KiB, else 2-stage. Eight ranks: LL up to 32768, then 1-stage below 80 KiB, else 2-stage. With hidden 4096 a decode step runs LL up to 7 tokens at four ranks and up to 8 tokens at two or eight ranks. | vLLM · aiter custom (the code path vLLM calls) · bf16 / fp16 | source only | changes only the summation order (TPA-13) | `aiter: csrc/include/custom_all_reduce.cuh:4514-4711` `CustomAllreduce::allreduce` |
| TPA-08 | vLLM custom picks 1-stage for two ranks always; on a fully connected mesh 1-stage below 512 KiB for up to four ranks and below 256 KiB for up to eight, else 2-stage. `VLLM_CUSTOM_ALLREDUCE_ALGO` overrides the choice. (The two-rank probe of TPA-11 always ran the 1-stage kernel.) | vLLM · vLLM custom | source only | changes only the summation order (TPA-13) | `csrc/custom_all_reduce.cuh:271-310` `CustomAllreduce::allreduce` |

## Summation dtype, order and store rounding

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| TPA-09 | aiter custom accumulates the stored partials in fp32 and casts once; on gfx94x that cast **truncates**: output = `truncate_bf16(fp32 sum)`. The library's default fp32 -> bf16 conversion is truncation there, and the all-reduce module is built without overriding it (the RMSNorm modules select round-to-nearest). The fused all-reduce + RMSNorm kernels sit in the same module and truncate too. | vLLM · aiter custom, aiter fused all-reduce + RMSNorm · bf16 · gfx942 | probe | a round-to-nearest store matches only about 73% of elements (the rest one bf16 ulp off); a bound must allow a whole ulp, biased toward zero, not half | `aiter: csrc/include/opus/opus.hpp:40-42`, `:1218-1229` `fp32_to_bf16`; `aiter: csrc/include/custom_all_reduce.cuh:75-90` `downcast_s`; `aiter: aiter/jit/optCompilerConfig.json:142-154` |
| TPA-10 | On gfx950 the same cast compiles to the hardware conversion, which rounds to nearest even: there the aiter all-reduce output is `round_nearest_even_bf16(fp32 sum)`. | vLLM · aiter custom · bf16 · gfx950 | source only | as TPA-09, mirrored | `aiter: csrc/include/opus/opus.hpp:1218-1229` `fp32_to_bf16` |
| TPA-11 | vLLM's custom kernel widens each partial to fp32, adds in pointer order and rounds once to nearest even: the output is `round_nearest_even_bf16(fp32 sum)`. | vLLM · vLLM custom (aiter off) · bf16 · gfx942 | probe | a truncating store disagrees on about a quarter of elements (TPA-09) | `csrc/custom_collective_common.cuh:292-298` `packed_reduce`, `:112-115` `downcast_s`; `csrc/custom_all_reduce.cuh:7-23` `cross_device_reduce_1stage` |
| TPA-12 | RCCL (`ncclAllReduce`, sum) on two ranks gives `round_nearest_even_bf16(fp32 sum)`. | vLLM · PyNccl / RCCL · bf16 · two ranks · gfx942 | probe | as TPA-11; more than two ranks unknown | `vllm/distributed/device_communicators/pynccl.py:273-300` `all_reduce` |
| TPA-13 | Summation order by kernel. aiter LL / LL128: own partial first, then rank + 1, rank + 2, ... modulo the rank count, so with more than two ranks every rank adds in another order and ranks need not agree bit for bit. 1-stage kernels (aiter and vLLM): rank order 0..n-1 on every rank. 2-stage kernels: each 1/n slice starts from its owning rank. Two ranks: one fp32 add, no order. | vLLM · aiter custom / vLLM custom · bf16 | source only (the insensitivity is a CPU simulation, not a measurement) | need not be matched: an exact-arithmetic simulation of 4 or 8 partials gave the same bf16 output in every order for exponent spreads up to 2^8 (both store roundings); at spreads of 2^12 to 2^20, 0.003% to 0.15% of elements changed | `aiter: csrc/include/custom_all_reduce.cuh:3509-3614` `ar_ll_gfx9`, `:376-581` `cross_device_reduce_1stage / _2stage`; `csrc/custom_all_reduce.cuh:7-80` |

## Fused all-reduce + RMSNorm

| id | Fact | Applies to | Status | Effect if wrong | Source |
|---|---|---|---|---|---|
| TPA-14 | The fusion pass is on by default in compiled deployments: optimization level 2 is the default, and at levels 2 and 3 `fuse_allreduce_rms` defaults to "aiter enabled, more than one rank, not batch-invariant". With `--enforce-eager` no pass runs and the two ops stay separate. | vLLM · compiled graph · aiter on · more than one rank | source only | decides between the unfused rules (TPA-09) and the fused ones (TPA-17) | `vllm/config/vllm.py:179-203` `enable_allreduce_rms_fusion`, `:297-323`, `:435-438`; `vllm/compilation/passes/pass_manager.py:167-171` |
| TPA-15 | What the pass replaces: all-reduce followed by RMSNorm, and all-reduce followed by add + RMSNorm (also the norm variant that scales by 1 + weight, and the forms with a per-group fp8 quant), for eps 1e-5 or 1e-6, within the compiled token range. It needs the aiter custom all-reduce. | vLLM · compiled graph · aiter on · more than one rank | source only | as TPA-14 | `vllm/compilation/passes/fusion/allreduce_rms_fusion.py:1570-1721` `RocmAiterAllReduceFusionPass`, `:1151-1260` |
| TPA-16 | Which fused kernel: 1-stage for at most 80 tokens, hidden a multiple of 8 and at most 8192, and (two ranks, or a fully connected mesh with a message under 256 KiB for up to four ranks or under 128 KiB for up to eight). Otherwise QuickReduce-fused if QuickReduce takes the message, else the 2-stage fused kernel. With hidden 4096 the 1-stage kernel covers up to 31 tokens at four ranks and up to 15 at eight. | vLLM · aiter fused all-reduce + RMSNorm · bf16 / fp16 | source only | selects TPA-17 or TPA-19 | `vllm/distributed/device_communicators/aiter_custom_all_reduce.py:90-122` `use_1stage_fused_ar_rms`; `vllm/_aiter_ops.py:1191-1268` `_rocm_aiter_fused_allreduce_rmsnorm_impl` |
| TPA-17 | 1-stage fused arithmetic: `s` = fp32 sum of the partials in rank order; `a = fp32(cast(s)) + fp32(residual)` (the sum is rounded to bf16 first, then widened again); `residual_out = cast(a)`; the norm reads the **unrounded fp32** `a`: `out = cast(a * rsqrt(mean(a^2) + eps) * w)`, one rounding (`1 + w` in the scale-by-1-plus-weight variant). `cast` truncates on gfx942 (TPA-09). | vLLM · aiter fused 1-stage · bf16 · gfx942 | probe (residual and output both bit for bit) | an unfused round-to-nearest emulation reproduces only about 60% of the residual elements and 45% of the normed outputs | `aiter: csrc/include/custom_all_reduce.cuh:1973-2090` `allreduce_fusion_kernel_1stage`, `:1660-1686` `ar_fusion_epilogue_rms_norm`, `:1829-1848` `ar_fusion_epilogue` |
| TPA-18 | Against the unfused all-reduce followed by add + RMSNorm, the fused 1-stage kernel differs in three places: the rounding mode of the residual and output stores (truncation, where the separate norm kernel rounds to nearest), the norm input (fp32 `a` instead of the stored bf16 residual) and the number of roundings on the way to the normed output (one instead of two). | vLLM · aiter fused 1-stage versus unfused · bf16 · gfx942 | probe (same run as TPA-17) | as TPA-17 | as TPA-17 |
| TPA-19 | 2-stage fused arithmetic: reduce-scatter in fp32 (rank order within each pack), cast to bf16, all-gather; then the residual is added **in bf16 arithmetic**, `residual_out` is stored and the norm runs on that stored bf16 residual. | vLLM · aiter fused 2-stage · bf16 | source only | not quantified | `aiter: csrc/include/custom_all_reduce.cuh:2836-2894` (body of `allreduce_fusion_kernel_2stage`) |

The 1-stage fused rule as arithmetic (written for this note; `cast` as in TPA-09):

```
s            = sum over ranks, in fp32, of the stored bf16 partials
a            = fp32(cast(s)) + fp32(residual)
residual_out = cast(a)
out          = cast(a * rsqrt(mean(a * a) + eps) * w)     # norm on fp32 a, one rounding
```

## What a candidate must reproduce at a TP boundary

A whole-layer kernel on more than one rank crosses this boundary once or twice per layer (after the attention
output projection and after the MLP / MoE). Pick the row by what the oracle ran, not by what is fastest.

| The oracle ran | Rule to reproduce | Facts |
|---|---|---|
| any backend | round each rank's partial to bf16 before summing (never sum fp32 partials); add a bias on rank 0 only | TPA-01, TPA-02 |
| aiter custom all-reduce, gfx942, unfused | accumulate in fp32, store `truncate_bf16(sum)` | TPA-09 |
| aiter custom all-reduce, gfx950 | accumulate in fp32, store round-to-nearest-even (unverified) | TPA-10 |
| vLLM custom all-reduce, or RCCL on two ranks | accumulate in fp32, store round-to-nearest-even | TPA-11, TPA-12 |
| RCCL on more than two ranks | unknown: neither read nor measured; decide it with the flip test | none |
| aiter fused 1-stage, gfx942 | truncate the sum to bf16, widen, add the residual in fp32, store the truncated residual, norm on the unrounded fp32 value with one truncating store | TPA-17, TPA-18 |
| aiter fused 2-stage | add the residual in bf16, norm on the stored bf16 residual (unverified) | TPA-19 |
| any | the summation order need not be matched | TPA-13 |

An oracle captured eagerly (`--enforce-eager`) holds the all-reduce and the norm as separate ops even when the
compiled deployment of the same model fuses them. Match the regime that produced the oracle. The difference
between the two regimes is real (TPA-18) and is not the candidate's to remove.

## Not covered by any check

- More than two ranks, on every backend: no bit-for-bit measurement. The four- and eight-rank orders are read
  from source plus a CPU simulation; RCCL's reduction order and per-hop rounding beyond two ranks were neither
  read nor measured.
- gfx950: the round-to-nearest hardware conversion in aiter (TPA-10) was read, not run.
- No model capture: every probe used synthetic bf16 partials.
- The fused probe ran eps 1e-6 only, up to 32 tokens of hidden 4096. Not run: eps 1e-5, the scale-by-1-plus-weight
  norm variant, the forms with a per-group fp8 quant, the 2-stage fused kernel, the QuickReduce-fused kernel.
- Whether the fusion pass matched a given model's graph is a per-deployment observation (the pass reports its
  replacement count at debug log level); the default-on rule (TPA-14) only says the pass was offered.
- QuickReduce codecs (integer and floating codecs, optional bf16 to fp16 cast) and the batch-invariant path.
- Partials in a dtype other than bf16 (fp16, fp32).
- The 2-stage kernels of both custom all-reduces and their size thresholds (TPA-07, TPA-08).
