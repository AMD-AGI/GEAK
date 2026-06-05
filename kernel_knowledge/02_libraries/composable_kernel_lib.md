# Composable Kernel as a consumed LIBRARY — GEMM / Grouped-GEMM / FMHA / MoE on MI300X (gfx942)

> AMD-only. This file treats Composable Kernel (CK, `ROCm/composable_kernel`) as a **library you consume** — picking and wiring prebuilt instances — *not* as the CK programming/DSL language. For writing CK/CK-Tile kernels, see the languages section. Target: gfx942 (MI300X/MI325X, CDNA3, native FP8-FNUZ). gfx950 (MI350/MI355) adds OCP FP8 + FP4/MXFP4.
>
> Repo note: CK is moving to `ROCm/rocm-libraries` and becoming **header-only** in an upcoming major ROCm release — `ckProfiler` and static libs will no longer be packaged/built by default but `ckProfiler` can still be built standalone. The instance/factory concepts below are unchanged.

CK is the broad, robust instance library underneath much of the AMD stack: **aiter, vLLM, and SGLang call CK** for GEMM, grouped-GEMM, FMHA (Flash Attention 2), and fused MoE. CK ships thousands of *instances* (concrete tile/pipeline configurations); the consumer's job is to (1) confirm an instance covers the target shape/dtype and (2) pick the fastest one. `ckProfiler` is the tool for both.

---

## 1. The CK mental model: instances & the instance factory

| Concept | Meaning |
|---|---|
| **Device op** | A templated kernel family, e.g. `ck::tensor_operation::device::DeviceGemm<ALayout,BLayout,CLayout,ADataType,BDataType,CDataType,AElementwiseOp,BElementwiseOp,CElementwiseOp>` |
| **Instance** | One fully-specialized device op: fixed tile (M/N/K block), MFMA instruction, pipeline (num stages), scheduler, vector widths. |
| **Instance factory** | Prebuilt lists of instances per op, queried at runtime: `add_device_gemm_xdl_..._instances(...)`. Headers like `grouped_gemm_tile_loop_multiply.hpp` expose them. |
| **`GetWorkSpaceSize` / `MakeArgument` / `MakeInvoker`** | Per-instance: size the scratch, build the argument, get a runnable invoker. |
| **`IsSupportedArgument`** | Per-instance predicate — returns false if the instance does **not** cover this shape/stride/dtype. This is the gate behind the infamous "device_gemm does not support this GEMM problem" crash. |

Consumer pattern (pseudocode):

```cpp
// 1. Get all instances for this op/dtype/layout from the factory
auto ops = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemm<...>>::GetInstances();

float best_ms = 1e30; int best = -1;
for (int i = 0; i < ops.size(); ++i) {
  auto arg     = ops[i]->MakeArgumentPointer(A, B, C, M, N, K, sA, sB, sC, aop, bop, cop);
  if (!ops[i]->IsSupportedArgument(arg.get())) continue;   // skip uncovered instances
  auto invoker = ops[i]->MakeInvokerPointer();
  float ms = invoker->Run(arg.get(), StreamConfig{nullptr, /*time=*/true});
  if (ms < best_ms) { best_ms = ms; best = i; }
}
// 2. Ship best instance index for this shape
```

This loop is exactly what `ckProfiler` automates — use it to choose, then hardwire the winning instance (or let aiter/vLLM's CK wrapper do it).

---

## 2. Building CK / ckProfiler for gfx942 + FP8 (fast)

```bash
git clone https://github.com/ROCm/composable_kernel && cd composable_kernel
mkdir build && cd build
cmake \
  -D CMAKE_PREFIX_PATH=/opt/rocm \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -D CMAKE_BUILD_TYPE=Release \
  -D GPU_TARGETS="gfx942" \
  -D DTYPES="fp8;bf16;fp16" \
  -D CK_PROFILER_OP_FILTER="gemm_universal" \
  ..
make -j ckProfiler
```

Build-time knobs that drastically cut compile time and binary size:

| CMake var | Effect |
|---|---|
| `GPU_TARGETS="gfx942"` | build only MI300X arch (use `gfx942;gfx950` for both) |
| `DTYPES="fp8;bf16"` | build only these dtype instances (subset of `fp64;fp32;tf32;fp16;fp8;bf16;int8`) |
| `CK_PROFILER_OP_FILTER` | regex of ops to compile (e.g. `"^grouped_gemm$"`, `"gemm_universal"`) |
| `CK_PROFILER_INSTANCE_FILTER` | regex to restrict which instances compile |
| `CK_USE_FP8_ON_UNSUPPORTED_ARCH` | OFF on gfx942 (it has native FP8); only needed on gfx908/gfx90a for functional FP8 |

> gfx942 has native FP8-FNUZ MFMA + hardware FP8 conversion (`fmed3f` clipping, packed 2-at-a-time convert). Do **not** set the unsupported-arch flag on MI300X.

---

## 3. ckProfiler — pick the best instance

`ckProfiler <op> <args...>` sweeps all matching compiled instances, verifies (optional), times them, and prints the fastest with its config and TFLOPS/GB/s.

### 3.1 Standard GEMM

```
./bin/ckProfiler gemm <dtype> <layout> <verify> <init> <print> <repeat> M N K StrideA StrideB StrideC
# dtype: 0=fp32 1=fp16   layout: 0=NN 1=NT 2=TN 3=TT
./bin/ckProfiler gemm 1 1 1 1 0 5 3840 4096 4096 4096 4096 4096
```

### 3.2 Universal GEMM — the FP8 path on gfx942 (recommended)

```
./bin/ckProfiler gemm_universal <dtype> <layout> <verify> <init> <print> <time> \
                 M N K StrideA StrideB StrideC <splitK> [warmup] [iters] [rotbuf_MB]
```

`<dtype>` (arg2):
| Val | Meaning |
|---|---|
| 0 | fp32 |
| 1 | fp16 |
| 2 | bf16 |
| 3 | int8 |
| 4 | f8 @ f16 (FP8 inputs, fp16 compute) |
| 5 | f16 @ f8 |
| 6 | f16 → f8 |
| 7 | f8 → bf16, comp f8 (**FP8 in, FP8 compute, bf16 out** — common LLM path) |
| 8 | f16 @ i4 |
| 9 | bf16 @ i4 |

`<layout>` (arg3): `0: A[m,k]·B[k,n]` (NN), `1: A[m,k]·B[n,k]` (NT — typical weight layout), `2: A[k,m]·B[k,n]` (TN), `3: A[k,m]·B[n,k]` (TT).

```bash
# bf16 4096^3, NT, split-K=1, 1 warmup, 10 iters, no rotating buffer
./bin/ckProfiler gemm_universal 2 1 1 1 0 1 4096 4096 4096 4096 4096 4096 1 1 10 0

# FP8 (f8 in / f8 compute / bf16 out) 4096^3 NT, rotating buffer 256 MB to defeat L2 reuse
./bin/ckProfiler gemm_universal 7 1 1 1 0 1 4096 4096 4096 4096 4096 4096 1 5 50 256
```

`<splitK>` (arg14) splits K across workgroups (helps tall-skinny / small-M decode GEMMs). `rotbuf_MB` (arg17) rotates input buffers so cache reuse doesn't inflate numbers — use a value ≥ L2 (MI300X L2 is 32 MB; AID/Infinity Cache larger) for realistic results.

### 3.3 Batched GEMM (multi-D)

```
./bin/ckProfiler batched_gemm_multi_d <dtype> <layout> <verify> <init> <print> <time> \
   M N K StrideA StrideB StrideC BatchStrideA BatchStrideB BatchStrideC BatchCount
# dtype: 0=fp16 1=int8
./bin/ckProfiler batched_gemm_multi_d 0 1 0 0 0 1 4096 4096 4096 4096 4096 4096 16777216 16777216 16777216 16
```

### 3.4 Grouped GEMM / FMHA / MoE

- **grouped_gemm**: instances exposed via factory headers (`grouped_gemm_tile_loop_multiply.hpp`, plus `grouped_gemm_fastgelu`, `grouped_gemm_tile_loop` variants). The profiler README does not give a fixed-arg line for it; consume it through the instance factory loop (Section 1) or via aiter's MoE path which wraps CK grouped GEMM. Build only base op with `CK_PROFILER_OP_FILTER="^grouped_gemm$"`.
- **FMHA**: CK-Tile FMHA uses **codegen**, not a fixed-arg profiler op. The generator `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` emits thousands of instances (tile sizes × warp configs × mask/bias variants). Run/benchmark via the example binary + smoke scripts:
  ```bash
  # built under example/ck_tile/01_fmha
  ./example/ck_tile/01_fmha/script/smoke_test_fwd.sh
  # example binary flags: -b batch -h heads -s seqlen -d head_dim -mask 1 ...
  ```
  FP8 FMHA on gfx942: supports fp8 dynamic tensor-wise quantization of the fp8 fmha-fwd kernel, and FP8 KV-cache for batch prefill.
- **MoE (fused)**: CK ships fused MoE (sorting + grouped GEMM + activation). It is normally consumed through **aiter `fused_moe`**, which selects a CK MoE instance by quant method.

### 3.5 Reading output / verifying

`ckProfiler` prints, per best instance: the instance name (tile/pipeline config), elapsed ms, **TFLOPS**, and **GB/s**. For steady-state numbers CK warms with ~50 launches then averages ~50. For deep analysis of the chosen instance, profile it with **rocprofv3 / rocprof-compute**. (Some accumulating kernels — grouped_conv_bwd_weight, col2img — warn against using CK's timer + verification simultaneously.)

---

## 4. How vLLM / SGLang / aiter call CK

| Consumer | CK usage |
|---|---|
| **aiter** | CK is the default fallback backend for GEMM / grouped-GEMM / FMHA / MoE when no hand-tuned asm path matches. FlyDSL MoE falls back to CK when FlyDSL absent. `CK_BLOCK_GEMM=1`, `SGLANG_ROCM_AITER_BLOCK_MOE=1` route to CK block-scale paths. |
| **vLLM** | CK Flash Attention 2 is available alongside Triton FA; switch with `VLLM_USE_FLASH_ATTN_TRITON=False` (CK) vs default Triton. FP8 linear ultimately reaches CK or hipBLASLt depending on shape. |
| **SGLang** | Uses CK FMHA / block GEMM on ROCm; AITER MoE wraps CK grouped GEMM. `CK_BLOCK_GEMM=1` enables CK block GEMM. |

So in practice you rarely call CK's C++ API by hand for serving — you (a) ensure the CK build packaged with aiter/vLLM contains instances for your shapes, and (b) use `ckProfiler` to confirm/select when chasing a regression.

---

## 5. Instance selection, tuning, and the coverage gotcha

The single most common production failure with CK is **missing instance coverage**:

```
device_gemm with the specified compilation parameters does not support this GEMM problem
```

This means **no compiled CK instance** satisfies `IsSupportedArgument` for your (M,N,K, strides, dtype, layout). Causes & fixes:

| Cause | Fix |
|---|---|
| Built with restricted `DTYPES` / `OP_FILTER` and your dtype/op was filtered out | rebuild including the needed dtype/op |
| Shape has odd strides / alignment the instances don't cover | pad M/N/K to a covered multiple; pick a different layout |
| FP8 model whose specific GEMM shapes have no instance | run `ckProfiler gemm_universal 7 ...` against the exact shapes to confirm an instance exists; if not, generate/build one |
| Wrong arch built | ensure `GPU_TARGETS` includes `gfx942` |

**Tuning workflow (library consumer):**
1. Extract the real (M,N,K, layout, dtype) GEMM shapes from your model.
2. For each, run `ckProfiler gemm_universal <fp8_mode> <layout> 1 1 0 1 M N K ... <splitK> <warmup> <iters> <rotbuf>`.
3. Record the winning instance + its TFLOPS; if none is supported, that shape needs a new instance (build/codegen) or a hipBLASLt path instead.
4. Wire the winner: either select that instance index in your factory loop, or rely on the wrapper (aiter/vLLM) whose internal CK selection should now find it.
5. Re-verify after ROCm/CK upgrades — instance lists and best configs change.

---

## 6. FP8 / FP4 CK GEMM specifics

- gfx942: FP8 = **FNUZ** (E4M3FNUZ / E5M2FNUZ). Use `gemm_universal` modes 4–7; mode 7 (f8 in, f8 compute, bf16 out) is the standard quantized-linear path.
- Hardware-accelerated FP8 conversion on gfx94x: clipping via `fmed3f`, packed 2-element convert — instances exploit this in the load/convert stage.
- gfx950: adds OCP FP8 and FP4/MXFP4 + block scaling; build with `GPU_TARGETS="gfx950"` and the corresponding dtype instances.
- Use the **rotating buffer** (arg17) when benchmarking FP8 GEMM so the small FP8 footprint doesn't sit entirely in cache and overstate TFLOPS.

---

## Sources
- ROCm Composable Kernel repository (instances, examples, `profiler/`): https://github.com/ROCm/composable_kernel
- ckProfiler README (op argument tables, examples): https://github.com/ROCm/composable_kernel/blob/develop/profiler/README.md
- ROCm docs, "Optimizing with Composable Kernel": https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- CK `DeviceGemm` struct reference (template params / `IsSupportedArgument`): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1tensor__operation_1_1device_1_1_device_gemm.html
- AMD ROCm Blog, "Hands-On with CK-Tile: Develop and Run Optimized GEMM on AMD GPUs": https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
- SGLang issue #16025 ("device_gemm does not support this GEMM problem" — CK MoE coverage on MI300X): https://github.com/sgl-project/sglang/issues/16025
