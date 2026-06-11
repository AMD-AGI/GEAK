---
title: allreduce on HIP (custom 1-shot/2-shot) — SOTA card
kind: sota_card
operator: allreduce
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, int4]
regimes: [decode, prefill, both]
status: sota
updated: 2026-06-09
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/custom_all_reduce.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/include/custom_all_reduce.cuh
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/include/quick_all_reduce.cuh
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
  - https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html
  - https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-2/README.html
  - https://github.com/vllm-project/vllm/pull/19744
  - https://github.com/ROCm/aiter/issues/1542
---

# allreduce × HIP (custom all-reduce)

## TL;DR
> The framework custom all-reduce is **HIP**: aiter's `custom_all_reduce.cu` (1-shot/2-shot xGMI P2P via IPC
> memory) and `quick_all_reduce.cu` (quantized 2-shot AR). These beat RCCL for **small/decode** messages by
> reading peers' buffers directly over the fully-connected xGMI mesh with no RCCL channel setup. HIP is the
> seam to own the 1-shot/2-shot algorithm, the IPC buffer registration, the quantization level, **and** the
> fused all-reduce+RMSNorm(+quant) epilogue. These are the first two tiers of the **3-way adaptive dispatch**:
> Custom AllReduce (small, <~512KB–2MB) → **QuickReduce** (mid/large, **up to 3× vs RCCL**, MI300-only, FP4
> added on MI355) → RCCL (largest). QuickReduce lifts **TTFT not TPOT**. Above a message-size threshold,
> switch to 2-shot / RCCL ring.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter custom AR (1-shot / 2-shot, IPC P2P) | `csrc/kernels/custom_all_reduce.cu` + `csrc/include/custom_all_reduce.cuh`, `aiter/ops/custom_all_reduce.py` | gfx942/950; bf16/fp16 | beats RCCL on small msgs (vendor/community) | decode TP all-reduce, small messages |
| aiter Quick/quantized AR (2-shot) = **QuickReduce** | `csrc/kernels/quick_all_reduce.cu` + `csrc/include/quick_all_reduce.cuh`, `aiter/ops/quick_all_reduce.py` (`quant_level`) | **MI300-only**; fp8/int8/int6/int4 (FP4 on MI355) | **up to 3× vs RCCL** (up to 2.25× on 2×/4× MI300X) on mid/large msgs; crossover ~1MB @TP2 / ~4MB @TP8; lifts **TTFT not TPOT** (vLLM PR #19744) | mid/large-msg AR; prefill TP comm |
| aiter fused AR+RMSNorm(+quant) | `custom_all_reduce.cu` (`allreduce_fusion_kernel_1stage*`), `aiter/ops/custom_all_reduce.py` (`fused_allreduce_rmsnorm[_quant][_per_group]`) | gfx942/950 | saves a kernel launch + a HBM round-trip | TP residual+norm after AR (see [[fused_allreduce_rmsnorm]]) |

### The kernels (on-box `a6bb49937`)
`custom_all_reduce.cuh` defines, all `__launch_bounds__(512,1)`:
- `cross_device_reduce_1stage` — **1-shot**: every rank reads all peers' input over IPC and reduces locally.
  Lowest latency, small messages.
- `cross_device_reduce_2stage` / `_write_mode` — **2-shot**: reduce-scatter then all-gather. Better bandwidth,
  mid messages.
- `allreduce_fusion_kernel_1stage` / `_per_group` — 1-shot AR fused with RMSNorm (+ optional fp8 quant).
- Signaling via `Signal{ start[kMaxBlocks][8]; end[kMaxBlocks][8]; _flag[kMaxBlocks]; }` with
  `kMaxBlocks = 80` — the cross-block / cross-rank handshake.

`quick_all_reduce.cuh` is **two-shot only** (`TWOSHOT_DISPATCH`), dispatching on `QuickReduceQuantLevel`:
```cpp
enum QuickReduceQuantLevel { FP8 = 1, INT6 = 2, INT4 = 3 };  // (0 = no quant)
// CodecFP8 (kFP8Max = 240.0f on MI300X fnuz), CodecQ6, CodecQ4 — templated on world_size
case FP8:  TWOSHOT_DISPATCH(CodecFP8) break;
case INT6: TWOSHOT_DISPATCH(CodecQ6)  break;
case INT4: TWOSHOT_DISPATCH(CodecQ4)  break;
```

## Config space / knobs
- **Algorithm**: 1-shot (every rank reads all peers — lowest latency, small msgs) vs 2-shot (reduce-scatter +
  all-gather — better bandwidth, mid msgs). Threshold by message size in the launcher.
- **IPC buffers**: `init_custom_ar` registers the workgroup; `init_custom_qr` / `qr_open_handles` register peer
  buffers (`qr_get_handle`); `qr_max_size` caps the registered region. Eager mode (input not IPC-registered)
  uses a temp-buffer copy path; `reg_inp_ptr==0` means the input itself is IPC-registered (faster).
- **`quant_level`** (quick AR): `FP8`/`INT6`/`INT4` — perf vs accuracy. fnuz fp8 (kFP8Max=240) on gfx942.
- **Fusion**: `fused_allreduce_rmsnorm`, `fused_allreduce_rmsnorm_quant`, `_per_group` — fold the post-AR norm
  (and fp8 quant) into the AR kernel.
- **xGMI**: fully-connected mesh; one block per CU reading its peers; `-munsafe-fp-atomics` if atomic-accumulating.

## Numerics / parity
custom AR fp16/bf16 = near-exact vs RCCL (reduction order differs → benign bf16 deltas). Quick AR quantizes
the reduction (fp8/int6/int4) → **accuracy gate required**. fnuz fp8 on gfx942. Stability: aiter AR has
segfaulted (#1542) — keep a fallback. See [numerics.md](../numerics.md).

## Integration (rebind seam)
- SGLang: `SGLANG_USE_AITER_AR=1` (needs `SGLANG_USE_AITER=1`).
- vLLM: wires AITER `CustomAllreduce` into the communicator.
- Kernels JIT/AOT into `aiter/jit/` (`module_custom_all_reduce`, `module_quick_all_reduce`). To own a custom
  algorithm: edit the `.cu`/`.cuh` + rebuild the module.

## Pitfalls & anti-patterns
- IPC handle exchange must complete before first use (init order); a stale handle → wrong reads / garbage.
- Quantized AR without an eval = silent accuracy regression — always gate INT4/INT6/FP8 quant.
- aiter AR segfault (#1542) → `SGLANG_USE_AITER_AR=0` fallback to RCCL.
- 1-shot on large messages is bandwidth-bad (each rank reads N×) — switch to 2-shot / RCCL ring above the
  threshold.
- Quick AR is **2-shot only** — there is no 1-shot quantized path; for the smallest messages use the unquantized
  1-shot custom AR.
- Message > `qr_max_size` registered region → out-of-bounds; size the region to the largest AR.

## How to verify
Numeric vs fp32 RCCL reference (custom AR ≈ exact; quick AR within quant tol); soak/loop for stability (#1542);
rocprof to confirm `cross_device_reduce_*` (or `allreduce_fusion_kernel_*`) ran and overlaps the next GEMM;
e2e decode tok/s vs RCCL.

## Worked example (Llama-70B TP=8 decode, MI300X)
hidden 8192, bf16, M=1 token (decode) → tiny AR message.
1. `SGLANG_USE_AITER_AR=1`; small message → `cross_device_reduce_1stage` (1-shot) fires.
2. rocprof confirms the custom kernel (not RCCL ring), overlapping the next attention/GEMM.
3. If decode latency is dominated by the post-AR RMSNorm, switch to `fused_allreduce_rmsnorm` (1 kernel).
4. For an aggressive latency budget, try quick AR `quant_level=FP8`; gate with a small eval; fall back to
   unquantized 1-shot if accuracy slips. Above ~1–4 MB messages (prefill), let RCCL ring or 2-shot take over.

## Alternatives / cross-links
[[allreduce]] · [rccl.md](rccl.md) (RCCL default) · [vllm_kernels.md](vllm_kernels.md) (Quick Reduce) ·
[[fused_allreduce_rmsnorm]] · [[reduce_scatter]] · [[allgather]] ·
[`languages/hip_cpp/overview.md`](../../../languages/hip_cpp/overview.md) · [overview.md](../overview.md) ·
[numerics.md](../numerics.md).

## Sources
- on-box: `ROCm/aiter@a6bb49937:csrc/kernels/{custom_all_reduce,quick_all_reduce}.cu`,
  `csrc/include/{custom_all_reduce,quick_all_reduce}.cuh` (1stage/2stage kernels, `Signal`/`kMaxBlocks=80`,
  `QuickReduceQuantLevel{FP8,INT6,INT4}`, `CodecFP8`/`CodecQ6`/`CodecQ4`),
  `aiter/ops/{custom_all_reduce,quick_all_reduce}.py` (`init_custom_ar`, `all_reduce`,
  `fused_allreduce_rmsnorm[_quant][_per_group]`, `qr_all_reduce(quant_level)`).
- xGMI fully-connected mesh: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
- QuickReduce up to 3× vs RCCL, two-shot + INT4/INT6/INT8 compression, MI300-only, TTFT-only: https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html ; FP4 on MI355: https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-2/README.html ; vLLM PR #19744: https://github.com/vllm-project/vllm/pull/19744
- AR segfault: https://github.com/ROCm/aiter/issues/1542
