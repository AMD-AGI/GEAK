---
title: dense_gemm on hipBLASLt — SOTA card
kind: sota_card
operator: dense_gemm
backend: hipblaslt
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
  - ROCm/aiter@a6bb4993:aiter/tuned_gemm.py
---

# dense_gemm × hipBLASLt

## TL;DR
hipBLASLt (Tensile-generated `Cijk_*` kernels) is the **default BLAS** and the kernel actually executed for
dense bf16/fp16/fp8 GEMM on Instinct — directly under PyTorch, and indirectly **under aiter**
(`hipb_gemm → hipb_mm`). For raw-torch paths, tune it **offline** (dump shapes → `hipblaslt-bench` search →
override file). On sglang/vllm it runs under aiter's dispatcher, so the override file is **not consulted** —
deploy the win through aiter's DB instead ([[operators/dense_gemm/backends/aiter]]).

## SOTA implementation
aiter wraps hipBLASLt directly; the chosen solution index travels in the tuned CSV (`solidx`). From
`/sgl-workspace/aiter/aiter/tuned_gemm.py` (`ROCm/aiter@a6bb4993`):

```python
def hipb_gemm(inp, weights, solidx, bias=None, otype=None,
              scale_a=None, scale_b=None, scale_c=None, bpreshuffle=False):
    if not extensions_created:
        hipb_create_extension(); ...
    return hipb_mm(inp, weights.t(), solidx, bias, otype,
                   scale_a, scale_b, scale_c, bpreshuffle)
```

`solidx = -1` means "let hipBLASLt heuristically pick"; a tuned positive index pins a specific Tensile
solution. Offline (raw-torch) the same indices are discovered by `hipblaslt-bench`.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| hipBLASLt Tensile `Cijk_*` (tuned solution index) | ROCm/hipBLASLt | gfx908–950; bf16/fp16/fp8 | bf16 **708 TFLOPS** (M4096,N4864,K32896) ≈54% of 1307 theo. peak; fp8 **1223.6 TFLOPS**; fp16 654 TFLOPS @ MI300X, ROCm 6.3, AMD Labs Feb-2025 | default dense GEMM; any BLAS-dispatch framework |
| hipBLASLt FP8 (CDNA4 no-tune bar) | ROCm/hipBLASLt | gfx950; fp8 | **~2750 TFLOPS** @ M=N=K=4096; **~3130** @ 8192 (MI355X, ROCm 7.1) — the bar HIP/C++ kernels target | the default FP8 GEMM bar on MI355X |
| TensileLite custom logic (generate new instances) | ROCm/hipBLASLt TensileLite | gfx942/950 | shape-specific; closes gaps where pooled solutions are weak | a hot shape with no good pooled solution |

Note: ~80% *software* efficiency (vs clock-adjusted peak) but only **~45%** of *theoretical* peak sustained
on MI300X across fp8/bf16/fp16 — the gap is power/clock throttle under the 750 W cap (third-party,
narrowing), not just kernel quality (Ambati & Diep 2025). On **CDNA4 (MI355X)** the picture is far
healthier: the FP8 hipBLASLt bar is ~2750@4096 / ~3130@8192 TFLOPS, and it is now **beaten without
assembly** by a HIP/C++ 8-wave ping-pong kernel (3204 @ 8192) — see
[[operators/dense_gemm/backends/asm]] / [[operators/scaled_quant_gemm/tuning]] and the ping-pong
scheduling prior in [[optimization/mfma_scheduling]].

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| `TORCH_BLAS_PREFER_HIPBLASLT` | 0 / 1 | route torch `addmm`/`mm` to hipBLASLt vs rocBLAS | 1 (recent ROCm) |
| `HIPBLASLT_LOG_MASK` | bitmask (32 = bench) | dump GEMM shapes for offline tuning | 0 |
| `hipblaslt-bench --algo_method` | all / index / heuristic | search all solutions vs replay a chosen index | heuristic |
| `HIPBLASLT_TUNING_FILE` | path | write/read tuned solution map (raw-torch only) | unset |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | path | force the tuned solution at runtime (raw-torch only) | unset |
| `OPTIMIZE_EPILOGUE` | 0 / 1 | fuse epilogue (bias/act) into the GEMM store | 0 |
| `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` | bytes | bound per-solution workspace | impl-default |
| solution tiling | `mfma_16x16` vs `32x32` | prefer 16×16 on MI300X for most shapes | per-solution |

MI300X levers: prefer `mfma_16x16` solutions, set `OPTIMIZE_EPILOGUE=1`, avoid 512-B leading-dim strides
(Tagram bank hotspot — pad to non-512-multiple).

## Numerics / parity
Same bf16/fp8 math, different tiling/solution → parity-safe up to accumulation order (cross-solution argmax
flips possible but rare). fp8 paths need a downstream accuracy gate ([../numerics.md](../numerics.md)).

## Integration (rebind seam)
- **raw torch path**: `TORCH_BLAS_PREFER_HIPBLASLT=1` + `HIPBLASLT_TUNING_OVERRIDE_FILE` — engages directly.
- **sglang/vllm**: hipBLASLt runs *under aiter*; the override file is NOT read on the live aiter path. The
  rebind seam is aiter's CSV (`solidx` per shape) — deploy via [[operators/dense_gemm/backends/aiter]].

## Pitfalls & anti-patterns
- Solution indices are **ROCm/hipBLASLt-version-locked** — re-tune on every upgrade; a stale index can map
  to a different or removed kernel.
- `hipblaslt-bench` CLI is **absent in some images** — then aiter/gradlib (which races the same solution
  pool) is your only tuner.
- On sglang, tuning the override file alone does nothing (aiter dispatch) — a known dead-end; verify
  engagement through aiter's `is tuned on cu_num` log, not hipBLASLt markers.
- Don't read theoretical 1307 TFLOPS as achievable — real bf16 lands ~708 (≈54%); plan headroom from there.

## How to verify (worked example)
```bash
# raw-torch offline tune of one shape
HIPBLASLT_LOG_MASK=32 python my_gemm.py 2> shapes.log          # dump shapes
hipblaslt-bench --algo_method all -m 4096 -n 4864 -k 32896 \
    --a_type bf16_r --b_type bf16_r --compute_type f32_r       # search solutions
TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TUNING_OVERRIDE_FILE=tuned.json python my_gemm.py
# serving path: prove via aiter engagement instead
grep -c 'is tuned on cu_num' server.log
```

## Alternatives / cross-links
[[operators/dense_gemm/backends/aiter]] (deploy seam on serving) ·
[[operators/dense_gemm/backends/ck]] (custom epilogue / int8) ·
[[operators/dense_gemm/backends/asm]] (peak) · [[operators/dense_gemm/backends/triton]] ·
[[operators/dense_gemm/overview]].

## Sources
- hipBLASLt FP8 bar ~2750@4096 / ~3130@8192 (MI355X, ROCm 7.1); HIP/C++ 8-wave ping-pong beats it (3204@8192, no asm): AMD CDNA4 GEMM blog (https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html).
- hipBLASLt offline tuning: https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html
- TensileLite custom tuning: https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html
- MI300X GEMM levers (mfma_16x16, epilogue): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- bf16 708 / fp8 1223.6 TFLOPS, ~45% sustained: https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html ; https://arxiv.org/pdf/2510.27583
- aiter wraps hipBLASLt (`hipb_mm`): `/sgl-workspace/aiter/aiter/tuned_gemm.py` (`ROCm/aiter@a6bb4993`).
