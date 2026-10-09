# AMD RDNA 3.5 — Strix Halo / Gorgon Halo SKU planning peaks

**Planning models, not guaranteed sustained rates.** This file covers **gfx1151** APUs for
RDNA WMMA downgrade / agent-compute UMA targets. Optimize via the RDNA WMMA path —
see `rdna-fork.md` and your DSL's WMMA reference. Unlike the other SKU files this one
defines no `saturation` line: grid-fill and the quantization tail are in
`roofline-models.md ## Saturation / wave quantization`, calibration in
`roofline-models.md ## Calibration`, microarchitecture in `planning-constants.md`. ISA PDF:
[RDNA3 shader ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023.pdf);
gfx1151 WMMA enablement: [ROCm/FlyDSL #567](https://github.com/ROCm/FlyDSL/commit/9861ab21bb7e9316fe6a3e31d297cf2ab2e04401).

Official / community sources: [AMD Ryzen AI Max+ 395 blog](https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-395-processor-breakthrough-ai-.html),
[AMD Gorgon Halo / Max PRO 400 blog](https://www.amd.com/en/blogs/2026/amd-powers-next-generation-agent-computers-with-new-ryzen-ai-hal.html),
[AMD Ryzen AI Max+ PRO 495 product page](https://www.amd.com/en/products/processors/laptop/ryzen-pro/ai-max-pro-400-series/amd-ryzen-ai-max-plus-pro-495.html),
[Notebookcheck Radeon 8060S](https://www.notebookcheck.net/AMD-Radeon-8060S-Benchmarks-and-Specs.942049.0.html),
llm-tracker Strix Halo GPU perf (`llm-tracker.info/AMD-Strix-Halo-(Ryzen-AI-Max+395)-GPU-Performance`, **404 as of this revision** -- numbers below are retained from it, the page is not),
[kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes),
[lhl/strix-halo-testing](https://github.com/lhl/strix-halo-testing) (vs DGX Spark).

Cross-ref NVIDIA UMA competitor: the `tile-programming-cutedsl` skill's `nvidia-sm12x.md` §GB10 (`DGX_SPARK`, `RTX_SPARK_N1X`).

## Gorgon Halo refresh (395 → 495) — summary

**Ryzen AI Max+ PRO 495** (May 2026, codename **Gorgon Halo**) is a **clock + memory-capacity
refresh**, not a new GPU ISA. Same **40 CU / gfx1151 / RDNA 3.5** WMMA stack as 395.

| Dimension | AI Max+ 395 (Strix Halo) | AI Max+ PRO 495 (Gorgon Halo) | Changed? |
| --- | --- | --- | --- |
| GPU | Radeon **8060S** @ 2.9 GHz | Radeon **8065S** @ **3.0 GHz** | Clock only |
| CUs | 40 | 40 | **No** |
| isa_class | gfx1151 | **gfx1151** (expect same ROCm target) | **No new ISA** |
| FP16/BF16 peak | 59.4 TF | **~61.4 TF** (+3.4%) | Clock only |
| LPDDR5x | 8000 MT/s | **8533 MT/s** (OEM / leak class) | **Yes** |
| Peak BW (theory) | 256 GB/s | **~273 GB/s** | **Yes (~7%)** |
| Max UMA | 128 GB | **192 GB** | **Yes** |
| VGM (GPU carve-out) | up to 96 GB / 128 GB | up to **160 GB** / 192 GB | **Yes** |
| NPU | ~50 TOPS | **55 TOPS** | Slight |
| CPU boost | ~5.1 GHz | **5.2 GHz** | +100 MHz |
| FP8 / FP4 GPU WMMA | ✗ | **✗** | **No** |

PassMark leaks: ~**3–10%** multi-thread vs 395 ([TechPowerUp](https://www.techpowerup.com/348739/amd-ryzen-ai-max-pro-495-gorgon-halo-apu-appears-with-radeon-8065s)). No measured
hipBLASLt / `rocm_bandwidth_test` on 495 in community repos yet — calibrate from trace.

## Instruction set (gfx1151 / RDNA 3.5)

Applies to **395 and 495** — Gorgon Halo does **not** add FP8 matrix or new WMMA shapes.

| Mechanism | Strix / Gorgon Halo (8060S / 8065S) | Notes |
| --- | --- | --- |
| Wave32 WMMA 16×16×16 | ✓ f16, bf16 | WMMA gfx1151 (`rdna-fork.md`; FlyDSL port exists) |
| WMMA INT8 / INT4 | ✓ i8, iu4 | rocWMMA gfx11 row |
| **FP8 matrix (WMMA)** | **✗** | **No native FP8 on gfx11** — needs gfx12 (RDNA 4) |
| **FP4 / NVFP4** | **✗** | No block-scaled FP4 tensor path |
| MFMA (CDNA) | ✗ | RDNA, not CDNA |
| LDS (shared) | ✓ | Strix Halo LDS sizing enabled in FlyDSL gfx1151 port |
| hipBLASLt GEMM | ✓ (ROCm 6.4+) | **Required** for competitive BF16/FP16 GEMM on gfx1151 |
| Vulkan (RADV/AMDVLK) | ✓ | Often best llama.cpp path for short context |
| XDNA 2 NPU | ✓ ~50 TOPS (395) / **55 TOPS** (495) | Separate from GPU; Copilot+ class |

**Compile / runtime**: `HCC_AMDGPU_TARGET=gfx1151`; ROCm 6.4+ with hipBLASLt;
`ROCBLAS_USE_HIPBLASLT=1` for GEMM. Variable Graphics Memory (VGM) up to 96 GB of 128 GB
for GPU workloads.

## Precision support (GPU tensor paths)

| Format | Ryzen AI Max+ 395 | Notes |
| --- | --- | --- |
| FP32 (SIMT) | ✓ | Per-CU SIMD |
| FP16 / BF16 WMMA | ✓ | **59.4 TF** theoretical peak @ 2.9 GHz |
| INT8 WMMA | ✓ | rocWMMA gfx11 |
| INT4 WMMA | ✓ | `wmma.i32.16x16x16.iu4` |
| **FP8** | **✗** (GPU WMMA) | Emulation only; native FP8 WMMA is **gfx12+** |
| **FP4** | **✗** | Use INT4 WMMA or quantized GGUF paths |
| NPU INT8/FP16 | ✓ | XDNA 2 ~50 TOPS — not in GPU ridge |

**Peak formula** (WMMA-saturated FP16/BF16):

```text
peak_FP16 = 512 ops/clock/CU × 40 CU × 2.9 GHz ≈ 59.4 TFLOPS
```

Without WMMA or wave32 VOPD, practical peak is **~half** (~29.7 TF).

## Quick reference

The two SKU rows GEAK carries are rendered from `perf_knowledge/hardware/data/sku.json`
(`extract_sku.py --sync-docs`; edit the json, never these cells). `AI_MAX_395` is the arch default
for a bare `gfx1151`. Its stored memory ceiling is the **measured** 0.212 TB/s
(`peak_hbm_basis: measured`, tool `rocm_bandwidth_test`); the 0.256 TB/s pin rate is
`datasheet_hbm_tb_s`, which the e2e roofline ranks against. Compute is **derived** from the published
clock (WMMA 512 ops/clk/CU, IU4 1024, dual-issue FP32 256 FLOP/clk/CU) — FP32 29.7 TF on 395, not
the unsourced 20.0 an earlier table carried. L2 is **2 MiB**, the Infinity Cache (MALL) **32 MiB**.

<!-- BEGIN GENERATED by extract_sku.py: table gfx1151 -->
| SKU | arch | CUs | clock (MHz) | mem BW TB/s | L2 / MALL (MB) | FP16 = BF16 | FP8 | FP4 | INT8 | FP32 | FP64 | basis | GEAK |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AI_MAX_395 | gfx1151 | 40 | boost 2900 | 0.212 measured (0.256 datasheet) | 2 / 32 | 59.4 | — | — | 59.4 | 29.7 | — | derived | yes |
| AI_MAX_495 | gfx1151 | 40 | boost 3000 | 0.273 | 2 / 32 | 61.4 | — | — | 61.4 | 30.7 | — | derived | yes |

Dense TFLOP/s (TOPS for INT8), no sparsity; `—` = no rate on record (a consumer refuses, never substitutes FP16). Generated from `perf_knowledge/hardware/data/sku.json` by `kernel_workflow/scripts/kernel_tools/extract_sku.py`; edit the json, then `extract_sku.py --sync-docs`.
<!-- END GENERATED by extract_sku.py -->

| SKU | isa_class | CUs | CPU | Mem | Peak BW | FP16/BF16 | FP8 | FP4 | INT8 | kernel_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ryzen AI Max+ **PRO 495** | gfx1151 | 40 | 16× Zen 5 | up to **192 GB** LPDDR5x UMA | **273 GB/s**†† | **61.4 TF** | — | — | WMMA | **Gorgon Halo**; 8065S @ 3.0 GHz |
| Ryzen AI Max+ 395 | gfx1151 | 40 | 16× Zen 5 | 32–128 GB LPDDR5x UMA | 256 GB/s (212†) | 59.4 TF (37‡) | — | — | WMMA | Radeon 8060S; Strix Halo |
| Ryzen AI Max 390 | gfx1151 | 32 | 12× Zen 5 | 32–128 GB | 256 GB/s | ~47.5 TF | — | — | WMMA | 8050S; scale ∝ CU |
| Ryzen AI Max 385 | gfx1151 | 32 | 8× Zen 5 | 32–128 GB | 256 GB/s | ~47.5 TF | — | — | WMMA | 8050S |

† **212 GB/s** measured on 395 (`rocm_bandwidth_test`, llm-tracker) — the `sku.json`
`peak_hbm_tb_s` for `AI_MAX_395` (`peak_hbm_basis: measured`, tool version unrecorded). ‡ **36.9 TF** hipBLASLt BF16 GEMM measured (395). †† **273 GB/s** = LPDDR5x-8533
× 256-bit theoretical ([topcpu.net](https://www.topcpu.net/en/cpu/amd-ryzen-ai-max-pro-495), AMD
blog); **not yet measured** on 495 — same class as DGX Spark package BW.

## Full SKU — Ryzen AI Max+ 395 (`AI_MAX_395`)

| Field | Value |
| --- | --- |
| SKU key (`sku.json`; `calc_perf.py` / `hw_budget.py --sku`) | `AI_MAX_395` |
| isa_class | gfx1151 (RDNA 3.5) |
| GPU | Radeon 8060S |
| Compute units | 40 |
| CPU | 16 cores / 32 threads Zen 5 (up to ~5.1 GHz) |
| Memory | 32 / 64 / 128 GB LPDDR5x-8000, 256-bit UMA |
| Peak memory BW (planning) | 212 GB/s measured; 256 GB/s theoretical max |
| FP16/BF16 tensor (theoretical) | 59.4 TFLOP/s @ 2.9 GHz WMMA |
| FP16/BF16 (hipBLASLt measured) | 36.9 TFLOP/s |
| FP8 (GPU WMMA) | **not supported** |
| FP4 | **not supported** |
| INT8 WMMA | supported (no separate marketing TOPS) |
| INT4 WMMA | supported |
| NPU | XDNA 2 ~50 TOPS |
| VGM (128 GB config) | up to 96 GB GPU-dedicated |
| kernel_notes | gfx1151 ROCm still maturing; llama.cpp **Vulkan** often beats raw HIP for pp512; **HIP+rocWMMA+FA** for long context. vs DGX Spark: slower prefill, competitive tg on large models ([lhl/strix-halo-testing](https://github.com/lhl/strix-halo-testing)). |

## Full SKU — Ryzen AI Max+ PRO 495 (`AI_MAX_495`)

| Field | Value |
| --- | --- |
| SKU key (`sku.json`; `calc_perf.py` / `hw_budget.py --sku`) | `AI_MAX_495` |
| isa_class | gfx1151 (RDNA 3.5) — **same as 395** |
| GPU | Radeon 8065S |
| Compute units | 40 |
| CPU | 16 cores / 32 threads Zen 5 (up to **5.2 GHz**) |
| Memory | up to **192 GB** LPDDR5x-8533, 256-bit UMA |
| Peak memory BW (planning) | **273 GB/s** theoretical (8533 MT/s); measure on silicon |
| FP16/BF16 tensor (theoretical) | **61.4 TFLOP/s** @ 3.0 GHz WMMA |
| FP8 (GPU WMMA) | **not supported** |
| FP4 | **not supported** |
| INT8 / INT4 WMMA | supported |
| NPU | XDNA 2 **55 TOPS** |
| VGM (192 GB config) | up to **160 GB** GPU-dedicated ([AMD GRHP-01](https://www.amd.com/en/blogs/2026/amd-powers-next-generation-agent-computers-with-new-ryzen-ai-hal.html)) |
| kernel_notes | **Refresh, not new ISA.** Main wins: **192 GB UMA**, faster LPDDR5x, +3.4% GPU clocks. Expect ~**225** FP16 ridge (61.4 TF / 273 GB/s). AMD claims 300B+ param @ 4-bit with VGM. Q3 2026 Halo dev platform. |

## Cross-SKU / cross-vendor impact

| Comparison | ISA | Planning impact |
| --- | --- | --- |
| **495 vs 395** | Same gfx1151 | **+3.4% TFLOPS**, **~6.6% BW theory** (256→273 GB/s), **+50% max RAM**; no FP8/FP4 |
| AI Max+ 495 vs DGX Spark | gfx1151 vs sm_121a | Spark ~3.5× FP16 peak; **495 matches Spark package BW** (273 GB/s) |
| AI Max+ 395 vs DGX Spark | gfx1151 vs sm_121a | Spark ~3.6× FP16 peak, similar UMA capacity; Spark faster prefill |
| AI Max+ 395 vs RTX 5090 | gfx1151 vs sm_120 | 5090 ~7× FP16 peak, ~7× BW; Strix wins on **VRAM capacity** at 128 GB UMA |
| AI Max+ 395 vs H20 | gfx1151 vs sm_90a | H20 more tensor TFLOPS; Strix often better **$/GB** for local LLM |
| 395 vs 390/385 | Same gfx1151 | Fewer CUs (32); scale tensor peaks by CU ratio |

## Ridge (FP16/BF16, planning)

| SKU | peak / BW | ridge (ops/byte) | notes |
| --- | --- | --- | --- |
| AI_MAX_395 | 59.4 TF / 212 GB/s | **~280** | theoretical WMMA |
| AI_MAX_395 | 36.9 TF / 212 GB/s | **~174** | hipBLASLt calibrated |
| AI_MAX_395 | 59.4 TF / 256 GB/s | **~232** | theoretical BW |
| AI_MAX_495 | 61.4 TF / 273 GB/s | **~225** | Gorgon Halo theory |
| DGX_SPARK (ref) | 214 TF / 273 GB/s | ~784 | same BW class; Spark higher compute ridge |

## GitHub benchmark anchors

| Repo | Finding |
| --- | --- |
| llm-tracker (**source page 404**, figures retained) | 59.4 TF theory; 36.9 TF hipBLASLt; 212 GB/s BW |
| [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) | hipBLASLt on by default; ROCm 6.4.4 vs 7.x variance |
| [lhl/strix-halo-testing](https://github.com/lhl/strix-halo-testing) | Spark pp2048 +68% to +446% vs Strix; tg converges at long ctx |
| [visorcraft/strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf) | 70B–235B models on 128 GB; tg 5–86 tok/s by model |
| [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) | gfx1151: f16/bf16 WMMA yes; **FP8 gated off gfx11** |

## Out of scope (this file)

| SKU | Route |
| --- | --- |
| MI300X / CDNA3 | `amd-cdna3-skus.md` (gfx942) |
| gfx1100 discrete (7900XTX / W7900) | `amd-rdna3-skus.md` |
| RDNA 4 gfx12xx discrete | `amd-rdna4-skus.md` |
| RTX Spark / DGX Spark | CuTeDSL skill — the `tile-programming-cutedsl` skill's `nvidia-sm12x.md` §GB10 |

## Optimize on Gluon
Gluon: `gl.amd.rdna3.wmma` — see `../gluon/rdna-wmma-reference.md`.
