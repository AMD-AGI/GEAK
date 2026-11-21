---
title: roofline and bottleneck classification
kind: technique
gens: [gfx950, gfx942]
dtypes: [bf16, fp16, fp8_e4m3, fp8_e4m3_fnuz, fp4_e2m1, int8]
regimes: [prefill, decode, training, both]
updated: 2026-10-07
sources:
  - https://rocm.github.io/rocprofiler-compute/performance_model.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# roofline and bottleneck classification

## TL;DR
Before optimizing, **classify the kernel**: compute-bound vs bandwidth-bound vs latency-bound,
decided by **arithmetic intensity** (FLOP per byte of HBM traffic) against the per-dtype machine
balance AND by how close the kernel actually gets to that roof. Optimizing a bandwidth-bound kernel
for MFMA occupancy (or a compute-bound kernel for coalescing) wastes effort. Reality check: MI300X
sustains only **~45–55% of theoretical peak** matrix throughput across fp8/bf16/fp16 — a
software-maturity ceiling, not hardware — so the bar is the **best tuned library kernel**, never the
datasheet peak. The method (basis pair, rank-vs-gate ladder, rocprof-compute caveats) is defined once
in `[[profiling/roofline_on_mi.md]]`; per-SKU peaks in `[[hardware/cdna4_mi350/peak_tables.md]]`
(gfx950, main line) and `[[hardware/cdna3_mi300/peak_tables.md]]` (gfx942 downgrade), both rendered
from `perf_knowledge/hardware/data/sku.json`; see also `[[hardware/shared/hbm_infinity_fabric.md]]`.

## Concepts
- **Arithmetic intensity (AI)** = FLOPs / HBM bytes moved. Compare to **machine balance** =
  peak FLOP/s[dtype] ÷ peak HBM BW (the roofline ridge point — per dtype and per product: MI355X FP16
  ≈ 312, MI350X ≈ 288, MI300X ≈ 247, MI325X ≈ 218 FLOP/B). `AI > balance` ⇒ the compute roof is the
  one the kernel walks toward; `AI < balance` ⇒ the bandwidth roof. Which roof actually **binds** is
  decided by utilization (next bullet), not by AI alone.
- **Utilization informs a hypothesis**. The 60/40/80% values in
  `perf_knowledge/hardware/data/thresholds.json` are local diagnostic heuristics.
  Low utilization alone does not establish latency or occupancy as the cause;
  check stalls, launch size and measured traffic before choosing an intervention.
- **Every "% of roofline" names its basis** — numerator `model` | `counters`, denominator
  `datasheet` | `empirical@<tool>-<version>` | `in-shape probe`. Local analytic tools
  always emit `analysis_role: planning`, `may_gate: false`, including with measured inputs.
  Native TraceLens owns its workload models, calibration and report semantics;
  acceptance still requires the workflow's correctness and performance validation
  (`[[profiling/roofline_on_mi.md]]`).
- **Roofline**: model ceiling = `min(peak_compute, AI × peak_BW)`. This identifies
  the limiting model axis; it does not by itself prove the measured bottleneck.
- **The ~45% reality**: MI300X delivers ~45–55% of theoretical matrix peak in practice (third-party
  bf16 ceiling ~890 TFLOP/s ≈ 68% of 1.3 PFLOP/s, power-limited at 750 W). Treat measured
  best-library throughput as the practical roof (`[[operators/dense_gemm/tuning.md]]`,
  `[[hardware/cdna3_mi300/clocks_power.md]]`).

## How to classify a kernel (procedure)
1. **Estimate AI analytically**: GEMM `M·N·K·2` FLOPs over operand reads, scale bytes
   (for MX formats) and output writes, each at its own storage width, assuming one HBM pass.
   Large square GEMM ⇒ high AI ⇒ a compute-bound hypothesis; GEMV/decode, norm,
   elementwise, copy ⇒ low AI ⇒ bandwidth-bound.
   Use the archetype model with a bracket rather than one number:
   `kernel_workflow/scripts/kernel_tools/hw_budget.py --sku MI355X --workload gemm --shapes M=..,N=..,K=.. --dtype bf16`.
2. **Measure**: use native TraceLens for trace-based roofline and reporting. Optional
   rocprof-compute counters (`VALUBusy`, MFMA-busy, HBM read+write BW and stalls) help test
   a hypothesis; busy percentages are not interchangeable with throughput percentages. On gfx950,
   do not read bytes from `FETCH_SIZE` / `TCC_BUBBLE` alone — they under-count; and `--roof-only` there
   needs rocprof-compute ≥ 3.6.0.
3. **Pick the lever set** from the table below.

## Bottleneck → lever map
| classification | symptom (counters) | levers |
|---|---|---|
| compute-bound | MFMA busy high, HBM low | `[[optimization/mfma_scheduling.md]]`, `[[optimization/occupancy_and_registers.md]]`, tile/MFMA shape (`[[operators/dense_gemm/tuning.md]]`) |
| bandwidth-bound | HBM near peak, MFMA idle | `[[optimization/vectorization_and_coalescing.md]]`, `[[optimization/xcd_l2_locality.md]]` (L2 reuse), `[[optimization/kernel_fusion_strategy.md]]` (cut traffic) |
| latency/occupancy-bound | both low, high stall cycles | `[[optimization/memory_pipelining.md]]`, more waves/EU, more workgroups (`[[optimization/wave_and_grid_sizing.md]]`) |
| LDS-bound | high `ds_*` stalls | `[[optimization/lds_and_bank_conflicts.md]]` (swizzle/padding) |

## Typical LLM classifications
- **Prefill GEMM / attention scores**: compute-bound (keep 256 / 304 CUs at high MFMA occupancy).
- **Decode GEMV / KV read / sampling**: bandwidth/latency-bound (coalesce, split-K to fill CUs).
- **RMSNorm / LayerNorm / elementwise / cast**: bandwidth-bound (fuse to cut passes,
  `[[optimization/kernel_fusion_strategy.md]]`).

## Pitfalls
- Optimizing the wrong roof (MFMA tuning a bandwidth-bound norm).
- Using theoretical peak as the denominator for "efficiency" — use measured best-library or an
  in-shape probe; a datasheet ratio is a ranking prior, labelled `denominator_basis: datasheet`.
- Quoting one SKU's peaks for another (MI355X for MI350X, MI300X bandwidth for MI325X), or the FP16
  peak for a dtype the SKU does not list.
- Ignoring the third roof: many real kernels are **latency-bound** (under-occupied / stalled), not
  cleanly compute- or BW-bound.
- Forgetting fusion *changes AI* — fusing two BW-bound kernels can push the result toward compute-bound.

## Verify
- Omniperf roofline plot: kernel point vs HBM and compute roofs (`[[profiling/]]`).
- Counters: MFMA/`VALUBusy` vs HBM BW; stall-cycle breakdown for the latency case.
- Recompute AI after any fusion and re-classify.

## Sources
- Unified method (basis pair, rank-vs-gate ladder, rocprof-compute ≥ 3.6.0 on gfx95x, gfx950
  `FETCH_SIZE`/`TCC_BUBBLE` under-count): `perf_knowledge/profiling/roofline_on_mi.md`.
- Thresholds: `perf_knowledge/hardware/data/thresholds.json` `bound_classification`; peaks:
  `perf_knowledge/hardware/data/sku.json`.
- Roofline / performance model, busy & BW counters: Omniperf performance-model docs.
- ~45–55% sustained-of-peak, ~890 TFLOP/s bf16 ceiling, 750 W power limit: ROCm workload guide + cited bench (see `[[operators/dense_gemm/tuning.md]]`).
- AI/operand-feed framing: ROCm matrix-cores-CDNA blog.
