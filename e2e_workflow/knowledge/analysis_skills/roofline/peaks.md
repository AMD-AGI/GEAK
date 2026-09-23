# Hardware peaks — roofline denominators

Pure data. One section per `gfx`. Extend by adding a section; nothing else needs to change.

Peaks are **dense, no-sparsity, sustained-achievable-ceiling** figures. HBM bandwidth is the
*theoretical pin* rate — real streaming kernels top out near 0.85–0.92 of it, which is exactly what
`target_eff` in `SKILL.md` encodes. Do not pre-derate the numbers here.

**Compute peaks need validation; the memory axis is the trustworthy one.** BF16 and FP16 MFMA run at
the same rate on these parts, so the two `flops` entries below must be **equal** — they are, and that
equality is the check to keep. Empirical MFMA microbenchmarks (e.g. from rocprof-compute) frequently
report a BF16 peak ~2× low, which inflates any BF16 compute-axis `roofline_pct` (sometimes above 100%,
where `SKILL.md` §6 L3 flags it `suspect`). At decode, prefer `hbm_util` and only rank on a
compute-axis number once its dtype peak has been validated against this equality.

## gfx950 — CDNA4, MI350X / MI355X class
```yaml
gfx: gfx950
cu: 256
hbm_bw_bytes_s: 8.0e12        # HBM3E, ~8 TB/s
flops:                         # dense matrix-core peaks, FLOP/s
  fp64: 7.86e13
  fp32: 1.57e14
  bf16: 2.5e15
  fp16: 2.5e15
  fp8:  5.0e15
  fp4:  1.0e16
l2_bytes: 4194304
```

## gfx942 — CDNA3, MI300X class
```yaml
gfx: gfx942
cu: 304
hbm_bw_bytes_s: 5.3e12        # HBM3, ~5.3 TB/s
flops:
  fp64: 1.63e14
  fp32: 1.63e14
  bf16: 1.31e15
  fp16: 1.31e15
  fp8:  2.61e15
l2_bytes: 4194304
```

## gfx1151 — RDNA3.5, Strix Halo / Radeon 8060S class (APU, unified LPDDR5X)

```yaml
gfx: gfx1151
cu: 40                        # torch reports multi_processor_count=20 = WGPs; 1 WGP = 2 CU
hbm_bw_bytes_s: 2.56e11       # LPDDR5X-8000 x 256-bit = 8000 MT/s * 32 B. NOT HBM.
flops:                        # WMMA (RDNA3.5 has no MFMA), dense, 40 CU @ 2.9 GHz
  fp64: 9.3e11               #   1:32 rate — UNVALIDATED, irrelevant to LLM serving
  fp32: 2.97e13              #   includes RDNA3 dual-issue VOPD; rarely achieved in practice
  bf16: 5.94e13              #   40 * 512 FLOP/clk/CU * 2.9e9; matches AMD's ~59 TFLOPS FP16 spec
  fp16: 5.94e13              #   equal to bf16 — the SKILL.md consistency check
l2_bytes: 2097152             # 2 MB L2
mall_bytes: 33554432          # 32 MB MALL / Infinity Cache — a THIRD roof, see SKILL.md §3 step 5
mall_bw_bytes_s: 7.9e11       # ~790 GB/s measured (read-read-write) = 3.4x the DRAM roof.
                              # Ops whose working set fits in 32 MB are NOT bound by
                              # hbm_bw_bytes_s above. Pure-read reaches 913-945 GB/s by ablation,
                              # so 7.9e11 is a conservative roof for a read-dominated kernel.
compute_empirical_ceiling: 3.85e13   # 38.35-38.52 TFLOP/s measured bf16/fp16 = 64.7% of the 5.94e13
                                     # datasheet figure. NOT a validated peak; see target_eff note
                                     # in SKILL.md §7 before ranking on any compute-axis number here.
```

**The compute axis on this part is a bracket, not a number.** No kernel measured on gfx1151 has
exceeded 38.35–38.52 TFLOP/s against the 59.4 datasheet entry above. Scoring `compute_util` against
5.94e13 therefore caps every kernel at ~0.65 and makes **every** gfx1151 GEMM read
`underperforming` forever — manufacturing headroom that does not exist. Report both ends (see
`perf_knowledge/hardware/rdna35_gfx1151/peak_tables.md`) and **prefer the memory axis**, exactly as
§4 already advises for BF16 on CDNA.

**Why the derived fallback is catastrophically wrong here — do not let it apply to gfx1151.**
`torch.cuda.get_device_properties` reports `memory_clock_rate = 1.0 GHz` / `memory_bus_width = 256`,
so the fallback's DDR formula yields `1.0e9 * 32 * 2 = 6.4e10` — **64 GB/s, 4x low**, because LPDDR5X
moves 8 bits per pin per clock, not 2. A 190 GB/s kernel then scores ~300% of roof, the roofline
self-marks `confidence: low / display-only`, and the whole memory axis silently stops being usable
for ranking.

Cross-check on the entry above: `2.56e11 * 0.91 = 233 GB/s`, which is exactly the sustained streaming
ceiling measured on this part (229-233 GB/s). A bf16 decode GEMM at 190 GB/s is therefore at **74% of
pin — genuinely below the 0.85-0.92 achievable band, i.e. real headroom**, not at the wall.

**Unified memory, so `total_memory` is not a GPU-only budget.** 96 GB is carved out of system RAM
shared with the host; check `MemAvailable` before sizing anything.

## Unknown gfx — derived fallback (confidence: low)

If the running `gfx` has no section above, DERIVE from `torch.cuda.get_device_properties(0)` and mark
`peaks.source="derived"`, `peaks.confidence="low"`:

```
hbm_bw_bytes_s ≈ memory_clock_rate_hz × (memory_bus_width_bits / 8) × 2     # DDR
flops[dtype]   ≈ multi_processor_count × clock_rate_hz × mfma_flops_per_cycle_per_cu[dtype]
```

The derived bandwidth is **frequently wrong for HBM3/3E** — the reported memory clock often understates
the effective pin rate (on gfx950 it derives ~4.1 TB/s against a real ~8 TB/s). Treat any derived-peak
result as `confidence: low`, which per `SKILL.md` means **display only, do not rank on it**.
