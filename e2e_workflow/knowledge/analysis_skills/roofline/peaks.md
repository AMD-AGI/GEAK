# Hardware peaks — roofline denominators

Pure data. One section per calibrated product (or per `gfx` when the ISA is the product).
Client RDNA4 peaks are **R9700-only**. A bare `gfx1201` lookup without `product: r9700`
must not inherit these numbers — several SKUs share that ISA.

Peaks are **dense, no-sparsity, datasheet-ceiling** figures. Memory bandwidth is the
*theoretical pin* rate — real streaming kernels top out near 0.85–0.92 of it, which is exactly what
`target_eff` in `SKILL.md` encodes. Do not pre-derate the numbers here. Do not put sparse (2:4)
rates in the yaml `flops` map.

**Compute peaks need validation; the memory axis is the trustworthy one.** BF16 and FP16 run at the
same rate on the matrix core of every part below (MFMA on CDNA, WMMA on RDNA), so the two `flops`
entries in each section must be **equal** — they are, and that equality is the check to keep.
Empirical matrix-core microbenchmarks (e.g. from rocprof-compute) frequently report a BF16 peak ~2×
low, which inflates any BF16 compute-axis `roofline_pct` (sometimes above 100%, where `SKILL.md` §6
L3 flags it `suspect`). At decode, prefer `hbm_util` and only rank on a compute-axis number once its
dtype peak has been validated against this equality.

## r9700 — Radeon AI PRO R9700 (Navi 48, gfx1201, 300 W)
```yaml
product: r9700
gfx: gfx1201
cu: 64                         # physical CUs from rocminfo; not PyTorch WGP count (32)
hbm_bw_bytes_s: 6.4e11         # 32 GB GDDR6, 256-bit at 20 Gbps, ~640 GB/s
flops:                         # dense matrix-core peaks, FLOP/s (or OPS/s for int8)
  fp32: 4.78e13
  bf16: 1.91e14
  fp16: 1.91e14
  fp8:  3.83e14
  int8: 3.83e14
```

Public dense peaks for this SKU: FP32 vector 47.8 TFLOPS; FP16/BF16 matrix 191 TFLOPS
(383 sparse 2:4); FP8 matrix 383 TFLOPS (766 sparse); INT8 383 TOPS (766 sparse);
INT4 766 TOPS (1531 sparse). There is **no** `fp4` key: this SKU has no block-scaled
MX path. Memory sits behind 64 MB Infinity Cache and 8 MB L2. Clocks: 1620 base /
2350 game / 2920 boost. Other gfx120x / gfx1201 products stay unknown until they have
their own `product:` section. `gfx1250` is CDNA5, not RDNA4 — do not inherit these numbers.

The FP16/BF16 ridge is ~298 FLOP/byte (`191e12 / 640e9`), between the tabulated
MI300X (~247) and MI350/355 (~312) ridges. A repeatedly reused working set that fits
the 64 MB Infinity Cache can report effective bandwidth above 640 GB/s; do not call
that impossible GDDR bandwidth. Measure the external-memory roof with a streaming
working set larger than the cache.

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

## gfx1151 — RDNA3.5, Radeon 8060S (Strix Halo APU) class
```yaml
gfx: gfx1151
cu: 40
hbm_bw_bytes_s: 256.0e9       # LPDDR5X-8000, 256-bit — pin rate, NOT HBM
flops:                         # dense WMMA peaks, FLOP/s — theoretical
  fp32: 3.0e13
  bf16: 6.0e13
  fp16: 6.0e13
  int8: 6.0e13
l2_bytes: 2097152
mall_bytes: 33554432
```

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
