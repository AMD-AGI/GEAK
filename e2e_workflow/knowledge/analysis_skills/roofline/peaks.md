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
