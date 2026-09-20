# Hardware peaks — roofline denominators

Pure data. One section per `gfx`. Extend by adding a section; nothing else needs to change.

Peaks are **dense, no-sparsity, sustained-achievable-ceiling** figures. HBM bandwidth is the
*theoretical pin* rate — real streaming kernels top out near 0.85–0.92 of it, which is exactly what
`target_eff` in `SKILL.md` encodes. Do not pre-derate the numbers here.

**These are datasheet figures, and on gfx950 an up-to-date microbenchmark now agrees with them.**
BF16 and FP16 MFMA run at the same rate on these parts, so the two `flops` entries below must be
**equal** — they are, and that equality is the check to keep.

The old warning that "empirical MFMA peaks read ~2× low" was real but is now **root-caused and fixed
upstream**: rocprof-compute's roofline microbenchmark measured gfx950 MFMA with the CDNA3 instructions
`v_mfma_f32_32x32x8_{f16,bf16}` and scored them at 16384 FLOP/iter, while CDNA4 has
`v_mfma_f32_32x32x16_{f16,bf16}` at 32768. Every FP16 / BF16 / INT8 empirical peak therefore came out
2–4× low and every compute-axis ratio built on one came out that much too high — production BF16 GEMMs
scored 196% and 219%, which is not a measurement but a disproof of the denominator. **Fixed in
rocprof-compute 3.6.0 (ROCm 7.13.0)**: *"Fixed roofline benchmark MFMA FP16/BF16/INT8 peaks for
MI 350"*. gfx942 is CDNA3 and was never affected.

Measured on gfx950, 3.4.0 → 3.6.0, GFLOP/s:

| dtype | old empirical | new empirical | ratio | vs the table below |
|---|---|---|---|---|
| bf16 | 614 663 | 2 435 517 | 3.96× | 97.4% of 2.5e15 |
| fp16 | 1 227 048 | 2 162 758 | 1.76× | 86.5% of 2.5e15 |
| int8 | 1 222 436 | 4 854 023 | 3.97× | — |
| fp4 (F6F4) | *not measured* | 9 680 756 | — | 96.8% of 1.0e16 |
| fp8 / fp32 / fp64 | — | — | **0.99–1.00×** | unchanged — the control |

fp8/fp32/fp64 not moving is what makes this a *fix* rather than microbenchmark drift. Note the new
FP16 number still sits ~11% under BF16 on identical silicon — the clean 2× is gone, but the
microbenchmark does not fully saturate FP16, so the equality check below still has something to say.
Another reason the table in this file, not a microbenchmark, is the denominator of record.

**So:** if you are reading empirical peaks from rocprof-compute on a gfx95x part, require **≥ 3.6.0**;
below that, use the datasheet values here instead. Keep the BF16 == FP16 equality as the standing
check for any peak source — it is what caught this bug, and it will catch the next one. The memory
axis stays the one to trust at decode: prefer `hbm_util` there regardless.

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
