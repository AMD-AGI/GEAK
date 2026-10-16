# Hardware peaks — roofline denominators

Pure data. One section per tabulated product (or per `gfx` for legacy tables).
MI350X and MI355X share `gfx950` but have different compute peaks. A bare `gfx950`
lookup has no table; select `product: mi350x` or `product: mi355x` from device identity.
Client RDNA4 peaks are **R9700-only**. A bare `gfx1201` lookup without `product: r9700`
must not inherit these numbers — several SKUs share that ISA.

Peaks are **dense, no-sparsity, datasheet-ceiling** figures. Memory bandwidth is the
*theoretical pin* rate. `target_eff` in `SKILL.md` is a generic op-class
implementation prior, not a measured product-specific bandwidth derate. Record
an on-box streaming measurement before claiming a product efficiency factor;
the measured R9700 streaming override is documented separately in `SKILL.md`
and selected by `target_eff_for`. Do
not pre-derate the numbers here. Do not put sparse (2:4)
rates in the yaml `flops` map.

Validate each denominator against the actual product, dtype, instruction unit and
clock/power conditions. The equal BF16/FP16 entries below follow the tabulated
matrix modes; equality alone does not validate an empirical benchmark. Memory
and compute models both need matching definitions and measurement boundaries.
The legacy `hbm_bw_bytes_s` field also holds GDDR/LPDDR pin rates where labeled.

## r9700 — Radeon AI PRO R9700 (Navi 48, gfx1201, 300 W)
```yaml
product: r9700
gfx: gfx1201
cu: 64                         # physical CUs from rocminfo; not PyTorch WGP count (32)
hbm_bw_bytes_s: 6.4e11         # 32 GB GDDR6, 256-bit at 20 Gbps, ~640 GB/s
flops:                         # dense peaks, FLOP/s (or OPS/s for int8)
  fp32: 4.78e13                # vector FP32 rate, not a matrix-core rate
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
MI300X (~247) and MI355X (~312) ridges; MI350X is ~288. A repeatedly reused working set that fits
the 64 MB Infinity Cache can report effective bandwidth above 640 GB/s; do not call
that impossible GDDR bandwidth. Measure the external-memory roof with a streaming
working set larger than the cache.

## mi350x — AMD Instinct MI350X (CDNA4, gfx950)
```yaml
product: mi350x
gfx: gfx950
cu: 256
source_url: https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html
source_checked: 2026-10-16
hbm_bw_bytes_s: 8.0e12        # HBM3E, 8 TB/s
flops:                         # dense matrix-core peaks, FLOP/s
  fp64: 7.21e13
  fp32: 1.442e14
  bf16: 2.3e15
  fp16: 2.3e15
  fp8:  4.6e15
  fp4:  9.2e15                # MXFP4, not an arbitrary FP4 instruction mode
```

## mi355x — AMD Instinct MI355X (CDNA4, gfx950)
```yaml
product: mi355x
gfx: gfx950
cu: 256
source_url: https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
source_checked: 2026-10-16
hbm_bw_bytes_s: 8.0e12        # HBM3E, 8 TB/s
flops:                         # dense matrix-core peaks, FLOP/s
  fp64: 7.86e13
  fp32: 1.573e14
  bf16: 2.5e15
  fp16: 2.5e15
  fp8:  5.0e15
  fp4:  1.01e16               # MXFP4, not an arbitrary FP4 instruction mode
```

These are AMD's published rounded per-GPU specifications, checked on the date
above: BF16/FP16 matrix, OCP-FP8 matrix (E5M2/E4M3), MXFP4 matrix, and FP32/FP64
matrix rates. AMD lists the same FP32/FP64 vector rates. The separate structured
sparsity rows are excluded. The listed peak engine clocks and typical board powers
are 2200 MHz / 1000 W for MI350X and 2400 MHz / 1400 W for MI355X. These are
conditional specification ceilings, not measured throughput or guaranteed headroom.

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

`derive_peaks_from_props()` can estimate external-memory pin bandwidth from
reported memory clock and bus width, with `confidence="low"`. Clock units and
DDR factors vary by device/tool; verify them before interpretation. The helper
does not derive compute throughput. These estimates are display-only, and
`resolve_peaks()` disables this fallback for unidentified `gfx950` and unknown
client `gfx120x` products.
