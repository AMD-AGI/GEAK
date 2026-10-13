# Assets

| File | Use |
| --- | --- |
| `example_arch.json` | **Synthetic** values for the `roofline_report.py` CPU-only example. Never use for real GPU results. |
| `example_cases.json` | Synthetic input for the same example. Reproduces 42.9497% → 53.6871% (and 85.8993% counter-derived). |
| `MI355X.measured.json` | A real microbench-derived MI355X arch file, with the fp4 backend split added by hand. |
| `arch_variants/` | Four MI355X arch files found in circulation on this cluster. Evidence that the arch JSON is **not a constant**. |

## The arch JSON is not a constant

Four distinct "MI355X" arch files exist here. They are not copies of one another and
they do not agree:

| File | md5 | `mem_bw_gbps` | `matrix_bf16` | `matrix_fp16` | `matrix_fp4` | `source` field | `matrix_*` keys |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `MI355X.bw6857_bf1449.json` | `c045ecc958e17ca6062de4970ddd3316` | 6857 | 1449 | 1314 | 5110 | benchmark-derived | 8 |
| `MI355X.bw6857_bf1452.json` | `278cb924d0f0d71ee7941c3af34d5694` | 6857 | 1452 | 1306 | 5004 | benchmark-derived | 8 |
| `MI355X.bw6877_bf1422.json` | `48743a3999d530e33f1364af2302f661` | 6877 | 1422 | 1276 | 4964 | benchmark-derived | 8 |
| `MI355X.bw8000_bf1686.json` | `dbd1c7ccb8ac28daeae995c36deec7f1` | 8000.0 | 1686.0 | 1686.0 | 5663.0 | **absent** | **5** |
| `MI355X.measured.json` | `68484cdc63ea6d95ed31691c20680a01` | 6851 | 1452 | 1310 | 5146 | benchmark-derived + hand note | 10 |

**bf16 dispersion across the four variants is 18.6%** (1422 → 1686). Pick the wrong
one and that is the error in every roofline percentage you publish from it.

### The outlier is handwritten

`MI355X.bw8000_bf1686.json` is the most-copied of the four and the one you are most
likely to be handed. Three tells, none of which is the filename:

1. **No `source` field.** The other three carry
   `"These are benchmark derived peak flops and bw"`.
2. **It exactly equals the datasheet pin** — 8000 GB/s is the spec HBM figure, not a
   measured one, and the measured files all land near 6857.
3. **Only five `matrix_*` keys** instead of eight, and `matrix_bf16 == matrix_fp16`
   exactly — which is right in hardware but is never what a microbench produces,
   since its fp16 path does not saturate the unit (the three measured files all show
   fp16 ~9% under bf16).

So: **do not infer provenance from the filename.** Record arch path + md5 + `source`
+ the peak key actually resolved on every output row.

### Reconciling the bandwidth figures

The three benchmark-derived variants read 6857 / 6857 / 6877 GB/s. The microbench
run that produced `MI355X.measured.json` read 6851. All four are the same GPU model
measured on different days — that ~0.4% spread is ordinary microbench scatter and is
*not* the thing to worry about. The 8000 entry is a different kind of number
entirely (a spec pin), and that is the 16.8% gap that matters.

### `matrix_fp4` hides two backends

The microbench measures fp4 on both CK and Triton and writes only the **max** into
the single `matrix_fp4` key. On the run behind `MI355X.measured.json` that was CK
5146 vs Triton 2722 TFLOP/s. A Triton or FlyDSL fp4 kernel scored against
`matrix_fp4` is therefore underestimated by **1.89×**.

`MI355X.measured.json` carries hand-added `matrix_fp4_ck` and `matrix_fp4_triton`
keys recovered from the same run's SUMMARY block. Use the backend-specific key when
the backend is known; its `source` field records the hand edit.

See [`../references/combination.md` §7](../references/combination.md) for how this
feeds the reporting rules.
