# Shape corpus and reference harness

Three files:

- **`shapes.py`** — the GEMM shape corpus, grouped by regime.
- **`run_case.py`** — a reference measurement harness applying the rules from
  `../tuning-core/measurement.md` and `../tuning-core/correctness_gates.md`.
- **`graph_harness.py`** — operator-agnostic HIP/CUDA graph capture-and-replay timing, with a
  capture-validity guard. Use this for anything on a graph-captured path; see
  `../tuning-core/graph_captured_benchmarking.md`.

`run_case.py` is **not a tuner**. It measures a shape with `torch.mm` so that every backend
in this skillset has a consistently-measured number to be compared against. The tuners live
in the per-language skills.

**`run_case.py` is also not a general-purpose harness.** It is hardwired to `torch.mm` on a
single operand pair, and it neither captures a graph nor rotates buffers. It is the calibration
reference for *methodology*, not something you can point at a library op. To time an aiter or CK
op on a serving decode path, use `graph_harness.py`: eager timing of a small kernel measures
launch overhead the kernel will never pay in production, and a change that is not present at
capture time does not take effect at all.

## Use

```bash
rocm-smi --showuse                    # find an idle GPU
export HIP_VISIBLE_DEVICES=4          # pin it

python3 shapes.py --smoke                       # inspect the corpus
python3 shapes.py --regime decode --format csv  # feed another tool
python3 run_case.py --smoke                     # validate the harness
python3 run_case.py --regime square --repeats 7 --rep 200
python3 run_case.py --M 4096 --N 4096 --K 4096
```

## Regimes

Grouped by regime rather than size, because regime — not raw dimensions — determines which
configs win. The measured evidence is in `../tuning-core/search_strategy.md`: the
compute-bound and decode winners sit at opposite corners of the same config space.

| regime | shape family | why it is separate |
| --- | --- | --- |
| `square` | N×N×N, 1K–8K | classic compute-bound; wants the largest viable tile |
| `tall_skinny` | large M, small N | prefill and narrow projections |
| `short_fat` | small M, large N | transpose of the above; different tile preference |
| `k_heavy` | deep K, narrow N | down-projection shaped; split-K becomes the lever |
| `decode` | M=1 GEMV | pure bandwidth; peak-FLOPS numbers are meaningless here |
| `batch_decode` | M ∈ 2..256 | the serving regime most often skipped; the bucketing ladder |

43 shapes total; `--smoke` gives one per regime for validating a harness cheaply.

## Reading the output

```
     M      N      K        regime       ms   TFLOPS     GB/s  spread%     err    SNR  status
  4096   4096   4096        square    0.233    590.0    432.1      8.0  0.0000   55.6  ok
     1   4096   4096        decode    0.017      2.0   2026.8     35.6  0.0000   55.6  NOISY mem-bound:judge-GB/s
```

- **TFLOPS vs GB/s** — the harness computes both and flags memory-bound shapes. That decode
  row's 2.0 TFLOPS looks catastrophic and is not: 2027 GB/s is close to what the memory
  system can deliver. Judge memory-bound shapes on bandwidth.
- **`spread%`** — max-min across independent samples, as a percentage of the median. Treat
  it as your noise floor: an improvement smaller than the spread is not measurable. Small
  shapes are noisiest; raising `--repeats`/`--rep` on the 1024³ case took it from 36% to 6%.
- **`err` / `SNR`** — relative gate (`err_ratio < 0.05`) and signal-to-noise in dB. Absolute
  error is deliberately not reported; it grows with K on correct kernels and cannot gate.
  A correct bf16 GEMM sits at ~55.6 dB regardless of K; fp16 sits near 74 dB.
- **`status`** — `FAIL-CORRECTNESS`, `IMPLAUSIBLE>peak` (above hardware peak, i.e. a broken
  harness — usually a missing synchronize), `NOISY` (spread >10%), or `ok`.

Exit status is non-zero if any case fails a gate, so it can be used in a check script.

## Extending

Add a generator function to `shapes.py` and register it in `_GENERATORS`. Keep regimes
meaningful: a new regime should be one where a *different config wins*, not merely a
different size.

When real serving shapes are available, capture them from the live workload (see the
per-framework skills) and tune those — they are strictly better targets than generated
ones. This corpus is the fallback and the smoke test.

## Architecture note

Generated for gfx942 (MI300X) and confirmed on gfx950 (MI355X). Both rows of `DTYPES_BY_ARCH`
in `shapes.py` are now measured on hardware — each dtype allocates *and* completes a GEMM,
which are separate checks. The shape families carry over unchanged; the dtypes and the tuned
results do not.

Two gfx950 notes that cost time to find. FP8 support **inverts** rather than extends: gfx950
computes OCP and refuses FNUZ, the mirror of gfx942. And FP4 needs `torch.empty`, not
`torch.zeros` — the latter raises `fill_cuda not implemented` on a part that has a working FP4
matrix core, so the obvious probe reports the dtype missing exactly where it exists.
