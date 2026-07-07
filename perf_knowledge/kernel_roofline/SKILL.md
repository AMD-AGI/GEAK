---
name: kernel_roofline
description: >
  Use when you need to roofline-profile a single GPU kernel from its unit
  test / performance harness on AMD hardware (CDNA / MI-series) with
  rocprof-compute. Produces arithmetic intensity, achieved-vs-peak HBM
  bandwidth and compute throughput, so you can tell whether a kernel is
  memory-bound or compute-bound before optimizing. Covers rocprof-compute
  install, the lightweight roof-only flow, and how it differs from a full
  rocprof-compute profile.
---

# Roofline — kernel roofline profiling KB

Given a kernel and the command that runs its unit test (its "performance"
harness), this skill collects a **roofline** with AMD `rocprof-compute` and
reports where the kernel sits relative to the GPU's peak compute and peak
memory bandwidth.

The helper `scripts/run_roofline.py` is self-contained: it shells out to
`rocprof-compute` directly and parses the roofline block, with no dependency
beyond the Python standard library and a working ROCm install.

## When to use

- You have a working kernel + a unit-test / perf command and want to know
  if it is **memory-bound or compute-bound** before spending effort.
- You want achieved arithmetic intensity (FLOP/byte), achieved HBM
  bandwidth %, and achieved compute % for the kernel's hot function.
- Target hardware is AMD **CDNA** (MI100/200/300...). RDNA is *not*
  supported by `rocprof-compute` roofline — see Limitations.

## Prerequisites: install rocprof-compute

`rocprof-compute` ships with ROCm (`/opt/rocm`) but is often not installed by
default (ROCm provides `rocprofv3`/`rocprof`, not `rocprof-compute`):

```bash
# 1. Is it already there?
rocprof-compute --version          # prints "rocprofiler-compute version: X.Y.Z"

# 2. If missing, install the package
sudo apt install rocprofiler-compute

# 3. Expose the binary on PATH
sudo update-alternatives --install /usr/bin/rocprof-compute \
     rocprof-compute /opt/rocm/bin/rocprof-compute 0

# 4. Install its Python deps
python3 -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt

# 5. Verify
rocprof-compute --version
```

Notes:
- Profiling needs access to the GPU performance counters. Run as a user
  with counter access (often root / `video` + `render` groups), or the
  profile step fails with a permissions error.
- Pin to one device with `HIP_VISIBLE_DEVICES=<idx>` so the harness and
  the profiler agree on which GPU to use.
- `rocprof-compute >= 3.3.1` supports the lightweight `--roof-only` path;
  older versions fall back to a full profile (handled in the script).

## Troubleshooting: common failures

These bit a real MI300X/gfx942 + ROCm 7.2.3 run; check them first when a
profile produces no numbers.

### "No profiling data found" / empty roofline (rocprof-compute 3.4.0 + pandas 3.x)

rocprof-compute 3.4.0 predates pandas 3, which defaults to the **pyarrow string
backend** (`future.infer_string=True`). Counter CSVs then load as immutable
`str` columns, which silently breaks two steps: the profile-time v3→v2 CSV
conversion writes *no* `pmc_perf.csv` ("Cannot write results ... no counter csv
files generated", "merge on str and int64 columns for key 'Agent_Id'"), and
`analyze` later dies in `eval_metric` (`Invalid value '0.0' for dtype 'str'`).
The symptom that surfaces is just `[analysis] No profiling data found.`

Two patches to the installed rocprof-compute (back up the files first):

1. Disable the arrow string default at the entry point. Edit the **real file
   behind the symlink** — `readlink -f $(command -v rocprof-compute)`, e.g.
   `/opt/rocm-<ver>/libexec/rocprofiler-compute/rocprof-compute` — and add near
   the top of the imports:

   ```python
   try:
       import pandas as _pd
       _pd.set_option("future.infer_string", False)
   except Exception:
       pass
   ```

2. Fix the `Agent_Id` dtype check in
   `<libexec>/rocprofiler-compute/utils/utils.py`
   (`v3_counter_csv_to_v2_csv`). It only converts `"Agent N"`→int when
   `dtype == "object"`, which is False for arrow strings, so the int64-vs-str
   merge crashes. Make it type-agnostic:

   ```python
   # was: if result["Agent_Id"].dtype == "object":
   if not pd.api.types.is_integer_dtype(result["Agent_Id"]):
       ... .apply(lambda x: int(re.search(r"Agent (\d+)", str(x)).group(1))) ...
   ```

Patch #1 alone fixes most of it; #2 is still needed for the conversion merge.
A cleaner long-term fix is to pin `pandas<3` in rocprof-compute's environment.

### "rocprof-compute: command not found" (not installed)

See Prerequisites — install `rocprofiler-compute`, expose it via
`update-alternatives`, and install its `requirements.txt`, then re-check
`rocprof-compute --version`.

### Profiler SIGSEGV on large-grid kernels (`aqlprofile_pmc_iterate_data`)

Some kernels crash rocprof-compute's PMC replay (`SIGSEGV` /
`terminate ... std::runtime_error` in `aqlprofile_pmc_iterate_data`) at large
launch grids — independent of dispatch count. If a kernel aborts, profile it at
a **smaller but representative shape** (AI / bandwidth / bound are usually
shape-stable) via a tiny driver that launches the kernel a few times, and note
the reduced shape in the output.

## Quick start

```bash
# Roofline a kernel whose unit test runs via its perf harness, on GPU 1
python3 scripts/run_roofline.py \
    --workdir /path/to/kernel_dir \
    --cmd "python3 test_my_kernel.py" \
    --output ./roofline_out \
    --gpu 1 \
    --hbm-peak-const 5300        # optional: fixed HBM peak for cross-run util

# Target one kernel when setup/RNG kernels dominate a 1-iter --profile run
python3 scripts/run_roofline.py \
    --workdir /path/to/tests \
    --cmd "python3 test_tilelang_main_kernel.py --profile --filter mqa_logits_prefill" \
    --gpu 1 --kernel mqa_logits --output ./roofline_out
```

`--workdir` is the directory the unit test runs from; `--cmd` is the command
that actually launches the kernel (a pytest invocation, a standalone
`python3 your_test.py`, etc.). Results land in `--output`:

- `<name>_roofline.txt` — parsed summary **+ empirical Roofline Eff. block**
- `<name>_roofline_raw.txt` — raw `rocprof-compute analyze -b 4` output
- `<name>_roofline_error.txt` — only on failure

The helper script is the runnable reference; read it for the exact API.

> Profiling multiple kernels from the **same** `--workdir` overwrites results
> unless you pass a distinct `--name` each time (the default name is the
> workdir basename).

## Modes (depth of analysis)

| `--mode` | profile / analyze | output |
|---|---|---|
| `roofline` (default) | `--roof-only` / `-b 4` | roofline summary + **empirical efficiency** |
| `full` | full / no `-b`, `--output-format txt` | raw persistent report across all blocks |

## Beyond roofline (`--mode full`)

Roofline tells you *which wall* you hit; the other rocprof-compute sections tell
you *why*. `--mode full` runs `analyze` with no `-b` filter, so it dumps **every
section as raw text** (the script does not parse these — read the report). When
chasing a low Roofline Eff., read these blocks:

| block | section | what it tells you |
|---|---|---|
| 2 | Speed-of-Light | % of peak per resource (VALU / MFMA / mem BW / LDS / occupancy) — fastest bottleneck glance |
| 7 | Wavefront | register / LDS allocation, **occupancy**, instrs per wave |
| 10–11 | Compute Units | instruction mix (VALU/MFMA/VMEM/SALU), FLOP counts, IPC |
| 16 | Vector L1 Cache | L1 hit rate, **coalescing**, L1→L2 stalls |
| 17 | L2 Cache | L2 hit rate, bandwidth, latency |

Typical reads: memory-bound + low Eff. → check 16/17 (cache hit, coalescing);
compute-bound + low Eff. → check 2/7 (occupancy, register pressure).

## Empirical Roofline efficiency (roofline mode)

In roofline mode the script post-parses the roofline block and, **per
non-noise kernel**, derives — all from the *same* rocprof roofline block, so
the peaks are this-run empirical, not datasheet:

```
Peak_BW_emp   = HBM Bandwidth "Peak (Empirical)" column        [GB/s]
Peak_Compute  = empirical peak of the kernel's dominant achieved
                dtype (see "Choosing Peak_Compute" below)      [GFLOP/s]
AI_HBM        = "AI HBM" arithmetic intensity                  [FLOP/byte]
Perf          = achieved "Performance"                         [GFLOP/s]

attainable    = min(Peak_Compute, AI_HBM × Peak_BW_emp)        [GFLOP/s]
Roofline Eff. = Perf / attainable
ridge(emp)    = Peak_Compute / Peak_BW_emp                     [FLOP/byte]
                AI_HBM < ridge → memory-bound; AI_HBM > ridge → compute-bound
```

Choosing `Peak_Compute` — **dtype-based** `--compute-peak auto` (default): the
script scans every FLOP/IOP rate metric and picks the empirical peak of the
dtype the kernel *actually ran on* (the dominant **achieved** counter). So a
BF16 attention/GEMM uses the MFMA-BF16 peak, an fp8 GEMM the MFMA-F8 peak, an
elementwise kernel the VALU-F32 peak — automatically, no per-kernel hinting.

`rocprof-compute` reports **no empirical peak for MFMA F6F4** (fp4/fp6) on some
versions. When the dominant dtype is F6F4 with an `N/A` peak, the script
**estimates** it as `2 × peak(MFMA-F8 empirical)` (fp4/fp6 packs ~2× the matrix
throughput of fp8); the peak label then reads `mfma_f6f4,est=2xF8emp` so the
estimate is explicit.

Force a specific dtype with `--compute-peak mfma_bf16|mfma_f8|valu_f32|...`
(any key in `COMPUTE_METRICS`). Or bypass the empirical peak entirely with
`--compute-peak-const <GFLOP/s>` — a fixed datasheet ceiling (e.g. a datasheet
fp4/fp8 MFMA peak), useful when the empirical peak is missing or unreliable.
`--compute-peak-const` overrides `--compute-peak`.

`HBM util (real)` = `achieved_HBM_BW / fixed machine-constant peak`
(`--hbm-peak-const`, e.g. `5300` for MI300X). It is reported for **cross-run
comparison only and does NOT enter Roofline Eff.** — the efficiency uses the
per-run empirical HBM peak, so a noisy peak measurement can't inflate it.
(Note the empirical HBM peak measured during a roofline run is typically well
below datasheet, e.g. ~4200 vs 5300 GB/s, because the roofline microbenchmark
does not fully saturate HBM — that's why util-real and Roofline Eff. differ.)

Reading it: **Roofline Eff. near 1.0** ⇒ at the empirical ceiling (the named
bound is the wall to attack). **Eff. ≪ 1.0 while memory-bound** ⇒ chase HBM
traffic/coalescing/reuse to lift AI past the ridge. **Eff. ≪ 1.0 while
compute-bound** ⇒ raise occupancy / use MFMA / cut redundant FLOPs.

## How roofline mode works (under the hood)

Roofline mode runs **two** `rocprof-compute` steps against the kernel's hot
function:

```bash
# Step 1 — collect ONLY roofline counters (single replay pass, fast)
rocprof-compute profile -n <name> --path <tmpdir> --roof-only -- <your unit test cmd>

# Step 2 — print ONLY the roofline analysis block (block 4)
rocprof-compute analyze -p <tmpdir> -b 4
```

The script then parses block `4.1 Roofline Rate Metrics` (achieved vs peak
HBM bandwidth and FLOP/IOP rates) and `4.2 Roofline AI Plot Points`
(arithmetic intensity + achieved performance) into a summary like:

```
kernel function name:
- <kernel>
HBM BANDWIDTH UTILIZATION:
- ... actual / peak / utilization_pct
COMPUTE UTILIZATION:
- ... actual / peak / utilization_pct
ARITHMETIC INTENSITY:
- Arithmetic Intensity: value FLOPs/byte
- Performance (TFLOPs): value
```

Interpretation: low AI + high HBM% ⇒ **memory-bound** (improve access
pattern / coalescing / reuse). High AI + high compute% ⇒ **compute-bound**
(reduce work, use MFMA, raise occupancy). Both low ⇒ **latency/occupancy
bound**.

## Difference from the full-profile commands

A common manual full-profile flow:

```bash
rocprof-compute profile -n full_run --path workloads/full_run \
    -- python3 profile_my_kernel.py
rocprof-compute analyze -p workloads/full_run \
    --output-format txt --output-name my_full_report
```

That is a **full profile**, not the roofline-only path. The differences:

| | Roofline mode (this skill) | Full profile |
|---|---|---|
| Profile flag | `--roof-only` | *(none)* — collects every counter |
| Replay passes | one (roofline counters only) | many (counters collected in groups) |
| Speed | fast | slow (re-runs the kernel many times) |
| Analyze scope | `-b 4` (roofline block only) | no `-b` ⇒ all blocks (Top stats, SoL, instruction mix, L1/L2, wavefront, roofline, ...) |
| Output | stdout, parsed to a summary | `--output-format txt --output-name` ⇒ persistent named `.txt` report file |
| Use it for | quick "memory- or compute-bound?" verdict | deep dive across all bottleneck sections |

So the full profile answers "everything about this kernel" at the cost of many
replay passes; roofline mode answers just "where is it on the roofline" in a
single pass. Get the full-report behavior from the helper with `--mode full`:

```bash
python3 scripts/run_roofline.py --mode full \
    --workdir workloads --name full_run \
    --cmd "python3 profile_my_kernel.py" \
    --output ./roofline_out
# -> ./roofline_out/full_run/full_run_report.txt
```

## Reading the output

- **Arithmetic Intensity (FLOP/byte)** — x-axis on the roofline plot.
- **Performance (TFLOPs)** — y-axis; compare to the peak compute ceiling.
- **HBM BANDWIDTH UTILIZATION utilization_pct** — how close to peak DRAM BW.
- **COMPUTE UTILIZATION utilization_pct** — how close to peak FLOP/IOP rate.
- By default the summary reports **every** kernel with a roofline block —
  nothing is filtered by name, so a rocBLAS `Cijk_` GEMM or an `at::native`
  kernel that *is* the kernel under test is never silently dropped. Kernel
  names come from the `analyze` block **headers** (`Kernel N: <name> (pct%)`),
  which are inherently 1:1 and in-order with the parsed 4.1/4.2 blocks;
  `pmc_kernel_top.csv` is only an optional fallback to un-truncate a long name.

### Filtering / targeting a specific kernel

- `--skip-noise` — opt-in dropping of genuine framework input-generation / init
  kernels (RNG fill, HIP memset, `arange`; see `NOISE_PATTERNS` in the script).
  Off by default. It deliberately does **not** list rocBLAS / `at::native`
  compute kernels, since those are often the kernel under test.
- `--kernel <substr>` — roofline **only** the kernel(s) whose name contains this
  (case-sensitive) substring. This is the right tool when setup/RNG kernels
  dominate a 1-iteration `--profile` run and bury the kernel under test (e.g. a
  tilelang / Triton kernel drowned by dozens of `torch.randn`/sort/fill
  kernels). It resolves the substring to rocprof kernel id(s) via
  `rocprof-compute analyze -p <dir> --list-stats`, then re-runs the block-4
  analyze with `-k <id> [<id> ...]`. If the substring matches no kernel, the
  script prints the available kernel names and exits nonzero. Default (no
  `--kernel`): analyze all kernels.

## Empirical vs datasheet peaks (`--peaks`)

The efficiency can be driven by two kinds of ceiling:

- **empirical** (default) — per-run peaks parsed from the roofline block
  (`Peak (Empirical)` column + the fp4=2×fp8 estimate). These come from
  rocprof's on-device microbenchmark and run **well below datasheet** because
  the microbench does not fully saturate the units.
- **datasheet** — fixed vendor peaks from `DATASHEET_PEAKS[arch]` in the script.
  For **gfx950 / MI350–MI355** (dense, no sparsity):

  | resource | peak |
  |---|---|
  | HBM | 8 TB/s |
  | MFMA FP8 | 5 PFLOP/s |
  | MFMA FP4/FP6 (F6F4) | 10 PFLOP/s (2× FP8) |
  | MFMA BF16 / FP16 | 2.5 PFLOP/s |

  The datasheet compute roof is selected by the kernel's **dominant achieved
  FP metric** (`MFMA FLOPs (F8)`→FP8, `(BF16)`→BF16, `VALU FLOPs (F32)`→F32,
  …), exactly as the empirical path does.

`--peaks {empirical,datasheet,both}` — **default `both`**: every run prints an
empirical block **and** a datasheet cross-check per kernel, i.e. the built-in
double-check is on by default (pass `empirical`/`datasheet` to get just one).
Because empirical fp8/fp4 peaks measured here are ≈ half datasheet,
**empirical Roofline Eff. ≈ 2× datasheet Eff.** for fp8/fp4 GEMMs; the two
ceilings agree on the **bound classification** (memory- vs compute-bound), which
is the robust takeaway. Datasheet Eff. is the conservative number to quote.

Formula (identical for both, per §10 of the pure-agent flow):

```text
ceiling      = min(dtype_compute_peak, AI_HBM * HBM_peak)
roofline_eff = achieved_perf / ceiling
```

## Multi-shape aggregation by coverage weight

A single kernel runs many shapes in a real workload. To roofline the *kernel*
(not one shape), profile each representative shape (one `--kernel`-targeted run
per shape), then aggregate by that shape's **coverage weight** — its share of
real dispatches / time from the serving/profiler trace (not from the roofline
run itself):

```text
weighted_metric = sum(metric(shape_i) * coverage_i) / sum(coverage_i)
```

Report, per kernel: number of profiled shapes; cumulative shape coverage;
weighted Roofline Eff.; weighted achieved TFLOP/s; weighted HBM GB/s; weighted
AI; and the per-shape roofline details. (The DSV4 `kernel_unit_tests/shapes/*.json`
give the shape *variants* per kernel; the coverage weights come from the kineto/
profiler trace that recorded how often each shape actually ran.)

## Pure-agent flow (no helper script)

The same result can be produced by an agent driving `rocprof-compute` directly —
the method the helper script encodes:

1. `rocprof-compute profile ... --roof-only -- <bench cmd>` on each **unit test**,
   never on a live multiprocess serving server (profiling wait-loops destabilizes
   startup and pollutes the roofline).
2. `rocprof-compute analyze -p <dir> -b 4 > raw.txt`.
3. Parse `raw.txt` **by block**: each `Kernel N: <name> (pct%)` header owns the
   `4.1 Roofline Rate Metrics` and `4.2 Roofline AI Plot Points` immediately
   below it — take name and metrics from the **same block**. Never `zip`
   `pmc_kernel_top.csv` names against the analyze text (order need not match).
4. Compute efficiency with the peaks above; aggregate by coverage weight.

## Limitations / pitfalls

- **RDNA is unsupported.** The script rejects RDNA archs (gfx10xx/11xx/12xx);
  use a CDNA / MI-series GPU.
- **RNG / setup pollution.** If the harness fills inputs with
  `torch.randn` on-GPU, those PyTorch kernels show up as top kernels and can
  outweigh the target (common in a 1-iteration `--profile` run). Either
  pre-generate inputs on CPU / outside the timed region, or target the kernel
  under test directly with `--kernel <substr>` (see above).
- **int8 (MFMA IOPs) kernels report Roofline Eff. ≈ 0.** The roofline is
  **FLOP-based**: an int8 GEMM does integer MFMA ops (MFMA IOPs), so rocprof's
  `Performance` (FLOP/s) reads ≈ 0 and AI/Perf collapse. The HBM-BW and
  IOP-utilization numbers are still meaningful, but the derived **Roofline Eff.
  is not** for int8 kernels — read the achieved IOP rate / bandwidth instead.
- **Counter access required.** A permission failure in step 1 surfaces as a
  profile error in `<name>_roofline_error.txt`.
- The unit-test command must actually **launch the kernel on the GPU**; a
  correctness-only test that exits before the kernel runs yields no roofline.

## Files

- `scripts/run_roofline.py` — CLI driver (roofline + full modes), self-contained.
