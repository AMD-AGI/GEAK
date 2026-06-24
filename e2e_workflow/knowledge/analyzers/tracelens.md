<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
See LICENSE for license information.
-->
# Analyzer recipe: TraceLens (default)

How to turn a PyTorch-profiler trace into the canonical `top_kernels.json` (see `_contract.md`) using the
public **TraceLens** package. This is the SINGLE place TraceLens specifics live — if TraceLens changes its
CLI/flags/columns, edit only this file.

> Inputs available to you: `EVAL_DIR`, `GPU_IDS`, `TRACELENS_INSTALL` (pip spec, optional), `OUTDIR`
> (default `${EVAL_DIR}/analysis`). All paths are discovered/derived — hardcode nothing.

## Step 1 — ensure TraceLens is importable (install if missing; non-fatal)
```bash
python3 -c "import TraceLens" 2>/dev/null \
  || pip install --quiet "${TRACELENS_INSTALL:-git+https://github.com/AMD-AGI/TraceLens.git}"
```
If the import still fails after install (no network / no access), STOP and return `{"ok":false,
"note":"TraceLens unavailable"}`. Do not hardcode a local checkout path.

## Step 2 — find the torch trace the Profiler already captured
The profiler drops a Chrome/torch trace under `${EVAL_DIR}/profile/`. Prefer the model-execution rank 0:
```bash
# Use find (no globstar dependency). Prefer the model-execution rank-0 torch trace.
TRACE=$(find "${EVAL_DIR}/profile" -name '*rank0*.pt.trace.json.gz' 2>/dev/null | sort | tail -1)
[ -z "$TRACE" ] && TRACE=$(find "${EVAL_DIR}/profile" \( -name '*.pt.trace.json.gz' -o -name '*.json.gz' -o -name '*.json' \) 2>/dev/null | grep -v async_llm | sort | tail -1)
```
If none exists, return `{"ok":false,"note":"no torch trace under EVAL_DIR/profile"}`.

## Step 3 — detect arch (bundled spec only; omit if unknown)
The public package bundles only **MI300X** and **MI325X** (`--gpu_arch_platform`). Detect the on-box card
(`rocminfo` / `rocm-smi --showproductname`, or reuse `${EVAL_DIR}/env_report.json`). Map gfx942→MI300X/MI325X.
If the card is not one of the bundled platforms (e.g. gfx950/MI355X), **omit `--gpu_arch_platform`** — the
report still gives time-share + shapes, only roofline % is skipped. Never hardcode a platform.

## Step 4 — run the inference perf report (it builds a trace tree; large traces are slow)
```bash
OUT="${OUTDIR:-$EVAL_DIR/analysis}"; mkdir -p "$OUT/perf_report_csvs"
python3 -m TraceLens.Reporting.generate_perf_report_pytorch_inference \
    --profile_json_path "$TRACE" \
    --output_csvs_dir   "$OUT/perf_report_csvs" \
    ${ARCH_PLATFORM:+--gpu_arch_platform "$ARCH_PLATFORM"}
```
Very large traces (millions of kernels) build a heavy tree and can be slow; you are time-bounded by the
agent timeout — if it does not finish, return `{"ok":false,"note":"tracelens report timed out"}`.

## Step 5 — normalize to the canonical `top_kernels.json` (column mapping lives here)
Read the produced CSVs in `$OUT/perf_report_csvs/` and map to the contract schema:
- **time share** ← `unified_perf_summary.csv` (preferred) or `ops_summary.csv`
  (`name`, `Percentage (%)`→`pct`, `total_direct_kernel_time_sum`(µs)→`gpu_time_us`, `Count`→`count`,
  `Categories`→`category`).
- **shapes** ← GEMM/MoE rows (`GEMM.csv` `param: M/N/K/dtype_A_B`) or `ops_unique_args.csv`.
- **roofline** ← per-category CSV columns (`Roofline Bound`, `TFLOPS/s` vs peak) when arch was passed.
- **total_gpu_time_us** ← sum, or `gpu_timeline.csv`.
Write `$OUT/top_kernels.json`. Keep `raw` CSVs in place for reference.

## Step 6 — return
`{"ok": true, "top_kernels_path": "$OUT/top_kernels.json", "summary_md_path": "$OUT/summary.md", "note": "..."}`
(the executor writes `summary.md` from `top_kernels.json` per `_contract.md`).
