# Profile Parsing — the Standardized Top-N Contract

The Profile phase MUST produce ONE canonical artifact so every downstream agent reads the
bottleneck identically. The tool is `scripts/parse_profile.py`; this file is its contract.

## How to produce it
```bash
# torch/sglang profiler trace → standardized Top-N with per-call analysis:
python3 $WF_DIR/scripts/parse_profile.py --torch-trace <trace.json.gz> --top 25 --out $EVAL_DIR/profile_topN

# force stdlib-only parsing (skip TraceLens even if installed):
python3 $WF_DIR/scripts/parse_profile.py --torch-trace <trace.json.gz> --top 25 --no-tracelens --out $EVAL_DIR/profile_topN
```
Writes `profile_topN.json` (canonical schema) + `profile_topN.md` (human table).

When TraceLens is installed, the parser automatically uses tree-based CPU→GPU linking, per-shape
breakdown, and roofline analysis. Without TraceLens, it falls back to stdlib-only flat-scan parsing
(shapes via `External id` linkage, no roofline).

## Canonical schema (profile_topN.json)
```
{ source, tracelens: bool,
  total_gpu_time_ms, num_kernel_launches, num_distinct_kernels,
  top_kernels: [ {
    rank, name, short_name, calls, total_ms, avg_us,
    pct_gpu_time,              // percentage of total GPU time
    shapes[], dtypes[],
    classification, backend_guess, editable, opt_hint,

    // per-call distribution (present when n≥2; informational — no de-inflation)
    per_call: { n, median_us, mean_us, std_us, min_us, max_us,
                p10_us, p90_us, p99_us, cov,
                distribution_type: "stable|moderate|high_variance" },

    // TraceLens-only fields (absent when tracelens=false)
    cpu_op,                    // the CPU op name (e.g. aten::mm)
    roofline: { tflops_s, tb_s, flops_byte, bound, pct_roofline },
  } ] }
```

## The classification field (this is the triage signal the Architect routes on)
- `library_gemm` — hipBLASLt/Tensile/rocBLAS GEMM. **Not source-editable.** Route to Config Tuner
  (backend/env/heuristics swap: aiter vs hipBLASLt vs CK GEMM, tuning DB) — NOT to the kernel squad.
- `library_attn` — CK/AITER/FlashAttn paged attention. Route to Config Tuner (`--attention-backend`
  swap, per-shape backend). Source-edit only if it resolves to a Triton attention.
- `triton` / `fused_custom` — **editable.** Route to Kernel Extractor → kernel squad. This is where
  the recursive single-kernel kernel_workflow runs.
- `elementwise_overhead` — fill/cast/activation/copy. Route to host_runtime fusion (Lever 1) or
  config (e.g. enable fused activation). Often cheap per-call but high call count.
- `reduction_norm` — rmsnorm/rope/softmax. Editable (often Triton); candidate for fusion.
- `memory` — memcpy/memset. Reduce via native layouts.
- `other` — inspect source to route (the Profiler should try to resolve these before finishing).

## Reading the result (how the Architect should think)
1. **Amdahl first.** A kernel at 52% gpu time with a plausible 1.3x is worth far more than a 5x on a
   2% kernel. Rank candidates by `pct_gpu_time × achievable_speedup × editable`.
2. **GEMM/attn usually dominate** prefill (big M). They are library calls → the highest-ROI early
   move is the Config Tuner sweep (backend/quant/tuning), NOT a source rewrite.
3. **Editable Triton/custom kernels** (mamba/gated-delta, norms, activations) are where the kernel
   squad earns its keep. Carry their `shapes` into the Extractor so the unittest replays real shapes.
4. **Check per-call distribution** (`per_call` field). In the TraceLens path, each entry is already
   scoped to a specific (name, Input Dims, Input type) group, so per-call durations within a group
   are directly comparable. A `high_variance` (CoV > 1.0) entry within a single shape group signals
   genuine instability (warmup outliers, JIT compilation, scheduling jitter). In the stdlib path,
   `per_call` groups by kernel name only (no shape separation) — `high_variance` there may simply
   reflect different shapes (e.g. prefill vs decode).
5. **High call-count tiny kernels** (e.g. elementwise at 1000s of calls) signal dispatch overhead →
   host_runtime fusion / cuda-graph.
6. **Roofline** (TraceLens only): when `roofline.pct_roofline > 85%`, the kernel is near hardware
   peak — optimization headroom is limited. When `roofline.bound == "MEMORY_BOUND"`, focus on
   fusion/memory access patterns rather than compute tuning.

## Per-call distribution analysis

`parse_profile.py` automatically computes per-call distribution statistics for every top kernel.
The stats are **informational only** — `pct_gpu_time` is NOT de-inflated.

**TraceLens path** (recommended): each top entry is already grouped by `(name, Input Dims, Input
type)` via TraceLens's `summarize_df_unified_perf_table()`. Per-call durations are extracted from the
unsummarized DataFrame, so each group contains only calls with the SAME shape. This means:
- `high_variance` (CoV > 1.0) in a single shape group is genuine instability
- Different shapes (prefill M=4096 vs decode M=1) are in SEPARATE entries, not mixed

**Stdlib path**: groups by kernel name only (no shape separation). Per-call durations may mix
different shapes under one name. `high_variance` here could simply reflect different shapes.

Fields in `per_call`:
- `n` — number of calls
- `median_us`, `mean_us`, `std_us` — central tendency and spread
- `min_us`, `max_us`, `p10_us`, `p90_us`, `p99_us` — range and percentiles
- `cov` — coefficient of variation (std/mean). `< 0.3` = stable, `> 1.0` = high variance
- `distribution_type` — `"stable"` (CoV < 0.3), `"moderate"` (0.3–1.0), `"high_variance"` (> 1.0)

### Manual deep-dive (optional)
```bash
# Extract per-call durations for a specific kernel from torch trace:
python3 -c "
import json, gzip, sys
path, core = sys.argv[1], sys.argv[2]
opener = gzip.open if path.endswith('.gz') else open
with opener(path, 'rt') as f:
    data = json.load(f)
events = data.get('traceEvents', data if isinstance(data, list) else [])
ds = [e['dur'] for e in events if isinstance(e, dict)
      and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')
      and core in e.get('name','') and 'dur' in e]
if len(ds) > 20:
    ds.sort(); n = len(ds); q = lambda p: ds[min(n-1,int(n*p))]
    m = ds[n//2]; mean = sum(ds)/n; std = (sum((x-mean)**2 for x in ds)/(n-1))**0.5
    print(f'n={n} median={m:.1f}us mean={mean:.1f}us cov={std/mean:.2f} '
          f'p10={q(.10):.1f} p90={q(.90):.1f} p99={q(.99):.1f} max={ds[-1]:.1f}us')
" <trace.json.gz> 'cross_device_reduce'
```

## `record_shapes` — required for shape enrichment

`parse_profile.py` links GPU kernel events to CPU ops via `External id` and reads `Input Dims` /
`Input type` from those CPU ops to populate each kernel's `shapes[]` and `dtypes[]`. **This data is
ONLY present when the torch profiler is started with `record_shapes=True`.**  Without it, every
kernel in the Top-N will have `shapes: []` — downstream agents (Extractor, kernel squad, op
benchmarker) lose the shape context they need for regime-specific unittests and shape-targeted tuning.

### How each backend enables it

**sglang** — enabled by default, no action needed.
The env var `SGLANG_PROFILE_RECORD_SHAPES` defaults to `True` (see `sglang/srt/environ.py`). When
`/start_profile` is called (which is what `bench_serving.py --profile` does), the HTTP/gRPC handler
reads this env var and passes `record_shapes=True` to `torch.profiler.profile()`. To explicitly
control it:
```bash
# already True by default — only set to override:
SGLANG_PROFILE_RECORD_SHAPES=true   # env var (default True)
# or in the /start_profile API body:
curl -X POST http://host:port/start_profile -d '{"record_shapes": true, "num_steps": 5}'
```
If the trace shows no `Input Dims` (see verification below), check the code path:
`srt/managers/tokenizer_control_mixin.py` → `srt/managers/scheduler_profiler_mixin.py:start_profile()`
→ `torch.profiler.profile(record_shapes=...)`. The fallback is `False` when `record_shapes is None`
(line 195 in `scheduler_profiler_mixin.py`), so the env var / API body must reach the scheduler.

**vllm** — **defaults to `False`, must explicitly enable.**
`ProfilerConfig.torch_profiler_record_shapes` defaults to `False` (see `vllm/config/profiler.py:62`).
The profiler config is set at **server launch** time, not at profile trigger time. Enable it by
adding to `EXTRA_SERVER_ARGS`:
```bash
# Option 1: via server launch flags (recommended for e2e_workflow)
vllm serve $MODEL ... \
    --profiler-config profiler=torch \
                      torch_profiler_dir=$PROFILE_DIR \
                      torch_profiler_record_shapes=true

# Option 2: via Python API
from vllm.config import ProfilerConfig
profiler_config = ProfilerConfig(
    profiler="torch",
    torch_profiler_dir="/path/to/traces",
    torch_profiler_record_shapes=True,   # <-- REQUIRED
)
```
The code path: `config/profiler.py:ProfilerConfig` → `profiler/wrapper.py:TorchProfilerWrapper` →
`torch.profiler.profile(record_shapes=profiler_config.torch_profiler_record_shapes)` (line 221).

**Standalone torch profiler** (manual profiling, or any custom backend):
```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,        # <-- populates "Input Dims" / "Input type" per cpu_op
    with_stack=True,           # optional: adds Python stack frames for source mapping
    profile_memory=False,      # optional: memory tracking (heavier)
) as prof:
    # run the workload (e.g. a few forward passes)
    for _ in range(5):
        model(input_ids)

prof.export_chrome_trace("trace.json")
# gzip it for parse_profile.py:  gzip trace.json
```

### How to verify a captured trace has shapes
```bash
python3 -c "
import json, gzip, sys
path = sys.argv[1]
opener = gzip.open if path.endswith('.gz') else open
with opener(path, 'rt') as f:
    data = json.load(f)
events = data.get('traceEvents', data if isinstance(data, list) else [])
cpu_ops = [e for e in events if isinstance(e, dict) and e.get('cat') == 'cpu_op']
with_dims = [e for e in cpu_ops if e.get('args', {}).get('Input Dims')]
print(f'cpu_op events: {len(cpu_ops)}, with Input Dims: {len(with_dims)}')
if not with_dims:
    print('WARNING: record_shapes was NOT enabled — shapes will be empty in the Top-N')
else:
    print('OK: record_shapes is active')
    # show an example
    ex = with_dims[0]['args']
    print(f'  example Input Dims: {ex.get(\"Input Dims\")}')
    print(f'  example Input type: {ex.get(\"Input type\")}')
" <trace.json.gz>
```

## TraceLens enhancement (optional)

When [TraceLens](https://github.com/AMD-AGI/TraceLens) is available, `parse_profile.py` automatically
uses its tree-based analysis for enhanced results. **If TraceLens is not installed, `parse_profile.py`
will attempt to `pip install TraceLens` automatically on first run.** Only if both the auto-install
and the import fail does it fall back to stdlib-only parsing.

- **Tree-based CPU→GPU linking**: handles graph launches and nested ops (more accurate than flat-scan
  `External id` linkage)
- **Per-shape breakdown**: each unique `(op_name, Input Dims, Input type)` group gets independent
  GPU-time stats → resolves the "5 shapes lumped together" problem
- **Roofline analysis**: TFLOPS/s, TB/s, FLOPS/Byte, compute- vs memory-bound classification,
  percent of hardware peak (`pct_roofline`)

TraceLens is **NOT a hard dependency** — if auto-install fails (e.g. no network, no pip),
`parse_profile.py` falls back to stdlib-only parsing (flat scan + `External id` shape enrichment).
The per-call distribution analysis works in both modes (it reads torch trace events directly, no
TraceLens needed).

The `tracelens` field in the output indicates which path was used.

### Install
`parse_profile.py` auto-installs TraceLens on first run if not already present. It searches in
order: (1) local sibling clone at `../TraceLens` relative to GEAK, (2) `~/TraceLens`, (3) git clone
from `https://github.com/AMD-AGI/TraceLens.git`. **Do NOT `pip install TraceLens`** — the PyPI
package with that name is a different project. To pre-install manually:
```bash
# from local clone (recommended):
pip install -e /path/to/TraceLens

# or clone + install:
git clone https://github.com/AMD-AGI/TraceLens.git
pip install -e TraceLens/
# TraceLens depends on numpy + pandas; they will be pulled in automatically.
```

### Usage with parse_profile.py
```bash
# TraceLens auto-detected — produces roofline + shape_breakdown in the output:
python3 $WF_DIR/scripts/parse_profile.py --torch-trace trace.json.gz --top 25 --out profile_topN
# → output includes: "tracelens": true, roofline metrics, shape_breakdown per kernel

# Force stdlib-only (skip TraceLens even if installed):
python3 $WF_DIR/scripts/parse_profile.py --torch-trace trace.json.gz --top 25 --no-tracelens --out profile_topN
# → output includes: "tracelens": false, no roofline or shape_breakdown
```

### Standalone TraceLens usage (outside parse_profile.py)
```python
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

# Build the analysis tree from a torch profiler trace
analyzer = TreePerfAnalyzer.from_file("trace.json.gz")

# Get the full per-op unified perf table (DataFrame)
df = analyzer.build_df_unified_perf_table(include_nccl=True)
print(df[["name", "op category", "Kernel Time (µs)_sum", "Percentage (%)",
          "Input Dims", "Input type"]].head(20))

# Get the summarized table grouped by (name, Input Dims, Input type)
summary = analyzer.summarize_df_unified_perf_table(df)
print(summary.head(10))

# Roofline columns in the DataFrame:
#   TFLOPS/s_mean, TB/s_mean, FLOPS/Byte, Roofline Bound, Pct Roofline
roofline_cols = ["name", "TFLOPS/s_mean", "TB/s_mean", "FLOPS/Byte_first", "Roofline Bound"]
print(df[roofline_cols].dropna(subset=["TFLOPS/s_mean"]).head(10))
```

### Output example — with vs without TraceLens

**With TraceLens** (`"tracelens": true`):
```json
{
  "rank": 1,
  "name": "Cijk_Ailk_Bljk_HHS_BH_MT128x64x64_MI16x16x16x1_SE_1LDSB0_APM1_...",
  "short_name": "Cijk_Ailk_Bljk_HHS_BH_MT128x64x64_MI16x16x16x1_SE_1LDSB0_APM1_",
  "cpu_op": "aten::mm",
  "calls": 200,
  "total_ms": 45.23,
  "avg_us": 226.15,
  "pct_gpu_time": 31.2,
  "classification": "library_gemm",
  "backend_guess": "hipblaslt",
  "editable": false,
  "per_call": {
    "n": 200, "median_us": 220.3, "mean_us": 226.1, "std_us": 12.5,
    "min_us": 198.0, "max_us": 260.1,
    "p10_us": 215.0, "p90_us": 235.8, "p99_us": 248.2,
    "cov": 0.055, "distribution_type": "stable"
  },
  "roofline": {
    "tflops_s": 312.5,
    "tb_s": 2.1,
    "flops_byte": 148.8,
    "bound": "COMPUTE_BOUND",
    "pct_roofline": 78.1
  }
}
```

**Without TraceLens** (`"tracelens": false`) — same kernel, stdlib fallback:
```json
{
  "rank": 1,
  "name": "Cijk_Ailk_Bljk_HHS_BH_MT128x64x64_MI16x16x16x1_SE_1LDSB0_APM1_...",
  "short_name": "Cijk_Ailk_Bljk_HHS_BH_MT128x64x64_MI16x16x16x1_SE_1LDSB0_APM1_",
  "calls": 200,
  "total_ms": 45.23,
  "avg_us": 226.15,
  "pct_gpu_time": 31.2,
  "classification": "library_gemm",
  "backend_guess": "hipblaslt",
  "editable": false,
  "per_call": {
    "n": 200, "median_us": 220.3, "mean_us": 226.1, "std_us": 12.5,
    "min_us": 198.0, "max_us": 260.1,
    "p10_us": 215.0, "p90_us": 235.8, "p99_us": 248.2,
    "cov": 0.055, "distribution_type": "stable"
  },
  "shapes": [[[4096, 5120]], [[1, 5120]]],
  "dtypes": ["Float16"]
}
```
Note: `roofline`, `cpu_op` are absent without TraceLens. `per_call` works in both modes — but in
the TraceLens path each entry is per-(name, shape) so per-call stats are shape-specific, while in
the stdlib path per-call groups by kernel name only (may mix shapes).

## Reliability notes
- Profile with the SAME ISL/OSL/concurrency as the throughput benchmark, after warmup.
- Use a short, bounded profiling window (`--profile-num-steps`) so traces stay parseable.
- `total_gpu_time_ms` is summed kernel duration in the captured window, not wall-clock — use it for
  RELATIVE ranking (%gpu), not as the throughput number.
- `pct_gpu_time` is the direct percentage of total GPU time (no de-inflation). Use `per_call` stats
  to identify instability (high CoV) and make informed routing decisions.
