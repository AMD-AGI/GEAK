---
name: roofline
description: Add conditional roofline and headroom estimates to a standardized kernel profile. Use for profile prioritization when work, traffic, timing and device capacities can be matched.
---

# Roofline estimates for head kernels

Annotate the Top-N to guide experiments. `pct_gpu_time` remains the primary ordering;
this skill never prunes a candidate or accepts an optimization. A high fraction of
an implementation's modeled roof does not prove its work or traffic is necessary.
Correctness and measured full-operation/workload results decide acceptance.

Use [roofline_tools.py](roofline_tools.py) for parsing and arithmetic when available.
Its labels and fixed thresholds are heuristics, not measured bottlenecks or
achievable limits. If a helper fails, compute from supported operands or emit an
unknown result and continue the run.

## Inputs and scope

Read `profile/round_<R>/profile_topN.json`, `env_report.json` and [peaks.md](peaks.md).
Use captured shapes/dtypes, model configuration and counters where available.
Select entries with `pct_gpu_time >= HEAD_THRESHOLD_PCT` (default 5), largest first,
up to 8. This is a work-budget default, not a universal noise threshold. Skipped
entries are absent, not modeling failures. Read the Top-N `editable` flag as a
source-editing hint. The task's edit boundary decides whether a library/binary
implementation can be replaced through another supported backend.

Identify the actual source/configuration, shape/dtype/strides, physical product,
runtime and measured phase. Match bytes/work to one launch's time, or aggregate
all stages and time of one logical operation. Do not divide a whole-call counter
total by one stage's duration. Keep decode and prefill separate. Choose the regime
from measured phase costs (`base_latency_ms[regime] * phase_calls_measured[regime]`),
then use its `base_latency_ms[regime]` for `t_ms`. The profile's
`serving.n_decode_steps` / `serving.n_prefill_steps` show phase coverage, not time shares.
A mixed-phase `avg_us` is not that regime's latency. If phase timing is missing,
leave the regime estimate unknown or explicitly label a whole-profile estimate.

Call `resolve_peaks(peaks_md_path, gfx, device=device_id, product=env_report.device_target)`
with this skill's `peaks.md` path and the measured GPU's device index. Use
`peak_flops_for()` for the actual arithmetic dtype/unit. `device_target=unknown`
does not restrict the existing ISA-keyed tables. Client `gfx120x` requires a known
product table (`r9700`); a bare `gfx1201` has no numeric fallback. `gfx125x` is CDNA5,
not that client family. Unknown dtype means unknown compute capacity, never the
largest available peak. The `dtype_bytes()` compatibility default is not evidence
for an unknown storage dtype. Derived peaks are low-confidence, display-only estimates.

## Model the work and traffic

Keep useful operation work, executed work and logical footprint separate. A logical
byte rate is not measured HBM utilization. Support cache residency with matching
cache/interface counters or a controlled warm/cold reuse comparison; fitting in
cache is insufficient. Counter bytes need documented interface,
units, read/write treatment, instance aggregation and call attribution.

Examples of logical models, with explicit assumptions:

| Operation | Work and footprint |
|---|---|
| Dense GEMM `[M,K] × [K,N]` | `W = 2*M*N*K`; `Q = M*K*a + K*N*w + M*N*c`, for input/weight/output bytes per element `a,w,c`, each transferred once |
| Grouped MoE | Work from routed token/expert pairs; weight bytes from experts actually used by the implementation. `E*(1-(1-1/E)^pairs)` assumes independent uniform routing and is only an estimate |
| Paged decode attention | KV footprint `batch*context*n_kv_heads*head_dim*2*kv_bytes`; work approximately `4*batch*n_q_heads*context*head_dim`, plus other stages when material |
| Elementwise/norm/quant | Count all inputs, outputs, scales, metadata and passes; estimate arithmetic separately |

For each decode launch, `context` is the mean current attended length across its
active sequences, not the maximum configured length. Model each launch with its
captured context/routing distribution and aggregate with matching time and weights.
If lengths are missing, `ISL + OSL/2` is a labeled, low-confidence approximation
only for a full-run fixed-length synthetic workload. Missing lengths for a trace
replay or partial profile window leave the context-dependent estimate unknown.
Include padding, rereads, workspace, quantization metadata and helper stages.
Name the byte method and its evidence: for example, a one-transfer footprint or
an implementation model with repeated passes. If reuse is uncertain, show those
scenarios as a range; do not choose whichever falls below a peak. Unknown classes
remain unmodeled.

For `W` FLOPs (FMA = 2), `Q` bytes, `T` seconds, `P` FLOP/s and `B` bytes/s:

```text
achieved_bw = Q/T; achieved_flops = W/T
AI = W/Q; ridge_point = P/B
hbm_util = (Q/T)/B; compute_util = (W/T)/P
roofline_pct = max(hbm_util, compute_util)
model_time = max(Q/B, W/P)
```

These are conditional model ratios. Use matching memory level, arithmetic mode,
dense/sparse convention and device conditions. An empirical microbenchmark is a
comparator for its own conditions, not a universal roof. Arithmetic intensity
selects the tighter model resource; low utilization alone does not prove latency
or occupancy is limiting. Use targeted evidence to distinguish causes.

Preserve raw out-of-range ratios; investigate units, scope, traffic assumptions
and capacities. Clamping for display never establishes saturation. Very short
launches need an applicable overhead measurement.

## Existing ranking fields

For compatibility, the helper uses these `target_eff` priors:

| Class | Prior |
|---|---:|
| dense GEMM | **0.90** |
| MoE / grouped GEMM | **0.90** |
| elementwise / norm / quant | 0.875 |
| attention decode (paged) | **0.50** |
| R9700 (`product=r9700`) MoE weight streaming / elementwise | **0.76** |

The R9700 override comes from a 485.8 GB/s median read+write copy result against
640 GB/s pin bandwidth (30 event-timed 1 GiB copies). Transfer to MoE/elementwise
is still a prior; it does not establish their attainable rates. Other priors are
not measured product efficiencies.

```text
attainable_speedup = max(1, target_eff / roofline_pct)
expected_e2e_gain_pct = pct_gpu_time * (1 - 1/attainable_speedup)
```

Despite its historical name, `attainable_speedup` is a scenario estimate.
`expected_e2e_gain_pct` estimates reduction in the profiled GPU-time sum; it predicts
workload latency only if that fraction matches the critical elapsed interval and
other costs/overlap remain unchanged. Measure the workload result separately.

The helper bands `headroom_class` at 0.9 and 0.6 times the prior. Its `latency`
label applies at `t <= 2*launch_overhead_s` (default `2*5 µs = 10 µs`). After the
sanity checks, it also applies when both utilization ratios are below 0.60. Treat these as hints; use
`unknown` when the claimed bottleneck or headroom is unsupported. High model
fractions can motivate fewer bytes/work, fusion or reuse; low fractions can motivate
scheduling or parallelism. Neither fixes the allowed implementation. Respect the
actual editable boundary and available backend; check integration before a rewrite.
Use `byte_reduction_levers` for concrete hypotheses such as removing an intermediate,
reusing weights/KV, skipping unused experts or changing packing. State which bytes
each could remove and its contract/integration constraints; an empty list is valid.

The supplied workload (`isl`, `osl`, concurrency/batch) and measurement contract
must not be changed. Do not introduce speculative decoding as a kernel optimization.
Precision changes need the existing numerical/accuracy gate.

## Output and degradation

Write `profile/round_<R>/profile_roofline.json` and a concise sibling
`profile_roofline.md`. Preserve the existing JSON field names:

- Top level: `skill: "roofline"`, `skill_version: "1"`, `gfx`, `peaks`, `stage`,
  `entries`, `ranking_by_pct`, `ranking_by_expected_gain`, `degraded`, `skill_errors`.
- Entry identity/scope: `name`, `short_name`, `pct_gpu_time` copied unchanged,
  `regime`, `op_class`, `modeled`, `t_ms`, `launches_per_step`.
- Model: `bytes_est`, `flops_est`, `achieved_bw_bytes_s`, `achieved_flops`, `hbm_util`,
  `compute_util`, `arithmetic_intensity`, `ridge_point`, `roofline_pct`, `target_eff`,
  `attainable_speedup`, `expected_e2e_gain_pct`.
- Interpretation: `bound_type` (`memory|compute|latency|unknown`), `headroom_class`
  (`underperforming|moderate|saturated|unknown`), `confidence`, `suspect`,
  `byte_reduction_levers`, and `notes` with assumptions, raw evidence and limits.

Use null for unavailable numbers. If displaying a clamped value, retain
`roofline_pct_raw` and `suspect:true`; the verdict stays unknown. Preserve the
helper's singular `note` diagnostic alongside the report's explanatory `notes`.
Its `bytes_upper_bound=B*T` and `flops_upper_bound=P*T` are conditional capacity
checks, not measured traffic/work or proven bounds on removable cost.

| Evidence | Stage/confidence | Use |
|---|---|---|
| Estimated shapes/model config | A / low | Display only |
| Captured shapes/dtypes with supported model | B / medium | Secondary prioritization |
| Matched counters and validated definitions/capacities | C / high for the supported quantities | Secondary prioritization; causal claims still need evidence |

Emit both rankings; low-confidence, suspect or unknown entries fall back to
`pct_gpu_time`. Refine when new evidence changes a decision, retaining source and
measurement identity. Counter availability alone does not raise confidence.
`FETCH_SIZE`/`WRITE_SIZE` convert by 1024 only where their definitions say KiB;
MFMA counters require the actual dtype/instruction work factor. Do not treat a
missing read/write component as measured zero or sum unrelated replays.
For `SQ_INSTS_VALU_MFMA_MOPS_F8`, verify the installed event definition and its units
and supply the verified conversion factor; otherwise leave the FLOP count unknown.
See the [counter and calibration guidance](../../../../perf_knowledge/skills/kernel-hardware-evidence/references/collection.md)
for version-specific limitations.

`analysis_skill=none` or unreadable skill: emit nothing. Missing peaks/model/counters:
leave affected quantities unknown, add the entry/reason to `degraded`, retain other
results. Unexpected failure: record it in `skill_errors` and continue. Compare
predictions with accepted results in the existing experiment/playbook notes;
a faster result can revise the assumptions without proving a particular cause.
Report isolated-kernel repeat variability separately from end-to-end variability.
A repeatable isolated gain can be too small to resolve in the workload measurement;
use that workload's measured uncertainty, not a universal percent cutoff.
