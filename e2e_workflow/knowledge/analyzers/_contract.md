<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
See LICENSE for license information.
-->
# Profile-analyzer contract (pluggable)

A *profile analyzer* turns a captured profiler trace into two artifacts the e2e workflow consumes as
**advisory** input: a structured top-kernel list and a human/LLM-readable summary. The orchestrator and
every downstream role depend ONLY on this contract — never on a specific analyzer's native output — so a
new analyzer (nsight, rocprof-native, a custom one, …) plugs in by adding **one markdown recipe** under
`knowledge/analyzers/<name>.md` that produces the same two files. Nothing in `e2e_workflow.js` changes.

Each analyzer recipe `knowledge/analyzers/<name>.md` MUST, given a trace, produce:

## 1. Canonical structured output — `<OUTDIR>/top_kernels.json`
Analyzer-agnostic schema (omit a field you cannot fill; never fabricate):
```json
{
  "schema_version": 1,
  "source": "<analyzer name>",
  "trace": "<path of the trace analyzed>",
  "gpu_arch": "MI300X | MI325X | <detected> | unknown",
  "total_gpu_time_us": 0.0,
  "top_kernels": [
    {
      "rank": 1,
      "name": "<kernel/op name>",
      "category": "GEMM | MoE | Attention | Comm | Norm | Elementwise | Other",
      "gpu_time_us": 0.0,
      "pct": 0.0,                       // % of total GPU time
      "count": 0,                       // launches in the analyzed window (optional)
      "shape": {"M": 0, "N": 0, "K": 0, "dtype": "..."},   // optional; for GEMM/MoE when known
      "roofline": {"bound": "compute|memory", "achieved_pct": 0.0}  // optional; when an arch spec is available
    }
  ]
}
```
Rank by `pct` descending; ~top 30 is enough. `category` is a coarse, analyzer-mapped bucket.

## 2. Summary markdown — `<OUTDIR>/summary.md`
Written by the executor agent (the workflow's own LLM) FROM `top_kernels.json` (+ any native detail the
recipe surfaced). Sections:
1. **Overview** — what the trace is, total GPU time, graph mode if detected.
2. **Where time goes** — a table of the top kernels by % share, with shape when known, and what each is.
3. **Bottlenecks** — dominant cost(s); roofline bound (compute vs memory, % of peak) when available;
   communication (all-reduce/nccl); small-kernel/launch overhead.
4. **Optimization suggestions** — 3-5 concrete, Amdahl-aware levers tied to the top kernels/shapes.
Keep it concise (~500 words), quantitative, and **advisory** — it ADDS candidates/priors; it never
overrides on-box measurement or the e2e gate.

## Fault tolerance (mandatory)
If the trace is missing, dependencies cannot be installed, or the analyzer fails, the executor returns
`{"ok": false, "note": "<reason>"}` and writes nothing. The run then proceeds exactly as if profile
analysis were disabled. Never block the run; never invent numbers.
