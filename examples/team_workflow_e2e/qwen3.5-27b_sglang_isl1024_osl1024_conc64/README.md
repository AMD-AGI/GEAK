# Example run — Qwen3.5-27B, sglang, ISL/OSL=1024, conc=64 (MI300X / gfx942)

A real end-to-end run of `team_workflow_e2e` on `Qwen-Qwen3.5-27B`. Invocation:

```
Workflow({
  scriptPath: "<…>/workflow_e2e_team/team_workflow_e2e.js",
  args: {
    model_path: "/path/to/Qwen-Qwen3.5-27B",
    workflow_dir: "<…>/workflow_e2e_team",
    backend: "sglang", isl: 1024, osl: 1024, conc: 64,
    gpu_ids: "0,1,2,3",          // optimization-parallelism pool (serving stays TP=1)
    budget: 4, min_kernel_tasks: 4, kernel_budget: 3,
    head_budget: 1, head_author_max: 1, e2e_repeats: 5,
    config_tune: "true", apply_to_original: "false"
  }
})
```

## Files
- **`final_report.md`** — the headline deliverable: complete timeline with the **Phases tree** +
  **artifact tree** modules, baseline + profile breakdown, every attempt (kept AND rejected) with
  isolated/e2e deltas and verdicts, summary table, and next steps.
- **`final_launch.sh`** — reproduces the optimized server + bench (carries the accepted config).

## Result (headline)
- Baseline **1485.4** → **1549.2 tok/s (+4.1%)**, accepted, output parity pass.
- Accepted lever: `--attention-backend triton`.
- Head dense GEMM (aiter DB tune + authored Triton) and the editable gated-delta/mamba kernels each
  had real *isolated* speedups but landed below the 0.5% e2e noise band (Amdahl-dominated by ~81%
  dense GEMM in this prefill regime) — see the report's Phases tree for the full per-step breakdown.

This example shows the workflow exercising every track (config sweep, head-GEMM bake-off + aiter tune
+ Triton author via the recursive kernel layer, the editable-kernel milestone loop with cumulative
stacking, and the e2e gate with interleaved A/B), and reporting honestly what did and did not convert.
