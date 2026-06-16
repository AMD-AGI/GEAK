# Example run — Qwen3.5-27B-**FP8**, sglang, ISL/OSL=1024, conc=64 (MI300X / gfx942) — **FlyDSL head-GEMM win**

A real end-to-end run of `team_workflow_e2e` on `Qwen-Qwen3.5-27B-FP8` (fp8 a8w8 blockscale). This is the
run where the **head dense GEMM was authored in FlyDSL and converted to a large e2e win** — the headline
example of the FlyDSL author path. Invocation:

```
Workflow({
  scriptPath: "<…>/workflow_e2e_team/team_workflow_e2e.js",
  args: {
    model_path: "/wekafs/models/Qwen-Qwen3.5-27B-FP8/",
    workflow_dir: "<…>/workflow_e2e_team",
    backend: "sglang", isl: 1024, osl: 1024, conc: 64,
    gpu_ids: "0,1,2,3",          // optimization-parallelism pool (serving stays TP=1, single GPU)
    budget: 6, head_budget: 3, apply_to_original: "false",
    task: "prioritize the fp8 a8w8 blockscale dense GEMM head; author a FlyDSL down-proj kernel …"
  }
})
```

## Files
- **`final_report.md`** — headline deliverable: Phases tree + artifact tree, baseline + profile, every
  attempt (kept AND rejected), the **single-kernel (unittest) breakdown** (triton / FlyDSL-R0-initial /
  FlyDSL-R3-optimized, with the R0 re-measurement), summary table, measurement caveats, next steps.
- **`final_launch.sh`** — reproduces the optimized server + bench (triton attn + fp8-kv + FlyDSL overlay).
- **`evidence/`** — the JSONs the report cites: `flydsl_integrate_result.json` (the e2e gate verdict,
  `gate: accepted`), `validate_{base,stack}_bench_summary.json` (the final same-session A/B).

## Result (headline)
- **FlyDSL down-proj GEMM = +14.17% e2e** (gate-accepted): matched same-session single-GPU A/B, 7 reps,
  ref 1170.01 → cand 1335.78 tok/s, non-overlapping, FlyDSL engaged on the live path, parity quality-preserved.
- **Full stack** (triton attn + fp8-kv-cache + FlyDSL) vs true baseline ≈ **+36.94%** (fast 2-rep
  same-session Validate: 988.03 → 1352.97; base leg noisier — treat as ~+35–39%).
- **Single-kernel (unittest):** FlyDSL geomean **1.79×** over the aiter Triton blockscale GEMM
  (prefill 2.04×, decode 1.56–1.79×). The **initial FlyDSL (R0) was ~0.997× — no speedup** (re-measured);
  all the gain came from optimization (R0 0.997× → R1 1.24× kills the dequant pass → R3 1.79×).

## What this example shows
- The **FlyDSL author path** end-to-end: reuse aiter `flydsl_preshuffle_gemm_a8` → full-K fused GEMM +
  operand pre-scaling → passive crash-safe seam → M-agnostic precompile into the decode CUDA graph →
  weight-cache ≥ num_layers → gate-accepted.
- **Honest reporting:** the run crashed mid-Milestone (machine went down); it was finished via a fast
  direct Validate (the remaining <3%-GPU milestone kernels were rejecting). Only same-session deltas are
  trusted; the box-drift and 2-rep-base caveats are stated in the report. The "initial FlyDSL == triton"
  claim was re-measured rather than assumed (R0 ≈ 0.997×, marginally slower — not identical).
