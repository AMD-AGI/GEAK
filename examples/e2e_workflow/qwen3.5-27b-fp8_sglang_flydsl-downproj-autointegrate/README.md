# Example run — Qwen3.5-27B-**FP8**, sglang, ISL/OSL=1024, conc=64 (MI300X / gfx942) — **FlyDSL auto-integrated, +67.4% e2e**

A fully **autonomous** end-to-end run of `team_workflow_e2e` on `Qwen-Qwen3.5-27B-FP8` (fp8 a8w8
blockscale). The workflow — with no hand-tuning — **authored a FlyDSL fused fp8 a8w8 blockscale GEMM,
bound it capture-safely over the live sglang decode path, and accepted it at the e2e gate**, lifting
serving throughput **+67.4%** (Director-validated). This is the headline FlyDSL auto-integration example
and **exceeds the prior hand-assisted +14.17% run** (`../qwen3.5-27b-fp8_sglang_flydsl-gemm`).

Invocation (exactly as run — autonomous, no per-head hints):

```
Workflow({
  scriptPath: "<…>/workflow_e2e_team/team_workflow_e2e.js",
  args: {
    model_path: "/wekafs/models/Qwen-Qwen3.5-27B-FP8/",
    workflow_dir: "<…>/workflow_e2e_team",
    backend: "sglang", isl: 1024, osl: 1024, conc: 64,
    gpu_ids: "0",
    head_author_max: 1            // FlyDSL-first per head (faster convergence)
  }
})
```

## Result (headline)
- **FlyDSL down-proj GEMM (N=5120, K=17408) = +60.09% on the head gate** (`gate: accepted`): authored
  fused fp8 core, isolated **2.432×** (decode M=1 6.76× / M=64 5.29×, prefill ~1.0–1.38×), parity pass.
- **Final stack (Director independent Validate): 931.6 → 1559.9 tok/s = +67.4% (1.674×)**, TPOT
  64.06 → 37.19 ms, output parity preserved. Stack = `--attention-backend triton` (+2.24%) + the FlyDSL
  fused core bound over the live decode CUDA-graph path.
- **Honest measurement** (no cheating): sequential single-GPU same-config A/B (mem-fraction 0.85,
  identical KV budget), identical output-token counts, engaged INSIDE the captured decode graph
  (283 decode batches `cuda graph: True`), +60% within the Amdahl ceiling (~80% GPU blockscale-GEMM
  coverage), parity probe 10/12 exact + 2 benign coherent diffs.
- **do-no-harm**: h0 up/gate (−0.26%) and h2 qkv/o (capture-unsafe nested graph) were correctly
  **rejected** — only a real, parity-safe, memory-neutral net win was stacked.

## Why this run succeeded where naive integration fails (the four real gaps it clears)
1. **Decode-aware head unittest** — the op is benchmarked at decode M (1, 64) AND prefill M (16384), so
   the optimizer cannot win a prefill-only kernel that regresses the decode/TPOT-bound steady state.
2. **CUDA-graph-safe overlay** — the kernel hot path is host-sync-free (data_ptr weight cache, no
   per-call `.item()`); JIT happens at warmup, not inside capture; `--watchdog-timeout` raised. The
   capture-safe seam is in `evidence/sitecustomize_capture_safe_seam.py`.
3. **Memory-bounded kernel** — a FUSED fp8 core (no bf16 weight re-materialization) keeps the weight
   cache compact, so it fits at mem-fraction 0.85 with the SAME KV budget as the reference (a bf16
   re-materializing kernel needed 92.6 GB → starved KV → net regression; that path is rejected).
4. **Resilient orchestration** — the run completed through 5 intermittent infra API failures
   (hung/4xx agents) via timeout→null guards + retries, instead of wedging.

## Files
- **`final_report.md`** — headline deliverable: Phases tree + artifact tree, baseline + profile, every
  attempt (accepted AND rejected), per-kernel isolated breakdown, the final validated A/B, caveats.
- **`final_launch.sh`** — reproduces the optimized server (triton attn + FlyDSL down-proj overlay).
- **`architect_report.md`** — concise summary.
- **`kernel/`** — the optimized FlyDSL kernel saved standalone (see `kernel/README.md`):
  - `gemm_a8w8_blockscale_flydsl.py` — the final authored fused fp8 down-proj kernel (2.432× isolated, shipped).
  - `sitecustomize.py` — the capture-safe overlay seam (lazy meta-path finder, precompile-before-capture
    warmup hook, host-sync-free data_ptr weight cache, one-shot engagement proof).
- **`evidence/`** — the JSONs the report cites:
  - `flydsl_downproj_integrate_result.json` — the e2e gate verdict (`gate: accepted`, +60.09%, parity pass).
  - `upgate_integrate_result_rejected.json` — the do-no-harm rejection of h0 (−0.26%).
  - `h1_downproj_director_validation.json` — independent per-case isolated validation (2.432×, correctness pass).
  - `validate_base_bench_summary.json` / `validate_stack_bench_summary.json` — the final 931.6 → 1559.9 A/B.
