---
title: Optimize an e2e model — the e2e_workflow serving-throughput flow
kind: workflow
gens: [gfx942, gfx950]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - GEAK/e2e_workflow/README.md
  - GEAK/e2e_workflow/PLAN.md
  - GEAK/e2e_workflow/roles/e2e_integrator.md
  - GEAK/e2e_workflow/roles/op_benchmarker.md
---

# Optimize an e2e model (serving throughput)

## TL;DR
Raise sglang/vLLM serving throughput by reasoning in **Amdahl mass** (`pct_gpu_time ×
achievable_speedup`), tuning the cheap config knobs first, then optimizing hot kernels via
the single-kernel layer and **overlaying them back reversibly**. Every change must
**engage the live path** and clear the **>0.5% e2e gate** (or **stack** if sub-threshold
and non-regressing). This is the `e2e_workflow` workflow
([`GEAK/e2e_workflow/`](../../e2e_workflow/)) made into a checklist.

## The invariants (do not break)
- **TP=1** for the tuning runs — single-GPU, deterministic Amdahl accounting and stable
  A/B. Multi-GPU collective config is tuned separately at bring-up
  ([`model_bringup_checklist.md`](model_bringup_checklist.md)).
- **>0.5% accept band** — a delta counts only if `delta% > NOISE_BAND_PCT` AND
  `cand_min > ref_max` (run distributions don't overlap). A 0.5% median gap with
  overlapping runs is noise → reject.
- **Cumulative stacking** — sub-threshold non-regressing wins (`cand_med >= ref_med`) are
  carried forward and gated as a **combined** stack against the TRUE baseline.
- **Engagement proof** — prove the change is used (server log), never infer from a wiggle.
- **Combined gate** — the Director's final validation runs the full accepted stack vs the
  true baseline under the tight protocol; that is the authoritative number.

## The phases

### 1. Setup (e2e Director)
Create an isolated eval dir; launch a warm server at the target `ISL/OSL/conc`; measure
the **TRUE baseline** throughput (median of ≥3 warm repeats, note spread). Fix the
workload, seed, and parity prompt set (≥10 prompts, greedy/temp=0). Everything later is
A/B'd against this baseline.

### 2. Profile (Profiler)
Warm-server trace (torch profiler + optional `rocprofv3`) → ONE standardized **Top-N**
artifact (per-kernel `pct_gpu_time` + captured shapes + prefill-vs-decode regime). See
[`../profiling/`](../profiling/). A same-named kernel in prefill vs decode = different
shape regimes → may need regime-specific variants.

### 3. Strategize (System Architect)
Route the Top-N by **Amdahl mass** into tracks:
- **config track** (Tier-0, cheap, run first) — flags/env/backend swaps that reshape the
  whole profile.
- **head-kernel track** — GEMM / attention (the bulk of GPU time). Tune AND author.
- **editable-kernel track** — norms/rope/act/gated-delta; small mass each → cluster +
  stack.
- **host track** — launch overhead, graph capture, scheduling.
Plan per-milestone with a stop rule; grow the persistent experience library
(`backend_playbook.md`).

### 4. ConfigSweep (Config Tuner, runs FIRST, default ON)
Tier-0 flag/env/backend sweep — no source rewrite. Examples:
`VLLM_ROCM_USE_AITER=1` (master switch), `--attention-backend <...>`
([`attention_backend_selection.md`](attention_backend_selection.md)), `--quantization
fp8`, RCCL / QuickReduce env, fused-norm flags. Config **reshapes the profile**, so
**re-profile after it** before extracting kernels.

### 5. HeadKernel (Kernel Extractor + Op Benchmarker → kernel layer)
For each head op: the Extractor captures **real shapes + a reference-I/O oracle** → an
**immutable** task dir. The Op Benchmarker runs the cheapest-first ladder
([`optimize_single_kernel.md`](optimize_single_kernel.md)): Tier-A bake-off, Tier-B tune
(aiter DB for GEMM — [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md)), and emits an
**author_plan** so the kernel `kernel_workflow` actually writes/optimizes a kernel. For a
GEMM head order FlyDSL author FIRST. Quant (Tier-D) only if `ENABLE_FP8`.

### 6. Milestone (e2e Integrator/Validator)
Overlay each verified kernel result back **reversibly** (PYTHONPATH overlay /
monkeypatch — never edit site-packages; never copy a whole package subtree, that shadows
the install). Wire-in seams: [`integrating_a_new_kernel.md`](integrating_a_new_kernel.md).
Then the **gate** — a change enters e2e only if ALL hold:
1. isolated speedup is real (oracle untampered),
2. **engagement proven** on the live path,
3. `delta% > NOISE_BAND_PCT` AND `cand_min > ref_max`,
4. output parity holds (≥10 prompts, greedy/temp=0; accuracy probe for quant).

**Three verdicts:** `accepted` (strong standalone), `stack` (engages + parity + non-
negative but sub-threshold → compound later), `rejected` (parity fail / no engagement /
regression).

**Tight 2-launch A/B protocol:** one reference block then one candidate block, back-to-
back on the same GPU, each running `E2E_REPEATS` (default 7) timed repeats on ONE server
(do NOT relaunch per repeat). Compute `ref_med, cand_med, ref_max, cand_min, delta%` from
all per-repeat rows.

### 7. Finalize (combined stack)
Assemble all `accepted` + `stack` changes into one config/overlay.

### 8. Report
Eval-dir timeline: every candidate, its gate verdict, the numbers, why rejected. A real
isolated speedup that didn't show up e2e is an **expected Amdahl outcome**, not a bug.

### 9. Validate (Director, the combined gate)
Run the **full combined stack vs the TRUE baseline** under the tight protocol — the
authoritative headline number. Re-check parity/accuracy on the combined stack. Only wins
that survive here are banked into [`../case_studies/by_model/`](../case_studies/by_model/).

## Worked anchor (validated)
On **Qwen3.5-27B @ MI300X gfx942 / sglang**, GEMM was ~78% of GPU time; the aiter DB GEMM
tune banked **+2.23% e2e** through the full flow above. Full recipe + traps:
[`gemm_tuning_workflow.md`](gemm_tuning_workflow.md).

## Pitfalls
- **Inferring engagement from a throughput wiggle** (TunableOp lesson) — always prove it.
- **Per-repeat server relaunch** — far too slow; use the 2-launch protocol.
- **Gating small kernels one-at-a-time** — banks none; cluster and stack.
- **Skipping the re-profile after ConfigSweep** — the profile changed under you.
- **Copying a package subtree into the overlay** — shadows the whole install.

## Cross-links
- Inner loop: [`optimize_single_kernel.md`](optimize_single_kernel.md)
- GEMM: [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md) · Attn: [`attention_backend_selection.md`](attention_backend_selection.md)
- Wire-in: [`integrating_a_new_kernel.md`](integrating_a_new_kernel.md)
- Bring-up: [`model_bringup_checklist.md`](model_bringup_checklist.md)
- Profiling: [`../profiling/`](../profiling/) · Optimization: [`../optimization/`](../optimization/)
- Skill: [`GEAK/e2e_workflow/`](../../e2e_workflow/)

## Sources
- Phases, roles, fractal two-altitude design: `GEAK/e2e_workflow/README.md`, `GEAK/e2e_workflow/PLAN.md`.
- The gate, three verdicts, tight 2-launch protocol, engagement proof: `GEAK/e2e_workflow/roles/e2e_integrator.md`.
- Head-op ladder + author_plan: `GEAK/e2e_workflow/roles/op_benchmarker.md`.
- aiter default backend / config levers: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
