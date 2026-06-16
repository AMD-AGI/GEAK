---
title: Optimize a single kernel — the cheapest-first backend ladder
kind: workflow
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - GEAK/e2e_workflow/roles/op_benchmarker.md
  - GEAK/kernel_workflow/  (single-kernel kernel_workflow)
  - https://github.com/ROCm/aiter
---

# Optimize a single kernel

## TL;DR
Build an **immutable unittest from real shapes**, then climb a **cheapest-first ladder**:
bench every available backend → tune the best backend → only if the winner is editable,
author/rewrite code via the kernel `kernel_workflow` → verify correctness AND perf on the
oracle → **bank the win**. Never start by writing code; start by measuring what already
exists. The ladder mirrors the Op-Benchmarker role
([`op_benchmarker.md`](../../e2e_workflow/roles/op_benchmarker.md)) and the kernel
layer ([`GEAK/kernel_workflow/`](../../kernel_workflow/)).

## Preconditions
- You know the operator family → see [`choosing_a_backend.md`](choosing_a_backend.md) and
  the matching `operators/<op>/overview.md`.
- You have a GPU pinned (`HIP_VISIBLE_DEVICES=<id>`), ROCm + the candidate backends
  installed (aiter, CK, Triton; FlyDSL via `is_flydsl_available()`).
- The shapes are **real** (captured from the model / profile), not guessed. Wrong shapes
  produce a tune that doesn't engage — the #1 trap (see step 1).

## Step 0 — Decide it's worth optimizing (Amdahl)
If this kernel is being optimized for an e2e win, first check `pct_gpu_time` from the
profile ([`optimize_e2e_model.md`](optimize_e2e_model.md)). Small mass (<0.5% each) →
don't gate alone; **stack** it with siblings. A head op (GEMM/attention, often the bulk
of GPU time — GEMM was ~78% on Qwen3.5-27B) always justifies the full ladder.

## Step 1 — Generate the unittest with REAL shapes (immutable oracle)
Create the task dir the kernel layer consumes: `unittest.py` (correctness + timing) +
`reference_io.pt` (golden inputs/outputs) + `meta.json` (shapes, dtype, `bias` flag,
`reference_io_sha256`).
- **Capture shapes live**, do not synthesize. For a GEMM the lookup key includes the
  **`bias` flag** and exact `M/N/K/dtype` — sglang issues most dense GEMMs with
  `bias=False` (bias applied separately). A synthesized `bias=True` set mismatches every
  live call → **0 engagement** (see [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md)).
- The oracle is **IMMUTABLE** (anti-cheating). Re-hash `reference_io.pt` vs
  `meta.json.reference_io_sha256` before trusting any later result.

## Step 2 — Tier A: backend bake-off (DISCOVER, no source)
Bench **every available backend** on the immutable oracle and record per-backend
`{available, correct, ms, max_rel_err}`, the winner, `winner_editable`,
`isolated_speedup` vs the default (hipBLASLt), and whether an editable impl exists.

```bash
HIP_VISIBLE_DEVICES=<id> python3 op_bench.py --task <OP_TASK_DIR> \
  --backends "<ranked,backends>" --repeats 50 --warmup 10 \
  --out <OP_TASK_DIR>/opbench_result.json
```

Candidate backends by family (see [`choosing_a_backend.md`](choosing_a_backend.md)):
- **dense GEMM** → `hipblaslt, aiter, flydsl, asm, ck, triton, tilelang`
  ([`operators/dense_gemm/backends/`](../operators/dense_gemm/backends/))
- **prefill attention** → `ck_tile, aiter, triton, tilelang, asm`
  ([`operators/attention_prefill_fmha/backends/`](../operators/attention_prefill_fmha/backends/))
- **decode attention / MLA** → `aiter (asm decode), triton, ck`
  ([`operators/mla_attention/`](../operators/mla_attention/), [`operators/attention_decode_paged/`](../operators/attention_decode_paged/))
- **editable custom (norms/rope/act/gated-delta)** → `triton` first
  ([`operators/rmsnorm/`](../operators/rmsnorm/), [`operators/rope/`](../operators/rope/),
  [`operators/linear_attention_gated_delta/`](../operators/linear_attention_gated_delta/))

Set **`best_known_ms`** = fastest correct backend's ms. This is the bar any authored
kernel must beat. A backend only counts if it **passes correctness** (dtype-appropriate
tolerance) AND is faster.

## Step 3 — Tier B: tune the promising backend(s) (no source)
- **GEMM** → the aiter per-shape DB is the lever (gradlib races
  hipBLASLt/asm/triton/skinny/flydsl per shape, so one tune covers per-backend GEMM
  tuning). Full recipe: [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md).
  ⚠ **Do NOT use PyTorch TunableOp / `HIPBLASLT_TUNING_FILE`** on sglang/aiter — they hook
  the torch dispatch the aiter live path bypasses (zero engagement).
- **attention** → the Tier-B lever is the `--attention-backend` swap
  ([`attention_backend_selection.md`](attention_backend_selection.md)).
- **FlyDSL env path** — FlyDSL is one of the backends the aiter DB tune races
  (`libtype=flydsl`); when `is_flydsl_available()` is true a normal aiter tune selects it
  for shapes where it wins with **zero extra code**. Confirm with
  `AITER_LOG_TUNED_CONFIG=1` (look for `libtype is flydsl`). See
  [`../backends/aiter/flydsl_path.md`](../backends/aiter/flydsl_path.md).

## Step 4 — Tier C: author or rewrite code (editable languages only)
Only if the winner is editable and there's headroom. Hand the op to the **kernel
`kernel_workflow`** ([`GEAK/kernel_workflow/`](../../kernel_workflow/)) which enforces the
immutable unittest. Two modes:
- **rewrite** (`mode=optimize`) — an editable impl already exists → optimize it.
- **author** (`mode=author`, `target_language=<lang>`) — no editable impl → write a fresh
  baseline, then optimize. Full author loop:
  [`authoring_a_kernel_with_geak.md`](authoring_a_kernel_with_geak.md).

Language priority for authoring:
- **Triton** is always a viable author target (fastest to iterate).
- **For a dense / quantized GEMM (fp8 / A4W4 / mxfp4), FlyDSL is the preferred author
  target** — it's aiter's SOTA GEMM DSL; the baseline reuses `flydsl_hgemm` /
  `flydsl_preshuffle_gemm_a8` and the optimize loop tunes tile/split_k/preshuffle (JIT, no
  build). See [`../languages/flydsl/`](../languages/flydsl/).
- **HIP/CK** only when headroom is large and the image supports the build.
- The experimental triton GEMM stub is **not** a real impl → treat "no editable triton
  kernel" as author-needed.

## Step 5 — Tier D: quantization (only if enabled)
fp8 GEMM / fp8 KV → **accuracy gate, not byte parity** (mark
`parity_note=needs_accuracy_gate`). On gfx950 prefer block-scaled MXFP8/MXFP6/MXFP4 (FP6
runs at FP4 rate). See [`../quantization/`](../quantization/) and
[`gemm_tuning_workflow.md`](gemm_tuning_workflow.md).

## Step 6 — Verify (the isolated gate)
A candidate is accepted into the bank only if **both** hold on the immutable oracle:
1. **Correct** — passes dtype-appropriate tolerance vs `reference_io.pt`. Note: same-dtype
   cross-backend swaps are expected near-identical but NOT byte-identical (a bf16 argmax
   flip is real) → flag the parity risk.
2. **Faster** — median of ≥3 warm repeats beats `best_known_ms`; note the spread.

## Step 7 — Bank the win
Record the winner as one of `{env, flag, patch, authored}` with its apply recipe, the
isolated speedup, the shapes it covers, and the parity note. This becomes the input to
e2e integration ([`integrating_a_new_kernel.md`](integrating_a_new_kernel.md)) and a
[`../case_studies/by_kernel/`](../case_studies/by_kernel/) entry. **An isolated win is not
an e2e win** until it engages on the live path and clears the Amdahl gate.

## Pitfalls
- **Synthesized shapes / wrong `bias`** → tune doesn't engage. Always capture live.
- **Stopping at "pick fastest default"** — for a head op, also tune AND author; let the
  gate pick best of {tuned, authored}.
- **Editing the oracle** — invalidates the result (anti-cheating); re-hash always.
- **Claiming an e2e win from an isolated speedup** — Amdahl can erase it.

## Cross-links
- e2e wrapper: [`optimize_e2e_model.md`](optimize_e2e_model.md)
- GEMM tune: [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md)
- Backend choice: [`choosing_a_backend.md`](choosing_a_backend.md)
- Author loop: [`authoring_a_kernel_with_geak.md`](authoring_a_kernel_with_geak.md)
- Wire-in: [`integrating_a_new_kernel.md`](integrating_a_new_kernel.md)
- Routing priors: [`../index/decision_trees.md`](../index/decision_trees.md)

## Sources
- The ladder (Tier A–D), `best_known_ms` bar, immutable-oracle discipline, FlyDSL dual path: `GEAK/e2e_workflow/roles/op_benchmarker.md`.
- Kernel layer contract (author/optimize modes): `GEAK/kernel_workflow/`, `GEAK/e2e_workflow/README.md`.
- aiter as default ROCm kernel backend: https://github.com/ROCm/aiter ; https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
