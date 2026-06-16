---
title: Authoring a kernel with GEAK / kernel_workflow — the unittest-first loop
kind: workflow
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1]
status: sota
updated: 2026-06-08
sources:
  - GEAK (AMD-AGI/GEAK) README
  - GEAK/kernel_workflow/roles/author_engineer.md
  - GEAK/e2e_workflow/roles/op_benchmarker.md
---

# Authoring a kernel with GEAK / kernel_workflow

## TL;DR
When **no editable backend impl exists** for a hot op (a library GEMM/attention, or an op
with no kernel on this image), you **author one from scratch** in a target language, then
let the optimize loop improve it — all gated by the **immutable correctness oracle**. Use
**GEAK** (the agent-driven kernel-authoring framework, supports HIP/Triton/FlyDSL) or the
kernel `kernel_workflow`'s **author mode**
([`author_engineer.md`](../../kernel_workflow/roles/author_engineer.md)). Correctness first,
performance second. The authored kernel only counts when it **passes the oracle AND beats
`best_known_ms`**.

## When to author (vs tune / swap)
From the single-kernel ladder ([`optimize_single_kernel.md`](optimize_single_kernel.md)):
- The Tier-A bake-off shows the fastest backend is a **library** (no Python seam) or the
  only editable candidate is a stub (the experimental triton GEMM stub is NOT a real impl →
  treat as author-needed).
- You want a code-level lever the env/flag tuning can't reach (a fused epilogue, a custom
  varlen path, a new dtype).
- For a **head op** the Op Benchmarker *always* emits an author_plan in addition to the
  tune — let the e2e gate pick best of {tuned, authored}.

## Language choice
- **Triton** — always supported, fastest to iterate; the default author target for norms,
  rope, act, gated-delta, and general attention. See [`../languages/triton_amd/`](../languages/triton_amd/).
- **FlyDSL** — aiter's Python kernel DSL, **JIT (no build step)**. The **preferred author
  target for dense / quantized GEMM (esp. fp8 / A4W4 / mxfp4)**: the simplest correct
  baseline calls aiter's production `flydsl_hgemm` / `flydsl_preshuffle_gemm_a8`
  (`out = a @ b.T (+bias)`), then the optimize loop tunes its **tile / split_k / preshuffle**
  knobs. Order FlyDSL FIRST for a GEMM head. See [`../languages/flydsl/`](../languages/flydsl/)
  and [`../backends/aiter/flydsl_path.md`](../backends/aiter/flydsl_path.md).
- **HIP / CK** — only when headroom is large and the image supports the build (CK build may
  be absent). See [`../languages/hip_cpp/`](../languages/hip_cpp/), [`../languages/composable_kernel/`](../languages/composable_kernel/).

## The unittest-first loop

### 0. Get the immutable oracle
The author works in the canonical `WORKSPACE` (a `kernel_src/`) built from the op `TASK_DIR`,
which holds the **IMMUTABLE** `unittest.py` + `reference_io.pt` + `meta.json` (anti-cheating;
re-hash `reference_io_sha256` before trusting anything). `OP_SPEC` from `meta.json` gives
`op_kind`, shapes, `transpose_b`, `bias`, `dtype`, `math_contract`, `regime`.

### 1. Load focused authoring knowledge (before writing a line)
- Language skeleton: this base's [`../languages/<lang>/`](../languages/) (annotated GEMM/FMHA
  skeletons). For **FlyDSL GEMM**, call `flydsl_hgemm` rather than hand-writing layout algebra.
- Op algorithm + shape-regime split: the matching
  [`../operators/<op>/overview.md`](../operators/).
- Hardware sanity for the FIRST cut only: right MFMA shape/dtype
  ([`../hardware/shared/`](../hardware/shared/)), FNUZ fp8 on gfx942, prefer
  `matrix_instr_nonkdim=16`. Don't over-tune the baseline.

### 2. Author the simplest implementation that PASSES the oracle
Correctness first. Commit the passing baseline — optimization happens afterwards, not here.

### 3. Optimize (the closed loop)
Hand the passing baseline to the optimize loop (GEAK's profiling→optimization→validation
loop, or the kernel `kernel_workflow`'s parallel specialist engineers: algorithm / memory /
compute / host_runtime). Each candidate patch is **independently re-benchmarked in a clean
workspace** — only verified, absolute-latency wins survive. Cross-round memory threads
learnings forward; dead-ends aren't retried.

### 4. The author gate
Accept the authored kernel only if it **passes the immutable oracle** (dtype-appropriate
tolerance) AND its verified median **beats `best_known_ms`** (the fastest pre-existing
backend). Note the parity risk for cross-backend/quant cases.

## GEAK quick start (the standalone framework)
GEAK auto-discovers or generates tests + harness, then runs the closed loop and produces a
reviewable patch. Supports HIP, Triton, FlyDSL; multi-agent search across isolated git
workspaces; repository-level (L3) workflows.
```bash
git clone https://github.com/AMD-AGI/GEAK && cd GEAK
make install                                  # core + MCP tools
AMD_LLM_API_KEY=<key> bash scripts/run-docker.sh   # or local make install
```
Use GEAK when you want a self-contained author+optimize run for one op; use the kernel
`kernel_workflow` author mode when the op is being driven by the e2e pipeline (it consumes the
extractor's immutable task dir and returns a `final_patch.diff` the Integrator overlays).

## Then: make e2e actually use it
An authored kernel has **no installed file to patch** → integration **rebinds** the op's
call site (`target_callable`) to the authored entry. If there's no Python seam →
`no_rebind_seam` (not a usable e2e win). Full seam mechanics:
[`integrating_a_new_kernel.md`](integrating_a_new_kernel.md).

## Pitfalls
- **Optimizing before it passes** — correctness is the gate; a fast wrong kernel is worth 0.
- **Over-engineering the baseline** — author the simplest correct version; let the loop tune.
- **Editing the oracle** — invalidates the result (anti-cheating).
- **Hand-writing FlyDSL layout algebra** when `flydsl_hgemm` already gives a correct baseline.
- **Authoring something with no rebind seam** — it can't reach the live path.

## Cross-links
- Ladder context: [`optimize_single_kernel.md`](optimize_single_kernel.md)
- Wire-in: [`integrating_a_new_kernel.md`](integrating_a_new_kernel.md)
- GEMM author target: [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md), [`../languages/flydsl/`](../languages/flydsl/)
- Kernel layer: [`GEAK/kernel_workflow/`](../../kernel_workflow/) · GEAK: https://github.com/AMD-AGI/GEAK

## Sources
- GEAK capabilities (HIP/Triton/FlyDSL, closed loop, L3, multi-agent): GEAK (AMD-AGI/GEAK) README, https://github.com/AMD-AGI/GEAK
- Author mode, FlyDSL-GEMM baseline reuse, correctness-first, immutable oracle: `GEAK/kernel_workflow/roles/author_engineer.md`.
- author_plan for head ops, best-of {tuned, authored}: `GEAK/e2e_workflow/roles/op_benchmarker.md`.
