---
title: Integrating a new kernel — the rebind seams that make e2e actually use it
kind: workflow
gens: [gfx942, gfx950]
status: sota
updated: 2026-06-08
sources:
  - GEAK/e2e_workflow/roles/e2e_integrator.md
  - GEAK/perf_knowledge/backends/aiter/integration.md
  - https://docs.vllm.ai/en/stable/design/custom_op/
---

# Integrating a new kernel

## TL;DR
A faster kernel is worthless until the **live server actually calls it**. Integration =
choose the **right seam** for the winner kind (`env` / `flag` / `patch` / `authored`),
overlay it **reversibly** (PYTHONPATH overlay, never edit site-packages), then **prove
engagement** from the server log before trusting any throughput number. If the call site
can't be cleanly rebound, the kernel is not a usable e2e win. This is the e2e Integrator
role ([`e2e_integrator.md`](../../e2e_workflow/roles/e2e_integrator.md)).

## The four winner kinds → their seams

### 1. `env` — a tuning DB / CSV (no overlay)
Candidate env = current env + the new var. Examples:
- GEMM: `AITER_CONFIG_GEMM_BF16=<tuned.csv>` (see
  [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md)).
- MoE: `AITER_CONFIG_FMOE=<tuned.csv>` (see [`../backends/aiter/fmoe.md`](../backends/aiter/fmoe.md)).
Keep the artifact under `$EVAL_DIR/config/` so it's reproducible. **Engagement proof:**
`AITER_LOG_TUNED_CONFIG=1` → `grep -c 'is tuned on cu_num' server.log` > 0.

### 2. `flag` — a server backend swap (no source)
Candidate flags = current + the flag. Examples: `--attention-backend ...`
([`attention_backend_selection.md`](attention_backend_selection.md)), `--quantization fp8`,
`VLLM_ROCM_USE_AITER=1` (master switch). **Engagement proof:** the backend banner / dispatch
log (`AITER_LOG_MORE=1` shows dispatch decisions).

### 3. `patch` — rewrite an existing installed module
A triton/hip/ck `code_patch` against an installed source file. Inject **only** the patched
submodule into the overlay (manifest `add-module`) — **NEVER copy a package subtree**, that
shadows the whole install:
```bash
CAND="$EVAL_DIR/overlay/cand_<short>"; cp -r "$CURRENT_OVERLAY"/. "$CAND"/ 2>/dev/null || mkdir -p "$CAND"
python3 overlay_setup.py add-module --overlay "$CAND" \
  --module "<dotted.module.of.patched.file>" \
  --patch "<code_patch>" --src-file "<installed source file>"
PYTHONPATH="$CAND" python3 overlay_setup.py check --module "<dotted.module>"
```
**Engagement proof:** an injected load banner / marker, or confirm the patched module is the
one imported.

### 4. `authored` — a from-scratch NEW kernel (rebind the call site)
There's no installed file to patch; you **rebind** the op's `target_callable` to the new
implementation. Materialize the optimized authored module (apply its `final_patch`), copy it
into the overlay as a standalone importable module, then rebind:
```bash
cp <authored kernel_src file(s)> "$CAND/<authored_pkg>/"
python3 overlay_setup.py add-rebind --overlay "$CAND" \
  --target "<target_callable e.g. pkg.mod:fn>" \
  --impl-module "<authored module dotted path>" --impl-attr "<authored entry fn>"
```
If the call site **cannot be cleanly rebound** (an inlined library call with no Python seam)
→ report `rejected` with reason `no_rebind_seam` and record it so the Architect learns the
seam is missing.

## The known rebind seams (MI-series serving stacks)
- **aiter dense GEMM** → `aiter.tuned_gemm:gemm_a16w16` (exposed as `tgemm.mm`). Tune via the
  DB env (`AITER_CONFIG_GEMM_BF16`) — the cleanest seam — or rebind for an authored GEMM.
  See [`../backends/aiter/tuned_gemm.md`](../backends/aiter/tuned_gemm.md).
- **aiter custom ops** are registered as **PyTorch custom ops with fake/meta impls**
  (`@torch_compile_guard` → `torch.library.Library` + `_register_fake`), the AMD equivalent
  of vLLM's `direct_register_custom_op`. So they survive `torch.compile`/Inductor — a rebind
  must preserve the registered schema + fake impl or Inductor will mis-trace.
- **vLLM custom-op registration** — vLLM's `CustomOp` dispatch routes to `forward_hip()` on
  ROCm (falling back to `forward_cuda()`); use `direct_register_custom_op` for a new op so it
  becomes opaque/traceable. Note: with `backend == "inductor"` vLLM disables custom ops in
  favor of Inductor Triton unless overridden.
  See [`../backends/vllm_kernels/`](../backends/vllm_kernels/).
- **sglang custom op** — register/monkeypatch the module on the overlay; sglang's FP8 linear
  routes through aiter (`torch._scaled_mm`/hipBLASLt). See
  [`../backends/sglang_kernels/`](../backends/sglang_kernels/).
- **MoE** → aiter `fused_moe` external API (auto-selects the quant kernel); tune via
  `AITER_CONFIG_FMOE`. See [`../operators/fused_moe_grouped_gemm/`](../operators/fused_moe_grouped_gemm/).

## Env flags that gate which kernel actually runs
| Var | Effect |
|---|---|
| `VLLM_ROCM_USE_AITER=1` | vLLM master switch (GEMM/RMSNorm/MoE/attn). Required even when forcing `--attention-backend`. |
| `VLLM_ROCM_USE_AITER_LINEAR/_MOE/_MLA/_BLOCK_GEMM` | per-family sub-switches (on by default with parent) |
| `VLLM_ROCM_USE_AITER=0` | disable aiter → Triton fallback (debugging) |
| `SGLANG_ROCM_AITER_BLOCK_MOE=1`, `CK_BLOCK_GEMM=1` | sglang block-MoE / CK block-GEMM |
| `AITER_LOG_MORE=1` | log JIT build + dispatch decisions (verify which backend ran) |
| `AITER_LOG_TUNED_CONFIG=1` | log every DB hit (`is tuned on cu_num`) — the engagement check |

## The integration gate (a change enters e2e only if ALL hold)
1. isolated speedup is real (oracle untampered — re-hash vs `meta.json`).
2. **engagement proven** on the live path (log evidence, not a throughput wiggle).
3. `delta% > NOISE_BAND_PCT (0.5%)` AND `cand_min > ref_max` (tight 2-launch A/B).
4. parity holds (greedy/temp=0, ≥10 prompts; accuracy probe for quant/same-dtype swaps).

Verdicts: `accepted` / `stack` (sub-threshold, non-regressing → compound) / `rejected`.
Full protocol: [`optimize_e2e_model.md`](optimize_e2e_model.md).

## Pitfalls
- **Copying a whole package subtree** into the overlay → shadows the entire install.
- **Inferring engagement** from throughput — the TunableOp lesson; always grep the log.
- **`HIPBLASLT_TUNING_FILE` / TunableOp** are inert under aiter — deploy via `AITER_CONFIG_*`.
- **Breaking the custom-op schema/fake impl** on a rebind → Inductor mis-trace under `torch.compile`.
- **First call pays JIT compile** — warm the server before benchmarking.
- **`--attention-backend triton` does NOT disable aiter GEMM/MoE** — that's why the GEMM tune
  stacks on a Triton attention backend.

## Cross-links
- e2e flow + gate: [`optimize_e2e_model.md`](optimize_e2e_model.md)
- GEMM env seam: [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md) · Attn flag: [`attention_backend_selection.md`](attention_backend_selection.md)
- Author then integrate: [`authoring_a_kernel_with_geak.md`](authoring_a_kernel_with_geak.md)
- aiter integration card: [`../backends/aiter/integration.md`](../backends/aiter/integration.md)

## Sources
- Overlay seams, four winner kinds, `no_rebind_seam`, the gate: `GEAK/e2e_workflow/roles/e2e_integrator.md`.
- aiter custom-op registration, rebind targets, env table: `GEAK/perf_knowledge/backends/aiter/integration.md`; ROCm/aiter@a6bb499 `aiter/jit/utils/torch_guard.py`.
- vLLM CustomOp dispatch / direct_register_custom_op: https://docs.vllm.ai/en/stable/design/custom_op/
