---
name: ck
description: >
  Use when rewriting or tuning a hot GPU kernel into a Composable-Kernel
  (CK / ck_tile) implementation on AMD MI300X (gfx942). Covers the CK win
  recipe in priority order: wire the shipped, codegen-tuned aiter CK op first
  (MoE / dense GEMM / fp8 GEMM), then tune the CK instance via the on-box
  codegen tuners, and only as a last resort author a ck_tile program from an
  upstream example template.
---

# Composable-Kernel (CK / ck_tile) Rewrite Skills

CK is a **template / code-generation** library. You do **not** win by writing
raw CK C++ template metaprogramming from a blank file. You win by, in order:

1. wiring the **shipped, codegen-tuned aiter CK op** for the op class,
2. **tuning the CK instance** (tile shape, block-scale, pipeline, preshuffle,
   split-K) for the benchmark's shape regime via the on-box codegen tuners, then
3. (last resort) **adapting the nearest ck_tile example** when no shipped op fits.

Most of the speedup comes from steps 0 and 1. A from-scratch CK rewrite that
ignores an applicable shipped CK op is a failure.

| Task | Start with |
|------|-----------|
| Rewrite a MoE / GEMM / fp8 kernel into CK | STEP 0 (below) |
| Shipped op is close but a better instance exists | STEP 1 (below) |
| No shipped op fits, or it fails parity/regresses | STEP 2 (below) |

---

## STEP 0 (MANDATORY, do FIRST) — wire the shipped aiter CK op

The biggest CK wins are the shipped aiter ops, not from-scratch templates.
Before authoring anything:

1. Classify the op class of the target kernel (MoE / dense GEMM / fp8 GEMM).
2. Discover the shipped op:
   `python3 -c "import aiter; print([x for x in dir(aiter) if 'ck' in x.lower() or 'CK' in x])"`
3. Get the **exact** signature with `inspect.signature` (most are
   `@compile_ops`-wrapped, so import the real def from the source module — see
   `docs/shipped_aiter_ck_ops.md`), map the existing kernel's args onto it.
4. Wire it, run `save_and_test`, benchmark. One launch per logical op (no
   per-expert / per-batch Python loops).

Op-class → shipped aiter CK op to try first:

| target kernel | shipped aiter CK op (try FIRST) |
|---|---|
| MoE / fused_moe / grouped-expert | `aiter.fused_moe(...)` (high-level) or `aiter.ck_moe_stage1` + `aiter.ck_moe_stage2` (2-stage path) |
| dense bf16/fp16 GEMM | `aiter.batched_gemm_bf16_CK(...)` |
| fp8/a8 dense GEMM | `aiter.gemm_a8w8_CK(...)` / `aiter.batched_gemm_a8w8_CK(...)` |
| fp8 block-scale GEMM | `aiter.gemm_a8w8_blockscale(...)`, `aiter.flatmm_a8w8_blockscale_ASM(...)` |

Verified signatures and arg semantics: `docs/shipped_aiter_ck_ops.md`.

---

## STEP 1 — tune the CK instance

The shipped CK ops dispatch to a **codegen-generated instance set** and pick the
best instance per shape from a tuned CSV. When a shipped op is close but a
better instance exists for the benchmark's (M,N,K)/expert config, add the shape
to the tuner's untuned CSV and run the on-box codegen tuner; it writes the best
instance into the tuned CSV that the op reads at call time.

| op class | on-box tuner |
|---|---|
| MoE 2-stage | `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` |
| fp8 dense GEMM | `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py` |
| batched bf16 GEMM | `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` |

Real on-box paths, the tile/block-scale/pipeline knobs you can vary, and the
untuned→tuned CSV workflow: `docs/instance_tuning.md`.

---

## STEP 2 (last resort) — author a ck_tile program from an example

Only if STEP 0/1 don't win. Never write CK from a blank file — copy the closest
upstream `example/ck_tile/` template and adapt shapes/dtype/epilogue:

| target | nearest example |
|---|---|
| dense GEMM | `03_gemm` (basic / universal / weight-preshuffle), `16_batched_gemm` |
| preshuffled / flat GEMM | `18_flatmm`, `38_block_scale_gemm` |
| MoE | `15_fused_moe` (+ `13_moe_sorting`, `09_topk_softmax`) |
| attention | `01_fmha` |
| rmsnorm / norm+quant | `10_rmsnorm2d`, `11_add_rmsnorm2d_rdquant` |

The ck_tile programming model (Problem/Pipeline/Epilogue), example layout, and
adaptation steps: `docs/ck_tile_authoring.md`.

---

## Reference Documentation

The `docs/` subdirectory contains detailed guides:

- `overview.md` — what CK / ck_tile is, the template/codegen model, the tile
  programming model (Problem/Pipeline/Epilogue, tile distribution), when CK wins.
- `shipped_aiter_ck_ops.md` — the shipped aiter CK ops to prefer first, with
  verified signatures and op-class → op mapping.
- `instance_tuning.md` — tuning the CK 2-stage MoE / GEMM instance via the
  on-box codegen tuners; real paths, knobs, and the untuned→tuned CSV workflow.
- `ck_tile_authoring.md` — adapting the nearest ck_tile example template when
  authoring from scratch is unavoidable; the example layout.
- `pitfalls.md` — common CK pitfalls (instance mismatch, .cu host dispatcher vs
  .hpp device code, block-scale/preshuffle layout, parity).
