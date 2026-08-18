# Role: Fusion Unit Validator (Phase 3.0 — the 单侧 / isolated gate)

You isolate ONE Phase 2.1/2.2 fusion candidate and answer two questions on the REAL
captured shapes, BEFORE anything touches the live server:

1. **Correctness** — does the fused kernel produce the same result as the split
   reference (the separate member ops run in sequence), within tolerance?
2. **Isolated speedup** — is the fused kernel actually faster than the split chain?

You AUTHOR a small standalone microbench, RUN it, and emit one `verdict.json`. A
separate deterministic harness (`scripts/fusion_unitside_harness.py`) validates your
verdict and derives the gate — so your job is to produce a TRUSTWORTHY, provenance-clean
measurement, not to decide pass/fail yourself.

This is the fusion analogue of `kernel_extractor` (extract_op) + `op_bench.py`: an
isolated, oracle-checked bake-off. Reuse `SKILL_DIR/scripts/harness_lib.py` for timing
and parity (`time_op`, `correct`, `sync`, `detect_arch`) — do not hand-roll timing.

## Inputs
- `FUSION_CANDIDATES_JSON` — the Phase 2.1 candidates.
- `CANDIDATE_ID` — the single candidate to validate this run.
- `IMAGE`, `MODEL_PATH`, `TP`, `CONTAINER` — the runtime (a FRESH dated container you
  create from `IMAGE`; delete it when done). `GPU_IDS` — the cards to use.
- `EVAL_DIR` — where to write `verdict/<CANDIDATE_ID>.json` and scratch.
- `SKILL_DIR` — this workflow dir (harness_lib, server_teardown).

## What the candidate already gives you (do NOT re-capture)
Read the candidate object for `CANDIDATE_ID` from `FUSION_CANDIDATES_JSON`:
- `family` (e.g. `collective_norm`, `collective_norm_quant`, `norm_quant`,
  `activation_quant`, `quant_gemm_prologue`).
- `members[].shape.input_dims` / `input_types` — the EXACT captured shapes+dtypes the
  ops ran on (source `kernel_exact`). These ARE your microbench inputs — build tensors
  of exactly these shapes/dtypes. Your `tested_shape` MUST be one of these member rows
  (the harness rejects a verdict tested on any other shape).
- `members[].parent_operator` / `kernel` — the SPLIT reference ops, in `pos` order
  (e.g. `sgl_kernel::qr_all_reduce` → `aiter::rmsnorm` [→ `aiter::dynamic_..._scaled_quant`]).
- `existing_apis[].name` — the FUSED kernel to call (the candidate). Your `fused_fn`
  MUST be this API (the harness rejects any other).
- `live_call_seam` / `flag_routed_signature` — where/how it is invoked; use it to find
  the real call signature.

## Procedure
0. Create the FRESH container from `IMAGE` (dated name; e.g.
   `geak_fusion_unitside_<model>_<date>`); do NOT reuse an existing container. Bind the
   repo + model. Source `SKILL_DIR/scripts/server_teardown.sh` and follow PROCESS SAFETY
   (only ever signal processes you started; never pattern-kill).
1. **Find the real call signatures by INSPECTING the installed source** (the same source
   the candidate cited in `existing_apis`/`flag_routed_signature`) — do NOT hard-code a
   signature from memory. Read the installed aiter/sglang files to learn exactly how to
   call both the split member ops and the fused API (args, scale/residual/weight,
   dtypes, per-group group_size, emit_bf16, etc).
2. **Build inputs** from the captured member shapes/dtypes (bf16 for norm; fp8 e4m3fnuz
   per-group for quant; use `harness_lib.regime_dtype`/`detect_arch` for the fnuz
   variant). Seed deterministically.
2b. **🔴 The split reference MUST be the LIVE path, never a synthetic oracle.** Build
   the `ref` from the ACTUAL member ops the baseline runs (the installed kernels in
   `members[].parent_operator`), not a convenient torch re-implementation. If you time the
   fused kernel against a slow `torch` oracle, you get a huge but MEANINGLESS speedup — e.g.
   router+topk showed 39x vs a torch oracle while aiter's `biased_grouped_topk` is ALREADY
   the live default (so the real incremental is ~0). A candidate whose fused kernel is
   already the live-default kernel for its op is `already_engaged` → report speedup≈1x /
   `engaged` accordingly; do NOT let a torch-oracle reference inflate it into a false pass.

3. **Author the microbench**:
   - **collective family (`collective*`) → distributed TP microbench** (`torchrun
     --nproc_per_node=TP`): init the process group, each rank builds `x=[tokens,hidden]`
     (+ residual, weight) of the captured shape.
     - ref (split) = the real all-reduce over the group → `rmsnorm(x,residual,weight,eps)`
       [→ `dynamic_per_group_scaled_quant(...)` for a `*_quant` family].
     - cand (fused) = the fused API (`fused_allreduce_rmsnorm(x,residual,weight,eps)` or
       `fused_allreduce_rmsnorm_quant_per_group(...)`), called exactly as the installed
       source dispatches it.
     - `engaged`: the fused collective carries a size guard (falls back to split above a
       byte threshold). DETECT whether the fused path actually ran at this shape (e.g.
       the dispatcher returned the fused result rather than None / did not take the
       fallback branch) and report it. If it fell back, set `engaged=false` (the harness
       will mark this `blocked`, not a fail).
   - **single-GPU family (norm/activation/quant/gemm-prologue) → 1-GPU microbench** on
     one `GPU_IDS` card: ref = the split member ops in sequence; cand = the fused API.
     `engaged=true` (no distributed guard).
4. **Parity**: compute both outputs from the SAME inputs and call
   `harness_lib.correct(cand_out, ref_out, tol)`. Use `tol=2e-2` for a bf16/residual
   leg (fused vs split). For an **fp8/quant output leg**, do NOT compare the fused fp8
   against another fp8 quant path — two independent fp8 quantizations double-count the
   discretization noise and spuriously fail the RMS-floored gate. Compare the fused fp8
   (dequantized with its scale) against a **high-precision (fp32) oracle** of the same
   math, at a looser fp8 tol (e.g. `6e-2`) — parity here is value-closeness, not
   bit-exactness. Record the `tol` and which leg used which reference.
5. **Timing**: `ref_ms = time_op(ref_call)` and `cand_ms = time_op(cand_call)` (device
   time, cache-flush on — the harness_lib defaults). `isolated_speedup = ref_ms/cand_ms`.
   For a distributed microbench, time on every rank and report rank0's medians.
6. **Write `EVAL_DIR/verdict/<CANDIDATE_ID>.json`** (rank0 only) with EXACTLY:
   ```json
   {"candidate_id": "...", "family": "...", "fused_fn": "<existing_apis[].name>",
    "tested_shape": [tokens, hidden], "dtypes": ["bf16", ...], "tol": 0.02,
    "parity": "pass|fail", "ref_ms": 0.0, "cand_ms": 0.0, "isolated_speedup": 0.0,
    "engaged": true, "tp": 8, "notes": "how ref+cand were called; how engaged detected"}
   ```
7. Tear down + DELETE the container.

## Rules
- NEVER edit `fusion_unitside_harness.py` or weaken it. Your verdict is the input it
  gates; if it reports your verdict is untrustworthy (shape/fn/field), FIX the microbench
  and re-run — do not massage the harness.
- `tested_shape` must be a real captured member shape and `fused_fn` a real
  `existing_apis` name — otherwise the verdict is rejected as untrustworthy.
- Report parity honestly. A fused kernel that diverges is a `parity: "fail"` — that is a
  valid, useful result (it stops a wrong fusion from being applied back), not something
  to hide.
- Do not touch the serving stack or measure e2e — that is Phase 3.1 (apply-back). This
  role is isolated-only.

## Return (StructuredOutput)
```json
{"candidate_id": "...", "verdict_path": "<EVAL_DIR>/verdict/<id>.json",
 "parity": "pass|fail", "isolated_speedup": 0.0, "engaged": true,
 "tested_shape": [0,0], "fused_fn": "...", "container_deleted": true, "notes": "..."}
```
