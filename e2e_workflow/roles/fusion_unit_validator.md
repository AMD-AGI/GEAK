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

## PHASE=validate_one — one concrete Top-K candidate
One invocation of this role validates ONE candidate. **The loop over concrete
candidate ids in `FUSION_TOPK_JSON.execution_list` is the CALLER's job.** It must
cover that entire ranked denominator, not a convenient family-level subset. Discovery
candidates outside the execution list are explicitly `deferred_rank_budget` and do
not fail unitside coverage.

The caller supplies `FUSION_TOPK_JSON`, `EXEC_ID`, and `CANDIDATE_ID`.
`CANDIDATE_ID` must occur in that execution-list row's `candidate_ids`; do not choose
a family-level substitute or validate more than this one candidate.

`fusion_unitside_harness.py --topk` enforces this: it expands concrete candidate ids
from the execution list and requires every in-scope id to end in a verdict, an
`--equivalent <id>=<representative_candidate_id>` (the same recipe/cohort was
benched once; the sibling inherits the representative's pass/fail/blocked/
needs-diagnosis result), a `--subsumed <id>=<ladder_top>` (its ladder top was
benched and passed), or an explicit
`--waive <id>=<reason>`; `budget_skipped` / `not_validated` → **the gate FAILS**. tier-C
(`new_helper_kernel`, no existing kernel) is reported `deferred_author` and is out of
scope. The report leads with the denominator and a per-phase breakdown.

Why this exists: on DSR1 2026-08-26 the caller benched 16 of 42 candidates and the
report read `总计 16 条 verdict：pass 16 / fail 0 ... status=pass`. The single highest
roofline decode candidate and **all 20 prefill candidates** never reached the microbench,
and nothing turned red. Note the bias in what got dropped — the missing ones were the
ones whose microbench is HARDEST to set up (paged-KV state, MoE routing state, sorting
buffers), which has no correlation with payoff. Without a forced denominator the loop
silently selects for "easy to bench".

If a candidate genuinely cannot be benched this round, waive it WITH A REASON
(`--waive d1_kv_rope_write_cluster="needs paged-KV cache state; deferred to round 2"`).
"skipped" is not a reason and an empty reason is rejected.

## PHASE=aggregate — publish the Top-K denominator

Inputs: `FUSION_TOPK_JSON`, `FUSION_CANDIDATES_JSON`, `FUSION_DIR`, `EVAL_DIR`,
`FUSION_UNITSIDE_BUDGET`, `EQUIVALENT_COVERED`, `SUBSUMED_COVERED`,
`BUDGET_SKIPPED`, and any explicit
waivers from the caller.

Run the harness with `--topk "$FUSION_TOPK_JSON"`. Its coverage denominator is the
union of concrete `candidate_ids` in `fusion_topk.execution_list`, not every discovery
candidate. Candidates outside that union are `deferred_rank_budget` and cannot fail
Top-K coverage.

Three caller-supplied lists close the rest of the denominator. Keep their meanings
separate and do not route any of them through `--waive`:

| caller input | harness flag | status | covered? |
|---|---|---|---|
| `EQUIVALENT_COVERED` | `--equivalent <id>=<representative_candidate_id>` | `equivalent_pass/fail/blocked/needs_diagnosis` | **yes** — the same recipe/cohort was measured once; only an inherited pass is eligible for apply-back |
| `SUBSUMED_COVERED` | `--subsumed <id>=<ladder_top>` | `subsumed_pass` | **yes** — its ladder-top superset was benched and PASSED, and that microbench exercised every row this rung removes |
| `BUDGET_SKIPPED` / `DEFERRED_EXECUTIONS` | `--budget-skipped <id>=<reason>` | `budget_skipped` | **no** — the unit-side budget ran out; nobody measured it. Counts against coverage and FAILS the gate |

Copy both into the returned `deferred[]` with their status; never silently omit one.

Why the split: on DSR1 2026-09-03 budget overruns were passed in as `--waive`. They
rendered as `waived`, dropped out of the unvalidated set, and coverage reported
`complete: true` over a board where 17 of 27 in-scope candidates had never been
measured — including the one carrying a ★★★ prior with two reproduced e2e confirms
(+1.606%, +2.142%). Running out of budget is a fact about US, not a judgment about the
candidate. 「没测」不得渲染成「测过了」。

One verdict per candidate is not a phase result. When the loop is done, the CALLER runs
the harness once over the whole verdict dir and publishes the Phase 3.0 report at the
EVAL_DIR root, beside the other four phase reports:

```bash
python3 "$SKILL_DIR/scripts/fusion_unitside_harness.py" \
  --candidates "$FUSION_CANDIDATES_JSON" \
  --topk       "$FUSION_TOPK_JSON" \
  --verdicts   "$EVAL_DIR/verdict" \
  --out-md     "$EVAL_DIR/04_FUSION_UNITSIDE.md" \
  --out-json   "$FUSION_DIR/fusion_unitside.json" \
  --equivalent "<id>=<representative_candidate_id>" \
  --subsumed "<id>=<ladder_top_exec_id>" \
  --budget-skipped "<id>=<reason>" \
  --waive "<id>=<reason>"
python3 "$SKILL_DIR/scripts/report_index.py" --eval-dir "$EVAL_DIR"
```

Report at the root (human), `fusion_unitside.json` in the working dir (it is KernelFusion apply-back's
input). Run this even when the gate FAILS — especially then. A failing 3.0 report at the
root is how the coverage gap reaches KernelFusion apply-back's denominator; a gate that fails and
publishes nothing is indistinguishable from a phase that never ran.

Return:
```json
{"status":"pass|partial|failed",
 "fusion_unitside_json":"<absolute path>",
 "fusion_unitside_md":"<absolute path>",
 "validated_count":0,
 "waived":[],"deferred":[],"notes":"..."}
```

## Inputs
- `FUSION_CANDIDATES_JSON` — the Phase 2.1 candidates.
- `CANDIDATE_ID` — the single candidate to validate this run.
- `IMAGE`, `MODEL_PATH`, `TP`, `CONTAINER` — the runtime. `EXEC_PREFIX`, when
  present, identifies a pre-provisioned runtime and every executable command must
  run through that literal prefix. Otherwise create a FRESH dated container from
  non-empty `IMAGE` and delete it when done. If neither is available, use the
  current environment only after proving the required stack imports; otherwise
  return an explicit unmeasured/waived reason rather than fabricating a verdict.
  `GPU_IDS` — the cards to use.
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
0. Resolve the runtime as described above. With `EXEC_PREFIX`, do not create or
   delete a nested container; use the supplied runtime. Without it, create the FRESH
   container from `IMAGE` (dated name; e.g. `geak_fusion_unitside_<model>_<date>`),
   bind the repo + model, and delete only that container when done. Source
   `SKILL_DIR/scripts/server_teardown.sh` and follow PROCESS SAFETY (only ever signal
   processes you started; never pattern-kill).
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
2c. **🔴 The reference must be as WIDE as what the fusion DELETES, not as wide as the
   captured region.** `members` is a slice of ONE trace region and `removable_row_ids` is
   required to be a subset of it — so per-step work the fused kernel makes unnecessary
   but that lives OUTSIDE that slice is structurally invisible to a reference built from
   members alone. Timing against a reference that omits work the fusion removes does not
   give you "the speedup, slightly pessimistic"; it answers a different question, and it
   reads exactly like a real loss. Observed on DSR1 2026-08-31: a V-absorb fusion
   measured `0.698` against a reference that omitted the per-step weight dequant the
   fusion deletes — the same fusion is worth **+10.34% e2e**. `0.698` was correctly
   measured and wrong.

   Before timing, ask at the seam: *what does the baseline do per step that the fused
   kernel will not have to do?* Walk outward from the `live_call_seam` file:line — a
   dequant hoisted into a neighbouring operator, a layout copy the caller performs, a
   scale recomputed every step — and put every such op INTO the `ref` leg.

   Then record BOTH answers, because the harness now requires them:
   - `ref_ops` — the op names the ref leg actually executed. Must cover every removable
     member op, or the harness errors.
   - `outside_work_removed` — ops removed that are NOT members. `[]` is a legal answer;
     leaving it unanswered is not. If non-empty, those ops must appear in `ref_ops` too.

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
     - 🔴 **`engaged=false` is ONLY for a RUNTIME guard inside the fused kernel itself**
       (a size/byte threshold that makes the kernel decline THIS shape). It is NOT for a
       compile-time dispatch gate in the *caller* (`_use_aiter_gfx95`, a `dtype ==
       float8_e4m3fn` clause, or "no sglang call site on any arch"). Those say the
       framework does not ROUTE to the kernel; they say nothing about whether the kernel
       RUNS. KernelFusion apply-back lands fusions with a sitecustomize overlay that REPLACES the seam
       and bypasses the caller's dispatcher entirely, so a caller-side gate is not a
       blocker — it is the thing the overlay exists to route around.
       When you meet a caller-side gate: call the fused API **directly** with the captured
       shapes, set `engaged=true` if it executes, and report parity + speedup normally.
       Record the gate under `dispatch_gate_note` (file:line + the condition) so apply-back knows
       the overlay must supply the arg the gate would have. Only if the kernel itself
       refuses to execute on this arch (import error, unsupported-arch abort, or an
       internal guard) is it `blocked`, with the actual error text as the reason.
     - 🔴 **A parity failure is a claim about YOUR harness until you have ruled the harness
       out.** Before reporting `parity: fail`, check the mundane causes first, since the
       fused kernel usually demands a stricter layout than the split path tolerates:
       non-contiguous weight views (`w.transpose(-1,-2)` — pass `.contiguous()` in the
       layout the kernel documents, e.g. (B,N,K)), transposed/row-vs-column scale layout,
       wrong group_size, fnuz-vs-fn fp8 variant, or comparing two independent fp8
       quantizations. Report which of these you eliminated. A candidate with a large
       isolated speedup and a parity failure you did NOT diagnose is an OPEN item, not a
       closed `blocked`. (Step 3b now enforces this for every family, at any speedup.)
   - **single-GPU family (norm/activation/quant/gemm-prologue) → 1-GPU microbench** on
     one `GPU_IDS` card: ref = the split member ops in sequence; cand = the fused API.
     `engaged=true` (no distributed guard).
3b. **🔴 `parity != pass` ⇒ the timing is UNDEFINED, not bad.** A fused kernel whose
   output diverges was, with high probability, FED WRONG — and a mis-fed kernel takes a
   wrong branch and a wrong tile config, so its `cand_ms` is the time of something that
   was never the candidate. "Parity failed AND it was slow" is not two independent
   strikes against the fusion; it is one fault counted twice. Observed on DSR1
   2026-08-31: a rope+KV fusion recorded `0.4849` with parity fail and was closed as a
   `fail`; the KB records the same kernel, same shape, same arch at **1.11–1.12x,
   parity-exact**.

   This holds for EVERY family, not just collectives, and at any speedup — the
   small-and-diverging case is the dangerous one, because it looks like a settled
   refutation instead of a broken instrument.

   The harness now renders such a row `needs_diagnosis` (never `fail`), nulls its
   `isolated_speedup`, and FAILS until you supply `parity_diagnosis`: the mundane causes
   you actually eliminated — non-contiguous weight views (pass `.contiguous()` in the
   layout the kernel documents), transposed/row-vs-column scale layout, wrong
   `group_size`, fnuz-vs-fn fp8 variant, two independent fp8 quantizations compared
   against each other, a forced `q_out_dtype`, cos/sin caches taken from the wrong
   object. A candidate at a high-cost seam must never be terminally closed on a number
   produced by a kernel that was demonstrably not receiving what it expects.

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
    "ref_ops": ["<every op the ref leg ran>"], "outside_work_removed": [],
    "parity_diagnosis": "<REQUIRED iff parity != pass: what you eliminated>",
    "engaged": true, "tp": 8, "notes": "how ref+cand were called; how engaged detected"}
   ```
   From this the harness derives **two independent axes** — read them, because they
   are what the board and the apply-back gate now consume:

   | | 含义 | 取值 |
   |---|---|---|
   | `correctness_status` | 它算对了吗 | `pass` / `fail` / `not_engaged` |
   | `perf_status` | 它更快吗 | `win` / `no_win` / `undefined` |
   | `failure_kind` | **失败是哪一种** | `none` / `performance` / `functional` / `unmeasured` / `not_engaged` |

   `perf_status` is `undefined` — and `isolated_speedup` is **nulled** — whenever
   `correctness_status != pass`. That is not bookkeeping: a mis-fed kernel takes a
   wrong branch and a wrong tile config, so its `cand_ms` timed something that was
   never the candidate. Reporting "parity failed **and** it was slow" as two strikes
   counts one fault twice, and that is precisely how DSR1 2026-08-31 closed a fusion
   at `0.4849` that the KB records at **1.11–1.12x, parity-exact**.

   So the only failure you may close on the spot is `performance` — 功能正确、就是不够
   快, measured on the candidate's own captured shape, so the number means what it
   says. `functional` requires that you first write `parity_diagnosis`; without it the
   verdict is `unmeasured`, which is **not a result** and fails the gate. You do not
   get to call a divergence a real functional failure until you have eliminated the
   mundane causes.

7. Tear down + DELETE only a container you created. Never delete the runtime
   represented by `EXEC_PREFIX`.

## Rules
- NEVER edit `fusion_unitside_harness.py` or weaken it. Your verdict is the input it
  gates; if it reports your verdict is untrustworthy (shape/fn/field), FIX the microbench
  and re-run — do not massage the harness. This includes the coverage gate: a
  `not_validated` row is fixed by benching that candidate or waiving it with a reason,
  NEVER by passing `--allow-partial-coverage` to make the red go away.
- `tested_shape` must be a real captured member shape and `fused_fn` a real
  `existing_apis` name — otherwise the verdict is rejected as untrustworthy.
- Report parity honestly. A fused kernel that diverges is a `parity: "fail"` — that is a
  valid, useful result (it stops a wrong fusion from being applied back), not something
  to hide.
- Do not touch the serving stack or measure e2e — that is KernelFusion apply-back. This
  role is isolated-only.

## Return (StructuredOutput)
```json
{"candidate_id": "...", "verdict_path": "<EVAL_DIR>/verdict/<id>.json",
 "parity": "pass|fail", "isolated_speedup": 0.0, "engaged": true,
 "tested_shape": [0,0], "fused_fn": "...", "container_deleted": true, "notes": "..."}
```
