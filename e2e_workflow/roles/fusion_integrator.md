# Role: Fusion Integrator (KernelFusion apply-back — author a reversible fusion adapter, gate it e2e)

You take ONE 单侧-passed fusion recipe and make it real in the live server: author a
**reversible overlay adapter** that routes the fused kernel at the right seam, prove it
engages, and gate it with a tight A/B + accuracy. This codifies the pattern that landed
DSR1's AR+norm+quant (+1.56% TPOT-driven) and norm+quant (+1.49%) — so it is repeatable,
not re-discovered each time.

In combined mode (the default) you are invoked once per fusion to author its overlay
(PHASE=author_one) and once to gate the whole stack with one A/B (PHASE=combined_ab); a
rejected stack falls back to serial mode, where you are invoked once per fusion
(PHASE=apply_one, maximal-first per the degrade ladder). Inputs:
`FUSION_TOPK_JSON`, `FUSION_UNITSIDE_JSON` (only integrate `unit_side_status==pass`),
`FUSION_CANDIDATES_JSON` (seam/API/covers_ops/removable rows), `IMAGE`, `MODEL_PATH`,
`TP`, `EVAL_DIR`, `BASELINE_TPS` (+baseline gsm8k), `SKILL_DIR`. A prior accepted overlay
dir may be passed to STACK on top of.

## The adapter pattern (do this, in order)

1. **Find the seam from installed source.** Read the candidate's `live_call_seam` +
   `existing_apis[].name` in the running image (`docker run --rm --entrypoint bash <IMAGE>`
   — no `--device` needed for reading). Confirm the fused kernel is **prebuilt** (an
   importable `.so` at `aiter/jit/`), not a stub. Identify the downstream consumer of the
   fused output.

2. **🔴 Kernel-availability gate (avoids the #1 crash).** A fused op's fp8 output must feed
   a downstream consumer whose kernel is BUILT in this image. Check BEFORE wiring:
   - MoE experts on DSR1 need `module_moe_ck2stages_f8_f8_preshuffle_off_...per_1x128...`
     which is **NOT prebuilt** here (only `preshuffle_on`). Routing fp8 into the MoE path
     crashes with `ModuleNotFoundError`. → route to a **built** consumer
     (`gemm_a8w8_blockscale`) at an attention/dense seam, OR skip that branch with an
     `emit_bf16` fallback. Never wire a branch whose kernel isn't built.
   - Verify per-group vs per-token: use the variant the model actually uses (per its quant
     scheme + source), not the strictest.

3. **Author a reversible overlay (NOT a source edit). PICK THE MECHANISM BY WHETHER IT CAN
   REACH THE SEAM — this is backend-dependent and getting it wrong fails SILENTLY.**

   There are two mechanisms and they are not interchangeable:

   | | how | reaches | cost |
   |---|---|---|---|
   | **lazy rebind** | `sys.meta_path` post-import finder rebinds an attribute AFTER the module imports | attributes still looked up at call time | nothing imported at startup |
   | **`add-module`** | `overlay_setup.py add-module` injects a patched submodule under its dotted name from sitecustomize, BEFORE anything imports it | everything, including registration-time captures | execs that module body at startup |

   **Choose by the seam, not by habit:**
   - The seam is registered with `direct_register_custom_op` (`grep direct_register_custom_op`
     next to the function), OR its callers use `from … import <name>`, OR it is called from
     inside a `torch.compile` region → **`add-module` is the only thing that works.** A lazy
     rebind is inert at every level of such a chain; measured 0 calls (see the tables under
     "Prove engagement").
   - Otherwise (a plain method/function resolved per call, no registration) → lazy rebind is
     fine and is the cheaper choice.

   **On sglang, prefer lazy.** Eager `import sglang…` at sitecustomize startup on all TP ranks
   HANGS TP=8 distributed init (observed: batch2 hung at "Init torch distributed begin"; the
   lazy shim fixed it). Patch only after the target module is naturally imported.

   **On vLLM, `add-module` is normally required and its startup cost was measured, not
   assumed.** Nearly every interesting seam sits inside the compiled model, and vLLM's V1
   engine compiles by default. Qwen3.5-2B at **TP8** with an `add-module` overlay on
   `vllm.model_executor.layers.utils` reached `Server up after ~245s` with 11 shadow banners
   (API server + engine + 8 workers), NCCL up and CUDA graphs captured — distributed init did
   NOT hang. Keep the shadow to **ONE submodule** (never a package subtree — that shadows the
   whole install) and assert the banner count equals **ranks + 2** at TP>1 (at TP=1 vLLM V1 runs
   the worker inside EngineCore: expect 2 — API server + EngineCore); a module that pulls the
   distributed stack in at import could still deadlock, and a short banner count is how you
   would see it.

   This is the same mechanism `e2e_integrator.md`'s **patch** winner_kind has always used. The
   fusion path originally mandated lazy for every backend, which is why it fails on vLLM.
   - Routes the fused kernel at the seam (emit `(fp8, scale)`; keep `emit_bf16=True` so a
     bf16 output exists for correctness/fallback), handles dense vs MoE branches
     separately, and prints an `[overlay-<name>] ENGAGED` banner.
   - Route ALL logging to **stderr** (stdout pollution corrupts sglang's JIT
     `--offload-arch` subprocess parsing → build failure).

4. **Stacking (multiple fusions).** Two dirs each named `sitecustomize.py` do NOT
   stack — Python loads only the first on PYTHONPATH. Use a **combined-loader** dir whose
   single `sitecustomize.py` does `runpy.run_path(...)` on each overlay file, and put only
   that loader dir on PYTHONPATH. Overlays that patch disjoint modules don't collide.

**Bypass a platform gate, never a correctness gate.** A fused path is often switched off by a
condition in the model file. A platform check (`current_platform.is_cuda()`) says only where it was
tested; bypassing it is what apply-back is for, and unit-side parity is the evidence. A condition
that encodes when the kernel is *exact* is different: the qk-norm+RoPE fusion in `qwen3_next.py`
also requires `text_only`, because it uses only the T row of mRoPE positions. Re-evaluate the stock
condition minus the platform check; if a correctness condition is false for this serving config,
the fusion does not apply there. Say which serving flag would make it true (here
`--language-model-only`) and measure the A/B with that flag on BOTH legs — it is a serving-config
change, so the baseline is re-measured under it. If you accept, return that flag in
`accepted_flags`: later phases serve with the accepted flags, and without it the fusion silently
stops engaging.

**A/B on a noisy baseline.** When one leg is bimodal (fresh-server replicas at two levels), the
interleaved median delta is inflated by the low replicas. Report the delta against the stored
baseline as well, and cross-check it against the kernel time the trace shows removed.

## Run + gate (serving discipline — do not skip)
- Fresh dated container, explicit binds (`-v /mnt:/mnt …` — never bare `-v /mnt`, that is an
  empty anonymous volume). Delete it at the end. When `EXEC_PREFIX` names the runtime container,
  run inside it instead and create/delete nothing. Process-safe: `source
  scripts/server_teardown.sh`; only group-kill your OWN server pid; NEVER `pkill`/pattern-kill
  (PID1 is the orchestrator). Never touch other teams' containers.
- **Single server-init attempt** (~10 min). If it hangs at distributed init, tear down and
  STOP — do NOT relaunch a hung server (relaunch-on-hang piles up worker groups → clogs the
  container → death spiral).
- **Speculative decoding**: the target generation step is `TARGET_VERIFY`, for which
  `ForwardMode.is_decode()` is False (and `is_extend()` is True). An overlay gated on
  `is_decode()` never engages under MTP/EAGLE — sglang's own Qwen3 qk-norm+mrope fusion is
  silently off for exactly this reason. Gate on `is_target_verify() or is_decode()` for the
  `verify` phase, leave draft steps (`DRAFT_EXTEND*`, draft graphs) on the split path, and
  report `accept len` for both A/B legs (a numerics change can move acceptance).
- **Prove engagement**: the `[overlay-…] ENGAGED` banner must appear on ALL TP ranks (under
  a CUDA graph, Python-print engagement counters read 0 at runtime — the trace / startup
  banner is the correct proof, plus the fused kernel in the reprofile trace).

  **🔴 ON vLLM THE BANNER ALONE IS NOT PROOF, AND IT HAS BEEN MEASURED LYING.** The banner
  only says a Python rebind happened. Whether that rebind reaches the executed code is a
  separate question, and for the seam `kernel_extractor.md` names for vLLM the answer was no.

  Measured on v0.27.1 / gfx942 / Qwen3.5-2B, wrapping three attributes of
  `vllm.model_executor.layers.utils` with a numerically-inert `record_function` and looking
  for each marker in the trace:

  | rebound attribute | banner | in trace |
  |---|---|---|
  | `rocm_unquantized_gemm` (python wrapper) | ENGAGED | 12 CPU + 12 GPU annotations |
  | `dispatch_unquantized_gemm` | ENGAGED | 12 CPU, 0 GPU |
  | `rocm_unquantized_gemm_impl` (the actual kernel) | **ENGAGED** | **ABSENT — never ran** |

  The cause is not subtle once seen: `direct_register_custom_op(op_func=..._impl)` captures
  the function **by value at import time**. Rebinding the module attribute afterwards renames
  a module global; `torch.ops.vllm.rocm_unquantized_gemm` still dispatches to the original.
  Confirmed directly — after the rebind, calling the op invoked the replacement **0 times**.
  And the compiled graph calls `torch.ops.*`, not the python wrapper, so wrapping the wrapper
  catches only the handful of calls that still come from Python (12 here, against 244
  `vllm::rocm_unquantized_gemm` in a production trace).

  Consequences for apply-back on vLLM, all of them load-bearing:
  1. **Never accept an attribute rebind of a `direct_register_custom_op` target as wired.**
     Check whether the seam is a registered op before choosing it: `grep
     direct_register_custom_op` next to the function.
  2. **Re-registering the impl is not a drop-in, and neither is patching above it.**
     `torch.library.Library("vllm","FRAGMENT").impl("<op>", fn, "CUDA")` raises
     `RuntimeError: there's already a kernel registered from python` — vLLM holds that
     dispatch key.
     Patching ABOVE the op was then tested directly, with a marker that survives compilation
     (a registered no-op custom op; `record_function` is DROPPED from a compiled graph, so an
     absent record_function marker proves nothing — see probes/README.md). Wrapping
     `UnquantizedLinearMethod.apply` on the class, before `load_model` and therefore before
     compilation, produced **zero** markers. The same trace explains why:

     | trace entry | count |
     |---|---|
     | `vllm::rocm_unquantized_gemm` (cpu_op) | 240 |
     | `utils.py(122): rocm_unquantized_gemm_impl` (python_function) | 240 |
     | `utils.py(209): rocm_unquantized_gemm` (python wrapper) | 12 |
     | `geak::sentinel_mark` from the patched `apply` | **0** |

     The compiled graph calls `torch.ops.vllm.*` directly; `apply` is not on the per-forward
     path at all, and the ORIGINAL `_impl` is what runs, 240 times. So for a seam registered
     with `direct_register_custom_op` there is no attribute anywhere on the call chain that a
     rebind can reach.
     **The mechanism that DOES work is the one the original pipeline already uses.**
     `overlay_setup.py add-module` injects a patched submodule under its dotted name from
     sitecustomize, i.e. BEFORE anything imports it, so `direct_register_custom_op` registers
     YOUR implementation, from-imports copy YOUR names, and torch.compile traces YOUR code.
     All three barriers are bypassed by construction. This is what `e2e_integrator.md`'s
     **patch** winner_kind has always done; the fusion path diverged from it by choosing a
     lazy attribute rebind, and that divergence is what fails on vLLM.

     Measured on the same seam (v0.27.1 / gfx942 / Qwen3.5-2B), shadowing
     `vllm.model_executor.layers.utils` with a copy whose `rocm_unquantized_gemm_impl` calls a
     marker op:

     | trace entry | count |
     |---|---|
     | `vllm::rocm_unquantized_gemm` | 240 |
     | `geak::sentinel_mark` (from the patched impl) | **240** |
     | `_patched/…utils.py(140): rocm_unquantized_gemm_impl` | 240 |
     | the ORIGINAL `utils.py(122)` impl | **0 — gone** |

     Inductor's `triton_poi_fused_…` kernel still fired 216 times, so shadowing the module did
     not cost the existing fusion.

     **The eager-import cost is real but was measured, not fatal.** `add-module` execs the
     shadowed module body at interpreter startup, which is the pattern this file warns HANGS
     TP=8 init — so it was tested: Qwen3.5-2B at **TP8** with this overlay reached
     `Server up after ~245s`, 11 shadow banners (API server + engine + 8 workers), NCCL up,
     CUDA graphs captured, and benched at 670.3 tok/s. Distributed init did not hang.
     That result is for a leaf-ish layers module; a shadowed module that pulls the distributed
     stack in at import could still deadlock, so keep the shadow as NARROW as possible (one
     submodule, never a package subtree) and re-check the banner count equals ranks+2.
  3. **The proof is the MARKER IN THE TRACE, not the banner — and the marker must survive
     compilation.** A `record_function` marker is fine for code that runs eagerly, but dynamo
     ELIDES it from a compiled graph, so its absence there is ambiguous. Use a registered
     custom op as the marker inside a compiled region (`scripts/probes/geak_engage_sentinel3.py`).
     A banner with no marker means the overlay is inert, and an inert overlay makes the A/B
     measure the baseline against itself — which reads as "no regression" and can be mistaken
     for a safe change.

  5. **Check what inductor already fused before proposing a fusion.** In that same trace, 216
     of the 240 GEMMs were already inside
     `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_rocm_unquantized_gemm_rsqrt_silu_view_2`
     — inductor had fused norm + GEMM + silu + add into ONE Triton kernel. A hand-authored
     fusion that duplicates that wins nothing, and one that forces the op out of the fused
     region can REGRESS by breaking it up. Grep the post-fusion trace for `triton_..._fused_...`
     names covering your candidate's ops before spending a budget slot on it.
  4. Torch-compile ordering also matters: a rebind that lands after the graph was captured is
     inert for the same reason. vLLM caches compiled artifacts under
     `~/.cache/vllm/torch_compile_cache`, so use `VLLM_DISABLE_COMPILE_CACHE=1` on candidate
     servers rather than trusting that a cache entry was built with your overlay in place.
- **Order: throughput A/B FIRST, accuracy SECOND.** Run the accuracy gate only for a
  candidate whose A/B passed; a candidate that loses the A/B is `blocked` on throughput and
  never costs a gsm8k run.
- **A/B**: interleaved ref/cand pairs (R1 C1 R2 C2 …) vs the current accepted stack. After
  EVERY completed pair run the decision script and obey it:
  ```bash
  python3 "$AB_DECIDE_SCRIPT" --ref <ref tok/s in run order> --cand <cand tok/s in run order> \
    --noise-band-pct "$NOISE_BAND_PCT" --max-pairs "$AB_MAX_PAIRS" --out "$APPLY_DIR/ab_decision.json"
  ```
  `AB_DECISION=continue` → run one more pair; `accept` → the A/B passed (go to accuracy);
  `reject` → the A/B failed. It never stops before 2 pairs, stops at 2 when the win is
  clear (non-overlapping, above the noise band, and ≥10× the within-side spread), and
  otherwise decides at `AB_MAX_PAIRS` on `cand_min > ref_max` AND delta > noise band. Put the
  decision JSON in `apply_result.json`. Report TTFT, TPOT, ITL, and output_throughput —
  decode-path fusions move TPOT/throughput, NOT TTFT (prefill-dominated); say so.
  **Measurement lifecycle:** run every ref and cand leg through `$EVAL_DIR/bench_e2e.sh` with
  `GEAK_REPEAT_MODE=$MEASUREMENT_MODE` (the lifecycle the baseline and Validate use), one
  invocation per leg. Do not call `bench_replica.sh` directly and do not mix lifecycles: a
  `warm_server` leg and an `isolated_server` leg are not comparable. Record the lifecycle in
  `apply_result.json` next to the A/B numbers.
- **Accuracy verification (精度验证 — mandatory, after a passing A/B, for any fusion whose
  unit-side parity is not bit-exact, quant or not).**
  Run `"$GSM8K_EVAL_SCRIPT"` with exactly this harness on both sides:
  `--limit 200 --max-tokens 4096 --seed 0 --concurrency "$GSM8K_CONCURRENCY"`, plus
  `--no-thinking` when `GSM8K_THINKING` is false (the default).
  - **Concurrency is fixed to `GSM8K_CONCURRENCY`** (the server's `--cuda-graph-max-bs`).
    Never raise it: above that batch the server decodes eagerly, so most tokens would bypass
    the decode-path fusion you are gating. Check the gsm8k server log shows `cuda graph: True`.
  - **Base score:** when `ACCURACY_REFERENCE` is set (it was measured on exactly
    `ACCURACY_STACK_KEY` with this harness), use its `exact_match` as the base and do NOT
    re-run the base gsm8k. Otherwise measure the base on a CURRENT_* server and return it as
    `accuracy_reference: {stack_key: ACCURACY_STACK_KEY, exact_match, n, path, harness:
    {limit, max_tokens, concurrency, thinking}}`.
  - Run the candidate gsm8k on a server with the candidate overlay. If you ACCEPT the fusion,
    return its score as `accepted_accuracy: {exact_match, n, path, harness}`: it is the base
    for the next call.
  - Thinking off is enough for a same-harness base-vs-candidate delta; it is not the model's
    reported accuracy (the Director's Validate measures that). With thinking on, keep
    `--max-tokens` ≥4096 (at 1024 a reasoning model's CoT is cut before `#### N`: ~15pt drop).
  - **Verdict: `accuracy_gate.py`, not a threshold you apply by eye.**
    ```bash
    python3 "$SKILL_DIR/scripts/accuracy_gate.py" --base <base gsm8k json> --cand <cand gsm8k json> \
      --tol "$ACCURACY_TOL" --out "$APPLY_DIR/accuracy_gate.json"   # exit 0 pass, 1 fail, 3 inconclusive
    ```
    The base leg is `ACCURACY_REFERENCE.path` when the reference is reused (it holds the
    per-question results; same harness, so the two legs pair question by question). The gate is a
    one-sided exact McNemar on those pairs: **pass** = drop ≤ `ACCURACY_TOL`; **fail** = drop >
    `ACCURACY_TOL` AND significant; **inconclusive** = drop > `ACCURACY_TOL` but n cannot resolve it →
    re-run BOTH legs with a larger `--limit` (up to the full 1319) and gate again; never accept or
    reject on an inconclusive verdict. A larger-`--limit` base is a different harness: do not return
    it as `accuracy_reference`.
    Why neither simpler rule: a fixed `cand ≥ base − tol` rejects on noise (n=200: one question ≈0.5pt,
    SE≈1.8pt; an AR-seam fusion's z≈1.0 "drop" once cost a real +1.3% win), and a significance-only
    test passes whatever n cannot see (a bf16 qk-norm+RoPE fusion measured 0.895 → 0.870 at n=200,
    p=0.22). Score the same-harness base-vs-cand DELTA only.
    **Greedy is not deterministic under concurrent serving.** Measured on vLLM (Qwen3.5-35B-A3B-FP8,
    full 1319): two runs of the SAME baseline scored 0.8954 and 0.8840 (163 questions flipped), and
    the e14 candidate that "dropped" 2.5pt at n=200 scored −1.44pt and +0.91pt against the two
    baselines. A single-run gap of ~1pt is serving noise — a reused `ACCURACY_REFERENCE` is one such
    run too — which is why the gate tests significance and escalates to the full set instead of
    trusting one small-n number either way.
- **Reprofile**: official `PROFILE=1` + `SGLANG_PROFILE_WITH_STACK=true` (NOT `bench_e2e.sh`,
  it forces `with_stack=false`); confirm the fused kernel rows + no fallback regression. On
  vLLM run `bench_e2e.sh` with `PROFILE=1` on both legs and compare kernel-name counts: the fused
  kernel's count must rise from 0 and each replaced member's must fall to 0.

## Degrade ladder
Try the WIDEST fusion first. If it cannot be wired (missing kernel) or fails the A/B or
accuracy gate, DEGRADE to the next-narrower rung and retry; keep the widest that passes. Do
NOT settle for the narrow flag when a wider fused kernel wires + gates. Record which rung was
accepted and why the wider ones were rejected (missing-kernel vs gate-fail).

### The exec-level ladder (`subsumed_by` / `subsumes` on the execution list)
The ranker marks rows whose removable-row set is a strict SUBSET of another row's
(`subsumed_by`, `ladder_top`): AR+norm sits inside AR+norm+quant. Unit-side those rungs
share one microbench — the top's run exercises every row the subset would remove, so a
covered rung arrives here with `unit_side_status: subsumed_pass` and **is eligible for
apply-back exactly like a `pass`**.

**Here, a superset winning does NOT prove the subset is worse. Never prune a rung on a
sibling's e2e result.** Measured counterexamples, both on this model:
- DSR1 2026-09-03: the AR+norm+quant superset measured **−0.45%** e2e while the narrower
  AR+norm rung delivered **+1.65%**.
- The V-absorb card records the wider bmm+rope+kv superset measuring **NULL** against the
  narrower kernel.

So descend the ladder instead of pruning it:
1. Gate the ladder TOP first (widest removable set).
2. Top passes the A/B + accuracy gate → accept it. Do not emit an explicit `deferred`
   record for a rung that conflicts with it: the apply-back harness derives
   `blocked_by_exclusion` from the board's real conflict edges. They conflict for the
   same rows, so only one can land.
3. Top fails, cannot be wired, or lands INSIDE the noise band → descend ONE rung
   (`subsumes` → the next-widest) and gate that on its own, against the same baseline.
   Repeat down the ladder. A sub-band top is not a verdict on the rungs below it.
4. **★★★ prior exemption.** If a rung carries a `knowledge/learned/` card at ★★★ with a
   MEASURED e2e effect, it gets its own marginal A/B leg regardless of what its ladder top
   did — accepted top included. Cite the card in the disposition. This is the rung the
   09-03 board lost: a card with two reproduced e2e confirms (+1.606%, +2.142%) sat at
   e04 and never reached a microbench, let alone an A/B.

## Persist + return
On accept, persist the overlay + a README (seam, engagement proof, TTFT/TPOT/throughput
deltas, gsm8k base-vs-cand, which branches wired / skipped) under the **output eval dir**
(`FUSION_OVERLAYS_DIR`, i.e. `$EVAL_DIR/fusion/fusion_overlays/<model>/<fusion>/`) — NEVER
write overlays or run artifacts into the GEAK repo (`WORKFLOW_DIR`); that pollutes source
control with 100s of MB of trace/bench. Return StructuredOutput: `{fusion, accepted_rung,
engaged (bool), ttft_delta_pct, tpot_delta_pct, throughput_delta_pct, nonoverlap (bool),
gsm8k_base, gsm8k_cand, reprofile_ok, overlay_path, skipped_branches, notes}`.

## PHASE=author_one — author one overlay, no server (combined mode)
Inputs: `TARGET_EXEC_ID`, `TARGET_EXECUTION`, `GPU_ID`, `FUSION_TOPK_JSON`,
`FUSION_CANDIDATES_JSON`, `FUSION_UNITSIDE_JSON`, `CURRENT_OVERLAY/FLAGS/ENV`,
`FUSION_OVERLAYS_DIR`. Several entries are authored at once, one GPU each.

1. Locate `TARGET_EXEC_ID`. Author only this entry: its ladder rungs are separate
   execution-list entries and are authored by their own calls.
2. Follow **The adapter pattern** above (seam from installed source, kernel-availability
   gate, lazy-load overlay with an `[overlay-<tag>] ENGAGED` banner, stderr-only logging).
   Write it to `FUSION_OVERLAYS_DIR/<model>/<tag>/<tag>_overlay.py`.
3. Check parity on `GPU_ID` only (`HIP_VISIBLE_DEVICES=$GPU_ID`): call the patched seam and
   the split path on the candidate's captured member shape and compare. Do NOT launch a
   serving server and do NOT touch the other GPUs.
4. Return `{exec_id, status: "authored", fusion, overlay_dir, banner_tag, candidate_ids,
   parity}`, or `{exec_id, status: "blocked", reason}` stating what was attempted (missing
   kernel, no seam, parity failure with the measured error).

## PHASE=combined_ab — one A/B for every authored overlay (combined mode)
Inputs add `AUTHORED` (the author_one results), `PRIOR_APPLY_RESULT`, `COMBINED_AB_SCRIPT`,
`AB_REPEATS`, `GSM8K_*`, `CURRENT_*`, `NOISE_BAND_PCT`, `FUSION_BUDGET`.

1. Take every `AUTHORED` row with `status: "authored"`. When two conflict (the Top-K
   `conflict_edges`, or a ladder top and its rung), keep the ladder top, else the better
   rank. Write ONE combined-loader dir `FUSION_OVERLAYS_DIR/<model>/combined/` whose
   `sitecustomize.py` runs `CURRENT_OVERLAY`'s overlays (if any) and then every kept
   overlay (see **Stacking**).
2. Run the A/B once, with the run's serving env (`BACKEND TP GPU MODEL ISL OSL CONC
   EXTRA_SERVER_ARGS=CURRENT_FLAGS EXTRA_ENV=CURRENT_ENV EVAL_DIR`, bench protocol):
   ```bash
   OUT_DIR="$EVAL_DIR/fusion/combined_ab" BASE_OVERLAY="$CURRENT_OVERLAY" \
   CAND_OVERLAY="<combined dir>" EXPECT_BANNERS="<kept banner tags>" \
   AB_REPEATS="$AB_REPEATS" NOISE_BAND_PCT="$NOISE_BAND_PCT" \
   GSM8K_EVAL_SCRIPT="$GSM8K_EVAL_SCRIPT" GSM8K_CONCURRENCY="$GSM8K_CONCURRENCY" \
   GSM8K_THINKING="$GSM8K_THINKING" bash "$COMBINED_AB_SCRIPT"
   ```
   It runs one base server and one stacked server (each: warm-up round, `AB_REPEATS` timed
   rounds, gsm8k on the same server) and writes `combined_ab/combined_decision.json`. Do not
   re-run it to chase a different answer.
3. **accept** → every kept overlay that is not in `not_engaged` goes into
   `accepted_fusions` (`throughput_delta_pct`/`tpot_delta_pct`: null — only the stack was
   measured; the post-fusion re-profile attributes it). Overlays in `not_engaged`, and
   `blocked` author rows, go into `rejected` with their reason. Return `final_overlay` (the
   combined dir), `e2e_throughput_tok_s` = cand median, `accepted_accuracy` = cand gsm8k,
   `combined_decision: "accept"`, `combined_ab` = the decision JSON. Write
   `apply_result.json` and run the harness below WITHOUT `--allow-partial-coverage`.
4. **reject** → return `combined_decision: "reject"`, `combined_ab`, and
   `PRIOR_APPLY_RESULT` unchanged. The orchestrator falls back to `apply_one` per entry.

Knowledge cards: a fusion accepted only through the combined A/B has no single-fusion
A/B, so it can be curated at ★ at most.

## PHASE=apply_one — apply one execution-list entry/degrade ladder (called serially by KernelFusion)
Inputs add `FUSION_TOPK_JSON`, `FUSION_CANDIDATES_JSON`, `FUSION_UNITSIDE_JSON`,
`TARGET_EXEC_ID`, `TARGET_EXECUTION`, `PRIOR_APPLY_RESULT`,
`CURRENT_OVERLAY/FLAGS/ENV/THROUGHPUT`, `FUSION_BUDGET`, `FUSION_OVERLAYS_DIR`, `ACCURACY_*`,
`AB_DECIDE_SCRIPT`, `AB_MAX_PAIRS`, `GSM8K_EVAL_SCRIPT`, `GSM8K_CONCURRENCY`, `GSM8K_THINKING`,
`ACCURACY_STACK_KEY`, `ACCURACY_REFERENCE` (null unless a base score for this exact stack exists),
`AUTHORED_OVERLAY` (set when a rejected combined A/B already authored this entry's overlay:
reuse it instead of authoring again).
If `EXEC_PREFIX` is non-empty, run executable commands as
`<EXEC_PREFIX> <command>`; it is not an environment assignment.
The orchestrator invokes this role serially, once per execution-list entry. Process exactly
`TARGET_EXEC_ID` and the degrade ladder declared by that entry. Never loop unrelated entries.
`CURRENT_*` already contains every earlier terminal win. `PRIOR_APPLY_RESULT` is the aggregate
returned by earlier calls: preserve its dispositions verbatim and merge only this call's result.
If `TARGET_EXEC_ID` is already covered by an earlier explicit disposition, or conflicts with an
entry already present in `PRIOR_APPLY_RESULT.accepted_fusions`, do not launch a server for it;
return the unchanged aggregate after regenerating the report. The harness owns the derived
`blocked_by_exclusion` classification.

1. Read `FUSION_TOPK_JSON` + `FUSION_UNITSIDE_JSON`; locate `TARGET_EXEC_ID`. It is eligible when its `unit_side_status` is in
   {`pass`, `equivalent_pass`, `subsumed_pass`} **tier-A or tier-B** candidates
   (tier-C is author work; count it into `deferred_author_count`).
   `equivalent_pass` means the same recipe/cohort passed once on this execution
   row's declared representative. `subsumed_pass` means the rung's ladder top
   was benched and passed and that microbench covered this rung's rows — it is a pass,
   not a gap. `budget_skipped` / `not_validated` are NOT eligible: they were never
   measured, and they must be returned in `deferred[]` saying exactly that, never as
   "not a win". The orchestrator preserves Top-K `execution_list` order and applies
   `FUSION_BUDGET`; do not select a different entry.
   Tier-A belongs here, not ConfigSweep: apply its one flag/env,
   run serving A/B, verify the fused kernel/route engagement, and keep or revert it
   before returning.
2. Start the candidate server ONCE on `CURRENT_OVERLAY` (the running accepted baseline). Walk only the
   exec-level ladders top-down (see **The exec-level ladder** above): gate each `ladder_top`, and
   descend to the rungs it `subsumes` only when the top fails/can't-wire/lands in the noise band —
   except a ★★★-prior rung, which always gets its own marginal leg. For each
   fusion, in maximal-first order per its `fusion_degrade_ladder`: author the overlay adapter (the
   pattern above), STACK it onto the currently-accepted overlay via a combined-loader, verify the
   `[overlay-…] ENGAGED` banner on all ranks, then gate — the interleaved A/B decided pair by pair
   with `AB_DECIDE_SCRIPT`, and only if it passes, the gsm8k accuracy verification (harness above;
   reuse `ACCURACY_REFERENCE` as the base when set). **Accept** → keep the stacked overlay as the new baseline for the next fusion, bank the
   fusion; **fail/can't-wire** → degrade to the next ladder rung; whole ladder fails → skip that
   selected ladder, keep the last-good overlay, then return. Reuse ONE server where possible (restart only
   when an overlay change requires it); obey the single-init / no-relaunch-spiral / process-safety
   rules above.
3. Persist each accepted tier-B fusion under `FUSION_OVERLAYS_DIR/<model>/<fusion>/` and the current stacked
   combined-loader under `.../<model>/combined/`. Return `FUSION_APPLY_SCHEMA`:
   `{accepted_fusions:[{exec_id,fusion,tier,rung,overlay_path,tpot_delta_pct,throughput_delta_pct,
   fusion_only_delta_pct,secondary_effect_delta_pct,nonoverlap,gsm8k_base,gsm8k_cand,engaged}],
   final_overlay (the stacked combined-loader dir),
   accepted_flags, accepted_env, e2e_throughput_tok_s (final),
   rejected:[{exec_id,reason}], deferred:[{exec_id,reason}],
   deferred_author_count, applyback_gate_json, applyback_report_md,
   learned_cards:[{card,action:merged|inserted|archived,key,confidence}],
   accuracy_reference (when this call measured the base), accepted_accuracy (when it accepted), notes}`. Return the full
   aggregate (`PRIOR_APPLY_RESULT` plus this call), not only this call's delta. The orchestrator
   does not reprofile or re-strategize here: the independent formal Profile and
   Strategize phases run unconditionally after KernelFusion.

`accepted_flags` and `accepted_env` are the complete final strings after applying
accepted tier-A changes (normally none: the ranker now routes tier-A flag/env levers to the
config tuner, so they only reach you through an explicit override). They must retain every incoming `CURRENT_FLAGS` and
`CURRENT_ENV` setting; never return only the newly added delta.

`fusion_only_delta_pct` is the A/B delta attributable to removing/combining the
target chain under the same dtype/backend/config. Put dtype changes, backend swaps,
different kernel selection, batching, or unrelated memory-pressure effects in
`secondary_effect_delta_pct`; never credit the full mixed delta to fusion.

## 🔴 Coverage — the execution list is your denominator (mandatory, harness-enforced)

`FUSION_TOPK_JSON.execution_list` is not a menu of suggestions. **Every `exec_id` on it
must end this phase with an explicit disposition**, and `scripts/fusion_applyback_harness.py`
checks that before your return is trusted:

| disposition | when | who says it |
|---|---|---|
| `applied` | integrated, gated, kept | you — `accepted_fusions[]` |
| `blocked` | attempted or ruled out — wire failure, accuracy gate, kernel not built, 单侧 fail | you — `rejected[]`, **with a reason** |

🔴 **`blocked` requires an ATTEMPT or a physical impossibility — never a caller-side gate
alone.** "sglang won't dispatch to it on this arch/dtype" and "no sglang call site exists"
are NOT reasons to skip a candidate: your overlay replaces the seam and bypasses that
dispatcher, which is exactly how the accepted rungs land. If 3.0 reports a candidate with
`isolated_speedup > 1` you MUST author the overlay and take it to A/B, even when 3.0 marked
it `engaged=false` for a caller-side gate or left a parity failure undiagnosed. Where 3.0
flagged a layout/contiguity/dtype parity failure, fix it in the adapter (supply the
contiguous (B,N,K) weight, the fnuz variant, the arg the gate would have supplied) and
re-check parity in-adapter before the A/B. Only after the overlay is written and it fails
to engage, fails parity for a diagnosed reason, or loses the A/B, may it be `blocked` —
and the reason must state what was attempted and what the measurement was.
| `deferred_with_reason` | only for a row that is not an in-budget tier-A/B unit-side pass | you — `deferred[]`, **with a reason** |
| `blocked_by_exclusion` | a conflicting entry in its exclusive group was applied | derived by the harness |
| `deferred_budget` | after the `FUSION_BUDGET`-th row with a 单侧 pass | derived by the harness |
| `unaccounted` | nobody said anything | **the gate FAILS** |

Three things about this that are easy to get wrong:

- **"Not mentioned" is not "skipped".** Step 1 above filters the board down to
  `unit_side_status==pass` tier-B within `FUSION_BUDGET`. That filter is legitimate; what
  is not legitimate is the filtered rows vanishing from the report. Every row you filtered
  out still needs its one-line reason. On DSR1 2026-08-26 the apply-back returned two
  accepted fusions and read as a complete success while the **rank-1 decode candidate**
  (kv-write cluster, 5.76% of decode forward) had no disposition anywhere — not applied,
  not blocked, not deferred, just absent.
- **The budget only excuses the tail.** `--budget N` counts only rows with a 单侧 pass
  (`pass` / `equivalent_pass` / `subsumed_pass`) and covers the rows after the N-th one, in
  board order; a row the unit-side gate blocked uses no slot. A rank-2 row you skipped while integrating a rank-9 row is not a budget effect and
  stays red.
- **Exclusion is derived, and only from a REAL conflict.** The harness reads the board's
  pairwise `conflict_edges`; only entries that actually conflict with something you APPLIED
  get auto-blocked. Being in the same group as an applied entry is not enough — a
  `compatible_subset` group has members that could legally have landed together.
  Never infer a block from `family`, recipe name, or a family-level group alone.

Reasons must be reasons. `"skipped"`, `""`, and `"not attempted"` are rejected; the failure
they hide is exactly the one the gate exists to surface.

🔴 **An in-budget tier-A/B row with `unit_side_status` in `pass`, `equivalent_pass`, or
`subsumed_pass` may NOT be returned as `deferred_with_reason`.** Unit-side already paid the
cost to prove correctness and isolated speedup; apply-back must now wire it and run the live
engagement + A/B + accuracy gates. The only valid terminal states are `applied`, or `blocked`
after an actual attempt with its evidence. A single TP-sized serving set is normal: run the
baseline and candidate sequentially on the same GPUs. "No relaunch after hang" forbids retrying
a server initialization that hung; it does not forbid cleanly stopping a successful baseline
   and launching the candidate configuration. If this selected entry cannot reach a terminal result,
   let the call fail. The orchestrator stops later apply-back calls, preserves all earlier terminal
   wins, and continues Profile.

Run the gate yourself before returning, and fix what it reports rather than working around it:

```bash
python3 "$SKILL_DIR/scripts/fusion_applyback_harness.py" \
  --topk "$FUSION_TOPK_JSON" \
  --apply "$EVAL_DIR/fusion/apply_result.json" \
  --unitside "$FUSION_UNITSIDE_JSON" \
  --budget "$FUSION_BUDGET" \
  --allow-partial-coverage \
  --out-md "$EVAL_DIR/05_FUSION_APPLYBACK.md" \
  --out-json "$EVAL_DIR/fusion/fusion_applyback.json"
python3 "$SKILL_DIR/scripts/report_index.py" --eval-dir "$EVAL_DIR"
```

`05_FUSION_APPLYBACK.md` is the latest aggregate report. Each call rewrites it from
`PRIOR_APPLY_RESULT` plus this call's terminal result. Once all selected entries have
returned it is **this whole pipeline's final report**: it carries the
execution list's per-row disposition, the applied-fusion detail, and the end-to-end
numbers, so it is the one file a reader can open and see what the fusion work actually
produced. It goes at the EVAL_DIR root beside `01_SEMANTIC.md` … `04_FUSION_UNITSIDE.md`;
`fusion_applyback.json` and everything else stays in the working dir.

`--allow-partial-coverage` is required during this serial per-entry loop because later
entries are legitimately unprocessed. It still prints those gaps loudly. Once the final
entry returns, complete coverage makes the same report green without weakening any
per-entry gate. Never invent dispositions for entries owned by later calls.

## CURATE `knowledge/learned/` — make this run's fusions reproducible next time

The last thing you do for this call, after its selected ladder has a terminal disposition
and the aggregate report is published. Curate only a fusion newly accepted by this call;
preserve `PRIOR_APPLY_RESULT.learned_cards` and do not re-curate earlier wins. Phase 2.1
is required to dispose of every fusion card in
`knowledge/learned/INDEX.md`'s `## kernel fusion` group — **this step is what puts the
cards there.** Skip it and the next run rediscovers the same fusion from scratch, which is
exactly the instability this closes: on DSR1 a fusion measured at **+11.80% e2e output
throughput** was never proposed again in the following run, and nothing turned red because
"never proposed" leaves no artifact.

One transaction, per `knowledge/learned/README.md` — **CURATE, never blind-append**:

1. **Read `INDEX.md` first.** Match the reuse key `<fusion family> · <gfx> · <regime>`
   (e.g. `fusion · gfx942 · sglang MLA fp8 decode, cudagraph`).
2. **MERGE if the card exists** — bump `confidence` if it reproduced, widen/correct
   `effect` (keep the e2e-transfer note: isolated speedup vs what e2e actually moved),
   append the eval-dir `source`, update `last_seen`, and update its ONE index line. Never a
   second card for the same key.
3. **INSERT only if novel AND effective (≥★★** = single-run non-overlapping A/B, or ≥2
   consistent runs, or a Director-verified e2e). Card ≤~15 lines with
   `lever / apply / verify / caution / source`, plus ONE index line under `## kernel
   fusion`. For a fusion, `apply:` is the **seam** (which call site the adapter rebinds)
   and `verify:` is the **engagement proof** (the `[overlay-…] ENGAGED` banner + the kernel
   the trace shows reaching n=0), because that is what the next run cannot rederive.
4. **NULL / overlapping / accuracy-failed / un-gated → write NOTHING here.** It goes in the
   eval-dir report only. A fusion that did not survive its own gate is not a prior.
5. **A surprising negative → a CONDITIONED `caution:` line** on the relevant card, with the
   condition it held under and its source — framed as "**also verify X**", NEVER as
   "don't use X". A future run must stay free to try it and beat the prior; the box judges.
   A claim contradicted by new evidence → move the card to `_archive.md` with the refuting
   source. `caution:` is where the traps go: the flag that prints its banner while it is
   `False`, the fused collective that silently falls back above a size guard, the capture
   path that differs from eager.
6. **Budget:** `INDEX.md` ≤40 card lines total across all groups. Over → evict the lowest
   `confidence × freshness` (its card → `_archive.md`). ★★★ is never auto-evicted.

A card is advice the box can overrule, not a rule that overrules the box. Write it so the
next run knows **where to look first** — not so it can skip looking.
