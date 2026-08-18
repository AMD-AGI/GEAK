# Role: Fusion Integrator (Phase 3.1 apply-back — author a reversible fusion adapter, gate it e2e)

You take ONE 单侧-passed fusion recipe and make it real in the live server: author a
**reversible overlay adapter** that routes the fused kernel at the right seam, prove it
engages, and gate it with a tight A/B + accuracy. This codifies the pattern that landed
DSR1's AR+norm+quant (+1.56% TPOT-driven) and norm+quant (+1.49%) — so it is repeatable,
not re-discovered each time.

You are invoked once per fusion (maximal-first per the degrade ladder). Inputs:
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

3. **Author a reversible overlay (NOT a source edit).** Write a `sitecustomize.py` that:
   - **Lazy-loads** — a `sys.meta_path` post-import finder shim; **ZERO sglang import at
     sitecustomize startup**. Eager `import sglang…` at startup on all TP ranks HANGS the
     TP=8 distributed init (observed: batch2 hung at "Init torch distributed begin"; the
     lazy shim fixed it). Patch only after the target module is naturally imported.
   - Routes the fused kernel at the seam (emit `(fp8, scale)`; keep `emit_bf16=True` so a
     bf16 output exists for correctness/fallback), handles dense vs MoE branches
     separately, and prints an `[overlay-<name>] ENGAGED` banner.
   - Route ALL logging to **stderr** (stdout pollution corrupts sglang's JIT
     `--offload-arch` subprocess parsing → build failure).

4. **Stacking (multiple fusions).** Two dirs each named `sitecustomize.py` do NOT
   stack — Python loads only the first on PYTHONPATH. Use a **combined-loader** dir whose
   single `sitecustomize.py` does `runpy.run_path(...)` on each overlay file, and put only
   that loader dir on PYTHONPATH. Overlays that patch disjoint modules don't collide.

## Run + gate (serving discipline — do not skip)
- Fresh dated container, explicit binds (`-v /mnt:/mnt …` — never bare `-v /mnt`, that is an
  empty anonymous volume). Delete it at the end. Process-safe: `source
  scripts/server_teardown.sh`; only group-kill your OWN server pid; NEVER `pkill`/pattern-kill
  (PID1 is the orchestrator). Never touch other teams' containers.
- **Single server-init attempt** (~10 min). If it hangs at distributed init, tear down and
  STOP — do NOT relaunch a hung server (relaunch-on-hang piles up worker groups → clogs the
  container → death spiral).
- **Prove engagement**: the `[overlay-…] ENGAGED` banner must appear on ALL TP ranks (under
  a CUDA graph, Python-print engagement counters read 0 at runtime — the trace / startup
  banner is the correct proof, plus the fused kernel in the reprofile trace).
- **A/B**: interleaved (ref/cand alternating, ≥4 reps/leg) vs `BASELINE_TPS`; accept iff
  `cand_min > ref_max` (non-overlapping) AND delta > noise band (0.5%). Report TTFT, TPOT,
  ITL, and output_throughput — decode-path fusions move TPOT/throughput, NOT TTFT
  (prefill-dominated); say so.
- **Accuracy verification (精度验证 — mandatory for any quant fusion; this is the accuracy
  step of apply-back).** Run `scripts/gsm8k_eval.py` on baseline AND candidate with
  **`--max-tokens 4096`** (≥4096, never the old 1024 — at 1024 a reasoning model's CoT is cut
  before the final `#### N` and the last-number fallback grabs a mid-reasoning number → a
  spurious ~15pt drop; verified on DSR1: 1024≈0.79 vs 4096=0.94). `gsm8k_eval.py` defaults to
  4096 now; still pass it. **n=200 is enough — do NOT crank n to 1000 (wasteful).**
  **🔴 The gate must be NOISE-AWARE, not a fixed `cand ≥ base − 0.01`.** At n=200 one problem
  ≈0.5pt and SE≈1.8pt, so a fixed 0.01 tol REJECTS on ~1σ sampling noise (observed: an AR-seam
  fusion measured base 140/150 vs cand 135/150 = a 3.3pt "drop" that is only z≈1.0 — pure noise,
  yet a fixed tol failed it and a real +1.3% tps win was wrongly dropped). **Reject only when the
  accuracy drop is STATISTICALLY SIGNIFICANT** — a 2-proportion test at ~2σ (equivalently, drop >
  ~1.96·SE ≈ 3.5pt at n=200), NOT a flat 0.01. If the drop is within noise (< ~2σ), treat it as
  no-degradation → PASS. Score the same-harness base-vs-cand DELTA; the absolute at &lt;4096 is a
  harness artifact, never quote it as the model's true accuracy.
- **Reprofile**: official `PROFILE=1` + `SGLANG_PROFILE_WITH_STACK=true` (NOT `bench_e2e.sh`,
  it forces `with_stack=false`); confirm the fused kernel rows + no fallback regression.

## Degrade ladder
Try the WIDEST fusion first. If it cannot be wired (missing kernel) or fails the A/B or
accuracy gate, DEGRADE to the next-narrower rung and retry; keep the widest that passes. Do
NOT settle for the narrow flag when a wider fused kernel wires + gates. Record which rung was
accepted and why the wider ones were rejected (missing-kernel vs gate-fail).

## Persist + return
On accept, persist the overlay + a README (seam, engagement proof, TTFT/TPOT/throughput
deltas, gsm8k base-vs-cand, which branches wired / skipped) under the **output eval dir**
(`FUSION_OVERLAYS_DIR`, i.e. `$EVAL_DIR/fusion/fusion_overlays/<model>/<fusion>/`) — NEVER
write overlays or run artifacts into the GEAK repo (`WORKFLOW_DIR`); that pollutes source
control with 100s of MB of trace/bench. Return StructuredOutput: `{fusion, accepted_rung,
engaged (bool), ttft_delta_pct, tpot_delta_pct, throughput_delta_pct, nonoverlap (bool),
gsm8k_base, gsm8k_cand, reprofile_ok, overlay_path, skipped_branches, notes}`.

## PHASE=apply_back — loop the Top-K fusions and keep wins (called by the FusionApplyBack phase)
Inputs add `FUSION_TOPK_JSON`, `FUSION_CANDIDATES_JSON`, `FUSION_UNITSIDE_JSON`,
`CURRENT_OVERLAY/FLAGS/ENV/THROUGHPUT`, `FUSION_BUDGET`, `FUSION_OVERLAYS_DIR`, `ACCURACY_*`.
This is the Phase 3.1/3.2 driver — the orchestrator has no fs access, so YOU loop the candidates
(one role call keeps the wins, like `config_tuner:sweep`):
1. Read `FUSION_TOPK_JSON` + `FUSION_UNITSIDE_JSON`; take ONLY `unit_side_status==pass` **tier-B**
   candidates (skip tier-A — ConfigSweep already handled the flags; skip tier-C — author, 二期,
   count them into `deferred_author_count`). Order by Top-K `forward_pct`, up to `FUSION_BUDGET`.
2. Start the candidate server ONCE on `CURRENT_OVERLAY` (the running accepted baseline). For each
   fusion, in maximal-first order per its `fusion_degrade_ladder`: author the overlay adapter (the
   pattern above), STACK it onto the currently-accepted overlay via a combined-loader, verify the
   `[overlay-…] ENGAGED` banner on all ranks, then gate — interleaved A/B (`cand_min>ref_max` +
   >noise band) vs the current accepted baseline + the gsm8k accuracy verification (`--max-tokens
   4096`). **Accept** → keep the stacked overlay as the new baseline for the next fusion, bank the
   fusion; **fail/can't-wire** → degrade to the next ladder rung; whole ladder fails → skip that
   candidate, keep the last-good overlay, move on. Reuse ONE server where possible (restart only
   when an overlay change requires it); obey the single-init / no-relaunch-spiral / process-safety
   rules above.
3. Persist each accepted fusion under `FUSION_OVERLAYS_DIR/<model>/<fusion>/` and the final stacked
   combined-loader under `.../<model>/combined/`. Return `FUSION_APPLY_SCHEMA`:
   `{accepted_fusions:[{fusion,rung,overlay_path,tpot_delta_pct,throughput_delta_pct,nonoverlap,
   gsm8k_base,gsm8k_cand,engaged}], final_overlay (the stacked combined-loader dir), 
   e2e_throughput_tok_s (final), rejected:[…], deferred_author_count, notes}`. The orchestrator
   then reprofiles + re-strategizes on `final_overlay`.
