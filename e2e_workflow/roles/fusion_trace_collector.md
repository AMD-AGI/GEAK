# Fusion Trace Collector

You are the trace-capture role owned exclusively by KernelFusion, which runs right
after the formal Profile. Produce the manifest of a raw production serving trace;
do not parse Top-N, classify kernels, enrich roofline data, or perform GEAK Profile work.

Inputs: `EVAL_DIR`, `MODEL_PATH`, `GPU_ID`, `WORKLOAD`, `ROUND`,
`CAPTURE_DIR`, `TRACE_MANIFEST_JSON`, `CAPTURE_REPEATS`,
`CAPTURE_NUM_PROMPTS`, `OVERLAY_PYTHONPATH`, `EXTRA_SERVER_ARGS`,
`EXTRA_ENV`, `BACKEND`, `PHASE_ANNOTATION` (`native` | `vllm_overlay_hook`),
optional `PROFILE_TRACE_DIR`, optional `TRACELENS_TRACE_FILE`,
`EXEC_PREFIX`, and `SKILL_DIR`.

When `EXEC_PREFIX` is non-empty, run executable commands as
`<EXEC_PREFIX> <command>`; do not treat it as an environment assignment.

0. **Reuse the formal Profile trace first.** When `PROFILE_TRACE_DIR` is set and
   holds the Profile's raw traces (captured on this same stack, with
   `SGLANG_PROFILE_WITH_STACK=true`), build the manifest from it and capture nothing:
   ```bash
   python3 "$SKILL_DIR/scripts/trace_capability.py" --trace-dir "$PROFILE_TRACE_DIR" \
     --auto-select-rank --require-phases prefill,decode --out "$TRACE_MANIFEST_JSON"
   ```
   Accept it only if the manifest has `status == "pass"` and a non-empty
   `analysis_rank_trace`; then return it with `source: "profile-trace"`. If it fails,
   say why in `notes` and fall through to step 1.
   **`--require-phases` is not optional.** The Profile is sized for a steady-state decode
   sample and may skip the prefill ramp to get one (`PROFILE_DELAY_ITERS`); such a trace
   has no prefill step at all. Measured on Kimi-K2.5 TP8 (vLLM 0.21): the reused trace was
   decode-only, and every Semantic table, candidate and Top-K row came out decode-only with
   "prefill ≈ 0 µs" — the prefill side of the workload was never looked at.
   `missing_required_phases` names what is absent; fall through and capture.
1. Reuse `TRACELENS_TRACE_FILE` only when it exists and contains a usable
   top-level serving trace. An analysis Markdown file is not a raw trace.
2. **Phase annotation** (only when you capture yourself in step 3). `PHASE_ANNOTATION=native`
   (the default, including for vllm) means the stack annotates its own steps and you do
   nothing here.

   On vLLM ≥ 0.27 that is the case: `gpu_worker` wraps every `execute_model` in
   `annotate_profile()` unconditionally and emits
   `execute_context_<nreq>(<ntok>)_generation_<nreq>(<ntok>)`, the legacy dialect
   `semantic_kernel_mapping` already reads. Nothing to enable.

   **If `PHASE_ANNOTATION=vllm_overlay_hook`, build the capture overlay BEFORE launching.**
   This is the fallback for a build that emits nothing (pre-0.27, or a fork that dropped
   `annotate_profile`) — a trace without step spans phase-tags **nothing**, so
   `semantic_kernel_mapping` builds no phase table and Phase 2.1 has no denominator. The
   capture server then runs with a small, capture-only overlay whose post-import hook makes
   vLLM emit sglang's dialect instead (`step[EXTEND bs=N toks=M]` / `step[DECODE bs=N]`):

   ```bash
   CAP_OVERLAY="$EVAL_DIR/fusion/capture_overlay"
   python3 "$SKILL_DIR/scripts/overlay_setup.py" add-vllm-phase-annotation \
     --overlay "$CAP_OVERLAY"
   # PREPEND it; the accepted-kernel overlay (OVERLAY_PYTHONPATH) must stay in effect.
   CAP_PYTHONPATH="$CAP_OVERLAY${OVERLAY_PYTHONPATH:+:$OVERLAY_PYTHONPATH}"
   ```
   Pass `OVERLAY_PYTHONPATH="$CAP_PYTHONPATH"` to `bench_e2e.sh`. The hook is inert unless
   `GEAK_VLLM_PHASE_ANNOTATE=1`, which the orchestrator already put in `EXTRA_ENV` — never
   add it to a measurement run, only to this capture.

   **Prove it engaged**: grep the server log for `[GEAK_PHASE_ANNOTATE] ENGAGED`. If it is
   absent the trace will be phase-blind; report that in `notes` and set
   `phase_evidence_status=unresolved` rather than passing off a prefill-only capture as
   full coverage.

   This overlay is for the CAPTURE only. Do not carry it into the A/B or apply-back
   servers: it wraps `execute_model` on every step and its cost is not in the baseline.

3. Otherwise (no usable Profile or TraceLens trace) run the existing `EVAL_DIR/bench_e2e.sh` serving capture with
   `OUT_DIR=CAPTURE_DIR`, `REPEATS=CAPTURE_REPEATS`,
   `NUM_PROMPTS=CAPTURE_NUM_PROMPTS`, and `PROFILE=1`, preserving the supplied
   overlay, flags, env, and Fusion-only profiler controls. `CAPTURE_DIR` is the
   one authoritative output directory; do not insert an extra `profile/` or
   `round_` component around it. The short pre-profile benchmark only warms the
   production path and supplies TPOT for profiler sizing; it is not an
   acceptance measurement, so do not expand its repeat or prompt count.
   `GEAK_FUSION_TRACE=1` switches the capture out of the native profiler's statistical
   window into fusion evidence mode; what that means is per-backend and the adapter owns it:
   - **sglang** — `PROFILE_NUM_STEPS=1` (one forward per separately captured stage, since
     `profile_by_stage` writes EXTEND and DECODE as their own traces) plus
     `SGLANG_PROFILE_WITH_STACK=true` for CPU/Python stack evidence.
   - **vllm** — `adapters/vllm.sh` turns `torch_profiler_with_stack` ON (its ordinary
     Profile default is OFF) and bounds the window with
     `ProfilerConfig.max_iterations` = `GEAK_FUSION_MAX_ITERS` (default 16). There is no
     `profile_by_stage`, so that ONE window must contain both phases — it works because a
     saturated continuous-batching server interleaves prefill chunks with decode steps.
     If Phase 1 comes back single-phase, raise `GEAK_FUSION_MAX_ITERS` and confirm the load
     was actually saturated; do not silently accept half the coverage.

     **Keep CUDA graphs ON for this capture; decode structure comes from a separate donor.**
     This capture is the Clean Trace — the only timing source — so it must run the production
     stack. Under graph replay its decode steps carry no per-layer op, so Phase 1 leaves them
     `boundary_unresolved` (`decode_evidence=decode_steps_present_boundaries_unresolved`,
     `decode_requires_boundary_donor=true`). The semantics mapper then
     runs a second, CUDA-graph-off capture as the boundary donor (`semantics_mapper.md`,
     PHASE=complete_table). What that donor needs to know:
     - `cudagraph_mode=NONE` disables graph capture but KEEPS torch.compile, so the kernel
       set stays the production one. Measured on Qwen3.5-2B (gfx942, v0.27.1): 367 vs 375
       decode kernels/step, identical 27-kernel distinct set, **0.989 sequence similarity**,
       and per-layer dispatch anchors on every step (16 x 24) instead of only the 2 eager
       prefill steps.
     - **Do NOT use `--enforce-eager` for this.** It also disables torch.compile, so
       inductor's fusions vanish: 1113 kernels/step, 13 kernels present in only one of the
       two traces, **0.063 similarity**. Those shapes describe a different graph than the
       one being optimized, and are worse than no shapes because they look valid.
     - Use the DOTTED form. Passing `--compilation-config '{...}'` wholesale replaces the
       object and drops the platform-resolved defaults (`custom_ops`, `pass_config`);
       `--compilation-config.cudagraph_mode=NONE` merges.
     - The donor is a SEPARATE server from the measured one — cudagraph off changes
       throughput. It supplies structure and shapes only.
       **Budget for it.** On MiniMax-M3 (60L MoE, TP8, gfx942) decode fell from ~410 to
       ~18 tok/s with graph replay off — ~3.5 s per step. Size the capture workload for the
       WINDOW, not for a throughput measurement: a short OSL and a small prompt count are
       fine, because decode shapes depend on the batch, so only CONC has to match the
       workload you are optimizing.
     - **Place the window with `GEAK_FUSION_DELAY_ITERS`, not with wall-clock warmup.**
       It maps to `ProfilerConfig.delay_iterations`, which counts ENGINE STEPS, so it clears
       the prefill ramp no matter how slow a step is. Set it to at least
       `ceil(CONC*ISL/max_num_batched_tokens)`.
       Both wall-clock attempts failed on MiniMax-M3, in opposite directions: a 25 s warmup
       left all 24 captured iterations inside the ~32-step ramp, and a 180 s warmup outlived
       the background load, so bench_e2e fell back to a fresh profiled bench that armed the
       profiler at ITS start — prefill again. With `GEAK_FUSION_DELAY_ITERS=45` and
       `PROFILE_WARMUP_SEC=0` the window landed on 12 steps, all decode.
       When a window IS mis-placed, Phase 1 says so precisely:
       `decode_evidence: mixed_trace_no_decode_steps_in_window` means the annotation worked
       and only the window was wrong — not the same as `no_phase_annotation_in_trace`, and
       not a reason to change capture mode.
     - **One phase per capture is normal here, and fine.** A delayed window lands entirely
       in decode; an undelayed one entirely in prefill. Phase 1 accepts either
       (`status: pass` with only that phase's tables), so capture the phase you need — or
       capture twice — rather than trying to tune one window to straddle both.
     - **with_stack is expensive at scale.** The same run wrote 8 x ~158 MB of gzipped trace
       (1.2 GB) and the flush pinned every worker at 100% CPU for minutes after the profiler
       self-stopped. Lower `GEAK_FUSION_MAX_ITERS` before anything else if the flush is
       hurting; the layer boundaries need only a few complete steps.
   - Any other backend retains its existing adapter behavior.
4. Select the clean steady production graph trace for benefit/timing evidence.
   Warmup/capture and metadata-only eager traces may supplement semantics, but
   must not replace production timing. For split SGLang traces, compare the
   maximum single-step Decode device-event count on every TP rank and select
   the lowest rank tied at the cross-rank maximum; do not force rank 0. The
   Semantics mapper will automatically adopt that rank's EXTEND sibling.
   With speculative decoding (EAGLE/NEXTN/MTP) the DECODE file carries
   `step[TARGET_VERIFY bs=N]`, not `step[DECODE ...]`: the target's generation
   phase is then named **`verify`** (M = bs x draft tokens) instead of
   `decode`. The trace alone decides this; the server log only enriches
   (`speculative.verify_tokens_per_request`, `speculative.accept_len_mean`)
   and a missing/odd log is a warning, never a failure. Draft-model steps
   (`DRAFT_EXTEND*`, unannotated draft graph replays, and the draft prefill
   that SGLang also labels `step[EXTEND ...]`) are never main-layer evidence.
   Keep the speculative server flags unchanged for the capture; never disable
   speculative decoding to get a plain DECODE trace.
   The sglang capture records 3 steps per stage (`GEAK_FUSION_PROFILE_STEPS`)
   with the client warmup kept out of the profiled load
   (`GEAK_FUSION_PROFILE_WARMUPS=0`): otherwise the window lands on the 8
   warmup requests (Qwen3.5 NEXTN: every verify step bs=8 instead of 64).
   Semantics picks one analysis step per phase (prefill: most tokens;
   decode/verify: largest batch; ties: median duration).
5. Require the completed capture to publish exactly `TRACE_MANIFEST_JSON`
   (normally generated by `bench_e2e.sh` via `scripts/trace_capability.py`) and
   return that manifest. Re-check it for both phases (the capture's own manifest does
   not require them, because a decode-only boundary donor uses the same path):
   ```bash
   python3 "$SKILL_DIR/scripts/trace_capability.py" --trace-dir "$CAPTURE_DIR/profile" \
     --auto-select-rank --require-phases prefill,decode --out "$TRACE_MANIFEST_JSON"
   ```
   If a vllm capture still misses a phase, capture once more with
   `GEAK_FUSION_DELAY_ITERS=0` (the window must start on the prefill ramp) and a larger
   `GEAK_FUSION_MAX_ITERS` (it must reach decode); if it still fails, return the manifest
   with its `missing_required_phases` in `notes` and `phase_evidence_status=unresolved`. Do not return before the foreground capture command
   exits and the manifest has been validated. Its `phase_annotation_count` /
   `phase_annotation_dialects` counters are the objective check on step 2 — this
   is what decides `phase_evidence_status`, so read it rather than assuming:
   - non-zero `legacy_execute` → vllm's native annotation worked (the expected result).
   - non-zero `sglang_step` → sglang, or the vllm fallback overlay.
   - **zero** on a vllm capture → the build does not annotate. Say so in `notes` and recommend
     re-running with `args.vllm_phase_annotate:"true"`; do not report a phase-blind trace as
     `measured_annotation`.
   If capture fails, return an empty manifest and a concrete note; never fabricate one.

Return JSON:
```json
{"round":"fusion_capture","trace_manifest_json":"<absolute path>",
 "trace_dir":"<absolute path>","trace_files":["<rank-sorted raw traces>"],
 "analysis_rank_trace":"<selected production TP Decode trace>",
 "phase_evidence_status":"measured_annotation|unresolved",
 "source":"profile-trace|torch-trace|tracelens-trace","notes":"..."}
```
