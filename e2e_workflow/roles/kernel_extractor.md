# Kernel Extractor

Build a self-contained, immutable correctness and timing task for the profiled operation. Do not
optimize it. Read `SKILL_DIR/knowledge/shape_capture.md` and `sglang_internals.md` for the capture
and overlay APIs. Return the JSON for the requested phase.

Inputs: `EVAL_DIR`, `MODEL_PATH`, `GPU_ID`, `WORKLOAD`, `KERNEL`, `CURRENT_OVERLAY`,
`CURRENT_FLAGS`, `CURRENT_ENV`, `SKILL_DIR`; optionally `PROFILE_WORKLOAD_JSON`, source/launcher
hints, `GEMM_SYNTH`, `DECODE_M_BUCKETS`, `REQUIRE_DECODE_BUCKET`, and `PREFILL_M_NOTE`.
Use the exact workload ISL/OSL/CONC; shorten capture with `REPEATS`/`CAPTURE_MAX`, never smaller shapes.

## Shared requirements

Resolve the live regime before capture or synthesis:
```bash
TASK="$EVAL_DIR/kernels/<short_name>_task"
mkdir -p "$TASK"
python3 "$SKILL_DIR/scripts/parse_regime.py" \
  --server-args "$CURRENT_FLAGS" --model-config "$MODEL_PATH/config.json" \
  --server-script "$EVAL_DIR/launch_baseline.sh" --backend "$BACKEND" \
  --server-log "<current benchmark server.log>" --out "$TASK/regime.json"
```
Merge the result into `meta.json.regime`. The launch script supplies flags omitted from
`CURRENT_FLAGS`, including chunked-prefill size.

- Use the live quantization seam and compact operand types/scales from `h.regime_spec(regime)`.
- Build paged KV with `h.synth_kv_cache(...)`; `h.pack_x(kv_dtype)` follows KV dtype, not compute
  dtype. Use actual quantized scales. `h.regime_dtype("fp8")` selects the device format; explicit
  checkpoint formats take precedence. Do not hardcode fnuz, bf16, or unit scales.
- Match deployment graph/compile behavior through the shared harness. If the live operation cannot
  be reproduced offline, return `editable:false` with the limitation.
- Determine `served_regimes` from the call graph or verified trace, and write it at top level in
  `meta.json`. A split prefill/decode operation gets only its own regime's cases. If the split is
  unresolved, report a task-generation defect; never assume both or neither.
- For a GEMM serving both regimes, include prefill and decode M≈CONC for every (N,K). Include each
  required `DECODE_M_BUCKETS` value and the prefill buckets from `PREFILL_M_NOTE`. No unserved cases.
- Treat TraceLens source/launcher/shape hints as advisory. Verify symbols and shapes against installed
  source plus live capture or the profiler torch trace; record corrections.
- Correctness requires complete live captured operands/layouts for value-dependent operations
  (attention, quantized packed layouts, MoE). If `oracle_complete:false` or capture is empty, do not
  freeze a partial oracle. Value-independent dense GEMM alone may use config-derived synthesis.

## Smoke status contract

`smoke` / `unittest_smoke` is exactly the lowercase string `"pass"` or `"fail"`.
Pass means exit 0 and correctness passed. Put commands, exit code, GPU, case counts and caveats in
`smoke_detail`. For exit 3 / `UT_HARNESS_INCOMPLETE`, regenerate the UT first as described below.

## PHASE=extract

Create:
```
<EVAL_DIR>/kernels/<short_name>_task/
  kernel_src/          # the only optimizer-writable directory
  baseline_overlay/    # frozen CURRENT_OVERLAY, the live timing baseline
  baseline_ref/*.orig  # unimportable source copies for diffing
  cases.py            # call, timing_cases, random_shapes, eager_cases
  reference_io.pt      # immutable captured inputs and golden outputs
  harness_lib.py       # vendored shared harness
  leg_runner.py        # vendored one-leg runner
  overlay_setup.py     # vendored candidate-overlay builder
  unittest.py          # immutable driver
  meta.json            # source, binding, regime, shapes, build, checksums, workload
```
Vendor before using the task helpers:
```bash
TASK="$EVAL_DIR/kernels/<short_name>_task"
mkdir -p "$TASK"
for f in harness_lib.py leg_runner.py overlay_setup.py; do cp "$SKILL_DIR/scripts/$f" "$TASK/"; done
```

### Select and capture the live operation

`KERNEL.device_kernel` is the exact GPU symbol to reach. Choose a bindable callable from
`KERNEL.seam_candidates[]`; `target_callable` is an initial hint and `live_call_seam` is prose.
Check supplied source/launcher hints first, then inspect imports and installed source. Append any
new candidate with its role, depth, matching device kernels and evidence.

- A standalone library call without an editable body belongs to config/tuning: return
  `editable:false`, `target_callable:""`; do not invent a source-rewrite proxy.
- For fused/grouped MoE, select the whole-operation `op_seam`, even if its implementation is native.
  Capture its full dispatch, including routing, GEMM1, activation, GEMM2 and reduction. Never substitute
  standalone dense GEMMs. A rebindable Python dispatcher is editable.
- For other operations, descend past wrappers to the deepest safe `inner_launcher`/`op_seam` reaching
  the symbol. Native/JIT `kernel_entry` objects are source evidence, not monkeypatch targets.

Capture on a copy of the current accepted overlay; multiple PYTHONPATH overlays do not compose
`sitecustomize`. Freeze that same stack in `baseline_overlay`:
   ```bash
   TASK="$EVAL_DIR/kernels/<short_name>_task"; mkdir -p "$TASK"
   # FREEZE the live serving stack as this task's baseline env, then hang the capture hook off a COPY
   # of it. --from is what stacks them: two overlay dirs on PYTHONPATH do NOT compound (only the first
   # sitecustomize is imported), so capturing on a bare hook overlay would silently capture the
   # PRISTINE install instead of the server the accepted kernels actually built.
   python3 "$SKILL_DIR/scripts/overlay_setup.py" add-capture \
     --overlay "$TASK/_capture_overlay" --from "$CURRENT_OVERLAY" \
     --target "<selected module:attr>" --out "$TASK" --max 5 \
     --capture-file "$SKILL_DIR/scripts/capture_shapes.py"
   # Repeat add-marker for every safe candidate, deepest first.
   python3 "$SKILL_DIR/scripts/overlay_setup.py" add-marker \
     --overlay "$TASK/_capture_overlay" --target "<candidate module:attr>" \
     --marker-file "$SKILL_DIR/scripts/seam_trace.py"
   if [ -n "$CURRENT_OVERLAY" ]; then
     [ -d "$CURRENT_OVERLAY" ] || { echo "Missing CURRENT_OVERLAY: $CURRENT_OVERLAY" >&2; exit 2; }
     mkdir -p "$TASK/baseline_overlay"
     cp -r "$CURRENT_OVERLAY"/. "$TASK/baseline_overlay"/
   else
     python3 -c "import sys;sys.path.insert(0,'$SKILL_DIR/scripts');import overlay_setup as o;o._ensure_overlay('$TASK/baseline_overlay')"
   fi

   BACKEND="<backend>" OUT_DIR="$TASK/_capture" GPU="$GPU_ID" MODEL="$MODEL_PATH" \
   ISL=<WORKLOAD.isl> OSL=<WORKLOAD.osl> CONC=<WORKLOAD.conc> REPEATS=0 PROFILE=0 \
   OVERLAY_PYTHONPATH="$TASK/_capture_overlay" EXTRA_SERVER_ARGS="$CURRENT_FLAGS" \
   EXTRA_ENV="$CURRENT_ENV CAPTURE_TARGET=<selected module:attr> CAPTURE_OUT=$TASK CAPTURE_MAX=5 GEAK_SELECTION_TRACE=$TASK/selection_trace.json CAPTURE_BYTE_BUDGET=${CAPTURE_BYTE_BUDGET:-8GiB} CAPTURE_CASE_BYTE_LIMIT=${CAPTURE_CASE_BYTE_LIMIT:-2GiB} CAPTURE_PERSIST_POLICY=${CAPTURE_PERSIST_POLICY:-share_large}" \
     bash "$EVAL_DIR/bench_e2e.sh" 2>&1 | tee "$EVAL_DIR/logs/capture_<short_name>.log"
   python3 "$SKILL_DIR/scripts/kernel_selection.py" \
     --target "<selected module:attr>" --device-kernel "<KERNEL.device_kernel>" \
     --capture-meta "$TASK"/capture.pid-*.rank-*/meta.json \
     --torch-trace "$TASK"/selection_trace.pid-*.rank-*.call-*.json \
     --candidate-target "<candidate module:attr>" \
     --task-dir "$TASK" \
     --out "$TASK/selection_validation.json"
   ```

Return the full `selection_validation.json` object verbatim as `selection_validation` in BOTH phases;
never replace it with a summary. It includes target/device identity, observed/marker/matched-call counts,
all tested candidates and live/deeper-candidate evidence required by the workflow.
Repeat `--candidate-target` for all relevant candidates. Require `selection_validation.ok:true` and
`deepest_verified:true`: every capture PID/rank must pass, including deeper-marker checks across calls.
`kernel_selection.py` promotes the selected oracle/meta and reclaims process-local captures. Before
retrying a failed capture, clean without promotion:
```bash
python3 "$SKILL_DIR/scripts/capture_shapes.py" --cleanup-task-dir "$TASK" --no-promote
```
Use `CAPTURE_BYTE_BUDGET=8GiB`, `CAPTURE_CASE_BYTE_LIMIT=2GiB` and
`CAPTURE_PERSIST_POLICY=share_large` unless supplied otherwise. Honor `capture_manifest.json` limits.
Load large shared oracles lazily with `h.iter_eager_cases_from_oracle` /
`h.check_correct_multi_lazy`, not all on GPU at once. Verify nonempty `reference_io.pt`, `meta.json`
and case coverage. On startup failure inspect `$OUT_DIR/server_start.json` for the actual reason.

### Seed and bind the candidate

Resolve source under the frozen overlay and copy only the defining file, never a package subtree:
```bash
MOD="$(python3 -c 'print("<module:attr>".split(":")[0])')"
SRC="$(PYTHONPATH="$TASK/baseline_overlay:$PYTHONPATH" \
       python3 "$TASK/overlay_setup.py" check --module "$MOD" --path-only)"
mkdir -p "$TASK/kernel_src" "$TASK/baseline_ref"
cp "$SRC" "$TASK/kernel_src/$(basename "$SRC")"
cp "$SRC" "$TASK/baseline_ref/$(basename "$SRC").orig"
```
Declare `meta.candidate_bind`:
```jsonc
// existing module replacement
{"kind":"module", "module":"<dotted module>", "file":"kernel_src/<basename>.py"}
// new implementation rebound to the public callable
{"kind":"rebind", "target":"<module:attr>", "impl_module":"<impl basename>",
 "impl_attr":"<entry fn>", "file":"kernel_src/<impl>.py"}
```
This is also the Integrator's deployment binding. Keep baseline files and oracle immutable.

### Write cases and the immutable unittest

Both legs run the same `cases.py` and `leg_runner.py`; only PYTHONPATH differs:
`baseline_overlay` versus `_cand_overlay`. The latter is rebuilt from `candidate_bind` by
`h.measure_legs`. The baseline implementation must resolve outside the task's editable source.
`baseline_ref/*.orig` is never a timing source. There is no separate `baseline_callable` in this phase.

`cases.py` imports no candidate source or other task module; use the supplied `h` and resolve only
`meta.target_callable` through importlib. Define:
```python
def call(args):             # preserve the entire return structure and fresh-output contract
    return _resolve(META["target_callable"])(**args)
def timing_cases(h, meta):  # one {sig, regime, m, args} per meta.workload.cases entry
    ...
def random_shapes(h, meta): # {sig, make_inputs(rng)}; fresh values at FIXED online dimensions
    ...
def eager_cases(h, meta):   # lazy {args, ref, sig} from reference_io.pt
    ...
```
`sig` must be stable across processes. Without workload metadata, time captured cases unweighted.
Reconstruct recorded dtype/layout/device/contiguity and restore pre-call buffers for in-place inputs.
Compare every output component, including tuple/dict members. Use dtype-appropriate tolerances
(bf16/fp16 rtol=atol=2e-2; fp32 tighter; document quantized tolerances). Do not expose reference outputs
to the candidate or allow persistent output buffers that overwrite earlier results.

Load the vendored harness before removing the task directory from `sys.path` to avoid shadowing
stdlib `unittest` when importing torch:
```python
import importlib.util, os
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("harness_lib", os.path.join(HERE, "harness_lib.py"))
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)
```
Use the shared correctness entrypoint, not hand-written allclose/timing loops:
```python
base_out = h.baseline_random_outputs(TASK, meta, draws=meta.get("random_draws", 3))
try:
    ok, report = h.run_correctness(
        meta["regime"], eager_cases=eager_cases, current_call=cases.call,
        random_shapes=cases.random_shapes(h, meta), tol=tol, baseline_outputs=base_out,
        draws=meta.get("random_draws", 3), replay=replay_bundle)
except h.HarnessIncompleteError:
    sys.exit(3)  # harness already prints UT_HARNESS_INCOMPLETE
```
Any eager/random mismatch fails correctness. Random draws vary values, not dimensions, and their
speedups are secondary only. Pass `baseline_outputs=`, never a candidate-process baseline closure.

For graph deployment (`h.deployment_graph_mode(meta['regime'])`), supply a replay bundle with at least
two boundary shapes: `{fill, run, read_out, cases, capture_idx}` per the harness docstring. Capture
the largest case and fill edge/smaller cases into the same static buffers with server padding.
Attention uses `h.boundary_decode_seq_lens(meta['geometry'], ISL+OSL)` and
`h.shuffled_block_table(...)`; GEMM uses family×M buckets. Use captured/baseline references. Also
check `meta.call_sequence` through `h.check_correct_sequence` for stale state across real interleaves.
`regime.cuda_graph`, not observed `meta.graph_replayed`, decides whether replay is required.
Compile parity is included by `h.run_correctness`: numeric drift fails; an isolated compile build
error is a surfaced note, not a regenerate signal. The end-to-end gate judges deployment compile behavior.

Timing is `per_case = h.measure_legs(TASK, meta)`. It checks distinct source resolution and uses
interleaved baseline/candidate subprocess pairs, the same graph/compile context, and median speedup
`baseline_ms/optimized_ms`. Keep adaptive repetition defaults; report `speedup_spread` for noisy
buckets. `h.time_op` uses device events (`inner=1`); do not replace it with host dispatch timing or
award an isolated graph trick against an eager baseline. Print baseline/optimized ms and speedup
per case. Surface `._geak_compile_error` in notes.

### Workload attribution and metric

Config M buckets define coverage; live capture augments/corrects them. Profile percentages select
heads and provide attribution priors, not the final within-operation weights. Tag every captured
case with `regime` (or `""` if none); GEMM/MoE bucket lists imply it. Binding and regime must agree.
When `PROFILE_WORKLOAD_JSON` is present, run:
```bash
python3 "$SKILL_DIR/scripts/attribute_weights.py" \
  --meta "$TASK/meta.json" --profile-weights "$PROFILE_WORKLOAD_JSON" \
  --name-match "<kernel base symbol>" --isl "$ISL" --osl "$OSL" --conc "$CONC" \
  --min-regime-share 0.3 --served-regimes "<verified regimes>" --out "$TASK/workload.json"
```
Merge into `meta.workload` and return `workload_path`. Supply optional `--ttft-ms`/`--tpot-ms` for
sanity checks only, and `--prefill-chunk` only to override the parsed launch budget. The tool can derive
served regimes from annotated traces, but set them explicitly for split kernels or incomplete traces.
The gate must agree in top-level `meta.served_regimes`, the oracle cases, and workload/analytic counts.
Read attribution notes; never hand-write weights. Without a signal, omit workload merge/path.

Compute the primary metric with `h.serving_weighted_speedup(per_case, meta)`, not stored profile
weights. It uses measured baseline latency × analytic calls, removes unserved regimes, and assigns
a regime's calls to its largest-M bucket (smaller transients receive 1 call):
- prefill calls = `CONC × ceil(ISL/chunk)`;
- decode calls = `OSL`, with CONC already represented in the shape;
- weighted speedup = total baseline lifecycle time / total candidate lifecycle time.
If `weighted is None` because all buckets are identity/untrusted, repair and remeasure instead of
reporting 1.0. Print `GEAK_WEIGHTED_SPEEDUP`, geomean and per-case numbers. TTFT/TPOT are whole-server
cross-checks, never per-kernel weights. For causal attention, measure accumulated-context prefill
chunks individually, or a full-ISL pass with `calls_prefill=CONC`; equal chunk costs are invalid.

### Finalize and smoke

Record source path, candidate backends, regime, build flag/command, `random_draws` (default 3), and
`reference_io_sha256`. Smoke the copied baseline candidate:
```bash
cd "$TASK" && bash "$SKILL_DIR/../kernel_workflow/scripts/gpu_lock.sh" "$GPU_ID" python3 unittest.py
```
Require correctness pass, speedup≈1, and distinct leg resolution printed by `h.assert_legs_differ`.
The paths differ even though the seed bytes match. Never fall back to candidate source as baseline.
Exit codes: 0 pass, 1 correctness/binding failure, 2 environment failure, 3 incomplete harness.
Exit 3 or `UT_HARNESS_INCOMPLETE` requires regenerating the missing replay bundle and retrying up to
three times. Only then return failure and start `smoke_detail` with `harness_incomplete_unrecoverable`.
Other unextractable cases return `editable:false` with the actual reason.

Return JSON (also include `selection_validation` from the verifier verbatim):
```json
{
  "short_name": "<short_name>",
  "editable": true,
  "task_dir": "<EVAL_DIR>/kernels/<short_name>_task",
  "source_path_in_sglang": "<abs path under site-packages>",
  "device_kernel": "<KERNEL.device_kernel verbatim>",
  "target_callable": "<module:attr>",
  "seam_candidates": [
    {"target_callable": "<module:attr>",
     "role": "outer_wrapper|dispatcher|op_seam|inner_launcher|kernel_entry",
     "device_kernels": ["<KERNEL.device_kernel>"], "depth": 0, "evidence": "source/runtime evidence"}
  ],
  "candidate_bind": {"kind": "module|rebind", "module": "<dotted>", "file": "kernel_src/<f>.py"},
  "baseline_overlay": "<task_dir>/baseline_overlay",
  "baseline_frozen": true,
  "num_cases": 0,
  "regimes_captured": ["prefill","decode"],
  "candidate_backends": ["triton","hip","ck"],
  "build": false,
  "unittest_smoke": "pass",
  "smoke_detail": "command run, exit code, cases passed — free text goes HERE, never in unittest_smoke",
  "reference_io_sha256": "...",
  "workload_path": "<task_dir>/workload.json",
  "notes": "granularity choice, hidden state captured, anything unusual"
}
```

## PHASE=extract_op

Create an operation task for a head GEMM/attention backend comparison; library `edit=N` is allowed.
For `op_kind=moe`, use the complete PHASE=extract contract above, return `op_kind:"moe"`, and set
`math_contract:"grouped per-expert GEMM + routing"`, `synthesized:false`. Capture real activations,
`topk_ids` and `topk_weights`; uniform/random routing is not an oracle. If a regime cannot be captured,
report that missing coverage. Keep the full fused dispatcher as the baseline and candidate boundary.

For GEMM/attention, emit `meta.json`, immutable `unittest.py`, vendored `harness_lib.py`, and
`reference_io.pt` (required for attention; optional for value-independent dense `GEMM_SYNTH`). Include
`op_kind`, `dtype`, `math_contract`, `regime`, `pct_gpu_time`, shapes/tensor specs and live bindings.

- Dense GEMM: determine `a_shape`, `b_shape`, `transpose_b`, bias and activation epilogue from source/
  trace. With `GEMM_SYNTH=true`, synthesize in-regime inputs; `op_bench.py` computes the default-backend
  oracle at load time. If activation distribution matters, capture `{A,B,bias,output}` instead.
- Quantized operands: candidate inputs contain only the compact live operands, scales, zero-points
  and shape/group metadata. Keep dequantized weights and golden output private to the harness. The
  baseline must perform the real compact-operand reconstruction/compute, not consume a precomputed
  dequantized weight. The signature must match the live operation; document `rebind_seam_note`.
- Attention: capture the active backend's forward, including q/k/v, KV cache, metadata and outputs for
  the served regimes. Record `captured_keys` and the paged attention math contract. Backend swaps are
  server flags; the task supports correctness and authored replacements.

Use `h.time_op` with deployment graph/compile parity and `h.check_correct_multi` (fresh outputs), plus
`h.check_random_vs_baseline(baseline_call, current_call, shapes, tol,
 draws=meta.get("random_draws", 3), graph=h.deployment_graph_mode(meta["regime"]))`.
Here `baseline_call` resolves `meta.baseline_callable` on the live stack outside editable task source.
Apply the shared regime, served-case coverage and workload metric rules above. Fold all correctness
results into PASS/FAIL; random speedups remain secondary. Save the oracle checksum when present.
Smoke GEMM with `op_bench.py --task <dir> --backends hipblaslt --repeats 5`; smoke attention against
its captured live path. Report unsupported or failed smoke honestly.

Resolve an authored candidate's live rebind seam by installed backend/dtype, then prove engagement
with a call counter on live forwards and the normal device-kernel/deepest-selection validation.
Zero hits requires a different seam or skipping authoring. Useful search seeds, not authorities:
- vLLM ROCm unquantized linear: `vllm.model_executor.layers.utils:rocm_unquantized_gemm_impl`;
- vLLM CUDA unquantized linear: `torch.nn.functional:linear`;
- ATOM fp8 blockscale: `atom.model_ops.linear:gemm_a8w8_blockscale_preshuffle_impl`;
- ATOM fused QKV-A: `atom.models.deepseek_v2:_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp8`.
Do not assume an aiter inner seam is active on every architecture. Inspect quantized/attention/MoE
paths directly. Empty `target_callable` is allowed only when no Python seam exists; an applicable
environment/tuning winner can still deploy. Preserve exact device-kernel identity and seam evidence.

Return JSON (also include `selection_validation` from the verifier verbatim):
```json
{
  "short_name": "<short_name>",
  "op_kind": "gemm|attn",
  "editable": true,
  "task_dir": "<EVAL_DIR>/kernels/<short_name>_task",
  "shapes": {"a_shape": [], "b_shape": [], "transpose_b": true, "bias": false},
  "dtype": "bf16",
  "synthesized": true,
  "regimes_captured": ["prefill"],
  "candidate_backends": ["aiter","hipblaslt","triton","ck"],
  "reference_io_sha256": "<or '' if synthesized>",
  "device_kernel": "<KERNEL.device_kernel verbatim>",
  "target_callable": "<module:attr rebind seam if one exists, else ''>",
  "seam_candidates": [
    {"target_callable": "<module:attr>",
     "role": "outer_wrapper|dispatcher|op_seam|inner_launcher|kernel_entry",
     "device_kernels": ["<KERNEL.device_kernel>"], "depth": 0, "evidence": "source/runtime evidence"}
  ],
  "baseline_callable": "<module:attr of the live default backend, resolved OUTSIDE the task dir>",
  "smoke": "pass",
  "smoke_detail": "command run, exit code, cases passed — free text goes HERE, never in smoke",
  "notes": "transpose/bias inference, regime, whether oracle was synthesized vs captured"
}
```
