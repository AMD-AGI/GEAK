# Oracle Freezer

Turn KERNEL_PATH into an isolated immutable task for comparison across backends.
Do not optimize. The input kernel's frozen baseline_src/ supplies both correctness
and the speedup denominator on every draw. Never substitute a naive reference.
Use live parity with regenerated inputs, not saved golden tensors; emit no
reference_io.pt and return reference_io_sha256:"". If the original or its inputs
cannot be run/reconstructed, return smoke:"fail", baseline_frozen:false and why.

Inputs: KERNEL_PATH, EXP_ROOT, KERNEL_NAME_HINT, GPU_ID, OP_SPEC,
WORKLOAD_SPEC_PATH, SKILL_DIR, HARNESS_LIB, GPU_LOCK and optional EXPECTED_GFX,
EXPECTED_TARGET, EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT.

Establish hardware identity before choosing policy:
```bash
python3 "$SKILL_DIR/../scripts/gpu_identity.py"
```
Return gfx, target, marketing_name and physical_cu_count as device_gfx,
device_target, device_name and physical_cu_count. Stop for a nonempty EXPECTED_GFX
or EXPECTED_DEVICE_NAME mismatch, an EXPECTED_TARGET=r9700 mismatch, or disagreement
with positive EXPECTED_PHYSICAL_CU_COUNT. Do not infer a product from gfx alone.

Create the task and logs:
```bash
TS=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="$EXP_ROOT/bakeoff_${KERNEL_NAME_HINT}_${TS}"
TASK="$EVAL_DIR/task"
mkdir -p "$TASK" "$EVAL_DIR/logs"
```
Emit kernel_src/ (editable candidate), baseline_src/ (frozen input), vendored
harness_lib.py, unittest.py and meta.json. Keep the task self-contained, using
candidate entry metadata rather than a hardcoded candidate backend.

Find the shipped driver/config/task_runner/tests first. Read its callable, actual
case file, input construction and launch constants. If absent, synthesize driver
plumbing using benchmark_engineer guidance, never its naive correctness fallback.
Record live_backend from actual source. Copy original source into kernel_src/ and
baseline_src/; bind meta.baseline_callable to baseline_src/, never kernel_src/.

Case and state contract

Record concrete dimensions, dtypes/layouts, scalar/quantization parameters, stable
sig and seed. Prefer the shipped driver's actual performance cases; then caller
workload cases; then OP_SPEC; synthesize small/medium/large only as a last resort.
Resolve symbolic dimensions before creating tensors. Preserve baseline launch
constants and input distributions, including value-dependent routing/masks.
Map workload weights onto the shipped timing cases by dimensions, dtypes and
required parameters, not names alone. Do not replace the shipped timing set or
silently renormalize unmatched workload entries. Report incompatible coverage;
if the required workload cannot be represented faithfully, fail the freeze.

Keep the shipped performance set exactly for its timing score. Preserve and
execute any additional correctness-only cases separately (for example in
meta.correctness_cases); do not dilute the timing score with them or drop their
checks. Each required case/sample and numerical/state check must run. Record actual
coverage and any unresolved gap; a note alone cannot replace correctness coverage.
A case entry may be {"sig":"c2_M2048","M":2048,"N":4096,"K":3072,"seed":42,
"regime":"prefill"}; retain all parameters required by the real callable.

Use shared build_args(case) logic for parity and timing, with each tensor's seed
independent of draw order. Both implementations and timing legs receive equivalent
inputs and state resets. Seeds establish within-run reproducibility, not cross-node
bit identity. Match the driver's timing distribution; harder parity distributions
are allowed when recorded separately. Preserve nested outputs and mutation/state
contracts, not just one returned tensor.

Harness and integrity

Always vendor the supplied current helper, even when reusing an earlier baseline:
```bash
cp "$HARNESS_LIB" "$TASK/harness_lib.py"
```
Verify its SHA256 matches HARNESS_LIB and retain both hashes. Record
baseline_src_sha256 over sorted relative paths/content, excluding __pycache__,
build, *.pyc, *.so and *.o; also record harness_lib_sha256 and unittest_sha256.
Write unittest.py before hashing it into meta.json. Recompute all three before
each run and fail on mismatch. Mark baseline_src/, harness_lib.py, unittest.py and
meta.json read-only after construction. No source outside the declared task closure
may silently replace the frozen baseline or candidate.

Use vendored harness_lib (h) for correctness and timing:
- h.check_random_vs_baseline(baseline_call, current_call, shapes, tol,
  draws=meta.get("random_draws", 5)); shapes contains sig and make_inputs(rng).
  Use fresh values at fixed required dimensions. Any failing draw fails validation.
- h.assert_independent_outputs(current_call, args_a, args_b) is also mandatory.
  Use the smallest case at the SAME dimensions with DIFFERENT values; different
  shapes can hide a shape-keyed static output buffer. Fold this into all_ok.
- Do not load golden tensors, fingerprint regenerated operands against saved
  values, or add an operand-drift gate when no captured input exists.
- d = h.time_op(call, warmup, repeats, detail=True). None means failure, never zero.
  Preserve d["ms"], d.get("primed"), d.get("host_ms") and cache_condition.mode.
  Time the complete required operation with correct resets; baseline_call always
  resolves to the frozen original. speedup = baseline_ms/current_ms.

For valid workload weights mapped onto the declared performance cases, emit
GEAK_WEIGHTED_SPEEDUP = sum(weight)/sum(weight/speedup) as primary, with geomean
secondary. Otherwise use the declared performance cases unweighted when that is
the requested scope. Never drop required correctness cases or workload coverage.

Retain timing qualifications. primed=True means device work dominates, not zero
overhead; amortize a significant window overhead with inner. primed=False means
host_bound. Absent primed means timer_unprimed, not false. Print measured times
with those labels; do not claim a clean device-only ratio when either leg lacks
priming. Emit this machine-readable receipt for both legs of every timed case:
```text
GEAK_TIMING_RECEIPT: {"all_primed": <bool>, "timer_unprimed": <bool>, "cache_mode": "<mode>",
 "cases": {"<case>": {"baseline": {"primed": ..., "host_ms": ...},
                      "current": {"primed": ..., "host_ms": ...}}}}
```
all_primed requires True for every leg; timer_unprimed records any missing primed.
Preserve the actual cache mode: current harness uses read-evict. Missing
cache_condition means unknown_write_evict, not no cache preparation. Disagreement
between legs is a fault, not an average. Preserve historical receipt labels when
reading old evidence; they do not change the supplied current harness.

meta.json includes op_kind, dtype, cases, correctness_cases when needed, geometry,
regime, entry callable, baseline_callable, baseline_frozen:true, tol, build/build
command, candidate_backends, random_draws (default 5), integrity hashes and optional
workload. Include the input backend; propose other backends only with actual support.
Keep baseline/candidate imports separate and source/build dependencies reproducible.

Smoke test current==baseline through the supplied GPU lock:
```bash
set -o pipefail
cd "$TASK" && bash "$GPU_LOCK" "$GPU_ID" python3 unittest.py 2>&1 | tee "$EVAL_DIR/logs/freeze_smoke.log"
```
Check the test exit status as well as output. Require runnable frozen baseline,
all correctness/state checks, matched timing near 1x, a complete receipt, and
current HARNESS_LIB hash. Compare every shipped performance case by dimensions,
not just sig; altered/missing cases fail. Require separate execution of all
correctness-only cases. Record coverage, duplicate shapes and timing/parity input
distributions in notes. No tensor dumps should be present. Report failures honestly;
smoke:"pass" permits downstream comparison and must not hide gaps.

Return JSON:

```json
{
  "eval_dir": "<abs run dir under EXP_ROOT>",
  "op_kind": "gemm|attn|elementwise|moe|other",
  "task_dir": "<abs op task dir>",
  "live_backend": "hip|triton|flydsl|ck|other",
  "candidate_backends": ["hip","triton","flydsl"],
  "baseline_frozen": true,
  "baseline_callable": "module:attr of the frozen input kernel",
  "device_gfx": "gfx950",
  "device_target": "r9700|mi350x|mi355x|unknown",
  "device_name": "<exact Marketing Name>",
  "physical_cu_count": 256,
  "reference_io_sha256": "",
  "op_spec": { "op_kind": "...", "shapes": {}, "dtype": "bf16", "regime": "both" },
  "workload_path": "<task_dir>/workload.json or ''",
  "smoke": "pass|fail",
  "notes": "driver source (which case file it resolved to), step-5 coverage verdict, timing vs parity value distributions, regime, any symbolic dims resolved, anything unusual"
}
```
