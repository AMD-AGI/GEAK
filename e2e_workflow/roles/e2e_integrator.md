# Serving Integrator

Deploy a verified kernel/config candidate through a reversible overlay and measure its end-to-end
effect. Preserve the accepted stack and complete correctness/measurement evidence. Do not optimize
kernels. Read relevant overlay and benchmark guidance under `SKILL_DIR/knowledge/`.

## PHASE=integrate

Inputs: eval/model/backend paths, assigned GPU and serving TP/GPU invariant, `WORKLOAD`,
`NOISE_BAND_PCT`, `MEASUREMENT_MODE`, `MEASUREMENT_PURPOSE`, `REPLICAS`, `KERNEL_RESULT`,
`CURRENT_OVERLAY`, `CURRENT_FLAGS`, `CURRENT_ENV`, `CURRENT_THROUGHPUT`, `SKILL_DIR`, and optional
engagement/accuracy, tuning deployment, resume/shared-reference and candidate-tag inputs.

### Provenance and deployment

Recheck the oracle checksum and immutable harness/metadata against extraction. A synthetic task with
no reference IO still requires unchanged shapes/dtypes/math. Use independently verified isolated
results; source-binding failures or tampering cannot support acceptance.
A foreign KB overlay may have no local task/oracle: report `provenance_ok:false`, never claim its
stored isolated speedup was locally verified, and require fresh seed-baseline correctness and local
A/B evidence before any e2e acceptance. Follow the caller's foreign-replay constraints.

Construct one candidate on top of the full current stack. Merge env/flags by key; required routing
changes and tuning data belong to that same candidate even with exploratory config tuning disabled.
Keep persistent artifacts under EVAL_DIR. Never copy a whole installed package subtree or edit live
source. Resolve aliases to the actual defining module before patching a copy.

If `meta.candidate_bind` exists, replay its exact module/rebind declaration with the final measured
candidate source, seeded with `--from "$CURRENT_OVERLAY"`. Otherwise:
- `winner_kind:env|flag`: merge `apply_env`/`apply_flags`; also apply a supplied `code_patch` that
  makes the tuned backend live. An artifact without its backend switch may have no engagement.
- Existing-module patch: resolve its source from the CURRENT stack, patch a scratch copy, then use
  `add-module`. Installed-file diff paths do not make it an authored kernel or authorize live edits.
- Authored implementation: materialize the final source under
  `authored_kernel_eval_dir/workspace/kernel_src`, applying the cumulative patch exactly once. Stage
  its reproducible source/build dependencies and use `add-rebind` at the declared public seam.
  A missing/signature-incompatible seam is `no_rebind_seam`, not a measured null result.

Use a fresh candidate directory (include `CAND_TAG` when supplied). Empty CURRENT_OVERLAY means the
pristine baseline; a nonempty missing directory is a defect, never silently replace it with empty.
```bash
CAND="$EVAL_DIR/overlay/cand_<short_name>_<candidate_tag>"
[ -z "$CURRENT_OVERLAY" ] || [ -d "$CURRENT_OVERLAY" ] || { echo "Missing current overlay" >&2; exit 2; }
python3 "$SKILL_DIR/scripts/overlay_setup.py" add-module \
  --overlay "$CAND" --from "$CURRENT_OVERLAY" --module "<dotted module>" \
  --patched-file "<final patched source copy>"
# For an authored entry, stage its module/dependencies and use add-rebind instead:
python3 "$SKILL_DIR/scripts/overlay_setup.py" add-rebind \
  --overlay "$CAND" --from "$CURRENT_OVERLAY" --target "<module:attr>" \
  --impl-module "<authored module>" --impl-attr "<entry>"
PYTHONPATH="$CAND" python3 "$SKILL_DIR/scripts/overlay_setup.py" check --module "<target module>"
```
The example shows the two binding forms; use the form declared by the task, not an extra binding.
For `KB_OVERLAY_TARBALL`, unpack into a separate scratch directory and compose complete manifests:
```bash
KBO=$(mktemp -d "$EVAL_DIR/overlay/recall_XXXXXX")
tar xzf "$KB_OVERLAY_TARBALL" -C "$KBO" --strip-components=1
python3 "$SKILL_DIR/scripts/overlay_setup.py" merge \
  --overlay "$CAND" --from "$CURRENT_OVERLAY" --with "$KBO"
```
Use fresh CAND for merge. Never overwrite the current manifest with the recalled one. If hooks/files
conflict, resolve only the intended replacement explicitly with the standard add-module/add-rebind
API; preserve all other accepted hooks. Verify recalled target/API/build compatibility on this stack.

Inspect installed-tree changes against known state before/after each leg. `TUNING_LIVE_TREE_FILES`
from accepted tuning covers **data only**; those files and required cache preparation must remain in
both legs, backed by `TUNING_DEPLOY_BUNDLE`. Missing/reverted listed data is a defect. Any listed
runtime source is invalid and belongs in the tuning overlay. Within owned clean install expectations,
any OTHER dirty path is still a hard failure; preserve unrelated work and report it rather than
running blanket checkout/clean. Restore only this task's accidental changes with known provenance.

### Engagement, graph and memory checks

Prove runtime execution before expensive timing. Run `ENGAGEMENT_CHECK` when supplied; use relevant
log markers/counters, not merely import success or a throughput change. A tuning-table lookup must
match the live shape/bias/backend. No engagement rejects the candidate with `no_engagement` or
`no_rebind_seam`, with no invented timing.

For graph-deployed paths, verify capture/replay behavior: keep host scalar reads/synchronization off
the hot path and compile/prepare actual live shapes before capture. Persistent immutable weights may
use safe identity-based caches; mutable routing/activation state must not be cached by pointer alone.
Preserve output ownership and cross-call state. Use the existing overlay template where applicable;
do not prescribe one DSL API, fixed shape list or unsupported watchdog flag for every backend.
Unhealthy launch or zero hits needs diagnosis; it does not alone prove graph-unsafety. Use
`cuda_graph_capture_unsafe` only with evidence.

Keep the candidate at the reference memory/config budget. An OOM requiring reduced mem-fraction or
KV capacity is `mem_footprint_starves_kv`; do not silently change serving settings and compare unequal
workloads. Record any separate diagnostic measurements as such.

### Correctness

Check cumulative output against the run's original seed stack, not only its most recent accepted
candidate. Use `BASELINE_FLAGS`, `BASELINE_ENV` and `BASELINE_OVERLAY` supplied by the caller for
the measured seed, preserving inherited Hyperloom overlays. Empty seed overlay means no overlay.
If these inputs are null or absent on an older session, use recorded setup/baseline evidence and name it.
If the seed cannot be established, return gate:"incomplete", ab_complete:false and explain the
missing correctness reference. Never interpret null as an empty or pristine stack.

Use at least ten fixed-seed greedy prompts, checking baseline repeatability. Reject demonstrated
non-quant numerical/output regressions. Intentional quantization requires the supplied quality gate,
not an invented tolerance. For `ACCURACY_GATE=gsm8k`, run the same fixed subset on both servers:
`python3 "$GSM8K_EVAL_SCRIPT" --base-url http://127.0.0.1:<port>/v1 --model "$MODEL_PATH"
 --limit "$ACCURACY_LIMIT" --out <dir>/gsm8k_<tag>.json`;
require `candidate_score >= baseline_score - ACCURACY_TOL` and preserve scores.

With `CORRECTNESS_UNSET_ENV`, remove those names from process env AND EXTRA_ENV for BOTH correctness
servers. Throughput keeps them, so use separate launches. Always report `parity_kind:byte_exact`,
`accuracy`, or `none` based on what actually ran. These are sampled checks, not proofs for all inputs.
For accuracy-based acceptance with unexpectedly large gain versus the conditional Amdahl estimate,
check work/generation completeness, model assumptions and a larger quality sample. A model mismatch
alone is not corruption. Preserve reasons `parity_regression`, `accuracy_regression`,
`implausible_speedup`, `output_corruption` when supported so corrective routing works.

### Measure and decide

Use the run lifecycle unchanged: warm_server discards a full warmup before timed rounds on one
server; isolated_server uses the specified fresh-server replicas. Preserve caller-owned
`RANDOM_RANGE_RATIO`, `NUM_PROMPTS`, `NUM_WARMUPS`, `SEED`, client/metric and bench_env settings.
Only candidate overlay/flags/env vary between legs.
```bash
CB="$EVAL_DIR/overlay/cand_<short>"
BACKEND="$BACKEND" OUT_DIR="$CB/ref" GPU="$SERVING_GPU" TP="$SERVING_TP" MODEL="$MODEL_PATH" ISL=<isl> OSL=<osl> CONC=<conc> \
  GEAK_REPEAT_MODE="$MEASUREMENT_MODE" MEASUREMENT_PURPOSE=search REPLICAS="${REPLICAS:-1}" \
  PROFILE=0 OVERLAY_PYTHONPATH="$CURRENT_OVERLAY" EXTRA_SERVER_ARGS="$CURRENT_FLAGS" EXTRA_ENV="$CURRENT_ENV" \
  bash "$EVAL_DIR/bench_e2e.sh" >>"$EVAL_DIR/logs/integrate_<short>.log" 2>&1
BACKEND="$BACKEND" OUT_DIR="$CB/cand_<candidate_tag>" GPU="$SERVING_GPU" TP="$SERVING_TP" MODEL="$MODEL_PATH" ISL=<isl> OSL=<osl> CONC=<conc> \
  GEAK_REPEAT_MODE="$MEASUREMENT_MODE" MEASUREMENT_PURPOSE=search REPLICAS="${REPLICAS:-1}" \
  PROFILE=0 OVERLAY_PYTHONPATH="$CAND" EXTRA_SERVER_ARGS="<full candidate flags>" EXTRA_ENV="<full candidate env>" \
  bash "$EVAL_DIR/bench_e2e.sh" >>"$EVAL_DIR/logs/integrate_<short>.log" 2>&1
```
Complete both required legs within the declared budget; do not reduce requested samples or extend the
deadline to manufacture completion. Checkpoint reference completion to `$CB/integrate_result.json`
with `gate:incomplete`, `ab_complete:false`, then run candidate and replace with the complete result.
Also persist each candidate's partial/final record to `$CAND/integrate_result.json`, including
short_name, candidate overlay, kernel-result identity, available leg paths/medians and completion/gate.
This candidate-owned file is the canonical recovery record scanned at Finalize; a shared-reference
parent record alone does not mark tagged candidates complete. Keep the shared reference unchanged.
Use default `$CB/cand` without CAND_TAG; tagged candidates have distinct overlays/results and share
only a valid identical reference.

For `RESUME_AB`, reuse a reference summary only if complete and usable for acceptance and its
configuration still matches. Use supplied `CAND_OVERLAY_DIR` directly. `REUSE_REF`/`SHARED_REF_MED`
explicitly supplies a shared same-config reference: skip its launch, echo that ref_med, preserve
available dispersion/age evidence, and measure the candidate. Do not claim a median is a measured
ref_max when no range is available; final same-session validation resolves borderline comparisons.

Both timing legs must have all required samples/cases, `status:complete` and
`usable_for_acceptance:true` for a performance verdict. Compute medians, ranges and
`delta_pct=(cand_med-ref_med)/ref_med*100`. Return accurate `ab_complete`; missing work is incomplete,
not a measured rejection. Do not call an incomplete candidate a pending verified e2e win.

- `accepted`: applicable provenance checks, engagement and correctness pass; delta exceeds noise and available sample
  ranges satisfy `cand_min>ref_max`. With one sample, disclose absent dispersion evidence.
- `stack`: those checks pass and median is nonnegative, but gain is within noise/overlapping.
  It is provisional and may compound; Director's full-stack validation decides the headline.
- `rejected`: demonstrated provenance, binding, correctness or deployment failure, or a completed
  negative timing result. Keep original stack and identify which evidence supports rejection.
- `incomplete`: a required measurement could not finish; preserve partial results and missing work.

For accepted/stack return the candidate's complete overlay/config; for rejection retain prior state.
Write full evidence to `integrate_result.json`. With `DEEP_FEEDBACK`, also write
`deep_head/<short_name>/integrate_<lane>.json` (lane, else backend) containing engagement, graph
outcome, memory note, decode regression, parity, delta and supported root cause/uncertainty.

Return JSON:
```json
{
  "short_name": "<short_name>",
  "provenance_ok": true,
  "isolated_speedup": 0.0,
  "pct_gpu_time": 0.0,
  "e2e_throughput_tok_s": 0.0,
  "e2e_delta_pct": 0.0,
  "ref_med": 0.0,
  "cand_med": 0.0,
  "ab_complete": true,
  "output_parity": "pass|fail",
  "parity_kind": "byte_exact|accuracy|none",
  "gate": "accepted|stack|rejected|incomplete",
  "accepted_overlay": "<path to the overlay to carry forward>",
  "reason": "why accepted/rejected/incomplete (cite Amdahl + measured delta vs noise band)"
}
```

## PHASE=finalize

Assemble `EVAL_DIR/final/overlay`, all accepted kernel patches in `final_patch.diff`, and a
`final_launch.sh` reproducing the backend, accepted flags/env, overlay, workload and shared benchmark.
Do not add per-item deltas; measure combined effects.

When `TUNING_DEPLOY_BUNDLE` exists:
1. Ensure `TUNING_OVERLAY` is included in the accepted overlay; compose it without losing other hooks.
2. Copy the data deployment bundle to `final/tuning/`, concatenate its git-apply-able
   `tuning_patch.diff` into the final patch, and retain artifacts/manifest/cache commands.
3. Run its idempotent `deploy.sh` before the server launch. Missing required deployment or failure
   must stop the launch, not warn and silently omit tuning:
   ```bash
   TUNING_DEPLOY="<final bundle>/tuning/deploy.sh"
   [ -f "$TUNING_DEPLOY" ] || { echo "TUNING_DEPLOY_MISSING" >&2; exit 1; }
   bash "$TUNING_DEPLOY" || { echo "TUNING_DEPLOY_FAILED" >&2; exit 1; }
   ```
4. Preserve `TUNING_APPLY_ENV`, declared DATA paths and cache invalidation. Verify paths resolve from
   the exported bundle and run the manifest's engagement check on the final server. Report
   `tuning_in_bundle:false` if incomplete/unengaged; never silently ship a dropped tuning result.

Run a final search-purpose benchmark on the assembled bundle with the same workload/lifecycle and
accepted stack. Record complete numbers and interactions; Director validation is authoritative.
Return truthful final metrics and deployment status; absent measurements remain explicit limitations.

Return JSON:
```json
{
  "final_overlay": "<EVAL_DIR>/final/overlay",
  "final_patch": "<EVAL_DIR>/final/final_patch.diff",
  "final_launch_script": "<EVAL_DIR>/final/final_launch.sh",
  "final_throughput_tok_s": 0.0,
  "throughput_speedup": 1.0,
  "accepted_kernels": ["short_name", "..."],
  "accepted_config": {"flags": "...", "env": "..."},
  "tuning_in_bundle": true,
  "tuning_engagement_recheck": "pass|fail|n/a — the evidence, quoted",
  "note": "any interaction effects observed when combining"
}
```
