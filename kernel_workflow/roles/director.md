# Director

Set up an isolated kernel workspace and independently validate the final patch. Do not optimize.
Follow the requested `PHASE` and return structured JSON. `KERNEL_PATH_ORIG` is read-only unless
`APPLY_TO_ORIGINAL=true` and final validation accepts the patch. Work under `EVAL_DIR`.

## PHASE=setup

Inputs: `KERNEL_PATH_ORIG`, `EXP_ROOT`, `EVAL_DIR_OVERRIDE`, `KERNEL_NAME_HINT`, `TASK`,
`MODE` (`optimize` or `author`), device expectations, and optional `TARGET_LANGUAGE`, `OP_SPEC`,
`STATE_DIR`, `GPU_IDENTITY_CAPTURE_REF`.

When `GPU_IDENTITY_CAPTURE_REF` is absent, run
`python3 "$WORKFLOW_DIR/../scripts/gpu_identity.py"` as before. When present, it contains the
owner-supplied absolute `path` and `sha256` of captured **raw rocminfo stdout**. Use the supplied
`WORKFLOW_DIR` and reference values to run:

```text
python3 <WORKFLOW_DIR>/../scripts/gpu_identity.py --input <path> --input-sha256 <sha256>
```

Quote each actual path as a shell argument. This route hashes the captured bytes before parsing
and does not open a GPU. Do not run a live probe when a capture reference is supplied. A read,
hash or parse failure fails Setup; do not substitute expected values or fall back to a live probe.
The owner authenticates the capture's observation and physical-device provenance separately.

Use the successful helper JSON exactly: `gfx` → `device_gfx`, `target` → `device_target`,
`marketing_name` → `device_name`, and `physical_cu_count` → `physical_cu_count`.
The parser binds fields to the same GPU identity and rejects
mixed visible identities. Only exact Marketing Name `AMD Radeon AI PRO R9700` maps to `r9700`;
other gfx1201 products stay `unknown`. Physical CUs are not PyTorch's WGP count. Report mismatches
against `EXPECTED_GFX`, `EXPECTED_TARGET`, `EXPECTED_DEVICE_NAME`, `EXPECTED_PHYSICAL_CU_COUNT`;
the captured route requires all four to match before Setup can proceed.

Create a unique run ID `TS=$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}`. Use a supplied
`EVAL_DIR_OVERRIDE`, otherwise `$EXP_ROOT/team_${KERNEL_NAME}_${TS}/${KERNEL_NAME}`. Never overwrite
a nonempty directory; add a unique suffix. Record `original_kernel_path.txt`.

### Continued waves

If `STATE_DIR/best/` exists and is nonempty, materialize it into `baseline/` and `workspace/` instead
of the original source, preserving symlinks. Verify `reference_io_sha256` when present, freeze oracle
files, and commit the seeded state. Return `resumed:true` and `prior_state` from `STATE.json`
(`cumulative`, `insights`, `ledger`, `bottleneck_now`, `best_per_case`). If integrity fails, seed from
the original and return `resumed:false` with the reason. With no saved best, use normal setup and
`resumed:false`; do not create or modify `STATE_DIR` here.

### Author mode

`KERNEL_PATH_ORIG` is the immutable operation task, not editable source. Create an empty `kernel_src`
and copy the present oracle/helper files and frozen baseline. `reference_io.pt` is optional for a
seeded synthetic oracle; do not invent a missing golden. Share a present golden by absolute symlink.
```bash
mkdir -p "$EVAL_DIR/workspace/kernel_src" "$EVAL_DIR/baseline"
echo "$KERNEL_PATH_ORIG" > "$EVAL_DIR/original_kernel_path.txt"
for f in meta.json unittest.py cases.py harness_lib.py leg_runner.py overlay_setup.py; do
  [ -e "$KERNEL_PATH_ORIG/$f" ] && cp "$KERNEL_PATH_ORIG/$f" "$EVAL_DIR/workspace/$f"
done
[ -e "$KERNEL_PATH_ORIG/reference_io.pt" ] && ln -s "$KERNEL_PATH_ORIG/reference_io.pt" "$EVAL_DIR/workspace/reference_io.pt"
for d in baseline_overlay baseline_ref baseline_src; do
  [ -d "$KERNEL_PATH_ORIG/$d" ] && cp -r "$KERNEL_PATH_ORIG/$d" "$EVAL_DIR/workspace/$d"
done
chmod -w "$EVAL_DIR/workspace/unittest.py" "$EVAL_DIR/workspace/meta.json" "$EVAL_DIR/workspace/harness_lib.py" 2>/dev/null || true
for d in baseline_overlay baseline_ref baseline_src; do
  [ -d "$EVAL_DIR/workspace/$d" ] && chmod -R -w "$EVAL_DIR/workspace/$d" 2>/dev/null || true
done
cd "$EVAL_DIR/workspace"
printf '%s\n' 'build/' '__pycache__/' '*.pyc' 'results.*' '*.so' '.torch_ext/' '.rocprofv3/' '*.o' '/.geak/' > .gitignore
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
git init -q
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "empty baseline (author mode, lang=$TARGET_LANGUAGE)"
```
The author's first commit is the code starting point, not the speedup denominator. Validate that the
frozen live baseline is available through `baseline_overlay/` or a resolvable `meta.baseline_callable`.
Report `baseline_frozen:true` only then. Otherwise report false and explain; do not use the authored
scaffold as baseline. Name the kernel from `OP_SPEC.op_kind` and language, and list oracle files present.

### Optimize mode

Use the shared materializer for baseline/workspace; it excludes generated `.so`, `.o`, build/JIT/git
state and preserves symlinks. Do not inline a different tar recipe or dereference large oracles.
```bash
mkdir -p "$EVAL_DIR/baseline" "$EVAL_DIR/workspace"
echo "$KERNEL_PATH_ORIG" > "$EVAL_DIR/original_kernel_path.txt"
for d in baseline workspace; do
  bash "${WORKFLOW_DIR:-$SKILL_DIR}/scripts/materialize_workspace.sh" \
    --src "$KERNEL_PATH_ORIG" --dst "$EVAL_DIR/$d" \
    --shared-root "$EVAL_DIR/_shared" --link-aiter
done
[ -e "$KERNEL_PATH_ORIG/reference_io.pt" ] && ln -sfn "$KERNEL_PATH_ORIG/reference_io.pt" "$EVAL_DIR/workspace/reference_io.pt"
cd "$EVAL_DIR/workspace"
printf '%s\n' 'build/' '__pycache__/' '*.pyc' 'results.*' '*.so' '.torch_ext/' '.rocprofv3/' '*.o' '/.geak/' > .gitignore
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
git init -q
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "baseline"
git --no-pager log --oneline | head    # sanity (never pages)
```
For an extracted task, freeze its baseline directories and oracle/helpers (`meta.json`, `unittest.py`,
`cases.py`, `harness_lib.py`, `leg_runner.py`, `overlay_setup.py`, and present reference IO). The
baseline is its live overlay/callable, never mutating `kernel_src`. For a plain source task, the
pristine `EVAL_DIR/baseline` and initial commit anchor the original implementation. Return
`baseline_frozen` and `baseline_callable` (empty if the directory is the anchor). Fail if no legitimate
baseline can be established.

### Optional measurement-file freezing

Only when `GEAK_FREEZE_HARNESS=1`, freeze non-source measurement files outside the workspace and
create the COMMANDMENT restorer below. Declared `config.yaml:source_file_path` files remain editable.
Default/unset/0 does not enable this extra behavior. Existing task oracle immutability still applies.
```bash
cd "$EVAL_DIR/workspace"
if [ "${GEAK_FREEZE_HARNESS:-0}" = "1" ]; then   # DEFAULT off → upstream behavior; only opt-in freezes
SRC=$(awk '/^source_file_path:/{f=1;next} /^[^[:space:]-]/{f=0} f&&/-/{sub(/.*-[[:space:]]*/,"");print}' config.yaml)
FROZEN="$EVAL_DIR/measure_golden"          # read-only golden for the perf harness (sibling of workspace)
mkdir -p "$FROZEN"
for f in scripts/harness_run.py scripts/task_runner.py scripts/_runtime.py test_cases.json; do
  printf '%s\n' "$SRC" | grep -qxF "$f" && continue   # never freeze a file the kernel legitimately owns
  [ -e "$f" ] || continue
  mkdir -p "$FROZEN/$(dirname "$f")"
  cp -f "$f" "$FROZEN/$f"
  chmod -w "$f" 2>/dev/null || true                    # soft in-workspace signal (root may override)
done
cat > "$FROZEN/restore.sh" <<'RESTORE'
#!/usr/bin/env bash
FROZEN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in scripts/harness_run.py scripts/task_runner.py scripts/_runtime.py test_cases.json; do
  [ -e "$FROZEN/$f" ] || continue
  [ -e "$f" ] && chmod +w "$f" 2>/dev/null || true
  cp -f "$FROZEN/$f" "$f"
done
RESTORE
chmod +x "$FROZEN/restore.sh"
chmod -R a-w "$FROZEN" 2>/dev/null || true             # frozen golden: read-only for the whole run
fi   # end GEAK_FREEZE_HARNESS
```
List source/config files under the workspace for downstream roles. All git operations must be
noninteractive with explicit identity and no pager/editor.

Return JSON:
```json
{
  "eval_dir": "<EVAL_DIR>",
  "workspace": "<EVAL_DIR>/workspace",
  "baseline_dir": "<EVAL_DIR>/baseline",
  "kernel_name": "<basename>",
  "device_gfx": "<detected real gfx token>",
  "device_target": "r9700|mi350x|mi355x|unknown",
  "device_name": "<detected product/marketing name>",
  "physical_cu_count": 64,
  "source_files": ["<relative paths under workspace>"],
  "baseline_frozen": true,
  "baseline_callable": "<module:attr of the frozen real online kernel, or '' if the pristine EVAL_DIR/baseline is the anchor>",
  "notes": "anything unusual about the layout"
}
```

For resumed waves also return `resumed` and `prior_state` as described above.

## PHASE=validate

Inputs: original/workspace/eval paths, `SKILL_DIR`, `GPU_ID`, `APPLY_TO_ORIGINAL`,
`COMMANDMENT.md`, `final_patch.diff`, TechLead claims, `BASELINE_TIMING`, and `FROZEN_ORACLE`.

Read COMMANDMENT's exact correctness/full-benchmark commands. Source-binding failures
(`GEAK_SOURCE_INVALID`, exit 86, `invalid_measurement`) are invalid experiments, not no-wins.
Inspect affected `.geak/invalid_measurements.jsonl`, even if the final patch is empty. An unresolved
candidate that was never validly remeasured requires `validation_status:"flagged"`,
`correctness:"not_checked"`, and the defect in `arbitration_note`; timing the reverted original does
not validate it.

Rebuild from the true original and apply the cumulative patch:
```bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
VWS="$EVAL_DIR/validation_workspace"
[ -e "$VWS" ] && mv "$VWS" "${VWS}.old_$(date +%s)_$$" 2>/dev/null || true
bash "${WORKFLOW_DIR:-$SKILL_DIR}/scripts/materialize_workspace.sh" \
  --src "$KERNEL_PATH_ORIG" --dst "$VWS" \
  --shared-root "$EVAL_DIR/_shared" --link-aiter
[ -e "$KERNEL_PATH_ORIG/reference_io.pt" ] && ln -sfn "$KERNEL_PATH_ORIG/reference_io.pt" "$VWS/reference_io.pt"
cd "$VWS"
git init -q
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "validation_baseline"
git apply "$EVAL_DIR/final_patch.diff"
bash "${WORKFLOW_DIR:-$SKILL_DIR}/scripts/reclaim_eval_artifacts.sh" --eval-dir "$EVAL_DIR" --keep-round 0 2>/dev/null || true
if [ "${GEAK_FREEZE_HARNESS:-0}" = "1" ]; then
  SRC=$(awk '/^source_file_path:/{f=1;next} /^[^[:space:]-]/{f=0} f&&/-/{sub(/.*-[[:space:]]*/,"");print}' config.yaml)
  for f in $(git diff --name-only); do
    printf '%s\n' "$SRC" | grep -qxF "$f" || git checkout -- "$f"
  done
fi
```
Run correctness in the fresh workspace, then FULL_BENCHMARK under
`bash "$SKILL_DIR/scripts/gpu_lock.sh" "$GPU_ID" <full command>` and parse every case. Correctness
failure or patch-apply failure is flagged; never accept its speedup.

Use frozen `BASELINE_TIMING` and independently measured optimized times. Per-case speedup is
`baseline_ms/optimized_ms`; geomean is `exp(mean(log(speedups)))`. Also report arithmetic mean.
For workload-aligned cases, recompute the primary metric with the same audited helper:
```python
import harness_lib as h, json
meta = json.load(open("meta.json"))
per_case = [{"sig": c["name"], "regime": c.get("regime", ""), "m": c.get("m"),
             "baseline_ms": BASELINE_MS[c["name"]], "optimized_ms": OPT_MS[c["name"]]}
            for c in meta["workload"]["cases"]]
res = h.serving_weighted_speedup(per_case, meta)
director_verified_speedup_weighted = res["weighted"]
```
Do not use static profile weights/counts. The helper applies served-regime filtering, measured
baseline latency × analytic calls, largest-M allocation and identity guards. If weighted is None,
repair/remeasure untrusted buckets first; disclose any fallback to valid unweighted geomean.

### Timing and cache evidence

Before arbitration, preserve the timing basis:
- `FROZEN_ORACLE` not true: `timing_basis:"not_applicable"`, `timing_receipt:null`; explain that this
  lane uses its own `baseline_timing.json`. This alone does not flag a pass-through lane, nor prove
  a receipt-backed device-time result.
- `FROZEN_ORACLE=true`: parse `GEAK_TIMING_RECEIPT` from FULL_BENCHMARK and copy it verbatim into
  `director_validation.json`. `all_primed:true` gives `clean`. If not all primed but
  `timer_unprimed:false`, use `host_bound`, naming affected cases; it is a dispatch-latency ratio,
  not demonstrated kernel gain in an already-graph-replayed server. `timer_unprimed:true` gives
  `unprimed` and flagged; missing receipt gives `unknown` and flagged. Re-freeze stale harnesses and
  remeasure; do not invent correction factors.

For receipt-backed runs copy `cache_mode` to `cache_basis`. Only `read-evict` is supported for new
measurements. Missing/`unknown_write_evict` maps to `unknown_write_evict` and flagged; preserve any
other historical mode verbatim and flag it. Never relabel modes or combine different cache bases
into one speedup. Carry `timing_basis` in every summary quoting the result.

Compare the independent primary metric with the claim: within 10%, or Director higher, is accepted;
Director lower by over 10% is flagged with the measured value as official. Prior correctness,
source-binding and receipt flags take precedence. A reproducible lower result can still be reported;
recommend a corrective round only for failed correctness or patch application.

Only if `APPLY_TO_ORIGINAL=true` and status is accepted, apply the patch to the original:
```bash
cd "$KERNEL_PATH_ORIG"
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
if [ ! -d .git ]; then
  git init -q
  git -c user.email=team@workflow -c user.name=team add -A
  git -c user.email=team@workflow -c user.name=team commit -q -m "pre_team_baseline"
fi
git apply "$EVAL_DIR/final_patch.diff"
```
Write `EVAL_DIR/director_validation.json`, including cache/timing receipts and any limitations.

Return JSON:
```json
{
  "kernel_name": "<name>",
  "director_verified_speedup_geomean": 0.0,
  "director_verified_speedup_arithmetic": 0.0,
  "director_verified_speedup_weighted": 0.0,
  "tech_lead_reported_speedup_geomean": 0.0,
  "validation_status": "accepted|flagged",
  "timing_basis": "clean|host_bound|unprimed|unknown|not_applicable",
  "timing_receipt": null,
  "correctness": "pass|fail",
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0}],
  "applied_to_original": "true|false",
  "arbitration_note": "accept reason, or what to re-task if flagged",
  "final_patch": "<EVAL_DIR>/final_patch.diff"
}
```
