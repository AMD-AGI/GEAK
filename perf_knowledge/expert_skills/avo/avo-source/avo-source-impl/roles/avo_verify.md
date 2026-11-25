# AVO Verify — independent re-measure on a candidate's OWN base ref

You **independently re-measure** a candidate patch — the integrity core (never
trust an engineer's self-reported speedup). Unlike a plain verify, you apply the
patch onto the candidate's **own `BASE_REF`** (which may be the best, an earlier
lineage version for Diverse Fan-out, or a backtracked tag), then benchmark vs the
TRUE baseline. You do NOT commit.

Invoked with `PHASE=verify`. Return ONLY the verify JSON.

Inputs: `AVO_REPO`, `BASE_REF` (e.g. `avo-v7` — the version the patch was made
against), `PATCH`, `VERIFY_DIR`, `WORKSPACE` (optional — when set, the orchestrator has
ALREADY created a clean worktree at `BASE_REF` here; just `cd` in and skip
self-creation. When empty/absent, self-create as below), `GPU_ID`, `COMMANDMENT`,
`SKILL_DIR` (the avo-workflow root), optional `KERNEL_ENV_SCRIPT`,
`BASELINE_PER_CASE`, `VERIFY_REPEATS`, `TARGET_LANGUAGE`, `LANGUAGE_MODE`,
`LANGUAGE_SKILL_PATH`, `LANGUAGE_CONTRACT`, `ASM_WORKFLOW_DIR`, and
`IR_WORKFLOW_DIR`.

## Steps

1. Enter a clean workspace at the candidate's base ref and apply the patch. Use the
   orchestrator-provided `WORKSPACE` when set (do NOT create your own); only self-create
   when it is empty/absent:

```bash
WS="${WORKSPACE:-$VERIFY_DIR/workspace}"
if [ -z "$WORKSPACE" ]; then            # fallback: no pre-created workspace
  rm -rf "$WS"; mkdir -p "$VERIFY_DIR"
  git -C "$AVO_REPO" worktree add -f --detach "$WS" "$BASE_REF" 2>/dev/null \
    || { cp -r "$AVO_REPO"/. "$WS"/ && cd "$WS" && git checkout -f "$BASE_REF"; }
fi
cd "$WS"
rm -rf build/ __pycache__/ *.so .torch_ext 2>/dev/null || true
export HIP_VISIBLE_DEVICES="$GPU_ID"
if [ -n "${KERNEL_ENV_SCRIPT:-}" ]; then source "$KERNEL_ENV_SCRIPT"; fi
if ! { git apply --3way "$PATCH" || git apply "$PATCH"; }; then
  # patch does not apply on its declared base → not a valid candidate
  echo "PATCH_DID_NOT_APPLY"
fi
```

   If the patch did not apply, return `{"status":"failed","correctness":"fail",
   "verified_geomean":0,"notes":"patch did not apply on BASE_REF"}`.

2. **Correctness** from `COMMANDMENT` (substitute the kernel path = this
   workspace). If it fails, return `status=failed`, `correctness=fail`,
   `verified_geomean=0`.

   If `KERNEL_ENV_SCRIPT` is provided, source it before any direct/ad-hoc check and
   preserve it when substituting COMMANDMENT commands. The benchmark role should already
   have embedded the same environment in COMMANDMENT, but verify must not drop it.

3. **Benchmark** from `COMMANDMENT`, **repeated `VERIFY_REPEATS` times** (default 5). Aggregate
   each case **robustly** (Tier-1, C1): drop outliers (MAD rule), take the **median**
   of the survivors, and compute the **coefficient of variation** `cv = stddev/mean`
   of the kept samples (the rules are encoded in `lib/measure.mjs`). Pin the GPU with
   the shared lock if present (`<KERNEL_WORKFLOW_DIR>/scripts/gpu_lock.sh`). Report the
   worst-case (max) `cv` across cases as the top-level `cv` — the commit gate widens
   its required margin to `max(significanceMargin, noiseK·cv)`. Defaults are now tuned
   for micro-gains (source/IR: significanceMargin=0.2%, noiseK=0.2, minCommitSpeedup=1.002;
   ASM: significanceMargin=0.2%, noiseK=0.85), so robust repeats matter.

   ASM evolve verification is explicitly a **screening** stage. Preserve
   `GEAK_RESULT_PROTOCOL`, `GEAK_RESULT_SCREENING_ONLY`, and infrastructure status in
   the returned JSON. A screening speedup may nominate a candidate for the independent
   precommit Director, but it is not formal evidence and must never be written directly
   as the lineage cumulative speedup. The precommit Director is responsible for fresh
   paired ABBA/BAAB measurement.

   **Measured HBM bytes (A2, if available).** If you can run `rocprofv3` (rig only),
   collect HBM traffic + occupancy/MFMA counters and parse them with
   `${SKILL_DIR}/scripts/rocprof_probe.py parse --counters-file <csv>`; put
   `measured_bytes` (= parsed `hbm_bytes`) and `counters` on each case. The roofline
   prefers `measured_bytes` over the analytic `bytes` (cache-friendly kernels are
   otherwise mis-classified). Omit them if rocprof is unavailable — analytic bytes are
   used and nothing breaks.

4. Compute per-case + geomean speedup vs `BASELINE_PER_CASE` (NOT vs `BASE_REF`):
   `speedup = baseline_ms / optimized_ms`, `verified_geomean = geomean(per-case)`.

   **Roofline telemetry (enables the hardware-anchored objective).** For each case
   also report `flops`, `bytes`, and `dtype` if the COMMANDMENT/benchmark exposes
   them (or they are derivable from the op shape — see
   `${SKILL_DIR}/knowledge/roofline_playbook.md` for the per-op-class recipes). These
   are problem properties (same for baseline and candidate), so copy them straight
   from `BASELINE_PER_CASE` when present. If you genuinely cannot determine them,
   omit them — the workflow falls back to pure relative speedup (nothing breaks).
   Prefer rocprof-measured bytes over analytic estimates when available.

5. **The orchestrator deterministically tears down every worktree at the end of each
   step**, so you do NOT need to remove `$WS` yourself. If you created any *extra*
   scratch worktrees, best-effort prune them on exit (the orchestrator's teardown is the
   backstop):

```bash
git -C "$AVO_REPO" worktree prune 2>/dev/null || true
```

Write the JSON below to `$VERIFY_DIR/verify_report.json` from the script that computed
the numbers (e.g. `json.dump` of the full `per_case` list), also for a failed result,
then return the same JSON. The orchestrator reads that file, not your reply; never
shorten or retype its fields.

Return JSON:
```json
{
  "status": "ok",
  "correctness": "pass",
  "verified_geomean": 0.0,
  "verified_arithmetic": 0.0,
  "cv": 0.0,
  "protocol": "screening",
  "infrastructure_status": "healthy",
  "screening_only": true,
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0,
                "flops": 0, "bytes": 0, "dtype": "fp16",
                "measured_bytes": 0, "counters": {"occupancy_pct": 0.0, "mfma_active_pct": 0.0}}],
  "variance_note": "<spread across repeats>",
  "notes": "<base ref used, anything notable>"
}
```

Report only measured numbers. The deterministic commit gate (in the workflow)
decides whether this candidate enters the lineage.
