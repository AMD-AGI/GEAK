# AVO Runner — run geak-avo and adapt its lineage to the contract

You drive GEAK's existing Python **`geak-avo`** (continuous single-lineage kernel
evolution) for ONE kernel, then **adapt its outputs** to the team_workflow
contract so the Director can validate them. This is the **Step 1 thin wrapper**;
Step 2 will replace it with a native deterministic loop (see PLAN.md).

Inputs: `KERNEL_PATH_ORIG`, `EVAL_DIR`, `KERNEL_NAME`, `SKILL_DIR`, `GPU_IDS`,
`BUDGET_S` (wall-clock seconds), `MIN_COMMITS`, `TASK`, `VERIFY_REPEATS`,
`USE_PROFILING`, `GEAK_ROOT`, `GEAK_CMD`.

> Background on what geak-avo produces (lineage, commit gate, `avo-v{N}` tags,
> rich commit messages, `avo_state/`): read
> `${SKILL_DIR}/knowledge/avo_evolution_playbook.md` and
> `${SKILL_DIR}/knowledge/lineage_usage.md`.

---

## 1. Locate the geak-avo CLI

Prefer `GEAK_CMD` (default `geak-avo`). If it is not on PATH and `GEAK_ROOT` is
given, use the module entry point from there:

```bash
if command -v "$GEAK_CMD" >/dev/null 2>&1; then
  AVO="$GEAK_CMD"
elif [ -n "$GEAK_ROOT" ]; then
  AVO="python3 -m minisweagent.run.avo.controller"   # or the repo's console-script
  export PYTHONPATH="$GEAK_ROOT/src:${PYTHONPATH:-}"
else
  echo "geak-avo not found"; # return status=failed with a clear note
fi
```

If geak-avo cannot be located, return `{"status":"failed","final_patch":"",
"reported_geomean":0,"notes":"geak-avo CLI not found; set args.geak_root or
args.geak_cmd"}` and stop.

## 2. Configure measurement rigor (Tier-1)

Export noise-robust verification so a noisy "win" cannot enter the lineage
(the paper measures 10x; we default to `VERIFY_REPEATS` >= 3):

```bash
export GEAK_GPU_DEVICE="$(echo "$GPU_IDS" | cut -d, -f1)"
export HIP_VISIBLE_DEVICES="$GPU_IDS"
# noise-robust commit gate (Tier-1): median over repeats + a significance margin
export AVO_VERIFY_REPEATS="$VERIFY_REPEATS"
```

## 3. Launch geak-avo (output INTO the eval dir)

Point geak-avo's output at `$EVAL_DIR/avo_run` so all lineage state lands under
our eval dir. Use a **wall-clock budget** (`BUDGET_S`); when called from e2e this
is a time slice, not literal days.

```bash
mkdir -p "$EVAL_DIR/avo_run"
$AVO \
  --repo "$KERNEL_PATH_ORIG" \
  --output "$EVAL_DIR/avo_run" \
  --task "${TASK:-Optimize the $KERNEL_NAME kernel. Metric: latency (lower is better).}" \
  --gpu-ids "$GPU_IDS" \
  --total-budget-s "$BUDGET_S" \
  $( [ "$USE_PROFILING" = "false" ] && echo "--no-profiling" )
```

geak-avo resumes automatically if `$EVAL_DIR/avo_run/avo_state/lineage.json`
already exists. It is non-fatal if individual steps fail — let it run to the
budget. Capture stdout/stderr to `$EVAL_DIR/avo_run/runner.log`.

## 4. Locate the lineage outputs

After it returns, the run dir holds (see `lineage_usage.md`):
- `avo_repo/` — the isolated clone with `.git` + `avo-v{N}` tags (the lineage)
- `avo_state/lineage.json` — committed versions + best id + per-version score
- `avo_state/patches/v{N}.patch` — per-version patches
- `final_report.json` — best result (canonical GEAK shape + an `avo` block)

```bash
RUN="$EVAL_DIR/avo_run"
AVO_STATE="$RUN/avo_state"
AVO_REPO="$RUN/avo_repo"
```

## 5. Produce the final patch (best avo-v{N} vs the TRUE baseline)

The Director validates `final_patch.diff` against the ORIGINAL baseline, so the
patch must be **baseline -> best version** (NOT incremental). Read the best
version id from `lineage.json` (`best_id`), then emit the cumulative diff from the
lineage's baseline tag `avo-v0` to the best tag:

```bash
BEST=$(python3 -c "import json;d=json.load(open('$AVO_STATE/lineage.json'));print(d.get('best_id',''))")
cd "$AVO_REPO"
git diff "avo-v0" "avo-$BEST" > "$EVAL_DIR/final_patch.diff"
```

If `best_id` is `v0` (no improvement committed), return `status=ok` with an empty
`final_patch` and `reported_geomean=1.0` so the Director records "no improvement".

## 6. Read the reported scores + a human-readable lineage table

Pull the verified geomean/per-case for the best version from `lineage.json` /
`final_report.json`. Generate the version table for the report (reuse GEAK's tool
if available, else summarize `lineage.json`):

```bash
python3 "$GEAK_ROOT/scripts/avo_lineage_report.py" "$AVO_STATE" --format md \
  > "$EVAL_DIR/lineage_report.md" 2>/dev/null || true
```

## 7. Return the adapted contract

```json
{
  "status": "ok",
  "final_patch": "<EVAL_DIR>/final_patch.diff",
  "reported_geomean": 0.0,
  "reported_arithmetic": 0.0,
  "best_version": "avo-v<N>",
  "num_commits": 0,
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0}],
  "avo_state_dir": "<EVAL_DIR>/avo_run/avo_state",
  "lineage_report": "<EVAL_DIR>/lineage_report.md",
  "notes": "<budget used, #attempts vs #commits, any step failures>"
}
```

Notes:
- Report **only verified** numbers from the lineage; never invent a speedup.
- The Director re-measures independently — your `reported_geomean` is provisional.
- Keep `runner.log` for debugging; surface any infra-failure abort in `notes`.
