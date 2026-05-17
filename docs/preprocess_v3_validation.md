# Manual validation: v3 preprocess cutover

## Goal

Commit set 5a wired the v3 LLM-driven preprocess orchestrator
(`PreprocessOrchestratorAgent`) into the CLI flow. The legacy 5-phase
pipeline under `src/minisweagent/run/preprocess/` and the
multi-process plumbing under `src/minisweagent/pipeline_workers/` are
**still on disk** but are no longer on the call graph from the CLI
entry. This runbook is the manual gate before the legacy code gets
deleted: run v3 end-to-end against representative workloads (Triton,
HIP, optionally PyTorch→FlyDSL), inspect the artifacts, and only then
ask for legacy removal.

## What changed at the call site

- `src/minisweagent/run/mini.py` line 37 (single-line import change).
- New module: `src/minisweagent/run/preprocess_v3/adapter.py`
  (`run_preprocess_v3(**legacy_kwargs) -> preprocess_ctx_dict`).
- Commit that flipped the import: `4810ab94`
  (`feat(preprocess-v3): wire PreprocessOrchestratorAgent into run_pipeline`).

Nothing else in `run/`, `agents/`, or `subagents/<name>/` changed.

## Quick sanity check (no GPU required)

```bash
cd /home/upandey/unification/GEAK
PYTHONPATH=src pytest tests/run/preprocess_v3 -q
```

Expected: **`209 passed, 2 skipped`** (181 before commit set 5a, +28 new
tests across the three commits in this set). Failures here mean the
adapter, the orchestrator, or the v3 subagent class regressed; do **not**
proceed to the end-to-end runs below.

## End-to-end validation

These runs require a GPU host with the AMD LLM router configured. Each
exercises a different language path through the v3 orchestrator.

### 1. Triton fixture

A small Triton kernel that already lives in a known-good repo with
benchmark + correctness scripts.

```bash
geak \
  -t "Optimize the add_kernel for B=2048 D=128 fp16." \
  --kernel <path-or-url-to-a-triton-kernel> \
  --output ./validation_runs/triton_v3 \
  --gpus 0
```

Expected artifacts under `./validation_runs/triton_v3/`:

- `CODEBASE_CONTEXT.md` (step 1 output)
- `test_harness.py` (or whatever the LLM names it — step 3 output)
- `baseline_metrics.json` (step 4 part 1 — adapter-written, schema
  matches the legacy shape: `median_ms`, `samples_ms`, `stdev_ms`,
  `duration_us`)
- `profile.json` (step 4 part 2 — profiler-mcp output)
- `compute_speedup.py` (step 5 output — verifies the baseline parses)
- `COMMANDMENT.md` (step 6 output)

### 2. HIP fixture

Same shape against a HIP/ROCm kernel (`.hip` / `.cpp` source).

```bash
geak \
  -t "Optimize the rocPRIM segmented_radix_sort benchmark target." \
  --kernel <path-or-url-to-a-hip-kernel> \
  --output ./validation_runs/hip_v3 \
  --gpus 0
```

Same artifact list. The `kernel_type` field in the resulting
`preprocess_ctx` should read `"hip"` rather than `"triton"` — the v3
language detector (`preprocess_v3.lang.detect_language`) picks this up
from the file extension + content hints.

### 3. PyTorch → FlyDSL fixture (optional — requires `FLYDSL_HOME`)

Translation runs as a **tool call** in v3 (`translate_to_flydsl`), not as
a subagent dispatch. The orchestrator calls it once when
`source_language != target_language and target_language == "flydsl"`.

```bash
[[ -n "$FLYDSL_HOME" ]] || { echo "FLYDSL_HOME not set; skip"; exit 0; }

geak \
  -t "Translate this PyTorch module to FlyDSL and optimize." \
  --kernel <path-to-pytorch-nn-Module> \
  --target-language flydsl \
  --output ./validation_runs/pytorch_to_flydsl_v3 \
  --gpus 0
```

Additional expected artifact: the translated `.fdsl` file in the
orchestrator's `output_dir` (the same one the
`translate_to_flydsl` tool writes to). The result dict will contain a
`v3_translation` entry carrying the `TranslationResult` projection.

If `FLYDSL_HOME` isn't set on the validation host, document the skip in
the run log and move on — this path is exercised by unit tests
(`tests/run/preprocess_v3/test_translate.py`) and the integration test
under `tests/run/preprocess_v3/test_orchestrator_integration.py` covers
the orchestrator side with a mocked translator.

## How to inspect the PreprocessResult

The v3 adapter projects the orchestrator's `PreprocessResult` into the
legacy `preprocess_ctx` dict so downstream consumers (the round loop,
the evaluation contract, planned-mode orchestrator) don't have to
change. The richer typed result is **also** carried on the dict under
`v3_subagent_runs` / `v3_elapsed_s` / optionally `v3_translation`.

To read the full result from a run's artifact directory:

```python
import json
from pathlib import Path

run_dir = Path("./validation_runs/triton_v3")
ctx_path = run_dir / "preprocess_ctx.json"   # if surfaced by your run wrapper
ctx = json.loads(ctx_path.read_text()) if ctx_path.exists() else {}
print("success:", bool(ctx.get("harness_path") and ctx.get("baseline_metrics_path")))
print("kernel_type:", ctx.get("kernel_type"))
print("harness_path:", ctx.get("harness_path"))
print("baseline_median_ms:", (ctx.get("baseline_metrics") or {}).get("median_ms"))
print("commandment_path:", ctx.get("commandment_path"))
print("subagent runs:")
for run in ctx.get("v3_subagent_runs", []):
    print(f"  - {run.get('name')}: success={run.get('success')} elapsed={run.get('elapsed_s')}s")
```

For a live run, the orchestrator log line `v3 preprocess completed in
<n>s (success=True/False, errors=<n>)` is the easiest sanity ping.

## Comparing v3 vs legacy outputs

To run the **legacy** preprocess against the same input for a side-by-side
artifact diff, temporarily revert the wiring commit (sha `4810ab94`):

```bash
git stash                    # save any uncommitted edits
git revert --no-commit 4810ab94  # re-points mini.py at the legacy shim
# Run the same `geak -t ... --output ./validation_runs/<name>_legacy ...` command
git reset --hard HEAD        # restore v3 wiring
git stash pop                # restore edits if any
```

Then diff the two artifact trees:

```bash
diff -r ./validation_runs/triton_v3 ./validation_runs/triton_legacy \
  | grep -v -E '\.log$|\.json$' | head
diff <(jq -S . ./validation_runs/triton_v3/baseline_metrics.json) \
     <(jq -S . ./validation_runs/triton_legacy/baseline_metrics.json)
```

Expect:

- `CODEBASE_CONTEXT.md`: same files listed, structurally equivalent
  rendering (the v3 explore wraps the same legacy generator).
- `baseline_metrics.json`: same `median_ms` ±1-2% (microbenchmark noise);
  `duration_us` derived as `median_ms * 1000`.
- `COMMANDMENT.md`: structurally compliant with the universal contract
  (same template family).
- `test_harness.py`: may differ in formatting/comments — both must pass
  `--correctness`, `--profile`, `--benchmark`, `--full-benchmark` modes
  and print `GEAK_RESULT_LATENCY_MS=<float>` as the last line of
  `--benchmark`.

## Path-A validation

Commit set 7 added a **Path-A short-circuit** to the orchestrator. When
the task prompt carries explicit run instructions, the orchestrator
calls the new `commandment_from_user_command` tool to render
`COMMANDMENT.md` directly from the user's command and SKIPS the
`harness-generator` / `harness-verifier` / `speedup-verify` subagent
dispatches. The decision is encoded in the orchestrator's `Step 0`
system-prompt section — it's LLM judgment, not a regex on the task.

### Example invocation

```bash
geak \
  -t 'run via python my_kernel.py --benchmark --shape 4096' \
  --kernel <path-to-my_kernel.py> \
  --output ./validation_runs/path_a_v3 \
  --gpus 0
```

The `run via ...` phrasing (or any explicit `python ...`,
`./bench ...`, or `make bench`-style invocation in the task) is what
triggers Path A.

### Expected artifacts (Path A)

Under `./validation_runs/path_a_v3/`:

- `CODEBASE_CONTEXT.md` — still produced (Path A calls
  `codebase_explore` after the short-circuit; the artifact is useful
  to downstream consumers regardless of path).
- `baseline_metrics.json` — still produced (Path A calls
  `collect_baseline` against the kernel using the user's command).
- `profile.json` — still produced (Path A calls `collect_profile`).
- `COMMANDMENT.md` — produced by `commandment_from_user_command` with
  the user's command projected into the 5 canonical sections.
- **No `test_harness.py`** — Path A does not generate a harness; the
  harness IS the user's run command.
- **No subagent_run entries** for `harness-generator`,
  `harness-verifier`, or `speedup-verify` in the resulting
  `preprocess_ctx`. The `v3_subagent_runs` list should be empty on a
  clean Path-A run.

### Confirming `path_taken="A"` from the run log

The orchestrator records the structural decision on
`PreprocessResult.path_taken` and surfaces it via the adapter into
`preprocess_ctx`. Grep the run output for either:

```bash
grep -E 'path_taken|commandment_from_user_command' \
  ./validation_runs/path_a_v3/*.log
```

or inspect the `preprocess_ctx.json` directly:

```python
import json
ctx = json.loads(open("./validation_runs/path_a_v3/preprocess_ctx.json").read())
print(ctx.get("v3_path_taken"))  # expect "A"
```

The `commandment_from_user_command` tool name appears in
`result.tool_calls` exactly once for a Path-A run, and never appears
for a Path-B run.

### Partial-mode-coverage warnings

If the user's command covers only one mode (e.g. `--benchmark`) and
the LLM asks the tool to infer the others, the rendered
`COMMANDMENT.md` carries `# PATH_A_PARTIAL_COVERAGE: <mode> inferred
from <source-mode>` markers in the inferred sections. Grep for them
on a Path-A run to confirm:

```bash
grep -c PATH_A_PARTIAL_COVERAGE ./validation_runs/path_a_v3/COMMANDMENT.md
```

Three matches is typical for a `--benchmark`-only source command
(correctness, profile, and full_benchmark all inferred).

### Reverting Path A if it misbehaves

The Path-A short-circuit is orchestrator-side only. The three always-on
subagents are unchanged. To temporarily disable Path A without
reverting all of commit set 7, revert the system-prompt commit (commit
3 of the set) so the LLM no longer knows about Step 0:

```bash
# Find the commit that updated the system prompt:
git log --oneline --grep='orchestrator system prompt teaches Path-A'
# Revert that single commit.
git revert <sha>
```

A full revert of the commit set (back to the 6-step-only flow) is:

```bash
# Walk back from HEAD to the commit before the set started:
git log --oneline -n 10
git revert <sha-1>..<sha-5>  # or one-by-one
```

## What to do if v3 fails

1. Capture the failure: copy the run's stdout/stderr and any
   `errors` array from the `preprocess_ctx` dict.
2. Revert the wiring commit to restore the legacy path:
   `git revert 4810ab94`.
3. File a note describing the failure mode (which step, which subagent,
   what the orchestrator log said immediately before the failure). The
   v3 orchestrator's subagent_runs list pinpoints which dispatch failed
   even on a partial-result run.
4. Re-run with the legacy path to confirm the same workload still works
   there — that bounds the regression to v3-only.

The `tests/run/preprocess_v3/test_orchestrator_integration.py` and
`tests/run/preprocess_v3/test_adapter.py` suites are the right places to
add a unit-level repro when the failure is reproducible.
