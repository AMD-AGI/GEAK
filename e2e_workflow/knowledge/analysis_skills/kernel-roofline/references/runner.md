# Runnable measurement and reporting workflow

Use the bundled runner rather than copying a historical campaign's `split()` or
roof calculator. Existing campaign rigs and their frozen results are read-only
evidence; they are not automatically repaired by updating this package.

## One case, either engine

The GPU environment needs PyTorch and Triton. The TL reporting environment can be
different, but must load the selected upstream main commit and its dependencies.

```bash
python3 scripts/case_runner_template.py \
  --ut-root /path/to/tasks --task glm52_add_rmsnorm_quant_task \
  --case pre_m10129_res --arm candidate \
  --cache cold-read --llc-mib 256 --execution eager \
  --iters 5 --warmup 5 --capture --output results/case-candidate

python3 scripts/tl_only.py \
  --measurement results/case-candidate/measurement.json \
  --tl-source /path/to/current-main/TraceLens --arch /path/to/frozen_arch.json \
  --output results/tl-only/candidate
```

Use the actual task/case and cache size. `--harness loader.py` optionally supplies
an existing `load_ut`/arm guard, but its flush, timing, and handwritten roof code
are not called. Without it, the loader imports `unittest_ori.py` or `unittest.py`.
Auto selects `_launch`/`_timed` before `_invoke`; a separate `_gather` without a
known timed entry is an error. `_BASEM`/`_CANDM` module-based UTs and ordinary
function targets are supported. An explicit `--runner invoke` is historical,
unverified scope and is not eligible for a scored fold.

For a different UT, supply `--adapter adapter.py` instead of UT loader arguments.
Its `build_case(task, case, arm)` returns:

```python
{
    "run": timed_zero_argument_callable,
    "metadata": {
        "operator_boundary": "complete operator, including required reductions",
        "implementation_source": "/path/to/selected/implementation.py",
        "case_metadata": {"regime": "decode", "share_of_op_calls": 1},
    },
    "validate": correctness_callable,          # optional; False means failure
    "expected_dispatches": {"exact_symbol": 1},  # optional stronger oracle
}
```

Construct inputs and correctness gathers outside `run`. Validate against the UT's
reference and tolerances; if validation is external, attach that evidence. Absence
of a validation callback is recorded as unverified, never as a passing test.

## Cache and execution contracts

- `cold-read` is the default conditioning protocol. It reads a separate buffer
  and writes only block sums. Size defaults to three times the declared relevant
  LLC and must be at least twice it. gfx950 defaults to the campaign's 256-MiB
  MALL assumption; an explicit `--llc-mib` takes precedence. Other devices need a
  suitable device property or explicit size. Record the assumption for partitions.
- A self-test checks the reduction results and records conditioning duration.
  Captured traces/counter passes must also contain the conditioning dispatch
  before each measured call. This validates the protocol, not a counter proof of
  every operand miss. `cold-write` is an explicit diagnostic with dirty-line
  writeback caveats. `warm` performs no eviction. No task-owned flush is called.
- Warmup, JIT, autotune, validation and import self-checks occur in the same process
  before measured markers. This avoids relying on autotuner caches across processes.
- Each iteration is conditioning, begin marker, complete operator, end marker.
  The runner synchronizes at boundaries to include all required streams. Marker
  and conditioning work is excluded by scope, not by broad symbol substrings.
- `--execution graph` captures exactly one operator per graph and conditions before
  every replay. Its capture trace is stored separately for native TL graph merging.
  If TL cannot recover that graph's model metadata, retain timings and the error;
  do not borrow eager model/roof values and label the result native graph analysis.
  Hardware validation against main `90f2f6f` found that the native merger recognizes
  framework execution annotations (vLLM/SGLang/ATOM), but not generic `krf.seam`
  ranges. A separate capture folder alone does not fix this: standalone graph
  timing can pass with zero native operation rows. Report `UNAVAILABLE` and retain
  both traces; do not fabricate a framework annotation to select a model.
- `--cpu-affinity auto` pins to allowed GPU-local CPUs when PCI/NUMA mapping is
  available, otherwise reports unverified affinity. `local` requires successful
  pinning; `unchanged` preserves affinity. `--gpu-pci-bus-id` resolves ambiguous
  mappings. No physical GPU index is inferred from a render-node arithmetic rule.

`measurement.json` preserves environment, cache protocol, selected closure,
correctness state, per-call current-stream event intervals, and device busy-time
samples. `trace.raw.json` is unsliced; `trace.json` retains selected GPU activity,
CPU model metadata, and both `ac2g` flow endpoints. `scope.json` lists counted and
excluded work with times. Dispatch multiplicity must agree across measured calls
and with an adapter's explicit oracle when supplied.

`tl_only.py` checks the source commit/import path and invokes only the native TL
reporter, with pseudo-ops and call stacks enabled. It preserves raw CSVs and
exports operation rows with their complete context. Its additional case rollup
uses native reference/time fields and is labeled separately from native percentage
statistics. Supply `--tl-commit` for the commit frozen at comparison start.
Exported rows describe native operation instances, not an invented model for the
entire UT closure. Their weights include native operation count per measured call.
If a decomposition changes native operation multiplicity, do not force a same-row
roofline comparison: preserve each native report and compare complete-closure
timings from the measurement manifests. Different A/B weights fail explicitly.
For a standalone UT without usable CPU operator metadata, perform the source
review in [ut-context.md](ut-context.md), then pass `tl_only.py --ut-context` the
bound identity/argument evidence. New UT measurements record `input_metadata` and
the UT source hash before timing; these descriptors also detect stale metadata.
Use `scripts/compare_measurements.py --baseline baseline/measurement.json
--candidate candidate/measurement.json` for that complete-closure latency
comparison, including unsupported TL operations. It checks matching case and
measurement conditions and does not invent a roofline reference.

## Counter mode

Omit `--capture` under rocprof; use the same runner. The driver supplies a manifest
path to the child and retains profile CSVs, logs, selected dispatch IDs, and JSON
metrics. `--scope measured-window` is the default. It checks marker pairs, cache
dispatch presence, per-call multiplicity, and cross-pass ID consistency before
calling `analyze -d ...`. Missing/inconsistent windows fail rather than analyzing
the whole process. If a tool version remaps IDs between passes, it fails explicitly
and retains the CSVs for a version-specific adapter.

The driver writes `workload.sh` and invokes `bash workload.sh` from its profile
directory. Keep this boundary: rocprof-compute 3.4 flattens workload arguments,
so `bash -c <command>` can silently execute only the first word. The script changes
to the requested workdir and preserves the original command's quoting. After
profiling, the driver verifies that the frozen peak CSV's hash is unchanged.
It enables profiler verbosity to retain individual `pmc_perf_<pass>.csv` files:
rocprof-compute 3.4 otherwise deletes them after joining, preventing cross-pass
window verification. Preserve these files alongside the merged CSV.

```bash
export ROOFLINE_MIBENCH_CACHE=/path/to/verified/frozen/roofline.csv
python3 scripts/run_roofline.py --workdir "$PWD" \
  --cmd "python3 scripts/case_runner_template.py --adapter /path/to/adapter.py --task op --case case1 --arm baseline --iters 3 --cache warm" \
  --name op_case1_baseline --output results/rocprof/leg --expected-calls 3
```

For a batch, `rocprof_leg_template.sh` consumes four columns:
`task case regime call_weight`. Set `UT_ROOT` or `ADAPTER`, `FROZEN_CSV`, and a
fresh `OUT`. It preserves the request manifest, propagates failures, and calls
`collect_results.py` to retain missing cases in both arms. Missing weights must be
resolved before the campaign; explicit equal weights are allowed but are not a
measured deployment distribution. Paths and command arguments are shell-quoted.

## Validation and dispatch diagnostics

```bash
python3 -m unittest discover -s tests -v
python3 tests/gpu_smoke.py --output /tmp/roofline-gpu-smoke
python3 scripts/dispatch_probe.py --cache cold-read --llc-mib 256 --output results/probe
```

GPU smoke exits 77 when no GPU is available. CPU tests include the six original
top-k raw reports, scope/flow retention, warmup exclusion, missing markers, changed
dispatch counts, coverage, and the original launch/gather failure. The dispatch
probe records eager and one-node graph device/event samples under the chosen
cache/NUMA condition; it is never substituted into a TL roof or used as a strict
bound. No universal 85% target is imposed on tiny/unsupported operations.
