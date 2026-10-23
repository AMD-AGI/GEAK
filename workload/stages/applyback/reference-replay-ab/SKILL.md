---
name: reference-replay-ab
description: Re-run the reference harness with cold-verified kernels, a fresh stock control, independent candidate arms, and actual serving-rank source evidence.
---

# Reference replay A/B

Use the original reference harness and native metric basis. The slot builds one
accepted source overlay from cold-verified task bindings, measures a fresh stock
control, then measures the configured candidate arms. Operator and E2E results
remain separate.

The agent must prepare and review a campaign-specific **one-shot serving source
observer before invoking `run.sh`**. There is no generic observer implementation
in this slot. `serving_source_proof.producer_file` and its reviewed SHA-256 are
required preconditions. Missing files, changed bytes or syntax errors block before
long replay. The builder installs that exact file as a bootstrap in each accepted
source copy. A producer bootstrap exception aborts the serving interpreter.

The observer must collect actual observations from every serving worker after lazy
computation loads have been exercised, inspect the callable references actually
used by callers, and remove itself before measured traffic. It must not substitute
launcher hashes, coordinator imports, or standalone UT receipts for serving proof.
Prepare and validate it under the agent-shaped applyback procedure, then record
the reviewed file's hash in configuration before running a serving arm.
See [SOURCE_PROOF.md](SOURCE_PROOF.md) for the producer/consumer contract.

## Source composition

The shared `../build_overlay.py` preserves each task's full `kernel_src` under a
stable per-task directory. Rebind entries are re-exported from that directory,
which preserves their original `__file__`-relative companion lookups. Declared
`candidate_bind.requires_env` paths and scenario `binding_options` supply source
root hooks. No task-name cases are encoded in the builder.

Local binding checks run in a preserved mirror, so lazy imports/native builds
cannot contaminate the accepted source overlay. They check the actual rebound
callable or JIT object, not only the unchanged defining module path. They are
**local preflight checks**, not serving-rank proof.

Each candidate arm gets a real, independent copy and its own case registration.
Native builders that write beside `__file__` therefore get separate per-arm roots.
No symlinks are used. Existing destinations and failed run/cache directories are
preserved; select fresh run tags instead of overwriting them. The source inventory
is frozen before imports. Later native build outputs do not become new input
source, but every declared prelaunch file must still exist with its recorded hash.

## Parameters

Existing parameters remain: `harness_dir`, `harness_cmd`, `case_file`, `case_name`,
`base_case`, `overlay_subdir`, `reference_run`, `reference_key`, `run_tag_prefix`,
`repeats`, `duration_s`, and optional `stop_cmd`/`port`.

- `reference_run` is the historical receipt used to report stock drift; the fresh
  stock control is the A/B denominator.
- `stock_run_tag` optionally overrides `<run_tag_prefix>_stock`.
- `overlay_build_command` is an optional argv prefix, defaulting to `["python3"]`,
  for running the shared builder in the pinned serving Python environment. The
  command receives the normal builder arguments without shell evaluation. The
  benchmark launcher may use a separate runtime adapter through `harness_cmd`.
- `cache_root` is required and must be declared in scenario config with appropriate
  node-local placement. Each arm gets independent Triton/XDG/Inductor/torch-extension
  directories. The reference harness retains its image-provided AITER cache policy.
- `binding_options.<task>.source_env` maps environment names to accepted-source
  path templates. `{kernel_src}` names the task's bundled `kernel_src`; `{task}`
  and legacy `<task>` name its accepted task bundle. Values must stay inside that
  bundle. Hooks are installed before module or rebind imports and remain relative
  to each arm's source copy. Conflicting global source-root names are rejected.
- `overlay_receipt` names the existing launcher receipt fields. These are checked
  as **source inventory**, never treated as runtime proof.
- `native_validation` names actual native validity/completion fields and duration
  evidence. Missing configured fields block admission; boolean values cannot stand
  in for numeric counters.
- `serving_source_proof` declares the prepared observer and required rank count.

For the current InferenceX-shaped reference harness, the validation settings are:

```json
{
  "overlay_receipt": {
    "file": "run_identity.json",
    "hashes_path": "server.python_overlay_files",
    "root_path": "server.python_overlay"
  },
  "native_validation": {
    "file": "aiperf_artifacts/profile_export_aiperf.json",
    "required_values": {
      "metadata.submission_valid": true,
      "was_cancelled": false,
      "error_summary": []
    },
    "aggregate_required_values": {
      "request_accounting.records_error_dropped": 0
    },
    "completion_receipt": {
      "file": "run_identity.json",
      "required_values": {"client_exit_code": 0}
    },
    "duration_coverage": {
      "path": "metadata.metric_duration_coverage",
      "phase": "profiling",
      "phase_field": "phase_name",
      "duration_field": "expected_duration_seconds",
      "threshold_field": "required_ratio",
      "ratio_fields": ["ttft_ratio", "inter_token_latency_ratio"]
    }
  },
  "serving_source_proof": {
    "file": "serving_source_proof.json",
    "world_size": 8,
    "producer_file": "configured path to the reviewed observer",
    "producer_sha256": "SHA-256 of that actual prepared file"
  }
}
```

The native producer's duration-coverage threshold is consumed from its receipt;
this slot does not invent a new tolerance. The target duration must equal the
configured duration, the native coverage gates must pass, and the aggregate must
cover at least that duration.

## Execution and outputs

`run.sh` validates configuration, builds/checks the accepted overlay, runs and
validates fresh stock, then registers/runs/validates each candidate arm. It stops
on the first invalid arm and retains its evidence. Completed arms may be reused
only after their retained native and source-proof records pass current checks.

Verify the runtime boundary before launch: environment variables set around a
host command do not automatically reach a container. A scenario's native launcher
must forward the proof directory and record the actual fresh cache placement.
When representative unmeasured requests are needed to exercise lazy paths, use a
bounded pre-measurement hook and identical conditioning for stock and candidates.
The hook must stop a failed primer or incomplete observer cleanup before the native
benchmark starts. Preserve the benchmark command and metric basis, and record the
extra conditioning separately from the historical reference.

`summarize.py` publishes a median only when every requested independent candidate arm is
complete and valid, every actual rank has source proof, the stock control is
valid, and metric bases agree. Valid slow repeats remain in the median. There is
no subset-median fallback. It records stock drift, TTFT/ITL context, all arms and
all invalid-arm reasons. The slot's unchanged `../finalize.py` owns the final A/B
verdict and `result.json` mapping.

Outputs include the accepted source overlay, preserved binding-check mirror,
`applied.json`, per-arm case registrations/validation records, `reference.json`,
`e2e/bench_summary.json`, and the slot's final artifacts. No E2E claim is valid
without the actual serving observer's evidence.
