<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Accepted upstream source

GEAK's baseline is the upstream orchestrator's latest accepted configuration.
It includes cumulative flags, environment changes, removals and supported
source changes. GEAK measures its incremental improvement against that state.
The orchestrator may separately report total improvement against the session's
original baseline. These denominators have different meanings: an original
100 tok/s, accepted framework configuration of 120 tok/s and GEAK final of
126 tok/s give a 5% GEAK increment and a 26% total improvement.

Sparse source directories on `PYTHONPATH` do not establish this contract.
Python may select an installed regular package instead, and an absent file in
a sparse overlay cannot remove an installed module. A source-bearing handoff
therefore needs the versioned `baseline_env_spec.source_materialization`
descriptor. Missing or unsupported source is an explicit
`unresolved_baseline_source` error; it is never silently treated as a stock
baseline.

## Materialization contract

```json
{
  "schema_version": 1,
  "status": "ready",
  "bundle_root": "/absolute/accepted-source",
  "manifest_path": "manifest.json",
  "manifest_sha256": "<64 lowercase hexadecimal characters>",
  "required_layer_ids": ["accepted-layer-1", "accepted-layer-2"],
  "pythonpath_prefixes": ["trees/framework/python"]
}
```

The manifest names the exact accepted Git trees, ordered accepted layers,
complete file hashes and modes, import roots, affected modules and deletions.
`ready` proves the source bundle is complete and valid. It does not prove a
serving process loaded that source. The producer must account for changes
between accepted commits and for source artifacts outside declared roots.

Version 1 supports regular pure-Python packages with unambiguous import
ownership. Unsupported compiled artifacts, namespace/package deletion,
startup hooks, unresolved layer coverage and unversioned files under import
roots are refused. Runtime/interpreter dependencies outside those trees are
not certified by a source manifest.

The interface stages complete copies under
`EVAL_DIR/source_materializations/<manifest_sha256>/`, content-addressed
requests under `EVAL_DIR/source_requests/`, and exact canonical helpers beside
`bench_e2e.sh`. An immutable `source_manifest.sha256` marker prevents that
benchmark from silently running without the staged source request. Staging
does not replace conflicting helpers, bundles or old measurement evidence.
A moved run can stage a new location-bound request while preserving its old
request bytes.

## Launch and measurement

The serving import order is verification bootstrap, authored overlay,
accepted upstream source, backend defaults, then inherited paths. The
authored overlay and accepted upstream source remain separate. Both reference
and candidate use the same upstream source; the candidate can add its own
authored overlay.

Linux serving Python processes register their own identities and answer
credential-bound local challenges. The runtime checks source hashes, module
origins, deletion resolution, absence of source bytecode, and the frozen
overlay inventory. Receipts distinguish modules actually loaded from modules
whose resolution was checked. Merely resolving a changed module is not proof
that it executed.

Preparation checks run after server health. The `ready` gate runs after
configured warmup and before timed hot rounds; `finished` runs after those
rounds and before the summary. Python worker creation is refused after ready.
Persistent unobserved workers, including shell wrappers, are unsupported.
The initial contract covers observed persistent processes and guarded Python
imports. Native process creation bypassing Python's audit mechanism and
arbitrary dynamic execution are outside that contract. Backend, interpreter,
entrypoint, model and workload identity still require their own checks.

`source_runtime/measurement.json` binds the request, both gate artifacts,
the exact launch capsule, raw runs and exact summary bytes. Gates and process
receipts also bind that capsule. Its frozen overlay inventory includes startup
hooks, standalone Python helpers and the overlay manifest; the verifier checks
their current content, so retaining the same directory name is insufficient.
Isolated-server aggregates must account for
every selected replica and recompute the reported median from those selected
measurements. The seal covers hot timed rounds. It does not certify cold
diagnostics or accuracy evaluation.

Persisted acceptance additionally requires confirmed teardown from the same
launch. The source teardown pins process identities, retains the owner while
cleaning descendants, and confirms their exits. Unproved cleanup leaves a
shared request-directory barrier, removes summary eligibility and returns 43;
another output directory cannot bypass it. A seal written before a hard kill
does not establish completed cleanup. Native post-measure callbacks, when
composed with this lifecycle, run before teardown; persisted verification runs
after the benchmark returns.

## Results and replay

Normalization verifies the selected Setup baseline and the fresh Director
reference/final pair. The initial supported provenance is `workflow_return`
or `disk_director_validation` with an actual
`validation/base/bench_summary.json`. Other recovery paths remain unavailable
until their exact selected measurements can be bound; normalization does not
search for a favorable matching throughput.

When source proof or replay staging is unavailable, the result is an error.
The original measurements and configuration remain under
`unverified_source_result`. They are diagnostics, not eligible gains. The
final report marks that refusal, and the kernel journey does not publish
acceptance from those claims. A verified final pair also does not establish
each per-kernel attribution; version 1 reports that attribution as unavailable.

A verified positive result uses an immutable replay wrapper. It exports the
exact staged request and accepted import roots, clears prior observation
context, and forwards all arguments to the original final launcher. The
original launcher and measured artifacts remain unchanged. Replay receives
fresh process observations through the canonical benchmark.

Source-bound global knowledge exports are withheld until their individual
measurement provenance is supported. Run-local measurements, failures and
checkpoints remain available. Existing handoffs without source changes keep
their existing launch and result behavior.
