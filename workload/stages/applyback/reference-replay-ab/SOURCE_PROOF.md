# Actual serving-source proof contract

This file specifies retained evidence consumed by `summarize.py`. It is not an
observer, a GPU verifier, or a recipe for replacing real observations with JSON.
The applyback agent prepares the campaign-specific observer and root/operator
reviews its exact bytes before a long replay is admitted.

## Bootstrap inputs

The configured `producer_file` is copied unchanged into the accepted source tree
as `_geak_serving_source_producer.py`. The builder checks `producer_sha256` and
syntax. The appended `sitecustomize` bootstrap executes it only when
`GEAK_SERVING_SOURCE_PROOF_RUN_DIR` is set. That environment variable names the
current arm's run directory and is supplied by the reference runner.

The producer finds `_geak_source_obligations.json` beside its own `__file__`. This
contains `world_size`, `proof_file`, and each applied target's callable,
`bundled_source_hashes`, entry path and native-proof requirement. Paths are relative
to this arm's real overlay root. The observer must resolve actual serving-worker
rank identity from the serving runtime; a parent PID, import helper or device
number alone is not a rank observation. Bootstrap helper descendants are guarded
against recursion; the campaign producer still owns correct worker-role detection.

The producer installs one-shot observers, records actual call/leaf/library sources
once their lazy paths execute, restores original callables/hooks before measured
traffic, and writes the configured proof file atomically in the run directory.
Temporary observation must not alter kernel math, dispatch or the timed workload.
The generic slot does not produce these observations for it.

## Required retained record

The proof JSON has:

- `contract: "serving_source_proof/v1"`;
- `run_tag`: the actual run-directory basename;
- `overlay_root`: the actual absolute root of this arm's copied overlay;
- `ranks`: exactly the configured rank set, numbered `0 .. world_size-1`.

Each rank row contains actual integer `rank` and positive integer `pid`,
`post_lazy_load: true`, `observers_removed: true`, and a `log_ref` with run-relative
`path` and SHA-256 of a nonempty raw worker log. These true fields describe measured
facts; they are not a waiver or user-configurable switch.

Each rank's `bindings` must cover every applied `target_callable` exactly once:

- `invocations`: a positive observed call count at the actual target/used alias;
- `resolved_file`: the actual callable's file, normalized relative to this arm's
  overlay, and present in the accepted source inventory;
- `leaf_source_files`: actual computation source paths, relative to this overlay;
- `source_hashes`: the observed accepted-source hash map matching the target's
  `bundled_source_hashes`;
- `native_libraries`, where required: the actually loaded library's absolute `path`,
  `sha256`, observed `namespace`, `loaded: true`, accepted native `source_hashes`, and
  `build_provenance_ref` pointing to its actual copied build-provenance record and
  SHA-256 in the run directory.

The observer must derive native source linkage from the loaded build's provenance,
not copy the expected map into a purported observation. Rank logs/build provenance
remain available for audit. Within an arm, ranks must agree on native namespace
and library hash. Across arms the library path or incidental relocated-build hash
may differ; neither is required to equal a standalone UT's former path/hash.

## Consumer gates

The existing summarizer checks complete ranks/targets, observed invocation counts,
worker-log hashes, source-file hashes against current bytes in that arm's real
source root, observer removal, native-library bytes/provenance, and within-arm
native build agreement. A proof naming another arm's root is invalid.

Launcher inventory is separate. `run_identity.json` copies the prelaunch inventory
from `case.json`. The consumer verifies that exact declared input map against the
untouched accepted source map and current per-arm bytes. It does not recursively
add later `run/native_build` or Python-cache outputs to the input map. Each arm is
copied from the untouched accepted source, so previous runtime outputs cannot be
inherited as new inputs.

These are evidence-consistency checks, not independent attestation that arbitrary
JSON is truthful. Missing or malformed proof blocks an E2E claim. Local import
checks, task UT receipts, bootstrap presence, and launcher hashes alone cannot
satisfy this contract. The agent must validate the actual campaign observer and
its restoration before scheduling measured serving work.
