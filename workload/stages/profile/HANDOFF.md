# Profile → UT handoff

`profile_handoff.json` uses `workload/kernel_profile_handoff/v1`. It is a reference
manifest over completed dispatch evidence, not an executable UT or a tensor archive.

| Field | Meaning |
| --- | --- |
| `profile_root` | Absolute location of the immutable evidence tree |
| `selection_params` | Top-K, strict overall GPU-time threshold, phases actually used |
| `files` | Relative structured-evidence paths with SHA-256 and byte size |
| `external_files` | Referenced census/probe/observation files outside that tree, with hashes |
| `observations` | Per-config raw scheduler-observation references |
| `points` | Measured point IDs, config, phase, achieved shape/context, represented instances, census and shape-manifest references |
| `kernels` | Every selected identity, operator/entry when located, points where it dispatched, source-map key and explicit gaps |
| `coverage` | Profile denominators, measured-point model summary and missing sampling mass; UT coverage is unset |
| `compile_evidence` | Whether compile-triggered call/variant evidence was recorded |

Required input files are `kernel_ranking.json`, `selection_targets.json`,
`sampling_points_final.json`, `evidence/source_map.json` and `evidence/adjacency.json`.
Every counted point needs a matching census and a shape manifest under `evidence/shapes/`.
Raw observations are required per config. Keep all unselected/unsupported/collective
work in the ranking denominator and inventory. File hashes are computed by the exporter,
not authored by the agent. Structured evidence includes compile/call descriptors when
present; raw traces and external tensor payloads remain referenced at their recorded
locations. Preserve those resources when a later source review/replay needs them.

`handoff.py export` checks the configured selection policy and measured denominators.
It indexes unresolved callable/source boundaries explicitly rather than fabricating a
test definition. Those gaps do not become UT successes: the UT stage must resolve them
from handed-off evidence and installed source, or report blockers. Missing census,
shape manifests, observations or inconsistent scope prevent exporting a complete handoff.

`ut/prepare.py` verifies hashes and creates a separate output directory. It references
heavy evidence and copies small top-level descriptors, never old task directories.
`profile_input.json` records the exact handoff hash. Re-running preparation is idempotent
for unchanged evidence; a different handoff cannot silently replace a UT run's inputs.
The UT finalizer verifies both original evidence and local descriptors again.

The profile stage's `ok` means dispatch evidence is ready for UT review. It does not
mean any UT exists. The UT stage's `ok` means the declared task inventory is accounted
for and all produced tasks passed the full task/source acceptance gates. Neither
status implies exact whole-serving coverage or an E2E improvement.

To reuse an older combined extraction, export its profile evidence using its recorded
selection parameters. The existing `tasks/` and receipts are excluded from the handoff;
UT generation can run independently without another GPU profile. To change selection,
rerun the CPU selector and export a new handoff in a new evidence output.
