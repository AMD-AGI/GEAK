# External UT in the existing native provider

`external-owner.patch` adds the supported GLM UT to the existing native helper.
It changes three source files; `external-owner-base.json` identifies their exact
pre-patch bytes. This is a source integration, not an installed or GPU-qualified
provider. It creates no allocator, store, service or recovery loop.

The callback remains `NativeCallbacks.launch/settle`. The existing root helper
owns source admission, study and evaluation clocks, device allocation, cutoff,
cleanup, costs and the `geak-arena-settlement-v1` result. Its source is unchanged.

## Prepare a new installation

The operational owner applies the patch to a new copy of the pinned provider,
checks its source hashes, and seals that installation through the existing
admission path. Do not apply it to an active study.

The new copy needs these additional inputs:

- `task/`: the exact ORIGINAL-only task packet. Each file in
  `contract.artifacts` must match its digest. Do not include copied optimized
  programs, prior histories, results or researcher evidence.
- `contract.json`: the validated external task contract, identical to the
  evaluator's contract. The sealed request/artifact/source-object roots must
  match the existing evaluator and store.
- `cleaned/geak_evo/`: the Python package from the selected pinned GEAK `evo`
  descendant. Include its complete Python source closure in `assets_sha256`;
  imports must not resolve to a writable checkout or candidate directory.

Keep the original `root_helper.py`, `adapter.py`, `gpu_lock.sh` and owner code
pinned in the admission. Set the image and device admission for the chosen
task; the prepared GLM packets use the immutable ac30 SGLang image and gfx950.
Provide enough existing session/evaluation allowance for all three UT
processes, first-use compilation and cleanup. The task's combined UT timeout
does not extend the owner's deadline.

The external path copies only the declared task files, then overlays exact
candidate source objects. It mounts no legacy fixture or shared setup tree.
Inside the owned container it keeps the verified singleton device mapping,
read-only source mounts and existing `gpu_lock.sh` busy check. It creates empty
private Triton, Torch, XDG and both AITER cache directories under `/results/cache`.
These are the supported cache settings already exercised by the fixed GLM
replay; they do not change the task or installed packages.

## Results and the GEAK control

The runner writes `external/START.json` and three process directories containing
`RECEIPT.json`, `ut_result.json` and `_runs_v2.jsonl`. The collector retains these
as `observer/...`, including partial failures. It reparses the raw files with
`read_external_report`; `OBSERVATIONS.json` is only a saved summary. Successful
raw reports alone do not establish owner execution, runtime identity or cleanup.

GEAK's existing measurement command must return the real Arena `candidate_id`
and `evaluation_id`. The latter is the request's `operation_id`; the former is
`request.binding.candidate_id`. These identify the same store and evaluator
that created the request. They are not values guessed from a workspace name or
from a model's summary. The control reader can then call
`result.read_evaluation(evaluator, candidate_id, evaluation_id=...)` and obtain
the checked native primary-phase score and gain gate. No extra lifecycle call
is needed to read a settled evaluation.

Before search, the owner must qualify ORIGINAL on the actual node/image through
this installed route. Confirm three fresh process records, unchanged UT/source
bytes, native phase folds, positive ordinary owner settlement and saved raw
artifacts. ORIGINAL may pass correctness and timing while its native gain gate
is false; that is expected. Any proposed improvement still needs that gain gate.

CPU checks applied the patch to all pinned bases, prepared exact candidate and
ORIGINAL bytes, checked the reduced mounts, and ran the real observer subprocess
on complete and incomplete synthetic reports. They verified that a forged
summary score is ignored and raw success cannot replace missing owner evidence.
They did not launch a container, task kernel or model call.
