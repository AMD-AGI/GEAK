# The GEAK-EVO engine

This package contains the evolutionary search and its native task adapter.
Importing it starts no work and changes no saved run.

## Current support

This is a code-review version. Source and CPU checks do not establish model,
GPU or full Hyperloom readiness. The host must supply and qualify its model,
workspace, evaluator and process-ownership services; the bootstrap wheel has
no standalone EVO runner.

| Path | Current support |
| --- | --- |
| [Native search and final measurements](../docs/evo/admitted-command.md) | Implemented with the existing owner and original budget. Numeric export supports native-v2 `task-report-v1` graph samples and phase cache receipts. |
| [Common optimizer dispatcher](../docs/evo/dispatch-adapter.md) | Native EVO adapter and owner-entry patch implemented; installation and live validation remain. |
| [Hyperloom-generated tasks](../docs/evo/evaluation-port.md#generated-paired-results) | Task reader, raw paired-report reader and result export implemented. The existing owner's runner/admission and live validation remain required. |
| [Archive export](../docs/evo/archive.md) | Exact sources, ideas, measurements and retained research statements in a deterministic folder view. Full runtime resume also needs the original store and pinned dependencies. |

## Code map

| Start here | What it does |
| --- | --- |
| [loop.py](loop.py) | Save a step, run it, and continue from saved state |
| [host.py](host.py) | Compose the search with existing admitted GEAK execution services |
| [command.py](command.py), [dispatch.py](dispatch.py) | Run an admitted task and connect it to the common dispatcher |
| [result.py](result.py) | Export the selected source and matching measurements |
| [planner/context.py](planner/context.py) | Give the planner the current ideas, code and evidence |
| [planner/native.py](planner/native.py) | Turn a planner reply into checked work |
| [engineer/assignment.py](engineer/assignment.py) | Prepare an engineer's source, budget and task |
| [engineer/investigation.py](engineer/investigation.py) | Run or recover its edit, test and repair work |
| [archive/store.py](archive/store.py) | Store exact code versions and append their history |
| [archive/records.py](archive/records.py) | Keep each idea's current state and earlier results |
| [evaluate/jobs.py](evaluate/jobs.py) | Load the required task and test contract |
| [evaluate/evaluator.py](evaluate/evaluator.py) | Check which source was tested and retain its results |
| [selection/final.py](selection/final.py) | Verify the chosen program against the saved record |

## Code, ideas and measurements

A source ID names exact files. A candidate adds its task, measurement context,
parents and donors. An idea can have several candidates, failed attempts and
unfinished work. A measurement keeps its source, test cases and raw evidence.

The [archive guide](../docs/evo/archive.md) describes what engineers save, how
the host indexes trials, and how to export a readable folder view.

Keeping a candidate in the archive does not claim it is faster. Choosing a
candidate does not claim a speedup either. The final workload measurement must
establish the gain against the correct baseline. Missing evidence stays missing.

## Runtime wiring

The [shared host](../docs/evo/shared-host.md) constructs the search using the
existing Arena native-v2 callbacks. It reuses the host's GPU ownership, evaluation
and process cleanup. The host must also verify the loaded runtime code and
permission to settle a call; these checks still need live qualification. See
the [planner notes](../docs/evo/planner-port.md).

`EvolutionLoop(store, campaign_id, ...)` reads an existing campaign. Its
constructor starts no work. Recovery uses the saved operation identity and
settlement record; it must not repeat a call that may already have started.

The host must pass [`program_lease_bound(policy)`](archive/policy.py) as `program_lease` when
constructing the loop. This retains the selected policy for continuing an
idea. The default is zero. The loop's prompts default to the packaged `prompts/`
folder; another explicitly selected version can be supplied with `prompt_dir`.
The native planner binds its own instructions separately.

[`PROMPT_SOURCES.json`](PROMPT_SOURCES.json) records the packaged prompt version
and each file's `current_sha256`. Its `source_ref` fields retain historical import
provenance and do not describe the revised bytes. Prompt changes apply to new
work; saved instruction references must keep their original bytes. The manifest
also records retired guidance. Native adapter instructions are versioned with
their source code separately.

Saved record formats remain readable. In-flight operations pin their evaluator,
backend and instructions and require the original runtime or a checked migration.
Older runtime references require their exact original bytes and a compatible
host; copying this package alone does not qualify a historical run.
Never rewrite those pins to force resume.

## Check a change

From the repository root:

```sh
PYTHONPATH="$PWD" python3 -m unittest discover -s tests/geak_evo -t tests/geak_evo
node --test tests/geak_evo/test_engineer_entry.mjs
```

These checks use CPU fixtures. Reference-gated compatibility checks can skip in
ordinary CI. A compatibility claim requires all three source references and zero
skips, as listed in the [cleanup validation notes](../docs/evo/cleanup.md#validation-and-remaining-work).
The [planner](../docs/evo/planner-port.md),
[engineer](../docs/evo/engineer-port.md) and
[evaluation](../docs/evo/evaluation-port.md) notes record source compatibility
and the remaining runtime work.
