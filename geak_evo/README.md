# The GEAK-EVO engine

This package contains the existing evolutionary search code, grouped by responsibility.
The Hyperloom adapter is a separate feature. Importing this package does not
start an engineer, reserve a GPU, or change a saved run.

| Start here | What it does |
| --- | --- |
| [loop.py](loop.py) | Save a step, run it, and continue from saved state |
| [host.py](host.py) | Compose the search with existing admitted GEAK execution services |
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

The host supplies model services, GPU ownership, workspace preparation,
evaluation and process cleanup. The package uses those existing services;
there is no second device manager or serving stack here.
The host must also verify the loaded runtime code and permission to settle a
call, as described in the [planner notes](../docs/evo/planner-port.md).
The [shared host](../docs/evo/shared-host.md) now constructs the cleaned native
search from those explicit services. It keeps the existing Arena native-v2
callback protocol; installation and live qualification remain separate.

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

The source move preserves the record formats. References to older runtime
files still need their exact original bytes and a compatible host; copying
only this package does not qualify a historical run for a new runtime.
Saved archives can be read across the two layouts. In-flight operation folders
also pin their evaluator, backend and instructions. Those folders must resume
with the original runtime or a separately checked migration. New code must not
rewrite their pins to make them pass.

## Check a change

From the repository root:

```sh
PYTHONPATH="$PWD" python3 -m unittest discover -s tests/geak_evo -t tests/geak_evo
node --test tests/geak_evo/test_engineer_entry.mjs
```

These checks use CPU fixtures. They do not establish model or GPU readiness.
Comparisons with the pre-port source are skipped unless `GEAK_EVO_ORIGINAL_SRC`
and `GEAK_EVO_OCT15_SOURCE` point to that saved source. The regular CI run does
not provide those paths; the cleanup review also ran with both references.
The [planner](../docs/evo/planner-port.md),
[engineer](../docs/evo/engineer-port.md) and
[evaluation](../docs/evo/evaluation-port.md) notes record source compatibility
and the remaining runtime work.
