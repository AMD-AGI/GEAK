# Opt-in evolution on GEAK main

This initial implementation adds local source retention and agent-selected parent
and donor references to GEAK main. It does not replace the workflow host, role
files, knowledge, skills, engineer allowances, objective, ORIGINAL measurement,
independent verifier, promotion gate, or final validation.

Omitted, null, empty, and `greedy` optimizer settings use the existing main worker.
Other supplied values must be strings; booleans, numbers, arrays and objects are
rejected consistently by the JavaScript entries and Python interface.
Only `optimizer: "evo"` selects the new worker. The Python E2E interface keeps
main's existing Claude/SDK versus optional Node host selection; selecting EVO
does not itself require Node. Python 3 and Git are needed by the EVO archive
helper, invoked through the existing agent/Bash tools. Helper-agent calls consume
time and tokens and must be included in resource comparisons.

## Invocation

Use the existing `kernel_workflow/kernel_workflow.js` entry point, with the same
main arguments plus:

```json
{
  "optimizer": "evo",
  "evolution_source_paths": ["kernel_src/kernel.py"]
}
```

Paths are relative to the canonical task workspace, not the GEAK repository.
The task owner must declare the editable regular files, including any anticipated
new files. Do not include tests, the oracle, workload weights or benchmarking
code. Directory, symlink, submodule and out-of-scope patch inputs are rejected.
The first implementation supports optimize/author lanes, not multi-language
bakeoff. An explicit custom `kernel_lane_script` cannot be combined with EVO.

The top-level direct lane entry point also dispatches on the flag. E2E selects the EVO lane
directly, avoiding an additional nested Workflow level, and forwards the explicit
source list. Only select E2E tasks for which that source contract is valid; this
patch does not infer a safe editable scope from an arbitrary extraction. Normal
HL+GEAK arguments and learned-KB policy remain unchanged when EVO is off.

Custom callers that already nest the legacy lane must select
`evo/kernel_lane.js` directly when enabling EVO, retaining `workflow_dir` as the
original `kernel_workflow` directory. Calling the legacy trampoline from an
already-nested workflow would exceed main's one-child-level limit. This patch
does not raise that limit or silently migrate the host.

## Retention and reuse

An archive is initialized in `eval_dir/evolution_archive` after preparation and
warm-start. `FOUNDER` names that source; it is not necessarily ORIGINAL.
`INCUMBENT` keeps the normal canonical starting source. Planners can query the
archive, select a previous entry as `parent_id`, and name `donor_ids` in a new
direction. Donors are source references, not automatic code combinations or
evidence of a semantic contribution.

Each engineer workspace is materialized by main's existing script. EVO prepares
its declared-source Git index before overlaying the selected parent, so candidate
diffs still apply to the current round's canonical source. The engineer must not
proceed if materialization omitted or changed any declared source: the helper
compares exact bytes, modes and absence against the round canonical first. Files
under main's excluded paths, such as `build/`, therefore fail closed. It must not
reinitialize or commit a new baseline. The immutable task/oracle is not replaced.
Source custody is checked before the ordinary verifier. A custody failure skips
that candidate rather than weakening verification or substituting a fallback
parent. Other work and canonical final validation can still finish.

Before promotion or cleanup, the archive separately retains best-patch bytes and
unfinished workspace bytes. Missing patches are explicit diagnostics, not a
reason to discard all development. Engineer/verifier reports are retained with
provenance; the archive is not an evaluator. A best-patch report never qualifies
different unfinished bytes. Archive failures are also visible in the workflow's
`evolution.errors` result. Original logs and full final endpoints remain required.

## Build and validation

`evo/kernel_lane.js` is generated only into this new directory from the reviewed
main lane at `a48ac8bef9aeab299ba5ac9cb44164d997e94524`. Its small substitution list
and extension are in `build_workflow.py` and `extension.js`. The build checks the
entire main source digest after removing the explicit opt-in dispatch hook; a
main update requires review and repinning, not a silent regenerated fork.

```bash
python3 evo/build_workflow.py
python3 evo/build_workflow.py --check
python3 -m unittest discover -s evo/tests -v
```

CPU dispatch mocks and real-Git custody witnesses establish only their tested
contracts. Actual native-host/tool conformance and complete GPU correctness and
delivery must be recorded separately before recommending this path. This patch
does not establish speedup or universal superiority over main. Migration,
question-linked evidence, learned skills and scheduling overlap remain separate
feature-driven experiments; historical broad-branch results do not qualify this
new implementation.
