# Opt-in evolution on GEAK main

This initial implementation adds local source retention and agent-selected parent
and donor references to GEAK main. It does not replace the workflow host, role
files, knowledge, skills, engineer allowances, objective, ORIGINAL measurement,
independent verifier, promotion gate, or final validation.

See [design diagrams](DESIGN.md) for the implemented single-lane comparison and
the **proposed, not implemented** persistent multi-lineage extension. Current EVO
uses one canonical incumbent with archive-backed source selection; it does not
yet maintain a population of independently developing incumbents.

Omitted, null, empty, and `greedy` optimizer settings use the existing main behavior.
Other supplied values must be strings; booleans, numbers, arrays and objects are
rejected consistently by the JavaScript entries and Python interface.
Only `optimizer: "evo"` enables archive hooks in the same main worker. The Python E2E interface keeps
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

The top-level direct lane, dispatcher and E2E all use the same main lane and
forward the explicit source list. Only select E2E tasks for which that source contract is valid; this
patch does not infer a safe editable scope from an arbitrary extraction. Normal
HL+GEAK arguments and learned-KB policy remain unchanged when EVO is off.

Custom callers keep using `kernel_workflow/kernel_lane.js` and its ordinary
`workflow_dir`. Enabling EVO does not add a workflow nesting level. This patch
does not raise main's nesting limit or silently migrate the host.

## Agent decisions and deterministic tools

There is one shared JavaScript workflow, not a generated EVO copy of GEAK main.
Small flag-gated hooks add archive initialization, parent materialization,
source-custody checks and retention at the existing phase boundaries. No
filesystem import, dynamic code evaluation or new workflow host is required.

The existing TechLead reads `evo/roles/planner.md` only for EVO planning. It
chooses investigations, parents and donors from evidence. Existing specialist
and deep engineers read `evo/roles/engineer.md` only for EVO engineering and
author the code themselves. These are addenda, not replacement roles or a new
population of managerial agents. Main's shared role files are unchanged.

`archive.py` is the deterministic Python tool implementation: it preserves
source bytes and provenance, queries records, prepares selected sources and
rejects out-of-scope patches. It does not choose parents, invent measurements,
write a crossover implementation or promote an unverified source. JavaScript
connects those operations to main's existing verification and promotion loop.

Today this CLI tool is exposed through ordinary Bash calls, not a dedicated
native or MCP tool registration. The workflow's helper-agent calls validate
returned JSON receipts; this agent-mediated bridge does not independently prove
that a command ran or its reported output is genuine. Real tool-call records,
source-bound artifacts and independent final measurements remain necessary.
A direct typed-tool adapter is a possible separate improvement if both existing
hosts can support it without changing ordinary main runs. Rewriting the workflow
in Python or introducing another agent framework would not itself make decisions
more agentic or improve that evidence boundary.

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

## Validation

No generated worker or build step is needed. CPU dispatch tests compare ordinary
main behavior against `a48ac8bef9aeab299ba5ac9cb44164d997e94524`, including rendered
prompts, schemas, calls, failures and terminal results. EVO uses the same lane
and host-selection code with the flag enabled. Real-Git witnesses separately
exercise archive/source custody and the existing workspace materializer.

```bash
python3 -m unittest discover -s evo/tests -v
```

CPU dispatch mocks and real-Git custody witnesses establish only their tested
contracts. Actual native-host/tool conformance and complete GPU correctness and
delivery must be recorded separately before recommending this path. This patch
does not establish speedup or universal superiority over main. Migration,
question-linked evidence, learned skills and scheduling overlap remain separate
feature-driven experiments; historical broad-branch results do not qualify this
new implementation.
