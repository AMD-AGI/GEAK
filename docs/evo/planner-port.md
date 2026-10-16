# Planner port from the frozen October 15 source

The initial port preserved search behavior with the API/defect changes and
host provenance prerequisite described below. The October 16 cleanup additionally
retires unused transport and settlement helpers, listed below. It does not change
parent selection, proposal admission, archive policy, prompt content, budgets, or
the treatment of an
unknown model outcome. No real model, GPU, or campaign process was invoked while
porting it.

## Source and interfaces

The reference is the verified `oct15_source` copy and its `SOURCE_PINS.json`.
The canonical package source is `evolution/src/geak_evo`, supplemented by the
recorded bridge overlays.

| Original source | Production module | Main interfaces |
| --- | --- | --- |
| `evolution/src/geak_evo/cli_planner.py` | `geak_evo.planner.context` | `current_claim_context`, `outcome_prompt_context`, exact freeze/navigation builder (internally named `CLIPlanner`) |
| `bridge/native_planner.py` | `geak_evo.planner.native` | `NativePlanner`, `NativePlannerPending`, model identity/configuration/observation checks, proposal admission and recovery |
| `bridge/planner_model_backend.py` | `geak_evo.planner.model` | `run`, request/route validation, `remaining_budget`, `NativeRoleEnvelope` |
| `bridge/failed_native_planner.py` | `geak_evo.planner.failure` | Failed-turn artifact/receipt validation |
| Clock access in those files | `geak_evo.clock` | `SystemClock.time()`, `monotonic()`, `monotonic_ns()`, `boot_id()` |

The active native planner parameters and defaults remain. Optional keyword-only
inputs make runtime services explicit:

- `NativePlanner(..., runtime=None, clock=None,
  program_lease_prompt_ref=None)` takes model services and an optional exact
  prompt reference. The default prompt is the packaged, source-pinned
  `prompts/program_lease_planner.md`.
- `ModelRuntime(qualified, transport, settings, clock=None, routes=None,
  events=None, boundary=None)` names the existing transport components directly.
  Its first three fields preserve the original `dependencies` tuple. Codex
  attribution and failed-turn verification additionally use the explicit route,
  event, and boundary components.
- Model/failure helper functions accept optional `runtime` or `clock` keywords
  where needed. Native calls without a supplied runtime still use the original
  backend `run(..., dependencies=...)` call shape.

`NativePlanner` now requires either a supplied `ModelRuntime` or the legitimate
three-service legacy `dependencies` tuple. Both constructor and model `run`
validate the required service interfaces before creating directories, attempt
files or launch intent. This deliberately repairs the port's missing-runtime
defect: previously construction succeeded, a later call persisted launch intent,
then missing services failed without a model invocation and left it pending.
This is a defect repair, not an original-source parity claim. Supplied valid
legacy tuples remain supported.

The unused, unpublished `model.main` CLI wrapper was removed. It could not obtain
the required runtime from a shell invocation. The supported API is library
`run(config, request, runtime=...)` or `run(..., dependencies=...)`; no loader or
replacement CLI was added.

When a clock is explicitly supplied to `NativePlanner`, it is also bound into
the model runtime. A supplied runtime is copied without changing its services;
the caller's original object is not mutated. An explicit clock with the legacy
dependency tuple uses those same services in `ModelRuntime`. This avoids mixing
the planner's test/deadline clock with the model boundary's default clock.

There is no runtime registry, source-path discovery, `sys.path` modification,
or import of `codex_native` in production planner code. The default clock keeps
the original boot identity and monotonic/wall-clock behavior. Tests can inject
fixed clocks without creating a new campaign clock.

## Dependencies and ownership

Context uses `archive.refs`, `archive.store`, and `archive.policy`. Native planning
uses the archive's cells/maturation modules, selection's work/advice/choices/
recommendation modules, and the common budget functions. Engineer assignment
and investigation imports are lazy. The transport receives concrete services
from the caller rather than loading a historical installation directory.

`engineer.backend` owns the preserved Claude command/event/route functions.
Model execution uses the explicit native runtime services above.

## Preserved behavior

- Prompt constants and assembled decision/freeze bytes remain source-equivalent.
  Large navigation fields are externalized in the original order, with every
  field and exact reference retained.
- Claim navigation keeps dependency ancestry, later corrections, exact candidate
  binding, deterministic ordering, and the distinction between interpretation
  and validated claims.
- Accepted, rejected, failed and unknown model calls remain distinct. Missing
  or incomplete output is never converted into a stop, winner, or redispatch.
- Model identity, exact actual configuration, receipts, prompt bytes, settlement
  and whole-operation costs remain checked. Unknown tokens or monetary costs
  remain unknown.
- The existing call directory, launch fence, completion witness and acceptance
  deadline prevent replay from creating a new invocation or renewed allowance.
- Generic failures remain actual `ValueError` instances, preserving saved
  `type(exc).__name__` fields. Existing named pending exceptions retain their
  `ValueError` ancestry while also inheriting `EvoError`.

The old import loader is not executed by this port. This is a changed trust
boundary, not equivalent enforcement:

- Before constructing a model runtime or passing legacy dependencies, the host
  must verify the exact admitted path and hash of every supplied executable
  service, its imported module identity, and the canonical, root-owned,
  nonwritable installation ancestry required by the original admission. For
  Codex this includes the transport, route, event and boundary components; the
  retained `codex_native_root` string alone proves none of these properties.

These host obligations are not qualified by the synthetic suites. Private
model-state permissions, current-user ownership, file content/reference checks,
and model/result attribution remain.
Legacy configuration and saved-record field names, including
`codex_native_root`, are retained even where runtime imports no longer use them.

The context module retains its small `_json` implementation because its Decimal
handling and unsupported-value error message are part of the original behavior.
The model/failure modules use the shared finite JSON/hash helpers where those
semantics match.

## October 16 prospective retirement

Repository and current-composition callers use `NativePlanner`. The unused
`CLIPlanner` transport lifecycle, its pending exception and `_object` helper
are retired; its exact `_freeze` evidence builder and context helpers remain.
The unused `ClaudePlannerRuntime` wrapper is also retired; command, route and
event functions remain available to the current runtime.

The orphan `settle_failed_planner`, `validate_failed_history`,
`historical_validator` and `read_root_authority` helper chain is retired.
Active native failed-call artifact, content-hash and receipt validation remains.
Old runs retain their original authenticated runtime; this release supplies no
historical settlement migration. Unused imports were removed from native
planning and artifact transport without changing record formats.

Current-context CPU coverage runs without the external snapshot. Historical
comparisons explicitly exclude retired CLI lifecycle bodies. Surviving helpers
covered by the original parity comparison remain checked, with its existing
exclusions. The initial validation counts below describe the port before these
intentional retirements.

## Initial port validation

The three focused suites compare the frozen source with the port using synthetic
inputs and fake transport/process interfaces:

```text
test_planner_context_port.py
test_planner_model_port.py
test_planner_failure_port.py
```

The integrated combined run passed **16 tests and 194 subtests**. It covers exact
claim/context/navigation/prompt/source-freeze bytes; proposal rejection and
unknown recovery; 20 model backend/lifecycle cases; authenticated failed and
incomplete Claude/Codex fixtures; mismatched or unavailable cleanup; budgets,
boot identity, native/CLI constructor defaults, shared injected-clock identity,
default Claude route validation without a runtime, private state and authority guards,
no-replay fences, and immutable result checks. Lifecycle fixture
comparisons allow only the relocated backend's own source reference and the
receipt hash derived from that reference to differ.
The added defect test proves missing and malformed services produce no planner
root, call directory or launch intent and never invoke the backend. Valid
construction parity uses the same legitimate synthetic legacy services on both
sides. The removed CLI is explicitly excluded from signature parity.

AST checks additionally compare 57 original algorithm bodies after normalizing
only import relocation, `require(..., ValueError)`, explicit clock calls and the
optional runtime argument. Constructors, route service injection and prompt
resource relocation are reviewed separately.

The initial run used actual dependency modules from the independently developed
engine worktree through a test-only package path. The final run used the combined
engine tree directly, without that fallback. Set `GEAK_EVO_OCT15_SOURCE` to run
the original source comparisons; no reference source is silently fabricated.

Independent review caught and corrected a missing optional constructor argument
list, the clock-propagation gap, missing generic ownership/privacy guards, and
the pure Claude validation fallback before delivery. The engineer owner reviewed
the planner and found no additional logic, cost or replay discrepancy. Reciprocal
review of engineer assignment, investigation and operation preparation found the
boot and publication guards preserved in the final source. Codex peer review is
not a claim of actual Claude researcher agreement.

Synthetic parity does not establish a successful live optimizer run. The real
transport adapters, current model service, process boundary and full Hyperloom
handoff still require the combined integration qualification.
