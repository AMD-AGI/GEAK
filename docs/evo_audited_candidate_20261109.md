# EVO audited research candidate — 2026-11-09

Runtime commit `b8bd4323a6fec076cd37e146a552df17b2fc1c34` is an experimental
research candidate based on `6b898e4db3c0b73a6c2d3f65834b7a131a525519`.
Compared with upstream `evo` at
`9f925de2902815d3125f0a51fd4d0dc2d7116f01`, it includes both evidence repairs
and changes to search policy. Disabling optional flags does not restore the
upstream behavior. This candidate makes no claim of performance superiority.

For `optimizer: "evo"`, the default schedule is **`first_stall`**, whose
efficacy remains **UNPROVEN**. It follows greedy rounds until a round neither
improves nor advances, then starts an evolution archive from the incumbent.
`evolution_start: "search_start"` instead initializes the archive at search
start. That selector does not undo inherited changes to initial planner and
engineer context, direction guidance, below-floor qualification, or dedicated
deep-engineer scheduling.

The integrated components are:

| Commit | Scope |
| --- | --- |
| `524aae35afa713b1e256c37ba7b9b8a8001a6507` + `379c376535fb295f46bd323bee6522377455bd7a` | Materialize a hash-pinned empty parent without applying an empty patch; preserve identity checks. |
| `3293b9658283c0c3753d82d41fd727c0705f0457` | Preserve complete settled verifier reports across winner, delivery and checkpoint handoffs. |
| `0703d63813c6fa147d981ad988cc816b56df6b3c` | Associate native case rows with their measurement cohort; preserve unavailable secondary metrics as null; add optional case presentation. |
| `be6341cb1ee6a67fab67a0ae4157095073dc2b99` | Bind Setup to its assigned evaluation and workspace paths. |
| `84035d75c2bfa26b2a6a6e09394fafa4908ce03d` | Add an optional return of an unmeasured development attempt while EVO is active. |

The merged handoffs retain complete verifier reports alongside compact case
metadata. Missing candidate rows clear stale rows instead of inheriting another
program's measurements; the initial ORIGINAL retains its baseline exception.
The common role, schema and cohort changes from `0703d638` apply even when its
experimental selector is absent. Metadata remains author-reported unless the
existing host measurement reader independently validates it.

## Arguments and defaults

| Direct-lane argument | Default | Behavior when selected |
| --- | --- | --- |
| `optimizer` | `"greedy"` | `"evo"` selects the evolutionary policy. |
| `evolution_start` | `"first_stall"` when using EVO | Accepts `"first_stall"` or `"search_start"`. |
| `evolution_preserve_founder_attempts` | `false` | Requests correct slower founder source while EVO is active and no retained source exists. |
| `evolution_return_development` | `false` | Permits one concrete unmeasured attempt when the ordinary return lacks new measured development; ordinary verification and promotion remain required. |
| `evolution_case_evidence_view` | `false` | Exposes retained per-case evidence in additional decision inputs. |
| `evolution_evidence_handoff` | `false` | Adds selected parent and donor evidence to engineer inputs. |
| `case_measurement_view` | Absent | Accepts `"current_mixed"` or `"native_consistent"` only for ordinary workload-weighted tasks; rejected with `external_primary`. |

The four boolean feature flags require literal JSON `true`; the string `"true"`
does not enable them. Both case-measurement views require matching supplied
native and frozen-reference tables. Missing tables produce an explicitly
unavailable view without changing canonical scalar scores. No automatic source
import or global initializer is introduced.

## Validation and limits

Existing CPU checks passed for generated-workflow freshness, simple evolution,
first-stall parity, external-primary handling, and delivery/cutoff behavior.
They validate scripted behavior, not stochastic efficacy, GPU performance or
workload gains. First-stall and optional policy efficacy require experiments.

The generic E2E argument forwarding includes `evolution_start`, founder
preservation and the two evidence flags. It does **not** forward
`evolution_return_development` or `case_measurement_view`; direct-lane support
does not establish serving reachability. Existing per-call arguments can
override defaults, so inspect the actual child invocation. Learned knowledge
also retains its existing entrypoint-specific defaults: enabled directly and
disabled through the E2E lane wrapper, subject to isolation policy.
