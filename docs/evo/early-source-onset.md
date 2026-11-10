# Retained-source handoff onset

`evolution_visibility` is an opt-in experiment for the model handoff and selection
of retained sources. Both settings create the same archive before search and use
the same source capture, preservation, verification, integration and delivery
paths. The setting changes when planning and engineering receive the explicit
retained-approach policy and archive handoff.

Use the common configuration below, changing only `evolution_visibility` between
`"first_source"` and `"first_stall"`:

```json
{
  "optimizer": "evo",
  "evolution_start": "search_start",
  "evolution_preserve_founder_attempts": true,
  "evolution_return_development": true,
  "evolution_visibility": "first_source"
}
```

The lane rejects another optimizer or start schedule, preservation settings other
than boolean `true`, and unknown visibility values before Setup. Omit
`evolution_visibility` to retain the existing behavior. This option does not set
engineering time, credits, stall limits, deadlines or concurrency; bind those
identically for a comparison. Source capture consumes the ordinary run allowance.

| Setting | First eligible planning boundary |
| --- | --- |
| `first_source` | The first round that begins with retained source. |
| `first_stall` | The first round that begins with retained source after a completed round neither improved nor advanced the search. |

The handoff state is fixed for the entire round, including forced replans,
engineers that start later than their peers, and the memory call. A source captured
during the round can enter handoff at the next planning boundary. A sourceless
stall is remembered even if a later round improves before source becomes
available. No stall counter is reset, and a stop limit or cutoff can leave no
eligible later round.

Before handoff opens, the planner uses its ordinary schema and inputs. Explicit
parent and donor requests in an out-of-schema response are ignored; workspace
preparation inherits the archive's verified incumbent, or ORIGINAL when there is
none. Once handoff opens, retained parents and donors use the existing selection
and source-identity checks. Both settings keep the same full source-preservation
and development-fallback instructions before and after handoff.

This is a model handoff and selection contrast, not filesystem access isolation.
Ordinary history, raw archive references, source custody and same-user filesystem
access remain available in both settings. Initial model-input parity is expected
when the common archive has no source before round one; imported peer source can
make `first_source` eligible immediately. `search_start` itself is not a claim of
input parity with the original greedy schedule.

The existing source predicate requires a retained patch reference without a
retention error. Archive creation or failed retention alone does not qualify. A
valid zero-byte patch still represents the unchanged archive base and satisfies
that existing predicate; it does not establish development or reuse of an
alternative. The host logs `[evolution_handoff]` records with the round, source
availability, visible state, first stalled round and onset round. The assigned
setting is omitted from runtime handoff diagnostics. The final lane result returns
`evolution_handoff` with the setting and onset, outside the model-facing history.
This does not hide configuration files or other same-user artifacts: an experiment
must record their actual accessibility rather than infer treatment blinding from
prompt parity.

CPU regressions cover initial prompt/schema/source/capture parity, onset timing,
staggered peer dispatch, unresolved hidden references, stop limits, failed and
zero-byte capture, and delivery while handoff stays closed. They establish
implementation behavior. Compare complete delivered fitness and cost to assess
the feature; earlier handoff, source-backed descendants and time to real reuse
are fidelity observations, not evidence of a performance benefit by themselves.
