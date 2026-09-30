# Bounded scheduling through the Hyperloom interface

The normal `interface/run_e2e.py` entry accepts explicit scheduling controls
without changing the warm-server benchmark protocol. The caller's actual
`time_budget_s` remains the hard envelope. Neither a larger role timeout nor a
reserve extends it.

An explicit `final_reserve_s` / `GEAK_FINAL_RESERVE_S` is honored in full. It must
be positive and smaller than the actual workflow budget. The old default remains
60 minutes capped at 20% when no reserve is supplied. For AgentX with the standard
3,600-second full passes and 900-second search passes, Finalize plus independent
Validate requires 16,200 seconds of client time alone. Setup takes 7,200 seconds
and one candidate's integration A/B takes 3,600 seconds. Server starts, internal
client warmups, profiling, optimization, correctness and reports take additional
time. Explicit AgentX schedules that cannot fund these lower bounds fail with
`error_class="invalid_schedule"` before launch preparation.

`search_deadline_s` is the elapsed time after workflow start when no new head,
milestone or deep wave may be dispatched. It must be inside the optimization
window before the final reserve. Existing work receives only the remaining
optimization time: nested lanes stop starting agents when it expires, and the
standalone runtime applies the per-agent timeout across queueing and schema
retries. Parent and nested clocks each retain a rounding margin. Explicit role
timeouts reach nested lanes too. On POSIX, an agent owns a separate process group;
deadline cleanup kills that group and waits for child closure/reaping before
returning. Runtime shutdown also cleans its owned agent groups.
Final agents remain bounded by actual time left. The existing outer
process/container supervisor still owns the hard stop and cleanup; a timeout is
an incomplete result, never evidence of a measured win.

Handoff keys take precedence over environment values, including explicit zero:

| Handoff key | Environment default | Meaning |
|---|---|---|
| `budget` | `GEAK_KERNEL_TASK_BUDGET` | Maximum editable-kernel tasks; zero disables them |
| `min_kernel_tasks` | `GEAK_MIN_KERNEL_TASKS` | Existing task floor, capped by `budget` |
| `kernel_budget` | `GEAK_KERNEL_ROUND_BUDGET` | Positive nested optimization-direction budget |
| `head_budget` | `GEAK_HEAD_BUDGET` | Maximum head operations; zero disables them |
| `head_author_max` | `GEAK_HEAD_AUTHOR_MAX` | Maximum author directions per head |
| `head_author_tries` | `GEAK_HEAD_AUTHOR_TRIES` | Positive author attempt limit |
| `head_corrective_max` | `GEAK_HEAD_CORRECTIVE_MAX` | Corrective re-author limit; zero disables it |
| `ab_finish_retries` | `GEAK_AB_FINISH_RETRIES` | Retries to finish the same incomplete A/B |
| `baseline_extract_retries` | `GEAK_BASELINE_EXTRACT_RETRIES` | Retries to obtain the frozen isolated baseline |
| `search_deadline_s` | `GEAK_SEARCH_DEADLINE_S` | Stop dispatching new candidates at this elapsed time |
| `tuning_skillset` | `GEAK_TUNING_SKILLSET` | Enable/disable the independent tuning track |
| `use_learned_kb` | `GEAK_USE_LEARNED_KB` | Enable/disable learned kernel references |
| `fast_mode`, `deep_mode` | `GEAK_FAST_MODE`, `GEAK_DEEP_MODE` | Existing workflow mode switches |
| `warm_start` | `GEAK_WARM_START` | Existing `on`, `off`, `reference`, or `return_after_read` mode |
| `final_reserve_s` | `GEAK_FINAL_RESERVE_S` | Final-stage time within the actual budget |
| `agent_timeout_ms` | `GEAK_AGENT_TIMEOUT_MS` | Finite role timeout, also bounded by stage time left |
| `time_tail_cap_s` | `GEAK_TIME_TAIL_CAP_S` | Existing default dispatch-tail calculation |

For an **at-most-one editable-kernel task**, explicitly request `budget=1`,
`min_kernel_tasks=1`, `kernel_budget=1`, `head_budget=0`,
`head_corrective_max=0`, `ab_finish_retries=0`, `tuning_skillset=false`, and
`warm_start=off`. This supplies one opportunity, not a promise of a candidate or
gain. A head-operation limit alone does not cap its authored/bakeoff candidates.
Warm start and standalone tuning can otherwise introduce their own measurements.

A reviewed 12-hour **actual delegated GEAK budget** can, for example, request a
six-hour final reserve, six-hour role timeout, and a six-hour dispatch deadline:

```sh
GEAK_FINAL_RESERVE_S=21600 GEAK_AGENT_TIMEOUT_MS=21600000 \
GEAK_SEARCH_DEADLINE_S=21600 \
GEAK_KERNEL_TASK_BUDGET=1 GEAK_MIN_KERNEL_TASKS=1 \
GEAK_KERNEL_ROUND_BUDGET=1 GEAK_HEAD_BUDGET=0 \
GEAK_HEAD_CORRECTIVE_MAX=0 GEAK_AB_FINISH_RETRIES=0 \
GEAK_TUNING_SKILLSET=false GEAK_WARM_START=off \
python interface/run_e2e.py handoff.json result.json --timeout-s 43200 --dry-run
```

Inspect `mapped_args.schedule_validation` and the actual mapped controls before
running. This is a duration admission check, not a GPU readiness test. Normal
Hyperloom may delegate less than an environment timeout because of its session,
kernel-phase, rebench and closing reserves; use that real delegated value.
Hyperloom's normal direct subprocess route inherits these environment controls.
A separate transport with an environment allowlist must explicitly carry them.

For A/B comparisons, retain both requested and effective settings. Older control
versions may ignore the new limits and cap an explicit reserve at 20%. Supplying
identical environment variables does not establish identical effective search
policy. Leave optional task/mode switches unset if the experiment intends to
retain the historical track selection; otherwise describe the whole revised PR,
including its effective scheduling changes, as the treatment. Never modify a
frozen control to conceal the difference. Full scored AgentX and GSM8K evidence
and independent same-hardware retests remain required for a performance claim.
