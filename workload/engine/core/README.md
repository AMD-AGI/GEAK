# Sequential driver

`workload/run.py` executes the config's `order`, revalidates retained results on resume
and runs each slot's finalizer. It is an executor, not the
[Reasoning controller](../../reasoning/SKILL.md). Its default route combines Profile
with UT and Apply back with Evaluate.

## Commands

```bash
python3 workload/run.py --config "$CFG" --dry-run
python3 workload/run.py --config "$CFG" all
```

An implementation is selected by `stages.<stage>.skill`; `runner` selects its agent CLI.
Keep credentials outside the repository through `runner_env.env_file`; see the
[runner contract](../../runners/CONTRACT.md).
Use `status` / `next` to inspect progress, `run <stage>` for one stage,
`prepare <stage>` for its environment and `validate <stage>` to record its result.
A blocked stage stops the order and returns evidence for controller diagnosis.
The driver may open one repair session for a blocked agent delivery, preserving the
original diagnosis if that session returns no valid result.

`all` starts `watch.py` by default; `--report-interval SEC`, `--no-report` and the
`report` config control its file reports. This monitor does not schedule Reasoning
or send messages to the user's conversation.

## Calling convention

A configured implementation has `stages/<stage>/<skill>/SKILL.md`.
The driver supplies its context through these environment variables:

| Variable | Meaning |
| --- | --- |
| `GEAK_ROOT`, `WORKLOAD_ROOT` | Repository and workload roots |
| `SKILL_ROOT` | Selected implementation directory; read-only |
| `STAGE`, `STAGE_SKILL` | Selected stage and implementation |
| `RUN_ROOT` | Run directory |
| `IN` | Previous entry in `order`'s output; empty for the first |
| `OUT` | This stage's artifact directory, `$RUN_ROOT/<stage>/out` |
| `CONFIG`, `PARAMS` | Merged config and stage parameters as read-only JSON |
| `RUNNER` | Agent CLI adapter |

Write artifacts to `OUT`. Parameters override implementation defaults.
Resolve other upstream inputs explicitly through `RUN_ROOT`, not by splicing paths
onto `IN`. Direct module calls supply the environment required by any tool they reuse,
without adopting the driver's entire sequence.

## Execution shapes

With `run.sh`, the driver runs `bash $SKILL_ROOT/run.sh`.
With no script, or `"mode": "agent"`, it supplies the skill, slot README and environment
to an agent via the runner. `core/prompt/` contains the unattended session instructions.
Both shapes have the same acceptance criterion.

Every slot used by this driver must provide `stages/<stage>/finalize.py`.
It runs unconditionally after execution, archives the implementation result and writes
the slot's verdict. A dry-run rejects a missing finalizer. Scoped modules such as Setup
or Report need no finalizer unless being adapted for this full-stage driver.

## Finishing: `$OUT/result.json`

```json
{
  "stage": "<stage>",
  "skill": "<implementation>",
  "status": "ok",
  "artifacts": {"report": "REPORT.md"},
  "metrics": {},
  "notes": ""
}
```

- `ok`: the configured delivery is complete and admissible; not an optimization win.
- `blocked`: a prerequisite is missing; name it and how to resume.
- `fail`: the work was attempted and did not succeed.

Artifact paths are relative to `OUT` and must exist. Use
`stages/_shared/emit_result.py` to write the envelope.
The slot's `finalize.py` owns acceptance and checks that configured policy took effect;
native tests own their numerical verdicts. The criterion is written once, in its owner.
`REPORT.md` is the stage's explanation; the run renderer includes accepted stage
reports and preserves driver and native-summary rejections.

The existing mappings live with [Baseline](../../stages/baseline/README.md),
[Profile/UT](../../stages/profile/README.md#driver-delivery),
[Roofline](../../stages/roofline/README.md), [Optimize](../../stages/optimize/README.md)
and [Apply back/Evaluate](../../stages/applyback/README.md). A replacement must provide
the same declared artifacts to use those slots.

## Identity and resume

Before invocation the driver records `inputs.json`: selected skill, parameters,
target and hashes of prior result receipts. Resume validates these before skipping work.
Changed inputs require a fresh delivery or the original inputs. Older standalone results
without this receipt get envelope validation only.
A fresh invocation retains prior declared file artifacts in `.previous_runs/` and
continues session numbering; an empty retry cannot inherit an earlier success.
Working directories remain available for continuation; their owner separates native
cache/run identities when making new measurements.

Record observed stack identity in `result.json.stack` or `stack_identity.json`.
`target.py` compares it with the configured target; contradictions reject the delivery
and missing observations remain unverified. Its filename/run-name gate is a driver
labeling policy. A deliberate image-change experiment needs separately declared
reference and candidate identities; one driver's global target must not mislabel both.
