# Standard recipe

Use the existing configured implementations and budgets:

`baseline → profile / UT → roofline → optimize → applyback / evaluate`

```bash
python3 workload/run.py --config "$CFG" --dry-run
python3 workload/run.py --config "$CFG" all
```

`CFG` names the scenario. The driver supplies each stage's inputs, parameters and output
location and runs its finalizer. Preserve the full profile delivery, kernel parallelism,
real credited search time, cold verification and E2E reservation. The selected skill's
own instructions still define the work; the root loop does not shorten its budget.

Resume with the same command: completed stages are revalidated before reuse. `status`
shows progress, `next` names the pending stage, and `run <stage>` runs one configured
stage. `prepare <stage>` prints its environment for manual work; `validate <stage>`
records its result. A blocked stage stops this recipe and returns its evidence to
reasoning. Retain prior receipts and repair the owner of the failed precondition.

`all` reports periodically by default. `--report-interval SEC`, `--no-report` and the
`report` config control it; per-task state is available through
`workload/engine/campaign/scheduler.py status --work-root <work-root>`.

An implementation is selected by `stages.<stage>.skill`; `runner` selects its agent CLI.
Keep credentials outside the repository through `runner_env.env_file`. The
[stage contract](../stages/CONTRACT.md#standard-recipe-adapter) and
[runner contract](../runners/CONTRACT.md) describe those interfaces.

Keep the validated optimized system as the floor when required, not merely the original
stock baseline. Retain its sources, runtime configuration and receipts. New candidates
replace it only after the declared comparison, including all required objectives, passes.
A known-good reference may be revalidated and reused; report that starting advantage and
any extra exploration budget when comparing search policies.

Self-check: `python3 workload/engine/tools/skill_manifest.py verify-all`, scenario dry-runs,
and CPU fixture calls through actual measurement hooks. Those establish compatibility;
controlled serving comparisons establish performance. Until the latter pass, the fixed
recipe remains the performance reference and the adaptive route remains experimental.
