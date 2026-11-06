# Profile and UT calls

Use the [invocation contract](../CONTRACT.md#direct-module-calls) and the relevant parts
of [kernel-extract](kernel-extract/SKILL.md), or the profiler named in the plan.

**Profile:** answer the stated bottleneck question. Collect the relevant serving trace,
CPU/device activity, queue, cache or communication evidence. Return the observations,
workload identity and uncertainty in `REPORT.md`; a diagnostic call need not extract UTs
or claim that the full profile stage has completed. Use roofline when a bound helps.

**UT:** use that evidence to declare the computation boundary and representative cases.
Run the extraction and delivery tools for the planned scope; qualify the frozen tasks
with their own native tests and retain their receipts. A changed boundary or harness
needs a separate qualified delivery; it does not inherit earlier correctness evidence.

`run.py run profile` still means the complete configured profile/UT delivery and its
existing finalizer. A scoped observation is not submitted to that full-delivery gate as
if the remaining tasks had been produced. Preserve the original census and deferred work.
