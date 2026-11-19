# Muse Spark: route guide

This is an English navigation summary of the retained historical analysis, not a
reconstructed sequence of independent GPU experiments. Each release can bundle changes.

| Branch / stage | Recorded changes | Transfer limit |
| --- | --- | --- |
| Day0 | Quark MXFP4 target, DFlash2, AITER attention and FlyDSL MoE; TP4/EP4 and TP8/EP8. | The existing capabilities are the reference, not a gain over an unknown earlier state. |
| High-concurrency cache | Enable a DRAM KV tier at selected C12/C16 points. | TP4 capacity benefits do not transfer automatically to TP8. |
| Corpus reset | Remove the 256k trace cap with the same image/recipe. | Re-establish the workload comparison basis before assigning a kernel effect. |
| 1020 | Reuse metadata, specialize attention, tune small-M GEMMs, fuse routing/norm, and shard large prefill work. | Scheduling/cache changes share the release; inspect actual path activation. |
| 1027 | Add larger small-step mono, with admission/graph, expert copies, quantized producers and communication changes. | Fusion coverage, memory and policy interact; the local mono pair is separate evidence. |
| 1106 / later replay | Balance unequal context work; expand supported attention/page/fusion routes and revise the workload grid. | Some output gains accompany P90 declines; there is no later public TP8/C8 point in the collected grid. |

The original detailed reports preserve intermediate versions, PRs, point tables, failures
and reversals. Archive links require the retained collection to be supplied as the local
view or through the task archive root; [raw catalogue](../raw/README.md).

- [1020.md](../raw/collections/agentx-mi355x-20261113/reports/muse-spark/1020.md)
- [1027.md](../raw/collections/agentx-mi355x-20261113/reports/muse-spark/1027.md)
- [1106.md](../raw/collections/agentx-mi355x-20261113/reports/muse-spark/1106.md)
- [FRAMEWORK_AND_PROVENANCE.md](../raw/collections/agentx-mi355x-20261113/reports/muse-spark/FRAMEWORK_AND_PROVENANCE.md)
- [POINTS.md](../raw/collections/agentx-mi355x-20261113/reports/muse-spark/POINTS.md)
- [PR_TRIALS.md](../raw/collections/agentx-mi355x-20261113/reports/muse-spark/PR_TRIALS.md)
- [ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/muse-spark/ROUTE.md)
- [muse-spark.md](../raw/collections/agentx-mi355x-20261113/reports/muse-spark.md)

Public source entry points:

- https://github.com/SemiAnalysisAI/InferenceX/pull/4127
- https://github.com/SemiAnalysisAI/InferenceX/pull/4201
- https://github.com/SemiAnalysisAI/InferenceX/pull/4287
- https://github.com/SemiAnalysisAI/InferenceX/pull/4295

For decisions, use the origin-labelled [case index](../INDEX.md) and current workload
evidence. Local toggle cards are later, separate evidence; they do not change the author
predecision information or turn a bundled historical gain into per-kernel attribution.
