# GLM-5.2: route guide

This is an English navigation summary of the retained historical analysis, not a
reconstructed sequence of independent GPU experiments. Each release can bundle changes.

| Branch / stage | Recorded changes | Transfer limit |
| --- | --- | --- |
| Initial MXFP4 | Start TP4/EP4 and TP8/EP8 with HiCache; explore topology, speculation, admission, graph and chunk settings. | Keep internal alternatives and cache-ratio reversals. |
| Selection and metrics | Fix health/CLI paths, migrate attention/indexer implementations and resolve model-hook flag overrides. | Input-plus-output/GPU claims are not total output-throughput gains. |
| Cache and ordinary kernels | Fix registered-address aliases and copy rounds, then tune prefill/sparse attention and writers. | KV row layout and enabled branches govern applicability. |
| Larger indexer fusion | Use a bounded four-kernel chain and shared workspace, then revisit ordinary fallback choices. | Wider rows and writer formats remain constraints; partial tests do not prove all cache writes. |
| Later release / DCP | Improve several kernels, then shard target KV with DCP while some other state remains replicated. | Capacity does not imply a serving gain; retain regressions and automatic host/device memory effects. |
| FP8 / successor branches | Track FP8 and GLM-5.3 separately in the archive. | The pooled dashboard query does not make successor results evidence for GLM-5.2. |

The original detailed reports preserve intermediate versions, PRs, point tables, failures
and reversals. Archive links require the retained collection to be supplied as the local
view or through the task archive root; [raw catalogue](../raw/README.md).

- [EVIDENCE_NOTES.md](../raw/collections/agentx-mi355x-20261113/reports/glm52/EVIDENCE_NOTES.md)
- [FP8_AND_SUCCESSOR.md](../raw/collections/agentx-mi355x-20261113/reports/glm52/FP8_AND_SUCCESSOR.md)
- [MXFP4_ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/glm52/MXFP4_ROUTE.md)
- [POINTS.md](../raw/collections/agentx-mi355x-20261113/reports/glm52/POINTS.md)
- [ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/glm52/ROUTE.md)
- [glm52.md](../raw/collections/agentx-mi355x-20261113/reports/glm52.md)

Public source entry points:

- https://github.com/sgl-project/sglang/pull/35233
- https://github.com/sgl-project/sglang/pull/37152
- https://github.com/sgl-project/sglang/pull/38583

For decisions, use the origin-labelled [case index](../INDEX.md) and current workload
evidence. Local toggle cards are later, separate evidence; they do not change the author
predecision information or turn a bundled historical gain into per-kernel attribution.
