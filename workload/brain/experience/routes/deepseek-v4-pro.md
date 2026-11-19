# DeepSeek V4 Pro: route guide

This is an English navigation summary of the retained historical analysis, not a
reconstructed sequence of independent GPU experiments. Each release can bundle changes.

| Branch / stage | Recorded changes | Transfer limit |
| --- | --- | --- |
| vLLM branch | Remove shard padding; improve sparse attention/metadata; fuse selected producers and explore heterogeneous experts, then DSpark/mono/workspace changes. | Quantization, topology, mode and actual activation distinguish paths; draft/indexer work can remain external. |
| Single-node SGLang | Explore projections/experts, TP4 vs TP8, cache/overlap, DSpark, FP8 KV, MegaMoE, PCP and small-step mono. | Different concurrency regimes select different implementations and resource layouts. |
| MoRI separated roles | Vary prefill capacity, DP/EP, TP, indexers, overlap/chunks, memory pools, speculation and role resources. | Preserve reversals and role-specific costs; larger host pools and newer draft releases can regress. |
| Version and metric audit | Check recovered image bytes, source/recipe/worker topology and client calibration. | Package labels and published total-resource fields do not prove equivalent implementations or GPU budgets. |

The original detailed reports preserve intermediate versions, PRs, point tables, failures
and reversals. Archive links require the retained collection to be supplied as the local
view or through the task archive root; [raw catalogue](../raw/README.md).

- [EVIDENCE_NOTES.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v4-pro/EVIDENCE_NOTES.md)
- [IMAGE_SOURCE_AUDIT.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v4-pro/IMAGE_SOURCE_AUDIT.md)
- [MORI_ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v4-pro/MORI_ROUTE.md)
- [POINTS.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v4-pro/POINTS.md)
- [PR2823_DEVELOPMENT.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v4-pro/PR2823_DEVELOPMENT.md)
- [ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v4-pro/ROUTE.md)
- [SGLANG_ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v4-pro/SGLANG_ROUTE.md)
- [VLLM_ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v4-pro/VLLM_ROUTE.md)
- [deepseek-v4-pro.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v4-pro.md)

Public source entry points:

- https://github.com/SemiAnalysisAI/InferenceX/pull/2823
- https://github.com/vllm-project/vllm/pull/51473
- https://github.com/vllm-project/vllm/pull/53161

For decisions, use the origin-labelled [case index](../INDEX.md) and current workload
evidence. Local toggle cards are later, separate evidence; they do not change the author
predecision information or turn a bundled historical gain into per-kernel attribution.
