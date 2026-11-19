# DeepSeek V4.1 Flash: route guide

This is an English navigation summary of the retained historical analysis, not a
reconstructed sequence of independent GPU experiments. Each release can bundle changes.

| Branch / stage | Recorded changes | Transfer limit |
| --- | --- | --- |
| vLLM initial correction | Resolve restrictive admission/graph geometry and change MoE backend. | Do not assign the full C1 release jump to the local kernel effect. |
| vLLM ordinary paths | Explore Engram placement/TP, scratch and KV budgets; align producer/scales/GEMM and paged scoring; withdraw faulty replay. | Representation, cache and correctness dependencies precede any transferable action. |
| vLLM larger fusion | Expand supported target layers and draft runners, then recover replay and revisit scheduling. | Coverage and fallback remain shape/graph dependent; target and draft are separate causes. |
| SGLang bring-up | Repair scratch, allocator, admission, graph and capacity in a native image after an unsuccessful port. | Retain the slow but qualified initial reference. |
| SGLang capacity and scheduling | Fix address arithmetic, revisit host placement/TP, prefill intervals, chunks and ordinary kernels. | Queue, TTFT, output and P90 tradeoffs are distinct. |
| SGLang split paths | Use bounded target/draft mono in one regime and row-group/page8 ordinary paths in another. | The mono scorer and alternate page layout are not interchangeable. |

The original detailed reports preserve intermediate versions, PRs, point tables, failures
and reversals. Archive links require the retained collection to be supplied as the local
view or through the task archive root; [raw catalogue](../raw/README.md).

- [EVIDENCE_NOTES.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v41-flash/EVIDENCE_NOTES.md)
- [POINTS.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v41-flash/POINTS.md)
- [ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v41-flash/ROUTE.md)
- [SGLANG_ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v41-flash/SGLANG_ROUTE.md)
- [VLLM_ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v41-flash/VLLM_ROUTE.md)
- [deepseek-v41-flash.md](../raw/collections/agentx-mi355x-20261113/reports/deepseek-v41-flash.md)

Public source entry points:

- https://github.com/SemiAnalysisAI/InferenceX/pull/4367
- https://github.com/vllm-project/vllm/pull/58456
- https://github.com/sgl-project/sglang/pull/45480

For decisions, use the origin-labelled [case index](../INDEX.md) and current workload
evidence. Local toggle cards are later, separate evidence; they do not change the author
predecision information or turn a bundled historical gain into per-kernel attribution.
