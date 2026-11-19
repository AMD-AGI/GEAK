# Kimi K3.1 Pro: route guide

This is an English navigation summary of the retained historical analysis, not a
reconstructed sequence of independent GPU experiments. Each release can bundle changes.

| Branch / stage | Recorded changes | Transfer limit |
| --- | --- | --- |
| Separate initial branches | SGLang DFlash2 and vLLM DSpark have different checkpoints/runner, cache and fusion contexts. | Do not combine them into one same-resource gain curve. |
| Initial vLLM trials | Repair stalls/memory; vary chunks, admission, draft lengths and host tier, preserving failed alternatives. | A qualified initial recipe already includes specialized low-concurrency work. |
| SGLang kernel/resource updates | Split attention work, shard replicated prefill FC, improve sorting/presort and share rank-group resources. | New device capacity can over-allocate derived host memory. |
| vLLM regime tuning | Revisit DCP, admission and speculation, then specialized DSA/fold and prefill cadence. | Local wins can conflict with CI results; multiple changes share each release. |
| SGLang later policy / separated roles | Adjust target/draft admission/cadence and introduce a distinct P/D branch with transport work. | P90 gains can worsen TTFT; a 16-GPU separated system is not an 8-GPU kernel comparison. |

The original detailed reports preserve intermediate versions, PRs, point tables, failures
and reversals. Archive links require the retained collection to be supplied as the local
view or through the task archive root; [raw catalogue](../raw/README.md).

- [EVIDENCE_NOTES.md](../raw/collections/agentx-mi355x-20261113/reports/kimi-k31-pro/EVIDENCE_NOTES.md)
- [POINTS.md](../raw/collections/agentx-mi355x-20261113/reports/kimi-k31-pro/POINTS.md)
- [ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/kimi-k31-pro/ROUTE.md)
- [SGLANG_ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/kimi-k31-pro/SGLANG_ROUTE.md)
- [VLLM_ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/kimi-k31-pro/VLLM_ROUTE.md)
- [kimi-k31-pro.md](../raw/collections/agentx-mi355x-20261113/reports/kimi-k31-pro.md)

Public source entry points:

- https://github.com/sgl-project/sglang/pull/48437
- https://github.com/sgl-project/sglang/pull/48439
- https://github.com/SemiAnalysisAI/InferenceX/pull/4928

For decisions, use the origin-labelled [case index](../INDEX.md) and current workload
evidence. Local toggle cards are later, separate evidence; they do not change the author
predecision information or turn a bundled historical gain into per-kernel attribution.
