# MiniMax M3.1 Flash: route guide

This is an English navigation summary of the retained historical analysis, not a
reconstructed sequence of independent GPU experiments. Each release can bundle changes.

| Branch / stage | Recorded changes | Transfer limit |
| --- | --- | --- |
| vLLM initial | Quark MXFP4 and DSpark, with regime-specific draft, KV, overlap and mono choices. | The original route already contains fusion; later code presence is not first activation. |
| SGLang initial / recovery | Bring up the original-checkpoint branch; withdraw faulty HiCache and semantic top-k reuse, then recover legal host allocation. | Capacity/allocator recovery is not an isolated kernel gain. |
| vLLM intermediate alternatives | Explore draft depth, checkpoint/SP alternatives and host capacity; retain selected pieces in the Quark line. | An SP flag can remain on while runtime lookup fails. |
| 4fab | Add indexer, norm/quant, sparse-work, GEMM, draft, frontend and replay improvements; remove overlap. | Keep withdrawn unsupported communication and slower trials. |
| 75bc | Replace older mono with a new compatible operator chain, layout and runtime; restore qualified overlap. | GEMM/MoE/norm/SP dependencies and automatic memory effects prevent adding individual gains. |
| CPU / prefix work | Cache safe token prefixes and protect contiguous CPU KV chains. | Encoding observations and simultaneous tier changes do not establish isolated serving contributions. |

The original detailed reports preserve intermediate versions, PRs, point tables, failures
and reversals. Archive links require the retained collection to be supplied as the local
view or through the task archive root; [raw catalogue](../raw/README.md).

- [EARLY_AND_SGLANG.md](../raw/collections/agentx-mi355x-20261113/reports/minimax-m31-flash/EARLY_AND_SGLANG.md)
- [EVIDENCE_NOTES.md](../raw/collections/agentx-mi355x-20261113/reports/minimax-m31-flash/EVIDENCE_NOTES.md)
- [POINTS.md](../raw/collections/agentx-mi355x-20261113/reports/minimax-m31-flash/POINTS.md)
- [ROUTE.md](../raw/collections/agentx-mi355x-20261113/reports/minimax-m31-flash/ROUTE.md)
- [VLLM_4FAB.md](../raw/collections/agentx-mi355x-20261113/reports/minimax-m31-flash/VLLM_4FAB.md)
- [VLLM_75BC.md](../raw/collections/agentx-mi355x-20261113/reports/minimax-m31-flash/VLLM_75BC.md)
- [minimax-m31-flash.md](../raw/collections/agentx-mi355x-20261113/reports/minimax-m31-flash.md)

Public source entry points:

- https://github.com/SemiAnalysisAI/InferenceX/pull/4694
- https://github.com/SemiAnalysisAI/InferenceX/pull/4866
- https://github.com/SemiAnalysisAI/InferenceX/pull/5035

For decisions, use the origin-labelled [case index](../INDEX.md) and current workload
evidence. Local toggle cards are later, separate evidence; they do not change the author
predecision information or turn a bundled historical gain into per-kernel attribution.
