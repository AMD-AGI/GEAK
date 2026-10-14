---
name: kernel-extract-glm-agentx-example
description: Apply the AgentX kernel extraction method to GLM deployments with prefix caching, speculative decoding, MoE or sparse attention. Consult only when those features are present.
---

# GLM AgentX example

Use alongside [kernel-extract](../kernel-extract/SKILL.md). This example maps
GLM features to evidence needed by the general method; it supplies no default
shapes, layer counts, multipliers, kernel names or performance thresholds.

| Observed feature | Establish before extracting |
|---|---|
| Prefix caching | Newly executed query tokens, per-request context and cache-hit distribution |
| Speculative decoding | Separate target/draft architectures, actual subphase boundaries, proposed/accepted tokens and graph-padded M |
| MoE | Captured routing, packed weights/scales, preshuffling and the live fused-operation boundary |
| Sparse attention/indexer | Which layers execute or reuse indices; separate indexer, selection and attention launches in each active phase |
| Shape-dependent dispatch | Per-shape kernels, full grids and call counts, including tails and eager fallbacks |

For a cached speculative request, nominal input length does not determine
prefill work, and request batch does not determine kernel M. Reconcile effective
configuration with the actual operands and launches before choosing cases.
Census active phases separately: a decode implementation need not be the prefill
implementation, and a draft model need not use the target model's kernels.

## Reconciling a supplied GLM UT delivery

Use the core skill's [selection coverage contract](../kernel-extract/references/selection-coverage.md).
The extraction target set is the union of PREFILL/DECODE Top-5 lists after
filtering each by overall workload share >3%. Prior deliveries are discovery
references; their full task lists do not override this rule.
For deliveries such as GLM-5.2 0917/0922, use the following families as search
leads only when the supplied artifacts and current model contain them:

| Family in the delivery | Distinction to preserve |
|---|---|
| Routed fused MoE | Whole fused op versus stage-1/stage-2 kernels and routing/quantization glue; real packed inputs and routing |
| Indexer decode score / top-k / joint | Separate editable leaves; joint overlaps both and cannot add coverage to them |
| Indexer prefill score / top-k chain | Separate phase and ragged/chunk metadata, even when decode dominates total time |
| Sparse MLA decode / prefill | Distinct dispatch branches; sharing a source module or callable does not imply shared UT coverage |
| MLA absorb BMM | Separate projection/attention work from indexer scoring |
| Add-RMSNorm-quant | Native source branch and fused boundary; small e2e headroom does not remove it from discovery |
| Indexer preamble | Projection, normalization, rotation, quantization and cache-write boundaries; establish which launches belong to the joint task |
| BF16 QKV/O projections, router and indexer WQ/WK | Distinguish by live call site and operand shape; a generic GEMM symbol or shared callable cannot allocate their shares |

When indexer score/top-k is selected, package its decode and prefill joint
branches in one unified task, as specified in
[joint indexer validation](../kernel-extract/references/joint-indexer-ut.md).
The old leaf tasks are useful diagnostic references; the old decode/prefill
joint tasks establish examples of the boundary. Preserve required companion
launches even when a companion's individual overall share is below 3%.

Check current handoff revisions before old smoke/score fields. Reconcile duplicate
copies and overlapping joint tasks across deliveries rather than counting
directories as independent kernels. Never import their old GPU-busy percentages
into a ranking with a different run, phase mix or denominator. Unmatched families
stay unresolved until current evidence establishes a mapping or absence.

If a UT matches device names, grids and outputs, still verify that its editable
leaf executes: a copied dispatcher can import the installed implementation
inside a branch. Apply the core skill's source-extraction checks and retain its
frozen overlay baseline and observed-call weighting.

Record the matching feature and current evidence in `kernel_workload.json` or
`meta.kernel_extraction`. Skip absent features and follow the calling role's
normal artifacts and return schema.
