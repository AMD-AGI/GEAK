# Modeling decisions and TraceLens integration

This reference is shared by all three modes, but its model corrections and
handwritten fallbacks do **not** apply inside `tl-only`, which preserves TL's own
models. Use it from `rocprof` or `combination`, or when building a frozen semantic
reference per [corrections.md](corrections.md).

Read the matching operator section. These rules define the model boundary; they
do not require every operator to use the same formula.

## Using TraceLens

Select a maintained checkout explicitly and record the commit and local diff.
Do not import whichever transient experiment directory happens to be on `sys.path`.
Install that checkout's declared dependencies in the chosen environment. A shape
model may be CPU-only even though the package import has additional dependencies.

The September 2026 comparison used TraceLens commit
`a59a9c165bb64c7c416fd7cf79149803d552e43c`. The following API signatures were checked
against that version; inspect the selected version before using them:

```python
from TraceLens.PerfModel.perf_model import GEMM

# Example: conventional BF16 C[M,N] = A[M,K] @ B[K,N], no bias.
M, N, K = 128, 4096, 4096
flops = GEMM.flops_func(M, N, K, False)
nbytes = GEMM.bytes_func(M, N, K, False, 2, 2, 2, 2)
# bytes_func argument order after bias:
# bpe_mat1, bpe_mat2, bpe_bias, bpe_output.
assert flops == 2 * M * N * K
assert nbytes == 2 * (M * K + K * N + M * N)
```

Other starting points in that checkout are
`TraceLens.PerfModel.extensions.attention_perf_model_extensions.InferenceAttention`
and `...moe_perf_model_extensions.UnfusedMoE_Up` / `UnfusedMoE_Down`.
Inspect their `flops_func` / `bytes_func` definitions: sparse masks, aliasing,
ragged sequence lengths, routing distributions, and fused epilogues need explicit
adaptation. Do not assume a similarly named class matches the measured kernel.

If TL is unavailable, a transparent handwritten formula can support a provisional
analysis. Label its source accordingly; do not report that TraceLens executed it.

## GEMM and batched GEMM

For conventional GEMM, count FMA as two FLOPs:

```text
F_matmul = 2 * M * N * K
B_ideal = M*K*s_A + K*N*s_B + M*N*s_C
```

Add required bias reads and epilogue work explicitly. If `beta != 0`, include the
old output read. For batched GEMM, distinguish distinct matrices from broadcast
weights that can ideally be reused. Do not multiply shared storage by batch blindly.
Logical dimensions define useful FLOPs; padded tile dimensions belong to execution
diagnostics. Packed FP4 payload uses half a byte per element before packing/alignment
and scale overhead; an FP4 dtype name alone is not a complete traffic model.

## Attention and sparse MLA

For fixed allowed query-key pairs, a conventional matrix-work component is:

```text
F_QK_PV = 2 * H_Q * sum_over_queries(valid_KV_count) * (D_QK + D_V)
```

Define masking, causal prefixes, sparse selection, and head sharing. If per-head
selection differs, sum the pairs per head explicitly. State whether softmax,
normalization, and auxiliary work are included; a matrix-only estimate is partial.

Distinguish semantic token selection from execution granularity. If selecting a
page is part of the operation, count its valid selected tokens. Masked tail work
executed because a tile/page is padded is overhead. Avoid replacing a sparse
selection with the dense context length.

In absorbed MLA, K and V can share one latent row; V may be a view of its first
columns. Model the union of required storage under the reuse assumption, rather
than charging separate full K and V tensors. Account for Q, final output, indices,
page metadata, and quantization scales that the contract requires.

For **implementation diagnostics**, split-K partial traffic may include:

```text
partial_output_write = sum(actual_splits_per_request) * H_Q * D_V * sizeof(partial)
partial_LSE_write    = sum(actual_splits_per_request) * H_Q * sizeof(LSE)
```

Then account for reduction reads and final writes. Use actual active splits, not
the capacity of the scratch allocation. These bytes are excluded from a fixed
whole-attention ideal model when the decomposition is removable. A stage-only
model instead includes that stage's required partial output, and is explicitly
specific to that stage contract.

## Indexer and absorb BMM

An indexer may scan the same KV data for multiple tiles/heads. Ideal reuse and
observed rereads describe different quantities. Low effective bandwidth together
with high measured external bandwidth suggests traffic expansion, not necessarily
unused physical bandwidth. Include required score/mask/top-k work if the boundary
is the full indexer; a logits-only benchmark must say so.

For absorb BMM with fused quantization, distinguish storage and dot precision.
Loading BF16 activations, converting on chip, and feeding FP8 matrix instructions
does not imply writing and rereading an FP8 global tensor. Count the original
external input and required output/scales. Add an intermediate only when it must
cross the agreed memory boundary. Keep conversion work explicit if omitted from
the matrix-work model.

## MoE

Fix routing/top-k and expert dimensions per case. Compute matrix work from the
actual routed token counts and logical expert GEMMs. Use known active experts to
estimate unique weight traffic; label uniform-routing expectations as estimates
when the route data is unavailable. Include packed weights, block scales, required
activations, routing metadata, and the agreed final output.

Separate unavoidable semantic outputs from a materialized `[tokens, topk, hidden]`
buffer used only between GEMM2 and reduction. Its write plus read can dominate
measured traffic even when a fused implementation can avoid it. Gating, activation,
quantization, and routing overhead must be itemized or stated as omitted.

## Lessons from the September 21 comparison

These are historical observations from one MI355X setup, not correction factors:

- Sparse MLA exposed shared-latent double counting and split-K scope differences.
- Absorb BMM exposed an invented external round trip for on-chip quantization.
- Indexer decode showed roughly 5.24 times the modeled external traffic.
- MoE prefill showed roughly 8.28 times the modeled traffic, including a large
  intermediate write/read pair.
- Small projection cases had much more counter-derived than useful matrix work;
  the exact factors require instruction/counter analysis, not an automatic padding
  explanation.

Preserve old results as `legacy_tl`; derive corrections from the operator contract
and storage layout, not by scaling a model until it matches the counters.
