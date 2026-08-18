#!/usr/bin/env python3
"""Single source of truth for the KB controlled vocabularies (plan Part 1.1-1.2).

Every KB tool imports THIS module so the vocabularies live in exactly one place:
  * _backfill_kb.py   — derives platforms/kernel_class from these tables.
  * _gen_index.py     — buckets files into views/ by these axes.
  * kb_resolve.py     — validates + orders query filters against these.
  * _validate_kb.py   — flags any frontmatter value that is not in these sets.

taxonomy.md is the HUMAN-readable mirror of the same tables; if you edit one, edit
both. _validate_kb.py cross-checks that taxonomy.md lists every id defined here.
"""

# --------------------------------------------------------------------------- #
# Hardware generations + SKUs (plan Part 1.1). `gen` gains gfx1250/CDNA5/MI450.
# The SKU axis exists only where SKUs differ materially (clock/TDP -> roofline
# denominator), e.g. MI350X vs MI355X.
# --------------------------------------------------------------------------- #
GENS = {
    "gfx906": {"arch": "GCN/pre-CDNA", "skus": ["mi50", "mi60"]},
    "gfx908": {"arch": "CDNA1", "skus": ["mi100"]},
    "gfx90a": {"arch": "CDNA2", "skus": ["mi210", "mi250", "mi250x"]},
    "gfx942": {"arch": "CDNA3", "skus": ["mi300a", "mi300x", "mi325x"]},
    "gfx950": {"arch": "CDNA4", "skus": ["mi350x", "mi355x"]},
    "gfx1250": {"arch": "CDNA5", "skus": ["mi450"]},
}
GEN_IDS = set(GENS)
SKU_IDS = {s for g in GENS.values() for s in g["skus"]}

# SKU -> gen reverse lookup.
SKU_TO_GEN = {s: g for g, d in GENS.items() for s in d["skus"]}

# --------------------------------------------------------------------------- #
# kernel_class (plan Part 1.1) — the ONE axis that unifies the ~50 operator ids,
# the learned/ hand-written groups, and kernel_workflow's free-text op_kind.
# Two-level dotted ids.
# --------------------------------------------------------------------------- #
KERNEL_CLASSES = {
    "gemm.dense", "gemm.batched", "gemm.grouped_moe", "gemm.splitk_streamk",
    "gemm.scaled_quant", "gemm.epilogue_fused", "gemm.skinny_decode",
    "attn.prefill", "attn.decode_paged", "attn.mla", "attn.gqa_mqa",
    "attn.sparse", "attn.linear", "attn.spec_decode",
    "norm_act", "positional", "quant",
    "moe.routing", "moe.dispatch",
    "collective", "data_movement", "elementwise_reduction", "conv",
    "embedding_sampling", "method",
}

# operator id (taxonomy.md) -> kernel_class. The full 50-row map; _backfill_kb.py
# uses it to auto-fill kernel_class on operator/sota cards.
OPERATOR_KERNEL_CLASS = {
    # GEMM family
    "dense_gemm": "gemm.dense",
    "batched_gemm": "gemm.batched",
    "grouped_gemm_moe": "gemm.grouped_moe",
    "splitk_streamk_gemm": "gemm.splitk_streamk",
    "scaled_quant_gemm": "gemm.scaled_quant",
    "gemm_epilogue_fused": "gemm.epilogue_fused",
    "skinny_gemv_decode": "gemm.skinny_decode",
    # Attention
    "attention_prefill_fmha": "attn.prefill",
    "attention_decode_paged": "attn.decode_paged",
    "mla_attention": "attn.mla",
    "gqa_mqa_attention": "attn.gqa_mqa",
    "sliding_window_attention": "attn.sparse",
    "sparse_attention_nsa": "attn.sparse",
    "linear_attention_gated_delta": "attn.linear",
    "chunked_prefill": "attn.prefill",
    "context_parallel_attention": "attn.prefill",
    "speculative_decode_verify": "attn.spec_decode",
    # Norm / Act
    "rmsnorm": "norm_act",
    "layernorm": "norm_act",
    "softmax": "norm_act",
    "act_and_mul_silu_gelu": "norm_act",
    "fused_add_rmsnorm": "norm_act",
    "fused_norm_quant": "norm_act",
    # Positional
    "rope": "positional",
    "mrope": "positional",
    "alibi": "positional",
    # Embedding / sampling
    "embedding": "embedding_sampling",
    "lm_head_logits": "embedding_sampling",
    "sampling_topk_topp": "embedding_sampling",
    # Elementwise / reduction / scan
    "elementwise": "elementwise_reduction",
    "reduction": "elementwise_reduction",
    "cumsum_scan": "elementwise_reduction",
    "argmax_topk": "elementwise_reduction",
    "cast_fill_copy": "elementwise_reduction",
    # Conv
    "causal_conv1d": "conv",
    "depthwise_conv": "conv",
    "conv2d": "conv",
    # Data movement
    "transpose": "data_movement",
    "gather_scatter": "data_movement",
    "all_to_all_dispatch_combine": "data_movement",
    "paged_kv_copy": "data_movement",
    "layout_shuffle": "data_movement",
    # Quant ops
    "quant_dequant_fp8": "quant",
    "quant_int8": "quant",
    "quant_fp4_mxfp": "quant",
    "kv_cache_quant": "quant",
    # MoE
    "moe_routing_topk": "moe.routing",
    "moe_dispatch_combine": "moe.dispatch",
    "fused_moe_grouped_gemm": "gemm.grouped_moe",
    "shared_expert_fusion": "moe.dispatch",
    # Collectives
    "allreduce": "collective",
    "allgather": "collective",
    "reduce_scatter": "collective",
    "fused_allreduce_rmsnorm": "collective",
}

# --------------------------------------------------------------------------- #
# lever (plan Part 1.1) — the optimization-MEANS vocabulary (the missing 3rd axis).
# --------------------------------------------------------------------------- #
LEVERS = {
    "tile.static-shape", "tile.autotune-db", "tile.splitk", "tile.streamk",
    "fusion.epilogue", "fusion.prologue", "fusion.norm-quant",
    "backend.swap", "config.per-shape-tune", "env.flag",
    "layout.swizzle", "layout.xcd-l2",
    "pipeline.async-lds", "pipeline.software-pipeline",
    "occupancy.vgpr", "vectorize.coalesce",
    "dtype.microscale", "dtype.downcast",
    "graph.cudagraph-safe", "host.launch-overhead", "host.kernel-merge",
}

# --------------------------------------------------------------------------- #
# cost ladder (plan Part 1.2) — "simple -> complex" axis. Ordered L0<L1<L2<L3;
# kb_resolve sorts ASCENDING so the cheapest lever is tried first.
# --------------------------------------------------------------------------- #
COSTS = ["L0", "L1", "L2", "L3"]
COST_RANK = {c: i for i, c in enumerate(COSTS)}
COST_MEANING = {
    "L0": "env var / flag, zero code change, minutes",
    "L1": "config / autotune, no source change, hours",
    "L2": "backend swap / off-the-shelf lib, integration seam",
    "L3": "rewrite kernel / DSL port, days",
}

RISKS = {"parity-safe", "numerics-affecting", "integration-risky"}

# --------------------------------------------------------------------------- #
# bound_type (plan Part 1.1) — the roofline routing key.
# --------------------------------------------------------------------------- #
BOUND_TYPES = {
    "hbm_bw", "mfma_compute", "lds_bank", "l2_locality",
    "launch_overhead", "occupancy", "sync", "host_bound",
}

# --------------------------------------------------------------------------- #
# lifecycle (plan Part 2.2) — the in/at/out-of-store state machine.
# --------------------------------------------------------------------------- #
LIFECYCLES = {"candidate", "active", "stale", "archived"}
LAYERS = {"reference", "learned", "artifact"}


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def gens_to_platforms(gens):
    """Normalize a frontmatter `gens:` list into a `platforms:` list (known gfx ids only)."""
    out = []
    for g in gens or []:
        g = str(g).strip().lower()
        if g in GEN_IDS and g not in out:
            out.append(g)
    return out


def kernel_class_for_operator(op):
    return OPERATOR_KERNEL_CLASS.get(str(op or "").strip())
