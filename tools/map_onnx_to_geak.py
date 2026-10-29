#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Map an ONNX model's operator set onto GEAK's EP-kernel optimization coverage.

Loads a `.onnx` graph (external weights are NOT needed — only the node list),
counts each `op_type`, and joins against tools/ep_op_registry.yaml to print a
three-column coverage table:

  ONNX op (count)  ->  HIP EP runs it on GPU?  ->  GEAK can optimize it today?

The "GEAK coverage" column is the point: it shows which of the model's kernels
GEAK already has a task for (covered), which is the gap being added (hero), and
which are roadmap. The "EP on GPU" column is a static known-good list from the
MorphiZen HIP-EP op inventory (coverage is ultimately per-node at MLIR
legalization time, so treat `confirm` entries as "verify node-by-node").

    python3 tools/map_onnx_to_geak.py path/to/model.onnx

Run in an env with `onnx` installed (e.g. the `oga` conda env).
"""
import argparse
import collections
import os
import sys

import yaml

try:
    import onnx
except ImportError:
    sys.exit("onnx not importable — run in an env with onnx installed "
             "(e.g. `conda activate oga`).")

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(TOOLS_DIR, "ep_op_registry.yaml")

# ONNX op_type -> GEAK registry op. Ops not listed are either roadmap (if the
# HIP EP runs them) or n/a (pure shape/data-movement glue, never an optimization
# target). A registry op present here means GEAK can generate a task for it.
ONNX_TO_REGISTRY = {
    "MatMulNBits": "matmul_nbits",
    "LayerNormalization": "layer_norm",
    "SimplifiedLayerNormalization": "rmsnorm_selfcheck",
    "SkipSimplifiedLayerNormalization": "rmsnorm_selfcheck",
    "RMSNormalization": "rmsnorm_selfcheck",
    "GatedRMSNorm": "rmsnorm_selfcheck",
}

# ops the HIP EP legalizes to a .hip kernel. `True` = inventoried; "confirm" =
# Qwen3-Next variant that should be checked node-by-node before claiming.
EP_ON_GPU = {
    "MatMulNBits": True, "LayerNormalization": True,
    "SimplifiedLayerNormalization": True, "SkipSimplifiedLayerNormalization": True,
    "RMSNormalization": True, "GatedRMSNorm": "confirm",
    "GroupQueryAttention": True, "MultiHeadAttention": True, "Attention": True,
    "RotaryEmbedding": True, "MRotaryEmbedding": "confirm",
    "Softmax": True, "Sigmoid": True, "Mul": True, "Add": True,
    "LinearAttention": "confirm", "LinearAttentionGate": "confirm",
    "CausalConvWithState": "confirm", "LpNormalization": "confirm",
    "QMoE": True, "Gather": True, "GatherBlockQuantized": True,
    "MatMul": True, "Gemm": True, "ReduceSum": True, "Sub": True,
}

# pure shape / data-movement ops — never optimization targets.
GLUE_OPS = {
    "Reshape", "Transpose", "Cast", "Split", "Concat", "Unsqueeze", "Squeeze",
    "Gather", "Slice", "Expand", "Shape", "Constant", "ConstantOfShape",
    "Identity", "Range", "Tile", "Where",
}


def load_registry_ops():
    with open(REGISTRY) as f:
        reg = yaml.safe_load(f)
    return {r["op"] for r in reg["ops"]}


def classify(op_type, registry_ops):
    """Return (ep_on_gpu_str, geak_coverage_str)."""
    ep = EP_ON_GPU.get(op_type)
    ep_str = {True: "yes", "confirm": "yes (confirm)", None: "—"}[ep]

    reg_op = ONNX_TO_REGISTRY.get(op_type)
    if reg_op and reg_op in registry_ops:
        if reg_op == "rmsnorm_selfcheck":
            geak = f"HERO -> {reg_op}"
        else:
            geak = f"covered ({reg_op})"
    elif op_type in GLUE_OPS:
        geak = "n/a (glue)"
    elif ep is not None:
        geak = "roadmap"
    else:
        geak = "roadmap (unmapped)"
    return ep_str, geak


def op_histogram(model_path):
    model = onnx.load(model_path, load_external_data=False)
    return collections.Counter(n.op_type for n in model.graph.node), len(model.graph.node)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to a .onnx model file")
    args = ap.parse_args()
    if not os.path.isfile(args.model):
        sys.exit(f"model not found: {args.model}")

    registry_ops = load_registry_ops()
    hist, total = op_histogram(args.model)

    rows = []
    for op_type, count in hist.most_common():
        ep_str, geak = classify(op_type, registry_ops)
        rows.append((count, op_type, ep_str, geak))

    w_op = max((len(r[1]) for r in rows), default=8)
    w_ep = max((len(r[2]) for r in rows), default=12)
    print(f"\n{os.path.basename(args.model)}: {total} nodes\n")
    print(f"{'count':>6}  {'ONNX op':<{w_op}}  {'EP on GPU':<{w_ep}}  GEAK coverage")
    print(f"{'-'*6}  {'-'*w_op}  {'-'*w_ep}  {'-'*13}")
    for count, op_type, ep_str, geak in rows:
        print(f"{count:>6}  {op_type:<{w_op}}  {ep_str:<{w_ep}}  {geak}")

    # summary: GEAK-optimizable node coverage
    covered = sum(c for c, o, _, g in rows if g.startswith("covered"))
    hero = sum(c for c, o, _, g in rows if g.startswith("HERO"))
    print(f"\nGEAK today: {covered} nodes covered; "
          f"+{hero} nodes unlocked by the hero op (rmsnorm_selfcheck).")


if __name__ == "__main__":
    main()
