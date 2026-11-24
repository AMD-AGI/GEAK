#!/usr/bin/env python3
"""Build a source-anchored tile/data dependency DAG for MegaMoEV2."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


_NODES = {
    "pre_dispatch_quant": {
        "stage": "pre_dispatch",
        "role": "convert BF16 token rows to FP8 plus block scales",
        "anchor": (
            "aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py",
            "x_q, scales = self.quantize(x_bf16)",
            0,
        ),
    },
    "stage1_dispatch_plan": {
        "stage": "stage1",
        "role": "build compact route/group metadata and expected readiness epochs",
        "anchor": (
            "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py",
            "emit_dispatch_plan(",
            0,
        ),
    },
    "stage1_dispatch_payload": {
        "stage": "stage1",
        "role": "copy routed token payload and scales to expert-rank symmetric buffers",
        "anchor": (
            "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py",
            "emit_dispatch_payload(",
            0,
        ),
    },
    "stage1_payload_acquire": {
        "stage": "stage1",
        "role": "wait for per-tile or per-expert payload publication",
        "anchor": (
            "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py",
            "def _wait_tile_payload(flat):",
            0,
        ),
    },
    "stage1_gemm1_activation_quant": {
        "stage": "stage1",
        "role": "GEMM1, activation, and FP8 quantization into Stage2 activation rows",
        "anchor": (
            "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py",
            "expert_of_flat, _do_scheduled_tile = build_fused_gemm1(",
            0,
        ),
    },
    "stage2_tile_schedule": {
        "stage": "stage2",
        "role": "map compact expert rows and hidden-dimension blocks to persistent work",
        "anchor": (
            "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py",
            "total_m_blocks = (cumsum0 + fx.Int32(BM - 1)) // fx.Int32(BM)",
            0,
        ),
    },
    "stage2_gemm2": {
        "stage": "stage2",
        "role": "consume Stage1 FP8 rows and produce one GEMM2 accumulator tile",
        "anchor": (
            "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py",
            "accm_vecs, m_row, n_block_idx, _n_out_rt = gemm2_compute_v2(",
            0,
        ),
    },
    "stage2_weight_quant_scatter": {
        "stage": "stage2",
        "role": "weight, optionally FP8-quantize, and scatter output rows to token-owner ranks",
        "anchor": (
            "aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py",
            "p2p_scatter_epilog(lds_base_i32, accm_vecs",
            0,
        ),
    },
    "combine_cross_rank_visibility": {
        "stage": "combine",
        "role": "publish and acquire cross-rank completion before reading P2P input",
        "anchor": (
            "aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py",
            "# Stage 2: CrossDeviceBarrier.",
            0,
        ),
    },
    "combine_topk_reduce": {
        "stage": "combine",
        "role": "read peer slots, dequantize, reduce top-k contributions, and store output",
        "anchor": (
            "aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py",
            "# Stage 3: local read + WarpAccum.",
            0,
        ),
    },
}


_EDGES = [
    {
        "from": "pre_dispatch_quant",
        "to": "stage1_dispatch_payload",
        "kind": "RAW",
        "object": "quantized_token_rows_and_scales",
    },
    {
        "from": "stage1_dispatch_plan",
        "to": "stage1_dispatch_payload",
        "kind": "CONTROL",
        "object": "route_plan_and_destination_offsets",
    },
    {
        "from": "stage1_dispatch_payload",
        "to": "stage1_payload_acquire",
        "kind": "RELEASE_ACQUIRE",
        "object": "payload_ready_epoch",
    },
    {
        "from": "stage1_dispatch_plan",
        "to": "stage1_gemm1_activation_quant",
        "kind": "RAW",
        "object": "sorted_expert_ids_token_map_weights_and_tile_row_base",
    },
    {
        "from": "stage1_payload_acquire",
        "to": "stage1_gemm1_activation_quant",
        "kind": "CONTROL",
        "object": "expert_tile_payload_visibility",
    },
    {
        "from": "stage1_dispatch_plan",
        "to": "stage2_tile_schedule",
        "kind": "RAW",
        "object": "compact_row_count_and_expert_tile_metadata",
    },
    {
        "from": "stage1_gemm1_activation_quant",
        "to": "stage2_gemm2",
        "kind": "RAW",
        "object": "stage1_fp8_activation_rows_and_scales",
    },
    {
        "from": "stage2_tile_schedule",
        "to": "stage2_gemm2",
        "kind": "CONTROL",
        "object": "persistent_m_n_tile_assignment",
    },
    {
        "from": "stage2_gemm2",
        "to": "stage2_weight_quant_scatter",
        "kind": "REGISTER_LDS",
        "object": "gemm2_accumulator_tile_and_route_weight",
    },
    {
        "from": "stage2_weight_quant_scatter",
        "to": "combine_cross_rank_visibility",
        "kind": "RELEASE_ACQUIRE",
        "object": "symmetric_combine_input_rows",
    },
    {
        "from": "combine_cross_rank_visibility",
        "to": "combine_topk_reduce",
        "kind": "CONTROL",
        "object": "all_peer_payload_visibility",
    },
]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_anchor(root: Path, spec: tuple[str, str, int]) -> dict:
    relative, needle, occurrence = spec
    path = root / relative
    text = path.read_text(encoding="utf-8")
    matches = [
        (line_number, line.strip())
        for line_number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    ]
    if occurrence >= len(matches):
        raise ValueError(
            f"{path}: expected occurrence {occurrence} of {needle!r}, "
            f"found {len(matches)}"
        )
    line_number, source = matches[occurrence]
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "line": line_number,
        "match": needle,
        "source": source,
    }


def _topological_order(nodes: set[str], edges: list[dict]) -> list[str]:
    incoming = {node: 0 for node in nodes}
    outgoing = {node: [] for node in nodes}
    for edge in edges:
        source = edge["from"]
        target = edge["to"]
        if source not in nodes or target not in nodes:
            raise ValueError(f"edge references an unknown node: {edge}")
        incoming[target] += 1
        outgoing[source].append(target)
    ready = sorted(node for node, count in incoming.items() if count == 0)
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in outgoing[node]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(nodes):
        raise ValueError("tile dependency graph contains a cycle")
    return order


def derive(source_root: Path, trace_dag_path: Path) -> dict:
    trace_dag = json.loads(trace_dag_path.read_text(encoding="utf-8"))
    nodes = {
        name: {
            "stage": spec["stage"],
            "role": spec["role"],
            "source_anchor": _source_anchor(source_root, spec["anchor"]),
        }
        for name, spec in _NODES.items()
    }
    order = _topological_order(set(nodes), _EDGES)
    trace_cases = trace_dag.get("cases", {})
    if not trace_cases:
        raise ValueError("runtime trace DAG has no cases")
    for case_id, case in trace_cases.items():
        if any(value != 0.0 for value in case["measured_pairwise_overlap_ms"].values()):
            raise ValueError(f"{case_id}: expected a serialized measured trace")
    return {
        "schema_version": "geak-tile-dependency-dag-v1",
        "status": "complete_for_isolated_operator_tile_scope",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
        "edges": _EDGES,
        "topological_order": order,
        "coverage": {
            "pre_dispatch": True,
            "stage1_dispatch_grouping_gemm1_activation_quant": True,
            "stage2_schedule_gemm2_weight_quant_scatter": True,
            "combine_visibility_dequant_topk_reduce": True,
        },
        "runtime_kernel_serialization": {
            "source": str(trace_dag_path.resolve()),
            "source_sha256": hashlib.sha256(trace_dag_path.read_bytes()).hexdigest(),
            "cases": trace_cases,
            "observed_overlap": "zero_for_all_category_pairs",
        },
        "scope_warning": (
            "The graph closes operator/tile RAW and synchronization dependencies. "
            "It does not model compiler instruction scheduling or per-wave dynamic latency."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trace-dag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = derive(args.source_root, args.trace_dag)
    _write(args.output, payload)
    print(f"GEAK_TILE_DAG_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
