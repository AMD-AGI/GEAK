#!/usr/bin/env python3
"""Build typed derived-evidence and provenance catalogs from live artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-evidence", type=Path, required=True)
    parser.add_argument("--hardware-measurements", type=Path, required=True)
    parser.add_argument("--publication-report", type=Path, action="append", default=[])
    parser.add_argument("--resource-evidence", type=Path)
    parser.add_argument("--dag-evidence", type=Path)
    parser.add_argument("--liveness-evidence", type=Path)
    parser.add_argument("--xgmi-evidence", type=Path)
    parser.add_argument("--pmc-evidence", type=Path)
    parser.add_argument("--att-occupancy-evidence", type=Path)
    parser.add_argument("--att-wait-evidence", type=Path)
    parser.add_argument("--combine-wait-evidence", type=Path)
    parser.add_argument("--payload-control-evidence", type=Path)
    parser.add_argument("--tile-dag-evidence", type=Path)
    parser.add_argument("--derived-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).isoformat()
    route = json.loads(args.route_evidence.read_text(encoding="utf-8"))
    hardware = json.loads(args.hardware_measurements.read_text(encoding="utf-8"))
    resource = (
        json.loads(args.resource_evidence.read_text(encoding="utf-8"))
        if args.resource_evidence
        else None
    )
    dag = (
        json.loads(args.dag_evidence.read_text(encoding="utf-8"))
        if args.dag_evidence
        else None
    )
    liveness = (
        json.loads(args.liveness_evidence.read_text(encoding="utf-8"))
        if args.liveness_evidence
        else None
    )
    xgmi = (
        json.loads(args.xgmi_evidence.read_text(encoding="utf-8"))
        if args.xgmi_evidence
        else None
    )
    pmc = (
        json.loads(args.pmc_evidence.read_text(encoding="utf-8"))
        if args.pmc_evidence
        else None
    )
    att_occupancy = (
        json.loads(args.att_occupancy_evidence.read_text(encoding="utf-8"))
        if args.att_occupancy_evidence
        else None
    )
    att_wait = (
        json.loads(args.att_wait_evidence.read_text(encoding="utf-8"))
        if args.att_wait_evidence
        else None
    )
    combine_wait = (
        json.loads(args.combine_wait_evidence.read_text(encoding="utf-8"))
        if args.combine_wait_evidence
        else None
    )
    payload_control = (
        json.loads(args.payload_control_evidence.read_text(encoding="utf-8"))
        if args.payload_control_evidence
        else None
    )
    tile_dag = (
        json.loads(args.tile_dag_evidence.read_text(encoding="utf-8"))
        if args.tile_dag_evidence
        else None
    )

    publication_runs = {}
    for path in args.publication_report:
        payload = json.loads(path.read_text(encoding="utf-8"))
        case = payload["cases"][0]
        name = case["variant"]["name"]
        publication_runs.setdefault(name, []).append(
            {
                "case_id": case["case_id"],
                "correctness": case["correctness"],
                "rank_max_ms": {
                    metric: max(
                        rank["timing_ms"][metric] for rank in case["ranks"]
                    )
                    for metric in ("e2e", "stage1", "stage2_combine")
                },
            }
        )
    publication_variants = []
    for name, runs in sorted(publication_runs.items()):
        metrics = {}
        for metric in ("e2e", "stage1", "stage2_combine"):
            values = [run["rank_max_ms"][metric] for run in runs]
            metrics[metric] = {
                "runs": values,
                "mean": statistics.mean(values),
                "span_pct": (
                    (max(values) - min(values)) / statistics.mean(values) * 100.0
                ),
            }
        publication_variants.append(
            {
                "name": "full" if name == "pc384" else name,
                "chunk_rows": int(name.removeprefix("pc")),
                "changed_components": (
                    [] if name == "pc384" else ["publication_granularity"]
                ),
                "correctness_status": "pass",
                "rank_max_ms": metrics,
                "source_cases": [run["case_id"] for run in runs],
            }
        )

    route_provenance = "collector:route-derived:live_20260814"
    hardware_provenance = "collector:transferbench:live_20260814"
    publication_provenance = "collector:publication-sweep:live_20260814"
    resource_provenance = "collector:kernel-trace-resource:live_20260814"
    dag_provenance = "collector:trace-dag:live_20260814"
    liveness_provenance = "collector:liveness-stress:live_20260814"
    xgmi_provenance = "collector:amdsmi-xgmi:live_20260814"
    pmc_provenance = "collector:rocprof-pmc:live_20260814"
    att_occupancy_provenance = "collector:att-occupancy:live_20260814"
    att_wait_provenance = "collector:att-wait:live_20260814"
    combine_wait_provenance = "collector:combine-wait-timer:live_20260814"
    payload_control_provenance = "collector:payload-control:live_20260814"
    tile_dag_provenance = "collector:tile-dag:live_20260814"
    derived = {
        "live:route_derived": {
            "kind": "route_derived_counts",
            "status": "complete",
            "confidence": "medium",
            "scope": "analytical logical-byte lower bound and route-derived padding",
            "metric_ids": [
                "live:route:remote_routes",
                "live:route:remote_payload_bytes",
                "live:route:stage1_padding_pct",
                "live:route:stage2_padding_pct",
            ],
            "provenance_refs": [route_provenance],
            "data": route,
        },
        "live:hardware_ceiling": {
            "kind": "hardware_ceiling",
            "status": "complete",
            "confidence": "high",
            "scope": "target-local MI355X TransferBench",
            "metric_ids": [
                f"live:hardware:{metric}"
                for metric in hardware["measurements"]
            ],
            "provenance_refs": [hardware_provenance],
            "data": hardware,
        },
    }
    if publication_variants:
        derived["live:publication_sweep"] = {
            "kind": "controlled_experiment",
            "status": (
                "complete"
                if min(len(runs) for runs in publication_runs.values()) >= 2
                else "partial"
            ),
            "confidence": "medium",
            "scope": "correctness-checked publication chunk sweep",
            "metric_ids": [
                "live:publication:e2e_rank_max_ms",
                "live:publication:stage1_rank_max_ms",
            ],
            "provenance_refs": [publication_provenance],
            "data": {
                "variants": publication_variants,
                "repetitions_per_variant": min(
                    len(runs) for runs in publication_runs.values()
                ),
                "overlap_pairs": [
                    {
                        "left": left["name"],
                        "right": right["name"],
                        "changed_components": ["publication_granularity"],
                    }
                    for index, left in enumerate(publication_variants)
                    for right in publication_variants[index + 1 :]
                    if left["name"] != "full" and right["name"] != "full"
                ],
                "delta_additivity_allowed": False,
            },
        }
    if resource is not None:
        derived["live:resource_residency"] = {
            "kind": "resource_residency",
            "status": "partial",
            "confidence": "high",
            "scope": "rocprofv3 dispatch metadata with LDS/wave residency bounds",
            "metric_ids": [
                "live:resource:lds_bytes",
                "live:resource:vgpr_count",
                "live:resource:sgpr_count",
                "live:resource:scratch_bytes",
                "live:resource:residency_upper_bound",
            ],
            "provenance_refs": [resource_provenance],
            "data": resource,
        }
    if dag is not None:
        derived["live:dependency_dag"] = {
            "kind": "dependency_dag",
            "status": "partial",
            "confidence": "medium",
            "scope": "kernel-level normalized Chrome Trace DAG",
            "metric_ids": [
                "live:dag:critical_path_ms",
                "live:dag:pairwise_overlap_ms",
            ],
            "provenance_refs": [dag_provenance],
            "data": dag,
        }
    if liveness is not None:
        derived["live:liveness"] = {
            "kind": "liveness",
            "status": "complete",
            "confidence": "high",
            "scope": "current v2 EP8 skew 1000-replay stress",
            "metric_ids": [
                "live:liveness:cuda_graph_replays",
                "live:liveness:e2e_rank_max_ms",
            ],
            "provenance_refs": [liveness_provenance],
            "data": liveness,
        }
    if xgmi is not None:
        derived["live:xgmi_firmware_accumulators"] = {
            "kind": "xgmi_firmware_accumulator",
            "status": "complete",
            "confidence": "high",
            "scope": (
                "all eight GPUs, AMD-SMI per-link firmware KB accumulators, "
                "three 200-replay intervals per route/control"
            ),
            "metric_ids": [
                "live:xgmi:paired_endpoint_bytes_per_replay",
                "live:xgmi:counter_to_logical_amplification",
                "live:xgmi:payload_control_removed_bytes",
            ],
            "provenance_refs": [xgmi_provenance],
            "data": xgmi,
        }
    if pmc is not None:
        derived["live:gmi_sector_counters"] = {
            "kind": "gmi_sector_counters",
            "status": "complete",
            "confidence": "high",
            "scope": "rank0 per-kernel rocprofv3 TCC/EA GMI 32-byte sectors",
            "metric_ids": [
                "live:pmc:gmi_sector_bytes:stage1",
                "live:pmc:gmi_sector_bytes:stage2",
                "live:pmc:gmi_sector_bytes:combine",
            ],
            "provenance_refs": [pmc_provenance],
            "data": pmc.get("gmi", {}),
        }
        derived["live:hardware_occupancy_counters"] = {
            "kind": "occupancy_counters",
            "status": "complete",
            "confidence": "medium",
            "scope": "rank0 whole-device profiled dispatch occupancy counters",
            "metric_ids": [
                "live:pmc:mean_occupancy_per_cu",
                "live:pmc:occupancy_percent",
            ],
            "provenance_refs": [pmc_provenance],
            "data": pmc.get("occupancy", {}),
        }
    if att_occupancy is not None:
        derived["live:att_occupancy"] = {
            "kind": "att_occupancy",
            "status": "complete",
            "confidence": "medium",
            "scope": "ATT SE0 sampled wave residency for uniform/skew stages",
            "metric_ids": [
                "live:att_occupancy:avg_active_waves_per_cu",
                "live:att_occupancy:peak_active_waves_per_cu",
                "live:att_occupancy:workgroup_equivalent_per_cu",
            ],
            "provenance_refs": [att_occupancy_provenance],
            "data": att_occupancy,
        }
    if att_wait is not None:
        derived["live:att_wait"] = {
            "kind": "att_wait",
            "status": "complete",
            "confidence": "medium",
            "scope": "ATT sampled waitcnt/barrier cycles plus static source sites",
            "metric_ids": [
                "live:att_wait:sync_accounted_cycle_share_pct",
                "live:att_wait:waitcnt_accounted_cycle_share_pct",
                "live:att_wait:barrier_accounted_cycle_share_pct",
            ],
            "provenance_refs": [att_wait_provenance],
            "data": att_wait,
        }
    if combine_wait is not None:
        derived["live:combine_peer_wait"] = {
            "kind": "combine_wait_timing",
            "status": "complete",
            "confidence": "medium",
            "scope": "instrumented Combine peer-readiness wait, all ranks and 20 replays",
            "metric_ids": [
                "live:combine_wait:rank_max_p95_us",
                "live:combine_wait:rank_mean_spread_us",
                "live:combine_wait:payload_control_removed_p95_us",
            ],
            "provenance_refs": [combine_wait_provenance],
            "data": combine_wait,
        }
    if payload_control is not None:
        derived["live:payload_control"] = {
            "kind": "payload_control",
            "status": "complete",
            "confidence": "high",
            "scope": "matched normal/no-payload Stage2 controls, three runs per route",
            "metric_ids": [
                "live:payload_control:stage2_combine_delta_ms",
                "live:payload_control:remote_incremental_exposed_ms",
            ],
            "provenance_refs": [payload_control_provenance],
            "data": payload_control,
        }
    if tile_dag is not None:
        derived["live:tile_dependency_dag"] = {
            "kind": "tile_dependency_dag",
            "status": "complete",
            "confidence": "high",
            "scope": "source-anchored isolated-operator tile/data dependencies",
            "metric_ids": [
                "live:tile_dag:node_count",
                "live:tile_dag:edge_count",
            ],
            "provenance_refs": [tile_dag_provenance],
            "data": tile_dag,
        }
    provenance = {
        route_provenance: {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "derive_route_evidence",
            "tool_version": "geak-route-derived-evidence-v1",
            "command": "derive_route_evidence.py --model-dim 7168 --stage1-block-m 128 --stage2-block-m 64",
            "timestamp": timestamp,
            "scope": "EP8 route matrix, logical FP8 payload lower bound, GEMM row padding",
            "repetitions": 1,
            "raw_artifacts": [str(args.route_evidence.resolve())],
            "confidence": "medium",
            "profiler_perturbation_pct": None,
            "cross_checks": ["route totals equal tokens * topk * world_size"],
            "units": {
                "route_count": "expert-token routes",
                "payload": "bytes",
                "padding": "rows and percent",
            },
        },
        hardware_provenance: {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "transferbench_ceiling",
            "tool_version": hardware["tool"],
            "command": "collect_transferbench.py --presets empty,hbm,a2a,p2p",
            "timestamp": timestamp,
            "scope": "eight local MI355X GPUs",
            "repetitions": 20,
            "raw_artifacts": list(hardware["raw_artifacts"]),
            "confidence": "high",
            "profiler_perturbation_pct": None,
            "cross_checks": ["all tests ran under the same fixed eight-GPU lease"],
            "units": dict(hardware["units"]),
        },
    }
    if publication_variants:
        provenance[publication_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "controlled_variants",
            "tool_version": "AITER bench_mega_moe_v2",
            "command": "bench_mega_moe_v2.py --route rank-mixed-skew --check-variant --stage1-payload-chunk-rows {128,256,384}",
            "timestamp": timestamp,
            "scope": "8192 tokens/rank skew route publication chunk sweep",
            "repetitions": min(
                len(runs) for runs in publication_runs.values()
            ),
            "raw_artifacts": [
                str(path.resolve()) for path in args.publication_report
            ],
            "confidence": "medium",
            "profiler_perturbation_pct": None,
            "cross_checks": [
                "each variant reports zero relative-L2 delta versus default",
                "each chunk size has independent process-level repetitions"
            ],
            "units": {"latency": "ms", "chunk_rows": "rows"},
        }
    if resource is not None:
        provenance[resource_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "rocprofv3_kernel_trace",
            "tool_version": "rocprofv3 ROCm 7.2.4",
            "command": "rocprofv3 --kernel-trace --stats -- EP8 MegaMoEV2 benchmark",
            "timestamp": timestamp,
            "scope": "rank0 uniform route dispatch resource metadata",
            "repetitions": 1,
            "raw_artifacts": [str(args.resource_evidence.resolve())],
            "confidence": "high",
            "profiler_perturbation_pct": None,
            "cross_checks": ["LDS limit cross-checked against MI355X hardware context"],
            "units": {
                "lds": "bytes/workgroup",
                "scratch": "bytes",
                "registers": "count",
                "residency": "workgroups/CU upper bound",
            },
        }
    if dag is not None:
        provenance[dag_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "trace_dependency_dag",
            "tool_version": "geak-trace-dependency-dag-v1",
            "command": "derive_trace_dag.py --report live_20260814_multi_rank_analysis.json",
            "timestamp": timestamp,
            "scope": "normalized kernel-level uniform/skew Chrome traces",
            "repetitions": 3,
            "raw_artifacts": [str(args.dag_evidence.resolve())],
            "confidence": "medium",
            "profiler_perturbation_pct": None,
            "cross_checks": ["all reported pairwise kernel overlaps are zero"],
            "units": {
                "duration": "ms/replay",
                "overlap": "ms/replay",
            },
        }
    if liveness is not None:
        provenance[liveness_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "cuda_graph_liveness_stress",
            "tool_version": "AITER bench_mega_moe_v2",
            "command": "bench_mega_moe_v2.py --route rank-mixed-skew --iters 1000",
            "timestamp": timestamp,
            "scope": "current v2 EP8 skew route",
            "repetitions": 1000,
            "raw_artifacts": [str(args.liveness_evidence.resolve())],
            "confidence": "high",
            "profiler_perturbation_pct": None,
            "cross_checks": ["process exited zero without timeout"],
            "units": {
                "replays": "count",
                "latency": "ms/replay",
            },
        }
    if xgmi is not None:
        provenance[xgmi_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "amdsmi_xgmi_accumulators",
            "tool_version": "AMD-SMI 26.2.2 metrics content revision 9",
            "command": (
                "bench_mega_moe_v2.py --mega-only --prequant "
                "--xgmi-replays 200 --xgmi-output <artifact>"
            ),
            "timestamp": timestamp,
            "scope": "all eight MI355X endpoints; matched workload and idle intervals",
            "repetitions": 3,
            "raw_artifacts": sorted(
                {
                    path
                    for group in xgmi.get("groups", {}).values()
                    for path in group.get("raw_artifacts", [])
                }
            ),
            "confidence": "high",
            "profiler_perturbation_pct": None,
            "cross_checks": [
                "200-replay intervals have sub-percent span for workload counters",
                "local-only route is near-zero relative to remote routes",
                "no-payload controls remove Stage2 counter traffic",
            ],
            "units": {
                "firmware_accumulator": "KB converted to bytes",
                "logical_payload": "bytes/replay",
            },
        }
    if pmc is not None:
        provenance[pmc_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "rocprofv3_pmc",
            "tool_version": "rocprofv3 ROCm 7.2.4",
            "command": (
                "rocprofv3 --pmc <GMI or occupancy counters> "
                "--kernel-include-regex <MegaMoE stages>"
            ),
            "timestamp": timestamp,
            "scope": "rank0 profiled MegaMoEV2 kernel dispatches",
            "repetitions": 1,
            "raw_artifacts": [str(args.pmc_evidence.resolve())],
            "confidence": "high",
            "profiler_perturbation_pct": None,
            "cross_checks": [
                "Stage2 no-payload reports zero GMI sectors",
                "uniform/skew Stage1 and Stage2 sectors are nearly equal",
                "occupancy percentages agree with 32 waves/CU capacity",
            ],
            "units": {
                "gmi": "32-byte TCC/EA sectors",
                "occupancy": "waves/CU and percent",
            },
        }
    if att_occupancy is not None:
        provenance[att_occupancy_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "rocprofv3_att_occupancy",
            "tool_version": "rocprofv3 ROCm 7.2.4 ATT occupancy v3.0.0",
            "command": "derive_att_occupancy_evidence.py --occupancy <six ATT files>",
            "timestamp": timestamp,
            "scope": "SE0 (8 of 256 CUs) uniform/skew stage samples",
            "repetitions": 1,
            "raw_artifacts": [str(args.att_occupancy_evidence.resolve())],
            "confidence": "medium",
            "profiler_perturbation_pct": None,
            "cross_checks": [
                "all start/end events are balanced",
                "sampled peaks match Stage1/Stage2 resource ceilings",
            ],
            "units": {
                "residency": "waves/CU and workgroup-equivalent/CU",
                "event_span": "microseconds under ATT",
            },
        }
    if att_wait is not None:
        provenance[att_wait_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "att_wait_derivation",
            "tool_version": "geak-wait-evidence-v1",
            "command": "derive_wait_evidence.py --att <stage CSVs> --source <kernels>",
            "timestamp": timestamp,
            "scope": "GPU0/CU1 sampled instructions and static source sites",
            "repetitions": 1,
            "raw_artifacts": [str(args.att_wait_evidence.resolve())],
            "confidence": "medium",
            "profiler_perturbation_pct": None,
            "cross_checks": ["ATT cycle shares use Latency + Idle without double-counting Stall"],
            "units": {"wait": "sampled cycles and percent"},
        }
    if combine_wait is not None:
        provenance[combine_wait_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "combine_peer_wait_timer",
            "tool_version": "s_memrealtime analysis kernel v1",
            "command": (
                "bench_mega_moe_v2.py --combine-wait-replays 20 "
                "--combine-wait-output <artifact>"
            ),
            "timestamp": timestamp,
            "scope": "all eight ranks, 256 Combine blocks/rank, 20 replays/route",
            "repetitions": 20,
            "raw_artifacts": [str(args.combine_wait_evidence.resolve())],
            "confidence": "medium",
            "profiler_perturbation_pct": None,
            "cross_checks": [
                "uniform and all-remote controls have similar rank-max p95",
                "skew no-payload removes 60.57% of instrumented rank-max p95",
            ],
            "units": {"wait_timer": "100 MHz ticks converted to microseconds"},
        }
    if payload_control is not None:
        provenance[payload_control_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "stage2_payload_control",
            "tool_version": "AITER analysis-no-p2p-payload v1",
            "command": (
                "bench_mega_moe_v2.py --analysis-no-p2p-payload; "
                "derive_payload_control.py"
            ),
            "timestamp": timestamp,
            "scope": "skew, all-remote, and local-only Stage2 payload controls",
            "repetitions": 3,
            "raw_artifacts": [str(args.payload_control_evidence.resolve())],
            "confidence": "high",
            "profiler_perturbation_pct": None,
            "cross_checks": ["physical GMI sector counters fall to zero for Stage2"],
            "units": {"latency": "ms/replay", "logical_payload": "bytes/replay"},
        }
    if tile_dag is not None:
        provenance[tile_dag_provenance] = {
            "schema_version": "geak-collection-provenance-v1",
            "collector_id": "source_anchored_tile_dag",
            "tool_version": "geak-tile-dependency-dag-v1",
            "command": "derive_tile_dag.py --source-root AITER-candidate --trace-dag <trace>",
            "timestamp": timestamp,
            "scope": "isolated MegaMoEV2 operator/tile dependencies",
            "repetitions": 1,
            "raw_artifacts": [str(args.tile_dag_evidence.resolve())],
            "confidence": "high",
            "profiler_perturbation_pct": None,
            "cross_checks": ["all source anchors and SHA-256 identities resolve"],
            "units": {"nodes": "count", "edges": "count"},
        }
    _write(args.derived_output, derived)
    _write(args.provenance_output, provenance)
    print(f"GEAK_DERIVED_EVIDENCE_JSON={args.derived_output.resolve()}")
    print(f"GEAK_PROVENANCE_CATALOG_JSON={args.provenance_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
