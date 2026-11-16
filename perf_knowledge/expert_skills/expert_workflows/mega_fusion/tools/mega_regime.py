#!/usr/bin/env python3
"""Regime map for a grouped-GEMM (MoE) operator: which resource bounds each guard's shape.

The same operator behaves like different problems at different token counts. At a few tokens per
rank every expert's weights are still read once, so the time is a weight stream plus fixed costs.
At thousands of tokens the matrix math and the cross-rank traffic dominate. A plan that is right
for one size can be wrong for the other, so the first planning step for every guard is to place
it on this map, from the shape and the measured baseline, before choosing what to build.

For each guard ``<tokens>_<route>[_mtpr<M>]`` the tool reports, per rank:

  * rows per expert (uniform-route average) and the padding efficiency of common row tiles;
  * weight bytes touched, matrix FLOPs and cross-rank bytes;
  * lower bounds for each resource (weight stream, math, links) and which one binds;
  * when a measured baseline time is given: how far above the binding bound it runs, and the
    part no single roofline explains (``unexplained_share``: fixed cost, latency, inefficiency);
  * a regime label and the lever *families* that regime responds to (principles, not recipes).

Guards whose labels agree form one bucket. A bucket is a hypothesis to measure, never a result:
the bounds are only as good as the hardware figures passed in. Prefer measured figures (for
example a copy kernel's achieved bandwidth) over datasheet peaks and say which you used.

All inputs are explicit command-line values; nothing about a particular kernel is built in.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys

SCHEMA = "geak-mega-regime-v1"
GUARD_RE = re.compile(r"^(?P<tokens>\d+)_(?P<route>[A-Za-z0-9-]+?)(?:_mtpr(?P<mtpr>\d+))?$")
ROW_TILES = (16, 32, 64, 128, 256)

LEVER_FAMILIES = {
    "weight_stream": [
        "the weight bytes are fixed per call: keep the stream at the achieved bandwidth "
        "(enough loads in flight on every compute unit, wide coalesced loads)",
        "do not compute or load padding rows; size row tiles to the rows an expert really has",
        "hide everything else (launches, host or device syncs, cross-rank handshakes, "
        "scheduling) under the weight stream, or remove it",
        "math-throughput tuning (bigger tiles, more overlap of matrix stages) buys little here",
    ],
    "compute": [
        "matrix-unit utilization: tile shapes per bucket, occupancy, operand reuse",
        "overlap independent stages so the matrix units never idle on a dependency",
        "padding still costs math: check the tile efficiency column",
    ],
    "communication": [
        "overlap cross-rank transfers with math",
        "send fewer bytes (narrower payload where accuracy allows) and fewer rounds",
        "balance per-rank traffic; the slowest rank sets the time",
    ],
    "fixed_cost": [
        "the time is mostly not explained by any roofline: count and shrink fixed costs first "
        "(launches, syncs, handshakes, the latency until the first useful work starts)",
        "measure the critical path of one call before tuning any throughput",
    ],
}


def parse_guard(guard):
    match = GUARD_RE.match(str(guard).strip())
    if not match:
        raise ValueError(f"guard {guard!r} is not <tokens>_<route>[_mtpr<M>]")
    tokens = int(match["tokens"])
    mtpr = int(match["mtpr"]) if match["mtpr"] else None
    return {"guard": str(guard).strip(), "tokens": tokens, "route": match["route"], "mtpr": mtpr}


def expected_touched(experts, pairs):
    """Expected distinct experts hit by ``pairs`` independent uniform choices among ``experts``."""
    if experts <= 0:
        return 0.0
    return experts * (1.0 - (1.0 - 1.0 / experts) ** max(0.0, pairs))


def tile_efficiency(rows_per_expert, touched, tile):
    """Useful rows / computed rows when each touched expert pads its rows up to ``tile``."""
    if touched <= 0 or rows_per_expert <= 0:
        return None
    per_expert = rows_per_expert
    padded = math.ceil(per_expert / tile) * tile
    return per_expert / padded


def classify(bounds, measured_us, unexplained_threshold, balance_ratio):
    ordered = sorted(bounds.items(), key=lambda kv: kv[1], reverse=True)
    binding, top = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    labels = [binding]
    if top > 0 and second >= balance_ratio * top:
        labels.append(ordered[1][0])
    unexplained = None
    if measured_us:
        unexplained = max(0.0, measured_us - top) / measured_us
        if unexplained >= unexplained_threshold:
            labels.insert(0, "fixed_cost")
    return binding, labels, unexplained


def analyse(args):
    hw = {
        "hbm_tbps": args.hbm_tbps,
        "math_tflops": args.math_tflops,
        "link_gbps": args.link_gbps,
        "source": args.hardware_source,
    }
    params_per_expert = args.params_per_expert or (
        args.gemm_matrices * args.hidden * args.inter if args.hidden and args.inter else 0)
    if params_per_expert <= 0:
        raise ValueError("give --params-per-expert or --hidden/--inter/--gemm-matrices")
    weight_bytes = args.weight_bytes_per_rank or (
        args.experts_per_rank * params_per_expert * args.weight_bits / 8.0)
    measured = {}
    for item in args.baseline_ms or []:
        guard, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--baseline-ms expects GUARD=MS, got {item!r}")
        measured[guard.strip()] = float(value)
    rows = []
    for guard in args.guards:
        g = parse_guard(guard)
        pairs = g["tokens"] * args.topk  # rows arriving per rank under a balanced route
        rows_per_expert = pairs / args.experts_per_rank
        touched = expected_touched(args.experts_per_rank, pairs)
        touched_bytes = weight_bytes * touched / args.experts_per_rank
        flops = 2.0 * pairs * params_per_expert
        remote = (args.world - 1) / args.world if args.world > 1 else 0.0
        link_bytes = g["tokens"] * args.topk * args.hidden * remote * (
            args.dispatch_bytes + args.combine_bytes)
        bounds = {
            "weight_stream": touched_bytes / (args.hbm_tbps * 1e12) * 1e6,
            "compute": flops / (args.math_tflops * 1e12) * 1e6,
        }
        if args.link_gbps > 0 and args.hidden:
            bounds["communication"] = link_bytes / (args.link_gbps * 1e9) * 1e6
        base_ms = measured.get(g["guard"])
        base_us = base_ms * 1e3 if base_ms else None
        binding, labels, unexplained = classify(
            bounds, base_us, args.unexplained_threshold, args.balance_ratio)
        row = {
            **g,
            "rows_per_rank": pairs,
            "rows_per_expert": rows_per_expert,
            "experts_touched": touched,
            "tile_efficiency": {str(t): tile_efficiency(rows_per_expert, touched, t)
                                for t in ROW_TILES},
            "weight_bytes_touched": touched_bytes,
            "matrix_flops": flops,
            "arithmetic_intensity_flop_per_weight_byte": flops / touched_bytes if touched_bytes else None,
            "link_bytes": link_bytes if "communication" in bounds else None,
            "bounds_us": bounds,
            "binding_resource": binding,
            "regime": "+".join(labels),
            "lever_families": [line for label in labels for line in LEVER_FAMILIES[label]],
        }
        if base_us:
            row.update(
                baseline_ms=base_ms,
                baseline_over_bound=base_us / bounds[binding] if bounds[binding] else None,
                unexplained_share=unexplained,
            )
        rows.append(row)
    buckets = {}
    for row in rows:
        buckets.setdefault(row["regime"], []).append(row["guard"])
    return {
        "schema_version": SCHEMA,
        "hardware": hw,
        "model": {
            "experts_per_rank": args.experts_per_rank, "topk": args.topk, "world": args.world,
            "hidden": args.hidden, "inter": args.inter, "params_per_expert": params_per_expert,
            "weight_bytes_per_rank": weight_bytes, "dispatch_bytes": args.dispatch_bytes,
            "combine_bytes": args.combine_bytes,
            "route_model": "balanced route: tokens*topk rows per rank, experts hit as independent "
                           "uniform draws",
        },
        "guards": rows,
        "buckets": [{"regime": k, "guards": v} for k, v in buckets.items()],
        "caveats": [
            "bounds assume the resource runs at the given figure; datasheet peaks overstate it",
            "a skewed route concentrates rows on fewer experts and ranks; read skew guards as "
            "the uniform shape plus imbalance, and measure them",
            "buckets are hypotheses to test with the paired A/B, not results",
        ],
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--guards", nargs="+", required=True, help="guard ids <T>_<route>[_mtpr<M>]")
    p.add_argument("--experts-per-rank", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--hidden", type=int, default=0, help="model dimension (rows of the token)")
    p.add_argument("--inter", type=int, default=0, help="expert intermediate dimension")
    p.add_argument("--gemm-matrices", type=int, default=3,
                   help="hidden x inter matrices per expert (gate+up+down = 3)")
    p.add_argument("--params-per-expert", type=float, default=0)
    p.add_argument("--weight-bits", type=float, default=16.0,
                   help="stored bits per weight including scales (e.g. 4.25 for 4-bit + 1/32 scale)")
    p.add_argument("--weight-bytes-per-rank", type=float, default=0,
                   help="measured weight bytes on one rank; overrides the estimate")
    p.add_argument("--dispatch-bytes", type=float, default=1.0, help="bytes per element sent")
    p.add_argument("--combine-bytes", type=float, default=2.0, help="bytes per element returned")
    p.add_argument("--hbm-tbps", type=float, required=True)
    p.add_argument("--math-tflops", type=float, required=True)
    p.add_argument("--link-gbps", type=float, default=0.0,
                   help="per-rank cross-rank bandwidth; 0 omits the communication bound")
    p.add_argument("--hardware-source", default="unspecified",
                   help="where the figures come from: measured (how) or datasheet")
    p.add_argument("--baseline-ms", nargs="*", default=[], help="GUARD=MS measured rank-max e2e")
    p.add_argument("--unexplained-threshold", type=float, default=0.30)
    p.add_argument("--balance-ratio", type=float, default=0.75)
    p.add_argument("--json-output", default="")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name in ("hbm_tbps", "math_tflops", "experts_per_rank", "topk", "world"):
        if not getattr(args, name) or getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    report = analyse(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        with open(args.json_output, "w") as f:
            f.write(text + "\n")
    for row in report["guards"]:
        extra = ""
        if row.get("baseline_ms"):
            extra = (f" baseline={row['baseline_ms']:.4f}ms x{row['baseline_over_bound']:.2f} of bound"
                     f" unexplained={row['unexplained_share']:.0%}")
        bounds = " ".join(f"{k}={v:.1f}us" for k, v in row["bounds_us"].items())
        print(f"{row['guard']}: rows/expert={row['rows_per_expert']:.2f} {bounds} "
              f"regime={row['regime']}{extra}")
    if not args.json_output:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
