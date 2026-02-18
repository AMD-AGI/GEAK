"""Format kernel-profile (MetrixTool) output into baseline_metrics.json for OpenEvolve.

This module is a **formatting layer**, not a decision-making layer.
Kernel selection — which kernel(s) are relevant to the optimisation task —
is the LLM agent's responsibility.  The agent reads the full profiler output,
reasons about which kernels matter, and passes those choices here.

Profiling itself is not limited to baseline extraction.  kernel-profile can
be invoked at any point: before optimisation (baseline), after optimisation
(validation), or by OpenEvolve during evolution (via COMMANDMENT.md PROFILE
section).  This module only handles the *formatting* of profiler output into
the JSON structure that ``run_openevolve.py --baseline-metrics`` expects.

Usage (Python — agent calls this after choosing kernels):
    from minisweagent.baseline_metrics import build_baseline_metrics, format_for_openevolve

    profiler_output = tool.profile(command=..., ...)
    all_kernels = list_kernels(profiler_output)  # show to agent
    # Agent picks kernel indices/names based on the task:
    baseline = build_baseline_metrics(profiler_output, kernel_names=["topk_stage1", "topk_stage2"])
    # Or by indices:
    baseline = build_baseline_metrics(profiler_output, kernel_indices=[0, 2])
    write_json(baseline, "baseline_metrics.json")

Usage (CLI):
    # List all kernels so the agent can choose
    python -m minisweagent.baseline_metrics list profiler_output.json

    # Build baseline from agent-chosen kernels
    python -m minisweagent.baseline_metrics build profiler_output.json --kernels "topk_stage1,topk_stage2"
    python -m minisweagent.baseline_metrics build profiler_output.json --indices 0,2
    python -m minisweagent.baseline_metrics build profiler_output.json --all  # include every kernel
"""

import json
import math
import sys
from pathlib import Path

# Metrics where summation is the correct aggregation (total cost).
_SUM_METRICS = {"duration_us"}
# All other numeric metrics use duration-weighted averaging.


def _sanitize_value(v):
    """Replace NaN/inf floats with None (JSON-safe) before serialization."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, dict):
        return {k: _sanitize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_sanitize_value(item) for item in v]
    return v


def list_kernels(metrixtool_result: dict, gpu_index: int = 0) -> list[dict]:
    """Return all kernels from a MetrixTool.profile() result, unchanged.

    This is a convenience accessor — the agent reads this listing, reasons
    about the task, and decides which kernels to include.

    Args:
        metrixtool_result: Dict from ``MetrixTool.profile()``.
        gpu_index: Which GPU result to read (default: 0).

    Returns:
        List of kernel dicts, each with keys:
        ``name``, ``duration_us``, ``metrics``, ``bottleneck``, ``observations``.

    Raises:
        ValueError: If the result structure is unexpected.
    """
    results = metrixtool_result.get("results")
    if not results or not isinstance(results, list):
        raise ValueError(
            "MetrixTool result missing 'results' list. "
            f"Got top-level keys: {list(metrixtool_result.keys())}"
        )
    if gpu_index >= len(results):
        raise ValueError(
            f"gpu_index={gpu_index} but only {len(results)} GPU result(s) available."
        )
    return metrixtool_result["results"][gpu_index].get("kernels", [])


def aggregate_metrics(kernels: list[dict]) -> dict[str, float]:
    """Aggregate metrics across multiple kernel dicts.

    - ``duration_us`` is **summed** (total wall-time of the kernel group).
    - All other numeric metrics are **duration-weighted averages** so that
      longer-running kernels contribute proportionally more.

    Args:
        kernels: List of kernel dicts (each must have a ``metrics`` sub-dict).

    Returns:
        Flat dict of metric name → aggregated value.
    """
    if not kernels:
        return {}
    if len(kernels) == 1:
        return dict(kernels[0].get("metrics", {}))

    total_duration = sum(
        k.get("duration_us", k.get("metrics", {}).get("duration_us", 0))
        for k in kernels
    )

    all_keys: set[str] = set()
    for k in kernels:
        all_keys.update(k.get("metrics", {}).keys())

    aggregated: dict[str, float] = {}
    for key in sorted(all_keys):
        if key in _SUM_METRICS:
            raw = sum(k.get("metrics", {}).get(key, 0) for k in kernels)
        elif total_duration > 0:
            weighted = sum(
                k.get("metrics", {}).get(key, 0)
                * k.get("duration_us", k.get("metrics", {}).get("duration_us", 0))
                for k in kernels
            )
            raw = weighted / total_duration
        else:
            values = [k.get("metrics", {}).get(key, 0) for k in kernels]
            raw = sum(values) / len(values)

        # Sanitize NaN/inf to None for JSON compatibility
        if isinstance(raw, float) and (math.isnan(raw) or math.isinf(raw)):
            aggregated[key] = None
        else:
            aggregated[key] = raw

    return aggregated


def build_baseline_metrics(
    metrixtool_result: dict,
    *,
    kernel_names: list[str] | None = None,
    kernel_indices: list[int] | None = None,
    include_all: bool = False,
    gpu_index: int = 0,
) -> dict:
    """Build a baseline_metrics dict from agent-chosen kernels.

    The agent must specify which kernels to include via exactly one of:
    - ``kernel_names``: list of exact kernel name strings
    - ``kernel_indices``: list of 0-based indices into the kernel list
    - ``include_all=True``: use every kernel (useful when only one exists)

    Args:
        metrixtool_result: Dict from ``MetrixTool.profile()``.
        kernel_names: Kernel names to include (exact match).
        kernel_indices: Kernel indices to include (0-based).
        include_all: If True, include all kernels.
        gpu_index: Which GPU result to read.

    Returns:
        Dict ready to be written as ``baseline_metrics.json``::

            {
                "duration_us": float,        # summed if multiple kernels
                "kernel_name": str,          # single name or "name1+N"
                "kernel_names": [str, ...],  # all included kernel names
                "metrics": {str: float},     # aggregated hardware metrics
                "bottleneck": str,           # from the longest-duration kernel
                "observations": [str, ...],  # merged, deduplicated
            }

    Raises:
        ValueError: If selection args are ambiguous, or no kernels match.
    """
    # --- Validate exactly one selection mode ---
    modes = sum([kernel_names is not None, kernel_indices is not None, include_all])
    if modes == 0:
        raise ValueError(
            "Specify how to select kernels: kernel_names=[...], "
            "kernel_indices=[...], or include_all=True."
        )
    if modes > 1:
        raise ValueError(
            "Specify only one of kernel_names, kernel_indices, or include_all."
        )

    all_kernels = list_kernels(metrixtool_result, gpu_index=gpu_index)
    if not all_kernels:
        raise ValueError("No kernels found in profiling results.")

    # --- Select kernels ---
    if include_all:
        selected = list(all_kernels)
    elif kernel_indices is not None:
        for idx in kernel_indices:
            if idx < 0 or idx >= len(all_kernels):
                raise ValueError(
                    f"Kernel index {idx} out of range (0..{len(all_kernels)-1}). "
                    f"Available: {[k['name'] for k in all_kernels]}"
                )
        selected = [all_kernels[i] for i in kernel_indices]
    else:
        assert kernel_names is not None
        name_set = set(kernel_names)
        selected = [k for k in all_kernels if k["name"] in name_set]
        found_names = {k["name"] for k in selected}
        missing = name_set - found_names
        if missing:
            available = [k["name"] for k in all_kernels]
            raise ValueError(
                f"Kernel(s) not found: {sorted(missing)}. "
                f"Available: {available}"
            )

    if not selected:
        raise ValueError("No kernels selected.")

    return _format_baseline(selected)


def _format_baseline(selected: list[dict]) -> dict:
    """Format selected kernel(s) into the baseline_metrics.json structure."""
    # Sort by duration descending for consistent dominant-kernel ordering
    selected = sorted(
        selected,
        key=lambda k: k.get("duration_us", k.get("metrics", {}).get("duration_us", 0)),
        reverse=True,
    )

    aggregated = aggregate_metrics(selected)
    dominant = selected[0]

    if len(selected) == 1:
        kernel_name = dominant["name"]
    else:
        kernel_name = f"{dominant['name']}+{len(selected) - 1}"

    # Merge observations (deduplicated, order-preserving)
    seen: set[str] = set()
    observations: list[str] = []
    for k in selected:
        for obs in k.get("observations", []):
            if obs not in seen:
                seen.add(obs)
                observations.append(obs)

    result = {
        "duration_us": aggregated.get("duration_us", 0),
        "kernel_name": kernel_name,
        "kernel_names": [k["name"] for k in selected],
        "metrics": aggregated,
        "bottleneck": dominant.get("bottleneck", "unknown"),
        "observations": observations,
    }
    # Sanitize NaN/inf values to ensure valid JSON output
    return _sanitize_value(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Format kernel-profile output for OpenEvolve",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- list ---
    p_list = sub.add_parser("list", help="List all profiled kernels (for the agent to inspect)")
    p_list.add_argument("input", nargs="?", default=None, help="MetrixTool JSON output (or '-' for stdin)")
    p_list.add_argument("--from-profile", default=None, metavar="FILE", help="Read profile JSON from kernel-profile output")
    p_list.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")

    # --- build ---
    p_build = sub.add_parser("build", help="Build baseline_metrics.json from chosen kernels")
    p_build.add_argument("input", nargs="?", default=None, help="MetrixTool JSON output (or '-' for stdin)")
    p_build.add_argument("--from-profile", default=None, metavar="FILE", help="Read profile JSON from kernel-profile output")
    p_build.add_argument("-o", "--output", default=None, help="Output path (default: stdout)")
    p_build.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    sel = p_build.add_mutually_exclusive_group(required=True)
    sel.add_argument("--kernels", help="Comma-separated kernel names to include")
    sel.add_argument("--indices", help="Comma-separated kernel indices to include")
    sel.add_argument("--all", action="store_true", dest="include_all", help="Include all kernels")

    args = parser.parse_args()

    # Resolve input: --from-profile takes precedence, then positional, then stdin
    input_source = args.from_profile or args.input
    if not input_source:
        parser.error("input is required (positional, --from-profile, or '-' for stdin)")

    # Load input
    if input_source == "-":
        data = json.load(sys.stdin)
    else:
        data = json.loads(Path(input_source).read_text())

    if args.command == "list":
        kernels = list_kernels(data, gpu_index=args.gpu)
        if not kernels:
            print("No kernels found.", file=sys.stderr)
            sys.exit(1)
        print(f"{'Idx':<4} {'Duration(µs)':<14} {'Bottleneck':<12} Name")
        print("-" * 70)
        for i, k in enumerate(kernels):
            dur = k.get("duration_us", k.get("metrics", {}).get("duration_us", 0))
            bn = k.get("bottleneck", "?")
            print(f"{i:<4} {dur:<14.2f} {bn:<12} {k['name']}")

    elif args.command == "build":
        kwargs: dict = {"gpu_index": args.gpu}
        if args.include_all:
            kwargs["include_all"] = True
        elif args.kernels:
            kwargs["kernel_names"] = [n.strip() for n in args.kernels.split(",")]
        elif args.indices:
            kwargs["kernel_indices"] = [int(i.strip()) for i in args.indices.split(",")]

        baseline = build_baseline_metrics(data, **kwargs)
        output_json = json.dumps(baseline, indent=2)

        if args.output:
            Path(args.output).write_text(output_json + "\n")
            print(f"Wrote {args.output}", file=sys.stderr)
        else:
            print(output_json)


if __name__ == "__main__":
    main()
