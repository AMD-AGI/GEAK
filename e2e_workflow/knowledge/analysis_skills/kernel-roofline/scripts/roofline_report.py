#!/usr/bin/env python3
"""Extended mode: report a frozen roofline from explicit models/timings.

Uses only the Python standard library. Does not collect GPU measurements or import
TraceLens. See references/report-input.md for units and required provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def required_text(obj, key):
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def metrics(flops, nbytes, latency_us, peak_tflops, bandwidth_gbs):
    compute_us = flops / (peak_tflops * 1e6)
    memory_us = nbytes / (bandwidth_gbs * 1e3)
    roof_us = max(compute_us, memory_us)
    if math.isclose(compute_us, memory_us, rel_tol=1e-12, abs_tol=0):
        axis = "balanced"
    else:
        axis = "compute" if compute_us > memory_us else "memory"
    return {
        "flops_per_call": flops,
        "bytes_per_call": nbytes,
        "ai_flops_per_byte": flops / nbytes if nbytes else None,
        "ridge_flops_per_byte": peak_tflops * 1e3 / bandwidth_gbs,
        "compute_reference_us": compute_us,
        "memory_reference_us": memory_us,
        "reference_us": roof_us,
        "model_roof_axis": axis,
        "effective_tflops": flops / (latency_us * 1e6) if latency_us else None,
        "effective_gbs": nbytes / (latency_us * 1e3) if latency_us else None,
        "reference_efficiency": roof_us / latency_us if latency_us else None,
        "distance_to_reference": latency_us / roof_us if latency_us and roof_us else None,
    }


def counter_metrics(counter, model, peak, bw):
    time = number(counter.get("latency_us"), "counter.latency_us", positive=True)
    required_text(counter, "source")
    required_text(counter, "coverage")
    required_text(counter, "timing_basis")
    f, b = counter.get("flops"), counter.get("bytes")
    if f is not None:
        number(f, "counter.flops")
    if b is not None:
        number(b, "counter.bytes")
    result = {
        "counter_latency_us": time,
        "counter_flops_per_call": f,
        "counter_bytes_per_call": b,
        "executed_tflops": f / (time * 1e6) if f is not None else None,
        "external_gbs": b / (time * 1e3) if b is not None else None,
        "flop_expansion": f / model["flops"] if f is not None and model["flops"] else None,
        "byte_expansion": b / model["bytes"] if b is not None and model["bytes"] else None,
        "counter_ai_flops_per_byte": None,
        "counter_reference_us": None,
        "counter_roof_axis": None,
        "counter_reference_efficiency": None,
    }
    # A partial counter collection cannot establish the two-resource roof.
    if f is not None and b is not None and f + b > 0:
        m = metrics(f, b, time, peak, bw)
        result.update({
            "counter_ai_flops_per_byte": m["ai_flops_per_byte"],
            "counter_reference_us": m["reference_us"],
            "counter_roof_axis": m["model_roof_axis"],
            "counter_reference_efficiency": m["reference_efficiency"],
        })
    return result


def build_rows(data, arch):
    if data.get("mode", "extended") != "extended":
        raise ValueError("this calculator supports only extended mode; use TraceLens native reporting for tl-only")
    if data.get("schema_version") != 1:
        raise ValueError("input schema_version must be 1")
    if not isinstance(data.get("provenance"), dict) or not data["provenance"]:
        raise ValueError("input provenance must be a nonempty object")
    bw = number(arch.get("mem_bw_gbps"), "mem_bw_gbps", positive=True)
    peaks = arch.get("max_achievable_tflops", {})
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    rows, seen = [], set()
    for case in cases:
        cid = required_text(case, "case_id")
        if cid in seen:
            raise ValueError(f"duplicate case_id: {cid}")
        seen.add(cid)
        contract = case["contract"]
        for key in ("id", "operator", "boundary", "precision", "memory_level", "cache_policy", "model_version"):
            required_text(contract, key)
        if not isinstance(case.get("workload"), dict) or not case["workload"]:
            raise ValueError(f"{cid}: workload must describe this case")
        model = case["model"]
        kind = required_text(model, "kind")
        if kind not in ("semantic_reference", "legacy_tl", "implementation_model"):
            raise ValueError(f"{cid}: unsupported model kind {kind}")
        required_text(model, "source")
        if not isinstance(model.get("assumptions"), list) or not model["assumptions"]:
            raise ValueError(f"{cid}: model assumptions must be a nonempty list")
        f = number(model.get("flops"), "model.flops")
        b = number(model.get("bytes"), "model.bytes")
        if f + b == 0:
            raise ValueError(f"{cid}: both FLOPs and bytes are zero; model is unavailable")
        key = required_text(model, "peak_key")
        if key not in peaks:
            raise ValueError(f"{cid}: peak {key!r} absent; available: {sorted(peaks)}")
        peak = number(peaks[key], f"peak {key}", positive=True)
        measurements = case.get("measurements", [])
        if not isinstance(measurements, list):
            raise ValueError(f"{cid}: measurements must be a list")
        implementations = set()
        for measurement in measurements or [None]:
            flags = []
            if measurement is None:
                impl, latency, correctness, timing = "model-only", None, "not_checked", "unavailable"
                flags.append("model_only")
            else:
                impl = required_text(measurement, "implementation")
                if impl in implementations:
                    raise ValueError(f"{cid}: duplicate implementation {impl}")
                implementations.add(impl)
                latency = number(measurement.get("latency_us"), "latency_us", positive=True)
                timing = required_text(measurement, "timing_basis")
                required_text(measurement, "source")
                correctness = measurement.get("correctness")
                if correctness not in ("pass", "fail", "not_checked"):
                    raise ValueError("correctness must be pass, fail, or not_checked")
                if correctness != "pass":
                    flags.append("correctness_" + correctness)
            row = {
                "mode": "extended",
                "case_id": cid, "contract_id": contract["id"], "model_kind": kind,
                "implementation": impl, "peak_key": key, "peak_tflops": peak,
                "bandwidth_gbs": bw, "latency_us": latency, "timing_basis": timing,
                "correctness": correctness,
                **metrics(f, b, latency, peak, bw),
            }
            if row["reference_efficiency"] is not None and row["reference_efficiency"] > 1:
                flags.append("reference_exceeded_investigate")
            if kind != "semantic_reference":
                flags.append("not_a_fixed_semantic_reference")
            if measurement is not None and measurement.get("counters") is not None:
                row.update(counter_metrics(measurement["counters"], model, peak, bw))
                efficiency = row["counter_reference_efficiency"]
                if efficiency is not None and efficiency > 1:
                    flags.append("counter_reference_exceeded_investigate")
            row["flags"] = flags
            rows.append(row)
    return rows


def cell(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        value = "; ".join(value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(rows, fields):
    return ["| " + " | ".join(label for _, label in fields) + " |",
            "| " + " | ".join("---" for _ in fields) + " |"] + [
        "| " + " | ".join(cell(row.get(key)) for key, _ in fields) + " |" for row in rows]


def write_report(output, data, arch, arch_hash, rows):
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "mode": "extended", "arch_sha256": arch_hash,
              "arch_snapshot": arch, "input_snapshot": data, "rows": rows}
    (output / "roofline.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "roofline.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({**r, "flags": ";".join(r["flags"])} for r in rows)
    lines = ["# Kernel roofline report", "", "Mode: **extended**. Engine: bundled report calculator (not TraceLens native reporting).", "", f"Frozen architecture SHA-256: `{arch_hash}`.", "",
             "FLOPs/bytes come from the supplied model. Latencies and correctness are supplied",
             "observations; this helper does not measure or validate a kernel. Efficiency values",
             "are fractions (0.5 = 50%). Axis labels are model classifications, not measured stalls.", ""]
    if data.get("provenance", {}).get("synthetic_example"):
        lines += ["**Synthetic example: these timings and peaks are not GPU measurements.**", ""]
    lines += table(rows, [("case_id", "Case"), ("implementation", "Implementation"),
        ("model_kind", "Model basis"), ("flops_per_call", "FLOPs/call"),
        ("bytes_per_call", "Bytes/call"), ("latency_us", "Time us"),
        ("effective_tflops", "Effective TFLOP/s"), ("effective_gbs", "Effective GB/s"),
        ("ai_flops_per_byte", "AI FLOP/byte")])
    lines += ["", "## Reference and validity", ""]
    lines += table(rows, [("case_id", "Case"), ("implementation", "Implementation"),
        ("compute_reference_us", "Compute us"), ("memory_reference_us", "Memory us"),
        ("reference_us", "Reference us"), ("model_roof_axis", "Axis"),
        ("reference_efficiency", "Efficiency fraction"), ("correctness", "Correctness"),
        ("flags", "Flags")])
    counters = [r for r in rows if "counter_latency_us" in r]
    if counters:
        lines += ["", "## Counter diagnostics", "",
                  "Counter rates use their own supplied timing and the same selected hardware references.",
                  "Missing counter values remain unavailable. Coverage and pass provenance are in JSON.", ""]
        lines += table(counters, [("case_id", "Case"), ("implementation", "Implementation"),
            ("counter_flops_per_call", "Counter FLOPs/call"), ("counter_bytes_per_call", "External bytes/call"),
            ("counter_latency_us", "Counter time us"), ("executed_tflops", "Executed TFLOP/s"),
            ("external_gbs", "External GB/s"), ("counter_ai_flops_per_byte", "Counter AI"),
            ("counter_roof_axis", "Counter axis"), ("counter_reference_efficiency", "Counter efficiency fraction")])
    lines += ["", "## Provenance and limitations", "",
              "The JSON includes the full model contract, workload, assumptions, measurement sources,",
              "and architecture snapshot. Review these before interpreting distance to reference as",
              "optimization headroom. No attainable or end-to-end speedup is inferred automatically.", ""]
    (output / "roofline.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["extended"], default="extended",
                        help="This helper implements extended mode only; tl-only uses TraceLens native reporting.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--arch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text())
        arch_bytes = args.arch.read_bytes()
        arch = json.loads(arch_bytes)
        rows = build_rows(data, arch)
        write_report(args.output, data, arch, hashlib.sha256(arch_bytes).hexdigest(), rows)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        parser.exit(2, f"roofline_report: {exc}\n")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
