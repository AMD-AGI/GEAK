#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI wrapper: aiperf ``profile_export_aiperf.json`` -> canonical result JSON.

Vendored from InferenceX ``benchmarks/map_aiperf.py`` so a standalone GEAK run
needs no InferenceX checkout. That checkout is a Hyperloom deployment artifact:
under the orchestrator its AgentX runtime copies this file into
``$INFERENCEX_PATH/benchmarks/``, but standalone nothing does, and the client
adapter used to abort with "map_aiperf.py not found" AFTER it had already
launched and warmed a server -- a ~20 minute detour to reach a one-line error.

Resolution order is unchanged and this copy is LAST (see agentx.sh): a real
InferenceX checkout still wins, so an orchestrated run keeps mapping its result
with the exact file its own runtime deployed, and this only fills the gap when
there is nothing to defer to.

The mapping is carried inline rather than imported from the orchestrator's
package, because GEAK takes no dependency on Hyperloom. The upstream inline
copy predates the interactivity fields, so this one also carries the
percentiles ``bench_summarize.py`` grades the interactivity axes on:
``p90_tpot_ms`` (InferenceX P90 interactivity is ``1000 / p90_tpot_ms``) and
``e2e_norm_intvty_p90`` / ``e2e_norm_intvty_p50`` (aiperf's P10 / P50 of the
per-request rate OSL/E2EL_s). Field names follow the InferenceX result schema,
so a row reads the same whichever mapper produced it.
"""

import json
import os
import sys


def _noncanonical_reasons():
    """Workload deviations the client detected; see agentx.sh.

    aiperf cannot judge these -- it has no concept of corpus size, and it only
    stamps a False verdict when --unsafe-override actually suppressed a
    violation -- so the client passes them here and the verdict is forced.
    """
    raw = (os.environ.get("AGENTX_NONCANONICAL_REASONS") or "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else []


def _stat(m, key, sub="avg", default=0.0):
    v = m.get(key)
    if isinstance(v, dict):
        return v.get(sub, v.get("avg", default))
    return v if v is not None else default


def _pct(m, key, sub):
    """``m[key][sub]``, or None. No fallback to the mean: a graded percentile that
    silently became an average would be read as the percentile it is named after."""
    v = m.get(key)
    if isinstance(v, dict):
        sv = v.get(sub)
        if isinstance(sv, (int, float)) and not isinstance(sv, bool):
            return sv
    return None


def _submission_outcome(export):
    # Tri-state: True / False / None(absent). Absent is NOT valid -- it means
    # no --scenario was requested or the aiperf build predates the field.
    md = export.get("metadata")
    if not isinstance(md, dict) or "submission_valid" not in md:
        return None, []
    reasons = md.get("submission_invalid_reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return bool(md.get("submission_valid")), [str(r) for r in reasons]


def map_aiperf(export, *, noncanonical_reasons=None):
    d = export
    _verdict, _reasons = _submission_outcome(d)
    _extra = [str(r) for r in (noncanonical_reasons or []) if str(r).strip()]
    if _extra:
        _verdict = False
        _reasons = [*_reasons, *_extra]
    m = d if ("time_to_first_token" in d or "output_token_throughput" in d) else d.get("metrics", d)
    out_tput = _stat(m, "output_token_throughput")
    in_tput = _stat(m, "input_token_throughput")
    total_tput = _stat(m, "total_token_throughput") or ((in_tput or 0) + (out_tput or 0))
    rc = int(_stat(m, "request_count") or 0)
    isl = _stat(m, "input_sequence_length")
    return {
        "request_throughput": _stat(m, "request_throughput"),
        "output_throughput": out_tput,
        "input_throughput": in_tput,
        "total_token_throughput": total_tput,
        "completed": rc,
        "total_input_tokens": int(_stat(m, "total_isl") or (isl * max(1, rc)) or 0),
        "total_output_tokens": int(_stat(m, "total_output_tokens") or _stat(m, "total_osl") or 0),
        "duration": _stat(m, "benchmark_duration"),
        "mean_ttft_ms": _stat(m, "time_to_first_token", "avg"),
        "median_ttft_ms": _stat(m, "time_to_first_token", "p50"),
        "p90_ttft_ms": _pct(m, "time_to_first_token", "p90"),
        "p99_ttft_ms": _stat(m, "time_to_first_token", "p99"),
        "std_ttft_ms": _stat(m, "time_to_first_token", "std"),
        "mean_tpot_ms": _stat(m, "inter_token_latency", "avg"),
        "median_tpot_ms": _stat(m, "inter_token_latency", "p50"),
        "p90_tpot_ms": _pct(m, "inter_token_latency", "p90"),
        "p99_tpot_ms": _stat(m, "inter_token_latency", "p99"),
        "std_tpot_ms": _stat(m, "inter_token_latency", "std"),
        # The P90 interactivity is the slow-tail request's own rate, i.e. the P10
        # of the per-request rate; the median needs no such inversion.
        "e2e_norm_intvty_p90": _pct(m, "e2e_output_token_throughput", "p10"),
        "e2e_norm_intvty_p50": _pct(m, "e2e_output_token_throughput", "p50"),
        "mean_itl_ms": _stat(m, "inter_token_latency", "avg"),
        "median_itl_ms": _stat(m, "inter_token_latency", "p50"),
        "p99_itl_ms": _stat(m, "inter_token_latency", "p99"),
        "std_itl_ms": _stat(m, "inter_token_latency", "std"),
        "mean_e2el_ms": _stat(m, "request_latency", "avg"),
        "median_e2el_ms": _stat(m, "request_latency", "p50"),
        "p99_e2el_ms": _stat(m, "request_latency", "p99"),
        "std_e2el_ms": _stat(m, "request_latency", "std"),
        "theoretical_prefix_cache_hit": _stat(m, "theoretical_prefix_cache_hit"),
        "submission_valid": _verdict,
        "submission_invalid_reasons": _reasons,
    }


def main(src, dst):
    with open(src) as f:
        data = json.load(f)
    res = map_aiperf(data, noncanonical_reasons=_noncanonical_reasons())
    with open(dst, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
