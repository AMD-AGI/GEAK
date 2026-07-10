#!/usr/bin/env python3
"""Count DECODE forward steps captured in the newest torch profiler trace under a directory.

Used by bench_e2e.sh's representativeness gate: a profiling window that under-captures decode biases
BOTH head selection (raw %GPU) and the decode weight-share. The gate needs a cheap decode-step count so
it can enlarge the window and re-capture when the count is below N = max(30, 5*ceil(OSL/CONC)).

PREFERRED source = the EXACT decode forward-step count measured from the trace's gpu_user_annotation
step spans (vLLM detailed_trace_annotation): parse_torch_trace returns phase_meta.n_decode_steps, the
true number of pure-decode steps — the unit the gate actually wants.

FALLBACK (traces without step annotations) = the launch count of the busiest COMPUTE kernel whose input
shapes are HIDDEN (graph-replayed decode launches; prefill runs eager with real Input Dims). NOTE this
proxy is decode_steps × num_layers (a per-layer kernel fires once per layer per step), i.e. it
OVER-counts steps by ~num_layers — it is NOT a decode-step count and must NOT be compared against a step
floor as if it were. It survives only as a coarse "did we capture *any* decode?" signal when annotations
are absent. Prints a single integer to stdout (0 on any error, so the caller degrades to
"re-capture / low-confidence" rather than crashing the bench).
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_SKIP = ("memcpy", "memset", "copy", "cast", "elementwise", "fill", "index_", "cat_")


def _newest_trace(d):
    pats = ("*.pt.trace.json*", "*.trace.json*", "*.json.gz")
    files = []
    for p in pats:
        files += glob.glob(os.path.join(d, "**", p), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    return max(files, key=os.path.getmtime) if files else ""


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        from parse_profile import parse_torch_trace  # reuse the exact trace reader
    except Exception:
        print(0)
        return
    tr = _newest_trace(d)
    if not tr:
        print(0)
        return
    try:
        agg, _total_us, _launches, _pmeta = parse_torch_trace(tr)
    except Exception:
        print(0)
        return
    # PREFERRED: exact decode-STEP count from the trace's gpu_user_annotation step spans. This is the
    # unit the gate compares against (N = max(30, 5*ceil(OSL/CONC)) decode steps) — no launches->steps
    # confusion. Only when the trace lacks step annotations do we fall back to the launch proxy below.
    try:
        if (_pmeta or {}).get("has_annotations"):
            print(int(_pmeta.get("n_decode_steps", 0)))
            return
    except Exception:
        pass
    # FALLBACK (no step annotations): rank compute kernels by GPU time; among the busiest few (the real per-forward compute heads, not a
    # high-multiplicity pointwise), take the max shape-hidden launch count. This is decode LAUNCHES (>=
    # decode forward steps, since a per-layer kernel fires once per layer) — a conservative floor that
    # reliably catches the pathological under-capture (hidden ~= 0) and guarantees >= N decode latency
    # samples for a stable decode weight-share.
    try:
        heads = sorted(
            ((info.get("total_us", 0.0), name, info) for name, info in agg.items()
             if not any(s in name.lower() for s in _SKIP)),
            reverse=True,
        )[:5]
        best = 0
        for _t, _name, info in heads:
            by_case = info.get("by_case") or {}
            hidden = sum(c.get("count", 0) for (sig, _dt), c in by_case.items() if not sig)
            if hidden > best:
                best = hidden
    except Exception:
        print(0)
        return
    print(int(best))


if __name__ == "__main__":
    main()
