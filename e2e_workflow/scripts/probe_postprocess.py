#!/usr/bin/env python3
"""Merge per-pid probe outputs + (optional) join profile per-kernel avg_us -> per_shape_probe.{json,md}.

MODEL-AGNOSTIC: kernels are discovered from the probe output files themselves (their `target` field),
not from a hard-coded map. Any kernel the probe captured shows up here — add a new model by pointing
PROBE_TARGETS at its hot kernels; no code change to this script.

Reads:
  --probe-dir     dir of probe_<pid>_<target>.json files (from capture_shapes_probe flush)
  --profile-topn  profile/round_0/profile_topN.json — STRONGLY RECOMMENDED (not required): supplies
                  %GPU (the Amdahl basis for "which kernel is worth optimizing") + a fallback per-shape
                  latency. If omitted, the script still emits shape+count(+measured latency) but warns
                  loudly and marks %GPU as "unknown".

For each captured kernel: sum per-shape counts across all pids; latency is the probe's measured
cuda.Event time when present (PROBE_TIME=1), else the profiled per-kernel avg_us spread across shapes
(approximate; CUDA graph replay can't be per-shape timed).

Stdlib only.
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

# "unknown" sentinel for %GPU when no profile is provided (distinct from a real None/absent kernel).
GPU_UNKNOWN = "unknown"


def label_from_target(target):
    """Derive a human label from a probe target string 'module.path:attr' -> 'attr'.
    Falls back to the whole target if it has no ':'."""
    return target.split(":")[-1] if ":" in target else target


def match_substrs_from_target(target):
    """Derive profile-name match substrings from a target with NO hard-coded map. Uses the attr name
    and its leaf module component, so e.g. 'triton_kernels.matmul_ogs:matmul_ogs' matches a profiled
    '_matmul_ogs' kernel. Deduped, lowercased-compare handled at match time."""
    mod, _, attr = target.partition(":")
    subs = []
    if attr:
        subs.append(attr)
    leaf = mod.split(".")[-1] if mod else ""
    if leaf and leaf not in subs:
        subs.append(leaf)
    return subs or [target]


def load_probe(probe_dir):
    """target -> {'total_calls': int, 'cases': {sig: {dims,dtypes,count}}} summed across pids."""
    merged = {}
    for path in sorted(glob.glob(os.path.join(probe_dir, "probe_*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        tgt = d["target"]
        m = merged.setdefault(tgt, {"total_calls": 0, "cases": {}})
        m["total_calls"] += d.get("total_calls", 0)
        for c in d.get("cases", []):
            # dtypes are PER-TENSOR (parallel to dims), NOT deduped — keep order so dtype[i]
            # corresponds to dims[i]. The dedup key must preserve that order too.
            key = json.dumps(c["dims"], sort_keys=True) + "|" + json.dumps(c.get("dtypes", []))
            cc = m["cases"].get(key)
            if cc is None:
                cc = {"dims": c["dims"], "dtypes": c.get("dtypes", []),
                      "arg_labels": c.get("arg_labels", []), "count": 0,
                      "_gpu_us_weighted": 0.0, "_timed_count": 0}
                m["cases"][key] = cc
            cc["count"] += c["count"]
            # measured GPU latency: accumulate weighted by timed_count across pids
            tc = c.get("timed_count", 0)
            if tc and c.get("gpu_us_avg") is not None:
                cc["_gpu_us_weighted"] += c["gpu_us_avg"] * tc
                cc["_timed_count"] += tc
    return merged


def load_profile(profile_topn):
    with open(profile_topn) as fh:
        p = json.load(fh)
    return p.get("top_kernels", [])


def match_profile(top_kernels, substrs):
    """Return list of profiled kernels whose name/short_name matches any substr."""
    out = []
    for k in top_kernels:
        name = (k.get("name", "") + " " + k.get("short_name", "")).lower()
        if any(s.lower() in name for s in substrs):
            out.append(k)
    return out


def build(probe_dir, profile_topn, workload=None):
    probe = load_probe(probe_dir)
    have_profile = bool(profile_topn) and os.path.exists(profile_topn)
    top = load_profile(profile_topn) if have_profile else []
    if not have_profile:
        sys.stderr.write(
            "\n"
            "############################################################################\n"
            "# [probe_postprocess] WARNING: no --profile-topn provided (or file missing).\n"
            "#   -> %GPU is UNKNOWN, so you CANNOT rank kernels by Amdahl importance.\n"
            "#   -> shape+count(+measured latency) are still emitted, but for optimization\n"
            "#      selection you should re-run with --profile-topn <profile_topN.json>.\n"
            "############################################################################\n\n")
    kernels = []
    # MODEL-AGNOSTIC: iterate the kernels the probe actually captured (discovered from probe_*.json),
    # NOT a hard-coded map. Any target present in the probe dir is reported.
    for tgt in sorted(probe.keys()):
        pm = probe[tgt]
        label = label_from_target(tgt)
        substrs = match_substrs_from_target(tgt)
        matched = match_profile(top, substrs) if top else []
        # weighted avg_us across matched profile call sites (by calls), as the per-shape latency basis
        if matched:
            tot_calls = sum(k.get("calls", 0) for k in matched) or 1
            avg_us = sum(k.get("avg_us", 0) * k.get("calls", 0) for k in matched) / tot_calls
            pct_gpu = sum(k.get("pct_gpu_time", 0) for k in matched)
            profile_calls = sum(k.get("calls", 0) for k in matched)
        elif have_profile:
            avg_us = None; pct_gpu = None; profile_calls = None  # profile given but no name match
        else:
            avg_us = None; pct_gpu = GPU_UNKNOWN; profile_calls = None  # no profile at all

        cases = sorted(pm["cases"].values(), key=lambda c: c["count"], reverse=True)
        tot = sum(c["count"] for c in cases) or 1
        any_measured = False
        clean_cases = []
        for c in cases:
            rec = {"dims": c["dims"], "dtypes": c["dtypes"],
                   "arg_labels": c.get("arg_labels", []), "count": c["count"],
                   "count_frac": round(c["count"] / tot, 6)}
            tc = c.get("_timed_count", 0)
            if tc:  # MEASURED per-shape GPU latency (cuda.Event, enforce-eager run)
                rec["latency_us_measured"] = round(c["_gpu_us_weighted"] / tc, 3)
                rec["timed_count"] = tc
                any_measured = True
            else:   # fallback: profile global avg spread across shapes (approximate)
                rec["latency_us_approx"] = round(avg_us, 3) if avg_us is not None else None
            clean_cases.append(rec)
        latency_basis = ("per_shape_measured (cuda.Event, enforce-eager; 真实 per-shape GPU 时间)"
                         if any_measured else
                         "profile_per_kernel_avg (per-shape 为近似摊分; CUDA graph 无法 per-shape 计时)")
        kernels.append({
            "target": tgt,
            "label": label,
            "probe_status": "captured",
            "pct_gpu": pct_gpu,
            "probe_total_calls": pm["total_calls"],
            "profile_calls": profile_calls,
            "kernel_avg_us": round(avg_us, 3) if avg_us is not None else None,
            "latency_basis": latency_basis,
            "num_distinct_shapes": len(clean_cases),
            "cases": clean_cases,
        })
    eager = any(any("latency_us_measured" in c for c in k.get("cases", []))
                for k in kernels if k.get("probe_status") == "captured")
    if eager:
        semantics = {
            "mode": "enforce-eager (CUDA graph OFF)",
            "shapes": "COMPLETE and REAL — every distinct input shape the kernel was called with.",
            "count": "REAL per-shape call frequency. With CUDA graph OFF every decode step runs the "
                     "kernel's Python entry, so the probe counts every real call — this IS the "
                     "steady-state traffic weight for this workload.",
            "latency": "MEASURED per-shape GPU time via cuda.Event (first/warmup sample per shape "
                       "dropped). Reflects real GPU kernel execution time, independent of graph.",
            "caveat": "Measured under enforce-eager, NOT the production graph config. Per-call Python "
                      "dispatch overhead is excluded (cuda.Event times GPU only); the SHAPE/COUNT/GPU "
                      "-time picture transfers to production, end-to-end throughput does not.",
        }
    else:
        semantics = {
            "mode": "CUDA graph ON (capture-phase only)",
            "shapes": "COMPLETE and REAL — includes shapes CUDA-graph replay hides from the profiler.",
            "count": "NOT real serving frequency — reflects graph CAPTURE counts only (replay skips "
                     "the Python entry). Use to see WHICH shapes exist, not traffic weight. Re-run "
                     "with EXTRA_SERVER_ARGS='--enforce-eager' + PROBE_TIME=1 for real count+latency.",
            "latency": "APPROXIMATE — profile global avg_us spread across shapes; not per-shape timed.",
            "count_vs_profile": "probe_calls << profile_calls by design (Python-call vs GPU-launch).",
        }
    return {
        "schema": "per-shape-probe-v1",
        "workload": workload if workload is not None else {"isl": None, "osl": None, "conc": None},
        "profile_source": profile_topn if have_profile else None,
        "data_semantics": semantics,
        "kernels": kernels,
        "note_coverage": "探针只覆盖'热点 kernel 在 Python 层有 tensor 入参入口'的算子。无 Python 入口的 "
                         "kernel(如 HIP C++ / hipBLASLt 等闭源库)探针无法采集,不出现在本报告中。",
    }


def _fmt_pct_gpu(v):
    """%GPU may be a float (from profile), the string 'unknown' (no profile), or None (profile given
    but no name match). Render each without crashing on the non-numeric cases."""
    if isinstance(v, (int, float)):
        return f"{v:.2f}%"
    if v == GPU_UNKNOWN:
        return "unknown (no profile)"
    return "n/a"


def to_md(summ):
    ds = summ.get("data_semantics", {})
    wl = summ.get("workload") or {}
    L = [f"# per-shape probe — {summ['schema']}",
         f"- workload: ISL={wl.get('isl')} OSL={wl.get('osl')} conc={wl.get('conc')}",
         f"- profile source: `{summ.get('profile_source')}`",
         f"- {summ.get('note_coverage','')}\n",
         "## ⚠️ 数据语义与局限（务必先读）\n",
         f"- **采集模式**：{ds.get('mode','')}"]
    for _key in ("shapes", "count", "latency", "caveat", "count_vs_profile"):
        if ds.get(_key):
            L.append(f"- **{_key}**：{ds[_key]}")
    L.append("")
    for k in summ["kernels"]:
        L.append(f"## {k['label']}  (`{k['target']}`)")
        if k["probe_status"] != "captured":
            L.append(f"- probe_status: **{k['probe_status']}** — {k.get('note','')}\n")
            continue
        pg = _fmt_pct_gpu(k.get("pct_gpu"))
        L.append(f"- %GPU: {pg} · probe_calls: {k['probe_total_calls']} · "
                 f"profile_calls: {k.get('profile_calls')} · distinct shapes: {k['num_distinct_shapes']}")
        L.append(f"- latency basis: {k['latency_basis']}")
        measured = any("latency_us_measured" in c for c in k["cases"])
        lat_col = "latency_us (measured)" if measured else "~latency_us (approx)"
        L.append(f"\n| shape (dims) | dtypes | count | count% | {lat_col} |")
        L.append("|---|---|---:|---:|---:|")
        for c in k["cases"][:50]:
            dims = json.dumps(c["dims"])
            dt = ",".join(c["dtypes"])
            lat = c.get("latency_us_measured", c.get("latency_us_approx"))
            L.append(f"| `{dims}` | {dt} | {c['count']} | {c['count_frac']*100:.2f}% | "
                     f"{lat if lat is not None else 'n/a'} |")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--profile-topn", default="",
                    help="profile_topN.json — STRONGLY RECOMMENDED (supplies %GPU). If omitted, warns "
                         "and marks %GPU unknown, but still emits shape+count.")
    ap.add_argument("--isl", type=int, default=None, help="workload input seq len (metadata)")
    ap.add_argument("--osl", type=int, default=None, help="workload output seq len (metadata)")
    ap.add_argument("--conc", type=int, default=None, help="workload concurrency (metadata)")
    ap.add_argument("--out", required=True, help="path prefix; writes <out>.json and <out>.md")
    args = ap.parse_args()

    workload = None
    if any(v is not None for v in (args.isl, args.osl, args.conc)):
        workload = {"isl": args.isl, "osl": args.osl, "conc": args.conc}
    summ = build(args.probe_dir, args.profile_topn, workload=workload)
    with open(args.out + ".json", "w") as fh:
        json.dump(summ, fh, indent=2)
    with open(args.out + ".md", "w") as fh:
        fh.write(to_md(summ))
    sys.stderr.write(f"wrote {args.out}.json and {args.out}.md\n")
    # quick console summary
    for k in summ["kernels"]:
        if k["probe_status"] == "captured":
            print(f"{k['label']}: {k['num_distinct_shapes']} shapes, "
                  f"probe_calls={k['probe_total_calls']} profile_calls={k.get('profile_calls')}")
        else:
            print(f"{k['label']}: {k['probe_status']}")


if __name__ == "__main__":
    main()
