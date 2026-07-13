#!/usr/bin/env python3
"""Merge per-pid probe outputs + join profile per-kernel avg_us -> per_shape_probe.{json,md}.

Reads:
  --probe-dir     dir of probe_<pid>_<target>.json files (from capture_shapes_probe flush)
  --profile-topn  profile/round_0/profile_topN.json (for %gpu, total calls, avg_us -> latency join)

For each hooked kernel: sum per-shape counts across all pids, then attach the profiled per-kernel
avg_us as an APPROXIMATE per-shape latency (plan §4: CUDA graph replay can't be per-shape timed, so
we spread the aggregate avg). Emits the schema from plan §6.4.

Kernel name mapping (probe target -> profile short_name substring), so the join can find the matching
profiled kernel. hipBLASLt #4 (Cijk) has no probe (no Python entry) and is reported as such.

Stdlib only.
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

# probe target "module:attr"  ->  (label, profile short_name match substrings)
# #1/#2 are the SAME hook (matmul_ogs) but two profiled kernels (call sites); we report the hook's
# aggregate and note both profiled call sites in the profile-join.
TARGET_MAP = {
    "triton_kernels.matmul_ogs:matmul_ogs": {
        "label": "matmul_ogs (MoE GEMM #1/#2)",
        "profile_match": ["_matmul_ogs"],
    },
    "aiter.ops.triton.unified_attention:unified_attention": {
        "label": "unified_attention (#3)",
        "profile_match": ["unified_attention", "kernel_unified_attention"],
    },
    "vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe:pack_bitmatrix": {
        "label": "pack_bitmatrix (#5)",
        "profile_match": ["pack_bitmatrix"],
    },
}


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


def build(probe_dir, profile_topn):
    probe = load_probe(probe_dir)
    top = load_profile(profile_topn) if profile_topn and os.path.exists(profile_topn) else []
    kernels = []
    for tgt, meta in TARGET_MAP.items():
        pm = probe.get(tgt)
        matched = match_profile(top, meta["profile_match"]) if top else []
        # weighted avg_us across matched profile call sites (by calls), as the per-shape latency basis
        if matched:
            tot_calls = sum(k.get("calls", 0) for k in matched) or 1
            avg_us = sum(k.get("avg_us", 0) * k.get("calls", 0) for k in matched) / tot_calls
            pct_gpu = sum(k.get("pct_gpu_time", 0) for k in matched)
            profile_calls = sum(k.get("calls", 0) for k in matched)
        else:
            avg_us = None; pct_gpu = None; profile_calls = None

        if not pm:
            kernels.append({
                "target": tgt, "label": meta["label"], "probe_status": "not_captured",
                "note": "no probe output for this target in this run",
                "pct_gpu": pct_gpu, "profile_calls": profile_calls,
            })
            continue

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
            "label": meta["label"],
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
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
        "profile_source": profile_topn,
        "data_semantics": semantics,
        "kernels": kernels,
        "note_hipblaslt": "#4 Cijk (hipBLASLt) 无 Python 入口, 未探针采集 (见 plan §6.2)",
    }


def to_md(summ):
    ds = summ.get("data_semantics", {})
    L = [f"# per-shape probe — {summ['schema']}",
         f"- workload: ISL={summ['workload']['isl']} OSL={summ['workload']['osl']} "
         f"conc={summ['workload']['conc']}",
         f"- profile source: `{summ['profile_source']}`",
         f"- {summ['note_hipblaslt']}\n",
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
        pg = f"{k['pct_gpu']:.2f}%" if k.get("pct_gpu") is not None else "n/a"
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
    ap.add_argument("--profile-topn", default="")
    ap.add_argument("--out", required=True, help="path prefix; writes <out>.json and <out>.md")
    args = ap.parse_args()

    summ = build(args.probe_dir, args.profile_topn)
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
