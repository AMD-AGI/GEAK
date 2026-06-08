#!/usr/bin/env python3
"""Standardized profile -> per-kernel Top-N summary.

Turns a profiler trace into ONE canonical, deterministic schema (JSON + Markdown) so every
downstream agent reads the bottleneck the same way. This is the "规范" contract for the e2e
workflow's Profile phase.

Two input sources (use either or both; merged when both given):
  --torch-trace  <file.json[.gz]>   sglang/torch profiler trace. Gives op names + per-launch
                                    shapes/dtypes (linked kernel->cpu_op via "External id").
  --rocprof-dir  <dir>              directory with rocprofv3 *kernel*stats*.csv (HW kernel
                                    durations; authoritative GPU time, no shapes).

When both are present, HW durations come from rocprofv3 and shapes/op-names are enriched from the
torch trace (matched by normalized kernel name).

Output (written next to --out, default stdout):
  <out>.json   the canonical schema below
  <out>.md     a human-readable Top-N table

Schema (json):
{
  "source": "torch-trace|rocprofv3|merged",
  "total_gpu_time_ms": float,
  "num_kernel_launches": int,
  "num_distinct_kernels": int,
  "top_kernels": [ {
     "rank", "name", "short_name", "calls", "total_ms", "avg_us", "pct_gpu_time",
     "shapes": [[...dims...], ...],          # up to 5 distinct input-dim sets
     "dtypes": [...],                        # distinct input dtypes seen
     "classification": "triton|library_gemm|library_attn|fused_custom|"
                       "elementwise_overhead|reduction_norm|memory|other",
     "backend_guess": "triton|hipblaslt|aiter|ck|rocblas|torch_native|unknown",
     "editable": bool,                       # can a source-level kernel swap touch it?
     "opt_hint": str
  } ... ]
}

Stdlib only.
"""
import argparse, csv, glob, gzip, json, os, re, sys
from collections import defaultdict


# --------------------------------------------------------------------------- #
# Classification heuristics. Order matters (first match wins).
# Each entry: (regex, classification, backend_guess, editable, hint)
# --------------------------------------------------------------------------- #
RULES = [
    (r"triton|_kernel_0d1d|tt\.|fused_.*kernel", "triton", "triton", True,
     "Triton kernel — extractable; try Triton tuning, or a CK/HIP rewrite if memory/compute bound."),
    (r"Cijk|Tensile|hipblaslt|_gemm|GemmEx|gemm_|hgemm|sgemm|f16_gemm|igemm",
     "library_gemm", "hipblaslt", False,
     "Library GEMM (hipBLASLt/Tensile). Tune via heuristics/env or swap to aiter/CK GEMM; rarely source-editable."),
    (r"aiter|ater::", "fused_custom", "aiter", True,
     "AITER kernel. Has source; compare aiter vs triton vs CK for this shape."),
    (r"flash|fmha|attention|attn|_mha_|paged|kv_cache|decode_attention|prefill",
     "library_attn", "ck", False,
     "Attention kernel (CK/AITER/FA). Try --attention-backend swap + per-shape backend; source-edit only if Triton attn."),
    (r"ck_|composable_kernel|CK::|ck::", "fused_custom", "ck", True,
     "Composable Kernel. Compare CK instance/config; source-tunable via instance selection."),
    (r"mamba|ssm|causal_conv|selective_scan|chunk_scan|chunk_fwd|chunk_gated|"
     r"gated_delta|delta_rule|state_passing|recompute_w|kkt_solve|l2norm|cumsum",
     "fused_custom", "triton", True,
     "Mamba/gated-delta linear-attn (hybrid model). Usually Triton — extractable; tune scan tiling."),
    (r"rms_?norm|layernorm|layer_norm|_norm_|rope|rotary|softmax|reduce|reduction",
     "reduction_norm", "triton", True,
     "Norm/rope/softmax. Often fusible into neighbor; try aiter/triton fused variant."),
    (r"silu|gelu|swiglu|activation|elementwise|FillFunctor|fill_|copy_|cast|"
     r"vectorized_elementwise|index_elementwise|scatter|gather|add_|mul_",
     "elementwise_overhead", "torch_native", True,
     "Elementwise/fill/cast. Candidate for fusion (Lever 1) to collapse dispatches."),
    (r"memcpy|memset|Memcpy|Memset|DtoH|HtoD|DtoD", "memory", "torch_native", False,
     "Memory op. Reduce via native layouts / fewer host roundtrips."),
]


def classify(name):
    for rx, cls, backend, editable, hint in RULES:
        if re.search(rx, name, re.IGNORECASE):
            return cls, backend, editable, hint
    # Fallback: a snake_case symbol ending in 'kernel' (and not a mangled C++ symbol) is almost
    # always a Triton/custom JIT kernel in sglang -> editable.
    if re.search(r"^[a-z0-9_]+kernel[a-z0-9_]*$", name) or re.search(r"_fwd_kernel|_bwd_kernel", name):
        return ("triton", "triton", True,
                "Snake_case JIT kernel (likely Triton). Extractable; tune or compare backends.")
    return "other", "unknown", True, "Unclassified — inspect source to route."


def short_name(name):
    """Best-effort readable short name from a mangled C++/triton symbol."""
    n = name
    # drop leading 'void ' and template/return noise
    n = re.sub(r"^void\s+", "", n)
    # take the first identifier-ish token before '(' or '<'
    m = re.match(r"[\w:]+", n)
    base = m.group(0) if m else n
    base = base.split("::")[-1]
    return base[:60]


# --------------------------------------------------------------------------- #
# torch / sglang trace
# --------------------------------------------------------------------------- #
def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def parse_torch_trace(path):
    with _open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data if isinstance(data, list) else [])

    # cpu_op External id -> (input_dims, input_types) for shape enrichment
    op_by_ext = {}
    for e in events:
        if not isinstance(e, dict) or e.get("cat") != "cpu_op":
            continue
        a = e.get("args", {})
        ext = a.get("External id")
        dims = a.get("Input Dims")
        if ext is not None and dims:
            # keep the op whose dims are non-trivial
            flat = [d for d in dims if d]
            if flat and (ext not in op_by_ext):
                op_by_ext[ext] = (dims, a.get("Input type"))

    agg = {}  # name -> dict
    total_us = 0.0
    launches = 0
    for e in events:
        if not isinstance(e, dict) or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        name = e.get("name", "?")
        dur = float(e.get("dur", 0.0) or 0.0)
        total_us += dur
        launches += 1
        d = agg.setdefault(name, {"calls": 0, "total_us": 0.0, "shapes": set(), "dtypes": set()})
        d["calls"] += 1
        d["total_us"] += dur
        ext = e.get("args", {}).get("External id")
        if ext in op_by_ext:
            dims, types = op_by_ext[ext]
            sig = json.dumps([x for x in dims if x])
            if len(d["shapes"]) < 5:
                d["shapes"].add(sig)
            if types:
                for t in types:
                    if t:
                        d["dtypes"].add(t)
    return agg, total_us, launches


# --------------------------------------------------------------------------- #
# rocprofv3 kernel stats csv
# --------------------------------------------------------------------------- #
def parse_rocprof_dir(d):
    csvs = []
    for pat in ("*kernel*stats*.csv", "*/*kernel*stats*.csv", "*.csv", "*/*.csv"):
        csvs += glob.glob(os.path.join(d, pat))
    csvs = sorted(set(csvs))
    agg = {}
    total_us = 0.0
    launches = 0
    for path in csvs:
        try:
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            continue
        if not rows:
            continue
        cols = {c.lower(): c for c in rows[0].keys()}
        name_c = cols.get("name") or cols.get("kernelname") or cols.get("kernel_name")
        # rocprofv3 stats csv: columns vary; common: Name, Calls, TotalDurationNs, AverageNs
        dur_c = next((cols[k] for k in cols if "totalduration" in k or k == "totaldurationns"
                      or "total_duration" in k), None)
        calls_c = next((cols[k] for k in cols if k in ("calls", "count")), None)
        if not (name_c and dur_c):
            continue
        for r in rows:
            name = r[name_c]
            ns = float(r[dur_c] or 0)
            calls = int(float(r[calls_c])) if calls_c and r.get(calls_c) else 1
            us = ns / 1000.0
            total_us += us
            launches += calls
            e = agg.setdefault(name, {"calls": 0, "total_us": 0.0, "shapes": set(), "dtypes": set()})
            e["calls"] += calls
            e["total_us"] += us
        break  # one stats file is the authoritative aggregate
    return agg, total_us, launches


def norm_key(name):
    """Loose key to match a HW kernel name to a torch op name for shape enrichment."""
    return re.sub(r"[^a-z0-9]", "", short_name(name).lower())


def build_summary(agg, total_us, launches, source, top_n, enrich=None):
    items = []
    for name, d in agg.items():
        items.append((name, d))
    items.sort(key=lambda kv: kv[1]["total_us"], reverse=True)

    enrich_by_key = {}
    if enrich:
        for name, d in enrich.items():
            enrich_by_key.setdefault(norm_key(name), d)

    top = []
    for rank, (name, d) in enumerate(items[:top_n], 1):
        cls, backend, editable, hint = classify(name)
        shapes = sorted(d["shapes"]) if d["shapes"] else []
        dtypes = sorted(d["dtypes"]) if d["dtypes"] else []
        if not shapes and enrich:
            ed = enrich_by_key.get(norm_key(name))
            if ed:
                shapes = sorted(ed["shapes"])
                dtypes = sorted(ed["dtypes"])
        top.append({
            "rank": rank,
            "name": name,
            "short_name": short_name(name),
            "calls": d["calls"],
            "total_ms": round(d["total_us"] / 1000.0, 4),
            "avg_us": round(d["total_us"] / max(d["calls"], 1), 3),
            "pct_gpu_time": round(100.0 * d["total_us"] / total_us, 2) if total_us else 0.0,
            "shapes": [json.loads(s) for s in shapes[:5]],
            "dtypes": dtypes[:8],
            "classification": cls,
            "backend_guess": backend,
            "editable": editable,
            "opt_hint": hint,
        })
    return {
        "source": source,
        "total_gpu_time_ms": round(total_us / 1000.0, 4),
        "num_kernel_launches": launches,
        "num_distinct_kernels": len(agg),
        "top_kernels": top,
    }


def to_markdown(summ):
    L = []
    L.append(f"# Profile Top-{len(summ['top_kernels'])} — standardized summary\n")
    L.append(f"- source: `{summ['source']}`")
    L.append(f"- total GPU time: **{summ['total_gpu_time_ms']:.2f} ms** "
             f"over {summ['num_kernel_launches']} launches, "
             f"{summ['num_distinct_kernels']} distinct kernels\n")
    L.append("| # | kernel | class | backend | edit | calls | total ms | %gpu | avg us | shapes |")
    L.append("|--|--------|-------|---------|------|-------|----------|------|--------|--------|")
    for k in summ["top_kernels"]:
        sh = "; ".join(json.dumps(s) for s in k["shapes"][:2]) if k["shapes"] else ""
        sh = (sh[:60] + "…") if len(sh) > 61 else sh
        L.append(f"| {k['rank']} | `{k['short_name']}` | {k['classification']} | "
                 f"{k['backend_guess']} | {'Y' if k['editable'] else 'N'} | {k['calls']} | "
                 f"{k['total_ms']:.3f} | {k['pct_gpu_time']:.1f} | {k['avg_us']:.1f} | `{sh}` |")
    L.append("\n## Opt hints (top entries)\n")
    for k in summ["top_kernels"][:12]:
        L.append(f"- **{k['rank']}. {k['short_name']}** ({k['pct_gpu_time']:.1f}% gpu, "
                 f"{k['classification']}/{k['backend_guess']}): {k['opt_hint']}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-trace", default="")
    ap.add_argument("--rocprof-dir", default="")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="", help="path prefix; writes <out>.json and <out>.md")
    args = ap.parse_args()

    if not args.torch_trace and not args.rocprof_dir:
        ap.error("provide --torch-trace and/or --rocprof-dir")

    torch_agg = torch_total = torch_launch = None
    if args.torch_trace:
        torch_agg, torch_total, torch_launch = parse_torch_trace(args.torch_trace)
    rp_agg = rp_total = rp_launch = None
    if args.rocprof_dir:
        rp_agg, rp_total, rp_launch = parse_rocprof_dir(args.rocprof_dir)

    if rp_agg and torch_agg:
        summ = build_summary(rp_agg, rp_total, rp_launch, "merged", args.top, enrich=torch_agg)
    elif rp_agg:
        summ = build_summary(rp_agg, rp_total, rp_launch, "rocprofv3", args.top)
    else:
        summ = build_summary(torch_agg, torch_total, torch_launch, "torch-trace", args.top)

    js = json.dumps(summ, indent=2)
    md = to_markdown(summ)
    if args.out:
        with open(args.out + ".json", "w") as fh:
            fh.write(js)
        with open(args.out + ".md", "w") as fh:
            fh.write(md)
        sys.stderr.write(f"wrote {args.out}.json and {args.out}.md\n")
    print(md)


if __name__ == "__main__":
    main()
