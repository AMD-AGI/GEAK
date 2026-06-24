#!/usr/bin/env python3
"""Standardized profile -> per-kernel Top-N summary.

Turns a torch profiler trace into ONE canonical, deterministic schema (JSON + Markdown) so every
downstream agent reads the bottleneck the same way. This is the "spec" contract for the e2e
workflow's Profile phase.

Input:
  --torch-trace  <file.json[.gz]>   sglang/vllm torch profiler trace. Gives op names + per-launch
                                    shapes/dtypes (linked kernel->cpu_op via "External id") and
                                    per-call kernel durations for distribution analysis.

Optional enhancement (auto-installed on first run, stdlib fallback on failure):
  TraceLens (AMD-AGI/TraceLens) — when importable, enables:
    - Tree-based CPU->GPU linking (handles graph launches, nested ops)
    - Per-shape breakdown (each unique shape group gets independent GPU-time stats)
    - Roofline analysis (TFLOPS/s, TB/s, FLOPS/Byte, compute- vs memory-bound)
  If TraceLens is absent, auto-installs from local clone or GitHub. Falls back to
  stdlib-only flat-scan parsing if install/import fails.

Output (written next to --out, default stdout):
  <out>.json   the canonical schema below
  <out>.md     a human-readable Top-N table

Schema (json):
{
  "source": "torch-trace",
  "tracelens": true|false,
  "total_gpu_time_ms": float,
  "num_kernel_launches": int,
  "num_distinct_kernels": int,
  "top_kernels": [ {
     "rank", "name", "short_name", "calls", "total_ms", "avg_us",
     "pct_gpu_time",                          # percentage of total GPU time
     "shapes": [[...dims...], ...],           # up to 5 distinct input-dim sets
     "dtypes": [...],                         # distinct input dtypes seen
     "classification": "triton|library_gemm|library_attn|fused_custom|"
                       "elementwise_overhead|reduction_norm|memory|other",
     "backend_guess": "triton|hipblaslt|aiter|ck|rocblas|torch_native|unknown",
     "editable": bool,
     "opt_hint": str,
     "per_call": {                              # per-call distribution (informational)
       "n", "median_us", "mean_us", "std_us", "min_us", "max_us",
       "p10_us", "p90_us", "p99_us", "cov", "distribution_type"
     },
     "roofline": { ... },                    # only when TraceLens available
     "shape_breakdown": [ ... ]              # only when TraceLens available
  } ... ]
}

Stdlib only (when TraceLens is absent).
"""
import argparse, gzip, json, os, re, sys

# ---------------------------------------------------------------------------
# TraceLens: try import → auto-install from local/git → retry import → give up
# ---------------------------------------------------------------------------
TRACELENS = False

# PyPI "TraceLens" is a DIFFERENT package (AI agent eval framework).
# AMD TraceLens must be installed from the local clone or the AMD-AGI GitHub repo.
_TRACELENS_LOCAL_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "TraceLens"),  # sibling to GEAK
    os.path.expanduser("~/TraceLens"),
]
_TRACELENS_GIT_URL = "https://github.com/AMD-AGI/TraceLens.git"


def _try_import_tracelens():
    global TRACELENS
    try:
        from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer  # noqa: F811
        from TraceLens.PerfModel.torch_op_mapping import categorize_torch_op  # noqa: F811
        TRACELENS = True
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _auto_install_tracelens():
    import subprocess
    # Try local paths first (fast, no network)
    for p in _TRACELENS_LOCAL_PATHS:
        p = os.path.realpath(p)
        if os.path.isfile(os.path.join(p, "setup.py")) or os.path.isfile(os.path.join(p, "pyproject.toml")):
            sys.stderr.write(f"TraceLens not found, installing from {p} ...\n")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-e", p, "-q"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                if _try_import_tracelens():
                    sys.stderr.write("TraceLens installed from local clone.\n")
                    return True
            except Exception:
                pass
    # Fallback: clone from GitHub
    try:
        import tempfile
        clone_dir = tempfile.mkdtemp(prefix="tracelens_")
        sys.stderr.write(f"TraceLens not found locally, cloning from {_TRACELENS_GIT_URL} ...\n")
        subprocess.check_call(
            ["git", "clone", "--depth=1", _TRACELENS_GIT_URL, clone_dir],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", clone_dir, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        if _try_import_tracelens():
            sys.stderr.write("TraceLens installed from GitHub.\n")
            return True
    except Exception as e:
        sys.stderr.write(f"TraceLens auto-install failed ({e}); falling back to stdlib.\n")
    return False


if not _try_import_tracelens():
    _auto_install_tracelens()

if TRACELENS:
    from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
    from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
    from TraceLens.PerfModel.torch_op_mapping import categorize_torch_op


# ---------------------------------------------------------------------------
# Classification heuristics. Order matters (first match wins).
# Each entry: (regex, classification, backend_guess, editable, hint)
# ---------------------------------------------------------------------------
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


def analyze_distribution(durations_us):
    """Per-call distribution stats from a list of durations (µs). Informational only."""
    if not durations_us or len(durations_us) < 2:
        return None
    n = len(durations_us)
    s = sorted(durations_us)
    mean_v = sum(s) / n
    median_v = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    variance = sum((x - mean_v) ** 2 for x in s) / (n - 1)
    std_v = variance ** 0.5
    cov = std_v / mean_v if mean_v > 0 else 0.0

    def pct(sorted_list, p):
        idx = int(len(sorted_list) * p / 100)
        return sorted_list[min(idx, len(sorted_list) - 1)]

    if cov < 0.3:
        dist_type = "stable"
    elif cov > 1.0:
        dist_type = "high_variance"
    else:
        dist_type = "moderate"

    return {
        "n": n,
        "median_us": round(median_v, 3),
        "mean_us": round(mean_v, 3),
        "std_us": round(std_v, 3),
        "min_us": round(s[0], 3),
        "max_us": round(s[-1], 3),
        "p10_us": round(pct(s, 10), 3),
        "p90_us": round(pct(s, 90), 3),
        "p99_us": round(pct(s, 99), 3),
        "cov": round(cov, 3),
        "distribution_type": dist_type,
    }


def classify(name):
    for rx, cls, backend, editable, hint in RULES:
        if re.search(rx, name, re.IGNORECASE):
            return cls, backend, editable, hint
    if re.search(r"^[a-z0-9_]+kernel[a-z0-9_]*$", name) or re.search(r"_fwd_kernel|_bwd_kernel", name):
        return ("triton", "triton", True,
                "Snake_case JIT kernel (likely Triton). Extractable; tune or compare backends.")
    return "other", "unknown", True, "Unclassified — inspect source to route."


def short_name(name):
    """Best-effort readable short name from a mangled C++/triton symbol."""
    n = name
    # TraceLens synthetic graph launches: "hipGraphLaunch->actual_kernel (Synthetic Op)"
    if '->' in n:
        n = n.split('->', 1)[1]
    n = re.sub(r'\s*\(Synthetic Op\)\s*$', '', n).strip()
    n = re.sub(r"^void\s+", "", n)
    m = re.match(r"[\w:]+", n)
    base = m.group(0) if m else n
    base = base.split("::")[-1]
    return base[:60]


# ---------------------------------------------------------------------------
# TraceLens category -> our classification mapping
# ---------------------------------------------------------------------------
TRACELENS_CATEGORY_MAP = {
    "GEMM_fwd": ("library_gemm", "hipblaslt", False),
    "GEMM_bwd": ("library_gemm", "hipblaslt", False),
    "GroupedGEMM_fwd": ("library_gemm", "hipblaslt", False),
    "GroupedGEMM_bwd": ("library_gemm", "hipblaslt", False),
    "CONV_fwd": ("library_gemm", "torch_native", False),
    "CONV_bwd": ("library_gemm", "torch_native", False),
    "SDPA_fwd": ("library_attn", "ck", False),
    "SDPA_bwd": ("library_attn", "ck", False),
    "InferenceAttention": ("library_attn", "ck", False),
    "NORM_fwd": ("reduction_norm", "triton", True),
    "NORM_bwd": ("reduction_norm", "triton", True),
    "SSM_fwd": ("fused_custom", "triton", True),
    "SSM_bwd": ("fused_custom", "triton", True),
    "RoPE_fwd": ("reduction_norm", "triton", True),
    "RoPE_bwd": ("reduction_norm", "triton", True),
    "MoE_fused": ("fused_custom", "triton", True),
    "MoE_unfused": ("fused_custom", "triton", True),
    "MoE_comm_fwd": ("fused_custom", "triton", True),
    "MoE_comm_bwd": ("fused_custom", "triton", True),
    "MoE_aux": ("fused_custom", "triton", True),
    "CrossEntropy_fwd": ("reduction_norm", "triton", True),
    "CrossEntropy_bwd": ("reduction_norm", "triton", True),
    "elementwise": ("elementwise_overhead", "torch_native", True),
    "reduce": ("reduction_norm", "triton", True),
    "triton": ("triton", "triton", True),
    "multi_tensor_apply": ("elementwise_overhead", "torch_native", True),
}


def classify_tracelens(op_category, kernel_names):
    """Map a TraceLens op category + actual GPU kernel names to our classification.

    The kernel_names list lets us override the default: e.g. an aten::mm that dispatches to a Triton
    GEMM should be classified as triton (editable), not library_gemm (non-editable).
    """
    base = TRACELENS_CATEGORY_MAP.get(op_category)
    if base is None:
        if kernel_names:
            return classify(kernel_names[0])
        return "other", "unknown", True, "Unclassified — inspect source to route."

    cls, backend, editable = base
    hint = next((h for _, _, _, _, h in RULES if _ == cls), "")

    if kernel_names:
        for kn in kernel_names:
            kcls, kbackend, keditable, khint = classify(kn)
            if keditable and not editable:
                return kcls, kbackend, keditable, khint
            if kbackend != "unknown" and backend in ("unknown", "torch_native"):
                backend = kbackend

    return cls, backend, editable, hint



# ---------------------------------------------------------------------------
# torch / sglang trace — load
# ---------------------------------------------------------------------------
def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def load_trace_events(path):
    """Load trace events from a torch profiler JSON trace, returning (events_list, raw_data)."""
    with _open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    return events, data



def parse_torch_trace(events):
    """Flat-scan parser: aggregate kernel stats + shape enrichment via External id."""
    op_by_ext = {}
    for e in events:
        if not isinstance(e, dict) or e.get("cat") != "cpu_op":
            continue
        a = e.get("args", {})
        ext = a.get("External id")
        dims = a.get("Input Dims")
        if ext is not None and dims:
            flat = [d for d in dims if d]
            if flat and (ext not in op_by_ext):
                op_by_ext[ext] = (dims, a.get("Input type"))

    agg = {}
    total_us = 0.0
    launches = 0
    kernel_events = []
    for e in events:
        if not isinstance(e, dict) or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        kernel_events.append(e)
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
    return agg, total_us, launches, kernel_events


# ---------------------------------------------------------------------------
# TraceLens enhanced parser
# ---------------------------------------------------------------------------
def _build_analyzer_from_data(data):
    """Build a TreePerfAnalyzer from already-loaded trace dict (skip re-reading file)."""
    trace_metadata = {k: v for k, v in data.items() if k != "traceEvents"}
    events = data["traceEvents"]
    tree = TraceToTree(events, event_to_category=TraceToTree.default_categorizer,
                       trace_metadata=trace_metadata)
    return TreePerfAnalyzer(tree, event_to_category=TraceToTree.default_categorizer)


def parse_with_tracelens(events, data, top_n):
    """Use TraceLens tree-based parser for enhanced analysis.

    events and data are from load_trace_events() — no second file read needed.
    """
    analyzer = _build_analyzer_from_data(data)
    df = analyzer.build_df_unified_perf_table(include_nccl=True)

    if df.empty:
        return None

    summary = analyzer.summarize_df_unified_perf_table(df)
    if summary.empty:
        return None

    # Build per-(name, shape, dtype) duration lists for per-call distribution analysis.
    # Each group matches a summary row — durations are comparable within a group.
    _group_durs = {}
    if "Kernel Time (µs)" in df.columns:
        for _, r in df.iterrows():
            kt = r.get("Kernel Time (µs)")
            if kt is None or (isinstance(kt, float) and kt != kt):
                continue
            key = (str(r.get("name", "")),
                   str(r.get("Input Dims", "")),
                   str(r.get("Input type", "")))
            _group_durs.setdefault(key, []).append(float(kt))

    # Compute totals from raw events (TraceLens summary may not cover all kernels)
    total_us = 0.0
    launches = 0
    distinct = set()
    for e in events:
        if isinstance(e, dict) and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            total_us += float(e.get("dur", 0))
            launches += 1
            distinct.add(e.get("name", ""))

    top = []
    for idx, (_, row) in enumerate(summary.head(top_n).iterrows()):
        if idx >= top_n:
            break

        op_name = row.get("name", "?")
        op_category = row.get("op category", "other")

        kd = row.get("kernel_details")
        kernel_names = []
        if isinstance(kd, (list, tuple)):
            kernel_names = [k.get("name", "") for k in kd if isinstance(k, dict)]
        elif isinstance(kd, str):
            kernel_names = [kd]

        primary_kernel = kernel_names[0] if kernel_names else op_name
        # TraceLens synthetic graph launches: "hipGraphLaunch->actual_kernel (Synthetic Op)"
        # Extract the actual kernel so classification and naming work correctly.
        if not kernel_names and '->' in primary_kernel:
            _actual = re.sub(r'\s*\(Synthetic Op\)\s*$', '', primary_kernel.split('->', 1)[1]).strip()
            if _actual:
                primary_kernel = _actual
                kernel_names = [_actual]
        cls, backend, editable, hint = classify_tracelens(op_category, kernel_names)

        calls = int(row.get("operation_count", row.get("Count", 1)))
        total_ms_val = float(row.get("Kernel Time (µs)_sum", 0)) / 1000.0
        avg_us_val = float(row.get("Kernel Time (µs)_mean", 0))
        pct = float(row.get("Percentage (%)", 0))

        shapes_raw = row.get("Input Dims")
        dtypes_raw = row.get("Input type")
        shapes = []
        if shapes_raw is not None:
            if isinstance(shapes_raw, (list, tuple)):
                shapes = [list(s) if isinstance(s, (list, tuple)) else s
                          for s in shapes_raw if s][:5]
            else:
                try:
                    parsed = json.loads(str(shapes_raw)) if isinstance(shapes_raw, str) else shapes_raw
                    if isinstance(parsed, list):
                        shapes = [parsed]
                except (json.JSONDecodeError, TypeError):
                    pass
        dtypes = []
        if dtypes_raw is not None:
            if isinstance(dtypes_raw, (list, tuple)):
                dtypes = sorted(set(str(t) for t in dtypes_raw if t))[:8]

        entry = {
            "rank": idx + 1,
            "name": primary_kernel,
            "short_name": short_name(primary_kernel),
            "cpu_op": op_name,
            "calls": calls,
            "total_ms": round(total_ms_val, 4),
            "avg_us": round(avg_us_val, 3),
            "pct_gpu_time": round(pct, 2),
            "shapes": shapes,
            "dtypes": dtypes,
            "classification": cls,
            "backend_guess": backend,
            "editable": editable,
            "opt_hint": hint,
        }

        tflops = row.get("TFLOPS/s_mean")
        tbs = row.get("TB/s_mean")
        flops_byte = row.get("FLOPS/Byte_first", row.get("FLOPS/Byte"))
        roofline_bound = row.get("Roofline Bound")
        pct_roofline = row.get("Pct Roofline")
        if tflops is not None or tbs is not None:
            roofline_data = {}
            if tflops is not None:
                roofline_data["tflops_s"] = round(float(tflops), 2)
            if tbs is not None:
                roofline_data["tb_s"] = round(float(tbs), 2)
            if flops_byte is not None:
                roofline_data["flops_byte"] = round(float(flops_byte), 2)
            if roofline_bound is not None:
                roofline_data["bound"] = str(roofline_bound)
            if pct_roofline is not None:
                roofline_data["pct_roofline"] = round(float(pct_roofline), 2)
            entry["roofline"] = roofline_data

        # Per-call distribution (per shape group — durations are comparable)
        gkey = (str(row.get("name", "")),
                str(row.get("Input Dims", "")),
                str(row.get("Input type", "")))
        durs = _group_durs.get(gkey, [])
        if len(durs) >= 2:
            entry["per_call"] = analyze_distribution(durs)

        top.append(entry)

    return {
        "source": "torch-trace",
        "tracelens": True,
        "total_gpu_time_ms": round(total_us / 1000.0, 4),
        "num_kernel_launches": launches,
        "num_distinct_kernels": len(distinct),
        "top_kernels": top,
    }


# ---------------------------------------------------------------------------
# Build summary from flat-scan aggregation (stdlib fallback)
# ---------------------------------------------------------------------------
def build_summary(agg, total_us, launches, top_n, kernel_events=None):
    items = sorted(agg.items(), key=lambda kv: kv[1]["total_us"], reverse=True)

    # Per-name duration lists for per-call distribution (stdlib: no shape separation)
    name_durs = {}
    if kernel_events:
        for e in kernel_events:
            nm = e.get("name", "?")
            dur = float(e.get("dur", 0) or 0)
            if dur > 0:
                name_durs.setdefault(nm, []).append(dur)

    top = []
    for rank, (name, d) in enumerate(items[:top_n], 1):
        cls, backend, editable, hint = classify(name)
        shapes = sorted(d["shapes"]) if d["shapes"] else []
        dtypes = sorted(d["dtypes"]) if d["dtypes"] else []
        pct = round(100.0 * d["total_us"] / total_us, 2) if total_us else 0.0

        entry = {
            "rank": rank,
            "name": name,
            "short_name": short_name(name),
            "calls": d["calls"],
            "total_ms": round(d["total_us"] / 1000.0, 4),
            "avg_us": round(d["total_us"] / max(d["calls"], 1), 3),
            "pct_gpu_time": pct,
            "shapes": [json.loads(s) for s in shapes[:5]],
            "dtypes": dtypes[:8],
            "classification": cls,
            "backend_guess": backend,
            "editable": editable,
            "opt_hint": hint,
        }

        durs = name_durs.get(name, [])
        if len(durs) >= 2:
            entry["per_call"] = analyze_distribution(durs)

        top.append(entry)

    return {
        "source": "torch-trace",
        "tracelens": False,
        "total_gpu_time_ms": round(total_us / 1000.0, 4),
        "num_kernel_launches": launches,
        "num_distinct_kernels": len(agg),
        "top_kernels": top,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def to_markdown(summ):
    L = []
    tl_tag = " (TraceLens)" if summ.get("tracelens") else " (stdlib)"
    L.append(f"# Profile Top-{len(summ['top_kernels'])} — standardized summary{tl_tag}\n")
    L.append(f"- source: `{summ['source']}`")
    L.append(f"- total GPU time: **{summ['total_gpu_time_ms']:.2f} ms** "
             f"over {summ['num_kernel_launches']} launches, "
             f"{summ['num_distinct_kernels']} distinct kernels\n")
    L.append("| # | kernel | class | backend | edit | calls | total ms | %gpu | avg us | shapes |")
    L.append("|--|--------|-------|---------|------|-------|----------|------|--------|--------|")
    for k in summ["top_kernels"]:
        sh = "; ".join(json.dumps(s) for s in k["shapes"][:2]) if k["shapes"] else ""
        sh = (sh[:50] + "…") if len(sh) > 51 else sh
        L.append(f"| {k['rank']} | `{k['short_name']}` | {k['classification']} | "
                 f"{k['backend_guess']} | {'Y' if k['editable'] else 'N'} | {k['calls']} | "
                 f"{k['total_ms']:.3f} | {k['pct_gpu_time']:.1f} | "
                 f"{k['avg_us']:.1f} | `{sh}` |")

    roofline_entries = [k for k in summ["top_kernels"][:12] if k.get("roofline")]
    if roofline_entries:
        L.append("\n## Roofline analysis (TraceLens)\n")
        L.append("| # | kernel | TFLOPS/s | TB/s | FLOPS/B | bound | %roofline |")
        L.append("|--|--------|----------|------|---------|-------|-----------|")
        for k in roofline_entries:
            r = k["roofline"]
            L.append(f"| {k['rank']} | `{k['short_name']}` | "
                     f"{r.get('tflops_s', '-')} | {r.get('tb_s', '-')} | "
                     f"{r.get('flops_byte', '-')} | {r.get('bound', '-')} | "
                     f"{r.get('pct_roofline', '-')} |")

    percall_entries = [k for k in summ["top_kernels"][:12] if k.get("per_call")]
    if percall_entries:
        L.append("\n## Per-call distribution (top entries)\n")
        L.append("| # | kernel | n | median µs | mean µs | std µs | p90 µs | CoV | type |")
        L.append("|--|--------|---|-----------|---------|--------|--------|-----|------|")
        for k in percall_entries:
            pc = k["per_call"]
            L.append(f"| {k['rank']} | `{k['short_name']}` | "
                     f"{pc['n']} | {pc['median_us']:.1f} | {pc['mean_us']:.1f} | "
                     f"{pc['std_us']:.1f} | {pc['p90_us']:.1f} | "
                     f"{pc['cov']:.2f} | {pc['distribution_type']} |")

    L.append("\n## Opt hints (top entries)\n")
    for k in summ["top_kernels"][:12]:
        L.append(f"- **{k['rank']}. {k['short_name']}** ({k['pct_gpu_time']:.1f}% gpu, "
                 f"{k['classification']}/{k['backend_guess']}): {k['opt_hint']}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-trace", required=True)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="", help="path prefix; writes <out>.json and <out>.md")
    ap.add_argument("--no-tracelens", action="store_true", help="force stdlib fallback even if TraceLens is installed")
    args = ap.parse_args()

    events, data = load_trace_events(args.torch_trace)
    use_tracelens = TRACELENS and not args.no_tracelens

    summ = None
    if use_tracelens:
        try:
            summ = parse_with_tracelens(events, data, args.top)
        except Exception as e:
            sys.stderr.write(f"TraceLens failed ({e}), falling back to stdlib parser\n")
            summ = None

    if summ is None:
        agg, total_us, launches, kernel_events = parse_torch_trace(events)
        summ = build_summary(agg, total_us, launches, args.top, kernel_events)

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
