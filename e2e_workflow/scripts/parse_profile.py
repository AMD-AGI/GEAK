#!/usr/bin/env python3
"""Standardized profile -> per-kernel Top-N summary.

Turns a profiler trace into ONE canonical, deterministic schema (JSON + Markdown) so every
downstream agent reads the bottleneck the same way. This is the "spec" contract for the e2e
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
import argparse, bisect, csv, glob, gzip, json, math, os, re, sys
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
# serving-phase (prefill/decode) accounting — MEASURED from the trace's own
# gpu_user_annotation step spans (vLLM detailed_trace_annotation). This is NOT
# "deciding a regime": it only reports which phase each launch was observed in.
# See skill kernel-phase-accounting for the derivation + the two use cases.
# --------------------------------------------------------------------------- #
def _seg(name, tag):
    """(lead, tokens, kv) for a 'context'/'generation' segment. Handles both annotation
    dialects: perfskills  context_<nreq>(<ntok>)  and  ..._<batch>(sq<q>sk<kv>...)."""
    m = re.search(tag + r"_(\d+)\(([^)]*)\)", name)
    if not m:
        return (0, 0, 0)
    lead = int(m.group(1))
    inner = m.group(2)
    sq = re.search(r"sq(\d+)", inner)
    sk = re.search(r"sk(\d+)", inner)
    if sq:
        return (lead, int(sq.group(1)), int(sk.group(1)) if sk else 0)
    return (lead, int(inner) if inner.isdigit() else 0, 0)


def _classify_step(name):
    """-> (is_prefill, ctx_tokens, decode_batch, kv) for an execute_* span, else None."""
    if "context_" not in name or "generation_" not in name:
        return None
    ctx = _seg(name, "context")
    gen = _seg(name, "generation")
    return (ctx[1] > 0, ctx[1], gen[0] or gen[1], gen[2])


def _collect_step_spans(events):
    """Sorted GPU-timeline step windows: [(ts, end, 'P'|'D', ctx_tok, dec_batch)]."""
    spans = []
    for e in events:
        if not isinstance(e, dict) or e.get("cat") != "gpu_user_annotation":
            continue
        nm = e.get("name")
        if not (isinstance(nm, str) and nm.startswith("execute_") and "dur" in e and "ts" in e):
            continue
        c = _classify_step(nm)
        if c:
            spans.append((e["ts"], e["ts"] + e["dur"],
                          "P" if c[0] else "D", c[1], c[2]))
    spans.sort()
    return spans


def snap_capture_size(x, sizes):
    """Round x UP to the nearest cudagraph capture size (what production actually pads/runs)."""
    if not sizes or not x:
        return x
    for s in sorted(sizes):
        if s >= x:
            return s
    return max(sizes)


def analytic_regime_calls(isl, osl, conc, chunk=None):
    """Analytic per-wave forward-pass count per serving phase (the denominator the short profile
    window cannot observe). Mirrors attribute_weights.estimate_serving_regime_calls so the topN
    est_calls == serving_weight_model.analytic_calls the immutable unittest consumes:
      prefill = CONC * ceil(ISL/chunk)   (CONC is in the COUNT)
      decode  = OSL                      (CONC is in the SHAPE M, not the count)
    Prefer the shared impl; fall back to this identical formula if the import is unavailable."""
    try:
        from attribute_weights import estimate_serving_regime_calls
        return estimate_serving_regime_calls(isl, osl, conc or 1, prefill_chunk=chunk)
    except Exception:
        if not (isl and osl):
            return {}
        chunk = chunk or isl
        return {"prefill": (conc or 1) * math.ceil(isl / chunk) if isl > 0 else 0,
                "decode": max(0, osl)}


def _kernel_phase(d):
    bp = d.get("by_phase") or {}
    p = bp.get("prefill", {}).get("count", 0) > 0
    de = bp.get("decode", {}).get("count", 0) > 0
    return "both" if (p and de) else "prefill" if p else "decode" if de else None


def _phase_latency_ms(d):
    """Measured per-launch latency (ms) in each phase — the base_latency the unittest weights by."""
    out = {}
    for ph, e in (d.get("by_phase") or {}).items():
        if e.get("count"):
            out[ph] = round(e["total_us"] / e["count"] / 1000.0, 6)
    return out


def _est_shape(d, conc, capture_sizes):
    """Per-phase M estimate: prefill = observed step-M (budget + remainders); decode = concurrency
    snapped to the nearest cudagraph capture size (decode M is hidden by CUDA graph at runtime)."""
    bp = d.get("by_phase") or {}
    out = {}
    pm = bp.get("prefill", {}).get("m") or {}
    if pm:
        top = max(pm, key=pm.get)
        out["prefill"] = {"M": top,
                          "M_dist": dict(sorted(pm.items(), key=lambda kv: -kv[1])[:5])}
    if bp.get("decode", {}).get("count"):
        out["decode"] = {"M": snap_capture_size(conc, capture_sizes) if conc else None,
                         "M_note": "concurrency snapped to cudagraph capture size (CUDA-graph-hidden)"}
    return out


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

    # Serving-phase step windows (empty if the trace has no execute_* annotations).
    spans = _collect_step_spans(events)
    _starts = [s[0] for s in spans]

    def _phase_of(ts):
        """(phase 'prefill'|'decode'|None, step_M) for a launch at time ts."""
        if not _starts or ts is None:
            return None, 0
        i = bisect.bisect_right(_starts, ts) - 1
        if i < 0 or ts >= spans[i][1]:
            return None, 0
        _, _, ph, ctx_tok, dec_batch = spans[i]
        if ph == "P":
            return "prefill", ctx_tok + dec_batch      # dense/MoE M = all tokens in the step
        return "decode", dec_batch

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
        d = agg.setdefault(name, {"calls": 0, "total_us": 0.0, "shapes": set(),
                                  "dtypes": set(), "by_case": {}, "by_phase": {}})
        d["calls"] += 1
        d["total_us"] += dur
        # attribute this launch to its serving phase (measured from the step span it falls in)
        phase, stepM = _phase_of(e.get("ts"))
        if phase:
            bp = d["by_phase"].setdefault(phase, {"count": 0, "total_us": 0.0, "m": {}})
            bp["count"] += 1
            bp["total_us"] += dur
            bp["m"][stepM] = bp["m"].get(stepM, 0) + 1
        ext = e.get("args", {}).get("External id")
        sig = ""           # shape signature (json of non-empty input dims)
        dtype_sig = ""     # dtype signature (json of per-tensor input types)
        if ext in op_by_ext:
            dims, types = op_by_ext[ext]
            sig = json.dumps([x for x in dims if x])
            if len(d["shapes"]) < 5:
                d["shapes"].add(sig)
            if types:
                dts = [t for t in types if t]
                if dts:
                    dtype_sig = json.dumps(dts)
                for t in dts:
                    d["dtypes"].add(t)
        # UNCAPPED per-(shape,dtype) breakdown — the workload model needs the full
        # call-count distribution, not just the first 5 distinct shapes. Empty sigs
        # (kernel launched with no linked cpu_op / no Input Dims, e.g. graph replay)
        # collapse into one "shape unknown" bucket that build_workload() flags.
        c = d["by_case"].setdefault((sig, dtype_sig), {"count": 0, "total_us": 0.0, "phase": {}})
        c["count"] += 1
        c["total_us"] += dur
        if phase:
            c["phase"][phase] = c["phase"].get(phase, 0) + 1

    pref = [s for s in spans if s[2] == "P"]
    dec = [s for s in spans if s[2] == "D"]
    phase_meta = {}
    if spans:
        phase_meta = {
            "has_annotations": True,
            "n_prefill_steps": len(pref),
            "n_decode_steps": len(dec),
            "prefill_tokens": sum(s[3] for s in pref),
            "decode_batches": [s[4] for s in dec],
        }
    return agg, total_us, launches, phase_meta


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


def build_summary(agg, total_us, launches, source, top_n, enrich=None,
                  conc=0, isl=0, osl=0, chunk=None, capture_sizes=None, phase_meta=None):
    items = []
    for name, d in agg.items():
        items.append((name, d))
    items.sort(key=lambda kv: kv[1]["total_us"], reverse=True)

    enrich_by_key = {}
    if enrich:
        for name, d in enrich.items():
            enrich_by_key.setdefault(norm_key(name), d)

    # SERVING-PHASE header: is the captured decode window at steady state (batch ~ concurrency)?
    # If not, per-launch TIME (%GPU, base_latency) is biased low for batch/KV-scaled kernels; the
    # analytic est_calls (COUNT) is still valid (count/step is batch-independent). See the skill.
    pm = phase_meta or {}
    est_calls = analytic_regime_calls(isl, osl, conc, chunk) if (isl and osl) else {}
    b_cap = 0
    if pm.get("decode_batches"):
        b_cap = max(set(pm["decode_batches"]), key=pm["decode_batches"].count)
    serving = None
    if pm.get("has_annotations"):
        b_ss = snap_capture_size(conc, capture_sizes) if conc else conc
        serving = {
            "n_prefill_steps": pm.get("n_prefill_steps", 0),
            "n_decode_steps": pm.get("n_decode_steps", 0),
            "decode_batch_captured": b_cap,
            "decode_batch_steady": b_ss,
            "steady": bool(b_cap and conc and b_cap >= 0.8 * conc),
            "analytic_calls": est_calls,   # == serving_weight_model.analytic_calls (unittest weight)
            "note": ("decode captured at steady batch — per-launch time trustworthy"
                     if (b_cap and conc and b_cap >= 0.8 * conc) else
                     f"decode batch {b_cap} << concurrency {conc}: COUNTS ok, per-launch TIME biased "
                     "low — prefer a steady-state profile before trusting decode %GPU"),
        }

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
        entry = {
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
        }
        # SERVING-PHASE annotation (only when the trace exposed step spans). Enrich from the
        # HW-name-matched torch agg when this agg (e.g. rocprof) has no by_phase of its own.
        pd = d if d.get("by_phase") else (enrich_by_key.get(norm_key(name)) if enrich else None)
        phase = _kernel_phase(pd) if pd else None
        if phase:
            entry["phase"] = phase                               # prefill | decode | both
            entry["served_regimes"] = ([phase] if phase != "both" else ["prefill", "decode"])
            entry["phase_calls_measured"] = {ph: pd["by_phase"][ph]["count"]
                                             for ph in pd["by_phase"]}
            entry["calls_per_step"] = {ph: round(pd["by_phase"][ph]["count"]
                                                 / max(pm.get(f"n_{ph}_steps", 0) or 1, 1), 2)
                                       for ph in pd["by_phase"]}
            entry["base_latency_ms"] = _phase_latency_ms(pd)     # per-launch ms, per phase
            entry["est_shape"] = _est_shape(pd, conc, capture_sizes)
            if est_calls:
                # count the immutable unittest weights each phase's cases by (baseline_ms x est_calls[phase]).
                entry["est_calls"] = {ph: est_calls.get(ph)
                                      for ph in entry["served_regimes"] if ph in est_calls}
        top.append(entry)
    out = {
        "source": source,
        "total_gpu_time_ms": round(total_us / 1000.0, 4),
        "num_kernel_launches": launches,
        "num_distinct_kernels": len(agg),
        "top_kernels": top,
    }
    if serving:
        out["serving"] = serving
    return out


def build_workload(agg, total_us, top_n, target=""):
    """Per-kernel WEIGHT SIGNAL: each kernel's profiled time + (when the trace exposed them) its
    per-(shape,dtype) call distribution. This is the raw input to `attribute_weights.py`, which JOINS
    it with the extractor's `meta.json` shape cases (op_kind-aware) to produce the final WORKLOAD_SPEC
    the harness consumes. This script stays kernel-type-agnostic: it never invents shapes and never
    decides regimes — it only reports what the trace measured.

    For each kernel and each distinct (input shapes, input dtypes) it was called with, emit:
      count                how many launches had this shape+dtype
      baseline_latency_ms  measured per-call latency of the ORIGINAL kernel for this case
      weight               total time this case contributes (= count * latency, in us); this is
                           the workload-time weight the harness uses for the time-weighted metric
      weight_source        "trace" when the shape was recovered, else "regime_prior" (shape hidden
                           behind a graph-replay launch — count/time are real, shape is not)

    This feeds the kernel_workflow harness so it benchmarks the SAME shapes/dtypes the workload
    hits, weighted by their real wall-clock contribution. Correctness is unaffected (it stays on
    the frozen reference_io.pt oracle); this is a performance-measurement model only.
    """
    items = sorted(agg.items(), key=lambda kv: kv[1]["total_us"], reverse=True)
    if target:
        items = [(n, d) for (n, d) in items if target.lower() in n.lower()]
    kernels = []
    for name, d in items[:top_n]:
        cls, backend, editable, hint = classify(name)
        by_case = d.get("by_case") or {}
        cases = []
        for (sig, dtype_sig), c in by_case.items():
            dims = json.loads(sig) if sig else []
            dtypes = json.loads(dtype_sig) if dtype_sig else []
            count = c["count"]
            tot_us = c["total_us"]
            case = {
                "dims": dims,
                "dtypes": dtypes,
                "count": count,
                "baseline_latency_ms": round(tot_us / 1000.0 / max(count, 1), 6),
                "weight": round(tot_us, 3),            # total us contributed in the workload
                "weight_source": "trace" if dims else "regime_prior",
            }
            # serving phase MEASURED from the trace (dominant phase of this case's launches).
            # attribute_weights prefers this over its M-bucket heuristic when present.
            phc = c.get("phase") or {}
            if phc:
                case["regime"] = max(phc, key=phc.get)   # "prefill" | "decode"
            cases.append(case)
        cases.sort(key=lambda x: x["weight"], reverse=True)
        wsum = sum(x["weight"] for x in cases) or 1.0
        for x in cases:
            x["weight_norm"] = round(x["weight"] / wsum, 6)
        kentry = {
            "name": name,
            "short_name": short_name(name),
            "classification": cls,
            "backend_guess": backend,
            "editable": editable,
            "calls": d["calls"],
            "total_ms": round(d["total_us"] / 1000.0, 4),
            "pct_gpu_time": round(100.0 * d["total_us"] / total_us, 2) if total_us else 0.0,
            "num_cases": len(cases),
            "cases": cases,
        }
        phase = _kernel_phase(d)
        if phase:
            kentry["phase"] = phase
            kentry["served_regimes"] = ([phase] if phase != "both" else ["prefill", "decode"])
        kernels.append(kentry)
    return {
        "schema": "workload-v1",
        "total_gpu_time_ms": round(total_us / 1000.0, 4),
        "num_kernels": len(kernels),
        "kernels": kernels,
    }


def to_markdown(summ):
    L = []
    L.append(f"# Profile Top-{len(summ['top_kernels'])} — standardized summary\n")
    L.append(f"- source: `{summ['source']}`")
    L.append(f"- total GPU time: **{summ['total_gpu_time_ms']:.2f} ms** "
             f"over {summ['num_kernel_launches']} launches, "
             f"{summ['num_distinct_kernels']} distinct kernels\n")
    sv = summ.get("serving")
    if sv:
        L.append(f"- serving phase: {sv['n_prefill_steps']} prefill + {sv['n_decode_steps']} decode "
                 f"steps; decode batch captured={sv['decode_batch_captured']} "
                 f"steady={sv['decode_batch_steady']} "
                 f"(**{'STEADY' if sv['steady'] else 'NOT steady — decode %/latency biased low'}**); "
                 f"analytic_calls={sv.get('analytic_calls')}\n")
    L.append("| # | kernel | class | backend | edit | phase | calls | total ms | %gpu | avg us | shapes |")
    L.append("|--|--------|-------|---------|------|-------|-------|----------|------|--------|--------|")
    for k in summ["top_kernels"]:
        sh = "; ".join(json.dumps(s) for s in k["shapes"][:2]) if k["shapes"] else ""
        sh = (sh[:60] + "…") if len(sh) > 61 else sh
        L.append(f"| {k['rank']} | `{k['short_name']}` | {k['classification']} | "
                 f"{k['backend_guess']} | {'Y' if k['editable'] else 'N'} | {k.get('phase','?')} | "
                 f"{k['calls']} | "
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
    ap.add_argument("--workload-out", default="",
                    help="also write the per-(shape,dtype) weighted WORKLOAD MODEL json here "
                         "(for the kernel_workflow harness; needs --torch-trace for shapes)")
    ap.add_argument("--target", default="",
                    help="optional kernel-name substring filter for --workload-out")
    # serving-phase accounting (optional; pass the SAME ISL/OSL/concurrency as the bench). These only
    # EXPOSE the analytic per-phase call model (est_calls == serving_weight_model.analytic_calls) and
    # let est_shape snap decode M to the capture size; they never rescale the profile `weight`.
    ap.add_argument("--isl", type=int, default=0)
    ap.add_argument("--osl", type=int, default=0)
    ap.add_argument("--conc", type=int, default=0, help="max_concurrency of the bench")
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="max_num_batched_tokens (chunked-prefill budget) from server.log")
    ap.add_argument("--capture-sizes", default="",
                    help="comma list of cudagraph_capture_sizes (to snap decode est_shape M)")
    args = ap.parse_args()

    if not args.torch_trace and not args.rocprof_dir:
        ap.error("provide --torch-trace and/or --rocprof-dir")

    capture_sizes = [int(x) for x in args.capture_sizes.split(",") if x.strip()]
    chunk = args.prefill_chunk or None
    phk = dict(conc=args.conc, isl=args.isl, osl=args.osl, chunk=chunk,
               capture_sizes=capture_sizes)

    torch_agg = torch_total = torch_launch = None
    torch_pmeta = {}
    if args.torch_trace:
        torch_agg, torch_total, torch_launch, torch_pmeta = parse_torch_trace(args.torch_trace)
    rp_agg = rp_total = rp_launch = None
    if args.rocprof_dir:
        rp_agg, rp_total, rp_launch = parse_rocprof_dir(args.rocprof_dir)

    if rp_agg and torch_agg:
        summ = build_summary(rp_agg, rp_total, rp_launch, "merged", args.top, enrich=torch_agg,
                             phase_meta=torch_pmeta, **phk)
    elif rp_agg:
        summ = build_summary(rp_agg, rp_total, rp_launch, "rocprofv3", args.top, **phk)
    else:
        summ = build_summary(torch_agg, torch_total, torch_launch, "torch-trace", args.top,
                             phase_meta=torch_pmeta, **phk)

    js = json.dumps(summ, indent=2)
    md = to_markdown(summ)
    if args.out:
        with open(args.out + ".json", "w") as fh:
            fh.write(js)
        with open(args.out + ".md", "w") as fh:
            fh.write(md)
        sys.stderr.write(f"wrote {args.out}.json and {args.out}.md\n")

    # Workload model (shape+dtype weighted cases). Per-shape data lives only in the torch trace;
    # rocprof alone yields shape-unknown buckets (weight_source=regime_prior). Prefer the trace agg.
    if args.workload_out:
        w_agg = torch_agg if torch_agg else rp_agg
        w_tot = torch_total if torch_agg else rp_total
        wl = build_workload(w_agg or {}, w_tot or 0.0, args.top, args.target)
        with open(args.workload_out, "w") as fh:
            fh.write(json.dumps(wl, indent=2))
        sys.stderr.write(f"wrote {args.workload_out} ({wl['num_kernels']} kernels)\n")

    print(md)


if __name__ == "__main__":
    main()
