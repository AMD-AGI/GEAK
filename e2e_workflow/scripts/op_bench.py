#!/usr/bin/env python3
"""Single-op multi-backend bake-off + autotune for head kernels (GEMM / attention).

The Op Benchmarker uses this to optimize the HIGHEST-pct_gpu_time kernels — dense GEMM and attention —
which are usually library calls and were previously skipped by the kernel squad. A fixed-shape GEMM is
highly tunable: this script runs every available backend against the IMMUTABLE correctness oracle,
times each, optionally autotunes the editable ones (Triton), and reports the fastest-correct backend
plus any tuning artifact. It does NOT touch a server or measure e2e — that is the e2e Integrator's job
(this is the isolated Tier-A/Tier-B bake-off; Tier-C code rewrites go to the recursive kernel_workflow).

Task-dir contract (written by the Kernel Extractor PHASE=extract_op):
  <op>_task/
    meta.json         # op_kind=gemm|attn, dtype, a_shape/b_shape/transpose_b/bias (gemm) OR captured
                      # tensor spec (attn), math_contract, reference_io_sha256, regime
    reference_io.pt   # OPTIONAL golden {inputs..., output} oracle. If absent for GEMM, this script
                      # synthesizes inputs from meta shapes+dtype and computes the oracle with the
                      # DEFAULT backend (GEMM perf is value-independent; correctness is C=A·B[ᵀ]).

Usage:
  python3 op_bench.py --task <op_task_dir> [--backends hipblaslt,tunableop,rocblas,aiter,triton]
                      [--repeats 50] [--warmup 10] [--tol 2e-2] [--out result.json]
                      [--triton-autotune] [--seed 0]

Exit 0 always (unless the task dir is unreadable); per-backend failures are captured in the JSON so an
unavailable backend on this image is a recorded "skipped", not a crash.
"""
import argparse, hashlib, json, math, os, sys, time, traceback

# Shared harness measurement library (single source of truth for timing + correctness + Amdahl).
# op_bench.py lives in scripts/ so a plain import resolves; keep a guarded fallback so an old
# checkout without harness_lib still runs (with the naive tight-loop timing).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import harness_lib as _hlib
except Exception:
    _hlib = None


def _torch():
    """Import torch with the CWD SHADOW removed, then put sys.path back exactly as it was.

    A frozen task dir contains generated modules whose names collide with the stdlib — notably
    `unittest.py`. Python puts the CWD first on sys.path, so with the task dir as CWD that file wins over
    the stdlib `unittest` PACKAGE, and `torch/_guards.py`'s `import unittest.mock` dies with
    "No module named 'unittest.mock'; 'unittest' is not a package". op_bench then can't bench anything.

    The shield is scoped to this import and REVERTED in `finally`: `_resolve_callable` imports entry
    points by dotted module name (e.g. `csrc.cpp_itfs.pa.pa_ragged:paged_attention_ragged`) that resolve
    through the CWD, and it swallows ImportError into a `None` return — so permanently dropping the CWD
    would silently downgrade real backends to "unavailable" instead of failing loudly. Restore, don't strip.
    """
    saved = list(sys.path)
    cwd = os.getcwd()
    sys.path[:] = [p for p in sys.path
                   if p not in ("", ".") and os.path.abspath(p) != cwd]
    try:
        import torch
    finally:
        sys.path[:] = saved
    return torch


# ----------------------------------------------------------------------------- timing / correctness
def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Deployment graph context for timing. When the task's meta carries a regime, main() sets this to
# harness_lib.deployment_graph_mode(regime): the LIVE server replays decode under a CUDA/HIP graph
# (default on; off only under --enforce-eager / --disable-cuda-graph). BOTH baseline and candidate are
# timed through _time_call, so timing them under the SAME deployment graph mode is the fix for the
# "eager baseline" strawman — a candidate can't be scored against a baseline the deployment never runs.
# Left False (eager amortized) when meta has no regime, so regime-less tasks are byte-identical to before.
_GRAPH_MODE = False


def _time_call(fn, warmup, repeats):
    """Return (event_ms, wall_ms): the PRIMARY metric is CUDA-EVENT DEVICE time (GPU-timeline duration,
    excludes host dispatch); wall-clock is a REFERENCE (host+device). Timed via harness_lib.time_op under
    the deployment graph context (_GRAPH_MODE) and with the cache flushed cold each sample, so a candidate
    cannot win by collapsing Python launch overhead (device time already excludes it) and a memory-bound
    kernel is measured against real HBM traffic. Falls back to a naive wall-clock loop (event_ms==wall_ms)
    only if harness_lib is absent. Returns (None, None) if `fn` raises."""
    if _hlib is not None:
        r = _hlib.time_op(fn, warmup=warmup, repeats=repeats, graph=_GRAPH_MODE, detail=True)
        if not r:
            return None, None
        return r.get("ms"), r.get("wall_ms")
    torch = _torch()
    try:
        for _ in range(max(1, warmup)):
            fn()
        _sync(torch)
        samples = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            fn()
            _sync(torch)
            samples.append((time.perf_counter() - t0) * 1e3)
        samples.sort()
        m = samples[len(samples) // 2]
        return m, m
    except Exception:
        return None, None


def _correct(torch, out, ref, tol):
    """allclose-style check: |out-ref| <= atol + tol*|ref|, with a SCALE-RELATIVE atol so near-zero
    output elements (created by bias cancellation + bf16 double-rounding) don't blow up a pure relative
    metric. err = max(|out-ref| / (|ref| + atol)) — bounded near zero, comparable to `tol`."""
    try:
        if out.shape != ref.shape:
            return False, float("inf")
        out = out.float(); ref = ref.float()
        atol = tol * ref.abs().max().clamp_min(1e-6)        # absolute floor tied to the tensor scale
        diff = (out - ref).abs()
        ok = bool((diff <= (atol + tol * ref.abs())).all())
        err = diff.div(ref.abs() + atol).max().item()
        return ok, err
    except Exception:
        return False, float("inf")


# ------------------------------------------------------------------- INV-1/INV-3: denominator identity
# A speedup is a RATIO. The numerator is always measured here; the denominator is only meaningful if it
# is the thing the deployment actually runs. Every previous silent `baseline or target` fallback made the
# ratio a self-comparison (or a comparison against a reference this script itself invented) while still
# printing a headline number. These helpers make the denominator's provenance an explicit, recorded fact,
# and let main() WITHHOLD the ratio rather than publish an unsourced one.
#
# Op-kind agnostic by construction: nothing below inspects op_kind, dtype, shapes or kernel names.

DEGRADATIONS = []


def note_degradation(where, what, detail=""):
    d = {"where": where, "what": what, "detail": str(detail)[:400]}
    DEGRADATIONS.append(d)
    sys.stderr.write(f"[op_bench][degraded] {where}: {what}" + (f" — {detail}\n" if detail else "\n"))
    return d


# severity: "ok"        -> the denominator is the deployed path; a ratio against it is publishable
#           "unverified"-> plausibly the deployed path, but nothing proved it; publishable only with
#                          --no-denominator-strict, and always carries its provenance
#           "invalid"   -> provably not a denominator (self-comparison, or a reference we synthesized)
DENOM_SEVERITY = {
    "measured_backend_default": "ok",          # baseline row is a real backend we timed on this box
    "verified_baseline":        "ok",          # meta.baseline_validation proved it is the live seam
    "unverified_baseline":      "unverified",  # meta.baseline_callable, but no seam_contract.py verdict
    "target_fallback":          "invalid",     # no baseline declared -> candidate vs itself
    "synthesized_reference":    "invalid",     # oracle was fabricated, not captured from the seam
    "none":                     "invalid",     # nothing to divide by
}


def resolve_denominator(meta):
    """Resolve the baseline spec AND why we believe it is one. Never falls back silently.

    Returns {spec, provenance, severity, why}. `spec` may be non-empty even when severity is "invalid":
    callers may still want to RUN it (a self-comparison is a valid sanity check), they just may not
    report a speedup against it.
    """
    bl = str(meta.get("baseline_callable") or "").strip()
    tgt = str(meta.get("target_callable") or "").strip()
    v = meta.get("baseline_validation")
    verified = (isinstance(v, dict) and v.get("contract") == "baseline_identity" and v.get("ok") is True)

    def out(spec, prov, why):
        return {"spec": spec, "provenance": prov, "severity": DENOM_SEVERITY[prov], "why": why}

    if meta.get("synthesized") is True:
        return out(bl or tgt, "synthesized_reference",
                   "meta.synthesized=true: the reference was written by the extractor, not captured "
                   "from the live seam, so it is an oracle for CORRECTNESS only, never a perf denominator")
    if bl and verified:
        return out(bl, "verified_baseline",
                   "seam_contract.py --mode baseline reported ok=true for this spec")
    if bl:
        return out(bl, "unverified_baseline",
                   "meta.baseline_callable is declared but carries no baseline_validation verdict "
                   "(run scripts/seam_contract.py --mode baseline)")
    if tgt:
        return out(tgt, "target_fallback",
                   "no baseline_callable in meta.json; using target_callable would compare the "
                   "candidate against ITSELF")
    return out("", "none", "meta.json declares neither baseline_callable nor target_callable")


def _delegation_status(meta):
    """Some heads have no in-process bake-off at all: their only lever is a server-level flag owned by
    the Config-Tuner track. That is a legitimate outcome ONLY while that track is actually running. If it
    is off, 'delegated' means 'nobody measured this', which must surface as a reject code, not as a pass.

    Recognised meta keys (any one of them, all optional):
      delegated_config_track: true|false      config_track_enabled: true|false
    Absent -> "unknown": we say so instead of assuming enabled.
    """
    for key in ("delegated_config_track", "config_track_enabled"):
        if key in meta and meta[key] is not None:
            return "enabled" if bool(meta[key]) else "disabled"
    return "unknown"


# --------------------------------------------------------------- INV-3: generic captured-oracle replay
def _rehydrate(obj, torch, device):
    """Inverse of capture_shapes._snapshot: {'__tensor__':True,'data':...} -> a device tensor.
    Structure-preserving over list/tuple/dict; scalars pass through; un-snapshottable objects
    (recorded as {'__repr__': ...}) come back as a sentinel so the caller can refuse to replay."""
    if isinstance(obj, dict) and obj.get("__tensor__"):
        return obj["data"].to(device)
    if isinstance(obj, dict) and "__repr__" in obj:
        return _UNREPLAYABLE
    if isinstance(obj, (list, tuple)):
        return type(obj)(_rehydrate(v, torch, device) for v in obj)
    if isinstance(obj, dict):
        return {k: _rehydrate(v, torch, device) for k, v in obj.items()}
    return obj


class _Unreplayable:
    def __repr__(self):
        return "<unreplayable: capture recorded only repr()>"


_UNREPLAYABLE = _Unreplayable()


def _has_unreplayable(obj):
    if obj is _UNREPLAYABLE:
        return True
    if isinstance(obj, (list, tuple)):
        return any(_has_unreplayable(v) for v in obj)
    if isinstance(obj, dict):
        return any(_has_unreplayable(v) for v in obj.values())
    return False


def load_oracle_records(task, torch, device):
    """Return (records, err). Each record: {sig, regime, args, kwargs, output} with tensors on `device`.
    `err` is a human string when the oracle cannot be replayed; records is then []."""
    iopath = os.path.join(task, "reference_io.pt")
    if not os.path.exists(iopath):
        return [], "no reference_io.pt in the task dir"
    try:
        blob = torch.load(iopath, map_location="cpu", weights_only=False)
    except Exception as e:
        return [], f"reference_io.pt unreadable: {e!r}"
    recs = blob.get("records") if isinstance(blob, dict) else None
    if not recs:
        return [], "reference_io.pt carries no 'records' (not a capture_shapes oracle)"
    out = []
    for r in recs:
        args = _rehydrate(r.get("args", ()), torch, device)
        kwargs = _rehydrate(r.get("kwargs", {}), torch, device)
        ref = _rehydrate(r.get("output", None), torch, device)
        if _has_unreplayable(args) or _has_unreplayable(kwargs):
            continue
        out.append({"sig": r.get("sig", ""), "regime": r.get("regime", ""),
                    "args": tuple(args) if isinstance(args, (list, tuple)) else (args,),
                    "kwargs": dict(kwargs) if isinstance(kwargs, dict) else {},
                    "ref": ref})
    if not out:
        return [], (f"all {len(recs)} recorded case(s) contain non-tensor arguments the capture could "
                    f"only store as repr() — cannot replay this seam in-process")
    return out, ""


def _out_param_of(rec, meta):
    """For an IN-PLACE seam (returns None, writes into a caller-supplied buffer) the oracle's `output`
    snapshot is None and the golden values live in one of the INPUTS after the call. Which one is a fact
    the extractor records; we do not guess from names here beyond a last-resort default, and we say so.

    Returns (kind, key) with kind in {"kwarg","arg",""}.
    """
    ev = meta.get("seam_runtime_evidence") or {}
    names = [str(n) for n in (ev.get("inplace_params") or []) if str(n)]
    for n in names:
        if n in rec["kwargs"]:
            return "kwarg", n
    if names:
        return "", ""            # declared, but not present in this record -> refuse, don't guess
    return "", ""


def _clone_inputs(rec, torch):
    """A fresh copy of args/kwargs so an in-place seam cannot poison the next timing iteration."""
    def cl(o):
        if torch.is_tensor(o):
            return o.clone()
        if isinstance(o, (list, tuple)):
            return type(o)(cl(v) for v in o)
        if isinstance(o, dict):
            return {k: cl(v) for k, v in o.items()}
        return o
    return cl(rec["args"]), cl(rec["kwargs"])


def bench_captured_replay(args, meta):
    """Time the CURRENT seam callable against its captured oracle by replaying the recorded call.

    This replaces the former attention stub, which returned {"correct": True, "ms": None} unconditionally
    — a pass with no measurement behind it. It is op-kind agnostic: it replays whatever capture_shapes
    recorded at whatever seam, so it serves attention, MoE routing, norms, or anything else that has an
    oracle but no cross-backend bake-off.

    Each result row carries `denominator_provenance` so main() can decide whether the ratio is
    publishable.
    """
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    denom = resolve_denominator(meta)
    delegation = _delegation_status(meta)
    results = []

    recs, err = load_oracle_records(args.task, torch, device)
    if err:
        note_degradation("bench_captured_replay", "no replayable oracle", err)
        return [{"backend": "current", "available": False, "correct": False, "ms": None,
                 "raised": False, "reason_code": "invalid_denominator",
                 "denominator_provenance": denom["provenance"],
                 "note": f"cannot measure this seam: {err}"}]

    # Bench the DOMINANT recorded case (most-repeated sig) — same convention as the blockscale path.
    recs.sort(key=lambda r: -int((meta.get("shape_counts_by_sig") or {}).get(r["sig"], 0)))
    rec = recs[0]

    plan = []
    tgt = str(meta.get("target_callable") or "").strip()
    if tgt:
        plan.append(("current", tgt))
    if denom["spec"] and denom["spec"] != tgt:
        plan.append(("baseline", denom["spec"]))

    if not plan:
        note_degradation("bench_captured_replay", "no callable to time", denom["why"])
        return [{"backend": "current", "available": False, "correct": False, "ms": None,
                 "raised": False, "reason_code": "invalid_denominator",
                 "denominator_provenance": denom["provenance"], "note": denom["why"]}]

    # In-place seam: capture_shapes snapshots the arguments AFTER invoking the original, so the recorded
    # out-buffer already holds the golden values. Replay therefore takes the GOLDEN from that buffer and
    # zeroes the copy it hands to the callee — otherwise the callee would be handed the answer.
    ip_kind, ip_key = _out_param_of(rec, meta)
    golden = rec["ref"]
    if ip_kind == "kwarg" and torch.is_tensor(rec["kwargs"].get(ip_key)):
        golden = rec["kwargs"][ip_key].clone()

    for name, spec in plan:
        fn = _resolve_callable(spec)
        if fn is None:
            results.append({"backend": name, "available": False, "correct": False, "ms": None,
                            "raised": False, "reason_code": "candidate_unresolvable",
                            "denominator_provenance": denom["provenance"],
                            "note": f"callable not importable: {spec}"})
            continue

        def call_once():
            a, k = _clone_inputs(rec, torch)
            if ip_kind == "kwarg" and torch.is_tensor(k.get(ip_key)):
                k[ip_key].zero_()
            r = fn(*a, **k)
            return r if r is not None else (k.get(ip_key) if ip_kind == "kwarg" else None)

        try:
            out = call_once(); _sync(torch)
            out = call_once()
        except Exception as e:
            results.append({"backend": name, "available": True, "correct": False, "ms": None,
                            "raised": True, "denominator_provenance": denom["provenance"],
                            "note": f"replay raised: {e!r}"})
            continue

        ref = golden
        if out is None or ref is None or not torch.is_tensor(out) or not torch.is_tensor(ref):
            # We CAN time it, but we cannot certify it. Say exactly that instead of asserting correct.
            ms, wall_ms = _time_call(call_once, args.warmup, args.repeats)
            why = ("seam returns None and meta.seam_runtime_evidence.inplace_params does not name the "
                   "output buffer" if out is None else
                   "oracle recorded no comparable output tensor for this case")
            note_degradation("bench_captured_replay", f"{name}: timed but unverified", why)
            results.append({"backend": name, "available": True, "correct": None, "ms": round(ms, 4) if ms else None,
                            "wall_ms": round(wall_ms, 4) if wall_ms else None, "raised": False,
                            "denominator_provenance": denom["provenance"],
                            "note": f"{spec} @ {rec['sig']} ({rec['regime']}) — UNVERIFIED: {why}"})
            continue

        ok, rel = _correct(torch, out, ref, args.tol)
        ms, wall_ms = _time_call(call_once, args.warmup, args.repeats)
        results.append({"backend": name, "available": True, "correct": bool(ok),
                        "max_rel_err": round(rel, 5) if math.isfinite(rel) else None,
                        "ms": round(ms, 4) if ms else None,
                        "wall_ms": round(wall_ms, 4) if wall_ms else None, "raised": False,
                        "denominator_provenance": denom["provenance"],
                        "note": f"{spec} @ {rec['sig']} ({rec['regime']})"})

    if delegation == "disabled":
        note_degradation("bench_captured_replay", "delegated config track is DISABLED",
                         "cross-backend comparison for this head has no owner; op-level timing above is "
                         "a reference only")
        results.append({"backend": "delegated_config_track", "available": False, "correct": False,
                        "ms": None, "raised": False, "reason_code": "delegated_track_disabled",
                        "denominator_provenance": denom["provenance"],
                        "note": "server-level backend comparison is delegated to the Config Tuner, and "
                                "that track is disabled for this run — nothing measured it"})
    elif delegation == "unknown":
        note_degradation("bench_captured_replay", "delegated config track state unknown",
                         "meta declares neither delegated_config_track nor config_track_enabled")
    return results


# ----------------------------------------------------------------------------- GEMM bake-off
def _dtype(torch, name):
    # Prefer the shared, ARCH-DRIVEN resolver so a bare "fp8"/"fp8_e4m3" picks the running GPU's fp8
    # variant (MI300 gfx942 -> fnuz; MI355 gfx950 -> OCP fn) instead of a hardcoded fnuz. Fall back to
    # the local (fnuz-default) table only on an old checkout without harness_lib.
    if _hlib is not None:
        return _hlib.regime_dtype(name, torch)
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
            "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32,
            "fp8": getattr(torch, "float8_e4m3fnuz", torch.bfloat16),
            "fp8_e4m3fnuz": getattr(torch, "float8_e4m3fnuz", torch.bfloat16),
            "fp8_e5m2fnuz": getattr(torch, "float8_e5m2fnuz", torch.bfloat16),
            "fp8_e4m3fn": getattr(torch, "float8_e4m3fn", torch.bfloat16),
            "fp8_e5m2": getattr(torch, "float8_e5m2", torch.bfloat16),
            }.get(str(name).lower(), torch.bfloat16)


def _resolve_shape(shape, meta):
    """Resolve a shape that may carry SYMBOLIC dims (e.g. "M" for a dynamic GEMM row count) into
    concrete ints. String dims are mapped via meta: "M"/"m*"/"-1"/None -> a representative value from
    meta["m_buckets"] (the dominant = LARGEST profiled bucket). Raises a CLEAR error if a symbolic dim
    cannot be resolved, so the caller records a meaningful harness error instead of a cryptic
    `randn(str, int, ...)` TypeError (the exact bug this guards against)."""
    if not shape:
        raise ValueError("empty/None shape")
    buckets = [int(b) for b in (meta.get("m_buckets") or []) if str(b).strip().lstrip("-").isdigit()]
    rep_m = max(buckets) if buckets else None
    out = []
    for d in shape:
        if isinstance(d, bool):
            raise ValueError(f"bool dim {d!r} in shape {shape}")
        if isinstance(d, int):
            out.append(int(d)); continue
        s = str(d).strip().lower()
        if s.lstrip("-").isdigit() and int(s) > 0:
            out.append(int(s)); continue
        if s in ("m", "-1", "none", "") or s.startswith("m"):
            if rep_m is None:
                raise ValueError(f"symbolic dim {d!r} in shape {shape} but meta has no usable m_buckets to resolve it")
            out.append(rep_m); continue
        raise ValueError(f"unresolvable symbolic dim {d!r} in shape {shape} (give ints, or an m_buckets list)")
    return out


def _resolve_callable(spec):
    """'module:attr' -> callable (or None if unimportable)."""
    if not spec or ":" not in str(spec):
        return None
    try:
        import importlib
        mod_name, attr = str(spec).split(":", 1)
        m = importlib.import_module(mod_name)
        return getattr(m, attr, None)
    except Exception:
        return None


def _is_blockscale_gemm(meta):
    """True for a quantized block-scaled GEMM (fp8 a8w8 blockscale etc.) — these CANNOT be benched by the
    generic dense torch-BLAS path (fp8 + per-block scales), so they take the dedicated blockscale path."""
    dt = str(meta.get("dtype", "")).lower()
    qs = str(meta.get("quant_scheme", "")).lower()
    is_fp8 = ("fp8" in dt or "e4m3" in dt or "e5m2" in dt)
    has_block = bool(meta.get("weight_block_size")) or "block" in qs
    return is_fp8 and has_block


def _is_grouped_or_quant_gemm(meta):
    """True for a grouped/packed-quant GEMM the dense torch-BLAS bake-off CANNOT represent:
    MoE fused-experts (3D [E,N,K] weights), int4/awq/gptq packed weights, or per-group quant.
    These are Tier-C authored grouped-GEMM ops (GEMM1->act->GEMM2 over experts on packed weights),
    NOT an F.linear bake-off candidate — forcing them through bench_gemm raises ValueError (no
    a_shape/b_shape) or RuntimeError (.t() on a 3D weight). We classify them out so op_bench records a
    clean "needs authored kernel" result instead of a spurious harness self-fault."""
    kc = str(meta.get("kernel_class", "")).lower()
    dt = str(meta.get("dtype", "")).lower()
    qs = str(meta.get("quant_scheme", "")).lower()
    if any(t in kc for t in ("moe", "grouped", "experts")):
        return True
    if any(t in dt for t in ("int4", "int8", "uint4", "awq", "gptq", "w4a16", "w8a16")):
        return True
    if any(t in qs for t in ("awq", "gptq", "int4", "compressed_tensors")):
        return True
    b = meta.get("b_shape")
    if isinstance(b, (list, tuple)) and len(b) >= 3:        # explicit 3D weight => grouped
        return True
    sh = meta.get("shape")
    if isinstance(sh, dict) and "E" in sh:                  # structured MoE shape block (E,N,K,...)
        return True
    return False


def _synth_blockscale_case(torch, meta, M, device, seed):
    """Synthesize an fp8 a8w8 blockscale GEMM case + its dequant oracle, MIRRORING the extracted
    unittest's `_synth_case` exactly (so op_bench's correctness target matches the immutable oracle).
    Returns {x(fp8 [M,K]), w(fp8 [N,K]), x_scale([M,sK]), w_scale([sN,sK]), ref(out_dt [M,N]), M, out_dt}."""
    N = int(meta["b_shape"][0]); K = int(meta["b_shape"][1])
    blk = meta.get("weight_block_size") or [128, 128]
    BLK_N, BLK_K = int(blk[0]), int(blk[1])
    fp8 = _dtype(torch, meta.get("dtype", "fp8"))
    out_dt = _dtype(torch, meta.get("out_dtype", "bf16"))
    sK = (K + BLK_K - 1) // BLK_K
    sN = (N + BLK_N - 1) // BLK_N
    fmax = float(torch.finfo(fp8).max)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x_hp = (torch.randn(M, K, generator=gen, dtype=torch.float32, device=device) * 0.1)
    w_hp = (torch.randn(N, K, generator=gen, dtype=torch.float32, device=device) * 0.05)
    padK = sK * BLK_K
    xb = torch.zeros(M, padK, device=device); xb[:, :K] = x_hp
    x_blk = xb.reshape(M, sK, BLK_K)
    x_scale = (x_blk.abs().amax(dim=2).clamp_min(1e-8) / fmax).to(torch.float32)          # [M, sK]
    x_q = (x_blk / x_scale[:, :, None]).clamp(-fmax, fmax).reshape(M, padK)[:, :K].to(fp8)
    padN = sN * BLK_N
    wb = torch.zeros(padN, padK, device=device); wb[:N, :K] = w_hp
    w_blk = wb.reshape(sN, BLK_N, sK, BLK_K)
    w_scale = (w_blk.abs().amax(dim=(1, 3)).clamp_min(1e-8) / fmax).to(torch.float32)      # [sN, sK]
    w_q = (w_blk / w_scale[:, None, :, None]).clamp(-fmax, fmax).reshape(padN, padK)[:N, :K].to(fp8)
    x_scale_full = x_scale.repeat_interleave(BLK_K, dim=1)[:, :K]
    x_deq = x_q.to(torch.float32) * x_scale_full
    w_scale_full = w_scale.repeat_interleave(BLK_N, dim=0).repeat_interleave(BLK_K, dim=1)[:N, :K]
    w_deq = w_q.to(torch.float32) * w_scale_full
    ref = (x_deq @ w_deq.t()).to(out_dt)
    return {"x": x_q, "w": w_q, "x_scale": x_scale, "w_scale": w_scale, "ref": ref, "M": M, "out_dt": out_dt}


def bench_blockscale_gemm(args, meta):
    """Bake-off for an fp8 a8w8 blockscale GEMM head. The generic dense torch-BLAS backends cannot
    represent this op (fp8 + per-block scales), so the candidates are the meta callable(s):
    the live baseline blockscale path + its bpreshuffle variant (same signature
    fn(x, w, x_scale, w_scale, dtype=out)). Benched at the DOMINANT (largest) m_bucket — the GPU-time
    mass. Returns the standard per-backend results list."""
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buckets = [int(b) for b in (meta.get("m_buckets") or []) if str(b).strip().lstrip("-").isdigit()]
    if not buckets:
        buckets = [int(_resolve_shape(meta.get("a_shape") or ["M"], meta)[0])]
    M = max(buckets)
    case = _synth_blockscale_case(torch, meta, M, device, args.seed)
    out_dt = case["out_dt"]

    def _call(fn):
        return fn(case["x"], case["w"], case["x_scale"], case["w_scale"], dtype=out_dt)

    results = []

    def record(name, fn, note=""):
        try:
            _call(fn); _sync(torch)          # warmup (compile/autotune) on a clean launch
            out = _call(fn)
        except Exception as e:
            # An EXCEPTION (not a slow/incorrect number) -> candidate could not run. The op_benchmarker
            # treats "all candidates raised" as a harness self-fault (see its role); we surface it clearly.
            results.append({"backend": name, "available": True, "correct": False, "ms": None,
                            "denominator_provenance": denom["provenance"],
                            "note": f"call raised: {e!r}", "raised": True})
            return
        ok, err = _correct(torch, out, case["ref"], args.tol)
        ms, wall_ms = _time_call(lambda: _call(fn), args.warmup, args.repeats)
        results.append({"backend": name, "available": True, "correct": bool(ok),
                        "max_rel_err": round(err, 5) if math.isfinite(err) else None,
                        "ms": round(ms, 4) if ms else None,
                        "wall_ms": round(wall_ms, 4) if wall_ms else None,
                        "denominator_provenance": denom["provenance"],
                        "note": note, "raised": False})

    # INV-1: the denominator is resolved WITH its provenance, never by a silent `baseline or target`.
    # We still run whatever spec we have (a self-comparison is a usable sanity check); what changes is
    # that main() now knows whether the resulting ratio may be published.
    denom = resolve_denominator(meta)
    if denom["severity"] != "ok":
        note_degradation("bench_blockscale_gemm", f"denominator is {denom['provenance']}", denom["why"])
    base_spec = denom["spec"]
    tgt_spec = meta.get("target_callable") or base_spec
    seen = set()
    plan = [("aiter_blockscale", base_spec)]
    if tgt_spec and tgt_spec != base_spec:
        plan.append(("aiter_blockscale_target", tgt_spec))
    # bpreshuffle variant (large-M prefill lever) lives at top-level aiter, NOT the triton submodule.
    # Plain-weight call may not match its preshuffled-weight signature -> guarded by record()'s try/except
    # + the correctness gate (a wrong-layout call fails correctness and is simply not a winner; the real
    # CK/asm/bpreshuffle race is the aiter DB tune in Tier-B, which the op_benchmarker drives).
    plan.append(("aiter_bpreshuffle", "aiter:gemm_a8w8_blockscale_bpreshuffle"))
    for name, spec in plan:
        fn = _resolve_callable(spec)
        if fn is None:
            results.append({"backend": name, "available": False, "correct": False, "ms": None,
                            "note": f"callable not importable: {spec}", "raised": False})
            continue
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        record(name, fn, note=f"{spec} @ M={M} (dominant bucket)")
    return results


def _load_or_synth_gemm(torch, task, meta, device, seed):
    """Return (A, B, bias, transpose_b, ref). Prefer the recorded oracle; else synthesize + compute ref
    with the default backend (perf is value-independent; this only fixes the correctness target)."""
    dt = _dtype(torch, meta.get("dtype", "bf16"))
    transpose_b = bool(meta.get("transpose_b", True))  # F.linear style by default
    use_bias = bool(meta.get("bias", False))
    iopath = os.path.join(task, "reference_io.pt")
    if os.path.exists(iopath):
        blob = torch.load(iopath, map_location=device)
        # accept a few shapes of recorded blob
        A = blob.get("A") if isinstance(blob, dict) else None
        B = blob.get("B") if isinstance(blob, dict) else None
        bias = blob.get("bias") if isinstance(blob, dict) else None
        ref = blob.get("output") if isinstance(blob, dict) else None
        if A is not None and B is not None:
            A = A.to(device); B = B.to(device)
            bias = bias.to(device) if bias is not None else None
            if ref is None:
                ref = (A @ (B.t() if transpose_b else B))
                if bias is not None:
                    ref = ref + bias
            return A.to(dt), B.to(dt), (bias.to(dt) if bias is not None else None), transpose_b, ref.float()
    # synthesize from shapes (resolve any SYMBOLIC dims like "M" via meta.m_buckets -> ints first)
    a_shape0 = meta.get("a_shape"); b_shape0 = meta.get("b_shape")
    if not (a_shape0 and b_shape0):
        raise ValueError("gemm task has neither reference_io.pt nor a_shape/b_shape in meta.json")
    a_shape = _resolve_shape(a_shape0, meta)
    b_shape = _resolve_shape(b_shape0, meta)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = (torch.randn(*a_shape, generator=g) * 0.1).to(device=device, dtype=dt)
    B = (torch.randn(*b_shape, generator=g) * 0.1).to(device=device, dtype=dt)
    bias = None
    if use_bias:
        n = a_shape[0] if False else (b_shape[0] if transpose_b else b_shape[-1])
        bias = (torch.randn(n, generator=g) * 0.1).to(device=device, dtype=dt)
    ref = (A.float() @ (B.float().t() if transpose_b else B.float()))
    if bias is not None:
        ref = ref + bias.float()
    return A, B, bias, transpose_b, ref


def _gemm_fn(torch, A, B, bias, transpose_b):
    """A canonical GEMM closure using torch (dispatches to whatever BLAS backend is active)."""
    if transpose_b:
        import torch.nn.functional as F
        return lambda: F.linear(A, B, bias)
    if bias is None:
        return lambda: torch.matmul(A, B)
    return lambda: torch.addmm(bias, A, B) if A.dim() == 2 else (torch.matmul(A, B) + bias)


def _set_prefer_blas(torch, lib):
    """Best-effort switch of torch's BLAS backend (ROCm: hipblaslt vs rocblas). Returns True if applied."""
    try:
        fn = torch.backends.cuda.preferred_blas_library
        fn(lib)  # 'hipblaslt' / 'cublaslt' map; 'cublas'/'rocblas' for the non-Lt path
        return True
    except Exception:
        return False


def _tunableop(torch, enable, tuning, filename=None):
    try:
        t = torch.cuda.tunable
        t.enable(bool(enable))
        t.tuning_enable(bool(tuning))
        if filename:
            try:
                t.set_filename(filename)
            except Exception:
                pass
        return True
    except Exception:
        return False


def bench_gemm(args, meta):
    torch = _torch()
    # Quantized block-scaled GEMM (fp8 a8w8 blockscale, …) takes the dedicated path — the generic dense
    # torch-BLAS backends below cannot represent fp8 + per-block scales (this is the head that used to die
    # on `randn(str,int,...)`; it is now benched with the immutable-oracle construction).
    if _is_blockscale_gemm(meta):
        return bench_blockscale_gemm(args, meta)
    # Grouped/packed-quant MoE GEMM (int4_w4a16 / awq / gptq / 3D [E,N,K] / structured shape:{E,..}):
    # the dense torch-BLAS bake-off below cannot represent packed/3D weights — it would raise ValueError
    # (no a_shape/b_shape) or RuntimeError (.t() on a 3D weight). Record a clean, non-raising skip so the
    # dominant head is reported as "needs a Tier-C authored grouped GEMM", NOT a harness self-fault.
    if _is_grouped_or_quant_gemm(meta):
        return [{
            "backend": "grouped_quant_gemm", "available": False, "correct": False,
            "ms": None, "raised": False,
            "note": ("grouped/quantized MoE GEMM (kernel_class=%s dtype=%s): not a dense torch-BLAS "
                     "bake-off candidate; requires a Tier-C authored fused-experts grouped GEMM "
                     "(GEMM1->act->GEMM2 over experts on packed weights). Skipped at op_bench "
                     "(dense path cannot represent packed/3D weights)."
                     % (meta.get("kernel_class"), meta.get("dtype"))),
        }]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A, B, bias, transpose_b, ref = _load_or_synth_gemm(torch, args.task, meta, device, args.seed)
    ref = ref.to(device)
    # Default excludes the experimental triton stub (it's a placeholder; real triton GEMM is a Tier-C
    # kernel-squad rewrite, not a bake-off candidate). Request it explicitly with --backends if wanted.
    want = [b.strip() for b in args.backends.split(",") if b.strip()] if args.backends else \
        ["hipblaslt", "tunableop", "rocblas", "ck", "aiter", "flydsl"]
    results = []

    def record(name, fn, note="", artifact=""):
        try:
            fn()            # warmup: triggers compile/autotune so correctness is checked on a CLEAN
            _sync(torch)    # launch, not on the autotune benchmarking pass (which returns a dirty buffer)
            out = fn()
        except Exception as e:
            results.append({"backend": name, "available": False, "correct": False, "ms": None,
                            "note": f"call failed: {e!r}", "artifact": artifact})
            return
        ok, err = _correct(torch, out, ref, args.tol)
        ms, wall_ms = _time_call(fn, args.warmup, args.repeats)
        results.append({"backend": name, "available": True, "correct": bool(ok),
                        "max_rel_err": round(err, 5) if math.isfinite(err) else None,
                        "ms": round(ms, 4) if ms else None,
                        "wall_ms": round(wall_ms, 4) if wall_ms else None,
                        "note": note, "artifact": artifact})

    base_fn = _gemm_fn(torch, A, B, bias, transpose_b)

    # hipBLASLt (default Lt path)
    if "hipblaslt" in want:
        _set_prefer_blas(torch, "hipblaslt"); _tunableop(torch, False, False)
        record("hipblaslt", base_fn, note="torch default Lt path")

    # PyTorch TunableOp — tune once, persist CSV, freeze. The CSV is the DEPLOYABLE artifact: loaded at
    # server startup (TUNING=0) it is baked into the cuda-graph capture, unlike tune-during-serving which
    # the graph bypasses. Explicit write_file() so the CSV actually persists for the integrate step.
    if "tunableop" in want:
        csv = os.path.join(args.task, "tunableop.csv")
        on = _tunableop(torch, True, True, csv)
        if on:
            base_fn()  # triggers a tuning pass for this shape
            _sync(torch)
            try:
                torch.cuda.tunable.write_file(csv)  # persist the tuned solution(s)
            except Exception:
                pass
            _tunableop(torch, True, False, csv)  # freeze: use tuned, stop searching
            record("tunableop", base_fn, note="PYTORCH TunableOp tuned (CSV deployable at startup)", artifact=csv)
            _tunableop(torch, False, False)
        else:
            results.append({"backend": "tunableop", "available": False, "correct": False,
                            "ms": None, "note": "torch.cuda.tunable API unavailable"})

    # hipBLASLt offline tuning: NOT reachable from a PyTorch process. The real env is
    # HIPBLASLT_TUNING_OVERRIDE_FILE (consume-only, generated by the `hipblaslt-bench` CLI which isn't
    # installed here), and libtorch_hip does not read it — PyTorch's GEMM tuner is TunableOp (which itself
    # enumerates hipBLASLt solutions, see PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED). So "tune hipBLASLt" == the
    # tunableop backend above. Only attempted if explicitly requested, and always reported as such.
    if "hipblaslt_tuned" in want:
        results.append({"backend": "hipblaslt_tuned", "available": False, "correct": False, "ms": None,
                        "note": "not a PyTorch-level lever: HIPBLASLT_TUNING_OVERRIDE_FILE is consume-only "
                                "(needs hipblaslt-bench, not installed) and libtorch_hip ignores it; "
                                "use the 'tunableop' backend, which enumerates hipBLASLt solutions"})

    # rocBLAS (non-Lt path)
    if "rocblas" in want:
        applied = _set_prefer_blas(torch, "cublas")  # maps to rocblas on ROCm
        record("rocblas", base_fn, note="torch non-Lt path" + ("" if applied else " (switch unconfirmed)"))
        _set_prefer_blas(torch, "hipblaslt")

    # CK / ck_tile GEMM — best-effort via torch's preferred-BLAS 'ck' (if this ROCm build exposes it).
    if "ck" in want:
        applied = _set_prefer_blas(torch, "ck")
        if applied:
            record("ck", base_fn, note="torch preferred_blas=ck")
        else:
            results.append({"backend": "ck", "available": False, "correct": False, "ms": None,
                            "note": "torch preferred_blas_library('ck') unsupported in this build"})
        _set_prefer_blas(torch, "hipblaslt")

    # aiter GEMM — scan the installed aiter API for a GEMM-like entrypoint (names vary by version).
    if "aiter" in want:
        record("aiter", lambda: _aiter_gemm(A, B, bias, transpose_b), note="aiter fused gemm (auto-probed)")

    # FlyDSL GEMM — aiter's Python kernel DSL hgemm (the SOTA fp8/MoE/quantized-GEMM author backend on
    # gfx942/950). This is a REAL implementation (unlike the retired triton stub), so it IS a first-class
    # bake-off candidate. Gated by is_flydsl_available(); unavailable -> recorded "skipped", not a crash.
    if "flydsl" in want:
        try:
            from aiter.ops.flydsl.utils import is_flydsl_available
            if not is_flydsl_available():
                results.append({"backend": "flydsl", "available": False, "correct": False, "ms": None,
                                "note": "is_flydsl_available()==False (flydsl not installed on this image)", "artifact": ""})
            else:
                record("flydsl", lambda: _flydsl_gemm(A, B, bias, transpose_b),
                       note="aiter flydsl_hgemm (a@b.T+bias, default tiling; per-shape knobs tuned in Tier-B/C)")
        except Exception as e:
            results.append({"backend": "flydsl", "available": False, "correct": False, "ms": None,
                            "note": f"flydsl unavailable: {e!r}", "artifact": ""})

    # Triton matmul — RETIRED as a bake-off candidate. This is a naive placeholder, NOT a real Triton
    # GEMM, and it is never in the default `want` list. A real Triton (or HIP/CK) implementation now
    # comes from the AUTHOR route: the Op Benchmarker emits an `author_plan` and the orchestrator runs
    # `kernel_workflow` mode=author/optimize to write + tune it against the immutable oracle. This stub is
    # kept ONLY for ad-hoc `--backends triton` debugging; do not rely on its number for routing.
    # The weight transpose (B[N,K]->[K,N]) is done ONCE here, NOT inside the timed loop.
    if "triton" in want:
        try:
            triton, _mm = _get_triton_mm()
            Kr = A.shape[-1]
            a2 = A.reshape(-1, Kr)
            Bm = (B.t() if transpose_b else B).contiguous()  # [K,N], once
            Mr, Nr = a2.shape[0], Bm.shape[-1]
            cbuf = torch.empty((Mr, Nr), device=A.device, dtype=A.dtype)

            def _tri():
                grid = lambda META: (triton.cdiv(Mr, META["BLOCK_M"]) * triton.cdiv(Nr, META["BLOCK_N"]),)
                _mm[grid](a2, Bm, cbuf, Mr, Nr, Kr, a2.stride(0), a2.stride(1),
                          Bm.stride(0), Bm.stride(1), cbuf.stride(0), cbuf.stride(1))
                out = cbuf.reshape(*A.shape[:-1], Nr)
                return (out + bias) if bias is not None else out

            record("triton", _tri,
                   note="triton placeholder (RETIRED; real triton comes from the author route, not this stub)")
        except Exception as e:
            results.append({"backend": "triton", "available": False, "correct": False, "ms": None,
                            "note": f"triton unavailable: {e!r}", "artifact": ""})

    return results


def _flydsl_gemm(A, B, bias, transpose_b):
    """aiter FlyDSL hgemm: out = a @ b.T (+bias), with a=[M,K], b=[N,K] (TN, linear-weight layout).
    Uses default tiling for a correctness-first bake-off number; the per-shape knobs (tile_m/n/k,
    split_k, b_preshuffle, ...) are what Tier-B/Tier-C tune. Value-independent perf, so the synthesized
    inputs from the oracle are fine for timing.

    flydsl_hgemm is **bf16/fp16 only**. For an fp8 (a8w8) head GEMM the flydsl path is
    `flydsl_preshuffle_gemm_a8(XQ, WQ, x_scale, w_scale, Out, ...)`, which needs the quantized operands +
    per-token/per-channel scales that this bake-off's plain (A,B,bias) synth does not carry. Rather than
    fabricate scales (a wrong number is worse than a skip), raise a clear guidance error so the harness
    records flydsl as a graceful "skipped" for fp8 — the live fp8-flydsl win is reached via the aiter
    per-shape DB tune (gradlib races `libtype=flydsl`; deploy `AITER_CONFIG_GEMM_BF16`) and/or the
    author route (`target_language=flydsl`, baseline = `flydsl_preshuffle_gemm_a8`)."""
    _t = _torch()
    if A.dtype in (getattr(_t, "float8_e4m3fnuz", None), getattr(_t, "float8_e5m2fnuz", None),
                   getattr(_t, "float8_e4m3fn", None), getattr(_t, "float8_e5m2", None)):
        raise RuntimeError(
            "flydsl_hgemm is bf16/fp16 only; fp8 a8w8 GEMM uses flydsl_preshuffle_gemm_a8 (needs "
            "x_scale/w_scale). Reach flydsl-fp8 via the aiter DB tune (libtype=flydsl) or the "
            "author route, not this plain bake-off probe.")
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
    Kr = A.shape[-1]
    a2 = A.reshape(-1, Kr).contiguous()
    b_nk = (B if transpose_b else B.t()).contiguous()  # ensure [N,K]
    out = flydsl_hgemm(a2, b_nk, bias=bias,
                       b_preshuffle=False, auto_shuffle_b=False)  # no preshuffle = simplest correct path
    return out.reshape(*A.shape[:-1], b_nk.shape[0])


def _aiter_gemm(A, B, bias, transpose_b):
    """Probe the installed aiter for a GEMM entrypoint. aiter's API name varies across versions, so scan
    module + submodules for callables whose name contains 'gemm'/'linear' and try the plausible signatures."""
    import aiter
    Bm = B if transpose_b else B.t()  # aiter linear-style usually wants weight [N,K]; matmul wants [K,N]
    cands = []
    mods = [aiter]
    for sub in ("ops", "ops.triton", "tuned_gemm"):
        m = aiter
        try:
            for part in sub.split("."):
                m = getattr(m, part)
            mods.append(m)
        except Exception:
            pass
    seen = set()
    for m in mods:
        for nm in dir(m):
            low = nm.lower()
            if ("gemm" in low or low in ("linear", "mm")) and not nm.startswith("_"):
                f = getattr(m, nm, None)
                if callable(f) and id(f) not in seen:
                    seen.add(id(f)); cands.append((f"{getattr(m,'__name__',m)}.{nm}", f))
    last = None
    for name, f in cands:
        for argset in ((A, Bm), (A, B), (A, B.t())):
            try:
                out = f(*argset)
                if out is not None and hasattr(out, "shape"):
                    return out if bias is None else (out + bias)
            except Exception as e:
                last = f"{name}: {e!r}"
    raise RuntimeError(f"no working aiter gemm entrypoint (tried {len(cands)}; last={last})")




# The Triton kernel is built ONCE and cached at module scope. Defining it inside the call (the old bug)
# made every invocation recompile + re-autotune -> ~600x slowdown in the timed loop.
_TRITON_MM = None


def _get_triton_mm():
    global _TRITON_MM
    if _TRITON_MM is not None:
        return _TRITON_MM
    import triton
    import triton.language as tl

    @triton.autotune(configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
    ], key=["M", "N", "K"])
    @triton.jit
    def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
            sam, sak, sbk, sbn, scm, scn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
        pid = tl.program_id(0)
        gm = tl.cdiv(M, BLOCK_M); gn = tl.cdiv(N, BLOCK_N)
        wig = GROUP_M * gn
        gid = pid // wig
        first = gid * GROUP_M
        gsize = tl.minimum(gm - first, GROUP_M)
        pm = first + ((pid % wig) % gsize)   # FIXED grouped pid_m (was pid % gsize -> wrong tiles)
        pn = (pid % wig) // gsize
        rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        ap = a_ptr + (rm[:, None] * sam + rk[None, :] * sak)
        bp = b_ptr + (rk[:, None] * sbk + rn[None, :] * sbn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a_t = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_K), other=0.0)
            b_t = tl.load(bp, mask=(rk[:, None] < K - k * BLOCK_K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a_t, b_t)
            ap += BLOCK_K * sak; bp += BLOCK_K * sbk
        cp = c_ptr + (rm[:, None] * scm + rn[None, :] * scn)
        tl.store(cp, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))

    _TRITON_MM = (triton, _mm)
    return _TRITON_MM


def _triton_matmul(torch, A, B, bias, transpose_b, autotune):
    """Triton matmul using the module-cached autotuned kernel (compiled once). Raises if triton is
    unavailable so the bake-off records it as skipped rather than crashing."""
    triton, _mm = _get_triton_mm()
    Bm = (B.t() if transpose_b else B).contiguous()  # [K, N], contiguous for clean strides
    M, K = A.shape[-2], A.shape[-1]
    N = Bm.shape[-1]
    a = A.reshape(-1, K)
    c = torch.empty((a.shape[0], N), device=A.device, dtype=A.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _mm[grid](a, Bm, c, a.shape[0], N, K,
              a.stride(0), a.stride(1), Bm.stride(0), Bm.stride(1), c.stride(0), c.stride(1))
    out = c.reshape(*A.shape[:-1], N)
    if bias is not None:
        out = out + bias
    return out


# --------------------------------------------------------- non-GEMM heads (captured-oracle replay path)
def bench_attn(args, meta):
    """Back-compat alias. The former body returned {"correct": True, "ms": None} for every attention
    task without calling anything — a green result with no measurement under it, which is how an
    unvalidated head reached the integrator. Timing now goes through the generic captured-oracle replay
    (see bench_captured_replay), which is not attention-specific."""
    return bench_captured_replay(args, meta)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="op task dir (with meta.json)")
    ap.add_argument("--backends", default="", help="comma list; default = all known")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--triton-autotune", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    # INV-1: refuse to publish a speedup whose denominator nobody verified. Off only for deliberate
    # exploratory runs, and even then the provenance travels with the number.
    ap.add_argument("--denominator-strict", dest="denominator_strict", action="store_true", default=True)
    ap.add_argument("--no-denominator-strict", dest="denominator_strict", action="store_false")
    a = ap.parse_args()

    meta_path = os.path.join(a.task, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    op_kind = str(meta.get("op_kind", "gemm")).lower()

    # Time baseline+candidate under the deployment's graph context (from the parsed regime): the live
    # server replays decode under a CUDA/HIP graph, so an EAGER baseline is a strawman. Honor the regime
    # when present (default graphed; eager only under enforce-eager/disable-cuda-graph); stay eager when
    # meta has no regime so regime-less tasks are unchanged. time_op(graph=True) falls back to eager if
    # capture is unavailable on this image, so this never hard-fails.
    global _GRAPH_MODE
    if _hlib is not None and meta.get("regime"):
        _GRAPH_MODE = _hlib.deployment_graph_mode(meta.get("regime"))

    try:
        # GEMM has a real cross-backend bake-off; every other op kind is timed by replaying its captured
        # oracle at its own seam. No op kind falls through to an unmeasured pass.
        results = bench_gemm(a, meta) if op_kind == "gemm" else bench_captured_replay(a, meta)
    except Exception as e:
        results = [{"backend": "ERROR", "available": False, "correct": False, "ms": None,
                    "note": f"{e!r}", "trace": traceback.format_exc()[-800:]}]

    correct = [r for r in results if r.get("correct") and r.get("ms")]
    correct.sort(key=lambda r: r["ms"])
    # Prefer an explicitly-named baseline row (the replay path emits one) over the library default.
    baseline = next((r for r in results if r["backend"] == "baseline" and r.get("ms")), None) or \
        next((r for r in results if r["backend"] in ("hipblaslt", "current", "aiter_blockscale") and r.get("ms")), None)
    winner = correct[0] if correct else None

    # ---- Harness self-fault signal (for op_benchmarker self-repair + orchestrator dominant-head guard).
    # If NOTHING produced a correct timed number AND every failure is an EXCEPTION (candidate/reference
    # raised, or the top-level synth ERROR), the harness itself is broken (bad input construction / call),
    # NOT a legitimately-no-faster backend. Surface it explicitly so a dominant head is never silently
    # written off as "no win". A backend that merely ran-but-slow / ran-but-incorrect does NOT trip this.
    ran_any = any(r.get("ms") for r in results)
    raised = [r for r in results if r.get("raised") or r.get("backend") == "ERROR"
              or "raised" in str(r.get("note", "")) or "failed:" in str(r.get("note", ""))]
    harness_suspect = bool(results) and (not ran_any) and len(raised) > 0
    harness_error = ""
    if harness_suspect:
        r0 = raised[0]
        harness_error = str(r0.get("note") or r0.get("trace") or "unknown harness error")[:400]
    speedup = (baseline["ms"] / winner["ms"]) if (winner and baseline and winner["ms"]) else (
        1.0 if winner else 0.0)

    # ---- INV-1: is that ratio publishable?
    # The denominator's provenance is whatever the bench function attached to the baseline row. A row we
    # actually timed as a deployed library backend (hipblaslt/rocblas/...) is self-evidently the deployed
    # path; a row sourced from meta is only as good as meta's validation.
    denom = resolve_denominator(meta)
    prov = (baseline or {}).get("denominator_provenance")
    if not prov:
        prov = "measured_backend_default" if baseline else denom["provenance"]
    severity = DENOM_SEVERITY.get(prov, "invalid")
    denom_why = denom["why"] if prov == denom["provenance"] else \
        f"baseline row '{(baseline or {}).get('backend')}' was timed on this box as the deployed default"
    withhold = (severity == "invalid") or (severity == "unverified" and a.denominator_strict)
    speedup_reason = ""
    if withhold:
        speedup_reason = (f"denominator provenance is '{prov}' ({severity}): {denom_why}. "
                          f"Reporting a ratio against it would be a number without a comparison.")
        note_degradation("main", "isolated_speedup WITHHELD", speedup_reason)
    reported_speedup = None if withhold else round(speedup, 4)
    wb = winner["backend"] if winner else None
    # Only triton/hip are source-editable (-> Tier-C kernel-squad rewrite). ck is a library backend.
    editable = bool(wb in ("triton", "hip"))
    art = (winner.get("artifact") if winner else "") or ""

    # The DEPLOYABLE recipe per winner: what env/flag the server must set so the win survives cuda-graph.
    apply_env, apply_flags, kind = "", "", "none"
    if wb in ("tunableop",):
        apply_env = f"PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME={art}"; kind = "env"
    elif wb == "rocblas":
        apply_env = "TORCH_BLAS_PREFER_HIPBLASLT=0"; kind = "env"
    elif wb == "ck":
        apply_env = ""; kind = "flag"   # ck deploy path is build/flag-dependent; integrate must verify
    elif editable:
        kind = "patch_candidate"
    elif wb == "hipblaslt":
        kind = "none"  # default already; no change to deploy

    # Amdahl ceiling: the MAX e2e delta this isolated speedup can produce given the kernel's GPU-time
    # share (from the Architect/meta). The e2e Integrator uses this to refuse crediting an e2e delta
    # that exceeds the ceiling (box drift, not the kernel). Annotated here so the number travels with
    # the bake-off result. pct_gpu_time absent -> ceiling omitted (None).
    pct_gpu = meta.get("pct_gpu_time", meta.get("pct_gpu", None))
    amdahl_ceiling_pct = None
    # A ceiling derived from a withheld speedup would launder the unpublishable number back in.
    if _hlib is not None and pct_gpu is not None and winner and not withhold:
        try:
            amdahl_ceiling_pct = round(_hlib.amdahl_ceiling(float(pct_gpu), float(speedup)), 3)
        except Exception:
            amdahl_ceiling_pct = None

    summary = {
        "op_kind": op_kind,
        "task": a.task,
        "harness_suspect": harness_suspect,
        "harness_error": harness_error,
        "results": results,
        "winner_backend": wb,
        "winner_ms": winner["ms"] if winner else None,
        "baseline_backend": baseline["backend"] if baseline else None,
        "baseline_ms": baseline["ms"] if baseline else None,
        "isolated_speedup": reported_speedup,
        "isolated_speedup_unpublishable": round(speedup, 4) if withhold else None,
        "speedup_withheld": bool(withhold),
        "speedup_withheld_reason": speedup_reason,
        "denominator": {"spec": denom["spec"], "provenance": prov, "severity": severity,
                        "why": denom_why, "strict": bool(a.denominator_strict)},
        "delegated_config_track": _delegation_status(meta),
        "degradations": DEGRADATIONS,
        "reason_code": next((r.get("reason_code") for r in results if r.get("reason_code")),
                            "invalid_denominator" if withhold else ""),
        "pct_gpu_time": pct_gpu,
        "amdahl_ceiling_e2e_pct": amdahl_ceiling_pct,
        "winner_editable": editable,
        "winner_kind": kind,
        "tuning_artifact": art,
        "apply_env": apply_env,
        "apply_flags": apply_flags,
        "deployable_note": ("env loaded at server startup is captured into the cuda-graph (deployable)"
                            if kind == "env" else
                            "hipblaslt default already in use; nothing to deploy" if wb == "hipblaslt" else
                            "verify deployability at the e2e gate"),
    }
    out = json.dumps(summary, indent=2, default=str)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(out)
    print(out)
    print(f"OPBENCH winner={summary['winner_backend']} "
          + (f"speedup=WITHHELD ({prov})" if withhold else f"speedup={summary['isolated_speedup']}x")
          + f" denominator={prov} "
          f"editable={summary['winner_editable']} kind={summary['winner_kind']} "
          f"harness_suspect={summary['harness_suspect']}"
          + (f" harness_error={summary['harness_error']!r}" if summary['harness_suspect'] else "")
          + (f" degradations={len(DEGRADATIONS)}" if DEGRADATIONS else ""))


if __name__ == "__main__":
    main()
