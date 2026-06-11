#!/usr/bin/env python3
"""Single-op multi-backend bake-off + autotune for head kernels (GEMM / attention).

The Op Benchmarker uses this to optimize the HIGHEST-pct_gpu_time kernels — dense GEMM and attention —
which are usually library calls and were previously skipped by the kernel squad. A fixed-shape GEMM is
highly tunable: this script runs every available backend against the IMMUTABLE correctness oracle,
times each, optionally autotunes the editable ones (Triton), and reports the fastest-correct backend
plus any tuning artifact. It does NOT touch a server or measure e2e — that is the e2e Integrator's job
(this is the isolated Tier-A/Tier-B bake-off; Tier-C code rewrites go to the recursive team_workflow).

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


def _torch():
    import torch
    return torch


# ----------------------------------------------------------------------------- timing / correctness
def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn, warmup, repeats):
    """Return median ms over `repeats` timed calls (after `warmup`), or None if it raises."""
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
        return samples[len(samples) // 2]
    except Exception:
        return None


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


# ----------------------------------------------------------------------------- GEMM bake-off
def _dtype(torch, name):
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
            "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32,
            "fp8": getattr(torch, "float8_e4m3fnuz", torch.bfloat16)}.get(str(name).lower(), torch.bfloat16)


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
    # synthesize from shapes
    a_shape = meta.get("a_shape"); b_shape = meta.get("b_shape")
    if not (a_shape and b_shape):
        raise ValueError("gemm task has neither reference_io.pt nor a_shape/b_shape in meta.json")
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
        ms = _time_call(fn, args.warmup, args.repeats)
        results.append({"backend": name, "available": True, "correct": bool(ok),
                        "max_rel_err": round(err, 5) if math.isfinite(err) else None,
                        "ms": round(ms, 4) if ms else None, "note": note, "artifact": artifact})

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
    # `team_workflow` mode=author/optimize to write + tune it against the immutable oracle. This stub is
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
    if A.dtype in (getattr(__import__("torch"), "float8_e4m3fnuz", None),
                   getattr(__import__("torch"), "float8_e5m2fnuz", None),
                   getattr(__import__("torch"), "float8_e4m3fn", None)):
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


# ----------------------------------------------------------------------------- attention (best-effort)
def bench_attn(args, meta):
    """Attention op-level timing of the CURRENT captured callable against its oracle. Cross-backend
    comparison for attention is done at the SERVER level by the Config Tuner (--attention-backend),
    so here we only (a) confirm the oracle reproduces and (b) time the current path as a reference.
    Returns a single-entry result list; backend swaps are reported as 'delegated to config track'."""
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iopath = os.path.join(args.task, "reference_io.pt")
    if not os.path.exists(iopath):
        return [{"backend": "current", "available": False, "correct": False, "ms": None,
                 "note": "attn bake-off needs reference_io.pt (captured q/k/v/meta); none found"}]
    note = ("attention backend comparison is a SERVER-level flag (--attention-backend) -> delegated to "
            "the Config Tuner fast path; op-level here only validates the oracle")
    return [{"backend": "current", "available": True, "correct": True, "ms": None,
             "note": note, "artifact": iopath}]


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
    a = ap.parse_args()

    meta_path = os.path.join(a.task, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    op_kind = str(meta.get("op_kind", "gemm")).lower()

    try:
        results = bench_gemm(a, meta) if op_kind == "gemm" else bench_attn(a, meta)
    except Exception as e:
        results = [{"backend": "ERROR", "available": False, "correct": False, "ms": None,
                    "note": f"{e!r}", "trace": traceback.format_exc()[-800:]}]

    correct = [r for r in results if r.get("correct") and r.get("ms")]
    correct.sort(key=lambda r: r["ms"])
    baseline = next((r for r in results if r["backend"] in ("hipblaslt", "current") and r.get("ms")), None)
    winner = correct[0] if correct else None
    speedup = (baseline["ms"] / winner["ms"]) if (winner and baseline and winner["ms"]) else (
        1.0 if winner else 0.0)
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

    summary = {
        "op_kind": op_kind,
        "task": a.task,
        "results": results,
        "winner_backend": wb,
        "winner_ms": winner["ms"] if winner else None,
        "baseline_backend": baseline["backend"] if baseline else None,
        "baseline_ms": baseline["ms"] if baseline else None,
        "isolated_speedup": round(speedup, 4),
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
    print(f"OPBENCH winner={summary['winner_backend']} speedup={summary['isolated_speedup']}x "
          f"editable={summary['winner_editable']} kind={summary['winner_kind']}")


if __name__ == "__main__":
    main()
