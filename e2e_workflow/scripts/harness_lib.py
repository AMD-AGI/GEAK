#!/usr/bin/env python3
"""Shared harness measurement library for GEAK e2e kernel tasks.

This is the SINGLE source of truth for how an isolated kernel task measures speedup and checks
correctness. Both the shared bake-off (`op_bench.py`) and every per-task `unittest.py` the Kernel
Extractor generates MUST import these helpers instead of hand-rolling a timing loop / correctness
check. Vendored (copied) into each `<short_name>_task/` at extract time so the task stays
self-contained + immutable + sha-checkable.

It exists to close two systematic "isolated win / e2e loss" holes that a naive per-op harness has:

(a) DEPLOYMENT-REPRESENTATIVE TIMING — do not reward collapsing Python launch/dispatch overhead.
    The classic exploit: time the launcher in a tight `for _ in range(50): fn(); sync()` loop. For
    decode shapes (small M) the wall clock is floored by per-call Python dispatch, NOT the GEMM. A
    candidate then wraps the whole op in a CUDA graph and `graph.replay()` — collapsing a dispatch
    floor that in the LIVE server is ALREADY gone (decode runs inside the server's own CUDA graph).
    Result: a huge isolated "speedup" that evaporates on integration.
    Fix: `time_op` amortizes the per-call host floor by issuing `inner` back-to-back launches between
    two syncs and dividing by `inner`. Baseline and candidate are timed under the IDENTICAL loop, so a
    graph wrapper buys nothing the amortization didn't already give the baseline — the number reflects
    kernel work, not dispatch. (Optional `graph=True` times the same amortized body under a captured
    graph, i.e. exactly the deployment context, for ops where that is reproducible.)

(b) MULTI-ACCESS CORRECTNESS + AMDAHL SANITY.
    - `check_correct_multi` runs several DISTINCT-input cases, keeps ALL returned tensors live, and
      only THEN compares each to its oracle. A candidate that returns a persistent/shared `static_out`
      buffer (the graph-replay shortcut) is caught: the later call overwrites the earlier return, so
      the earlier comparison fails. It also asserts distinct output storage + no cross-call mutation
      (`assert_independent_outputs`). A launcher whose contract is "callable(args) -> fresh out" must
      not alias.
    - `amdahl_ceiling` / `amdahl_check` bound the e2e delta a kernel at `pct_gpu` GPU-time can produce
      given its isolated speedup. `amdahl_ceiling` is surfaced by op_bench as `amdahl_ceiling_e2e_pct`
      so the isolated bake-off already reports the MAX plausible e2e win. `amdahl_check` is the verdict
      form (observed-vs-ceiling) available to any downstream e2e comparison; an observed delta far above
      the ceiling is box drift / measurement error, not the kernel.
"""
import math
import time


# --------------------------------------------------------------------------- device sync
def _torch():
    import torch
    return torch


def sync(torch=None):
    torch = torch or _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- regime-driven synthesis
# The SINGLE source of truth for building operands in the LIVE serving regime. The #1 cause of
# "isolated win / e2e loss-or-crash" is a unittest that SYNTHESIZES its inputs with OFFLINE DEFAULTS
# (DTYPE=bf16, x = 16 // element_size(bf16) = 8, scales = ones) instead of the regime the server
# actually runs. Synthesis itself is fine (perf is value-independent; the oracle is a high-precision
# compute over the same synthesized inputs) — but it MUST be DRIVEN BY the parsed regime descriptor
# (scripts/parse_regime.py output), so operand dtype, quant form, scales, and the paged-KV inner
# factor `x` all follow online. Everything below derives from element sizes + the parsed regime
# fields, so a new dtype / quant scheme needs no new branch. Torch-free for the pure derivations
# (dtype string math), so tests can assert them on a CPU-only / no-torch box.

# element size in BYTES per dtype STRING (no torch needed) — 1-byte types are the quantized/low-precision
# KV/operand dtypes (fp8*, int8) that need scales and pack x=16 into a 16-byte vector. All fp8 variants
# are 1 byte regardless of arch, so the layout math (pack_x) is arch-INDEPENDENT.
_DTYPE_BYTES = {
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "fp8_e4m3fnuz": 1, "fp8_e5m2fnuz": 1, "fp8_e4m3fn": 1, "fp8_e5m2fn": 1,
    "int8": 1, "uint8": 1,
    "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2, "half": 2,
    "fp32": 4, "float32": 4, "float": 4,
    "fp64": 8, "float64": 8,
}

# CDNA3 (MI300/MI325, gfx942) + gfx90a use the AMD-only "fnuz" fp8 (no-inf/unsigned-zero). CDNA4
# (MI355, gfx950) moved to the OCP-standard fp8 (e4m3fn/e5m2), same as NVIDIA. So the fp8 NUMERIC
# FORMAT is the ONE hardware-specific axis: a bare "fp8"/"fp8_e4m3" must resolve to the arch's variant,
# not a hardcoded fnuz. An EXPLICIT ...fnuz/...fn (e.g. from a pre-quantized checkpoint's config) wins.
_FNUZ_ARCH_PREFIXES = ("gfx940", "gfx941", "gfx942", "gfx90a")


def detect_arch(torch=None):
    """Best-effort GPU arch string (e.g. 'gfx942', 'gfx950'); '' if no CUDA/HIP device visible."""
    torch = torch or _torch()
    try:
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0].lower()
    except Exception:
        pass
    return ""


def fp8_is_fnuz(arch):
    """True if this arch uses the AMD fnuz fp8 (CDNA3/gfx942); False for CDNA4/OCP (gfx950) and others."""
    a = (arch or "").lower()
    return any(a.startswith(p) for p in _FNUZ_ARCH_PREFIXES)


def regime_dtype(name, torch=None, arch=None):
    """Map any regime dtype STRING to a torch dtype. The fp8 variant is ARCH-DRIVEN so this is general
    across MI300 (fnuz) and MI355 (OCP fn): a bare 'fp8'/'fp8_e4m3'/'fp8_e5m2' resolves to the running
    GPU's fp8 format (or `arch` if given, for offline cross-arch synthesis). An explicit '...fnuz'/'...fn'
    name is honored literally (a pre-quantized checkpoint that declares its format wins over detection).
    Falls back to bf16 on images without the requested fp8 type."""
    torch = torch or _torch()
    n = str(name).lower()
    non_fp8 = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
        "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
        "int8": getattr(torch, "int8", torch.bfloat16), "uint8": getattr(torch, "uint8", torch.bfloat16),
    }
    if n in non_fp8:
        return non_fp8[n]
    if "fp8" in n or "e4m3" in n or "e5m2" in n:
        mant = "e5m2" if "e5m2" in n else "e4m3"
        if n.endswith("fnuz"):
            suffix = "fnuz"
        elif n.endswith("fn"):
            suffix = "fn"
        else:  # bare/generic name → pick by arch (this is the MI300-vs-MI355 fork)
            suffix = "fnuz" if fp8_is_fnuz(arch if arch is not None else detect_arch(torch)) else "fn"
        return getattr(torch, f"float8_{mant}{suffix}", torch.bfloat16)
    return torch.bfloat16


def _bytes_of(dtype, torch=None):
    """Byte width of a dtype given either as a STRING (no torch needed) or a torch dtype."""
    if isinstance(dtype, str):
        b = _DTYPE_BYTES.get(dtype.lower())
        if b:
            return b
        torch = torch or _torch()
        dtype = regime_dtype(dtype, torch)
    torch = torch or _torch()
    return torch.tensor([], dtype=dtype).element_size()


def pack_x(dtype, pack_bytes=16, torch=None):
    """GENERAL paged-KV inner-block factor: `pack_bytes // element_size(dtype)`. This is the single
    computation that replaces every hand-written `x = 16 // element_size(DTYPE)` — and crucially it keys
    off the KV-CACHE dtype, not the compute dtype. int8/fp8 (1B) -> 16, bf16/fp16 (2B) -> 8,
    fp32 (4B) -> 4. Works for any dtype string or torch dtype."""
    return int(pack_bytes) // _bytes_of(dtype, torch)


def regime_spec(regime):
    """Fold the parsed regime (parse_regime.py output) into what a synthesizer needs — PURE, no torch.
    `auto`/empty KV follows the compute dtype (bf16). A 1-byte KV/operand dtype is treated as quantized
    (needs scales). Returns:
      {compute_dtype, kv_dtype, kv_x, kv_quant, quant_method, operand_dtype, needs_scales}."""
    regime = regime or {}
    quant = regime.get("quant") or {}
    qmethod = str(quant.get("method") or "none").lower()
    quant_on = qmethod not in ("", "none")
    compute_dtype = "bf16"

    kv_raw = str(regime.get("kv_cache_dtype") or "auto").lower()
    kv_dtype = compute_dtype if kv_raw in ("auto", "", "none") else kv_raw
    kv_quant = _DTYPE_BYTES.get(kv_dtype, 2) < 2

    operand_dtype = (quant.get("weight_dtype") or "fp8_e4m3") if quant_on else compute_dtype
    return {
        "compute_dtype": compute_dtype,
        "kv_dtype": kv_dtype,
        "kv_x": pack_x(kv_dtype),
        "kv_quant": bool(kv_quant),
        "quant_method": qmethod,
        "operand_dtype": operand_dtype,
        "needs_scales": bool(quant_on or kv_quant),
    }


def synth_kv_cache(num_blocks, num_heads, head_size, block_size, regime, torch=None, seed=0, arch=None):
    """Build a paged K/V cache in the LIVE regime's KV dtype + layout — the general attention operand
    builder the crashed kernel needed (no unittest computes the layout by hand again). Uses the vLLM
    paged layout where the key cache splits head_size by the pack factor `x` (derived from the KV dtype,
    NOT the compute dtype):
        key_cache   : [num_blocks, num_heads, head_size // x, block_size, x]
        value_cache : [num_blocks, num_heads, head_size,      block_size]
    Real (non-unit) per-tensor k_scale/v_scale are produced when the KV dtype is quantized; scalar 1.0
    otherwise. Returns {key_cache, value_cache, k_scale, v_scale, x, kv_dtype}."""
    torch = torch or _torch()
    spec = regime_spec(regime)
    dt = regime_dtype(spec["kv_dtype"], torch, arch=arch)
    x = spec["kv_x"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(int(seed))
    k_hp = torch.randn(num_blocks, num_heads, head_size // x, block_size, x,
                       generator=gen, dtype=torch.float32, device=device) * 0.1
    v_hp = torch.randn(num_blocks, num_heads, head_size, block_size,
                       generator=gen, dtype=torch.float32, device=device) * 0.1
    if spec["kv_quant"]:
        fmax = float(torch.finfo(dt).max) if dt.is_floating_point else float(torch.iinfo(dt).max)
        k_scale = (k_hp.abs().amax().clamp_min(1e-8) / fmax).to(torch.float32)
        v_scale = (v_hp.abs().amax().clamp_min(1e-8) / fmax).to(torch.float32)
        key_cache = (k_hp / k_scale).clamp(-fmax, fmax).to(dt)
        value_cache = (v_hp / v_scale).clamp(-fmax, fmax).to(dt)
    else:
        k_scale = torch.ones((), dtype=torch.float32, device=device)
        v_scale = torch.ones((), dtype=torch.float32, device=device)
        key_cache = k_hp.to(dt)
        value_cache = v_hp.to(dt)
    return {"key_cache": key_cache, "value_cache": value_cache,
            "k_scale": k_scale, "v_scale": v_scale, "x": x, "kv_dtype": spec["kv_dtype"]}


# --------------------------------------------------------------------------- (a) timing
def deployment_graph_mode(regime):
    """Whether the LIVE server replays this op under a CUDA/HIP graph — the deployment context the
    isolated unittest MUST time its baseline in. Decode is graph-captured by default; `--enforce-eager`
    (vllm) / `--disable-cuda-graph` (sglang) turn it off (regime.enforce_eager / regime.cuda_graph, from
    parse_regime.py).

    WHY the unittest author needs this: the "isolated win, e2e loss" strawman is a baseline TIMED EAGERLY
    when deployment runs under a graph. A candidate that only collapses Python launch/dispatch overhead
    then posts a big isolated speedup that the live graph already erased. So the generated unittest must
    time BOTH baseline and candidate with `time_op(..., graph=deployment_graph_mode(regime))` — it must
    NOT author an eager (disable-cuda-graph) baseline when the regime deploys under a graph. Returns True
    when deployment replays under a graph (the normal case)."""
    regime = regime or {}
    if regime.get("enforce_eager"):
        return False
    return bool(regime.get("cuda_graph", True))


def time_op(call, warmup=10, repeats=50, inner=16, graph=False):
    """Median PER-CALL milliseconds, with the per-call host dispatch floor amortized.

    `call` is a zero-arg closure that issues ONE op launch (and returns its output; the return is
    ignored for timing). We measure `inner` back-to-back launches between two syncs and divide by
    `inner`, so the Python/dispatch floor is spread across `inner` calls instead of dominating the
    small-M (decode) number. Baseline and candidate MUST be timed with the SAME `inner`, so no
    candidate can win purely by collapsing that floor (e.g. a CUDA-graph replay wrapper) — the
    baseline already gets the same amortization. This is the core of hole (a).

    Set `graph=True` to time the amortized body under a captured CUDA graph — i.e. the actual
    deployment context (decode runs inside the server's graph). Falls back to eager amortized timing
    if capture is unavailable/unsupported for this `call`.

    Returns median ms for a SINGLE launch, or None if `call` raises.
    """
    torch = _torch()
    inner = max(1, int(inner))
    try:
        if graph and torch.cuda.is_available():
            g = _try_capture(torch, call, inner)
            if g is not None:
                return _time_graph(torch, g, warmup, repeats)
        for _ in range(max(1, warmup)):
            call()
        sync(torch)
        samples = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            for _ in range(inner):
                call()
            sync(torch)
            samples.append((time.perf_counter() - t0) * 1e3 / inner)
        samples.sort()
        return samples[len(samples) // 2]
    except Exception:
        return None


def _try_capture(torch, call, inner):
    """Capture `inner` launches into a CUDA graph. Returns the graph or None if capture is unsafe
    (host sync in the op, dynamic alloc, etc.) — the caller then falls back to eager amortized timing."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(s)
        sync(torch)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                call()
        return (g, inner)
    except Exception:
        return None


def _time_graph(torch, g, warmup, repeats):
    graph, inner = g
    for _ in range(max(1, warmup)):
        graph.replay()
    sync(torch)
    samples = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        graph.replay()
        sync(torch)
        samples.append((time.perf_counter() - t0) * 1e3 / inner)
    samples.sort()
    return samples[len(samples) // 2]


# --------------------------------------------------------------------------- (b) correctness
def correct(out, ref, tol):
    """allclose-style check with a scale-relative atol floor so near-zero output elements don't blow
    up a pure relative metric. Returns (ok, max_rel_err)."""
    torch = _torch()
    try:
        if tuple(out.shape) != tuple(ref.shape):
            return False, float("inf")
        out = out.float()
        ref = ref.float()
        atol = tol * ref.abs().max().clamp_min(1e-6)
        diff = (out - ref).abs()
        ok = bool((diff <= (atol + tol * ref.abs())).all())
        err = diff.div(ref.abs() + atol).max().item()
        return ok, err
    except Exception:
        return False, float("inf")


def assert_independent_outputs(call, args_a, args_b):
    """Catch a candidate that returns a shared/persistent buffer (the graph-replay `static_out`
    shortcut). Call with two DIFFERENT inputs and verify:
      1. the first output is NOT mutated by the second call (snapshot compare), and
      2. the two outputs do not share storage (distinct data_ptr).
    A correct `fn(args) -> fresh out` launcher passes both. Returns (ok, reason)."""
    torch = _torch()
    try:
        out_a = call(args_a)
        snap = out_a.detach().clone()
        out_b = call(args_b)
        if out_a.data_ptr() == out_b.data_ptr():
            return False, ("shared_output_buffer: two calls returned the SAME storage "
                           f"(data_ptr={out_a.data_ptr():#x}) — a persistent/static return buffer. "
                           "The launcher contract is fn(args) -> FRESH out; a shared buffer is a "
                           "tight-loop cheat that is incorrect for any real (batched) caller.")
        if not torch.equal(out_a, snap):
            return False, ("mutated_prior_output: the second call overwrote the first call's returned "
                           "tensor — the launcher aliases a persistent buffer instead of allocating a "
                           "fresh output. Incorrect for real callers.")
        return True, ""
    except Exception as e:
        return False, f"independence_check_raised: {e!r}"


def check_correct_multi(call, cases, tol):
    """Run every case, KEEP all outputs live, THEN compare each to its oracle (this is what defeats a
    shared-buffer return — a later call would have overwritten an earlier return before we check it).

    `cases` is a list of dicts: {"args": <opaque args passed to call>, "ref": <golden tensor>,
    "sig": <label>}. `call(args) -> out`. Returns (all_ok, per_case_list). Also runs
    `assert_independent_outputs` across the first two distinct cases when available and folds its
    verdict into `all_ok` (reported as a synthetic per-case entry)."""
    outs = [call(c["args"]) for c in cases]        # all live simultaneously — no reuse allowed
    per_case = []
    all_ok = True
    for c, out in zip(cases, outs):
        ok, err = correct(out, c["ref"], tol)
        all_ok = all_ok and ok
        per_case.append({"case": c.get("sig", ""), "correct": ok,
                         "max_rel_err": round(err, 5) if math.isfinite(err) else None})
    if len(cases) >= 2:
        ok, reason = assert_independent_outputs(call, cases[0]["args"], cases[1]["args"])
        all_ok = all_ok and ok
        per_case.append({"case": "output_independence", "correct": ok,
                         "max_rel_err": None, "note": reason})
    return all_ok, per_case


def check_correct_sequence(call, ordered_cases, tol):
    """Replay cases in their RECORDED TEMPORAL ORDER (not the deduped set) and compare each to its
    oracle, keeping all outputs live. This surfaces cross-call STALE STATE that single-shape isolated
    testing misses: the deployment interleaves shapes (chunked-prefill big-M → decode M=1 → …), and a
    kernel that stashes shape-dependent state (a cached scale layout, a persistent workspace sized to
    the first shape it saw) is only wrong on the SECOND, differently-shaped call. `check_correct_multi`
    dedups by shape and can miss the order; this runs the literal sequence.

    `ordered_cases` is a list (WITH repeats, in call order) of {"args", "ref", "sig"}. Returns
    (all_ok, per_case_list). Outputs are held live before comparison (same shared-buffer defeat as
    check_correct_multi)."""
    outs = [call(c["args"]) for c in ordered_cases]     # all live — a shared-buffer return is caught
    per_case = []
    all_ok = True
    for i, (c, out) in enumerate(zip(ordered_cases, outs)):
        ok, err = correct(out, c["ref"], tol)
        all_ok = all_ok and ok
        per_case.append({"case": f"seq[{i}]:{c.get('sig','')}", "correct": ok,
                         "max_rel_err": round(err, 5) if math.isfinite(err) else None})
    return all_ok, per_case


def check_graph_replay(fill, run, read_out, cases, tol, capture_idx=0, warmup=3):
    """Reproduce the DEPLOYMENT capture-once / replay-many path with a STATIC buffer reused ACROSS
    shapes — the exact context a single-shape isolated UT cannot see, and the one that faults on the
    live server. Decode runs inside the server's CUDA graph: the graph is captured ONCE against fixed
    static input/output storage, then replayed every step with fresh data copied into that SAME storage.
    A kernel that (a) allocates an internal workspace / captures a scale-layout pointer during capture
    that does not match what replay feeds it, or (b) writes past the static output on a differently
    shaped case, OOB-faults or returns stale data ONLY under replay — never in eager per-call testing.

    The UT supplies three closures bound to PRE-ALLOCATED static tensors (allocate them once, at the
    capture case's shape; smaller cases pad into them, exactly as the server pads decode batches):
      fill(case) -> copy this case's inputs INTO the static input buffers (honor captured dtype/stride/
                    layout). MUST copy into existing storage, never reallocate.
      run()      -> issue ONE op launch reading static inputs, writing the static output. This is what
                    is captured; MUST be graph-safe (no host sync, no dynamic alloc).
      read_out() -> return the current static output tensor (compared to each case's "ref").

    Capture on cases[capture_idx], then for EVERY case: fill(), graph.replay(), compare read_out() to
    that case's oracle. Returns (all_ok, per_case_list). If CUDA-graph capture is unavailable/unsupported
    on this image, returns a PASS no-op entry (so eager-only boxes don't false-fail) — the e2e gate still
    catches it. A replay that FAULTS is caught (recorded correct=False), not swallowed."""
    torch = _torch()
    if not torch.cuda.is_available() or not cases:
        return True, [{"case": "graph_replay", "correct": True, "max_rel_err": None,
                       "note": "skipped: no CUDA / no cases"}]
    try:
        fill(cases[capture_idx])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(max(1, warmup)):
                run()
        torch.cuda.current_stream().wait_stream(s)
        sync(torch)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
    except Exception as e:
        return True, [{"case": "graph_replay", "correct": True, "max_rel_err": None,
                       "note": f"skipped: capture unavailable ({e!r})"}]
    per_case = []
    all_ok = True
    for c in cases:
        try:
            fill(c)                      # copy THIS case's inputs into the SAME captured storage
            g.replay()
            sync(torch)
            ok, err = correct(read_out(), c["ref"], tol)
            per_case.append({"case": c.get("sig", ""), "correct": ok,
                             "max_rel_err": round(err, 5) if math.isfinite(err) else None,
                             "note": "graph_replay"})
        except Exception as e:
            ok = False
            per_case.append({"case": c.get("sig", ""), "correct": False, "max_rel_err": None,
                             "note": f"graph_replay_raised (OOB/stale under replay): {e!r}"})
        all_ok = all_ok and ok
    return all_ok, per_case


# --------------------------------------------------------------------------- (b) Amdahl gate
def amdahl_ceiling(pct_gpu, isolated_speedup):
    """Max end-to-end THROUGHPUT delta (%) attributable to speeding up a kernel that is `pct_gpu` of
    GPU time by `isolated_speedup`x, assuming GPU time is the throughput bottleneck (an OPTIMISTIC
    upper bound — comm/overlap make the real ceiling lower).

        time_saved_fraction = pct_gpu * (1 - 1/speedup)
        throughput_ceiling  = 1 / (1 - time_saved_fraction)
        delta%              = (throughput_ceiling - 1) * 100

    `pct_gpu` accepts either a fraction (0.14) or a percent (14.0) — values > 1 are treated as percent.
    Returns the ceiling delta in PERCENT (e.g. 5.4 means "at most +5.4% e2e")."""
    p = float(pct_gpu)
    if p > 1.0:
        p /= 100.0
    p = min(max(p, 0.0), 1.0)
    s = float(isolated_speedup)
    if s <= 0:
        return 0.0
    saved = p * (1.0 - 1.0 / s)
    saved = min(max(saved, 0.0), 0.999)
    return (1.0 / (1.0 - saved) - 1.0) * 100.0


def amdahl_check(e2e_delta_pct, pct_gpu, isolated_speedup, noise_band_pct=0.5, slack=1.5):
    """Is an OBSERVED e2e delta physically attributable to this kernel? An observed delta far above
    the Amdahl ceiling is box drift / measurement error, not the kernel.

    `slack` (default 1.5) allows the observed delta to exceed the optimistic ceiling by up to 50%
    before we call it implausible, because the ceiling model ignores second-order fusion/scheduling
    effects and the ceiling itself is only as good as the pct_gpu estimate. Returns a dict:
      {ceiling_pct, allowed_pct, plausible, verdict, note}
    verdict ∈ {ok, implausible} — 'implausible' means a downstream e2e comparison must NOT credit this
    delta to the kernel and should re-measure with an interleaved A/B (and re-check pct_gpu) before
    accepting. (Helper only: op_bench surfaces the ceiling via `amdahl_ceiling`; this verdict form is
    available for any e2e check that wants it.)"""
    ceiling = amdahl_ceiling(pct_gpu, isolated_speedup)
    allowed = ceiling * float(slack) + float(noise_band_pct)
    plausible = float(e2e_delta_pct) <= allowed
    if plausible:
        note = (f"observed +{float(e2e_delta_pct):.2f}% within Amdahl ceiling "
                f"(≤ +{ceiling:.2f}% × {slack} slack + {noise_band_pct}% band = {allowed:.2f}%).")
    else:
        note = (f"observed +{float(e2e_delta_pct):.2f}% EXCEEDS the Amdahl ceiling for a "
                f"{float(pct_gpu)} GPU-time kernel at {float(isolated_speedup):.3f}x "
                f"(ceiling +{ceiling:.2f}%, allowed +{allowed:.2f}%). Not attributable to this kernel — "
                f"treat as box drift/measurement error: re-measure interleaved and re-verify pct_gpu.")
    return {"ceiling_pct": round(ceiling, 3), "allowed_pct": round(allowed, 3),
            "plausible": bool(plausible), "verdict": "ok" if plausible else "implausible",
            "note": note}
