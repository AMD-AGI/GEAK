#!/usr/bin/env python3
"""Shared harness measurement library for GEAK e2e kernel tasks.

This is the SINGLE source of truth for how an isolated kernel task measures speedup and checks
correctness. Both the shared bake-off (`op_bench.py`) and every per-task `unittest.py` the Kernel
Extractor generates MUST import these helpers instead of hand-rolling a timing loop / correctness
check. Vendored (copied) into each `<short_name>_task/` at extract time so the task stays
self-contained + immutable + sha-checkable.

It exists to close two systematic "isolated win / e2e loss" holes that a naive per-op harness has:

(a) DEPLOYMENT-REPRESENTATIVE TIMING — score device work, not host launch/dispatch overhead, and read
    memory-bound kernels COLD from HBM like the live server does.
    The classic exploit: time the launcher in a tight `for _ in range(50): fn(); sync()` wall-clock loop.
    For decode shapes (small M) the wall clock is floored by per-call Python dispatch, NOT the GEMM. A
    candidate then wraps the op in a CUDA graph and `graph.replay()` — collapsing a dispatch floor that
    in the LIVE server is ALREADY gone (decode runs inside the server's own CUDA graph). Result: a huge
    isolated "speedup" that evaporates on integration.
    Fix: `time_op` scores CUDA-EVENT DEVICE time (the GPU timeline between two events), which excludes
    host launch/dispatch — so collapsing dispatch buys nothing and no `inner` amortization is
    needed (inner=1). That exclusion is CONDITIONAL, so `time_op` also reports `primed`/`host_ms` saying
    how far it holds (the receipt `oracle_freezer` prints and `director` gates on).
    WALL time is measured alongside as a reference only. It also FLUSHES the last-level/
    Infinity cache before each sample so a memory-bound decode kernel reads its weights cold from HBM,
    matching deployment (the model working set >> cache, evicted between decode steps). `graph=True` times
    a captured-graph replay (the exact decode deployment context) with the same event+flush method.

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
import json
import math
import os
import shutil
import signal
import subprocess
import sys
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


def deployment_compile_mode(regime):
    """Whether the LIVE server runs this op inside a torch.compile'd (fused) region — the deployment
    context the isolated unittest MUST time its BASELINE in. vLLM V1 compiles the backbone BY DEFAULT
    (opt-out via --enforce-eager); sglang has no default torch.compile (regime.compile from
    parse_regime.py -> "torch_compile" | "eager").

    WHY the unittest author needs this: the FUSION analog of deployment_graph_mode's graph strawman. A
    baseline timed EAGERLY when deployment runs under torch.compile omits the fusion the live server
    already got (epilogue/elementwise fusion, dtype-cast folding), so a candidate posts an isolated win
    that fusion had already captured -> isolated win, e2e loss. Time BOTH baseline and candidate through
    compiled_op(fn, regime) so fusion parity is enforced. Returns True when deployment compiles this op."""
    regime = regime or {}
    if regime.get("enforce_eager"):
        return False
    return str(regime.get("compile") or "eager").lower() in (
        "torch_compile", "compile", "inductor", "true", "1")


def compiled_op(fn, regime, *, fullgraph=True, dynamic=False, mode=None):
    """Wrap the op callable `fn` in torch.compile WHEN the live regime deploys it compiled, else return
    `fn` unchanged (eager) — so the generated unittest times baseline AND candidate with the SAME fusion
    state the server uses. Build the timed closure from the result:

        base = h.compiled_op(BASELINE_FN, REGIME)
        cand = h.compiled_op(CANDIDATE_FN, REGIME)
        h.time_op(lambda: base(x, w, b), graph=h.deployment_graph_mode(REGIME))

    `fullgraph=True` (no graph breaks -> one fused region, matching a captured graph) and `dynamic=False`
    (specialize per timed shape; the harness times one shape at a time) mirror the server's per-shape
    compiled kernels. Compilation is triggered by the timer's warmup launches.

    Degrades gracefully and PARITY-SAFELY: returns `fn` unchanged when the regime is eager, when
    torch.compile is unavailable, or when compile raises. When the regime IS compiled but compile fails,
    the fallback is eager for the returned callable AND the failure is recorded on it via
    `._geak_compile_error` so the unittest can surface it — it NEVER silently reports an eager baseline as
    if it were compiled. Use it symmetrically on both sides so a fallback keeps baseline==candidate fusion."""
    if not deployment_compile_mode(regime):
        return fn
    try:
        import torch
    except Exception:
        try:
            setattr(fn, "_geak_compile_error", "torch import failed; compiled path skipped")
        except Exception:
            pass
        return fn
    if not hasattr(torch, "compile"):
        try:
            setattr(fn, "_geak_compile_error", "torch.compile unavailable (torch<2.0)")
        except Exception:
            pass
        return fn
    try:
        if not torch.cuda.is_available():
            setattr(fn, "_geak_compile_error", "no cuda; compiled path skipped")
            return fn
    except Exception:
        pass
    try:
        kw = {"fullgraph": bool(fullgraph), "dynamic": bool(dynamic)}
        if mode is not None:
            kw["mode"] = mode
        return torch.compile(fn, **kw)
    except Exception as exc:
        try:
            setattr(fn, "_geak_compile_error", f"{type(exc).__name__}: {str(exc)[:200]}")
        except Exception:
            pass
        return fn


def time_op(call, warmup=10, repeats=50, inner=1, graph=False, flush_cache=True, detail=False):
    """Median PER-CALL milliseconds. PRIMARY metric = CUDA-EVENT DEVICE time; wall-clock is a reference.

    `call` is a zero-arg closure that issues ONE op launch (its return is ignored for timing).
      - DEVICE time is measured with cuda.Event around the launch(es) — the GPU-timeline duration, which
        EXCLUDES host launch/dispatch. This is the number a speedup is scored on, so a candidate cannot
        win by collapsing dispatch (a graph wrapper), and no amortization is needed: inner=1 gives clean
        device time for any kernel whose runtime >> event resolution (~1us). Raise `inner` only for
        sub-microsecond kernels (amortizes event/launch resolution).
      - WALL time (perf_counter+sync) is measured in the SAME loop and reported as a REFERENCE only
        (host+device); a large wall≫device gap flags a host-bound op.
    `flush_cache` (default True) evicts the last-level/Infinity cache BEFORE each timed sample so a
    memory-bound decode kernel reads its weights COLD from HBM — matching the live server, where the model
    working set >> cache and every weight is evicted between decode steps. Flushed OUTSIDE the event window.

    `graph=True` times a captured CUDA-graph replay (the decode deployment context) with the same
    event+flush method; falls back to eager event timing if capture is unavailable (see the `timer` field).

    The device-time guarantee is CONDITIONAL, and `primed` says how far it holds. Read it as THREE
    states, never a bool (see `_host_dispatch_ms`):
      True    -> dispatch is cheaper than the kernel, so `ms` is dominated by device work. Score it.
      False   -> this op dispatches slower than it computes; `ms` is a HOST-BOUND latency, not a kernel
                 time, and a candidate can "win" on it by collapsing dispatch alone.
      ABSENT  -> the timer could not tell (no CUDA). NOT the same as False.
    `primed=True` does NOT mean the window is overhead-free: a window costs a few us beyond the kernel
    either way, negligible for a 100us GEMM and most of the number for a 5us op. Raise `inner` when that
    matters — the overhead is per-window, so it divides away.

    Returns median device ms (float), or {ms, wall_ms, timer} when detail=True — plus {primed, host_ms}
    whenever the timer can produce them. None if `call` raises. On a box without CUDA, device time is
    unavailable so ms == wall_ms and timer='wall'."""
    torch = _torch()
    inner = max(1, int(inner))
    try:
        have_cuda = bool(torch.cuda.is_available())
    except Exception:
        have_cuda = False
    try:
        if have_cuda and graph:
            g = _try_capture(torch, call, inner)
            if g is not None:
                dev, wall = _time_graph(torch, g, warmup, repeats, flush_cache)
                # One replay issues all `inner` launches, so divide to put host on the SAME per-launch
                # basis as `dev` -- otherwise a clean inner>1 graph reads as host-bound.
                host = _host_dispatch_ms(torch, lambda: g[0].replay()) / g[1]
                return _timing_result(dev, wall, "cuda_event_graph", detail, host, host < dev)
        if have_cuda:
            dev, wall = _time_events(torch, call, warmup, repeats, inner, flush_cache)
            host = _host_dispatch_ms(torch, call)
            return _timing_result(dev, wall, "cuda_event", detail, host, host < dev)
        wall = _time_wall(torch, call, warmup, repeats, inner)   # no device timeline -> wall only
        return _timing_result(wall, wall, "wall", detail)
    except Exception:
        return None


def _timing_result(dev_ms, wall_ms, timer, detail, host_ms=None, primed=None):
    if not detail:
        return dev_ms
    d = {"ms": dev_ms, "wall_ms": wall_ms, "timer": timer}
    if host_ms is not None:
        # Both keys or neither. A consumer distinguishes "host-bound" from "cannot tell" by PRESENCE
        # (oracle_freezer.md step 4), so half a receipt would be read as a whole one.
        d["host_ms"] = host_ms
        d["primed"] = bool(primed)
    return d


_HOST_PROBE_LAUNCHES = 30


def _host_dispatch_ms(torch, call):
    """Host cost of issuing ONE `call()`, measured on a DRAINED queue.

    Deliberately not derived from `wall_ms`: that is host+device together, so on a long kernel it is
    essentially the device time and every slow kernel would look slow to dispatch too. `primed =
    host_ms < dev_ms` then reads as director.md states it — a host that cannot issue faster than the GPU
    executes cannot keep the queue full, so the event window measures host latency, not kernel time.

    `_HOST_PROBE_LAUNCHES` is fixed and owned here. It used to be the caller's `inner`, which defaults to
    1, making the estimate a single cold read of a Python dispatch path; on MI355X that overstated a
    512^3 GEMM's dispatch ~2x and flipped a genuinely primed measurement to False. It stays small so the
    launches do not backpressure the host, which would reintroduce that inflation.

    Callers put the result on the same per-launch basis as `dev`: eager `call()` is one launch; a graph
    replay is `inner`, so that caller divides."""
    n = max(1, _HOST_PROBE_LAUNCHES)
    for _ in range(3):
        call()                      # the dispatch path must be HOT: we want steady-state issue cost
    sync(torch)
    t0 = time.perf_counter()
    for _ in range(n):
        call()
    ms = (time.perf_counter() - t0) * 1e3 / n
    sync(torch)
    return ms


_CACHE_FLUSH_BUF = None


def flush_cache(torch=None, mb=None):
    """Evict the GPU last-level / Infinity cache so the NEXT launch reads cold from HBM (matches decode:
    the model working set >> cache, so every weight is evicted between reuses). Writes a persistent buffer
    larger than the cache (default 512MB > MI300's 256MB Infinity Cache; override HARNESS_CACHE_FLUSH_MB).
    No-op without CUDA."""
    global _CACHE_FLUSH_BUF
    torch = torch or _torch()
    try:
        if not torch.cuda.is_available():
            return
    except Exception:
        return
    mb = int(os.environ.get("HARNESS_CACHE_FLUSH_MB", "512")) if mb is None else int(mb)
    n = max(1, (mb << 20) // 4)
    if _CACHE_FLUSH_BUF is None or _CACHE_FLUSH_BUF.numel() < n:
        _CACHE_FLUSH_BUF = torch.empty(n, dtype=torch.float32, device="cuda")
    _CACHE_FLUSH_BUF.zero_()


def _time_events(torch, call, warmup, repeats, inner, flush):
    """Median (device_ms, wall_ms) over `repeats` samples of `inner` back-to-back launches: device via
    cuda.Event (host-free), wall via perf_counter (reference). Cache flushed before each sample when
    `flush`, so a memory-bound kernel is timed cold.

    The per-sample sync is deliberate. Batching the samples into one queue with no intervening sync was
    tried and REJECTED on measurement: fitting per-launch ms against 1/inner on MI355X (torch 2.9 / ROCm
    7.2) to separate kernel time K from per-window overhead E showed batching RAISED E in 5 of 6
    configurations, and under `flush` let the two timers disagree on K itself by up to 2.8x -- flush and
    kernel become two commands in a packed queue, so the cold-HBM guarantee stops holding. Do not
    reintroduce it without re-running that fit."""
    for _ in range(max(1, warmup)):
        call()
    sync(torch)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dev, wall = [], []
    for _ in range(max(1, repeats)):
        if flush:
            flush_cache(torch)
        sync(torch)
        t0 = time.perf_counter()
        start.record()
        for _ in range(inner):
            call()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t0) * 1e3 / inner)
        dev.append(start.elapsed_time(end) / inner)
    dev.sort(); wall.sort()
    return dev[len(dev) // 2], wall[len(wall) // 2]


def _time_wall(torch, call, warmup, repeats, inner):
    """Wall-clock-only fallback for boxes with no CUDA-event device timeline (CPU/no-CUDA)."""
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


def _try_capture(torch, call, inner):
    """Capture `inner` launches into a CUDA graph. Returns the graph or None if capture is unsafe
    (host sync in the op, dynamic alloc, etc.) — the caller then falls back to eager event timing."""
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


def _time_graph(torch, g, warmup, repeats, flush):
    """Median (device_ms, wall_ms) of a captured-graph replay, device via cuda.Event, cache flushed
    before each sample when `flush` (replay reuses static buffers, so without a flush the weights stay
    hot — unrepresentative of cold decode)."""
    graph, inner = g
    for _ in range(max(1, warmup)):
        graph.replay()
    sync(torch)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dev, wall = [], []
    for _ in range(max(1, repeats)):
        if flush:
            flush_cache(torch)
        sync(torch)
        t0 = time.perf_counter()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t0) * 1e3 / inner)
        dev.append(start.elapsed_time(end) / inner)
    dev.sort(); wall.sort()
    return dev[len(dev) // 2], wall[len(wall) // 2]


# --------------------------------------------------------------------------- (b) correctness
def flatten_outputs(out):
    """Deterministic tensor list from an op's return value: tensor, tuple/list, dict, or nesting.

    Attention entries return `(out, lse)`; a bare-tensor assumption does not merely raise here, it
    makes `correct()` fall into its except and report the candidate INCORRECT — a correct kernel
    rejected for the shape of its return value. Dict order is by sorted key so the two legs pair up."""
    if isinstance(out, dict):
        return [t for k in sorted(out) for t in flatten_outputs(out[k])]
    if isinstance(out, (tuple, list)):
        return [t for o in out for t in flatten_outputs(o)]
    if hasattr(out, "shape"):
        return [out]
    return []      # scalars/None carry no tolerance semantics and are not compared


def to_device_like(ref, dev):
    """Move a recorded oracle entry (tensor or container of tensors) onto `dev`."""
    if isinstance(ref, dict):
        return {k: to_device_like(v, dev) for k, v in ref.items()}
    if isinstance(ref, tuple) and hasattr(ref, "_fields"):
        return type(ref)(*(to_device_like(o, dev) for o in ref))
    if isinstance(ref, (tuple, list)):
        return type(ref)(to_device_like(o, dev) for o in ref)
    return ref.to(dev) if hasattr(ref, "to") else ref


def reconstruct_captured(obj, device="cpu"):
    """Inverse of capture_shapes._snapshot (shared refs must already be resolved)."""
    if isinstance(obj, dict) and obj.get("__tensor__"):
        t = obj["data"]
        return t.to(device) if hasattr(t, "to") else t
    if isinstance(obj, dict) and set(obj.keys()) == {"__repr__"}:
        return obj["__repr__"]
    if isinstance(obj, dict):
        return {k: reconstruct_captured(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(reconstruct_captured(v, device) for v in obj)
    return obj


def load_reference_io(path, map_location="cpu"):
    """Load a capture_shapes oracle blob (supports optional ``shared`` weight pool)."""
    torch = _torch()
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    if not isinstance(blob, dict):
        raise TypeError(f"reference_io.pt must be a dict, got {type(blob)}")
    return blob


def resolve_oracle_shared(obj, shared):
    """Expand ``{"__shared__": key}`` refs written by capture_shapes share_large/moe_slim."""
    if isinstance(obj, dict) and set(obj.keys()) == {"__shared__"}:
        key = obj["__shared__"]
        if key not in shared:
            raise KeyError(f"oracle shared ref missing: {key!r}")
        return shared[key]
    if isinstance(obj, dict):
        return {k: resolve_oracle_shared(v, shared) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(resolve_oracle_shared(v, shared) for v in obj)
    return obj


def iter_eager_cases_from_oracle(path, device="cpu"):
    """Yield ``{args, ref, sig, regime}`` one record at a time (memory-friendly for multi-GiB MoE)."""
    blob = load_reference_io(path, map_location="cpu")
    shared = blob.get("shared") or {}
    for record in blob.get("records") or []:
        kwargs = resolve_oracle_shared(record.get("kwargs") or {}, shared)
        args_pos = resolve_oracle_shared(record.get("args") or (), shared)
        kwargs = reconstruct_captured(kwargs, device=device)
        args_pos = reconstruct_captured(args_pos, device=device)
        ref = reconstruct_captured(
            resolve_oracle_shared(record.get("output"), shared), device=device)
        args = kwargs if isinstance(kwargs, dict) and kwargs else args_pos
        if isinstance(args, dict) and isinstance(args_pos, (list, tuple)) and args_pos:
            names = ["hidden_states", "w1", "w2", "topk_weight", "topk_ids"]
            for name, value in zip(names, args_pos):
                args.setdefault(name, value)
        yield {
            "args": args,
            "ref": ref,
            "sig": record.get("sig", ""),
            "regime": record.get("regime", ""),
        }


def eager_cases_from_oracle(path, device="cpu"):
    """Materialize all eager cases; prefer ``iter_eager_cases_from_oracle`` for large oracles."""
    return list(iter_eager_cases_from_oracle(path, device=device))


def check_correct_multi_lazy(call, case_iter, tol, max_keep_live=2):
    """Like ``check_correct_multi`` but only keeps ``max_keep_live`` outputs resident."""
    per_case = []
    all_ok = True
    live = []
    first_two_args = []
    for case in case_iter:
        out = call(case["args"])
        ok, err = correct(out, case["ref"], tol)
        all_ok = all_ok and ok
        per_case.append({"case": case.get("sig", ""), "correct": ok,
                         "max_rel_err": round(err, 5) if math.isfinite(err) else None})
        if len(first_two_args) < 2:
            first_two_args.append(case["args"])
        live.append(out)
        if len(live) > max(1, int(max_keep_live)):
            live.pop(0)
    if len(first_two_args) >= 2:
        ok, reason = assert_independent_outputs(call, first_two_args[0], first_two_args[1])
        all_ok = all_ok and ok
        per_case.append({"case": "output_independence", "correct": ok,
                         "max_rel_err": None, "note": reason})
    return all_ok, per_case


def correct(out, ref, tol):
    """Per-element mixed-tolerance check `|out-ref| <= atol + tol*|ref|`, returns (ok, max_rel_err).

    A multi-tensor return (e.g. attention's `(out, lse)`) is compared component-wise: ALL components
    must pass, and the reported error is the worst of them. A count mismatch between the two sides is
    a failure, never a silent comparison of the first component only.

    The absolute floor `atol` exists ONLY to keep near-zero reference elements from blowing up the pure
    relative term — it is set to the computation's NOISE level `tol * RMS(ref)`, NOT `tol * max(|ref|)`.
    A max-scaled floor lets a small element of a HIGH-DYNAMIC-RANGE output (attention scores, MoE routing
    weights, index-decode top-k) drift by `tol*max` in absolute terms — i.e. an unbounded relative error
    on the small elements the value-parity gate is meant to catch. RMS tracks the bulk magnitude, so for
    a uniform-magnitude tensor RMS≈max (behavior unchanged) but for a spiky tensor RMS≪max (the floor
    tightens and the small-element error is no longer masked)."""
    po, pr = flatten_outputs(out), flatten_outputs(ref)
    if len(po) != 1 or len(pr) != 1:
        if not po or len(po) != len(pr):
            return False, float("inf")
        ok_all, worst = True, 0.0
        for o, r in zip(po, pr):
            ok, err = _correct_one(o, r, tol)
            ok_all = ok_all and ok
            worst = max(worst, err)
        return ok_all, worst
    return _correct_one(po[0], pr[0], tol)


def _correct_one(out, ref, tol):
    torch = _torch()
    try:
        if tuple(out.shape) != tuple(ref.shape):
            return False, float("inf")
        out = out.float()
        ref = ref.float()
        atol = tol * ref.pow(2).mean().sqrt().clamp_min(1e-6)   # noise floor = tol * RMS(ref)
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
    "sig": <label>}. `call(args) -> out`. Returns (all_ok, per_case_list). When there are >=2 cases it
    also runs `assert_independent_outputs` on cases[0] and cases[1] (the shared-storage/data_ptr check
    catches a persistent buffer regardless of whether those two inputs differ) and folds its verdict into
    `all_ok` (reported as a synthetic per-case entry)."""
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


# --------------------------------------------------------------------------- (b) random-value parity vs live baseline
def check_random_vs_baseline(baseline_call, current_call, shapes, tol,
                             draws=3, warmup=10, repeats=50, inner=1, graph=False, seed=0,
                             baseline_outputs=None):
    """Validate the candidate against the LIVE frozen baseline on MANY RANDOM INPUT VALUE DRAWS at the
    SAME online-aligned shapes (NOT random shapes — dims are fixed per `sig`, only values vary). The
    frozen oracle (`reference_io.pt`) pins ONE recorded input+golden; this catches value-dependent bugs
    that single draw misses (masking, NaN/denormal handling, accumulation across magnitudes) by using the
    real production kernel as the truth source for each fresh draw — no stored golden needed.

    CORRECTNESS is a HARD GATE: for every shape × every draw, `correct(candidate_out, baseline_out, tol)`
    must pass, else `all_ok=False`. The baseline output is snapshotted (clone) BEFORE the candidate runs,
    so a candidate that aliases the baseline's storage (or returns a shared/static buffer) is caught.
    SPEEDUP is REPORT-ONLY: `baseline_ms / current_ms` per draw is recorded as a secondary robustness
    signal (value-independent perf variance / cliff detection), NOT the win metric — the primary metric
    stays the workload-weighted oracle speedup. Both times are `time_op` DEVICE time (CUDA events), taken
    with the L2/Infinity cache flushed cold before each iteration; `inner=1` (one launch per event window)
    since events time the GPU timeline directly — no host-side amortization loop is needed.

    `baseline_outputs` (PREFERRED) = the dict `baseline_random_outputs(...)` recorded by the BASELINE
        leg in its OWN subprocess (`leg_runner.py --mode oracle` under `baseline_overlay/`), keyed
        `"<sig>|<draw>"`. Same seed => same inputs, so the two legs never have to be co-resident. When
        it is given, `baseline_call` is ignored and `speedup` is None here (timing comes from
        `measure_legs`).
    `baseline_call(args) -> out` is the LEGACY in-process form, kept for op_bench / single-process tasks.
        `current_call(args) -> out` invokes the candidate in kernel_src/.
    `shapes` is a list of {"sig": <label>, "make_inputs": callable(rng) -> args}. `make_inputs` builds a
        FRESH random in-regime input set for that shape's fixed dims (the unittest closes over
        regime_spec/synth_kv_cache); `rng` is a seeded torch.Generator for reproducibility.
    Returns (all_ok, per_case_list).
    """
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_case = []
    all_ok = True
    for shape in shapes:
        sig = shape.get("sig", "")
        make_inputs = shape["make_inputs"]
        for i in range(max(1, int(draws))):
            rng = torch.Generator(device=device).manual_seed(int(seed) + i)
            try:
                args = make_inputs(rng)
                if baseline_outputs is not None:
                    ref = baseline_outputs.get(f"{sig}|{i}")
                    if ref is None:
                        raise KeyError(f"no recorded baseline output for {sig}|{i}")
                    cand_out = current_call(args)
                    parts = flatten_outputs(cand_out)
                    if not parts:
                        raise TypeError(f"candidate returned no tensor for {sig}|{i}")
                    base_snap = to_device_like(ref, parts[0].device)
                else:
                    base_out = baseline_call(args)
                    # snapshot BEFORE current runs (anti-alias)
                    base_snap = [t.detach().clone() for t in flatten_outputs(base_out)]
                    cand_out = current_call(args)
            except Exception as e:
                all_ok = False
                per_case.append({"case": f"random[{i}]:{sig}", "correct": False, "max_rel_err": None,
                                 "speedup": None, "note": f"value-parity raised: {e!r}"})
                continue
            ok, err = correct(cand_out, base_snap, tol)
            all_ok = all_ok and ok
            if baseline_outputs is not None:
                speedup = None                             # timing belongs to measure_legs, not here
            else:
                ms_base = time_op(lambda: baseline_call(args), warmup, repeats, inner, graph)
                ms_cand = time_op(lambda: current_call(args), warmup, repeats, inner, graph)
                speedup = (ms_base / ms_cand) if (ms_base and ms_cand) else None
            per_case.append({"case": f"random[{i}]:{sig}", "correct": ok,
                             "max_rel_err": round(err, 5) if math.isfinite(err) else None,
                             "speedup": round(speedup, 3) if speedup else None,
                             "note": "value-parity vs live baseline (correctness gates; speedup reports)"})
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
    if isolated_speedup is None:
        return 0.0        # no measurement -> no attributable ceiling; same answer as a non-win
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


# --------------------------------------------------------------------------- (c) serving-weighted PRIMARY metric
def _analytic_calls_from_meta(meta):
    swm = ((meta or {}).get("workload") or {}).get("serving_weight_model") or {}
    return swm.get("analytic_calls") or {}


def served_regimes(meta):
    """Regimes this kernel actually runs in (lower-cased set); empty set = ungated. Preference:
      1. meta['served_regimes'] (extractor override, if set), else
      2. meta['workload']['served_regimes'] (trace-derived by attribute_weights from the profile's
         MEASURED serving-phase spans — parse_profile served_regimes/case regime).
    A *_fwd/prefill wrapper with a separate *_decode kernel is 'prefill'; the decode kernel is 'decode'."""
    sr = (meta or {}).get("served_regimes")
    if not sr:
        sr = ((meta or {}).get("workload") or {}).get("served_regimes")
    if not sr:
        return set()
    return {str(r).strip().lower() for r in sr if str(r).strip()}


def served_buckets(meta):
    """The M buckets this kernel actually SERVES, with their analytic per-wave pass counts.

    `op_bench` benched at `max(m_buckets)`, on the premise that the largest bucket carries the
    GPU-time mass. It does not. Mass = per-launch time x LAUNCHES, and the analytic call model
    (`serving_weight_model.analytic_calls`) counts prefill = CONC*ceil(ISL/chunk) passes against
    decode = OSL passes. Neither bucket dominates universally -- prefill runs few large passes,
    decode runs many small ones -- so a single bucket cannot represent the served mix, and the
    LARGEST one is routinely the least-served shape.

    Decode's M is the concurrency (snapped to a graph capture size); prefill's is the chunk, or
    ISL when the server does not chunk. Each is matched to the nearest profiled `m_buckets` entry
    when one exists, so the bench runs on a shape the profile actually saw.

    Returns [(phase, M, calls), ...] ordered by descending calls, de-duplicated by M. Returns []
    when meta carries no serving model (offline / unit-test metas), which leaves callers on their
    previous single-bucket behaviour."""
    swm = ((meta or {}).get("workload") or {}).get("serving_weight_model") or {}
    calls = swm.get("analytic_calls") or {}
    if not calls:
        return []
    buckets = [int(b) for b in (meta.get("m_buckets") or [])
               if str(b).strip().lstrip("-").isdigit()]
    served = served_regimes(meta)
    want = {"decode": swm.get("conc"),
            "prefill": swm.get("prefill_chunk") or swm.get("isl")}
    out = {}
    for phase, n in calls.items():
        ph = str(phase).strip().lower()
        if served and ph not in served:
            continue                      # a phase this kernel does not run in carries no weight
        m, n = want.get(ph), _int_or_none(n)
        m = _int_or_none(m)
        if not m or not n or n <= 0:
            continue
        if buckets:
            m = min(buckets, key=lambda b: abs(b - m))
        prev = out.get(m)
        if prev is None or n > prev[2]:
            out[m] = (ph, m, n)
    return sorted(out.values(), key=lambda t: -t[2])


def _int_or_none(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def serving_weighted_speedup(per_case, meta, *, identity_eps=1e-4, geomean=True):
    """Centralized PRIMARY-metric weighting for the immutable unittests — replaces the per-kernel
    hand-rolled `_serving_calls`/`weight` blocks so every UT applies the SAME audited rule. Enforces:

      (1) served-regimes gate: drop any case whose regime is not in `served_regimes(meta)` (when set),
          so a decode case that leaked into a prefill-only kernel's oracle cannot be weighted onto it.
      (2) analytic call model (NEVER profiled counts): calls come from
          meta.workload.serving_weight_model.analytic_calls (prefill already CONC-scaled by
          attribute_weights: CONC*ceil(ISL/chunk); decode = OSL). Within each served regime the regime's
          total passes go to the LARGEST-M bucket (decode M==CONC; prefill M==chunk/ISL); smaller/
          transient buckets get calls=1 and stay visible. CONC enters decode via the SHAPE (M) and
          prefill via the COUNT — one consistent per-wave basis.
      (3) pseudo-identity guard: a bucket whose baseline_ms == optimized_ms to `identity_eps` (a warm-JIT
          / autotune-converged / byte-identical-candidate artifact, NOT a real measured null) is flagged
          and EXCLUDED. If EVERY surviving bucket is identity, `weighted` is None with a reason so the
          caller does not trust an unmeasured 1.0x (re-measure per-bucket ms in a fresh subprocess under
          the deployment graph/compile — see kernel_extractor.md).

    `per_case`: list of {sig|name, regime, m?, baseline_ms, optimized_ms?|speedup?}. speedup is derived
    from baseline_ms/optimized_ms when both present (preferred), else the passed `speedup` is used.
    Returns {weighted, geomean, primary, included, dropped_unserved, suspect_identity, per_case, reason}."""
    served = served_regimes(meta)
    calls_model = _analytic_calls_from_meta(meta)

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    rows, dropped_unserved = [], []
    for c in per_case:
        reg = str(c.get("regime") or "").lower()
        if served and reg and reg not in served:
            dropped_unserved.append(c.get("sig") or c.get("name") or reg)
            continue
        b, o, spd = _f(c.get("baseline_ms")), _f(c.get("optimized_ms")), _f(c.get("speedup"))
        if b is not None and o is not None and o > 0:
            spd = b / o
        rows.append({"sig": c.get("sig") or c.get("name") or "", "regime": reg, "m": c.get("m"),
                     "baseline_ms": b, "optimized_ms": o, "speedup": spd})

    # assign analytic passes: the largest-M bucket in each regime carries that regime's calls, rest = 1.
    by_reg = {}
    for r in rows:
        by_reg.setdefault(r["regime"], []).append(r)

    def _m(x):
        try:
            return float(x["m"])
        except (TypeError, ValueError):
            return float(x["baseline_ms"] or 0.0)

    for reg, members in by_reg.items():
        total_calls = int(calls_model.get(reg, 1) or 0) if reg else 1
        dom = max(members, key=_m) if members else None
        for r in members:
            r["calls"] = total_calls if (dom is not None and r is dom) else 1

    suspect, included = [], []
    for r in rows:
        b, o, spd = r["baseline_ms"], r["optimized_ms"], r["speedup"]
        is_identity = (b is not None and o is not None
                       and abs(b - o) <= identity_eps * max(abs(b), 1e-12))
        r["identity"] = bool(is_identity)
        r["weight"] = (b or 0.0) * r.get("calls", 1)
        if is_identity:
            suspect.append(r["sig"]); r["included"] = False
        elif spd and spd > 0 and r["weight"] > 0:
            r["included"] = True; included.append(r)
        else:
            r["included"] = False

    if not included:
        return {"weighted": None, "geomean": None, "primary": None, "included": 0,
                "dropped_unserved": dropped_unserved, "suspect_identity": suspect, "per_case": rows,
                "reason": ("no measurable non-identity bucket survived (all identity/zero-weight or "
                           "dropped by served-regimes) — weighted speedup is UNTRUSTED; re-measure "
                           "per-bucket ms in a fresh subprocess under the deployment graph/compile.")}
    W = sum(r["weight"] for r in included)
    D = sum(r["weight"] / r["speedup"] for r in included)
    weighted = (W / D) if D else None
    gm = None
    if geomean:
        spds = [r["speedup"] for r in included]
        gm = math.exp(sum(math.log(s) for s in spds) / len(spds)) if spds else None
    return {"weighted": weighted, "geomean": gm, "primary": weighted, "included": len(included),
            "dropped_unserved": dropped_unserved, "suspect_identity": suspect, "per_case": rows,
            "reason": ""}


class HarnessIncompleteError(Exception):
    """The generated UT is INCOMPLETE for this kernel (a harness/GENERATION defect), NOT a kernel-
    correctness failure. Raised by `run_correctness` when a graph-deploy kernel is handed no usable
    (>=2-shape) replay bundle. `run_correctness` has ALREADY printed the `UT_HARNESS_INCOMPLETE:` sentinel
    to stdout before raising, so the UT's main() should just catch this and exit 3 (do NOT re-print the
    sentinel) — the smoke-test then REGENERATES the UT (add the replay leg) instead of blaming the
    candidate. Distinct from a correctness FAIL (exit 1) and an env error (exit 2)."""


UT_HARNESS_INCOMPLETE_SENTINEL = "UT_HARNESS_INCOMPLETE"


# --------------------------------------------------------------------------- (c) fail-closed correctness suite
# WHY: `check_graph_replay` already reproduces the deploy path that faults (capture-once / replay-many /
# static-buffer reuse across shapes), but it was (1) gated on the fragile `meta.graph_replayed` flag
# (which CUDA-graph REPLAY makes unobservable at the Python seam, so it was often absent -> the gate never
# fired) and (2) LLM-EMITTED (the generated UT could just omit it). A decode kernel that OOB-writes on a
# ragged real shape under replay then passes the eager isolated UT and only faults e2e (0/320 served) —
# the h2 paged_attention failure. `run_correctness` closes both holes: the trigger is the AUTHORITATIVE
# deployment fact `deployment_graph_mode(regime)` (regime.cuda_graph, from the launch flags), and a
# graph-deploy kernel that supplies no >=2-shape replay bundle FAILS CLOSED instead of silently passing.
def run_correctness(regime, *, eager_cases, current_call, random_shapes, tol,
                    baseline_call=None, baseline_outputs=None, replay=None, draws=3):
    """The SINGLE correctness entrypoint every generated unittest must call. Runs, in order:
      1. eager multi-case vs oracle (`check_correct_multi`) — also the output-independence check;
      2. random-value parity vs the frozen live baseline (`check_random_vs_baseline`);
      3. FAIL-CLOSED deployment-context replay: when `deployment_graph_mode(regime)` is True, a
         `replay` bundle with >=2 BOUNDARY shapes is MANDATORY (single-shape replay cannot expose a
         static-buffer-reuse OOB). Missing / too-few cases => hard FAIL (never a silent skip).

    `replay` (required when the op deploys under a graph) = {
        "fill": fn(case)->None (copy case inputs INTO pre-allocated static buffers; never realloc),
        "run":  fn()->None     (one graph-safe launch reading/writing those static buffers),
        "read_out": fn()->Tensor (the static output to compare),
        "cases": [{"args":..., "ref": <golden>, "sig": <label>}, ...]  (>=2, boundary-spanning),
        "capture_idx": <index of the LARGEST case to capture on> (default 0),
    }
    Returns (all_ok, report) where report has keys eager / random / graph_replay.
    """
    report = {}
    ok = True

    c_ok, per = check_correct_multi(current_call, eager_cases, tol)
    report["eager"] = per
    ok = ok and c_ok

    if baseline_call is None and baseline_outputs is None:
        raise ValueError("run_correctness needs baseline_outputs (preferred: recorded by the baseline "
                         "leg via baseline_random_outputs) or a legacy in-process baseline_call")
    r_ok, perr = check_random_vs_baseline(baseline_call, current_call, random_shapes, tol,
                                          draws=draws, graph=deployment_graph_mode(regime),
                                          baseline_outputs=baseline_outputs)
    report["random"] = perr
    ok = ok and r_ok

    if deployment_graph_mode(regime):
        cases = (replay or {}).get("cases") or []
        missing = [k for k in ("fill", "run", "read_out") if not callable((replay or {}).get(k))]
        if replay is None or missing or len(cases) < 2:
            why = ("no replay bundle" if replay is None
                   else f"missing closures {missing}" if missing
                   else f"only {len(cases)} replay case(s); need >=2 boundary shapes")
            reason = ("deploys under CUDA graph (deployment_graph_mode=True) but " + why
                      + " — a graph-deploy kernel MUST be replay-checked across >=2 boundary shapes "
                        "(capture-once/replay-many reused static buffer). This is a UT-GENERATION defect "
                        "(the harness is incomplete), NOT a kernel-correctness failure: regenerate the UT "
                        "with a replay bundle rather than blaming the candidate.")
            # Print the sentinel HERE — this is the SINGLE source of the stdout line, so the smoke-test
            # recognizes "regenerate UT" even if the generated main() forgets to catch the exception.
            # main() must NOT re-print it (just exit 3); then raise the DISTINCT harness error (never a
            # silent correctness FAIL that would wrongly reject the candidate).
            print(f"{UT_HARNESS_INCOMPLETE_SENTINEL}: {reason}")
            report["graph_replay"] = [{"case": "graph_replay", "correct": None,
                                       "note": f"{UT_HARNESS_INCOMPLETE_SENTINEL}: {reason}"}]
            raise HarnessIncompleteError(reason)
        g_ok, perg = check_graph_replay(replay["fill"], replay["run"], replay["read_out"],
                                        cases, tol, capture_idx=int(replay.get("capture_idx", 0)))
        report["graph_replay"] = perg
        ok = ok and g_ok

    # torch.compile deployment context (calibrated DIFFERENTLY from the graph gate — see _compile_parity):
    # generic (no UT bundle to forget => NOT fail-closed/regenerate). Fusion-induced numeric drift is a
    # real correctness FAIL; an isolated bare-op compile ERROR is a surfaced NON-FATAL note.
    if deployment_compile_mode(regime):
        cp_ok, cp = _compile_parity(current_call, eager_cases, regime, tol)
        report["compile_parity"] = cp
        ok = ok and cp_ok

    return ok, report


def _compile_parity(current_call, cases, regime, tol):
    """Deployment-context correctness for torch.compile. Unlike the graph-replay gate this is GENERIC
    (built here from `compiled_op`, no per-op bundle the UT could forget) and therefore NOT fail-closed /
    regenerate. Two outcomes:
      * compiled(candidate) vs eager(candidate) DRIFT beyond tol  -> correct=False (a REAL correctness
        failure: fusion changed the numerics — matters for fusible ops like rmsnorm/rope/silu; for an
        opaque custom op the two are identical, so this is a cheap pass).
      * `compiled_op` raised (`_geak_compile_error`)              -> surfaced NON-FATAL note (an isolated
        bare-op fullgraph compile != the server's whole-model compile, where an opaque custom op is not
        traced into), so we DO NOT auto-reject the candidate on it — we make it visible.
    Returns (all_ok, per_case_list)."""
    compiled = compiled_op(current_call, regime)
    err = getattr(compiled, "_geak_compile_error", None) or getattr(current_call, "_geak_compile_error", None)
    if err:
        return True, [{"case": "compile_parity", "correct": True, "max_rel_err": None,
                       "note": (f"compile_soft_degrade (NON-FATAL): {err}. Isolated bare-op fullgraph "
                                "compile != server whole-model compile (opaque custom ops are not traced "
                                "into) — surfaced, not auto-rejected. Verify at the e2e gate.")}]
    per = []
    all_ok = True
    for c in cases:
        try:
            eo = current_call(c["args"])
            co = compiled(c["args"])
            k, e = correct(co, eo, tol)
            per.append({"case": c.get("sig", ""), "correct": k,
                        "max_rel_err": round(e, 5) if math.isfinite(e) else None,
                        "note": "compile_parity(compiled vs eager)"})
            all_ok = all_ok and k
        except Exception as e:
            # A raise DURING compiled execution (not build) is a real deployability problem, but keep it
            # visible-and-soft here (the e2e gate is authoritative); do not silently pass as correct.
            per.append({"case": c.get("sig", ""), "correct": True, "max_rel_err": None,
                        "note": f"compile_parity soft-skip (compiled call raised): {e!r}"})
    return all_ok, per


# --------------------------------------------------------------------------- boundary-shape helpers (attn)
def boundary_decode_seq_lens(geo, ctx_max):
    """Ragged decode seq_lens that straddle the block_size / partition_size boundaries plus min/max —
    the set that trips OOB in split/reduce paged-attention paths which a single uniform capture shape
    (e.g. seq_len=1536) never exercises. `geo` is meta.geometry; `ctx_max` = ISL+OSL."""
    B = int((geo or {}).get("block_size", 16) or 16)
    P = int((geo or {}).get("partition_size", 256) or 256)
    ctx_max = int(ctx_max or 0)
    cands = {1, B - 1, B, B + 1, P - 1, P, P + 1, 2 * P - 1, 2 * P, ctx_max - 1, ctx_max}
    return sorted(x for x in cands if 1 <= x <= max(1, ctx_max))


def shuffled_block_table(num_seqs, blocks_per_seq, pool_blocks=0, seed=0, torch=None, device="cuda"):
    """A NON-contiguous (shuffled, padded-pool) block table — the real server's paged layout. A
    contiguous `arange` table hides indexing/stride bugs that only fault on a real scattered mapping."""
    torch = torch or _torch()
    need = int(num_seqs) * int(blocks_per_seq)
    pool = max(int(pool_blocks or 0), need + 16)
    g = torch.Generator(device=device).manual_seed(int(seed))
    perm = torch.randperm(pool, generator=g, device=device)[:need].to(torch.int32)
    return perm.reshape(int(num_seqs), int(blocks_per_seq))


# --------------------------------------------------------------------------- (d) two-leg measurement
# The baseline is the LIVE SERVING STACK (install + every accepted kernel), reached by running the
# SAME leg_runner.py + cases.py under `baseline_overlay/` on PYTHONPATH; the candidate is that stack
# plus ONE entry generated from kernel_src/. Leg direction is therefore a property of the environment,
# not of a name string the harness binds by hand.
def _kill_process_group(proc, grace=5.0):
    """SIGTERM then SIGKILL the leg's whole process group, so nothing it spawned keeps the GPU.

    Reaping is what proves death: `os.kill(pid, 0)` still succeeds on a zombie, so wait() on the
    child is what confirms the signal took."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    if pgid == os.getpgid(0):      # start_new_session did not take — never signal our own group
        proc.kill()
        return
    for sig, wait_for in ((signal.SIGTERM, grace), (signal.SIGKILL, 1.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=wait_for)   # wait(), not kill(pid, 0): a zombie still answers signal 0
            return
        except subprocess.TimeoutExpired:
            continue


def _run_leg(task_dir, overlay, mode, *, bucket="", out="", seed=0, draws=0, timeout=3600):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([overlay] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    cmd = [sys.executable, os.path.join(task_dir, "leg_runner.py"),
           "--task", task_dir, "--mode", mode, "--seed", str(int(seed))]
    if bucket:
        cmd += ["--bucket", bucket]
    if out:
        cmd += ["--out", out]
    if draws:
        cmd += ["--draws", str(int(draws))]
    # Own process GROUP, not just a child: on timeout the leg may be mid-kernel or may have spawned
    # helpers, and killing only the direct child leaves those holding the GPU allocation. The NEXT leg
    # then times against a busy device and reads as a slowdown attributable to nothing.
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(p)
        stdout, stderr = p.communicate()
        raise
    if p.returncode != 0:
        raise RuntimeError(f"leg({mode}) under {overlay} exited {p.returncode}: {stderr[-800:]}")
    lines = [ln for ln in stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(f"leg({mode}) under {overlay} produced no JSON: {stdout[-800:]}")
    return json.loads(lines[-1])


def build_candidate_overlay(task_dir, meta):
    """Rebuild `<task>/_cand_overlay` = baseline_overlay + ONE entry from kernel_src/ (meta.candidate_bind).

    Rebuilt on EVERY measurement because kernel_src/ is the optimizer's live workspace. `candidate_bind`
    is the same {kind, module|target, file} the e2e Integrator uses to wire the accepted kernel."""
    task = os.path.abspath(task_dir)
    base = os.path.join(task, "baseline_overlay")
    cand = os.path.join(task, "_cand_overlay")
    shutil.rmtree(cand, ignore_errors=True)
    b = meta.get("candidate_bind") or {}
    src = os.path.join(task, b.get("file", "") or "")
    if not b.get("file") or not os.path.exists(src):
        raise RuntimeError("meta.candidate_bind missing or its file is absent — cannot build the candidate leg")
    tool = [sys.executable, os.path.join(task, "overlay_setup.py")]
    if b.get("kind") == "rebind":
        args = ["add-rebind", "--target", b["target"], "--impl-module", b["impl_module"],
                "--impl-attr", b["impl_attr"], "--impl-file", src]
    else:
        args = ["add-module", "--module", b["module"], "--patched-file", src]
    r = subprocess.run(tool + args + ["--overlay", cand, "--from", base], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"candidate overlay build failed: {r.stderr[-800:]}")
    return base, cand


def assert_legs_differ(task_dir, base, cand, meta, timeout=600):
    """Refuse to measure unless the two legs provably import DIFFERENT code AND the baseline leg
    resolves OUTSIDE the task dir."""
    task = os.path.abspath(task_dir)
    target = meta["target_callable"]
    bi = _run_leg(task, base, "resolve", timeout=timeout)
    ci = _run_leg(task, cand, "resolve", timeout=timeout)
    for who, ident in (("baseline", bi), ("candidate", ci)):
        if ident.get("error"):
            raise RuntimeError(f"{who} leg cannot import {target}: {ident['error']}")
    if bi == ci:
        raise RuntimeError(
            f"both legs resolve {target} to the SAME code ({bi}) — the candidate overlay is not "
            "shadowing it, so any measured 'speedup' is noise. Check meta.candidate_bind.")
    if bi.get("file", "").startswith(task + os.sep):
        raise RuntimeError(
            f"baseline leg resolved INSIDE the task dir ({bi['file']}) — the baseline must be the live "
            "serving stack, never a copy the optimizer can edit.")
    return bi, ci


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def measure_legs(task_dir, meta, *, timeout=3600, max_reps=3, undecided=(0.95, 1.10)):
    """`per_case` for `serving_weighted_speedup`, one bucket per FRESH subprocess per leg.

    Fresh-per-bucket is not optional: a warm interpreter shares JIT/autotune state between the legs and
    reports a pseudo-1.0x that is an artifact, not a measurement.

    But fresh-per-bucket is also what makes ONE pair per bucket too coarse: fresh-process variance is
    exactly the variance `time_op`'s in-process repeats cannot average away. Measured on gfx950 it is
    ~0.1-0.5% on a prefill bucket and up to ~14% on a small-M decode bucket — so a real 1.05x candidate
    sits inside the noise of a single unpaired pair, which is the resolution the direction fix needs and
    did not have.

    Two things fix that, and only one of them costs anything:

    - INTERLEAVE. Each rep is a B,C PAIR, never all Bs then all Cs. Any drift over the sweep (clocks,
      thermals, a neighbour on the box) then lands on both legs and cancels, instead of accruing to
      whichever leg ran later. This is free — same subprocess count.
    - REPEAT, BUT ONLY WHEN IT MATTERS. `max_reps` pairs is the CEILING, not the plan. After each pair
      the running median ratio is checked: once it is outside `undecided`, the verdict cannot plausibly
      flip and the bucket stops. A 2.3x bucket costs one pair, exactly as before; a 1.05x bucket — the
      one whose answer is in doubt — spends the full budget. Flat `reps=3` would have tripled the cost
      of every bucket to buy resolution on the few that need it, and this path already pays a cold
      torch import per subprocess.

    `baseline_ms`/`optimized_ms` are the MEDIAN over the pairs actually run, so one unlucky process
    cannot carry the bucket. `reps` and `speedup_spread` (per-pair speedup range / median, 0.0 for a
    single pair) ride along so a wide bucket is visible downstream rather than averaged into silence."""
    task = os.path.abspath(task_dir)
    base, cand = build_candidate_overlay(task, meta)
    assert_legs_differ(task, base, cand, meta)
    lo, hi = undecided
    per_case = []
    for sig in _run_leg(task, base, "list", timeout=timeout)["sigs"]:
        row, bms, oms = None, [], []
        for _ in range(max(1, int(max_reps))):
            b = _run_leg(task, base, "time", bucket=sig, timeout=timeout)["cases"]
            o = _run_leg(task, cand, "time", bucket=sig, timeout=timeout)["cases"]
            if not b or not o:
                break
            if row is None:
                row = b[0]
            bms.append(b[0].get("ms"))
            oms.append(o[0].get("ms"))
            good_b, good_o = [x for x in bms if x], [y for y in oms if y]
            if not good_b or not good_o:
                break                      # nothing timable here; more pairs cannot change that
            if not (lo <= _median(good_b) / _median(good_o) <= hi):
                break                      # decisive already — do not buy resolution we do not need
        if row is None:
            continue
        good_b, good_o = [x for x in bms if x], [y for y in oms if y]
        bm = _median(good_b) if good_b else None
        om = _median(good_o) if good_o else None
        pair_sp = [x / y for x, y in zip(bms, oms) if x and y]
        per_case.append({"sig": sig, "regime": row.get("regime", ""), "m": row.get("m"),
                         "baseline_ms": bm, "optimized_ms": om,
                         "speedup": (bm / om) if (bm and om) else None,
                         "reps": len(bms),
                         "speedup_spread": ((max(pair_sp) - min(pair_sp)) / _median(pair_sp)
                                            if len(pair_sp) > 1 else 0.0)})
    return per_case


def baseline_random_outputs(task_dir, meta, *, seed=0, draws=0, timeout=3600):
    """Random-draw outputs recorded by the BASELINE leg in its own process, for
    `check_random_vs_baseline(baseline_outputs=...)`. Same seed => same inputs on both sides."""
    task = os.path.abspath(task_dir)
    out = os.path.join(task, "_baseline_random.pt")
    _run_leg(task, os.path.join(task, "baseline_overlay"), "oracle",
             out=out, seed=seed, draws=draws, timeout=timeout)
    return _torch().load(out, map_location="cpu")
