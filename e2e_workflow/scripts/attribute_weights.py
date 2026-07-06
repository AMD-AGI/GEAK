#!/usr/bin/env python3
"""Unified, op_kind-aware WEIGHT ATTRIBUTION (the "Tier 2" of workload alignment).

The shape/case set for EVERY kernel type comes from the extractor's `meta.json` (the same shapes the
oracle + unittest use — config-derived M-buckets for GEMM, captured tensor cases for attn/editable
kernels). The profile only supplies a WEIGHT SIGNAL (per-kernel wall-clock time, from
`parse_profile.py --workload-out`). This script JOINS the two: it attributes the profiled time onto
meta's shape cases so the harness can benchmark exactly those cases, weighted by their real
contribution. ALL kernel-type-specific logic lives here, in ONE place, behind an op_kind switch.

It NEVER fabricates a shape — shapes only ever come from meta. When it cannot attribute precisely it
falls back to a coarser, explicitly-labelled weight (`weight_source`) so downstream knows the fidelity:
  trace       - matched to a real per-call shape in the profile (precise per-case weight)
  regime      - profiled time split decode/prefill (measured per regime), distributed within the regime
                by a documented size prior (decode -> full-batch bucket; prefill -> larger chunks)
  count_prior - a regime's MEASURED total time distributed within the regime by observed capture CALL
                COUNTS (count x size). Counts != time, so this only DISTRIBUTES a measured regime total;
                it never overrides a per-shape `trace`. Better than the size-only split when the capture
                recorded per-case frequencies.
  prior       - no usable profile signal; even weight across the meta cases (logged, low confidence)

An ORTHOGONAL correction (not a weight_source — it rescales counts, not shapes): when --isl/--osl are
given, each regime's weight is rescaled from the SHORT profiling window (which sees only ~PROFILE_NUM_STEPS
forward steps, so it under-counts decode) to the full serving lifecycle (prefill=ceil(isl/chunk) passes,
decode=osl passes). Per-call latency is measured; only the biased CALL COUNT is corrected. This directly
fixes the inaccurate shape weight-share for decode-heavy (large-OSL) workloads. See
estimate_serving_regime_calls.

Output: a workload-v1 json (the WORKLOAD_SPEC kernel_workflow's benchmark_engineer consumes):
  {schema:"workload-v1", op_kind, kernel, num_cases,
   cases:[{name, dims:[[...]], dtypes:[...], count, weight, weight_norm, weight_source, regime}], notes}

Stdlib only.
"""
import argparse, json, math, re, sys


# --------------------------------------------------------------------------- #
# profile weight signal (parse_profile.py --workload-out output)
# --------------------------------------------------------------------------- #
def load_profile_entries(path, name_match):
    """Return the profile 'kernel' entries whose name matches (substring, case-insensitive).
    For a triton GEMM these are the many autotune-specialized names of ONE logical kernel."""
    with open(path) as fh:
        wl = json.load(fh)
    nm = (name_match or "").lower()
    out = []
    for k in wl.get("kernels", []):
        if not nm or nm in k.get("name", "").lower() or nm in k.get("short_name", "").lower():
            out.append(k)
    return out


def _field(name, key):
    m = re.search(key + r"_(\d+)", name)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# SERVING-LIFECYCLE COUNT CORRECTION (fixes inaccurate shape weight-share from a short profiling window)
# --------------------------------------------------------------------------- #
# The profiler captures only PROFILE_NUM_STEPS (~40) forward steps. A request emits OSL (e.g. 1000)
# decode tokens but runs prefill exactly once, so the window sees the prefill pass in full while it
# catches only a sliver of decode. The per-case `weight` (= count x avg_us) therefore UNDER-counts
# decode. Per-call latency (weight/count = avg_us) is measured fine even in a short window; the biased
# quantity is the CALL COUNT, and that is analytically derivable from the workload params. So we rescale
# each regime's weight to what its FULL serving lifecycle would produce:
#     corrected_weight[r] = observed_per_call_latency[r] x analytic_calls[r]
#                         = observed_weight[r] x (analytic_calls[r] / observed_calls[r])
# The scale is applied uniformly within a regime, so the within-regime split is untouched and only the
# cross-regime (decode:prefill) ratio is corrected. No-op when ISL/OSL (or observed counts) are absent.
def estimate_serving_regime_calls(isl, osl, prefill_chunk=None):
    """Analytic per-request forward-pass counts per regime for a steady-state serving run — the
    denominator the short profiling window cannot observe. Per request:
      prefill: ceil(isl / chunk) forward passes (chunk = chunked-prefill token budget; default = isl,
               i.e. a single prefill pass over the whole prompt).
      decode : osl forward passes (one emitted token per step).
    The batched GEMM/attn kernel fires once per forward pass, so these ARE the per-regime call counts.
    Concurrency scales BOTH regimes equally, so it cancels in the decode:prefill ratio (not needed here).
    Returns {'prefill': int, 'decode': int}, or {} when isl/osl are missing/non-numeric."""
    try:
        isl = int(isl) if isl is not None else 0
        osl = int(osl) if osl is not None else 0
    except (TypeError, ValueError):
        return {}
    if isl <= 0 and osl <= 0:
        return {}
    chunk = int(prefill_chunk) if prefill_chunk else isl
    chunk = max(1, chunk)
    return {"prefill": math.ceil(isl / chunk) if isl > 0 else 0, "decode": max(0, osl)}


def serving_regime_scale(regime_calls_obs, workload, notes):
    """Per-regime weight multiplier converting a regime's WINDOW-observed weight to its full serving-
    lifecycle weight (see the block comment above). Returns {regime: scale}, normalized so the least-
    adjusted regime stays at 1.0 (only ratios matter after within-kernel normalization). No-op ({})
    when workload params or observed counts are missing."""
    if not workload:
        return {}
    est = estimate_serving_regime_calls(workload.get("isl"), workload.get("osl"),
                                        workload.get("prefill_chunk"))
    if not est:
        return {}
    scale = {}
    for r, obs in regime_calls_obs.items():
        analytic = est.get(r)
        if obs and obs > 0 and analytic is not None:
            scale[r] = analytic / obs
    if not scale:
        notes.append("serving-count correction requested (ISL/OSL given) but no observed per-regime "
                     "call counts in the profile/capture — skipped (weights use the raw window split).")
        return {}
    base = min(scale.values())
    if base > 0:
        scale = {r: s / base for r, s in scale.items()}
    notes.append(
        "serving-count correction (window→lifecycle): analytic calls "
        + ", ".join(f"{r}={est.get(r)}" for r in sorted(est))
        + "; observed " + ", ".join(f"{r}={regime_calls_obs.get(r, 0)}" for r in sorted(regime_calls_obs))
        + "; regime weight scale " + ", ".join(f"{r}×{scale[r]:.2f}" for r in sorted(scale))
        + " (per-call latency measured; call count from ISL/OSL — fixes short-window decode under-count).")
    return scale


def _apply_serving_scale(cases, regime_calls_obs, workload, notes):
    """Multiply every case's weight by its regime's serving-count scale. Uniform within a regime, so
    the within-regime split is preserved and only the cross-regime ratio changes. weight_source labels
    (shape/time provenance) are LEFT UNCHANGED — the count correction is an orthogonal axis, reported in
    `notes` and the top-level `serving_weight_model`. Returns the (mutated) cases."""
    scale = serving_regime_scale(regime_calls_obs, workload, notes)
    if not scale:
        return cases
    for c in cases:
        s = scale.get(c.get("regime"))
        if s:
            c["weight"] = round(c.get("weight", 0.0) * s, 3)
    return cases


def _regime_obs_calls(mcases, matched):
    """Observed per-regime call counts for the case-based paths: the capture window's per-case count,
    or the profiled trace count when the shape matched the profile (trace count is the harder signal)."""
    obs = {}
    for c in mcases:
        r = c.get("regime") or ""
        cnt = c.get("count")
        m = matched.get(c["name"])
        if m is not None and m.get("count") is not None:
            cnt = m.get("count")
        obs[r] = obs.get(r, 0) + int(cnt or 0)
    return obs


# --------------------------------------------------------------------------- #
# GEMM: cases = config M-buckets x (fixed N,K); weight = profiled time split by regime
# --------------------------------------------------------------------------- #
def attribute_gemm(meta, entries, notes, workload=None):
    a_shape = meta.get("a_shape") or ["M", None]   # [M, K]
    b_shape = meta.get("b_shape") or [None, None]  # [N, K]
    K = a_shape[1] if len(a_shape) > 1 else None
    N = b_shape[0] if b_shape else None
    in_dt = meta.get("dtype", "")
    decode = list(meta.get("decode_m_buckets") or [])
    prefill = list(meta.get("prefill_m_buckets") or [])
    if not (decode or prefill):
        # no regime split in meta -> treat every m_bucket as its own (prefill-like) case
        prefill = list(meta.get("m_buckets") or [])

    # ---- classify each profiled specialized-name into decode/prefill, sum its time ----
    # A triton GEMM hides M behind the launch grid. We do NOT need exact M, only regime:
    # M_blocks ~ GRID_MN / ceil(N/BLOCK_N); M_blocks<=~1 => decode (single M-tile), else prefill.
    decode_us = prefill_us = 0.0
    decode_calls = prefill_calls = 0     # observed WINDOW launch counts per regime (for serving correction)
    matched_by_shape = {}   # bucket_M -> summed weight, when the profile DID expose a real shape
    matched_calls = {}      # bucket_M -> summed observed count
    grid_vals = []          # unresolved launches (grid, weight, count, entry) for the median second pass
    for k in entries:
        kcases = k.get("cases", [])
        kw = sum(c.get("weight", 0.0) for c in kcases)
        kcount = sum(int(c.get("count") or 0) for c in kcases)
        # (a) precise: profile exposed a real input shape for this launch
        real = next((c for c in kcases if c.get("dims")), None)
        if real and real["dims"]:
            m = real["dims"][0][0] if real["dims"][0] else None
            if isinstance(m, int):
                bucket = _nearest(m, decode + prefill)
                matched_by_shape[bucket] = matched_by_shape.get(bucket, 0.0) + kw
                matched_calls[bucket] = matched_calls.get(bucket, 0) + kcount
                continue
        # (b) regime via grid magnitude
        name = k.get("name", "")
        grid = _field(name, "GRID_MN")
        bn = _field(name, "BLOCK_SIZE_N")
        is_decode = None
        if grid and bn and N:
            nblk = math.ceil(N / bn)
            mblk = grid / nblk
            is_decode = mblk <= 1.5
        if is_decode is None:
            grid_vals.append((grid, kw, kcount, k))  # mark for second-pass median split
            continue
        if is_decode:
            decode_us += kw; decode_calls += kcount
        else:
            prefill_us += kw; prefill_calls += kcount

    # second pass: any launches we couldn't classify by N/BLOCK_N -> split by GRID_MN median
    if grid_vals:
        gs = sorted(g[0] for g in grid_vals if g[0])
        med = gs[len(gs) // 2] if gs else 0
        notes.append(f"{len(grid_vals)} launches classified by GRID_MN median split (median={med}); "
                     "N/BLOCK_N not parseable for them.")
        for grid, kw, kcount, _k in grid_vals:
            if grid and grid <= med:
                decode_us += kw; decode_calls += kcount
            else:
                prefill_us += kw; prefill_calls += kcount

    cases = []

    def emit(M, regime, weight, src):
        dims = [[M, K], [N, K]]
        cases.append({
            "name": f"{regime}_M{M}",
            "dims": dims,
            "dtypes": [in_dt, in_dt],
            "count": None,
            "weight": round(weight, 3),
            "weight_source": src,
            "regime": regime,
            "m": M,
        })

    # ---- precise per-bucket weights where the profile gave real shapes ----
    used_shape_buckets = set()
    for bucket, w in matched_by_shape.items():
        regime = "decode" if bucket in decode else "prefill"
        emit(bucket, regime, w, "trace")
        used_shape_buckets.add(bucket)
        if regime == "decode":
            decode_calls += matched_calls.get(bucket, 0)
        else:
            prefill_calls += matched_calls.get(bucket, 0)

    # ---- regime totals distributed across the remaining buckets ----
    rem_decode = [m for m in decode if m not in used_shape_buckets]
    rem_prefill = [m for m in prefill if m not in used_shape_buckets]
    for buckets, total, regime in ((rem_decode, decode_us, "decode"),
                                   (rem_prefill, prefill_us, "prefill")):
        if not buckets:
            continue
        if total <= 0:
            # The profile window showed ZERO time for a regime that meta says exists. This is almost
            # always a capture-window artifact (e.g. a prefill-dominated profiling window misses decode),
            # NOT proof the regime is free. Emit the cases at weight 0 (prior) so they are still
            # benchmarked + visible, and warn loudly. Use --min-regime-share to floor it for serving.
            notes.append(f"WARNING: regime '{regime}' has meta buckets {buckets} but ZERO profiled "
                         f"time — likely a prefill/decode-biased profiling window. Decode-critical "
                         f"serving should set --min-regime-share to avoid ignoring it.")
            for M in buckets:
                emit(M, regime, 0.0, "prior")
            continue
        for M, frac in _within_regime_split(buckets, regime):
            emit(M, regime, total * frac, "regime")
    _apply_serving_scale(cases, {"decode": decode_calls, "prefill": prefill_calls}, workload, notes)
    return cases


def _within_regime_split(buckets, regime):
    """Distribute a regime's measured total time across its config buckets (documented prior, since
    the profile gives the regime total but not per-bucket counts for a shape-hidden GEMM).
      decode  -> steady-state serving runs at ~full batch, so the largest decode bucket (==CONC)
                 dominates; tiny M is transient.
      prefill -> larger chunks carry proportionally more time (more FLOPs), so split ~proportional to M.
    """
    buckets = sorted(buckets)
    if regime == "decode":
        # 80% on the full-batch bucket, the rest spread over the smaller ones
        if len(buckets) == 1:
            return [(buckets[0], 1.0)]
        big = buckets[-1]
        rest = buckets[:-1]
        out = [(big, 0.8)]
        for M in rest:
            out.append((M, 0.2 / len(rest)))
        return out
    # prefill: proportional to M
    s = float(sum(buckets)) or 1.0
    return [(M, M / s) for M in buckets]


def _nearest(m, buckets):
    return min(buckets, key=lambda b: abs(b - m)) if buckets else m


# --------------------------------------------------------------------------- #
# Case-based op_kinds (attn / linear-attn-recurrent / norm / elementwise / editable): meta carries
# explicit shape cases, EACH TAGGED WITH A `regime` by the extractor. They all share ONE distribution
# engine (`_distribute`); each op_kind differs only in its thin REGIME CLASSIFIER — how it splits the
# kernel's profiled time into per-regime totals. GEMM/MoE keep the precise grid-based path above; this
# is the unification for everything the trace can't pin to a shape (e.g. HIP/CUDA-graph decode).
# --------------------------------------------------------------------------- #
def _case_size(dims):
    """Size proxy for a case = element count of its first (principal) operand (e.g. tokens x feature,
    batch x packed-dim). time ~ this proxy, so within a regime the larger-batch case gets more weight."""
    for t in dims:
        if t and all(isinstance(x, int) for x in t):
            p = 1
            for x in t:
                p *= x
            return p
    return 1


def _norm_meta_cases(meta):
    """Normalize meta.cases -> [{name, dims, dtypes, regime, size}] (regime tagged by the extractor)."""
    out = []
    for mc in meta.get("cases") or []:
        dims = mc.get("input_shapes") or mc.get("dims") or []
        out.append({
            "name": mc.get("sig") or mc.get("name") or _shape_name(dims),
            "dims": dims,
            "dtypes": mc.get("input_dtypes") or mc.get("dtypes") or [],
            "regime": (mc.get("regime") or "").lower(),
            "size": _case_size(dims),
            "count": mc.get("count"),   # observed capture call count (None if absent) — a within-regime prior
        })
    return out


def _members_split(members):
    """Fractions to split a regime's MEASURED total time across its member cases. Returns
    (fractions, used_count).

    Prefer observed capture CALL COUNT × size (time ~ count × per-call FLOPs) when the capture recorded
    counts — a strictly better prior than size alone, which assumes every case was called exactly once.
    A case present in meta but with count 0 (e.g. injected from config, not seen in the short capture
    window) is floored to 1 so it still gets a size-proportional sliver rather than being zeroed by a
    capture-window artifact. Falls back to size, then even. `used_count=True` lets the caller label the
    resulting weights `count_prior` (counts ≠ time, so this is still a PRIOR that never overrides a
    measured per-shape `trace`)."""
    sizes = [m["size"] for m in members]
    counts = [int(m.get("count") or 0) for m in members]
    if any(c > 0 for c in counts):
        w = [max(c, 1) * s for c, s in zip(counts, sizes)]
        wsum = sum(w)
        if wsum > 0:
            return [(m, wi / wsum) for m, wi in zip(members, w)], True
    ssum = sum(sizes)
    if ssum <= 0:
        return [(m, 1.0 / len(members)) for m in members], False
    return [(m, s / ssum) for m, s in zip(members, sizes)], False


def _distribute(mcases, regime_us, matched, notes, src="regime"):
    """THE shared case-based engine. For each meta case: if a profiled shape matched it -> trace
    weight; else split its regime's unmatched total (`regime_us[regime]`) across that regime's
    unmatched members by the size prior. A regime meta declares but the profile timed at ZERO ->
    weight 0 prior + loud warning (a capture-window artifact, not proof it's free; floor it via
    --min-regime-share). `src` labels the prior weights (regime|regime_prior)."""
    out = []
    by_regime = {}
    for c in mcases:
        if c["name"] in matched:
            m = matched[c["name"]]
            out.append({"name": c["name"], "dims": c["dims"], "dtypes": c["dtypes"],
                        "count": m.get("count"), "weight": round(m.get("weight", 0.0), 3),
                        "weight_source": "trace", "regime": c["regime"]})
        else:
            by_regime.setdefault(c["regime"], []).append(c)
    for regime, members in by_regime.items():
        total = regime_us.get(regime, 0.0)
        if total <= 0:
            if regime:
                notes.append(f"WARNING: regime '{regime}' present in meta but ZERO profiled time — "
                             "likely a capture-biased window; set --min-regime-share to keep it.")
            for c in members:
                out.append({"name": c["name"], "dims": c["dims"], "dtypes": c["dtypes"],
                            "count": None, "weight": 0.0, "weight_source": "prior", "regime": regime})
            continue
        fracs, used_count = _members_split(members)
        lbl = "count_prior" if used_count else src
        for c, frac in fracs:
            out.append({"name": c["name"], "dims": c["dims"], "dtypes": c["dtypes"],
                        "count": c.get("count"), "weight": round(total * frac, 3),
                        "weight_source": lbl, "regime": regime})
    return out


def _count_time_crosscheck(mcases, regime_us, notes):
    """Cross-check the two independent signals: the capture CALL-COUNT regime split vs the profiler
    TIME regime split. They measure different things (frequency vs GPU-time), so some gap is expected —
    but a LARGE divergence means one source is unrepresentative (a short/biased capture window, or a
    mis-attributed profile). Surface it so a downstream reader knows the weight (which uses TIME) rests
    on a signal the capture disagrees with. Does not change any weight."""
    cby = {}
    for c in mcases:
        r = c.get("regime") or ""
        cby[r] = cby.get(r, 0) + int(c.get("count") or 0)
    ctot = sum(cby.values())
    ttot = sum(v for v in regime_us.values())
    if ctot <= 0 or ttot <= 0:
        return
    for r in sorted(set(cby) | set(regime_us)):
        cf = cby.get(r, 0) / ctot
        tf = regime_us.get(r, 0.0) / ttot
        if abs(cf - tf) >= 0.3:
            notes.append(f"CROSS-CHECK regime '{r or '?'}': {cf:.0%} of capture CALLS vs {tf:.0%} of "
                         f"profiled TIME (count≠time; large gap ⇒ biased capture window or "
                         f"mis-attributed profile — weight uses TIME).")


def _collect_prof(entries):
    prof = []
    for k in entries:
        for c in k.get("cases", []):
            if c.get("dims"):
                prof.append(c)
    return prof


def _total_time(entries):
    return sum(c.get("weight", 0.0) for k in entries for c in k.get("cases", []))


def _shape_match_pass(mcases, prof):
    """Match each meta case to a profiled (real-shape) case. Returns {case_name: prof_case} and the
    summed matched weight."""
    matched, matched_w = {}, 0.0
    for c in mcases:
        m = _best_shape_match(c["dims"], prof)
        if m is not None:
            matched[c["name"]] = m
            matched_w += m.get("weight", 0.0)
    return matched, matched_w


def _classify_attn(mcases, entries, matched_w, total_w, notes):
    """Attention regime classifier: a serving attn kernel runs in two regimes that the kernel NAME
    discriminates — prefill (`...prefill...`, big-q causal FMHA) vs decode (`...paged...`/`...decode...`,
    q=1 over the KV cache, usually graph-hidden). Split the UNMATCHED profiled time into those two
    regime totals by name; whatever can't be named falls to the regime mix present in meta by size."""
    decode_us = prefill_us = other_us = 0.0
    for k in entries:
        kw = sum(c.get("weight", 0.0) for c in k.get("cases", []) if not c.get("dims"))  # unmatched only
        name = (k.get("name", "") + " " + k.get("short_name", "")).lower()
        if any(t in name for t in ("decode", "paged", "_gqa", "mqa_decode")):
            decode_us += kw
        elif any(t in name for t in ("prefill", "context", "varlen", "fwd")):
            prefill_us += kw
        else:
            other_us += kw
    regime_us = {"decode": decode_us, "prefill": prefill_us}
    if other_us > 0:  # spread unnamed remainder across whatever regimes meta declares, by size
        regs = {c["regime"] for c in mcases if c["regime"]}
        sz = {r: sum(c["size"] for c in mcases if c["regime"] == r) for r in regs}
        ssum = sum(sz.values()) or 1.0
        for r in regs:
            regime_us[r] = regime_us.get(r, 0.0) + other_us * sz[r] / ssum
        notes.append(f"attn: {other_us:.0f}us of unnamed launches spread across meta regimes by size.")
    return regime_us


def _classify_fallback(mcases, entries, matched_w, total_w, notes):
    """Generic classifier (recurrent / norm / elementwise / editable): no name-based regime signal.
    Assign ALL unmatched profiled time to the regime(s) the extractor tagged on the cases — if the
    cases share a single regime (e.g. a pure-decode recurrent kernel) it all lands there; if they
    span regimes (or are untagged) it is pooled and the within-regime size prior splits it. This is
    what lets a HIP/CUDA-graph kernel (shapes hidden) still get a real time-proportional weight."""
    remainder = total_w - matched_w
    regs = [r for r in {c["regime"] for c in mcases}]
    if remainder <= 0:
        return {}
    # pool everything; distribute across regimes proportional to each regime's total size, so a single
    # tagged regime gets 100% and multi-regime splits by size (then _members_split splits within).
    sz = {r: sum(c["size"] for c in mcases if c["regime"] == r) for r in regs}
    ssum = sum(sz.values()) or 1.0
    out = {r: remainder * (sz[r] / ssum) for r in regs}
    notes.append(f"distributed {remainder:.0f}us of unattributed kernel time across "
                 f"{len(mcases)} shape-hidden case(s) by size prior (regime_prior) — shapes absent "
                 "from the trace (e.g. HIP/CUDA-graph decode); larger-batch case dominates.")
    return out


def attribute_attn(meta, entries, notes, workload=None):
    mcases = _norm_meta_cases(meta)
    if not mcases:                      # no explicit cases -> degrade to pass-through of profiled shapes
        return _passthrough(entries, notes)
    prof = _collect_prof(entries)
    matched, matched_w = _shape_match_pass(mcases, prof)
    total_w = _total_time(entries)
    regime_us = _classify_attn(mcases, entries, matched_w, total_w, notes)
    _count_time_crosscheck(mcases, regime_us, notes)
    cases = _distribute(mcases, regime_us, matched, notes, src="regime")
    _apply_serving_scale(cases, _regime_obs_calls(mcases, matched), workload, notes)
    return cases


def attribute_moe(meta, entries, notes, workload=None):
    """MoE grouped-GEMM = a GEMM whose effective M per expert = tokens*top_k/num_experts (routing-
    dependent). The extractor bakes that effective M into decode/prefill m_buckets, so MoE reuses the
    precise grid-based GEMM engine; routing skew makes the weights lower-confidence (noted)."""
    notes.append("op_kind=moe: per-expert token counts are routing-dependent; effective-M buckets "
                 "from meta drive a GEMM-style regime split. Treat weights as lower-confidence.")
    return attribute_gemm(meta, entries, notes, workload=workload)


def attribute_generic(meta, entries, notes, workload=None):
    mcases = _norm_meta_cases(meta)
    if not mcases:
        return _passthrough(entries, notes)
    prof = _collect_prof(entries)
    matched, matched_w = _shape_match_pass(mcases, prof)
    total_w = _total_time(entries)
    regime_us = _classify_fallback(mcases, entries, matched_w, total_w, notes)
    _count_time_crosscheck(mcases, regime_us, notes)
    src = "regime_prior" if regime_us else "prior"
    cases = _distribute(mcases, regime_us, matched, notes, src=src)
    _apply_serving_scale(cases, _regime_obs_calls(mcases, matched), workload, notes)
    return cases


def _passthrough(entries, notes):
    """No meta cases at all -> emit the profile's own per-(shape,dtype) weights verbatim."""
    cases = []
    for c in _collect_prof(entries):
        cases.append({"name": _shape_name(c["dims"]), "dims": c["dims"], "dtypes": c.get("dtypes", []),
                      "count": c.get("count"), "weight": c.get("weight", 0.0),
                      "weight_source": "trace", "regime": ""})
    if not cases:
        notes.append("no meta cases and no profiled shapes; nothing to weight.")
    return cases


def _shape_name(dims):
    return "x".join("_".join(str(d) for d in t) for t in dims if t)[:60] or "case"


def _best_shape_match(dims, prof):
    """Match a meta case's shapes to a profiled case. Exact first; else first-operand outer-dim nearest."""
    key = json.dumps([d for d in dims if d])
    for c in prof:
        if json.dumps([d for d in c["dims"] if d]) == key:
            return c
    # fuzzy: same first-operand trailing dims, nearest leading (token) dim
    if dims and dims[0]:
        lead, tail = dims[0][0], dims[0][1:]
        best, bestd = None, None
        for c in prof:
            if c["dims"] and c["dims"][0][1:] == tail and isinstance(c["dims"][0][0], int):
                d = abs(c["dims"][0][0] - lead) if isinstance(lead, int) else 0
                if bestd is None or d < bestd:
                    best, bestd = c, d
        return best
    return None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="extractor meta.json (the shape contract)")
    ap.add_argument("--profile-weights", required=True,
                    help="parse_profile.py --workload-out json (the weight signal)")
    ap.add_argument("--name-match", default="",
                    help="substring to select this kernel's profile entries (default: meta short_name)")
    ap.add_argument("--min-regime-share", type=float, default=0.0,
                    help="floor: guarantee each regime (decode/prefill) present in meta gets at least "
                         "this fraction of total weight, even if the profile under-captured it (e.g. a "
                         "prefill-only window). 0 (default) = faithful to the profile. For decode-"
                         "critical serving, set e.g. 0.3 so decode is never optimized away.")
    ap.add_argument("--served-regimes", default="",
                    help="comma-separated serving regimes THIS kernel actually executes in "
                         "(e.g. 'prefill', 'decode', or 'prefill,decode'). Cases whose regime is not "
                         "in this set are dropped BEFORE the floor/normalize, so decode shapes are "
                         "never synthesized/floored onto a prefill-only kernel (or vice-versa). This is "
                         "a kernel->regime gate: capture/inference decides shape WITHIN a served "
                         "regime; this flag decides WHICH regimes exist for the kernel. Empty (default) "
                         "= no gate = faithful to prior behavior. The extractor sets it per kernel: a "
                         "*_fwd_kernel/prefill wrapper that has a separate *_decode_kernel is "
                         "'prefill'; the decode kernel is 'decode'.")
    ap.add_argument("--isl", type=int, default=None,
                    help="input seq len (prompt tokens). With --osl, corrects the regime weight split "
                         "from the SHORT profiling window (sees only ~PROFILE_NUM_STEPS decode steps) to "
                         "the full serving lifecycle: prefill=ceil(isl/chunk) passes vs decode=osl passes. "
                         "Per-call latency is measured; only the call COUNT is rescaled. Omit = raw window.")
    ap.add_argument("--osl", type=int, default=None,
                    help="output seq len (tokens generated per request) = the true decode forward-pass "
                         "count the profiling window under-captures. Needs --isl to take effect.")
    ap.add_argument("--prefill-chunk", type=int, default=None,
                    help="chunked-prefill token budget (chunked_prefill_size / max_num_batched_tokens). "
                         "Default: isl (one prefill pass over the whole prompt).")
    ap.add_argument("--live-pct-min", type=float, default=2.0,
                    help="if the matched seam's total %%GPU in the (online-captured) profile is below "
                         "this, flag regime_warning: the seam is probably NOT the live kernel under the "
                         "online regime (e.g. an unquantized GEMM that only serves lm_head when the "
                         "server runs --quantization fp8). Default 2%%.")
    ap.add_argument("--out", required=True, help="output workload-v1 json")
    args = ap.parse_args()

    with open(args.meta) as fh:
        meta = json.load(fh)
    op_kind = (meta.get("op_kind") or "").lower()
    regime = meta.get("regime") or {}      # written by the extractor from parse_regime.py

    # workload params drive the serving-lifecycle count correction (see estimate_serving_regime_calls).
    # Empty when neither --isl nor --osl is given -> correction is a no-op (byte-identical to before).
    # The chunked-prefill budget falls back to the regime's parsed prefill_chunk (from the launch
    # script / server flags) when --prefill-chunk is not given explicitly.
    prefill_chunk = args.prefill_chunk if args.prefill_chunk is not None else regime.get("prefill_chunk")
    workload = ({"isl": args.isl, "osl": args.osl, "prefill_chunk": prefill_chunk}
                if (args.isl is not None or args.osl is not None) else None)
    name_match = args.name_match or _base_token(meta.get("short_name", ""))
    entries = load_profile_entries(args.profile_weights, name_match)

    notes = []
    if not entries:
        notes.append(f"no profile entries matched name '{name_match}'; weights are prior only.")

    # op_kind-aware attribution. gemm/moe use the precise grid/bucket engine; attn and the case-based
    # kinds (recurrent / norm / elementwise / editable) share the _distribute engine, differing only
    # in their thin regime classifier. All roads produce the same {..., regime, weight_source} schema.
    if op_kind == "gemm":
        cases = attribute_gemm(meta, entries, notes, workload=workload)
    elif op_kind == "moe":
        cases = attribute_moe(meta, entries, notes, workload=workload)
    elif op_kind == "attn":
        cases = attribute_attn(meta, entries, notes, workload=workload)
    else:
        cases = attribute_generic(meta, entries, notes, workload=workload)

    # kernel->regime gate (capture-over-inference): keep only the serving regimes THIS kernel
    # actually runs in. A prefill-only kernel (a *_fwd_kernel with a separate *_decode_kernel) must
    # NOT get decode shapes/weights synthesized or floored onto it (and vice-versa) — that is what
    # produced "optimize a decode win on a prefill kernel -> isolated speedup, e2e regression".
    # MUST run BEFORE _apply_regime_floor so the floor cannot re-inject a dropped regime. Empty
    # --served-regimes = no gate = byte-identical to prior behavior.
    served = {r.strip().lower() for r in (args.served_regimes or "").split(",") if r.strip()}
    if served:
        kept = [c for c in cases if (not c.get("regime")) or str(c.get("regime")).lower() in served]
        dropped = sorted({str(c.get("regime")) for c in cases
                          if c.get("regime") and str(c.get("regime")).lower() not in served})
        if not kept:
            notes.append(f"served-regimes gate: filtering to {sorted(served)} would drop ALL cases; "
                         f"kept original set (check --served-regimes vs the kernel's meta regimes).")
        else:
            if dropped:
                notes.append(f"served-regimes gate: kernel serves {sorted(served)}; dropped "
                             f"{len(cases) - len(kept)} case(s) in unserved regime(s) {dropped} before "
                             f"floor/normalize (prevents synthesizing/flooring a regime this kernel "
                             f"never runs, e.g. decode shapes on a prefill-only kernel).")
            cases = kept

    # optional regime floor (serving decode-protection): redistribute so each regime present in meta
    # gets >= min_regime_share of the total. Applied BEFORE normalization, on raw weights.
    if args.min_regime_share > 0:
        _apply_regime_floor(cases, args.min_regime_share, notes)

    # normalize weights within the kernel
    cases.sort(key=lambda c: c.get("weight", 0.0), reverse=True)
    wsum = sum(c.get("weight", 0.0) for c in cases) or 1.0
    for c in cases:
        c["weight_norm"] = round(c.get("weight", 0.0) / wsum, 6)

    # ---- REGIME: per-operand dtype/quant so the harness builds the SAME operands the live kernel sees ----
    quant = _quant_block(meta, regime)
    for c in cases:                        # stamp quant onto each case so the harness builds the
        c.setdefault("quant", quant)       # SAME operands online uses (fp8 + scales, not bf16)

    # live %GPU of THIS seam under the online regime (profile was captured on the real server).
    live_pct = round(sum(float(k.get("pct_gpu_time", 0.0)) for k in entries), 3)
    regime_warning = _regime_warnings(regime, op_kind, entries, live_pct, args.live_pct_min, notes)

    out = {
        "schema": "workload-v1",
        "op_kind": op_kind,
        "kernel": meta.get("short_name", ""),
        "name_match": name_match,
        "regime": regime,                  # quant / kv_cache_dtype / compile / attention_backend
        "quant": quant,                    # per-operand dtypes + scales for THIS kernel
        "live_pct_gpu": live_pct,          # this seam's share of GPU time under the online regime
        "regime_warning": regime_warning,  # non-empty => seam/regime mismatch; don't trust the weight
        "serving_weight_model": (          # None unless ISL/OSL drove the window→lifecycle correction
            {"isl": args.isl, "osl": args.osl, "prefill_chunk": prefill_chunk,
             "analytic_calls": estimate_serving_regime_calls(args.isl, args.osl, prefill_chunk)}
            if workload else None),
        "num_cases": len(cases),
        "weights_provenance": _provenance(cases),
        "cases": cases,
        "notes": " ".join(notes),
    }
    with open(args.out, "w") as fh:
        fh.write(json.dumps(out, indent=2))
    sys.stderr.write(f"wrote {args.out}: {len(cases)} cases, provenance={out['weights_provenance']}\n")
    print(json.dumps({"out": args.out, "num_cases": len(cases),
                      "weights_provenance": out["weights_provenance"], "notes": out["notes"]}))


def _apply_regime_floor(cases, floor, notes):
    """Ensure each regime present in meta holds >= `floor` of total weight. Within a floored regime,
    distribute by the same documented prior as _within_regime_split. Only meaningful with >=2 regimes."""
    regimes = sorted({c.get("regime") for c in cases if c.get("regime")})
    if len(regimes) < 2:
        return
    total = sum(c.get("weight", 0.0) for c in cases) or 1.0
    floored = [r for r in regimes
               if sum(c["weight"] for c in cases if c.get("regime") == r) / total < floor]
    if not floored:
        return
    if floor * len(floored) >= 1.0:
        notes.append(f"min-regime-share {floor} x {len(floored)} floored regimes >= 1.0; skipped.")
        return
    # Each floored regime -> exactly floor*total; the non-floored regimes share the remainder,
    # scaled down proportionally to their current weights.
    for r in floored:
        share = floor * total
        members = [c for c in cases if c.get("regime") == r]
        ms = [c["m"] for c in members if "m" in c]
        if len(ms) == len(members):                       # GEMM-style: split by M bucket
            frac_by_m = dict(_within_regime_split(ms, r))
            for c in members:
                c["weight"] = share * frac_by_m.get(c["m"], 1.0 / len(members))
                if c.get("weight_source") == "prior":
                    c["weight_source"] = "regime_floor"
        else:                                             # no per-case M: even split
            for c in members:
                c["weight"] = share / len(members)
                if c.get("weight_source") == "prior":
                    c["weight_source"] = "regime_floor"
    rest_regimes = [r for r in regimes if r not in floored]
    rest_total = sum(c["weight"] for c in cases if c.get("regime") in rest_regimes)
    keep = total * (1.0 - floor * len(floored))
    if rest_total > 0:
        scale = keep / rest_total
        for c in cases:
            if c.get("regime") in rest_regimes:
                c["weight"] *= scale
    notes.append(f"applied --min-regime-share {floor}: floored regimes {floored}.")


def _quant_block(meta, regime):
    """Per-operand dtypes + quant so the harness builds the SAME inputs the live kernel sees.
    meta (the captured/synthesized op) wins on operand specifics; regime fills gaps from launch flags."""
    rq = (regime or {}).get("quant") or {}
    return {
        "scheme": meta.get("quant_scheme") or rq.get("method") or "none",
        "weight_dtype": meta.get("dtype") or rq.get("weight_dtype") or "",
        "act_dtype": rq.get("act_dtype") or meta.get("dtype") or "",
        "out_dtype": meta.get("out_dtype") or "",
        "weight_block_size": meta.get("weight_block_size") or rq.get("block_size"),
        "scale_dtype": "float32",
        "kv_cache_dtype": (regime or {}).get("kv_cache_dtype", ""),
    }


def _regime_warnings(regime, op_kind, entries, live_pct, live_pct_min, notes):
    """Catch the 'isolated win, e2e loss' class BEFORE optimization. Returns a warning string ('' if ok).
    The profile is captured on the REAL online server, so these cross-check that THIS seam is actually the
    live kernel under the deployed regime (quant/KV-dtype/fusion) — not an out-of-regime look-alike."""
    warns = []
    q = (regime or {}).get("quant") or {}
    method = (q.get("method") or "none")
    # (1) live-seam guard: the profile is captured on the REAL online server, so the live kernel carries
    #     the GPU time. A near-zero share means this seam isn't what the workload actually runs (the
    #     classic unquantized-GEMM-serving-only-lm_head trap under --quantization fp8).
    if entries and live_pct < live_pct_min:
        warns.append(f"seam is only {live_pct}% GPU under the online regime (quant={method}); it is "
                     f"probably NOT the live kernel — verify the seam (e.g. unquantized GEMM serving "
                     f"only lm_head). Do NOT trust its weight/speedup for e2e.")
    # (2) attention KV dtype: a kernel hardcoding bf16 KV will fault under fp8 KV (bf16 stride over fp8).
    if op_kind in ("attn", "attention") and (regime or {}).get("kv_cache_dtype") in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        warns.append("online kv-cache-dtype=fp8: the attention oracle + kernel MUST use the fp8 KV "
                     "layout/stride, else a bf16-stride read over fp8 bytes faults. Verify the captured "
                     "KV dtype matches.")
    # (3) compile/graph baseline: if the online server fuses via torch.compile OR replays decode under a
    #     CUDA/HIP graph, an unfused/eager perf baseline is a strawman — the speedup won't survive e2e.
    #     enforce-eager is the explicit strawman flag; torch.compile fusion is the same trap for norm/
    #     elementwise ops whose gain comes from fusion the compiled path already has.
    if (regime or {}).get("enforce_eager"):
        warns.append("online baseline is enforce-eager/disable-cuda-graph: launch-overhead savings look "
                     "large here but vanish once decode replays under a graph. Any e2e speedup measured "
                     "against this baseline is a strawman — re-baseline with graphs/compile enabled.")
    if (regime or {}).get("compile") == "torch_compile" and op_kind in ("norm", "reduction_norm", "elementwise", "rmsnorm", ""):
        warns.append("online uses torch.compile fusion: the perf baseline must be the COMPILED/fused "
                     "path, not unfused eager, or the speedup is a strawman.")
    if warns:
        notes.extend(warns)
    return " ".join(warns)


def _base_token(short_name):
    """A stable substring for matching specialized kernel names (drop trailing shape/dim noise).
    Keeps embedded digits that are part of the name (e.g. a8w8) — only strips a trailing _NNN suffix."""
    t = short_name.strip() if short_name else ""
    t = t.split()[0]  # drop anything after whitespace (autotune params)
    t = re.sub(r"_\d+$", "", t)  # strip trailing numeric suffix (_128, _2048)
    return t or short_name


def _provenance(cases):
    srcs = sorted({c.get("weight_source", "prior") for c in cases})
    return srcs[0] if len(srcs) == 1 else "mixed(" + "+".join(srcs) + ")"


if __name__ == "__main__":
    main()
