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
  trace   - matched to a real per-call shape in the profile (precise per-case weight)
  regime  - profiled time split decode/prefill (measured per regime), distributed within the regime
            by a documented prior (decode -> the full-batch bucket; prefill -> larger chunks)
  prior   - no usable profile signal; even weight across the meta cases (logged, low confidence)

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
# GEMM: cases = config M-buckets x (fixed N,K); weight = profiled time split by regime
# --------------------------------------------------------------------------- #
def attribute_gemm(meta, entries, notes):
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
    decode_us = 0.0
    prefill_us = 0.0
    matched_by_shape = {}   # bucket_M -> summed weight, when the profile DID expose a real shape
    grid_vals = []
    for k in entries:
        kw = sum(c.get("weight", 0.0) for c in k.get("cases", []))
        # (a) precise: profile exposed a real input shape for this launch
        real = next((c for c in k.get("cases", []) if c.get("dims")), None)
        if real and real["dims"]:
            m = real["dims"][0][0] if real["dims"][0] else None
            if isinstance(m, int):
                bucket = _nearest(m, decode + prefill)
                matched_by_shape[bucket] = matched_by_shape.get(bucket, 0.0) + kw
                continue
        # (b) regime via grid magnitude
        name = k.get("name", "")
        grid = _field(name, "GRID_MN")
        bn = _field(name, "BLOCK_SIZE_N")
        grid_vals.append((grid, kw))
        is_decode = None
        if grid and bn and N:
            nblk = math.ceil(N / bn)
            mblk = grid / nblk
            is_decode = mblk <= 1.5
        if is_decode is None:
            grid_vals[-1] = (grid, kw, k)  # mark for second-pass median split
            continue
        if is_decode:
            decode_us += kw
        else:
            prefill_us += kw

    # second pass: any launches we couldn't classify by N/BLOCK_N -> split by GRID_MN median
    unresolved = [g for g in grid_vals if len(g) == 3]
    if unresolved:
        gs = sorted(g[0] for g in unresolved if g[0])
        med = gs[len(gs) // 2] if gs else 0
        notes.append(f"{len(unresolved)} launches classified by GRID_MN median split (median={med}); "
                     "N/BLOCK_N not parseable for them.")
        for g in unresolved:
            grid, kw = g[0], g[1]
            if grid and grid <= med:
                decode_us += kw
            else:
                prefill_us += kw

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
        })
    return out


def _members_split(members):
    """Fractions to split a regime's total time across its member cases, proportional to the size
    proxy (time ~ elements). Falls back to even split when sizes are unknown."""
    sizes = [m["size"] for m in members]
    ssum = sum(sizes)
    if ssum <= 0:
        return [(m, 1.0 / len(members)) for m in members]
    return [(m, s / ssum) for m, s in zip(members, sizes)]


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
        for c, frac in _members_split(members):
            out.append({"name": c["name"], "dims": c["dims"], "dtypes": c["dtypes"],
                        "count": None, "weight": round(total * frac, 3),
                        "weight_source": src, "regime": regime})
    return out


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


def attribute_attn(meta, entries, notes):
    mcases = _norm_meta_cases(meta)
    if not mcases:                      # no explicit cases -> degrade to pass-through of profiled shapes
        return _passthrough(entries, notes)
    prof = _collect_prof(entries)
    matched, matched_w = _shape_match_pass(mcases, prof)
    total_w = _total_time(entries)
    regime_us = _classify_attn(mcases, entries, matched_w, total_w, notes)
    return _distribute(mcases, regime_us, matched, notes, src="regime")


def attribute_moe(meta, entries, notes):
    """MoE grouped-GEMM = a GEMM whose effective M per expert = tokens*top_k/num_experts (routing-
    dependent). The extractor bakes that effective M into decode/prefill m_buckets, so MoE reuses the
    precise grid-based GEMM engine; routing skew makes the weights lower-confidence (noted)."""
    notes.append("op_kind=moe: per-expert token counts are routing-dependent; effective-M buckets "
                 "from meta drive a GEMM-style regime split. Treat weights as lower-confidence.")
    return attribute_gemm(meta, entries, notes)


def attribute_generic(meta, entries, notes):
    mcases = _norm_meta_cases(meta)
    if not mcases:
        return _passthrough(entries, notes)
    prof = _collect_prof(entries)
    matched, matched_w = _shape_match_pass(mcases, prof)
    total_w = _total_time(entries)
    regime_us = _classify_fallback(mcases, entries, matched_w, total_w, notes)
    src = "regime_prior" if regime_us else "prior"
    return _distribute(mcases, regime_us, matched, notes, src=src)


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
    ap.add_argument("--out", required=True, help="output workload-v1 json")
    args = ap.parse_args()

    with open(args.meta) as fh:
        meta = json.load(fh)
    op_kind = (meta.get("op_kind") or "").lower()
    regime = meta.get("regime") or {}      # written by the extractor from parse_regime.py
    name_match = args.name_match or _base_token(meta.get("short_name", ""))
    entries = load_profile_entries(args.profile_weights, name_match)

    notes = []
    if not entries:
        notes.append(f"no profile entries matched name '{name_match}'; weights are prior only.")

    # op_kind-aware attribution. gemm/moe use the precise grid/bucket engine; attn and the case-based
    # kinds (recurrent / norm / elementwise / editable) share the _distribute engine, differing only
    # in their thin regime classifier. All roads produce the same {..., regime, weight_source} schema.
    if op_kind == "gemm":
        cases = attribute_gemm(meta, entries, notes)
    elif op_kind == "moe":
        cases = attribute_moe(meta, entries, notes)
    elif op_kind == "attn":
        cases = attribute_attn(meta, entries, notes)
    else:
        cases = attribute_generic(meta, entries, notes)

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

    out = {
        "schema": "workload-v1",
        "op_kind": op_kind,
        "kernel": meta.get("short_name", ""),
        "name_match": name_match,
        "regime": regime,                  # quant / kv_cache_dtype / compile / attention_backend
        "quant": quant,                    # per-operand dtypes + scales for THIS kernel
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
