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
            is_decode = False  # resolved after the loop via median split if needed
            grid_vals[-1] = (grid, kw, k)  # mark for second pass
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
            # always a capture-口径 artifact (e.g. a prefill-dominated profiling window misses decode),
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
# Generic (attn / norm / elementwise / linear-attn / editable): meta carries captured cases with
# explicit shapes; the profile usually exposes the same shapes -> match per case.
# --------------------------------------------------------------------------- #
def attribute_generic(meta, entries, notes):
    meta_cases = meta.get("cases") or []
    # gather all profiled cases (with real dims) across the matched kernel entries
    prof = []
    for k in entries:
        for c in k.get("cases", []):
            if c.get("dims"):
                prof.append(c)
    cases = []
    if not meta_cases:
        # no explicit meta case list -> just pass the profile's own per-(shape,dtype) weights through
        for c in prof:
            cases.append({
                "name": _shape_name(c["dims"]),
                "dims": c["dims"], "dtypes": c.get("dtypes", []),
                "count": c.get("count"), "weight": c.get("weight", 0.0),
                "weight_source": "trace", "regime": "",
            })
        if not cases:
            notes.append("no meta cases and no profiled shapes; nothing to weight.")
        return cases

    for mc in meta_cases:
        dims = mc.get("input_shapes") or mc.get("dims") or []
        dtypes = mc.get("input_dtypes") or mc.get("dtypes") or []
        match = _best_shape_match(dims, prof)
        if match is not None:
            cases.append({
                "name": mc.get("sig") or _shape_name(dims),
                "dims": dims, "dtypes": dtypes,
                "count": match.get("count"), "weight": match.get("weight", 0.0),
                "weight_source": "trace", "regime": "",
            })
        else:
            cases.append({
                "name": mc.get("sig") or _shape_name(dims),
                "dims": dims, "dtypes": dtypes, "count": None, "weight": 0.0,
                "weight_source": "prior", "regime": "",
            })
    if any(c["weight_source"] == "prior" for c in cases):
        notes.append("some meta cases had no matching profiled shape -> weight=0 prior (logged).")
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
    name_match = args.name_match or _base_token(meta.get("short_name", ""))
    entries = load_profile_entries(args.profile_weights, name_match)

    notes = []
    if not entries:
        notes.append(f"no profile entries matched name '{name_match}'; weights are prior only.")

    if op_kind == "gemm":
        cases = attribute_gemm(meta, entries, notes)
    else:
        # attn / moe / norm / elementwise / editable all carry explicit shapes in meta -> generic path.
        if op_kind == "moe":
            notes.append("op_kind=moe: per-expert token counts are routing-dependent; weights are "
                         "shape-matched where possible, else prior. Treat as low-confidence.")
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

    out = {
        "schema": "workload-v1",
        "op_kind": op_kind,
        "kernel": meta.get("short_name", ""),
        "name_match": name_match,
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


def _base_token(short_name):
    """A stable substring for matching specialized kernel names (drop shape/dim noise)."""
    t = re.split(r"[ \d]", short_name.strip())[0] if short_name else ""
    return t or short_name


def _provenance(cases):
    srcs = sorted({c.get("weight_source", "prior") for c in cases})
    return srcs[0] if len(srcs) == 1 else "mixed(" + "+".join(srcs) + ")"


if __name__ == "__main__":
    main()
