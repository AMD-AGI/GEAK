#!/usr/bin/env python3
"""Triton-family performance calculators (gfx950 first; gfx942 and the client parts by --sku).

A GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/; the Gluon pack keeps a shim at
scripts/calc_perf.py). Closed-form helpers for the Budget and Profile phases. SKU peaks are READ
from perf_knowledge/hardware/data/sku.json (the single per-SKU source; SKU_PEAKS below is built
from it, never typed in). Microarch constants: the pack's references/hardware/planning-constants.md;
formulas: references/hardware/roofline-models.md; entry: references/hardware/atlas.md.

NO SILENT TARGET, NO SILENT DTYPE. Without --sku, --arch or --peak-tflops/--peak-hbm-bw the
roofline subcommands REFUSE (they used to assume gfx950), and a dtype the SKU row does not list is
REFUSED rather than priced at the fp16 rate. Every percent-of-peak printed carries its basis pair
(numerator model; denominator datasheet unless you typed a peak) -- a datasheet denominator may
rank, never gate or close.

Subcommands:
  tflops      TFLOPS from M,N,K and measured time
  attn-tflops TFLOPS for attention backward (2.5*4*BH*S^2*D convention)
  attn-fwd    TFLOPs + roofline for attention FORWARD (causal/non-causal, real HBM model)
  roofline    arithmetic intensity, ridge point, and bound-class hint
  budget      closed-form LDS bytes, R_acc, occupancy, in-flight for a GEMM tile
  occ         waves/SIMD from KD next_free_vgpr (the REGISTER TERM only; and not the
              profiler meta VGPR -- pass --lds to get the second limiter)
  mfma        MFMA efficiency = mfma_cycles_in_loop / avg_iter_duration

`occ` and `budget` are ARCH-SENSITIVE: the register file is 512/SIMD with a 512/wave cap on
CDNA but 1536/SIMD with a 256/wave cap on RDNA, so the CDNA formula under-reports RDNA
occupancy 2-3x. Both go through amd_occupancy.py; when you have the `.s`, LLVM's own
`; Occupancy:` comment beats any model here FOR THE REGISTER TERM -- read it with
`amd_occupancy.py --asm kernel.s` (or the loop audit's KD block). It is not the kernel's
occupancy on its own: the backend computes it against the LDS it can see at compile time, which
on a launch-sized group segment is zero. `occ --lds` and `budget` supply the second limiter, and
the binding occupancy is the min of the two.

Examples:
  calc_perf.py roofline --M 4096 --N 4096 --K 8192 --dtype fp16 --sku MI355X
  calc_perf.py roofline --M 4096 --N 4096 --K 8192 --dtype fp4 --sku MI355X
  calc_perf.py roofline --M 4096 --N 4096 --K 8192 --dtype fp16 --sku MI300X   # gfx942 downgrade
  calc_perf.py tflops --M 4096 --N 4096 --K 8192 --time-us 24000 --sku MI350X
  calc_perf.py attn-fwd --bh 32 --seqlen 8192 --d 128 --time-us 6549 --causal --sku R9700
  calc_perf.py budget --bm 256 --bn 256 --bk 64 --num-warps 8 --stages 2 --dtype fp16 --sku MI350X
  calc_perf.py occ --vgpr 264 --lds 32768 --sku MI350X
  calc_perf.py occ --vgpr 249 --arch gfx1201
  calc_perf.py mfma --mfma-cycles 16 --num-mfma 256 --iter-cycles 4200
  calc_perf.py --selftest
"""
import argparse
import importlib
import json
import math
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import _hwdata  # noqa: E402  (sibling locator for perf_knowledge/hardware/data)


def _load_amd_occupancy():
    """The occupancy model is a sibling GEAK kernel tool (kernel_tools/amd_occupancy.py).
    Returns None only when it is not importable."""
    try:
        return importlib.import_module("amd_occupancy")
    except ImportError:
        return None


_OCC = _load_amd_occupancy()


def _load_sku_json():
    """perf_knowledge/hardware/data/sku.json -- THE per-SKU peak source -- or None if absent."""
    p = _hwdata.find("sku.json")
    if p is None:
        return None
    with open(p) as fh:
        return json.load(fh)


def _emit_json(a, res):
    """If --json was passed, print the result dict and signal caller to return."""
    if getattr(a, "json", False):
        res.setdefault("sku", getattr(a, "sku_label", None))
        print(json.dumps(res, indent=2))
        return True
    return False


def _build_sku_peaks():
    """SKU_PEAKS, READ from sku.json. There is deliberately no hard-coded copy here any more: the
    old in-file dict and sku.json were two transcriptions of one datasheet, and a correction applied
    to one silently left the other quoting the old number. Every row field is carried through, so
    lds_per_cu_kib stays null on RDNA (the per-WGP pool lives in hw_constants.json -- see sku.json
    _lds_doc) and vgpr_per_simd (register FILE) stays distinct from vgpr_per_wave (per-wave cap)."""
    doc = _load_sku_json()
    if doc is None:
        return {}
    return {name: dict(row) for name, row in doc["skus"].items()}


# `lds_per_cu_kib` is None on every RDNA row, and that is the recorded answer rather than a gap.
# RDNA allocates shared memory per WGP (128 KiB) with a separate per-workgroup cap (64 KiB), so a
# bare per-CU scalar is ambiguous between CU mode and WGP mode. Consumers below therefore WITHHOLD
# the LDS verdict on these parts and name `--lds-per-cu` as the remedy; nothing substitutes a number.
SKU_PEAKS = _build_sku_peaks()
_QUIET = False   # selftest silences the per-row reference-only note


def arch_default_sku(gfx):
    """The sku.json row flagged roofline_default_for_arch for `gfx`, or None (gfx1201 has none ON
    PURPOSE: R9700 peaks are product-scoped; unsupported rows are never defaults)."""
    for name, row in SKU_PEAKS.items():
        if row.get("arch") == gfx and row.get("roofline_default_for_arch") \
                and row.get("geak_support") != "unsupported":
            return name
    return None


DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1, "bf8": 1, "fp4": 0.5, "mxfp4": 0.5,
               "fp32": 4, "fp64": 8, "int8": 1, "int4": 0.5}

# sku.json peak_tflops keys for the dtype spellings above (an alias, never a substitution)
_PEAK_KEY = {"bf8": "fp8", "mxfp4": "fp4"}


def _detect_gfx():
    """Best-effort device arch from rocminfo; None if unavailable."""
    import re as _re
    import subprocess as _sp
    try:
        out = _sp.check_output(["rocminfo"], text=True, stderr=_sp.DEVNULL)
        m = _re.search(r"\b(gfx9\d{2}|gfx1\d{3})\b", out)
        return m.group(1) if m else None
    except Exception:
        return None


def skus_for_arch(gfx):
    """SKU keys whose arch matches gfx -- what to suggest when the default peaks are wrong."""
    return sorted(k for k, v in SKU_PEAKS.items() if v["arch"] == gfx)


def _arch_facts(arch):
    """The hw_constants.json block for `arch`, via the shared occupancy model, or {}.

    Read through `amd_occupancy` rather than loading hw_constants.json here: a second reader
    is a second transcription, and the divergence this whole cluster is fixing began as two
    files disagreeing about one number. Returns {} when the model is unavailable, which sends
    every consumer down the NOT-PRICED path -- the safe direction.
    """
    if _OCC is None or not hasattr(_OCC, "_arch_facts") or not arch:
        return {}
    try:
        return _OCC._arch_facts(arch) or {}
    except Exception:
        return {}


def _lds_pool(arch, per_cu_kib=None):
    """(pool_kib, scope, priced, basis): the LDS pool occupancy divides into, WITH its scope.

    Mirrors `kernel_tools/hw_budget.py` rather than re-deciding
    the question here. Three quantities have all been called "the LDS size" and they are not
    interchangeable:

      * `lds_per_cu_kib`  -- CDNA's per-CU pool. This IS the occupancy pool there.
      * `lds_per_wgp_kib` -- RDNA's per-WGP pool (a WGP is 2 CUs). This is what occupancy
                             divides into on RDNA.
      * `lds_per_wg_kib`  -- RDNA's per-WORKGROUP allocation ceiling. It caps one workgroup and
                             prices NO occupancy: two workgroups on a WGP share one pool, they
                             do not get one ceiling each.

    Halving the WGP pool to publish a per-CU scalar is what produced `lds_per_cu_kib: 64.0` for
    RDNA -- numerically identical to gfx942's genuine per-CU figure, so nothing downstream ever
    looked wrong. The scope therefore travels WITH the number: a bare `64` is unreadable. Where
    the pool is not recorded (gfx1151 has no sourced per-WGP figure) `priced` is False and there
    is simply no LDS limiter to apply -- which is not a licence to divide the ceiling by 2.
    """
    facts = _arch_facts(arch)
    is_rdna = (_OCC.family_for(arch) or "").startswith("rdna") if _OCC and arch else False
    per_wgp, per_wg = facts.get("lds_per_wgp_kib"), facts.get("lds_per_wg_kib")
    pool = per_wgp if is_rdna else per_cu_kib
    scope = ("WGP" if is_rdna else "CU") if pool else None
    if pool:
        basis = (f"{pool} KiB per {scope}"
                 + (f" (a WGP is 2 CUs; one workgroup may allocate at most {per_wg} KiB, an "
                    f"allocation ceiling that does NOT price how many workgroups fit)"
                    if is_rdna and per_wg else ""))
    elif is_rdna:
        basis = (f"RDNA pools LDS per WGP and no lds_per_wgp_kib is recorded for {arch}"
                 + (f"; only the {per_wg} KiB per-workgroup ALLOCATION LIMIT is known, and "
                    f"halving it would manufacture a per-CU quantity that does not exist at "
                    f"that granularity" if per_wg else ""))
    else:
        basis = f"no LDS pool on record for {arch or 'an unnamed target'}"
    return pool, scope, pool is not None, basis


_NO_PEAKS = {"peak_tflops_fp16": None, "peak_hbm_tb_s": None, "cus": None, "l2_mb": None}


def _refuse(msg):
    raise SystemExit(f"[calc_perf] REFUSED: {msg}")


def peak_for_dtype(entry, dtype, sku_name="?"):
    """sku.json peak for `dtype`, or a refusal. NEVER another dtype's rate: the old fallback to the
    fp16 peak is wrong in the dangerous direction for every native low-precision dtype (CDNA4 fp4 is
    4x the fp16 rate) and silently absorbed typos."""
    key = _PEAK_KEY.get(dtype, dtype)
    peaks = entry.get("peak_tflops") or {}
    if key not in peaks:
        _refuse(f"{sku_name} has no {dtype} peak in sku.json (listed: {', '.join(sorted(peaks))}). "
                f"An absent rate is not the fp16 rate -- measure it and pass --peak-tflops.")
    return peaks[key]


def resolve_sku(sku=None, dtype="fp16", peak_tflops=None, peak_hbm_bw=None,
                lds_per_cu=None, vgpr_per_simd=None, cus=None, l2_mb=None, wave_size=None,
                arch=None, need_peaks=True):
    """(label, cfg). Peaks come from sku.json only. With neither --sku nor --arch nor manual peaks
    this REFUSES: the old silent gfx950 default turned every other part's percent-of-peak into
    fiction, and GEAK tools that need an arch and were not given one refuse."""
    if not SKU_PEAKS and (sku or (arch and peak_tflops is None)):
        _refuse("sku.json not found ($GEAK_HW_DATA_DIR or perf_knowledge/hardware/data)")
    if sku:
        key = sku.upper()
        if key not in SKU_PEAKS:
            _refuse(f"unknown SKU {sku!r}; keys: {', '.join(SKU_PEAKS)}")
        entry = SKU_PEAKS[key]
        if entry.get("geak_support") == "unsupported" and not _QUIET:
            print(f"[calc_perf] NOTE: {key} ({entry['arch']}) is kept in sku.json for reference "
                  f"only -- GEAK does not support or validate this part.", file=sys.stderr)
        base = dict(entry)
        base["peak_tflops_fp16"] = (peak_for_dtype(entry, dtype, key)
                                    if (need_peaks and peak_tflops is None) else
                                    (entry.get("peak_tflops") or {}).get(_PEAK_KEY.get(dtype, dtype)))
        label = key
    elif arch:
        # --arch without --sku: peaks from the arch's DEFAULT row when sku.json flags one
        # (gfx950 -> MI355X, gfx942 -> MI300X, gfx1151 -> AI_MAX_395), labelled as such; geometry
        # (register file, wave size, LDS pool) always from the named arch. No default row (gfx1201:
        # R9700 is product-scoped) -> no peaks at all, and the roofline subcommands refuse.
        dflt = arch_default_sku(arch)
        if dflt:
            base = dict(SKU_PEAKS[dflt])
            base["peak_tflops_fp16"] = (peak_for_dtype(base, dtype, dflt)
                                        if (need_peaks and peak_tflops is None) else None)
            label = f"{arch}-default({dflt})"
        else:
            base = dict(_NO_PEAKS)
            label = f"{arch}-geometry"
        base["arch"] = arch
        m = _OCC.model_for(arch) if _OCC else None
        if m:
            # vgpr_per_wave comes from the model, which states it per-arch (gfx1250 records it as
            # UNKNOWN, None -- a family literal would print RDNA's 256 there).
            base.update({"vgpr_per_simd": m["vgpr_file_per_simd"], "wave_size": m["wave_size"],
                         "vgpr_per_wave": m.get("vgpr_per_wave")})
        # The LDS cap is per-arch and must come from the arch, or be withheld (None on RDNA).
        base["lds_per_cu_kib"] = (_arch_facts(arch) or {}).get("lds_per_cu_kib")
    elif peak_tflops is None and peak_hbm_bw is None:
        _gfx = _detect_gfx()
        cands = skus_for_arch(_gfx) if _gfx else []
        _refuse("no --sku, no --arch and no --peak-tflops/--peak-hbm-bw: this tool needs a target "
                "and will not assume one (it used to assume gfx950)."
                + (f" Device reports {_gfx}; pass --sku {'/'.join(cands)}" if cands else
                   " Pass --sku <KEY> (e.g. --sku MI355X) or --arch <gfx>.")
                + f" Keys: {', '.join(SKU_PEAKS)}")
    else:
        base = dict(_NO_PEAKS)
        base.update({"arch": None, "lds_per_cu_kib": None, "vgpr_per_simd": None,
                     "vgpr_per_wave": None, "wave_size": None})
        label = "manual-peaks"
    if peak_tflops is not None:
        base["peak_tflops_fp16"] = peak_tflops
    if peak_hbm_bw is not None:
        base["peak_hbm_tb_s"] = peak_hbm_bw
    if lds_per_cu is not None:
        base["lds_per_cu_kib"] = lds_per_cu
    if vgpr_per_simd is not None:
        base["vgpr_per_simd"] = vgpr_per_simd
    if arch:
        base["arch"] = arch
    if cus is not None:
        base["cus"] = cus
    if l2_mb is not None:
        base["l2_mb"] = l2_mb
    if wave_size is not None:
        base["wave_size"] = wave_size
    if need_peaks and (base.get("peak_tflops_fp16") is None or base.get("peak_hbm_tb_s") is None):
        _refuse(f"{label}: no peak on record for this target -- pass --sku, or --peak-tflops and "
                f"--peak-hbm-bw.")
    return label, base


def apply_sku_to_args(a, need_peaks=True):
    # Captured BEFORE resolve_sku folds it into cfg, because afterwards an explicitly supplied
    # figure is indistinguishable from a table lookup -- and they do not have the same standing.
    explicit_lds = getattr(a, "lds_per_cu", None)
    a._explicit_peak = getattr(a, "peak_tflops", None) is not None
    label, cfg = resolve_sku(
        getattr(a, "sku", None), getattr(a, "dtype", "fp16"),
        getattr(a, "peak_tflops", None), getattr(a, "peak_hbm_bw", None),
        explicit_lds, getattr(a, "vgpr_per_simd", None),
        getattr(a, "cus", None), getattr(a, "l2_mb", None), getattr(a, "wave_size", None),
        getattr(a, "arch", None), need_peaks=need_peaks,
    )
    a.sku_label = label
    a.arch = cfg.get("arch")
    a.peak_tflops = cfg["peak_tflops_fp16"]
    a.peak_hbm_bw = cfg["peak_hbm_tb_s"]
    a.lds_per_cu = cfg["lds_per_cu_kib"]
    # An explicit --lds-per-cu is the caller asserting a per-CU figure they read off the part;
    # it wins and carries CU scope. Otherwise resolve the pool + scope + priced triple.
    if explicit_lds is not None:
        a.lds_pool_kib, a.lds_pool_scope, a.lds_priced = explicit_lds, "CU", True
        a.lds_basis = f"{explicit_lds} KiB per CU (supplied via --lds-per-cu)"
    else:
        a.lds_pool_kib, a.lds_pool_scope, a.lds_priced, a.lds_basis = _lds_pool(
            cfg.get("arch"), cfg["lds_per_cu_kib"])
    a.vgpr_per_simd = cfg["vgpr_per_simd"]
    # May be None: gfx1250's per-wave cap is not recorded. Kept as None rather than backfilled
    # from the file size, which would state a cap the model declines to state.
    a.vgpr_per_wave = cfg.get("vgpr_per_wave")
    a.cus = cfg["cus"]
    a.l2_mb = cfg["l2_mb"]
    a.wave_size = cfg["wave_size"]


def print_sku_context(a):
    label = getattr(a, "sku_label", None)
    if label and label.endswith("-geometry"):
        # --arch without --sku on an arch with no default row: only the register model and the
        # LDS pool are known; there are no peaks / CU / L2 numbers at all.
        print(f"arch             = {a.arch}  (register geometry + LDS pool only; no peak row for "
              f"this arch -- pass --sku for a roofline)")
    elif label and "-default(" in label:
        print(f"SKU              = {label}  (arch default row of sku.json; pass --sku for the exact "
              f"product)")
    elif label:
        print(f"SKU              = {label}")
    print(f"CUs              = {a.cus}  |  L2 = {a.l2_mb} MiB")
    if getattr(a, "grid_tiles", None) and a.cus:
        sat = a.grid_tiles / max(a.cus, 1)
        print(f"grid saturation  = {a.grid_tiles}/{a.cus} = {sat:.2f} tiles/CU"
              f"  ({'under-utilized' if sat < 1 else 'ok' if sat <= 2 else 'high'})")


def _basis(a):
    """The roofline basis pair every percent-of-peak here carries. FLOPs are a closed-form model
    over a measured time (numerator = model); the peak is the sku.json datasheet cell unless the
    caller typed one (empirical, origin unknown). Datasheet denominators may RANK only."""
    den = "empirical@hand-entered" if getattr(a, "_explicit_peak", False) else "datasheet"
    return {"numerator_basis": "model", "denominator_basis": den}


def tflops(a):
    apply_sku_to_args(a)
    flops = 2.0 * a.M * a.N * a.K
    val = flops / (a.time_us * 1e6)
    peak = a.peak_tflops
    if _emit_json(a, {"kind": "tflops", "flops": flops, "tflops": val, "dtype": a.dtype,
                      "peak_tflops": peak, "pct_of_peak": 100 * val / peak, **_basis(a)}):
        return
    print_sku_context(a)
    print(f"FLOPs            = {flops:.3e}")
    print(f"TFLOPS           = {val:.1f}")
    print(f"% of {a.dtype} peak ({peak:.0f}) = {100*val/peak:.1f}%   "
          f"[numerator {_basis(a)['numerator_basis']}, denominator {_basis(a)['denominator_basis']}]")


def roofline(a):
    apply_sku_to_args(a)
    b = DTYPE_BYTES[a.dtype]
    flops = 2.0 * a.M * a.N * a.K
    min_bytes = (a.M * a.K + a.K * a.N) * b + a.M * a.N * (2 if b <= 2 else 4)
    intensity = flops / min_bytes
    ridge = a.peak_tflops / a.peak_hbm_bw
    if _emit_json(a, {
            "kind": "roofline", "flops": flops, "min_bytes": min_bytes,
            "intensity": intensity, "intensity_lower": intensity, "intensity_upper": intensity,
            "ridge": ridge, "regime": "compute" if intensity >= ridge else "memory",
            "note": "gemm min-bytes = UPPER (optimistic) intensity bound", **_basis(a)}):
        return
    print_sku_context(a)
    print(f"peak {a.dtype} matrix = {a.peak_tflops:.1f} TF  |  peak HBM = {a.peak_hbm_bw:.2f} TB/s"
          f"  (datasheet denominators: rank only -- see roofline_on_mi.md)")
    print(f"arithmetic intensity = {intensity:.1f} ops/byte")
    print(f"ridge point          = {ridge:.1f} ops/byte")
    if intensity >= ridge:
        print("=> COMPUTE-bound: optimize MFMA continuity first (pipeline/slicing/compiler)")
    else:
        print("=> MEMORY-bound: optimize the memory path first (coalescing/in-flight)")


def _need_arch(a, what):
    if not getattr(a, "arch", None):
        _refuse(f"`{what}` needs a target arch (register file, wave size, LDS pool): pass --sku or "
                f"--arch.")


def budget(a):
    apply_sku_to_args(a, need_peaks=False)
    _need_arch(a, "budget")
    b = DTYPE_BYTES[a.dtype]
    a_tile = a.bm * a.bk * b
    b_tile = a.bk * a.bn * b
    lds = a.stages * (a_tile + b_tile)
    r_acc = a.bm * a.bn / (a.num_warps * a.wave_size)
    print_sku_context(a)
    print(f"A_tile/B_tile bytes  = {a_tile/1024:.1f} KiB / {b_tile/1024:.1f} KiB")
    if a.lds_priced:
        lds_cap = a.lds_pool_kib * 1024
        print(f"LDS_bytes ({a.stages}-stage) = {lds/1024:.1f} KiB  "
              f"(pool {a.lds_pool_kib:.0f} KiB/{a.lds_pool_scope})"
              f"  {'OK' if lds <= lds_cap else 'OVER!'}")
    else:
        print(f"LDS_bytes ({a.stages}-stage) = {lds/1024:.1f} KiB  (pool NOT PRICED)")
    if a.vgpr_per_wave:
        print(f"R_acc/thread         = {r_acc:.0f} dwords  (cap {a.vgpr_per_wave} VGPR/wave"
              f" on {a.arch})  {'OK' if r_acc <= a.vgpr_per_wave else 'check R_total'}")
    else:
        print(f"R_acc/thread         = {r_acc:.0f} dwords  (per-wave VGPR cap NOT RECORDED "
              f"for {a.arch}; pass --vgpr-per-simd or check the ISA guide)")
    if a.lds_priced:
        print(f"wg_per_{a.lds_pool_scope} by LDS    = {int((a.lds_pool_kib * 1024) // max(lds, 1))}")
    else:
        print(f"wg per CU/WGP by LDS = NOT PRICED -- {a.lds_basis}.")
        print(f"                       Pass --lds-per-cu <KiB> to price it.")
    if getattr(a, "M", None) and getattr(a, "N", None):
        tiles = math.ceil(a.M / a.bm) * math.ceil(a.N / a.bn)
        print(f"grid tiles (M,N)     = {tiles}  -> saturation {tiles/a.cus:.2f} tiles/CU")


def mfma(a):
    # a pure ratio of two cycle counts: no peak, no arch, nothing to resolve
    eff = a.mfma_cycles * a.num_mfma / a.iter_cycles
    print(f"MFMA efficiency = {100*eff:.2f}%  (target ~98%+ for compute-bound hot loop)")


def attn_tflops(a):
    apply_sku_to_args(a)
    flops = 2.5 * 4.0 * a.bh * a.seqlen * a.seqlen * a.d
    val = flops / (a.time_us * 1e6)
    peak = a.peak_tflops
    print_sku_context(a)
    print(f"attn-bwd FLOPs   = {flops:.3e}  (convention 2.5 * 4 * BH * S^2 * D)")
    print(f"TFLOPS           = {val:.1f}")
    print(f"% of {a.dtype} peak ({peak:.0f}) = {100*val/peak:.1f}%   "
          f"[numerator {_basis(a)['numerator_basis']}, denominator {_basis(a)['denominator_basis']}]")


def attn_fwd(a):
    """Forward flash-attention TFLOPs + roofline (causal & non-causal).

    FLOPs:
      non-causal = 4 * B * H * S^2 * D   (full Q@K^T + softmax@V)
      causal     = 2 * B * H * S^2 * D   (lower triangle only)

    Arithmetic intensity uses the REAL attention HBM-traffic model (not GEMM
    min-bytes): Q + O read/written once, K/V re-read once per Q-tile (causal
    halves the N-tiles on average). This is a LOWER bound on achievable
    intensity — L2 caches part of K/V on re-reads, so actual HBM traffic is
    lower and achieved TFLOPs can meet/slightly-exceed the HBM ceiling (that
    is the saturation signal, not an inconsistency). For a self-consistent
    upper bound use % of compute peak (always <= 100%)."""
    apply_sku_to_args(a)
    b = DTYPE_BYTES[a.dtype]
    S = a.seqlen
    flops = (2.0 if a.causal else 4.0) * a.bh * S * S * a.d
    val = flops / (a.time_us * 1e6)
    peak = a.peak_tflops
    # bytes: Q + O once; K and V re-read per Q-tile (causal averages half N-tiles)
    bytes_QO = 2 * a.bh * S * a.d * b
    q_tiles = max(1, S // a.block_m)
    n_rows_per_qtile = (S // 2) if a.causal else S
    bytes_KV = 2 * a.bh * q_tiles * n_rows_per_qtile * a.d * b
    total_bytes = bytes_QO + bytes_KV
    intensity = flops / total_bytes
    ridge = a.peak_tflops / a.peak_hbm_bw
    hbm_ceiling = min(a.peak_hbm_bw * intensity, peak)
    if _emit_json(a, {
            "kind": "attn-fwd", "causal": bool(a.causal), "flops": flops,
            "bytes_hbm": total_bytes, "bytes_min": bytes_QO,
            "intensity": intensity, "intensity_lower": intensity,
            "intensity_upper": flops / bytes_QO,
            "ridge": ridge, "hbm_ceiling_tf": hbm_ceiling, "tflops": val,
            "regime": "compute" if intensity >= ridge else "memory",
            "note": "attn HBM-only reread = LOWER intensity bound; upper = Q/O-once", **_basis(a)}):
        return
    print_sku_context(a)
    print(f"mode             = {'causal (lower triangle)' if a.causal else 'non-causal (full)'}")
    print(f"attn-fwd FLOPs   = {flops:.3e}  ({'2' if a.causal else '4'} * BH * S^2 * D)")
    print(f"attn-fwd bytes   = {total_bytes:.3e}  (Q+O once + K/V x {q_tiles} q-tiles"
          f"{', causal halves N-tiles' if a.causal else ', full N per tile'})")
    print(f"TFLOPS           = {val:.2f}")
    print(f"% of peak compute ({peak:.0f} TF) = {100*val/peak:.1f}%   "
          f"(self-consistent upper-bound reference)")
    print()
    print(f"arithmetic intensity = {intensity:.1f} ops/byte  (HBM-only model, "
          f"excludes L2 reuse -> LOWER bound)")
    print(f"ridge point          = {ridge:.1f} ops/byte")
    print(f"HBM-only ceiling     = {hbm_ceiling:.1f} TF  "
          f"(min(HBM_bw*intensity, compute_peak))")
    print(f"% of HBM ceiling     = {100*val/hbm_ceiling:.1f}%   "
          f"(may exceed 100% if L2 caches K/V re-reads; that signals HBM saturation)")
    if intensity >= ridge:
        print("=> COMPUTE-bound: optimize matrix-engine continuity first")
    else:
        print("=> MEMORY-bound: optimize the memory path first (more in-flight HBM / occupancy)")


def occ(a):
    apply_sku_to_args(a, need_peaks=False)
    _need_arch(a, "occ")
    if _OCC is None:
        raise SystemExit("[calc_perf] amd_occupancy.py not importable -- `occ` has no arch "
                         "model to use. Run from a composed pack, or read LLVM's "
                         "`; Occupancy:` straight out of the .s.")
    vgpr = max(a.vgpr, 1)
    waves, model = _OCC.waves_by_vgpr(vgpr, a.arch)
    print_sku_context(a)
    is_cdna = _OCC.family_for(a.arch) == "cdna"
    print(f"next_free_vgpr     = {a.vgpr}  (KD .amdhsa_next_free_vgpr"
          f"{' = ArchVGPR + AGPR' if is_cdna else '; RDNA has no AGPR file'})")
    print(f"waves/SIMD by VGPR = {waves}")
    print(f"model              = {model}")
    if not is_cdna:
        print("                     if you have the .s, LLVM's `; Occupancy:` comment is the "
              "authority FOR THE REGISTER TERM -- `amd_occupancy.py --asm kernel.s`")
    if a.lds > 0:
        # The pool and its SCOPE are printed together. A bare "wg/CU = 5" is unreadable on a
        # part whose LDS is pooled per WGP, and was flatly wrong on gfx942 back when this
        # divided by the gfx950 default regardless of --arch.
        if a.lds_priced:
            by_lds = int((a.lds_pool_kib * 1024) // max(a.lds, 1))
            print(f"wg/{a.lds_pool_scope} by LDS      = {by_lds}  (LDS/wg = {a.lds/1024:.1f} KiB, "
                  f"pool {a.lds_pool_kib:.0f} KiB/{a.lds_pool_scope})")
        else:
            print(f"wg by LDS          = NOT PRICED  (LDS/wg = {a.lds/1024:.1f} KiB) -- "
                  f"{a.lds_basis}.")
            print(f"                     Pass --lds-per-cu <KiB> to price it.")


def _selftest():
    global _QUIET
    _QUIET = True
    """Assert the MATH, not any measured kernel: synthetic integer inputs whose expected
    values are hand-computable, so a refactor that breaks a formula fails here instead of
    in a round's numbers."""
    class A:
        pass

    def _args(**kw):
        a = A()
        for k, v in dict(sku=None, arch=None, dtype="fp16", peak_tflops=None, peak_hbm_bw=None,
                         lds_per_cu=None, vgpr_per_simd=None, cus=None, l2_mb=None,
                         wave_size=None, json=True, grid_tiles=None).items():
            setattr(a, k, v)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    import contextlib
    import io

    def _run(fn, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(_args(**kw))
        return json.loads(buf.getvalue())

    shape = dict(bh=8, seqlen=1024, d=128, time_us=1000.0, block_m=128, dtype="bf16",
                 sku="MI350X")
    full = _run(attn_fwd, causal=False, **shape)
    half = _run(attn_fwd, causal=True, **shape)
    # causal is exactly the lower triangle -> exactly half the FLOPs of the full attention
    assert half["flops"] * 2 == full["flops"], (half["flops"], full["flops"])
    # 4 * BH * S^2 * D, hand-computed
    assert full["flops"] == 4 * 8 * 1024 * 1024 * 128, full["flops"]
    # K/V re-read scales linearly with the number of Q-tiles: 2x the tiles -> 2x the KV bytes
    kv1 = _run(attn_fwd, causal=False, **dict(shape, block_m=128))
    kv2 = _run(attn_fwd, causal=False, **dict(shape, block_m=64))
    assert (kv2["bytes_hbm"] - kv2["bytes_min"]) == 2 * (kv1["bytes_hbm"] - kv1["bytes_min"])
    assert kv1["bytes_min"] == kv2["bytes_min"], "Q+O traffic must not depend on BLOCK_M"
    # % of compute peak is a self-consistent upper bound: it can never exceed 100
    for t in (1.0, 10.0, 1e5):
        r = _run(attn_fwd, causal=False, **dict(shape, time_us=t))
        assert r["tflops"] / r["hbm_ceiling_tf"] > 0
        assert min(r["tflops"] / (SKU_PEAKS["MI350X"]["peak_tflops"]["bf16"]), 1.0) <= 1.0
    slow = _run(attn_fwd, causal=False, **dict(shape, time_us=1e6))
    assert 100 * slow["tflops"] / SKU_PEAKS["MI350X"]["peak_tflops"]["bf16"] <= 100.0
    # bound class flips exactly at the ridge: same SKU, intensity crossed by shrinking K
    lo = _run(roofline, M=64, N=64, K=64, dtype="fp16", sku="MI350X")
    hi = _run(roofline, M=8192, N=8192, K=8192, dtype="fp16", sku="MI350X")
    assert lo["intensity"] < lo["ridge"] and lo["regime"] == "memory", lo
    assert hi["intensity"] >= hi["ridge"] and hi["regime"] == "compute", hi
    assert abs(lo["ridge"] - hi["ridge"]) < 1e-9, "the ridge does not depend on the SHAPE"
    # ridge = peak_tflops / peak_hbm, hand-checkable
    assert abs(hi["ridge"] - 2300.0 / 8.0) < 1e-6, hi["ridge"]
    # ...but it DOES depend on the dtype, because the numerator is that dtype's peak. Same SKU,
    # same shape, three crossovers an order apart -- hw_budget.py used to read one stored fp16
    # ridge for all of them and put every fp32 kernel of moderate intensity on the wrong side.
    r8 = _run(roofline, M=8192, N=8192, K=8192, dtype="fp8", sku="MI350X")
    r32 = _run(roofline, M=8192, N=8192, K=8192, dtype="fp32", sku="MI350X")
    assert abs(r8["ridge"] - 2.0 * hi["ridge"]) < 1e-6, (r8["ridge"], hi["ridge"])
    assert r32["ridge"] < hi["ridge"] / 10, (r32["ridge"], hi["ridge"])

    # ONE SOURCE. SKU_PEAKS is READ from sku.json -- it used to be a second hand-typed copy that a
    # correction could reach on one side only. Assert there is no copy left to drift: every row
    # and every field is the file's own.
    ref = _load_sku_json()
    assert ref is not None, "sku.json must be reachable through _hwdata (perf_knowledge/hardware/data)"
    assert list(ref["skus"]) == list(SKU_PEAKS), "SKU_PEAKS must be exactly sku.json's rows, in order"
    for name, row in SKU_PEAKS.items():
        assert row == ref["skus"][name], f"{name}: SKU_PEAKS diverged from sku.json"
    # gfx950 is the main line: its rows lead the table
    assert [SKU_PEAKS[n]["arch"] for n in list(SKU_PEAKS)[:2]] == ["gfx950", "gfx950"], list(SKU_PEAKS)

    # NO SILENT DTYPE: an absent rate is refused, never priced at fp16.
    for sku, dt in (("MI300X", "fp4"), ("R9700", "fp4"), ("AI_MAX_395", "fp8")):
        try:
            resolve_sku(sku, dtype=dt)
            raise AssertionError(f"{sku}/{dt}: absent rate was not refused")
        except SystemExit as e:
            assert "no " + dt + " peak" in str(e), e
    # ...and a rate that IS listed is the row's own, not fp16's
    assert resolve_sku("MI355X", dtype="fp4")[1]["peak_tflops_fp16"] == 10000.0
    assert resolve_sku("MI350X", dtype="fp4")[1]["peak_tflops_fp16"] == 9200.0
    # NO SILENT TARGET: no --sku / --arch / manual peaks is refused (it used to assume gfx950)
    _saved_detect = globals()["_detect_gfx"]
    try:
        globals()["_detect_gfx"] = lambda: None
        try:
            resolve_sku(None)
            raise AssertionError("no target was not refused")
        except SystemExit as e:
            assert "REFUSED" in str(e) and "--sku" in str(e), e
    finally:
        globals()["_detect_gfx"] = _saved_detect
    # --arch alone takes the arch's DEFAULT row (labelled), and gfx1201 has none (product-scoped)
    assert resolve_sku(arch="gfx950")[0] == "gfx950-default(MI355X)"
    assert resolve_sku(arch="gfx942")[0] == "gfx942-default(MI300X)"
    assert arch_default_sku("gfx1201") is None and arch_default_sku("gfx1200") is None
    try:
        resolve_sku(arch="gfx1201")
        raise AssertionError("gfx1201 without --sku must not get peaks")
    except SystemExit:
        pass

    # --- the LDS pool, its SCOPE, and whether it is priced at all -------------------------
    # Written so a WRONG value has to FAIL. The original defect was a scope-free per-CU scalar
    # that was RIGHT on gfx942, so any check that only reads a number back passes under it; the
    # scope assertions and the not-priced cases are what reject it.
    assert _lds_pool("gfx942", 64.0)[:3] == (64.0, "CU", True), _lds_pool("gfx942", 64.0)
    assert _lds_pool("gfx950", 160.0)[:3] == (160.0, "CU", True), _lds_pool("gfx950", 160.0)
    # RDNA prices from the per-WGP pool, NOT from any per-CU figure, and says "WGP" out loud.
    for rdna in ("gfx1100", "gfx1200", "gfx1201"):
        pool, scope, priced, _ = _lds_pool(rdna, None)
        assert (pool, scope, priced) == (128, "WGP", True), (rdna, pool, scope, priced)
    # The halved figure must NOT come back even if a caller hands one in: on RDNA the per-CU
    # column is not the pool, so passing 64.0 changes nothing about the answer.
    assert _lds_pool("gfx1201", 64.0)[:3] == (128, "WGP", True), _lds_pool("gfx1201", 64.0)
    # gfx1151 has no sourced per-WGP figure: NOT PRICED, and the reason names the ceiling it
    # refused to halve. hw_constants.json marks this arch NOT PRICED on purpose.
    pool, scope, priced, basis = _lds_pool("gfx1151", None)
    assert (pool, scope, priced) == (None, None, False), (pool, scope, priced)
    assert "ALLOCATION LIMIT" in basis and "halving" in basis, basis
    assert _lds_pool("gfx9999", None)[2] is False
    assert _lds_pool(None, None)[2] is False
    # The LOOKUP'S OWN FAILURE PATH. `_arch_facts` swallows a raising model and returns {},
    # and every assertion above reaches it only on the happy path -- so a literal parked in
    # that `except` satisfies all of them while turning the one fail-SAFE branch in this file
    # into a fail-dangerous one. It must degrade to NOT PRICED, never to a plausible number.
    class _Raising:
        MODELS = _OCC.MODELS
        family_for = staticmethod(_OCC.family_for)

        @staticmethod
        def _arch_facts(_a):
            raise RuntimeError("hw_constants unreadable")
    _saved_occ = _OCC
    try:
        globals()["_OCC"] = _Raising
        assert _arch_facts("gfx950") == {}, _arch_facts("gfx950")
        _broken = _lds_pool("gfx950", None)
        assert _broken[2] is False, (
            f"with the hardware table unreadable the pool must be NOT PRICED, got {_broken}")
        _broken_rdna = _lds_pool("gfx1201", None)
        assert _broken_rdna[2] is False, _broken_rdna
    finally:
        globals()["_OCC"] = _saved_occ
    assert _OCC is _saved_occ
    # CU and WGP pools must not be confusable: same number, different scope, different meaning.
    assert _lds_pool("gfx1201", None)[1] != _lds_pool("gfx942", 64.0)[1]

    # Defect A regression: --arch without --sku must price LDS from THAT arch, not from the
    # gfx950 default. A constant-cap implementation returns 160.0 for every arch and fails here.
    for gfx, want_pool, want_scope in (("gfx942", 64, "CU"), ("gfx950", 160, "CU"),
                                       ("gfx1201", 128, "WGP")):
        _a = _args(arch=gfx, vgpr=128, lds=32768)
        apply_sku_to_args(_a, need_peaks=False)
        assert (_a.lds_pool_kib, _a.lds_pool_scope) == (want_pool, want_scope), (
            f"--arch {gfx}: priced {_a.lds_pool_kib} KiB/{_a.lds_pool_scope}, "
            f"expected {want_pool} KiB/{want_scope}")
    # the whole point of the regression: gfx942 and gfx950 must not resolve to the same pool
    _a942, _a950 = _args(arch="gfx942"), _args(arch="gfx950")
    apply_sku_to_args(_a942, need_peaks=False); apply_sku_to_args(_a950, need_peaks=False)
    assert _a942.lds_pool_kib != _a950.lds_pool_kib, "the LDS cap is not tracking --arch"
    # every RDNA SKU refuses rather than quoting a number; every CDNA SKU prices in CU scope
    for name, row in SKU_PEAKS.items():
        _a = _args(sku=name)
        apply_sku_to_args(_a, need_peaks=False)
        if (_OCC.family_for(row["arch"]) or "").startswith("rdna"):
            assert row["lds_per_cu_kib"] is None, (
                f"{name}: a per-CU LDS scalar on an RDNA part is the WGP pool halved")
            # priced only where a per-WGP pool is on record; gfx1151 has none
            if _arch_facts(row["arch"]).get("lds_per_wgp_kib"):
                assert (_a.lds_priced, _a.lds_pool_scope) == (True, "WGP"), name
            else:
                assert _a.lds_priced is False, f"{name}: no per-WGP pool, must be NOT PRICED"
        else:
            assert (_a.lds_priced, _a.lds_pool_scope) == (True, "CU"), name
    # an explicit --lds-per-cu prices a part the tables decline to price -- the stated remedy
    # has to actually work, or it is a consolation rather than a fix
    _a = _args(sku="AI_MAX_395", lds_per_cu=64.0)
    apply_sku_to_args(_a, need_peaks=False)
    assert (_a.lds_priced, _a.lds_pool_kib, _a.lds_pool_scope) == (True, 64.0, "CU"), _a.lds_basis

    # --- the per-wave VGPR cap comes from the model, which states it per-arch ---------------
    # A family literal (`256 if family != "cdna" else 512`) passes on gfx942/gfx950/gfx1201 and
    # is wrong on gfx1250: CDNA5 fails `!= "cdna"`, so it took the RDNA branch and printed a
    # 256/wave cap that the model deliberately records as UNKNOWN. gfx1250 is the case that
    # separates "read the model" from "re-derive it from the family name".
    for gfx, want in (("gfx942", 512), ("gfx950", 512), ("gfx1201", 256), ("gfx1250", None)):
        _a = _args(arch=gfx)
        apply_sku_to_args(_a, need_peaks=False)
        assert _a.vgpr_per_wave == want, (
            f"--arch {gfx}: per-wave VGPR cap {_a.vgpr_per_wave!r}, expected {want!r} "
            f"(the model states it; do not re-derive it from the family)")

    # --- the pool-scope branch must have an opinion about every family the model can return ---
    # _lds_pool splits on `family.startswith("rdna")`. That branch is only safe while the set of
    # family keys is the set it was written against: a family key nobody classified would fall
    # through to the per-CU path and be priced against a pool that is not per-CU -- silently,
    # with a plausible number. So pin the vocabulary rather than the spelling of the test.
    assert _OCC is not None, "amd_occupancy.py must be importable (sibling in a composed pack)"
    _pool_scope_by_family = {"cdna": "CU", "cdna5": "CU", "rdna": "WGP"}
    assert set(_OCC.MODELS) == set(_pool_scope_by_family), (
        f"ISA family vocabulary changed: {sorted(set(_OCC.MODELS) ^ set(_pool_scope_by_family))}. "
        f"_lds_pool decides the occupancy pool per family -- classify the new family's pool "
        f"scope (CU vs WGP) there before shipping; the fall-through prices it per-CU.")

    # occupancy: the per-arch step tables, both ends of every step. Same numbers as
    # amd_occupancy's own selftest -- here to catch a SKU row wired to the wrong geometry.
    for sku, steps in (
            # The 2-wave rung ends at 256, not 255: at granule 8 the allocation is ceil8(v),
            # and 2 waves needs ceil8(v) <= 512//2 = 256, so 256 is the LAST v on this rung.
            ("MI350X", [(64, 8), (72, 7), (80, 6), (96, 5), (128, 4), (168, 3), (256, 2)]),
            ("R9700", [(96, 16), (120, 12), (144, 10), (168, 9), (192, 8), (216, 7),
                       (240, 6), (256, 5)])):
        arch = SKU_PEAKS[sku]["arch"]
        per_wave = SKU_PEAKS[sku]["vgpr_per_wave"]
        lo_v = 1
        for hi_v, want in steps:
            for v in (lo_v, hi_v):
                got, _ = _OCC.waves_by_vgpr(v, arch)
                assert got == want, f"{sku}/{arch} vgpr={v}: {got} waves, expected {want}"
            # `waves(hi) == want` alone CANNOT catch a rung top that is too LOW -- a value one
            # short of the real boundary satisfies it just as well, which is exactly how a
            # 255-for-256 survived. Assert maximality too: hi+1 must fall off the rung.
            # Bounded by vgpr_per_wave, because past the per-wave addressable cap (256 on RDNA)
            # there is no allocation to ask about and the arithmetic keeps answering anyway.
            if hi_v + 1 <= per_wave:
                nxt, _ = _OCC.waves_by_vgpr(hi_v + 1, arch)
                assert nxt != want, (
                    f"{sku}/{arch}: rung top {hi_v} is NOT maximal -- vgpr={hi_v + 1} still "
                    f"gives {want} waves. The table understates the boundary.")
            lo_v = hi_v + 1
    # every SKU row's declared geometry must match the arch model it claims
    for name, row in SKU_PEAKS.items():
        m = _OCC.model_for(row["arch"])
        assert m, f"{name}: arch {row['arch']} has no occupancy model"
        assert row["vgpr_per_simd"] == m["vgpr_file_per_simd"], name
        assert row["wave_size"] == m["wave_size"], name
        assert row["vgpr_per_wave"] <= row["vgpr_per_simd"], name
    print("[calc_perf] SELFTEST PASS")
    return 0


def main():
    if "--selftest" in sys.argv:
        return _selftest()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true", help="check the formulas and step tables")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--sku", default=None, choices=list(SKU_PEAKS),
                        help="SKU key from perf_knowledge/hardware/data/sku.json (preferred). "
                             "Without --sku/--arch/--peak-* the tool REFUSES (no gfx950 default)")
        sp.add_argument("--peak-tflops", type=float, default=None)
        sp.add_argument("--peak-hbm-bw", type=float, default=None)
        sp.add_argument("--lds-per-cu", type=float, default=None)
        sp.add_argument("--arch", default=None,
                        help="gfx target for the register/occupancy model (e.g. gfx1201); "
                             "implied by --sku")
        sp.add_argument("--vgpr-per-simd", type=int, default=None,
                        help="register FILE size per SIMD (not the per-wave cap)")
        sp.add_argument("--cus", type=int, default=None)
        sp.add_argument("--l2-mb", type=int, default=None)
        sp.add_argument("--wave-size", type=int, default=None)
        sp.add_argument("--grid-tiles", type=int, default=None)
        sp.add_argument("--json", action="store_true",
                        help="emit a JSON result dict instead of text (for normalize/resolve)")

    t = sub.add_parser("tflops"); common(t)
    t.add_argument("--M", type=int, required=True); t.add_argument("--N", type=int, required=True)
    t.add_argument("--K", type=int, required=True); t.add_argument("--time-us", type=float, required=True)
    t.add_argument("--dtype", default="fp16", choices=DTYPE_BYTES,
                   help="dtype whose sku.json peak is the denominator (absent rate -> refused)")
    t.set_defaults(func=tflops)

    r = sub.add_parser("roofline"); common(r)
    r.add_argument("--M", type=int, required=True); r.add_argument("--N", type=int, required=True)
    r.add_argument("--K", type=int, required=True); r.add_argument("--dtype", default="fp16", choices=DTYPE_BYTES)
    r.set_defaults(func=roofline)

    bd = sub.add_parser("budget"); common(bd)
    bd.add_argument("--bm", type=int, required=True); bd.add_argument("--bn", type=int, required=True)
    bd.add_argument("--bk", type=int, required=True); bd.add_argument("--num-warps", type=int, required=True)
    bd.add_argument("--stages", type=int, default=2); bd.add_argument("--dtype", default="fp16", choices=DTYPE_BYTES)
    bd.add_argument("--M", type=int, default=None); bd.add_argument("--N", type=int, default=None)
    bd.set_defaults(func=budget)

    m = sub.add_parser("mfma"); common(m)
    m.add_argument("--mfma-cycles", type=float, required=True)
    m.add_argument("--num-mfma", type=int, required=True)
    m.add_argument("--iter-cycles", type=float, required=True)
    m.set_defaults(func=mfma)

    at = sub.add_parser("attn-tflops"); common(at)
    at.add_argument("--bh", type=int, required=True); at.add_argument("--seqlen", type=int, required=True)
    at.add_argument("--d", type=int, required=True); at.add_argument("--time-us", type=float, required=True)
    at.add_argument("--dtype", default="bf16", choices=DTYPE_BYTES,
                    help="dtype whose sku.json peak is the denominator (absent rate -> refused)")
    at.set_defaults(func=attn_tflops)

    af = sub.add_parser("attn-fwd"); common(af)
    af.add_argument("--bh", type=int, required=True, help="B*H (batch * num_heads)")
    af.add_argument("--seqlen", type=int, required=True, help="sequence length S")
    af.add_argument("--d", type=int, required=True, help="head dim D")
    af.add_argument("--time-us", type=float, required=True, help="measured kernel time (us)")
    af.add_argument("--causal", action="store_true", help="causal attention (lower triangle, FLOPs=2*BH*S^2*D)")
    af.add_argument("--block-m", type=int, default=128, help="BLOCK_M tile size for K/V reread model (default 128)")
    af.add_argument("--dtype", default="bf16", choices=DTYPE_BYTES, help="dtype (default bf16)")
    af.set_defaults(func=attn_fwd)

    o = sub.add_parser("occ"); common(o)
    o.add_argument("--vgpr", type=int, required=True)
    o.add_argument("--lds", type=int, default=0)
    o.set_defaults(func=occ)

    a = p.parse_args()
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
