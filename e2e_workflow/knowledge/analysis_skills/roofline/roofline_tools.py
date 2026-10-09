#!/usr/bin/env python3
"""Mechanical primitives for the `roofline` analysis skill.

OPTIONAL helpers only. The skill is defined by SKILL.md and an agent can execute it by hand; this
module exists so the boring parts (peak-table parsing, counter parsing, unit math) are not retyped.

Design rules, per knowledge/analysis_skills/INDEX.md:
  * no hard third-party dependency (stdlib only; torch is imported lazily and optionally)
  * every helper is defensive and returns a value or None -- it never raises at the caller
  * nothing here decides anything; the routing rules live in SKILL.md

Data: peaks come from perf_knowledge/hardware/data/sku.json (GEAK's single per-SKU source; peaks.md
next to this file is a GENERATED rendering of it), keyed by the gpu_identity `sku` (then `target`) when
there is one and by the row sku.json flags as the arch default otherwise. Bound-classification
thresholds come from perf_knowledge/hardware/data/thresholds.json (`bound_classification`).
`$GEAK_HW_DATA_DIR` overrides the data directory. Every metric carries numerator_basis /
denominator_basis; a datasheet denominator ranks, it never gates or closes.

Self-test:  python3 roofline_tools.py --selftest
"""
from __future__ import annotations

import json
import math
import os
import re

SKILL_NAME = "roofline"
SKILL_VERSION = "2"

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
_DATA_REL = os.path.join("perf_knowledge", "hardware", "data")


def hw_data_file(name):
    """Path of a GEAK shared hardware data file ($GEAK_HW_DATA_DIR first, then this checkout's
    perf_knowledge/hardware/data). Returns the checkout path even when it is absent, so a caller
    that opens it gets a clean miss (None from the loaders) rather than an exception."""
    env = os.environ.get("GEAK_HW_DATA_DIR")
    if env and os.path.isfile(os.path.join(env, name)):
        return os.path.join(env, name)
    return os.path.join(_REPO_ROOT, _DATA_REL, name)


#: THE per-SKU peak table. peaks.md is its generated rendering (extract_sku.py --sync-docs).
SKU_JSON = hw_data_file("sku.json")
THRESHOLDS_JSON = hw_data_file("thresholds.json")
PEAKS_MD = os.path.join(_HERE, "peaks.md")


def _load_bound_classification(path=None):
    """thresholds.json `bound_classification`, or {} (the module then keeps its built-in values,
    which the test suite asserts equal to the file's)."""
    try:
        with open(path or THRESHOLDS_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh).get("bound_classification") or {}
    except (OSError, ValueError, AttributeError):
        return {}


_BC = _load_bound_classification()


def _bc(block, key, default):
    try:
        v = _BC[block][key]
        return type(default)(v)
    except (KeyError, TypeError, ValueError):
        return default

# element size in bytes, keyed by the dtype spellings that show up in profiles/configs
_DTYPE_BYTES = {
    "fp4": 0.5, "float4": 0.5, "mxfp4": 0.5, "e2m1": 0.5,
    "fp8": 1, "float8": 1, "e4m3": 1, "e5m2": 1, "mxfp8": 1, "fp8_e4m3": 1, "int8": 1, "uint8": 1,
    "bf16": 2, "bfloat16": 2, "c10::bfloat16": 2, "fp16": 2, "float16": 2, "half": 2, "int16": 2,
    "fp32": 4, "float32": 4, "float": 4, "int32": 4,
    "fp64": 8, "float64": 8, "double": 8, "int64": 8, "long": 8,
}

_FP8_NAMES = frozenset({
    "e4m3", "e4m3fn", "e4m3fnuz",
    "e5m2", "e5m2fn", "e5m2fnuz",
    "fp8", "float8", "mxfp8", "w8a8",
})


def _normalize_dtype_name(name):
    """Return the canonical profile dtype spelling without guessing its peak."""
    raw = str(name or "").strip().lower().removeprefix("torch.")
    if raw in _FP8_NAMES or raw.startswith(("fp8_", "float8_")):
        return "fp8"
    return raw


# target_eff priors -- SKILL.md section 7 is the source of truth; keep the two in sync.
TARGET_EFF = {
    "gemm": 0.90,
    "moe": 0.90,
    "elementwise": 0.875,
    "attn": 0.50,
}
PRODUCT_TARGET_EFF = {
    # 1 GiB source + 1 GiB destination, 30 event-timed copies on R9700:
    # median 485.8 GB/s read+write against the 640 GB/s datasheet pin rate.
    "r9700": {"moe": 0.76, "elementwise": 0.76},
}


def target_eff_for(op_class, product=None):
    """Return a product override when measured, otherwise the op-class prior."""
    op = str(op_class or "").strip().lower()
    prod = str(product or "").strip().lower()
    return PRODUCT_TARGET_EFF.get(prod, {}).get(op, TARGET_EFF.get(op))

#: Only kernels big enough for a headroom estimate to change a decision are worth analysing.
#: Below this the Amdahl ceiling is under the noise band anyway, so modelling them adds failure
#: modes without adding information. Callers should pass the run's HEAD_THRESHOLD_PCT.
DEFAULT_MIN_PCT_GPU = 5.0
DEFAULT_TOP_N = 8

#: Kernel launch + scheduling overhead. A launch whose duration is within a small multiple of this
#: is timed by dispatch, not by its transfer or its math, so a roofline ratio is meaningless for it
#: (its lever is fusion / graph capture, not kernel tuning). Order-of-magnitude runtime constant,
#: not a per-model tuning knob -- override per box if measured.
LAUNCH_OVERHEAD_S = _bc("dispatch", "launch_overhead_s", 5e-6)
LATENCY_BOUND_FACTOR = _bc("dispatch", "latency_bound_factor", 2.0)

#: A roof (memory or compute) counts as the binding limiter only when the kernel actually gets near
#: it. Below this utilization on EVERY tabulated axis -- and above the dispatch floor -- the kernel is
#: latency/occupancy-bound, NOT bandwidth- or compute-bound. This is the single most common mislabel:
#: a small arithmetic intensity does not by itself make a kernel memory-bound.
#: (thresholds.json bound_classification.roof_util -- the same 60 cut point as the SoL ladder.)
UTIL_BOUND_THRESHOLD = _bc("roof_util", "bound_min", 0.60)
UTIL_SATURATED_THRESHOLD = _bc("roof_util", "saturated_min", 0.80)
HEADROOM_SATURATED = _bc("headroom_vs_target_eff", "saturated_min", 0.90)
HEADROOM_MODERATE = _bc("headroom_vs_target_eff", "moderate_min", 0.60)
FEASIBLE_MIN = _bc("feasibility", "min", 0.001)
FEASIBLE_MAX = _bc("feasibility", "max", 1.0)

#: numerator_basis vocabulary (bytes/FLOPs) and denominator_basis vocabulary (the peak).
NUMERATOR_BASES = ("model", "counters")
DATASHEET = "datasheet"

#: The only bound types this skill may emit. Anything else is a modelling escape and must degrade
#: to "unknown" rather than inventing a category the consumer has no routing rule for.
BOUND_TYPES = ("memory", "compute", "latency", "unknown")


def select_entries(entries, min_pct_gpu=DEFAULT_MIN_PCT_GPU, top_n=DEFAULT_TOP_N):
    """Head-scoped selection: the entries worth a roofline estimate, biggest first.

    Everything below the bar is SKIPPED (absent from the artifact), not "degraded" -- a kernel too
    small to matter is not a modelling failure and should not read as one.
    """
    try:
        sel = [e for e in (entries or []) if float(e.get("pct_gpu_time") or 0) >= float(min_pct_gpu)]
    except (TypeError, ValueError, AttributeError):
        return []
    sel.sort(key=lambda e: -float(e.get("pct_gpu_time") or 0))
    return sel[:int(top_n)] if top_n else sel


def dtype_bytes(name, default=2):
    """Element size in bytes for a dtype spelling. Unknown -> `default` (never raises)."""
    if name is None:
        return default
    if isinstance(name, (int, float)):
        return float(name)
    key = _normalize_dtype_name(name)
    if key in _DTYPE_BYTES:
        return _DTYPE_BYTES[key]
    for k, v in _DTYPE_BYTES.items():          # substring fallback: "c10::BFloat16", "torch.float8_e4m3fnuz"
        if k in key:
            return v
    return default


# ---------------------------------------------------------------- peaks

def _want_product(product):
    p = str(product or "").strip().lower()
    return "" if p == "unknown" else p


def product_from_identity(identity=None, sku=None, target=None):
    """The peak-table product key from a gpu_identity record: `sku` first (the Instinct /
    R9700 row name), then `target` (the r9700|unknown product pin), else "" (-> arch default).
    Accepts the identity dict or the two fields (e.g. env_report device_sku / device_target)."""
    if isinstance(identity, dict):
        sku = identity.get("sku", sku)
        target = identity.get("target", target)
    for v in (sku, target):
        w = _want_product(v)
        if w:
            return w
    return ""


def _peaks_from_row(name, row, product=""):
    """The roofline view of one sku.json row. Memory = the DATASHEET pin rate (a rank-only
    denominator); a row whose stored ceiling is measured (gfx1151) still ranks on its pin rate, and
    the measured figure stays available as measured_ceilings."""
    try:
        bw = float(row.get("datasheet_hbm_tb_s") or row["peak_hbm_tb_s"]) * 1e12
        l2 = row.get("l2_per_xcd_mb") or row.get("l2_mb")
        out = {
            "gfx": row["arch"], "sku": name, "cu": int(row["cus"]),
            "hbm_bw_bytes_s": bw,
            "flops": {dt: float(tf) * 1e12 for dt, tf in (row.get("peak_tflops") or {}).items()},
            "l2_bytes": int(l2 * (1 << 20)) if l2 else None,
            "mall_bytes": int(row["mall_mb"] * (1 << 20)) if row.get("mall_mb") else None,
            "source": "table",
            "confidence": "medium" if "orc-derived" in (row.get("basis"), row.get("peak_hbm_basis"))
                          else "high",
            "basis": row.get("basis"),
            "denominator_basis": DATASHEET,
            "measured_ceilings": list(row.get("measured_ceilings") or []),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if product:
        out["product"] = product
    return out


def _peaks_from_sku(doc, gfx, product=None):
    skus = (doc or {}).get("skus") or {}
    want = _want_product(product)
    gfx = str(gfx or "").strip().lower()
    supported = [(n, r) for n, r in skus.items()
                 if isinstance(r, dict) and r.get("geak_support") == "supported"
                 and str(r.get("arch", "")).lower() == gfx]
    if want:
        for n, r in supported:
            if str(r.get("identity_target") or "").lower() == want:
                return _peaks_from_row(n, r, product=want)
    for n, r in supported:
        if r.get("roofline_default_for_arch"):
            return _peaks_from_row(n, r)
    return None


def _peaks_from_md(text, gfx, product=None):
    """Fallback parser for the GENERATED peaks.md yaml blocks (used only when sku.json itself is
    unreachable). Same selection rule as _peaks_from_sku."""
    want = _want_product(product)
    default = None
    for block in re.findall(r"```yaml\s*\n(.*?)```", text, re.S):
        if not re.search(r"^\s*gfx:\s*%s\s*$" % re.escape(str(gfx)), block, re.M):
            continue
        out = {"flops": {}, "source": "table", "confidence": "high",
               "denominator_basis": DATASHEET}
        in_flops = False
        for line in block.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            body = line.split("#", 1)[0].rstrip()
            if re.match(r"^\s*flops:\s*$", body):
                in_flops = True
                continue
            m = re.match(r"^(\s*)([A-Za-z0-9_]+):\s*(.*)$", body)
            if not m:
                continue
            indent, key, val = m.group(1), m.group(2), m.group(3).strip()
            if in_flops and len(indent) >= 2 and val:
                try:
                    out["flops"][key] = float(val)
                except ValueError:
                    pass
                continue
            in_flops = False
            if not val:
                continue
            out[key] = float(val) if re.match(r"^[-+0-9.eE]+$", val) else val
        if not out.get("hbm_bw_bytes_s"):
            continue
        block_product = str(out.pop("product", "") or "").strip().lower()
        is_default = str(out.pop("default_for_gfx", "")).lower() == "true"
        if "cu" in out:
            out["cu"] = int(out["cu"])
        for k in ("l2_bytes", "mall_bytes"):
            out[k] = int(out[k]) if k in out else None
        if out.get("basis") == "orc-derived":
            out["confidence"] = "medium"
        if want and block_product == want:
            out["product"] = want
            return out
        if is_default and default is None:
            default = out
    return default


def load_peaks(peaks_path, gfx, product=None):
    """Peaks for `gfx` (and a product key -- gpu_identity `sku`, else `target` -- if any), or None.

    `peaks_path` names the peak table: sku.json itself, or the generated peaks.md next to this file
    (then the sku.json it was rendered from is read; the md blocks are parsed only if that json is
    unreachable). A missing `peaks_path` -> None, never an exception.

    Selection: the supported row whose identity_target == product (and whose arch == gfx); else the
    row flagged roofline_default_for_arch for gfx (gfx950 -> MI355X, gfx942 -> MI300X, gfx1151 ->
    AI_MAX_395). gfx1201 has no default ON PURPOSE: R9700 peaks need product='r9700'.
    ``product="unknown"`` is no product constraint. Rows marked geak_support=unsupported (gfx1100,
    gfx1200) are never returned.

    Returns {"hbm_bw_bytes_s", "flops": {dtype: FLOP/s}, "cu", "gfx", "sku", "l2_bytes",
    "mall_bytes", "source": "table", "confidence", "denominator_basis": "datasheet", ...} plus
    "product" when the row was selected by product.
    """
    try:
        if not peaks_path or not os.path.isfile(peaks_path):
            return None
        if str(peaks_path).endswith(".json"):
            with open(peaks_path, "r", encoding="utf-8") as fh:
                return _peaks_from_sku(json.load(fh), gfx, product)
        if os.path.isfile(SKU_JSON):
            with open(SKU_JSON, "r", encoding="utf-8") as fh:
                return _peaks_from_sku(json.load(fh), gfx, product)
        with open(peaks_path, "r", encoding="utf-8") as fh:
            return _peaks_from_md(fh.read(), gfx, product)
    except (OSError, UnicodeDecodeError, ValueError, AttributeError, TypeError):
        return None


def derive_peaks_from_props(device=0):
    """Fallback peaks from torch device properties. confidence='low' -- see peaks.md.

    The derived HBM figure is frequently wrong for HBM3/3E (understates the pin rate), which is
    exactly why the caller must treat this as display-only.
    """
    try:
        import torch  # noqa: PLC0415 - optional, lazily imported on purpose
        p = torch.cuda.get_device_properties(device)
    except Exception:
        return None
    try:
        bw = float(getattr(p, "memory_clock_rate", 0)) * 1e3 * (float(getattr(p, "memory_bus_width", 0)) / 8.0) * 2.0
    except Exception:
        bw = 0.0
    if bw <= 0:
        return None
    return {
        "hbm_bw_bytes_s": bw,
        "flops": {},
        "cu": int(getattr(p, "multi_processor_count", 0) or 0),
        "source": "derived",
        "confidence": "low",
        "denominator_basis": "derived",
    }


def is_client_rdna4(gfx):
    """True for Navi 48 client RDNA4 (gfx1200/gfx1201/...).

    gfx1250 is CDNA5 / MI450-class and must not match this family. A bare
    ``gfx12*`` prefix is therefore wrong.
    """
    return bool(re.fullmatch(r"gfx120\d+", str(gfx or "").lower()))


def resolve_peaks(peaks_path, gfx, device=0, product=None, identity=None):
    """Resolve tabulated peaks (sku.json, see load_peaks); never derive one for client RDNA4.

    Navi 48 reports properties that are insufficient to derive WMMA throughput,
    and its GDDR clock interpretation is toolchain-dependent. An absent table
    entry on gfx120x is therefore a hard unknown rather than a low-confidence
    numeric denominator. R9700 peaks require product='r9700'. CDNA5 gfx1250
    is not in that family and may derive.
    """
    if identity is not None:
        # sku first, then target; an explicit `product` still wins when it names something
        product = _want_product(product) or product_from_identity(identity)
    peaks = load_peaks(peaks_path, gfx, product=product)
    if peaks:
        return peaks
    if is_client_rdna4(gfx):
        return None
    return derive_peaks_from_props(device)


_FLOP_KEYS = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16", "half": "fp16",
    "fp32": "fp32", "float32": "fp32", "float": "fp32",
    "fp64": "fp64", "float64": "fp64", "double": "fp64",
    "fp8": "fp8", "float8": "fp8",
    "int8": "int8",
    "fp4": "fp4", "float4": "fp4", "mxfp4": "fp4",
}


def peak_flops_for(peaks, dtype_name):
    """Peak FLOP/s for a canonical dtype key. None if that dtype is not tabulated."""
    if not peaks:
        return None
    flops = peaks.get("flops") or {}
    if not flops:
        return None
    raw = _normalize_dtype_name(dtype_name)
    key = _FLOP_KEYS.get(raw, raw)
    if key in flops:
        try:
            return float(flops[key])
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------- op models

def experts_hit(num_experts, pairs):
    """Expected number of DISTINCT experts touched by `pairs` = M*top_k routed token-expert pairs.

    E*(1-(1-1/E)^pairs). At decode this is what decides MoE weight traffic.
    """
    try:
        E, n = float(num_experts), float(pairs)
        if E <= 0 or n <= 0:
            return 0.0
        return E * (1.0 - math.pow(1.0 - 1.0 / E, n))
    except (TypeError, ValueError, OverflowError):
        return 0.0


# ---------------------------------------------------------------- the metric

def roofline_metrics(bytes_moved, flops, t_seconds, peak_bw, peak_flops, target_eff,
                     pct_gpu_time=0.0, launch_overhead_s=LAUNCH_OVERHEAD_S,
                     numerator_basis="model", denominator_basis=DATASHEET):
    """Core arithmetic of SKILL.md section 3 step 5. Returns a dict, or None on unusable input.

    Every result carries its basis pair: `numerator_basis` (model | counters -- where bytes/FLOPs
    came from) and `denominator_basis` (datasheet | empirical@<tool>-<version> | in-shape probe |
    derived -- where the peak came from). `may_gate` is True only for a counters numerator over a
    non-datasheet, non-derived denominator: a datasheet roofline RANKS, it never gates or closes.

    A missing compute peak (dtype not tabulated) makes compute_util None -- unknown, never 0.0. With
    FLOPs present and the memory axis below the bound threshold, the bound is then "unknown" with no
    verdict: reading the unknown axis as idle is how a compute-bound kernel was sold as
    latency-bound with the memory axis's headroom.

    Two outcomes deliberately produce NO verdict (`headroom_class="unknown"`), because in both the
    ratio is not evidence about the kernel:
      * dispatch-bound -- the launch is timed by overhead, not by its transfer or its math;
      * infeasible (`raw_pct` outside (0,1]) -- the byte/FLOP model is wrong. A clamped 100% must
        never be read as "saturated"; that would turn a modelling failure into a routing decision.
    """
    try:
        b, f, t = float(bytes_moved or 0), float(flops or 0), float(t_seconds or 0)
        pbw, pfl, tgt = float(peak_bw or 0), float(peak_flops or 0), float(target_eff or 0)
    except (TypeError, ValueError):
        return None
    if t <= 0 or pbw <= 0 or tgt <= 0 or (b <= 0 and f <= 0):
        return None

    achieved_bw = b / t
    achieved_flops = (f / t) if f > 0 else 0.0
    ai = (f / b) if b > 0 else float("inf")
    ridge = (pfl / pbw) if (pfl > 0 and pbw > 0) else None
    num_basis = str(numerator_basis or "model")
    den_basis = str(denominator_basis or DATASHEET)
    basis = {"numerator_basis": num_basis, "denominator_basis": den_basis,
             "may_gate": bool(num_basis == "counters" and den_basis not in (DATASHEET, "derived"))}

    # AI picks which roof the kernel is walking TOWARD (the ceiling it would hit if it stopped
    # stalling); utilization tells whether it is actually near that roof. Both are needed -- a small
    # AI does NOT by itself mean memory-bound. roofline_pct is measured on the AI-selected roof.
    compute_bound = bool(ridge is not None and pfl > 0 and ai > ridge)
    hbm_util = achieved_bw / pbw
    # None = no tabulated peak for this dtype (unknown), NOT "the compute axis is idle"
    compute_util = (achieved_flops / pfl) if pfl > 0 else None
    raw_pct = compute_util if compute_bound else hbm_util
    roof_axis = "compute" if compute_bound else "memory"

    out = {
        "bytes_est": b, "flops_est": f, "t_ms": t * 1e3,
        "achieved_bw_bytes_s": achieved_bw, "achieved_flops": achieved_flops,
        "hbm_util": hbm_util, "compute_util": compute_util,
        "arithmetic_intensity": ai, "ridge_point": ridge,
        "roofline_pct_raw": raw_pct, "target_eff": tgt, "suspect": False,
        **basis,
    }

    # (1) Dispatch-bound by time: the launch is timed by scheduling overhead, not by its own transfer
    # or math, so no roofline ratio applies. No verdict; the lever is fusion / graph capture.
    if t <= float(launch_overhead_s or 0) * LATENCY_BOUND_FACTOR:
        out.update(bound_type="latency", roofline_pct=min(max(raw_pct, 0.0), 1.0),
                   attainable_speedup=1.0, expected_e2e_gain_pct=0.0,
                   headroom_class="unknown",
                   note="per-launch time is within launch-overhead scale -> dispatch-bound; "
                        "roofline not applicable, lever is fusion / graph capture")
        return out

    # (2) Infeasible: raw_pct outside (0,1] means the byte/FLOP model is wrong, NOT that the kernel is
    # at the wall -- so it must not yield a verdict. (A compute-axis >100% is usually an unvalidated
    # peak, e.g. the BF16 MFMA microbench reading ~2x low.) Clamp for display, refuse to classify, and
    # hand back the feasibility bound the model violated so stage C knows what to measure.
    if not (FEASIBLE_MIN <= raw_pct <= FEASIBLE_MAX):
        out.update(bound_type=roof_axis, roofline_pct=min(max(raw_pct, 0.0), 1.0),
                   attainable_speedup=1.0, expected_e2e_gain_pct=0.0,
                   headroom_class="unknown", suspect=True,
                   bytes_upper_bound=pbw * t, flops_upper_bound=(pfl * t) if pfl > 0 else None,
                   note="model infeasible (raw %.3f outside (0,1]) -> NOT a saturation verdict; "
                        "re-estimate with a tighter model or measure with counters (stage C)"
                        % raw_pct)
        return out

    # (3) True limiter by utilization. If NEITHER roof is near its ceiling (and we already ruled out
    # the dispatch floor), the kernel is latency/occupancy-bound. It still has recoverable headroom --
    # keep the verdict, so a low-utilization head like paged attention still ranks by its headroom --
    # but the lever is occupancy / shorter dependency chains / fusion, NOT byte reduction (which only
    # helps a genuinely bandwidth-bound, high-util head).
    if f > 0 and compute_util is None and hbm_util < UTIL_BOUND_THRESHOLD:
        # The compute axis is UNKNOWN (no peak for this dtype) and the memory axis is not near its
        # roof: it may be compute-bound or latency-bound, and nothing here can say which. No verdict.
        out.update(bound_type="unknown", roofline_pct=raw_pct, attainable_speedup=1.0,
                   expected_e2e_gain_pct=0.0, headroom_class="unknown",
                   note="no compute peak for this dtype (compute_util unknown) and memory below "
                        "%.2f of its roof -> compute- vs latency-bound undecidable; tabulate or "
                        "measure the dtype's peak" % UTIL_BOUND_THRESHOLD)
        return out
    if (compute_util is None or compute_util < UTIL_BOUND_THRESHOLD) \
            and hbm_util < UTIL_BOUND_THRESHOLD:
        bound = "latency"
    else:
        bound = roof_axis

    attainable = max(1.0, tgt / raw_pct)
    out.update(bound_type=bound, roofline_pct=raw_pct, attainable_speedup=attainable,
               expected_e2e_gain_pct=float(pct_gpu_time or 0.0) * (1.0 - 1.0 / attainable),
               headroom_class=classify_headroom(raw_pct, tgt))
    return out


def classify_headroom(roofline_pct, target_eff):
    """SKILL.md section 3 step 6.

    Banded against `target_eff`, NOT against the raw roofline: what matters is the distance to what
    a good implementation of this class can realistically reach. Within 10% of that target means
    tuning has nothing left to give (88% vs a 90% target is saturated, not "nearly there").
    """
    try:
        p, t = float(roofline_pct), float(target_eff)
    except (TypeError, ValueError):
        return "unknown"
    if p <= 0 or t <= 0:
        return "unknown"
    if p >= HEADROOM_SATURATED * t:
        return "saturated"
    if p >= HEADROOM_MODERATE * t:
        return "moderate"
    return "underperforming"


# ---------------------------------------------------------------- counters (stage C)

#: rocprofv3 counters to request. FETCH_SIZE/WRITE_SIZE are in KiB. fp8 has no MfmaFlopsF8 on
#: current builds -- SQ_INSTS_VALU_MFMA_MOPS_F8 stands in. Availability varies by ROCm build;
#: probe with `rocprofv3 --list-avail` and drop whatever is missing (degrades to L4, never fatal).
COUNTERS = ["FETCH_SIZE", "WRITE_SIZE", "MfmaFlops", "MfmaFlopsBF16", "MfmaFlopsF16",
            "SQ_INSTS_VALU_MFMA_MOPS_F8", "MemUnitStalled", "MfmaUtil", "OccupancyPercent"]


def parse_counter_csv(path, kernel_substr=None):
    """Sum rocprofv3 counter-collection CSV rows into {counter: value}.

    Tolerates the several column spellings rocprofv3 has shipped. Returns {} on any problem.
    """
    import csv  # noqa: PLC0415 - stdlib, kept local so a broken import can't kill module load
    out = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                low = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
                name = low.get("kernel_name") or low.get("name") or low.get("kernel") or ""
                if kernel_substr and kernel_substr.lower() not in name.lower():
                    continue
                ctr = (low.get("counter_name") or low.get("counter") or "").strip()
                val = low.get("counter_value") or low.get("value") or ""
                if not ctr:
                    continue
                try:
                    out[ctr] = out.get(ctr, 0.0) + float(val)
                except ValueError:
                    continue
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}
    return out


def bytes_from_counters(counters):
    """(FETCH_SIZE + WRITE_SIZE) KiB -> bytes. None when neither counter was collected."""
    if not counters:
        return None
    fetch, write = counters.get("FETCH_SIZE"), counters.get("WRITE_SIZE")
    if fetch is None and write is None:
        return None
    return (float(fetch or 0.0) + float(write or 0.0)) * 1024.0


def flops_from_counters(counters, mfma_flops_per_mop_f8=512.0):
    """Measured FLOPs from MFMA counters. None when no usable counter is present.

    `MfmaFlops*` are already FLOP counts. fp8 only exposes an MFMA *ops* count, so it needs the
    per-instruction FLOP factor of the MFMA shape in play -- pass the right one for the kernel.
    """
    if not counters:
        return None
    for key in ("MfmaFlops", "MfmaFlopsBF16", "MfmaFlopsF16", "MfmaFlopsF32"):
        if counters.get(key):
            return float(counters[key])
    mops = counters.get("SQ_INSTS_VALU_MFMA_MOPS_F8")
    if mops:
        return float(mops) * float(mfma_flops_per_mop_f8)
    return None


# ---------------------------------------------------------------- self-test

def _selftest():
    """Reproduces the SKILL.md section 9 worked example from real measured numbers."""
    ok = True

    peaks = load_peaks(SKU_JSON, "gfx950")
    assert peaks == load_peaks(PEAKS_MD, "gfx950"), "peaks.md path must resolve to the same sku.json row"
    assert peaks and abs(peaks["hbm_bw_bytes_s"] - 8.0e12) < 1e9, peaks
    assert abs(peak_flops_for(peaks, "fp8") - 5.0e15) < 1e12, peaks["flops"]
    assert abs(peak_flops_for(peaks, "fp8_e4m3") - 5.0e15) < 1e12
    assert abs(peak_flops_for(peaks, "float8_e4m3fn") - 5.0e15) < 1e12
    assert abs(peak_flops_for(peaks, "half") - 2.5e15) < 1e12
    assert load_peaks(SKU_JSON, "gfxNOPE") is None                              # L1 path
    assert load_peaks(SKU_JSON, "gfx950", product="mi350x")["flops"]["fp16"] == 2.3e15
    print("peaks: gfx950 %.2f TB/s, fp8 %.2f PFLOP/s; unknown gfx -> None  OK"
          % (peaks["hbm_bw_bytes_s"] / 1e12, peak_flops_for(peaks, "fp8") / 1e15))

    E, M, tk, H, I, L = 256, 64, 8, 2048, 512, 40
    hit = experts_hit(E, M * tk)
    assert 215 <= hit <= 225, hit
    wbytes = hit * (2 * I * H + H * I) * dtype_bytes("fp8")
    t_layer = 2 * 49.471e-6                                   # 80 launches/step over 40 layers
    moe = roofline_metrics(wbytes, 2 * M * tk * (2 * I * H + H * I), t_layer,
                           peaks["hbm_bw_bytes_s"], peak_flops_for(peaks, "fp8"),
                           TARGET_EFF["moe"], pct_gpu_time=26.45)
    print("MoE   : hit %.0f/%d, %.0f MB/layer -> %.0f%% of roofline, %s, %.3fx, +%.2f%% e2e (%s)"
          % (hit, E, wbytes / 1e6, 100 * moe["roofline_pct"], moe["bound_type"],
             moe["attainable_speedup"], moe["expected_e2e_gain_pct"], moe["headroom_class"]))
    ok &= (moe["bound_type"] == "memory" and moe["headroom_class"] == "saturated"
           and 0.85 <= moe["roofline_pct"] <= 0.92 and moe["expected_e2e_gain_pct"] < 1.0)

    B, S, kvh, hd = 64, 1024 + 1024 // 2, 2, 256
    attn = roofline_metrics(B * S * kvh * hd * 2 * dtype_bytes("bf16"),
                            2 * B * 16 * S * hd * 2, 141.112e-6,
                            peaks["hbm_bw_bytes_s"], peak_flops_for(peaks, "bf16"),
                            TARGET_EFF["attn"], pct_gpu_time=8.86)
    print("Attn  : %.0f%% of roofline (hbm_util %.0f%%, compute_util %.1f%%), %s, %.2fx, +%.2f%% e2e (%s)"
          % (100 * attn["roofline_pct"], 100 * attn["hbm_util"], 100 * attn["compute_util"],
             attn["bound_type"], attn["attainable_speedup"], attn["expected_e2e_gain_pct"],
             attn["headroom_class"]))
    # low util on both axes above the dispatch floor -> latency/occupancy-bound, but the verdict is
    # KEPT (real headroom) so the ranking inversion below survives.
    ok &= (attn["bound_type"] == "latency" and attn["headroom_class"] == "underperforming"
           and attn["attainable_speedup"] > 1.4
           and attn["expected_e2e_gain_pct"] > moe["expected_e2e_gain_pct"])

    # the whole point: the two rankings disagree, and roofline is the one that matched reality
    ok &= (26.45 > 8.86) and (attn["expected_e2e_gain_pct"] > moe["expected_e2e_gain_pct"])
    print("Rank  : by pct -> MoE first; by expected gain -> Attn first (measured: MoE -0.064%% e2e, "
          "Attn 1.56x isolated)  OK")

    # L3 sanity band: all-expert bytes overshoot the roofline -> clamped + suspect, not discarded
    allx = roofline_metrics(E * (2 * I * H + H * I) * 1, 1.0, t_layer, peaks["hbm_bw_bytes_s"],
                            peak_flops_for(peaks, "fp8"), TARGET_EFF["moe"], pct_gpu_time=26.45)
    assert allx["suspect"] and allx["roofline_pct"] <= 1.0, allx
    print("L3    : all-expert bytes -> raw %.2f clamped to %.2f, suspect=True  OK"
          % (allx["roofline_pct_raw"], allx["roofline_pct"]))

    # degradation: unusable inputs return None instead of raising
    for bad in [(0, 0, 1e-6, 1e12, 1e15, 0.9), (1, 1, 0, 1e12, 1e15, 0.9), (None, None, None, None, None, None)]:
        assert roofline_metrics(*bad) is None, bad
    assert dtype_bytes("who-knows") == 2 and parse_counter_csv("/nonexistent") == {}
    assert bytes_from_counters({}) is None and flops_from_counters({}) is None
    assert bytes_from_counters({"FETCH_SIZE": 1024.0, "WRITE_SIZE": 1024.0}) == 2 * 1024 * 1024
    print("L2/L4/L5: unusable input -> None/{} without raising  OK")

    print("\nSELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="reproduce the SKILL.md worked example")
    ap.add_argument("--peaks", metavar="GFX", help="print the peak table entry for GFX as JSON")
    ap.add_argument("--product", default=None,
                    help="gpu_identity `sku` (mi355x, mi350x, ...) or `target` (r9700) that "
                         "selects a product row")
    a = ap.parse_args()
    if a.peaks:
        p = resolve_peaks(SKU_JSON, a.peaks, product=a.product)
        print(json.dumps(p or {"error": "no calibrated peaks for %s" % a.peaks}, indent=2))
        raise SystemExit(0)
    raise SystemExit(_selftest() if a.selftest else ap.print_help())
