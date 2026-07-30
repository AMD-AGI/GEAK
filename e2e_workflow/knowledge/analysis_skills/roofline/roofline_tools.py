#!/usr/bin/env python3
"""Mechanical primitives for the `roofline` analysis skill.

OPTIONAL helpers only. The skill is defined by SKILL.md and an agent can execute it by hand; this
module exists so the boring parts (peak-table parsing, counter parsing, unit math) are not retyped.

Design rules, per knowledge/analysis_skills/INDEX.md:
  * no hard third-party dependency (stdlib only; torch is imported lazily and optionally)
  * every helper is defensive and returns a value or None -- it never raises at the caller
  * nothing here decides anything; the routing rules live in SKILL.md

Self-test:  python3 roofline_tools.py --selftest
"""
from __future__ import annotations

import json
import math
import os
import re

SKILL_NAME = "roofline"
SKILL_VERSION = "1"

# element size in bytes, keyed by the dtype spellings that show up in profiles/configs
_DTYPE_BYTES = {
    "fp4": 0.5, "float4": 0.5, "mxfp4": 0.5, "e2m1": 0.5,
    "fp8": 1, "float8": 1, "e4m3": 1, "e5m2": 1, "mxfp8": 1, "fp8_e4m3": 1, "int8": 1, "uint8": 1,
    "bf16": 2, "bfloat16": 2, "c10::bfloat16": 2, "fp16": 2, "float16": 2, "half": 2, "int16": 2,
    "fp32": 4, "float32": 4, "float": 4, "int32": 4,
    "fp64": 8, "float64": 8, "double": 8, "int64": 8, "long": 8,
}

# target_eff priors -- SKILL.md section 7 is the source of truth; keep the two in sync.
TARGET_EFF = {
    "gemm": 0.90,
    "moe": 0.90,
    "elementwise": 0.875,
    "attn": 0.50,
}

#: Only kernels big enough for a headroom estimate to change a decision are worth analysing.
#: Below this the Amdahl ceiling is under the noise band anyway, so modelling them adds failure
#: modes without adding information. Callers should pass the run's HEAD_THRESHOLD_PCT.
DEFAULT_MIN_PCT_GPU = 5.0
DEFAULT_TOP_N = 8

#: Kernel launch + scheduling overhead. A launch whose duration is within a small multiple of this
#: is timed by dispatch, not by its transfer or its math, so a roofline ratio is meaningless for it
#: (its lever is fusion / graph capture, not kernel tuning). Order-of-magnitude runtime constant,
#: not a per-model tuning knob -- override per box if measured.
LAUNCH_OVERHEAD_S = 5e-6
LATENCY_BOUND_FACTOR = 2.0

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
    key = str(name).strip().lower().lstrip("torch.")
    if key in _DTYPE_BYTES:
        return _DTYPE_BYTES[key]
    for k, v in _DTYPE_BYTES.items():          # substring fallback: "c10::BFloat16", "torch.float8_e4m3fnuz"
        if k in key:
            return v
    return default


# ---------------------------------------------------------------- peaks

def load_peaks(peaks_md_path, gfx):
    """Parse the ```yaml blocks in peaks.md and return the one matching `gfx`.

    Returns {"hbm_bw_bytes_s": float, "flops": {dtype: float}, "cu": int, "source": "table",
             "confidence": "high"} or None when the file or the section is absent.
    """
    try:
        with open(peaks_md_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError):
        return None

    for block in re.findall(r"```yaml\s*\n(.*?)```", text, re.S):
        if not re.search(r"^\s*gfx:\s*%s\s*$" % re.escape(str(gfx)), block, re.M):
            continue
        out = {"flops": {}, "source": "table", "confidence": "high"}
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
            try:
                out[key] = float(val) if re.match(r"^[-+0-9.eE]+$", val) else val
            except ValueError:
                out[key] = val
        if out.get("hbm_bw_bytes_s"):
            return out
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
    }


def peak_flops_for(peaks, dtype_name):
    """Peak FLOP/s for a dtype, falling back to the largest tabulated value. None if unknown."""
    if not peaks:
        return None
    flops = peaks.get("flops") or {}
    if not flops:
        return None
    key = str(dtype_name or "").strip().lower()
    for k, v in flops.items():
        if k and k in key:
            return float(v)
    return float(max(flops.values()))


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
                     pct_gpu_time=0.0, launch_overhead_s=LAUNCH_OVERHEAD_S):
    """Core arithmetic of SKILL.md section 3 step 5. Returns a dict, or None on unusable input.

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

    compute_bound = bool(ridge is not None and pfl > 0 and ai > ridge)
    if compute_bound:
        raw_pct, bound = achieved_flops / pfl, "compute"
    else:
        raw_pct, bound = achieved_bw / pbw, "memory"

    out = {
        "bytes_est": b, "flops_est": f, "t_ms": t * 1e3,
        "achieved_bw_bytes_s": achieved_bw, "achieved_flops": achieved_flops,
        "arithmetic_intensity": ai, "ridge_point": ridge,
        "roofline_pct_raw": raw_pct, "target_eff": tgt, "suspect": False,
    }

    # A launch timed by dispatch rather than by its own transfer/math: a roofline ratio says nothing
    # about it. Report it as such instead of a bogus efficiency; the lever is fusion, not tuning.
    if t <= float(launch_overhead_s or 0) * LATENCY_BOUND_FACTOR:
        out.update(bound_type="latency", roofline_pct=min(max(raw_pct, 0.0), 1.0),
                   attainable_speedup=1.0, expected_e2e_gain_pct=0.0,
                   headroom_class="unknown",
                   note="per-launch time is within launch-overhead scale -> dispatch-bound; "
                        "roofline not applicable, lever is fusion / graph capture")
        return out

    # L3 sanity band. An impossible ratio means the byte/FLOP model is wrong, NOT that the kernel is
    # at the wall -- so it must not yield a verdict. Clamp for display, refuse to classify, and hand
    # back the feasibility bound that the model violated so stage C knows what to measure.
    if not (0.001 <= raw_pct <= 1.0):
        out.update(bound_type=bound, roofline_pct=min(max(raw_pct, 0.0), 1.0),
                   attainable_speedup=1.0, expected_e2e_gain_pct=0.0,
                   headroom_class="unknown", suspect=True,
                   bytes_upper_bound=pbw * t, flops_upper_bound=(pfl * t) if pfl > 0 else None,
                   note="model infeasible (raw %.3f outside (0,1]) -> NOT a saturation verdict; "
                        "re-estimate with a tighter model or measure with counters (stage C)"
                        % raw_pct)
        return out

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
    if p >= 0.9 * t:
        return "saturated"
    if p >= 0.6 * t:
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
    here = os.path.dirname(os.path.abspath(__file__))
    ok = True

    peaks = load_peaks(os.path.join(here, "peaks.md"), "gfx950")
    assert peaks and abs(peaks["hbm_bw_bytes_s"] - 8.0e12) < 1e9, peaks
    assert abs(peak_flops_for(peaks, "fp8") - 5.0e15) < 1e12, peaks["flops"]
    assert load_peaks(os.path.join(here, "peaks.md"), "gfxNOPE") is None      # L1 path
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
    print("Attn  : %.0f%% of roofline, %s, %.2fx, +%.2f%% e2e (%s)"
          % (100 * attn["roofline_pct"], attn["bound_type"], attn["attainable_speedup"],
             attn["expected_e2e_gain_pct"], attn["headroom_class"]))
    ok &= (attn["headroom_class"] == "underperforming" and attn["attainable_speedup"] > 1.4
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
    a = ap.parse_args()
    if a.peaks:
        p = load_peaks(os.path.join(os.path.dirname(os.path.abspath(__file__)), "peaks.md"), a.peaks)
        print(json.dumps(p or derive_peaks_from_props() or {"error": "no peaks for %s" % a.peaks}, indent=2))
        raise SystemExit(0)
    raise SystemExit(_selftest() if a.selftest else ap.print_help())
