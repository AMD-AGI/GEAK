#!/usr/bin/env python3
"""Profile one case with rocprof-compute and analyze only measured dispatch windows.

Use case_runner_template.py (or a runner with the same markers) and an existing
ROOFLINE_MIBENCH_CACHE. Setup/autotune/warmup and conditioning stay outside the
selected windows. Raw profiles, logs, numeric results and provenance are retained.
This counter engine is independent of TL-native models and reports.

--mode roofline collects roof-only counters; --mode full collects all counters.
--scope whole-process is explicitly diagnostic. No peak is silently substituted.
"""
import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counter_metrics import score, case_rows
from peak_audit import audit_rocprof_peaks
from rocprof_window import select_dispatch_windows, save_window_audit

# Genuine framework *input-generation / init* noise: RNG fill, buffer memset,
# arange, etc. — NOT compute kernels. These are only dropped when the caller
# passes --skip-noise. By default NO kernel is filtered by name.
#
# IMPORTANT: rocBLAS ("Cijk_") and PyTorch "at::native" / "vectorized_elementwise"
# kernels are deliberately NOT in this list. In many unit tests they ARE the
# kernel under test (a rocBLAS GEMM, an at::native elementwise copy), so filtering
# them by name would silently drop the very kernel you want to roofline.
NOISE_PATTERNS = [
    "distribution_",                  # torch.randn / rand input generation
    "randperm",                       # torch.randperm setup
    "random_",                        # in-place RNG fill
    "fillBuffer",                     # HIP buffer memset
    "copyBuffer",                     # HIP buffer copy (H2D/D2D setup)
    "FillFunctor",                    # torch fill_ / zeros_/ ones_
    "elementwise_kernel_with_index",  # torch.arange
]

# Metric key (CLI-friendly) -> rocprof-compute roofline rate-metric name.
# The order also defines the display order for the achieved-dtype scan.
COMPUTE_METRICS = {
    "valu_f16":  "VALU FLOPs (F16)",
    "valu_f32":  "VALU FLOPs (F32)",
    "valu_f64":  "VALU FLOPs (F64)",
    "mfma_f64":  "MFMA FLOPs (F64)",
    "mfma_f32":  "MFMA FLOPs (F32)",
    "mfma_f16":  "MFMA FLOPs (F16)",
    "mfma_bf16": "MFMA FLOPs (BF16)",
    "mfma_f8":   "MFMA FLOPs (F8)",
    "mfma_f6f4": "MFMA FLOPs (F6F4)",
    "mfma_int8": "MFMA IOPs (Int8)",
}

# Vendor *datasheet* peaks per arch (dense, no sparsity) for a fixed-peak roofline
# cross-check, complementing the per-run EMPIRICAL peaks parsed from the roofline
# block. Empirical peaks come from rocprof's microbenchmark and run well below
# datasheet (the microbench does not fully saturate the units), so datasheet
# efficiency is the more conservative (lower) number. Values in GFLOP/s (compute)
# and GB/s (HBM). Add other CDNA archs as needed.
DATASHEET_PEAKS = {
    "gfx950": {  # MI350 / MI355X class
        "hbm": 8000.0,          # 8 TB/s
        "compute": {
            "mfma_f8":   5_000_000.0,   # FP8  5 PFLOP/s
            "mfma_f6f4": 10_000_000.0,  # FP4/FP6 10 PFLOP/s (2x FP8)
            "mfma_bf16": 2_500_000.0,   # BF16 2.5 PFLOP/s
            "mfma_f16":  2_500_000.0,   # FP16 2.5 PFLOP/s
        },
    },
}

# How far below an equal-rate sibling an empirical peak may sit before it is
# treated as a bad microbenchmark rather than a real ceiling (see
# sanity_check_peak). 0.90 passes ordinary run-to-run scatter -- rocprof's own
# HBM microbench moved 5% between runs on one GPU in one day -- and catches the
# failure actually observed, which was a clean factor of 2.
PEAK_SIBLING_TOL = 0.90


# --------------------------------------------------------------------------- #
# rocprof-compute / GPU environment helpers
# --------------------------------------------------------------------------- #
def detect_gpu_arch() -> str:
    """Return the GFX architecture string (e.g. 'gfx942') or '' on failure."""
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if "gfx" in line.lower() and "name:" in line.lower():
                for p in line.split():
                    if p.startswith("gfx"):
                        return p
    except Exception:
        pass
    return ""


def is_rdna(arch: str) -> bool:
    """rocprof-compute roofline does not support RDNA (gfx10xx/11xx/12xx)."""
    return arch.startswith(("gfx10", "gfx11", "gfx12"))


def rocprof_compute_version() -> str | None:
    """Return the installed rocprof-compute version string, or None if absent."""
    r = subprocess.run("rocprof-compute --version", capture_output=True, text=True, shell=True)
    if r.returncode != 0:
        return None
    m = re.search(r"rocprofiler-compute\s+version:\s*([0-9]+\.[0-9]+\.[0-9]+)", r.stdout)
    return m.group(1) if m else None


def _version_lt(a: str, b: str) -> bool:
    """True if version string ``a`` < ``b`` (numeric dotted compare)."""
    def parts(v):
        return [int(x) for x in re.findall(r"\d+", v)]
    return parts(a) < parts(b)


# Architectures whose MFMA empirical peaks are wrong before rocprof-compute 3.6.0.
# The microbenchmark issues the CDNA3 instructions v_mfma_f32_32x32x8_{f16,bf16}
# and scores them at 16384 FLOP/iter; CDNA4 has v_mfma_f32_32x32x16_{f16,bf16} at
# 32768, so every fp16/bf16/int8 empirical peak comes out ~2-4x low and the
# Roofline Eff. built on it comes out that much too high. Fixed in 3.6.0
# (ROCm 7.13.0): "Fixed roofline benchmark MFMA FP16/BF16/INT8 peaks for MI 350".
MICROBENCH_FIXED_VERSION = "3.6.0"
MICROBENCH_BROKEN_ARCHS = ("gfx95",)


def _stale_microbench_warning(version: str | None, arch: str) -> str:
    """Warn when this rocprof-compute's microbenchmark mis-measures MFMA peaks.

    Returns "" when the installed version is new enough, the arch is unaffected,
    or the peaks are being supplied from a cache (ROOFLINE_MIBENCH_CACHE) rather
    than measured here -- in the cache case the ceilings came from whatever
    produced the cache, so this version's bug is irrelevant.
    """
    if not version or not arch:
        return ""
    if not arch.startswith(MICROBENCH_BROKEN_ARCHS):
        return ""
    if not _version_lt(version, MICROBENCH_FIXED_VERSION):
        return ""
    if Path(os.environ.get("ROOFLINE_MIBENCH_CACHE", "")).is_file():
        return ""
    return (
        f"[roofline] WARNING: rocprof-compute {version} < {MICROBENCH_FIXED_VERSION} on {arch}.\n"
        f"[roofline]   Its roofline microbenchmark measures gfx95x MFMA with CDNA3 instructions,\n"
        f"[roofline]   so the empirical MFMA FP16 / BF16 / INT8 peaks read ~2-4x LOW (gfx950:\n"
        f"[roofline]   bf16 614663 vs the correct 2435517 GFLOP/s) and every Roofline Eff. built\n"
        f"[roofline]   on them reads that much too HIGH -- >100% rows are the giveaway.\n"
        f"[roofline]   Fix, in order of preference:\n"
        f"[roofline]     1. Upgrade to rocprof-compute >= {MICROBENCH_FIXED_VERSION} (ROCm >= 7.13.0).\n"
        f"[roofline]     2. Keep this version's counters and take only the CEILINGS from a newer\n"
        f"[roofline]        microbench: run it standalone (it needs only libamdhip64 + libhiprtc\n"
        f"[roofline]        from 3.5.0 on, NOT rocprofiler-sdk) and point ROOFLINE_MIBENCH_CACHE\n"
        f"[roofline]        at the roofline.csv it writes. Verified equivalent to re-profiling:\n"
        f"[roofline]        the counters are untouched, only the peak column moves.\n"
        f"[roofline]     3. Read the datasheet column / --compute-peak-const and ignore empirical.\n"
        f"[roofline]   Do NOT publish the empirical numbers from this version as headroom."
    )


# --------------------------------------------------------------------------- #
# rocprof-compute output parsing (roofline block 4)
# --------------------------------------------------------------------------- #
def _num(s: str):
    """Parse a rocprof cell to float, or None for 'N/A'/empty/non-numeric."""
    try:
        value = float(s)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def parse_top_kernels(profile_path: Path) -> list[str]:
    """Hottest-first kernel names from ``pmc_kernel_top.csv`` (optional fallback)."""
    try:
        with open(profile_path / "pmc_kernel_top.csv", newline="") as fh:
            return [row["Kernel_Name"] for row in csv.DictReader(fh)]
    except (OSError, KeyError):
        return []


# ``Kernel 3: void at::native::...gpu_kernel... (6.0%)`` -> name is the text
# between ``Kernel N: `` and the trailing ``(pct%)``. Long names are truncated
# by rocprof with a ``...`` suffix.
_KERNEL_HEADER_RE = re.compile(r"^Kernel\s+\d+:\s*(.*?)\s*\(\d+(?:\.\d+)?%\)\s*$")


def parse_kernel_blocks(content: str) -> list[tuple[str, dict, dict]]:
    """Segment ``analyze -b 4`` output by kernel; parse each kernel's 4.1/4.2 blocks.

    Returns one ``(name, rates, ai)`` per kernel, in order, where
    ``rates`` maps metric -> ``(value, peak_or_None, unit)`` and
    ``ai`` maps metric -> ``(value, unit)``.

    Segmenting per kernel (using the ``Kernel N: <name> (pct%)`` headers as
    delimiters) is what keeps names, rates and AI aligned. The previous design
    parsed rates and AI in two independent global passes and zipped them by
    position, which silently misaligns whenever any single 4.1 or 4.2 block is
    empty (one list gets a shorter length than the other). It also relied on a
    global ``break`` at the first ``4.3 Roofline Plot`` line, which dropped every
    kernel after the first when 4.3 is emitted per-kernel. Both are avoided here:
    each row is attributed to the current kernel/table, and 4.3 rows simply match
    no active table state.
    """
    kernels: list[list] = []          # each entry: [name, rates, ai]
    cur: list | None = None
    state: str | None = None          # None | "rates" | "ai"

    def _start(name: str) -> None:
        nonlocal cur
        cur = [name, {}, {}]
        kernels.append(cur)

    for line in content.split("\n"):
        m = _KERNEL_HEADER_RE.match(line.strip())
        if m:
            _start(m.group(1).strip())
            state = None
            continue
        if "4.1 Roofline Rate Metrics" in line:
            if cur is None:                       # blocks before any header
                _start(f"kernel_{len(kernels)}")
            state = "rates"
            continue
        if "4.2 Roofline AI Plot Points" in line:
            if cur is None:
                _start(f"kernel_{len(kernels)}")
            state = "ai"
            continue
        if "╘═" in line:                          # end of the current table
            state = None
            continue
        if cur is None or state is None or "│" not in line:
            continue
        parts = [p.strip() for p in line.split("│")]
        if state == "rates" and "4.1." in line and len(parts) >= 6:
            # Keep the row whenever the achieved value parses; the peak may be
            # "N/A" (e.g. F6F4/fp4 has no empirical peak on some rocprof
            # versions) -> store peak as None instead of dropping the row.
            val = _num(parts[3])
            if val is not None:
                cur[1][parts[2]] = (val, _num(parts[5]), parts[4])
        elif state == "ai" and "4.2." in line and len(parts) >= 5:
            metric_name, value, unit = parts[2], _num(parts[3]), parts[4]
            if value is None or not ("AI" in metric_name or "Performance" in metric_name):
                continue
            if "Performance" in metric_name:
                # Normalize achieved compute to a canonical TFLOP/s key regardless
                # of the reported prefix (rocprof usually emits Gflop/s, but be
                # robust to M/G/T/Pflop/s) so downstream can always read
                # "Performance (TFLOPs)".
                scale = {"m": 1e-6, "g": 1e-3, "t": 1.0, "p": 1e3}.get(
                    unit.strip()[:1].lower(), 1e-3)
                value, unit, metric_name = value * scale, "TFLOPS", "Performance (TFLOPs)"
            cur[2][metric_name] = (value, unit)

    return [(name, rates, ai) for name, rates, ai in kernels]


def _recover_full_name(header_name: str, top_names: list[str]) -> str:
    """Recover a fuller (untruncated) name for a truncated header name.

    Header names are truncated with a trailing ``...``; ``pmc_kernel_top.csv``
    (``top_names``) holds the full names. Prefix-match the truncated stem against
    the CSV names and return the full one; if nothing matches (or the header name
    is not truncated) keep the header name as-is.
    """
    if not header_name.endswith("...") or not top_names:
        return header_name
    stem = header_name[:-3]
    matches = sorted(set(full for full in top_names if full.startswith(stem)))
    if len(matches)>1:
        raise ValueError('ambiguous truncated kernel name: '+header_name)
    return matches[0] if matches else header_name


def parse_list_stats(content: str) -> list[tuple[int, str]]:
    """Parse ``analyze --list-stats`` 'Detected Kernels' table -> [(index, name)].

    The table rows look like ``│   0 │ _fused_mla_decode_kernel ...│``. Only the
    first table ('Detected Kernels', keyed on ``index``) is used; ``-k`` filters
    on that index. Parsing stops at the following 'Dispatch list' table.
    """
    out: list[tuple[int, str]] = []
    in_section = False
    for line in content.split("\n"):
        if "Detected Kernels" in line:
            in_section = True
            continue
        if in_section and "Dispatch list" in line:
            break
        if in_section and "│" in line:
            parts = [p.strip() for p in line.split("│")]
            if len(parts) >= 4 and parts[1].isdigit():
                out.append((int(parts[1]), parts[2]))
    return out


def skip_filtered_rows(profile_path: Path, content: str, skip_noise: bool = False):
    """Per-kernel ``(names, rates, ai)`` lists, aligned by construction.

    Kernel names come from ``parse_kernel_blocks`` (the analyze ``Kernel N: <name>
    (pct%)`` headers, which own the 4.1/4.2 blocks below them);
    ``pmc_kernel_top.csv`` is only an optional fallback to un-truncate a long
    header name. Because the three lists are produced from the same per-kernel
    segmentation they are always the same length and 1:1 — no kernel is silently
    dropped, and none can shift onto another kernel's data.

    By default NO kernel is filtered by name (the tool must not silently drop a
    kernel the user actually wants to roofline — e.g. a rocBLAS ``Cijk_`` GEMM or
    an ``at::native`` copy that is the kernel under test). When ``skip_noise`` is
    set, only genuine input-generation / init noise (``NOISE_PATTERNS``) is
    dropped.
    """
    top = parse_top_kernels(profile_path)
    names, rates_out, ai_out = [], [], []
    for name, rates, ai in parse_kernel_blocks(content):
        name = _recover_full_name(name, top)
        if skip_noise and any(p in name for p in NOISE_PATTERNS):
            continue
        names.append(name)
        rates_out.append(rates)
        ai_out.append(ai)
    return names, rates_out, ai_out


def roofline_summary(names, rates_list, ai_list) -> str:
    """Human-readable per-kernel HBM/compute utilization + arithmetic intensity."""
    out = ["\nBelow is the roofline information of the kernel:"]
    for kernel, rates, ai in zip(names, rates_list, ai_list):
        bw, compute = {}, {}
        for metric, (value, peak, unit) in rates.items():
            util = (value / peak * 100) if (peak and peak > 0) else 0
            if "HBM" in metric and "Bandwidth" in metric:
                bw[metric] = (value, peak, unit, util)
            elif ("FLOPs" in metric or "IOPs" in metric) and value > 0:
                compute[metric] = (value, peak, unit, util)
        out.append("\nkernel function name:")
        out.append(f"- {kernel}")
        if bw:
            out.append("HBM BANDWIDTH UTILIZATION:")
            for m, (v, p, u, util) in bw.items():
                out.append(f"- {m}: actual: {v} peak: {p} utilization_pct: {util}")
        if compute:
            out.append("COMPUTE UTILIZATION:")
            for m, (v, p, u, util) in compute.items():
                out.append(f"- {m}: actual: {v} peak: {p} utilization_pct: {util}")
        if ai:
            out.append("ARITHMETIC INTENSITY:")
            for m, (v, u) in ai.items():
                out.append(f"- {m}: value: {v} {u}" if u else f"- {m}: value: {v}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Empirical roofline efficiency
# --------------------------------------------------------------------------- #
def _f(x):
    return f"{x:.8g}" if isinstance(x, (int, float)) else "N/A"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _rate_get(rates: dict, wanted: str):
    """Tolerant metric lookup: exact -> whitespace/case-normalized -> substring.

    rocprof-compute metric labels are matched by exact string elsewhere, which
    breaks silently if a version tweaks the label (e.g. ``HBM Bandwidth`` vs
    ``HBM Bandwidth (GB/s)``): the parsed summary uses substring matching and
    still shows the row, while an exact ``dict.get`` in the efficiency path
    returns None and reports "insufficient data". This normalizes the lookup so
    both paths agree.
    """
    if wanted in rates:
        return rates[wanted]
    w = _norm(wanted)
    for k, v in rates.items():
        if _norm(k) == w:
            return v
    for k, v in rates.items():
        if w in _norm(k):
            return v
    return None


def dominant_dtype(rates: dict):
    """Return the dtype key (COMPUTE_METRICS) with the largest achieved value.

    That is the dtype the kernel actually ran on; None if no FLOP/IOP achieved.
    """
    best_key, best_ach = None, 0.0
    for key, metric in COMPUTE_METRICS.items():
        m = _rate_get(rates, metric)
        if m and m[0] is not None and m[0] > best_ach:
            best_key, best_ach = key, m[0]
    return best_key


def equal_rate_siblings(key: str, arch: str) -> list:
    """dtypes that MUST have the same peak as ``key`` on ``arch``, per datasheet.

    The datasheet table is the authority: if DATASHEET_PEAKS[arch] gives two MFMA
    dtypes the same peak, the hardware runs them at the same rate, so any
    disagreement between their *empirical* peaks is a microbenchmark artefact
    rather than a property of the silicon. Deriving the rule from the table
    instead of hard-coding "bf16 == fp16" means it extends to whatever arch is
    added next, and stays silent on an arch with no datasheet entry.
    """
    table = (DATASHEET_PEAKS.get(arch) or {}).get("compute") or {}
    mine = table.get(key)
    if mine is None:
        return []
    return [k for k, v in table.items() if k != key and v == mine]


def sanity_check_peak(rates, key, peak, arch, peak_of):
    """Flag implausible same-/scaled-rate references without inventing a peak."""
    if peak is None:
        return None, key + ':peakN/A'
    table = (DATASHEET_PEAKS.get(arch) or {}).get('compute', {})
    expected = []
    if table.get(key):
        for sibling, ds in table.items():
            value = peak_of(sibling)
            if sibling != key and value and ds:
                expected.append((value * table[key] / ds, sibling))
    if expected:
        largest, sibling = max(expected)
        if peak < PEAK_SIBLING_TOL * largest:
            return peak, f'{key},suspect=scaled_{sibling}'
    return peak, key


def select_compute_peak(rates: dict, compute_peak: str, arch: str = ""):
    """Select the requested/dominant resource peak, leaving absent peaks absent."""
    def peak_of(key):
        m = _rate_get(rates, COMPUTE_METRICS[key])
        return m[1] if m else None

    if compute_peak != "auto":
        # Forced dtype is sanity-checked too: the peak is wrong or right
        # independently of how the dtype was chosen.
        return sanity_check_peak(rates, compute_peak, peak_of(compute_peak),
                                 arch, peak_of)

    best_key = dominant_dtype(rates)
    if best_key is None:
        return None, "auto:none"

    peak = peak_of(best_key)
    if peak is not None:
        return sanity_check_peak(rates, best_key, peak, arch, peak_of)
    return None, f"{best_key}:peakN/A"


def select_datasheet_peak(rates: dict, arch: str):
    """Pick the DATASHEET compute ceiling for the kernel's dominant dtype.

    Returns ``(peak_gflops_or_None, hbm_gbps_or_None, label)`` from
    DATASHEET_PEAKS[arch]. Used for the fixed-peak roofline cross-check.
    """
    table = DATASHEET_PEAKS.get(arch)
    if not table:
        return None, None, f"{arch or 'unknown-arch'}:no-datasheet"
    hbm = table.get("hbm")
    key = dominant_dtype(rates)
    if key is None:
        return None, hbm, "datasheet:none"
    peak = table.get("compute", {}).get(key)
    return peak, hbm, (key if peak is not None else f"{key}:no-datasheet")


def _eff_line(ai_hbm, perf, peak_bw, peak_compute, achieved_bw=None):
    record = score(ai_hbm, perf, achieved_bw, peak_bw, peak_compute)
    return record['ridge'], record['attainable_gflops'], record['efficiency'], record['bound']


def efficiency_records(names, rates_list, ai_list, compute_peak='auto', compute_peak_const=None, peaks='both', arch=''):
    records = []
    for name, rates, ai in zip(names, rates_list, ai_list):
        hbm = _rate_get(rates, 'HBM Bandwidth') or (None, None, None)
        perf_t = (_rate_get(ai, 'Performance (TFLOPs)') or (None,))[0]
        perf = perf_t * 1000 if perf_t is not None else None
        ai_display = (_rate_get(ai, 'AI HBM') or (None,))[0]
        peak, kind = ((compute_peak_const, 'explicit_constant') if compute_peak_const is not None
                      else select_compute_peak(rates, compute_peak, arch))
        refs = []
        if peaks in ('empirical', 'both'):
            refs.append(('empirical', peak, hbm[1], kind))
        if peaks in ('datasheet', 'both'):
            ds_peak, ds_bw, ds_kind = select_datasheet_peak(rates, arch)
            refs.append(('datasheet', ds_peak, ds_bw, ds_kind))
        record = {'kernel': name, 'rates': rates, 'ai_report': ai, 'references': {}}
        for ref, p, bw, label in refs:
            value = score(ai_display, perf, hbm[0], bw, p, dominant_dtype(rates) == 'mfma_int8')
            value['compute_spec'] = label
            if ',suspect=' in label:
                value['peak_warning'] = 'empirical_peak_inconsistent_with_scaled_sibling; verify calibration'
            record['references'][ref] = value
        records.append(record)
    return records


def compute_roofline_efficiency(names, rates_list, ai_list, compute_peak, hbm_peak_const,
                                compute_peak_const=None, peaks='empirical', arch=''):
    records = efficiency_records(names, rates_list, ai_list, compute_peak, compute_peak_const, peaks, arch)
    out = ['', f'=== Roofline efficiency ({peaks}) ===']
    for row in records:
        out.append('\nkernel: ' + row['kernel'])
        for ref, value in row['references'].items():
            out.extend([
                f"  reference           : {ref}",
                f"  AI (HBM, displayed) : {_f(value['ai_display'])}",
                f"  AI (HBM, derived)   : {_f(value['ai_derived'])}",
                f"  Perf (achieved)     : {_f(value['achieved_gflops'])} GFLOP/s",
                f"  Peak_BW ({ref})     : {_f(value['peak_bw_gbps'])} GB/s",
                f"  Peak_Compute [{value['compute_spec']}]: {_f(value['peak_compute_gflops'])} GFLOP/s",
                f"  bound               : {value['bound']}",
                f"  basis               : {value['basis']}",
                f"  >> Roofline Eff.{('(ds)' if ref == 'datasheet' else '    ')}: {_f(value['efficiency'])}",
                f"  HBM utilization     : {_f(value['memory_efficiency'])}",
            ])
            if value['reason']:
                out.append('  reason              : ' + value['reason'])
            if value.get('peak_warning'):
                out.append('  WARN                : ' + value['peak_warning'])
        hbm = _rate_get(row['rates'], 'HBM Bandwidth')
        if hbm_peak_const and hbm and hbm[0] is not None:
            out.append(f"  HBM util (real)     : {_f(hbm[0] / hbm_peak_const * 100)} % [diagnostic constant only]")
    return '\n'.join(out)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _guard_arch(arch: str) -> str | None:
    if arch and is_rdna(arch):
        return (f"rocprof-compute roofline does not support RDNA ({arch}). "
                "Use a CDNA / MI-series GPU.")
    return None


def _both_streams(r) -> str:
    """Everything a failed subprocess said, stdout AND stderr, each labelled.

    `stdout or stderr` is the obvious idiom and it is wrong here. rocprof-compute narrates
    several hundred INFO lines to stdout on every run, so stdout is ALWAYS non-empty and the
    `or` therefore always discards stderr -- which is the only place the Python traceback that
    actually explains the failure appears. Observed cost: a seven-minute profile whose three
    counter passes had all succeeded failed in the PDF-export step, and the saved error file
    ended at an innocuous "Checking for roofline.csv" line with no indication of why.
    """
    parts = []
    for label, text in (("stdout", r.stdout), ("stderr", r.stderr)):
        text = (text or "").strip()
        if text:
            parts.append(f"===== {label} =====\n{text}")
    return "\n\n".join(parts) or f"(no output; returncode={r.returncode})"


def resolve_kernel_ids(profile_path: Path, workdir, kernel: str) -> tuple[list[int], str]:
    """Map a ``--kernel <substr>`` to rocprof ``-k`` id(s) via ``--list-stats``.

    Returns ``(ids, error)``. ``error`` is non-empty (and ids empty) if the
    list-stats call failed or no kernel name contained ``kernel`` (case-sensitive
    substring); the message then lists the available kernel names.
    """
    ls_cmd = ["rocprof-compute", "analyze", "-p", str(profile_path), "--list-stats"]
    r = subprocess.run(ls_cmd, cwd=workdir, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        return [], _both_streams(r)
    stats = parse_list_stats(r.stdout)
    ids = [idx for idx, kname in stats if kernel in kname]
    if not ids:
        avail = "\n".join(f"  [{idx}] {kname}" for idx, kname in stats)
        return [], (f"--kernel {kernel!r} matched no kernel. Available kernels:\n{avail}")
    return ids, ""


def run_roofline_mode(workdir, cmd, out_dir, name, compute_peak, hbm_peak_const,
                      compute_peak_const=None, skip_noise=False, kernel=None,
                      peaks='both', scope='measured-window', expected_calls=None,
                      allow_unverified_peaks=False, full=False):
    arch = detect_gpu_arch()
    version = rocprof_compute_version()
    if _guard_arch(arch) or version is None:
        raise ValueError(_guard_arch(arch) or 'rocprof-compute is not installed')
    if skip_noise:
        raise ValueError('--skip-noise name filtering is retired; select measured windows')
    if Path(name).name != name or name in ('.','..'):
        raise ValueError('run name must be a single path component')
    if (out_dir / f'{name}_roofline.txt').exists():
        raise ValueError('report already exists; use a fresh name/output directory')
    cache_text = os.environ.get('ROOFLINE_MIBENCH_CACHE', '').strip()
    cache = Path(cache_text).resolve() if cache_text else None
    if cache is None or not cache.is_file():
        raise ValueError('supply an existing frozen ROOFLINE_MIBENCH_CACHE; per-case calibration is disabled')
    peak_info = audit_rocprof_peaks(cache, arch)
    if peak_info['warnings'] and not allow_unverified_peaks:
        raise ValueError('CALIBRATION_SUSPECT: ' + '; '.join(peak_info['warnings']) +
                         '; validate the references or explicitly use --allow-unverified-peaks for diagnosis')
    profile_path = Path(tempfile.mkdtemp(prefix=name + '_profile_', dir=out_dir)).resolve()
    manifest_path = profile_path / 'measurement_manifest.json'
    env = dict(os.environ, KRF_MEASUREMENT_MANIFEST=str(manifest_path))
    shutil.copyfile(cache, profile_path / 'roofline.csv')
    # rocprof-compute 3.4 joins the workload argv into a shell string. A
    # `bash -c <one argument>` loses its boundary and runs just the first word.
    # Use a simple relative script name from the profile directory instead.
    workload = profile_path / 'workload.sh'
    workload.write_text('#!/bin/bash\nset -e\ncd -- ' + shlex.quote(str(Path(workdir).resolve())) + '\n' + cmd + '\n')
    metadata = {'tool':'rocprof-compute','version':version,'arch':arch,'scope':scope,
                'command':cmd,'workdir':str(workdir),'peak_reference':peak_info,
                'profile_path':str(profile_path),'workload_script':str(workload),'status':'STARTED'}
    t0 = time.time()
    try:
        # 3.4 otherwise deletes pmc_perf_<pass>.csv after joining counters.
        # Retain them so every pass's marker/ID sequence can be checked.
        command = ['rocprof-compute','profile','-V','-n',name,'--path',str(profile_path)]
        if not full and not _version_lt(version, '3.3.1'):
            command.append('--roof-only')
        command += ['--','bash',workload.name]
        metadata['profile_command'] = command
        result = subprocess.run(command,cwd=profile_path,env=env,capture_output=True,text=True,timeout=21600)
        (out_dir / f'{name}_profile.log').write_text(_both_streams(result))
        if result.returncode:
            raise RuntimeError('profile failed: ' + _both_streams(result))
        if hashlib.sha256((profile_path / 'roofline.csv').read_bytes()).hexdigest() != peak_info['sha256']:
            raise ValueError('FROZEN_PEAK_CHANGED: profiler did not preserve the supplied peak file')
        metadata['frozen_peak_reused'] = True
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        metadata['measurement'] = manifest
        selected_ids = []
        audit = None
        if scope == 'measured-window':
            calls = expected_calls or (manifest or {}).get('measured_calls')
            if not calls:
                raise ValueError('missing expected calls; use the bundled runner or --expected-calls')
            if manifest and manifest.get('status') != 'OK':
                raise ValueError('runner manifest did not complete successfully')
            audit = select_dispatch_windows(profile_path, calls, (manifest or {}).get('expected_dispatches'),
                       (manifest or {}).get('cache',{}).get('conditioning_kernel'))
            save_window_audit(audit, out_dir / f'{name}_dispatches.json')
            selected_ids = audit['selected_ids']
            metadata['window'] = {k:v for k,v in audit.items() if k != 'passes'}
        else:
            metadata['warning'] = 'whole process includes setup/warmup; not a production score'
        command = ['rocprof-compute','analyze','-p',str(profile_path)]
        if not full:
            command += ['-b','4']
        if selected_ids:
            command += ['-d'] + [str(i) for i in selected_ids]
        if kernel:
            ids, error = resolve_kernel_ids(profile_path,workdir,kernel)
            if error:
                raise ValueError(error)
            command += ['-k'] + [str(i) for i in ids]
            metadata['kernel_filter'] = kernel
        metadata['analyze_command'] = command
        result = subprocess.run(command,cwd=workdir,capture_output=True,text=True,timeout=21600)
        (out_dir / f'{name}_analyze.log').write_text(_both_streams(result))
        if result.returncode:
            raise RuntimeError('analysis failed: ' + _both_streams(result))
        content = result.stdout
        (out_dir / f'{name}_roofline_raw.txt').write_text(content)
        names,rates,ais = skip_filtered_rows(profile_path,content,False)
        if audit:
            names=[_recover_full_name(n,list(audit['per_call_dispatches'])) for n in names]
        if not names:
            raise ValueError('no roofline blocks in analyze output')
        records = efficiency_records(names,rates,ais,compute_peak,compute_peak_const,peaks,arch)
        report = roofline_summary(names,rates,ais) + '\n' + compute_roofline_efficiency(
            names,rates,ais,compute_peak,hbm_peak_const,compute_peak_const,peaks,arch)
        if scope != 'measured-window' or peak_info['warnings']:
            report = 'DIAGNOSTIC_ONLY: unverified scope or reference\n' + report
        (out_dir / f'{name}_roofline.txt').write_text(report)
        rows_by_reference=case_rows(records,manifest,audit)
        for value in rows_by_reference.values():
            value['measurement_contract'].update({'peak_file_sha256':peak_info['sha256'],'compute_peak_policy':compute_peak,'compute_peak_constant':compute_peak_const})
        (out_dir / f'{name}_roofline.json').write_text(json.dumps({'kernels':records,'scope':scope,
            'rows_by_reference':rows_by_reference,'rows':[rows_by_reference['empirical']] if 'empirical' in rows_by_reference else []},indent=2,allow_nan=False)+'\n')
        metadata['status'] = 'OK'
        print(f'[roofline] SUCCESS in {time.time()-t0:.1f}s; raw profile retained at {profile_path}')
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        metadata['status'] = 'FAILED'
        metadata['error'] = str(error)
        (out_dir / f'{name}_roofline_error.txt').write_text(str(error))
        print(f'[roofline] FAILED: {error}',file=sys.stderr)
        return 1
    finally:
        metadata['elapsed_seconds'] = time.time()-t0
        (out_dir / f'{name}_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, help="Directory to run the unit test from (cwd for rocprof).")
    ap.add_argument("--cmd", required=True, help="Unit-test / performance command that launches the kernel.")
    ap.add_argument("--mode", choices=["roofline", "full"], default="roofline")
    ap.add_argument("--name", default=None, help="Profiling run name (default: basename of --workdir).")
    ap.add_argument("--output", default="./roofline_out", help="Directory for results.")
    ap.add_argument("--gpu", default=None, help="GPU index -> HIP_VISIBLE_DEVICES (profile one device).")
    ap.add_argument("--compute-peak", choices=["auto"] + list(COMPUTE_METRICS), default="auto",
                    help="Compute ceiling for Roofline Eff. (roofline mode). 'auto' (default) uses "
                         "the empirical peak for the dominant achieved resource. Missing peaks "
                         "remain unavailable; no automatic sibling substitution.")
    ap.add_argument("--compute-peak-const", type=float, default=None,
                    help="Fixed compute ceiling [GFLOP/s] overriding --compute-peak (e.g. a datasheet "
                         "fp4/fp8 MFMA peak). Use when the empirical peak is missing or unreliable.")
    ap.add_argument("--hbm-peak-const", type=float, default=None,
                    help="Fixed machine HBM peak [GB/s] for cross-run 'HBM util (real)' (NOT used in Roofline Eff.).")
    ap.add_argument("--kernel", default=None,
                    help="Roofline only the kernel(s) whose name contains this (case-sensitive) "
                         "substring. Resolves via 'analyze --list-stats' to rocprof -k id(s). Use "
                         "this when setup/RNG kernels dominate a 1-iteration --profile run and bury "
                         "the kernel under test. Default: analyze all kernels.")
    ap.add_argument("--skip-noise", action="store_true",
                    help="Retired: use measured-window selection instead of name filtering.")
    ap.add_argument("--peaks", choices=["empirical", "datasheet", "both"], default="both",
                    help="Which peaks drive Roofline Eff.: 'both' (default) prints the empirical block AND "
                         "a datasheet cross-check per kernel; 'empirical' = per-run peaks from the roofline "
                         "block only; 'datasheet' = fixed vendor peaks from DATASHEET_PEAKS[arch] only. "
                         "Datasheet peaks run above empirical, so datasheet Eff. is the conservative "
                         "(lower) number; when the arch/dtype has no datasheet peak the cross-check is N/A.")
    ap.add_argument('--scope', choices=['measured-window','whole-process'], default='measured-window')
    ap.add_argument('--expected-calls', type=int, help='Required if a custom runner does not write a manifest')
    ap.add_argument('--allow-unverified-peaks', action='store_true', help='Diagnostic output only; never changes supplied peaks')
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.workdir).resolve().name

    try:
        return run_roofline_mode(args.workdir,args.cmd,out_dir,name,args.compute_peak,
            args.hbm_peak_const,args.compute_peak_const,args.skip_noise,args.kernel,args.peaks,
            args.scope,args.expected_calls,args.allow_unverified_peaks,full=args.mode == 'full')
    except (ValueError,OSError) as error:
        (out_dir / f'{name}_roofline_error.txt').write_text(str(error))
        print(str(error),file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
