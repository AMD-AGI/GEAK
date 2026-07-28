#!/usr/bin/env python3
"""Roofline-profile a single GPU kernel from its unit test, using AMD rocprof-compute.

Runnable entry point for the ``roofline`` skill. Self-contained: it shells out
to ``rocprof-compute`` directly and parses the roofline block, so it has no
dependency beyond the Python standard library and a working ROCm install.

Given a kernel and the command that runs its unit test / perf harness, it
collects a roofline with ``rocprof-compute`` and reports where the kernel sits
relative to the GPU's peak compute and peak memory bandwidth.

Modes (``--mode``)
------------------
  roofline (default)  ``profile --roof-only`` + ``analyze -b 4``. One replay
                      pass, roofline counters only. Fast. Prints the parsed
                      AI / HBM-BW / compute summary AND the derived empirical
                      Roofline Efficiency + ridge point (see below).

  full                Raw two-command flow (every counter, no ``-b`` filter),
                      writing a persistent ``--output-format txt`` report. Slow
                      (many replay passes); use for a deep cross-section dive.

Roofline efficiency (roofline mode)
-----------------------------------
For each kernel, computes from the SAME rocprof roofline block:

  Peak_BW_emp   = HBM Bandwidth "Peak (Empirical)"            [GB/s]
  Peak_Compute  = empirical peak of the dtype the kernel actually ran on
                  (the dominant achieved MFMA/VALU FLOP/IOP counter); fp4/fp6
                  (mfma_f6f4) has no empirical peak on some rocprof versions and
                  is estimated as 2x the fp8 empirical peak. Override with
                  --compute-peak <dtype> or --compute-peak-const.          [GFLOP/s]
  AI_HBM        = Arithmetic Intensity vs HBM                 [FLOP/byte]
  Perf          = achieved Performance                        [GFLOP/s]

  attainable    = min(Peak_Compute, AI_HBM * Peak_BW_emp)     [GFLOP/s]
  Roofline Eff. = Perf / attainable
  ridge(emp)    = Peak_Compute / Peak_BW_emp                  [FLOP/byte]
                  AI < ridge -> memory-bound; AI > ridge -> compute-bound

"HBM util (real)" = achieved_HBM_BW / fixed machine-constant peak. It is for
cross-run comparison ONLY and does NOT enter Roofline Eff. (the empirical
per-run HBM peak does). Pass the constant via ``--hbm-peak-const``.

Examples
--------
  python3 run_roofline.py --workdir KDIR \
      --cmd "python3 test_my_kernel.py" --gpu 1

  python3 run_roofline.py --mode full --workdir KDIR --name my_kernel \
      --cmd "python3 test_my_kernel.py" --output ./roofline_out
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

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


# --------------------------------------------------------------------------- #
# rocprof-compute output parsing (roofline block 4)
# --------------------------------------------------------------------------- #
def _num(s: str):
    """Parse a rocprof cell to float, or None for 'N/A'/empty/non-numeric."""
    try:
        return float(s)
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
    for full in top_names:
        if full.startswith(stem):
            return full
    return header_name


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
    return f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"


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


def select_compute_peak(rates: dict, compute_peak: str):
    """Pick the EMPIRICAL compute ceiling from the dtype the kernel actually used.

    Returns ``(peak_gflops_or_None, label)``.

    ``auto`` selects the dtype with the largest *achieved* value and uses that
    dtype's empirical peak. rocprof-compute does not report an empirical peak for
    MFMA F6F4 (fp4/fp6) on some versions, so when the dominant dtype's peak is N/A
    we estimate it from a measured sibling: fp4/fp6 packs ~2x the matrix
    throughput of fp8, so peak(f6f4) ≈ 2 * peak(f8_empirical). The label records
    when the peak is estimated.

    An explicit ``--compute-peak <dtype>`` forces that dtype's empirical peak.
    """
    def peak_of(key):
        m = _rate_get(rates, COMPUTE_METRICS[key])
        return m[1] if m else None

    if compute_peak != "auto":
        return peak_of(compute_peak), compute_peak

    best_key = dominant_dtype(rates)
    if best_key is None:
        return None, "auto:none"

    peak = peak_of(best_key)
    if peak is not None:
        return peak, best_key
    # dominant dtype has no empirical peak (typically mfma_f6f4) -> estimate.
    if best_key == "mfma_f6f4":
        f8_peak = peak_of("mfma_f8")
        if f8_peak is not None:
            return 2.0 * f8_peak, "mfma_f6f4,est=2xF8emp"
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


def _eff_line(ai_hbm, perf, peak_bw, peak_compute):
    """(ridge, attainable, eff, bound) from the roofline inputs, or Nones."""
    if None in (ai_hbm, perf, peak_bw, peak_compute) or peak_bw == 0:
        return None, None, None, None
    bw_bound = ai_hbm * peak_bw
    attainable = min(peak_compute, bw_bound)
    ridge = peak_compute / peak_bw
    eff = perf / attainable if attainable > 0 else None
    bound = "memory-bound" if ai_hbm < ridge else "compute-bound"
    return ridge, attainable, eff, bound


def compute_roofline_efficiency(names, rates_list, ai_list, compute_peak: str,
                                hbm_peak_const: float | None,
                                compute_peak_const: float | None = None,
                                peaks: str = "empirical", arch: str = "") -> str:
    """Derive empirical Roofline Eff. + ridge point per kernel from the parsed block.

    ``peaks`` controls which ceilings drive the efficiency:
      * ``empirical`` (default): per-run peaks parsed from the roofline block.
      * ``datasheet``: fixed vendor peaks from DATASHEET_PEAKS[arch].
      * ``both``: print the empirical block AND a datasheet cross-check per kernel.
    """
    _hdr = {"empirical": "empirical, per-run peaks",
            "datasheet": "datasheet vendor peaks",
            "both": "empirical + datasheet cross-check"}.get(peaks, peaks)
    out = ["", f"=== Roofline efficiency ({_hdr}) ==="]
    if not names:
        out.append("No kernel with a roofline block was found.")
        return "\n".join(out)

    for name, rates, ai in zip(names, rates_list, ai_list):
        out.append(f"\nkernel: {name}")

        hbm = _rate_get(rates, "HBM Bandwidth")
        peak_bw = hbm[1] if hbm else None
        achieved_bw = hbm[0] if hbm else None

        # Compute ceiling from the dtype the kernel actually used (see
        # select_compute_peak); an explicit --compute-peak-const overrides it.
        if compute_peak_const is not None:
            peak_compute, kind = compute_peak_const, "const"
        else:
            peak_compute, kind = select_compute_peak(rates, compute_peak)

        ai_hbm = (_rate_get(ai, "AI HBM") or (None,))[0]
        perf_t = (_rate_get(ai, "Performance (TFLOPs)") or (None,))[0]
        perf = perf_t * 1000.0 if perf_t is not None else None

        out.append(f"  AI (HBM)            : {_f(ai_hbm)} FLOP/byte")
        out.append(f"  Perf (achieved)     : {_f(perf)} GFLOP/s")
        out.append(f"  Peak_BW_emp (HBM)   : {_f(peak_bw)} GB/s")
        out.append(f"  Peak_Compute [{kind}]: {_f(peak_compute)} GFLOP/s")

        # int8 kernels do integer MFMA ops (MFMA IOPs); rocprof's FLOP-based
        # Performance then reads ~0, so the derived Roofline Eff. is meaningless.
        if dominant_dtype(rates) == "mfma_int8":
            out.append("  NOTE: int8 (MFMA IOPs) kernel — Perf is FLOP-based (≈0), so "
                       "Roofline Eff. below is NOT meaningful; read the achieved IOP "
                       "rate / HBM BW instead.")

        if peaks in ("empirical", "both"):
            ridge, attainable, eff, bound = _eff_line(ai_hbm, perf, peak_bw, peak_compute)
            if ridge is None:
                out.append("  -> insufficient data for efficiency (missing peak/AI/perf)")
            else:
                bw_bound = ai_hbm * peak_bw
                out.append(f"  ridge(emp)          : {_f(ridge)} FLOP/byte  ({bound})")
                out.append(f"  attainable          : {_f(attainable)} GFLOP/s  "
                           f"(min of compute {_f(peak_compute)} , BW*AI {_f(bw_bound)})")
                out.append(f"  >> Roofline Eff.    : {_f(eff)}  "
                           f"({_f(eff * 100) if eff is not None else 'N/A'} %)")

        # Datasheet cross-check (fixed vendor peaks) — the conservative number.
        if peaks in ("datasheet", "both"):
            ds_compute, ds_hbm, ds_label = select_datasheet_peak(rates, arch)
            ridge, attainable, eff, bound = _eff_line(ai_hbm, perf, ds_hbm, ds_compute)
            out.append(f"  Peak_BW_ds (HBM)    : {_f(ds_hbm)} GB/s")
            out.append(f"  Peak_Compute_ds [{ds_label}]: {_f(ds_compute)} GFLOP/s")
            if ridge is None:
                out.append(f"  -> datasheet Eff. N/A (no datasheet peak for arch {arch!r})")
            else:
                out.append(f"  ridge(ds)           : {_f(ridge)} FLOP/byte  ({bound})")
                out.append(f"  >> Roofline Eff.(ds): {_f(eff)}  "
                           f"({_f(eff * 100) if eff is not None else 'N/A'} %)")

        if hbm_peak_const and achieved_bw is not None:
            out.append(f"  HBM util (real)     : {_f(achieved_bw / hbm_peak_const * 100)} %"
                       f"  [vs fixed {hbm_peak_const} GB/s, cross-run only]")
        else:
            out.append("  HBM util (real)     : (pass --hbm-peak-const for cross-run util)")

    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _guard_arch(arch: str) -> str | None:
    if arch and is_rdna(arch):
        return (f"rocprof-compute roofline does not support RDNA ({arch}). "
                "Use a CDNA / MI-series GPU.")
    return None


def resolve_kernel_ids(profile_path: Path, workdir, kernel: str) -> tuple[list[int], str]:
    """Map a ``--kernel <substr>`` to rocprof ``-k`` id(s) via ``--list-stats``.

    Returns ``(ids, error)``. ``error`` is non-empty (and ids empty) if the
    list-stats call failed or no kernel name contained ``kernel`` (case-sensitive
    substring); the message then lists the available kernel names.
    """
    ls_cmd = f"rocprof-compute analyze -p {profile_path} --list-stats"
    r = subprocess.run(ls_cmd, shell=True, cwd=workdir, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        return [], (r.stdout.strip() or r.stderr.strip() or "list-stats failed")
    stats = parse_list_stats(r.stdout)
    ids = [idx for idx, kname in stats if kernel in kname]
    if not ids:
        avail = "\n".join(f"  [{idx}] {kname}" for idx, kname in stats)
        return [], (f"--kernel {kernel!r} matched no kernel. Available kernels:\n{avail}")
    return ids, ""


def run_roofline_mode(workdir, cmd, out_dir, name, compute_peak, hbm_peak_const,
                      compute_peak_const=None, skip_noise=False, kernel=None,
                      peaks="both") -> int:
    arch = detect_gpu_arch()
    err = _guard_arch(arch)
    if err:
        print(f"[roofline] {err}")
        (out_dir / f"{name}_roofline_error.txt").write_text(err)
        return 1

    version = rocprof_compute_version()
    if version is None:
        msg = ("rocprof-compute is not installed / not on PATH. Install it:\n"
               "  apt install rocprofiler-compute\n"
               "  update-alternatives --install /usr/bin/rocprof-compute "
               "rocprof-compute /opt/rocm/bin/rocprof-compute 0\n"
               "  pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt")
        print(f"[roofline] {msg}")
        (out_dir / f"{name}_roofline_error.txt").write_text(msg)
        return 1

    # rocprof-compute < 3.3.1 has no --roof-only fast path; collect a full profile.
    roof_only = "" if _version_lt(version, "3.3.1") else "--roof-only"
    profile_path = Path(tempfile.mkdtemp(prefix="rocprof_")).resolve()

    print(f"[roofline] workdir={workdir}\n[roofline] cmd={cmd!r}\n[roofline] rocprof-compute {version}")
    t0 = time.time()
    try:
        prof_cmd = f"rocprof-compute profile -n {name} --path {profile_path} {roof_only} -- {cmd}".strip()
        r = subprocess.run(prof_cmd, shell=True, cwd=workdir, capture_output=True, text=True, timeout=3600 * 6)
        if r.returncode != 0:
            elapsed = time.time() - t0
            error = r.stdout.strip() or r.stderr.strip()
            print(f"[roofline] profile FAILED in {elapsed:.1f}s\n{error}")
            (out_dir / f"{name}_roofline_error.txt").write_text(error)
            return 1

        # Optionally target only the kernel(s) whose name contains --kernel, so
        # a 1-iter --profile run's setup/RNG kernels don't bury the kernel under
        # test. Resolve substr -> rocprof -k id(s) via --list-stats.
        kernel_flag = ""
        if kernel:
            ids, kerr = resolve_kernel_ids(profile_path, workdir, kernel)
            if kerr:
                print(f"[roofline] {kerr}")
                (out_dir / f"{name}_roofline_error.txt").write_text(kerr)
                return 1
            print(f"[roofline] --kernel {kernel!r} -> -k {' '.join(map(str, ids))}")
            kernel_flag = " -k " + " ".join(str(i) for i in ids)

        ana_cmd = f"rocprof-compute analyze -p {profile_path} -b 4{kernel_flag}"
        r = subprocess.run(ana_cmd, shell=True, cwd=workdir, capture_output=True, text=True, timeout=3600 * 6)
        elapsed = time.time() - t0
        if r.returncode != 0:
            error = r.stdout.strip() or r.stderr.strip()
            print(f"[roofline] analyze FAILED in {elapsed:.1f}s\n{error}")
            (out_dir / f"{name}_roofline_error.txt").write_text(error)
            return 1

        content = r.stdout
        names, rates_list, ai_list = skip_filtered_rows(profile_path, content, skip_noise)
        summary = roofline_summary(names, rates_list, ai_list)
        eff = compute_roofline_efficiency(names, rates_list, ai_list, compute_peak,
                                          hbm_peak_const, compute_peak_const,
                                          peaks, arch)
        report = summary + "\n" + eff
        (out_dir / f"{name}_roofline.txt").write_text(report)
        (out_dir / f"{name}_roofline_raw.txt").write_text(content)
    finally:
        shutil.rmtree(profile_path, ignore_errors=True)

    print(f"[roofline] SUCCESS in {elapsed:.1f}s")
    print("=" * 70)
    print(report)
    print("=" * 70)
    print(f"[roofline] summary -> {out_dir / (name + '_roofline.txt')}")
    print(f"[roofline] raw     -> {out_dir / (name + '_roofline_raw.txt')}")
    return 0


def run_full_mode(workdir, cmd, out_dir, name) -> int:
    """Raw full profile + persistent txt report (every counter, all blocks)."""
    err = _guard_arch(detect_gpu_arch())
    if err:
        print(f"[full] {err}")
        return 1
    path = out_dir / name
    cmds = [
        ("profile", f"rocprof-compute profile -n {name} --path {path} -- {cmd}"),
        ("analyze", f"rocprof-compute analyze -p {path} --output-format txt --output-name {name}_report"),
    ]
    for label, c in cmds:
        print(f"[full:{label}] {c}")
        r = subprocess.run(c, shell=True, cwd=workdir, timeout=3600 * 6)
        if r.returncode != 0:
            print(f"[full:{label}] FAILED rc={r.returncode}")
            return r.returncode
    print(f"[full] report -> {path}/{name}_report.txt")
    return 0


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
                         "the empirical peak of the dtype the kernel actually ran on (dominant "
                         "achieved FLOP/IOP counter); fp4/fp6 (mfma_f6f4) with no empirical peak is "
                         "estimated as 2x the fp8 empirical peak. Or force a specific dtype.")
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
                    help="Drop framework input-generation/init noise kernels (RNG fill, memset, arange; "
                         "see NOISE_PATTERNS). Default: report every kernel, filtering nothing by name — "
                         "so a rocBLAS Cijk_ / at::native kernel that IS the kernel under test is kept.")
    ap.add_argument("--peaks", choices=["empirical", "datasheet", "both"], default="both",
                    help="Which peaks drive Roofline Eff.: 'both' (default) prints the empirical block AND "
                         "a datasheet cross-check per kernel; 'empirical' = per-run peaks from the roofline "
                         "block only; 'datasheet' = fixed vendor peaks from DATASHEET_PEAKS[arch] only. "
                         "Datasheet peaks run above empirical, so datasheet Eff. is the conservative "
                         "(lower) number; when the arch/dtype has no datasheet peak the cross-check is N/A.")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.workdir).resolve().name

    if args.mode == "roofline":
        return run_roofline_mode(args.workdir, args.cmd, out_dir, name, args.compute_peak,
                                 args.hbm_peak_const, args.compute_peak_const, args.skip_noise,
                                 args.kernel, args.peaks)
    return run_full_mode(args.workdir, args.cmd, out_dir, name)


if __name__ == "__main__":
    raise SystemExit(main())
