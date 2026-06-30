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
For each (non-noise) kernel, computes from the SAME rocprof roofline block:

  Peak_BW_emp   = HBM Bandwidth "Peak (Empirical)"            [GB/s]
  Peak_Compute  = MFMA BF16 peak (attention/GEMM) OR
                  VALU F32 peak (elementwise)                 [GFLOP/s]
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
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Top kernels matching these substrings are framework/setup noise (RNG fill,
# elementwise copies, etc.), not the kernel under test. They are skipped in the
# roofline summary and efficiency report.
SKIP_PATTERNS = [
    "vectorized_elementwise",
    "distribution_",
    "reduce_kernel",
    "fillBuffer",
    "copyBuffer",
    "Cijk_",
    "at::native",
]


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
def parse_top_kernels(profile_path: Path) -> list[str]:
    """Hottest-first kernel names from ``pmc_kernel_top.csv``."""
    try:
        with open(profile_path / "pmc_kernel_top.csv", newline="") as fh:
            return [row["Kernel_Name"] for row in csv.DictReader(fh)]
    except (OSError, KeyError):
        print("Warning: could not read pmc_kernel_top.csv")
        return []


def parse_roofline_rates(content: str) -> list[dict[str, tuple[float, float, str]]]:
    """Parse every ``4.1 Roofline Rate Metrics`` block: metric -> (value, peak, unit)."""
    kernel_rates: list[dict[str, tuple[float, float, str]]] = []
    rates: dict[str, tuple[float, float, str]] = {}
    in_section = False
    for line in content.split("\n"):
        if "4.1 Roofline Rate Metrics" in line:
            in_section, rates = True, {}
            continue
        if in_section and "╘═" in line:
            in_section = False
            if rates:
                kernel_rates.append(rates)
            continue
        if "4.3 Roofline Plot" in line:
            break
        if in_section and "│" in line and "4.1." in line:
            parts = [p.strip() for p in line.split("│")]
            if len(parts) >= 6:
                try:
                    rates[parts[2]] = (float(parts[3]), float(parts[5]), parts[4])
                except (ValueError, IndexError):
                    continue
    if in_section and rates:
        kernel_rates.append(rates)
    return kernel_rates


def parse_roofline_ai(content: str) -> list[dict[str, tuple[float, str]]]:
    """Parse every ``4.2 Roofline AI Plot Points`` block: metric -> (value, unit)."""
    kernel_metrics: list[dict[str, tuple[float, str]]] = []
    ai_metrics: dict[str, tuple[float, str]] = {}
    in_section = False
    for line in content.split("\n"):
        if "4.2 Roofline AI Plot Points" in line:
            in_section, ai_metrics = True, {}
            continue
        if in_section and "╘═" in line:
            in_section = False
            if ai_metrics:
                kernel_metrics.append(ai_metrics)
            continue
        if "4.3 Roofline Plot" in line:
            break
        if in_section and "│" in line and "4.2." in line:
            parts = [p.strip() for p in line.split("│")]
            if len(parts) >= 5:
                try:
                    metric_name, value, unit = parts[2], float(parts[3]), parts[4]
                except (ValueError, IndexError):
                    continue
                if "AI" in metric_name or "Performance" in metric_name:
                    if "Performance" in metric_name and "Gflop" in unit:
                        value, unit, metric_name = value / 1000.0, "TFLOPS", "Performance (TFLOPs)"
                    ai_metrics[metric_name] = (value, unit)
    if in_section and ai_metrics:
        kernel_metrics.append(ai_metrics)
    return kernel_metrics


def skip_filtered_rows(profile_path: Path, content: str):
    """``zip(top_kernels, rates, ai)`` rows whose kernel name is not noise."""
    top = parse_top_kernels(profile_path)
    rates = parse_roofline_rates(content)
    ai = parse_roofline_ai(content)
    names, rates_out, ai_out = [], [], []
    for k, r, a in zip(top, rates, ai):
        if any(p in k for p in SKIP_PATTERNS):
            continue
        names.append(k)
        rates_out.append(r)
        ai_out.append(a)
    return names, rates_out, ai_out


def roofline_summary(names, rates_list, ai_list) -> str:
    """Human-readable per-kernel HBM/compute utilization + arithmetic intensity."""
    out = ["\nBelow is the roofline information of the kernel:"]
    for kernel, rates, ai in zip(names, rates_list, ai_list):
        bw, compute = {}, {}
        for metric, (value, peak, unit) in rates.items():
            util = (value / peak * 100) if peak > 0 else 0
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


def compute_roofline_efficiency(names, rates_list, ai_list, compute_peak: str,
                                hbm_peak_const: float | None) -> str:
    """Derive empirical Roofline Eff. + ridge point per kernel from the parsed block."""
    out = ["", "=== Roofline efficiency (empirical, per-run peaks) ==="]
    if not names:
        out.append("No non-noise kernel with a roofline block was found.")
        return "\n".join(out)

    for name, rates, ai in zip(names, rates_list, ai_list):
        out.append(f"\nkernel: {name}")

        hbm = rates.get("HBM Bandwidth")
        peak_bw = hbm[1] if hbm else None
        achieved_bw = hbm[0] if hbm else None

        mfma_bf16 = rates.get("MFMA FLOPs (BF16)")
        valu_f32 = rates.get("VALU FLOPs (F32)")
        mfma_bf16_peak = mfma_bf16[1] if mfma_bf16 else None
        valu_f32_peak = valu_f32[1] if valu_f32 else None

        kind = compute_peak
        if kind == "auto":
            mfma_ach = sum(v[0] for k, v in rates.items() if k.startswith("MFMA") and v[0] is not None)
            valu_ach = sum(v[0] for k, v in rates.items() if k.startswith("VALU") and v[0] is not None)
            kind = "mfma_bf16" if (mfma_ach > 0 and mfma_ach >= valu_ach) else "valu_f32"
        peak_compute = mfma_bf16_peak if kind == "mfma_bf16" else valu_f32_peak

        ai_hbm = ai.get("AI HBM", (None,))[0]
        perf_t = ai.get("Performance (TFLOPs)", (None,))[0]
        perf = perf_t * 1000.0 if perf_t is not None else None

        out.append(f"  AI (HBM)            : {_f(ai_hbm)} FLOP/byte")
        out.append(f"  Perf (achieved)     : {_f(perf)} GFLOP/s")
        out.append(f"  Peak_BW_emp (HBM)   : {_f(peak_bw)} GB/s")
        out.append(f"  Peak_Compute [{kind}]: {_f(peak_compute)} GFLOP/s")

        if None in (ai_hbm, perf, peak_bw, peak_compute) or peak_bw == 0:
            out.append("  -> insufficient data for efficiency (missing peak/AI/perf)")
            continue

        bw_bound = ai_hbm * peak_bw
        attainable = min(peak_compute, bw_bound)
        ridge = peak_compute / peak_bw
        eff = perf / attainable if attainable > 0 else None
        bound = "memory-bound" if ai_hbm < ridge else "compute-bound"

        out.append(f"  ridge(emp)          : {_f(ridge)} FLOP/byte  ({bound})")
        out.append(f"  attainable          : {_f(attainable)} GFLOP/s  "
                   f"(min of compute {_f(peak_compute)} , BW*AI {_f(bw_bound)})")
        out.append(f"  >> Roofline Eff.    : {_f(eff)}  ({_f(eff * 100) if eff is not None else 'N/A'} %)")

        if hbm_peak_const and achieved_bw is not None:
            out.append(f"  HBM util (real)     : {_f(achieved_bw / hbm_peak_const * 100)} %"
                       f"  [vs fixed {hbm_peak_const} GB/s, cross-run only]")
        else:
            out.append("  HBM util (real)     : (pass --hbm-peak-const for cross-run util)")

    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _guard_arch() -> str | None:
    arch = detect_gpu_arch()
    if arch and is_rdna(arch):
        return (f"rocprof-compute roofline does not support RDNA ({arch}). "
                "Use a CDNA / MI-series GPU.")
    return None


def run_roofline_mode(workdir, cmd, out_dir, name, compute_peak, hbm_peak_const) -> int:
    err = _guard_arch()
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

        ana_cmd = f"rocprof-compute analyze -p {profile_path} -b 4"
        r = subprocess.run(ana_cmd, shell=True, cwd=workdir, capture_output=True, text=True, timeout=3600 * 6)
        elapsed = time.time() - t0
        if r.returncode != 0:
            error = r.stdout.strip() or r.stderr.strip()
            print(f"[roofline] analyze FAILED in {elapsed:.1f}s\n{error}")
            (out_dir / f"{name}_roofline_error.txt").write_text(error)
            return 1

        content = r.stdout
        names, rates_list, ai_list = skip_filtered_rows(profile_path, content)
        summary = roofline_summary(names, rates_list, ai_list)
        eff = compute_roofline_efficiency(names, rates_list, ai_list, compute_peak, hbm_peak_const)
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
    err = _guard_arch()
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
    ap.add_argument("--compute-peak", choices=["auto", "mfma_bf16", "valu_f32"], default="auto",
                    help="Which empirical peak to use as the compute ceiling (roofline mode).")
    ap.add_argument("--hbm-peak-const", type=float, default=None,
                    help="Fixed machine HBM peak [GB/s] for cross-run 'HBM util (real)' (NOT used in Roofline Eff.).")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.workdir).resolve().name

    if args.mode == "roofline":
        return run_roofline_mode(args.workdir, args.cmd, out_dir, name, args.compute_peak, args.hbm_peak_const)
    return run_full_mode(args.workdir, args.cmd, out_dir, name)


if __name__ == "__main__":
    raise SystemExit(main())
