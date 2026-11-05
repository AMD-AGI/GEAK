#!/usr/bin/env python3
"""Backport ROCm commit aa5dfb98 ("MI350 Fix L2 cache to HBM read counters/
metrics", rocm-systems #2501) into a private copy of the installed
rocprofiler-compute 3.4.0 tree.

Why this is needed here: on gfx950 the installed build infers the number of
128B L2->HBM read requests indirectly from TCC_BUBBLE. Measured on this machine
with a 2 GiB memcopy whose traffic is known exactly, TCC_BUBBLE reads 0 on every
dispatch while TCC_EA0_RDREQ_128B reads 16777216 -- i.e. every read request is a
full 128B line but the installed formula charges all of them at 64B. Reported
HBM read bytes come out at exactly 0.500x ground truth; the upstream formula
lands on 1.000x. Reads only; the write terms are untouched by the fix.

The tree is copied rather than patched in place so the container image and any
other session using it are left alone, and so the delta is reproducible.

    python3 rocprof_hbm_patch.py [--src <installed tree>] [--dst <patched copy>]

`--dst` defaults to ./rocprof_compute_hbmfix, and the PATH shim to
./rocprof_hbmfix_bin, both under the CURRENT directory -- never a path baked in
at authoring time, so two campaigns cannot silently share one patched tree.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path


def _default_src() -> Path:
    """The installed rocprofiler-compute tree, found rather than assumed.

    Hard-coding /opt/rocm-7.2.0 breaks on every other ROCm point release, and the
    failure is a confusing "file not found" rather than "wrong ROCm".
    """
    cands = sorted(Path("/opt").glob("rocm*/libexec/rocprofiler-compute"))
    return cands[-1] if cands else Path("/opt/rocm/libexec/rocprofiler-compute")

# (file, buggy pattern, fixed replacement, expected minimum hit count).
# Patterns are matched with all whitespace collapsed to \s+ because these live
# inside YAML plain scalars that the writer hard-wrapped at ~80 columns.
EDITS = [
    ("rocprof_compute_soc/analysis_configs/gfx950/0400_roofline.yaml",
     "(TCC_BUBBLE_sum * 128) + (TCC_EA0_RDREQ_32B_sum * 32) + ((TCC_EA0_RDREQ_sum "
     "- TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum) * 64)",
     "(TCC_EA0_RDREQ_128B_sum * 128) + (TCC_EA0_RDREQ_64B_sum * 64) + "
     "(TCC_EA0_RDREQ_32B_sum * 32)", 2),
    ("rocprof_compute_soc/analysis_configs/gfx950/0200_system_speed_of_light.yaml",
     "128 * TCC_BUBBLE_sum + 64 * (TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum "
     "- TCC_EA0_RDREQ_32B_sum) + 32 * TCC_EA0_RDREQ_32B_sum",
     "128 * TCC_EA0_RDREQ_128B_sum + 64 * TCC_EA0_RDREQ_64B_sum "
     "+ 32 * TCC_EA0_RDREQ_32B_sum", 2),
]

# counter_defs.yaml is NOT arch-scoped per file: FETCH_SIZE lists gfx940/941/942
# /gfx950 under one expression, and every arch's AST is built from the same
# entry. Rewriting it in place makes gfx942 fail at startup with "Unable to
# lookup metric TCC_EA0_RDREQ_128B_sum" (that counter does not exist there), so
# gfx950 has to be split out into its own definition instead.
FETCH_SIZE_OLD = """    - architectures:
      - gfx940
      - gfx941
      - gfx942
      - gfx950
      expression: (TCC_BUBBLE_sum*128 + (TCC_EA0_RDREQ_sum-TCC_BUBBLE_sum-TCC_EA0_RDREQ_32B_sum)*64 + TCC_EA0_RDREQ_32B_sum*32)/1024
"""
FETCH_SIZE_NEW = """    - architectures:
      - gfx950
      expression: (TCC_EA0_RDREQ_128B_sum*128 + TCC_EA0_RDREQ_64B_sum*64 + TCC_EA0_RDREQ_32B_sum*32)/1024
    - architectures:
      - gfx940
      - gfx941
      - gfx942
      expression: (TCC_BUBBLE_sum*128 + (TCC_EA0_RDREQ_sum-TCC_BUBBLE_sum-TCC_EA0_RDREQ_32B_sum)*64 + TCC_EA0_RDREQ_32B_sum*32)/1024
"""

#: Appended to the patched `roofline.py`; the substitution above rewrites both
#: `X.write_image(...)` calls to go through it.
SAFE_WRITE_IMAGE = '''

def _safe_write_image(fig, path, **kw):
    """Write a plotly figure, but never let a missing renderer kill a finished profile.

    Every counter pass is already on disk by the time this runs and the PDF is decorative,
    so a renderer failure here is a warning, not an exit code.
    """
    try:
        fig.write_image(path, **kw)
    except Exception as exc:  # noqa: BLE001 -- any renderer failure, not just Kaleido's
        print(f"[roofline] PDF export skipped ({type(exc).__name__}: {exc}); "
              f"counter data is unaffected", flush=True)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None,
                    help="Installed rocprofiler-compute tree (default: newest /opt/rocm*/libexec).")
    ap.add_argument("--dst", type=Path, default=Path.cwd() / "rocprof_compute_hbmfix",
                    help="Destination for the patched copy (default: ./rocprof_compute_hbmfix).")
    ap.add_argument("--bindir", type=Path, default=None,
                    help="Where to write the PATH shim (default: <dst>/../rocprof_hbmfix_bin).")
    a = ap.parse_args()

    SRC = (a.src or _default_src()).resolve()
    DST = a.dst.resolve()
    BIN = (a.bindir or DST.parent / "rocprof_hbmfix_bin").resolve()
    if not SRC.is_dir():
        sys.exit(f"FAIL: --src is not a directory: {SRC}\n"
                 f"      pass the installed rocprofiler-compute tree explicitly")
    print(f"src -> {SRC}")

    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, symlinks=True)
    for rel, bad, good, want in EDITS:
        p = DST / rel
        txt = p.read_text()
        pat = re.compile(r"\s+".join(re.escape(t) for t in bad.split()))
        txt2, n = pat.subn(good.replace("\\", "\\\\"), txt)
        if n < want:
            sys.exit(f"FAIL {rel}: matched {n}, expected >= {want}")
        p.write_text(txt2)
        print(f"  patched {rel}: {n} site(s)")

    p = DST / "rocprof_compute_soc/profile_configs/counter_defs.yaml"
    txt = p.read_text()
    if txt.count(FETCH_SIZE_OLD) != 1:
        sys.exit(f"FAIL counter_defs.yaml: FETCH_SIZE block matched "
                 f"{txt.count(FETCH_SIZE_OLD)} times, expected 1")
    p.write_text(txt.replace(FETCH_SIZE_OLD, FETCH_SIZE_NEW))
    print("  patched counter_defs.yaml FETCH_SIZE: gfx950 split out, 1 site")

    # roofline_calc.py builds the same read term in Python for the plot points.
    # Its branch is `else` to MI200, i.e. it also serves gfx942, where the new
    # counters do not exist -- so the choice is made on what the dataframe
    # actually carries rather than on arch, which keeps the copy usable
    # everywhere the stock tree was.
    p = DST / "utils/roofline_calc.py"
    txt = p.read_text()
    old = """                # Use TCC_BUBBLE_sum to calculate hbm_data
                hbm_data += (
                    (df["TCC_BUBBLE_sum"][idx] * 128)
                    + (df["TCC_EA0_RDREQ_32B_sum"][idx] * 32)
                    + (
                        (
                            df["TCC_EA0_RDREQ_sum"][idx]
                            - df["TCC_BUBBLE_sum"][idx]
                            - df["TCC_EA0_RDREQ_32B_sum"][idx]
                        )
                        * 64
                    )
"""
    new = """                # Read side: prefer the directly counted size classes
                # (ROCm aa5dfb98); TCC_BUBBLE is only a proxy for the 128B
                # request count and reads 0 on gfx950, which charges every
                # 128B line at 64B and halves the reported read traffic.
                if "TCC_EA0_RDREQ_128B_sum" in df:
                    hbm_read = (
                        (df["TCC_EA0_RDREQ_128B_sum"][idx] * 128)
                        + (df["TCC_EA0_RDREQ_64B_sum"][idx] * 64)
                        + (df["TCC_EA0_RDREQ_32B_sum"][idx] * 32)
                    )
                else:
                    hbm_read = (
                        (df["TCC_BUBBLE_sum"][idx] * 128)
                        + (df["TCC_EA0_RDREQ_32B_sum"][idx] * 32)
                        + (
                            (
                                df["TCC_EA0_RDREQ_sum"][idx]
                                - df["TCC_BUBBLE_sum"][idx]
                                - df["TCC_EA0_RDREQ_32B_sum"][idx]
                            )
                            * 64
                        )
                    )
                hbm_data += (
                    hbm_read
"""
    if txt.count(old) != 1:
        sys.exit(f"FAIL roofline_calc.py: read term matched {txt.count(old)}, expected 1")
    p.write_text(txt.replace(old, new))
    print("  patched utils/roofline_calc.py: 1 site")

    # The PDF export at the end of `profile --roof-only` must not be able to fail the
    # profile. plotly needs Kaleido to rasterise, and the exception escapes
    # post_profiling(), so the WHOLE run exits non-zero AFTER every counter pass has
    # completed and pmc_perf.csv is on disk -- minutes of GPU work discarded for a
    # decorative plot. Installing `kaleido>=1` is not the fix: 1.x renders through
    # headless Chrome (absent in most ROCm images) AND rocprof-compute pins
    # `kaleido==0.2.1` in its own requirements check, so upgrading makes the launcher
    # refuse to start at all. The numbers come from `analyze`, not from the PDF.
    p = DST / "roofline.py"
    txt = p.read_text()
    txt2, n = re.subn(r"(\w+)\.write_image\(\n",
                      r"_safe_write_image(\n                    \1,\n", txt)
    if n < 1:
        sys.exit(f"FAIL roofline.py: no `X.write_image(` call found -- upstream changed "
                 f"shape; re-read it rather than let this patch silently no-op")
    p.write_text(txt2 + SAFE_WRITE_IMAGE)
    print(f"  patched roofline.py write_image: {n} site(s) made non-fatal")

    left = [f for f in DST.rglob("*")
            if f.is_file() and f.suffix in (".yaml", ".py")
            and "gfx950" in str(f) and "TCC_BUBBLE" in f.read_text(errors="ignore")]
    print(f"remaining gfx950 files still mentioning TCC_BUBBLE: "
          f"{[str(f.relative_to(DST)) for f in left]}")
    print(f"patched tree -> {DST}")

    # The PATH shim. The installed `rocprof-compute` launcher resolves its own
    # tree from its location, so the only way to make an unmodified command line
    # pick up the patched copy is to shadow the binary name earlier on PATH.
    BIN.mkdir(parents=True, exist_ok=True)
    shim = BIN / "rocprof-compute"
    shim.write_text(
        "#!/bin/bash\n"
        "# Generated by rocprof_hbm_patch.py -- routes rocprof-compute at the\n"
        "# gfx950 HBM-read-byte-patched tree. Prepend this dir to PATH.\n"
        f'exec python3 "{DST}/rocprof-compute" "$@"\n')
    shim.chmod(0o755)
    entry = DST / "rocprof-compute"
    if not entry.is_file():
        print(f"  WARN: {entry} not found -- inspect the tree and fix the shim by hand")
    print(f"PATH shim  -> {shim}\n"
          f"  export PATH={BIN}:$PATH\n"
          f"  then verify:  command -v rocprof-compute   (must print the shim)")


if __name__ == "__main__":
    main()
