# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preflight the rocprof-compute install before any memory number is believed.

On gfx950 (MI350/MI355X), rocprof-compute builds L2->HBM read bytes by counting 128B
reads via ``TCC_BUBBLE`` and *inferring* 64B reads by subtraction::

    (TCC_BUBBLE_sum * 128) + (TCC_EA0_RDREQ_32B_sum * 32)
      + ((TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum) * 64)

``TCC_BUBBLE`` is not a 128B-read counter on gfx950, so every metric derived from HBM
bytes is wrong: SoL ``L2-Fabric Read BW`` (which the profile_engineer reads as "effective
HBM bandwidth"), and -- wherever the roofline block is collected -- ``HBM Bandwidth`` and
``AI HBM``. Measured against a memcopy of exactly known traffic the stock formula was
**-25.0% on total bytes (-50.0% on reads)**; the corrected formula was exact.

Upstream fixed this by using the explicit per-size counters::

    (TCC_EA0_RDREQ_128B_sum * 128) + (TCC_EA0_RDREQ_32B_sum * 32)
      + (TCC_EA0_RDREQ_64B_sum * 64)

  ROCm/rocm-systems commit aa5dfb98f96ab080de97259df64ba0d36f6796b4, released in
  rocprof-compute 3.5.0 (ROCm 7.12.0) as "Corrected the formula for metrics related to
  reads from L2 cache to HBM for AMD Instinct MI350 Series GPUs".

Images pinned to an older ROCm still carry the bug -- ROCm 7.2.0 ships rocprof-compute
3.4.0 -- and it fails silently: the profiler returns a plausible number that is simply
25% low. This module detects that and, by default, absorbs the upstream fix into the
installed tree so the numbers GEAK reports are the corrected ones.

Version is *reported* but the **content scan is authoritative**: an install may have been
patched in place, and a version new enough to be fixed still gets verified rather than
assumed.

Usage::

    python3 rocprof_compute_env.py --check              # report only; rc=1 if untrustworthy
    python3 rocprof_compute_env.py --apply              # patch when needed (idempotent)
    python3 rocprof_compute_env.py --apply --json r.json

Everything is fail-soft: a probe that cannot run, or a patch that cannot be written (the
tree is root-owned and GEAK may not be root), degrades to a reported status. Profiling
still proceeds -- with the measurement basis recorded, which is the point.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: The upstream fix, for provenance in the emitted report.
UPSTREAM_FIX = {
    "commit": "aa5dfb98f96ab080de97259df64ba0d36f6796b4",
    "repo": "ROCm/rocm-systems",
    "released_in": "rocprof-compute 3.5.0 (ROCm 7.12.0)",
    "summary": (
        "Corrected the formula for metrics related to reads from L2 cache to HBM "
        "for AMD Instinct MI350 Series GPUs"
    ),
}

#: First rocprof-compute release carrying the fix. Advisory only -- see module docstring.
FIXED_FROM_VERSION = (3, 5, 0)

#: Only gfx950 shipped the broken formula; other architectures are left alone.
AFFECTED_ARCH = "gfx950"

# The two spellings of the buggy read-bytes expression, whitespace-normalized. Both
# compute 128B-via-BUBBLE + 32B + 64B-by-subtraction.
_BAD_YAML_A = re.compile(
    r"\(\s*TCC_BUBBLE_sum\s*\*\s*128\s*\)\s*\+\s*"
    r"\(\s*TCC_EA0_RDREQ_32B_sum\s*\*\s*32\s*\)\s*\+\s*"
    r"\(\s*\(\s*TCC_EA0_RDREQ_sum\s*-\s*TCC_BUBBLE_sum\s*-\s*TCC_EA0_RDREQ_32B_sum\s*\)\s*\*\s*64\s*\)"
)
_GOOD_YAML_A = (
    "(TCC_EA0_RDREQ_128B_sum * 128) + (TCC_EA0_RDREQ_32B_sum * 32) "
    "+ (TCC_EA0_RDREQ_64B_sum * 64)"
)

_BAD_YAML_B = re.compile(
    r"128\s*\*\s*TCC_BUBBLE_sum\s*\+\s*"
    r"64\s*\*\s*\(\s*TCC_EA0_RDREQ_sum\s*-\s*TCC_BUBBLE_sum\s*-\s*TCC_EA0_RDREQ_32B_sum\s*\)\s*\+\s*"
    r"32\s*\*\s*TCC_EA0_RDREQ_32B_sum"
)
_GOOD_YAML_B = (
    "128 * TCC_EA0_RDREQ_128B_sum + 64 * TCC_EA0_RDREQ_64B_sum "
    "+ 32 * TCC_EA0_RDREQ_32B_sum"
)

# roofline_calc.py builds the same sum in Python, across several lines.
_BAD_PY = re.compile(
    r"\(df\[\"TCC_BUBBLE_sum\"\]\[idx\]\s*\*\s*128\)\s*\+\s*"
    r"\(df\[\"TCC_EA0_RDREQ_32B_sum\"\]\[idx\]\s*\*\s*32\)\s*\+\s*"
    r"\(\s*\(\s*df\[\"TCC_EA0_RDREQ_sum\"\]\[idx\]\s*"
    r"-\s*df\[\"TCC_BUBBLE_sum\"\]\[idx\]\s*"
    r"-\s*df\[\"TCC_EA0_RDREQ_32B_sum\"\]\[idx\]\s*\)\s*\*\s*64\s*\)",
    re.S,
)
_GOOD_PY = (
    '(df["TCC_EA0_RDREQ_128B_sum"][idx] * 128)\n'
    '                    + (df["TCC_EA0_RDREQ_32B_sum"][idx] * 32)\n'
    '                    + (df["TCC_EA0_RDREQ_64B_sum"][idx] * 64)'
)

# The comment above that accumulator names the counter the code no longer uses; left
# alone it tells the next reader the opposite of what the file now does.
_BAD_PY_COMMENT = re.compile(r"#\s*Use TCC_BUBBLE_sum to calculate hbm_data")
_GOOD_PY_COMMENT = "# Use the explicit per-size RDREQ counters to calculate hbm_data"

#: Marker proving a site carries the corrected formula, for the idempotence check.
_FIXED_MARKER = "TCC_EA0_RDREQ_128B_sum"


def _patch_sites(root: Path, arch: str) -> list[tuple[Path, list, list]]:
    """The files carrying the HBM read-bytes formula.

    Each entry is ``(path, rules, cosmetic)``. Only ``rules`` decide whether a tree is
    buggy -- ``cosmetic`` is tidied up while patching but must never make a correct tree
    read as broken.
    """
    configs = root / "rocprof_compute_soc" / "analysis_configs" / arch
    yaml_rules = [(_BAD_YAML_A, _GOOD_YAML_A), (_BAD_YAML_B, _GOOD_YAML_B)]
    # Only the READ term is wrong. The write term in the same expressions
    # (TCC_EA0_WRREQ_sum / TCC_EA0_WRREQ_64B_sum) is correct and is left untouched, and so
    # is 1800_l2_cache_per_channel.yaml, where TCC_BUBBLE appears in its own right as the
    # per-channel L2 bubble metric rather than as a stand-in for a 128B read count.
    return [
        # 4.1 HBM Bandwidth and 4.2 AI HBM (two occurrences). `--roof-only` derives its
        # counter set by parsing this file, so patching it also makes rocprof collect the
        # 128B/64B counters the corrected formula needs.
        (configs / "0400_roofline.yaml", yaml_rules, []),
        # 2.x SoL L2-Fabric Read BW, value + percent-of-peak -- read even under --no-roof,
        # which is the path kernel_workflow profiling actually takes.
        (configs / "0200_system_speed_of_light.yaml", yaml_rules, []),
        (root / "utils" / "roofline_calc.py", [(_BAD_PY, _GOOD_PY)],
         [(_BAD_PY_COMMENT, _GOOD_PY_COMMENT)]),
    ]


def _run(argv: list[str], timeout: float = 15.0) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (out.stdout or "") + (out.stderr or "")


def detect_gpu_arch() -> str | None:
    """The gfx target of the local accelerator, or None when rocminfo is unavailable."""
    match = re.search(r"\bgfx[0-9a-f]+\b", _run(["rocminfo"]))
    return match.group(0) if match else None


def detect_rocm_version(root: Path | None = None) -> str | None:
    """ROCm version, from the versioned install path or /opt/rocm's version file."""
    if root is not None:
        match = re.search(r"/opt/rocm-([0-9][0-9.]*)", str(root))
        if match:
            return match.group(1)
    for candidate in (Path("/opt/rocm/.info/version"), Path("/opt/rocm/.info/version-dev")):
        try:
            return candidate.read_text(encoding="utf-8").strip().split("-")[0] or None
        except OSError:
            continue
    return None


def locate_install(tool: str = "rocprof-compute") -> Path | None:
    """Resolve the rocprof-compute launcher to the libexec tree that holds its configs."""
    path = shutil.which(tool)
    if path:
        real = Path(path).resolve().parent
        if (real / "rocprof_compute_soc").is_dir():
            return real
    for pattern in ("/opt/rocm-*/libexec/rocprofiler-compute", "/opt/rocm/libexec/rocprofiler-compute"):
        for candidate in sorted(Path("/").glob(pattern.lstrip("/")), reverse=True):
            if (candidate / "rocprof_compute_soc").is_dir():
                return candidate
    return None


def detect_tool_version(root: Path | None, tool: str = "rocprof-compute") -> tuple[str | None, str | None]:
    """Return ``(version, build_commit)``.

    The install tree's VERSION file is tried FIRST: ``rocprof-compute --version`` runs in
    the tool's own interpreter and aborts before printing anything when one of its many
    optional dependencies (plotext, astunparse, dash, ...) is missing -- which is the
    common case in a lean inference image, and would otherwise read as "not installed".
    """
    if root is not None:
        try:
            lines = [ln.strip() for ln in (root / "VERSION").read_text(encoding="utf-8").splitlines()]
            lines = [ln for ln in lines if ln]
            if lines:
                commit = lines[1] if len(lines) > 1 else None
                return lines[0], commit
        except OSError:
            pass
    match = re.search(r"version:?\s*([0-9]+\.[0-9]+\.[0-9]+)", _run([tool, "--version"]), re.I)
    return (match.group(1) if match else None), None


def _version_tuple(version: str | None) -> tuple[int, ...] | None:
    if not version:
        return None
    parts = re.findall(r"\d+", version)
    return tuple(int(p) for p in parts[:3]) if parts else None


def scan(root: Path, arch: str) -> list[dict]:
    """Per-file formula status: ``buggy`` | ``patched`` | ``absent`` | ``unknown``."""
    sites = []
    for path, rules, _cosmetic in _patch_sites(root, arch):
        entry: dict = {"path": str(path)}
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            entry["status"] = "absent"
            sites.append(entry)
            continue
        normalized = " ".join(text.split())
        if any(rx.search(normalized) or rx.search(text) for rx, _ in rules):
            entry["status"] = "buggy"
        elif _FIXED_MARKER in text:
            entry["status"] = "patched"
        else:
            # Neither spelling present: a refactored upstream that computes the bytes
            # some third way. Not something to rewrite blindly.
            entry["status"] = "unknown"
        sites.append(entry)
    return sites


def apply_fix(root: Path, arch: str) -> list[dict]:
    """Rewrite the buggy expression in place. Idempotent; keeps a ``.orig`` backup."""
    results = []
    for path, rules, cosmetic in _patch_sites(root, arch):
        entry: dict = {"path": str(path)}
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            entry.update(status="absent", changed=0)
            results.append(entry)
            continue
        patched, changed = text, 0
        for rx, replacement in rules:
            patched, count = rx.subn(replacement, patched)
            changed += count
        for rx, replacement in cosmetic:
            patched, _ = rx.subn(replacement, patched)
        if patched == text:
            entry.update(status="patched" if _FIXED_MARKER in text else "unknown", changed=0)
            results.append(entry)
            continue
        try:
            backup = path.with_suffix(path.suffix + ".orig")
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(patched, encoding="utf-8")
        except OSError as exc:
            entry.update(status="buggy", changed=0, error=str(exc))
            results.append(entry)
            continue
        entry.update(status="patched", changed=changed)
        results.append(entry)
    return results


def probe(tool: str = "rocprof-compute", arch: str | None = None, root: Path | None = None) -> dict:
    """Describe the install and whether its HBM byte counts can be trusted."""
    root = root or locate_install(tool)
    arch = arch or detect_gpu_arch()
    version, commit = detect_tool_version(root, tool)
    report: dict = {
        "tool": shutil.which(tool),
        "tool_version": version,
        "tool_build": commit,
        "install_root": str(root) if root else None,
        "rocm_version": detect_rocm_version(root),
        "gpu_arch": arch,
        "affected_arch": AFFECTED_ARCH,
        "upstream_fix": UPSTREAM_FIX,
        "action": "none",
        "sites": [],
        "notes": [],
    }
    version_tuple = _version_tuple(version)
    report["version_predicts_fixed"] = (
        None if version_tuple is None else version_tuple >= FIXED_FROM_VERSION
    )

    if root is None:
        report["hbm_byte_formula"] = "unknown"
        report["hbm_bytes_trustworthy"] = None
        report["notes"].append("rocprof-compute install tree not found; nothing to verify")
        return report
    if arch != AFFECTED_ARCH:
        report["hbm_byte_formula"] = "not_applicable"
        report["hbm_bytes_trustworthy"] = True
        report["notes"].append(
            f"arch {arch or 'unknown'} is unaffected; only {AFFECTED_ARCH} shipped the broken formula"
        )
        return report

    report["sites"] = scan(root, arch)
    statuses = {site["status"] for site in report["sites"]}
    if "buggy" in statuses:
        report["hbm_byte_formula"] = "buggy"
        report["hbm_bytes_trustworthy"] = False
    elif statuses <= {"patched", "absent"} and "patched" in statuses:
        report["hbm_byte_formula"] = "patched"
        report["hbm_bytes_trustworthy"] = True
    else:
        report["hbm_byte_formula"] = "unknown"
        report["hbm_bytes_trustworthy"] = None
        report["notes"].append(
            "HBM read-bytes formula matched neither the known-bad nor the known-good shape; "
            "inspect before trusting HBM bandwidth / AI HBM"
        )
    if report["version_predicts_fixed"] is False and report["hbm_byte_formula"] == "patched":
        report["notes"].append(
            f"version {version} predates {'.'.join(map(str, FIXED_FROM_VERSION))} but the tree is "
            "already corrected (patched in place)"
        )
    return report


def ensure(tool: str = "rocprof-compute", arch: str | None = None, root: Path | None = None,
           mode: str = "auto") -> dict:
    """Probe and, when ``mode`` allows, absorb the upstream fix. Never raises."""
    try:
        report = probe(tool, arch, root)
    except Exception as exc:  # pragma: no cover - defensive; a probe must not fail a run
        return {"action": "failed", "error": f"{type(exc).__name__}: {exc}",
                "hbm_bytes_trustworthy": None, "notes": ["probe raised; profiling continues"]}
    if mode == "off" or report.get("hbm_byte_formula") != "buggy":
        report["action"] = "skipped" if mode == "off" else "none"
        return report
    try:
        report["sites"] = apply_fix(Path(report["install_root"]), report["gpu_arch"])
    except Exception as exc:  # pragma: no cover - defensive
        report["action"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    if any(site["status"] == "buggy" for site in report["sites"]):
        report["action"] = "failed"
        report["hbm_bytes_trustworthy"] = False
        report["notes"].append(
            "could not rewrite the formula (the install tree is usually root-owned); "
            "HBM bandwidth / AI HBM from this run are ~25% low -- do not compare them "
            "against numbers taken on a corrected install"
        )
    else:
        report["action"] = "patched"
        report["hbm_byte_formula"] = "patched"
        report["hbm_bytes_trustworthy"] = True
        report["notes"].append("absorbed upstream fix " + UPSTREAM_FIX["commit"][:7])
    return report


def format_report(report: dict) -> str:
    """One compact human/agent-readable block for profile_report.txt."""
    lines = [
        "=== rocprof-compute environment preflight ===",
        f"tool           : {report.get('tool') or 'not found'}",
        f"version        : {report.get('tool_version') or 'unknown'}"
        + (f" (build {report['tool_build']})" if report.get("tool_build") else ""),
        f"install_root   : {report.get('install_root') or 'unknown'}",
        f"rocm_version   : {report.get('rocm_version') or 'unknown'}",
        f"gpu_arch       : {report.get('gpu_arch') or 'unknown'}",
        f"hbm_byte_formula: {report.get('hbm_byte_formula', 'unknown')}  "
        f"(action: {report.get('action', 'none')})",
        f"hbm_bytes_trustworthy: {report.get('hbm_bytes_trustworthy')}",
    ]
    for site in report.get("sites", []):
        lines.append(f"  - {site['status']:<8} {site['path']}")
    for note in report.get("notes", []):
        lines.append(f"  note: {note}")
    if report.get("hbm_bytes_trustworthy") is False:
        lines.append(
            "  !!! HBM bandwidth / AI HBM in this report are UNDER-REPORTED (~25% on total "
            "bytes, ~50% on reads). Classify on other evidence, or re-run on a corrected install."
        )
    lines.append(
        "  upstream       : %s %s -- %s"
        % (UPSTREAM_FIX["repo"], UPSTREAM_FIX["commit"][:7], UPSTREAM_FIX["released_in"])
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tool", default="rocprof-compute", help="profiler launcher to inspect")
    parser.add_argument("--arch", default=None, help="override the detected gfx arch")
    parser.add_argument("--root", default=None, help="override the rocprofiler-compute install root")
    parser.add_argument("--apply", action="store_true", help="absorb the upstream fix when needed")
    parser.add_argument("--check", action="store_true", help="report only (default)")
    parser.add_argument("--json", dest="json_path", default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    mode = "auto" if args.apply and not args.check else "off"
    root = Path(args.root) if args.root else None
    report = ensure(args.tool, args.arch, root, mode=mode) if args.apply else probe(args.tool, args.arch, root)

    print(format_report(report))
    if args.json_path:
        try:
            Path(args.json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"  note: could not write {args.json_path}: {exc}", file=sys.stderr)
    # rc=1 means "the numbers this profiler is about to produce are known-wrong". Callers
    # that must not fail on it (profile_kernel.sh) simply ignore the code.
    return 0 if report.get("hbm_bytes_trustworthy") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
