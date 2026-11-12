#!/usr/bin/env python3
"""Checks a fusion region's accepted overlay on disk before the e2e workflow banks it.

    python3 fusion_overlay_check.py --overlay <accepted overlay dir> --initial <the overlay it was built on> \
        --root <EVAL_DIR>

Prints {"ok", "problems", "manifest_sha256", "files"} as one JSON line and exits 0 (ok) or 1. The workflow
script cannot read files, so when an integrate A/B accepts a fusion region (KernelFusion switched on) it
runs this through one agent and banks the region only on "ok": true.

1. it is a directory inside the run dir with a sitecustomize.py;
2. every .py in it compiles (compiled only, never imported or run here);
3. every absolute path it names under the run's parent dir (the fused kernels it loads) exists;
4. existing optimizations are kept: every file of the initial overlay is present and byte-identical,
   and the initial sitecustomize.py is either kept as is or copied under another name that the new
   sitecustomize.py imports;
5. it is the overlay the A/B measured: an integrate_result.json accepted exactly this directory with a
   complete A/B, and no code in it (_CODE_SUFFIXES) changed after that A/B's candidate leg finished. The
   A/B's own records (e2e_validation.json, manifests.json, ...) land here after the A/B and are not
   overlay content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Benchmark / correctness output the integrator writes next to the overlay sources; not part of the overlay.
_OVERLAY_OUTPUT_DIRS = {"__pycache__", "cand", "ref", "correctness"}
# The integrator's A/B records, written into the candidate dir during and after the A/B
# (roles/e2e_integrator.md: CB=$EVAL_DIR/overlay/cand_<short>). Records, not overlay content.
_AB_RECORDS = {"integrate_result.json", "e2e_validation.json", "manifests.json"}
# What the server imports or loads. Only these must be unchanged since the A/B's candidate leg: the
# integrator may still write records next to them.
_CODE_SUFFIXES = (".py", ".so", ".hip", ".cu", ".cpp", ".cc", ".h", ".hpp", ".hsaco", ".co")
_ABS_PATH = re.compile(r"""["'](/[^"'\s]+)["']""")


def _read_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _overlay_files(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _OVERLAY_OUTPUT_DIRS]
        for f in filenames:
            if f.endswith((".pyc", ".log", ".out")) or f in _AB_RECORDS:
                continue
            p = Path(dirpath) / f
            out[str(p.relative_to(root))] = p
    return out


def _ab_record(overlay: Path, run_dir: Path) -> tuple[dict | None, Path | None]:
    """The integrate_result.json whose A/B accepted exactly this overlay directory: the overlay's own
    record first, then any other under the run dir."""
    own = overlay / "integrate_result.json"
    for p in ([own] if own.is_file() else []) + sorted(run_dir.rglob("integrate_result.json")):
        d = _read_json(p) or {}
        acc = d.get("accepted_overlay")
        if acc and Path(acc).resolve() == overlay:
            return d, p
    return None, None


def validate_overlay(kf_overlay: str, initial_overlay: str, run_dir: Path) -> dict:
    """The checks in the module docstring. Returns {"ok", "problems", "manifest_sha256", "files"}."""
    problems: list[str] = []
    run_dir = Path(run_dir).resolve()
    overlay = Path(kf_overlay).resolve() if kf_overlay else Path()
    if not kf_overlay or not overlay.is_dir():
        return {"ok": False, "problems": [f"overlay {kf_overlay!r} is not a directory"]}
    if run_dir not in overlay.parents:
        problems.append(f"overlay {overlay} is outside the run dir {run_dir}")
    files = _overlay_files(overlay)
    site = overlay / "sitecustomize.py"
    if not site.is_file():
        problems.append("no sitecustomize.py at the overlay root")
    for rel, p in sorted(files.items()):
        if not rel.endswith(".py"):
            continue
        try:
            src = p.read_text(encoding="utf-8")
            compile(src, str(p), "exec")
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as e:
            problems.append(f"{rel} does not compile: {type(e).__name__}: {e}"[:300])
            continue
        for ref in _ABS_PATH.findall(src):
            if ref.startswith(str(run_dir.parent) + os.sep) and not Path(ref).exists():
                problems.append(f"{rel} names {ref}, which does not exist")
    site_src = site.read_text(encoding="utf-8", errors="replace") if site.is_file() else ""
    for root in [Path(x) for x in str(initial_overlay or "").split(os.pathsep) if x.strip()]:
        if not root.is_dir():
            continue
        for rel, p in sorted(_overlay_files(root).items()):
            want = p.read_bytes()
            if rel == "sitecustomize.py":
                if site.is_file() and site.read_bytes() == want:
                    continue
                kept = [r for r, q in files.items() if r != "sitecustomize.py" and q.read_bytes() == want]
                imported = [r for r in kept if re.search(
                    r"\b" + re.escape(Path(r).with_suffix("").as_posix().replace("/", ".")) + r"\b", site_src)]
                if not kept:
                    problems.append(f"the initial overlay's sitecustomize.py ({root}) is not kept")
                elif not imported:
                    problems.append(f"the initial overlay's sitecustomize.py is kept as {kept[0]} "
                                    "but the new sitecustomize.py does not import it")
            elif rel not in files:
                problems.append(f"initial overlay file {rel} ({root}) is missing")
            elif files[rel].read_bytes() != want:
                problems.append(f"initial overlay file {rel} ({root}) was changed")
    rec, rec_path = _ab_record(overlay, run_dir)
    if rec is None:
        problems.append("no integrate_result.json accepted this overlay directory")
    elif not rec.get("ab_complete"):
        problems.append(f"{rec_path}: the A/B is not complete")
    else:
        leg = Path(((rec.get("measurement_evidence") or {}).get("cand_summary")) or "")
        stamp = (leg if leg.is_file() else rec_path).stat().st_mtime
        changed = [r for r, p in files.items() if r.endswith(_CODE_SUFFIXES) and p.stat().st_mtime > stamp + 2]
        if changed:
            problems.append(f"overlay sources changed after the A/B measured it: {', '.join(sorted(changed)[:5])}")
    h = hashlib.sha256()
    for rel, p in sorted(files.items()):
        h.update(rel.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return {"ok": not problems, "problems": problems, "manifest_sha256": h.hexdigest(), "files": len(files)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overlay", required=True, help="the accepted overlay dir")
    ap.add_argument("--initial", default="", help="the overlay(s) it was built on (os.pathsep-separated)")
    ap.add_argument("--root", required=True, help="the run's EVAL_DIR")
    a = ap.parse_args(argv)
    try:
        out = validate_overlay(a.overlay, a.initial, Path(a.root))
    except Exception as e:  # a check that cannot run is a failed check, never an accept
        out = {"ok": False, "problems": [f"check failed: {type(e).__name__}: {e}"[:500]]}
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
