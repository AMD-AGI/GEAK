"""Deterministic rocprof counter probe for AVO (the parse half of A2).

Closes the roofline's biggest accuracy gap: analytic FLOPs/bytes can mis-classify a
cache-friendly kernel as memory-bound. Real **measured HBM bytes** (+ occupancy /
MFMA-active counters) from rocprofv3 give a trustworthy arithmetic intensity → regime.

Two halves, by design (mirrors scripts/ir_probe.py):
  - PARSE (pure, deterministic, unit-tested on fixtures): rocprof counter CSV ->
    a small counter vector {hbm_bytes, occupancy_pct, mfma_active_pct, ...}.
  - COLLECT (rig-only, best-effort): run `rocprofv3 -i <counters> -- <bench>` and emit
    the CSV. Needs ROCm, so NOT unit-tested here; on failure -> ok=false (inert).

The caller feeds `hbm_bytes` into the per-case telemetry as `measured_bytes`, which the
roofline model prefers over analytic `bytes` (see lib/roofline.mjs). All graceful: no
counters -> ok=false -> the loop falls back to analytic / relative mode.

Subcommands
-----------
parse    --counters-file F [--size-unit kb|bytes]
         Parse a rocprof counter CSV into the counter vector JSON (testable path).
collect  --workspace D --bench CMD ... (rig-only) -> emits ok=false stub here.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path


# Known counter aliases -> canonical field. Names vary across rocprof versions, so we
# match case-insensitively against this alias map.
_FETCH = {"fetch_size", "fetchsize", "tcc_ea_rdreq_bytes"}
_WRITE = {"write_size", "writesize", "tcc_ea_wrreq_bytes"}
_HBM = {"hbm_bytes"}
_OCC = {"occupancy", "occupancypct", "occupancy_pct"}
_MFMA = {"mfmautil", "mfma_active_pct", "mfmaactive", "mfma_util"}
_VALU = {"valuutilization", "valu_util", "valuutil"}


def _to_float(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def parse_counters_csv(text: str) -> dict:
    """Parse a rocprof CSV (header + one-or-more data rows) into {name_lower: value},
    averaging numeric columns across rows (rocprof emits one row per dispatch)."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return {}
    sums, counts = {}, {}
    for row in reader:
        for k, v in row.items():
            if k is None:
                continue
            f = _to_float(v)
            if f is None:
                continue
            key = k.strip().lower()
            sums[key] = sums.get(key, 0.0) + f
            counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def _first(counters: dict, aliases: set):
    for k, v in counters.items():
        if k in aliases:
            return v
    return None


def _norm_pct(v):
    """Normalize an occupancy/util to a 0-100 percentage (accept 0-1 fractions)."""
    if v is None:
        return None
    return round(v * 100, 2) if v <= 1.0 else round(v, 2)


def counters_to_vector(counters: dict, size_unit: str = "kb") -> dict:
    """Derive the canonical counter vector. `size_unit` is the unit of FETCH/WRITE_SIZE
    (rocprofv3 reports KB by default)."""
    mult = 1024.0 if size_unit == "kb" else 1.0
    hbm = _first(counters, _HBM)
    if hbm is None:
        fetch = _first(counters, _FETCH)
        write = _first(counters, _WRITE)
        if fetch is not None or write is not None:
            hbm = ((fetch or 0.0) + (write or 0.0)) * mult
    return {
        "hbm_bytes": int(hbm) if hbm is not None else None,
        "occupancy_pct": _norm_pct(_first(counters, _OCC)),
        "mfma_active_pct": _norm_pct(_first(counters, _MFMA)),
        "valu_util_pct": _norm_pct(_first(counters, _VALU)),
    }


def build_vector(counters: dict | None, ok: bool = True, notes: str = "") -> dict:
    has = bool(counters) and any(v is not None for v in counters.values())
    return {"source": "rocprofv3", "ok": bool(ok and has), "counters": counters or {}, "notes": notes}


def probe_text(text: str, size_unit: str = "kb") -> dict:
    raw = parse_counters_csv(text)
    if not raw:
        return build_vector(None, ok=False, notes="no parseable rocprof CSV")
    vec = counters_to_vector(raw, size_unit=size_unit)
    return build_vector(vec, ok=True)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_parse(args) -> int:
    p = Path(args.counters_file)
    if not p.exists():
        print(json.dumps(build_vector(None, ok=False, notes=f"file not found: {args.counters_file}")))
        return 0
    text = p.read_text(encoding="utf-8", errors="ignore")
    print(json.dumps(probe_text(text, size_unit=args.size_unit), indent=2, default=str))
    return 0


def cmd_collect(args) -> int:
    """(rig-only) Run rocprofv3 to produce the counter CSV, then parse it. Without ROCm
    this returns ok=false — by design the loop then uses analytic bytes / relative mode."""
    print(json.dumps(build_vector(None, ok=False,
        notes="collect is rig-only (rocprofv3 required); run `parse` on a collected CSV"),
        indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic AVO rocprof probe (parse half).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse")
    p.add_argument("--counters-file", required=True)
    p.add_argument("--size-unit", default="kb", choices=["kb", "bytes"])
    p.set_defaults(func=cmd_parse)

    c = sub.add_parser("collect")
    c.add_argument("--workspace", required=True)
    c.add_argument("--bench", default="")
    c.add_argument("--size-unit", default="kb", choices=["kb", "bytes"])
    c.set_defaults(func=cmd_collect)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
