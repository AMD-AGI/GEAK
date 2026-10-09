"""Locate GEAK's shared hardware data and the Gluon pack from any kernel tool.

The per-arch / per-SKU tables (`sku.json`, `hw_constants.json`, `thresholds.json`,
`workload_models.json` and the ISA encoding databases) live once, in
`perf_knowledge/hardware/data/`, and every tool reads them through here so no two tools
can disagree about which copy is canonical.

Lookup order for a data file:
  1. `$GEAK_HW_DATA_DIR/<name>`                      explicit override
  2. `<repo>/perf_knowledge/hardware/data/<name>`    this checkout
  3. an ancestor's `perf_knowledge/hardware/data/`   tool copied elsewhere inside a checkout
  4. `<here>/data/<name>`, `<here>/../references/hardware/<name>`  tool copied out alone
The ancestor walk is bounded (MAX_UP levels) so a shallow copy never walks the filesystem.

The Gluon pack (references the tools print or read, e.g. `lever-cards.json`) resolves from
`$GEAK_GLUON_PACK_DIR`, else `<repo>/perf_knowledge/expert_skills/skills/gluon_authoring`.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]          # kernel_workflow/scripts/kernel_tools -> repo root
DATA_REL = Path("perf_knowledge") / "hardware" / "data"
PACK_REL = Path("perf_knowledge") / "expert_skills" / "skills" / "gluon_authoring"
MAX_UP = 6


def _candidates(name: str):
    env = os.environ.get("GEAK_HW_DATA_DIR")
    if env:
        yield Path(env) / name
    yield REPO_ROOT / DATA_REL / name
    for anc in list(HERE.parents)[:MAX_UP]:
        yield anc / DATA_REL / name
    yield HERE / "data" / name
    yield HERE.parent / "references" / "hardware" / name


def find(name: str) -> Path | None:
    """Path of a hardware data file, or None if no candidate exists."""
    for p in _candidates(name):
        if p.is_file():
            return p
    return None


def data_dir() -> Path | None:
    """Directory holding `hw_constants.json` (the anchor file), or None."""
    p = find("hw_constants.json")
    return p.parent if p else None


def load_json(name: str):
    """Parse a data file (transparently gunzipping `*.gz`). Raises FileNotFoundError."""
    p = find(name)
    if p is None:
        raise FileNotFoundError(f"{name}: not found under $GEAK_HW_DATA_DIR or {DATA_REL}")
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as fh:
            return json.load(fh)
    with open(p) as fh:
        return json.load(fh)


def pack_dir() -> Path | None:
    """Root of the gluon_authoring pack, or None if it is not reachable."""
    env = os.environ.get("GEAK_GLUON_PACK_DIR")
    for p in ([Path(env)] if env else []) + [REPO_ROOT / PACK_REL]:
        if (p / "skill.md").is_file():
            return p
    return None


def pack_file(rel: str) -> Path | None:
    """A file inside the pack (e.g. 'references/hardware/lever-cards.json'), or None."""
    root = pack_dir()
    if root is None:
        return None
    p = root / rel
    return p if p.exists() else None


def _selftest() -> int:
    fails = []
    for n in ("sku.json", "hw_constants.json", "thresholds.json", "workload_models.json",
              "gfx950-encoding.json.gz"):
        if find(n) is None:
            fails.append(f"missing {n}")
    try:
        if "gfx950" not in json.dumps(load_json("hw_constants.json")):
            fails.append("hw_constants.json has no gfx950")
    except Exception as e:  # noqa: BLE001
        fails.append(f"load_json: {e}")
    if pack_dir() is None:
        fails.append("pack dir not found")
    old = os.environ.get("GEAK_HW_DATA_DIR")
    os.environ["GEAK_HW_DATA_DIR"] = "/nonexistent-geak-hw"
    try:
        if find("sku.json") is None:
            fails.append("override to a missing dir must fall through, not fail")
    finally:
        if old is None:
            os.environ.pop("GEAK_HW_DATA_DIR")
        else:
            os.environ["GEAK_HW_DATA_DIR"] = old
    for f in fails:
        print("FAIL", f)
    print("[_hwdata] SELFTEST " + ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest() if "--selftest" in sys.argv else (print(data_dir()) or 0))
