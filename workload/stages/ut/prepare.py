#!/usr/bin/env python3
"""Verify profile evidence and prepare a separate UT workspace, without GPU work."""
import argparse
import os
from pathlib import Path
import shutil
import sys

PROFILE = Path(__file__).resolve().parents[1] / "profile"
sys.path.insert(0, str(PROFILE))
from handoff import verify, read, write, sha, REQUIRED, OPTIONAL


def prepare(handoff_path, out):
    handoff_path, out = Path(handoff_path).resolve(), Path(out).resolve()
    doc = verify(handoff_path)
    source = Path(doc["profile_root"]).resolve()
    if out == source or out in source.parents or source in out.parents:
        raise ValueError("UT output must be separate from the profile evidence tree")
    receipt_path = out / "profile_input.json"
    receipt = {"contract": "workload/kernel_ut_input/v1", "handoff": str(handoff_path),
               "handoff_sha256": sha(handoff_path), "profile_root": str(source),
               "selection_params": doc["selection_params"], "local_evidence": {}}
    if receipt_path.exists():
        old = read(receipt_path)
        if old["handoff_sha256"] != receipt["handoff_sha256"] or old["profile_root"] != str(source):
            raise ValueError("UT workspace belongs to different profile evidence; use a new output directory")
    # Copy small top-level descriptors. Heavy structured evidence is referenced once;
    # none of the UT preparation/authoring scripts writes under evidence/.
    out.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED + OPTIONAL:
        if "/" in name or name not in doc["files"]:
            continue
        src, dest = source / name, out / name
        if dest.exists() and sha(dest) != doc["files"][name]["sha256"]:
            raise ValueError("local profile descriptor was changed: " + str(dest))
        if not dest.exists():
            shutil.copyfile(src, dest)
        receipt["local_evidence"][name] = doc["files"][name]["sha256"]
    for name in ("evidence", "configs", "projected"):
        src, dest = source / name, out / name
        if not src.exists():
            continue
        if dest.is_symlink() and dest.resolve() == src.resolve():
            continue
        if dest.exists() or dest.is_symlink():
            raise ValueError("UT input path conflicts with profile evidence: " + str(dest))
        dest.symlink_to(src, target_is_directory=True)
    write(receipt_path, receipt)
    return receipt


def verify_input(out):
    out = Path(out)
    receipt = read(out / "profile_input.json")
    if receipt.get("contract") != "workload/kernel_ut_input/v1":
        raise ValueError("unsupported UT input contract")
    if sha(receipt["handoff"]) != receipt["handoff_sha256"]:
        raise ValueError("profile handoff changed after UT preparation")
    doc = verify(receipt["handoff"])
    expected = {name: doc["files"][name]["sha256"] for name in REQUIRED + OPTIONAL
                if "/" not in name and name in doc["files"]}
    if receipt.get("local_evidence") != expected:
        raise ValueError("UT input receipt does not pin all local profile descriptors")
    for name, digest in receipt["local_evidence"].items():
        if sha(out / name) != digest:
            raise ValueError("local profile evidence changed during UT generation: " + name)
    for name in ("evidence", "configs", "projected"):
        source = Path(doc["profile_root"]) / name
        if source.exists() and (out / name).resolve() != source.resolve():
            raise ValueError("UT input reference changed: " + name)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", default=None, help="default: $IN/profile_handoff.json")
    parser.add_argument("--out", default=os.environ.get("OUT"))
    parser.add_argument("--check", action="store_true", help="verify an already prepared UT input")
    args = parser.parse_args()
    if not args.out:
        parser.error("--out or OUT is required")
    try:
        if args.check:
            verify_input(args.out)
        else:
            path = args.handoff or str(Path(os.environ["IN"]) / "profile_handoff.json")
            prepare(path, args.out)
        print("UT input verified; profile evidence remains read-only")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, "UT preparation: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
