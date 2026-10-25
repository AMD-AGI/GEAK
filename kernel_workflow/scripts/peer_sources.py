#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot ordinary peer archives and privately copy only selected native sources."""

import json
import re
import sys
from pathlib import Path, PurePosixPath

sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.native_runtime.direct_external_command import digest, identities, need, snapshot
from geak.native._util import canonical_json
from geak.native.contract import job_digest
from geak.native.external import validate_external_task
from geak.native.source import _patched, _read


def native_row(row):
    if not isinstance(row, dict):
        return False
    verifier = row.get("verifier") or {}
    # record() owns the verdict; native evidence still binds a selected source.
    return (isinstance(verifier, dict) and row.get("verified") is True
            and not row.get("retention_error") and row.get("patch_ref")
            and verifier.get("external_evaluation"))


def snapshot_peers(peers, directory):
    directory.mkdir(parents=True, exist_ok=True)
    result = {"peers": [], "offers": [], "errors": []}
    for peer in peers:
        try:
            root = Path(peer["root"])
            need(root.is_absolute() and root.resolve(strict=True) == root, "canonical peer root required")
            raw = _read(root, "archive.json")[0]
            archive = json.loads(raw)
            need(archive["schema"] == "geak.evolution/1", "unsupported peer archive")
            sha = digest(raw)
            name = f"peer_{peer['id']}.archive.json"
            with (directory / name).open("xb") as stream:
                stream.write(raw)
            offers = []
            for row in archive["entries"]:
                if native_row(row):
                    need(re.fullmatch(r"r[1-9][0-9]*_(?:d[0-9]+|integrated)", row["id"])
                         and len(row["id"]) <= 100, "invalid peer source ID")
                    offers.append({"id": f"peer_{peer['id']}_{sha}_{row['id']}",
                        "peer_id": peer["id"], "origin_id": row["id"],
                        "title": row.get("title"), "approach": row.get("approach"),
                        "status": row.get("status"), "foreign_speedup": row.get("speedup")})
            result["peers"].append({**peer, "archive_sha256": sha})
            result["offers"].extend(offers)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["errors"].append({"peer_id": peer["id"], "error": str(error)})
    raw = canonical_json(result) + b"\n"
    path = directory / "peer_sources.json"
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path), "sha256": digest(raw), **result}


def import_source(request):
    reference = request["snapshot"]
    index = Path(reference["path"])
    raw = _read(index.parent, index.name)[0]
    need(digest(raw) == reference["sha256"], "peer snapshot changed")
    frozen = json.loads(raw)
    offers = [row for row in frozen["offers"] if row["id"] == request["id"]]
    need(len(offers) == 1, "selected peer source is absent or ambiguous in this round snapshot")
    offer = offers[0]
    peer = next(row for row in frozen["peers"] if row["id"] == offer["peer_id"])
    archive_raw = _read(index.parent, f"peer_{peer['id']}.archive.json")[0]
    need(digest(archive_raw) == peer["archive_sha256"], "peer archive snapshot changed")
    archive = json.loads(archive_raw)
    rows = [row for row in archive["entries"] if row["id"] == offer["origin_id"]]
    need(len(rows) == 1 and native_row(rows[0]), "selected peer source lacks native verification")
    row = rows[0]
    root, task, base = map(Path, (peer["root"], request["task_root"], request["base"]))
    need(root.resolve(strict=True) == root, "peer root changed")
    task_raw = _read(task, "task.json")[0]
    contract = validate_external_task(json.loads(task_raw), task)
    # Embedded paths describe the publisher's mount. Only their relative suffix
    # is usable under the owner's declared read-only peer root.
    published_base = PurePosixPath(archive["base"])
    need(published_base.is_absolute() and published_base.name == "base", "invalid peer base path")
    patch_name = PurePosixPath(row["patch_ref"]["path"]).relative_to(published_base.parent).as_posix()
    need(patch_name == f"patches/{row['id']}.diff", "peer patch is outside its archive")
    for actual in (base, root / "base"):
        need(_read(actual, "task.json")[0] == task_raw, "peer or recipient task differs")
    original = {name: _read(task, original) for name, original in contract["original_source_map"].items()}
    _, original_id, _ = identities(contract, peer["native_epoch"], original)
    local_base = snapshot(base, contract["source_paths"])
    peer_base = snapshot(root / "base", contract["source_paths"])
    need(local_base == original and peer_base == original, "actual run bases must equal protected ORIGINAL")
    patch = _read(root, patch_name)[0]
    need(digest(patch) == row["patch_ref"]["sha256"], "selected peer patch changed")
    _, source_id, candidate_id = identities(contract, peer["native_epoch"], _patched(peer_base, patch))
    native = row["verifier"]["external_evaluation"]
    need(native.get("candidate_id") == candidate_id
         and isinstance(native.get("evaluation_id"), str) and native["evaluation_id"],
         "peer source differs from its native verifier identity")
    target = Path(request["archive_root"]) / "patches" / (offer["id"] + ".diff")
    try:
        with target.open("xb") as stream:
            stream.write(patch)
    except FileExistsError:
        need(_read(target.parent, target.name)[0] == patch, "private imported patch differs")
    return {"id": offer["id"], "title": row.get("title", ""), "approach": row.get("approach"),
        "parent_id": "ORIGINAL", "donor_ids": [], "base_commit": request["base_commit"],
        "patch_ref": {"path": str(target), "sha256": digest(patch), "base_commit": request["base_commit"]},
        "verified": False, "speedup": None, "status": "unverified", "retention_error": None,
        "origin": {"peer_id": peer["id"], "id": row["id"], "source_id": source_id,
            "original_source_id": original_id, "job_id": job_digest(contract), "task_sha256": digest(task_raw),
            "native_epoch": peer["native_epoch"], "external_evaluation": native,
            "snapshot": reference, "archive_sha256": peer["archive_sha256"],
            "base_commit": archive["base_commit"], "parent_id": row.get("parent_id"),
            "donor_ids": row.get("donor_ids", []), "status": row.get("status"), "speedup": row.get("speedup")}}


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        result = (snapshot_peers(request["peers"], Path(request["directory"]))
                  if sys.argv[1] == "snapshot" else import_source(request))
        print(canonical_json(result).decode())
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
