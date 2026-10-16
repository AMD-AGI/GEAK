"""File references: a plain file's absolute path plus the SHA-256 of its bytes.

``file_ref`` makes a reference, ``read_ref`` reads the bytes back and fails if
they changed. ``source_reference`` and ``_source_ref`` do the same for a saved
candidate and its source manifest inside a ResearchStore.
"""

import hashlib
import json
from pathlib import Path

from .._util import EvoError, is_sha256, require


class EngineerError(EvoError, ValueError):
    """A file or source reference is malformed, missing or changed.

    The name is kept from the engineer module these helpers came from, so
    callers that catch ``EngineerError`` keep working.
    """


def _need(condition, message):
    require(condition, message, EngineerError)


def _sha(value):
    _need(is_sha256(value), "invalid SHA256")
    return value


def _plain_file(path):
    _need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
          "reference must be a canonical plain file")


def file_ref(path):
    """Reference a plain file by path and hash, without executing or importing it."""
    path = Path(path)
    _plain_file(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_ref(ref):
    """Return the referenced file's bytes; fail if they no longer match the hash."""
    _need(type(ref) is dict and set(ref) == {"path", "sha256"}, "invalid file reference")
    _sha(ref["sha256"])
    path = Path(ref["path"])
    _plain_file(path)
    data = path.read_bytes()
    _need(hashlib.sha256(data).hexdigest() == ref["sha256"], "immutable reference changed")
    return data


_read_ref = read_ref  # Old name; existing callers import it.


def source_reference(store, candidate_id):
    """Reference a saved candidate's exact source manifest in ``store``."""
    candidate = store.get_candidate(candidate_id)
    return {"candidate_id": candidate_id, "source_id": candidate["source_id"],
            "manifest": file_ref((store.root / "objects" / candidate["source_id"]).resolve())}


def _source_ref(source):
    """Check a ``source_reference`` against the files it names; return the candidate."""
    _need(type(source) is dict and set(source) == {"candidate_id", "source_id", "manifest"},
          "invalid source reference")
    _sha(source["candidate_id"])
    _sha(source["source_id"])
    manifest = json.loads(read_ref(source["manifest"]))
    _need(source["manifest"]["sha256"] == source["source_id"] and manifest.get("kind") == "source",
          "source manifest identity mismatch")
    objects = Path(source["manifest"]["path"]).parent
    candidate = json.loads(read_ref({"path": str(objects / source["candidate_id"]),
                                     "sha256": source["candidate_id"]}))
    _need(candidate.get("kind") == "candidate" and candidate.get("source_id") == source["source_id"],
          "candidate and source identity mismatch")
    for entry in manifest["files"]:
        if entry["blob_id"] is not None:
            read_ref({"path": str(objects / _sha(entry["blob_id"])), "sha256": entry["blob_id"]})
    return candidate
