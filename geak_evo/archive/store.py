"""The saved research history of one search: source snapshots, candidates,
evidence and the current state of each research direction.

Files under the store root:

- ``objects/<sha256>``: immutable bytes named by their hash. These hold source
  files, source manifests (kind "source"), candidates (kind "candidate") and
  evidence records (kind "evidence").
- ``records/<sequence>-<sha256>.json``: the append-only journal. Each record
  holds one direction's new state and the hash of the previous record.
- ``closures/<sha256>``: the source-file layouts a job is allowed to use.
- ``index.json``: a cache of the journal replay; the journal is the truth.

Every candidate is bound to a ``(job_id, measurement_epoch_id)`` pair. Only
pairs listed in ``allowed_epochs`` are readable, so evidence measured under one
evaluator setup is never mixed with candidates from another.

Writers serialize on an exclusive file lock, and each update names the
revision it expects, so a stale writer fails instead of overwriting. The store
does not evaluate or select anything; the host supplies ``verify_evidence``.
"""

import contextlib
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile

from .._util import EvoError, SHA256_PATTERN, canonical_json, sha256_hex


class StateError(EvoError, ValueError):
    """A store read or write broke a storage rule."""


# Other modules import these private helpers by name.
_json = canonical_json
_digest = sha256_hex


def _path(value):
    """Return ``value`` if it is a canonical relative POSIX path."""
    if (not isinstance(value, str) or not value or "\\" in value or "\0" in value
            or PurePosixPath(value).is_absolute() or ".." in value.split("/")
            or PurePosixPath(value).as_posix() != value or value == "."):
        raise StateError("source/artifact path must be canonical and relative")
    return value


def _read(root, name):
    """Read a regular file without following symlinks, including parent directories.

    Returns ``(bytes, executable_bits)``. Raises if the file changed while read.
    """
    parts = _path(name).split("/")
    with contextlib.ExitStack() as stack:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, fd)
        for part in parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            stack.callback(os.close, fd)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        stream = stack.enter_context(os.fdopen(fd, "rb"))
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise StateError("only regular files may be archived")
        data = stream.read()
        after = os.fstat(fd)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
        if any(getattr(before, k) != getattr(after, k) for k in stable) or len(data) != after.st_size:
            raise StateError("file changed during capture")
        return data, stat.S_IMODE(after.st_mode) & 0o111


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path, data, immutable=False):
    """Durably write ``data`` to ``path``.

    Immutable files are hard-linked into place, so an existing file is never
    replaced (``FileExistsError``). Mutable files are atomically replaced.
    """
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444 if immutable else 0o600)
            os.fsync(stream.fileno())
        if immutable:
            os.link(name, path)
        else:
            os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class ResearchStore:
    """Content-addressed research history with a hash-chained journal.

    ``verify_evidence(artifacts: dict[str, bytes], expected_binding: dict)`` is
    supplied by the host. Its return value records the independent / settled /
    correct / eligible checks; this class cannot authenticate observations.

    ``commit_frontier`` is an optional observer. When given, it writes each
    journal record through ``write_record(path, data, sequence=...)``.
    """

    def __init__(self, root, *, allowed_epochs, source_paths, verify_evidence=None,
                 commit_frontier=None):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True, mode=0o700)
        if self.root.is_symlink():
            raise StateError("store root cannot be a symlink")
        self.allowed_epochs = frozenset(tuple(pair) for pair in allowed_epochs)
        self.source_paths = {job: sorted({_path(p) for p in paths})
                             for job, paths in source_paths.items()}
        self.verify_evidence = verify_evidence
        self.commit_frontier = commit_frontier
        for name in ("objects", "records", "closures"):
            (self.root / name).mkdir(exist_ok=True, mode=0o700)
            if (self.root / name).is_symlink():
                raise StateError("store directories cannot be symlinks")
        _sync_directory(self.root)
        _sync_directory(self.root.parent)
        for job_id, paths in self.source_paths.items():
            self.register_source_paths(job_id, paths)

    # -- source layouts ---------------------------------------------------

    def register_source_paths(self, job_id, paths):
        """Allow one exact file layout for a job. Earlier layouts stay valid."""
        paths = sorted({_path(p) for p in paths})
        if not paths or job_id not in {pair[0] for pair in self.allowed_epochs}:
            raise StateError("source closure requires paths and an allowlisted job")
        data = _json({"job_id": job_id, "paths": paths})
        identity = _digest(data)
        with self._lock():
            try:
                _write(self.root / "closures" / identity, data, immutable=True)
            except FileExistsError:
                if _read(self.root / "closures", identity)[0] != data:
                    raise StateError("source closure registration hash mismatch")
        return identity

    def _check_source_paths(self, job_id, paths):
        data = _json({"job_id": job_id, "paths": paths})
        try:
            registered, _ = _read(self.root / "closures", _digest(data))
        except OSError as exc:
            raise StateError("source closure is not registered by the host") from exc
        if registered != data:
            raise StateError("source closure registration hash mismatch")

    @contextlib.contextmanager
    def _lock(self):
        """Exclusive lock shared by every writer process of this store."""
        fd = os.open(self.root / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    # -- immutable objects ------------------------------------------------

    def _put(self, data):
        if not isinstance(data, bytes):
            raise StateError("archive accepts bytes, never mutable artifact paths")
        identity = _digest(data)
        try:
            _write(self.root / "objects" / identity, data, immutable=True)
        except FileExistsError:
            self._bytes(identity)  # The existing copy must still match its hash.
        return identity

    def _bytes(self, identity):
        if not isinstance(identity, str) or not SHA256_PATTERN.fullmatch(identity):
            raise StateError("invalid content identity")
        try:
            data, _ = _read(self.root / "objects", identity)
        except OSError as exc:
            raise StateError("missing or unsafe immutable object") from exc
        if _digest(data) != identity:
            raise StateError("immutable object hash mismatch")
        return data

    def _object(self, identity, kind):
        value = json.loads(self._bytes(identity))
        if not isinstance(value, dict) or value.get("kind") != kind:
            raise StateError("wrong object kind")
        return value

    # -- sources ----------------------------------------------------------

    def capture_source(self, workspace, paths):
        """Save the listed files of ``workspace``; a missing file is recorded as absent."""
        os.close(os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        paths = sorted({_path(p) for p in paths})
        if not paths:
            raise StateError("source closure is empty")

        def capture(path):
            try:
                data, executable = _read(workspace, path)
                return {"path": path, "blob_id": self._put(data), "executable": executable}
            except FileNotFoundError:
                return {"path": path, "blob_id": None, "executable": 0}

        # Read twice so a change between two files is also detected.
        files = [capture(path) for path in paths]
        if files != [capture(path) for path in paths]:
            raise StateError("source closure changed during capture")
        return self._put(_json({"kind": "source", "files": files}))

    def source(self, identity):
        value = self._object(identity, "source")
        paths = [item["path"] for item in value["files"]]
        if not paths or paths != sorted(set(paths)):
            raise StateError("source paths must be unique and sorted")
        for item in value["files"]:
            _path(item["path"])
            if type(item["executable"]) is not int or item["executable"] & ~0o111:
                raise StateError("invalid executable bits")
            if item["blob_id"] is not None:
                self._bytes(item["blob_id"])
        return value

    def materialize(self, source_id, destination):
        """Write a saved source into a new directory ``destination``."""
        source = self.source(source_id)
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise StateError("materialization destination must not exist")
        staging = Path(tempfile.mkdtemp(prefix=".source-", dir=destination.parent))
        try:
            for item in source["files"]:
                if item["blob_id"] is not None:
                    path = staging / item["path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(self._bytes(item["blob_id"]))
                    path.chmod(0o644 | item["executable"])
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    # -- candidates and evidence ------------------------------------------

    def _binding(self, candidate):
        return {"job_id": candidate["job_id"],
                "measurement_epoch_id": candidate["measurement_epoch_id"],
                "candidate_source_id": candidate["source_id"]}

    def _candidate(self, identity):
        """Load and check a candidate and everything it cites (each node once)."""
        pending, seen, result = [identity], set(), None
        while pending:
            reference = pending.pop()
            if reference in seen:
                continue
            seen.add(reference)
            value = self._object(reference, "candidate")
            if (value["job_id"], value["measurement_epoch_id"]) not in self.allowed_epochs:
                raise StateError("job/measurement epoch is not allowlisted")
            source = self.source(value["source_id"])
            self._check_source_paths(value["job_id"], [item["path"] for item in source["files"]])
            pending.extend(value["parents"] + value["donors"])
            for claim in value["contribution_claims"]:
                for evidence_id in claim.get("evidence", []):
                    evidence = self._object(evidence_id, "evidence")
                    bound = self._object(evidence["candidate_id"], "candidate")
                    if evidence["binding"] != self._binding(bound):
                        raise StateError("contribution evidence binding mismatch")
                    pending.append(evidence["candidate_id"])
                    for blob in evidence["artifacts"].values():
                        self._bytes(blob)
            if reference == identity:
                result = value
        return result

    def get_candidate(self, identity):
        return self._candidate(identity)

    def candidate(self, job_id, measurement_epoch_id, source_id, *, parents=(), donors=(),
                  contribution_claims=()):
        """Save a candidate. Ancestry is recorded; it never implies a contribution."""
        value = {"kind": "candidate", "job_id": job_id, "measurement_epoch_id": measurement_epoch_id,
                 "source_id": source_id, "parents": list(parents), "donors": list(donors),
                 "contribution_claims": list(contribution_claims)}
        cited = value["parents"] + value["donors"]
        for identity in cited:
            self._candidate(identity)
        for claim in value["contribution_claims"]:
            if claim["source_candidate_id"] not in cited:
                raise StateError("contribution claim must identify a cited parent/donor")
            if not claim.get("claim") or not claim.get("scope"):
                raise StateError("contribution claims require claim and scope")
            for evidence_id in claim.get("evidence", []):
                self.evidence(evidence_id)
        identity = self._put(_json(value))
        self._candidate(identity)
        return identity

    def archive_evidence(self, candidate_id, artifacts):
        """Save raw artifact bytes as evidence bound to one candidate."""
        candidate = self._candidate(candidate_id)
        if not artifacts:
            raise StateError("evidence has no raw artifacts")
        value = {"kind": "evidence", "candidate_id": candidate_id,
                 "binding": self._binding(candidate),
                 "artifacts": {_path(name): self._put(data) for name, data in artifacts.items()}}
        return self._put(_json(value))

    def _bound_evidence(self, evidence_id):
        value = self._object(evidence_id, "evidence")
        if value["binding"] != self._binding(self._candidate(value["candidate_id"])):
            raise StateError("evidence binding mismatch")
        return value

    def evidence_artifact(self, evidence_id, name):
        """Hash-check one artifact without loading the others."""
        value = self._bound_evidence(evidence_id)
        return value, self._bytes(value["artifacts"][name])

    def evidence(self, evidence_id):
        value = self._bound_evidence(evidence_id)
        return value, {name: self._bytes(identity) for name, identity in value["artifacts"].items()}

    # -- the direction journal --------------------------------------------

    def _validate(self, state):
        candidate = self._candidate(state["tip"])
        if (state["job_id"], state["measurement_epoch_id"]) != (
                candidate["job_id"], candidate["measurement_epoch_id"]):
            raise StateError("direction binding mismatch")
        for identity in state["evidence"]:
            evidence, _ = self.evidence(identity)
            if any(evidence["binding"][k] != state[k] for k in ("job_id", "measurement_epoch_id")):
                raise StateError("direction evidence job/epoch mismatch")
        anchor = state["anchor"]
        if anchor:
            evidence, _ = self.evidence(anchor["evidence_id"])
            bound = self._candidate(anchor["candidate_id"])
            if (evidence["candidate_id"] != anchor["candidate_id"] or
                    (bound["job_id"], bound["measurement_epoch_id"]) !=
                    (state["job_id"], state["measurement_epoch_id"]) or
                    any(anchor["checks"].get(k) is not True for k in
                        ("independent", "settled", "correct", "eligible"))):
                raise StateError("anchor lacks bound independent eligibility checks")

    def _replay(self):
        """Rebuild the index from the journal, checking the hash chain."""
        index = {"head": None, "sequence": 0, "directions": {}}
        for path in sorted((self.root / "records").glob("*.json")):
            data, _ = _read(path.parent, path.name)
            record = json.loads(data)
            sequence = index["sequence"] + 1
            if (path.name != "%08d-%s.json" % (sequence, _digest(data)) or
                    record["previous"] != index["head"]):
                raise StateError("journal hash/sequence mismatch")
            state = record["state"]
            previous = index["directions"].get(state["direction_id"])
            if state["revision"] != (previous["revision"] + 1 if previous else 1):
                raise StateError("direction revision mismatch")
            self._validate(state)
            index["directions"][state["direction_id"]] = state
            index.update(head=_digest(data), sequence=sequence)
        # A cached index that is ahead of the journal means records were deleted.
        try:
            cache = json.loads(_read(self.root, "index.json")[0])
        except (OSError, ValueError):
            cache = {}
        if isinstance(cache, dict) and type(cache.get("sequence")) is int:
            if cache["sequence"] > index["sequence"]:
                raise StateError("journal truncated behind previously recorded index")
        return index

    def _commit(self, index, state):
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._validate(state)
        data = _json({"previous": index["head"], "state": state})
        sequence = index["sequence"] + 1
        path = self.root / "records" / ("%08d-%s.json" % (sequence, _digest(data)))
        if self.commit_frontier is None:
            _write(path, data, immutable=True)  # The commit point; index.json is a cache.
        else:
            # The observer performs the same write and keeps its failures;
            # it never re-stamps a record.
            self.commit_frontier.write_record(path, data, sequence=sequence)
        index["directions"][state["direction_id"]] = state
        index.update(head=_digest(data), sequence=sequence)
        _write(self.root / "index.json", _json(index))
        return state

    def read_current(self, direction_id):
        """Read the current revision without rewriting the index cache."""
        with self._lock():
            return self._replay()["directions"][direction_id]

    def get(self, direction_id=None):
        with self._lock():
            index = self._replay()
            _write(self.root / "index.json", _json(index))
            return index["directions"] if direction_id is None else index["directions"][direction_id]

    def get_revision(self, direction_id, revision):
        """Read one committed historical revision without rewriting the index cache."""
        if type(revision) is not int or revision < 1:
            raise StateError("positive historical revision required")
        with self._lock():
            index = self._replay()
            current = index["directions"].get(direction_id)
            if current is None or revision > current["revision"]:
                raise StateError("historical direction revision is not committed")
            for path in sorted((self.root / "records").glob("*.json")):
                state = json.loads(_read(path.parent, path.name)[0])["state"]
                if state["direction_id"] == direction_id and state["revision"] == revision:
                    return state
        raise StateError("historical direction revision is missing")

    def _prior(self, index, direction_id, expected_revision):
        previous = index["directions"].get(direction_id)
        if expected_revision != (previous["revision"] if previous else 0):
            raise StateError("stale direction revision")
        return previous

    def checkpoint(self, direction_id, candidate_id, *, note, expected_revision,
                   status="active", resources=None, evidence=()):
        """Record a direction's new tip, note, status and cumulative resource use."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", direction_id):
            raise StateError("invalid direction ID")
        if any(not isinstance(note.get(k), str) or not note[k].strip()
               for k in ("claim", "scope", "next_test")) or not isinstance(status, str) or not status:
            raise StateError("a checkpoint requires claim, scope, next_test and status")
        with self._lock():
            index = self._replay()
            previous = self._prior(index, direction_id, expected_revision) or {}
            candidate = self._candidate(candidate_id)
            old_usage = previous.get("resources", {})
            usage = dict(resources if resources is not None else old_usage)
            if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in usage.values())
                    or any(usage.get(k, -1) < v for k, v in old_usage.items())):
                raise StateError("cumulative resource usage must be finite and nondecreasing")
            state = dict(previous, direction_id=direction_id, revision=expected_revision + 1,
                         tip=candidate_id, anchor=previous.get("anchor"), note=note,
                         status=status, resources=usage, evidence=list(evidence))
            for key in ("job_id", "measurement_epoch_id"):
                if previous and previous[key] != candidate[key]:
                    raise StateError("new job/epoch requires a new direction")
                state[key] = candidate[key]
            return self._commit(index, state)

    def promote(self, direction_id, candidate_id, evidence_id, *, expected_revision):
        """Set the direction's anchor to a candidate the host verifier accepts."""
        with self._lock():
            index = self._replay()
            state = dict(self._prior(index, direction_id, expected_revision) or {})
            evidence, artifacts = self.evidence(evidence_id)
            candidate = self._candidate(candidate_id)
            if not state or evidence["candidate_id"] != candidate_id or self.verify_evidence is None:
                raise StateError("promotion requires a direction and bound host verifier")
            checks = self.verify_evidence(artifacts, self._binding(candidate))
            state.update(anchor={"candidate_id": candidate_id, "evidence_id": evidence_id,
                                 "checks": checks}, revision=expected_revision + 1)
            return self._commit(index, state)
