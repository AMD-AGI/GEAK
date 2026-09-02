#!/usr/bin/env python3
"""A KB Store held on disk, in the shape the service uses.

This is the local plane of the same two-plane design KernelForge already ships
(`kernel_agents/rewrite_by_flydsl/record_store.py`: one `RewriteRecordStore` protocol,
`LocalRewriteRecords` on disk and `KBStoreRewriteRecords` over HTTP, chosen by
`KNOWLEDGE_STORE_MODE`). The layout, the ranking rule and the champion file are copied from
it deliberately, so the whole read/apply/optimize/write-back loop can be exercised offline and
switching to the service later is a change of backend, not of behaviour:

    <root>/geak/kernel/geak/moe_stage1/rocm/7.2/ck/gfx950/
        champion.json                     {"session_id", "metric": "speedup", "value"}
        sessions/<session id>/
            knowledge.json                producer / speedup / identity / value
            files/patch.diff, files/report.md

Three properties are load-bearing and must not drift from upstream:

  * ranking is `speedup` descending and nothing else. The store does not know what a bench key
    is, so it will happily order a `b:` measurement against a `b2:` one. Comparability is the
    caller's job, which is why `resolve-remote` filters on it client-side.
  * `candidates()` reads knowledge documents only. Artifacts are fetched by `materialize()`, for
    the selected few. Patches here reach 240KB and travel through an agent's tool result.
  * every mutation lands by atomic rename, and a session id repeated means overwrite. Session ids
    are content-addressed upstream, so re-recording one port updates one candidate rather than
    growing a new one per run.

Stdlib only, like experience_store.py, so a lane agent can call it over Bash.
"""

import errno
import json
import os
import re
import shutil
import tempfile
import uuid

try:
    import fcntl
except ImportError:            # pragma: no cover - POSIX only in practice
    fcntl = None

KNOWLEDGE_FILENAME = "knowledge.json"
CHAMPION_FILENAME = "champion.json"
RECIPE_FILENAME = "recipe.json"
LOCK_FILENAME = ".lock"
CHAMPION_METRIC = "speedup"
ARTIFACT_KIND = "rewrite"

# Both copied verbatim from upstream. Widening either one produces records the service would
# reject, which is exactly the failure this local plane exists to catch early.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SEGMENT_RE = re.compile(r"^[a-z0-9_][a-z0-9._+-]*$")


class KBStoreError(RuntimeError):
    """Anything that makes a record unusable: a bad id, an unsafe path, a broken document."""


def finite_speedup(value):
    """The ranking key, or None when the document does not carry a usable one.

    `True` is an int in Python and would sort as 1.0; a string "1.5" would raise. Both mean the
    producer wrote something we cannot rank, so both read as absent rather than as a number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def validate_session_id(session_id: str) -> str:
    """Reject ids that cannot be both a URL segment and a directory name."""
    raw = str(session_id or "").strip()
    if not _SESSION_ID_RE.fullmatch(raw):
        raise KBStoreError("unusable session id: %r" % (session_id,))
    return raw


# The character class every caller that MANGLES a filename toward safe_rel_path() builds its regex
# from. This validator raises rather than sanitizes, and it raises while the whole upload map is being
# built, so one shape-derived table name (`gemm_m:1024...csv`) aborts the write for every other file
# in the record — which is why two lanes each grew a sanitizer (experience_store._safe, with '-' and
# an 80-char cap; e2e_store._safe_basename, with '_'). Those two spellings are NOT interchangeable and
# are deliberately not merged: the mangled name is hashed into a tuned artifact's signature and stored
# in records, so swapping either lane's substitute re-addresses entries already in the store. What
# must not drift is WHICH characters are unsafe, so that lives here, once, next to the validator.
SAFE_COMPONENT_CHARS = "A-Za-z0-9._-"


def safe_rel_path(rel_path: str) -> str:
    """Reject artifact paths that could escape the record's files directory."""
    if not isinstance(rel_path, str):
        raise KBStoreError("unsafe artifact path: %r" % (rel_path,))
    parts = rel_path.split("/")
    if (not rel_path or "\0" in rel_path or "\\" in rel_path or rel_path.startswith("/")
            or ":" in rel_path or any(p in ("", ".", "..") for p in parts)):
        raise KBStoreError("unsafe artifact path: %r" % (rel_path,))
    return "/".join(parts)


def canonical_segments(canonical_id: str):
    """Render an identity as nested directory names, scheme first."""
    parts = str(canonical_id or "").split(":")
    if len(parts) < 2 or any(not _SEGMENT_RE.fullmatch(p) for p in parts):
        raise KBStoreError("unusable canonical id: %r" % (canonical_id,))
    return parts


class Candidate(object):
    """One recorded port, ranked by the speedup its own document claims."""

    __slots__ = ("session_id", "knowledge", "speedup", "is_champion")

    def __init__(self, session_id, knowledge, speedup, is_champion):
        self.session_id = session_id
        self.knowledge = knowledge
        self.speedup = speedup
        self.is_champion = is_champion

    @property
    def value(self):
        """The producer-owned half — opaque to the store, everything to the caller."""
        v = self.knowledge.get("value")
        return v if isinstance(v, dict) else {}


def _write_json(path: str, document) -> None:
    _atomic_bytes(path, json.dumps(document, ensure_ascii=False, indent=2,
                                   sort_keys=True).encode("utf-8"))


def _atomic_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    _fsync_dir(directory)


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _replace_directory(staging: str, destination: str) -> None:
    """Install `staging` at `destination`, putting the previous tree back if that fails."""
    parent = os.path.dirname(destination) or "."
    backup = ""
    if os.path.lexists(destination):
        if os.path.islink(destination) or not os.path.isdir(destination):
            raise KBStoreError("existing session is not a safe directory: " + destination)
        backup = os.path.join(parent, ".%s.backup-%s" % (os.path.basename(destination),
                                                         uuid.uuid4().hex))
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup:
            os.replace(backup, destination)
            backup = ""
        raise
    finally:
        if backup:
            shutil.rmtree(backup, ignore_errors=True)
    _fsync_dir(parent)


class LocalKBStore(object):
    """Read and write one producer's candidates under a canonical identity, on disk."""

    def __init__(self, root: str, metric: str = CHAMPION_METRIC, promote_floor: float = 1.0):
        self.root = os.path.abspath(os.path.expanduser(str(root)))
        # Which flat top-level `knowledge.<name>` scalar ranks this identity, and the value a
        # candidate must beat to be worth recording as champion at all. `speedup` above 1.0 is the
        # kernel lane's rule and stays the default. The e2e lane's exact-workload rung ranks on
        # absolute `throughput_tok_s` instead, where 1.0 would be a nonsense floor: two runs there
        # share a workload but not necessarily a baseline, so the higher speedup can easily be the
        # slower server. Coarser e2e rungs go back to `speedup`, because their workloads differ and
        # absolute numbers across them are not comparable at all.
        self.metric = str(metric or CHAMPION_METRIC)
        self.promote_floor = float(promote_floor)

    # -- addressing ----------------------------------------------------------------------

    def identity_dir(self, canonical_id: str) -> str:
        return os.path.join(self.root, *canonical_segments(canonical_id))

    def session_dir(self, canonical_id: str, session_id: str) -> str:
        return os.path.join(self.identity_dir(canonical_id), "sessions",
                            validate_session_id(session_id))

    def identities(self):
        """Every canonical id this store holds. Only useful locally — the service has an index."""
        found = []
        for current, directories, _files in os.walk(self.root, followlinks=False):
            if "sessions" not in directories:
                continue
            rel = os.path.relpath(current, self.root)
            if rel == ".":
                continue
            found.append(":".join(rel.split(os.sep)))
        return sorted(found)

    # -- read ----------------------------------------------------------------------------

    def candidates(self, canonical_id: str, limit: int = 3):
        """Rank this identity's candidates. Reads no artifact bytes."""
        sessions_dir = os.path.join(self.identity_dir(canonical_id), "sessions")
        if not os.path.isdir(sessions_dir) or os.path.islink(sessions_dir):
            return []
        champion_id = str(self.champion(canonical_id).get("session_id") or "")
        found = []
        for name in sorted(os.listdir(sessions_dir)):
            entry = os.path.join(sessions_dir, name)
            if not os.path.isdir(entry) or os.path.islink(entry) or name.startswith("."):
                continue
            validate_session_id(name)
            document = os.path.join(entry, KNOWLEDGE_FILENAME)
            if not os.path.isfile(document) or os.path.islink(document):
                continue
            try:
                with open(document, "r", errors="replace") as handle:
                    knowledge = json.load(handle)
            except (OSError, ValueError):
                continue                       # a half-written document is a miss, not a crash
            if not isinstance(knowledge, dict):
                continue
            found.append(Candidate(name, knowledge, finite_speedup(knowledge.get(self.metric)),
                                   name == champion_id))
        # Ties keep the session id order so two runs over one store rank identically.
        found.sort(key=lambda c: (-(c.speedup if c.speedup is not None else float("-inf")),
                                  c.session_id))
        return found[: max(0, int(limit))] if limit else found

    def get_session(self, canonical_id: str, session_id: str):
        document = os.path.join(self.session_dir(canonical_id, session_id), KNOWLEDGE_FILENAME)
        try:
            with open(document, "r", errors="replace") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def session_files(self, canonical_id: str, session_id: str):
        """Relative paths of one session's artifacts, without reading them."""
        root = os.path.join(self.session_dir(canonical_id, session_id), "files")
        found = []
        for current, _dirs, filenames in os.walk(root, followlinks=False):
            for name in filenames:
                path = os.path.join(current, name)
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                found.append(os.path.relpath(path, root).replace(os.sep, "/"))
        return sorted(found)

    def materialize(self, canonical_id: str, candidate, destination: str) -> str:
        """Lay one selected candidate out as the standard bundle: recipe.json + files/."""
        session_id = candidate.session_id if isinstance(candidate, Candidate) else str(candidate)
        source = self.session_dir(canonical_id, session_id)
        if os.path.islink(source) or not os.path.isdir(source):
            raise KBStoreError("candidate session is not a safe directory: " + source)
        knowledge = self.get_session(canonical_id, session_id)
        if knowledge is None:
            raise KBStoreError("candidate knowledge is unreadable: " + source)
        if not isinstance(candidate, Candidate):
            champion_id = str(self.champion(canonical_id).get("session_id") or "")
            candidate = Candidate(session_id, knowledge,
                                  finite_speedup(knowledge.get("speedup")),
                                  session_id == champion_id)

        os.makedirs(destination, exist_ok=True)
        bundle = os.path.join(destination, session_id)
        staging = tempfile.mkdtemp(prefix="." + session_id + "-", dir=destination)
        try:
            files_root = os.path.join(staging, "files")
            os.makedirs(files_root, exist_ok=True)
            for rel in self.session_files(canonical_id, session_id):
                target = os.path.join(files_root, *safe_rel_path(rel).split("/"))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copyfile(os.path.join(source, "files", *rel.split("/")), target)
            recipe = dict(knowledge)
            recipe.update({"canonical_id": canonical_id, "session_id": session_id,
                           "is_champion": candidate.is_champion,
                           "champion": candidate.is_champion})
            recipe.setdefault("speedup", candidate.speedup)
            _write_json(os.path.join(staging, RECIPE_FILENAME), recipe)
            _replace_directory(staging, bundle)
            staging = ""
        finally:
            if staging and os.path.isdir(staging):
                shutil.rmtree(staging, ignore_errors=True)
        return bundle

    def champion(self, canonical_id: str):
        path = os.path.join(self.identity_dir(canonical_id), CHAMPION_FILENAME)
        if not os.path.isfile(path) or os.path.islink(path):
            return {}
        try:
            with open(path, "r", errors="replace") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def champion_speedup(self, canonical_id: str):
        champion = self.champion(canonical_id)
        # A champion recorded under a different metric is not a weaker incumbent, it is an
        # incomparable one. Returning None makes the caller treat the slot as empty rather than
        # rank tokens-per-second against a ratio.
        if str(champion.get("metric") or "") != self.metric:
            return None
        return finite_speedup(champion.get("value"))

    # -- write ---------------------------------------------------------------------------

    def write(self, canonical_id: str, session_id: str, knowledge, files=None) -> str:
        """Record one candidate and its artifacts under an identity, atomically.

        The whole session lands or none of it does: a reader must never see a knowledge document
        that references bytes this store does not hold yet.
        """
        if not isinstance(knowledge, dict):
            raise KBStoreError("knowledge is not an object")
        session_id = validate_session_id(session_id)
        identity_dir = self.identity_dir(canonical_id)
        sessions_dir = os.path.join(identity_dir, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        named = {safe_rel_path(rel): src for rel, src in (files or {}).items()}

        with self._lock(identity_dir):
            staging = tempfile.mkdtemp(prefix="." + session_id + ".staging-", dir=sessions_dir)
            try:
                files_root = os.path.join(staging, "files")
                os.makedirs(files_root, exist_ok=True)
                for rel in sorted(named):
                    target = os.path.join(files_root, *rel.split("/"))
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    shutil.copyfile(named[rel], target)
                _write_json(os.path.join(staging, KNOWLEDGE_FILENAME), knowledge)
                destination = os.path.join(sessions_dir, session_id)
                _replace_directory(staging, destination)
                staging = ""
            finally:
                if staging and os.path.isdir(staging):
                    shutil.rmtree(staging, ignore_errors=True)
        return os.path.join(sessions_dir, session_id)

    def promote(self, canonical_id: str, session_id: str, speedup: float) -> None:
        """Point the identity's champion at one session. The caller owns the policy."""
        document = {"session_id": validate_session_id(session_id),
                    "metric": self.metric, "value": float(speedup)}
        identity_dir = self.identity_dir(canonical_id)
        os.makedirs(identity_dir, exist_ok=True)
        with self._lock(identity_dir):
            _write_json(os.path.join(identity_dir, CHAMPION_FILENAME), document)

    def maybe_promote(self, canonical_id: str, session_id: str, speedup) -> bool:
        """Upstream's gate, verbatim: only a real win, and only over the incumbent."""
        speedup = finite_speedup(speedup)
        if speedup is None or speedup <= self.promote_floor:
            return False
        incumbent = self.champion_speedup(canonical_id)
        if incumbent is not None and speedup <= incumbent:
            return False
        self.promote(canonical_id, session_id, speedup)
        return True

    # -- locking -------------------------------------------------------------------------

    class _Lock(object):
        def __init__(self, path):
            self.path = path
            self.handle = None

        def __enter__(self):
            if fcntl is None:
                return self
            try:
                self.handle = open(self.path, "a+")
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
            except OSError as error:
                # A read-only or lock-less filesystem must not make the store unusable; the
                # atomic renames below are still atomic, we just lose writer serialization.
                if self.handle is not None:
                    self.handle.close()
                    self.handle = None
                if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOSYS,
                                       errno.ENOLCK):
                    raise
            return self

        def __exit__(self, *exc):
            if self.handle is not None:
                try:
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                finally:
                    self.handle.close()
                    self.handle = None
            return False

    def _lock(self, identity_dir: str):
        return LocalKBStore._Lock(os.path.join(identity_dir, LOCK_FILENAME))


__all__ = ["ARTIFACT_KIND", "CHAMPION_METRIC", "Candidate", "KBStoreError", "LocalKBStore",
           "canonical_segments", "finite_speedup", "safe_rel_path", "validate_session_id"]
