"""Bounded local source retention; never an evaluator or a candidate selector."""

import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import sys
import tempfile


MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_OBJECT_BYTES = 32 * 1024 * 1024
MAX_PATCH_BYTES = 8 * 1024 * 1024
MAX_REPORT_BYTES = 1024 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
MAX_PATHS = 256
MAX_REPORTS = 128
MAX_RECORDS = 4096
MAX_OBJECTS = 32768
SCHEMA_VERSION = 2
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_CAVEAT = (
    "Only declared source files are bound. Runtime, dependencies, benchmark inputs, "
    "case weights and measurement provenance are not independently verified."
)


class ArchiveError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

    def diagnostic(self):
        return {"code": self.code, "message": str(self)}


def fail(code, message):
    raise ArchiveError(code, message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (ValueError, TypeError, RecursionError):
        fail("invalid_json", "JSON must be finite, serializable and bounded in depth")


def unique_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            fail("invalid_json", "Duplicate JSON object key")
        result[name] = value
    return result


def parse_json(data):
    try:
        parsed = json.loads(data, object_pairs_hook=unique_pairs,
                            parse_constant=lambda value: fail("invalid_json", "Nonfinite JSON number"))
        json_bytes(parsed)
        return parsed
    except (ValueError, UnicodeError, RecursionError):
        fail("invalid_json", "Malformed JSON or excessive nesting")


def identifier(value, report=False):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        fail("invalid_id", "IDs must match [A-Za-z0-9][A-Za-z0-9_-]{0,95}")
    if report and (len(value) > 64 or value in ("FOUNDER", "INCUMBENT", "ORIGINAL")):
        fail("invalid_id", "Report IDs have at most 64 characters and cannot be reserved")
    return value


def source_path(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        fail("unsafe_path", "Source paths must be nonempty relative paths of at most 1024 characters")
    components = value.split("/")
    if any(component in ("", ".", "..") or component.lower() == ".git"
           for component in components) or "\\" in value or any(
               ord(character) < 32 or ord(character) > 126 for character in value):
        fail("unsafe_path", "Source paths cannot traverse directories, name .git, or contain control characters")
    return value


def source_scope(paths):
    if not isinstance(paths, list) or not 1 <= len(paths) <= MAX_PATHS:
        fail("source_limit", "Declare between 1 and 256 source files")
    validated = sorted(source_path(value) for value in paths)
    if len(set(validated)) != len(validated):
        fail("unsafe_path", "Duplicate source path")
    for path in validated:
        if any(path.startswith(other + "/") for other in validated if other != path):
            fail("unsafe_path", "A declared file cannot be another source's directory")
    return validated


def absolute_path(value, required=False):
    if not isinstance(value, (str, Path)) or not str(value) or "\x00" in str(value):
        fail("unsafe_path", "Expected a filesystem path")
    path = Path(value)
    if required and not path.is_absolute():
        fail("unsafe_path", "This path must be absolute")
    if ".." in path.parts:
        fail("unsafe_path", "Filesystem paths cannot contain traversal components")
    return Path(os.path.abspath(path))


@contextlib.contextmanager
def directory_fd(path):
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        yield descriptor
    except OSError:
        fail("unsafe_path", "Directory must exist without symlink components: " + str(path))
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def source_parent(root, relative, create=False):
    with directory_fd(root) as root_descriptor:
        descriptor = os.dup(root_descriptor)
        try:
            for component in relative.split("/")[:-1]:
                if create:
                    try:
                        os.mkdir(component, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                try:
                    following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                        dir_fd=descriptor)
                except FileNotFoundError:
                    yield None, relative.split("/")[-1]
                    return
                except OSError:
                    fail("unsafe_path", "Source directory is a symlink or non-directory: " + relative)
                os.close(descriptor)
                descriptor = following
            yield descriptor, relative.split("/")[-1]
        finally:
            os.close(descriptor)


def read_regular(root, relative, limit, missing=False):
    with source_parent(root, relative) as (parent_descriptor, name):
        if parent_descriptor is None:
            if missing:
                return None
            fail("missing_file", "File is absent: " + relative)
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=parent_descriptor)
        except FileNotFoundError:
            if missing:
                return None
            fail("missing_file", "File is absent: " + relative)
        except OSError:
            fail("unsafe_path", "Cannot safely open regular file: " + relative)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o7000:
                fail("nonregular_source", "Expected regular file without special permission bits: " + relative)
            if before.st_size > limit:
                fail("size_limit", "File exceeds byte limit: " + relative)
            data = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
            if len(data) > limit:
                fail("size_limit", "File exceeds byte limit: " + relative)
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                fail("stale_source", "File changed during capture: " + relative)
            return data, stat.S_IMODE(before.st_mode)


def atomic_file(path, data, mode=0o600, immutable=False):
    descriptor, temporary = tempfile.mkstemp(prefix=".archive-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            os.fchmod(stream.fileno(), mode)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        with directory_fd(path.parent) as parent_descriptor:
            os.fsync(parent_descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bound_git_process():
    for kind, limit in ((resource.RLIMIT_CPU, 20), (resource.RLIMIT_AS, 512 * 1024 * 1024),
                        (resource.RLIMIT_FSIZE, 64 * 1024 * 1024), (resource.RLIMIT_CORE, 0)):
        unused_soft, hard = resource.getrlimit(kind)
        bounded = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(kind, (bounded, bounded))


def git(workspace, *arguments, input_data=None, output_limit=MAX_SOURCE_BYTES):
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    environment.update({"PATH": os.defpath, "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
                        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_ATTR_NOSYSTEM": "1",
                        "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
    command = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
               "-c", "core.attributesFile=/dev/null", "-c", "core.quotePath=true",
               "-c", "core.pager=cat", "-C", str(workspace)] + list(arguments)
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            completed = subprocess.run(command, input=input_data, stdout=output, stderr=errors,
                                       env=environment, timeout=30, check=False, preexec_fn=bound_git_process)
        except subprocess.TimeoutExpired:
            fail("git_timeout", "Git operation exceeded 30 seconds")
        except subprocess.SubprocessError:
            fail("git_error", "Git subprocess could not start with resource limits")
        output.seek(0)
        data = output.read(output_limit + 1)
        if len(data) > output_limit:
            fail("size_limit", "Git output exceeds byte limit")
        if completed.returncode:
            errors.seek(0)
            message = errors.read(2048).decode("utf-8", "replace").strip()
            fail("git_error", message or "Git operation failed")
        return data


def workspace_path(value, paths):
    workspace = absolute_path(value, required=True)
    with directory_fd(workspace):
        pass
    metadata = workspace / ".git"
    if metadata.is_symlink():
        fail("unsafe_path", "Workspace .git cannot be a symlink")
    top = git(workspace, "rev-parse", "--show-toplevel", output_limit=4096).decode().strip()
    if top != str(workspace):
        fail("invalid_workspace", "Workspace must be the Git working-tree root")
    ancestors = sorted({"/".join(path.split("/")[:depth]) for path in paths
                        for depth in range(1, len(path.split("/")) + 1)})
    tracked = git(workspace, "--literal-pathspecs", "ls-files", "--stage", "-z", "--", *ancestors,
                  output_limit=MAX_INDEX_BYTES)
    for record in tracked.split(b"\x00"):
        if record.startswith(b"160000 ") and record.split(b"\t", 1)[1].decode() in ancestors:
            fail("submodule_source", "Declared sources cannot include or traverse submodules")
    return workspace


def index_identity(workspace):
    location = git(workspace, "rev-parse", "--git-path", "index", output_limit=4096).decode().strip()
    path = absolute_path(workspace / location)
    captured = read_regular(path.parent, path.name, MAX_SOURCE_BYTES, missing=True)
    return digest(captured[0]) if captured is not None else None


def index_baseline(workspace, allowed_intent_to_add):
    records = git(workspace, "ls-files", "--stage", "-z", output_limit=MAX_INDEX_BYTES).split(b"\x00")
    retained = []
    allowed_paths = {path.encode("ascii") for path in allowed_intent_to_add}
    for record in records:
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        if raw_path in allowed_paths:
            debug = git(workspace, "--literal-pathspecs", "ls-files", "--debug", "--", raw_path.decode("ascii"),
                        output_limit=4096)
            flags = re.findall(rb"flags: ([0-9a-f]+)\n", debug)
            if len(flags) == 1 and int(flags[0], 16) & 0x20000000:
                mode, object_id, stage = header.split()
                empty_blob = git(workspace, "hash-object", "--stdin", input_data=b"").strip()
                if mode in (b"100644", b"100755") and stage == b"0" and object_id == empty_blob:
                    continue
                fail("stale_prepare_receipt", "Invalid intent-to-add placeholder")
        retained.append(record)
    try:
        head = git(workspace, "rev-parse", "--verify", "--quiet", "HEAD", output_limit=4096).decode().strip()
    except ArchiveError as error:
        if error.code != "git_error":
            raise
        head = None
    return {"entries_sha256": digest(b"\x00".join(retained)), "head": head,
            "allowed_intent_to_add": allowed_intent_to_add}


def capture(workspace, paths):
    files = {}
    blobs = {}
    total = 0
    for path in paths:
        captured = read_regular(workspace, path, MAX_SOURCE_BYTES - total, missing=True)
        if captured is None:
            files[path] = {"present": False}
        else:
            data, mode = captured
            total += len(data)
            identity = digest(data)
            blobs[identity] = data
            files[path] = {"present": True, "sha256": identity, "size": len(data),
                           "mode": 0o755 if mode & stat.S_IXUSR else 0o644}
    manifest = {"schema_version": SCHEMA_VERSION, "files": files}
    return manifest, blobs


class Store:
    def __init__(self, root):
        self.root = root
        self.objects = root / "objects"
        self.sizes = {}
        self.total_bytes = 0
        with directory_fd(self.objects):
            pass
        with os.scandir(self.objects) as objects:
            for item in objects:
                if len(self.sizes) >= MAX_OBJECTS or not DIGEST_PATTERN.fullmatch(item.name):
                    fail("corrupt_archive", "Invalid object inventory")
                data = self.get(item.name)
                self.sizes[item.name] = len(data)
                self.total_bytes += len(data)
                if self.total_bytes > MAX_OBJECT_BYTES:
                    fail("size_limit", "Archive object storage exceeds 32 MiB")

    def get(self, identity):
        if not isinstance(identity, str) or not DIGEST_PATTERN.fullmatch(identity):
            fail("corrupt_archive", "Invalid content identity")
        captured = read_regular(self.objects, identity, MAX_OBJECT_BYTES)
        if digest(captured[0]) != identity:
            fail("corrupt_archive", "Object SHA256 mismatch: " + identity)
        return captured[0]

    def put(self, data):
        identity = digest(data)
        if identity in self.sizes:
            if self.get(identity) != data:
                fail("corrupt_archive", "Content-addressed object mismatch")
            return identity
        if len(self.sizes) >= MAX_OBJECTS or self.total_bytes + len(data) > MAX_OBJECT_BYTES:
            fail("size_limit", "Archive immutable objects are limited to 32 MiB and 32768 objects")
        atomic_file(self.objects / identity, data, mode=0o444, immutable=True)
        self.sizes[identity] = len(data)
        self.total_bytes += len(data)
        return identity

    def save_source(self, manifest, blobs):
        encoded = json_bytes(manifest)
        pending = dict(blobs)
        pending[digest(encoded)] = encoded
        needed = {identity: data for identity, data in pending.items() if identity not in self.sizes}
        if (self.total_bytes + sum(len(data) for data in needed.values()) > MAX_OBJECT_BYTES
                or len(self.sizes) + len(needed) > MAX_OBJECTS):
            fail("size_limit", "Source would exceed archive storage limit")
        for identity in sorted(pending):
            self.put(pending[identity])
        return digest(encoded)

    def source(self, identity, paths):
        manifest = parse_json(self.get(identity))
        if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}:
            fail("corrupt_archive", "Invalid source manifest")
        files = manifest["files"]
        if manifest["schema_version"] != SCHEMA_VERSION or not isinstance(files, dict) or sorted(files) != paths:
            fail("stale_source", "Source manifest differs from declared scope")
        blobs = {}
        total = 0
        for metadata in files.values():
            if metadata == {"present": False}:
                continue
            if not isinstance(metadata, dict) or set(metadata) != {"present", "sha256", "size", "mode"}:
                fail("corrupt_archive", "Invalid source file metadata")
            if (metadata["present"] is not True or type(metadata["mode"]) is not int
                    or metadata["mode"] not in (0o644, 0o755) or type(metadata["size"]) is not int):
                fail("corrupt_archive", "Invalid regular file mode or size")
            data = self.get(metadata["sha256"])
            total += len(data)
            if len(data) != metadata["size"] or total > MAX_SOURCE_BYTES:
                fail("corrupt_archive", "Source size mismatch or source cap exceeded")
            blobs[metadata["sha256"]] = data
        return manifest, blobs


def validate_index(index):
    if not isinstance(index, dict):
        fail("corrupt_archive", "Invalid archive index")
    if index.get("schema_version") != SCHEMA_VERSION:
        fail("unsupported_archive_version", "Git-style source identity requires a fresh schema-2 archive; preserve older archives unchanged")
    required = {"schema_version", "source_paths", "initial_canonical", "original_benchmark",
                "entries", "reports", "preparations", "workspace_receipts", "batches"}
    if set(index) != required or not isinstance(index["initial_canonical"], str):
        fail("corrupt_archive", "Invalid archive index fields")
    if source_scope(index.get("source_paths")) != index["source_paths"]:
        fail("corrupt_archive", "Unsorted source scope")
    for name in ("entries", "reports", "preparations", "workspace_receipts", "batches"):
        if not isinstance(index.get(name), dict) or len(index[name]) > MAX_RECORDS:
            fail("index_limit", "Invalid or full index collection: " + name)
    for entry_id, entry in index["entries"].items():
        if (not isinstance(entry, dict) or entry.get("id") != entry_id
                or entry.get("source_retention") not in ("retained", "failed")):
            fail("corrupt_archive", "Invalid archive entry metadata")
    for receipt_id, receipt in index["preparations"].items():
        if not isinstance(receipt, dict) or digest(json_bytes(receipt)) != receipt_id:
            fail("corrupt_archive", "Invalid preparation receipt metadata")
    if len(json_bytes(index)) > MAX_INDEX_BYTES:
        fail("index_limit", "Index exceeds 8 MiB")


@contextlib.contextmanager
def locked_archive(value):
    root = absolute_path(value)
    with directory_fd(root) as root_descriptor:
        try:
            descriptor = os.open("lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 0o600, dir_fd=root_descriptor)
        except OSError:
            fail("unsafe_path", "Archive lock cannot be opened safely")
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                fail("unsafe_path", "Archive lock is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fail("archive_busy", "Another archive operation holds the lock; retry")
            captured = read_regular(root, "index.json", MAX_INDEX_BYTES, missing=True)
            index = parse_json(captured[0]) if captured else None
            if index is not None:
                validate_index(index)
            yield root, index
        finally:
            os.close(descriptor)


def save_index(root, index):
    validate_index(index)
    atomic_file(root / "index.json", json_bytes(index))


def require_index(index):
    if index is None:
        fail("uninitialized_archive", "Archive has no committed index")


def separate_archive(root, workspace):
    if root == workspace or root in workspace.parents or workspace in root.parents:
        fail("unsafe_path", "Archive and source workspace must be separate directory trees")


def resolve_entry(store, index, entry_id):
    identifier(entry_id)
    entry = index["entries"].get(entry_id)
    if not isinstance(entry, dict) or entry.get("source_retention") != "retained":
        fail("unknown_source", "Unknown or unretained source entry: " + entry_id)
    store.source(entry.get("source_id"), index["source_paths"])
    if entry.get("id") != entry_id:
        fail("corrupt_archive", "Entry identity mismatch")
    return entry


def donor_references(store, index, donors):
    if not isinstance(donors, list) or len(donors) > MAX_PATHS or len(set(map(str, donors))) != len(donors):
        fail("invalid_donors", "Donors must be a bounded list of unique retained entry IDs")
    return [{"id": donor, "source_id": resolve_entry(store, index, donor)["source_id"]}
            for donor in donors]


def initialize(archive, canonical, source_paths):
    paths = source_scope(source_paths)
    canonical = workspace_path(canonical, paths)
    manifest, blobs = capture(canonical, paths)
    root = absolute_path(archive, required=True)
    separate_archive(root, canonical)
    with directory_fd(root.parent) as parent_descriptor:
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_descriptor)
            created = True
        except FileExistsError:
            created = False
    with locked_archive(root) as (root, index):
        if index is None:
            if not created or set(os.listdir(root)) != {"lock"}:
                fail("init_conflict", "Initialization requires a new directory or an exact completed retry")
            os.mkdir(root / "objects", 0o700)
        store = Store(root)
        identity = digest(json_bytes(manifest))
        if index is not None:
            if (index["source_paths"] != paths or index.get("initial_canonical") != str(canonical)
                    or resolve_entry(store, index, "FOUNDER")["source_id"] != identity):
                fail("init_conflict", "Initialization differs from captured founder")
            return {"ok": True, "command": "init", "founder_source_id": identity,
                    "source_retention": "retained", "verification": "unknown", "reused": True}
        store.save_source(manifest, blobs)
        index = {"schema_version": SCHEMA_VERSION, "source_paths": paths, "initial_canonical": str(canonical),
                 "original_benchmark": {"status": "external_not_captured", "source_id": None,
                                        "note": "Main owns ORIGINAL; FOUNDER may include warmstart changes."},
                 "entries": {"FOUNDER": {"id": "FOUNDER", "kind": "founder", "source_id": identity,
                                         "source_retention": "retained", "verification": {"status": "unknown"}}},
                 "reports": {}, "preparations": {}, "workspace_receipts": {}, "batches": {}}
        save_index(root, index)
        return {"ok": True, "command": "init", "founder_source_id": identity,
                "source_retention": "retained", "verification": "unknown", "reused": False}


def query(archive, entry=None):
    with locked_archive(archive) as (root, index):
        require_index(index)
        store = Store(root)
        if entry is not None:
            identifier(entry)
            if entry not in index["entries"]:
                fail("unknown_source", "Unknown entry: " + entry)
            result = index["entries"][entry]
            details = {}
            if result.get("source_retention") == "retained":
                resolve_entry(store, index, entry)
                manifest, unused_blobs = store.source(result["source_id"], index["source_paths"])
                details = {"source_manifest": manifest, "source_manifest_path": str(store.objects / result["source_id"]),
                           "blob_directory": str(store.objects)}
            return {"ok": True, "command": "query", "entry": result, **details}
        for entry_id, value in index["entries"].items():
            if value.get("source_retention") == "retained":
                resolve_entry(store, index, entry_id)
        return {"ok": True, "command": "query", "index": index}


def overlay(workspace, manifest, blobs):
    for relative, metadata in sorted(manifest["files"].items()):
        with source_parent(workspace, relative, create=metadata["present"]) as (parent_descriptor, name):
            if parent_descriptor is None:
                continue
            if not metadata["present"]:
                try:
                    os.unlink(name, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
                continue
            temporary = ".evo-archive-" + digest(relative.encode())
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=parent_descriptor)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(blobs[metadata["sha256"]])
                    os.fchmod(stream.fileno(), metadata["mode"])
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, name, src_dir_fd=parent_descriptor, dst_dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass


def prepare(archive, workspace, parent="INCUMBENT", donors=None, initialize_index=False, canonical=None):
    with locked_archive(archive) as (root, index):
        require_index(index)
        store = Store(root)
        paths = index["source_paths"]
        workspace = absolute_path(workspace, required=True)
        with directory_fd(workspace):
            pass
        separate_archive(root, workspace)
        canonical_paths = {index["initial_canonical"]} | {
            batch["round_canonical"] for batch in index["batches"].values()}
        if str(workspace) in canonical_paths:
            fail("canonical_workspace", "Prepare requires a separate materialized workspace, not canonical")
        identifier(parent)
        references = donor_references(store, index, donors or [])
        missing_git = not os.path.lexists(workspace / ".git")
        if not missing_git or not initialize_index:
            workspace_path(workspace, paths)
        before, before_blobs = capture(workspace, paths)
        canonical_source_id = None
        if canonical is not None:
            canonical = workspace_path(canonical, paths)
            separate_archive(root, canonical)
            if canonical == workspace:
                fail("canonical_workspace", "Prepare requires a separate materialized workspace")
            canonical_manifest, unused_blobs = capture(canonical, paths)
            if before != canonical_manifest:
                fail("materialization_mismatch", "Declared source differs from round canonical before parent overlay")
            canonical_source_id = digest(json_bytes(canonical_manifest))
        if parent == "INCUMBENT":
            manifest, blobs = before, before_blobs
        else:
            selected = resolve_entry(store, index, parent)
            manifest, blobs = store.source(selected["source_id"], paths)
        source_id = store.save_source(manifest, blobs)
        if missing_git:
            git(workspace, "init", "--quiet", "--template=", "--object-format=sha1")
            seed_index(workspace, before, before_blobs)
        allowed_intent_to_add = [relative for relative, metadata in before["files"].items() if not metadata["present"]]
        baseline = index_baseline(workspace, allowed_intent_to_add)
        initialization = {"status": "created_declared_source_index" if missing_git else "existing_index_preserved"}
        if missing_git:
            initialization["base_source_id"] = store.save_source(before, before_blobs)
        previous_id = index["workspace_receipts"].get(str(workspace))
        if previous_id:
            previous = index["preparations"][previous_id]
            prior_baseline = previous["workspace_index_baseline"]
            if index_baseline(workspace, prior_baseline["allowed_intent_to_add"]) == prior_baseline:
                baseline = prior_baseline
                initialization = previous["index_initialization"]
        index_before = index_identity(workspace)
        receipt = {"workspace": str(workspace), "requested_parent_id": parent,
                   "materialized_canonical_source_id": canonical_source_id,
                   "prepared_source_id": source_id, "donors": references,
                   "workspace_index_sha256": index_before,
                   "workspace_index_baseline": baseline, "index_initialization": initialization,
                   "donor_policy": "references_only", "verification": "not_performed"}
        receipt_id = digest(json_bytes(receipt))
        index["preparations"][receipt_id] = receipt
        index["workspace_receipts"][str(workspace)] = receipt_id
        validate_index(index)
        store.put(json_bytes(receipt))
        if parent != "INCUMBENT":
            try:
                overlay(workspace, manifest, blobs)
                actual, unused_blobs = capture(workspace, paths)
                if digest(json_bytes(actual)) != source_id or index_identity(workspace) != index_before:
                    fail("stale_source", "Prepared source or Git index changed during preparation")
            except (ArchiveError, OSError):
                overlay(workspace, before, before_blobs)
                raise
        else:
            actual, unused_blobs = capture(workspace, paths)
            if digest(json_bytes(actual)) != source_id or index_identity(workspace) != index_before:
                fail("stale_source", "Incumbent source or Git index changed during preparation")
        save_index(root, index)
        return {"ok": True, "command": "prepare", "receipt_id": receipt_id, "receipt": receipt,
                "source_retention": "retained", "verification": "not_performed"}


def git_mode(metadata):
    return "100755" if metadata["mode"] & 0o100 else "100644"


def seed_index(repository, manifest, blobs):
    for relative, metadata in sorted(manifest["files"].items()):
        if metadata["present"]:
            object_id = git(repository, "hash-object", "-w", "--stdin",
                            input_data=blobs[metadata["sha256"]]).decode().strip()
            git(repository, "update-index", "--add", "--cacheinfo", git_mode(metadata), object_id, relative)


def patch_source(base, base_blobs, patch):
    if not patch.strip():
        fail("invalid_patch", "Best patch is empty")
    for line in patch.splitlines():
        mode_match = re.fullmatch(rb"(?:old mode|new mode|new file mode|deleted file mode) ([0-9]+)", line)
        index_match = re.fullmatch(rb"index [0-9a-f]+\.\.[0-9a-f]+ ([0-9]+)", line)
        matched = mode_match or index_match
        if matched and matched.group(1) not in (b"100644", b"100755"):
            fail("nonregular_patch", "Patch references a nonregular file mode")
    paths = sorted(base["files"])
    with tempfile.TemporaryDirectory(prefix="evo-archive-git-") as temporary:
        repository = Path(temporary)
        git(repository, "init", "--quiet", "--template=", "--object-format=sha1")
        seed_index(repository, base, base_blobs)
        summary = git(repository, "apply", "--numstat", "-z", "--", "-", input_data=patch,
                      output_limit=MAX_INDEX_BYTES)
        records = summary.split(b"\x00")
        position = 0
        touched = []
        while position < len(records) - 1:
            columns = records[position].split(b"\t", 2)
            position += 1
            if len(columns) != 3:
                fail("invalid_patch", "Unrecognized patch path summary")
            if columns[2]:
                touched.append(columns[2])
            else:
                if position + 1 >= len(records) - 1:
                    fail("invalid_patch", "Truncated rename path summary")
                touched.extend(records[position:position + 2])
                position += 2
        if not touched:
            fail("invalid_patch", "Best patch has no file changes")
        for raw_path in touched:
            try:
                relative = raw_path.decode("ascii")
            except UnicodeError:
                fail("out_of_scope_patch", "Patch path is outside declared ASCII source scope")
            source_path(relative)
            if relative not in paths:
                fail("out_of_scope_patch", "Patch touches undeclared source: " + relative)
        git(repository, "apply", "--cached", "--whitespace=nowarn", "--", "-", input_data=patch)
        staged = git(repository, "ls-files", "--stage", "-z", output_limit=MAX_INDEX_BYTES)
        files = {relative: {"present": False} for relative in paths}
        blobs = {}
        total = 0
        for record in staged.split(b"\x00"):
            if not record:
                continue
            header, raw_path = record.split(b"\t", 1)
            mode, object_id, stage = header.decode("ascii").split()
            relative = raw_path.decode("ascii")
            if mode not in ("100644", "100755") or stage != "0":
                fail("nonregular_patch", "Patch creates a symlink, submodule or nonregular mode")
            if relative not in files:
                fail("out_of_scope_patch", "Patch creates undeclared source: " + relative)
            data = git(repository, "cat-file", "blob", object_id, output_limit=MAX_SOURCE_BYTES - total)
            total += len(data)
            permissions = int(mode[-3:], 8)
            identity = digest(data)
            blobs[identity] = data
            files[relative] = {"present": True, "sha256": identity, "size": len(data), "mode": permissions}
        return {"schema_version": SCHEMA_VERSION, "files": files}, blobs


def parent_binding(store, index, report, workspace):
    parent = report.get("parent_id", "INCUMBENT")
    identifier(parent)
    donors = donor_references(store, index, report.get("donor_ids", []))
    if parent != "INCUMBENT":
        selected = resolve_entry(store, index, parent)
    receipt_id = index["workspace_receipts"].get(str(workspace))
    if receipt_id is None:
        if parent != "INCUMBENT" or donors:
            fail("missing_prepare_receipt", "Explicit parent/donors require a successful prepare receipt")
        return {"status": "unknown_no_prepare_receipt", "requested_parent_id": parent, "donors": donors}
    receipt = index["preparations"].get(receipt_id)
    if receipt is None or digest(json_bytes(receipt)) != receipt_id or store.get(receipt_id) != json_bytes(receipt):
        fail("corrupt_archive", "Preparation receipt identity mismatch")
    store.source(receipt["prepared_source_id"], index["source_paths"])
    if (receipt["requested_parent_id"] != parent or receipt["donors"] != donors
            or receipt["workspace"] != str(workspace)
            or (parent != "INCUMBENT" and receipt["prepared_source_id"] != selected["source_id"])):
        fail("stale_prepare_receipt", "Report parent/donors differ from actual preparation")
    baseline = receipt["workspace_index_baseline"]
    if index_baseline(workspace, baseline["allowed_intent_to_add"]) != baseline:
        fail("stale_prepare_receipt", "Workspace staged modes/blobs/stages or HEAD changed after prepare")
    return {"status": "prepared", "receipt_id": receipt_id, "prepared_source_id": receipt["prepared_source_id"],
            "requested_parent_id": parent, "donors": donors}


def check_patch(archive, canonical, patch):
    with locked_archive(archive) as (root, index):
        require_index(index)
        store = Store(root)
        canonical = workspace_path(canonical, index["source_paths"])
        separate_archive(root, canonical)
        base, base_blobs = capture(canonical, index["source_paths"])
        patch_path = absolute_path(patch, required=True)
        patch_data = read_regular(patch_path.parent, patch_path.name, MAX_PATCH_BYTES)[0]
        manifest, blobs = patch_source(base, base_blobs, patch_data)
        base_id = store.save_source(base, base_blobs)
        source_id = store.save_source(manifest, blobs)
        patch_id = store.put(patch_data)
        patch_object_path = str(store.objects / patch_id)
        receipt = {"patch_sha256": patch_id, "round_canonical_source_id": base_id,
                   "canonical_source_id": base_id, "source_id": source_id, "source_paths": index["source_paths"],
                   "canonical": str(canonical), "patch": str(patch_path), "patch_object_path": patch_object_path}
        receipt_id = store.put(json_bytes(receipt))
        return {"ok": True, "command": "check-patch", "receipt_id": receipt_id,
                "receipt": receipt, "source_scope_valid": True,
                "patch_object_path": patch_object_path, "patch_sha256": patch_id,
                "canonical_source_id": base_id, "source_id": source_id, "entry_recorded": False,
                "source_retention": "retained", "verification": "not_performed"}


def observe_report(store, index, report, base_id, base, base_blobs):
    report_id = identifier(report["id"], report=True)
    if len(json_bytes(report)) > MAX_REPORT_BYTES // 4:
        fail("size_limit", "Individual report exceeds 256 KiB")
    workspace = absolute_path(report.get("workspace"), required=True)
    binding = parent_binding(store, index, report, workspace)
    entries = []
    for kind, suffix in (("best_patch", "best"), ("development", "development")):
        entry = {"id": report_id + "-" + suffix, "report_id": report_id, "kind": kind,
                 "title": report.get("title"), "parent_binding": binding,
                 "round_canonical_source_id": base_id, "workspace": str(workspace),
                 "source_retention": "failed", "source_id": None,
                 "author_reported_evidence": {"engineer": report.get("engineer"), "verifier": report.get("verifier"),
                                              "label": "author_reported_not_independently_validated"},
                 "verification": {"status": "unknown", "independently_verified": False},
                 "evidence_source_binding_caveat": SOURCE_CAVEAT}
        try:
            if kind == "best_patch":
                patch_value = report.get("patch")
                if not patch_value:
                    fail("missing_patch", "No best-patch path was supplied")
                patch_path = absolute_path(patch_value, required=True)
                patch_data = read_regular(patch_path.parent, patch_path.name, MAX_PATCH_BYTES)[0]
                entry["patch_sha256"] = store.put(patch_data)
                entry["patch_path"] = str(patch_path)
                manifest, blobs = patch_source(base, base_blobs, patch_data)
            else:
                workspace_path(workspace, index["source_paths"])
                manifest, blobs = capture(workspace, index["source_paths"])
            entry["source_id"] = store.save_source(manifest, blobs)
            entry["source_retention"] = "retained"
            entry["verification"] = {
                "status": "author_reported_only" if kind == "best_patch" and report.get("verifier") is not None
                else ("unverified_development" if kind == "development" else "unknown"),
                "independently_verified": False,
                "evidence_applies_to": "bound_best_patch_only" if kind == "best_patch" else "no_measurement_binding",
                "bound_source_id": entry["source_id"] if kind == "best_patch" else None,
            }
        except (ArchiveError, OSError) as error:
            entry["diagnostic"] = error.diagnostic() if isinstance(error, ArchiveError) else {
                "code": "filesystem_error", "message": str(error)}
        entries.append(entry)
    return entries


def record_batch(archive, canonical, report_json):
    if not isinstance(report_json, str) or len(report_json.encode("utf-8")) > MAX_REPORT_BYTES:
        fail("size_limit", "Batch JSON exceeds 1 MiB")
    reports = parse_json(report_json)
    if not isinstance(reports, list) or not 1 <= len(reports) <= MAX_REPORTS:
        fail("invalid_reports", "Report JSON must be a list of 1 to 128 report objects")
    with locked_archive(archive) as (root, index):
        require_index(index)
        store = Store(root)
        canonical = workspace_path(canonical, index["source_paths"])
        separate_archive(root, canonical)
        base, base_blobs = capture(canonical, index["source_paths"])
        base_id = store.save_source(base, base_blobs)
        raw_id = store.put(report_json.encode("utf-8"))
        counts = collections.Counter(report.get("id") for report in reports
                                     if isinstance(report, dict) and isinstance(report.get("id"), str))
        results = []
        for ordinal, report in enumerate(reports):
            result = {"ordinal": ordinal, "id": report.get("id") if isinstance(report, dict) else None,
                      "status": "failed", "entries": [], "source_retention": {}, "verification": "not_performed"}
            try:
                if not isinstance(report, dict) or "id" not in report:
                    fail("invalid_report", "Each report must be an object with an ID")
                report_id = identifier(report["id"], report=True)
                if counts[report_id] != 1:
                    fail("duplicate_id", "Duplicate report ID within batch: " + report_id)
                entries = observe_report(store, index, report, base_id, base, base_blobs)
                retained_report = {"report": report, "round_canonical": str(canonical),
                                   "round_canonical_source_id": base_id, "entries": entries}
                previous = index["reports"].get(report_id)
                if previous is not None and previous != retained_report:
                    fail("id_conflict", "Report ID already binds different bytes, evidence or context: " + report_id)
                for entry in entries:
                    previous_entry = index["entries"].get(entry["id"])
                    if previous_entry is not None and previous_entry != entry:
                        fail("id_conflict", "Entry ID already binds different content: " + entry["id"])
                index["reports"][report_id] = retained_report
                for entry in entries:
                    index["entries"][entry["id"]] = entry
                result["entries"] = [entry["id"] for entry in entries]
                result["source_retention"] = {entry["kind"]: entry["source_retention"] for entry in entries}
                result["diagnostics"] = [{"entry_id": entry["id"], **entry["diagnostic"]}
                                         for entry in entries if "diagnostic" in entry]
                result["status"] = "recorded" if not result["diagnostics"] else (
                    "partial" if "retained" in result["source_retention"].values() else "failed")
            except (ArchiveError, OSError) as error:
                result["diagnostics"] = [error.diagnostic() if isinstance(error, ArchiveError)
                                         else {"code": "filesystem_error", "message": str(error)}]
                if isinstance(error, ArchiveError) and error.code in ("id_conflict", "duplicate_id"):
                    result["status"] = "conflict"
            results.append(result)
        batch = {"round_canonical": str(canonical), "round_canonical_source_id": base_id,
                 "raw_report_json_sha256": raw_id, "results": results}
        batch_id = digest(json_bytes(batch))
        index["batches"][batch_id] = batch
        save_index(root, index)
        return {"ok": True, "command": "record-batch", "batch_id": batch_id,
                "has_failures": any(result["status"] != "recorded" for result in results),
                "results": results, "verification": "not_performed"}


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        fail("arguments", message)


def main(argv=None):
    try:
        parser = JsonParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True, parser_class=JsonParser)
        initialize_parser = commands.add_parser("init")
        initialize_parser.add_argument("--archive", required=True)
        initialize_parser.add_argument("--canonical", required=True)
        initialize_parser.add_argument("--source-path", action="append", required=True)
        query_parser = commands.add_parser("query")
        query_parser.add_argument("--archive", required=True)
        query_parser.add_argument("--entry")
        prepare_parser = commands.add_parser("prepare")
        prepare_parser.add_argument("--archive", required=True)
        prepare_parser.add_argument("--workspace", required=True)
        prepare_parser.add_argument("--canonical")
        prepare_parser.add_argument("--parent", default="INCUMBENT")
        prepare_parser.add_argument("--donor", action="append", default=[])
        prepare_parser.add_argument("--initialize-index", action="store_true")
        record_parser = commands.add_parser("record-batch")
        record_parser.add_argument("--archive", required=True)
        record_parser.add_argument("--canonical", required=True)
        record_parser.add_argument("--report-json", required=True)
        check_parser = commands.add_parser("check-patch")
        check_parser.add_argument("--archive", required=True)
        check_parser.add_argument("--canonical", required=True)
        check_parser.add_argument("--patch", required=True)
        arguments = parser.parse_args(argv)
        if arguments.command == "init":
            result = initialize(arguments.archive, arguments.canonical, arguments.source_path)
        elif arguments.command == "query":
            result = query(arguments.archive, arguments.entry)
        elif arguments.command == "prepare":
            result = prepare(arguments.archive, arguments.workspace, arguments.parent, arguments.donor,
                             arguments.initialize_index, arguments.canonical)
        elif arguments.command == "check-patch":
            result = check_patch(arguments.archive, arguments.canonical, arguments.patch)
        else:
            result = record_batch(arguments.archive, arguments.canonical, arguments.report_json)
        result["archive"] = str(absolute_path(arguments.archive))
        print(json_bytes(result).decode("ascii"))
        return 0
    except (ArchiveError, OSError, ValueError, RecursionError) as error:
        diagnostic = error.diagnostic() if isinstance(error, ArchiveError) else {
            "code": "filesystem_or_input_error", "message": str(error)}
        print(json_bytes({"ok": False, "error": diagnostic, "verification": "not_performed"}).decode("ascii"))
        return 2
    except (KeyError, TypeError, AttributeError):
        print(json_bytes({"ok": False, "error": {"code": "invalid_metadata", "message": "Malformed archive or input metadata"},
                          "verification": "not_performed"}).decode("ascii"))
        return 2


if __name__ == "__main__":
    sys.exit(main())
