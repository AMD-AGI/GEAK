#!/usr/bin/env python3
"""Validate and read frozen native planner metadata; never execute its payloads."""

import copy
import hashlib
import json
import math
import os
import re
import stat
import sys


MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = MAX_OUTPUT_BYTES
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_READ_BYTES = 256 * 1024 * 1024
MAX_REFS = 1024
MAX_SAFE_INTEGER = 9007199254740991
REF_KEYS = {"path", "sha256", "bytes"}
SEED_KEYS = {
    "schema", "id", "original_archive", "archive", "source_refs", "task_root",
    "task_files", "context",
}
OPTIONAL_SEED_KEYS = {"context_refs"}
CONTEXT_KEYS = {
    "eval_dir", "round", "budget_remaining", "cumulative_speedup",
    "baseline_geomean_ms", "baseline_per_case", "current_best_per_case",
    "current_best_ref", "final_winner", "profile_summary", "history", "gfx",
    "device_target", "physical_cu_count", "roofline_status", "kk_operator",
    "kk_language", "kk_refs", "research_brief_path", "facts",
}
OPTIONAL_CONTEXT_KEYS = {"profile_evidence_inputs", "initial_original"}


class SeedError(Exception):
    """A fixed, payload-free validation error."""


def require(condition, code):
    if not condition:
        raise SeedError(code)


def canonical_path(value, *, root_allowed=False):
    require(type(value) is str and value.startswith("/")
            and not any(ord(char) < 32 or ord(char) == 127 for char in value),
            "invalid_path")
    if root_allowed and value == "/":
        return value
    require(all(part and part not in (".", "..") for part in value[1:].split("/")),
            "noncanonical_path")
    return value


def reference(value, *, source=False):
    require(type(value) is dict
            and set(value) == REF_KEYS | ({"entry_id"} if source else set()),
            "invalid_reference_fields")
    canonical_path(value["path"])
    require(type(value["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
            "invalid_reference_hash")
    require(type(value["bytes"]) is int and 0 <= value["bytes"] <= MAX_FILE_BYTES,
            "invalid_reference_size")
    if source:
        require(type(value["entry_id"]) is str and bool(value["entry_id"]),
                "invalid_source_entry_id")
    return value


def _open_path(path, *, directory=False):
    """Open each path component without following links, using Linux O_PATH.

    O_PATH pins a leaf without opening a device or blocking on a FIFO. A checked
    regular leaf can then be read through its own /proc/self/fd descriptor.
    """
    canonical_path(path, root_allowed=directory)
    flags = os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags | os.O_DIRECTORY)
    try:
        parts = [] if path == "/" else path[1:].split("/")
        for index, part in enumerate(parts):
            child_flags = flags
            if index < len(parts) - 1 or directory:
                child_flags |= os.O_DIRECTORY
            child = os.open(part, child_flags, dir_fd=fd)
            os.close(fd)
            fd = child
        mode = os.fstat(fd).st_mode
        require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode),
                "not_directory" if directory else "not_regular_file")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_uid,
            info.st_gid, info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)


class Reader:
    def __init__(self):
        self.verified_refs = []
        self.total_bytes = 0
        self.path_refs = {}

    def read(self, ref, *, source=False, json_file=False):
        reference(ref, source=source)
        require(len(self.verified_refs) < MAX_REFS, "reference_count_limit")
        size = ref["bytes"]
        require(not json_file or size <= MAX_JSON_BYTES, "json_size_limit")
        require(self.total_bytes + size <= MAX_TOTAL_READ_BYTES, "read_size_limit")
        prior = self.path_refs.get(ref["path"])
        identity = (ref["sha256"], size)
        require(prior is None or prior == identity, "conflicting_path_references")
        path_fd = _open_path(ref["path"])
        read_fd = None
        try:
            before = os.fstat(path_fd)
            require(before.st_size == size, "file_size_mismatch")
            read_fd = os.open(f"/proc/self/fd/{path_fd}",
                              os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            require(_identity(os.fstat(read_fd)) == _identity(before), "file_changed")
            digest = hashlib.sha256()
            chunks = [] if json_file else None
            count = 0
            while count <= size:
                chunk = os.read(read_fd, min(262144, size + 1 - count))
                if not chunk:
                    break
                count += len(chunk)
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            require(count == size, "file_size_mismatch")
            require(_identity(os.fstat(read_fd)) == _identity(before)
                    and _identity(os.fstat(path_fd)) == _identity(before), "file_changed")
            current_fd = _open_path(ref["path"])
            try:
                require(_identity(os.fstat(current_fd)) == _identity(before), "path_changed")
            finally:
                os.close(current_fd)
            require(digest.hexdigest() == ref["sha256"], "file_hash_mismatch")
        finally:
            if read_fd is not None:
                os.close(read_fd)
            os.close(path_fd)
        self.total_bytes += size
        self.path_refs[ref["path"]] = identity
        self.verified_refs.append(copy.deepcopy(ref))
        return decode_json(b"".join(chunks)) if json_file else None


def decode_json(raw):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def floating(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite_json_number")
        return number

    def invalid_constant(_value):
        raise SeedError("nonfinite_json_number")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs,
                          parse_float=floating, parse_constant=invalid_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise SeedError("invalid_json") from error


def same_json(left, right):
    """Preserve missingness, order, and types (not Python's True == 1)."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(same_json(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(same_json(a, b) for a, b in zip(left, right))
    return left == right


def archive_entries(archive):
    require(type(archive) is dict and archive.get("schema") == "geak.evolution/1",
            "invalid_archive_schema")
    rows = archive.get("entries")
    require(type(rows) is list and bool(rows), "empty_archive")
    require(len(rows) <= MAX_REFS, "archive_entry_limit")
    entries = {}
    for row in rows:
        require(type(row) is dict and type(row.get("id")) is str and bool(row["id"]),
                "invalid_archive_entry")
        require(row["id"] != "ORIGINAL" and row["id"] not in entries,
                "duplicate_or_reserved_archive_id")
        entries[row["id"]] = row
        patch = row.get("patch_ref")
        if patch is not None:
            require(type(patch) is dict and type(patch.get("path")) is str
                    and bool(patch["path"]) and type(patch.get("sha256")) is str
                    and re.fullmatch(r"[0-9a-f]{64}", patch["sha256"]) is not None,
                    "invalid_archive_patch_reference")
    require(type(archive.get("best_id")) is str
            and archive["best_id"] in {"ORIGINAL", *entries}, "invalid_archive_best_id")
    return entries


def positive_number(value, *, integer=False, nullable=False):
    if nullable and value is None:
        return True
    if integer:
        return type(value) is int and 0 < value <= MAX_SAFE_INTEGER
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_context(context, entries):
    require(type(context) is dict and CONTEXT_KEYS <= set(context)
            and set(context) <= CONTEXT_KEYS | OPTIONAL_CONTEXT_KEYS, "invalid_context_fields")
    canonical_path(context["eval_dir"])
    require(positive_number(context["round"], integer=True), "invalid_context_positive_integer")
    require(positive_number(context["budget_remaining"], integer=True, nullable=True),
            "invalid_context_positive_integer")
    for key in ("cumulative_speedup", "baseline_geomean_ms"):
        require(positive_number(context[key], nullable=True), "invalid_context_scalar")
    require(positive_number(context["physical_cu_count"], integer=True, nullable=True),
            "invalid_context_cu_count")
    for key in ("baseline_per_case", "current_best_per_case"):
        rows = context[key]
        require(rows is None or (type(rows) is list and bool(rows)
                                and all(type(row) is dict and bool(row) for row in rows)),
                "invalid_context_case_rows")
    for key in ("profile_summary", "history"):
        value = context[key]
        require(type(value) is dict or (key == "profile_summary" and value is None),
                "invalid_observation_context")
        if type(value) is dict and "unavailable" in value:
            require(set(value) in ({"unavailable"}, {"unavailable", "scope"})
                    and type(value["unavailable"]) is str and bool(value["unavailable"].strip())
                    and ("scope" not in value or (type(value["scope"]) is str and bool(value["scope"].strip()))),
                    "invalid_unavailable_context")
    for key in ("current_best_ref", "final_winner"):
        value = context[key]
        require(value is None or type(value) is dict, "invalid_current_best_context")
        if value is not None:
            for identity_key in ("id", "entry_id", "candidate_id"):
                if identity_key in value:
                    require(type(value[identity_key]) is str
                            and value[identity_key] in {"ORIGINAL", *entries},
                            "unknown_context_entry_id")
            identities = [value[key] for key in ("id", "entry_id", "candidate_id") if key in value]
            require(len(set(identities)) <= 1, "conflicting_context_entry_ids")
    for key in ("gfx", "device_target", "roofline_status", "kk_operator", "kk_language",
                "research_brief_path"):
        require(context[key] is None or type(context[key]) is str, "invalid_context_string")
    require(type(context["kk_refs"]) is list
            and all(type(item) is str for item in context["kk_refs"]), "invalid_context_knowledge_refs")
    require(type(context["facts"]) is dict, "invalid_context_facts")
    if "initial_original" in context:
        require(type(context["initial_original"]) is bool, "invalid_initial_original")
        if context["initial_original"]:
            require(context["final_winner"] is None and context["cumulative_speedup"] == 1
                    and context["baseline_per_case"] is not None
                    and same_json(context["baseline_per_case"], context["current_best_per_case"]),
                    "initial_original_context_mismatch")
    if "profile_evidence_inputs" in context:
        value = context["profile_evidence_inputs"]
        require(type(value) is dict
                and set(value) <= {"PROFILE_SUMMARY", "EVIDENCE_STATUS", "EVIDENCE_HANDOFF"},
                "invalid_profile_evidence_inputs")
        for key, item in value.items():
            require(type(item) is dict or (key == "PROFILE_SUMMARY" and item is None),
                    "invalid_profile_evidence_value")


def read_seed(seed_ref):
    reader = Reader()
    seed = reader.read(seed_ref, json_file=True)
    require(type(seed) is dict and SEED_KEYS <= set(seed)
            and set(seed) <= SEED_KEYS | OPTIONAL_SEED_KEYS
            and seed["schema"] == "geak.native_planner_seed/1", "invalid_seed_schema")
    require(type(seed["id"]) is str and bool(seed["id"].strip()), "invalid_seed_id")
    canonical_path(seed["task_root"])
    task_fd = _open_path(seed["task_root"], directory=True)
    os.close(task_fd)
    require(type(seed["source_refs"]) is list and type(seed["task_files"]) is list
            and bool(seed["task_files"]), "invalid_seed_ref_lists")
    context_refs = seed.get("context_refs", [])
    require(type(context_refs) is list, "invalid_context_refs")
    archive_ref_count = 0 if seed["archive"] is None and seed["original_archive"] is None else 2
    require(1 + archive_ref_count + len(seed["source_refs"]) + len(seed["task_files"])
            + len(context_refs) <= MAX_REFS,
            "reference_count_limit")
    if seed["archive"] is None or seed["original_archive"] is None:
        require(seed["archive"] is None and seed["original_archive"] is None
                and not seed["source_refs"], "inconsistent_absent_archive")
        archive, entries = None, {}
    else:
        original = reader.read(seed["original_archive"], json_file=True)
        archive = reader.read(seed["archive"], json_file=True)
        entries = archive_entries(original)
        archive_entries(archive)
        expected = copy.deepcopy(original)
        expected["base"] = seed["task_root"]
        expected["base_commit"] = None
        if type(expected.get("baseline")) is dict and "source" in expected["baseline"]:
            expected["baseline"]["source"] = seed["task_root"]
        source_refs = {}
        for ref in seed["source_refs"]:
            reference(ref, source=True)
            entry_id = ref["entry_id"]
            require(entry_id not in source_refs, "duplicate_source_entry_id")
            source_refs[entry_id] = ref
        patch_ids = {row["id"] for row in original["entries"] if row.get("patch_ref") is not None}
        require(bool(patch_ids) and set(source_refs) == patch_ids, "source_reference_closure_mismatch")
        for row in expected["entries"]:
            if "base_commit" in row:
                row["base_commit"] = None
            patch = row.get("patch_ref")
            if patch is not None:
                ref = source_refs[row["id"]]
                require(ref["sha256"] == patch["sha256"], "original_patch_hash_mismatch")
                if "bytes" in patch:
                    require(type(patch["bytes"]) is int and patch["bytes"] == ref["bytes"],
                            "original_patch_size_mismatch")
                patch["path"] = ref["path"]
                if "base_commit" in patch:
                    patch["base_commit"] = None
        require(same_json(expected, archive), "archive_view_mismatch")
    validate_context(seed["context"], entries)
    task_paths = set()
    prefix = seed["task_root"] + "/"
    for ref in seed["task_files"]:
        reference(ref)
        require(ref["path"].startswith(prefix), "task_reference_outside_root")
        relative = ref["path"][len(prefix):]
        require(relative not in task_paths, "duplicate_task_reference")
        task_paths.add(relative)
    # Bind the declared files without imposing one task layout. This is a
    # decision context, not certification of an executable replay dependency closure.
    for ref in seed["source_refs"]:
        reader.read(ref, source=True)
    for ref in seed["task_files"]:
        reader.read(ref)
    for ref in context_refs:
        reader.read(ref)
    return {"seed": seed, "archive": archive, "verified_refs": reader.verified_refs}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(len(argv) == 3, "expected_path_sha256_bytes")
        require(re.fullmatch(r"0|[1-9][0-9]*", argv[2]) is not None, "invalid_cli_size")
        seed_ref = {"path": argv[0], "sha256": argv[1], "bytes": int(argv[2])}
        result = read_seed(seed_ref)
        output = (json.dumps(result, ensure_ascii=True, allow_nan=False,
                             separators=(",", ":")) + "\n").encode("utf-8")
        require(len(output) <= MAX_OUTPUT_BYTES, "output_size_limit")
        sys.stdout.buffer.write(output)
        return 0
    except (SeedError, OSError, ValueError, OverflowError, RecursionError) as error:
        code = str(error) if isinstance(error, SeedError) else "metadata_read_failed"
        sys.stderr.write("native planner seed rejected: " + code + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
