"""Native transport packing; callers authenticate the original bundle first.

No runtime imports, filesystem operations, lifecycle actions, or input mutation.
The internal status is not a new receipt field. A required_overflow response is
the full failure bundle for trusted handling, never a bounded public payload.
"""
import copy
import json

from .._util import canonical_json, sha256_hex


MAX_ARTIFACT_COUNT = 4096
MAX_ARTIFACT_BYTES = 64 * 1024**2
MAX_ARTIFACT_TOTAL_BYTES = 256 * 1024**2


def artifacts_fit(artifacts, *, max_count=MAX_ARTIFACT_COUNT,
                  max_file_bytes=MAX_ARTIFACT_BYTES, max_total_bytes=MAX_ARTIFACT_TOTAL_BYTES):
    return (type(artifacts) is dict and len(artifacts) <= max_count
            and all(type(raw) is bytes and len(raw) <= max_file_bytes for raw in artifacts.values())
            and sum(map(len, artifacts.values())) <= max_total_bytes)


def canonical(value):
    return canonical_json(value)


def digest(raw):
    return sha256_hex(raw)


def pack(raw_artifacts, host_receipt, arena_result, request, *, max_count=MAX_ARTIFACT_COUNT,
         max_file_bytes=MAX_ARTIFACT_BYTES, max_total_bytes=MAX_ARTIFACT_TOTAL_BYTES):
    """Return {status, response}, preserving the original lifecycle conclusion.

    Original host receipt, raw map, and Arena result must already be validated.
    The caller first retains canonical(host_receipt)+b"\\n" at the existing
    collection/unpacked_host_receipt.json, then writes projected receipt bytes.
    The host evidence envelope is already in raw_artifacts; three evaluator records
    are added here. Optional retained-build files are the last eviction group.
    """
    if any(type(n) is not int or n < 0 for n in
           (max_count, max_file_bytes, max_total_bytes)):
        raise ValueError("nonnegative integer transport limits required")
    original = dict(raw_artifacts)
    hashes = {name: digest(raw) for name, raw in original.items()}
    original_receipt_sha = digest(canonical(host_receipt) + b"\n")
    refs = set(host_receipt["cleanup"]["evidence"])
    refs.update(host_receipt["runtime"]["evidence"])
    for row in host_receipt["observations"].values():
        refs.update(row["evidence"])
    for row in arena_result.get("native_observations", {}).values():
        refs.update(row["evidence"])
    refs.update(arena_result.get("observed_runtime", {}).get("evidence", []))
    if not refs <= original.keys() or hashes != host_receipt["artifacts_sha256"]:
        raise ValueError("original evidence closure/hash map differs")
    def producer_report(name):
        parts = name.split("/")
        return (parts[0] == "native" and
                ((len(parts) in (2, 3) and parts[-1].endswith(".json")) or
                 (len(parts) >= 4 and parts[2] == "raw-reports")))

    required = refs | {name for name in original
                       if name.startswith(("host/", "observer/")) or producer_report(name)}
    optional = sorted(original.keys() - required,
                      key=lambda name: ("retained-build" in name.split("/"), -len(name.split("/")), name))
    kept, omitted, cursor = dict(original), [], 0

    def build(selected, inventory, overflow=False):
        host, result = copy.deepcopy(host_receipt), copy.deepcopy(arena_result)
        host["artifacts_sha256"] = {name: hashes[name] for name in sorted(selected)}
        if inventory or overflow:
            collection = result["collection"]
            location = json.dumps(collection)
            if len(location) > 4096:
                location = "see receipt.collection; path_sha256=" + digest(canonical(collection))
            summary = ("Transport incomplete: " +
                       ("required bundle exceeds bounds; full bundle returned for trusted handling; "
                        if overflow else "optional payloads omitted; ") +
                       f"omitted_count={len(inventory)}; omitted_bytes={sum(r[1] for r in inventory)}; "
                       "omitted_inventory_sha256=" + digest(canonical(sorted(inventory))) +
                       "; original_collection=" + location +
                       "; original_host_receipt_sha256=" + original_receipt_sha +
                       "; original receipt: collection/unpacked_host_receipt.json; "
                       "returned host receipt is the transport projection" +
                       "; packing changed no owner files or lifecycle conclusion.")
            host["errors"].append(summary)
            result["host_errors"] = list(host["errors"])
            result["native_suite_pass_observed"] = False
        artifacts = {"raw/" + name: selected[name] for name in sorted(selected)}
        artifacts.update({"host_receipt.json": canonical(host) + b"\n",
                          "arena_evaluation.json": canonical(result) + b"\n",
                          "request.json": canonical(request) + b"\n"})
        return {"receipt": result, "artifacts": artifacts,
                "costs": copy.deepcopy(result["costs"])}

    while True:
        response = build(kept, omitted)
        sizes = [len(raw) for raw in response["artifacts"].values()]
        total = sum(sizes)
        if artifacts_fit(response["artifacts"], max_count=max_count,
                         max_file_bytes=max_file_bytes, max_total_bytes=max_total_bytes):
            return {"status": "incomplete" if omitted else "complete", "response": response}
        if cursor == len(optional):
            return {"status": "required_overflow", "response": build(original, [], overflow=True)}
        # Batch count/byte deficits, then serialize again to include exact JSON
        # sizes, hashes, and the one bounded omission summary in both receipts.
        remove_count, deficit = max(1, len(sizes) - max_count), max(0, total - max_total_bytes)
        removed = freed = 0
        while cursor < len(optional) and (removed < remove_count or freed < deficit):
            name = optional[cursor]
            cursor += 1
            raw = kept.pop(name)
            omitted.append([name, len(raw), hashes[name]])
            removed += 1
            freed += len(raw)
            # Recount serialized metadata before entering the preferred group:
            # even a zero-byte removal can save a large hash-map entry.
            if (cursor < len(optional) and "retained-build" not in name.split("/")
                    and "retained-build" in optional[cursor].split("/")):
                break
