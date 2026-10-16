"""Build exact planner evidence views, claim context and bounded navigation.

NativePlanner uses the retained freeze builder; transport and recovery live in
planner.native and planner.model.
"""

import copy
from decimal import Decimal
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from .._util import require
from ..archive.refs import file_ref, read_ref as _read_ref
from ..archive.store import _write, _sync_directory
from ..archive.policy import lineage_candidates, policy_contract, validate_action, validate_read_scope


def _json(value):
    def decimal(item):
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError("non-JSON planner value")
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=decimal).encode()


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate proposal field", ValueError)
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    _json(value)
    return value


def _put(path, value):
    _write(path, value if type(value) is bytes else _json(value), immutable=True)


def _put_ref(path, ref):
    """Materialize exact bytes in a distinct inode; CoW is an optional optimization."""
    raw = _read_ref(ref)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w+b") as output:
            source_fd = os.open(ref["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                require(stat.S_ISREG(os.fstat(source_fd).st_mode), "reference must be a regular file", ValueError)
                try:
                    # Linux FICLONE clones extents into our own newly created inode.
                    fcntl.ioctl(output.fileno(), 0x40049409, source_fd)
                except OSError as exc:
                    if exc.errno not in (errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTTY,
                                         errno.EINVAL, errno.ENOSYS):
                        raise
                    # A failed ioctl must not leave bytes or offsets in the fallback.
                    output.seek(0)
                    output.truncate(0)
                    output.write(raw)
                else:
                    # The source may have changed after _read_ref; authenticate the clone.
                    output.seek(0)
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: output.read(1024 * 1024), b""):
                        digest.update(chunk)
                    require(digest.hexdigest() == ref["sha256"], "cloned immutable reference changed", ValueError)
            finally:
                os.close(source_fd)
            output.flush()
            os.fchmod(output.fileno(), 0o444)
            os.fsync(output.fileno())
        os.link(temporary, path)  # Publish our temporary inode; never link to the source.
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# Keep model context bounded independently of the retained archive size.
DECISION_NAVIGATION_BYTES = 24 * 1024
DECISION_PROMPT_BYTES = 64 * 1024
DECISION_RETRIEVAL_SCOPE = """The decision navigation preserves every field of the full frozen brief. Only
fields named in _navigation.externalized_fields use navigation wrappers: their
full_value_ref stores the exact JSON field value. Other fields are ordinary
inline data even if they contain a key named full_value_ref. Wrapper type, byte
size and item count are navigation, not a scientific ranking. Follow
_navigation.complete_brief_ref only with targeted JSON queries.
Do not print whole catalogs, indexes, snapshots, briefs or logs. Use Bash/Python
to inspect keys/counts, then query fields, exact identities or bounded ranges.
Keep each tool response within 8 KiB and each textual excerpt within 4 KiB;
include the source reference and omitted item count. Continue with another
bounded query when necessary. Do not hide failures or infer missing evidence
from an unqueried range. Code and hardware evidence remain available for exact
inspection; retrieve relevant functions or ranges with the same output bounds.
Choose parents, donors, corrections and research direction yourself. Catalog
position and inline status do not express host preference. Read current budget,
action contract and applicable corrections before deciding. These output bounds
are instructions, not a guarantee that the CLI will enforce a context budget.
"""


def _bounded_decision_navigation(brief_text, public, index):
    """Move largest exact fields to files until the decision navigation fits."""
    brief = _decode(brief_text.encode())
    require(type(brief) is dict and "_navigation" not in brief,
          "frozen decision brief must be an object without reserved navigation metadata", ValueError)
    full_ref = file_ref(public / "brief.json")
    require(_read_ref(full_ref) == brief_text.encode(), "full frozen decision brief differs", ValueError)
    index.append({"original_ref": full_ref, "readable_ref": full_ref})
    navigation = dict(brief)
    navigation["_navigation"] = {
        "schema": "geak-native-decision-navigation-v2",
        "complete_brief_ref": full_ref,
        "field_count": len(brief),
        "externalized_fields": [],
        "navigation_byte_limit": DECISION_NAVIGATION_BYTES,
        "selection": "All fields retained; largest serialized value first, ties by field name, until navigation fits."}
    values = [(name, _json(value)) for name, value in brief.items()]
    rendered = _json(navigation).decode()
    for name, raw in sorted(values, key=lambda item: (-len(item[1]), item[0])):
        if len(rendered.encode()) <= DECISION_NAVIGATION_BYTES:
            break
        destination = public / hashlib.sha256(raw).hexdigest()
        if not destination.exists():
            _put(destination, raw)
        reference = file_ref(destination)
        require(_read_ref(reference) == raw, "decision field differs from frozen exact value", ValueError)
        index.append({"original_ref": reference, "readable_ref": reference})
        value = brief[name]
        kind = ("array" if type(value) is list else "object" if type(value) is dict
                else "string" if type(value) is str else "scalar")
        navigation["_navigation"]["externalized_fields"].append(name)
        navigation[name] = {"full_value_ref": reference, "json_type": kind,
                            "serialized_bytes": len(raw),
                            "item_count": len(value) if type(value) in (list, dict) else None}
        rendered = _json(navigation).decode()
    require(len(rendered.encode()) <= DECISION_NAVIGATION_BYTES,
          "decision navigation exceeds its explicit model-input bound", ValueError)
    return rendered


def _private(path):
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
          and not info.st_mode & 0o077 and path.resolve() == path, "private canonical planner directory required", ValueError)


def _evidence_refs(value):
    if type(value) is dict:
        if set(value) == {"evidence_id", "artifact"}:
            yield value
        else:
            for item in value.values():
                yield from _evidence_refs(item)
    elif type(value) is list:
        for item in value:
            yield from _evidence_refs(item)


OUTCOME_CONTEXT_BYTES = 24 * 1024


def current_claim_context(view, *, binding=None):
    """Project current claim states; source association is not a prose dependency.

    With no binding, show corrections and every inactive claim, including stale
    transitive dependents. A candidate binding selects its claims and declared
    dependency ancestry, together with corrections touching that ancestry.
    """
    # Frozen snapshots sort mapping keys; parent material must not depend on
    # whether this view was freshly replayed or decoded from that snapshot.
    entries = sorted(view.get("claims", {}).values(), key=lambda entry: _json(entry["ref"]))
    indexed = {_json(entry["ref"]): entry for entry in entries}
    corrections = view.get("corrections", [])

    def correction_keys(event):
        refs = list(event["targets"])
        if event.get("replacement") is not None:
            refs.append(event["replacement"])
        return {_json(ref) for ref in refs}

    if binding is None:
        selected = {_json(entry["ref"]) for entry in entries if entry["state"] != "active"}
        source_refs = None
    else:
        require(type(binding) is dict and set(binding) == {"job_id", "source_id", "measurement_epoch_id"},
              "claim navigation requires an exact candidate binding", ValueError)
        source_refs = [entry["ref"] for entry in entries
                       if all(entry["record"]["binding"][key] == value for key, value in binding.items())]
        selected = {_json(ref) for ref in source_refs}
        pending = list(selected)
        while pending:
            key = pending.pop()
            require(key in indexed, "claim navigation dependency is absent from the current view", ValueError)
            for ref in indexed[key]["record"]["depends_on"]:
                dependency = _json(ref)
                if dependency not in selected:
                    selected.add(dependency)
                    pending.append(dependency)
        # Follow later disputes/supersessions of a replacement too. Otherwise a
        # parent could show that replacement's current state without its cause.
        changed = True
        while changed:
            previous = set(selected)
            for event in corrections:
                keys = correction_keys(event)
                if keys & selected:
                    selected.update(keys)
            changed = selected != previous
        corrections = [event for event in corrections if correction_keys(event) & selected]
    for event in corrections:
        selected.update(correction_keys(event))
    require(selected <= set(indexed), "claim navigation correction is absent from the current view", ValueError)
    return copy.deepcopy({
        "claim_head": view.get("head"), "binding_filter": binding,
        "source_bound_claim_refs": source_refs,
        "claim_states": [entry for entry in entries if _json(entry["ref"]) in selected],
        "corrections_newest_first": list(reversed(corrections)),
        "note_dependency_status": "undeclared; no note dependency is inferred from source identity or prose",
        "check_scope": "Current claim state and declared claim dependencies only; producer authorization "
                       "does not establish current validity, scientific interpretation, or native eligibility."})


def _bounded_json_object(value, limit):
    """Never separate evidence from context by projecting part of a JSON object."""
    raw = _json(value)
    if len(raw) <= limit:
        return raw
    excerpt = {"whole_object_omitted": True, "original_bytes": len(raw),
               "instruction": "Read the exact raw reference above before using claims from this object."}
    encoded = _json(excerpt)
    require(len(encoded) <= limit, "object omission metadata exceeds the prompt bound", ValueError)
    return encoded


def outcome_prompt_context(store, context, snapshot_ref, *, direction_id=None,
                           parent_candidate_id=None, search_policy=None, campaign_root=None):
    """Copy preceding outcome/current interpretation into input, without certifying uptake."""
    validate_read_scope(search_policy, campaign_root, [store.root.resolve(), snapshot_ref["path"]])
    frozen = _decode(_read_ref(snapshot_ref))
    require(all(frozen.get(key) == context.get(key) for key in ("direction_notes", "claims")),
          "outcome context differs from its frozen notes/corrections", ValueError)
    state, notes = context["state"], context["direction_notes"]
    history = [row for row in state["history"] if row.get("settled") is True
               and all(key in row for key in ("attempt_id", "candidate_id", "transport_ref", "collection_ref"))]
    owners = {row["opportunity_id"]: row["direction_id"] for row in state["opportunities"]}
    if direction_id is not None:
        matching = [row for row in history if owners.get(row["attempt_id"]) == direction_id]
        history = matching or [row for row in history if row["candidate_id"] == parent_candidate_id]
    previous = history[-1] if history else None
    if (direction_id is None or direction_id not in notes) and previous is not None:
        direction_id = owners.get(previous["attempt_id"])
    parts = []
    if previous is not None:
        candidate = store.get_candidate(previous["candidate_id"])
        require(all(candidate[key] == value for key, value in state["binding"].items()),
              "preceding outcome belongs to another job/epoch", ValueError)
        for role in ("transport", "collection"):
            ref = previous[role + "_ref"]
            envelope, artifacts = store.evidence(ref["evidence_id"])
            require(envelope["candidate_id"] == previous["candidate_id"] and ref["artifact"] in artifacts,
                  "preceding outcome reference differs from its captured candidate", ValueError)
            names = [ref["artifact"]] if role == "transport" else [
                name for name in ("native_idea_context.json", "native_evaluation.json", "native_not_launched.json",
                                   "arena_evaluation.json", "arena_not_launched.json",
                                   "baseline/native_evaluation.json", "paired_feedback.json") if name in artifacts]
            if not names:
                names = [ref["artifact"]]
            for name in names:
                source_ref = {"evidence_id": ref["evidence_id"], "artifact": name}
                path = store.root.resolve() / "objects" / envelope["artifacts"][name]
                validate_read_scope(search_policy, campaign_root, [path])
                raw_ref = file_ref(path)
                value = _decode(_read_ref(raw_ref))
                if role == "transport":
                    keys = ("attempt_id", "process_status", "exit_code", "terminal", "transport_complete",
                            "settlement", "captured_tip_source_id", "errors", "protocol_errors")
                    value = {key: value[key] for key in keys if key in value}
                elif name.endswith("native_evaluation.json"):
                    value = dict(value)
                    for row in value.get("operations", []):
                        if row.get("assessment", {}).get("success") is True:
                            continue
                        collection = Path(row.get("collection", ""))
                        prefix = name[:-len("native_evaluation.json")] + "native/" + "/".join(collection.parts[-2:]) + "/"
                        result_name = prefix + "result.json"
                        if result_name not in artifacts:
                            parts.append(("Failed-operation streams unknown", {"evidence_ref": source_ref, "file": raw_ref},
                                          {"operation_id": row.get("request", {}).get("operation_id"),
                                           "reason": "Exact archived operation collection is unavailable.",
                                           "expected_artifact": result_name}))
                            continue
                        require(_decode(artifacts[result_name]) == row,
                              "failed-operation collection differs from the selected native receipt row", ValueError)
                        for stream in ("stderr.bin", "stdout.bin"):
                            stream_name = prefix + "artifacts/" + stream
                            digest = row.get("receipt", {}).get("artifacts_sha256", {}).get(stream)
                            if stream_name not in artifacts or digest is None:
                                parts.append(("Failed-operation " + stream + " unknown",
                                              {"evidence_id": ref["evidence_id"], "artifact": result_name},
                                              {"reason": "Stream or its receipt hash is unavailable."}))
                                continue
                            stream_path = store.root.resolve() / "objects" / envelope["artifacts"][stream_name]
                            validate_read_scope(search_policy, campaign_root, [stream_path])
                            stream_ref = file_ref(stream_path)
                            stream_raw = _read_ref(stream_ref)
                            require(stream_ref["sha256"] == digest, "failed-operation stream differs from its receipt hash", ValueError)
                            progress = [line for line in stream_raw.splitlines()
                                        if line.startswith(b"GEAK_CORRECTNESS_PROGRESS: ")] if stream == "stdout.bin" else []
                            selected = b"\n".join(progress) if progress else stream_raw
                            parts.append(("Exact failed-operation " + stream + " excerpt; no cause inferred",
                                          {"evidence_ref": {"evidence_id": ref["evidence_id"], "artifact": stream_name},
                                           "file": stream_ref, "selected_operation": row["request"]["operation_id"],
                                           "collection_artifact": result_name},
                                          {"selection": "correctness-progress lines, tail" if progress else "stream tail",
                                           "original_bytes": len(stream_raw), "selected_bytes": len(selected),
                                           "truncated": len(selected) > 4096, "excerpt_limit_bytes": 4096,
                                           "utf8_errors": "replace", "text": selected[-4096:].decode("utf-8", errors="replace")}))
                    value["operations"] = [
                        {"mode": row.get("request", {}).get("mode"),
                         "operation_id": row.get("request", {}).get("operation_id"),
                         "assessment": row.get("assessment"), "adapter_errors": row.get("adapter_errors"),
                         "boundary": {key: row.get("receipt", {}).get(key) for key in
                                      ("execution_started", "settled", "exit_code", "errors",
                                       "evaluation_integrity_valid", "measurement_valid", "peer_exception_active")}}
                        for row in value.get("operations", [])]
                parts.append((role + ": " + name, {"evidence_ref": source_ref, "file": raw_ref}, value))
    current = notes.get(direction_id, {})
    parts.append(("Current agent review note; interpretation, not validation",
                  {"file": snapshot_ref, "field": "direction_notes"},
                  {"direction_id": direction_id, "current_note": current or None}))
    parts.append(("Current corrections; newest first, historical scopes retained, not validation",
                  {"file": snapshot_ref, "field": "claims"},
                  current_claim_context(context.get("claims", {}))))
    header = ("Source-bound preceding outcome and current interpretation. Copied fields are not new validation.\n"
              "Missing fields are unknown. Read the raw references for omitted fields/logs and before relying on an excerpt.\n"
              "Explain whether this evidence changes the next action before editing or repeating work.\n"
              + _json({"context_byte_limit": OUTCOME_CONTEXT_BYTES, "campaign_id": context["campaign_id"],
                       "binding": state["binding"], "preceding_attempt": previous,
                       "outcome_status": "retained" if previous else "No applicable settled outcome; native result unknown."}).decode()
              + "\n")
    headings = ["\n" + label + "\nRaw reference: " + _json(ref).decode() + "\n"
                for label, ref, value in parts]
    remaining = OUTCOME_CONTEXT_BYTES - len(header.encode()) - sum(len(item.encode()) for item in headings)
    limit = remaining // len(parts)
    require(limit >= 256, "outcome context references exceed the explicit prompt bound", ValueError)
    blocks = []
    for heading, (_, _, value) in zip(headings, parts):
        raw = _bounded_json_object(value, limit)
        blocks.append(heading + raw.decode())
    rendered = header + "".join(blocks)
    require(len(rendered.encode()) <= OUTCOME_CONTEXT_BYTES, "outcome context exceeded its prompt bound", ValueError)
    return rendered


_PROMPT = """You are the continuing research coordinator for this one campaign.
Choose one scientific action for the current episode. Your final message must
be exactly the proposal JSON object accepted by the action documentation, or
literal JSON null for an explicit justified stop. No Markdown or cost envelope.
Never use null for missing/invalid output, an unfinished thought, or a transport
failure. You may explicitly stop this finite campaign when a missing external
prerequisite prevents useful work; do not invent speculative actions to avoid
stopping. State the missing prerequisite and the condition for resuming.
Before an explicit null stop, explain the reason in a commentary message so the
retained CLI events preserve the rationale separately from the final null.

The brief below names current legal directions and pending obligations. The
campaign record is orchestration metadata, never an engineer direction. Read
the current notes/corrections and indicated observations before choosing; your
memory and older notes may be superseded. Inspect the relevant exact source and
raw result through the file index. Explain in the proposal what the evidence
changes, or why it leaves the next action justified. Current notes and reported
flags are evidence to examine, not automatically validated scientific findings.
Numerical premises require exact artifact/field, units, denominator, measurement
kind and applicable job/source/case/epoch. Unknowns stay unknown. Failed code or
transport does not refute an algorithm. Inspect the prior source/result and say
what changed before repeating work. Never infer latency from qualification time.

You may inspect the supplied read-only files and use CPU analysis in scratch.
Every Python invocation must use PYTHONDONTWRITEBYTECODE=1 and -B. Do not run
native/GPU work, models, other agents, or candidate code. Do not alter source,
evidence, oracle, campaign journals or host metadata. Scratch notes are fallible
memory, not archive updates. The host alone admits actions and meters resources.
Read action documentation as needed; do not invent fields, identities or grants.
No review quartet is imposed here. A pending research_review must still be
resolved through its actual agent-authored action before another implementation.
"""


class CLIPlanner:
    """Legacy-named evidence builder used internally by NativePlanner.

    Explicit read-only files and named evidence groups are copied or indexed;
    the view does not grant access to the whole archive.
    """

    def _freeze(self, context, call):
        if self.search_policy is not None:
            validate_action(self.search_policy, context, None)
            # Check originals BEFORE copying into the view: a copied sibling
            # result would otherwise appear to be a local admissible input.
            validate_read_scope(self.search_policy, self.campaign_root,
                                [ref["path"] for ref in self.read_only] + [context["instructions"]["ref"]["path"]])
        view = call / "view"
        view.mkdir(mode=0o700)
        # Only byte-identical input copies belong here; generated records stay in view.
        replicas = view / "replicas"
        replicas.mkdir(mode=0o700)
        snapshot = copy.deepcopy(context)
        instructions = snapshot.pop("instructions")
        require(_read_ref(instructions["ref"]).decode("utf-8") == instructions["text"], "action instructions changed", ValueError)
        _put(view / "actions.md", instructions["text"].encode())
        _put(view / "snapshot.json", snapshot)
        _put(view / "notes.json", {key: context.get(key) for key in
             ("direction_notes", "claims", "maturation", "owed_maturation_reviews")})
        refs, index, observations = {}, [], []

        def add(label, ref, evidence=None):
            validate_read_scope(self.search_policy, self.campaign_root, [ref["path"]])
            raw = _read_ref(ref)
            refs[ref["path"]] = ref
            row = {"label": label, "file": ref}
            if evidence is not None:
                row["evidence_ref"] = evidence
            index.append(row)
            # Copy only explicit Boolean/status fields, never manufacture a
            # score, latency, scientific validity or derived numerical premise.
            if len(raw) <= 4 * 1024 * 1024:
                try:
                    value = _decode(raw)
                except (ValueError, UnicodeError):
                    value = None
                keys = ("correctness_observed", "correct", "eligible", "independent", "provisional",
                        "measurement_valid", "evaluation_integrity_valid", "workload_capture")
                if type(value) is dict and any(key in value for key in keys):
                    flags = {key: value[key] for key in keys if type(value.get(key)) is bool}
                    observations.append({"file": ref, "evidence_ref": evidence, "reported_flags": flags,
                                         "binding": value.get("binding"), "instruction": "Read raw result and limits; flags are not a performance claim."})

        for number, ref in enumerate(self.read_only):
            copied = replicas / ("input-%04d-" % number + Path(ref["path"]).name)
            _put_ref(copied, ref)
            add("Host input " + ref["path"], file_ref(copied))
            index[-1]["original_ref"] = ref
        directions = []
        for name, row in sorted(context["archive"]["directions"].items()):
            if name == context["campaign_id"]:
                continue
            current = self.store.get(name)
            require(current["revision"] == row["revision"] and current["tip"] == row["tip"], "direction changed before freeze", ValueError)
            note = context["direction_notes"][name]["note"]
            directions.append({"direction_id": name, "revision": row["revision"], "tip": row["tip"],
                               "research_status": note.get("research_status"),
                               "latest_transport_ref": note.get("latest_transport_ref"),
                               "research_update_ref": note.get("research_update_ref")})
            candidates = {row["tip"]}
            if row.get("anchor"):
                candidates.add(row["anchor"]["candidate_id"])
            if row.get("card", {}) and row["card"].get("last_faithful"):
                candidates.add(row["card"]["last_faithful"]["candidate_id"])
            if self.search_policy is not None:
                candidates = set(lineage_candidates(self.store, candidates))
            for candidate_id in sorted(candidates):
                candidate = self.store.get_candidate(candidate_id)
                source = self.store.source(candidate["source_id"])
                for entry in source["files"]:
                    if entry["blob_id"]:
                        add("Source " + candidate_id + " / " + entry["path"],
                            file_ref(self.store.root.resolve() / "objects" / entry["blob_id"]))
                    else:
                        index.append({"label": "Source " + candidate_id + " / " + entry["path"], "file": None,
                                      "read_status": "unavailable"})
        # Recommendation records share an archive group with native planner
        # receipts/events. The native adapter exposes only their named public
        # records; navigation links must not expand that whole private group.
        requested = [ref for ref in list(_evidence_refs(context)) + self.evidence_refs
                     if ref["artifact"] != "recommendation-binding.json"]
        for name in context["direction_notes"]:
            requested += [{"evidence_id": eid, "artifact": None} for eid in self.store.get(name)["evidence"]]
        groups = sorted({ref["evidence_id"] for ref in requested})
        for identity in groups:
            envelope, artifacts = self.store.evidence(identity)
            for ref in requested:
                if ref["evidence_id"] == identity and ref["artifact"] is not None:
                    require(ref["artifact"] in artifacts, "requested evidence artifact absent", ValueError)
            for name, blob in sorted(envelope["artifacts"].items()):
                add("Evidence " + name, file_ref(self.store.root.resolve() / "objects" / blob),
                    {"evidence_id": identity, "artifact": name})
        _put(view / "files.json", index)
        brief = {"campaign_id": context["campaign_id"], "campaign_is_not_a_direction": True,
                 "binding": context["state"]["binding"], "legal_existing_directions": directions,
                 "new_direction": "Agent may choose a new direction ID with expected_revision=0 and a permitted parent.",
                 "pending_maturation": context.get("owed_maturation_reviews", []),
                 "current_notes_and_corrections": str(view / "notes.json"), "observations_to_inspect": observations,
                 "snapshot": str(view / "snapshot.json"), "action_docs": str(view / "actions.md"),
                 "source_and_evidence_index": str(view / "files.json"),
                 "resource_limits": context["state"]["limits"], "deadline": context["state"]["deadline"]}
        brief["preceding_outcome_context"] = outcome_prompt_context(
            self.store, context, file_ref(view / "snapshot.json"),
            search_policy=self.search_policy, campaign_root=self.campaign_root)
        if self.search_policy is not None:
            brief["search_policy"] = policy_contract(self.search_policy)
            if self.search_policy["arm"] == "G":
                brief["new_direction"] = "Use a fresh native episode ID for the serial G line; parent must be a reviewed source role of the preceding episode (seed initially). Earlier own history remains readable for restoration and composition."
        _put(view / "brief.json", brief)
        for path in sorted(view.iterdir()):
            if path == replicas:
                continue  # Each input copy was already indexed by add().
            refs[str(path)] = file_ref(path)
        replicas.chmod(0o555)
        _sync_directory(call)
        return brief, list(refs.values())
