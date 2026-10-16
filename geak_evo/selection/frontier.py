"""Time-stamped journal commits and one-shot horizon seals of the final pick.

A ``CommitFrontier`` is registered on a new store before its first commit. From
then on every journal write is bracketed by two clock samples and the pair is
saved as a commit receipt. At a declared horizon (H4, H8, H12 or END),
``seal_horizon`` takes exactly the commits that finished before the cutoff,
runs ``final.validate_final_recommendation`` on that prefix and writes an
immutable seal with status ``selected``, ``missing`` or ``unresolved``. There
is no fallback: any error gives ``unresolved`` with no pick. A commit whose
interval crosses the cutoff, or a missing receipt, also gives ``unresolved``.
Receipt-writing problems never stop research; they only show up as a gap when
a horizon is read.

Clock: every "now" comes from the injected ``clock`` callable, which returns
``(boot_id, monotonic_ns)``. The default ``sample_clock`` reads the kernel boot
id before and after ``time.monotonic_ns()``. Monotonic times only compare within
one boot, so every sample, receipt and seal must carry the campaign clock's
``boot_id``; a reboot makes registration, receipts and horizons fail.

The ``authenticate_choice(frozen_store, context, choice)`` callback must
regenerate the exact choice with ``choices.authenticate_choice`` and the real
episode reader. Every file read during validation, plus the files named in the
chosen program, is copied into ``commit-frontier/objects`` so the seal can be
re-checked later without the live workspace.
"""

import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import time

from .._util import EvoError, canonical_json, is_sha256, require, sha256_hex
from ..archive.store import _read, _sync_directory, _write
from .final import validate_final_recommendation


HOURS = {"H4": 4, "H8": 8, "H12": 12}
SUPPORTED_HORIZONS = frozenset(HOURS) | {"END"}
NS = 1_000_000_000
CONTRACT_SCHEMA = "geak-committed-frontier-contract-v1"
RECEIPT_SCHEMA = "geak-durable-journal-commit-receipt-v1"
SEAL_SCHEMA = "geak-committed-horizon-endpoint-v1"
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_FRONTIER_KEYS = {"sequence", "head", "observed_monotonic_ns", "boot_id"}
_RECEIPT_KEYS = {"schema", "contract_ref", "boot_id", "journal_ref", "sequence", "direction_id", "revision",
                 "previous_journal_sha256", "lower_monotonic_ns", "upper_monotonic_ns"}
_SEAL_KEYS = {"schema", "contract_ref", "horizon", "cutoff_monotonic_ns", "status", "recommendation",
              "frontier", "proof_refs", "witness_refs", "error", "mechanism_history_completeness",
              "fresh_timing_authorized"}


class FrontierError(EvoError, ValueError):
    pass


def need(value, reason):
    require(value, reason, FrontierError)


def sample_clock():
    """Default clock: ``(boot_id, time.monotonic_ns())``; fails if the boot id changes meanwhile."""
    boot = BOOT_ID_PATH.read_text().strip()
    instant = time.monotonic_ns()
    need(boot and BOOT_ID_PATH.read_text().strip() == boot, "clock_boot_changed")
    return boot, instant


def _sample(clock):
    boot, instant = clock()
    need(type(boot) is str and boot and type(instant) is int and instant >= 0, "clock_sample_differs")
    return boot, instant


def horizon_cutoffs(clock, horizons):
    """Monotonic cutoffs: start + 4/8/12 hours, plus END = the clock's own deadline.

    ``clock`` is the campaign clock record. END uses its integer ``deadline_ns``,
    never a fresh sample; ``wall_seconds`` (decimal JSON) must describe exactly the
    same interval. Fraction avoids float rounding (e.g. ``int(1.001 * NS)``).
    """
    need(type(horizons) in (tuple, list) and horizons
         and all(type(name) is str and name in SUPPORTED_HORIZONS for name in horizons)
         and len(set(horizons)) == len(horizons), "declared_horizons_required")
    start = clock["started_monotonic_ns"]
    need(type(start) is int and start >= 0, "registration_clock_or_horizon_differs")
    selected = {name: start + hours * 3600 * NS for name, hours in HOURS.items() if name in horizons}
    if "END" in horizons:
        deadline = clock.get("deadline_ns")
        need(type(deadline) is int and deadline > start, "terminal_clock_deadline_required")
        wall = clock.get("wall_seconds")
        need(type(wall) in (int, float) and math.isfinite(wall) and wall > 0
             and Fraction(str(wall)) * NS == deadline - start, "terminal_clock_duration_differs")
        selected["END"] = deadline
    return selected


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    canonical_json(value)
    return value


def file_ref(path, raw):
    return {"path": str(path), "sha256": sha256_hex(raw)}


def checked_ref(ref, read_ref):
    need(type(ref) is dict and set(ref) == {"path", "sha256"}
         and type(ref["path"]) is str and Path(ref["path"]).is_absolute()
         and is_sha256(ref["sha256"]), "exact_file_reference_required")
    raw = read_ref(copy.deepcopy(ref))
    need(type(raw) is bytes and sha256_hex(raw) == ref["sha256"], "evidence_identity_differs")
    return raw


def _object_ref(root, identity):
    return {"path": str(root / "objects" / identity), "sha256": identity}


def _visit_refs(value, read, root):
    """Call ``read`` on every literal file ref in ``value``; an evidence pair reads
    its envelope and the one named artifact. Never follows refs inside file bodies."""
    if type(value) is dict:
        if {"path", "sha256"} <= set(value):
            read({key: value[key] for key in ("path", "sha256")})
        elif set(value) == {"evidence_id", "artifact"}:
            envelope = decode(read(_object_ref(root, value["evidence_id"])))
            read(_object_ref(root, envelope["artifacts"][value["artifact"]]))
        else:
            for child in value.values():
                _visit_refs(child, read, root)
    elif type(value) is list:
        for child in value:
            _visit_refs(child, read, root)


def _native_check(frozen, authenticate_choice, read, root):
    """Wrap the caller's choice authenticator and read the chosen program's files."""
    def native(context, choice):
        need(callable(authenticate_choice), "native_authenticator_required")
        actual = authenticate_choice(frozen, context, copy.deepcopy(choice))
        need(canonical_json(actual) == canonical_json(choice), "native_choice_differs")
        candidate = frozen.get_candidate(choice["candidate_id"])
        need(candidate["source_id"] == choice["source_id"], "selected_source_differs")
        for identity in (choice["candidate_id"], choice["source_id"]):
            read(_object_ref(root, identity))
        for row in frozen.source(choice["source_id"])["files"]:
            if row["blob_id"] is not None:
                read(_object_ref(root, row["blob_id"]))
        _visit_refs(choice, read, root)
        return actual
    return native


def _campaign_states(prefix, campaign):
    states = [record["state"] for record in prefix if record["state"]["direction_id"] == campaign]
    for committed in states:
        state = committed["note"]["evolution"]
        need(type(state["history"]) is list and type(state["opportunities"]) is list
             and type(state["recommendation_events"]) is list
             and "recommendation_head" in state, "incomplete_campaign_history")
    return states


def _snapshot(campaign, latest):
    return {"campaign_id": campaign, "revision": latest["revision"],
            "state": copy.deepcopy(latest["note"]["evolution"])}


class FrozenStore:
    """Read-only view of the store limited to the revisions before a horizon."""

    def __init__(self, store, records):
        self.root = store.root
        self._store = store
        self._revisions = {(r["state"]["direction_id"], r["state"]["revision"]): copy.deepcopy(r["state"])
                           for r in records}

    def get_revision(self, direction_id, revision):
        need((direction_id, revision) in self._revisions, "revision_outside_horizon")
        return copy.deepcopy(self._revisions[direction_id, revision])

    def get_candidate(self, identity):
        return self._store.get_candidate(identity)

    def source(self, identity):
        return self._store.source(identity)

    def evidence(self, identity):
        return self._store.evidence(identity)

    def evidence_artifact(self, identity, name):
        return self._store.evidence_artifact(identity, name)


class CommitFrontier:
    def __init__(self, root, contract, raw, *, clock=sample_clock):
        self.root = Path(root)
        self.directory = self.root / "commit-frontier"
        self.contract = contract
        self.contract_ref = file_ref(self.directory / "CONTRACT.json", raw)
        self.clock = clock
        self.last_witness_error = None

    @classmethod
    def register(cls, store, *, expected_binding, clock_ref, clock_raw,
                 horizons=("H4", "H8", "H12"), registration_ref=None, registration_raw=None,
                 clock=sample_clock):
        """Create the frontier on a store whose journal is still empty."""
        need(type(expected_binding) is dict and set(expected_binding) ==
             {"campaign_id", "job_id", "measurement_epoch_id"}
             and all(type(v) is str and v for v in expected_binding.values()), "binding_required")
        need(type(clock_raw) is bytes, "clock_bytes_required")
        checked_ref(clock_ref, lambda _: clock_raw)
        if registration_ref is None:
            need(registration_raw is None, "registration_reference_required")
        else:
            checked_ref(registration_ref, lambda _: registration_raw)
        campaign_clock = decode(clock_raw)
        selected = horizon_cutoffs(campaign_clock, horizons)
        boot, instant = _sample(clock)
        start = campaign_clock["started_monotonic_ns"]
        need(type(start) is int and start >= 0 and boot == campaign_clock["boot_id"]
             and start <= instant < min(selected.values())
             and campaign_clock["wall_seconds"] >= max((HOURS[n] for n in horizons if n in HOURS), default=0) * 3600,
             "registration_clock_or_horizon_differs")
        for key, value in expected_binding.items():
            need(key not in campaign_clock or campaign_clock[key] == value, "clock_binding_differs")
        contract = {"schema": CONTRACT_SCHEMA,
                    "binding": copy.deepcopy(expected_binding), "clock_ref": copy.deepcopy(clock_ref),
                    "registration_ref": copy.deepcopy(registration_ref),
                    "clock": campaign_clock, "registered_monotonic_ns": instant,
                    "horizons": selected,
                    "boundary": "journal_upper_strictly_before_horizon",
                    "missing_or_crossing_receipt": "unresolved_no_head",
                    "witness_failure_stops_research": False}
        raw = canonical_json(contract)
        with store._lock():
            need(not any((store.root / "records").iterdir()), "registration_requires_new_journal")
            directory = store.root / "commit-frontier"
            directory.mkdir(mode=0o700)  # Create-only: never attached to an existing journal.
            for name in ("receipts", "seals", "objects"):
                (directory / name).mkdir(mode=0o700)
            _write(directory / "CONTRACT.json", raw, immutable=True)
            _sync_directory(directory)
            _sync_directory(store.root)
        result = cls(store.root, contract, raw, clock=clock)
        store.commit_frontier = result
        return result

    @classmethod
    def open(cls, root, contract_ref, *, clock=sample_clock):
        """Attach to the exact existing contract; creates no missing receipts."""
        root = Path(root)
        path = root / "commit-frontier" / "CONTRACT.json"
        need(contract_ref.get("path") == str(path), "contract_path_differs")
        raw, _ = _read(path.parent, path.name)
        checked_ref(contract_ref, lambda _: raw)
        result = cls(root, decode(raw), raw, clock=clock)
        result._contract()
        return result

    def _contract(self):
        raw, _ = _read(self.directory, "CONTRACT.json")
        need(file_ref(self.directory / "CONTRACT.json", raw) == self.contract_ref
             and decode(raw) == self.contract, "contract_changed")
        need(self.contract["schema"] == CONTRACT_SCHEMA
             and type(self.contract["horizons"]) is dict and self.contract["horizons"]
             and set(self.contract["horizons"]) <= SUPPORTED_HORIZONS
             and self.contract["horizons"] == horizon_cutoffs(self.contract["clock"], list(self.contract["horizons"])),
             "horizons_changed")

    def write_record(self, path, raw, *, sequence):
        """Write one journal record (always attempted once; only its own errors
        propagate) and, separately, its commit receipt."""
        lower = None
        try:
            self._contract()
            need(path.parent == self.root / "records"
                 and path.name == "%08d-%s.json" % (sequence, sha256_hex(raw)), "journal_path_differs")
            record = decode(raw)
            boot, lower = _sample(self.clock)
            need(boot == self._boot_id() and lower >= self.contract["registered_monotonic_ns"],
                 "commit_clock_differs")
        except Exception as exc:
            lower = None
            self.last_witness_error = type(exc).__name__
        _write(path, raw, immutable=True)  # The journal commit; its errors are never swallowed.
        if lower is None:
            return
        try:
            upper_boot, upper = _sample(self.clock)
            need(upper_boot == boot and lower <= upper, "commit_interval_differs")
            receipt = {"schema": RECEIPT_SCHEMA, "contract_ref": self.contract_ref, "boot_id": boot,
                       "journal_ref": file_ref(path, raw), "sequence": sequence,
                       "direction_id": record["state"]["direction_id"],
                       "revision": record["state"]["revision"], "previous_journal_sha256": record["previous"],
                       "lower_monotonic_ns": lower, "upper_monotonic_ns": upper}
            proof = canonical_json(receipt)
            _write(self.directory / "receipts" / (path.stem + "-" + sha256_hex(proof) + ".json"), proof,
                   immutable=True)
        except Exception as exc:
            # A later horizon read reports the gap; no restamp, retry or search stop.
            self.last_witness_error = type(exc).__name__

    def _boot_id(self):
        return self.contract["clock"]["boot_id"]

    def _check_receipt(self, receipt, *, journal_ref, sequence, state, head, previous_upper, boot, observed,
                       cutoff):
        lower, upper = receipt["lower_monotonic_ns"], receipt["upper_monotonic_ns"]
        need(set(receipt) == _RECEIPT_KEYS and receipt["schema"] == RECEIPT_SCHEMA
             and receipt["contract_ref"] == self.contract_ref and receipt["boot_id"] == boot
             and receipt["journal_ref"] == journal_ref and receipt["sequence"] == sequence
             and receipt["direction_id"] == state["direction_id"] and receipt["revision"] == state["revision"]
             and receipt["previous_journal_sha256"] == head
             and type(lower) is int and type(upper) is int
             and previous_upper <= lower <= upper <= observed, "commit_receipt_differs")
        need(upper < cutoff or lower >= cutoff, "commit_straddles_horizon")
        return upper

    def _capture(self, store, horizon):
        """Journal records committed before the cutoff, with their proof refs."""
        self._contract()
        need(store.root == self.root, "store_root_differs")
        cutoff = self.contract["horizons"][horizon]
        records, receipts, prefix = [], [], []
        with store._lock():
            boot, observed = _sample(self.clock)
            need(boot == self._boot_id() and observed >= cutoff, "horizon_not_observed")
            index = store._replay()  # The store's own journal and source checks stay strict.
            head, previous_upper = None, self.contract["registered_monotonic_ns"]
            journal_paths = sorted((self.root / "records").glob("*.json"))
            proof_paths = {}
            for path in (self.directory / "receipts").glob("*.json"):
                match = re.fullmatch(r"([0-9]{8}-[0-9a-f]{64})-([0-9a-f]{64})\.json", path.name)
                need(match is not None, "receipt_name_differs")
                journal_name = match[1] + ".json"
                need(journal_name not in proof_paths, "duplicate_commit_receipt")
                proof_paths[journal_name] = (path, match[2])
            need(set(proof_paths) == {path.name for path in journal_paths}, "receipt_inventory_differs")
            for sequence, path in enumerate(journal_paths, 1):
                raw, _ = _read(path.parent, path.name)
                record = decode(raw)
                need(path.name == "%08d-%s.json" % (sequence, sha256_hex(raw))
                     and record["previous"] == head, "journal_frontier_differs")
                proof_path, proof_sha = proof_paths[path.name]
                proof, _ = _read(proof_path.parent, proof_path.name)
                need(sha256_hex(proof) == proof_sha, "receipt_hash_differs")
                upper = self._check_receipt(decode(proof), journal_ref=file_ref(path, raw), sequence=sequence,
                                            state=record["state"], head=head, previous_upper=previous_upper,
                                            boot=boot, observed=observed, cutoff=cutoff)
                if upper < cutoff:
                    prefix.append(record)
                records.append(record)
                receipts.append({"journal_ref": file_ref(path, raw), "receipt_ref": file_ref(proof_path, proof)})
                head, previous_upper = sha256_hex(raw), upper
            need(len(records) == index["sequence"] and head == index["head"], "journal_tip_differs")
        return prefix, receipts, {"sequence": len(records), "head": head, "observed_monotonic_ns": observed,
                                  "boot_id": boot}

    def _saved_prefix(self, store, saved, cutoff, read_ref):
        """Re-read the journal prefix named by a saved seal (no recapture)."""
        observed = saved["frontier"]
        need(type(observed) is dict and set(observed) == _FRONTIER_KEYS
             and type(observed["sequence"]) is int and observed["sequence"] >= 0
             and type(observed["observed_monotonic_ns"]) is int
             and observed["observed_monotonic_ns"] >= cutoff
             and observed["boot_id"] == self._boot_id(), "saved_frontier_differs")
        need(type(saved["proof_refs"]) is list
             and len(saved["proof_refs"]) == observed["sequence"], "saved_proof_inventory_differs")
        prefix, directions = [], {}
        head, previous_upper = None, self.contract["registered_monotonic_ns"]
        for sequence, refs in enumerate(saved["proof_refs"], 1):
            need(type(refs) is dict and set(refs) == {"journal_ref", "receipt_ref"}, "saved_proof_reference_differs")
            journal_ref, receipt_ref = refs["journal_ref"], refs["receipt_ref"]
            journal_raw = checked_ref(journal_ref, read_ref)
            journal_name = "%08d-%s" % (sequence, sha256_hex(journal_raw))
            need(journal_ref["path"] == str(self.root / "records" / (journal_name + ".json")),
                 "saved_journal_path_differs")
            record = decode(journal_raw)
            need(set(record) == {"previous", "state"} and record["previous"] == head, "journal_frontier_differs")
            state = record["state"]
            direction, revision = state["direction_id"], state["revision"]
            need(type(direction) is str and direction and type(revision) is int
                 and revision == directions.get(direction, 0) + 1, "direction_revision_differs")
            store._validate(state)  # The store's own object/source/evidence checks; no replay or cache.
            directions[direction] = revision
            proof = checked_ref(receipt_ref, read_ref)
            need(receipt_ref["path"] == str(self.directory / "receipts" /
                                            (journal_name + "-" + sha256_hex(proof) + ".json")),
                 "saved_receipt_path_differs")
            upper = self._check_receipt(decode(proof), journal_ref=journal_ref, sequence=sequence, state=state,
                                        head=head, previous_upper=previous_upper, boot=observed["boot_id"],
                                        observed=observed["observed_monotonic_ns"], cutoff=cutoff)
            if upper < cutoff:
                prefix.append(record)
            head, previous_upper = sha256_hex(journal_raw), upper
        need(head == observed["head"], "saved_frontier_head_differs")
        return prefix

    def _saved_witnesses(self, saved, read_ref):
        witnesses = {}
        need(type(saved["witness_refs"]) is list, "saved_witness_inventory_differs")
        for witness in saved["witness_refs"]:
            need(type(witness) is dict and set(witness) == {"origin_ref", "retained_ref", "bytes"},
                 "saved_witness_reference_differs")
            origin, retained = witness["origin_ref"], witness["retained_ref"]
            body = checked_ref(retained, read_ref)
            checked_ref(origin, lambda _: body)
            need(retained == file_ref(self.directory / "objects" / sha256_hex(body), body)
                 and type(witness["bytes"]) is int and witness["bytes"] == len(body),
                 "saved_witness_binding_differs")
            key = (origin["path"], origin["sha256"])
            need(key not in witnesses, "duplicate_saved_witness")
            witnesses[key] = body
        return witnesses

    def authenticate_existing_horizon(self, store, horizon, seal_ref, *, read_ref, authenticate_choice,
                                      runtime=None):
        """Re-check one saved ``selected``/``missing`` seal without recapturing.

        Returns ``(seal_raw, frozen_store, snapshot, recommendation)``; snapshot and
        recommendation are None when no campaign revision preceded the horizon.
        Unresolved seals raise. Only the seal's own journal prefix and its retained
        copies are read, so later commits and the current clock change nothing.
        The choice authenticator is rerun through the caller's callback.
        """
        self._contract()
        need(store.root == self.root, "store_root_differs")
        need(horizon in self.contract["horizons"], "undeclared_horizon")
        path = self.directory / "seals" / (horizon + ".json")
        need(type(seal_ref) is dict and seal_ref.get("path") == str(path), "seal_path_differs")
        raw = checked_ref(seal_ref, read_ref)
        saved = decode(raw)
        cutoff = self.contract["horizons"][horizon]
        need(saved["schema"] == SEAL_SCHEMA
             and saved["contract_ref"] == self.contract_ref and saved["horizon"] == horizon
             and saved["cutoff_monotonic_ns"] == cutoff
             and saved["status"] in ("selected", "missing") and saved["error"] is None
             and saved["mechanism_history_completeness"] == "not_established"
             and saved["fresh_timing_authorized"] is False, "saved_horizon_differs")
        prefix = self._saved_prefix(store, saved, cutoff, read_ref)
        need(decode(checked_ref(self.contract["clock_ref"], read_ref)) == self.contract["clock"],
             "clock_binding_differs")
        witnesses = self._saved_witnesses(saved, read_ref)
        used = set()

        def retained_read(ref):
            def lookup(exact):
                key = (exact["path"], exact["sha256"])
                need(key in witnesses, "saved_witness_missing")
                used.add(key)
                return witnesses[key]
            return checked_ref(ref, lookup)

        frozen = FrozenStore(store, prefix)
        campaign = self.contract["binding"]["campaign_id"]
        states = _campaign_states(prefix, campaign)
        snapshot, recommendation = None, None
        if not states:
            need(set(saved) == _SEAL_KEYS | {"missing_reason"} and saved["status"] == "missing"
                 and saved["missing_reason"] == "no_campaign_revision_before_horizon"
                 and saved["recommendation"] is None, "saved_empty_horizon_differs")
        else:
            snapshot = _snapshot(campaign, states[-1])
            need(set(saved) == _SEAL_KEYS | {"snapshot"} and saved["snapshot"] == snapshot,
                 "saved_snapshot_differs")
            recommendation = validate_final_recommendation(
                frozen, snapshot, expected_binding=self.contract["binding"], clock_ref=self.contract["clock_ref"],
                read_ref=retained_read,
                authenticate_choice=_native_check(frozen, authenticate_choice, retained_read, self.root),
                runtime=runtime)
            need(canonical_json(recommendation) == canonical_json(saved["recommendation"])
                 and recommendation["status"] == saved["status"], "saved_recommendation_differs")
        need(used == set(witnesses), "saved_witness_inventory_differs")
        return raw, frozen, snapshot, recommendation

    def seal_horizon(self, store, horizon, *, read_ref, authenticate_choice, runtime=None):
        """Write the one seal for ``horizon`` and return ``(seal, seal_ref)``.

        Callback: ``authenticate_choice(frozen_store, context, choice)``. Any error
        gives an immutable ``unresolved`` seal with no pick (no fallback). This
        does not start, wait for or settle any planner or engineer work.
        """
        need(horizon in self.contract["horizons"], "undeclared_horizon")
        boot, now = _sample(self.clock)
        need(boot == self._boot_id() and now >= self.contract["horizons"][horizon], "horizon_not_observed")
        path = self.directory / "seals" / (horizon + ".json")
        need(not path.exists(), "horizon_already_sealed")
        result = {"schema": SEAL_SCHEMA, "contract_ref": self.contract_ref,
                  "horizon": horizon, "cutoff_monotonic_ns": self.contract["horizons"][horizon],
                  "status": "unresolved", "recommendation": None, "frontier": None,
                  "proof_refs": [], "witness_refs": [], "error": None,
                  "mechanism_history_completeness": "not_established",
                  "fresh_timing_authorized": False}
        captured = {}

        def retain(ref):
            raw = checked_ref(ref, read_ref)
            blob = self.directory / "objects" / sha256_hex(raw)
            try:
                _write(blob, raw, immutable=True)
            except FileExistsError:
                need(_read(blob.parent, blob.name)[0] == raw, "retained_evidence_changed")
            captured[(ref["path"], ref["sha256"])] = {"origin_ref": copy.deepcopy(ref),
                                                     "retained_ref": file_ref(blob, raw), "bytes": len(raw)}
            return raw

        try:
            prefix, proofs, frontier = self._capture(store, horizon)
            result["frontier"], result["proof_refs"] = frontier, proofs
            frozen = FrozenStore(store, prefix)
            campaign = self.contract["binding"]["campaign_id"]
            states = _campaign_states(prefix, campaign)
            if not states:
                result["status"] = "missing"
                result["missing_reason"] = "no_campaign_revision_before_horizon"
            else:
                snapshot = _snapshot(campaign, states[-1])
                recommendation = validate_final_recommendation(
                    frozen, snapshot, expected_binding=self.contract["binding"],
                    clock_ref=self.contract["clock_ref"], read_ref=retain,
                    authenticate_choice=_native_check(frozen, authenticate_choice, retain, self.root),
                    runtime=runtime)
                result["recommendation"] = recommendation
                result["status"] = recommendation["status"]
                result["snapshot"] = snapshot
            result["witness_refs"] = list(captured.values())
        except Exception as exc:
            result.update(status="unresolved", recommendation=None, error=type(exc).__name__,
                          witness_refs=list(captured.values()))
            if isinstance(exc, FrontierError):
                result["error_reason"] = str(exc)  # A fixed reason string; no bodies or tracebacks.
        _write(path, canonical_json(result), immutable=True)
        return result, file_ref(path, canonical_json(result))
