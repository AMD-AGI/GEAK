"""The evolution loop: one planner decision, then one continuing engineer, repeated.

The loop keeps its whole state in a campaign record of the research store and
saves every step before it calls anything outside. After a crash a new
EvolutionLoop reads that state and continues: a call that may already have
started is never started again; the host must hand back its result (the
``recover_*`` methods) or leave the campaign blocked.

There is no default planner, engineer, evaluator, meter or GPU owner. The
callbacks passed in are trusted host code, not tools offered to a candidate.
Prompt files, the clock and the program lease bound are constructor options.
"""

import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from ._util import EvoError, canonical_json, require, sha256_hex
from . import budget
from .archive.actions import (index_result, perform_action, prepare_maturation, preview_action,
                              preview_maturation, without_timestamp)
from .archive.claims import ClaimLog, check_premises, check_repeat
from .archive.exchange import prepare_engineer_request, resolve_artifact, validate_exchange, validate_receipt
from .archive.maturation import (checkpoint_maturation, pending_maturation_reviews, read_maturation,
                                 validate_protection)
from .archive.records import archive_checkpoint, read_archive, read_record
from .archive.refs import file_ref, source_reference
from .archive.store import StateError
from .selection.recommendation import commit_record
from .selection.work import SelectionError, account_opportunities, advise, record_decision


class EvolutionError(EvoError, ValueError):
    pass


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
RESEARCH_ACTIONS = ("accept_anchor", "nominate", "revive", "charter", "membership", "concept", "exchange")
ENGINEER_FIELDS = {"direction_id", "expected_revision", "parent_candidate_id", "island_id", "engineer_id",
                   "question", "reason", "evidence_uptake", "reservation", "review_at", "premises",
                   "repeat_delta", "exchange_ref", "protection_ref", "override_reason", "membership_ref"}
TRANSPORT_FILES = ("prompt.txt", "report.md", "events.jsonl", "stderr.txt", "native_receipt.json",
                   "native_events.jsonl", "native_entry_result.json", "native_prepared.json")
PREMISE_NOTE = (b"\nThe planner question/reason/evidence_uptake are agent interpretation. "
                b"Use registered facts only for their checked structured quantity; test unverified causal premises.\n")
UNREVIEWED_FINDINGS = {
    "claim": "Engineer outcome awaits research review in the retained report and collection.",
    "scope": "Source-bound continuation; transport does not establish a scientific finding.",
    "next_test": "Inspect the retained outcome and reconcile its premises before continuing.",
    "evidence_uptake": "No post-turn research update was supplied; prior predictions remain unreviewed."}


def _need(condition, message):
    require(condition, message, EvolutionError)


def _text(value):
    _need(isinstance(value, str) and bool(value.strip()), "nonempty research text required")


def _json(value):
    return canonical_json(value, allow_decimal=True)


def _instruction(path):
    ref = file_ref(path)
    data = Path(path).read_bytes()
    _need(sha256_hex(data) == ref["sha256"], "instruction changed while binding")
    return ref, data


def _check_transport(collect, launch_worker, settle_worker, capture_tip, backend_options, run_investigation):
    turn = run_investigation is None and all(callable(fn) for fn in (launch_worker, settle_worker, capture_tip))
    native = callable(run_investigation) and all(
        fn is None for fn in (launch_worker, settle_worker, capture_tip, backend_options))
    _need(callable(collect) and (turn or native),
          "choose one qualified turn transport or one native investigation callback")


def _matches(record, binding):
    return all(record[key] == value for key, value in binding.items())


def _after_lease(lease):
    return "lease_blocked" if lease is not None and lease["status"] == "blocked" else "idle"


class EvolutionLoop:
    """One writer per campaign, on top of the store's journal and immutable files.

    Claim corrections go through update_claims or a collected research_update.
    The loop lock covers admission through dispatch. The store is the host's
    trusted record, not a security boundary against a privileged writer.
    A new instance only reads state; it never starts work from its constructor.

    Options:
      clock            wall-clock seconds; every deadline check uses it.
      monotonic        elapsed-time source for advice preparation timing.
      prompt_dir       directory with planner.md, engineer.md, crossover.md and
                       program_lease_engineer.md; defaults to the packaged
                       geak_evo/prompts, as the original read its own prompts.
      program_lease    program lease bound, 0 (off) or 2. The host must pass
                       archive.policy.program_lease_bound(policy), as the
                       original coordinator did; the default 0 is that
                       function's result for a policy without the opt-in.
    """

    def __init__(self, store, campaign_id, *, clock=time.time, verify_quantity=None, frozen_cases=None,
                 prompt_dir=PROMPT_DIR, program_lease=0, monotonic=time.monotonic):
        self.store, self.campaign_id = store, campaign_id
        self.clock, self.verify_quantity = clock, verify_quantity
        self.monotonic = monotonic
        _need(type(program_lease) is int and program_lease in (0, 2), "program lease bound must be 0 or 2")
        self.prompt_dir = Path(prompt_dir)
        self.program_lease = program_lease
        _, state = self._state()  # Fail if this is not an existing campaign.
        registered = state.get("frozen_cases")
        _need(frozen_cases is None or frozen_cases == registered, "campaign frozen case registry changed")
        self.frozen_cases = copy.deepcopy(registered)

    @classmethod
    def create(cls, store, campaign_id, seed_candidate_id, *, limits, deadline,
               claim_head=None, clock=time.time, verify_quantity=None, frozen_cases=None, **options):
        candidate = store.get_candidate(seed_candidate_id)
        binding = {key: candidate[key] for key in ("job_id", "measurement_epoch_id")}
        account_opportunities([], binding=binding, limits=limits, deadline=deadline, now=clock())
        ClaimLog(store, claim_head, verify_quantity=verify_quantity).view()
        state = {"schema_version": 1, "binding": binding, "limits": limits,
                 "frozen_cases": frozen_cases,
                 "deadline": deadline, "claim_head": claim_head, "phase": "idle",
                 "opportunities": [], "history": [], "latest_receipts": {}, "pending": None,
                 "recommendation_head": None, "recommendation_events": []}
        store.checkpoint(campaign_id, seed_candidate_id, expected_revision=0,
                         note={"claim": "Host research campaign state; no performance claim.",
                               "scope": "Planner and engineer lifecycle only.",
                               "next_test": "Select a source-bound discriminating experiment.",
                               "evolution": state})
        return cls(store, campaign_id, clock=clock, verify_quantity=verify_quantity,
                   frozen_cases=frozen_cases, **options)

    # --- state, lock and accounting ---------------------------------------

    @contextlib.contextmanager
    def _lock(self):
        name = "loop-" + sha256_hex(self.campaign_id.encode()) + ".lock"
        fd = os.open(self.store.root / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise EvolutionError("campaign already has a writer") from exc
            yield
        finally:
            os.close(fd)

    def _state(self):
        record = self.store.get(self.campaign_id)
        state = copy.deepcopy(record["note"].get("evolution"))
        _need(isinstance(state, dict) and state.get("schema_version") == 1, "not an evolution campaign")
        return record, state

    def _save(self, record, state):
        return self.store.checkpoint(self.campaign_id, record["tip"],
                                     note=dict(record["note"], evolution=state),
                                     expected_revision=record["revision"])

    def _account(self, state):
        return account_opportunities(state["opportunities"], binding=state["binding"],
                                     limits=state["limits"], deadline=state["deadline"], now=self.clock(),
                                     commitments=budget.commitments(state))

    def _claims(self, state):
        return ClaimLog(self.store, state["claim_head"], verify_quantity=self.verify_quantity)

    def _check_premises(self, premises, binding, view, *, expected_head, candidate_id):
        resolver = getattr(self.verify_quantity, "case_bindings", None)
        cases = resolver(candidate_id) if callable(resolver) else None
        return check_premises(premises, binding, view, expected_head=expected_head,
                              expected_case_bindings=cases)

    def _prompt(self, name):
        return _instruction(self.prompt_dir / name)

    def _artifact(self, candidate_id, name, value, artifacts=None):
        files = dict(artifacts or {})
        _need(name not in files, "host artifact name collides with supplied artifact")
        files[name] = _json(value)
        return {"evidence_id": self.store.archive_evidence(candidate_id, files), "artifact": name}

    def snapshot(self):
        record, state = self._state()
        directions = self.store.get()
        result = {"campaign_id": self.campaign_id, "revision": record["revision"],
                  "state": state, "archive": read_archive(self.store, frozen_cases=self.frozen_cases),
                  "claims": self._claims(state).view(), "frozen_cases": copy.deepcopy(self.frozen_cases)}
        result["maturation"] = {name: read_maturation(self.store, name) for name, direction in directions.items()
                                if direction["note"].get("maturation_ref") is not None}
        result["direction_notes"] = {name: {"revision": direction["revision"], "note": direction["note"]}
                                     for name, direction in directions.items() if name != self.campaign_id}
        try:
            result["accounting"] = self._account(state)
            result["owed_maturation_reviews"] = pending_maturation_reviews(self.store, result["accounting"])
        except SelectionError as exc:
            result["accounting_error"] = str(exc)
        return result

    # --- reviews ------------------------------------------------------------

    def review_maturation(self, direction_id, *, decision, expected_revision,
                          observation_ref=None, verify_host=None):
        """Apply a host-checked observation or an agent review; starts no work."""
        with self._lock():
            _, state = self._state()
            _need(state["phase"] in ("idle", "planned"), "resolve active work before maturation review")
            direction = self.store.get(direction_id)
            _need(_matches(direction, state["binding"]), "maturation direction job/epoch mismatch")
            _need(decision.get("action") in ("observe", "review"), "use the explicit maturation lifecycle")
            return checkpoint_maturation(self.store, direction_id, decision=decision,
                                         expected_revision=expected_revision, accounting=self._account(state),
                                         observation_ref=observation_ref, verify_host=verify_host)

    def _membership(self, proposal, direction):
        ref = proposal["membership_ref"]
        if proposal["island_id"] == "unassigned":
            _need(ref is None and proposal["exchange_ref"] is None,
                  "unassigned work has no island or crossover attribution")
            return None
        _need(ref is not None, "assigned island requires an exact membership reference")
        member = read_record(self.store, ref, frozen_cases=self.frozen_cases)
        _need(member["kind"] == "evo_membership" and member["candidate_id"] == proposal["parent_candidate_id"]
              and member["island_id"] == proposal["island_id"], "selected island membership mismatch")
        if direction is not None:
            card = read_record(self.store, direction["note"]["archive_card_ref"], frozen_cases=self.frozen_cases)
            _need(ref in card.get("membership_refs", []), "selected membership is not in the current direction card")
        return member

    def review_research(self, direction_id, *, expected_revision, transport_ref, collection_ref, research_update):
        """Commit a review of exact saved outcomes; no worker is called."""
        with self._lock():
            record, state = self._state()
            return self._begin_research_review(record, state, direction_id, expected_revision,
                                               transport_ref, collection_ref, research_update)

    def _begin_research_review(self, record, state, direction_id, expected_revision,
                               transport_ref, collection_ref, research_update, planner_action=None):
        _need(state["phase"] in ("idle", "planned"), "resolve active work before research review")
        direction = self.store.get(direction_id)
        status = direction["note"].get("research_status")
        _need(direction["revision"] == expected_revision and status in ("pending_review", "reported"),
              "research review requires the current pending or reported direction")
        _need(_matches(direction, state["binding"]), "research review job/epoch mismatch")
        matching = [item for item in state["history"]
                    if item.get("attempt_id") == direction["note"].get("evolution_operation_id")]
        _need(len(matching) == 1 and matching[0]["settled"] and matching[0]["transport_ref"] == transport_ref
              and matching[0]["collection_ref"] == collection_ref, "review must name the exact settled outcome")
        update = self._research_update(research_update, state["claim_head"], state["binding"])
        _need(update is not None, "research review requires an explicit note and claim head")
        lease = budget.active_lease(state)
        lease_review = (planner_action is not None and lease is not None
                        and lease["direction_id"] == direction_id
                        and matching[0]["attempt_id"] in lease["attempt_ids"]
                        and matching[0]["attempt_id"] not in lease["reviewed_attempt_ids"])
        _need(status == "pending_review" or lease_review or update["claim_head"] != state["claim_head"],
              "a reported review amendment must append or correct claims")
        if planner_action is not None:
            budget.planner_action(copy.deepcopy(state), planner_action["proposal"],
                                  planner_action["planner_operation_id"], attempt_id=matching[0]["attempt_id"],
                                  now=self.clock())
        intent = {"direction_id": direction_id, "expected_revision": expected_revision,
                  "transport_ref": transport_ref, "collection_ref": collection_ref,
                  "research_update": update, "claim_head_before": state["claim_head"],
                  "planner_action": planner_action,
                  "previous_phase": state["phase"], "previous_pending": state["pending"]}
        intent["review_id"] = sha256_hex(_json(intent))
        state.update(phase="reviewing", pending=intent)
        self._save(record, state)
        return self._apply_research_review()

    def _apply_research_review(self):
        record, state = self._state()
        _need(state["phase"] == "reviewing", "no research review to replay")
        pending = state["pending"]
        for key in ("transport_ref", "collection_ref"):
            resolve_artifact(self.store, pending[key])
        update = self._research_update(pending["research_update"], pending["claim_head_before"], state["binding"])
        direction = self.store.get(pending["direction_id"])
        if direction["note"].get("research_review_id") != pending["review_id"]:
            _need(direction["revision"] == pending["expected_revision"], "direction changed during research review")
            note = dict(direction["note"], **update["note"])
            note.update(research_status="reported", research_review_id=pending["review_id"],
                        reviewed_outcome_refs=[pending["transport_ref"], pending["collection_ref"]])
            self.store.checkpoint(pending["direction_id"], direction["tip"], note=note,
                                  expected_revision=direction["revision"], status=direction["status"],
                                  resources=direction["resources"], evidence=direction["evidence"])
        state.update(claim_head=update["claim_head"], phase=pending["previous_phase"],
                     pending=pending["previous_pending"])
        if pending.get("planner_action") is not None:
            self._consume_review(state, pending["planner_action"])
        self._save(record, state)
        return self.snapshot()

    def _consume_review(self, state, action):
        direction = self.store.get(action["proposal"]["direction_id"])
        lease = budget.planner_action(state, action["proposal"], action["planner_operation_id"],
                                      attempt_id=direction["note"].get("evolution_operation_id"), now=self.clock())
        state["history"].append(dict(action, review_kind=action["proposal"]["kind"],
                                     direction_revision=direction["revision"], claim_head=state["claim_head"]))
        state.update(phase=_after_lease(lease), pending=None)

    def _planned_proposal(self, state, planner_ref, message):
        _need(state["phase"] == "planned" and state["pending"]["planner_ref"] == planner_ref, message)
        return state["pending"]["proposal"]

    def execute_review(self, *, planner_ref, research_update=None, observation_ref=None, verify_host=None):
        """Apply the planner's selected review using host-supplied evidence."""
        with self._lock():
            record, state = self._state()
            proposal = self._planned_proposal(state, planner_ref, "review does not match the current planner decision")
            _need(type(proposal) is dict, "planner review must be an object")
            common = {"kind", "direction_id", "expected_revision"}
            kind = proposal.get("kind")
            extra = {"research_review": {"transport_ref", "collection_ref", "note", "claim_requests"},
                     "maturation_review": {"maturation_ref", "decision"}}
            optional = {"lease_release"} if kind == "research_review" and self.program_lease == 2 else set()
            _need(kind in extra and common | extra[kind] <= set(proposal) <= common | extra[kind] | optional,
                  "invalid planner review fields")
            if "lease_release" in proposal:
                _text(proposal["lease_release"])
            action = {"planner_ref": planner_ref, "planner_operation_id": state["pending"]["operation_id"],
                      "proposal": copy.deepcopy(proposal)}
            if kind == "research_review":
                _need(type(proposal["claim_requests"]) is list and research_update is not None
                      and research_update.get("note") == proposal["note"]
                      and observation_ref is None and verify_host is None,
                      "host research review changed the selected note or supplied unrelated authority")
                _need(proposal["claim_requests"] or research_update["claim_head"] == state["claim_head"],
                      "claim changes require an explicit planner request")
                return self._begin_research_review(
                    record, state, proposal["direction_id"], proposal["expected_revision"],
                    proposal["transport_ref"], proposal["collection_ref"], research_update, planner_action=action)
            _need(research_update is None and proposal["decision"].get("action") in ("observe", "review"),
                  "maturation selection requires an observation or review decision")
            direction = self.store.get(proposal["direction_id"])
            _need(direction["revision"] == proposal["expected_revision"]
                  and direction["note"].get("maturation_ref") == proposal["maturation_ref"],
                  "stale selected maturation review")
            _need(_matches(direction, state["binding"]), "maturation review job/epoch mismatch")
            budget.planner_action(copy.deepcopy(state), proposal, action["planner_operation_id"],
                                  attempt_id=direction["note"].get("evolution_operation_id"), now=self.clock())
            # An invalid decision stays planned and can be charged as a
            # rejection. Verify once, then save the exact result.
            prepared = prepare_maturation(self.store, proposal["direction_id"], decision=proposal["decision"],
                                          expected_revision=proposal["expected_revision"],
                                          accounting=self._account(state),
                                          observation_ref=observation_ref, verify_host=verify_host)
            state.update(phase="maturation_reviewing", pending={"planner_action": action,
                         "observation_ref": observation_ref, "prepared": prepared})
            self._save(record, state)
            return self._apply_maturation_review(verify_host)

    def _apply_maturation_review(self, verify_host):
        record, state = self._state()
        _need(state["phase"] == "maturation_reviewing", "no selected maturation review to replay")
        pending = state["pending"]
        proposal = pending["planner_action"]["proposal"]
        direction = self.store.get(proposal["direction_id"])
        prepared = pending.get("prepared")
        if prepared is not None:
            target = prepared["checkpoint"]
            if without_timestamp(direction) != target:
                _need(direction["revision"] == proposal["expected_revision"]
                      and direction["note"].get("maturation_ref") == proposal["maturation_ref"],
                      "direction changed after prepared maturation review")
                identity = self.store.archive_evidence(prepared["record"]["candidate_id"],
                                                       {"maturation.json": _json(prepared["record"])})
                _need(prepared["ref"] == {"evidence_id": identity, "artifact": "maturation.json"},
                      "prepared maturation produced different immutable evidence")
                committed = self.store.checkpoint(proposal["direction_id"], target["tip"], note=target["note"],
                                                  expected_revision=proposal["expected_revision"],
                                                  status=target["status"], resources=target["resources"],
                                                  evidence=target["evidence"])
                _need(without_timestamp(committed) == target, "maturation checkpoint differs from prepared transition")
        elif direction["revision"] == proposal["expected_revision"]:
            # Journals saved before decisions carried prepared evidence.
            _need(direction["note"].get("maturation_ref") == proposal["maturation_ref"], "maturation head changed")
            checkpoint_maturation(self.store, proposal["direction_id"], decision=proposal["decision"],
                                  expected_revision=proposal["expected_revision"], accounting=self._account(state),
                                  observation_ref=pending["observation_ref"], verify_host=verify_host)
        else:
            latest = read_maturation(self.store, proposal["direction_id"])
            _need(direction["revision"] == proposal["expected_revision"] + 1 and latest is not None
                  and latest["previous_ref"] == proposal["maturation_ref"] and latest["decision"] == proposal["decision"],
                  "direction changed after selected maturation review")
            if proposal["decision"]["action"] == "observe":
                _need(latest["observation_ref"] == pending["observation_ref"], "replayed observation changed")
        self._consume_review(state, pending["planner_action"])
        self._save(record, state)
        return self.snapshot()

    def resume_maturation_review(self, *, verify_host=None):
        with self._lock():
            return self._apply_maturation_review(verify_host)

    def resume_research_review(self):
        with self._lock():
            return self._apply_research_review()

    def execute_lease_release(self, *, planner_ref):
        """Commit the planner's explicit (paid) lease release; no claim change."""
        with self._lock():
            record, state = self._state()
            _need(self.program_lease == 2, "program commitment is disabled")
            proposal = self._planned_proposal(state, planner_ref, "release does not match the current planner decision")
            _need(type(proposal) is dict and set(proposal) == {"kind", "direction_id", "expected_revision", "reason"}
                  and proposal["kind"] == "lease_release", "invalid commitment release fields")
            _text(proposal["reason"])
            direction = self.store.get(proposal["direction_id"])
            _need(type(proposal["expected_revision"]) is int and direction["revision"] == proposal["expected_revision"]
                  and _matches(direction, state["binding"]), "release requires the current campaign direction")
            operation_id = state["pending"]["operation_id"]
            lease = budget.planner_action(state, proposal, operation_id,
                                          attempt_id=direction["note"].get("evolution_operation_id"), now=self.clock())
            state["history"].append({"kind": "lease_release", "planner_ref": planner_ref,
                                     "planner_operation_id": operation_id, "proposal": copy.deepcopy(proposal)})
            state.update(phase=_after_lease(lease), pending=None)
            self._save(record, state)
            return self.snapshot()

    # --- research actions -----------------------------------------------------

    def execute_research_action(self, *, planner_ref, binding):
        """Apply one exact agent choice using a host-bound claim scope, no worker."""
        with self._lock():
            record, state = self._state()
            proposal = self._planned_proposal(state, planner_ref,
                                              "research action does not match current planner decision")
            required = {"kind", "action", "direction_id", "expected_revision", "candidate_id",
                        "expected_claim_head", "premises", "payload"}
            _need(type(proposal) is dict and set(proposal) == required and proposal["kind"] == "research_action",
                  "invalid research action proposal fields")
            _text(proposal["direction_id"])
            _need(type(proposal["action"]) is str and proposal["action"] in RESEARCH_ACTIONS, "unknown research action")
            _need(type(proposal["candidate_id"]) is str and type(proposal["premises"]) is list,
                  "research action requires a candidate identity and premise list")
            _need(proposal["direction_id"] != self.campaign_id, "campaign is not a research direction")
            directions = self.store.get()
            _need(proposal["direction_id"] in directions, "research action requires an existing direction")
            direction = directions[proposal["direction_id"]]
            candidate = self.store.get_candidate(proposal["candidate_id"])
            _need(type(proposal["expected_revision"]) is int and proposal["expected_revision"] == direction["revision"]
                  and direction["tip"] == proposal["candidate_id"], "stale research action direction")
            _need(_matches(candidate, state["binding"]), "research action job/epoch mismatch")
            _need(all(binding[key] == candidate[key] for key in ("job_id", "source_id", "measurement_epoch_id")),
                  "host research action binding disagrees with source")
            _need(proposal["expected_claim_head"] == state["claim_head"], "stale research action claim head")
            gate = self._check_premises(proposal["premises"], binding, self._claims(state).view(),
                                        expected_head=state["claim_head"], candidate_id=proposal["candidate_id"])
            _need(not gate["rejected"], "unsupported or corrected research action premise")
            operation_id = state["pending"]["operation_id"]
            budget.planner_action(copy.deepcopy(state), proposal, operation_id, now=self.clock())
            prepared = preview_action(self.store, proposal, operation_id, accounting=self._account(state),
                                      frozen_cases=self.frozen_cases, records=state.get("research_refs", {}))
            action = {"planner_ref": planner_ref, "planner_operation_id": operation_id,
                      "proposal": copy.deepcopy(proposal)}
            state.update(phase="research_acting", pending={"planner_action": action, "binding": binding,
                         "premise_gate": gate, "prepared": prepared})
            self._save(record, state)  # Exact inputs saved before the action changes anything.
            return self._apply_research_action()

    def _apply_research_action(self):
        record, state = self._state()
        _need(state["phase"] == "research_acting", "no research action to replay")
        pending = state["pending"]
        action, prepared = pending["planner_action"], pending["prepared"]
        proposal = action["proposal"]
        _need(state["claim_head"] == proposal["expected_claim_head"], "research action claim head changed")
        gate = self._check_premises(proposal["premises"], pending["binding"], self._claims(state).view(),
                                    expected_head=state["claim_head"], candidate_id=proposal["candidate_id"])
        _need(not gate["rejected"], "research action premises no longer valid")
        direction = self.store.get(proposal["direction_id"])
        already_applied = prepared["checkpoint"] is not None and without_timestamp(direction) == prepared["checkpoint"]
        if not already_applied:
            refs = perform_action(self.store, proposal, action["planner_operation_id"], accounting=self._account(state),
                                  frozen_cases=self.frozen_cases, records=state.get("research_refs", {}))
            _need(refs == prepared["result_refs"], "research action produced different immutable refs")
            direction = self.store.get(proposal["direction_id"])
            _need(prepared["checkpoint"] is None or without_timestamp(direction) == prepared["checkpoint"],
                  "research action checkpoint differs from admitted intent")
        for ref in prepared["result_refs"].values():
            if ref is not None:
                resolve_artifact(self.store, ref)
        index_result(state.setdefault("research_refs", {}), proposal, prepared["result_refs"])
        lease = budget.planner_action(state, proposal, action["planner_operation_id"], now=self.clock())
        state["history"].append(dict(action, research_action=proposal["action"], result_refs=prepared["result_refs"],
                                     direction_revision=direction["revision"], claim_head=state["claim_head"]))
        state.update(phase=_after_lease(lease), pending=None)
        self._save(record, state)
        return self.snapshot()

    def resume_research_action(self):
        with self._lock():
            return self._apply_research_action()

    # --- claims and progress ---------------------------------------------------

    def _research_update(self, update, base_head, binding):
        if update is None:
            return None
        _need(type(update) is dict and set(update) == {"note", "claim_head"}, "invalid post-turn research update")
        _need(type(update["note"]) is dict and set(update["note"]) == {"claim", "scope", "next_test", "evidence_uptake"},
              "post-turn note requires claim, scope, next_test and evidence_uptake")
        for value in update["note"].values():
            _text(value)
        ClaimLog(self.store, update["claim_head"], verify_quantity=self.verify_quantity).view()
        head = update["claim_head"]
        while head != base_head:
            _need(head is not None, "research claim head must descend from admitted head")
            envelope, _ = self.store.evidence(head["evidence_id"])
            _need(_matches(envelope["binding"], binding), "collected claim event belongs to another job/epoch")
            head = json.loads(resolve_artifact(self.store, head))["previous"]
        return copy.deepcopy(update)

    def update_claims(self, update):
        """Commit append-only claims or corrections against the current claim head.

        update(ClaimLog) may append or correct; it must not start work.
        Corrections made after planning are rechecked before any engineer starts.
        """
        with self._lock():
            record, state = self._state()
            _need(state["phase"] in ("idle", "planned"), "resolve active call before changing claims")
            log = self._claims(state)
            result = update(log)
            log.view()
            state["claim_head"] = log.head
            self._save(record, state)
            return result

    def record_progress_review(self, operation_id, ref, *, verify_review):
        """Mark a settled opportunity's outcome review complete, without inventing gain.

        The artifact is {kind:'evolution_progress_review', binding,
        opportunity_id, complete:bool}. The host verifier checks complete
        coverage; selection.work.advise separately checks every claimed gain.
        """
        with self._lock():
            record, state = self._state()
            _need(state["phase"] in ("idle", "planned"), "review requires no active call")
            raw = resolve_artifact(self.store, ref)
            value = json.loads(raw)
            expected = {"kind": "evolution_progress_review", "binding": state["binding"], "opportunity_id": operation_id}
            if "gain_kind" in value:
                _need(value["gain_kind"] == "native_local_frontier_observed_log_v1",
                      "unsupported progress coverage contract")
                expected["gain_kind"] = value["gain_kind"]
            if "outcome_kind" in value:
                _need(value.get("gain_kind") == "native_local_frontier_observed_log_v1"
                      and value["outcome_kind"] in ("observed_comparison", "baseline_initialization", "unknown")
                      and value.get("complete") is (value["outcome_kind"] != "unknown"),
                      "invalid typed native progress outcome")
                expected["outcome_kind"] = value["outcome_kind"]
            _need(set(value) == set(expected) | {"complete"} and type(value["complete"]) is bool
                  and all(value[key] == item for key, item in expected.items()), "misbound progress review")
            opportunity = next((item for item in state["opportunities"] if item["opportunity_id"] == operation_id), None)
            _need(opportunity is not None and opportunity["settled"], "progress review needs settled work")
            _need(callable(verify_review) and verify_review(ref, raw, expected) is True,
                  "host coverage verification required")
            opportunity["progress_complete"] = value["complete"]
            for field, name in (("gain_kind", "progress_gain_kind"), ("outcome_kind", "progress_outcome_kind")):
                if field in value:
                    opportunity[name] = value[field]
                else:
                    opportunity.pop(name, None)
            refs = opportunity.setdefault("progress_review_refs", [])
            if ref not in refs:
                refs.append(ref)
            self._save(record, state)
            return self.snapshot()

    def record_advice_preparation(self, value, *, expected_revision):
        """Save the optional advice-preparation status; core write failures stay fatal."""
        _need(type(value) is dict and set(value) == {
            "input_revision", "status", "error_type", "reason", "projected_episode_count", "elapsed_seconds"}
            and type(value["input_revision"]) is int and value["input_revision"] > 0
            and value["status"] in ("available", "unavailable")
            and type(value["elapsed_seconds"]) in (int, float)
            and 0 <= value["elapsed_seconds"] < float("inf"), "invalid native advice preparation status")
        if value["status"] == "available":
            _need(value["error_type"] is None and value["reason"] is None
                  and type(value["projected_episode_count"]) is int and value["projected_episode_count"] >= 0,
                  "available native advice preparation cannot carry failure")
        else:
            _need(type(value["error_type"]) is str and value["error_type"]
                  and type(value["reason"]) is str and value["projected_episode_count"] is None,
                  "unavailable native advice preparation needs exact failure")
        with self._lock():
            record, state = self._state()
            _need(record["revision"] == expected_revision and value["input_revision"] <= expected_revision
                  and state["phase"] == "idle", "native advice preparation requires the exact reviewed idle state")
            self._account(state)
            event = dict(copy.deepcopy(value), kind="native_advice_preparation")
            state["history"].append(event)
            state["native_advice_preparation"] = event
            self._save(record, state)
            return self.snapshot()

    # --- opportunities and planning --------------------------------------------

    def _reserve(self, state, *, direction, island, engineer, question, reservation, review_at,
                 planner_capacity=None, planning_window_seconds=None, lease_proposal=None):
        _need(self.clock() < state["deadline"], "campaign deadline reached")
        opportunity = dict(state["binding"], opportunity_id=str(uuid.uuid4()),
                           direction_id=direction, island_id=island, engineer_id=engineer,
                           question=question, opened_at=self.clock(), review_at=review_at,
                           requested=reservation, reserved=reservation, receipts=[],
                           settled=False, completed_sequence=None, progress_complete=False)
        if (lease_proposal is None and engineer == "planner" and self.program_lease == 2
                and planning_window_seconds is not None):
            _need(type(planning_window_seconds) in (int, float)
                  and 0 < planning_window_seconds < float("inf")
                  and 0 < review_at - opportunity["opened_at"] <= planning_window_seconds,
                  "planner must fit its prospectively declared window")
            opportunity["program_lease_planning_seconds"] = planning_window_seconds
        if lease_proposal is not None:
            budget.admit_engineer(state, lease_proposal, opportunity,
                                  bound=self.program_lease, now=opportunity["opened_at"])
        elif engineer == "planner":
            budget.prepare_planner(state, opportunity, planner_capacity, opportunity["opened_at"])
        state["opportunities"].append(opportunity)
        self._account(state)  # Aggregate reservations include all unfinished calls.
        return opportunity

    def _planner_advice(self, context, opportunity, accounting, advice_inputs, offers, progress,
                        verify_progress, advice_config):
        began = self.monotonic()
        try:
            _need(callable(advice_inputs) and not offers and not progress and verify_progress is None,
                  "advice adapter cannot replace separately supplied progress")
            context["planned_advice_offer"] = copy.deepcopy(opportunity)
            inputs = advice_inputs(copy.deepcopy(context))
            _need(type(inputs) is dict and set(inputs) <= {"offers", "progress", "verify_progress"},
                  "invalid advice input adapter")
            advice = advise(inputs.get("offers", ()), accounting, inputs.get("progress", ()),
                            config=advice_config, verify_progress=inputs.get("verify_progress"))
        except StateError:
            raise
        except Exception as error:
            if (type(advice_config) is not dict or
                    advice_config.get("gain_kind") != "native_local_frontier_observed_log_v1"):
                raise
            advice = advise((), accounting, (), config=advice_config)
            advice["advice_unavailable"] = {"stage": "planner_inputs",
                                            "error_type": type(error).__name__, "reason": str(error)[:1000]}
        advice["preparation_elapsed_seconds"] = self.monotonic() - began
        return advice

    def plan(self, planner, *, reservation, review_at, offers=(), progress=(),
             advice_config, verify_progress=None, planner_capacity=None, advice_inputs=None,
             planning_window_seconds=None):
        """Reserve resources for one planner call, save the intent, then call it once.

        planner(context, operation_id) returns {operation_id, proposal, costs,
        settled, artifacts}. proposal=None ends the campaign, except that a
        trusted adapter may add host_disposition for a checked failed call: a
        charged decision that leaves the loop idle. The host owns metering and
        settlement; artifacts are raw bytes. An exception leaves a saved
        planning intent; recover_planner is the only way to settle that call.
        """
        _need(callable(planner), "explicit planner required")
        with self._lock():
            record, state = self._state()
            _need(state["phase"] == "idle", "pending call must be reconciled, never redispatched")
            if budget.check_remaining(state, now=self.clock()):
                state.update(phase="lease_blocked", pending=None)
                self._save(record, state)
                return None
            accounting = self._account(state)
            if advice_inputs is None:
                advice = advise(offers, accounting, progress, config=advice_config,
                                verify_progress=verify_progress)
            if planner_capacity is not None:
                _need(type(planner_capacity) is dict
                      and set(planner_capacity) == {"total_limit", "steps_after_current"}
                      and type(planner_capacity["total_limit"]) is int and planner_capacity["total_limit"] > 0
                      and type(planner_capacity["steps_after_current"]) is int
                      and planner_capacity["steps_after_current"] >= 0,
                      "bounded host planner capacity required")
            opportunity = self._reserve(state, direction=self.campaign_id, island="campaign",
                                        engineer="planner", question="Choose the next scientific action.",
                                        reservation=reservation, review_at=review_at,
                                        planner_capacity=planner_capacity,
                                        planning_window_seconds=planning_window_seconds)
            context = self.snapshot()
            if planner_capacity is not None:
                context["planner_capacity"] = copy.deepcopy(planner_capacity)
            if advice_inputs is not None:
                advice = self._planner_advice(context, opportunity, accounting, advice_inputs, offers, progress,
                                              verify_progress, advice_config)
            context["advice"] = advice
            ref, data = self._prompt("planner.md")
            context["instructions"] = {"ref": ref, "text": data.decode("utf-8")}
            state.update(phase="planning", pending={"operation_id": opportunity["opportunity_id"],
                                                    "context": context, "advice": advice})
            self._save(record, state)  # Saved before any model call.
            result = planner(copy.deepcopy(context), opportunity["opportunity_id"])
            return self._finish_planner(result)

    def _cost_receipt(self, state, operation_id, ref, costs, settled, *, reconcile=False):
        _need(type(costs) is dict and set(costs) == set(state["limits"]), "complete actual-cost vector required")
        _need(type(settled) is bool, "explicit host settlement required")
        opportunity = next(item for item in state["opportunities"] if item["opportunity_id"] == operation_id)
        if reconcile:
            _need(not opportunity["settled"] and len(opportunity["receipts"]) == 1,
                  "only one unresolved cumulative charge can be reconciled")
            previous = opportunity["receipts"][0]
            for key, value in previous["costs"].items():
                _need(value is None or (costs[key] is not None and costs[key] >= value),
                      "recovery cannot erase known incurred cost")
            opportunity.setdefault("superseded_receipts", []).append(previous)
        else:
            _need(not opportunity["receipts"], "operation already has an accounted receipt")
        opportunity["receipts"] = [{"receipt_id": ref["evidence_id"], "costs": costs}]
        opportunity["settled"] = settled
        opportunity["completed_sequence"] = (sum(item["settled"] for item in state["opportunities"]) - 1
                                             if settled else None)

    def _check_planner_result(self, result, pending, reconcile):
        failed = result.get("host_disposition")
        if "host_disposition" in result:
            # Only a trusted adapter sets host_disposition after checking the raw
            # failed call; model text reaches this loop only inside proposal.
            _need(not reconcile and result["settled"] is True and result["proposal"] is None
                  and type(failed) is dict and set(failed) == {"kind", "failure"}
                  and failed["kind"] in ("planner_native_invocation_failed", "planner_response_rejected"),
                  "failed planner settlement must be exact and final")
        if reconcile:
            _need(result["proposal"] == pending["proposal"], "recovery cannot replace the planner's proposal")
            _need(result.get("host_recommendation") == pending.get("host_recommendation"),
                  "recovery cannot replace the planner's recommendation")
        artifacts = dict(result["artifacts"])
        recommendation = result.get("host_recommendation")
        if recommendation is not None:
            _need(failed is None or failed["kind"] == "planner_response_rejected",
                  "failed native invocation cannot carry a recommendation")
            _need(artifacts.get("recommendation-binding.json") == _json(recommendation),
                  "recommendation differs from its durable acceptance artifact")
            _need(not result["settled"] or "completion.json" in artifacts,
                  "settled recommendation requires durable whole-operation completion")
        _need("planner_context.json" not in artifacts, "planner context artifact collision")
        return failed, recommendation, artifacts

    def _finish_planner(self, result):
        record, state = self._state()
        pending = state["pending"]
        reconcile = state["phase"] == "unsettled" and "request" not in pending
        _need((state["phase"] == "planning" or reconcile) and result["operation_id"] == pending["operation_id"],
              "planner receipt does not match pending intent")
        _need(set(result) - {"host_disposition", "host_recommendation"}
              == {"operation_id", "proposal", "costs", "settled", "artifacts"},
              "invalid planner result fields")
        failed, recommendation, artifacts = self._check_planner_result(result, pending, reconcile)
        artifacts["planner_context.json"] = _json(pending["context"])
        payload = {key: value for key, value in result.items() if key != "artifacts"}
        if reconcile:
            payload["recovery_of"] = pending["planner_ref"]
        ref = self._artifact(record["tip"], "planner.json", payload, artifacts)
        self._cost_receipt(state, pending["operation_id"], ref, result["costs"], result["settled"], reconcile=reconcile)
        if reconcile:
            pending.setdefault("prior_planner_refs", []).append(pending["planner_ref"])
        lease_blocked = False
        lease_error = None
        if result["settled"] and (failed is not None or result["proposal"] is None):
            try:
                lease_blocked = budget.reject_planner(
                    state, pending["operation_id"],
                    "Planner invocation failed or returned no explicit commitment disposition.", now=self.clock())
            except budget.LeaseError as exc:
                # Keep the delivered receipt even if an overrun makes the held
                # ledger invalid; never lose costs to a refund attempt.
                lease_error = str(exc)
        accounting_error = lease_error
        try:
            self._account(state)
        except SelectionError as exc:
            accounting_error = str(exc)
        if recommendation is not None and result["settled"]:
            opportunity = next(row for row in state["opportunities"]
                               if row["opportunity_id"] == pending["operation_id"])
            in_budget = accounting_error is None and all(
                result["costs"][key] is not None and result["costs"][key] <= value
                for key, value in opportunity["reserved"].items())
            commit_record(state, recommendation, planner_ref=ref, context=pending["context"],
                          commit_revision=record["revision"] + 1, in_budget=in_budget)
        if failed is not None:
            # A charged decision with no proposal: keep its evidence, show it to
            # the next planner, and continue under the same clock and ceiling.
            state["history"].append({"kind": failed["kind"], "operation_id": pending["operation_id"],
                                     "binding": state["binding"], "settled": True, "costs": result["costs"],
                                     "failure_ref": ref, "failure": failed["failure"]})
            state.update(phase="idle", pending=None)
        else:
            state["pending"].update(proposal=result["proposal"], planner_ref=ref)
            if recommendation is not None:
                state["pending"]["host_recommendation"] = recommendation
            state["phase"] = ("unsettled" if not result["settled"] else
                              "idle" if result["proposal"] is None and budget.active_lease(state) else
                              "stopped" if result["proposal"] is None else "planned")
            if state["phase"] == "idle":
                state["pending"] = None
        if lease_blocked:
            state["phase"] = "lease_blocked"
        if accounting_error is not None:
            state.update(phase="accounting_blocked", accounting_error=accounting_error)
        self._save(record, state)
        return result["proposal"]

    def recover_planner(self, result, *, verify_recovery):
        with self._lock():
            _, state = self._state()
            _need(callable(verify_recovery) and verify_recovery(state["pending"], result) is True,
                  "host recovery identity/settlement verification required")
            return self._finish_planner(result)

    def reject_plan(self, reason):
        """Record why a proposal cannot be admitted; the planner cost stays charged."""
        _text(reason)
        with self._lock():
            record, state = self._state()
            _need(state["phase"] == "planned", "only an unexecuted proposal may be rejected")
            state["history"].append({"planner_ref": state["pending"]["planner_ref"],
                                     "rejected": reason, "proposal": state["pending"]["proposal"]})
            blocked = budget.reject_planner(state, state["pending"]["operation_id"], reason, now=self.clock())
            state.update(phase="lease_blocked" if blocked else "idle", pending=None)
            self._save(record, state)

    # --- engineer admission ------------------------------------------------------

    def _check_selection(self, state, proposal, admit_protection):
        """Check the planner's engineer proposal against the current archive."""
        optional = {"lease"} if self.program_lease == 2 else set()
        _need(type(proposal) is dict
              and ENGINEER_FIELDS <= set(proposal) <= ENGINEER_FIELDS | {"grant_intent", "semantic_donors"} | optional,
              "invalid planner proposal fields")
        grant_intent = proposal.get("grant_intent")
        if proposal["protection_ref"] is None:
            _need(grant_intent is None and admit_protection is None, "unprotected work cannot carry a grant")
        else:
            fields = {"reason", "next_test", "evidence_refs", "release_condition", "displaced_work"}
            _need(type(grant_intent) is dict and fields <= set(grant_intent) <= fields | {"last_faithful"},
                  "protected work requires explicit agent grant intent")
        for key in ("direction_id", "island_id", "engineer_id", "question", "reason", "evidence_uptake"):
            _text(proposal[key])
        _need(proposal["direction_id"] != self.campaign_id, "campaign is not a research direction")
        candidate = self.store.get_candidate(proposal["parent_candidate_id"])
        _need(all(candidate[key] == state["binding"][key] for key in state["binding"]), "parent job/epoch mismatch")
        directions = self.store.get()
        direction = directions.get(proposal["direction_id"])
        _need(proposal["expected_revision"] == (direction["revision"] if direction else 0), "stale selected direction")
        _need(direction is None or direction["tip"] == proposal["parent_candidate_id"],
              "continuation must preserve current tip; another parent requires a new direction")
        if proposal["protection_ref"] is not None:
            _need(direction is not None and direction["note"].get("maturation_ref") == proposal["protection_ref"],
                  "selected protection ref is not the current nomination/review/revival")
        _need(not any(item["tip"] == proposal["parent_candidate_id"]
                      and item["note"].get("research_status") == "pending_review"
                      for item in directions.values()),
              "pending research outcome requires an explicit review before more engineer work")
        _need(not pending_maturation_reviews(self.store, self._account(state)),
              "settled protection requires authenticated observation and agent review before more engineer work")
        membership = self._membership(proposal, direction)
        return candidate, direction, membership

    def _check_request(self, state, proposal, direction, request, run_investigation, validate_request):
        """Check the host's engineer request against the selected proposal."""
        request = validate_request(request)
        _need(request["direction_id"] == proposal["direction_id"]
              and request["tip"]["candidate_id"] == proposal["parent_candidate_id"], "request/selection mismatch")
        expected_anchor = direction["anchor"]["candidate_id"] if direction and direction["anchor"] else None
        _need((request["anchor"]["candidate_id"] if request["anchor"] else None) == expected_anchor,
              "request must carry current validated anchor")
        _need(request["continuation"]["prior_receipt"] == state["latest_receipts"].get(proposal["direction_id"]),
              "request must continue latest registered engineer receipt")
        semantic_donors = proposal.get("semantic_donors", [])
        _need(type(semantic_donors) is list and all(type(identity) is str for identity in semantic_donors)
              and len(semantic_donors) == len(set(semantic_donors))
              and proposal["parent_candidate_id"] not in semantic_donors,
              "distinct agent-selected semantic donor IDs required")
        if semantic_donors:
            _need(callable(run_investigation) and proposal.get("exchange_ref") is None,
                  "native semantic combination cannot bypass formal exchange")
            # _membership already binds the parent to its island. Semantic donors
            # are explicit ancestry; they get no formal exchange credit.
            _need(request["donors"] == [source_reference(self.store, identity) for identity in semantic_donors]
                  and all(_matches(self.store.get_candidate(identity), state["binding"])
                          for identity in semantic_donors),
                  "semantic donors differ from agent selection or campaign domain")
        else:
            _need(not request["donors"], "donors must enter through a validated exchange or explicit native selection")
        if proposal["exchange_ref"] is not None:
            exchange = validate_exchange(self.store, json.loads(resolve_artifact(self.store, proposal["exchange_ref"])),
                                         frozen_cases=self.frozen_cases)
            _need(exchange["recipient"]["island_id"] == proposal["island_id"], "selected island disagrees with exchange")
            _need(exchange["recipient"]["membership_ref"] == proposal["membership_ref"],
                  "exchange membership differs from selection")
            request = prepare_engineer_request(self.store, proposal["exchange_ref"], request,
                                               frozen_cases=self.frozen_cases)
        _need(request["next_question"] == proposal["question"], "request changed agent scientific question")
        return request

    def _engineer_prompt(self, state, proposal, opportunity, packet, premise_gate, prompt_context, request):
        """Assemble the engineer prompt bytes; fixes their order."""
        names = ["engineer.md"] + (["crossover.md"] if proposal["exchange_ref"] else [])
        lease = budget.active_lease(state)
        if lease is not None and opportunity["opportunity_id"] in lease["attempt_ids"]:
            packet["program_lease"] = copy.deepcopy(lease)
            names.append("program_lease_engineer.md")
        instructions = [self._prompt(name) for name in names]
        packet["instruction_refs"] = [ref for ref, data in instructions]
        prompt = b"\n\n".join(data for ref, data in instructions)
        prompt += b"\n\nDeclared evidence premises and verification scope\n" + _json(premise_gate) + PREMISE_NOTE
        if "program_lease" in packet:
            prompt += b"\n\nAdmitted adaptive commitment and remaining held resources\n" + _json(packet["program_lease"])
        if prompt_context is not None:
            _need(callable(prompt_context), "prompt context must be constructed under the admission lock")
            context_text, context_ref = prompt_context(self.snapshot())
            _text(context_text)
            request["read_only"].append(context_ref)
            prompt += b"\n\n" + context_text.encode("utf-8")
        return prompt

    def _object_ref(self, ref):
        envelope, _ = self.store.evidence(ref["evidence_id"])
        return file_ref((self.store.root / "objects" / envelope["artifacts"][ref["artifact"]]).resolve())

    def execute(self, request, *, binding, attempt_binding, launch_worker=None, settle_worker=None,
                capture_tip=None, collect, attempts_root, backend_options=None,
                admit_protection=None, prompt_context=None, run_investigation=None):
        """Admit and run one continuing engineer, then collect its result evidence.

        binding and attempt_binding come from the host's registered job and
        protocol identity, never from unchecked planner fields.
        collect(result, candidate_id) returns {costs, artifacts, card,
        exchange_receipt} and optionally research_update. It must meter all
        work in this opportunity, including any evaluator it calls, and its
        call is saved separately so it can be recovered.
        """
        from .engineer.request import validate_request
        _check_transport(collect, launch_worker, settle_worker, capture_tip, backend_options, run_investigation)
        with self._lock():
            record, state = self._state()
            _need(state["phase"] == "planned", "no admitted plan; ambiguous calls must be reconciled")
            proposal = copy.deepcopy(state["pending"]["proposal"])
            candidate, direction, membership = self._check_selection(state, proposal, admit_protection)
            for actual in (binding, attempt_binding):
                _need(all(actual[key] == candidate[key] for key in ("job_id", "source_id", "measurement_epoch_id")),
                      "host experiment binding disagrees with selected source")
            # Read the current claim head under the campaign writer lock.
            current_head = self._state()[1]["claim_head"]
            claims = ClaimLog(self.store, current_head, verify_quantity=self.verify_quantity).view()
            premise_gate = self._check_premises(proposal["premises"], binding, claims,
                                                expected_head=current_head,
                                                candidate_id=proposal["parent_candidate_id"])
            _need(not premise_gate["rejected"], "unsupported or corrected factual premise")
            history = [item for item in state["history"] if "attempt_id" in item]
            repeat_gate = check_repeat(attempt_binding, history, proposal["repeat_delta"],
                                       resolve_evidence=self.store.evidence)
            request = self._check_request(state, proposal, direction, request, run_investigation, validate_request)
            donors = [item["candidate_id"] for item in request["donors"]]
            selection = {key: proposal[key] for key in ("direction_id", "island_id", "engineer_id")}
            selection.update(working_source_id=candidate["source_id"],
                             donor_source_ids=list(dict.fromkeys(item["source_id"] for item in request["donors"])))
            decision = record_decision(state["pending"]["advice"], selection,
                                       reason=proposal["reason"], override_reason=proposal["override_reason"])
            opportunity = self._reserve(state, direction=proposal["direction_id"], island=proposal["island_id"],
                                        engineer=proposal["engineer_id"], question=proposal["question"],
                                        reservation=proposal["reservation"], review_at=proposal["review_at"],
                                        lease_proposal=proposal)
            opportunity["membership_ref"] = proposal["membership_ref"]
            if "wall_seconds" in proposal["reservation"]:
                _need(request["budget"]["wall_seconds"] <= proposal["reservation"]["wall_seconds"],
                      "engineer wall budget exceeds reservation")
            protection_intent = None
            if proposal["protection_ref"] is not None:
                grant_decision = dict(proposal["grant_intent"], action="grant",
                                      opportunity_id=opportunity["opportunity_id"])
                protection_intent = preview_maturation(self.store, proposal["direction_id"], decision=grant_decision,
                                                       expected_revision=proposal["expected_revision"],
                                                       accounting=self._account(state))
                protection_intent["decision"] = grant_decision
            packet = {"proposal": proposal, "decision": decision, "premise_gate": premise_gate,
                      "repeat_gate": repeat_gate, "claims": claims, "claim_head": current_head,
                      "planner_ref": state["pending"]["planner_ref"], "opportunity": opportunity,
                      "membership": membership}
            prompt = self._engineer_prompt(state, proposal, opportunity, packet, premise_gate,
                                           prompt_context, request)
            packet_ref = self._artifact(proposal["parent_candidate_id"], "assignment.json", packet,
                                        {"engineer_prompt.txt": prompt})
            request["read_only"].append(self._object_ref(packet_ref))
            request = validate_request(request)
            state.update(phase="admitted", pending={"operation_id": opportunity["opportunity_id"],
                         "proposal": proposal, "request": request, "packet_ref": packet_ref,
                         "attempt_binding": attempt_binding, "binding": binding, "donors": donors,
                         "claim_head_before": current_head, "protection_intent": protection_intent})
            self._save(record, state)
            # A legacy host hook may commit only the exact selected decision.
            # The ordinary path adds the reserved identity and commits directly.
            protection_ref = (admit_protection(proposal, opportunity, self._account(state))
                              if proposal["protection_ref"] is not None and admit_protection is not None else None)
            return self._dispatch(protection_ref=protection_ref, launch_worker=launch_worker,
                                  settle_worker=settle_worker, capture_tip=capture_tip, collect=collect,
                                  attempts_root=attempts_root, backend_options=backend_options,
                                  run_investigation=run_investigation)

    # --- dispatch and settlement ----------------------------------------------

    def _add_read_only(self, request, readable):
        if readable not in request["read_only"]:
            request["read_only"].append(readable)

    def _grant_protection(self, state, pending, proposal, direction, protection_ref, validate_request):
        """Commit the admitted protection grant and give the engineer its sources."""
        request = pending["request"]
        intent = pending.get("protection_intent")
        _need(intent is not None, "admitted protected work lacks agent grant intent")
        _need(protection_ref is None or protection_ref == intent["ref"], "host changed selected grant decision")
        if without_timestamp(direction) != intent["checkpoint"]:
            _need(direction["revision"] == proposal["expected_revision"]
                  and direction["note"].get("maturation_ref") == proposal["protection_ref"],
                  "admitted protection state changed")
            result = checkpoint_maturation(self.store, proposal["direction_id"], decision=intent["decision"],
                                           expected_revision=proposal["expected_revision"],
                                           accounting=self._account(state))
            _need(result["ref"] == intent["ref"], "grant differs from admitted intent")
            direction = self.store.get(proposal["direction_id"])
        _need(without_timestamp(direction) == intent["checkpoint"], "grant checkpoint differs from selected intent")
        protection_ref = intent["ref"]
        protection = validate_protection(self.store, protection_ref, direction_id=proposal["direction_id"],
                                         candidate_id=proposal["parent_candidate_id"],
                                         opportunity_id=pending["operation_id"], accounting=self._account(state))
        proposal["expected_revision"] = direction["revision"]
        pending["granted_protection_ref"] = protection_ref
        seed = protection.get("faithful_seed", protection["last_faithful"])
        material = {"grant_ref": protection_ref, "grant": protection,
                    "faithful_seed": source_reference(self.store, seed["candidate_id"]),
                    "witness": seed}
        anchor = protection["comparison_anchor"]
        material["comparison_anchor_source"] = source_reference(self.store, anchor["candidate_id"]) if anchor else None
        for source in (material["faithful_seed"], material["comparison_anchor_source"]):
            if source is not None:
                identities = [source["candidate_id"], source["source_id"]] + [
                    item["blob_id"] for item in self.store.source(source["source_id"])["files"]
                    if item["blob_id"] is not None]
                for identity in identities:
                    self._add_read_only(request, {"path": str((self.store.root / "objects" / identity).resolve()),
                                                  "sha256": identity})
        material_ref = self._artifact(proposal["parent_candidate_id"], "protected_assignment.json", material)
        refs = [material_ref] + seed["evidence_refs"]
        if anchor:
            _, artifacts = self.store.evidence(anchor["evidence_id"])
            refs += [{"evidence_id": anchor["evidence_id"], "artifact": name} for name in artifacts]
        for ref in refs:
            self._add_read_only(request, self._object_ref(ref))
        request = validate_request(request)
        pending.update(request=request, protection_packet_ref=material_ref)
        return direction

    def _dispatch(self, *, protection_ref, launch_worker=None, settle_worker=None, capture_tip=None,
                  collect, attempts_root, backend_options=None, run_investigation=None):
        from .engineer.request import validate_request
        record, state = self._state()
        _need(state["phase"] == "admitted", "only an unstarted admitted call may dispatch")
        _check_transport(collect, launch_worker, settle_worker, capture_tip, backend_options, run_investigation)
        pending = state["pending"]
        proposal = pending["proposal"]
        direction = self.store.get().get(proposal["direction_id"])
        _need(direction is None or direction["tip"] == proposal["parent_candidate_id"], "admitted parent changed")
        _need(not any(item["tip"] == proposal["parent_candidate_id"]
                      and item["note"].get("research_status") == "pending_review"
                      for item in self.store.get().values()), "research review remains pending")
        if proposal["protection_ref"] is not None:
            direction = self._grant_protection(state, pending, proposal, direction, protection_ref, validate_request)
        else:
            _need(protection_ref is None, "unexpected protection grant")
            _need((direction["revision"] if direction else 0) == proposal["expected_revision"],
                  "admitted direction changed")
        request = pending["request"]
        anchor = direction["anchor"]["candidate_id"] if direction and direction["anchor"] else None
        _need((request["anchor"]["candidate_id"] if request["anchor"] else None) == anchor, "admitted anchor changed")
        claims = self._claims(state).view()
        gate = self._check_premises(proposal["premises"], pending["binding"], claims,
                                    expected_head=state["claim_head"], candidate_id=proposal["parent_candidate_id"])
        _need(not gate["rejected"], "admitted premises no longer valid")
        self._membership(proposal, direction)
        _need(not pending_maturation_reviews(self.store, self._account(state)), "maturation review remains owed")
        _need(self.clock() < min(state["deadline"], proposal["review_at"]), "admitted opportunity expired")
        state["phase"] = "dispatching"
        self._save(record, state)
        envelope, _ = self.store.evidence(pending["packet_ref"]["evidence_id"])
        prompt_path = self.store.root / "objects" / envelope["artifacts"]["engineer_prompt.txt"]
        if run_investigation is None:
            from .engineer.runner import run_turn
            result = run_turn(request, attempts_root, launch_worker=launch_worker, settle_worker=settle_worker,
                              capture_tip=capture_tip, backend_options=backend_options, prompt_path=prompt_path)
        else:
            # The native entry owns the whole edit/test/repair loop and its own
            # settlement; it returns the usual transport envelope.
            result = run_investigation(request, attempts_root, prompt_path=prompt_path,
                                       operation_id=pending["operation_id"])
        candidate_id = self._retain_transport(result)
        collected = collect(result, candidate_id)
        return self._finish_engineer(collected)

    def dispatch_admitted(self, *, protection_ref=None, prompt_context=None, **runtime):
        """Continue an admitted call that never reached dispatching.

        A protection grant is committed or recognised from the saved intent;
        a supplied ref must match it. No new reservation is made. The engineer
        prompt saved at admission is used as is; a new prompt_context is ignored.
        """
        with self._lock():
            return self._dispatch(protection_ref=protection_ref, **runtime)

    def _retain_transport(self, result):
        record, state = self._state()
        pending = state["pending"]
        reconcile = state["phase"] == "unsettled" and "request" in pending
        _need(state["phase"] == "dispatching" or reconcile, "transport was already recorded")
        receipt = result["receipt"]
        if reconcile:
            _need(receipt["attempt_id"] == pending["receipt"]["attempt_id"]
                  and receipt.get("recovery_of") == pending["receipt_ref"],
                  "recovery must append to the exact unresolved receipt")
            pending.setdefault("prior_results", []).append(
                {key: pending[key] for key in ("receipt_ref", "transport_ref", "collection_ref", "costs")})
            pending["reconciling"] = True
        _need(file_ref(result["receipt_ref"]["path"]) == result["receipt_ref"]
              and json.loads(Path(result["receipt_ref"]["path"]).read_bytes()) == receipt,
              "transport receipt bytes changed")
        _need(receipt["request"] == pending["request"], "transport receipt belongs to another intent")
        parent = pending["proposal"]["parent_candidate_id"]
        source = receipt["captured_tip_source_id"]
        candidate_id = parent if source is None else self.store.candidate(
            state["binding"]["job_id"], state["binding"]["measurement_epoch_id"], source,
            parents=[parent], donors=pending["donors"])
        artifacts = {"transport.json": _json(receipt)}
        for name in TRANSPORT_FILES:
            path = Path(result["attempt_dir"]) / name
            if path.is_file():
                ref = file_ref(path)
                data = path.read_bytes()
                _need(sha256_hex(data) == ref["sha256"], "transport artifact changed")
                artifacts[name] = data
        evidence_id = self.store.archive_evidence(candidate_id, artifacts)
        pending.update(candidate_id=candidate_id, transport_ref={"evidence_id": evidence_id, "artifact": "transport.json"},
                       receipt_ref=result["receipt_ref"], receipt=receipt)
        state["phase"] = "collecting"
        self._save(record, state)
        return candidate_id

    def _finish_engineer(self, collected):
        record, state = self._state()
        _need(state["phase"] == "collecting", "collection was already recorded")
        fields = {"costs", "artifacts", "card", "exchange_receipt"}
        _need(fields <= set(collected) <= fields | {"research_update"}, "invalid host collection result")
        pending = state["pending"]
        receipt = pending["receipt"]
        update = collected.get("research_update")
        _need(update is None or receipt["settlement"].get("settled") is True,
              "settle the engineer before accepting its research update")
        _need(state["claim_head"] == pending["claim_head_before"], "claim head changed during the admitted turn")
        update = self._research_update(update, pending["claim_head_before"], state["binding"])
        if collected["exchange_receipt"] is not None:
            _need(pending["proposal"]["exchange_ref"] is not None, "unexpected crossover receipt")
            validate_receipt(self.store, pending["proposal"]["exchange_ref"], collected["exchange_receipt"],
                             child_candidate_id=pending["candidate_id"] if receipt["captured_tip_source_id"] else None,
                             frozen_cases=self.frozen_cases)
        ref = self._artifact(pending["candidate_id"], "collection.json",
                             {key: value for key, value in collected.items() if key != "artifacts"},
                             collected["artifacts"])
        pending.update(collection_ref=ref, costs=collected["costs"], card=collected["card"], research_update=update)
        state["phase"] = "settling"
        self._save(record, state)  # Saved inputs make the settlement replayable.
        return self._apply_settlement()

    def _checkpoint_direction(self, pending, proposal, receipt, direction, update):
        """Move the direction to the engineer's new tip, with its costs and notes."""
        _need((direction["revision"] if direction else 0) == proposal["expected_revision"],
              "direction changed during execution; retain settlement for reconciliation")
        resources = dict(direction["resources"] if direction else {})
        for key, cost in pending["costs"].items():
            if cost is not None:
                resources[key] = resources.get(key, 0) + cost
        card = pending["card"]
        if card is None:
            _need(direction and direction["note"].get("archive_card_ref"), "new direction requires an agent research card")
            old = read_record(self.store, direction["note"]["archive_card_ref"], frozen_cases=self.frozen_cases)
            card = {"idea": old["idea"], "historical_refs": [direction["note"]["archive_card_ref"]]}
            if old.get("last_faithful"):
                card["last_faithful"] = old["last_faithful"]
        # Attribution comes from the admitted assignment and retained transport,
        # never from an engineer's self-reported result or an inherited card.
        card = dict(card, author={"engineer_id": proposal["engineer_id"],
                                  "attempt_ref": pending["transport_ref"]})
        findings = update["note"] if update else UNREVIEWED_FINDINGS
        note = dict(direction["note"] if direction else {}, **findings)
        note.update(evolution_operation_id=pending["operation_id"],
                    research_status="reported" if update else "pending_review",
                    planner_intent={key: proposal[key] for key in ("reason", "question", "evidence_uptake")},
                    latest_transport_ref=pending["transport_ref"], research_update_ref=pending["collection_ref"])
        archive_checkpoint(self.store, proposal["direction_id"], pending["candidate_id"], card=card,
                           note=note, expected_revision=proposal["expected_revision"], resources=resources,
                           frozen_cases=self.frozen_cases,
                           status="active" if receipt["transport_complete"] else "unfinished",
                           evidence=list(direction["evidence"] if direction else []) +
                           [pending["transport_ref"]["evidence_id"], pending["collection_ref"]["evidence_id"]])

    def _apply_settlement(self):
        record, state = self._state()
        _need(state["phase"] == "settling", "no replayable settlement")
        pending = state["pending"]
        proposal, receipt = pending["proposal"], pending["receipt"]
        update = self._research_update(pending.get("research_update"), pending["claim_head_before"], state["binding"])
        settled = receipt["settlement"].get("settled") is True
        self._cost_receipt(state, pending["operation_id"], pending["collection_ref"], pending["costs"], settled,
                           reconcile=pending.get("reconciling", False))
        direction = self.store.get().get(proposal["direction_id"])
        applied = direction and direction["note"].get("evolution_operation_id") == pending["operation_id"]
        if receipt["captured_tip_source_id"] is not None and not applied:
            self._checkpoint_direction(pending, proposal, receipt, direction, update)
        elif applied:
            _need(direction["tip"] == pending["candidate_id"], "replayed operation has a different tip")
        if pending.get("reconciling"):
            old = [item for item in state["history"] if item.get("attempt_id") == pending["operation_id"]]
            _need(len(old) == 1 and not old[0]["settled"], "recovery lost its unresolved attempt")
            state.setdefault("superseded_attempts", []).extend(old)
            state["history"] = [item for item in state["history"] if item.get("attempt_id") != pending["operation_id"]]
        state["history"].append({"attempt_id": pending["operation_id"], "binding": pending["attempt_binding"],
                                 "settled": settled, "candidate_id": pending["candidate_id"],
                                 "packet_ref": pending["packet_ref"], "transport_ref": pending["transport_ref"],
                                 "collection_ref": pending["collection_ref"],
                                 "prior_results": pending.get("prior_results", [])})
        if settled:
            state["latest_receipts"][proposal["direction_id"]] = pending["receipt_ref"]
        if update is not None:
            state["claim_head"] = update["claim_head"]
        state.update(phase="idle" if settled else "unsettled", pending=None if settled else pending)
        if settled and budget.check_remaining(state, now=self.clock()):
            state["phase"] = "lease_blocked"
        try:
            self._account(state)
        except SelectionError as exc:
            state.update(phase="accounting_blocked", accounting_error=str(exc))
        self._save(record, state)
        return self.snapshot()

    def recover_engineer(self, *, result=None, collected=None, verify_recovery):
        """Settle one interrupted engineer call; never start a replacement.

        The verifier must establish receipt ownership, process/resource
        settlement and metering. A missing return from a killed caller does not
        mean the call never ran; there is no built-in PID scan.
        """
        with self._lock():
            _, state = self._state()
            _need(callable(verify_recovery) and verify_recovery(state["pending"], result, collected) is True,
                  "host recovery identity/settlement verification required")
            if state["phase"] in ("dispatching", "unsettled"):
                _need(result is not None, "exact recovered transport result required")
                self._retain_transport(result)
            if collected is not None:
                return self._finish_engineer(collected)
            return self.snapshot()

    def resume_settlement(self):
        """Replay only saved settlement inputs; no model or tool call."""
        with self._lock():
            return self._apply_settlement()

    # --- driver ---------------------------------------------------------------

    def run(self, planner, host_assignment, *, planning_reservation, planning_seconds,
            advice_config, max_steps, advice_inputs=None, after_review=None, planner_decision_limit=None,
            stop_before_deadline=False):
        """Run up to max_steps planner decisions; the planner sees every settled result.

        host_assignment returns execute() arguments for engineer work, only
        research_update / observation_ref+verify_host for a selected review, or
        binding for a selected research action. It must not start unmetered
        work. Review reasoning belongs to the metered planner call.
        """
        _need(type(max_steps) is int and max_steps > 0 and callable(host_assignment), "bounded host driver required")
        if planner_decision_limit is not None:
            consumed = sum(row["engineer_id"] == "planner" for row in self.snapshot()["state"]["opportunities"])
            _need(type(planner_decision_limit) is int and planner_decision_limit > 0
                  and max_steps <= planner_decision_limit - consumed,
                  "run would exceed the original total planner ceiling")
        for step_index in range(max_steps):
            state = self.snapshot()["state"]
            if state["phase"] == "idle":
                if stop_before_deadline and self.clock() + planning_seconds >= state["deadline"]:
                    break  # Leave the settled snapshot available for final selection.
                capacity = ({"total_limit": planner_decision_limit,
                             "steps_after_current": max_steps - step_index - 1}
                            if planner_decision_limit is not None else None)
                self.plan(planner, reservation=planning_reservation,
                          review_at=min(self.clock() + planning_seconds, state["deadline"]),
                          advice_config=advice_config, planner_capacity=capacity, advice_inputs=advice_inputs,
                          planning_window_seconds=planning_seconds)
            state = self.snapshot()["state"]
            if state["phase"] == "idle":
                continue  # A settled failed decision used this step.
            if state["phase"] != "planned":
                break
            try:
                self._run_selected(state, host_assignment, after_review)
            except ValueError as exc:
                if self._state()[1]["phase"] != "planned":
                    raise  # Admission/dispatch ambiguity is never a new search.
                self.reject_plan(type(exc).__name__ + ": " + str(exc))
        return self.snapshot()

    def _run_selected(self, state, host_assignment, after_review):
        proposal = state["pending"]["proposal"]
        planner_ref = state["pending"]["planner_ref"]
        arguments = host_assignment(copy.deepcopy(proposal), self.snapshot())
        kind = proposal.get("kind")
        if kind == "research_action":
            self.execute_research_action(planner_ref=planner_ref, **arguments)
        elif kind == "lease_release":
            self.execute_lease_release(planner_ref=planner_ref, **arguments)
        elif kind in ("research_review", "maturation_review"):
            reviewed = self.execute_review(planner_ref=planner_ref, **arguments)
            if after_review is not None:
                _need(callable(after_review), "host review completion hook must be callable")
                after_review(reviewed)
        else:
            self.execute(**arguments)
