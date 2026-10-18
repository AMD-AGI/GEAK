"""Development cells: group archived programs by agent-defined categories.

An island's charter declares a small categorical descriptor (named axes with
a fixed list of values). A membership record places one program in a cell of
that descriptor. Programs in the same cell and timing context compete on
their measured development quality (median of per-trial geometric means of
case times); the fastest is the cell's development elite.

The host injects the episode reader (NativeEvidence.episode) and the episode
review resolver. This module accepts no caller-supplied fitness numbers and
does no model call, GPU work, source change or store promotion.
"""

import copy
import json
import math
import statistics

from .._util import EvoError, canonical_json, require, sha256_hex
from .policy import proposal_ref
from .records import read_record
from .refs import read_ref


class CellError(EvoError, ValueError):
    """Host evidence for a cell is inconsistent."""


class CellProposalError(CellError):
    """A deterministic error in an agent-submitted cell action; not host corruption."""


def _need(condition, reason):
    require(condition, reason, CellError)


def _proposal_need(condition, reason):
    require(condition, reason, CellProposalError)


def _proposal_fields(value, required, optional=()):
    _proposal_need(type(value) is dict and set(required) <= set(value) <= set(required) | set(optional),
                   "invalid submitted cell fields")


def _hash(value):
    return sha256_hex(canonical_json(value))


def _text(value):
    return type(value) is str and bool(value.strip())


def _geomean(times, cases):
    return math.exp(math.fsum(math.log(times[case]) for case in cases) / len(cases))


# ---------------------------------------------------------------- descriptors

def validate_descriptor_spec(spec):
    """A small agent-defined finite categorical map; no built-in kernel axes."""
    _proposal_need(type(spec) is dict and set(spec) == {"schema", "descriptor_id", "revision", "axes"}
                   and spec["schema"] == "native-categorical-cells-v1"
                   and _text(spec["descriptor_id"]) and type(spec["revision"]) is int
                   and spec["revision"] > 0, "versioned cell descriptor specification required")
    _proposal_need(type(spec["axes"]) is list and spec["axes"], "cell descriptor needs explicit axes")
    names = []
    for axis in spec["axes"]:
        _proposal_need(type(axis) is dict and set(axis) == {"name", "meaning", "values"}
                       and _text(axis["name"]) and _text(axis["meaning"]),
                       "descriptor axis needs a name and operational meaning")
        values = axis["values"]
        _proposal_need(type(values) is list and values and all(_text(item) for item in values)
                       and len(set(values)) == len(values), "descriptor categories must be explicit unique text")
        names.append(axis["name"])
    _proposal_need(len(set(names)) == len(names), "duplicate descriptor axis")
    return copy.deepcopy(spec)


def descriptor_coordinates(spec, assignment):
    """Return (coordinates, None) for a supported assignment, else (None, reason)."""
    spec = validate_descriptor_spec(spec)
    if assignment["status"] != "supported":
        return None, "descriptor assignment is " + assignment["status"]
    values = assignment["descriptors"]
    if set(values) != {axis["name"] for axis in spec["axes"]}:
        return None, "descriptor assignment does not cover exactly the declared axes"
    if any(values[axis["name"]] not in axis["values"] for axis in spec["axes"]):
        return None, "descriptor value is unknown or outside its declared categories"
    if not assignment["evidence_refs"]:
        return None, "descriptor support is missing"
    return [values[axis["name"]] for axis in spec["axes"]], None


# ------------------------------------------------------------------- quality

def _quality_context(request, host):
    """The timing context two observations must share to be compared.

    Returns (context, None), or (None, reason) when runtime/evaluator identity
    is missing. Also used by geak_evo.selection.advice.
    """
    contract, runtime = request["contract"], host.get("runtime", {})
    cases, mapping = request["benchmark_case_ids"], request["benchmark_report_ids"]
    _need(type(cases) is list and cases and len(set(cases)) == len(cases)
          and set(cases) == set(mapping) and len(set(mapping.values())) == len(mapping),
          "benchmark cases/report mapping are incomplete or ambiguous")
    for key in ("image", "config_digest", "physical_devices"):
        if not runtime.get(key):
            return None, "physical runtime identity is missing: " + key
    _need(type(runtime["physical_devices"]) is list
          and all(_text(item) for item in runtime["physical_devices"]), "invalid physical device identity")
    if not request.get("evaluator_sha256") or not contract.get("timing_policy"):
        return None, "evaluator or actual timing-policy identity is missing"
    binding = request["binding"]
    return {"job_id": binding["job_id"], "measurement_epoch_id": binding["measurement_epoch_id"],
            "correctness_case_ids": request["correctness_case_ids"], "benchmark_case_ids": cases,
            "benchmark_report_ids": mapping, "phases": request["phases"],
            "evaluator_sha256": request["evaluator_sha256"], "contract_sha256": _hash(contract),
            "timing_policy": contract["timing_policy"],
            "runtime": {key: runtime[key] for key in ("image", "config_digest", "physical_devices")},
            "statistic": "median-of-complete-trial-equal-case-geomeans-ms-v1"}, None


def development_quality(observations):
    """A disclosed descriptive statistic over complete vectors, never a CI/LCB.

    Uses all unique same-source/same-context observations (not the fastest
    one, not an order-dependent pick) and keeps every vector with its geomean.
    """
    _need(observations, "development quality requires observations")
    first = observations[0]
    cases = first["comparison"]["benchmark_case_ids"]
    unique, scores = {}, []
    for row in observations:
        _need(row["source_id"] == first["source_id"] and row["comparison"] == first["comparison"],
              "cannot pool different sources or comparison contexts")
        operation = row["operation_id"]
        if operation in unique:
            _need(unique[operation] == row, "conflicting duplicate native operation")
            continue
        times = row["per_case_times_ms"]
        _need(set(times) == set(cases) and all(type(value) in (int, float)
              and math.isfinite(value) and value > 0 for value in times.values()),
              "quality requires every registered timing case with positive finite milliseconds")
        unique[operation] = row
        scores.append(_geomean(times, cases))
    retained = [dict(unique[key], trial_geomean_ms=_geomean(unique[key]["per_case_times_ms"], cases))
                for key in sorted(unique)]
    return {"statistic": first["comparison"]["statistic"], "value_ms": statistics.median(scores),
            "lower_is_better": True, "observation_count": len(unique),
            "observed_trial_geomean_range_ms": [min(scores), max(scores)],
            "observations": retained, "uncertainty": None,
            "scope": "native checked development observations; drift, retest selection and interference "
                     "remain possible; no independent endpoint qualification"}


# --------------------------------------------------------------------- cells

def reduce_cells(members):
    """Recompute each cell's elite from current evidence; keep every competitor."""
    groups = {}
    for member in members:
        groups.setdefault(member["cell_id"], []).append(member)
    cells = []
    for identity, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: (row["quality"]["value_ms"], row["source_id"], row["candidate_id"]))
        sources = {}
        for row in rows:
            source = row["source_id"]
            option = {key: row[key] for key in ("candidate_id", "direction_id", "membership_ref", "retained_roles")}
            if source not in sources:
                sources[source] = dict(copy.deepcopy(row), membership_options=[])
            _need(sources[source]["quality"] == row["quality"], "same cell source has conflicting quality")
            if option not in sources[source]["membership_options"]:
                sources[source]["membership_options"].append(option)
        rows = list(sources.values())
        best = rows[0]
        cells.append({"cell_id": identity, "island_id": best["island_id"],
                      "charter_ref": best["charter_ref"], "descriptor_spec": best["descriptor_spec"],
                      "coordinates": best["coordinates"], "comparison": best["comparison"],
                      "development_elite": copy.deepcopy(best), "contenders": copy.deepcopy(rows),
                      "tie_policy": "equal observed statistic ties use source ID then candidate ID; "
                                    "no superiority claim"})
    return cells


def select_cell(view, cell_id, expected_source_id):
    """Resolve an agent-chosen cell to the existing native parent fields."""
    matches = [row for row in view["cells"] if row["cell_id"] == cell_id]
    _need(len(matches) == 1, "cell is absent or ambiguous")
    elite = matches[0]["development_elite"]
    _need(elite["source_id"] == expected_source_id, "cell elite changed; use the current exact source")
    return {"parent_candidate_id": elite["candidate_id"], "island_id": elite["island_id"],
            "membership_ref": copy.deepcopy(elite["membership_ref"])}


# --------------------------------------------------------------- memberships

def _committed_charter(store, event, proposal, frozen_cases):
    ref = event["result_refs"]["charter_ref"]
    charter = read_record(store, ref, frozen_cases=frozen_cases)
    _need(charter["candidate_id"] == proposal["candidate_id"]
          and all(charter[key] == proposal["payload"][key] for key in
                  ("island_id", "question", "descriptor_spec", "previous_ref")),
          "committed charter differs from agent action")
    return ref, charter


def committed_memberships(store, state, frozen_cases):
    """Resolve committed charter/membership actions, versions and withdrawals.

    Returns (charters by ref hash, members by (direction, ref hash), retired keys).
    """
    charters, members, retired = {}, {}, set()
    for position, event in enumerate(state["history"]):
        action = event.get("research_action")
        if action not in ("charter", "membership"):
            continue
        proposal = event["proposal"]
        _need(proposal["action"] == action, "research action identity differs")
        if action == "charter":
            ref, charter = _committed_charter(store, event, proposal, frozen_cases)
            charters[_hash(ref)] = charter
            continue
        direction = proposal["direction_id"]
        checkpoint = store.get_revision(direction, event["direction_revision"])
        _need(event["direction_revision"] == proposal["expected_revision"] + 1
              and checkpoint["tip"] == proposal["candidate_id"]
              and checkpoint["note"]["archive_card_ref"] == event["result_refs"]["card_ref"],
              "membership differs from its immutable committed checkpoint")
        prior = store.get_revision(direction, proposal["expected_revision"])
        old = read_record(store, prior["note"]["archive_card_ref"], frozen_cases=frozen_cases)
        retained = proposal["payload"]["retain_refs"]
        retired.update((direction, _hash(ref))
                       for ref in old.get("membership_refs", []) + old.get("source_membership_refs", [])
                       if ref not in retained)
        ref = event["result_refs"]["membership_ref"]
        if ref is None:
            continue
        member = read_record(store, ref, frozen_cases=frozen_cases)
        submitted = proposal["payload"]["membership"]
        subject = submitted.get("candidate_id", proposal["candidate_id"])
        _need(member["candidate_id"] == subject
              and all(member[key] == submitted[key] for key in
                      ("island_id", "charter_ref", "assignment", "previous_ref")),
              "committed membership differs from agent action")
        _need(_hash(member["charter_ref"]) in charters, "membership charter is not a committed prior action")
        card = read_record(store, event["result_refs"]["card_ref"], frozen_cases=frozen_cases)
        _need(ref in card["membership_refs"] + card.get("source_membership_refs", []),
              "membership is absent from its committed card")
        owner = (direction, _hash(ref))
        retired.discard(owner)  # A later admitted identical assignment can rejoin.
        members[owner] = {"ref": ref, "record": member, "position": position, "direction_id": direction}
    return charters, members, retired


def _selected_membership(store, snapshot, membership_ref):
    """Resolve only the exact submitted membership in the frozen history."""
    proposal_ref(membership_ref)
    _, members, retired = committed_memberships(store, snapshot["state"], snapshot["frozen_cases"])
    identity = _hash(membership_ref)
    active = [item for key, item in members.items() if key[1] == identity and key not in retired]
    _proposal_need(active, "selected membership is absent or retired")
    return active[0]["record"]


def bind_engineer_identity(store, snapshot, proposal):
    """Fill in absent identity fields without choosing a parent or membership.

    The original proposal is not modified. An explicit non-null island is
    never replaced, even when its membership is missing.
    """
    if proposal is None or proposal.get("kind") is not None:
        return proposal
    bound = dict(proposal)
    for key in ("repeat_delta", "exchange_ref", "protection_ref", "override_reason"):
        bound.setdefault(key, None)  # absent and null both mean "not used"
    island, membership = proposal.get("island_id"), proposal.get("membership_ref")
    if membership is None:
        bound["membership_ref"] = None
        if island is None:
            bound["island_id"] = "unassigned"
    else:
        member = _selected_membership(store, snapshot, membership)
        _proposal_need(island is None or island == member["island_id"],
                       "explicit island differs from selected membership")
        if island is None:
            bound["island_id"] = member["island_id"]
    return bound


# ---------------------------------------------------------- action checking

def _validate_charter(store, snapshot, payload):
    _proposal_fields(payload, {"island_id", "question", "descriptor_spec", "previous_ref", "reason"})
    spec = validate_descriptor_spec(payload["descriptor_spec"])
    previous = payload["previous_ref"]
    if previous is None:
        _proposal_need(spec["revision"] == 1, "new descriptor starts at revision one")
        return
    prior = read_record(store, previous, frozen_cases=snapshot["frozen_cases"])
    old = validate_descriptor_spec(prior["descriptor_spec"])
    _proposal_need(spec["descriptor_id"] == old["descriptor_id"] and spec["revision"] == old["revision"] + 1,
                   "descriptor revision must follow its exact previous definition")


def _membership_subject_is_reviewed(store, snapshot, subject, episode_material, episode_review):
    """True if ``subject`` is a source role of a settled, reviewed native episode."""
    owners = {row["opportunity_id"]: row["direction_id"] for row in snapshot["state"]["opportunities"]}
    for event in snapshot["state"]["history"]:
        if event.get("settled") is not True or not event.get("collection_ref"):
            continue
        _, artifacts = store.evidence(event["collection_ref"]["evidence_id"])
        if "native_source_roles.json" not in artifacts:
            continue
        roles = json.loads(artifacts["native_source_roles.json"])
        selected_roles = [role for role, identity in roles.items() if identity == subject]
        if not selected_roles:
            continue
        candidate = store.get_candidate(subject)
        _need(all(candidate[key] == value for key, value in snapshot["state"]["binding"].items()),
              "membership subject belongs to another source domain")
        episode_review(store, snapshot, event, owners[event["attempt_id"]])
        _, material = episode_material(snapshot, event)
        _need(all(material["harvest"][role]["source_id"] == candidate["source_id"] for role in selected_roles),
              "native membership subject differs from authenticated retained source")
        return True
    return False


def _validate_membership(store, snapshot, proposal, episode_material, episode_review):
    _proposal_fields(proposal["payload"], {"retain_refs", "membership", "reason"})
    member = proposal["payload"]["membership"]
    if member is None:
        return
    _proposal_fields(member, {"island_id", "charter_ref", "assignment", "previous_ref"}, {"candidate_id"})
    assignment = member["assignment"]
    _proposal_fields(assignment, {"status", "descriptors", "evidence_refs", "reason"})
    _proposal_need(assignment["status"] in ("unknown", "tentative", "supported")
                   and type(assignment["descriptors"]) is dict
                   and type(assignment["evidence_refs"]) is list and _text(assignment["reason"]),
                   "invalid submitted descriptor assignment")
    _proposal_need(assignment["status"] != "supported" or assignment["evidence_refs"],
                   "supported assignment requires source-bound evidence")
    _need(callable(episode_material) and callable(episode_review),
          "native membership requires authenticated episode and review readers")
    subject = member.get("candidate_id", proposal["candidate_id"])
    _proposal_need(_text(subject), "membership subject must be an exact retained candidate ID")
    _proposal_need(_membership_subject_is_reviewed(store, snapshot, subject, episode_material, episode_review),
                   "membership subject is not a settled reviewed native source role")


def validate_cell_action(store, snapshot, proposal, *, episode_material=None, episode_review=None):
    """Check charter/membership actions and an engineer's chosen membership."""
    if proposal is None:
        return
    if proposal.get("kind") == "research_action" and proposal["action"] == "charter":
        _validate_charter(store, snapshot, proposal["payload"])
    if proposal.get("kind") == "research_action" and proposal["action"] == "membership":
        _validate_membership(store, snapshot, proposal, episode_material, episode_review)
    if proposal.get("kind") is not None or proposal["island_id"] == "unassigned":
        return
    state = snapshot["state"]
    member = _selected_membership(store, snapshot, proposal["membership_ref"])
    _proposal_need(member["candidate_id"] == proposal["parent_candidate_id"]
                   and member["island_id"] == proposal["island_id"]
                   and state.get("research_refs", {}).get("charters", {}).get(member["island_id"])
                   == member["charter_ref"],
                   "selected membership source, island or current charter differs")


# ----------------------------------------------------------------- projector

class NativeCellProjector:
    """Planner callback: snapshot -> (public cell view, exact file refs read).

    ``episode_material`` is the host's NativeEvidence.episode callback and
    ``episode_review`` is geak_evo.archive.succession.episode_review. There is
    deliberately no fallback when either is missing.
    """

    def __init__(self, store, *, episode_material, episode_review):
        self.store = store
        _need(callable(episode_material) and callable(episode_review),
              "host episode and immutable-review readers required")
        self.episode_material, self.episode_review = episode_material, episode_review

    def validate_action(self, snapshot, proposal):
        return validate_cell_action(self.store, snapshot, proposal,
                                    episode_material=self.episode_material, episode_review=self.episode_review)

    def _check_snapshot(self, snapshot):
        current = self.store.get(snapshot["campaign_id"])
        current_state = current["note"]["evolution"]
        committed = current["revision"] == snapshot["revision"] and current_state == snapshot["state"]
        reserved = (current_state.get("phase") == "planning"
                    and current_state.get("pending", {}).get("context") == snapshot)
        _need(committed or reserved,
              "cell projection requires the current committed campaign snapshot or exact reserved planner context")

    def _collect_episode(self, snapshot, event, direction, review_index, read_file, found):
        """Record source roles, failures and timing observations of one reviewed episode."""
        state = snapshot["state"]
        review = self.episode_review(self.store, snapshot, event, direction)
        matching_reviews = [index for index in review_index
                            if state["history"][index]["proposal"]["direction_id"] == direction
                            and state["history"][index]["direction_revision"] == review["revision"]]
        _need(len(matching_reviews) == 1, "exact authenticated episode review is ambiguous")
        # The optional third callback argument is the ASSIGNED parent,
        # not this history event's captured terminal candidate.
        _, material = self.episode_material(snapshot, event)
        envelope, artifacts = self.store.evidence(event["collection_ref"]["evidence_id"])
        _need(envelope["candidate_id"] == event["candidate_id"], "native role collection owner differs")
        role_ids = json.loads(artifacts["native_source_roles.json"])
        _need(set(role_ids) <= {"selected", "faithful", "unfinished", "terminal"} and "terminal" in role_ids,
              "native role inventory is invalid")
        for role, candidate_id in role_ids.items():
            candidate = self.store.get_candidate(candidate_id)
            _need(candidate["source_id"] == material["harvest"][role]["source_id"]
                  and all(candidate[key] == value for key, value in state["binding"].items()),
                  "native role source or domain differs")
            found["roles"].setdefault(candidate_id, []).append({
                "role": role, "attempt_id": event["attempt_id"], "direction_id": direction,
                "source_id": candidate["source_id"], "review_revision": review["revision"],
                "collection_ref": event["collection_ref"]})
            found["reviewed"][candidate_id] = min(found["reviewed"].get(candidate_id, len(state["history"])),
                                                  matching_reviews[0])
        for trial in material["native_trials"]:
            verified = trial["verified"]
            if verified is None:
                feedback = read_file(trial["feedback_ref"])
                found["failures"].setdefault(feedback["source_id"], {})[feedback["operation_id"]] = {
                    "feedback_ref": trial["feedback_ref"], "error": trial["error"]}
                continue
            source = verified["source_id"]
            index = {row["name"]: {key: row[key] for key in ("path", "sha256")}
                     for row in verified["artifact_index"]}
            request, host = read_file(index["request.json"]), read_file(index["host_receipt.json"])
            _need(request["binding"]["candidate_source_id"] == source
                  and all(request["binding"][key] == value for key, value in state["binding"].items()),
                  "quality source or campaign domain differs")
            context, reason = _quality_context(request, host)
            if context is None:
                found["unplaced"].append({"source_id": source, "feedback_ref": trial["feedback_ref"],
                                          "reason": reason})
                continue
            row = {"source_id": source, "operation_id": verified["operation_id"], "comparison": context,
                   "per_case_times_ms": verified["per_case_times_ms"], "feedback_ref": trial["feedback_ref"],
                   "request_ref": index["request.json"], "host_receipt_ref": index["host_receipt.json"],
                   "qualification_flags": {key: verified.get(key) for key in
                                           ("measurement_valid", "custody_proven", "reward_eligible")}}
            key = (source, _hash(context), verified["operation_id"])
            observations = found["observations"]
            _need(key not in observations or observations[key] == row, "conflicting repeated native observation")
            observations[key] = row

    def _place_members(self, state, charters, members, retired, found):
        """Split memberships into cell competitors, unplaced and historical."""
        eligible, historical = [], []
        heads = state.get("research_refs", {}).get("charters", {})
        failures = found["failures"]
        for identity, item in members.items():
            member, ref = item["record"], item["ref"]
            candidate_id = member["candidate_id"]
            source = self.store.get_candidate(candidate_id)["source_id"]
            charter = charters[_hash(member["charter_ref"])]
            base = {"candidate_id": candidate_id, "source_id": source, "membership_ref": ref,
                    "charter_ref": member["charter_ref"], "island_id": member["island_id"],
                    "direction_id": item["direction_id"], "retained_roles": found["roles"].get(candidate_id, [])}
            if identity in retired or heads.get(member["island_id"]) != member["charter_ref"]:
                historical.append(dict(base, reason="membership retired or charter superseded"))
                continue
            reason = None
            if candidate_id not in found["reviewed"] or found["reviewed"][candidate_id] >= item["position"]:
                reason = "membership lacks an earlier reviewed authenticated native source role"
            try:
                coordinates, descriptor_reason = descriptor_coordinates(charter["descriptor_spec"], member["assignment"])
            except CellError as error:
                coordinates, descriptor_reason = None, str(error)
            # Semantic membership and timing eligibility answer different questions.
            # Keep the source-supported assignment visible even when this exact
            # implementation cannot enter development-quality competition.
            semantic = {"coordinates": copy.deepcopy(coordinates),
                        "descriptor_status": member["assignment"]["status"]}
            quality_status = "unreviewed" if reason else "descriptor_unsupported"
            reason = reason or descriptor_reason
            if source in failures:
                reason = "same source has failed native validation; retained pending adjudication"
                quality_status = "native_trial_error"
            if reason:
                found["unplaced"].append(dict(base, **semantic, reason=reason,
                                              quality_status=quality_status,
                                              failed_trials=list(failures.get(source, {}).values())))
                continue
            grouped = {}
            for (observed_source, context_id, _), observation in found["observations"].items():
                if observed_source == source:
                    grouped.setdefault(context_id, []).append(observation)
            if not grouped:
                found["unplaced"].append(dict(base, **semantic, quality_status="no_comparable_timing",
                                              reason="no complete comparable native timing observation"))
            for context_id, rows in grouped.items():
                cell_binding = {"island_id": member["island_id"], "charter_ref": member["charter_ref"],
                                "coordinates": coordinates, "comparison_id": context_id}
                eligible.append(dict(base, cell_id=_hash(cell_binding), coordinates=coordinates,
                                     descriptor_spec=charter["descriptor_spec"], comparison=rows[0]["comparison"],
                                     quality=development_quality(rows)))
        return eligible, historical

    def __call__(self, snapshot):
        self._check_snapshot(snapshot)
        state, refs = snapshot["state"], {}
        owners = {row["opportunity_id"]: row["direction_id"] for row in state["opportunities"]}
        found = {"observations": {}, "failures": {}, "roles": {}, "reviewed": {}, "unplaced": []}
        charters, members, retired = committed_memberships(self.store, state, snapshot["frozen_cases"])

        def read_file(ref):
            raw = read_ref(ref)
            refs[(ref["path"], ref["sha256"])] = copy.deepcopy(ref)
            return json.loads(raw)

        for event in state["history"]:
            if event.get("settled") is not True or not event.get("collection_ref"):
                continue
            review_index = [index for index, row in enumerate(state["history"])
                            if row.get("review_kind") == "research_review"
                            and all(row["proposal"].get(key) == event[key]
                                    for key in ("transport_ref", "collection_ref"))]
            if not review_index:
                found["unplaced"].append({"attempt_id": event["attempt_id"],
                                          "reason": "native episode has no committed research review"})
                continue
            self._collect_episode(snapshot, event, owners[event["attempt_id"]], review_index, read_file, found)

        eligible, historical = self._place_members(state, charters, members, retired, found)
        cells = reduce_cells(eligible)
        view = {"schema": "native-development-cells-v1", "campaign_revision": snapshot["revision"],
                "cells": cells, "unplaced": found["unplaced"], "historical_memberships": historical,
                "source_roles": found["roles"], "failed_native_sources": found["failures"],
                "descriptor_scope": "agent-reviewed source-supported categories; semantic truth and diversity "
                                    "benefit are not proved",
                "quality_scope": "provisional native development competition, not validated anchors or "
                                 "independent endpoint claims"}
        view["offers"] = [dict(cell_id=cell["cell_id"], source_id=cell["development_elite"]["source_id"],
                               **select_cell(view, cell["cell_id"], cell["development_elite"]["source_id"]))
                          for cell in cells]
        return view, list(refs.values())
