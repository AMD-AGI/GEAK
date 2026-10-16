"""Inputs for the optional allocation advice (``selection.work.advise``).

Three parts:

* ``NativeDevelopmentAdvice`` measures, after each reviewed engineer episode,
  how much its selected program improved on the best comparable program
  measured before the episode started (observed log ratio of development
  quality). The result is saved once as a review plus a proof record; later
  planner turns only read those saved records.
* ``NativeAdviceReview`` runs that measurement after each committed review
  and records whether it was available.
* ``NativeAdviceInputs`` turns the current reported continuations into
  offers with the exact reservation the next engineer would receive, using
  an explicitly uniform prior. Offers do not create work or choose parents;
  the ordinary proposal checks still decide.
"""

import copy
import json
import math
from pathlib import Path

from .._util import canonical_json, require, sha256_hex
from ..archive.cells import _quality_context, development_quality
from ..archive.refs import read_ref
from ..archive.store import StateError
from ..archive.succession import episode_review, parent_outcomes
from ..budget import (active_lease, bounded_engineer_reservation, commitments, planner_followup_reserve,
                      prepare_planner)
from .work import OBSERVED_GAIN_KIND, OBSERVED_OUTCOME_KINDS, account_opportunities


REVIEW_ARTIFACT = "evolution_progress_review.json"
PROOF_ARTIFACT = "native_development_gain_proof.json"
GAIN_KIND = OBSERVED_GAIN_KIND
NS = 1_000_000_000
ADVICE_SCOPE = '''native_allocation_advice is optional PUCT-like advice over exact current
continuations, using an explicitly uniform policy prior. Its scored progress is
nonnegative observed log improvement over the best current per-source median
estimate from applicable local measurements, frozen before an investigation.
The comparator can change as repeated observations change per-source medians;
gains do not telescope into a cumulative improvement bound. It is not a confidence lower bound,
causal contribution, endpoint fitness, or a prediction of an unfinished idea's
potential. Authenticated baseline initialization establishes history without an
observed gain or credit; its costs count in ordinary accounting and exploration
service, and are excluded from the gain/cost rate numerator and denominator.
Complete slower/zero outcomes
retain exploration; missing coverage or cost remains unknown. Prior costs are
counted once from the ordinary ledger;
planner/review overhead remains campaign-level, not assigned to each direction.
Next cost is the exact reserved offer ceiling, not a predicted realized cost.
The standard-cost floor makes the next-cost divisor constant across these
bounded offers; past delivered costs still enter the score.
Only reported continuations with receipts receive offers, so none is untried.
New directions, donor combinations, and assigned cell work are not_comparable;
owed maturation reviews temporarily suppress offers. Agreement with a top row
records a choice, not proof that the advice influenced it or improved search.
Inspect the decomposed evidence and missingness. You may choose another source,
new direction, donor combination, review or stop. When choosing a scored current
continuation below the highest score, explain why in existing override_reason;
a new/unmatched direction is not_comparable. No new response field is required.
'''
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")



def _need(condition, message):
    require(condition, message, ValueError)


def _hash(value):
    return sha256_hex(canonical_json(value))


def _ref_key(ref):
    _need(type(ref) is dict and set(ref) == {"path", "sha256"}, "exact native file ref required")
    return ref["path"], ref["sha256"]


def _reviews(state, event, direction):
    return [row for row in state["history"] if row.get("review_kind") == "research_review"
            and row["proposal"].get("direction_id") == direction
            and all(row["proposal"].get(key) == event[key] for key in ("transport_ref", "collection_ref"))]


def _outcome(value, proof):
    """Check typed outcome coverage; older committed proofs keep their meaning."""
    if "outcome_kind" not in value:
        _need("outcome_kind" not in proof, "legacy native proof has new outcome coverage")
        return None
    kind = value["outcome_kind"]
    _need(kind in OBSERVED_OUTCOME_KINDS and proof.get("outcome_kind") == kind
          and value["complete"] is (kind != "unknown"), "native outcome coverage differs")
    if kind == "baseline_initialization":
        _need(proof["selected_source_id"] is not None and proof["selected_quality"] is not None
              and proof["baseline"] is None and proof["signed_frontier_log_ratio"] is None
              and proof["nonnegative_frontier_log_ratio"] is None and proof["credit"] is None
              and not proof["unknown_reasons"], "baseline initialization cannot claim a ratio or gain")
    elif kind == "observed_comparison":
        _need(proof["selected_source_id"] is not None and proof["selected_quality"] is not None
              and proof["baseline"] is not None and proof["signed_frontier_log_ratio"] is not None
              and proof["nonnegative_frontier_log_ratio"] is not None and not proof["unknown_reasons"],
              "observed comparison has incomplete native evidence")
    else:
        _need(proof["credit"] is None and proof["unknown_reasons"],
              "unknown native outcome cannot claim gain or complete coverage")
    return kind


def _timing_row(verified, trial, index, request, host):
    """One complete timing observation in the shape development_quality expects."""
    context, reason = _quality_context(request, host)
    if reason:
        return None, reason
    return {"source_id": verified["source_id"], "operation_id": verified["operation_id"],
            "comparison": context, "per_case_times_ms": verified["per_case_times_ms"],
            "feedback_ref": trial["feedback_ref"], "request_ref": index["request.json"],
            "host_receipt_ref": index["host_receipt.json"],
            "qualification_flags": {key: verified.get(key) for key in
                                    ("measurement_valid", "custody_proven", "reward_eligible")}}, None


# ------------------------------------------------------- development gains

class NativeDevelopmentAdvice:
    """Produce and read the observed development gain of each reviewed episode."""

    def __init__(self, loop, native_evidence):
        self.loop, self.store, self.native_evidence = loop, loop.store, native_evidence

    def _artifact(self, ref):
        _need(type(ref) is dict and set(ref) == {"evidence_id", "artifact"}, "exact stored evidence ref required")
        envelope, files = self.store.evidence(ref["evidence_id"])
        _need(ref["artifact"] in files, "stored advice artifact absent")
        return envelope, files, json.loads(files[ref["artifact"]])

    def _committed(self, snapshot):
        record = self.store.get_revision(snapshot["campaign_id"], snapshot["revision"])
        _need(record["note"]["evolution"] == snapshot["state"], "advice requires an exact committed snapshot")

    def _before(self, snapshot, event, opportunity):
        """The planner context frozen before the producing opportunity opened."""
        _, _, packet = self._artifact(event["packet_ref"])
        _need(packet["opportunity"]["opportunity_id"] == event["attempt_id"], "advice assignment owner differs")
        _, artifacts, _ = self._artifact(packet["planner_ref"])
        frozen = json.loads(artifacts["planner_context.json"])
        self._committed(frozen)
        _need(frozen["campaign_id"] == snapshot["campaign_id"]
              and frozen["state"]["binding"] == snapshot["state"]["binding"]
              and frozen["accounting"]["now"] <= opportunity["opened_at"],
              "advice comparator snapshot is not before the producing opportunity")
        _need(not any(row.get("attempt_id") == event["attempt_id"] for row in frozen["state"]["history"]),
              "producing episode appears in its own comparator history")
        return packet, frozen, {"evidence_id": packet["planner_ref"]["evidence_id"],
                                "artifact": "planner_context.json"}

    def _owned_operations(self, native, event):
        _need(native["invocation_id"] == event["attempt_id"], "native gain episode identity differs")
        owned, operation_ids = {}, set()
        for item in native["native_results"]:
            key = _ref_key(item["feedback_ref"])
            operation = item.get("operation_id")
            _need(type(operation) is str and operation and key not in owned and operation not in operation_ids,
                  "native gain operation inventory is missing or duplicated")
            owned[key] = operation
            operation_ids.add(operation)
        return owned

    def _episode_rows(self, snapshot, event, opportunity):
        """Timing rows, failures and errors for the trials this episode owns."""
        review = episode_review(self.store, snapshot, event, opportunity["direction_id"])
        _, material = self.native_evidence.episode(snapshot, event)
        owned = self._owned_operations(material["native_receipt"], event)
        rows, failures, errors, seen = [], {}, [], set()
        for trial in material["native_trials"]:
            key = _ref_key(trial["feedback_ref"])
            if key not in owned:
                continue  # A continuing harvest also retains prior episodes.
            _need(key not in seen, "duplicate owned native trial")
            seen.add(key)
            verified = trial["verified"]
            if verified is None:
                feedback = json.loads(read_ref(trial["feedback_ref"]))
                source = feedback.get("source_id")
                if type(source) is str and source:
                    failures.setdefault(source, []).append({"feedback_ref": trial["feedback_ref"],
                                                            "error": trial["error"]})
                else:
                    errors.append("failed native trial has no source binding")
                continue
            _need(verified["operation_id"] == owned[key] and verified["turn_id"] == event["attempt_id"],
                  "native gain trial belongs to another operation/episode")
            index = {}
            for item in verified["artifact_index"]:
                _need(item["name"] not in index, "duplicate native artifact name")
                index[item["name"]] = {name: item[name] for name in ("path", "sha256")}
            request = json.loads(read_ref(index["request.json"]))
            host = json.loads(read_ref(index["host_receipt.json"]))
            _need(request["binding"]["candidate_source_id"] == verified["source_id"]
                  and all(request["binding"][name] == value for name, value in snapshot["state"]["binding"].items()),
                  "native gain source or measurement domain differs")
            row, reason = _timing_row(verified, trial, index, request, host)
            if reason:
                errors.append(reason)
                continue
            rows.append(row)
        for ref in material["diagnostic_refs_not_revalidated"]:
            key = _ref_key(ref)
            if key in owned:
                _need(key not in seen, "native trial/diagnostic identity overlaps")
                seen.add(key)
        _need(seen == set(owned), "owned native gain inventory has a gap")
        return {"event": event, "opportunity": opportunity, "review": review,
                "selected": material["harvest"].get("selected"), "rows": rows,
                "failures": failures, "errors": errors, "host_provenance": material["host_provenance"]}

    def _episode_material(self, snapshot, events, opportunities):
        """Rows of every reviewed settled episode; a failed check becomes unknown evidence."""
        state, material, owned_ops = snapshot["state"], {}, set()
        for event in events:
            opportunity = opportunities[event["attempt_id"]]
            if not _reviews(state, event, opportunity["direction_id"]):
                continue
            try:
                item = self._episode_rows(snapshot, event, opportunity)
            except StateError:
                raise  # Core journal/object integrity is not missing native measurement evidence.
            except (ValueError, OSError, KeyError, TypeError) as error:
                # Failed validation is unknown evidence, never a zero gain.
                # Keep its charged opportunity and exact host refs visible.
                item = {"event": event, "opportunity": opportunity, "review": None,
                        "selected": None, "rows": [], "failures": {},
                        "errors": [type(error).__name__ + ": " + str(error)],
                        "host_provenance": {key: event[key] for key in
                                            ("packet_ref", "transport_ref", "collection_ref")}}
            for row in item["rows"]:
                _need(row["operation_id"] not in owned_ops, "native operation credited through multiple episodes")
                owned_ops.add(row["operation_id"])
            material[event["attempt_id"]] = item
        return material

    def _prior(self, frozen, material, opportunity):
        """Reviewed episodes visible in the frozen pre-opportunity context."""
        prior_events = {row["attempt_id"]: row for row in frozen["state"]["history"]
                        if row.get("attempt_id") and row.get("settled") is True and row.get("collection_ref")}
        rows, failures, refs, errors = [], {}, [], []
        for key, prior_event in prior_events.items():
            prior = material.get(key)
            if prior is None:
                continue
            _need(prior_event == prior["event"], "historical comparator event changed")
            if not _reviews(frozen["state"], prior_event, prior["opportunity"]["direction_id"]):
                continue
            _need(prior["opportunity"]["opened_at"] < opportunity["opened_at"],
                  "comparator episode does not precede target")
            rows.extend(prior["rows"])
            failures.update(prior["failures"])
            errors.extend(prior["errors"])
            frozen_review = episode_review(self.store, frozen, prior_event, prior["opportunity"]["direction_id"])
            refs.append({"attempt_id": key, "collection_ref": prior_event["collection_ref"],
                         "transport_ref": prior_event["transport_ref"], "review": frozen_review})
        return rows, failures, refs, errors

    def _new_proof(self, snapshot, event, item, material, credited_sources):
        """Build the (review, proof) pair for one episode with no saved proof yet."""
        state, opportunity = snapshot["state"], item["opportunity"]
        packet, frozen, context_ref = self._before(snapshot, event, opportunity)
        prior_rows, prior_failures, prior_refs, prior_errors = self._prior(frozen, material, opportunity)
        reasons = list(item["errors"]) + list(prior_errors)
        selected = item["selected"]
        source = selected["source_id"] if selected is not None else None
        selected_rows = [row for row in item["rows"] if row["source_id"] == source]
        selected_contexts = {_hash(row["comparison"]) for row in selected_rows}
        quality, baseline, signed, positive = None, None, None, None
        if source is None:
            reasons.append("no authenticated selected source")
        elif source in item["failures"] or source in prior_failures:
            reasons.append("selected source has failed native validation")
        elif not selected_rows or len(selected_contexts) != 1:
            reasons.append("selected source has no single complete current-episode timing context")
        else:
            quality = development_quality(selected_rows)
            context = selected_rows[0]["comparison"]
            grouped = {}
            for row in prior_rows:
                if row["comparison"] == context:
                    grouped.setdefault(row["source_id"], []).append(row)
            qualities = [(development_quality(rows), identity) for identity, rows in grouped.items()]
            if not qualities:
                # Authenticated selected measurements establish an initial
                # comparator without inventing a ratio or reward. Failed
                # prior validation cannot establish the absence of one.
                if prior_failures:
                    reasons.append("prior native validation failures prevent baseline initialization")
            else:
                baseline_quality, baseline_source = min(qualities, key=lambda pair: (pair[0]["value_ms"], pair[1]))
                baseline = {"source_id": baseline_source, "quality": baseline_quality}
                if baseline_source in prior_failures or baseline_source in item["failures"]:
                    reasons.append("baseline source has failed native validation")
                else:
                    signed = math.log(baseline_quality["value_ms"] / quality["value_ms"])
                    positive = max(0.0, signed)
        complete = not reasons
        if not complete:
            outcome_kind = "unknown"
        else:
            outcome_kind = "baseline_initialization" if baseline is None else "observed_comparison"
        repeated = source in credited_sources or any(row["source_id"] == source for row in prior_rows)
        credit = None
        if complete and positive is not None and positive > 0 and not repeated:
            credit = dict(state["binding"], opportunity_id=event["attempt_id"],
                          credit_id=_hash({"kind": OBSERVED_GAIN_KIND, "opportunity_id": event["attempt_id"],
                                           "source_id": source}), source_id=source,
                          baseline_source_id=baseline["source_id"], gain_log_observed=positive,
                          gain_kind=OBSERVED_GAIN_KIND)
            credited_sources.add(source)
        parent = self.store.get_candidate(packet["proposal"]["parent_candidate_id"])["source_id"]
        parent_rows = [row for row in prior_rows if row["source_id"] == parent
                       and selected_rows and row["comparison"] == selected_rows[0]["comparison"]]
        parent_ratio = (math.log(development_quality(parent_rows)["value_ms"] / quality["value_ms"])
                        if quality is not None and parent_rows
                        and parent not in prior_failures and parent not in item["failures"] else None)
        review = {"kind": "evolution_progress_review", "binding": state["binding"],
                  "opportunity_id": event["attempt_id"], "complete": complete, "gain_kind": OBSERVED_GAIN_KIND,
                  "outcome_kind": outcome_kind}
        proof = {"schema": "native-local-frontier-observed-proof-v2", "gain_kind": OBSERVED_GAIN_KIND,
                 "outcome_kind": outcome_kind,
                 "binding": state["binding"], "opportunity_id": event["attempt_id"],
                 "candidate_id": event["candidate_id"], "assignment_ref": event["packet_ref"],
                 "preassignment_context_ref": context_ref, "preassignment_revision": frozen["revision"],
                 "current_host_provenance": item["host_provenance"], "research_review": item["review"],
                 "prior_episodes": prior_refs, "selected_source_id": source, "selected_quality": quality,
                 "baseline": baseline, "signed_frontier_log_ratio": signed,
                 "nonnegative_frontier_log_ratio": positive, "parent_source_id": parent,
                 "parent_relative_signed_log_ratio_context_only": parent_ratio,
                 "repeated_local_source": repeated, "unknown_reasons": reasons,
                 "failed_current_sources": item["failures"], "failed_prior_sources": prior_failures,
                 "credit": credit, "progress_review": review, "uncertainty": None,
                 "scope": "Descriptive development advice only; no confidence bound, endpoint or causal "
                          "ancestry credit."}
        _outcome(review, proof)
        return review, proof

    def _project(self, snapshot):
        self._committed(snapshot)
        existing_inputs = self.inputs(snapshot)  # Authenticate saved numerical records first.
        state = snapshot["state"]
        opportunities = {row["opportunity_id"]: row for row in state["opportunities"]}
        events = [event for event in state["history"] if event.get("attempt_id")
                  and event.get("settled") is True and event.get("collection_ref")]
        events.sort(key=lambda event: (opportunities[event["attempt_id"]]["opened_at"], event["attempt_id"]))
        material = self._episode_material(snapshot, events, opportunities)
        # A first review may arrive after a later opportunity already claimed this
        # source. Keep every committed claim before projecting any new row.
        projected, credited_sources = [], {row["source_id"] for row in existing_inputs["progress"]}
        for event in events:
            item = material.get(event["attempt_id"])
            if item is None:
                continue
            existing = self._existing(item["opportunity"])
            if existing is not None:
                # Later note corrections do not rewrite an earlier numerical
                # observation or bind it to a different review.
                projected.append((event, existing[2], existing[3]))
                if existing[3]["credit"] is not None:
                    credited_sources.add(existing[3]["credit"]["source_id"])
                continue
            review, proof = self._new_proof(snapshot, event, item, material, credited_sources)
            projected.append((event, review, proof))
        return projected

    def _existing(self, opportunity):
        """The saved (ref, envelope, review, proof) for an opportunity, or None."""
        matches = []
        for ref in opportunity.get("progress_review_refs", []):
            envelope, artifacts, value = self._artifact(ref)
            if value.get("gain_kind") != OBSERVED_GAIN_KIND:
                continue
            _need(ref["artifact"] == REVIEW_ARTIFACT and PROOF_ARTIFACT in artifacts,
                  "typed native progress has no producer proof")
            matches.append((ref, envelope, value, json.loads(artifacts[PROOF_ARTIFACT])))
        _need(len(matches) <= 1, "duplicate typed native progress production")
        return matches[0] if matches else None

    def record_reviewed_progress(self, snapshot):
        projected = self._project(snapshot)
        return self._record_projected(snapshot, projected)

    def _record_projected(self, snapshot, projected):
        """Commit a checked projection; journal/evidence write failures stay fatal."""
        for event, value, proof in projected:
            opportunity = next(row for row in snapshot["state"]["opportunities"]
                               if row["opportunity_id"] == event["attempt_id"])
            existing = self._existing(opportunity)
            if existing is not None:
                _need(existing[2] == value and existing[3] == proof, "replayed native progress proof changed")
                continue
            identity = self.store.archive_evidence(event["candidate_id"], {
                REVIEW_ARTIFACT: canonical_json(value), PROOF_ARTIFACT: canonical_json(proof)})
            ref = {"evidence_id": identity, "artifact": REVIEW_ARTIFACT}

            def verify(actual_ref, raw, expected):
                envelope, artifacts = self.store.evidence(identity)
                return (actual_ref == ref and raw == canonical_json(value) and artifacts[REVIEW_ARTIFACT] == raw
                        and artifacts[PROOF_ARTIFACT] == canonical_json(proof)
                        and envelope["candidate_id"] == event["candidate_id"]
                        and all(value.get(key) == item for key, item in expected.items()))

            self.loop.record_progress_review(event["attempt_id"], ref, verify_review=verify)
            snapshot = self.loop.snapshot()
        return snapshot

    def _checked_credit(self, state, opportunity, events, sources):
        """Check one saved review/proof against committed state; return (ref, proof) or None."""
        existing = self._existing(opportunity)
        if existing is None:
            return None
        ref, envelope, value, proof = existing
        event = events.get(opportunity["opportunity_id"])
        _need(event is not None and event.get("settled") is True
              and envelope["candidate_id"] == event["candidate_id"]
              and proof["candidate_id"] == event["candidate_id"]
              and proof["binding"] == state["binding"]
              and proof["opportunity_id"] == opportunity["opportunity_id"]
              and proof["gain_kind"] == OBSERVED_GAIN_KIND
              and proof["assignment_ref"] == event["packet_ref"]
              and proof["progress_review"] == value
              and value["binding"] == state["binding"]
              and value["opportunity_id"] == opportunity["opportunity_id"]
              and opportunity.get("progress_gain_kind") == OBSERVED_GAIN_KIND
              and opportunity["progress_complete"] == value["complete"]
              and opportunity.get("progress_outcome_kind") == value.get("outcome_kind"),
              "committed native progress proof or coverage differs")
        _outcome(value, proof)
        credit = proof["credit"]
        if credit is None:
            return None
        _need(value["complete"] is True and credit["gain_kind"] == OBSERVED_GAIN_KIND
              and credit["opportunity_id"] == opportunity["opportunity_id"]
              and all(credit[key] == item for key, item in state["binding"].items())
              and credit["source_id"] == proof["selected_source_id"]
              and credit["baseline_source_id"] == proof["baseline"]["source_id"]
              and credit["gain_log_observed"] == proof["nonnegative_frontier_log_ratio"]
              and credit["source_id"] not in sources, "native gain credit/proof differs or duplicates a source")
        return ref, proof

    def inputs(self, snapshot):
        """Read saved proofs only (no native validation, no writes).

        Returns {"progress": credits, "verify_progress": callback} for ``advise``.
        """
        self._committed(snapshot)
        state, progress, approved, sources = snapshot["state"], [], {}, set()
        events = {row["attempt_id"]: row for row in state["history"] if row.get("attempt_id")}
        for opportunity in state["opportunities"]:
            checked = self._checked_credit(state, opportunity, events, sources)
            if checked is None:
                continue
            ref, proof = checked
            credit = proof["credit"]
            evidence_ref = {"evidence_id": ref["evidence_id"], "artifact": PROOF_ARTIFACT}
            expected = copy.deepcopy(credit)
            progress.append(dict(expected, evidence_ref=evidence_ref))
            approved[(evidence_ref["evidence_id"], evidence_ref["artifact"])] = (expected, copy.deepcopy(proof))
            sources.add(credit["source_id"])

        def verify_progress(ref, expected):
            key = (ref.get("evidence_id"), ref.get("artifact"))
            if key not in approved:
                return False
            bound, proof = approved[key]
            _, _, actual = self._artifact(ref)
            return expected == bound and actual == proof

        return {"progress": progress, "verify_progress": verify_progress}


# ------------------------------------------------------------------ offers

def native_advice_config(limits, engineer_limit):
    """Freeze declared dimensionless policy scales without tuning to outcomes."""
    _need(set(limits) == set(engineer_limit) == {"wall_seconds", "evaluations"}
          and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                  for value in limits.values()), "explicit finite advice capacities required")
    weights = {"wall_seconds": 0.5, "evaluations": 0.5}
    reference = math.fsum(weights[key] * engineer_limit[key] / limits[key] for key in weights)
    return {"weights": weights, "c_ref": reference, "half_life": 4 * reference,
            "epsilon": 0.01, "beta": 0.1, "gain_kind": GAIN_KIND}


def program_lease_accounting(context, intent):
    """Reproduce the exact planner transfer from its frozen pre-transfer state."""
    state = copy.deepcopy(context["state"])
    row = copy.deepcopy(intent["program_lease_planner_opportunity"])
    _need(row["opportunity_id"] == intent["operation_id"]
          and row["direction_id"] == context["campaign_id"] and row["engineer_id"] == "planner"
          and row["island_id"] == "campaign" and row["reserved"] == intent["reservation"]
          and row["requested"] == row["reserved"] and row["receipts"] == []
          and row["settled"] is False and row["completed_sequence"] is None
          and row["progress_complete"] is False
          and all(row[key] == value for key, value in state["binding"].items())
          and all(item["opportunity_id"] != row["opportunity_id"] for item in state["opportunities"]),
          "recorded program lease planner opportunity differs")
    prepare_planner(state, row, context["planner_capacity"], row["opened_at"])
    state["opportunities"].append(row)
    accounting = account_opportunities(state["opportunities"], binding=state["binding"],
                                       limits=state["limits"], deadline=state["deadline"],
                                       now=row["opened_at"], commitments=commitments(state))
    return state, accounting


def program_lease_offer(context, intent, campaign_end, state, *, engineer_limit):
    """Offer only the reviewed direction's held second engineer, with all owed later reviews."""
    lease = active_lease(state)
    if not (lease is not None and lease["status"] == "active" and lease["engineers_remaining"] == 1
            and lease["selection_pending"] and lease["attempt_ids"]
            and lease["attempt_ids"][-1] in lease["reviewed_attempt_ids"]):
        return None
    draws = [row for row in lease["planner_draws"] if row["operation_id"] == intent["operation_id"]]
    _need(len(draws) == 1 and draws[0]["funding"] in ("surplus", "commitment") and draws[0]["resolved"] is False,
          "committed engineer offer needs the exact current planner draw")
    # A surplus-funded selection releases its unused held selection tranche only
    # when the engineer is admitted. A commitment-funded one already transferred it.
    later_planners = lease["planner_tranches"] - int(draws[0]["funding"] == "surplus")
    _need(later_planners >= 1, "committed engineer lacks its held final review")
    followup = planner_followup_reserve(context, intent)
    held_engineer = lease["engineer_reservation"]
    _need(set(held_engineer) == set(engineer_limit)
          and all((type(value) in (int, float) and math.isfinite(value)) and value == int(value)
                  and 0 < value <= engineer_limit[key] for key, value in held_engineer.items()),
          "invalid held native engineer envelope")
    reservation = {key: int(value) for key, value in held_engineer.items()}
    followup_reservation = {key: later_planners * value for key, value in lease["planner_reservation"].items()}
    calendar = later_planners * lease["planner_seconds"]
    _need(all(lease["remaining_reservation"][key] >= value + followup_reservation[key]
              for key, value in reservation.items()), "invalid held engineer reservation")
    available = (reservation["wall_seconds"] > intent["admission"]["native_envelope_seconds"] + 30
                 and followup["decisions_after_current"] >= later_planners
                 and intent["deadline_monotonic_ns"] + math.ceil((reservation["wall_seconds"] + calendar) * NS)
                 < campaign_end)
    return {"schema": "native-program-lease-offer-v1", "direction_id": lease["direction_id"],
            "commitment_id": lease["commitment_id"], "predecessor_attempt_id": lease["attempt_ids"][-1],
            "reservation": reservation, "available": available,
            "followup_planner_reserve": {"reservation": followup_reservation,
                                         "decisions_after_current": followup["decisions_after_current"]},
            "calendar_after_engineer_seconds": calendar,
            "scope": "Only the bound reviewed direction may consume this existing E2 reservation. "
                     "All remaining ordinary review windows stay held; this is no new grant."}


def _current_boot_id():
    return _BOOT_ID_PATH.read_text().strip()


def engineer_offer_binding(snapshot, clock_ref, native_envelope_seconds, engineer_limit, *, boot_id=None):
    """The engineer reservation the planner would actually offer next.

    The loop supplies ``snapshot["planned_advice_offer"]`` after building the
    exact planner reservation; ``snapshot["state"]`` is the committed state
    before it. The planner recomputes these bounds before model delivery.
    The campaign clock's monotonic times are only meaningful on the boot that
    wrote them: ``boot_id`` (default: the running kernel's) must match.
    """
    upcoming = snapshot.get("planned_advice_offer")
    _need(type(upcoming) is dict and upcoming.get("engineer_id") == "planner"
          and upcoming.get("direction_id") == snapshot["campaign_id"]
          and upcoming.get("opportunity_id") not in {
              row["opportunity_id"] for row in snapshot["state"]["opportunities"]},
          "exact upcoming planner reservation required")
    _need(upcoming["review_at"] <= upcoming["opened_at"] + upcoming["reserved"]["wall_seconds"],
          "advice planner window exceeds its reserved duration")
    clock = json.loads(read_ref(clock_ref))
    _need(clock["boot_id"] == (_current_boot_id() if boot_id is None else boot_id),
          "advice offer clock belongs to another boot")
    state, accounting = snapshot["state"], snapshot["accounting"]
    _need(set(accounting["limits"]) == {"wall_seconds", "evaluations"}
          and all(upcoming[key] == value for key, value in state["binding"].items()),
          "advice reservation or resource domain differs")
    intent = {"operation_id": upcoming["opportunity_id"], "reservation": upcoming["reserved"],
              "program_lease_planner_opportunity": upcoming,
              "admission": {"native_envelope_seconds": native_envelope_seconds}}
    prepared = None
    if state.get("program_leases") or "program_lease_planning_seconds" in upcoming:
        prepared, reserved = program_lease_accounting(snapshot, intent)
        available = reserved["available"]
    else:
        available = {key: limit - math.fsum(
            [row["committed"][key] for row in accounting["opportunities"]] + [upcoming["reserved"][key]])
            for key, limit in accounting["limits"].items()}
        _need(all(value >= 0 for value in available.values()), "planner reservation exceeds advice allowance")
    start_ns, start_unix = clock["started_monotonic_ns"], clock["started_unix"]
    campaign_end = min(start_ns + int(clock["wall_seconds"] * NS),
                       start_ns + int((state["deadline"] - start_unix) * NS))
    planner_end = min(campaign_end, start_ns + int((upcoming["review_at"] - start_unix) * NS))
    intent["deadline_monotonic_ns"] = planner_end
    followup = planner_followup_reserve(snapshot, intent)
    reservation, usable = bounded_engineer_reservation(available, campaign_end, planner_end,
                                                       native_envelope_seconds, engineer_limit=engineer_limit,
                                                       followup=followup)
    result = {"reservation": reservation, "available": usable, "clock_ref": copy.deepcopy(clock_ref),
              "planner_operation_id": upcoming["opportunity_id"],
              "followup_planner_reserve": followup, "campaign_deadline_monotonic_ns": campaign_end}
    held = (program_lease_offer(snapshot, intent, campaign_end, prepared, engineer_limit=engineer_limit)
            if prepared is not None else None)
    if held is not None:
        result["program_lease_offer"] = held
    return result


class NativeAdviceReview:
    """After-review hook: project development advice, record the outcome.

    A numerical projection error is recorded as ``unavailable`` and research
    goes on. Store integrity errors (``StateError``) and the archive and
    checkpoint writes are outside that boundary: they propagate and are never
    retried as new research.
    """

    def __init__(self, loop, development):
        self.loop, self.development = loop, development

    def __call__(self, snapshot):
        began = self.loop.monotonic()
        event = {"input_revision": snapshot["revision"], "status": "available",
                 "error_type": None, "reason": None, "projected_episode_count": None}
        try:
            projected = self.development._project(snapshot)
        except StateError:
            raise
        except Exception as error:
            event.update(status="unavailable", error_type=type(error).__name__, reason=str(error)[:1000])
        else:
            snapshot = self.development._record_projected(snapshot, projected)
            event["projected_episode_count"] = len(projected)
        event["elapsed_seconds"] = self.loop.monotonic() - began
        return self.loop.record_advice_preparation(event, expected_revision=snapshot["revision"])


class NativeAdviceInputs:
    """Loop advice hook: snapshot -> {"progress", "verify_progress", "offers"}."""

    def __init__(self, loop, development, *, clock_ref, native_envelope_seconds, engineer_limit, boot_id=None):
        self.loop, self.store, self.development = loop, loop.store, development
        self.clock_ref = copy.deepcopy(clock_ref)
        self.native_envelope_seconds = native_envelope_seconds
        self.engineer_limit = copy.deepcopy(engineer_limit)
        self.boot_id = boot_id

    def _offer(self, snapshot, direction_id, row, pending_tips, directions, offered):
        """One continuation offer for a reported direction, or None if not offered."""
        direction = directions[direction_id]
        _need(direction["tip"] == row["tip"], "advice source differs from its committed direction")
        if direction["note"].get("research_status") != "reported" or row["tip"] in pending_tips:
            return None
        prior = snapshot["state"]["latest_receipts"].get(direction_id)
        if prior is None:
            return None
        receipt = json.loads(read_ref(prior))
        histories = [history for history in snapshot["state"]["history"]
                     if history.get("attempt_id") == receipt["attempt_id"]]
        _need(len(histories) == 1 and histories[0].get("settled") is True
              and histories[0]["candidate_id"] == row["tip"],
              "advice continuation lacks its exact settled terminal history")
        outcomes = parent_outcomes(self.store, snapshot, row["tip"])
        _need(outcomes, "advice continuation must be an exact reviewed native role")
        opportunities = [item for item in snapshot["state"]["opportunities"]
                         if item["opportunity_id"] == receipt["attempt_id"]]
        _need(len(opportunities) == 1 and opportunities[0]["direction_id"] == direction_id,
              "advice engineer identity differs from producing opportunity")
        candidate = self.store.get_candidate(row["tip"])
        return dict(snapshot["state"]["binding"], direction_id=direction_id,
                    island_id="unassigned", engineer_id=opportunities[0]["engineer_id"],
                    priority_weight=1, estimated_costs=copy.deepcopy(offered["reservation"]),
                    offer_binding={"snapshot_revision": snapshot["revision"],
                                   "direction_revision": row["revision"], "parent_candidate_id": row["tip"],
                                   "source_id": candidate["source_id"], "membership_ref": None,
                                   "clock_ref": copy.deepcopy(self.clock_ref),
                                   "reservation": copy.deepcopy(offered["reservation"])})

    def __call__(self, snapshot):
        committed = self.store.get_revision(snapshot["campaign_id"], snapshot["revision"])
        _need(committed["note"]["evolution"] == snapshot["state"],
              "advice inputs require the exact committed campaign snapshot")
        prepared = snapshot["state"].get("native_advice_preparation")
        _need(not prepared or prepared["status"] != "unavailable",
              "native advice projection unavailable: " +
              (str(prepared.get("error_type")) + ": " + str(prepared.get("reason")) if prepared else "unknown"))
        inputs = self.development.inputs(snapshot)
        _need(type(inputs) is dict and set(inputs) <= {"progress", "verify_progress"},
              "development adapter must supply only authenticated progress")
        envelope = engineer_offer_binding(snapshot, self.clock_ref, self.native_envelope_seconds,
                                          self.engineer_limit, boot_id=self.boot_id)
        offers = []
        held = envelope.get("program_lease_offer")
        any_available = envelope["available"] or (held is not None and held["available"])
        if any_available and not snapshot.get("owed_maturation_reviews"):
            archive = snapshot["archive"]["directions"]
            directions = {name: self.store.get_revision(name, row["revision"]) for name, row in archive.items()}
            pending_tips = {direction["tip"] for direction in directions.values()
                            if direction["note"].get("research_status") == "pending_review"}
            for direction_id, row in sorted(archive.items()):
                if direction_id == snapshot["campaign_id"]:
                    continue
                offered = held if held is not None and direction_id == held["direction_id"] else envelope
                if not offered["available"]:
                    continue
                offer = self._offer(snapshot, direction_id, row, pending_tips, directions, offered)
                if offer is not None:
                    offers.append(offer)
        return dict(inputs, offers=offers)


def verify_delivered_advice(advice, context, actual_offer, store):
    """Recheck offered identities and exact reservation at real planner delivery."""
    _need(advice.get("gain_kind") == GAIN_KIND and advice.get("advisory_only") is True,
          "typed observed native allocation advice required")
    for row in advice["rows"]:
        binding = row["offer_binding"]
        held = actual_offer.get("program_lease_offer")
        offered = held if held is not None and row["direction_id"] == held["direction_id"] else actual_offer
        direction = store.get_revision(row["direction_id"], binding["direction_revision"])
        candidate = store.get_candidate(binding["parent_candidate_id"])
        _need(binding["snapshot_revision"] == context["revision"]
              and direction["tip"] == binding["parent_candidate_id"]
              and candidate["source_id"] == binding["source_id"]
              and binding["clock_ref"] == actual_offer["clock_ref"]
              and binding["reservation"] == offered["reservation"]
              and offered["available"] is True,
              "delivered allocation offer differs from actual native source or reservation")
    return copy.deepcopy(advice)
