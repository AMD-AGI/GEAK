"""Bounded extra work for a promising but slower idea ("maturation").

A direction can nominate an idea, receive one finite work grant from the
host's opportunity ledger, record what the host observed, and be reviewed,
shelved, refuted, realized or revived. Each step is a JSON record saved in the
store and chained to the previous one through the direction's note.

The engineer names a "last faithful" source: the latest code that still
implements the idea. That is the engineer's claim, not a verified fact. Facts
about delivered work or results come only from records that the host's
verify_host callable accepts; flags in the agent's decision never count. This
module cannot launch work, promote an anchor or make a candidate eligible.

opportunity_view() builds the read-only summary of these witnesses that the
planner sees next to the ordinary engineer offer.
"""

import copy
import json

from .._util import EvoError, canonical_json, require, sha256_hex
from .claims import resolve_artifact


class MaturationError(EvoError, ValueError):
    pass


def _require(condition, message):
    require(condition, message, MaturationError)


def _text(value):
    _require(type(value) is str and bool(value.strip()), "expected nonempty scientific rationale")


def _json(value):
    try:
        return canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise MaturationError("record must be finite JSON") from exc


def _loads(data):
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, "duplicate JSON field")
            value[key] = item
        return value
    try:
        value = json.loads(data, object_pairs_hook=pairs)
        _json(value)
        return value
    except (ValueError, UnicodeError) as exc:
        raise MaturationError("invalid maturation JSON") from exc


def _binding(candidate):
    return {key: candidate[key] for key in ("job_id", "measurement_epoch_id")}


def _candidate(store, candidate_id, binding):
    candidate = store.get_candidate(candidate_id)
    _require(_binding(candidate) == binding, "candidate job/epoch mismatch")
    return candidate


def _raw(store, ref, binding, candidate_id=None):
    envelope, data = resolve_artifact(ref, store.evidence)
    _require(_binding(envelope["binding"]) == binding, "evidence job/epoch mismatch")
    if candidate_id is not None:
        _require(envelope["candidate_id"] == candidate_id, "evidence candidate mismatch")
    return envelope, data


def _refs(store, refs, binding, candidate_id=None):
    _require(type(refs) is list, "evidence references must be a list")
    for ref in refs:
        _raw(store, ref, binding, candidate_id)


def _faithful(store, witness, binding):
    if witness is None:
        return
    _require(type(witness) is dict and set(witness) == {"candidate_id", "claim", "evidence_refs"},
             "invalid last-faithful witness fields")
    _candidate(store, witness["candidate_id"], binding)
    _text(witness["claim"])
    _require(bool(witness["evidence_refs"]), "faithfulness needs exact source evidence")
    _refs(store, witness["evidence_refs"], binding, witness["candidate_id"])


def _comparison_anchor(store, anchor, binding):
    """Check the anchor saved at nomination; never replace it with a later one."""
    if anchor is None:
        return
    _require(type(anchor) is dict and set(anchor) == {"candidate_id", "evidence_id", "checks"},
             "invalid comparison anchor snapshot")
    _candidate(store, anchor["candidate_id"], binding)
    envelope, _ = store.evidence(anchor["evidence_id"])
    _require(envelope["candidate_id"] == anchor["candidate_id"]
             and _binding(envelope["binding"]) == binding, "comparison anchor evidence mismatch")
    _require(type(anchor["checks"]) is dict and all(anchor["checks"].get(key) is True
             for key in ("independent", "settled", "correct", "eligible")),
             "comparison anchor lacks its recorded eligibility checks")


def _check_retraction(store, record, decision, binding):
    _require(decision["action"] == "review"
             and decision["retract_faithful_ref"] == record.get("previous_ref")
             and decision.get("last_faithful", False) is None
             and record.get("last_faithful") is None,
             "invalid fidelity retraction target or transition")
    _text(decision["reason"])
    _require(bool(decision["evidence_refs"]), "fidelity retraction requires correction evidence")
    _, prior_data = _raw(store, decision["retract_faithful_ref"], binding)
    _require(_loads(prior_data).get("last_faithful") is not None,
             "retraction target has no current faithful witness")


def _check_history_record(store, record, envelope, direction_id, binding):
    """Check one saved maturation record; fills in missing comparison anchors as None."""
    _require(record.get("kind") == "maturation" and record.get("schema_version") == 1,
             "unknown maturation record")
    _require(record.get("direction_id") == direction_id and record.get("binding") == binding
             and record.get("candidate_id") == envelope["candidate_id"], "maturation identity mismatch")
    _require(record.get("native_eligible") is False, "maturation cannot grant native eligibility")
    _faithful(store, record.get("last_faithful"), binding)
    # Older records did not save a comparator. They stay readable with an
    # explicit unknown; never rebuild one from today's anchor.
    record.setdefault("comparison_anchor", None)
    _comparison_anchor(store, record["comparison_anchor"], binding)
    if record.get("grant") is not None:
        record["grant"].setdefault("comparison_anchor", None)
        _require(record["grant"]["comparison_anchor"] == record["comparison_anchor"],
                 "grant comparison anchor changed")
        if "faithful_seed" in record["grant"]:
            _require(record["grant"]["faithful_seed"] is not None, "grant requires a faithful seed witness")
            _faithful(store, record["grant"]["faithful_seed"], binding)
        _candidate(store, record["grant"]["start_candidate_id"], binding)
    _refs(store, record["decision"]["evidence_refs"], binding)
    if "retract_faithful_ref" in record["decision"]:
        _check_retraction(store, record, record["decision"], binding)
    for name in ("observation_ref", "result_ref"):
        if record.get(name) is not None:
            _raw(store, record[name], binding)


def read_maturation(store, direction_id):
    """Return the direction's latest committed maturation record, or None.

    Walks and checks the whole chain of earlier records.
    """
    state = store.get(direction_id)
    ref = state["note"].get("maturation_ref")
    if ref is None:
        return None
    _require(ref["evidence_id"] in state["evidence"], "maturation head is not committed evidence")
    binding, seen, latest = _binding(state), set(), None
    while ref is not None:
        key = _json(ref)
        _require(key not in seen, "maturation history cycle")
        seen.add(key)
        envelope, data = _raw(store, ref, binding)
        record = _loads(data)
        _check_history_record(store, record, envelope, direction_id, binding)
        if latest is None:
            latest = record
        else:
            _require(all(latest[key] == record[key] for key in ("proposition", "scope")),
                     "a changed proposition requires a new nomination/direction")
            _require(latest["comparison_anchor"] == record["comparison_anchor"],
                     "fixed comparison anchor changed across maturation history")
        ref = record.get("previous_ref")
    return latest


def _retraction_targets(store, record, binding):
    """Yield each record whose witness a later record retracted (newest first)."""
    while record is not None:
        ref = record.get("previous_ref")
        prior = _loads(_raw(store, ref, binding)[1]) if ref is not None else None
        if "retract_faithful_ref" in record["decision"]:
            yield prior
        record = prior


def _new_witness_evidence(store, witness, prior, binding):
    """A witness that was once withdrawn needs at least one new evidence payload."""
    def payloads(refs):
        return {sha256_hex(_raw(store, ref, binding)[1]) for ref in refs}
    withdrawn_payloads = set()
    for target in _retraction_targets(store, prior, binding):
        withdrawn = target["last_faithful"]
        if witness["candidate_id"] == withdrawn["candidate_id"]:
            withdrawn_payloads.update(payloads(withdrawn["evidence_refs"]))
    if withdrawn_payloads:
        _require(bool(payloads(witness["evidence_refs"]) - withdrawn_payloads),
                 "a withdrawn witness needs new source evidence before reinstatement")


def _opportunity_rows(accounting):
    rows = {row["opportunity_id"]: row for row in accounting["opportunities"]}
    _require(len(rows) == len(accounting["opportunities"]), "duplicate opportunity")
    return rows


def _ledger(record, accounting):
    _require(type(accounting) is dict and accounting.get("binding") == record["binding"],
             "current host accounting for this job/epoch is required")
    rows = _opportunity_rows(accounting)
    for key in record["used_opportunities"]:
        _require(key in rows and rows[key]["direction_id"] == record["direction_id"],
                 "previous opportunity cannot disappear or change direction")
    grant = record.get("grant")
    if grant is not None:
        _require(grant["opportunity_id"] in rows, "missing granted opportunity")
        row = rows[grant["opportunity_id"]]
        for key in ("direction_id", "engineer_id", "island_id", "requested", "reserved", "opened_at", "review_at"):
            _require(row[key] == grant[key], "granted identity/envelope changed")
    return rows


def pending_maturation_reviews(store, accounting):
    """List grants in this ledger whose work has settled and still needs review.

    Never infers delivered work. New work for the direction should wait for a
    host observation and an explicit agent review. Grants from other ledgers
    are ignored.
    """
    _require(type(accounting) is dict and type(accounting.get("binding")) is dict,
             "current host accounting is required")
    rows = _opportunity_rows(accounting)
    result = []
    for direction_id, state in store.get().items():
        if _binding(state) != accounting["binding"] or state["note"].get("maturation_ref") is None:
            continue
        record = read_maturation(store, direction_id)
        grant = record.get("grant")
        if (record["lifecycle"] not in ("active", "interrupted", "review_due") or grant is None
                or grant["opportunity_id"] not in rows):
            continue
        row = _ledger(record, accounting)[grant["opportunity_id"]]
        if row["settled"]:
            result.append({"direction_id": direction_id, "maturation_ref": state["note"]["maturation_ref"],
                           "opportunity_id": row["opportunity_id"],
                           "assigned_candidate_id": grant["start_candidate_id"],
                           "needs_observation": record["observation"] is None or not record["observation"]["settled"],
                           "needs_review": True})
    return result


def _verified(store, ref, binding, expected, verify_host):
    envelope, data = _raw(store, ref, binding)
    _require(callable(verify_host) and verify_host(ref, data, expected) is True,
             "authentic host receipt verification required")
    return envelope, data


def validate_maturation_decision(decision):
    """Check the agent's decision JSON: fields and text only, no store access."""
    _require(type(decision) is dict, "agent decision must be an object")
    decision = _loads(_json(decision))
    action = decision.get("action")
    required = {
        "nominate": {"proposition", "scope"},
        "grant": {"opportunity_id", "release_condition", "displaced_work"},
        "observe": set(),
        "review": {"outcome", "changed_since_prior", "revival_condition"},
        "revive": {"trigger_kind", "trigger_reason"},
    }
    common = {"action", "reason", "next_test", "evidence_refs"}
    optional = {"last_faithful"} | ({"result_ref", "retract_faithful_ref"} if action == "review" else set())
    _require(type(action) is str and action in required and common | required[action] <= set(decision)
             and set(decision) <= common | required[action] | optional,
             "invalid agent decision fields; host facts are not agent inputs")
    for key in (common | required[action]) - {"evidence_refs"}:
        _text(decision[key])
    _require(type(decision["evidence_refs"]) is list, "evidence references must be a list")
    return decision


def _apply_grant(record, decision, state, direction_id, accounting):
    _require(record["lifecycle"] in ("nominated", "reviewed", "revived"), "review or revival required before another grant")
    _require(record["last_faithful"] is not None, "protection needs an exact faithful source witness")
    rows = _ledger(record, accounting)
    key = decision["opportunity_id"]
    _require(key in rows and key not in record["used_opportunities"], "grant requires a fresh reserved opportunity")
    row = rows[key]
    _require(row["direction_id"] == direction_id and not row["settled"] and not row["expired"],
             "grant must be current, unsettled and assigned to this direction")
    _require(any(value > 0 for value in row["remaining_reservation"].values()), "grant has no remaining opportunity")
    record["grant"] = {key: row[key] for key in ("opportunity_id", "direction_id", "engineer_id", "island_id",
                      "requested", "reserved", "opened_at", "review_at")}
    # Work continues in the current workspace. Going back to a different
    # faithful seed is the engineer's choice, never a host rollback.
    record["grant"]["start_candidate_id"] = state["tip"]
    record["grant"]["faithful_seed"] = _loads(_json(record["last_faithful"]))
    record["grant"]["comparison_anchor"] = record["comparison_anchor"]
    record["used_opportunities"].append(decision["opportunity_id"])
    record.update(lifecycle="active", observation=None, observation_ref=None, result_ref=None)


WORK_KINDS = {"none": 0, "review_only": 1, "implementation_or_diagnosis": 2}


def _apply_observe(store, record, direction_id, binding, accounting, observation_ref, verify_host):
    _require(record["lifecycle"] in ("active", "interrupted", "review_due"), "no live grant to observe")
    row = _ledger(record, accounting)[record["grant"]["opportunity_id"]]
    expected = dict(kind="maturation_observation", binding=binding, direction_id=direction_id,
                    opportunity_id=row["opportunity_id"], assigned_candidate_id=record["grant"]["start_candidate_id"])
    envelope, data = _verified(store, observation_ref, binding, expected, verify_host)
    observed = _loads(data)
    fields = set(expected) | {"candidate_id", "delivered", "work_kind", "settled", "interrupted"}
    _require(type(observed) is dict and set(observed) == fields
             and all(observed[key] == value for key, value in expected.items()), "observation binding/fields mismatch")
    _candidate(store, observed["candidate_id"], binding)
    _require(observed["candidate_id"] == envelope["candidate_id"], "observed source is not the archived source")
    _require(all(type(observed[key]) is bool for key in ("delivered", "settled", "interrupted")), "observation facts need explicit booleans")
    _require(observed["work_kind"] in WORK_KINDS and (observed["delivered"] or observed["work_kind"] == "none"), "work requires actual source delivery")
    _require(observed["settled"] == row["settled"], "observation and host settlement disagree")
    previous = record["observation"]
    if previous:
        _require((not previous["delivered"] or observed["delivered"])
                 and WORK_KINDS[observed["work_kind"]] >= WORK_KINDS[previous["work_kind"]]
                 and (not previous["settled"] or observed["settled"]), "cumulative work/settlement cannot regress")
    record.update(observation=observed, observation_ref=observation_ref,
                  lifecycle="interrupted" if observed["interrupted"] else ("review_due" if observed["settled"] else "active"))


def _apply_review(store, record, decision, state, direction_id, binding, accounting, retracting, verify_host):
    correction_only = decision["outcome"] == "withdraw_fidelity"
    _require(not correction_only or retracting, "withdraw_fidelity requires an explicit witness retraction")
    _require(correction_only or record["lifecycle"] in ("nominated", "active", "review_due", "interrupted"),
             "no pending maturation review")
    if record["grant"] is not None and not correction_only:
        row = _ledger(record, accounting)[record["grant"]["opportunity_id"]]
        _require(row["settled"] and record["observation"] is not None and record["observation"]["settled"],
                 "review requires authenticated settlement; interruption is not cleanup")
    outcomes = {"continue": "reviewed", "shelve": "shelved", "abandon": "abandoned", "refute": "refuted",
                "realize": "realized", "withdraw_fidelity": record["lifecycle"]}
    _require(decision["outcome"] in outcomes, "unknown scoped review outcome")
    _require(decision["outcome"] in ("refute", "realize") or "result_ref" not in decision,
             "result evidence must name a scoped refutation or realization")
    if decision["outcome"] in ("refute", "realize"):
        target = record["last_faithful"]["candidate_id"] if record["last_faithful"] else state["tip"]
        _require(decision["outcome"] != "realize" or record["last_faithful"] is not None, "realization requires a faithful witness")
        expected = dict(kind=decision["outcome"], binding=binding, direction_id=direction_id,
                        proposition=record["proposition"], scope=record["scope"], candidate_id=target,
                        source_id=_candidate(store, target, binding)["source_id"])
        _raw(store, decision.get("result_ref"), binding, target)
        _verified(store, decision.get("result_ref"), binding, expected, verify_host)
        record["result_ref"] = decision["result_ref"]
    record["lifecycle"] = outcomes[decision["outcome"]]


def _apply_revive(record, decision, prior):
    _require(record["lifecycle"] in ("shelved", "abandoned", "refuted", "realized"), "revival requires an ended direction obligation")
    kind = decision["trigger_kind"]
    _require(kind in ("changed_evidence", "scoped_replication"), "revival needs changed evidence or scoped replication")
    _require(bool(decision["evidence_refs"]), "revival requires evidence for its stated scope")
    if kind == "changed_evidence":
        _require(any(ref not in prior["decision"]["evidence_refs"] for ref in decision["evidence_refs"]), "changed evidence cannot be the unchanged prior references")
    record["lifecycle"] = "revived"


def checkpoint_maturation(store, direction_id, *, decision, expected_revision,
                          accounting=None, observation_ref=None, verify_host=None):
    """Save one agent decision; the direction's tip and anchor stay unchanged.

    accounting is the host's current account_opportunities() result, never an
    agent attachment. verify_host(ref, raw_bytes, expected) must confirm who
    produced an observation/result record and that it matches ``expected``;
    without it, observations and results are refused.
    """
    state = store.get(direction_id)
    _require(state["revision"] == expected_revision, "stale direction revision")
    prior = read_maturation(store, direction_id)
    decision = validate_maturation_decision(decision)
    action = decision["action"]
    _require(action == "observe" or observation_ref is None, "observation receipts require an observation event")
    binding = _binding(state)
    _refs(store, decision["evidence_refs"], binding)
    if prior is None:
        _require(action == "nominate", "nomination required before protection")
        record = dict(proposition=decision["proposition"], scope=decision["scope"],
                      lifecycle="nominated", last_faithful=None, grant=None, observation=None,
                      observation_ref=None, result_ref=None, used_opportunities=[],
                      comparison_anchor=_loads(_json(state["anchor"])))
        _comparison_anchor(store, record["comparison_anchor"], binding)
    else:
        _require(action != "nominate", "renomination cannot reset an existing lifecycle")
        record = _loads(_json(prior))
    record.update(schema_version=1, kind="maturation", direction_id=direction_id,
                  binding=binding, candidate_id=state["tip"], native_eligible=False,
                  previous_ref=state["note"].get("maturation_ref"), decision=decision)
    retracting = "retract_faithful_ref" in decision
    if retracting:
        _require(prior is not None and prior["last_faithful"] is not None
                 and decision["retract_faithful_ref"] == state["note"].get("maturation_ref")
                 and decision.get("last_faithful", False) is None,
                 "fidelity retraction must name the exact current witness record")
        _require(bool(decision["evidence_refs"]), "fidelity retraction requires correction evidence")
    if "last_faithful" in decision:
        witness = decision["last_faithful"]
        _require(witness is not None or record["last_faithful"] is None or retracting,
                 "a fallback cannot erase the last faithful source without an explicit correction")
        _faithful(store, witness, binding)
        if witness is not None and prior is not None:
            _new_witness_evidence(store, witness, prior, binding)
        record["last_faithful"] = witness
    if action == "grant":
        _apply_grant(record, decision, state, direction_id, accounting)
    elif action == "observe":
        _apply_observe(store, record, direction_id, binding, accounting, observation_ref, verify_host)
    elif action == "review":
        _apply_review(store, record, decision, state, direction_id, binding, accounting, retracting, verify_host)
    elif action == "revive":
        _apply_revive(record, decision, prior)
    data = _json(record)
    ref = {"evidence_id": store.archive_evidence(state["tip"], {"maturation.json": data}), "artifact": "maturation.json"}
    note = dict(state["note"], maturation_ref=ref,
                claim="%s: %s. %s" % (record["lifecycle"], record["proposition"], decision["reason"]),
                scope=record["scope"], next_test=decision["next_test"])
    committed = store.checkpoint(direction_id, state["tip"], note=note, expected_revision=expected_revision,
                                 status=state["status"], resources=state["resources"], evidence=list(state["evidence"]) + [ref["evidence_id"]])
    return {"state": committed, "ref": ref, "record": record}


def validate_protection(store, ref, *, direction_id, candidate_id, opportunity_id, accounting):
    """Check that a grant is still active for this candidate and opportunity.

    Returns the ledger row plus the witness details; never native eligibility.
    """
    state = store.get(direction_id)
    _require(state["note"].get("maturation_ref") == ref, "protection reference is stale or uncommitted")
    record = read_maturation(store, direction_id)
    _require(record is not None and record["lifecycle"] == "active", "protection requires an active grant")
    _require(record["grant"]["opportunity_id"] == opportunity_id, "protection opportunity mismatch")
    _candidate(store, candidate_id, record["binding"])
    _require(record["grant"]["start_candidate_id"] == candidate_id, "protected starting workspace cannot be replaced")
    seed = record["grant"].get("faithful_seed", record["last_faithful"])
    _require(record["last_faithful"] is not None
             and not any(target.get("grant") is not None
                         and target["grant"]["opportunity_id"] == opportunity_id
                         and seed is not None
                         and seed["candidate_id"] == target["last_faithful"]["candidate_id"]
                         for target in _retraction_targets(store, record, record["binding"])),
             "protected seed fidelity has been withdrawn; retain settlement and review obligations")
    _faithful(store, seed, record["binding"])
    _require(seed is not None and ("faithful_seed" in record["grant"] or seed["candidate_id"] == candidate_id),
             "legacy grant requires its exact faithful seed witness before dispatch")
    row = _ledger(record, accounting)[opportunity_id]
    _require(not row["settled"] and not row["expired"] and any(v > 0 for v in row["remaining_reservation"].values()),
             "protected grant is expired, settled or exhausted")
    return dict(row, protected_candidate_id=candidate_id, last_faithful=record["last_faithful"],
                faithful_seed=seed,
                faithful_seed_source_id=_candidate(store, seed["candidate_id"], record["binding"])["source_id"],
                comparison_anchor=record["comparison_anchor"], native_eligible=False, contract_checked=True)


# ---------------------------------------------------------------------------
# Planner view (was bridge/native_maturation.py)

COMMON_SCOPE = """The brief's native_maturation_opportunity displays existing named faithful
witnesses and the same finite native_engineer_offer used for ordinary work.
These are engineer designations, not certified fidelity, correctness or speed.
When a slower faithful source has an actionable unresolved question, consider
whether another bounded investigation is worthwhile. A changed launch signature
or a profile is not required. Verify any claimed speed comparison against exact
source/case/method evidence; missing evidence remains unknown. You choose the
source, question and acceptance, deferral or rejection in the existing reason,
next_test and review fields. Do not add a new proposal field for this display.
The enabled flag states whether the existing protection actions are available.
Ordinary continuation and access to older faithful/failed work remain available
in both arms. Protection consumes the ordinary finite offer inside the same
campaign ceiling; it is not a bonus allocation or an automatic follow-up.

"""

_VIEW_SCOPE = (
    'Current named witnesses only; full earlier roles and corrections remain in the existing source/evidence index. '
    'A witness row is navigation, not scientific eligibility or a dispatched grant. '
    'Archived designations may coexist with a withdrawn current maturation witness; retain and inspect the correction. '
    'After nomination/review, admission uses the then-current finite offer and existing grant validators.')

_LEASE_SCOPE = (
    'Existing protection may use this same once-charged E2 for its bound reviewed direction. '
    'Admission additionally requires resources, time and two decisions for the ordinary '
    'protection observation and review. The existing final research review stays held. '
    'The current nomination, source and protection validators still decide eligibility.')


def _named_witnesses(context, candidates):
    """One row per faithful witness named by an archive card or a maturation record."""
    maturation = context.get('maturation', {})
    named = []
    for direction_id, direction in sorted(context['archive']['directions'].items()):
        if direction_id == context['campaign_id']:
            continue
        record = maturation.get(direction_id)
        witnesses = [('archive_card', (direction.get('card') or {}).get('last_faithful'))]
        if record is not None:
            witnesses.append(('maturation_record', record.get('last_faithful')))
        for designation, witness in witnesses:
            if witness is None:
                continue
            candidate = candidates.get(witness['candidate_id'])
            named.append({'direction_id': direction_id, 'direction_revision': direction['revision'],
                'current_tip_candidate_id': direction['tip'], 'designation_record': designation,
                'witness': copy.deepcopy(witness),
                'source_id': None if candidate is None else candidate['source_id'],
                'source_catalog_match': candidate is not None,
                'maturation_lifecycle': None if record is None else record['lifecycle'],
                'maturation_current_witness_candidate_id': None if record is None or record.get('last_faithful') is None
                    else record['last_faithful']['candidate_id'],
                'faithfulness_verified': False, 'slower_than_comparator': None,
                'slowness_status': 'Inspect matching source-bound native evidence; no comparison inferred.',
                'grant_eligibility_asserted': False})
    return named


def opportunity_view(context, engineer_offer, parent_candidates, *, enabled):
    """Planner-facing list of named faithful witnesses plus the current offer.

    Pure display: no store writes, scoring or grants. The caller has already
    frozen ``context`` and checked the parent identities. A witness whose
    candidate is not in ``parent_candidates`` is shown as unknown and never
    authorizes work; nomination and grant checks above still decide.
    """
    # This view raised a plain ValueError before the port (not MaturationError).
    require(type(enabled) is bool, 'explicit protection treatment required', ValueError)
    candidates = {row['candidate_id']: row for row in parent_candidates}
    named = _named_witnesses(context, candidates)
    result = {'schema': 'native-maturation-opportunity-v1', 'protected_work_enabled': enabled,
        'ordinary_offer_available': engineer_offer['available'],
        'reservation': copy.deepcopy(engineer_offer['reservation']),
        'offer_operation_id': engineer_offer['operation_id'],
        'named_faithful_witnesses': named,
        'required_question': 'Agent chooses an actionable unresolved question and exact evidence.',
        'signature_difference_required': False, 'profiling_required_for_offer': False,
        'automatic_grant': False, 'new_resource_credit': False,
        'pending_maturation_reviews': copy.deepcopy(context.get('owed_maturation_reviews', [])),
        'scope': _VIEW_SCOPE}
    held = engineer_offer.get('program_lease_offer')
    if held is not None:
        from ..budget import active_lease
        lease = active_lease(context['state'])
        require(lease is not None and held['commitment_id'] == lease['commitment_id']
                and held['direction_id'] == lease['direction_id'],
                'held protection option differs from the current commitment', ValueError)
        result['program_lease_protected_option'] = {
            'commitment_id': held['commitment_id'], 'direction_id': held['direction_id'],
            'engineer_offer_available': held['available'],
            'engineer_reservation': copy.deepcopy(held['reservation']),
            'additional_planner_reservation': {key: 2 * value for key, value in lease['planner_reservation'].items()},
            'additional_planner_window_seconds': 2 * lease['planner_seconds'],
            'additional_planner_decisions': 2, 'grant_eligibility_asserted': False,
            'scope': _LEASE_SCOPE}
    return result
