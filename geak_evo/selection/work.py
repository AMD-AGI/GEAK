"""Cost accounting and optional advice on which line of work to continue.

``account_opportunities`` checks the complete list of work opportunities
(reservations, cost receipts, settlement) against the campaign limits and
derives what was delivered and what is still reserved.

``advise`` scores the offered continuations with a PUCT-like rule: recent
observed gain per unit cost plus an exploration bonus. The result is advice
only; it never admits work or chooses a parent.

``record_decision`` records the agent's actual choice next to that advice.

Inputs must come from a host-controlled complete ledger. A host verifier
authenticates progress claims, not these Python mappings.
"""

import json
import math
from pathlib import PurePosixPath

from .._util import EvoError, canonical_json, is_sha256, require, sha256_hex


OBSERVED_GAIN_KIND = "native_local_frontier_observed_log_v1"
OBSERVED_OUTCOME_KINDS = frozenset({"observed_comparison", "baseline_initialization", "unknown"})
OBSERVED_GAIN_SCOPE = (
    "Descriptive native development gain relative to the minimum comparable local "
    "source value frozen before the producing opportunity; not a confidence bound, "
    "endpoint result or causal ancestry credit."
)
_BINDING_PARAMETERS = {"weights", "c_ref", "half_life", "epsilon", "beta"}


class SelectionError(EvoError, ValueError):
    """An allocation record is inconsistent or cannot support its claim."""


def _need(condition, message):
    require(condition, message, SelectionError)


# The three helpers below are also used by geak_evo.budget.
def _number(value, name, positive=False):
    _need(not isinstance(value, bool) and isinstance(value, (int, float)), name + " must be a finite number")
    try:
        value = float(value)
    except OverflowError:
        raise SelectionError(name + " must be finite") from None
    _need(math.isfinite(value) and not (value <= 0 if positive else value < 0),
          name + " must be finite and nonnegative/positive")
    return value


def _vector(value, dimensions, unknown=False):
    """One number per resource dimension; ``unknown`` allows missing/None entries."""
    _need(isinstance(value, dict) and not set(value) - set(dimensions), "unexpected resource dimensions")
    _need(unknown or set(value) == set(dimensions), "missing resource dimension")
    return {k: None if unknown and value.get(k) is None else _number(value[k], k)
            for k in dimensions}


def _sum(values):
    """Exact float sum; None if any value is unknown."""
    if any(v is None for v in values):
        return None
    try:
        return _number(math.fsum(values), "sum")
    except OverflowError:
        raise SelectionError("sum must be finite") from None


def _text(value, name):
    _need(isinstance(value, str) and value.strip(), name + " must be a nonempty string")
    return value


def _digest(value):
    _need(is_sha256(value), "expected lowercase SHA256")
    return value


def _ref(value):
    _need(isinstance(value, dict) and set(value) == {"evidence_id", "artifact"},
          "expected evidence_id/artifact reference")
    name = _text(value["artifact"], "artifact")
    path = PurePosixPath(name)
    _need(not (path.is_absolute() or ".." in path.parts or str(path) != name
               or "\\" in name or "\0" in name or name == "."),
          "artifact must be a canonical relative file name")
    return {"evidence_id": _digest(value["evidence_id"]), "artifact": name}


def _hash(value):
    return sha256_hex(canonical_json(value))


def _binding(value, expected=None):
    result = {"job_id": _digest(value.get("job_id")),
              "measurement_epoch_id": _text(value.get("measurement_epoch_id"), "measurement epoch")}
    _need(expected is None or result == expected, "mixed job or measurement epoch")
    return result


def _offer_binding(value, dimensions):
    """Check a frozen offer's structure; the host verifies its content separately."""
    fields = {"snapshot_revision", "direction_revision", "parent_candidate_id",
              "source_id", "membership_ref", "clock_ref", "reservation"}
    _need(isinstance(value, dict) and set(value) == fields, "exact observed offer_binding required")
    for key in ("snapshot_revision", "direction_revision"):
        _need(type(value[key]) is int and value[key] >= 1,
              "positive offered snapshot/direction revision required")
    for key in ("parent_candidate_id", "source_id"):
        _digest(value[key])
    if value["membership_ref"] is not None:
        _ref(value["membership_ref"])
    clock = value["clock_ref"]
    _need(isinstance(clock, dict) and set(clock) == {"path", "sha256"}, "exact offered clock reference required")
    path = PurePosixPath(_text(clock["path"], "clock path"))
    _need(path.is_absolute() and str(path) == clock["path"] and ".." not in path.parts
          and "\\" not in clock["path"] and "\0" not in clock["path"],
          "canonical absolute offered clock path required")
    _digest(clock["sha256"])
    _vector(value["reservation"], dimensions)
    return json.loads(json.dumps(value, allow_nan=False))


# ---------------------------------------------------------------- accounting

def _account_row(item, binding, limits, deadline, now, ids, receipts, sequences):
    """Check one opportunity record and derive its delivered/remaining cost."""
    _binding(item, binding)
    row = {k: _text(item.get(k), k) for k in
           ("opportunity_id", "direction_id", "island_id", "engineer_id", "question")}
    _need(row["opportunity_id"] not in ids, "duplicate opportunity")
    ids.add(row["opportunity_id"])
    start, end = _number(item.get("opened_at"), "opened_at"), _number(item.get("review_at"), "review_at")
    _need(start < end <= deadline and start <= now, "opportunity must have a finite campaign-bounded review")
    settled, complete = item.get("settled"), item.get("progress_complete")
    _need(type(settled) is bool and type(complete) is bool,
          "settled and progress_complete must be explicit booleans")
    sequence = item.get("completed_sequence")
    if settled:
        _need(type(sequence) is int and sequence >= 0 and sequence not in sequences,
              "settlement requires unique nonnegative completed_sequence")
        sequences.add(sequence)
    else:
        _need(sequence is None, "unsettled opportunity cannot have completed_sequence")
    requested = _vector(item.get("requested"), limits)
    reserved = _vector(item.get("reserved"), limits)
    _need(isinstance(item.get("receipts"), list), "explicit cost receipts required")
    costs = []
    for receipt in item["receipts"]:
        receipt_id = _text(receipt.get("receipt_id"), "receipt_id")
        _need(receipt_id not in receipts, "duplicate cost receipt")
        receipts.add(receipt_id)
        costs.append(_vector(receipt.get("costs"), limits, True))
    # A settled opportunity without any receipt has unknown (not zero) cost.
    delivered = {k: None if settled and not costs else _sum([v[k] for v in costs]) for k in limits}
    known = {k: _sum([v[k] for v in costs if v[k] is not None]) for k in limits}
    remaining, committed = {}, {}
    for k in limits:
        _need(reserved[k] <= requested[k] and reserved[k] <= limits[k], "reservation exceeds request or budget")
        _need(known[k] <= reserved[k], "delivery exceeds reservation; retain overrun evidence")
        # Unknown cost keeps its reservation even after settlement.
        remaining[k] = 0.0 if settled and delivered[k] is not None else reserved[k] - known[k]
        committed[k] = known[k] + remaining[k]
    row.update(binding, opened_at=start, review_at=end, expired=now >= end,
               requested=requested, reserved=reserved, delivered=delivered, known_delivered=known,
               remaining_reservation=remaining, committed=committed, settled=settled,
               completed_sequence=sequence, progress_complete=complete)
    if "progress_gain_kind" in item:
        row["progress_gain_kind"] = _text(item["progress_gain_kind"], "progress gain kind")
    if "progress_outcome_kind" in item:
        row["progress_outcome_kind"] = _text(item["progress_outcome_kind"], "progress outcome kind")
    return row


def _commitment_holds(commitments, limits):
    message = "commitments must be an iterable of reservation records"
    _need(not isinstance(commitments, (str, bytes, dict)), message)
    try:
        items = iter(commitments)
    except TypeError:
        raise SelectionError(message) from None
    holds, seen = [], set()
    for item in items:
        _need(isinstance(item, dict) and set(item) == {"commitment_id", "reserved"},
              "exact commitment_id/reserved commitment required")
        key = _text(item["commitment_id"], "commitment_id")
        _need(key not in seen, "duplicate commitment")
        seen.add(key)
        holds.append({"commitment_id": key, "reserved": _vector(item["reserved"], limits)})
    return holds


def account_opportunities(records, *, binding, limits, deadline, now, commitments=()):
    """Check a full host ledger; derive delivery and unconsumed reservations.

    Expiry does not release an unsettled reservation. Unknown actual cost keeps
    its reservation even after settlement. Commitments reserve unstarted work
    and never expire here; the host reduces or removes a hold when it opens the
    ordinary opportunity or explicitly releases the unstarted work. This checks
    record arithmetic only, not meters, cleanup, lifecycle or authorization.
    """
    binding = _binding(binding)
    _need(isinstance(limits, dict) and limits, "explicit nonempty resource limits required")
    # A zero limit is allowed (e.g. a tool-free planner has zero evaluations);
    # it permits zero demand only.
    limits = {k: _number(v, k) for k, v in limits.items()}
    deadline, now = _number(deadline, "deadline", True), _number(now, "now")
    ids, receipts, sequences = set(), set(), set()
    rows = [_account_row(item, binding, limits, deadline, now, ids, receipts, sequences)
            for item in records]
    holds = _commitment_holds(commitments, limits)
    committed = {k: _sum([r["committed"][k] for r in rows] + [h["reserved"][k] for h in holds])
                 for k in limits}
    _need(all(committed[k] <= limits[k] for k in limits), "aggregate commitments exceed budget")
    result = {"schema_version": 1, "binding": binding, "limits": limits, "deadline": deadline, "now": now,
              "opportunities": rows, "delivered": {k: _sum([r["delivered"][k] for r in rows]) for k in limits},
              "known_delivered": {k: _sum([r["known_delivered"][k] for r in rows]) for k in limits},
              "remaining_reservation": {k: _sum([r["remaining_reservation"][k] for r in rows]
                                                + [h["reserved"][k] for h in holds]) for k in limits},
              "available": {k: limits[k] - committed[k] for k in limits}}
    if holds:
        result.update(commitments=holds,
                      unstarted_reservation={k: _sum([h["reserved"][k] for h in holds]) for k in limits})
    return result


# -------------------------------------------------------------------- advice

def _advice_config(config, limits):
    observed = "gain_kind" in config
    _need(set(config) == _BINDING_PARAMETERS | ({"gain_kind"} if observed else set()),
          "all advice parameters must be explicit")
    _need(not observed or config["gain_kind"] == OBSERVED_GAIN_KIND, "unsupported explicit gain_kind")
    weights = _vector(config["weights"], limits)
    _need(all(v > 0 for v in weights.values())
          and math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12),
          "positive resource weights must sum to one")
    c_ref, half = _number(config["c_ref"], "c_ref", True), _number(config["half_life"], "half_life", True)
    epsilon, beta = _number(config["epsilon"], "epsilon", True), _number(config["beta"], "beta")
    _need(epsilon <= 1, "epsilon must not exceed one")
    return observed, weights, c_ref, half, epsilon, beta


def _advice_episodes(accounting, binding, observed):
    episodes = {}
    for row in accounting["opportunities"]:
        _binding(row, binding)
        _need(observed or row.get("progress_gain_kind") != OBSERVED_GAIN_KIND,
              "observed progress coverage cannot authorize strict LCB advice")
        if "progress_outcome_kind" in row:
            outcome = row["progress_outcome_kind"]
            _need(observed, "observed outcome coverage cannot authorize strict LCB advice")
            _need(row.get("progress_gain_kind") == OBSERVED_GAIN_KIND
                  and outcome in OBSERVED_OUTCOME_KINDS
                  and row["progress_complete"] is (outcome != "unknown"),
                  "inconsistent typed observed outcome coverage")
        _need(row["opportunity_id"] not in episodes, "duplicate opportunity")
        episodes[row["opportunity_id"]] = row
    return episodes


def _verified_gains(progress, episodes, binding, observed, verify_progress):
    """Sum host-verified gain credits per opportunity; each source counts once."""
    gains, credit_ids, sources, evidence_ids = {}, set(), set(), set()
    for item in progress:
        _binding(item, binding)
        key = _text(item.get("opportunity_id"), "opportunity_id")
        _need(key in episodes and episodes[key]["settled"], "credit must identify a settled producing opportunity")
        credit = _text(item.get("credit_id"), "credit_id")
        if observed:
            _need(episodes[key].get("progress_outcome_kind") != "baseline_initialization",
                  "baseline initialization cannot receive gain credit")
            _need(item.get("gain_kind", OBSERVED_GAIN_KIND) == OBSERVED_GAIN_KIND
                  and not {"frontier_before_source_id", "gain_log_lcb"} & set(item),
                  "mixed or mismatched progress gain contract")
            baseline_field, gain_field = "baseline_source_id", "gain_log_observed"
        else:
            _need(not {"gain_kind", "baseline_source_id", "gain_log_observed"} & set(item),
                  "observed progress cannot be used as strict LCB credit")
            baseline_field, gain_field = "frontier_before_source_id", "gain_log_lcb"
        source, before = _digest(item.get("source_id")), _digest(item.get(baseline_field))
        ref = _ref(item.get("evidence_ref"))
        evidence_key = (ref["evidence_id"], ref["artifact"])
        _need(not (credit in credit_ids or source in sources or evidence_key in evidence_ids or source == before),
              "duplicate credit, source or evidence; DAG gain counts once")
        gain = _number(item.get(gain_field), gain_field)
        expected = dict(binding, opportunity_id=key, credit_id=credit, source_id=source,
                        **{baseline_field: before, gain_field: gain})
        if observed:
            expected["gain_kind"] = OBSERVED_GAIN_KIND
        _need(verify_progress is not None and verify_progress(ref, expected) is True,
              "progress requires host verification")
        credit_ids.add(credit)
        sources.add(source)
        evidence_ids.add(evidence_key)
        gains[key] = _sum([gains.get(key, 0.0), gain])
    return gains


def _direction_history(episodes, gains, scalar, total_cost, half, observed):
    """Walk settled episodes in completion order and build per-direction stats.

    Recent gain and cost are discounted by cost spent since (half-life in
    normalized cost). A direction with unknown cost or incomplete progress
    has no rate. Baseline initialization advances the cost clock but never
    enters the gain/cost rate.
    """
    stats = {"recent": {}, "unknown": set(), "initialization_counts": {}, "comparison_counts": {},
             "initialization_costs": {}, "recent_initialization_costs": {}, "rate_episode_counts": {}}
    elapsed = 0.0
    settled = sorted((r for r in episodes.values() if r["settled"]), key=lambda r: r["completed_sequence"])
    for episode in settled:
        direction, key = episode["direction_id"], episode["opportunity_id"]
        cost = scalar(episode["delivered"])
        complete = episode["progress_complete"] and (
            not observed or episode.get("progress_gain_kind") == OBSERVED_GAIN_KIND)
        initialized = observed and complete and episode.get("progress_outcome_kind") == "baseline_initialization"
        if observed:
            if initialized:
                costs = stats["initialization_costs"]
                costs[direction] = _sum([costs.get(direction, 0.0), cost])
            else:
                counts = stats["rate_episode_counts"]
                counts[direction] = counts.get(direction, 0) + 1
        if observed and complete:
            counts = stats["initialization_counts" if initialized else "comparison_counts"]
            counts[direction] = counts.get(direction, 0) + 1
        if cost is None or not complete:
            stats["unknown"].add(direction)
        if cost is not None:
            elapsed = _sum([elapsed, cost])
        if total_cost is not None and cost is not None:
            age = 2 ** (-max(0.0, total_cost - elapsed) / half)
            if initialized:
                recent = stats["recent_initialization_costs"]
                recent[direction] = _sum([recent.get(direction, 0.0), age * cost])
                continue
            pair = stats["recent"].setdefault(direction, [0.0, 0.0])
            pair[0] = _sum([pair[0], age * gains.get(key, 0.0)])
            pair[1] = _sum([pair[1], age * cost])
    return stats


def advise(offers, accounting, progress, *, config, verify_progress=None):
    """Return unsorted, decomposed advice, never an executed choice.

    ``accounting`` must be the host's ``account_opportunities`` result. For
    every credit, ``verify_progress(evidence_ref, expected)`` must return
    literal True after checking the evidence for that exact binding; without
    a callback no credit can be claimed.

    Default (strict) mode credits ``gain_log_lcb`` against the frontier before.
    With ``config["gain_kind"] == OBSERVED_GAIN_KIND`` it instead credits the
    descriptive nonnegative observed log ratio ``gain_log_observed`` against the
    minimum comparable local source frozen before the producing opportunity;
    this is not a confidence bound. Complete zero/slower outcomes may have no
    credit; unknown coverage must stay progress_complete=False. Baseline
    initialization is complete but has no comparator and no credit: its cost
    counts in service and recency, not in the gain/cost rate.
    """
    binding, limits = _binding(accounting["binding"]), accounting["limits"]
    _need(not any(_number(value, key) == 0 for key, value in limits.items()),
          "normalized allocation advice requires positive resource capacities")
    offers = list(offers)
    observed, weights, c_ref, half, epsilon, beta = _advice_config(config, limits)

    def scalar(vector):
        if any(vector[k] is None for k in limits):
            return None
        return _sum([weights[k] * vector[k] / limits[k] for k in limits])

    episodes = _advice_episodes(accounting, binding, observed)
    gains = _verified_gains(progress, episodes, binding, observed, verify_progress)
    total_cost = scalar(accounting["delivered"])
    service = {}
    for episode in episodes.values():
        direction = episode["direction_id"]
        service[direction] = _sum([service.get(direction, 0.0), scalar(episode["committed"]) / c_ref])
    count = _sum(list(service.values()))
    stats = _direction_history(episodes, gains, scalar, total_cost, half, observed)
    recent, unknown = stats["recent"], stats["unknown"]

    priors, seen, offer_bindings = [], set(), {}
    for offer in offers:
        _binding(offer, binding)
        direction = _text(offer.get("direction_id"), "direction_id")
        _need(direction not in seen, "one compared offer per direction")
        seen.add(direction)
        if observed:
            offer_bindings[direction] = _offer_binding(offer.get("offer_binding"), limits)
        priors.append(_number(offer.get("priority_weight"),
                              "uniform policy prior weight" if observed else "agent priority weight"))
    _need(not (observed and priors and any(value != priors[0] for value in priors)),
          "observed development advice requires uniform policy prior weights")
    prior_sum = _sum(priors)

    output = []
    for offer, raw in zip(offers, priors):
        direction = offer["direction_id"]
        estimate = scalar(_vector(offer.get("estimated_costs"), limits, True))
        prior = (1 - epsilon) * (raw / prior_sum if prior_sum else 1 / len(offers)) + epsilon / len(offers)
        observed_gain, observed_cost = recent.get(direction, [0.0, 0.0])
        rate = None if total_cost is None or direction in unknown else observed_gain / max(c_ref, observed_cost)
        n = service.get(direction, 0.0)
        bonus = beta * prior * math.sqrt(1 + count) / (1 + n)
        score = None if rate is None or estimate is None else _number(rate + bonus / max(c_ref, estimate),
                                                                      "derived score")
        reasons = ((["group cost unknown"] if total_cost is None else [])
                   + (["direction cost/progress unknown"] if direction in unknown else [])
                   + (["next cost unknown"] if estimate is None else []))
        row = {"direction_id": direction, "island_id": _text(offer.get("island_id"), "island_id"),
               "engineer_id": _text(offer.get("engineer_id"), "engineer_id"), "priority_weight": raw,
               "prior_mass": prior, "recent_gain": None if rate is None else observed_gain,
               "recent_cost": None if rate is None else observed_cost, "unavailable_reasons": reasons,
               "rate": rate, "service_count": n, "exploration_gain": bonus,
               "estimated_cost": estimate, "score": score,
               "status": "unavailable" if score is None else ("observed" if direction in recent else "untried")}
        if observed:
            initializations = stats["initialization_counts"].get(direction, 0)
            comparisons = stats["comparison_counts"].get(direction, 0)
            row["offer_binding"] = offer_bindings[direction]
            row["baseline_initialization_count"] = initializations
            row["baseline_initialization_cost"] = stats["initialization_costs"].get(direction, 0.0)
            row["recent_initialization_cost"] = (None if total_cost is None else
                                                 stats["recent_initialization_costs"].get(direction, 0.0))
            row["observed_comparison_count"] = comparisons
            row["rate_episode_count"] = stats["rate_episode_counts"].get(direction, 0)
            if score is not None and initializations and not comparisons:
                row["status"] = "baseline_initialization"
        output.append(row)
    result = {"schema_version": 1, "advisory_only": True, "binding": binding,
              "config": json.loads(json.dumps(config)), "config_id": _hash(config),
              "total_service_count": count, "rows": output}
    if observed:
        metric = {"gain_kind": OBSERVED_GAIN_KIND, "uncertainty": None,
                  "scope": OBSERVED_GAIN_SCOPE, "prior_origin": "uniform_policy"}
        result.update(metric)
        for row in output:
            row.update(metric)
    return result


# ------------------------------------------------------------------ decision

def record_decision(advice, selection, *, reason, override_reason=None, evidence_refs=()):
    """Record an agent choice separately from the advice and from whether the host accepts it."""
    _need(advice.get("advisory_only") is True, "expected advisory result")
    chosen = {k: _text(selection.get(k), k) for k in ("direction_id", "island_id", "engineer_id")}
    chosen["working_source_id"] = _digest(selection.get("working_source_id"))
    donors = selection.get("donor_source_ids")
    _need(isinstance(donors, list), "donor_source_ids must be a unique list")
    chosen["donor_source_ids"] = [_digest(v) for v in donors]
    _need(len(set(donors)) == len(donors), "donor_source_ids must be a unique list")
    observed = advice.get("gain_kind") == OBSERVED_GAIN_KIND
    scored = [r for r in advice["rows"] if r["score"] is not None]
    fields = ("direction_id", "island_id") if observed else ("direction_id", "island_id", "engineer_id")

    def matches(row):
        if not all(row[k] == chosen[k] for k in fields):
            return False
        # Observed advice binds an exact source; donors make it a different offer.
        return not observed or (not chosen["donor_source_ids"]
                                and row["offer_binding"]["source_id"] == chosen["working_source_id"])

    matching = next((r for r in scored if matches(r)), None)
    if matching is None:
        comparison = "not_comparable"
    else:
        comparison = "override" if matching["score"] < max(r["score"] for r in scored) else "top_or_tied"
    if comparison == "override":
        _text(override_reason, "ranked override reason")
    if override_reason is not None:
        _text(override_reason, "override_reason")
    return {"schema_version": 1, "authority": "agent_choice_only", "binding": _binding(advice["binding"]),
            "advice_id": _hash(advice), "selection": chosen, "reason": _text(reason, "reason"),
            "comparison": comparison, "override_reason": override_reason,
            "evidence_refs": [_ref(v) for v in evidence_refs]}
