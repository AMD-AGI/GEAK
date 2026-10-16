"""Budget arithmetic for the evolution loop: program leases and planner reserves.

A program lease lets the planner commit to two engineer episodes on one idea
and holds the resources for the episodes, their reviews and the decision in
between. The functions here are pure: they edit the state dict they are given
and never save it, call a planner or read a clock. The caller works on a copy,
saves it together with the ordinary opportunity or action, and throws the copy
away if a check fails. Time passing never releases a hold by itself.

The planner-reserve helpers size one engineer offer so that a later ordinary
planner decision still fits inside the campaign deadline and allowance.
"""

import copy
import math

from ._util import EvoError, require
from .selection.work import SelectionError, _number, _sum, _vector, account_opportunities


NS = 1_000_000_000


class LeaseError(EvoError, ValueError):
    """A program reservation cannot support the proposed transition."""


def _need(condition, message):
    require(condition, message, LeaseError)


def _text(value, name):
    _need(type(value) is str and bool(value.strip()), name + " must be nonempty text")
    return value


def _counter(value):
    _need(type(value) is int and value >= 0, "invalid remaining lease counter")
    return value


# --- program lease -----------------------------------------------------------

def _remaining(lease):
    engineers, planners = _counter(lease["engineers_remaining"]), _counter(lease["planner_tranches"])
    return {key: _sum([engineers * value, planners * lease["planner_reservation"][key]])
            for key, value in lease["engineer_reservation"].items()}


def _sync(lease):
    lease["remaining_reservation"] = _remaining(lease)


def _leases(state):
    leases = state.get("program_leases", [])
    _need(type(leases) is list, "program_leases must be a retained list")
    ids = set()
    for lease in leases:
        _need(type(lease) is dict and lease.get("schema") == 1, "invalid program lease schema")
        key = _text(lease.get("commitment_id"), "commitment_id")
        _need(key not in ids, "duplicate program lease")
        ids.add(key)
        _need(lease.get("status") in ("active", "releasing", "completed", "released", "blocked"),
              "invalid program lease status")
        _vector(lease["engineer_reservation"], state["limits"])
        _vector(lease["planner_reservation"], state["limits"])
        _need(_vector(lease["remaining_reservation"], state["limits"]) == _remaining(lease),
              "lease hold differs from remaining tranches")
        if lease["status"] in ("completed", "released"):
            _need(not any(lease["remaining_reservation"].values()), "terminal lease retains unexplained reservation")
    return leases


def active_lease(state):
    """Return the outstanding active, releasing or blocked lease, without edits."""
    current = [lease for lease in _leases(state) if lease["status"] in ("active", "releasing", "blocked")]
    _need(len(current) <= 1, "only one outstanding program lease is permitted")
    return current[0] if current else None


def commitments(state):
    """The unstarted holds of the outstanding lease, in the accounting format."""
    lease = active_lease(state)
    return ([] if lease is None else [{"commitment_id": lease["commitment_id"],
                                      "reserved": dict(lease["remaining_reservation"])}])


def held_calendar(lease):
    return _sum([lease["engineers_remaining"] * lease["engineer_seconds"],
                 lease["planner_tranches"] * lease["planner_seconds"]])


def check_remaining(state, now):
    """Block a lease whose remaining work no longer fits; never release its hold."""
    lease = active_lease(state)
    if lease is None:
        return False
    if lease["status"] == "blocked":
        return True
    now = _number(now, "now")
    planners = [row for row in state["opportunities"] if row["engineer_id"] == "planner"]
    slots = lease["planner_total_limit"] - len(planners)
    resolved = [draw for draw in lease["planner_draws"] if draw["resolved"]]
    if resolved:
        latest = resolved[-1]
        positions = [index for index, row in enumerate(planners) if row["opportunity_id"] == latest["operation_id"]]
        _need(len(positions) == 1, "resolved planner draw requires its ordinary row")
        slots = min(slots, latest["steps_after_current"] - (len(planners) - positions[0] - 1))
    if now + held_calendar(lease) >= state["deadline"] or slots < lease["planner_tranches"]:
        lease.update(status="blocked", reason="Remaining committed work no longer fits the original calendar or planner ceiling.")
        return True
    return False


def _account(state, now, extra=()):
    try:
        return account_opportunities(list(state["opportunities"]) + list(extra), binding=state["binding"],
                                     limits=state["limits"], deadline=state["deadline"], now=now,
                                     commitments=commitments(state))
    except SelectionError as exc:
        raise LeaseError(str(exc)) from exc


def _capacity(state, capacity, *, current_in_ledger):
    _need(type(capacity) is dict and set(capacity) == {"total_limit", "steps_after_current"}
          and type(capacity["total_limit"]) is int and capacity["total_limit"] > 0
          and type(capacity["steps_after_current"]) is int and capacity["steps_after_current"] >= 0,
          "frozen ordinary planner capacity required")
    consumed = sum(row["engineer_id"] == "planner" for row in state["opportunities"])
    remaining = capacity["total_limit"] - consumed - (not current_in_ledger)
    _need(remaining >= 0, "planner exceeds its original decision ceiling")
    return min(remaining, capacity["steps_after_current"])


def _planner(state, operation_id):
    rows = [row for row in state["opportunities"] if row["opportunity_id"] == operation_id]
    _need(len(rows) == 1 and rows[0]["engineer_id"] == "planner", "exact ordinary planner row required")
    return rows[0]


def _window(opportunity):
    observed = _number(opportunity["review_at"] - opportunity["opened_at"], "planner window", True)
    nominal = _number(opportunity.get("program_lease_planning_seconds", observed), "declared planner window", True)
    _need(observed <= nominal, "actual planner window exceeds its declared window")
    return nominal


def _draw(lease, operation_id):
    draws = [draw for draw in lease["planner_draws"] if draw["operation_id"] == operation_id]
    _need(len(draws) == 1 and draws[0]["resolved"] is False, "one unresolved current planner draw required")
    return draws[0]


def _resolve(lease, draw, *, obligation):
    if obligation and draw["funding"] == "surplus":
        _need(lease["planner_tranches"] > 0, "no held planner tranche for this obligation")
        lease["planner_tranches"] -= 1
    draw["resolved"] = True
    _sync(lease)


def _complete(lease):
    if lease["status"] == "releasing" and not lease["protection_obligations"]:
        _need(lease["planner_tranches"] == 0, "released lease has unmatched planner reservation")
        lease.update(status="released")
    elif (lease["status"] == "active" and lease["engineers_remaining"] == 0
          and not lease["selection_pending"] and len(lease["attempt_ids"]) == 2
          and lease["reviewed_attempt_ids"] == lease["attempt_ids"] and not lease["protection_obligations"]):
        _need(lease["planner_tranches"] == 0, "completed lease has unmatched planner reservation")
        lease.update(status="completed", reason="Both engineer episodes and their reviews completed.")
    _sync(lease)


def _release(lease, reason):
    _text(reason, "release reason")
    _need(lease["reviewed_attempt_ids"] == lease["attempt_ids"], "research review is owed before release")
    lease.update(engineers_remaining=0, selection_pending=False,
                 planner_tranches=len(lease["protection_obligations"]),
                 status="releasing" if lease["protection_obligations"] else "released", reason=reason)
    _sync(lease)


def prepare_planner(state, opportunity, capacity, now):
    """Fund one ordinary planner row before the caller appends it to the ledger.

    A caller replaying a saved context may run this on a deepcopy with the same
    complete row to reproduce the original transfer. Spare resources may fund
    a planner with a different window; drawing on the lease needs the exact
    planner reservation and window the lease was opened with.
    """
    lease = active_lease(state)
    if lease is None:
        return None
    _need(lease["status"] != "blocked", "program lease is blocked")
    operation_id = _text(opportunity.get("opportunity_id"), "planner operation_id")
    _need(opportunity.get("engineer_id") == "planner"
          and all(row["opportunity_id"] != operation_id for row in state["opportunities"]),
          "planner preparation must precede its ordinary row")
    _need(not any(not draw["resolved"] for draw in lease["planner_draws"])
          and all(draw["operation_id"] != operation_id for draw in lease["planner_draws"]),
          "previous planner draw must resolve exactly once")
    slots = _capacity(state, capacity, current_in_ledger=False)
    _need(capacity["total_limit"] == lease["planner_total_limit"], "original planner decision ceiling changed")
    reservation = _vector(opportunity.get("reserved"), state["limits"])
    window = _window(opportunity)
    available = _account(state, now)["available"]
    surplus = (all(available[key] >= value for key, value in reservation.items())
               and opportunity["review_at"] + held_calendar(lease) < state["deadline"]
               and slots >= lease["planner_tranches"])
    if not surplus:
        _need(lease["planner_tranches"] > 0 and reservation == lease["planner_reservation"]
              and window == lease["planner_seconds"], "commitment draw requires the original planner envelope")
        _need(slots >= lease["planner_tranches"] - 1
              and opportunity["review_at"] + held_calendar(lease) - lease["planner_seconds"] < state["deadline"],
              "remaining committed work cannot fit after planner draw")
        lease["planner_tranches"] -= 1
        _sync(lease)
    draw = {"operation_id": operation_id, "funding": "surplus" if surplus else "commitment",
            "reservation": reservation, "window_seconds": window, "resolved": False,
            "steps_after_current": slots}
    lease["planner_draws"].append(draw)
    _account(state, now, [opportunity])
    return copy.deepcopy(draw)


def _terms(proposal, bound):
    terms = proposal.get("lease")
    _need(type(bound) is int and bound == 2 and type(terms) is dict
          and set(terms) == {"episodes", "release_condition", "displaced_work"}
          and type(terms["episodes"]) is int and terms["episodes"] == 2,
          "program lease requires exactly two episodes under the enabled bound")
    _text(terms["release_condition"], "release_condition")
    _text(terms["displaced_work"], "displaced_work")
    return copy.deepcopy(terms)


def _open_lease(state, proposal, planner, reservation, *, bound, now):
    terms = _terms(proposal, bound)
    operation_id = state["pending"]["operation_id"]
    capacity = state["pending"]["context"].get("planner_capacity")
    slots = _capacity(state, capacity, current_in_ledger=True)
    planner_reservation = _vector(planner["reserved"], state["limits"])
    planner_seconds = _window(planner)
    engineer_seconds = _number(reservation.get("wall_seconds"), "engineer wall_seconds", True)
    extras = 2 if proposal.get("protection_ref") is not None else 0
    tranches = 3 + extras
    _need(slots >= tranches and proposal["review_at"] + engineer_seconds + tranches * planner_seconds < state["deadline"],
          "opening lease cannot fit its reviews, selection and second engineer")
    lease = {"schema": 1, "commitment_id": "lease:" + operation_id, "direction_id": proposal["direction_id"],
             "terms": terms, "opening_planner_operation_id": operation_id,
             "engineer_reservation": reservation, "planner_reservation": planner_reservation,
             "planner_seconds": planner_seconds, "engineer_seconds": engineer_seconds,
             "planner_total_limit": capacity["total_limit"], "engineers_remaining": 2,
             "planner_tranches": tranches, "attempt_ids": [], "reviewed_attempt_ids": [],
             "protection_obligations": [], "selection_pending": True, "planner_draws": [],
             "status": "active", "reason": None}
    _sync(lease)
    available = _account(state, now)["available"]
    _need(all(available[key] >= value for key, value in lease["remaining_reservation"].items()),
          "opening lease exceeds available resources")
    state.setdefault("program_leases", []).append(lease)
    return lease


def admit_engineer(state, proposal, opportunity, *, bound, now):
    """Move one held engineer tranche to a new engineer row (before it is appended)."""
    lease = active_lease(state)
    if lease is None and "lease" not in proposal:
        return None
    _need(type(bound) is int and bound == 2, "program commitment is disabled")
    direction = _text(proposal.get("direction_id"), "direction_id")
    operation_id = state["pending"]["operation_id"]
    planner = _planner(state, operation_id)
    _need(planner["settled"] is True, "engineer admission requires a settled planner")
    reservation = _vector(opportunity.get("reserved"), state["limits"])
    _need(opportunity["direction_id"] == direction and opportunity["engineer_id"] != "planner"
          and opportunity["reserved"] == proposal["reservation"]
          and opportunity["review_at"] == proposal["review_at"], "engineer differs from the actual selected assignment")
    _need(all(row["opportunity_id"] != opportunity["opportunity_id"] for row in state["opportunities"]),
          "engineer transfer must precede its ordinary row")
    if lease is not None:
        _need(lease["status"] == "active", "outstanding program lease cannot admit an engineer")
        draw = _draw(lease, operation_id)
        if direction != lease["direction_id"]:
            _need(not lease["protection_obligations"], "protected review is owed before changing investigation")
            _release(lease, "A different investigation was selected.")
            draw["resolved"] = True
            lease = None
        else:
            _need("lease" not in proposal, "same-direction continuation cannot reset its lease")
            _need(lease["engineers_remaining"] == 1 and lease["selection_pending"]
                  and lease["attempt_ids"] and lease["attempt_ids"][-1] in lease["reviewed_attempt_ids"],
                  "committed continuation requires its preceding research review and paid selection")
            _need(all(reservation[key] <= value for key, value in lease["engineer_reservation"].items()),
                  "engineer exceeds the held E tranche")
            _resolve(lease, draw, obligation=True)
            lease["selection_pending"] = False
    if lease is None:
        if "lease" not in proposal:
            _account(state, now, [opportunity])
            return None
        lease = _open_lease(state, proposal, planner, reservation, bound=bound, now=now)
    elif proposal.get("protection_ref") is not None:
        capacity = state["pending"]["context"].get("planner_capacity")
        slots = _capacity(state, capacity, current_in_ledger=True)
        _need(capacity["total_limit"] == lease["planner_total_limit"], "original planner decision ceiling changed")
        lease["planner_tranches"] += 2
        _need(slots >= lease["planner_tranches"], "protected continuation exceeds planner capacity")
    lease["engineers_remaining"] -= 1
    _need(opportunity["opportunity_id"] not in lease["attempt_ids"], "duplicate committed engineer attempt")
    lease["attempt_ids"].append(opportunity["opportunity_id"])
    if proposal.get("protection_ref") is not None:
        lease["protection_obligations"].extend(
            {"attempt_id": opportunity["opportunity_id"], "action": action} for action in ("observe", "review"))
    _sync(lease)
    _need(proposal["review_at"] + held_calendar(lease) < state["deadline"],
          "selected engineer leaves no time for its committed remainder")
    _account(state, now, [opportunity])
    return lease


def planner_action(state, proposal, operation_id, *, attempt_id=None, now):
    """Settle a planner draw: a committed review, a metadata action or an early release."""
    lease = active_lease(state)
    if lease is None:
        _need(proposal.get("kind") != "lease_release" and "lease_release" not in proposal,
              "no program lease to release")
        return None
    _need(lease["status"] != "blocked", "program lease is blocked")
    _need(_planner(state, operation_id)["settled"] is True, "planner action requires settlement")
    draw = _draw(lease, operation_id)
    kind = proposal.get("kind")
    same = proposal.get("direction_id") == lease["direction_id"]
    if kind == "lease_release":
        _need(same, "release must name the bound investigation")
        _need(draw["funding"] == "surplus" or lease["selection_pending"],
              "release cannot consume a planner tranche owed to protection")
        _release(lease, proposal.get("reason"))
        draw["resolved"] = True
    elif (kind == "research_review" and same and attempt_id in lease["attempt_ids"]
          and attempt_id not in lease["reviewed_attempt_ids"]):
        _need(attempt_id == lease["attempt_ids"][len(lease["reviewed_attempt_ids"])], "committed reviews must follow attempt order")
        _resolve(lease, draw, obligation=True)
        lease["reviewed_attempt_ids"].append(attempt_id)
        if "lease_release" in proposal:
            _release(lease, proposal["lease_release"])
    elif (kind == "maturation_review" and same
          and {"attempt_id": attempt_id, "action": proposal["decision"]["action"]} in lease["protection_obligations"]):
        action = proposal["decision"]["action"]
        obligation = {"attempt_id": attempt_id, "action": action}
        _need(action != "review" or {"attempt_id": attempt_id, "action": "observe"} not in lease["protection_obligations"],
              "protected review must follow its observation")
        _resolve(lease, draw, obligation=True)
        lease["protection_obligations"].remove(obligation)
    else:
        _need("lease_release" not in proposal
              or (kind == "research_review" and same and attempt_id in lease["reviewed_attempt_ids"]),
              "release must name a committed research review")
        _need(draw["funding"] == "surplus", "metadata cannot consume a mandatory planner tranche")
        _resolve(lease, draw, obligation=False)
        if "lease_release" in proposal:
            _release(lease, proposal["lease_release"])
    _complete(lease)
    _account(state, now)
    check_remaining(state, now)
    return lease


def reject_planner(state, operation_id, reason, now):
    """Keep the costs; refund a failed lease-funded planner only from real spare resources.

    Returns True when the remaining promise cannot be funded or fit. A blocked
    lease keeps its existing unstarted hold and never gets a retry slot.
    """
    lease = active_lease(state)
    if lease is None:
        return False
    if lease["status"] == "blocked":
        return True
    _text(reason, "rejection reason")
    _planner(state, operation_id)
    draw = _draw(lease, operation_id)
    available = _account(state, now)["available"]
    replenish = int(draw["funding"] == "commitment")
    tranches = lease["planner_tranches"] + replenish
    consumed = sum(row["engineer_id"] == "planner" for row in state["opportunities"])
    slots = min(draw["steps_after_current"], lease["planner_total_limit"] - consumed)
    feasible = (slots >= tranches and now + held_calendar(lease) + replenish * lease["planner_seconds"] < state["deadline"]
                and all(available[key] >= replenish * value for key, value in lease["planner_reservation"].items()))
    draw.update(resolved=True, rejection_reason=reason)
    if feasible:
        lease["planner_tranches"] = tranches
    else:
        lease.update(status="blocked", reason=reason)
    _sync(lease)
    _account(state, now)
    return not feasible


# --- planner reserve ---------------------------------------------------------

# The planner-reserve checks raised a plain ValueError before the port, not
# LeaseError; callers record the class name, so keep it.
def _reserve_need(condition, message):
    require(condition, message, ValueError)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def planner_followup_reserve(context, intent):
    """Size the next ordinary planner decision from the saved planner context.

    EvolutionLoop.run supplies steps_after_current from its loop index. The
    context is saved before the current planner row is added, so that one
    decision is subtracted here. Failed and unsettled decisions count. The
    current planner reservation is also the window used for the later planner.
    """
    capacity = context.get('planner_capacity')
    _reserve_need(type(capacity) is dict
          and set(capacity) == {'total_limit', 'steps_after_current'}
          and type(capacity['total_limit']) is int and capacity['total_limit'] > 0
          and type(capacity['steps_after_current']) is int
          and capacity['steps_after_current'] >= 0,
          'frozen ordinary planner capacity required')
    opportunities = context['state']['opportunities']
    _reserve_need(type(opportunities) is list
          and all(type(row) is dict and type(row.get('engineer_id')) is str
                  and row.get('opportunity_id') != intent['operation_id']
                  for row in opportunities),
          'frozen opportunities must precede the current planner reservation')
    consumed = sum(row['engineer_id'] == 'planner' for row in opportunities)
    after_current = capacity['total_limit'] - consumed - 1
    _reserve_need(after_current >= 0, 'current planner exceeds its original decision ceiling')
    reservation = intent['reservation']
    _reserve_need(type(reservation) is dict
          and set(reservation) == {'wall_seconds', 'evaluations'}
          and _finite(reservation['wall_seconds']) and reservation['wall_seconds'] > 0
          and _finite(reservation['evaluations']) and reservation['evaluations'] == 0,
          'configured ordinary planner reservation required')
    return {'reservation': dict(reservation),
            'decisions_after_current': min(after_current, capacity['steps_after_current'])}


def bounded_engineer_reservation(available, campaign_end, planner_end,
                                 native_envelope_seconds, *, engineer_limit, followup):
    """Fit a fresh engineer and one later ordinary planner inside both deadlines.

    The engineer allowance assumes the engineer settles on time and the loop
    continues; it never creates a later call or decision slot. If no later
    decision fits, fresh engineering is unavailable (the current planner can
    still pick another action or stop).
    """
    _reserve_need(type(engineer_limit) is dict
          and set(engineer_limit) == {'wall_seconds', 'evaluations'}
          and all(type(value) is int and value > 0 for value in engineer_limit.values())
          and type(available) is dict and set(available) == set(engineer_limit)
          and all(_finite(value) and value >= 0 for value in available.values())
          and type(campaign_end) is int and type(planner_end) is int
          and _finite(native_envelope_seconds) and native_envelope_seconds > 0,
          'finite remaining allowance and native envelope required')
    _reserve_need(type(followup) is dict and set(followup) == {'reservation', 'decisions_after_current'}
          and type(followup['decisions_after_current']) is int
          and followup['decisions_after_current'] >= 0,
          'frozen followup planner capacity required')
    planner_reservation = followup['reservation']
    _reserve_need(type(planner_reservation) is dict
          and set(planner_reservation) == set(engineer_limit)
          and _finite(planner_reservation['wall_seconds'])
          and planner_reservation['wall_seconds'] > 0
          and _finite(planner_reservation['evaluations'])
          and planner_reservation['evaluations'] == 0,
          'configured followup planner reservation required')
    if followup['decisions_after_current'] == 0:
        return {'wall_seconds': 0, 'evaluations': 0}, False
    planner_wall = planner_reservation['wall_seconds']
    # Round the held planner window up and the engineer allowance down; keep
    # the strict one-nanosecond boundary of the original offer rule.
    planner_ns = math.ceil(planner_wall * NS)
    wall = max(0, min(engineer_limit['wall_seconds'],
                      math.floor(available['wall_seconds'] - planner_wall),
                      (campaign_end - planner_end - planner_ns - 1) // NS))
    calls = min(engineer_limit['evaluations'], math.floor(available['evaluations']))
    reservation = {'wall_seconds': wall, 'evaluations': calls}
    # The engineer's completion margin stays inside its own allowance; the
    # later planner window already includes the coordinator's margin.
    return reservation, wall > native_envelope_seconds + 30 and calls >= 1
