"""Bind claim append/correction requests to exact settled engineer outcomes."""

from .._util import require


def _need(condition, reason):
    require(condition, reason, ValueError)


def claim_update(store, proposal, snapshot, bindings, *, verify_quantity=None):
    """Prepare append/correct events; EvolutionLoop adopts note and head together."""
    from ..archive.claims import ClaimLog

    state = snapshot["state"]
    direction = store.get(proposal["direction_id"])
    status = direction["note"].get("research_status")
    _need(direction["revision"] == proposal["expected_revision"]
          and status in ("pending_review", "reported"),
          "claim review requires the current pending or reported direction")
    matches = [row for row in state["history"]
               if row.get("attempt_id") == direction["note"].get("evolution_operation_id")]
    _need(len(matches) == 1 and matches[0]["settled"]
          and all(matches[0][key] == proposal[key] for key in ("transport_ref", "collection_ref")),
          "claim review must name its exact settled outcome")
    _need(type(proposal["claim_requests"]) is list, "claim_requests must be a list")
    _need(status == "pending_review" or bool(proposal["claim_requests"]),
          "a reported review amendment requires an explicit claim append or correction")
    log = ClaimLog(store, state["claim_head"], verify_quantity=verify_quantity)
    for request in proposal["claim_requests"]:
        _need(type(request) is dict, "claim request must be an object")
        action = request.get("action")
        fields = ({"action", "candidate_id", "record"} if action == "append" else
                  {"action", "candidate_id", "targets", "scope", "reason", "consequence", "replacement", "status"})
        _need(action in ("append", "correct") and set(request) == fields,
              "claim request must use exact ClaimLog append or correct fields")
        candidate = store.get_candidate(request["candidate_id"])
        _need(all(candidate[key] == value for key, value in state["binding"].items()),
              "claim candidate belongs to another campaign job/epoch")
        binding = bindings(request["candidate_id"])[0]
        case_bindings = getattr(verify_quantity, "case_bindings", None)
        if callable(case_bindings):
            allowed = case_bindings(request["candidate_id"])
            _need(allowed.get(binding["case_id"]) == binding, "host claim case bindings change default scope")
            record = request["record"] if action == "append" else {"binding": request["scope"]}
            scope = record.get("binding") if type(record) is dict else None
            _need(type(scope) is dict and type(scope.get("case_id")) is str and scope["case_id"] in allowed,
                  "claim case is outside host scope")
            binding = allowed[scope["case_id"]]
        if action == "append":
            log.append(request["candidate_id"], request["record"], binding)
        else:
            _need(request["scope"] == binding, "correction differs from host-bound candidate scope")
            log.correct(request["candidate_id"], **{key: value for key, value in request.items()
                                                   if key not in ("action", "candidate_id")})
    return {"note": proposal["note"], "claim_head": log.head}
