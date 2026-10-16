"""Check a campaign's final recommendation by replaying its whole saved chain.

Given the final committed campaign snapshot, ``validate_final_recommendation``
re-reads every recommendation event from the store, rebuilds each record from
its saved planner inputs and reply, and checks that the records, the planner
bookkeeping and the recommendation head all agree. It returns ``selected``
with the exact chosen program, or ``missing``. There is no fallback: a
``clear`` leaves the result missing, and an earlier pick, the latest program,
the fastest program or ORIGINAL is never substituted. Anything inconsistent
raises ``RecommendationValidationError``.

The caller must pass the complete committed journal (a chosen historical
prefix proves nothing) and two callbacks:

* ``read_ref(ref)`` returns the bytes of an absolute ``{path, sha256}`` file
  reference;
* ``authenticate_choice(context, choice)`` must call
  ``choices.authenticate_choice(store, context, choice, episode)`` with the real
  episode reader and return the regenerated choice. A plain hash check does not
  implement it.

This module writes nothing, calls no model and evaluates nothing. The planner
reply must come from the model, route and transport fixed by the campaign's
original admission (``model_config``): a Claude route must report exactly the
requested model; a Codex route is checked with the caller's ``runtime`` services
(``runtime.routes`` and ``runtime.transport``, as in ``planner.model``), whose
modules must load from the admitted ``codex_native_root``.
"""

import copy
import functools
import json
import math
from pathlib import PurePosixPath

from .._util import EvoError, is_sha256, require, sha256_hex
from ..model_profile import LEGACY_UNROUTED_CLAUDE_MODEL
from .recommendation import digest, encoded, extract_structured_response, make_record, split_response
from .work import SelectionError, account_opportunities


_EVENT_KEYS = {"record_ref", "commit_revision", "applied", "in_budget"}
_PROOF_ARTIFACTS = {"planner.json", "planner_context.json", "context.json", "intent.json",
                    "recommendation-binding.json", "recommendation-acceptance.json",
                    "recommendation-choices.json", "request.json", "native/receipt.json",
                    "native-result.json", "response.txt", "raw_proposal.json", "proposal.json",
                    "outcome.json", "completion.json", "proposal-binding.json"}
_OPPORTUNITY_KEYS = ("job_id", "measurement_epoch_id", "opportunity_id", "direction_id", "island_id",
                     "engineer_id", "question", "opened_at", "review_at", "requested", "reserved")
NS = 1_000_000_000


class RecommendationValidationError(EvoError, ValueError):
    """The final recommendation does not have an exact, consistent saved chain."""


def _need(value, message):
    require(value, message, RecommendationValidationError)


def _shape_errors(function):
    @functools.wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
            raise RecommendationValidationError("incomplete or malformed recommendation proof") from exc
    return checked


def _decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value, "duplicate JSON field in recommendation evidence")
            value[key] = item
        return value
    _need(type(raw) is bytes, "recommendation evidence must contain exact bytes")
    value = json.loads(raw, object_pairs_hook=pairs)
    encoded(value)  # Rejects non-finite numbers.
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _ref_bytes(ref, read_ref):
    _need(type(ref) is dict and set(ref) == {"path", "sha256"}
          and type(ref["path"]) is str and PurePosixPath(ref["path"]).is_absolute()
          and is_sha256(ref["sha256"]),
          "exact absolute file reference required")
    raw = read_ref(copy.deepcopy(ref))
    _need(type(raw) is bytes and sha256_hex(raw) == ref["sha256"], "referenced recommendation bytes differ")
    return raw


def _check_route_field(value, message):
    require(type(value) is str and value and len(value) <= 256 and value.strip() == value
            and not any(ord(c) < 32 for c in value), message, ValueError)


def _validate_claude_route(profile):
    """One explicit Claude route; model names are provider reports, not proof of weights."""
    required = {"schema", "workflow_backend", "wire_protocol", "requested_model", "accepted_reported_models"}
    require(type(profile) is dict and required <= set(profile) <= required | {"requested_reasoning_effort"},
        "exact native model route fields required", ValueError)
    require(profile["schema"] == "geak-native-model-route-v1"
            and profile["workflow_backend"] == "claude"
            and profile["wire_protocol"] == "anthropic_messages", "unsupported native model route", ValueError)
    requested, accepted = profile["requested_model"], profile["accepted_reported_models"]
    _check_route_field(requested, "explicit bounded requested model required")
    require(type(accepted) is list and accepted, "explicit bounded reported model identities required", ValueError)
    for item in accepted:
        _check_route_field(item, "explicit bounded reported model identities required")
    require(len(accepted) == len(set(accepted)) and requested in accepted,
            "reported model identities must include the CLI requested identity without duplicates", ValueError)
    return copy.deepcopy(profile)


def _codex_runtime(runtime):
    """A Codex record is checked with the caller's route and transport services."""
    require(runtime is not None and runtime.routes is not None and runtime.transport is not None,
            "explicit model route services required", ValueError)


def _codex_module_paths(config, **modules):
    """Each loaded Codex module must be the one under this record's admitted root."""
    root = config.get("codex_native_root")
    _need(type(root) is str and PurePosixPath(root).is_absolute(),
          "absolute admitted Codex package root required")
    for name, module in modules.items():
        _need(getattr(module, "__file__", None) == str(PurePosixPath(root) / (name + ".py")),
              "loaded Codex module differs from this record's admitted package root")


def _model_identity(config, read_ref, runtime):
    """Return (requested model, route ref) fixed by the original admission."""
    _need(type(config) is dict, "original admitted model configuration required")
    backend = config.get("workflow_backend", "claude")
    _need(backend in ("claude", "codex"), "unsupported admitted model backend")
    if "model_route_ref" not in config:
        _need(backend == "claude", "Codex recommendation requires its admitted route")
        return LEGACY_UNROUTED_CLAUDE_MODEL, None
    if backend == "codex":
        _codex_runtime(runtime)
        _codex_module_paths(config, responses_route=runtime.routes)
        validate_route = runtime.routes.validate
    else:
        validate_route = _validate_claude_route
    reference = config["model_route_ref"]
    route = validate_route(_decode(_ref_bytes(reference, read_ref)))
    if backend == "claude":
        _need(route["accepted_reported_models"] == [route["requested_model"]],
              "native recommendation requires one exact requested and reported model")
    return route["requested_model"], copy.deepcopy(reference)


def _actual_model_config(original, artifacts, receipt, read_ref):
    """A Codex call's actual pre-call config; only workspace fields may differ."""
    if original.get("workflow_backend", "claude") != "codex":
        return original
    actual = _decode(artifacts["model-config.json"])
    reference = _decode(artifacts["native/intent.json"]).get("model_config_ref")
    _need(receipt.get("model_config_ref") == reference
          and _decode(_ref_bytes(reference, read_ref)) == actual,
          "actual model configuration lacks the matching native pre-call reference")
    _need(type(actual) is dict and set(actual) == set(original),
          "actual model configuration fields differ from original admission")
    derived = {"work_root", "state_root", "read_roots", "codex_runtime_ref"}
    _need(all(actual[key] == value for key, value in original.items() if key not in derived),
          "actual model configuration changes an immutable admitted field")
    if "codex_runtime_family_ref" not in original:
        _need(actual.get("codex_runtime_ref") == original.get("codex_runtime_ref"),
              "exact-only admission cannot derive a runtime descriptor")
    return actual


def _descriptor_observation(config, transport, descriptor, audit, read_ref):
    validation = transport.get("descriptor_validation")
    if (type(validation) is not dict
            or descriptor.get("workspace") != config.get("work_root")
            or descriptor.get("read_roots") != config.get("read_roots")):
        return False
    if "codex_runtime_family_ref" not in config:
        return (validation == {"mode": "exact_descriptor",
                               "runtime_descriptor_ref": config["codex_runtime_ref"]}
                and audit.get("runtime_descriptor_ref") == config["codex_runtime_ref"])
    policy_ref = config["codex_runtime_family_ref"]
    policy = _decode(_ref_bytes(policy_ref, read_ref))
    return (policy.get("schema") == "geak-codex-descriptor-family-v1"
            and policy.get("model_route_ref") == config["model_route_ref"]
            and policy.get("seed_audit_ref") == config["tool_audit_ref"]
            and audit.get("runtime_descriptor_ref") == policy.get("seed_descriptor_ref")
            and validation == {
                "mode": "owner_descriptor_family", "family_policy_ref": policy_ref,
                "seed_audit_ref": policy["seed_audit_ref"],
                "seed_descriptor_ref": policy["seed_descriptor_ref"],
                "relocation_evidence_ref": policy["relocation_evidence_ref"],
                "current_workspace_exact_tool_tested": False})


def _model_observation(config, receipt, observed, expected_model, read_ref, runtime):
    """The completed call used the admitted model (Claude) or route and transport (Codex).

    Codex events prove framing and tool use, not the served model, so a Codex
    call is checked against its pinned route, runtime descriptor and tool audit.
    """
    if config.get("workflow_backend", "claude") == "claude":
        return (observed.get("reported_models") == [expected_model]
                and observed.get("reported_tools") == ["Bash"])
    _codex_runtime(runtime)
    responses_route, codex_runtime = runtime.routes, runtime.transport
    _codex_module_paths(config, responses_route=responses_route, codex_runtime=codex_runtime)
    route = responses_route.validate(_decode(_ref_bytes(config["model_route_ref"], read_ref)))
    transport = receipt.get("transport") or {}
    descriptor = _decode(_ref_bytes(config["codex_runtime_ref"], read_ref))
    audit = _decode(_ref_bytes(config["tool_audit_ref"], read_ref))
    return (codex_runtime.completed(observed)
            and _descriptor_observation(config, transport, descriptor, audit, read_ref)
            and transport.get("backend") == "codex"
            and transport.get("model_route_ref") == config["model_route_ref"]
            and transport.get("route_id") == responses_route.route_id(route)
            and "codex_runtime_ref" in config
            and transport.get("runtime_descriptor_ref") == config["codex_runtime_ref"]
            and transport.get("requested_model") == expected_model
            and transport.get("requested_reasoning_effort") == route["requested_reasoning_effort"]
            and receipt.get("tool_audit") == config["tool_audit_ref"]
            and audit.get("schema") == "geak-codex-native-tool-audit-v1"
            and audit.get("qualified") is True
            and audit.get("route_id") == transport["route_id"]
            and type(audit.get("admitted_tool_profile")) is dict
            and transport.get("tool_profile") == audit["admitted_tool_profile"]
            and transport.get("cli") == descriptor["executable_ref"]
            and transport["cli"] in audit["code_bindings"]
            and type(transport.get("transport")) is dict
            and transport["transport"].get("path") == str(PurePosixPath(config["codex_native_root"]) / "codex_runtime.py")
            and transport["transport"] in audit["code_bindings"]
            and observed.get("reported_models") == []
            and observed.get("reported_tools") is None
            and observed.get("reported_reasoning_effort") is None)


def _state(store, campaign_id, revision, binding):
    _need(type(revision) is int and revision > 0, "positive committed revision required")
    committed = store.get_revision(campaign_id, revision)
    _need(committed.get("revision") == revision, "store returned another campaign revision")
    if "direction_id" in committed:
        _need(committed["direction_id"] == campaign_id, "store returned another campaign")
    state = committed["note"]["evolution"]
    _need(type(state) is dict and state["binding"] == binding,
          "committed recommendation campaign binding differs")
    return state


def _evidence(store, identity, binding, read_ref, *, for_committed_record=False):
    """Read only the named proof artifacts of one planner evidence group
    (plus the model-config artifacts a committed routed record needs)."""
    _need(is_sha256(identity), "exact planner evidence identity required")
    # store.evidence() would read the whole group, including raw model events.
    objects = PurePosixPath(str(store.root)) / "objects"
    _need(objects.is_absolute(), "absolute immutable store root required")
    envelope = _decode(_ref_bytes({"path": str(objects / identity), "sha256": identity}, read_ref))
    _need(type(envelope) is dict and envelope.get("kind") == "evidence"
          and digest(envelope) == identity
          and all(envelope["binding"][key] == value for key, value in binding.items()),
          "planner evidence envelope or campaign binding differs")
    candidate = store.get_candidate(envelope["candidate_id"])
    _need(all(candidate[key] == value for key, value in binding.items())
          and envelope["binding"] == dict(binding, candidate_source_id=candidate["source_id"]),
          "planner evidence source binding differs from its immutable candidate")
    _need(type(envelope["artifacts"]) is dict, "planner evidence artifact inventory required")
    inventory = envelope["artifacts"]
    _need("recommendation-acceptance.json" not in inventory or "recommendation-binding.json" in inventory,
          "recommendation acceptance artifact has no binding record")
    _need(not ({"recommendation-binding.json", "recommendation-acceptance.json"} & set(inventory))
          or "planner.json" in inventory,
          "recommendation artifacts have no planner result")
    artifacts = {}

    def read(name):
        content_id = inventory[name]
        _need(is_sha256(content_id), "exact planner artifact identity required")
        artifacts[name] = _ref_bytes({"path": str(objects / content_id), "sha256": content_id}, read_ref)

    if "planner.json" not in inventory:
        return artifacts
    read("planner.json")
    planner = _decode(artifacts["planner.json"])
    if "host_recommendation" not in planner and "recommendation-binding.json" not in inventory:
        return artifacts
    for name in sorted((_PROOF_ARTIFACTS & set(inventory)) - {"planner.json"}):
        read(name)
    if for_committed_record:
        model_config = _decode(artifacts["intent.json"])["admission"]["model_config"]
        _need(type(model_config) is dict, "original admitted model configuration required")
        if "model_route_ref" in model_config:
            for name in ("model-config.json", "native/intent.json"):
                read(name)
    return artifacts


def _operation(state, operation_id, campaign_id, binding):
    rows = [row for row in state["opportunities"] if row["opportunity_id"] == operation_id]
    _need(len(rows) == 1, "planner opportunity must be unique and committed")
    row = rows[0]
    _need(row["engineer_id"] == "planner" and row["direction_id"] == campaign_id
          and row["island_id"] == "campaign"
          and all(row[key] == value for key, value in binding.items()),
          "recommendation opportunity is not its campaign planner")
    return row


def _accounted(state):
    try:
        account_opportunities(state["opportunities"], binding=state["binding"],
                              limits=state["limits"], deadline=state["deadline"], now=state["deadline"])
    except SelectionError:
        return False
    return True


def _response(raw):
    _need(len(raw) <= 262144, "oversized recommendation response")
    inner_raw = extract_structured_response(raw, decode=_decode)
    response = _decode(inner_raw)
    _need(response is None or type(response) is dict, "structured recommendation response required")
    inner, directive = split_response(response)
    if type(response) is dict and "proposal" in response:
        inner_raw = encoded(inner)
    return inner_raw, inner, directive


def _check_context(store, artifacts, *, binding, campaign_id, clock_ref):
    context = _decode(artifacts["context.json"])
    intent = _decode(artifacts["intent.json"])
    planner = _decode(artifacts["planner.json"])
    choices = _decode(artifacts["recommendation-choices.json"])
    _need(type(context) is dict and context["campaign_id"] == campaign_id
          and context["state"] == _state(store, campaign_id, context["revision"], binding)
          and _decode(artifacts["planner_context.json"]) == context
          and intent["campaign_id"] == campaign_id and intent["binding"] == binding
          and intent["context_sha256"] == digest(context)
          and intent["operation_id"] == intent["invocation_id"] == planner["operation_id"]
          and intent["admission"]["clock_ref"] == clock_ref,
          "recommendation frozen context or original intent differs")
    _need(type(choices) is list and all(type(choice) is dict for choice in choices),
          "complete frozen recommendation choice list required")
    handles = [choice["choice_handle"] for choice in choices]
    _need(len(handles) == len(set(handles)), "duplicate frozen recommendation handle")
    for choice in choices:
        _need(choice["binding"] == dict(binding, campaign_id=campaign_id),
              "frozen recommendation choice belongs to another campaign")
    return context, intent, planner, choices


def _check_planner_call(artifacts, intent, clock_ref, read_ref, runtime):
    """The saved planner request, result and receipt describe one completed call,
    by the admitted model/route/transport, whose reply text is ``response.txt``."""
    request = _decode(artifacts["request.json"])
    returned = _decode(artifacts["native-result.json"])
    receipt_raw = artifacts["native/receipt.json"]
    receipt = _decode(receipt_raw)
    observed = receipt.get("observed") or {}
    model_config = intent["admission"]["model_config"]
    expected_model, model_route_ref = _model_identity(model_config, read_ref, runtime)
    if "model_profile" in receipt:
        from ..model_profile import ModelProfile
        route = _decode(_ref_bytes(model_route_ref, read_ref))
        expected_profile = ModelProfile.from_route(route, model_route_ref,
            workflow_backend=model_config.get("workflow_backend", "claude"))
        _need(receipt["model_profile"] == expected_profile.as_dict(), "recorded model profile differs from original admission")
    actual_model_config = _actual_model_config(model_config, artifacts, receipt, read_ref)
    _need(receipt.get("model_route_ref") == model_route_ref
          and (model_route_ref is None or (
              _decode(artifacts["model-config.json"]).get("model_route_ref") == model_route_ref
              and _decode(artifacts["native/intent.json"]).get("model_route_ref") == model_route_ref)),
          "native recommendation model route differs from original admission")
    receipt_ref = returned["receipt_ref"]
    _need(_ref_bytes(receipt_ref, read_ref) == receipt_raw,
          "native planner receipt is not the original retained receipt")
    _need(request["schema"] == "geak-standalone-model-request-v1"
          and returned["schema"] == "geak-standalone-model-result-v1"
          and receipt["schema"] == "geak-standalone-model-receipt-v1"
          and receipt["invocation_id"] == returned["invocation_id"] == request["invocation_id"] == intent["operation_id"]
          and receipt["clock_ref"] == clock_ref
          and receipt["model_backend_ref"] == intent["admission"]["backend_ref"]
          and receipt["requested_model"] == request["model"] == expected_model
          and receipt["label"] == request["label"]
          and receipt["original_native_timeout_ms"] == request["timeout_ms"],
          "native recommendation receipt identity differs from original admission")
    _need(receipt["settlement"].get("settled") is True and returned.get("settled") is True
          and receipt.get("status") == "completed" and receipt.get("returncode") == 0
          and observed.get("identity_valid") is True
          and _model_observation(actual_model_config, receipt, observed, expected_model, read_ref, runtime)
          and observed.get("terminal") == "turn.completed"
          and not observed.get("protocol_errors") and type(returned.get("text")) is str
          and returned["text"] == observed.get("report_text")
          and artifacts["response.txt"] == returned["text"].encode(),
          "recommendation lacks authenticated native completion and exact response")
    _need(_number(receipt["elapsed_seconds"]) and receipt["elapsed_seconds"] >= 0
          and returned["elapsed_seconds"] == receipt["elapsed_seconds"],
          "native recommendation duration differs")
    return receipt, receipt_ref


def _check_acceptance(artifacts, record, intent):
    acceptance = _decode(artifacts["recommendation-acceptance.json"])
    _need(type(acceptance) is dict and set(acceptance) == {
              "schema", "operation_id", "binding_sha256", "durable_before_monotonic_ns"}
          and acceptance["schema"] == "geak-member-recommendation-acceptance-v1"
          and acceptance["operation_id"] == intent["operation_id"]
          and acceptance["binding_sha256"] == digest(record)
          and type(acceptance["durable_before_monotonic_ns"]) is int
          and record["accepted_monotonic_ns"] <= acceptance["durable_before_monotonic_ns"]
          < min(record["planner_deadline_monotonic_ns"], record["campaign_deadline_monotonic_ns"]),
          "recommendation lacks its exact durable pre-cutoff acceptance witness")
    return acceptance


def _check_cost(artifacts, planner, intent, receipt, acceptance):
    _need("completion.json" in artifacts,
          "recommendation lacks durable whole-operation completion; omitted cost is not zero")
    outcome = _decode(artifacts["completion.json"])
    _need(_number(outcome["wall_seconds"]) and outcome["wall_seconds"] >= receipt["elapsed_seconds"],
          "recommendation operation cost omits native duration")
    cost = max(outcome["wall_seconds"],
               (acceptance["durable_before_monotonic_ns"] - intent["started_monotonic_ns"]) / NS)
    if "proposal-binding.json" in artifacts:
        bound = _decode(artifacts["proposal-binding.json"])
        _need(type(bound["accepted_monotonic_ns"]) is int, "integer action acceptance required")
        cost = max(cost, (bound["accepted_monotonic_ns"] - intent["started_monotonic_ns"]) / NS)
    _need(planner["settled"] is True and planner["costs"] == {"wall_seconds": cost, "evaluations": 0},
          "recommendation planner cost or settlement differs")


def _check_disposition(artifacts, planner, inner):
    disposition = planner.get("host_disposition")
    if disposition is not None:
        _need(type(disposition) is dict and set(disposition) == {"kind", "failure"}
              and disposition["kind"] == "planner_response_rejected" and planner["proposal"] is None,
              "failed native invocation cannot author a recommendation")
        failure = disposition["failure"]
        _need(type(failure) is dict and set(failure) == {"stage", "kind", "message"}
              and failure["stage"] in ("structured_response", "proposal_admission"),
              "recommendation action rejection is not an exact host rejection")
    else:
        _need(inner is None or type(inner) is dict, "unrejected planner action must be object or null")
        _need(_decode(artifacts["proposal.json"]) == planner["proposal"],
              "retained bound planner action differs")


def _authenticate_record(store, artifacts, record, *, binding, campaign_id,
                         clock_ref, clock, read_ref, authenticate_choice, runtime):
    """Rebuild one saved record from its planner inputs and reply; check it matches."""
    context, intent, planner, choices = _check_context(
        store, artifacts, binding=binding, campaign_id=campaign_id, clock_ref=clock_ref)
    receipt, receipt_ref = _check_planner_call(artifacts, intent, clock_ref, read_ref, runtime)
    inner_raw, inner, directive = _response(artifacts["response.txt"])
    _need(artifacts["raw_proposal.json"] == inner_raw,
          "recommendation inner proposal differs from authenticated response")
    actual = make_record(context=context, intent=intent, clock_ref=clock_ref, clock=clock,
                         receipt_ref=receipt_ref, response_raw=artifacts["response.txt"],
                         proposal_raw=inner_raw, directive=directive, choices=choices,
                         accepted_ns=record["accepted_monotonic_ns"])
    _need(record == actual and planner["host_recommendation"] == actual,
          "recommendation binding differs from its exact deterministic record")
    if record["disposition"] == "set":
        choice = record["choice"]
        _need(encoded(authenticate_choice(copy.deepcopy(context), copy.deepcopy(choice))) == encoded(choice),
              "recommendation choice lacks exact authenticated native SELECTED witness")
    _need(record["planner_started_monotonic_ns"] <= record["accepted_monotonic_ns"]
          < record["planner_deadline_monotonic_ns"] <= record["campaign_deadline_monotonic_ns"]
          and clock["started_monotonic_ns"] <= record["planner_started_monotonic_ns"],
          "recommendation was not durably accepted before both original cutoffs")
    acceptance = _check_acceptance(artifacts, record, intent)
    _check_cost(artifacts, planner, intent, receipt, acceptance)
    _check_disposition(artifacts, planner, inner)
    return context, intent, planner


def _campaign_clock(clock_ref, read_ref, expected_binding):
    clock = _decode(_ref_bytes(clock_ref, read_ref))
    _need(type(clock["started_monotonic_ns"]) is int and clock["started_monotonic_ns"] >= 0
          and _number(clock["wall_seconds"]) and clock["wall_seconds"] > 0
          and _number(clock["started_unix"]) and clock["started_unix"] >= 0,
          "authenticated original campaign clock required")
    for key in ("boot_id", "session_id"):  # Monotonic times only compare within one boot.
        if key in clock:
            _need(type(clock[key]) is str and bool(clock[key]), "original clock identity is incomplete")
    for key, value in expected_binding.items():
        if key in clock:
            _need(clock[key] == value, "original clock belongs to another campaign binding")
    return clock


def _planner_cutoff(clock, intent, opportunity, committed):
    start = clock["started_monotonic_ns"]
    return min(start + int(clock["wall_seconds"] * NS),
               intent["started_monotonic_ns"] + int(opportunity["reserved"]["wall_seconds"] * NS),
               start + int((min(opportunity["review_at"], committed["deadline"]) - clock["started_unix"]) * NS))


def _replay_event(store, state, events, index, *, head, binding, campaign_id, clock_ref, clock,
                  read_ref, authenticate_choice, event_operations, runtime):
    """Check one event against its commit; return (applied, ref, record, operation_id)."""
    event = events[index]
    ref, revision = event["record_ref"], event["commit_revision"]
    artifacts = _evidence(store, ref["evidence_id"], binding, read_ref, for_committed_record=True)
    record = _decode(artifacts[ref["artifact"]])
    context, intent, planner = _authenticate_record(
        store, artifacts, record, binding=binding, campaign_id=campaign_id, clock_ref=clock_ref,
        clock=clock, read_ref=read_ref, authenticate_choice=authenticate_choice, runtime=runtime)
    operation_id = intent["operation_id"]
    _need(operation_id not in event_operations and context["revision"] < revision
          and record["previous_head"] == head
          and context["state"].get("recommendation_head") == head
          and context["state"].get("recommendation_events", []) == events[:index],
          "recommendation frozen context does not extend the complete prior chain")
    prior = _state(store, campaign_id, revision - 1, binding)
    committed = _state(store, campaign_id, revision, binding)
    _need(prior["phase"] in ("planning", "unsettled")
          and prior["pending"]["operation_id"] == operation_id
          and prior["pending"]["context"] == context
          and prior.get("recommendation_head") == head
          and prior.get("recommendation_events", []) == events[:index]
          and committed.get("recommendation_events", []) == events[:index + 1],
          "recommendation was not committed from its exact pending planner context")
    if prior["phase"] == "unsettled":
        _need(prior["pending"].get("host_recommendation") == record
              and prior["pending"]["proposal"] == planner["proposal"]
              and planner.get("recovery_of") == prior["pending"]["planner_ref"],
              "settled recovery changed the pending recommendation or proposal")
    opportunity = _operation(committed, operation_id, campaign_id, binding)
    prior_op = _operation(prior, operation_id, campaign_id, binding)
    terminal_op = _operation(state, operation_id, campaign_id, binding)
    _need(all(prior_op[key] == opportunity[key] == terminal_op[key] for key in _OPPORTUNITY_KEYS)
          and prior_op["settled"] is False
          and all(row["opportunity_id"] != operation_id for row in context["state"]["opportunities"]),
          "recommendation planner reservation changed after its frozen context")
    _need(intent["deadline_monotonic_ns"] == _planner_cutoff(clock, intent, opportunity, committed),
          "recommendation planner cutoff differs from its original reservation")
    receipt = {"receipt_id": ref["evidence_id"], "costs": planner["costs"]}
    _need(opportunity["settled"] is True and opportunity["receipts"] == [receipt]
          and terminal_op["settled"] is True and terminal_op["receipts"] == [receipt]
          and intent["reservation"] == opportunity["reserved"]
          and committed["limits"] == context["state"]["limits"] == state["limits"]
          and committed["deadline"] == context["state"]["deadline"] == state["deadline"],
          "recommendation lacks its original accounted planner settlement")
    accounted = _accounted(committed)
    in_budget = accounted and all(planner["costs"][key] is not None and planner["costs"][key] <= value
                                  for key, value in opportunity["reserved"].items())
    applied = in_budget and record["disposition"] in ("set", "clear")
    _need(event["in_budget"] == in_budget and event["applied"] == applied,
          "recommendation application disagrees with committed accounting")
    phase = ("accounting_blocked" if not accounted else "idle" if planner.get("host_disposition")
             else "stopped" if planner["proposal"] is None else "planned")
    _need(committed["phase"] == phase, "recommendation commit phase differs from planner result/accounting")
    new_head = copy.deepcopy(ref) if applied else head
    _need(committed.get("recommendation_head") == new_head, "committed recommendation head differs from replay")
    return applied, ref, record, operation_id


def _check_ledger(store, state, binding, read_ref, event_operations):
    """Every planner result that carries a recommendation must have its event, so
    dropping event metadata cannot revive an earlier pick or hide a clear."""
    ledger_operations, ledger_receipts = set(), set()
    for opportunity in state["opportunities"]:
        operation_id = opportunity["opportunity_id"]
        _need(operation_id not in ledger_operations, "duplicate terminal opportunity")
        ledger_operations.add(operation_id)
        if opportunity["engineer_id"] != "planner":
            continue
        for receipt in opportunity["receipts"]:
            identity = receipt["receipt_id"]
            _need(identity not in ledger_receipts, "duplicate terminal planner receipt")
            ledger_receipts.add(identity)
            artifacts = _evidence(store, identity, binding, read_ref)
            if "planner.json" not in artifacts:
                continue  # An explicit no-call closure has no model result.
            planner = _decode(artifacts["planner.json"])
            has_record = "host_recommendation" in planner
            _need(has_record == ("recommendation-binding.json" in artifacts),
                  "planner recommendation payload/artifact presence differs")
            if not has_record:
                continue
            if opportunity["settled"] is True:
                _need(event_operations.get(operation_id) == identity,
                      "terminal planner recommendation is missing its unique committed event")
                continue
            # An unsettled planner result grants nothing until its settled
            # recovery; the previous applied head stays in force.
            pending = state["pending"]
            _need(state["phase"] in ("unsettled", "accounting_blocked")
                  and type(pending) is dict and operation_id not in event_operations
                  and planner["settled"] is False and planner["operation_id"] == operation_id
                  and opportunity["receipts"] == [{"receipt_id": identity, "costs": planner["costs"]}]
                  and pending["operation_id"] == operation_id
                  and pending["planner_ref"] == {"evidence_id": identity, "artifact": "planner.json"}
                  and pending["proposal"] == planner["proposal"]
                  and pending.get("host_recommendation") == planner["host_recommendation"]
                  == _decode(artifacts["recommendation-binding.json"])
                  and pending["context"] == _decode(artifacts["context.json"])
                  == _decode(artifacts["planner_context.json"]),
                  "unsettled recommendation differs from its exact pending planner result")


@_shape_errors
def validate_final_recommendation(store, terminal_snapshot, *, expected_binding,
                                  clock_ref, read_ref, authenticate_choice, runtime=None):
    """Return ``{status: selected|missing, head, choice, reason, record,
    recorded_configuration}`` or raise ``RecommendationValidationError``.

    ``store`` provides read-only ``get_revision``, ``get_candidate``, ``source``
    and ``root``. The final phase may be planning, unsettled or blocked; that does
    not erase an earlier applied recommendation. Each event is checked against
    its own commit. An applied clear stays missing: no earlier, latest, best or
    ORIGINAL fallback.
    """
    _need(type(expected_binding) is dict
          and set(expected_binding) == {"campaign_id", "job_id", "measurement_epoch_id"}
          and all(type(value) is str and value for value in expected_binding.values()),
          "exact campaign/job/epoch binding required")
    _need(callable(read_ref) and callable(authenticate_choice), "trusted read and native witness callbacks required")
    campaign_id = expected_binding["campaign_id"]
    binding = {key: expected_binding[key] for key in ("job_id", "measurement_epoch_id")}
    _need(type(terminal_snapshot) is dict and terminal_snapshot["campaign_id"] == campaign_id,
          "root-final snapshot belongs to another campaign")
    terminal_revision = terminal_snapshot["revision"]
    state = _state(store, campaign_id, terminal_revision, binding)
    _need(terminal_snapshot["state"] == state, "root-final snapshot is not the committed campaign state")
    clock = _campaign_clock(clock_ref, read_ref, expected_binding)
    events = state.get("recommendation_events", [])
    _need(type(events) is list, "complete committed recommendation event list required")
    head, current, previous_revision = None, None, 0
    seen, event_operations = set(), {}
    for index, event in enumerate(events):
        _need(type(event) is dict and set(event) == _EVENT_KEYS
              and type(event["applied"]) is bool and type(event["in_budget"]) is bool,
              "exact recommendation event required")
        ref, revision = event["record_ref"], event["commit_revision"]
        _need(type(ref) is dict and set(ref) == {"evidence_id", "artifact"}
              and ref["artifact"] == "recommendation-binding.json"
              and ref["evidence_id"] not in seen and type(revision) is int
              and previous_revision < revision <= terminal_revision,
              "recommendation event order or exact reference differs")
        seen.add(ref["evidence_id"])
        previous_revision = revision
        applied, ref, record, operation_id = _replay_event(
            store, state, events, index, head=head, binding=binding, campaign_id=campaign_id,
            clock_ref=clock_ref, clock=clock, read_ref=read_ref,
            authenticate_choice=authenticate_choice, event_operations=event_operations, runtime=runtime)
        if applied:
            head, current = copy.deepcopy(ref), copy.deepcopy(record)
        event_operations[operation_id] = ref["evidence_id"]
    _check_ledger(store, state, binding, read_ref, event_operations)
    _need(state.get("recommendation_head") == head, "root-final recommendation head differs from complete replay")
    selected = current is not None and current["disposition"] == "set"
    choice = copy.deepcopy(current["choice"]) if selected else None
    return {"status": "selected" if selected else "missing", "head": head,
            "choice": choice, "reason": None if current is None else current["reason"],
            "record": current,
            "recorded_configuration": None if choice is None else copy.deepcopy(choice["recorded_configuration"])}
