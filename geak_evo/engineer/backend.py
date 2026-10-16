"""Claude Code command/event adapter; the caller still owns worker containment.

Raw Claude JSONL is never rewritten into Codex events. The small common terminal
vocabulary is only a host interpretation, with native identity/usage retained.
"""

import copy
import json
from pathlib import Path
import uuid


from .._util import require
from ..model_profile import ModelProfile


SETTINGS = {"enableWorkflows": True, "ultracode": True}
TOOLS = frozenset(("Bash", "Read", "Write"))


def _need(condition, message):
    require(condition, message, ValueError)


def _session(value):
    _need(type(value) is str and str(uuid.UUID(value)) == value,
          "Claude session must be a canonical UUID")
    return value


def validate_model_route(profile):
    """One explicit native route; identities are provider reports, not weight proof."""
    required = {"schema", "workflow_backend", "wire_protocol", "requested_model", "accepted_reported_models"}
    _need(type(profile) is dict and required <= set(profile) <= required | {"requested_reasoning_effort"},
        "exact native model route fields required")
    _need(profile["schema"] == "geak-native-model-route-v1"
          and profile["workflow_backend"] == "claude"
          and profile["wire_protocol"] == "anthropic_messages", "unsupported native model route")
    requested, accepted = profile["requested_model"], profile["accepted_reported_models"]
    _need(type(requested) is str and requested and len(requested) <= 256
          and requested.strip() == requested and not any(ord(c) < 32 for c in requested),
          "explicit bounded requested model required")
    _need(type(accepted) is list and accepted and all(type(item) is str and item
          and len(item) <= 256 and item.strip() == item and not any(ord(c) < 32 for c in item)
          for item in accepted), "explicit bounded reported model identities required")
    _need(len(accepted) == len(set(accepted)) and requested in accepted,
          "reported model identities must include the CLI requested identity without duplicates")
    return copy.deepcopy(profile)


def claude_command(request, report_path, *, executable, session_directory,
                   model_profile, model_route=None, tools=("Bash", "Read", "Write"), sandbox_settings=None,
                   env_executable="/usr/bin/env"):
    """Build native CLI argv without credentials or an implicit fallback model.

    This retains the existing CLI and ultracode settings; the model is selected
    by the explicit profile. EVO does not invoke GEAK's recursive Workflow runtime: its own
    admitted turns own allocation. G and EVO must share this exact tool set.
    CLI flags limit tools/config discovery, not filesystem or GPU authority.
    An external qualified launcher must supply the private provider environment,
    writable session storage, read-only evidence and the bounded worktree.
    """
    del report_path  # Final text is extracted from native result AFTER settlement.
    _need(isinstance(model_profile, ModelProfile) and model_profile.workflow_backend == "claude",
          "explicit Claude model profile required")
    route = model_profile.verify()
    _need(model_route is None or model_route == route, "model route differs from explicit profile")
    model = validate_model_route(route)["requested_model"]
    _need(model_profile.requested_reasoning_effort == "ultracode", "unqualified effort route")
    _need(type(tools) in (list, tuple) and len(tools) == len(set(tools))
          and set(tools) <= TOOLS, "unsupported Claude tool set")
    directory = Path(session_directory)
    _need(directory.is_absolute(), "explicit absolute session storage required")
    _need(Path(executable).is_absolute(), "explicit pinned Claude binary required")
    command = [env_executable, "CLAUDE_CONFIG_DIR=" + str(directory),
               "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1", "PYTHONDONTWRITEBYTECODE=1",
               str(executable), "--print", "--verbose", "--bare",
               "--output-format", "stream-json", "--model", model,
               "--settings", json.dumps(SETTINGS, sort_keys=True, separators=(",", ":")),
               "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
               "--disable-slash-commands", "--no-chrome", "--tools", ",".join(tools),
               "--permission-mode", "bypassPermissions"]
    if sandbox_settings is not None:
        _need(type(sandbox_settings) is dict and tuple(tools) == ("Bash",), "qualified sandbox exposes Bash only")
        _need(sandbox_settings.get("enabled") is True and sandbox_settings.get("failIfUnavailable") is True
              and sandbox_settings.get("allowUnsandboxedCommands") is False
              and sandbox_settings.get("excludedCommands") == [], "sandbox must fail closed")
        command[command.index("--settings") + 1] = json.dumps(dict(SETTINGS, sandbox=sandbox_settings),
                                                             sort_keys=True, separators=(",", ":"))
    continuation = request["continuation"]
    _need(continuation["mode"] in ("fresh", "resume", "handoff"), "invalid continuation")
    if continuation["mode"] == "resume":
        from ..archive.refs import read_ref as _read_ref
        prior = json.loads(_read_ref(continuation["prior_receipt"]))
        _need(prior.get("backend") == "claude" and prior.get("requested_model") == model,
              "resume requires the same actual Claude backend/model")
        _need(prior.get("model_profile") == model_profile.as_dict(), "resume cannot change the original model profile")
        command += ["--resume", _session(prior["thread_id"])]
    return command


def inspect_claude_events(path, *, model_route, expected_thread=None, forbidden_thread=None,
                          max_line_bytes=8 * 1024 * 1024):
    """Inspect original stream-json events; leave raw usage and costs unqualified."""
    profile = validate_model_route(model_route)
    result = {"thread_id": None, "identity_valid": False, "terminal": None,
              "native_terminal": None, "requested_model": profile["requested_model"],
              "reported_models": [], "reported_tools": None, "usage": None, "usage_status": "unknown",
              "native_model_usage": None, "native_total_cost_usd": None,
              "report_text": None, "protocol_errors": []}
    errors = result["protocol_errors"]
    initial, terminal, sessions, models, order = [], [], [], [], []
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline(max_line_bytes + 1)
            if not line:
                break
            if len(line) > max_line_bytes:
                errors.append("oversized_event")
                break
            try:
                event = json.loads(line.decode("utf-8"))
                _need(type(event) is dict and type(event.get("type")) is str, "invalid event")
            except (ValueError, UnicodeError):
                errors.append("invalid_jsonl_event")
                continue
            kind = event["type"]
            api_error_message = kind == "assistant" and event.get("is_api_error_message") is True
            if event.get("session_id") is not None:
                sessions.append(event["session_id"])
            if kind == "system" and event.get("subtype") == "init":
                initial.append(event)
                order.append("init")
                models.append(event.get("model"))
            elif kind == "result":
                terminal.append(event)
                order.append("result")
                if type(event.get("modelUsage")) is dict:
                    models.extend(event["modelUsage"])
            elif kind == "assistant":
                if order != ["init"]:
                    errors.append("assistant_outside_turn")
                message = event.get("message")
                if type(message) is dict and message.get("model"):
                    usage = message.get("usage")
                    # The CLI's flagged, zero-token API-error envelope is local;
                    # its label is not a provider model attribution.
                    local_error = (api_error_message and type(usage) is dict and all(
                        type(usage.get(key)) is int and usage[key] == 0 for key in
                        ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")))
                    if not local_error:
                        models.append(message["model"])
            # api_retry reports an in-turn recovery attempt, not a failed turn.
            # Completion still requires the existing unique successful result;
            # missing/failed terminals and all other backend errors stay rejected.
            recoverable_retry = (kind == "system" and event.get("subtype") == "api_retry"
                                 and order == ["init"])
            if kind == "error" or api_error_message or (event.get("error") is not None and not recoverable_retry):
                errors.append("backend_error_event")
    if len(initial) == 1:
        result["reported_tools"] = initial[0].get("tools")
        if not (type(result["reported_tools"]) is list
                and all(type(tool) is str and tool in TOOLS for tool in result["reported_tools"])):
            errors.append("unexpected_or_missing_tool_set")
        try:
            identity = _session(initial[0].get("session_id"))
            result["thread_id"] = identity
            result["identity_valid"] = (bool(sessions) and all(item == identity for item in sessions)
                and (expected_thread is None or identity == expected_thread) and identity != forbidden_thread)
        except (ValueError, AttributeError):
            pass
    if not result["identity_valid"]:
        errors.append("conversation_identity_mismatch_or_missing")
    if order != ["init", "result"]:
        errors.append("invalid_turn_event_sequence")
    result["reported_models"] = sorted({item for item in models if type(item) is str})
    if not models or any(item not in profile["accepted_reported_models"] for item in models):
        errors.append("reported_model_mismatch_or_missing")
    if len(terminal) != 1:
        errors.append("terminal_event_missing_or_duplicated")
    else:
        event = terminal[0]
        if event.get("session_id") != result["thread_id"]:
            errors.append("terminal_session_mismatch")
        subtype = event.get("subtype")
        result["native_terminal"] = {key: event.get(key) for key in
                                     ("type", "subtype", "is_error", "stop_reason", "session_id")}
        success = subtype == "success" and event.get("is_error") is False
        result["terminal"] = "turn.completed" if success else "turn.failed"
        usage = event.get("usage")
        if type(usage) is dict:
            # Claude cache counters are separate input buckets; do not impose
            # Codex's cached_input_tokens <= input_tokens or add cumulative turns.
            counters = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
            if all(type(usage.get(key)) is int and usage[key] >= 0 for key in counters):
                result.update(usage=usage, usage_status="reported_native_terminal_unqualified_incremental")
            else:
                errors.append("invalid_terminal_usage")
        elif usage is not None:
            errors.append("invalid_terminal_usage")
        result["native_model_usage"] = event.get("modelUsage")
        result["native_total_cost_usd"] = event.get("total_cost_usd")
        text = event.get("result")
        if success and type(text) is str:
            result["report_text"] = text
        elif success:
            errors.append("terminal_result_text_missing")
    return result
