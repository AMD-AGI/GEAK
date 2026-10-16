"""Validate failed native planner evidence without settling or repeating a turn.

Exact records bind the call, costs and process cleanup. This module never
starts a model, worker or clock.
"""
import math
import os
from pathlib import Path
import stat

from .._util import canonical_json, require, sha256_hex
from .context import _decode

SCHEMA = 'geak-failed-native-planner-v1'
LOCAL_NAMES = ('intent.json', 'context.json', 'model-config.json', 'request.json',
               'public-manifest.json', 'launch-intent.json', 'failure.json',
               'outcome.json', 'engineer-offer.json')
NATIVE_NAMES = ('request.json', 'intent.json', 'receipt.json', 'prompt.txt',
                'events.jsonl', 'stderr.txt', 'worker_plan.json',
                'worker_admission.json', 'worker_settlement.json', 'worker_supervisor.py')
ABSENT_NAMES = ('native-result.json', 'completion.json', 'proposal-binding.json')


def need(value, message):
    """Compatibility name for the original validator's error helper."""
    require(value, message, ValueError)


def encoded(value):
    return canonical_json(value)


def digest(value):
    return sha256_hex(encoded(value))


def file_ref(path):
    return {'path': str(path), 'sha256': sha256_hex(Path(path).read_bytes())}


def checked_bytes(ref):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'exact file reference required', ValueError)
    path = Path(ref['path'])
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
         'canonical nonsymlink file required', ValueError)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'regular file required', ValueError)
    raw = path.read_bytes()
    after = path.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_uid')
    require(all(getattr(before, k) == getattr(after, k) for k in keys), 'file changed during read', ValueError)
    require(sha256_hex(raw) == ref['sha256'], 'file reference hash changed', ValueError)
    return raw


def artifact_paths(call, operation):
    paths = {name: call / name for name in LOCAL_NAMES}
    paths.update({'native/' + name: call / 'model/attempts' / operation / name
                  for name in NATIVE_NAMES})
    config = _decode(checked_bytes(file_ref(call / 'model-config.json')))
    if config.get('workflow_backend', 'claude') == 'codex':
        attempt = call / 'model/attempts' / operation
        paths['native/model-config.json'] = attempt / 'model-config.json'
        if os.path.lexists(attempt / 'final-message.txt'):
            paths['native/final-message.txt'] = attempt / 'final-message.txt'
    return paths


def codex_failed_observation(config, receipt, attempt, artifacts, *, runtime=None):
    """Authenticate retained Codex failure without inventing served identities."""
    require(runtime is not None and runtime.routes is not None and runtime.events is not None
            and runtime.boundary is not None and runtime.transport is not None,
            'explicit Codex failure observation services required', ValueError)
    responses_route, codex_events = runtime.routes, runtime.events
    codex_runtime, codex_cpu_boundary = runtime.transport, runtime.boundary
    route = responses_route.validate(_decode(checked_bytes(config['model_route_ref'])))
    descriptor = _decode(checked_bytes(config['codex_runtime_ref']))
    transport = receipt.get('transport') or {}
    require(transport.get('backend') == 'codex'
         and transport.get('model_route_ref') == config['model_route_ref']
         and transport.get('route_id') == responses_route.route_id(route)
         and transport.get('runtime_descriptor_ref') == config['codex_runtime_ref']
         and transport.get('requested_model') == route['requested_model']
         and transport.get('requested_reasoning_effort') == route['requested_reasoning_effort']
         and receipt['tool_audit'] == config['tool_audit_ref']
         and transport.get('transport') == file_ref(Path(codex_runtime.__file__).resolve()),
         'failed Codex receipt differs from admitted runtime/model/descriptor', ValueError)
    validation = transport.get('descriptor_validation')
    if 'codex_runtime_family_ref' in config:
        require(type(validation) is dict and validation.get('mode') == 'owner_descriptor_family'
             and validation.get('family_policy_ref') == config['codex_runtime_family_ref']
             and validation.get('seed_audit_ref') == config['tool_audit_ref']
             and validation.get('current_workspace_exact_tool_tested') is False,
             'failed Codex descriptor family differs from original admission', ValueError)
    else:
        require(validation in (None, {'mode': 'exact_descriptor', 'runtime_descriptor_ref': config['codex_runtime_ref']}),
             'failed Codex exact descriptor identity differs', ValueError)
    observed = codex_events.inspect_codex_events(attempt / 'events.jsonl', report_path=attempt / 'final-message.txt')
    require(observed == receipt['observed'] and observed['identity_valid'] is True
         and observed['reported_models'] == [] and observed['reported_tools'] is None
         and observed['reported_reasoning_effort'] is None and observed['report_text'] is None,
         'failed Codex observation differs from actual retained events', ValueError)
    events = [_decode(line) for line in artifacts['native/events.jsonl'].splitlines() if line.strip()]
    framing = [event['type'] for event in events if event.get('type') in codex_events.TURN_EVENTS]
    explicit = observed['terminal'] == 'turn.failed'
    allowed_errors = {'backend_error_event', 'unfinished_tool_item'}
    if explicit:
        require(framing == ['thread.started', 'turn.started', 'turn.failed'],
             'explicit failed Codex terminal is ambiguous', ValueError)
    else:
        require(observed['terminal'] is None and framing == ['thread.started', 'turn.started'],
             'incomplete Codex turn has terminal or ambiguous framing', ValueError)
        allowed_errors.update({'invalid_turn_event_sequence', 'terminal_event_missing_or_duplicated'})
    require(set(observed['protocol_errors']) <= allowed_errors,
         'failed Codex turn has unqualified protocol or tool evidence', ValueError)
    plan = _decode(artifacts['native/worker_plan.json'])
    require(plan['route_id'] == transport['route_id'] == responses_route.verify_command(route, plan['argv'])
         and plan['expected_argv_sha256'] == sha256_hex(responses_route.canonical(plan['argv']))
         and plan['executable_ref'] == descriptor['executable_ref'] == transport['cli']
         and plan['cwd'] == descriptor['workspace'] == config['work_root']
         and descriptor['read_roots'] == config['read_roots']
         and all(plan[key] == descriptor[key] for key in ('private_home', 'private_temp', 'private_codex_home'))
         and plan['transport_boundary_source'] == file_ref(Path(codex_cpu_boundary.__file__).resolve())
         and plan['responses_route_source'] == file_ref(Path(responses_route.__file__).resolve())
         and plan['source_sha256'] == codex_cpu_boundary.BASE_SOURCE_SHA256,
         'failed Codex worker command or source differs from native admission', ValueError)
    return explicit


def validate_failed_invocation(call, campaign_id, binding, clock_ref, artifacts, *, runtime=None):
    """Authenticate a failed, positively settled turn from its original records."""
    intent = _decode(artifacts['intent.json'])
    operation = intent['operation_id']
    require(call.name == sha256_hex(operation.encode())
         and intent['invocation_id'] == operation and intent['campaign_id'] == campaign_id
         and intent['binding'] == binding, 'failed invocation identity differs', ValueError)
    for name, path in artifact_paths(call, operation).items():
        require(checked_bytes(file_ref(path)) == artifacts[name], 'failed invocation artifact changed: ' + name, ValueError)
    require(all(name not in artifacts and not os.path.lexists(call / name) for name in ABSENT_NAMES),
         'native result or accepted proposal exists; failed-turn closure forbidden', ValueError)
    request = _decode(artifacts['request.json'])
    receipt = _decode(artifacts['native/receipt.json'])
    native_intent = _decode(artifacts['native/intent.json'])
    config = _decode(artifacts['model-config.json'])
    from .native import planner_model_identity
    backend = config.get('workflow_backend', 'claude')
    require(backend in ('claude', 'codex'), 'unsupported failed native planner backend', ValueError)
    expected_model, model_route_ref = planner_model_identity(intent['admission']['model_config'], runtime=runtime)
    require(config.get('model_route_ref') == model_route_ref
         and receipt.get('model_route_ref') == model_route_ref
         and native_intent.get('model_route_ref') == model_route_ref,
         'failed planner model route differs from original admission', ValueError)
    attempt = call / 'model/attempts' / operation
    require(intent['admission']['clock_ref'] == receipt['clock_ref'] == clock_ref
         and intent['admission']['campaign_root'] == str(call.parent.parent.parent)
         and config['state_root'] == str(call / 'model')
         and _decode(artifacts['native/request.json']) == request
         and request['invocation_id'] == receipt['invocation_id'] == operation
         and receipt['schema'] == 'geak-standalone-model-receipt-v1'
         and request['schema'] == 'geak-standalone-model-request-v1'
         and receipt['requested_model'] == request['model'] == expected_model
         and receipt['label'] == request['label']
         and receipt['model_backend_ref'] == intent['admission']['backend_ref']
         and receipt['original_native_timeout_ms'] == request['timeout_ms'],
         'failed native receipt differs from original admission', ValueError)
    for key in ('clock_ref', 'invocation_id', 'label', 'model_backend_ref',
                'original_native_timeout_ms', 'requested_model', 'started_monotonic_ns',
                'effective_wall_seconds', 'tool_audit', 'transport'):
        require(native_intent[key] == receipt[key], 'native admission/receipt differs: ' + key, ValueError)
    if backend == 'codex':
        from .native import validate_actual_planner_config
        validate_actual_planner_config(intent['admission']['model_config'], config, call, native_intent, receipt)
        require(_decode(artifacts['native/model-config.json']) == config,
             'archived failed Codex model configuration differs', ValueError)
    launch = _decode(artifacts['launch-intent.json'])
    require(launch == {'operation_id': operation, 'request_sha256': sha256_hex(artifacts['request.json'])},
         'launch fence differs from original request', ValueError)
    failure = _decode(artifacts['failure.json'])
    observed = receipt['observed']
    require(receipt['status'] == 'failed' and type(receipt['returncode']) is int and receipt['returncode'] != 0
         and failure['retry_permitted'] is False and type(failure['error_kind']) is str
         and receipt['error'].startswith(failure['error_kind'] + ': ')
         and observed['identity_valid'] is True and observed['terminal'] in (None, 'turn.failed')
         and observed['report_text'] is None, 'not an authenticated incomplete or explicitly failed turn', ValueError)
    events = [_decode(line) for line in artifacts['native/events.jsonl'].splitlines() if line.strip()]
    results = [event for event in events if event.get('type') == 'result']
    explicit = observed['terminal'] == 'turn.failed'
    if backend == 'codex':
        explicit = codex_failed_observation(config, receipt, attempt, artifacts, runtime=runtime)
    elif explicit:
        require(observed['reported_models'] == [expected_model] and observed['reported_tools'] == ['Bash'],
             'failed Claude model or tools differ', ValueError)
        # The backend's own unique failed terminal. It settles exactly like an
        # incomplete turn: charged, never accepted, never redispatched. A
        # success-shaped, duplicated or receipt-inconsistent terminal stays rejected.
        terminal = results[0] if len(results) == 1 else {}
        require(len(results) == 1
             and not (terminal.get('subtype') == 'success' and terminal.get('is_error') is False)
             and terminal.get('session_id') == observed['thread_id']
             and observed['native_terminal'] == {key: terminal.get(key) for key in
                                                 ('type', 'subtype', 'is_error', 'stop_reason', 'session_id')},
             'explicit failed terminal is ambiguous or differs from its receipt', ValueError)
    else:
        require(observed['reported_models'] == [expected_model] and observed['reported_tools'] == ['Bash'],
             'failed Claude model or tools differ', ValueError)
        require(events and not results, 'native result event exists; incomplete-turn closure forbidden', ValueError)
    worker = _decode(artifacts['native/worker_settlement.json'])
    admission = _decode(artifacts['native/worker_admission.json'])
    plan = _decode(artifacts['native/worker_plan.json'])
    settlement = receipt['settlement']
    empty = worker['empty_observation']
    require(settlement['settled'] is worker['settled'] is True and worker['child_reaped'] is True
         and worker['child_status'] in ('timed_out', 'exited') and type(worker['child_exit_code']) is int
         and empty['empty'] is True and empty['procs'] == empty['threads'] == []
         and empty['events']['populated'] == '0' and settlement['empty_observation'] == empty
         and settlement['receipt_sha256'] == sha256_hex(artifacts['native/worker_settlement.json'])
         and settlement['receipt_path'] == str(attempt / 'worker_settlement.json')
         and worker['operation_id'] == admission['operation_id'] == plan['operation_id'] == settlement['operation_id']
         and worker['worker_identity'] == admission['worker_identity'] == settlement['cgroup_identity']
         and worker['worker_path'] == admission['worker_path'] == settlement['cgroup_path']
         and worker['child_pid'] == admission['child_pid']
         and worker['child_start_ticks'] == admission['child_start_ticks']
         and worker['unit'] == admission['unit'] == plan['unit']
         and plan['attempt'] == str(attempt)
         and plan['source_sha256'] == sha256_hex(artifacts['native/worker_supervisor.py']),
         'positive original worker closure differs', ValueError)
    if worker['child_status'] == 'timed_out':
        require(worker['owned_cgroup_kill'] is True and worker['child_exit_code'] == -9,
             'timed-out worker lacks positive kill/reap closure', ValueError)
    outcome = _decode(artifacts['outcome.json'])
    elapsed = (worker['elapsed_seconds'], receipt['elapsed_seconds'], outcome['wall_seconds'])
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in elapsed)
         and elapsed[0] <= elapsed[1] <= elapsed[2] and outcome['execution_may_have_started'] is True,
         'known nested elapsed costs differ', ValueError)
    reported_cost = observed['native_total_cost_usd']
    if backend == 'codex':
        # The real Codex parser retained the exact terminal usage above. Standard
        # events do not report price or per-model cost, so those stay unknown.
        require(reported_cost is None and observed['native_model_usage'] is None,
             'failed Codex price or per-model usage was invented', ValueError)
        provider_usage = observed['usage']
    elif explicit:
        # Accounting is bound exactly to the raw failed terminal. It keeps the
        # reported, unqualified incremental scope only: charged costs, fitness and
        # usage authority are unchanged.
        reported = observed['usage_status'] == 'reported_native_terminal_unqualified_incremental'
        require(reported_cost == terminal.get('total_cost_usd')
             and (reported_cost is None or (type(reported_cost) in (int, float)
                  and math.isfinite(reported_cost) and reported_cost >= 0))
             and observed['native_model_usage'] == terminal.get('modelUsage')
             and (type(observed['usage']) is dict and observed['usage'] == terminal.get('usage') if reported
                  else observed['usage_status'] == 'unknown' and observed['usage'] is None
                  and terminal.get('usage') is None),
             'explicit failed terminal accounting differs from its raw terminal', ValueError)
        provider_usage = observed['usage']
    else:
        require(observed['usage_status'] == 'unknown' and observed['usage'] is None
             and observed['native_model_usage'] is None and reported_cost is None,
             'unknown provider accounting changed; separate known-usage closure required', ValueError)
        provider_usage = None
    failed = {'error_kind': failure['error_kind'], 'error': receipt['error'],
              'native_status': receipt['status'], 'returncode': receipt['returncode']}
    if explicit:
        failed.update(terminal='turn.failed', native_terminal=observed['native_terminal'],
                      usage_status=observed['usage_status'])
    return {'costs': {'wall_seconds': outcome['wall_seconds'], 'evaluations': 0},
            'failure': failed, 'provider_usage': provider_usage, 'reported_model_cost_usd': reported_cost}
