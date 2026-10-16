"""Run one planner role through caller-supplied model and process services.

Prompt bytes pass unchanged on stdin. Every call keeps its requested identity,
original deadline and positive process settlement. GPU evaluation is separate.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess

from .._util import canonical_json, require, sha256_hex
from ..model_profile import ModelProfile


CLEANUP_SECONDS = 30
MAX_REQUEST_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ModelRuntime:
    """The caller's model services and monotonic clock.

    The three services have the same interfaces as the original ``dependencies``
    tuple. The clock supplies ``boot_id``, ``monotonic`` and ``monotonic_ns``.
    Optional route, event and boundary services validate retained Codex failures.
    No model or process starts when this binding is created.

    The host must verify the admitted source identities and installation custody
    of these services before supplying them. Injection does not verify imports;
    see docs/evo/planner-port.md for the exact host prerequisite.
    """

    qualified: object
    transport: object
    settings: object
    clock: object = None
    routes: object = None
    events: object = None
    boundary: object = None

    @property
    def dependencies(self):
        return self.qualified, self.transport, self.settings


def need(condition, reason):
    """Compatibility name for callers of the original backend."""
    require(condition, reason, ValueError)


def encoded(value):
    return canonical_json(value)


def decode(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON field', ValueError)
            result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def read(path):
    return decode(Path(path).read_bytes())


def put(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_hex(path.read_bytes())}


def canonical_dir(path, *, create=False):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'absolute canonical directory required', ValueError)
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(path.resolve(strict=True) == path and path.is_dir(), 'aliased directory is forbidden', ValueError)
    return path


def workflow_backend(config):
    backend = config.get('workflow_backend', 'claude')
    require(type(backend) is str and backend in ('claude', 'codex'), 'unsupported model backend', ValueError)
    return backend


def validate_runtime_dependencies(config, *, dependencies=None, runtime=None):
    """Check supplied services before creating any call or attempt files."""
    if dependencies is None:
        require(isinstance(runtime, ModelRuntime), 'explicit model runtime dependencies required', ValueError)
        dependencies = runtime.dependencies
    require(type(dependencies) in (tuple, list) and len(dependencies) == 3,
            'model dependencies must contain qualified, transport and settings services', ValueError)
    qualified, transport, settings = dependencies
    require(callable(getattr(transport, 'validate_model_route', None)),
            'model transport requires route validation', ValueError)
    if workflow_backend(config) == 'codex':
        require(all(callable(getattr(transport, name, None)) for name in
                    ('budget_policy', 'create_runtime', 'inspect_codex_events', 'completed')),
                'Codex model transport services are incomplete', ValueError)
    else:
        require(callable(getattr(qualified, 'QualifiedClaudeTransport', None))
                and callable(getattr(settings, 'FileSettingsRuntime', None))
                and callable(getattr(transport, 'inspect_claude_events', None)),
                'Claude model transport services are incomplete', ValueError)
    return qualified, transport, settings


def codex_imports(config, *, runtime=None):
    """Use the explicitly supplied Codex services without source-path loading."""
    require(workflow_backend(config) == 'codex', 'Codex model backend required', ValueError)
    return imports(runtime=runtime)


def resolve_model_route(config, *, transport=None, runtime=None):
    reference = config.get('model_route_ref')
    require(type(reference) is dict and set(reference) == {'path', 'sha256'},
         'required pinned native model route is missing or malformed', ValueError)
    path = Path(reference['path'])
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
         'canonical native model route file required', ValueError)
    raw = path.read_bytes()
    require(len(raw) <= MAX_REQUEST_BYTES and sha256_hex(raw) == reference['sha256'],
         'native model route reference changed', ValueError)
    if transport is None:
        if workflow_backend(config) == 'codex':
            transport = codex_imports(config, runtime=runtime)[1]
        elif runtime is not None:
            transport = runtime.transport
        else:
            from ..engineer import backend as transport
    return transport.validate_model_route(decode(raw)), dict(reference)


def validate_request(config, request, *, transport=None, runtime=None):
    workflow_backend(config)
    require(config.get('schema') == 'geak-standalone-model-config-v1', 'wrong model config', ValueError)
    require(config.get('launch_admitted') is True, 'model config is not launch-admitted', ValueError)
    require(set(request) == {'schema', 'invocation_id', 'prompt', 'label', 'cwd', 'model', 'timeout_ms'},
         'unexpected model request fields', ValueError)
    require(request['schema'] == 'geak-standalone-model-request-v1', 'wrong model request', ValueError)
    require(type(request['invocation_id']) is str and
         re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', request['invocation_id']), 'invalid invocation identity', ValueError)
    profile, _ = resolve_model_route(config, transport=transport, runtime=runtime)
    require(request['model'] in (None, profile['requested_model']), 'model override differs from pinned route', ValueError)
    require(type(request['prompt']) is str and 0 < len(request['prompt'].encode()) <= MAX_REQUEST_BYTES,
         'bounded nonempty role prompt required', ValueError)
    require(type(request['label']) is str and len(request['label']) <= 512, 'invalid role label', ValueError)
    require(type(request['timeout_ms']) in (int, float) and math.isfinite(request['timeout_ms'])
         and 5000 <= request['timeout_ms'] <= 14400000, 'finite native role deadline required', ValueError)
    work = canonical_dir(config['work_root'])
    cwd = canonical_dir(request['cwd'] or str(work))
    require(cwd == work or work in cwd.parents, 'model cwd escaped writable work root', ValueError)
    return work, cwd


def remaining_budget(config, request, *, now_ns=None, boot_id=None, clock=None):
    if clock is None:
        from ..clock import SystemClock
        clock = SystemClock()
    record = read(config['clock_path'])
    if now_ns is None:
        now_ns = clock.monotonic_ns()
    if boot_id is None:
        boot_id = clock.boot_id()
    start = record.get('started_monotonic_ns')
    wall = record.get('wall_seconds')
    require(record.get('boot_id') == boot_id and type(start) is int and 0 < start <= now_ns,
         'clock is not this running experiment', ValueError)
    # A campaign can fund several fresh investigations. The native request
    # still has its independent <=4h cap; this immutable clock is the full
    # admitted campaign horizon and is never renewed by the engineer.
    require(type(wall) in (int, float) and math.isfinite(wall) and wall > 0,
         'campaign horizon must be finite and positive', ValueError)
    remaining = (start + int(wall * 1e9) - now_ns) / 1e9
    budget = min(request['timeout_ms'] / 1000, remaining - CLEANUP_SECONDS)
    require(budget >= 5, 'original experiment deadline leaves no role and cleanup budget', ValueError)
    return budget, ref(config['clock_path'])


def imports(*, runtime=None):
    """Return caller-supplied services; production does not search study paths."""
    require(isinstance(runtime, ModelRuntime), 'explicit model runtime dependencies required', ValueError)
    return runtime.dependencies


class NativeRoleEnvelope:
    """Explicit constructor-policy extension; launch/settle remain inherited.

    The qualified CPU constructor caps calls at 570s. The standalone role budget
    may be longer, bounded by the original experiment clock. This wrapper changes
    only the two deadline fields before any launch; it is a prospective timing
    policy revision, not a claim that a 4h role was live-qualified previously.
    """
    def __init__(self, runtime, seconds, workspace, inputs, absolute_work_deadline):
        self.inner = runtime.boundary(min(seconds, 570), workspace, inputs)
        # QualifiedClaudeTransport returns this actual boundary; apply the file
        # settings verification in our caller before launch, not a hidden proxy.
        require(hasattr(self.inner, 'work_deadline_monotonic') and self.inner.plan is None,
             'unexpected CPU boundary implementation', ValueError)
        self.inner.work_deadline_monotonic = min(self.inner.created_monotonic + seconds,
                                                 absolute_work_deadline)
        self.inner.wall_seconds = self.inner.work_deadline_monotonic - self.inner.created_monotonic
        require(self.inner.wall_seconds >= 5, 'transport setup consumed the remaining role budget', ValueError)

    def launch(self, argv, **kwargs):
        return self.inner.launch(argv, **kwargs)

    def settle(self, process, attempt):
        return self.inner.settle(process, attempt)


def run(config, request, *, dependencies=None, runtime=None):
    is_codex = workflow_backend(config) == 'codex'
    qualified, transport, settings = validate_runtime_dependencies(config, dependencies=dependencies, runtime=runtime)
    clock_source = runtime.clock if runtime is not None else None
    if clock_source is None:
        from ..clock import SystemClock
        clock_source = SystemClock()
    work, cwd = validate_request(config, request, transport=transport)
    model_route, model_route_ref = resolve_model_route(config, transport=transport)
    profile = ModelProfile.from_route(model_route, model_route_ref, workflow_backend=workflow_backend(config))
    if is_codex:
        transport.budget_policy(config, {'mode': 'fresh', 'prior_receipt_ref': None})
        require(cwd == work, 'Codex role cwd must equal its admitted workspace', ValueError)
    state = canonical_dir(config['state_root'], create=True)
    require(state != work and work not in state.parents and state not in work.parents,
         'model state overlaps tool workspace', ValueError)
    require(state.stat().st_mode & 0o077 == 0, 'model state must be private', ValueError)
    require(not (state / 'STOP_UNSETTLED').exists(), 'an earlier model invocation remains unsettled', ValueError)
    attempt = state / 'attempts' / request['invocation_id']
    attempt.parent.mkdir(mode=0o700, exist_ok=True)
    attempt.mkdir(mode=0o700)  # Durable no-replay fence.
    put(attempt / 'request.json', request)
    if is_codex:
        # Bind the actual call configuration before the native intent or model
        # launch. The planner's earlier template admission remains unchanged.
        put(attempt / 'model-config.json', config)
        (attempt / 'model-config.json').chmod(0o400)
        model_config_ref = ref(attempt / 'model-config.json')
    budget, clock_ref = remaining_budget(config, request, clock=clock_source)
    # Freeze the absolute deadline before transport preparation. Constructor
    # audits must not shift the original experiment or requested role cutoff.
    clock = read(config['clock_path'])
    require(ref(config['clock_path']) == clock_ref, 'original clock changed during preparation', ValueError)
    absolute_work_deadline = min(clock_source.monotonic() + request['timeout_ms'] / 1000,
        clock['started_monotonic_ns'] / 1e9 + clock['wall_seconds'] - CLEANUP_SECONDS)
    roots = [canonical_dir(path) for path in config['read_roots']]
    require(all(root != work and root not in work.parents and work not in root.parents for root in roots),
         'read-only roots overlap tool workspace', ValueError)
    report_path = attempt / 'final-message.txt' if is_codex else None
    if is_codex:
        runtime = transport.create_runtime(config, model_route, model_route_ref, config['tool_audit_ref'])
        descriptor = runtime.descriptor
        require(descriptor['workspace'] == str(work) and descriptor['read_roots'] == [str(root) for root in roots],
             'Codex descriptor differs from actual role workspace/read roots', ValueError)
        require(any(state == Path(path) or Path(path) in state.parents for path in descriptor['operational_roots']),
             'Codex tool policy does not exclude actual model state', ValueError)
        base_runtime, tool_audit = runtime, config['tool_audit_ref']
    else:
        route = qualified.QualifiedClaudeTransport(state / 'private' / request['invocation_id'],
                    read_roots=lambda workspace: [str(root) for root in roots],
                    model_route=model_route, model_route_ref=model_route_ref)
        base_runtime = route.runtime(work)
        runtime = settings.FileSettingsRuntime(base_runtime)
        tool_audit = route.tool_audit
    runtime.audit(work, tool_audit)
    command = runtime.command({'mode': 'fresh', 'prior_receipt': None}, report_path)
    boundary = NativeRoleEnvelope(base_runtime, budget, work, [], absolute_work_deadline)
    budget = boundary.inner.wall_seconds
    receipt = {'schema': 'geak-standalone-model-receipt-v1', 'invocation_id': request['invocation_id'],
               'label': request['label'], 'requested_model': model_route['requested_model'],
               'model_route_ref': model_route_ref, 'clock_ref': clock_ref,
               'model_profile': profile.as_dict(),
               'transport': runtime.identity(), 'tool_audit': tool_audit,
               'model_backend_ref': ref(__file__), 'effective_wall_seconds': budget,
               'original_native_timeout_ms': request['timeout_ms'],
               'timing_policy_revision': 'native-role-envelope-with-original-experiment-deadline-v1',
               'started_monotonic_ns': clock_source.monotonic_ns(), 'settlement': {'settled': False}}
    if is_codex:
        receipt['requested_reasoning_effort'] = model_route['requested_reasoning_effort']
        receipt['model_config_ref'] = model_config_ref
    put(attempt / 'intent.json', receipt)
    (attempt / 'prompt.txt').write_bytes(request['prompt'].encode())
    process = None
    failure = None
    interruption = None
    began = clock_source.monotonic()
    try:
        with (attempt / 'prompt.txt').open('rb') as stdin, (attempt / 'events.jsonl').open('xb') as stdout, \
                (attempt / 'stderr.txt').open('xb') as stderr:
            runtime._verify()
            if is_codex:
                require(ref(attempt / 'model-config.json') == model_config_ref
                     and read(attempt / 'model-config.json') == config, 'actual model configuration changed before launch', ValueError)
            # The admitted writable workspace is explicit; the selected cwd is
            # within it and never taken from a model-generated command string.
            process = boundary.launch(command, cwd=str(cwd), stdin=stdin, stdout=stdout,
                                      stderr=stderr, shell=False, start_new_session=True)
            try:
                code = process.wait(timeout=budget + CLEANUP_SECONDS)
                receipt['returncode'] = code
            except subprocess.TimeoutExpired:
                # Do not start a retry while the supervisor is still live.
                receipt['transport_timeout'] = True
                failure = 'CPU supervisor exceeded role plus cleanup bound'
    except BaseException as exc:
        failure = type(exc).__name__ + ': ' + str(exc)[:400]
        if not isinstance(exc, Exception):
            interruption = exc
    finally:
        try:
            receipt['settlement'] = boundary.settle(process, attempt)
        except BaseException as exc:
            # A failed settlement observer is an unresolved operation, never a
            # completed invocation that permits a later model retry.
            receipt['settlement'] = {'settled': False,
                'error': type(exc).__name__ + ': ' + str(exc)[:400]}
            failure = failure or 'CPU settlement observer failed'
            if not isinstance(exc, Exception):
                interruption = exc
        receipt['elapsed_seconds'] = clock_source.monotonic() - began
    if receipt['settlement'].get('settled') is not True:
        failure = failure or 'CPU descendants lack positive settlement'
        marker = state / 'STOP_UNSETTLED'
        try:
            put(marker, {'invocation_id': request['invocation_id'], 'reason': failure})
        except FileExistsError:
            pass
    elif (state / 'STOP_UNSETTLED').exists():
        failure = failure or 'a concurrent model invocation remains unsettled'
    try:
        if is_codex:
            require(ref(attempt / 'model-config.json') == model_config_ref,
                 'actual model configuration changed during invocation', ValueError)
        observed = (transport.inspect_codex_events(attempt / 'events.jsonl', report_path=report_path)
                    if is_codex else transport.inspect_claude_events(attempt / 'events.jsonl', model_route=model_route))
        receipt['observed'] = observed
        complete = (transport.completed(observed) if is_codex else
                    observed.get('identity_valid') is True and observed.get('reported_models')
                    and all(model in model_route['accepted_reported_models'] for model in observed['reported_models'])
                    and observed.get('reported_tools') == ['Bash']
                    and observed.get('terminal') == 'turn.completed' and not observed.get('protocol_errors')
                    and type(observed.get('report_text')) is str)
        require(receipt.get('returncode') == 0 and not failure and complete,
             'model turn did not complete authentically', ValueError)
    except BaseException as exc:
        failure = failure or type(exc).__name__ + ': ' + str(exc)[:400]
        if not isinstance(exc, Exception):
            interruption = exc
    receipt['status'] = 'failed' if failure else 'completed'
    receipt['error'] = failure
    put(attempt / 'receipt.json', receipt)
    if interruption is not None:
        raise interruption
    require(not failure, failure, ValueError)
    return {'schema': 'geak-standalone-model-result-v1', 'invocation_id': request['invocation_id'],
            'text': observed['report_text'], 'receipt_ref': ref(attempt / 'receipt.json'),
            'elapsed_seconds': receipt['elapsed_seconds'], 'settled': True}
